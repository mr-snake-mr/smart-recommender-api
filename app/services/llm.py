"""LLM service layer: the three prompt calls from the blueprint.

- extract_constraints  → Prompt 1 (Phase 1)
- judge               → Prompt 2A (Phase 3, Path A)
- pivot               → Prompt 2B (Phase 3, Path B)

Uses an OpenAI-compatible async client (works with OpenAI, DeepSeek, Ollama,
etc. via OPENAI_BASE_URL). When LLM_MOCK_MODE=true it falls back to a
deterministic rule-based mock so the whole pipeline runs with zero API keys —
handy for demos, CI, and the /admin dashboard smoke tests.
"""
import json
import re
from typing import Any

from app import prompts
from app.config import get_settings
from app.schemas import DEFAULT_MAX_PRICE, VALID_CATEGORIES, ExtractedConstraints

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_BUDGET_KEYWORDS = ("under", "below", "less than", "budget", "max", "within", "at most", "no more than")
_NEGATION_MARKERS = ("no", "without", "absolutely no", "not", "never", "avoid", "excluding", "except", "no refurbished")
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "electronics": ("laptop", "computer", "phone", "headphone", "earbud", "monitor", "keyboard", "mouse", "camera", "tv", "tablet", "gaming"),
    "jewelery": ("jewel", "ring", "necklace", "bracelet", "earring", "diamond", "gold", "silver"),
    "men's clothing": ("men", "shirt", "polo", "jacket", "hoodie", "sweatshirt", "jeans", "t-shirt", "tshirt", "male"),
    "women's clothing": ("women", "dress", "blouse", "skirt", "bra", "female"),
}

# Prompts that produce non-JSON filler text before the payload.
_STRIP_PREFIXES = ("```json", "```", "json")


class LLMResponseError(RuntimeError):
    """Raised when the LLM returns something that cannot be parsed into the schema."""


# ── JSON parsing helpers ──────────────────────────────────────────────────────
def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    for prefix in _STRIP_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    cleaned = cleaned.strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise LLMResponseError("LLM output could not be parsed as JSON")


def _force_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _force_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Real LLM client ───────────────────────────────────────────────────────────
class LLMService:
    def __init__(self) -> None:
        settings = get_settings()
        self._mock = settings.llm_mock_mode
        self._client: Any = None
        if not self._mock and settings.openai_api_key:
            from openai import AsyncOpenAI  # lazy: only needed in real-LLM mode

            self._client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self._model = settings.llm_model

    # Prompt 1 ──────────────────────────────────────────────────────────────────
    async def extract_constraints(self, raw_user_query: str) -> ExtractedConstraints:
        if self._mock or self._client is None:
            return _mock_extract(raw_user_query)

        data = await self._complete(prompts.format_extractor(raw_user_query))
        return ExtractedConstraints(
            max_price=_force_float(data.get("max_price"), DEFAULT_MAX_PRICE),
            category=_normalize_category(str(data.get("category", "electronics"))),
            must_not_include=[str(k).lower().strip() for k in _force_str_list(data.get("must_not_include"))],
            soft_preferences=_force_str_list(data.get("soft_preferences")),
        )

    # Prompt 2A ─────────────────────────────────────────────────────────────────
    async def judge(self, soft_preferences: list[str], top_10_products: list[dict[str, Any]]) -> dict[str, Any]:
        if self._mock or self._client is None:
            return _mock_judge(soft_preferences, top_10_products)

        data = await self._complete(prompts.format_judge(soft_preferences, top_10_products))
        return _validate_judge(data)

    # Prompt 2B ─────────────────────────────────────────────────────────────────
    async def pivot(self, failed_constraints: dict[str, Any]) -> dict[str, Any]:
        if self._mock or self._client is None:
            return _mock_pivot(failed_constraints)

        data = await self._complete(prompts.format_pivot(failed_constraints))
        return _validate_pivot(data)

    # ── internals ──────────────────────────────────────────────────────────────
    async def _complete(self, prompt: str) -> dict[str, Any]:
        assert self._client is not None
        last_error: Exception | None = None
        for _ in range(2):  # one retry on malformed output
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                }
                try:  # some OpenAI-compatible endpoints reject response_format
                    kwargs["response_format"] = {"type": "json_object"}
                    resp = await self._client.chat.completions.create(**kwargs)
                except Exception:
                    kwargs.pop("response_format", None)
                    resp = await self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or "{}"
                return parse_json(content)
            except LLMResponseError as exc:
                last_error = exc
        raise LLMResponseError(f"LLM returned unparseable output twice: {last_error}")


# ── Mock implementations (deterministic, no API calls) ────────────────────────
def _normalize_category(category: str) -> str:
    cat = (category or "").strip().lower()
    if cat in VALID_CATEGORIES:
        return cat
    if cat == "jewelry":
        return "jewelery"
    return "electronics"


def _mock_extract(query: str) -> ExtractedConstraints:
    text = query.lower().strip()
    tokens = re.findall(r"[a-z0-9]+", text)

    # Budget: pick the largest currency amount mentioned next to a budget word.
    amounts = [float(m) for m in re.findall(r"\$?\s?(\d+(?:\.\d+)?)", text) if float(m) > 0]
    mentioned = [a for a in amounts if any(kw in text for kw in _BUDGET_KEYWORDS)]
    max_price = max(mentioned) if mentioned else DEFAULT_MAX_PRICE

    # Category by keyword match.
    category = "electronics"
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            category = cat
            break

    # Dealbreakers: noun phrases after negation markers, plus known red flags.
    must_not: list[str] = []
    for marker in ("absolutely no", "no refurbished", "without", "never", "avoid", "no", "not", "excluding", "except"):
        idx = text.find(marker)
        if idx == -1:
            continue
        tail = text[idx + len(marker):]
        phrase = re.match(r"\s*([a-z][a-z \-']*?)(?=\s*(?:and|or|but|,|\.|$))", tail)
        if phrase:
            word = phrase.group(1).strip().split()[0] if phrase.group(1).strip() else ""
            if word and word not in must_not and word not in _STOPWORDS_EXTRA:
                must_not.append(word)
    if "refurbished" in text and "refurbished" not in must_not:
        must_not.append("refurbished")
    if "used" in text and "used" not in must_not:
        must_not.append("used")

    # Soft preferences: leftover descriptive tokens minus stopwords/numbers.
    stop = {"under", "below", "less", "than", "budget", "max", "within", "at", "most",
            "no", "more", "absolutely", "without", "not", "never", "avoid", "excluding",
            "except", "with", "and", "or", "a", "an", "the", "for", "to", "of", "i", "need", "want", "looking"}
    soft: list[str] = []
    for tok in tokens:
        if tok in stop or tok.isdigit() or tok in must_not:
            continue
        if tok not in soft:
            soft.append(tok)
    return ExtractedConstraints(
        max_price=max_price,
        category=category,
        must_not_include=[m for m in must_not],
        soft_preferences=soft,
    )


_STOPWORDS_EXTRA = {"and", "or", "but", "with", "that", "this", "for", "any", "some", "please"}


def _mock_judge(soft_preferences: list[str], top_10: list[dict[str, Any]]) -> dict[str, Any]:
    prefs = ", ".join(soft_preferences) if soft_preferences else "your requirements"
    prefs_lower = prefs.lower()
    picks = top_10[:2]
    matches = {}
    for slot, item in zip(("match_1", "match_2"), picks):
        pid = int(item.get("id", 0))
        name = str(item.get("title", f"Product #{pid}"))
        desc = str(item.get("description", "")).lower()
        found = [p for p in soft_preferences if p.lower() in desc or p.lower() in name.lower()]
        pros = [f"Matches your '{p}' preference" for p in found[:2]]
        while len(pros) < 2:
            pros.append("Within your stated budget")
            break
        cons = [f"Not explicitly tailored to '{prefs}'"] if prefs_lower and prefs != "your requirements" else ["Only two finalists selected"]
        matches[slot] = {
            "id": pid,
            "name": name,
            "why_it_won": f"Best TF-IDF match for: {prefs}",
            "pros": pros[:2],
            "cons": cons[:1],
        }
    if len(picks) < 2:  # only one survivor
        matches["match_2"] = {
            "id": -1,
            "name": "No second finalist in the filtered set",
            "why_it_won": "Only one product met every hard constraint",
            "pros": [],
            "cons": [],
        }
    return matches


def _mock_pivot(failed_constraints: dict[str, Any]) -> dict[str, Any]:
    max_price = _force_float(failed_constraints.get("max_price"), DEFAULT_MAX_PRICE)
    must_not = failed_constraints.get("must_not_include") or []
    if max_price < DEFAULT_MAX_PRICE:
        suggestion = f"Try increasing your budget above ${max_price:,.0f}, or removing the 'must not include' terms {must_not}."
    elif must_not:
        suggestion = f"Consider relaxing one of your dealbreakers ({', '.join(must_not)}), which excluded every catalog item."
    else:
        suggestion = "Try searching a different category — nothing in this one matched your request."
    return {
        "status": "unmet_constraint",
        "message": "We couldn't find any products that meet every one of your constraints in our current catalog.",
        "suggested_action": suggestion,
    }


# ── Output validation for real-LLM responses ──────────────────────────────────
def _validate_judge(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for slot in ("match_1", "match_2"):
        entry = data.get(slot)
        if not isinstance(entry, dict):
            raise LLMResponseError(f"Judge response missing '{slot}'")
        out[slot] = {
            "id": int(entry.get("id", 0)),
            "name": str(entry.get("name", "")),
            "why_it_won": str(entry.get("why_it_won", "")),
            "pros": _force_str_list(entry.get("pros"))[:2],
            "cons": _force_str_list(entry.get("cons"))[:1],
        }
    return out


def _validate_pivot(data: dict[str, Any]) -> dict[str, Any]:
    message = str(data.get("message", "")).strip()
    suggestion = str(data.get("suggested_action", "")).strip()
    if not message or not suggestion:
        raise LLMResponseError("Pivot response missing message/suggested_action")
    return {
        "status": "unmet_constraint",
        "message": message,
        "suggested_action": suggestion,
    }

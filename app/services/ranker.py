"""Phase 2 of the blueprint: strict filtering + TF-IDF / cosine similarity ranking.

No AI here — pure Python math. The strict filter enforces hard constraints
(budget, category, dealbreakers); the ranker scores the survivors against the
user's soft preferences and keeps the top N.
"""
import math
import re
from collections import Counter

from app.schemas import ExtractedConstraints, Product, RankedProduct

VALID_CATEGORIES = {"electronics", "jewelery", "men's clothing", "women's clothing"}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "with", "for", "to", "of", "in", "on",
    "at", "by", "from", "under", "over", "below", "above", "less", "than", "more",
    "absolutely", "no", "not", "without", "never", "avoid", "excluding", "except",
    "i", "need", "want", "looking", "great", "good", "really", "very", "that",
    "this", "is", "are", "it", "my", "me", "do", "does", "has", "have", "be",
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_category(category: str) -> str:
    """Map user-facing category names onto the FakeStore API vocabulary."""
    cat = (category or "").strip().lower()
    if cat == "jewelry":  # API misspells it; users usually don't
        return "jewelery"
    return cat


# ── Strict filter ─────────────────────────────────────────────────────────────
def strict_filter(products: list[Product], constraints: ExtractedConstraints) -> list[Product]:
    """Drop anything over budget, in the wrong category, or matching a dealbreaker."""
    cat = normalize_category(constraints.category)
    forbidden = [k.strip().lower() for k in constraints.must_not_include if k and k.strip()]

    survivors: list[Product] = []
    for p in products:
        if p.price > constraints.max_price:
            continue
        if cat in VALID_CATEGORIES and normalize_category(p.category) != cat:
            continue
        if forbidden:
            haystack = p.searchable_text.lower()
            if any(keyword in haystack for keyword in forbidden):
                continue
        survivors.append(p)
    return survivors


# ── TF-IDF + cosine similarity ────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _tfidf_vectors(corpus: list[list[str]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Return (tfidf vector per doc, idf per token). Add-one-smoothed idf."""
    n_docs = len(corpus)
    df: Counter = Counter()
    for doc in corpus:
        df.update(set(doc))
    idf = {token: math.log((1 + n_docs) / (1 + count)) + 1.0 for token, count in df.items()}

    vectors: list[dict[str, float]] = []
    for doc in corpus:
        tf = Counter(doc)
        total = sum(tf.values()) or 1
        vectors.append({token: (count / total) * idf.get(token, 0.0) for token, count in tf.items()})
    return vectors, idf


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def tfidf_rank(
    products: list[Product],
    soft_preferences: list[str],
    top_n: int = 10,
) -> list[RankedProduct]:
    """Score products against the soft preferences and return the top N.

    Falls back to a popularity tie-break (rating desc, price asc) when there
    are no soft preferences to score against.
    """
    if not products:
        return []

    pref_text = " ".join(p for p in soft_preferences if p)
    corpus = [_tokenize(pref_text)] + [_tokenize(p.searchable_text) for p in products]
    vectors, _ = _tfidf_vectors(corpus)
    query_vec, product_vecs = vectors[0], vectors[1:]

    ranked = [
        RankedProduct(product=p, score=_cosine(query_vec, vec))
        for p, vec in zip(products, product_vecs)
    ]
    ranked.sort(key=lambda rp: (-rp.score, -rp.product.rating_value, rp.product.price))
    return ranked[:top_n]

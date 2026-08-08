# Smart Recommender API (FastAPI)

Backend for the orchestration blueprint: one `POST /api/recommend` call chains
**user query → LLM constraint extraction → strict math filter → TF-IDF ranking →
LLM expert judge (or polite pivot)** and returns a frontend-ready JSON payload.

## How the pipeline maps to the blueprint

```
 Phase 1                     Phase 2 (no AI)                 Phase 3 (fork)
┌────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌────────────────────────┐
│ raw query  │──▶│ LLM Call 1    │──▶│ FakeStore fetch  │   │ len(top_10) > 0        │
│            │   │ The Extractor │   │ strict filter    │   │   LLM Call 2A (Judge)  │
│            │   │ (Prompt 1)    │   │ TF-IDF + cosine  │──▶│   → Vs. Grid + scores  │
│            │   └───────────────┘   │ top 10           │   │                        │
│            │                       └──────────────────┘   │ len(top_10) == 0      │
│            │                                              │   analytics push       │
│            │                                              │   LLM Call 2B (Pivot)  │
│            │                                              │   → advisory banner    │
└────────────┘                                              └────────────────────────┘
```

| Blueprint phase | Implementation |
|---|---|
| Phase 1 — Extractor (Prompt 1) | `app/services/llm.py::LLMService.extract_constraints` |
| Phase 2 — fetch / filter / rank | `app/services/catalog.py`, `app/services/ranker.py` |
| Phase 3 Path A — Judge (Prompt 2A) | `LLMService.judge` + `app/main.py::_path_a` |
| Phase 3 Path B — analytics + Pivot (Prompt 2B) | `app/services/analytics.py` + `_path_b` |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your LLM key

uvicorn app.main:app --reload   # http://localhost:8000
```

Try it:

```bash
curl -s http://localhost:8000/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"query": "I need a coding laptop under $1000 with a great screen, no refurbished models."}'
```

**Mock mode (no API key needed):** set `LLM_MOCK_MODE=true` in `.env`. The
pipeline then runs end-to-end with deterministic rule-based extraction and
canned judge/pivot responses — ideal for demos, CI, and frontend development.

Smoke test (no network, no keys):

```bash
source .venv/bin/activate
python scripts/smoke_test.py
```

## API contract

### `POST /api/recommend`

Request:

```json
{ "query": "I need a coding laptop under $1000 with a great screen, no refurbished models." }
```

Success response (Path A — render the Vs. Grid):

```json
{
  "outcome": "success",
  "match_1": {
    "id": 2, "name": "Dell XPS 13 Laptop",
    "why_it_won": "...", "pros": ["...", "..."], "cons": ["..."],
    "similarity_score": 0.61
  },
  "match_2": { "id": 4, "name": "Samsung 4K Monitor", "...": "..." }
}
```

Unmet-constraint response (Path B — render the advisory banner):

```json
{
  "outcome": "unmet_constraint",
  "status": "unmet_constraint",
  "message": "We couldn't find any products that meet every one of your constraints...",
  "suggested_action": "Try increasing your budget above $20..."
}
```

Branch on `outcome` in Next.js: `success` → Vs. Grid; `unmet_constraint` →
banner. Error responses: `502` (LLM output unparseable), `503` (catalog API down).

### `GET /health` → `{"status": "ok"}`

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | — / `https://api.openai.com/v1` / `gpt-4o-mini` | Any OpenAI-compatible endpoint (OpenAI, DeepSeek, Ollama...) |
| `LLM_MOCK_MODE` | `false` | `true` = deterministic mock, zero API calls |
| `FAKESTORE_API_URL` | `https://fakestoreapi.com/products` | Primary catalog source |
| `FALLBACK_CATALOG_API_URL` | `https://dummyjson.com/products?limit=100` | Fallback source used when the primary source is unavailable |
| `CATALOG_CACHE_TTL_SECONDS` | `300` | In-process catalog cache TTL |
| `ANALYTICS_BACKEND` | (empty) | `redis` or `supabase` to log unmet constraints for the `/admin` dashboard |
| `REDIS_URL` / `SUPABASE_URL` / `SUPABASE_ANON_KEY` / `SUPABASE_TABLE` | — | Backend-specific credentials |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated Next.js origins |

## Project layout

```
app/
  main.py            # FastAPI app + POST /api/recommend (the chain + fork)
  config.py          # pydantic-settings, env-driven
  schemas.py         # request/response contracts (Vs. Grid, pivot)
  prompts.py         # Prompt 1 / 2A / 2B templates (verbatim from blueprint)
  services/
    llm.py           # LLM calls + JSON parsing + mock mode
    catalog.py       # FakeStore fetch with TTL cache
    ranker.py        # strict filter + pure-Python TF-IDF / cosine similarity
    analytics.py     # fire-and-forget Redis/Supabase logging (Path B)
scripts/
  smoke_test.py      # end-to-end test of both forks (mock LLM + stubbed catalog)
```

## Notes & next steps

- The ranker is pure Python (no scikit-learn) so the math is dependency-free;
  swap in `sklearn.feature_extraction.text.TfidfVectorizer` if you want
  n-grams/stopword corpora for a larger catalog.
- FakeStore categories use the API's own spelling (`jewelery`); the extractor
  normalizes `jewelry` → `jewelery` on both sides.
- `/admin` dashboard: subscribe to the `unmet_constraints` Redis list or the
  Supabase table — every Path B request inserts a row immediately (fire-and-forget,
  never blocks the API response).

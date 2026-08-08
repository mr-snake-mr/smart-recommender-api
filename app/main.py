"""FastAPI backend implementing the full recommendation orchestration blueprint.

POST /api/recommend chains:
  Phase 1  LLM constraints extraction  (Prompt 1)
  Phase 2  FakeStore fetch → strict filter → TF-IDF + cosine rank (top 10)
  Phase 3  fork:
             Path A (matches)  → LLM judge (Prompt 2A) → Vs. Grid + math scores
             Path B (no match) → analytics logging → LLM pivot (Prompt 2B)
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import MatchEntry, PivotResponse, Product, RecommendRequest, VsGridResponse
from app.services import analytics, catalog, llm, ranker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

settings = get_settings()
app = FastAPI(title="Smart Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_service = llm.LLMService()


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Provide a concise production landing response for the public API URL."""
    return JSONResponse(
        {
            "service": "Smart Recommender API",
            "status": "ok",
            "health": "/health",
            "recommend": "/api/recommend",
            "docs": "/docs",
        }
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/recommend", response_model=VsGridResponse | PivotResponse)
async def recommend(req: RecommendRequest) -> VsGridResponse | PivotResponse:
    # ── Phase 1: constraint extraction (LLM Call 1) ──────────────────────────
    try:
        constraints = await llm_service.extract_constraints(req.query)
    except llm.LLMResponseError as exc:
        raise HTTPException(status_code=502, detail=f"Constraint extraction failed: {exc}") from exc

    # ── Phase 2: fetch, strict filter, TF-IDF rank (pure Python) ─────────────
    try:
        products: list[Product] = await catalog.fetch_products()
    except catalog.CatalogFetchError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    survivors = ranker.strict_filter(products, constraints)
    top_10 = ranker.tfidf_rank(survivors, constraints.soft_preferences, top_n=10)

    # ── Phase 3: fork ─────────────────────────────────────────────────────────
    # The Vs. Grid contract requires two real products. A single survivor is an
    # insufficient comparison set, so pivot instead of returning a fake card.
    if len(top_10) >= 2:
        return await _path_a(constraints, top_10)

    return await _path_b(constraints)


# ── Path A: the expert judge (LLM Call 2A) ────────────────────────────────────
async def _path_a(constraints, top_10) -> VsGridResponse:
    top_10_payload = [rp.product.model_dump() for rp in top_10]
    scores = {rp.product.id: round(rp.score, 4) for rp in top_10}

    try:
        judge_result = await llm_service.judge(constraints.soft_preferences, top_10_payload)
    except llm.LLMResponseError as exc:
        raise HTTPException(status_code=502, detail=f"Judge call failed: {exc}") from exc

    def _to_match(entry: dict) -> MatchEntry:
        return MatchEntry(
            id=int(entry["id"]),
            name=str(entry["name"]),
            why_it_won=str(entry["why_it_won"]),
            pros=list(entry["pros"]),
            cons=list(entry["cons"]),
            similarity_score=scores.get(int(entry["id"])),  # math score merged in
        )

    return VsGridResponse(
        match_1=_to_match(judge_result["match_1"]),
        match_2=_to_match(judge_result["match_2"]),
    )


# ── Path B: analytics logging + the pivot (LLM Call 2B) ───────────────────────
async def _path_b(constraints) -> PivotResponse:
    analytics.fire_and_forget(constraints.model_dump())  # /admin dashboard flex

    try:
        pivot_result = await llm_service.pivot(constraints.model_dump())
    except llm.LLMResponseError as exc:
        raise HTTPException(status_code=502, detail=f"Pivot call failed: {exc}") from exc

    return PivotResponse(
        status="unmet_constraint",
        message=str(pivot_result["message"]),
        suggested_action=str(pivot_result["suggested_action"]),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

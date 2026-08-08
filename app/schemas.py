"""Pydantic schemas for the request/response contracts of the recommend pipeline.

The shapes mirror the blueprint's output schemas so the Next.js frontend can
render the Vs. Grid or the advisory banner without transformation.
"""
from typing import Literal

from pydantic import BaseModel, Field

# FakeStore API only carries these four categories (note the API's spelling).
VALID_CATEGORIES = ("electronics", "jewelery", "men's clothing", "women's clothing")

DEFAULT_MAX_PRICE = 99_999.0


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Raw user shopping query")


# ── Phase 1 output ────────────────────────────────────────────────────────────
class ExtractedConstraints(BaseModel):
    max_price: float = Field(default=DEFAULT_MAX_PRICE, ge=0)
    category: str = Field(default="electronics")
    must_not_include: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)


# ── Phase 2 artifacts ─────────────────────────────────────────────────────────
class Product(BaseModel):
    id: int
    title: str
    price: float
    category: str
    description: str = ""
    rating: dict = Field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return f"{self.title} {self.description}"

    @property
    def rating_value(self) -> float:
        rate = self.rating.get("rate") if isinstance(self.rating, dict) else None
        return float(rate) if rate is not None else 0.0


class RankedProduct(BaseModel):
    product: Product
    score: float = 0.0


# ── Phase 3 / Path A output (the "Vs. Grid") ─────────────────────────────────
class MatchEntry(BaseModel):
    id: int
    name: str
    why_it_won: str
    pros: list[str]
    cons: list[str]
    similarity_score: float | None = None  # math score merged by FastAPI


class VsGridResponse(BaseModel):
    outcome: Literal["success"] = "success"
    match_1: MatchEntry
    match_2: MatchEntry


# ── Phase 3 / Path B output (the pivot / advisory banner) ────────────────────
class PivotResponse(BaseModel):
    outcome: Literal["unmet_constraint"] = "unmet_constraint"
    status: str = "unmet_constraint"
    message: str
    suggested_action: str


RecommendResponse = VsGridResponse | PivotResponse

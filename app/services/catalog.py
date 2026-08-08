"""Catalog fetching with a primary provider, fallback provider, and TTL cache."""
import time
from typing import Any

import httpx

from app.config import get_settings
from app.schemas import Product

_cache: dict[str, object] = {"products": None, "fetched_at": 0.0}


class CatalogFetchError(RuntimeError):
    """Raised when every configured catalog source is unavailable."""


async def fetch_products(force_refresh: bool = False) -> list[Product]:
    """Return a normalized catalog, using a fallback if the primary source fails."""
    settings = get_settings()
    now = time.time()

    if (
        not force_refresh
        and _cache["products"] is not None
        and now - float(_cache["fetched_at"]) < settings.catalog_cache_ttl_seconds
    ):
        return list(_cache["products"])  # type: ignore[arg-type]

    errors: list[str] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for url, adapter in (
            (settings.fakestore_api_url, _coerce_fakestore_products),
            (settings.fallback_catalog_api_url, _coerce_dummyjson_products),
        ):
            if not url:
                continue
            try:
                response = await client.get(url)
                response.raise_for_status()
                products = adapter(response.json())
                if not products:
                    raise ValueError("catalog contained no usable products")
                _cache["products"] = products
                _cache["fetched_at"] = now
                return products
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                errors.append(f"{url}: {exc}")

    # Keep serving the last valid catalog during transient upstream outages.
    if _cache["products"] is not None:
        return list(_cache["products"])  # type: ignore[arg-type]
    raise CatalogFetchError("Failed to fetch a catalog from configured sources: " + "; ".join(errors))


def _coerce_fakestore_products(raw: Any) -> list[Product]:
    if not isinstance(raw, list):
        raise ValueError("expected a list of products")
    return [_coerce_product(item) for item in raw if isinstance(item, dict)]


def _coerce_dummyjson_products(raw: Any) -> list[Product]:
    if not isinstance(raw, dict) or not isinstance(raw.get("products"), list):
        raise ValueError("expected a DummyJSON products payload")

    normalized: list[Product] = []
    for item in raw["products"]:
        if not isinstance(item, dict):
            continue
        category = "electronics" if str(item.get("category", "")).lower() == "laptops" else str(item.get("category", ""))
        normalized.append(
            Product(
                id=int(item.get("id", 0)),
                title=str(item.get("title", "")),
                price=float(item.get("price", 0.0)),
                category=category,
                description=str(item.get("description", "")),
                rating={"rate": item.get("rating", 0.0), "count": item.get("stock", 0)},
            )
        )
    return normalized


def _coerce_product(item: dict[str, Any]) -> Product:
    return Product(
        id=int(item.get("id", 0)),
        title=str(item.get("title", "")),
        price=float(item.get("price", 0.0)),
        category=str(item.get("category", "")),
        description=str(item.get("description", "")),
        rating=item.get("rating") or {},
    )

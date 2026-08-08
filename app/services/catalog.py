"""Catalog fetching from the FakeStore API with an in-process TTL cache."""
import time

import httpx

from app.config import get_settings
from app.schemas import Product

_cache: dict[str, object] = {"products": None, "fetched_at": 0.0}


class CatalogFetchError(RuntimeError):
    """Raised when the FakeStore API cannot be reached."""


async def fetch_products(force_refresh: bool = False) -> list[Product]:
    """Return the full catalog, reusing a cached copy for TTL seconds."""
    settings = get_settings()
    now = time.time()

    if (
        not force_refresh
        and _cache["products"] is not None
        and now - float(_cache["fetched_at"]) < settings.catalog_cache_ttl_seconds
    ):
        return list(_cache["products"])  # type: ignore[arg-type]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(settings.fakestore_api_url)
            resp.raise_for_status()
            raw = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        # Fall back to a stale cache if we have one; otherwise surface the error.
        if _cache["products"] is not None:
            return list(_cache["products"])  # type: ignore[arg-type]
        raise CatalogFetchError(f"Failed to fetch catalog from FakeStore API: {exc}") from exc

    products = [_coerce_product(item) for item in raw if isinstance(item, dict)]
    _cache["products"] = products
    _cache["fetched_at"] = now
    return products


def _coerce_product(item: dict) -> Product:
    return Product(
        id=int(item.get("id", 0)),
        title=str(item.get("title", "")),
        price=float(item.get("price", 0.0)),
        category=str(item.get("category", "")),
        description=str(item.get("description", "")),
        rating=item.get("rating") or {},
    )

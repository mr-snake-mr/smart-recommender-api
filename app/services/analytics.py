"""Path B analytics logging ("the hackathon flex").

Fire-and-forget push of failed constraints to Redis or Supabase so the
/admin seller dashboard updates immediately without blocking the API response.
"""
import asyncio
import json
import logging
import time
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


def build_payload(constraints: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "max_price": constraints.get("max_price"),
        "category": constraints.get("category"),
        "must_not_include": constraints.get("must_not_include", []),
        "soft_preferences": constraints.get("soft_preferences", []),
    }


async def log_unmet_constraints(constraints: dict[str, Any]) -> None:
    """Persist a failed-constraints event. Returns immediately in off mode."""
    settings = get_settings()
    if not settings.analytics_enabled:
        return

    payload = build_payload(constraints)
    try:
        if settings.analytics_backend == "redis":
            await _push_redis(payload)
        elif settings.analytics_backend == "supabase":
            await _push_supabase(payload)
    except Exception as exc:  # analytics must never break the main flow
        logger.warning("Analytics push failed: %s", exc)


def fire_and_forget(constraints: dict[str, Any]) -> None:
    """Schedule the push on the running loop without awaiting it."""
    asyncio.get_running_loop().create_task(log_unmet_constraints(constraints))


async def _push_redis(payload: dict[str, Any]) -> None:
    import redis.asyncio as aioredis  # optional dependency

    settings = get_settings()
    client = aioredis.from_url(settings.redis_url)
    try:
        await client.lpush(settings.analytics_queue_name, json.dumps(payload))
    finally:
        await client.aclose()


async def _push_supabase(payload: dict[str, Any]) -> None:
    import httpx

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_anon_key:
        logger.warning("Supabase analytics configured but SUPABASE_URL/KEY missing")
        return
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{settings.supabase_table}"
    headers = {
        "apikey": settings.supabase_anon_key,
        "Authorization": f"Bearer {settings.supabase_anon_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()

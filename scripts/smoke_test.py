"""End-to-end smoke test for the recommend pipeline.

Runs against the mock LLM (no API keys) with a stubbed catalog (no network).
Exercises both Phase 3 forks: Path A (Vs. Grid) and Path B (pivot + analytics).

Usage:  python scripts/smoke_test.py
"""
import os

os.environ["LLM_MOCK_MODE"] = "true"  # must be set before importing app.config
os.environ["ANALYTICS_BACKEND"] = ""

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.schemas import Product  # noqa: E402
from app.services import catalog  # noqa: E402
import app.main as main_mod  # noqa: E402

# ── Stub catalog (FakeStore-shaped) ───────────────────────────────────────────
SAMPLE_CATALOG = [
    {"id": 1, "title": "MacBook Pro 16", "price": 1099.0, "category": "electronics",
     "description": "flagship pro laptop", "rating": {"rate": 4.8, "count": 100}},
    {"id": 2, "title": "Dell XPS 13 Laptop", "price": 999.0, "category": "electronics",
     "description": "high resolution screen, thin and light, great for coding", "rating": {"rate": 4.7, "count": 200}},
    {"id": 3, "title": "HP Refurbished Laptop 15", "price": 450.0, "category": "electronics",
     "description": "refurbished business laptop with 1080p screen", "rating": {"rate": 3.9, "count": 80}},
    {"id": 4, "title": "Samsung 4K Monitor", "price": 320.0, "category": "electronics",
     "description": "crisp 4K screen for developers and designers", "rating": {"rate": 4.5, "count": 150}},
    {"id": 5, "title": "Gold Ring", "price": 199.0, "category": "jewelery",
     "description": "solid gold band", "rating": {"rate": 4.2, "count": 60}},
    {"id": 6, "title": "Wireless Earbuds", "price": 29.0, "category": "electronics",
     "description": "basic wireless earbuds for everyday listening", "rating": {"rate": 3.5, "count": 300}},
]


async def fake_fetch_products(force_refresh: bool = False) -> list[Product]:
    return [Product(**item) for item in SAMPLE_CATALOG]


catalog.fetch_products = fake_fetch_products  # no network

analytics_calls: list[dict] = []
main_mod.analytics.fire_and_forget = lambda constraints: analytics_calls.append(constraints)

client = TestClient(main_mod.app)
failures: list[str] = []


def check(label: str, cond: bool, extra: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        failures.append(label)
        print(f"  FAIL  {label} {extra}")


# ── Test 1: Path A — success, Vs. Grid ────────────────────────────────────────
print("\n[1] Path A: 'I need a coding laptop under $1000 with a great screen, no refurbished models.'")
r = client.post("/api/recommend", json={"query": "I need a coding laptop under $1000 with a great screen, no refurbished models."})
body = r.json()
check("HTTP 200", r.status_code == 200, str(r.status_code))
check("outcome == success", body.get("outcome") == "success", str(body.get("outcome")))
check("match_1 is Dell XPS (id 2)", body.get("match_1", {}).get("id") == 2, str(body.get("match_1")))
check("match_2 is Samsung 4K (id 4)", body.get("match_2", {}).get("id") == 4, str(body.get("match_2")))
check("math score merged into match_1", isinstance(body.get("match_1", {}).get("similarity_score"), float), str(body))
check("match_1 has 2 pros + 1 con", len(body.get("match_1", {}).get("pros", [])) == 2 and len(body.get("match_1", {}).get("cons", [])) == 1, str(body.get("match_1")))
check("refurbished item (id 3) was filtered out", all(m["id"] != 3 for m in (body.get("match_1"), body.get("match_2"))), str(body))

# ── Test 2: Path B — unmet constraints → pivot + analytics ────────────────────
print("\n[2] Path B: 'wireless earbuds under $20' (nothing that cheap exists)")
r = client.post("/api/recommend", json={"query": "wireless earbuds under $20"})
body = r.json()
check("HTTP 200", r.status_code == 200, str(r.status_code))
check("outcome == unmet_constraint", body.get("outcome") == "unmet_constraint", str(body.get("outcome")))
check("status field present", body.get("status") == "unmet_constraint", str(body))
check("message non-empty", bool(body.get("message")), str(body))
check("suggested_action non-empty", bool(body.get("suggested_action")), str(body))
check("analytics event fired once", len(analytics_calls) == 1, str(analytics_calls))
check("analytics captured max_price=20", analytics_calls and analytics_calls[0]["max_price"] == 20.0, str(analytics_calls))

# ── Test 3: validation + health ───────────────────────────────────────────────
print("\n[3] Validation & health")
r = client.post("/api/recommend", json={"query": ""})
check("empty query -> 422", r.status_code == 422, str(r.status_code))
r = client.get("/health")
check("health ok", r.status_code == 200 and r.json()["status"] == "ok", str(r.text))

print()
if failures:
    print(f"SMOKE TEST FAILED: {len(failures)} check(s) failed -> {failures}")
    sys.exit(1)
print("SMOKE TEST PASSED — both forks behave as the blueprint specifies.")

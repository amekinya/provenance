"""Client for our fine-tuned matcher served on a Nebius Serverless Endpoint (see train/server.py)."""
from __future__ import annotations

import time

import httpx

from app.ai.client import CallRecord, LiveModelUnavailable
from app.config import settings


def available() -> bool:
    return bool(settings.matcher_url)


def score(model: str, pairs: list[tuple[str, str]], calls: list[CallRecord] | None = None) -> list[float]:
    if not available():
        raise LiveModelUnavailable("MATCHER_URL not set")
    headers = {"Authorization": f"Bearer {settings.matcher_token}"} if settings.matcher_token else {}
    out: list[float] = []
    t0 = time.perf_counter()
    with httpx.Client(timeout=300) as c:
        for i in range(0, len(pairs), 2048):
            r = c.post(settings.matcher_url.rstrip("/") + "/score", headers=headers, json={"model": model, "pairs": [list(p) for p in pairs[i:i + 2048]]})
            if r.status_code != 200:
                raise LiveModelUnavailable(f"matcher HTTP {r.status_code}: {r.text[:200]}")
            out += r.json()["scores"]
    if calls is not None:
        # Endpoint time isn't billed per token; record it as a call with GPU-seconds for the ledger.
        calls.append(CallRecord("matcher", f"{model} (Nebius endpoint)", time.perf_counter() - t0, len(pairs), 0, 0, 0.0, "Score sentence-detail pairs"))
    return out

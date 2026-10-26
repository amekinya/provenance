"""Thin wrapper around Nebius Token Factory (OpenAI-compatible API).

Responsibilities:
- one place that talks to the API
- per-call usage + cost accounting (shown in the UI's pipeline view and the eval report)
- a daily spend cap so a public demo can't drain credits
- reasoning on/off for Nemotron models
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.config import ModelSpec, settings


class LiveModelUnavailable(RuntimeError):
    """Raised when there is no API key or the spend cap has been reached."""


@dataclass
class CallRecord:
    role: str
    model_id: str
    seconds: float
    tokens_in: int
    tokens_out: int
    cached_in: int
    cost_usd: float
    purpose: str


@dataclass
class UsageLedger:
    """Tracks spend for the current process (reset daily) and per-request call logs."""

    day: date = field(default_factory=date.today)
    spent_usd: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, cost: float) -> None:
        with self.lock:
            if self.day != date.today():
                self.day, self.spent_usd = date.today(), 0.0
            self.spent_usd += cost

    def remaining(self) -> float:
        with self.lock:
            if self.day != date.today():
                self.day, self.spent_usd = date.today(), 0.0
            return settings.daily_spend_cap_usd - self.spent_usd


ledger = UsageLedger()
_client = None


def _get_client():
    global _client
    if not settings.live_models:
        raise LiveModelUnavailable("NEBIUS_API_KEY is not set")
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=settings.nebius_base_url, api_key=settings.nebius_api_key, timeout=180)
    return _client


def cost_of(spec: ModelSpec, tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * spec.price_in + tokens_out * spec.price_out) / 1_000_000


def chat(
    spec: ModelSpec,
    messages: list[dict[str, Any]],
    *,
    purpose: str,
    json_schema: dict[str, Any] | None = None,
    reasoning: bool = False,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    calls: list[CallRecord] | None = None,
    enforce_cap: bool = True,
) -> str:
    """Run one chat completion and return the text content.

    `calls` (optional) collects a CallRecord so the caller can show what each step cost.
    """
    if enforce_cap and ledger.remaining() <= 0:
        raise LiveModelUnavailable("daily spend cap reached")
    client = _get_client()

    kwargs: dict[str, Any] = {
        "model": spec.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": json_schema, "strict": True},
        }
    # Nemotron models toggle reasoning through the chat template.
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": reasoning}}

    import openai  # noqa: PLC0415

    t0 = time.perf_counter()
    for attempt in range(3):  # rate limits and transient 5xx get two retries with backoff
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            if attempt == 2:
                raise LiveModelUnavailable(f"{type(e).__name__}: {e}") from e
            time.sleep(1.5 * (attempt + 1))
        except openai.APIStatusError as e:
            raise LiveModelUnavailable(f"{type(e).__name__}: {e}") from e
    seconds = time.perf_counter() - t0

    usage = getattr(resp, "usage", None)
    tin = int(getattr(usage, "prompt_tokens", 0) or 0)
    tout = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
    cost = cost_of(spec, tin, tout)
    ledger.add(cost)
    if calls is not None:
        calls.append(CallRecord(spec.role, spec.model_id, seconds, tin, tout, cached, cost, purpose))

    content = resp.choices[0].message.content or ""
    return content


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def chat_json(spec: ModelSpec, messages: list[dict[str, Any]], *, schema: dict[str, Any], **kw) -> dict[str, Any]:
    """Chat with a strict JSON schema; tolerate code fences. An empty or cut-off reply is retried once with
    twice the token budget, then reported as LiveModelUnavailable so the caller's fallback runs."""
    for attempt in range(2):
        text = chat(spec, messages, json_schema=schema, **kw)
        try:
            return _parse_json(text)
        except ValueError:
            kw["max_tokens"] = int(kw.get("max_tokens", 2048)) * 2
    raise LiveModelUnavailable(f"{spec.model_id} returned no valid JSON")

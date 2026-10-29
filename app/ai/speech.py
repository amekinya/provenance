"""A recording -> transcript turns, on our Nebius Serverless Endpoint (train/server.py):
NVIDIA Parakeet TDT 0.6B v3 (words with times) and NVIDIA Nemotron-3-Diarization (who spoke when). Code aligns the two
(app/engine/speech.py); Nemotron 3.5 Lightning only proposes which speaker is the suspect, and the user confirms.
Nothing is stored: the upload lives in a temporary file for the length of the request."""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from app.ai.client import CallRecord, LiveModelUnavailable, chat_json
from app.config import settings
from app.engine import speech as S

MAX_SECONDS = 30 * 60
OFF = "Speech recognition needs our Nebius endpoint, which is off right now. You can still paste a transcript."
_down_until = 0.0


def available() -> bool:
    return bool(settings.matcher_url) and time.monotonic() >= _down_until


def _mark_down() -> None:
    global _down_until
    _down_until = time.monotonic() + 120


def to_audio(src: Path) -> tuple[bytes, float]:
    """Any audio or video file -> 16 kHz mono Ogg Vorbis (small enough to send), and its length in seconds."""
    if not shutil.which("ffmpeg"):
        raise LiveModelUnavailable("ffmpeg isn't installed on this server")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(src)],
                           capture_output=True, text=True, timeout=60)
    try:
        seconds = float(json.loads(probe.stdout)["format"]["duration"])
    except Exception as e:  # noqa: BLE001
        raise ValueError("That file doesn't look like audio or video we can read.") from e
    if seconds > MAX_SECONDS:
        raise ValueError(f"Recordings up to {MAX_SECONDS // 60} minutes for now (this one is {seconds / 60:.0f}).")
    with tempfile.NamedTemporaryFile(suffix=".ogg") as out:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "libvorbis", "-q:a", "3", out.name], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise ValueError("Couldn't find an audio track in that file.")
        return Path(out.name).read_bytes(), seconds


def _post(path: str, payload: dict, timeout: float) -> dict:
    if not available():
        raise LiveModelUnavailable(OFF)
    headers = {"Authorization": f"Bearer {settings.matcher_token}"} if settings.matcher_token else {}
    try:
        r = httpx.post(settings.matcher_url.rstrip("/") + path, json=payload, headers=headers, timeout=httpx.Timeout(timeout, connect=4.0))
    except httpx.HTTPError as e:
        _mark_down()
        raise LiveModelUnavailable(OFF) from e
    if r.status_code in (502, 503, 504):
        _mark_down()
        raise LiveModelUnavailable(OFF)
    if r.status_code != 200:
        raise LiveModelUnavailable(f"speech service HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except Exception as e:  # noqa: BLE001
        raise LiveModelUnavailable(f"unexpected reply from the speech service: {r.text[:120]!r}") from e


def transcribe(audio: bytes, calls: list[CallRecord] | None = None) -> dict:
    """-> {"turns", "speakers" (summary), "audio_seconds", "models"}; speech recognition and diarization run in parallel."""
    b64 = base64.b64encode(audio).decode()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(2) as ex:
        fa = ex.submit(_post, "/asr", {"audio_b64": b64, "timestamps": True}, 600)
        fd = ex.submit(_post, "/diarize", {"audio_b64": b64}, 600)
        a, d = fa.result(), fd.result()
    if not isinstance(a.get("words"), list) or not isinstance(d.get("segments"), list):
        raise LiveModelUnavailable("the speech service didn't return word times and speaker segments")
    if calls is not None:
        calls.append(CallRecord("asr", f"{a.get('model', 'parakeet')} + {d.get('model', 'diarization')} (Nebius endpoint)",
                                time.perf_counter() - t0, 0, 0, 0, 0.0, "Transcribe and split speakers"))
    turns = S.align(a["words"], d["segments"])
    return {"turns": turns, "speakers": S.speaker_summary(turns), "audio_seconds": a.get("audio_seconds"),
            "models": {"speech_recognition": a.get("model"), "diarization": d.get("model")},
            "endpoint_seconds": {"asr": a.get("seconds"), "diarize": d.get("seconds")}}


ROLE_SCHEMA = {"type": "object", "properties": {"speakers": {"type": "array", "items": {"type": "object", "properties": {
    "speaker": {"type": "string"}, "role": {"type": "string", "enum": ["investigator", "suspect", "other"]},
    "why": {"type": "string", "description": "a few words, from what they say"}},
    "required": ["speaker", "role", "why"], "additionalProperties": False}}},
    "required": ["speakers"], "additionalProperties": False}


def propose_roles(turns: list[dict], calls: list[CallRecord] | None = None) -> dict:
    """Nemotron 3.5 Lightning proposes each speaker's role from what they say; code checks the reply and falls back to
    a simple rule. -> {"by": "lightning"|"code", "roles": {speaker: {"role", "why"}}}"""
    summary = S.speaker_summary(turns)
    fallback = {"by": "code", "roles": S.propose_roles_by_code(summary)}
    if not settings.live_models or len(summary) < 2:
        return fallback
    lines = []
    for t in turns[:60]:
        lines.append(f"{t['speaker']}: {t['text'][:220]}")
    try:
        out = chat_json(settings.lightning, [
            {"role": "system", "content": "You read the opening of a police interview transcript whose speakers are numbered. "
                                          "Say which speaker is the person being questioned (suspect), which are investigators "
                                          "(officers, detectives, agents asking the questions) and which are anyone else. "
                                          "Judge only from what they say. This is a proposal a person will check."},
            {"role": "user", "content": "\n".join(lines)}],
            schema=ROLE_SCHEMA, purpose="Propose speaker roles", reasoning=False, max_tokens=600, calls=calls)
    except LiveModelUnavailable:
        return fallback
    known = {s["speaker"] for s in summary}
    roles = {x["speaker"].strip().upper(): {"role": x["role"], "why": x["why"][:120]} for x in out.get("speakers", [])
             if x.get("speaker", "").strip().upper() in known}
    if set(roles) != known or sum(r["role"] == "suspect" for r in roles.values()) != 1:
        return fallback  # a proposal that misses a speaker, or names no suspect or two, isn't used
    return {"by": "lightning", "roles": roles}

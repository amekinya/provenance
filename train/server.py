"""Our models on a Nebius Serverless Endpoint (one L40S): NVIDIA speech recognition + our fine-tuned matchers.

Runs on a stock PyTorch image; Nebius injects this file and the container command pip-installs what it needs.
  GET  /health                 -> which models are loaded
  POST /asr     {audio_b64}    -> {"text": ...}   NVIDIA Parakeet TDT 0.6B v3 (25 languages incl. English, Spanish)
  POST /score   {model, pairs: [[query, passage], ...]} -> {"scores": [...]}  a fine-tuned NVIDIA reranker
Fine-tuned weights live in the mounted bucket at /models/<name>; if missing (e.g. "base-reranker"), the base reranker is
used and reported. PRELOAD (comma list, e.g. "asr,faithful-checker,provenance-matcher") loads models at startup so the
first request isn't slow. Auth is handled by the endpoint's bearer token.
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_RERANKER = "nvidia/llama-nemotron-rerank-1b-v2"
ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="nemotron-tools")
_asr = None
_rerankers: dict[str, tuple] = {}


def asr():
    global _asr
    if _asr is None:
        from transformers import pipeline

        _asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=0 if DEVICE == "cuda" else -1,
                        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)
    return _asr


def reranker(name: str):
    if name not in _rerankers:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = MODELS_DIR / name
        src = str(path) if (path / "config.json").exists() else BASE_RERANKER
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(src, trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32).to(DEVICE).eval()
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.eos_token_id
        _rerankers[name] = (tok, model, src)
    return _rerankers[name]


def template(q: str, p: str) -> str:
    """The reranker's own prompt format (model card)."""
    return f"question:{q} \n \n passage:{p}"


@app.on_event("startup")
def _preload() -> None:
    import threading

    def go():
        for name in filter(None, (x.strip() for x in os.environ.get("PRELOAD", "").split(","))):
            try:
                asr() if name == "asr" else reranker(name)
                print("preloaded", name, flush=True)
            except Exception as e:  # noqa: BLE001
                print("preload failed", name, e, flush=True)
    threading.Thread(target=go, daemon=True).start()


@app.get("/health")
def health():
    return {"device": DEVICE, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
            "asr_loaded": _asr is not None, "rerankers": {k: v[2] for k, v in _rerankers.items()},
            "available": sorted(p.name for p in MODELS_DIR.glob("*") if (p / "config.json").exists()) if MODELS_DIR.exists() else []}


class AsrIn(BaseModel):
    audio_b64: str


@app.post("/asr")
def transcribe(body: AsrIn):
    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(io.BytesIO(base64.b64decode(body.audio_b64)), dtype="float32")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Couldn't read audio: {e}") from e
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    t0 = time.perf_counter()
    out = asr()({"raw": np.asarray(data), "sampling_rate": 16000})
    return {"text": out["text"].strip(), "seconds": round(time.perf_counter() - t0, 3), "audio_seconds": round(len(data) / 16000, 2),
            "model": ASR_MODEL}


class ScoreIn(BaseModel):
    model: str
    pairs: list[list[str]]


@app.post("/score")
def score(body: ScoreIn):
    if not body.pairs or len(body.pairs) > 4096:
        raise HTTPException(400, "Send 1-4096 pairs.")
    tok, model, src = reranker(body.model)
    t0 = time.perf_counter()
    scores: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(body.pairs), 64):
            batch = body.pairs[i:i + 64]
            enc = tok([template(q, p) for q, p in batch], padding=True, truncation=True, max_length=320, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits.float().view(-1)
            scores += torch.sigmoid(logits).tolist()
    return {"scores": [round(s, 4) for s in scores], "model": src, "seconds": round(time.perf_counter() - t0, 3)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

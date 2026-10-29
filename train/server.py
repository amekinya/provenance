"""Our models on a Nebius Serverless Endpoint (one L40S): NVIDIA speech recognition + our fine-tuned matchers.

Runs on a stock PyTorch image; Nebius injects this file and the container command pip-installs what it needs.
  GET  /health                 -> which models are loaded
  POST /asr     {audio_b64}    -> {"text": ...}   NVIDIA Parakeet TDT 0.6B v3 (25 languages incl. English, Spanish)
  POST /asr     {audio_b64, timestamps: true} -> {"text", "words": [{"w", "start", "end"}]}  long audio (up to 30 min),
                                  transcribed in chunks cut at quiet points, with word times in seconds
  POST /diarize {audio_b64}    -> {"segments": [{"speaker", "start", "end"}]}  NVIDIA Nemotron-3-Diarization (who spoke when,
                                  up to 8 speakers, numbered in order of first appearance)
  POST /score   {model, pairs: [[query, passage], ...]} -> {"scores": [...]}  a fine-tuned NVIDIA reranker
Fine-tuned weights live in the mounted bucket at /models/<name>; if missing (e.g. "base-reranker"), the base reranker is
used and reported. PRELOAD (comma list, e.g. "asr,faithful-checker,provenance-matcher") loads models at startup so the
first request isn't slow. Auth is handled by the endpoint's bearer token.
"""
from __future__ import annotations

import base64
import json
import io
import os
import re
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_RERANKER = "nvidia/llama-nemotron-rerank-1b-v2"
ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DIAR_MODEL = "nvidia/Nemotron-3-Diarization"
LONG_MAX_SECONDS = 30 * 60      # recordings for /asr with timestamps and /diarize
LONG_MAX_B64 = 48_000_000       # about 30 minutes of compressed 16 kHz mono audio
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="nemotron-tools")
_asr = None
_tdt = None   # (processor, model) for word timestamps; the model is shared with the pipeline
_diar = None  # (processor, model)
_rerankers: dict[str, tuple] = {}


def asr():
    global _asr
    with _lock:
        if _asr is None:
            from transformers import pipeline

            _asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=0 if DEVICE == "cuda" else -1,
                            dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)
    return _asr


def tdt():
    """Parakeet with its processor, for word timestamps (the pipeline doesn't return them for TDT models)."""
    global _tdt
    pipe = asr()
    with _lock:
        if _tdt is None:
            from transformers import AutoProcessor

            _tdt = (AutoProcessor.from_pretrained(ASR_MODEL), pipe.model)
    return _tdt


def diarizer():
    global _diar
    with _lock:
        if _diar is None:
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor

            _diar = (AutoProcessor.from_pretrained(DIAR_MODEL),
                     AutoModelForAudioFrameClassification.from_pretrained(DIAR_MODEL).to(DEVICE).eval())
    return _diar


_lock = threading.Lock()
NAMES = set(filter(None, os.environ.get("MODEL_NAMES", "faithful-checker,provenance-matcher,secondlook-matcher,base-reranker").split(",")))


def model_dir(name: str) -> Path | None:
    """Where a fine-tuned model lives. The bucket mount can't rename or overwrite files, so each training run copies its
    finished model into a fresh folder <name>@<time> and writes DONE last; the newest finished folder wins. A plain
    <name> folder with weights (older layout) is the fallback."""
    runs = sorted(p for p in MODELS_DIR.glob(f"{name}@*") if (p / "DONE").exists() and (p / "config.json").exists())
    if runs:
        return runs[-1]
    plain = MODELS_DIR / name
    if (plain / "config.json").exists() and any(plain.glob("*.safetensors")):
        return plain
    return None


def reranker(name: str):
    """Load a reranker once (thread-safe). Only names on the allow-list; only 'base-reranker' means the untuned model.
    A fine-tuned name whose weights are missing is an error, never a silent fallback to the base model."""
    if name not in NAMES or not re.fullmatch(r"[a-z0-9-]+", name):
        raise HTTPException(404, f"Unknown model {name!r}")
    with _lock:
        if name in _rerankers:
            return _rerankers[name]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = model_dir(name)
        if name == "base-reranker":
            src = BASE_RERANKER
        elif path is not None:
            src = str(path)
        else:
            raise HTTPException(503, f"Fine-tuned weights for {name!r} aren't in {MODELS_DIR} yet")
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(src, trust_remote_code=True,
                                                                   dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32).to(DEVICE).eval()
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.eos_token_id
        _rerankers[name] = (tok, model, src)
    return _rerankers[name]


def template(q: str, p: str) -> str:
    """The reranker's own prompt format (model card)."""
    return f"question:{q} \n \n passage:{p}"


@app.on_event("startup")  # noqa: (deprecated hook, still supported; keeps this file self-contained)
def _preload() -> None:
    import threading

    def go():
        names = [x.strip() for x in os.environ.get("PRELOAD", "").split(",") if x.strip()]
        if os.environ.get("PRELOAD_DIARIZER", "1") == "1" and "diarizer" not in names:
            names.append("diarizer")  # small (0.4 GB); loaded so a recording upload isn't slowed by a download
        for name in names:
            try:
                asr() if name == "asr" else diarizer() if name == "diarizer" else reranker(name)
                print("preloaded", name, flush=True)
            except Exception as e:  # noqa: BLE001
                print("preload failed", name, e, flush=True)
    threading.Thread(target=go, daemon=True).start()


@app.get("/health")
def health():
    return {"device": DEVICE, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
            "asr_loaded": _asr is not None, "diarizer_loaded": _diar is not None, "rerankers": {k: v[2] for k, v in _rerankers.items()},
            "available": {n: model_dir(n).name for n in sorted(NAMES) if n != "base-reranker" and model_dir(n)} if MODELS_DIR.exists() else {}}


@app.get("/metrics/{name}")
def metrics(name: str):
    """The training job's report (metrics.json next to the weights), so results can be published without bucket access."""
    if name not in NAMES or name == "base-reranker":
        raise HTTPException(404, "unknown model")
    d = model_dir(name)
    f = d / "metrics.json" if d else MODELS_DIR / name / "metrics.json"
    if not f.exists():
        raise HTTPException(404, "no training report yet")
    return json.loads(f.read_text())


class AsrIn(BaseModel):
    audio_b64: str
    timestamps: bool = False


def _read_audio(b64: str, max_b64: int, max_seconds: int):
    """base64 audio (wav, flac, ogg or mp3) -> mono float32 at 16 kHz."""
    import numpy as np
    import soundfile as sf

    if len(b64) > max_b64:
        raise HTTPException(413, "Audio too large.")
    try:
        data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Couldn't read audio: {e}") from e
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) > max_seconds * sr:
        raise HTTPException(413, f"Audio longer than {max_seconds // 60 if max_seconds >= 120 else max_seconds} "
                                 f"{'minutes' if max_seconds >= 120 else 'seconds'}.")
    if sr != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return np.asarray(data, dtype=np.float32)


def _cut_points(data, sr: int = 16000, target: float = 16.0, window: float = 4.0) -> list[int]:
    """Chunk boundaries near every `target` seconds, each moved to the quietest 50 ms within +-window/2 seconds, so a
    word is rarely cut in half. 16 s had the lowest word error on a separate dev recording (Faithful's voiced turns,
    3.4 min, each chunk padded with silence as below): 16 s 5.9%, 28 s 6.3%."""
    import numpy as np

    n, hop = len(data), int(0.05 * sr)
    cuts, pos = [0], 0
    while n - pos > int((target + window) * sr):
        lo, hi = pos + int((target - window / 2) * sr), pos + int((target + window / 2) * sr)
        seg = data[lo:hi]
        frames = len(seg) // hop
        energy = (seg[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1)
        pos = lo + int(np.argmin(energy)) * hop + hop // 2
        cuts.append(pos)
    cuts.append(n)
    return cuts


def _words(tokens: list[dict], offset: float) -> list[dict]:
    """Parakeet sub-word tokens with times -> words. A token starting with a space (or the first one) starts a word."""
    out: list[dict] = []
    for t in tokens:
        piece = t["token"]
        if not piece.strip():
            continue
        if piece.startswith(" ") or not out:
            out.append({"w": piece.strip(), "start": round(max(0.0, t["start"] + offset), 2), "end": round(max(0.0, t["end"] + offset), 2)})
        else:
            out[-1]["w"] += piece
            out[-1]["end"] = round(max(out[-1]["end"], t["end"] + offset), 2)
    return out


PAD_SECONDS = 0.5  # silence added around each piece: without it Parakeet sometimes skipped a whole sentence mid-piece
NOISE_FLOOR = 1e-4  # -80 dBFS, far below any microphone's own noise. Exact digital silence (synthetic speech, noise gates,
# some call recorders) sometimes made Parakeet emit nothing for a whole chunk of clear speech.


def _transcribe_pieces(data, pieces: list[tuple[int, int]], pad_seconds: float = PAD_SECONDS) -> list[list[dict]]:
    """Each (start, end) sample range -> its words with times in the whole recording, batched 8 at a time."""
    import numpy as np

    proc, model = tdt()
    pad = np.zeros(int(pad_seconds * 16000), dtype=np.float32)
    out_words: list[list[dict]] = []
    for b in range(0, len(pieces), 8):
        batch = pieces[b:b + 8]
        inputs = proc([np.concatenate([pad, data[a:z], pad]) for a, z in batch], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(DEVICE, dtype=model.dtype) if DEVICE == "cuda" else inputs
        with torch.inference_mode():
            out = model.generate(**inputs, return_dict_in_generate=True)
        for i, (a, _) in enumerate(batch):
            _, stamps = proc.decode(out.sequences[i:i + 1], durations=out.durations[i:i + 1], skip_special_tokens=True)
            out_words.append(_words(stamps[0], a / 16000 - pad_seconds))
    return out_words


def _voiced_frames(seg):
    """Per 50 ms frame: louder than -40 dBFS? A rough measure of where someone is speaking."""
    import numpy as np

    hop = 800
    frames = len(seg) // hop
    if not frames:
        return np.zeros(0, dtype=bool)
    return np.sqrt((seg[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1)) > 0.01


def _voiced_seconds(seg) -> float:
    return float(_voiced_frames(seg).sum()) * 0.05


def _uncovered_speech(data, words: list[dict], min_voiced: float = 0.6) -> list[tuple[int, int]]:
    """Stretches of voiced audio with no word over them (words widened by 0.3 s); pauses under 1 s don't split a
    stretch. -> sample ranges (widened by 0.25 s) holding at least `min_voiced` seconds of voice."""
    import numpy as np

    voiced = _voiced_frames(data)
    covered = np.zeros(len(voiced), dtype=bool)
    for w in words:
        covered[max(0, int((w["start"] - 0.3) / 0.05)): int((w["end"] + 0.3) / 0.05) + 1] = True
    gaps, start, last, count = [], None, None, 0
    for i in np.flatnonzero(voiced & ~covered):
        if start is not None and i - last > 20:
            gaps.append((start, last, count))
            start, count = None, 0
        start = i if start is None else start
        last, count = i, count + 1
    if start is not None:
        gaps.append((start, last, count))
    return [(max(0, (a - 5) * 800), min(len(data), (z + 6) * 800)) for a, z, c in gaps if c * 0.05 >= min_voiced]


def _transcribe_long(data) -> tuple[str, list[dict]]:
    """Whole recording -> words with times, in chunks cut at quiet points. Two safety nets, because Parakeet sometimes
    skipped stretches of clear (synthetic) speech: a chunk with seconds of voice but almost no words is transcribed again
    with more silence around it; then any stretch of voice still without words is transcribed on its own, and the words
    found there that don't overlap existing ones are added."""
    import numpy as np

    data = (data + NOISE_FLOOR * np.random.default_rng(0).standard_normal(len(data))).astype(np.float32)
    cuts = _cut_points(data)
    pieces = list(zip(cuts, cuts[1:]))
    per_piece = _transcribe_pieces(data, pieces)
    for k, (a, z) in enumerate(pieces):
        voiced = _voiced_seconds(data[a:z])
        if voiced >= 2.0 and len(per_piece[k]) < voiced / 3:
            again = _transcribe_pieces(data, [(a, z)], pad_seconds=1.0)[0]
            if len(again) > len(per_piece[k]):
                per_piece[k] = again
    words = [w for ws in per_piece for w in ws]
    gaps = _uncovered_speech(data, words)
    for found in _transcribe_pieces(data, gaps) if gaps else []:
        for w in found:  # keep words that don't overlap one we already have
            if not any(min(w["end"], o["end"]) - max(w["start"], o["start"]) > 0.05 for o in words):
                words.append(w)
    words.sort(key=lambda w: w["start"])
    return " ".join(w["w"] for w in words), words


@app.post("/asr")
def transcribe(body: AsrIn):
    t0 = time.perf_counter()
    if body.timestamps:
        data = _read_audio(body.audio_b64, LONG_MAX_B64, LONG_MAX_SECONDS)
        text, words = _transcribe_long(data)
        return {"text": text, "words": words, "seconds": round(time.perf_counter() - t0, 3), "audio_seconds": round(len(data) / 16000, 2),
                "model": ASR_MODEL}
    data = _read_audio(body.audio_b64, 8_000_000, 90)  # the short-clip path, unchanged (Faithful's per-turn audio)
    t0 = time.perf_counter()
    out = asr()({"raw": data, "sampling_rate": 16000})
    return {"text": out["text"].strip(), "seconds": round(time.perf_counter() - t0, 3), "audio_seconds": round(len(data) / 16000, 2),
            "model": ASR_MODEL}


class DiarizeIn(BaseModel):
    audio_b64: str


@app.post("/diarize")
def diarize(body: DiarizeIn):
    data = _read_audio(body.audio_b64, LONG_MAX_B64, LONG_MAX_SECONDS)
    proc, model = diarizer()
    t0 = time.perf_counter()
    inputs = proc(data, sampling_rate=16000).to(DEVICE)
    with torch.inference_mode():
        logits = model(**inputs).logits
    segs = proc.extract_speaker_dict(logits.float(), inputs.attention_mask)[0]
    segments = [{"speaker": int(x["Speaker"]), "start": round(float(x["Start"]), 2), "end": round(float(x["End"]), 2)} for x in segs]
    return {"segments": segments, "speakers": len({x["speaker"] for x in segments}), "seconds": round(time.perf_counter() - t0, 3),
            "audio_seconds": round(len(data) / 16000, 2), "model": DIAR_MODEL}


class ScoreIn(BaseModel):
    model: str
    pairs: list[list[str]]


@app.post("/score")
def score(body: ScoreIn):
    if not body.pairs or len(body.pairs) > 4096:
        raise HTTPException(400, "Send 1-4096 pairs.")
    if any(len(p) != 2 or len(p[0]) + len(p[1]) > 6000 for p in body.pairs):
        raise HTTPException(400, "Each pair is [query, passage], under 6000 characters together.")
    tok, model, src = reranker(body.model)
    t0 = time.perf_counter()
    scores: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(body.pairs), 64):
            batch = body.pairs[i:i + 64]
            enc = tok([template(q, p) for q, p in batch], padding=True, truncation=True, max_length=320, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits.float().view(-1)
            scores += torch.sigmoid(logits).tolist()
    return {"scores": [round(s, 4) for s in scores], "model": src, "fine_tuned": src != BASE_RERANKER, "name": body.model,
            "seconds": round(time.perf_counter() - t0, 3)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

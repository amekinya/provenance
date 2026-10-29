"""Measure the recording path on the synthetic test interrogation (eval/audio_test: script.json, truth.json,
interrogation.mp3, made by scripts/make_test_audio.py).

Runs the same server code the Nebius endpoint runs (train/server.py, loaded locally) behind the app's own client and
alignment code, then scores:
  - word error rate against the script (lower case, punctuation removed)
  - speaker attribution: share of script words given to the right speaker, after mapping each numbered speaker to the
    script speaker it mostly covers
  - turns: how many turns came out, and how many script turns were kept whole under one speaker
Writes eval/audio_test/results.json and eval/audio_test/transcript.txt.
Usage: python -m scripts.audio_eval [--name hard] [--server path/to/server.py]   (needs torch, transformers>=5.21, soundfile, librosa)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "audio_test"


def norm(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def align_words(ref: list[str], hyp: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment -> pairs (ref index or None, hyp index or None)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1], d[n][m]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=str(ROOT / "train" / "server.py"))
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    suf = f"_{a.name}" if a.name else ""
    os.environ.setdefault("MODELS_DIR", "/nonexistent")
    os.environ["PRELOAD"] = ""
    os.environ["PRELOAD_DIARIZER"] = "0"
    spec = importlib.util.spec_from_file_location("server", a.server)
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    from fastapi.testclient import TestClient

    from app.ai import speech

    client = TestClient(srv.app)

    def local_post(path: str, payload: dict, timeout: float) -> dict:  # the endpoint, run in this process
        r = client.post(path, json=payload)
        r.raise_for_status()
        return r.json()

    speech._post = local_post
    truth = json.loads((OUT / f"truth{suf}.json").read_text())
    t0 = time.perf_counter()
    audio, seconds = speech.to_audio(OUT / f"interrogation{suf}.mp3")
    out = speech.transcribe(audio)
    wall = time.perf_counter() - t0

    ref, ref_spk = [], []
    for t in truth["turns"]:
        for w in norm(t["text"]):
            ref.append(w); ref_spk.append(t["speaker"])
    hyp, hyp_spk = [], []
    for t in out["turns"]:
        for w in norm(t["text"]):
            hyp.append(w); hyp_spk.append(t["speaker"])
    pairs, dist = align_words(ref, hyp)
    wer = dist / max(1, len(ref))
    matched = [(i, j) for i, j in pairs if i is not None and j is not None]
    votes: dict[str, dict[str, int]] = {}
    for i, j in matched:
        votes.setdefault(hyp_spk[j], {}).setdefault(ref_spk[i], 0)
        votes[hyp_spk[j]][ref_spk[i]] += 1
    mapping = {h: max(v, key=v.get) for h, v in votes.items()}
    right = sum(1 for i, j in matched if mapping.get(hyp_spk[j]) == ref_spk[i])
    # script turns kept whole: every matched word of the turn under one numbered speaker that maps to the right person
    by_turn: dict[int, set] = {}
    turn_of = []
    for ti, t in enumerate(truth["turns"]):
        for _ in norm(t["text"]):
            turn_of.append(ti)
    for i, j in matched:
        by_turn.setdefault(turn_of[i], set()).add(mapping.get(hyp_spk[j]) == ref_spk[i] and hyp_spk[j])
    whole = sum(1 for ti, labels in by_turn.items() if len(labels) == 1 and next(iter(labels)))
    res = {
        "audio_seconds": truth["audio_seconds"], "script_turns": len(truth["turns"]), "script_words": len(ref),
        "speakers_in_script": len({t["speaker"] for t in truth["turns"]}), "speakers_found": len({t["speaker"] for t in out["turns"]}),
        "turns_found": len(out["turns"]), "word_error_rate": round(wer, 4),
        "speaker_attribution": round(right / max(1, len(matched)), 4), "script_turns_kept_whole": f"{whole}/{len(truth['turns'])}",
        "speaker_mapping": mapping, "wall_seconds_cpu": round(wall, 1), "endpoint_seconds_cpu": out.get("endpoint_seconds"),
        "models": out.get("models"), "roles_by_code": speech.S.propose_roles_by_code(out["speakers"]),
        "note": "Synthetic voices (Kokoro-82M), two speakers, no background noise: an easy case. Real interrogation audio "
                "(one room microphone, cross-talk, a poor recording) will do worse; this checks that the path works end to end.",
    }
    (OUT / f"results{suf}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"transcript{suf}.txt").write_text("\n".join(f"{t['speaker']}: {t['text']}" for t in out["turns"]) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""Voice the synthetic test interrogation (eval/audio_test/script.json) with Kokoro-82M, so the recording path can be
measured against a script whose words, speakers and turn times are known exactly.

Text-to-speech only MAKES test audio; Provenance never speaks. Writes eval/audio_test/interrogation.mp3 and
eval/audio_test/truth.json (each turn's speaker, text, start and end in seconds).
Usage (Python 3.12 with kokoro, soundfile; ffmpeg and espeak-ng installed):
  python scripts/make_test_audio.py                      (script.json -> interrogation.mp3, truth.json)
  python scripts/make_test_audio.py --name hard          (script_hard.json -> interrogation_hard.mp3, truth_hard.json)
"""
from __future__ import annotations

import json
import random
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "audio_test"


def main() -> None:
    import argparse

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    name = ap.parse_args().name
    suf = f"_{name}" if name else ""
    script = json.loads((OUT / f"script{suf}.json").read_text())
    pipe = KPipeline(lang_code="a")
    sr = 24000
    rng = random.Random(7)
    audio, truth, t = [], [], 0.0
    lead = np.zeros(int(0.5 * sr), dtype=np.float32)
    audio.append(lead)
    t += len(lead) / sr
    for speaker, text in script["turns"]:
        chunks = [np.asarray(a, dtype=np.float32) for _, _, a in pipe(text, voice=script["voices"][speaker], speed=1.0)]
        clip = np.concatenate(chunks)
        truth.append({"speaker": speaker, "text": text, "start": round(t, 3), "end": round(t + len(clip) / sr, 3)})
        audio.append(clip)
        t += len(clip) / sr
        gap = np.zeros(int(rng.uniform(0.45, 1.1) * sr), dtype=np.float32)  # a natural pause between turns
        audio.append(gap)
        t += len(gap) / sr
    full = np.concatenate(audio)
    with tempfile.NamedTemporaryFile(suffix=".wav") as w:
        sf.write(w.name, full, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w.name, "-ac", "1", "-b:a", "64k", str(OUT / f"interrogation{suf}.mp3")], check=True)
    (OUT / f"truth{suf}.json").write_text(json.dumps({"audio_seconds": round(len(full) / sr, 2), "turns": truth}, indent=1))
    print(f"{len(truth)} turns, {len(full) / sr:.1f} s -> {OUT / f'interrogation{suf}.mp3'}")


if __name__ == "__main__":
    main()

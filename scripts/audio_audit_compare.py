"""Does the audit come out the same from the recording as from the script it was voiced from?

Sends two transcripts of the synthetic test interrogation to a running Provenance (/api/analyze), with the same facts:
  1. the script itself ("DET: ..." / "SUSPECT: ...")
  2. the transcript the recording path produced (eval/audio_test/transcript*.txt), with speakers named the way the
     "Who is who?" step names them from the proposed roles (SUSPECT, DET 1, ...)
and compares each fact's origin, corroboration and first mentions. Writes eval/audio_test/audit_compare*.json.
The audit's tagging runs on Nemotron 3 Ultra via Token Factory, so each run costs a few cents.
Usage: python -m scripts.audio_audit_compare [--name hard] [--base https://provenance-2401.onrender.com]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "audio_test"


def ui_names(roles: dict[str, dict]) -> dict[str, str]:
    """The names app.js gives speakers once their roles are confirmed."""
    names, inv, oth = {}, 0, 0
    for spk in sorted(roles, key=lambda s: int(s.split()[-1])):
        role = roles[spk]["role"]
        if role == "suspect":
            names[spk] = "SUSPECT"
        elif role == "investigator":
            inv += 1
            names[spk] = f"DET {inv}"
        else:
            oth += 1
            names[spk] = f"OTHER {oth}"
    return names


def summary(rep: dict) -> dict:
    out = {}
    for f in rep["facts"]:
        first_s = f.get("first_suspect") or {}
        first_i = f.get("first_interrogator") or {}
        out[f["fact"]["label"]] = {"origin": f["origin"], "corroboration": f["corroboration"],
                                   "first_suspect_quote": (first_s.get("quote") or "")[:90],
                                   "first_investigator_quote": (first_i.get("quote") or "")[:90]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--base", default="https://provenance-2401.onrender.com")
    a = ap.parse_args()
    suf = f"_{a.name}" if a.name else ""
    script = json.loads((OUT / f"script{suf}.json").read_text())
    res = json.loads((OUT / f"results{suf}.json").read_text())
    facts = "\n".join(f["label"] for f in script["facts"])
    investigators = sorted({s for s, _ in script["turns"] if s != "SUSPECT"})
    script_text = "\n".join(f"{s}: {t}" for s, t in script["turns"])
    names = ui_names(res["roles_by_code"])
    asr_lines = []
    for line in (OUT / f"transcript{suf}.txt").read_text().splitlines():
        m = re.match(r"(SPEAKER \d+): (.*)", line)
        if m:
            asr_lines.append(f"{names[m.group(1)]}: {m.group(2)}")
    runs = {}
    for label, text, inv in (("script", script_text, ", ".join(investigators)),
                             ("recording", "\n".join(asr_lines), ", ".join(n for n in names.values() if n.startswith("DET")))):
        r = httpx.post(a.base.rstrip("/") + "/api/analyze", json={"transcript": text, "facts": facts, "suspect": "SUSPECT",
                                                                  "interrogators": inv}, timeout=600)
        r.raise_for_status()
        rep = r.json()
        runs[label] = {"headline": rep.get("headline"), "counts": rep.get("counts"), "facts": summary(rep),
                       "cost_usd": rep["run"]["cost_usd"], "seconds": rep["run"]["total_seconds"]}
    same = {k: runs["script"]["facts"][k]["origin"] == runs["recording"]["facts"].get(k, {}).get("origin") and
            runs["script"]["facts"][k]["corroboration"] == runs["recording"]["facts"].get(k, {}).get("corroboration")
            for k in runs["script"]["facts"]}
    out = {"facts_with_same_origin_and_corroboration": f"{sum(same.values())}/{len(same)}", "same_counts": runs["script"]["counts"] == runs["recording"]["counts"],
           "differences": [k for k, v in same.items() if not v], "speaker_names_used": names, "runs": runs}
    (OUT / f"audit_compare{suf}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    for k in runs["script"]["facts"]:
        print(f"  {k[:45]:45s} script={runs['script']['facts'][k]['origin']:22s} recording={runs['recording']['facts'].get(k, {}).get('origin')}")
    print("cost", round(runs["script"]["cost_usd"] + runs["recording"]["cost_usd"], 4), "USD")


if __name__ == "__main__":
    main()

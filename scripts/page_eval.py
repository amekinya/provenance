"""How well is a scanned page read? Compares Nemotron 3.5 Super VL's reading and plain Tesseract OCR with hand
transcriptions (data/cases/<case>/gold_pages.json).

Two numbers per reader: word error rate (dotted 'inaudible' runs and punctuation ignored) and speaker attribution
(the share of correctly read words that were credited to the right speaker, which is what the ledger depends on).
Super VL readings come from the ingest cache (pages/<session>/<file>.json); Tesseract is run here.
Usage: python -m scripts.page_eval dassey   (needs poppler-utils and tesseract-ocr)
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.ingest import _canon, case_dir


def words_with_speaker(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out = []
    for spk, text in turns:
        text = re.sub(r"\([^)]*\)", " ", text)  # stage directions
        for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'")):
            out.append((w, spk))
    return out


def score(gold: list[tuple[str, str]], pred: list[tuple[str, str]]) -> dict:
    g, p = words_with_speaker(gold), words_with_speaker(pred)
    sm = difflib.SequenceMatcher(a=[w for w, _ in g], b=[w for w, _ in p], autojunk=False)
    matched = right_spk = 0
    edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                matched += 1
                right_spk += g[i1 + k][1] == p[j1 + k][1]
        else:
            edits += max(i2 - i1, j2 - j1)
    return {"gold_words": len(g), "wer": round(edits / max(len(g), 1), 4), "speaker_attribution": round(right_spk / max(matched, 1), 4)}


def turns_from_page(pg: dict, roster: dict[str, str]) -> list[tuple[str, str]]:
    turns: list[list[str]] = []
    for ln in pg.get("lines", []):
        if ln.get("kind") == "header" or not ln.get("text", "").strip():
            continue
        name, _ = _canon(ln.get("speaker", ""), roster)
        if not name and turns:
            turns[-1][1] += " " + ln["text"]
        else:
            turns.append([name or "UNKNOWN", ln["text"]])
    return [tuple(t) for t in turns]


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else "dassey"
    cd = case_dir(case)
    cfg = json.loads((cd / "case.json").read_text())
    roster = cfg["roster"]
    canon_gold = lambda turns: [(_canon(s, roster)[0], t) for s, t in turns]  # noqa: E731
    gold = json.loads((cd / "gold_pages.json").read_text())["pages"]
    from scripts.ingest_ocr import parse_page  # noqa: PLC0415

    rows, tot = [], {"super_vl": [0, 0.0, 0.0], "tesseract": [0, 0.0, 0.0]}
    pdf_of = {s["id"]: cd / s["pdf"] for s in cfg["sessions"] if s.get("pdf")}
    for gp in gold:
        g = canon_gold(gp["turns"])
        row = {"session": gp["session"], "page": gp["page_label"]}
        vl = cd / "pages" / gp["session"] / f"{gp['file']}.json"
        if vl.exists():
            row["super_vl"] = score(g, turns_from_page(json.loads(vl.read_text()), roster))
        n = int(gp["file"].split("-")[1])
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-r", "200", "-gray", "-png", "-f", str(n), "-l", str(n), str(pdf_of[gp["session"]]), f"{tmp}/p"], check=True)
            img = next(Path(tmp).glob("p-*.png"))
            txt = subprocess.run(["tesseract", str(img), "-", "--psm", "6"], capture_output=True, text=True, check=True).stdout
        row["tesseract"] = score(g, turns_from_page(parse_page(txt, roster), roster))
        for k in tot:
            if k in row:
                w = row[k]["gold_words"]
                tot[k][0] += w; tot[k][1] += row[k]["wer"] * w; tot[k][2] += row[k]["speaker_attribution"] * w
        rows.append(row)
        print(row)
    summary = {k: {"words": n, "wer": round(e / n, 4), "speaker_attribution": round(s / n, 4)} for k, (n, e, s) in tot.items() if n}
    out = {"pages": rows, "overall": summary, "note": "Word error rate and speaker attribution against hand transcriptions; lower WER is better."}
    (cd / "page_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

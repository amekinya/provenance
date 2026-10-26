"""Plain-OCR reading of the transcripts (Tesseract, no model), in the same page format Super VL produces.

Two uses: a no-model baseline to measure Super VL's page reading against (speaker-turn agreement), and a fast local
reading for development. Usage: python -m scripts.ingest_ocr dassey [--out utterances_ocr.json] [--text-dir DIR]
With --text-dir, reads existing '<pdf stem>.ocr.txt' files ('### p-NN' page markers) instead of running Tesseract.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from scripts.ingest import _canon, case_dir, merge, render

LABEL = re.compile(r"^[\W_]{0,4}([A-Z][A-Za-z'\-]{2,15})\s*[:;]\s*(.*)$")
HEADER = re.compile(r"(CALUMET COUNTY|SHERIFF'?S DEPARTMENT|Complaint No|File Number|Interview of|Reporting Officer|^\W*Page\W*$|^\W*\d{1,4}\W*$)", re.I)
NOTE = re.compile(r"^\W*[\(\[](.{1,80})[\)\]]\W*$")


def parse_page(text: str, roster: dict[str, str]) -> dict:
    lines, label = [], ""
    for raw in text.splitlines():
        t = raw.strip()
        if not t:
            continue
        if re.fullmatch(r"\W*\d{1,4}\W*", t) and not label:
            label = re.sub(r"\D", "", t)
        if HEADER.search(t):
            lines.append({"speaker": "", "text": t, "kind": "header"})
            continue
        m = LABEL.match(t)
        if m and _canon(m.group(1), roster)[1] in ("interrogator", "suspect"):
            lines.append({"speaker": m.group(1).upper(), "text": m.group(2).strip(" _"), "kind": "dialogue"})
        elif NOTE.match(t):
            lines.append({"speaker": "", "text": NOTE.match(t).group(1), "kind": "note"})
        else:
            lines.append({"speaker": "", "text": t.strip(" _;:'\"“”"), "kind": "dialogue"})
    return {"page_label": label, "lines": lines}


def pages_from_text(path: Path, roster: dict[str, str]) -> list[dict]:
    parts = re.split(r"^### p-\d+\s*$", path.read_text(), flags=re.M)
    return [parse_page(p, roster) | {"file": f"p-{i:03d}"} for i, p in enumerate(parts[1:], 1)]


def pages_from_pdf(pdf: Path, roster: dict[str, str], tmp: Path) -> list[dict]:
    out = []
    for img in render(pdf, tmp / pdf.stem, dpi=200):
        txt = subprocess.run(["tesseract", str(img), "-", "--psm", "6"], capture_output=True, text=True, check=True).stdout
        out.append(parse_page(txt, roster) | {"file": img.name})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--out", default="utterances_ocr.json")
    ap.add_argument("--text-dir", default="")
    ap.add_argument("--map", default="", help="session=filename.ocr.txt,... when using --text-dir")
    a = ap.parse_args()
    cd = case_dir(a.case)
    cfg = json.loads((cd / "case.json").read_text())
    roster = cfg["roster"]
    names = dict(x.split("=", 1) for x in a.map.split(",") if "=" in x)
    utts: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, s in enumerate(cfg["sessions"]):
            if not s.get("pdf"):
                continue
            if a.text_dir:
                f = Path(a.text_dir) / names.get(s["id"], Path(s["pdf"]).stem + ".ocr.txt")
                if not f.exists():
                    print(f"{s['id']}: no OCR text at {f}")
                    continue
                pages = pages_from_text(f, roster)
            else:
                pages = pages_from_pdf(cd / s["pdf"], roster, Path(tmp))
            u = merge(s["id"], pages, roster, (idx + 1) * 100000)
            print(f"{s['id']}: {len(pages)} pages -> {len(u)} turns "
                  f"{ {r: sum(1 for x in u if x['role'] == r) for r in ('interrogator', 'suspect', 'other')} }")
            utts += u
    (cd / a.out).write_text(json.dumps(utts, indent=1))


if __name__ == "__main__":
    main()

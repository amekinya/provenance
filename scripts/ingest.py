"""Turn a case's scanned transcript PDFs into ordered, speaker-labeled utterances.

Nemotron 3.5 Super VL reads each page image (parallel, cached per page). Code then merges continuation lines,
normalizes speaker labels against the case roster, assigns roles and a global order.
Usage: python -m scripts.ingest dassey [--sessions S1,S2] [--max-pages N]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.ai import tasks
from app.ai.client import CallRecord

ROOT = Path(__file__).resolve().parent.parent


def case_dir(case_id: str) -> Path:
    return ROOT / "data" / "cases" / case_id


def render(pdf: Path, out: Path, dpi: int = 150) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    if not any(out.glob("p-*.png")):
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", str(pdf), str(out / "p")], check=True)
    return sorted(out.glob("p-*.png"))


def read_pages(images: list[Path], cache: Path, calls: list[CallRecord]) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)

    def one(img: Path) -> dict:
        f = cache / (img.stem + ".json")
        if f.exists():
            return json.loads(f.read_text())
        local: list[CallRecord] = []
        for attempt in range(2):
            try:
                d = tasks.read_page(img.read_bytes(), local)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 1:
                    print(f"  page {img.name} failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                    return {"page_label": "", "lines": [], "error": str(e)[:200]}
        d["file"] = img.name
        d["cost_usd"] = round(sum(c.cost_usd for c in local), 5)
        f.write_text(json.dumps(d, indent=1))
        calls.extend(local)
        return d

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, images))


def _canon(label: str, roster: dict[str, str]) -> tuple[str, str]:
    """Speaker label as printed -> (canonical name, role). Tolerates OCR slips like WIGERT for WIEGERT."""
    k = re.sub(r"[^A-Z]", "", (label or "").upper())
    if not k:
        return "", ""
    if k in roster:
        return k, roster[k]
    best = min(roster, key=lambda r: _dist(k, r))
    if _dist(k, best) <= max(1, len(best) // 4):
        return best, roster[best]
    return k, "other"


def _dist(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def merge(session_id: str, pages: list[dict], roster: dict[str, str], start_seq: int) -> list[dict]:
    """Page lines -> utterances. A line without a speaker continues the previous turn (across page breaks too)."""
    utts: list[dict] = []
    for pg in pages:
        label = pg.get("page_label") or pg.get("file", "")
        for ln in pg.get("lines", []):
            if ln.get("kind") == "header" or not ln.get("text", "").strip():
                continue
            text = ln["text"].strip()
            if ln.get("kind") == "note":
                text = f"({text.strip('()[] ')})"
            name, role = _canon(ln.get("speaker", ""), roster)
            if not name and utts:
                utts[-1]["text"] = (utts[-1]["text"] + " " + text).strip()
                continue
            if not name:
                name, role = "UNKNOWN", "other"
            utts.append({"id": "", "session": session_id, "seq": 0, "speaker": name, "role": role, "text": text,
                         "page": str(label), "sensitive": False})
    for i, u in enumerate(utts):
        u["id"] = f"{session_id}-{i + 1:04d}"
        u["seq"] = start_seq + i
    return utts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--pages", default="", help="a page range to read for a test, e.g. 60-66 (doesn't write utterances.json)")
    a = ap.parse_args()
    cd = case_dir(a.case)
    cfg = json.loads((cd / "case.json").read_text())
    roster = cfg["roster"]
    want = set(filter(None, a.sessions.split(",")))
    calls: list[CallRecord] = []
    all_utts: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, s in enumerate(cfg["sessions"]):
            if not s.get("pdf") or (want and s["id"] not in want):
                continue
            imgs = render(cd / s["pdf"], Path(tmp) / s["id"])
            if a.max_pages:
                imgs = imgs[: a.max_pages]
            if a.pages:
                lo, hi = (int(x) for x in a.pages.split("-"))
                imgs = imgs[lo - 1: hi]
            pages = read_pages(imgs, cd / "pages" / s["id"], calls)
            utts = merge(s["id"], pages, roster, (idx + 1) * 100000)  # session order fixes the global order
            all_utts += utts
            roles = {r: sum(1 for u in utts if u["role"] == r) for r in ("interrogator", "suspect", "other")}
            print(f"{s['id']}: {len(imgs)} pages -> {len(utts)} utterances {roles}")
    if a.pages:
        print("test range only; utterances.json not changed")
        for u in all_utts[:40]:
            print(f"  {u['speaker']:11s} {u['text'][:110]}")
    elif not want:
        (cd / "utterances.json").write_text(json.dumps(all_utts, indent=1))
    else:  # merge into the existing file, replacing only the sessions we re-read
        old = json.loads((cd / "utterances.json").read_text()) if (cd / "utterances.json").exists() else []
        keep = [u for u in old if u["session"] not in want]
        (cd / "utterances.json").write_text(json.dumps(sorted(keep + all_utts, key=lambda u: u["seq"]), indent=1))
    cost = sum(c.cost_usd for c in calls)
    print(f"model calls this run: {len(calls)}, ${cost:.4f}")


if __name__ == "__main__":
    main()

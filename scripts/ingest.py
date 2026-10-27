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


TITLES = r"^(?:INV(?:ESTIGATOR)?|SA|S/A|SPECIAL AGENT|AGENT|DET(?:ECTIVE)?|SGT|SERGEANT|LT|OFFICER|OFC|DEPUTY|DEP|CPT|CAPT|MR|MRS|MS)\.?\s+"


def _canon(label: str, roster: dict[str, str]) -> tuple[str, str]:
    """Speaker label as printed -> (canonical name, role). Tolerates titles ('INV. WIEGERT'), full names ('BRENDAN DASSEY'),
    and OCR or typing slips ('WIGERT', 'WEIGERT')."""
    raw = re.sub(TITLES, "", (label or "").upper().strip(), flags=re.I)
    words = [re.sub(r"[^A-Z]", "", w) for w in re.split(r"[\s.]+", raw)]
    words = [w for w in words if w]
    if not words:
        return "", ""
    for w in [*words, "".join(words)]:  # exact word match first ('BRENDAN DASSEY' -> BRENDAN)
        if w in roster:
            return w, roster[w]
    best_name, best_d = None, 99
    for w in words:
        for r in roster:
            d = _dist(w, r)
            if d < best_d:
                best_name, best_d = r, d
    if best_name and best_d <= max(1, len(best_name) // 4):
        return best_name, roster[best_name]
    return "".join(words), "other"


def _dist(a: str, b: str) -> int:
    """Damerau-Levenshtein (optimal string alignment): a swapped pair of letters counts as one edit."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _nod_split(text: str, suspect_names: set[str]) -> list[tuple[str, str]]:
    """Split '(Brendan nods "yes")' out of an investigator's turn: a nod is the suspect's answer, not the investigator's words.
    -> [("speaker"|"suspect", text), ...] in order."""
    names = "|".join(sorted(suspect_names, key=len, reverse=True)) or "SUSPECT"
    rx = re.compile(rf"\((?:{names})\s+(nods?|shakes?)[^)]*\)", re.I)
    out, last = [], 0
    for m in rx.finditer(text):
        before = text[last:m.start()].strip()
        if before:
            out.append(("speaker", before))
        out.append(("suspect", "(" + m.group(0).strip("()").split(None, 1)[1] + ")"))
        last = m.end()
    rest = text[last:].strip()
    if rest:
        out.append(("speaker", rest))
    return out or [("speaker", text)]


def printed_pages(pages: list[dict]) -> list[str]:
    """The page number printed on each page ('Page 588'), from the model's page_label or a header line. Where a page's
    number wasn't read, infer it from its neighbours (printed number minus PDF index is constant within a transcript);
    failing that, cite the PDF page."""
    idx = [int(re.sub(r"\D", "", p.get("file", "")) or 0) for p in pages]
    found: list[int | None] = []
    for p in pages:
        texts = [str(p.get("page_label") or "")] + [ln.get("text", "") for ln in p.get("lines", []) if ln.get("kind") == "header"]
        m = next((re.search(r"\bPage\s*(\d{1,5})\b", t, re.I) for t in texts if re.search(r"\bPage\s*\d{1,5}\b", t, re.I)), None)
        if m is None and re.fullmatch(r"\s*\d{1,5}\s*", texts[0]):
            m = re.search(r"\d+", texts[0])
        found.append(int(m.group(1) if m.lastindex else m.group(0)) if m else None)
    offsets = [f - i for f, i in zip(found, idx) if f is not None and i]
    off = max(set(offsets), key=offsets.count) if offsets else None
    out = []
    for f, i in zip(found, idx):
        if f is not None and (off is None or f - i == off):
            out.append(str(f))
        elif off is not None and i:
            out.append(str(i + off))
        else:
            out.append(f"PDF p. {i}" if i else "?")
    return out


def merge(session_id: str, pages: list[dict], roster: dict[str, str], start_seq: int, page_offset: int | None = None) -> list[dict]:
    """Page lines -> utterances. A line without a speaker continues the previous turn (across page breaks too).
    Stage directions continue the current turn, except a suspect's nod or head shake, which becomes his own turn.
    Pages whose printed number wasn't read are cited as 'PDF p. N'."""
    suspect_names = {k for k, v in roster.items() if v == "suspect"}
    suspect_key = next(iter(sorted(suspect_names)), "SUSPECT")
    utts: list[dict] = []
    unknown: dict[str, int] = {}
    labels = printed_pages(pages)
    for pg, label in zip(pages, labels):
        for ln in pg.get("lines", []):
            if ln.get("kind") == "header" or not ln.get("text", "").strip():
                continue
            text = ln["text"].strip()
            if ln.get("kind") == "note":
                text = f"({text.strip('()[] ')})"
                name, role = "", ""  # a stage direction never starts a new speaker turn
            else:
                name, role = _canon(ln.get("speaker", ""), roster)
                if role == "other" and name:
                    unknown[name] = unknown.get(name, 0) + 1
            parts = _nod_split(text, suspect_names)
            for k, (who, chunk) in enumerate(parts):
                if who == "suspect":
                    utts.append({"id": "", "session": session_id, "seq": 0, "speaker": suspect_key, "role": "suspect", "text": chunk,
                                 "page": str(label), "sensitive": False, "from_note": True})
                    continue
                if k == 0 and not name and utts:
                    utts[-1]["text"] = (utts[-1]["text"] + " " + chunk).strip()
                    continue
                if k > 0 and parts[k - 1][0] == "suspect":  # words after a nod belong to whoever was speaking
                    spk = next((u for u in reversed(utts) if not u.get("from_note")), None)
                    nm, rl = (name, role) if name else ((spk["speaker"], spk["role"]) if spk else ("UNKNOWN", "other"))
                else:
                    nm, rl = (name, role) if name else ("UNKNOWN", "other")
                utts.append({"id": "", "session": session_id, "seq": 0, "speaker": nm, "role": rl, "text": chunk,
                             "page": str(label), "sensitive": False})
    for i, u in enumerate(utts):
        u["id"] = f"{session_id}-{i + 1:04d}"
        u["seq"] = start_seq + i
        u.pop("from_note", None)
    if unknown:
        print(f"  {session_id}: labels not in the roster (role 'other'): {dict(sorted(unknown.items(), key=lambda x: -x[1])[:10])}")
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
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for idx, s in enumerate(cfg["sessions"]):
            if not s.get("pdf") or (want and s["id"] not in want):
                continue
            imgs = render(cd / s["pdf"], Path(tmp) / s["id"])
            if a.max_pages and not a.pages:
                imgs = imgs[: a.max_pages]
            if a.pages:
                lo, hi = (int(x) for x in a.pages.split("-"))
                imgs = imgs[lo - 1: hi]
            pages = read_pages(imgs, cd / "pages" / s["id"], calls)
            failed += sum(1 for p in pages if p.get("error"))
            utts = merge(s["id"], pages, roster, (idx + 1) * 100000)  # session order fixes the global order
            all_utts += utts
            roles = {r: sum(1 for u in utts if u["role"] == r) for r in ("interrogator", "suspect", "other")}
            print(f"{s['id']}: {len(imgs)} pages -> {len(utts)} utterances {roles}")
    if failed:
        print(f"{failed} page(s) could not be read; utterances.json NOT changed. Re-run to retry (read pages are cached).")
        sys.exit(1)
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

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


NOT_SPEAKERS = {"", "note", "notes", "header", "stage direction", "direction", "narration", "none", "unknown"}


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
    for w in [*words, "".join(words)] if len(words) > 1 else words:  # 'FASAS BENDER' is one name split in two
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


# An investigator reading the suspect's written statement aloud: the words are the suspect's, not the investigator's.
READ_ANNOUNCE = re.compile(r"\bread (it|this|that)\b|\bread what you (wrote|just added|added)\b|\bread this last part\b", re.I)
WROTE_CUE = re.compile(r"\byou wrote\b|\byou (just )?added\b|\byour statement\b|\bthe statement\b|\bwritten\b|\bwrite\b", re.I)
CHECK_Q = re.compile(r"\s*((is|was) (that|this) (your|what you|right|correct|true)[^?]*\?)\s*$", re.I)


def split_read_aloud(utts: list[dict], suspect_key: str) -> list[dict]:
    """Where an investigator reads the suspect's written statement aloud, the statement becomes its own line: role
    suspect, marked '(written statement, read aloud by X)', same page. Two layouts occur:
    - inline: 'I'm going to read what you wrote. (Is I got off the bus ... Is that your statement Brendan?)'
    - next line: 'I'll read this. Is that okay?' / 'Yeah.' / '<the statement text>' by the same investigator.
    The announcement must refer to what the suspect wrote (its own words or the 2 lines before it), so reading a
    Miranda form isn't mistaken for his statement."""
    out: list[dict] = []
    pending: dict | None = None  # an announcement waiting for the read text on a following line
    for k, u in enumerate(utts):
        cue_text = " ".join(x["text"] for x in utts[max(0, k - 2):k + 1])
        announce = u["role"] == "interrogator" and READ_ANNOUNCE.search(u["text"]) and WROTE_CUE.search(cue_text)
        if announce:
            m = re.search(r"\(([^()]{80,})\)?\s*$", u["text"], re.S)
            if m and len(m.group(1).split()) >= 15:  # inline layout
                body = m.group(1).rstrip(") ")
                q = CHECK_Q.search(body)
                tail = q.group(1) if q else ""
                body = body[: q.start()] if q else body
                before = u["text"][: m.start()].strip()
                body_start = m.start(1)
                tail_start = body_start + (q.start(1) if q else len(body))
                if before:
                    out.append(u | {"text": before, "_segs": _span(u, 0, m.start())})
                out.append(_statement(u, body, suspect_key) | {"_segs": _span(u, body_start, tail_start)})
                if tail:
                    out.append(u | {"text": tail, "_segs": _span(u, tail_start, len(u["text"]))})
                pending = None
                continue
            out.append(u)
            pending = {"speaker": u["speaker"], "left": 3}
            continue
        if pending:
            pending["left"] -= 1
            if u["role"] == "interrogator" and u["speaker"] == pending["speaker"] and "?" not in u["text"] and len(u["text"].split()) >= 12:
                out.append(_statement(u, u["text"], suspect_key))
                pending = None
                continue
            if pending["left"] <= 0 or (u["role"] == "interrogator" and u["speaker"] != pending["speaker"]):
                pending = None
        out.append(u)
    return out


def _span(u: dict, start: int, end: int) -> list:
    """The page segments of u's text between two character offsets (for a turn split in two)."""
    segs = u.get("_segs") or [[0, u["page"]]]
    first = max((sg for sg in segs if sg[0] <= start), key=lambda sg: sg[0])
    return [[0, first[1]]] + [[sg[0] - start, sg[1]] for sg in segs if start < sg[0] < end]


def _statement(u: dict, body: str, suspect_key: str) -> dict:
    reader = u["speaker"].title()
    return u | {"speaker": suspect_key, "role": "suspect",
                "text": f"(read aloud by {reader} from {suspect_key.title()}'s written statement; printed speaker: {u['speaker']}) {body.strip()}",
                "read_aloud_by": u["speaker"]}


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
            name, role = _canon(ln.get("speaker", ""), roster)
            # The reader sometimes marks a whole page's dialogue as "note". A line is a stage direction only when it has
            # no speaker of its own: a roster name, or any other printed name in front of words that aren't in brackets.
            spk_label = re.sub(r"[^A-Za-z ]", "", ln.get("speaker", "")).strip().lower()
            spoken = role in ("interrogator", "suspect") or (spk_label not in NOT_SPEAKERS and not text.startswith(("(", "[")))
            if ln.get("kind") == "note" and not spoken:
                text = f"({text.strip('()[] ')})"
                name, role = "", ""  # a stage direction never starts a new speaker turn
            elif role == "other" and name:
                unknown[name] = unknown.get(name, 0) + 1
            parts = _nod_split(text, suspect_names)
            for k, (who, chunk) in enumerate(parts):
                if who == "suspect":
                    utts.append({"id": "", "session": session_id, "seq": 0, "speaker": suspect_key, "role": "suspect", "text": chunk,
                                 "page": str(label), "sensitive": False, "from_note": True, "_segs": [[0, str(label)]]})
                    continue
                if k == 0 and not name and utts:
                    prev = utts[-1]
                    if prev["_segs"][-1][1] != str(label):  # the turn continues onto the next printed page
                        prev["_segs"].append([len(prev["text"]) + 1, str(label)])
                    prev["text"] = (prev["text"] + " " + chunk).strip()
                    continue
                if k > 0 and parts[k - 1][0] == "suspect":  # words after a nod belong to whoever was speaking
                    spk = next((u for u in reversed(utts) if not u.get("from_note")), None)
                    nm, rl = (name, role) if name else ((spk["speaker"], spk["role"]) if spk else ("UNKNOWN", "other"))
                else:
                    nm, rl = (name, role) if name else ("UNKNOWN", "other")
                utts.append({"id": "", "session": session_id, "seq": 0, "speaker": nm, "role": rl, "text": chunk,
                             "page": str(label), "sensitive": False, "_segs": [[0, str(label)]]})
    utts = split_read_aloud(utts, suspect_key)
    for i, u in enumerate(utts):
        u["id"] = f"{session_id}-{i + 1:04d}"
        u["seq"] = start_seq + i
        segs = u.pop("_segs", None) or [[0, u["page"]]]
        u["page"], u["page_end"] = segs[0][1], (segs[-1][1] if segs[-1][1] != segs[0][1] else "")
        if u.pop("from_note", None):
            u["split"] = True  # a nod pulled out of the investigator's printed turn: not a turn of its own
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
    else:
        old = json.loads((cd / "utterances.json").read_text()) if (cd / "utterances.json").exists() else []
        # A re-read keeps the graphic-content flags the tag step set, for lines whose id and text haven't changed.
        flagged = {(u["id"], u["text"]) for u in old if u.get("sensitive")}
        flagged_long = {(u["session"], u["text"]) for u in old if u.get("sensitive") and len(u["text"].split()) >= 8}  # ids can shift
        for u in all_utts:
            body = re.sub(r"^\([^)]*\)\s*", "", u["text"]) if u.get("read_aloud_by") else None  # the note before a read-aloud statement may change
            u["sensitive"] = (u.get("sensitive", False) or (u["id"], u["text"]) in flagged or (u["session"], u["text"]) in flagged_long
                              or (body is not None and any(i == u["id"] and re.sub(r"^\([^)]*\)\s*", "", t) == body for i, t in flagged)))
        keep = [u for u in old if want and u["session"] not in want]  # with --sessions, only those sessions are replaced
        (cd / "utterances.json").write_text(json.dumps(sorted(keep + all_utts, key=lambda u: u["seq"]), indent=1))
    cost = sum(c.cost_usd for c in calls)
    print(f"model calls this run: {len(calls)}, ${cost:.4f}")


if __name__ == "__main__":
    main()

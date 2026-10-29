"""Provenance web app. Stateless: nothing a user pastes is stored."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ai import tasks
from app.ai.client import CallRecord, LiveModelUnavailable, ledger
from app.config import settings
from app.engine.ledger import audit, load_case

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
BENCH = Path(__file__).resolve().parent.parent / "eval" / "results.json"
CASES = ROOT / "data" / "cases"

app = FastAPI(title="Provenance")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _read(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return default


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"live_models": settings.live_models, "spend_remaining_today": round(ledger.remaining(), 4),
            "models": {s.role: s.model_id for s in (settings.lightning, settings.super_vl, settings.ultra)},
            "matcher": bool(settings.matcher_url), "draft_key_words": KT_DRAFT}


# Drafted key words are offered in the Try form only if they measure at least as well as no key words on the dev slice.
# They didn't (eval/RESULTS_dev.md, Oct 29: 61% vs 84% right first speaker), so drafting is off by default; hand-written
# key words can always be typed. KEYWORD_DRAFTING=1 turns the button back on.
KT_DRAFT = os.environ.get("KEYWORD_DRAFTING", "0") == "1"


@app.get("/api/cases")
def cases():
    out = []
    show_test = os.environ.get("SHOW_TEST_CASES") == "1"
    for d in sorted(CASES.iterdir()):
        if d.name.startswith("_") and not show_test:  # scratch cases built for UI testing are never listed
            continue
        cfg, rep = _read(d / "case.json"), _read(d / "audit.json")
        if cfg and rep:
            out.append({"id": cfg["id"], "title": cfg["title"], "suspect": cfg["suspect"], "headline": rep["headline"],
                        "sessions": len([s for s in cfg["sessions"] if s.get("recorded", True)]), "facts": len(rep["facts"])})
    return out


@app.get("/api/case/{cid}")
def case(cid: str):
    if not re.fullmatch(r"[a-z0-9_-]+", cid):
        raise HTTPException(404)
    d = CASES / cid
    rep = _read(d / "audit.json")
    if not rep:
        raise HTTPException(404, "Unknown case.")
    run = _read(d / "run.json", {}) or {}
    keep = ("tag_seconds", "total_seconds", "cost_usd", "by_model", "by_step", "labeler", "read_by", "unverified", "safety_search_added",
            "code_agreements_added", "verifier_reasoning", "investigator_side", "sense_check_cost_usd", "memo_cost_usd")
    one = _read(d / "one_prompt.json")
    return rep | {"memo": _read(d / "memo.json"), "run": {k: run.get(k) for k in keep},
                  "config": _read(d / "case.json"), "eval": _read(d / "eval.json") or _read(BENCH),
                  "one_prompt": {k: v for k, v in one.items() if k != "raw"} if one else None}


@app.get("/api/case/{cid}/lines")
def lines(cid: str, around: str, before: int = 6, after: int = 4):
    """Context around one sentence, for reading a quote in place."""
    if not re.fullmatch(r"[a-z0-9_-]+", cid):
        raise HTTPException(404)
    utts = _read(CASES / cid / "utterances.json", []) or []
    order = sorted(utts, key=lambda u: u["seq"])
    idx = next((i for i, u in enumerate(order) if u["id"] == around), None)
    if idx is None:
        raise HTTPException(404, "Unknown line.")
    sess = order[idx]["session"]
    window = [u for u in order[max(0, idx - before): idx + after + 1] if u["session"] == sess]
    return {"lines": window, "focus": around}


# ---------------------------------------------------------------------------------------------------------------------
# Try it on your own transcript (live models, spend-capped)
# ---------------------------------------------------------------------------------------------------------------------
class AnalyzeIn(BaseModel):
    transcript: str = Field(..., max_length=120_000)
    facts: str = Field(..., max_length=6_000)
    suspect: str = Field("", max_length=80)
    interrogators: str = Field("", max_length=200)
    # One line per detail, in the same order: 'cut/chop/hack + hair' (groups joined by +, alternates by / or ,). Optional.
    key_words: list[str] = Field(default_factory=list, max_length=20)
    private: bool = False  # our fine-tuned matcher on your own endpoint + code only; nothing goes to an outside AI service


def parse_key_words(line: str) -> list[list[str]]:
    from app.ai.tasks import normalize_must  # noqa: PLC0415

    return normalize_must([re.split(r"\s*[/,|]\s*", g) for g in re.split(r"\s*\+\s*", line or "") if g.strip()])


def key_words_text(must: list[list[str]]) -> str:
    return " + ".join("/".join(g) for g in must)


class KeyWordsIn(BaseModel):
    facts: str = Field(..., max_length=6_000)


@app.post("/api/keywords")
def draft_key_words(body: KeyWordsIn):
    """Nemotron 3 Ultra drafts each detail's key words from its wording only; the user edits them before running."""
    labels = [x.strip(" -•\t") for x in body.facts.splitlines() if x.strip()][:20]
    if not labels:
        raise HTTPException(400, "List the key crime details, one per line.")
    if not KT_DRAFT:
        raise HTTPException(404, "Drafted key words are switched off; type them by hand.")
    if not settings.live_models or ledger.remaining() <= 0.05:
        raise HTTPException(503, "Drafting isn't available right now; you can type key words by hand.")
    calls: list[CallRecord] = []
    try:
        kt = tasks.draft_key_terms([{"id": f"F{i:02d}", "label": lab, "detail": ""} for i, lab in enumerate(labels, 1)], calls)
    except LiveModelUnavailable as e:
        raise HTTPException(503, f"The model service isn't available right now ({e}).") from e
    return {"key_words": [key_words_text(kt.get(f"F{i:02d}", [])) for i in range(1, len(labels) + 1)],
            "cost_usd": round(sum(c.cost_usd for c in calls), 4), "model": settings.ultra.model_id}


LINE = re.compile(r"^\s*([A-Z][A-Za-z .'\-]{0,30}?)\s*:\s*(.+)$")


def parse_transcript(text: str, suspect: str, interrogators: str) -> list[dict]:
    """'NAME: words' lines -> utterances. Unlabeled lines continue the previous turn."""
    inv = {re.sub(r"[^A-Z]", "", x.upper()) for x in re.split(r"[,;]", interrogators) if x.strip()}
    sus = re.sub(r"[^A-Z]", "", suspect.upper())
    utts: list[dict] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = LINE.match(raw)
        if m:
            key = re.sub(r"[^A-Z]", "", m.group(1).upper())
            role = "suspect" if sus and (key == sus or sus.startswith(key) or key.startswith(sus)) else \
                "interrogator" if key in inv or any(key.startswith(i) or i.startswith(key) for i in inv if i) else \
                "interrogator" if key in {"Q", "DET", "DETECTIVE", "OFFICER", "INV", "AGENT", "SGT"} else \
                "suspect" if key in {"A", "SUSPECT"} else "other"
            utts.append({"id": f"U-{len(utts) + 1:04d}", "session": "S1", "seq": len(utts) + 1, "speaker": m.group(1).strip().upper(),
                         "role": role, "text": m.group(2).strip(), "page": "", "sensitive": False})
        elif utts:
            utts[-1]["text"] += " " + raw.strip()
    return utts


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    from scripts.tag import by_step, case_dict, check_labels, tag_all  # noqa: PLC0415

    utts = parse_transcript(body.transcript, body.suspect, body.interrogators)
    if len(utts) < 4:
        raise HTTPException(400, "Paste the transcript as lines like 'DET: question' and 'SUSPECT: answer'.")
    if not any(u["role"] == "suspect" for u in utts) or not any(u["role"] == "interrogator" for u in utts):
        raise HTTPException(400, "Tell us which speaker is the suspect and which are investigators.")
    facts = [{"id": f"F{i:02d}", "label": ln.strip(" -•\t"), "detail": "", "aliases": []}
             for i, ln in enumerate([x for x in body.facts.splitlines() if x.strip()][:20], 1)]
    if not facts:
        raise HTTPException(400, "List the key crime details, one per line.")
    for f, kw in zip(facts, body.key_words):
        must = parse_key_words(kw)
        if must:
            f["must"] = must
    if body.private:
        return analyze_private(utts, facts, body)
    if not settings.live_models or ledger.remaining() <= 0.05:
        raise HTTPException(503, "Live analysis isn't available right now. The sample cases still work.")
    cfg = {"id": "yours", "title": "Your transcript", "suspect": body.suspect or "Suspect",
           "sessions": [{"id": "S1", "label": "Your transcript", "day": "2000-01-01"}], "notes": []}
    calls: list[CallRecord] = []
    t0 = time.perf_counter()
    try:
        mentions, graphic = tag_all(utts, facts, 18, calls, spec=settings.ultra)  # the same pipeline as the case audits
        for u in utts:
            u["sensitive"] = graphic.get(u["id"], False)
        agreed, searched, unverified = check_labels(cfg, utts, facts, mentions, calls, rounds=3)
        if any(f.get("must") for f in facts):  # with key words: the suspect-side search and sense check, as in the case audits
            from scripts.tag import search_suspect, sense_check  # noqa: PLC0415
            search_suspect(cfg, utts, facts, mentions, calls)
            sense_check(cfg, utts, facts, mentions, calls, rounds=3)
    except LiveModelUnavailable as e:
        raise HTTPException(503, f"The model service isn't available right now ({e}).") from e
    rep = audit(load_case(case_dict(cfg, utts, facts, mentions)))
    return JSONResponse(rep | {"utterances": utts, "run": {"labeler": "ultra", "unverified": unverified, "safety_search_added": searched,
                                                           "investigator_side": "key words (code)" if any(f.get("must") for f in facts) else "",
                                                           "code_agreements_added": agreed, "by_step": by_step(calls),
                                                           "total_seconds": round(time.perf_counter() - t0, 1),
                                                           "cost_usd": round(sum(c.cost_usd for c in calls), 4),
                                                           "calls": [asdict(c) | {"cost_usd": round(c.cost_usd, 5)} for c in calls]}})


def analyze_private(utts: list[dict], facts: list[dict], body: AnalyzeIn):
    """Private mode: our fine-tuned matcher (on a Nebius endpoint in the user's own project) labels the lines, code adds
    bare agreements and the ledger decides. No Token Factory call, so nothing leaves the user's infrastructure."""
    from scripts.tag import add_code_assents, by_step, case_dict, tag_with_matcher  # noqa: PLC0415

    if not settings.matcher_url:
        raise HTTPException(503, "Private mode needs your own matcher endpoint (ENDPOINT_URL); it isn't configured on this server.")
    cfg = {"id": "yours", "title": "Your transcript", "suspect": body.suspect or "Suspect",
           "sessions": [{"id": "S1", "label": "Your transcript", "day": "2000-01-01"}], "notes": []}
    calls: list[CallRecord] = []
    t0 = time.perf_counter()
    th = (_read(ROOT / "eval" / "training.json", {}) or {}).get("threshold", 0.5)
    try:
        mentions, _ = tag_with_matcher(utts, facts, calls, threshold=float(th))
    except LiveModelUnavailable as e:
        raise HTTPException(503, f"Your matcher endpoint isn't reachable right now ({e}).") from e
    agreed = add_code_assents(utts, mentions)
    rep = audit(load_case(case_dict(cfg, utts, facts, mentions)))
    return JSONResponse(rep | {"utterances": utts, "run": {"labeler": "matcher", "private": True, "unverified": 0, "code_agreements_added": agreed,
                                                           "investigator_side": "key words (code)" if any(f.get("must") for f in facts) else "",
                                                           "by_step": by_step(calls), "total_seconds": round(time.perf_counter() - t0, 1),
                                                           "cost_usd": 0.0, "calls": [asdict(c) | {"cost_usd": 0.0} for c in calls]}})

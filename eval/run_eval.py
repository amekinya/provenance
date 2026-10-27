"""Measure who-said-it-first accuracy on synthetic interrogations with known answers.

Methods (same interrogations, same detail lists):
  keyword       first line containing the detail's words (no model)
  one_prompt    Nemotron 3 Ultra reads the whole interrogation in one prompt and answers per detail
  provenance    Nemotron 3.5 Lightning labels lines -> code ledger -> Nemotron 3 Ultra verifies deciding lines
  provenance+matcher (when ENDPOINT_URL is set) our fine-tuned matcher proposes the labels instead of Lightning
Metrics: first-mention speaker (investigator / suspect / never) and exact first line; plus the corroboration call.
Gold: "interrogator" = investigators named the detail before the suspect stated or agreed to it; "suspect" = the suspect said
it first; "never" = the suspect never stated or agreed to it.
Splits: test (held out, different writer models; the published numbers) or dev / train / trainval (the training writer's
interrogations, for diagnosing and tuning; never tune on test). Per-fact predictions go to eval/details_<split>.json.
Usage: python -m eval.run_eval [--split test] [--n 30] [--methods keyword,one_prompt,provenance]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.ai import tasks
from app.ai.client import CallRecord, LiveModelUnavailable
from app.engine.ledger import audit, load_case
from scripts.tag import add_code_assents, case_dict, search_investigator_first, tag_all, verify_deciding

ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = ROOT / "data" / "synth"
STOP = set("a an the of to in on at for with and or his her its their was were is be by from that this it he she they".split())


def gold(rec: dict) -> dict[str, dict]:
    """First investigator line and first suspect adoption per fact, from the writer's labels (consistency-checked)."""
    out = {}
    for f in rec["facts"]:
        fi = fs = None
        for u in rec["utterances"]:
            for c in u["carries"]:
                if c["fact"] != f["id"]:
                    continue
                if fi is None and u["role"] == "interrogator" and c["kind"] in ("asks", "asserts"):
                    fi = u["id"]
                if fs is None and u["role"] == "suspect" and c["kind"] in ("asserts", "assents"):
                    fs = u["id"]
        seq = {u["id"]: u["seq"] for u in rec["utterances"]}
        first = "never" if fs is None else ("interrogator" if fi is not None and seq[fi] < seq[fs] else "suspect")
        out[f["id"]] = {"first": first, "line": fi if first == "interrogator" else fs}
    return out


def _cfg(rec):
    return {"id": rec["id"], "title": rec["crime"], "suspect": "SUSPECT", "sessions": [{"id": "S1", "label": "Interview", "day": "2020-01-01"}]}


def predict_from_report(rep: dict) -> dict[str, dict]:
    out = {}
    for f in rep["facts"]:
        fi, fs = f["first_interrogator"], f["first_suspect"]
        first = "never" if not fs else ("interrogator" if fi and fi["seq"] < fs["seq"] else "suspect")
        out[f["fact"]["id"]] = {"first": first, "line": (fi or {}).get("utterance") if first == "interrogator" else (fs or {}).get("utterance")}
    return out


def keyword(rec: dict) -> dict[str, dict]:
    from app.engine.ledger import ASSENT, spoken  # noqa: PLC0415

    mentions = []
    utts = rec["utterances"]
    for f in rec["facts"]:
        words = [w for w in re.findall(r"[a-z]+", (f["label"]).lower()) if w not in STOP and len(w) > 2]
        hit_prev = False
        for u in utts:
            t = u["text"].lower()
            hit = bool(words) and sum(w in t for w in words) >= max(1, (len(words) + 1) // 2)
            if hit:
                kind = "asks" if u["role"] == "interrogator" and "?" in u["text"] else "asserts"
                mentions.append({"utterance": u["id"], "fact": f["id"], "kind": kind, "quote": "", "by": "keyword", "verified": None})
            elif hit_prev and u["role"] == "suspect" and ASSENT.match(spoken(u["text"])):
                # a fair baseline: a bare 'yeah' right after a question that carried the detail counts as agreeing to it
                mentions.append({"utterance": u["id"], "fact": f["id"], "kind": "assents", "quote": "", "by": "keyword", "verified": None})
            hit_prev = hit and u["role"] == "interrogator"
    facts = [{"id": f["id"], "label": f["label"], "detail": f["detail"]} for f in rec["facts"]]
    return predict_from_report(audit(load_case(case_dict(_cfg(rec), rec["utterances"], facts, mentions))))


def one_prompt(rec: dict, calls: list[CallRecord]) -> dict[str, dict]:
    rows = tasks.single_prompt([{"id": u["id"], "speaker": u["speaker"], "role": u["role"], "text": u["text"]} for u in rec["utterances"]],
                               rec["facts"], calls)
    out = {f["id"]: {"first": "never", "line": None, "answered": False} for f in rec["facts"]}
    by_label = {f["label"].strip().lower(): f["id"] for f in rec["facts"]}
    unmatched = []
    for r in rows:
        fid = str(r.get("fact", "")).strip()
        m = re.match(r"^(F\d+)\b", fid)
        fid = m.group(1) if m else by_label.get(fid.lower(), fid)
        if fid in out:
            out[fid] = {"first": r["first_said_by"], "line": (r.get("utterance_id") or "").strip() or None, "answered": True}
        else:
            unmatched.append(r.get("fact"))
    missing = [k for k, v in out.items() if not v["answered"]]
    if missing or unmatched:
        print(f"  one_prompt {rec['id']}: {len(rows)} rows; no answer for {missing}; unmatched ids {unmatched[:5]}")
    return out


TRACE: dict[str, list[dict]] = {}  # rec id -> the labels the method ended with (for eval/details_<split>.json)


def _with_trace(rec, utts, facts, mentions, calls, full: bool = True):
    if full:
        add_code_assents(utts, mentions)
    verify_deciding(_cfg(rec), utts, facts, mentions, calls, rounds=3)
    if full and search_investigator_first(_cfg(rec), utts, facts, mentions, calls):
        verify_deciding(_cfg(rec), utts, facts, mentions, calls, rounds=3)
    TRACE[rec["id"]] = [{k: m.get(k) for k in ("utterance", "fact", "kind", "quote", "by", "verified")} for m in mentions]
    return predict_from_report(audit(load_case(case_dict(_cfg(rec), utts, facts, mentions))))


def _inputs(rec):
    utts = [{k: v for k, v in u.items() if k != "carries"} | {"page": "", "sensitive": False} for u in rec["utterances"]]
    facts = [{"id": f["id"], "label": f["label"], "detail": f["detail"], "aliases": []} for f in rec["facts"]]
    return utts, facts


def provenance(rec: dict, calls: list[CallRecord]) -> dict[str, dict]:
    utts, facts = _inputs(rec)
    mentions, _ = tag_all(utts, facts, 18, calls)
    return _with_trace(rec, utts, facts, mentions, calls)


def provenance_ultra(rec: dict, calls: list[CallRecord]) -> dict[str, dict]:
    from app.config import settings  # noqa: PLC0415

    utts, facts = _inputs(rec)
    mentions, _ = tag_all(utts, facts, 18, calls, spec=settings.ultra)
    return _with_trace(rec, utts, facts, mentions, calls)


def provenance_basic(rec: dict, calls: list[CallRecord]) -> dict[str, dict]:
    """Ablation: Lightning labels + Ultra check only (no code agreements, no safety search), as before Oct 27."""
    utts, facts = _inputs(rec)
    mentions, _ = tag_all(utts, facts, 18, calls)
    return _with_trace(rec, utts, facts, mentions, calls, full=False)


def provenance_matcher(rec: dict, calls: list[CallRecord]) -> dict[str, dict]:
    from scripts.tag import tag_with_matcher  # noqa: PLC0415

    utts, facts = _inputs(rec)
    mentions, _ = tag_with_matcher(utts, facts, calls)
    return _with_trace(rec, utts, facts, mentions, calls)


def load_split(split: str) -> list[dict]:
    from scripts.build_pairs import trainval_split  # noqa: PLC0415

    f = SYNTH_DIR / ("interrogations_test.jsonl" if split == "test" else "interrogations_trainval.jsonl")
    recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    if split in ("dev", "train"):
        sp = trainval_split([r["id"] for r in recs])
        recs = [r for r in recs if sp[r["id"]] == split]
    return recs


def score(golds: list[dict], preds: list[dict]) -> dict:
    n = sp = ln = 0
    conf = {}
    failed = 0
    for g, p in zip(golds, preds):
        if p is None:  # the method crashed on this interrogation: every fact counts as wrong
            failed += 1
            n += len(g)
            conf["failed"] = conf.get("failed", 0) + len(g)
            continue
        for fid, gv in g.items():
            pv = p.get(fid, {"first": "never", "line": None})
            n += 1
            sp += pv["first"] == gv["first"]
            ln += pv["first"] == gv["first"] and (gv["first"] == "never" or pv["line"] == gv["line"])
            conf[f"{gv['first']}->{pv['first']}"] = conf.get(f"{gv['first']}->{pv['first']}", 0) + 1
    # The costly error for counsel: calling a detail 'suspect said it first' when investigators supplied it.
    false_corrob = conf.get("interrogator->suspect", 0)
    return {"n_facts": n, "failed_interrogations": failed, "speaker_acc": round(sp / max(n, 1), 3), "line_acc": round(ln / max(n, 1), 3),
            "investigator_first_called_suspect_first": false_corrob, "confusion": conf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--methods", default="keyword,one_prompt,provenance")
    ap.add_argument("--split", default="test", choices=["test", "dev", "train", "trainval"])
    a = ap.parse_args()
    recs = load_split(a.split)[: a.n]
    if not recs:
        raise SystemExit(f"no interrogations in split {a.split}")
    golds = [gold(r) for r in recs]
    results, costs, details = {}, {}, {}
    for m in a.methods.split(","):
        calls: list[CallRecord] = []
        t0 = time.perf_counter()
        fn = {"keyword": lambda r: keyword(r), "one_prompt": lambda r: one_prompt(r, calls), "provenance": lambda r: provenance(r, calls),
              "provenance_basic": lambda r: provenance_basic(r, calls),
              "provenance_ultra": lambda r: provenance_ultra(r, calls),
              "provenance_matcher": lambda r: provenance_matcher(r, calls)}[m]

        def safe(r, fn=fn):
            try:
                return fn(r)
            except LiveModelUnavailable as e:
                print(f"  {m} failed on {r['id']}: {e}")
                return None
        TRACE.clear()
        with ThreadPoolExecutor(4 if m != "keyword" else 1) as ex:
            preds = list(ex.map(safe, recs))
        results[m] = score(golds, preds)
        details[m] = [{"id": r["id"], "failed": p is None, "labels": TRACE.get(r["id"]),
                       "facts": {fid: {"gold": g[fid], "pred": (p or {}).get(fid)} for fid in g}} for r, g, p in zip(recs, golds, preds)]
        costs[m] = {"seconds": round(time.perf_counter() - t0, 1), "cost_usd": round(sum(c.cost_usd for c in calls), 4), "calls": len(calls)}
        print(m, results[m], costs[m])
    names = {"keyword": "Keyword match (no model)", "one_prompt": "Nemotron 3 Ultra, one prompt per interview",
             "provenance": "Provenance: Lightning labels + code ledger + Ultra check + safety search",
             "provenance_basic": "Provenance without code agreements or safety search (ablation)",
             "provenance_ultra": "Provenance with Nemotron 3 Ultra as the labeler + code ledger + Ultra check + safety search",
             "provenance_matcher": "Provenance with our fine-tuned matcher (Nebius) + code ledger + Ultra check"}
    writers = sorted({r["generator"] for r in recs})
    who = " and ".join(writers) + (" (model families different from the training writer)" if a.split == "test" else " (the training writer; diagnostic split)")
    out = {"split": a.split, "n_interrogations": len(recs), "generator": ", ".join(writers), "results": results, "costs": costs,
           "rows": [{"method": names.get(k, k), **v, **costs[k]} for k, v in results.items()],
           "note": f"{len(recs)} synthetic interrogations ({a.split} split) written by {who}, {sum(len(g) for g in golds)} details with known first speakers."}
    suffix = "" if a.split == "test" else f"_{a.split}"
    (ROOT / "eval" / f"results{suffix}.json").write_text(json.dumps(out, indent=1))
    (ROOT / "eval" / f"details_{a.split}.json").write_text(json.dumps({"recs": {r["id"]: {"facts": r["facts"], "plan": r.get("plan"),
                                                                                         "utterances": [{k: u[k] for k in ("id", "role", "text", "carries")} for u in r["utterances"]]}
                                                                               for r in recs}, "methods": details}, indent=1))
    md = ["# Provenance benchmark", "", out["note"], "", "| Method | Right first speaker | Right first line | Investigator-supplied details wrongly credited to the suspect | Time | Cost |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['method']} | {r['speaker_acc']:.0%} | {r['line_acc']:.0%} | {r['investigator_first_called_suspect_first']} | {r['seconds']} s | ${r['cost_usd']:.3f} |" for r in out["rows"]]
    (ROOT / "eval" / f"RESULTS{suffix}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Private-mode accuracy computed from a saved benchmark run, with no model calls.

Private mode = our fine-tuned matcher's labels + code's bare-agreement rule + the ledger; no Ultra check, no safety
search, no Token Factory. The benchmark's `provenance_matcher` method saves every matcher label (including the ones its
Ultra check later rejected) in eval/details_<split>.json, so private mode can be recomputed exactly from that file.
Usage: python -m eval.private_from_saved [--split test]   -> eval/private_mode.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.engine.ledger import audit, load_case
from eval import run_eval as R
from scripts.tag import add_code_assents, case_dict

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    det = json.loads((ROOT / "eval" / f"details_{a.split}.json").read_text())
    saved = det["methods"].get("provenance_matcher")
    if not saved:
        raise SystemExit(f"eval/details_{a.split}.json has no provenance_matcher run to recompute from")
    recs = {r["id"]: r for r in R.load_split(a.split)}
    golds, preds = [], []
    for x in saved:
        rec = recs[x["id"]]
        utts, facts = R._inputs(rec)
        ms = [dict(m, verified=None) for m in (x["labels"] or []) if m.get("by") == "matcher"]
        add_code_assents(utts, ms)
        preds.append(R.predict_from_report(audit(load_case(case_dict(R._cfg(rec), utts, facts, ms)))))
        golds.append(R.gold(rec))
    res = R.score(golds, preds)
    results = json.loads((ROOT / "eval" / ("results.json" if a.split == "test" else f"results_{a.split}.json")).read_text())
    out = {"split": a.split, "method": "Private mode: fine-tuned matcher labels + code (no outside AI service)",
           "computed_from": f"eval/details_{a.split}.json, provenance_matcher labels of run {results.get('run_number', '?')} "
                            f"({results.get('generated', '?')}, commit {results.get('commit') or '?'}), no new model calls",
           "result": res, "same_run_with_ultra_check": results["results"].get("provenance_matcher")}
    (ROOT / "eval" / "private_mode.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"private": {k: res[k] for k in ("n_facts", "speaker_acc", "line_acc", "investigator_first_called_suspect_first")},
                      "with_ultra_check": {k: out["same_run_with_ultra_check"][k] for k in ("speaker_acc", "line_acc", "investigator_first_called_suspect_first")}}))


if __name__ == "__main__":
    main()

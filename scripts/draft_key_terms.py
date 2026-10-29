"""Draft key words for the synthetic interrogations' details with Nemotron 3 Ultra, from each detail's label and
description only (never the transcript), and cache them, so the benchmark can measure the pipeline with drafted key
words without paying for them twice.
Writes data/synth/key_terms_<file>.json = {interrogation id: {fact id: must}}.
Usage: python -m scripts.draft_key_terms --file trainval        (the test file only for a final, recorded test run)
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.ai import tasks
from app.ai.client import CallRecord, LiveModelUnavailable

ROOT = Path(__file__).resolve().parent.parent
SYNTH = ROOT / "data" / "synth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="trainval", choices=["trainval", "test"])
    a = ap.parse_args()
    recs = [json.loads(x) for x in (SYNTH / f"interrogations_{a.file}.jsonl").read_text().splitlines() if x.strip()]
    out_file = SYNTH / f"key_terms_{a.file}.json"
    cache = json.loads(out_file.read_text()) if out_file.exists() else {}
    todo = [r for r in recs if r["id"] not in cache or set(cache[r["id"]]) != {f["id"] for f in r["facts"]}]
    calls: list[CallRecord] = []

    def run(r):
        local: list[CallRecord] = []
        try:
            return r["id"], tasks.draft_key_terms([{"id": f["id"], "label": f["label"], "detail": f.get("detail", "")} for f in r["facts"]], local), local
        except LiveModelUnavailable as e:
            print(f"  {r['id']}: {e}")
            return r["id"], None, local
    with ThreadPoolExecutor(6) as ex:
        for rid, kt, local in ex.map(run, todo):
            calls.extend(local)
            if kt is not None:
                cache[rid] = kt
    out_file.write_text(json.dumps(cache, indent=1))
    n_f = sum(len(v) for v in cache.values())
    empty = sum(1 for v in cache.values() for g in v.values() if not g)
    print(f"key words for {len(cache)} interrogations, {n_f} details ({empty} with no usable groups); {len(calls)} calls, "
          f"${sum(c.cost_usd for c in calls):.3f}")


if __name__ == "__main__":
    main()

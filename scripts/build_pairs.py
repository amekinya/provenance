"""Training pairs for our matcher from the synthetic interrogations (split by interrogation so test interviews are unseen).
query = the detail; passage = the previous line + the line; label = the line carries the detail (asks/asserts/denies/assents).
Writes data/synth/pairs.jsonl. Usage: python -m scripts.build_pairs"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "synth" / "interrogations.jsonl"
OUT = ROOT / "data" / "synth" / "pairs.jsonl"


def query(f: dict) -> str:
    return f"{f['label']}. {f.get('detail', '')}".strip()


def passage(prev: dict | None, u: dict) -> str:
    who = lambda x: "Investigator" if x["role"] == "interrogator" else "Suspect" if x["role"] == "suspect" else "Other"  # noqa: E731
    head = f"{who(prev)}: {prev['text']}\n" if prev else ""
    return f"{head}{who(u)}: {u['text']}"


def main() -> None:
    recs = [json.loads(x) for x in SRC.read_text().splitlines() if x.strip()]
    rng = random.Random(1)
    ids = [r["id"] for r in recs]
    rng.shuffle(ids)
    n = len(ids)
    split = {i: ("test" if k < n * 0.15 else "dev" if k < n * 0.3 else "train") for k, i in enumerate(ids)}
    rows = []
    for r in recs:
        for k, u in enumerate(r["utterances"]):
            prev = r["utterances"][k - 1] if k else None
            carried = {c["fact"] for c in u["carries"]}
            for f in r["facts"]:
                rows.append({"query": query(f), "passage": passage(prev, u), "label": int(f["id"] in carried), "split": split[r["id"]], "group": r["id"]})
    OUT.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    by = {s: (sum(1 for x in rows if x["split"] == s), sum(x["label"] for x in rows if x["split"] == s)) for s in ("train", "dev", "test")}
    print(f"{len(rows)} pairs (pairs, positives) by split: {by}")


if __name__ == "__main__":
    main()

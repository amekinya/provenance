"""Training pairs for our matcher from the synthetic interrogations (split by interrogation so test interviews are unseen).
query = the detail; passage = the previous line + the line; label = the line carries the detail (asks/asserts/denies/assents).
Writes data/synth/pairs.jsonl. Usage: python -m scripts.build_pairs"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# reads data/synth/interrogations_trainval.jsonl and interrogations_test.jsonl
OUT = ROOT / "data" / "synth" / "pairs.jsonl"


def query(f: dict) -> str:
    return f"{f['label']}. {f.get('detail', '')}".strip()


def passage(prev: dict | None, u: dict) -> str:
    who = lambda x: "Investigator" if x["role"] == "interrogator" else "Suspect" if x["role"] == "suspect" else "Other"  # noqa: E731
    head = f"{who(prev)}: {prev['text']}\n" if prev else ""
    return f"{head}{who(u)}: {u['text']}"


def trainval_split(ids: list[str]) -> dict[str, str]:
    """Interrogation id -> 'train' or 'dev' (20%, fixed seed). The benchmark's --split dev uses the same split."""
    rng = random.Random(1)
    ids = list(ids)
    rng.shuffle(ids)
    return {i: ("dev" if k < len(ids) * 0.2 else "train") for k, i in enumerate(ids)}


def main() -> None:
    tv = ROOT / "data" / "synth" / "interrogations_trainval.jsonl"
    te = ROOT / "data" / "synth" / "interrogations_test.jsonl"
    recs = [json.loads(x) for x in tv.read_text().splitlines() if x.strip()]
    tests = [json.loads(x) for x in te.read_text().splitlines() if x.strip()] if te.exists() else []
    split = trainval_split([r["id"] for r in recs])
    split |= {r["id"]: "test" for r in tests}  # test interrogations come from a different writer model
    recs += tests
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

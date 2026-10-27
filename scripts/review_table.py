"""A table for an independent reviewer: every detail's verdict and the lines it rests on.
Writes data/cases/<case>/review_table.md. Usage: python -m scripts.review_table dassey"""
from __future__ import annotations

import json
import sys

from scripts.ingest import case_dir


def main() -> None:
    cd = case_dir(sys.argv[1] if len(sys.argv) > 1 else "dassey")
    rep = json.loads((cd / "audit.json").read_text())
    text = {u["id"]: u["text"] for u in json.loads((cd / "utterances.json").read_text())}
    run = json.loads((cd / "run.json").read_text()) if (cd / "run.json").exists() else {}
    cell = lambda ev: f"{ev['utterance']} (p. {ev.get('page') or '?'}): {' '.join(text[ev['utterance']].split())[:160]}" if ev else "none"  # noqa: E731
    md = [f"# Review table: {rep['case']['title']}", "", f"Headline: {rep['headline']}", "",
          f"Run: {run.get('total_seconds', '?')} s, ${run.get('cost_usd', '?')}, labeler {run.get('labeler', '?')}, "
          f"verifier reasoning {run.get('verifier_reasoning', '?')}, deciding lines not re-checked: {run.get('unverified', '?')}", "",
          "| Fact | Label | Verdict | First investigator line | First suspect line |", "|---|---|---|---|---|"]
    for f in rep["facts"]:
        md.append(f"| {f['fact']['id']} | {f['fact']['label']} | {f['origin_label']} | {cell(f['first_interrogator'])} | {cell(f['first_suspect'])} |".replace("\n", " "))
    (cd / "review_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

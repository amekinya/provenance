"""Draft a case's key-detail list from its case documents with Nemotron 3 Ultra (counsel reviews and edits it).
Writes data/cases/<id>/facts_draft.json. Usage: python -m scripts.facts dassey"""
from __future__ import annotations

import json
import subprocess
import sys

from app.ai import tasks
from app.ai.client import CallRecord
from scripts.ingest import case_dir


def main() -> None:
    cd = case_dir(sys.argv[1])
    cfg = json.loads((cd / "case.json").read_text())
    text = "\n\n".join(subprocess.run(["pdftotext", "-layout", str(cd / d), "-"], capture_output=True, text=True, check=True).stdout
                       for d in cfg.get("case_documents", []))
    calls: list[CallRecord] = []
    facts = tasks.draft_facts(text, calls)
    for i, f in enumerate(facts, 1):
        f["id"] = f"F{i:02d}"
    (cd / "facts_draft.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))
    print(f"{len(facts)} facts; ${sum(c.cost_usd for c in calls):.4f}")


if __name__ == "__main__":
    main()

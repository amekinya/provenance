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
    from app.engine.ledger import pages_of  # noqa: PLC0415
    cell = lambda ev: (f"{ev['utterance']} ({pages_of(ev)}): " + (f"[{ev['read_aloud']}] " if ev.get("read_aloud") else "")  # noqa: E731
                       + " ".join((ev.get("quote") or text[ev["utterance"]]).split())[:160]) if ev else "none"
    md = [f"# Review table: {rep['case']['title']}", "", f"Headline: {rep['headline']}", "",
          f"Run: {run.get('total_seconds', '?')} s, ${run.get('cost_usd', '?')}, labeler {run.get('labeler', '?')}, "
          f"verifier reasoning {run.get('verifier_reasoning', '?')}, deciding lines not re-checked: {run.get('unverified', '?')}", "",
          "| Fact | Label | Verdict | First investigator line | First suspect line | Note |", "|---|---|---|---|---|---|"]
    for f in rep["facts"]:
        note = f["why"]  # the card's explanation, with its qualifiers
        more = [x for x in f.get("investigator_lines_before", []) if not f["first_interrogator"] or x["utterance"] != f["first_interrogator"]["utterance"]]
        for n in f.get("court_notes", []):
            note += " Court: " + n["text"]
        if f.get("earlier_related_mentions"):
            note += " Earlier related investigator mentions: " + "; ".join(
                f"{x['utterance']} ({pages_of(x)}): {' '.join(x['quote'].split())[:60]}" for x in f["earlier_related_mentions"]) + "."
        if more:
            note += (" " if note else "") + "Also before his first statement: " + "; ".join(f"{x['utterance']} ({pages_of(x)})" for x in more) + "."
        if f.get("nearby_prompts"):
            note += (" " if note else "") + "Nearby investigator prompts: " + "; ".join(
                f"{x['utterance']} ({pages_of(x)}): {' '.join(x['quote'].split())[:60]}" for x in f["nearby_prompts"]) + "."
        for w in f.get("label_words_from_investigators", []):
            note += (" " if note else "") + f"'{w['word']}' came from investigators: " + "; ".join(f"{x['utterance']} ({pages_of(x)})" for x in w["lines"]) + "."
        if f.get("related_statements"):
            note += (" " if note else "") + "His related earlier statements: " + "; ".join(
                f"{x['utterance']} ({pages_of(x)}): {' '.join(x['quote'].split())[:70]}" for x in f["related_statements"]) + "."
        if f.get("his_responses"):
            note += (" " if note else "") + "His answers when it was put to him: " + "; ".join(
                f"{x['utterance']} ({pages_of(x)}): {' '.join(x['quote'].split())[:60]}" for x in f["his_responses"]) + "."
        md.append(f"| {f['fact']['id']} | {f['fact']['label']} | {f['origin_label']} | {cell(f['first_interrogator'])} | {cell(f['first_suspect'])} | {note} |".replace("\n", " "))
    memo = cd / "memo.json"
    if memo.exists():
        import re  # noqa: PLC0415
        pages = {str(u["page"]) for u in json.loads((cd / "utterances.json").read_text())}
        text = re.sub(r"Decision and Order,? pp?\. ?[\w\u2013-]+", "", memo.read_text())  # the court's pages, not the transcripts'
        cited = set(re.findall(r"pp?\. ?(\w+)", text)) | set(re.findall(r"pp\. \w+\u2013(\w+)", text))
        md += ["", f"Memo page citations: {len(cited)} distinct; not in the transcripts: {sorted(cited - pages) or 'none'}"]
    (cd / "review_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

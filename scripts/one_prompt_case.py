"""The one-prompt baseline on a real case: Nemotron 3 Ultra reads each session's whole transcript in ONE prompt and names,
per detail, the first investigator line and the first suspect line that state it. Code combines the sessions in
order (the same rule the ledger uses) and compares the result with the audit.
Writes data/cases/<case>/one_prompt.json. Usage: python -m scripts.one_prompt_case dassey
"""
from __future__ import annotations

import argparse
import json
import time

from app.ai.client import CallRecord, chat_json
from app.config import settings
from scripts.ingest import case_dir

SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "object", "properties": {
        "fact": {"type": "string", "description": "the detail's id exactly as given"},
        "first_investigator_line": {"type": "string", "description": "id of the first investigator line that names the detail (a question counts), or empty"},
        "first_suspect_line": {"type": "string", "description": "id of the first suspect line that states the detail or agrees to it, or empty"}},
        "required": ["fact", "first_investigator_line", "first_suspect_line"], "additionalProperties": False}}},
    "required": ["facts"], "additionalProperties": False,
}


def ask(session_utts: list[dict], facts: list[dict], calls: list[CallRecord]) -> dict[str, dict]:
    record = "\n".join(f"[{u['id']}] {u['speaker']} ({u['role']}): {u['text']}" for u in session_utts)
    fact_list = "\n".join(f"{f['id']}: {f['label']}" for f in facts)
    out = chat_json(settings.ultra, [{"role": "user", "content":
        f"Interrogation transcript:\n{record}\n\nCrime details:\n{fact_list}\n\nFor EVERY detail id: the id of the first investigator line "
        "that names the detail (a question naming it counts), and the id of the first suspect line that states the detail or agrees to "
        "it ('yeah' to a question naming it counts; a denial doesn't). Use an empty string when there is none."}],
        schema=SCHEMA, purpose="One-prompt baseline (real case)", reasoning=True, max_tokens=16000, calls=calls)
    return {r["fact"].split(":")[0].strip(): r for r in out.get("facts", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    a = ap.parse_args()
    cd = case_dir(a.case)
    cfg = json.loads((cd / "case.json").read_text())
    utts = sorted(json.loads((cd / "utterances.json").read_text()), key=lambda u: u["seq"])
    facts = json.loads((cd / "facts.json").read_text())
    seq = {u["id"]: u["seq"] for u in utts}
    role = {u["id"]: u["role"] for u in utts}
    calls: list[CallRecord] = []
    t0 = time.perf_counter()
    first: dict[str, dict] = {f["id"]: {"inv": None, "sus": None} for f in facts}
    per_session = {}
    for s in cfg["sessions"]:
        su = [u for u in utts if u["session"] == s["id"]]
        if not su:
            continue
        ans = ask(su, facts, calls)
        per_session[s["id"]] = ans
        for fid, r in ans.items():
            if fid not in first:
                continue
            for key, want, field in (("inv", "interrogator", "first_investigator_line"), ("sus", "suspect", "first_suspect_line")):
                lid = (r.get(field) or "").strip()
                if lid in seq and role[lid] == want and (first[fid][key] is None or seq[lid] < seq[first[fid][key]]):
                    first[fid][key] = lid
    audit = json.loads((cd / "audit.json").read_text()) if (cd / "audit.json").exists() else {"facts": []}
    ours = {}
    for fl in audit["facts"]:
        fi, fs = fl["first_interrogator"], fl["first_suspect"]
        ours[fl["fact"]["id"]] = {"first": "never" if not fs else ("interrogator" if fi and fi["seq"] < fs["seq"] else "suspect"),
                                  "inv": fi and fi["utterance"], "sus": fs and fs["utterance"], "origin": fl["origin"]}
    rows, agree = [], 0
    for f in facts:
        x = first[f["id"]]
        verdict = "never" if not x["sus"] else ("interrogator" if x["inv"] and seq[x["inv"]] < seq[x["sus"]] else "suspect")
        o = ours.get(f["id"], {})
        same = verdict == o.get("first")
        agree += same
        rows.append({"fact": f["id"], "label": f["label"], "one_prompt": verdict, "one_prompt_inv": x["inv"], "one_prompt_sus": x["sus"],
                     "audit": o.get("first"), "audit_origin": o.get("origin"), "audit_inv": o.get("inv"), "audit_sus": o.get("sus"), "agree": same})
    out = {"model": settings.ultra.model_id, "sessions": len(per_session), "facts": len(facts), "agree": agree,
           "cost_usd": round(sum(c.cost_usd for c in calls), 4), "seconds": round(time.perf_counter() - t0, 1),
           "tokens_in": sum(c.tokens_in for c in calls), "rows": rows, "raw": per_session}
    (cd / "one_prompt.json").write_text(json.dumps(out, indent=1))
    print(f"one prompt per session: {agree}/{len(facts)} details get the same first speaker as the audit; ${out['cost_usd']}, "
          f"{out['tokens_in']} tokens in, {out['seconds']} s")
    for r in rows:
        if not r["agree"]:
            print(f"  {r['fact']} {r['label'][:40]}: one prompt says {r['one_prompt']} (inv {r['one_prompt_inv']}, sus {r['one_prompt_sus']}); "
                  f"audit says {r['audit']} (inv {r['audit_inv']}, sus {r['audit_sus']})")


if __name__ == "__main__":
    main()

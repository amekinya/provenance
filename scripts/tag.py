"""Label every sentence with the case details it carries, verify the sentences that decide verdicts, and save the audit.

1. Nemotron 3 Ultra (reasoning off) tags sentences in parallel chunks (with a few lines of context). Lightning is an
   option; on our diagnostic interrogations it missed about half of the suspect's adoptions.
2. Code drops any label whose quote isn't really in the sentence, and adds the bare agreements: a suspect's "yeah"
   right after an investigator line carrying a detail adopts that detail.
3. The ledger (plain code) finds each detail's first investigator mention and first suspect adoption.
4. Nemotron 3 Ultra re-reads only those deciding sentences; rejected ones are dropped and the ledger re-runs,
   until every deciding sentence has been checked (at most 8 rounds).
5. Safety search: before the suspect is credited with any detail, code searches EVERY earlier investigator line for the
   detail's words and Ultra checks each candidate, so a labeler miss can't turn an investigator's detail into the
   suspect's.
6. Ultra writes a memo from the ledger's findings only.
Writes mentions.json, audit.json, memo.json and run.json (calls, cost, timings).
Usage: python -m scripts.tag dassey [--chunk 18] [--no-memo]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from app.ai import tasks
from app.ai.client import CallRecord, LiveModelUnavailable
from app.engine.ledger import ASSENT, audit, has_terms, load_case, plain, spoken, term_window
from scripts.ingest import case_dir


def is_assent_text(text: str) -> bool:
    t = plain(text)
    return bool(ASSENT.match(spoken(t)) or re.fullmatch(r"\W*\(?\s*(\w+\s+)?nods?[^)]*\)?\W*(yeah|yes|mm[- ]?huh)?\W*", t.strip(), re.I)
                or re.match(r"^\W*\((\w+\s+)?nods?[^)]*\)\s*\W*(yeah|yes|mm[- ]?huh|uh[- ]?huh)?\W*$", t.strip(), re.I))


def load(cid: str) -> tuple[dict, list[dict], list[dict]]:
    cd = case_dir(cid)
    cfg = json.loads((cd / "case.json").read_text())
    utts = json.loads((cd / "utterances.json").read_text())
    facts = json.loads((cd / "facts.json").read_text())
    return cfg, utts, facts


def case_dict(cfg: dict, utts: list[dict], facts: list[dict], mentions: list[dict]) -> dict:
    fact_fields = ("id", "label", "detail", "source", "public_before", "holdback", "planted", "prior_statement", "must")
    return {"id": cfg["id"], "title": cfg["title"], "suspect": cfg["suspect"], "notes": cfg.get("notes", []),
            "sessions": [{k: s.get(k, d) for k, d in (("id", ""), ("label", ""), ("day", ""), ("recorded", True), ("source", ""))}
                         for s in cfg["sessions"]],
            "utterances": utts, "facts": [{k: f.get(k) for k in fact_fields if k in f} for f in facts], "mentions": mentions}


GRAPHIC = re.compile(r"\b(stab\w*|throat|rap(e|ed|ing)|naked|blood\w*|bones?|flesh|body parts?|skull|toes?|choke\w*|cut (her|off)|shot her|kill\w*)\b", re.I)


def tag_all(utts: list[dict], facts: list[dict], chunk: int, calls: list[CallRecord], failed: list[str] | None = None,
            spec=None) -> tuple[list[dict], dict[str, bool]]:
    """The labeler (Ultra by default) reads the record in chunks that never cross a session boundary. Chunks that fail twice are listed in
    `failed`; their lines keep a code-based graphic flag so nothing graphic shows unmasked."""
    order = sorted(utts, key=lambda u: u["seq"])
    jobs = []
    i = 0
    while i < len(order):
        sess = order[i]["session"]
        j = i
        while j < len(order) and j - i < chunk and order[j]["session"] == sess:
            j += 1
        tgt = order[i:j]
        ctx = [u for u in order[max(0, i - 4):i] if u["session"] == sess]
        jobs.append((tgt, ctx))
        i = j

    def run(job):
        tgt, ctx = job
        local: list[CallRecord] = []
        slim = lambda xs: [{"id": u["id"], "speaker": f"{u['speaker']} ({u['role']})", "text": u["text"]} for u in xs]  # noqa: E731
        for attempt in range(2):
            try:
                rows = tasks.tag_mentions(slim(tgt), slim(ctx), facts, local, spec=spec)
                return rows, local
            except LiveModelUnavailable as e:
                if attempt == 1:
                    print(f"  chunk at {tgt[0]['id']} failed: {e}")
                    if failed is not None:
                        failed.append(tgt[0]["id"])
        return [], local

    mentions, graphic = [], {}
    text_of = {u["id"]: u["text"] for u in utts}
    dropped = 0
    with ThreadPoolExecutor(8) as ex:
        for rows, local in ex.map(run, jobs):
            calls.extend(local)
            for r in rows:
                graphic[r["id"]] = bool(r.get("graphic")) or bool(GRAPHIC.search(text_of.get(r["id"], "")))
                for m in r["mentions"]:
                    if m["kind"] == "assents" and not is_assent_text(text_of[r["id"]]):
                        # "Yeah, the red one" is a statement, not a bare agreement: keep it as one only if the quoted
                        # words are really there and say more than "yeah"; otherwise drop it.
                        if tasks.quote_in(m["quote"], text_of[r["id"]]) and not is_assent_text(m["quote"]):
                            m["kind"] = "asserts"
                        else:
                            dropped += 1
                            continue
                    if m["kind"] != "assents" and not tasks.quote_in(m["quote"], text_of[r["id"]]):
                        dropped += 1  # code check: the quote isn't in the sentence
                        continue
                    mentions.append({"utterance": r["id"], "fact": m["fact"], "kind": m["kind"], "quote": m["quote"], "by": "model", "verified": None})
    print(f"tagged {len(utts)} sentences in {len(jobs)} chunks: {len(mentions)} labels kept, {dropped} dropped (quote not in sentence)")
    return mentions, graphic


def add_code_assents(utts: list[dict], mentions: list[dict]) -> int:
    """A suspect's bare agreement right after an investigator line that carries a detail adopts that detail (the
    labeler often leaves a lone 'yeah' unlabeled). Code adds these; the ledger still only counts an agreement that
    follows the investigator's mention within 3 turns, and Ultra checks it if it decides a verdict."""
    order = sorted(utts, key=lambda u: u["seq"])
    carried: dict[str, set[str]] = {}
    for m in mentions:
        if m["kind"] in ("asks", "asserts") and m.get("verified") is not False:
            carried.setdefault(m["utterance"], set()).add(m["fact"])
    have = {(m["utterance"], m["fact"]) for m in mentions}
    added = 0
    for prev, u in zip(order, order[1:]):
        if u["role"] != "suspect" or prev["role"] != "interrogator" or prev["session"] != u["session"] or not is_assent_text(u["text"]):
            continue
        for fid in sorted(carried.get(prev["id"], ())):
            if (u["id"], fid) not in have:
                mentions.append({"utterance": u["id"], "fact": fid, "kind": "assents", "quote": spoken(u["text"])[:80] or u["text"][:80],
                                 "by": "code", "verified": None})
                have.add((u["id"], fid))
                added += 1
    return added


STOP = set("a an the of to in on at for with and or his her him its their was were is be by from that this it he she they there "
           "then them out into onto over under about after before when what who did does had has have not off".split())


def detail_stems(f: dict) -> list[str]:
    """Distinctive word stems of a detail: its label and aliases, minus function words; 5-letter stems catch plurals."""
    words = re.findall(r"[a-z0-9]+", " ".join([f.get("label", "")] + list(f.get("aliases") or [])).lower())
    stems = {w[:5] for w in words if w not in STOP and (len(w) >= 3 or w.isdigit())}
    return sorted(stems)


def candidate_lines(utts: list[dict], f: dict, before_seq: int) -> list[dict]:
    """Investigator lines before `before_seq` (any session) that could carry the detail. With key terms (`must`), a line
    qualifies only if it passes the same key-term check the ledger applies. Without them, a line qualifies if it contains
    one of the detail's distinctive stems (in a long record, stems on more than 2% of lines don't count as distinctive)."""
    if f.get("must"):
        order = sorted(utts, key=lambda u: u["seq"])
        return [u for i, u in enumerate(order) if u["role"] == "interrogator" and u["seq"] < before_seq and has_terms(term_window(order, i), f["must"])]
    stems = detail_stems(f)
    if len(utts) >= 200:
        texts = [plain(u["text"]).lower() for u in utts]
        cap = max(5, int(0.02 * len(utts)))
        stems = [s for s in stems if sum(1 for t in texts if re.search(rf"\b{re.escape(s)}", t)) <= cap]
    if not stems:
        return []
    rx = re.compile(r"\b(" + "|".join(re.escape(s) for s in stems) + r")", re.I)
    return [u for u in sorted(utts, key=lambda u: u["seq"]) if u["role"] == "interrogator" and u["seq"] < before_seq and rx.search(plain(u["text"]))]


def search_investigator_first(cfg, utts, facts, mentions, calls: list[CallRecord], per_fact: int = 15, passes: int = 3) -> int:
    """Safety search. For every detail the suspect stated or agreed to, look for an investigator line BEFORE the earliest
    known mention that carries it: code finds candidates by the detail's words, earliest first, plus the investigator
    lines just before the suspect's first adoption (a short answer like 'In the garage.' usually echoes the question
    right before it, whatever its words); Ultra checks each one.
    Confirmed lines are added as verified investigator mentions; rejected ones are kept as rejected so they're not
    asked again. Returns how many investigator mentions were added."""
    by_id = {u["id"]: u for u in utts}
    order = sorted(utts, key=lambda u: u["seq"])
    prev_of = {u["id"]: (order[i - 1]["text"] if i and order[i - 1]["session"] == u["session"] else "") for i, u in enumerate(order)}
    pos = {u["id"]: i for i, u in enumerate(order)}
    fact_of = {f["id"]: f for f in facts}
    added = 0
    for _ in range(passes):
        rep = audit(load_case(case_dict(cfg, utts, facts, mentions)))
        asked = {(m["utterance"], m["fact"]) for m in mentions}
        items = []
        for fl in rep["facts"]:  # noqa: PLR1702
            fs, fi = fl["first_suspect"], fl["first_interrogator"]
            # Public, planted and prior-statement details get their verdict before investigator order is considered.
            if not fs or fl["origin"] in ("public", "planted_adopted") or fl["fact"].get("prior_statement"):
                continue
            limit = min(fs["seq"], fi["seq"]) if fi else fs["seq"]
            fid = fl["fact"]["id"]
            p = pos[fs["utterance"]]
            just_before = [u for u in order[max(0, p - 4):p] if u["role"] == "interrogator" and u["session"] == fs["session"] and u["seq"] < limit
                           and (not fact_of[fid].get("must") or has_terms(term_window(order, pos[u["id"]]), fact_of[fid]["must"]))]
            kw = [u for u in candidate_lines(utts, fact_of[fid], limit) if (u["id"], fid) not in asked][:per_fact]
            pool = {u["id"]: u for u in kw + [u for u in just_before if (u["id"], fid) not in asked]}
            cands = sorted(pool.values(), key=lambda u: u["seq"])
            items += [{"key": f"{u['id']}|{fid}", "fact_label": fact_of[fid]["label"], "fact_detail": fact_of[fid].get("detail", ""),
                       "speaker": f"{u['speaker']} ({u['role']})", "text": u["text"], "prev": prev_of[u["id"]]} for u in cands]
        if not items:
            break
        results: dict[str, dict] = {}

        def run(b):
            local: list[CallRecord] = []
            try:
                return tasks.verify(b, local), local
            except LiveModelUnavailable as e:
                print(f"  safety check failed: {str(e)[:140]}")
                return {}, local
        with ThreadPoolExecutor(4) as ex:
            for res, local in ex.map(run, [items[i:i + 3] for i in range(0, len(items), 3)]):
                calls.extend(local)
                results.update(res)
        confirmed = 0
        for it in items:
            r = results.get(it["key"])
            if r is None:
                continue
            uid, fid = it["key"].split("|")
            ok = bool(r["carries_detail"]) and r["kind"] in ("asks", "asserts")
            mentions.append({"utterance": uid, "fact": fid, "kind": r["kind"] if ok else "asks", "quote": by_id[uid]["text"][:160],
                             "by": "search", "verified": ok})
            confirmed += ok
        added += confirmed
        print(f"safety search: {len(items)} earlier investigator lines checked, {sum(1 for i in items if i['key'] in results)} answered, "
              f"{confirmed} carry the detail")
    return added


DENY = __import__("re").compile(r"^\W*(no\b|nope|nah|never|i don'?t know|i dunno|i didn'?t|i don'?t remember|not that i)", __import__("re").I)


def tag_with_matcher(utts: list[dict], facts: list[dict], calls: list[CallRecord], threshold: float = 0.5) -> tuple[list[dict], dict[str, bool]]:
    """Our fine-tuned matcher (Nebius endpoint) scores every sentence x detail pair; code assigns the kind;
    Ultra later verifies the sentences that decide verdicts, exactly as with the Lightning labeler."""
    from app.ai import matcher  # noqa: PLC0415
    from app.engine.ledger import ASSENT, spoken  # noqa: PLC0415
    from scripts.build_pairs import passage, query  # noqa: PLC0415

    order = sorted(utts, key=lambda u: u["seq"])
    pairs, keys = [], []
    for k, u in enumerate(order):
        prev = order[k - 1] if k and order[k - 1]["session"] == u["session"] else None
        for f in facts:
            pairs.append((query(f), passage(prev, u)))
            keys.append((u, f))
    scores = matcher.score("provenance-matcher", pairs, calls)
    mentions = []
    for (u, f), sc in zip(keys, scores):
        if sc < threshold:
            continue
        t = spoken(u["text"])
        kind = ("asks" if "?" in u["text"] else "asserts") if u["role"] == "interrogator" else \
            "assents" if ASSENT.match(t) else "denies" if DENY.match(t) else "asserts"
        mentions.append({"utterance": u["id"], "fact": f["id"], "kind": kind, "quote": t[:160], "by": "matcher", "verified": None, "score": round(sc, 3)})
    print(f"matcher scored {len(pairs)} pairs; {len(mentions)} above {threshold}")
    return mentions, {}


def verify_deciding(cfg, utts, facts, mentions, calls: list[CallRecord], rounds: int = 8) -> int:
    """Ultra re-reads the lines that decide verdicts; rejected labels are dropped and the ledger re-runs, until every
    deciding line has been checked or `rounds` is used up. Returns how many deciding lines are still unchecked."""
    by_id = {u["id"]: u for u in utts}
    order = sorted(utts, key=lambda u: u["seq"])
    prev_of = {u["id"]: (order[i - 1]["text"] if i and order[i - 1]["session"] == u["session"] else "") for i, u in enumerate(order)}
    fact_of = {f["id"]: f for f in facts}
    for rnd in range(rounds):
        rep = audit(load_case(case_dict(cfg, utts, facts, mentions)))
        deciding = set()
        for f in rep["facts"]:
            for ev in (f["first_interrogator"], f["first_suspect"]):
                if ev:
                    deciding.add((ev["utterance"], f["fact"]["id"]))
        todo = [m for m in mentions if (m["utterance"], m["fact"]) in deciding and m["verified"] is None]
        if not todo:
            print(f"verification done after {rnd} round(s)")
            return 0
        items = [{"key": f"{m['utterance']}|{m['fact']}", "fact_label": fact_of[m["fact"]]["label"], "fact_detail": fact_of[m["fact"]].get("detail", ""),
                  "speaker": f"{by_id[m['utterance']]['speaker']} ({by_id[m['utterance']]['role']})", "text": by_id[m["utterance"]]["text"],
                  "prev": prev_of[m["utterance"]]} for m in todo]
        results: dict[str, dict] = {}
        batches = [items[i:i + 3] for i in range(0, len(items), 3)]  # small batches keep Ultra's reasoning short enough to finish

        def run(b):
            local: list[CallRecord] = []
            try:
                return tasks.verify(b, local), local
            except LiveModelUnavailable as e:
                print(f"  verifier failed: {str(e)[:140]}")
                return {}, local

        with ThreadPoolExecutor(4) as ex:
            for res, local in ex.map(run, batches):
                calls.extend(local)
                results.update(res)
        rejected = 0
        for m in todo:
            r = results.get(f"{m['utterance']}|{m['fact']}")
            if r is None:
                continue
            m["verified"] = bool(r["carries_detail"])
            if m["verified"] and r["kind"] != m["kind"]:
                m["kind"] = r["kind"]
            rejected += not m["verified"]
        got = sum(1 for m in todo if m["verified"] is not None)
        print(f"round {rnd + 1}: {len(todo)} deciding labels to check, {got} checked, {rejected} rejected")
        if got == 0:
            print("  the verifier answered nothing this round; stopping")
            break
    rep = audit(load_case(case_dict(cfg, utts, facts, mentions)))
    left = sum(1 for f in rep["facts"] for ev in (f["first_interrogator"], f["first_suspect"]) if ev and ev.get("verified") is None)
    print(f"{left} deciding line(s) not re-checked")
    return left


STEP_OF = {"Tag details in sentences": "label", "Verify deciding sentences": "check", "Write the memo": "memo", "Read a transcript page": "read"}


def by_step(calls: list[CallRecord]) -> dict[str, dict]:
    """Calls and cost per pipeline step (labeling and checking can both be Ultra, so the model alone doesn't say which)."""
    out: dict[str, dict] = {}
    for c in calls:
        k = STEP_OF.get(c.purpose, c.role)
        r = out.setdefault(k, {"model": c.model_id, "calls": 0, "seconds": 0.0, "cost_usd": 0.0})
        r["calls"] += 1; r["seconds"] = round(r["seconds"] + c.seconds, 2); r["cost_usd"] = round(r["cost_usd"] + c.cost_usd, 5)
    return out


def check_labels(cfg, utts, facts, mentions, calls: list[CallRecord], rounds: int = 8) -> tuple[int, int, int]:
    """Everything after labeling: code agreements, Ultra's check of the deciding lines, and the safety search.
    Returns (agreements added, investigator lines found by the safety search, deciding lines left unchecked)."""
    agreed = add_code_assents(utts, mentions)
    unverified = verify_deciding(cfg, utts, facts, mentions, calls, rounds=rounds)
    searched = search_investigator_first(cfg, utts, facts, mentions, calls)
    if searched:
        unverified = verify_deciding(cfg, utts, facts, mentions, calls, rounds=rounds)
    return agreed, searched, unverified


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--chunk", type=int, default=18)
    ap.add_argument("--no-memo", action="store_true")
    ap.add_argument("--reaudit", action="store_true", help="recompute the audit from the saved labels (mentions.json) with the current code; no labeling")
    ap.add_argument("--labeler", default="ultra", choices=["ultra", "lightning", "matcher"],
                    help="ultra (default): our diagnostic runs found Lightning missed about half of the suspect's adoptions")
    a = ap.parse_args()
    cd = case_dir(a.case)
    cfg, utts, facts = load(a.case)
    calls: list[CallRecord] = []
    if a.reaudit:
        mentions = json.loads((cd / "mentions.json").read_text())
        known = {u["id"] for u in utts}
        if any(m["utterance"] not in known for m in mentions):
            sys.exit("mentions.json refers to lines that aren't in utterances.json; run the full tag step instead")
        rep = audit(load_case(case_dict(cfg, utts, facts, mentions)))
        (cd / "audit.json").write_text(json.dumps(rep, indent=1))
        print("re-audited from saved labels:", rep["headline"])
        return
    t0 = time.perf_counter()
    failed: list[str] = []
    if a.labeler == "matcher":
        mentions, graphic = tag_with_matcher(utts, facts, calls)
    else:
        from app.config import settings  # noqa: PLC0415
        mentions, graphic = tag_all(utts, facts, a.chunk, calls, failed, spec=settings.ultra if a.labeler == "ultra" else None)
    if not utts:
        sys.exit("no utterances: run ingest first")
    if failed and len(failed) > max(1, len(utts) // 400):
        sys.exit(f"{len(failed)} labeling chunks failed; nothing written. Re-run when the model service is healthy.")
    for u in utts:
        u["sensitive"] = graphic.get(u["id"], bool(GRAPHIC.search(u["text"])) or u.get("sensitive", False))
    (cd / "utterances.json").write_text(json.dumps(utts, indent=1))
    t_tag = time.perf_counter() - t0
    agreed, searched, unverified = check_labels(cfg, utts, facts, mentions, calls)
    print(f"code added {agreed} bare agreements; the safety search found {searched} earlier investigator lines")
    (cd / "mentions.json").write_text(json.dumps(mentions, indent=1))
    rep = audit(load_case(case_dict(cfg, utts, facts, mentions)))
    (cd / "audit.json").write_text(json.dumps(rep, indent=1))
    print(rep["headline"])
    if not a.no_memo:
        try:
            memo = tasks.write_memo(rep, calls)
            (cd / "memo.json").write_text(json.dumps(memo, indent=1))
        except LiveModelUnavailable as e:
            print("memo unavailable:", e)
    roles = {}
    for c in calls:
        r = roles.setdefault(c.role, {"model": c.model_id, "calls": 0, "seconds": 0.0, "cost_usd": 0.0})
        r["calls"] += 1; r["seconds"] += c.seconds; r["cost_usd"] += c.cost_usd
    run = {"labeler": a.labeler, "read_by": "super_vl", "failed_chunks": failed, "unverified": unverified, "safety_search_added": searched,
           "code_agreements_added": agreed, "verifier_reasoning": tasks.VERIFY_REASONING, "by_step": by_step(calls),
           "tag_seconds": round(t_tag, 1), "total_seconds": round(time.perf_counter() - t0, 1),
           "cost_usd": round(sum(c.cost_usd for c in calls), 4), "by_model": roles,
           "calls": [asdict(c) | {"seconds": round(c.seconds, 2), "cost_usd": round(c.cost_usd, 5)} for c in calls]}
    (cd / "run.json").write_text(json.dumps(run, indent=1))
    print(f"done in {run['total_seconds']}s, {len(calls)} model calls, ${run['cost_usd']}")


if __name__ == "__main__":
    main()

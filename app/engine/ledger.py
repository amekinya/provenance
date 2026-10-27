"""The provenance ledger: who said each crime fact first, and what that means for corroboration.

Models only label sentences (which fact, asked or asserted). Everything here is plain code you can test and audit:
the order of the record decides, never a model's opinion. This is a corroboration audit for counsel. It never says a
confession is false; it says which details can't show inside knowledge, and why.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from datetime import date

from app.engine.models import Case, Fact, Mention, Utterance

# Origin of a fact in the record (one per fact).
ORIGINS = {
    "interrogator_first": "Investigators said it first",
    "suspect_first": "Suspect said it first",
    "prior_contact": "May trace to an earlier statement with no public transcript",
    "public": "Reported publicly before the interview",
    "planted_adopted": "Suspect adopted a detail investigators knew was false",
    "never_adopted": "Suspect never adopted it",
    "not_mentioned": "Not mentioned in the record",
}
# What that means for using the detail as corroboration of guilt.
CORROBORATION = {
    "interrogator_first": "cannot_corroborate",
    "public": "cannot_corroborate",
    "planted_adopted": "cannot_corroborate",
    "prior_contact": "unclear",
    "suspect_first": "may_corroborate",
    "never_adopted": "not_adopted",
    "not_mentioned": "not_adopted",
}

# Investigator lines that point to an earlier conversation we can't read ("the jeans Tom talked to you about the other night").
# Narrow on purpose: a reference to earlier in the SAME taped interview ("you told me before that...") isn't one.
PRIOR_CONTACT = re.compile(
    r"\b(talked (to|with) you about the other (day|night)|you (told|said to|mentioned to) (us|me|tom|mark|him|them|the officer)\b[^.?!]{0,40}"
    r"\b(the other (day|night)|last (week|time|night)|at (the|your) (school|station|house))|when (we|you|they) (talked|spoke) (to you )?(the other|last|at the)|"
    r"off (the )?tape|before (we|the) (turned|started) the (tape|recorder))", re.I)
# Bare agreement: the suspect adopts what was put to him without adding anything. Transcripts spell it many ways.
ASSENT = re.compile(r"^\W*(yeah|yes|yep|yup|ya|yah|uh[- ]?huh|mm+[- ]?(hmm|huh|hum)|mim huh|mhm|ok(ay)?|right|i guess|sure|nods)\W*$", re.I)
NEGATIVE = {"no", "nope", "nah", "uh uh", "uh-uh", "i don't know", "i dunno", "dunno", "i don't remember", "no sir"}
# Interrogation tactics researchers associate with false confessions (Kassin et al. 2010 white paper):
# minimization, promises, 'we already know', and pressure to change answers.
TACTICS = {
    "we_already_know": re.compile(r"\bwe (already )?know\b|\bwe('ve| have) (got|the) (evidence|proof)\b", re.I),
    "minimization": re.compile(r"\b(it'?s (ok|okay|not your fault)|not your fault|we('re| are) not (mad|here to)|you'?re not in trouble|it'?s gonna be (ok|okay|alright))\b", re.I),
    "promises": re.compile(r"\b(we('ll| will) (help|go to bat|stand behind)|you('ll| will) (be|get to) (able to )?go home|honesty (is|will)|set you free)\b", re.I),
    "demand_truth": re.compile(r"\b(be honest|tell (us|me) the truth|that'?s not (the truth|what happened)|come on)\b", re.I),
}


def ordered(case: Case) -> list[Utterance]:
    return sorted(case.utterances, key=lambda u: u.seq)


def plain(text: str) -> str:
    """Straight quotes for matching: transcripts use \u2019 and \u201c, models usually write ' and \"."""
    return text.translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'}))


def spoken(text: str) -> str:
    """The words actually said: stage directions like '(pause)' or '(nods)' removed."""
    return re.sub(r"\s+", " ", re.sub(r"[(\[][^)\]]*[)\]]", " ", plain(text))).strip()


def is_assent(u: Utterance) -> bool:
    return bool(ASSENT.match(spoken(u.text)) or re.fullmatch(r"\W*\(?\s*(\w+\s+)?nods?[^)]*\)?\W*", plain(u.text).strip(), re.I))


def fact_ledger(case: Case, fact: Fact) -> dict:
    """Everything the record says about one fact, in order, and the code's verdict."""
    by_id = {u.id: u for u in case.utterances}
    sessions = {s.id: s for s in case.sessions}
    order = ordered(case)
    pos = {u.id: i for i, u in enumerate(order)}
    ms = sorted((m for m in case.mentions if m.fact == fact.id and m.utterance in by_id and m.verified is not False),
                key=lambda m: by_id[m.utterance].seq)
    events = []
    first_inv = first_sus = None
    inv_before = 0
    inv_pos: list[int] = []  # where investigators put this fact into the record
    for m in ms:
        u = by_id[m.utterance]
        ev = {"utterance": u.id, "seq": u.seq, "session": u.session, "page": u.page, "role": u.role, "speaker": u.speaker,
              "kind": m.kind, "quote": m.quote or u.text[:200], "sensitive": u.sensitive, "verified": m.verified}
        if u.role == "interrogator" and m.kind in ("asserts", "asks"):
            inv_pos.append(pos[u.id])
            if first_inv is None:
                first_inv = ev
            if first_sus is None:
                inv_before += 1
        if u.role == "suspect" and first_sus is None:
            if m.kind == "asserts":
                first_sus = ev
            elif m.kind == "assents":
                # A bare "yeah" adopts a detail only if an investigator just put that detail to him (same session,
                # within the last 3 turns). Otherwise it's not evidence he adopted anything.
                p = pos[u.id]
                if any(p - 3 <= q < p and order[q].session == u.session for q in inv_pos):
                    first_sus = ev
                else:
                    ev["kind_note"] = "agreement without a question about this detail just before it; not counted"
        events.append(ev)

    origin, why = _origin(case, fact, first_inv, first_sus, by_id, sessions)
    if first_inv:
        first_inv["first"] = "interrogator"
    if first_sus:
        first_sus["first"] = "suspect"
    adopted_by_assent = bool(first_sus and first_sus["kind"] == "assents")
    return {
        "fact": asdict(fact) | {"public_before": fact.public_before.isoformat() if fact.public_before else None},
        "origin": origin, "origin_label": ORIGINS[origin], "corroboration": CORROBORATION[origin], "why": why,
        "first_interrogator": first_inv, "first_suspect": first_sus,
        "interrogator_mentions_before_suspect": inv_before if first_sus else None,
        "adopted_by_assent": adopted_by_assent,
        "crossed_sessions": bool(first_inv and first_sus and first_inv["seq"] < first_sus["seq"] and first_inv["session"] != first_sus["session"]),
        "events": events,
    }


def _origin(case: Case, fact: Fact, first_inv: dict | None, first_sus: dict | None, by_id: dict, sessions: dict) -> tuple[str, str]:
    if first_inv is None and first_sus is None:
        return "not_mentioned", "Nobody mentions this detail in the transcripts."
    if first_sus is None:
        return "never_adopted", f"Investigators raised it ({_cite(first_inv, sessions)}), but the suspect never stated or agreed to it."
    if fact.planted:
        return "planted_adopted", (f"Investigators used this as a false detail; the suspect went along with it ({_cite(first_sus, sessions)}). "
                                   "Read the exact lines: a hedged answer is not the same as adopting it.")
    inv_first = first_inv is not None and first_inv["seq"] < first_sus["seq"]
    also = f" Investigators also mentioned it before he did ({_cite(first_inv, sessions)})." if inv_first else ""
    s_day = sessions[first_sus["session"]].day if first_sus["session"] in sessions else min((s.day for s in sessions.values()), default=None)
    if fact.public_before and s_day and fact.public_before < s_day:
        return "public", f"It was reported publicly on {fact.public_before:%b %d, %Y}, before the suspect first said it ({_cite(first_sus, sessions)}).{also}"
    if fact.prior_statement:
        return "prior_contact", (f"The record shows he described this earlier, in a statement with no public transcript: {fact.prior_statement}. "
                                 f"In the transcripts it first appears at {_cite(first_inv if inv_first else first_sus, sessions)}.")
    if inv_first:
        cue = _prior_cue(case, first_inv)
        if cue:
            return "prior_contact", (f"Investigators first said it ({_cite(first_inv, sessions)}), but in words that may refer to something he said "
                                     f"earlier with no public transcript (p. {cue.page or '?'}: \"{plain(cue.text)[:90]}\").")
        how = "by agreeing to a question" if first_sus["kind"] == "assents" else "after that"
        return "interrogator_first", (f"Investigators first said it ({_cite(first_inv, sessions)}); the suspect first adopted it {how} "
                                      f"({_cite(first_sus, sessions)}).")
    cue = _prior_cue(case, first_sus, before=12)
    if cue:
        return "prior_contact", (f"The suspect said it first in the transcripts ({_cite(first_sus, sessions)}), but just before, the investigator "
                                 f"may have referred to an earlier conversation with no public transcript (p. {cue.page or '?'}: \"{plain(cue.text)[:90]}\").")
    return "suspect_first", f"The suspect said it first ({_cite(first_sus, sessions)}), and nothing earlier in the transcripts supplies it."


def _prior_cue(case: Case, ev: dict, before: int = 2) -> Utterance | None:
    """An investigator line at or just before this event that points to an earlier conversation we can't read."""
    window = [u for u in ordered(case) if u.session == ev["session"] and ev["seq"] - before <= u.seq <= ev["seq"]]
    return next((u for u in reversed(window) if u.role == "interrogator" and PRIOR_CONTACT.search(plain(u.text))), None)


def _cite(ev: dict, sessions: dict) -> str:
    s = sessions.get(ev["session"])
    return f"{s.label if s else ev['session']}, p. {ev['page'] or '?'}"


def session_stats(case: Case) -> list[dict]:
    """Per-session pressure measures, all counted by code."""
    out = []
    mentions_by_u: dict[str, list[Mention]] = {}
    for m in case.mentions:
        if m.verified is not False:
            mentions_by_u.setdefault(m.utterance, []).append(m)
    for s in case.sessions:
        us = [u for u in ordered(case) if u.session == s.id]
        inv = [u for u in us if u.role == "interrogator"]
        sus = [u for u in us if u.role == "suspect"]
        words = lambda xs: sum(len(u.text.split()) for u in xs)  # noqa: E731
        questions = [u for u in inv if "?" in u.text]
        fact_questions = [u for u in questions if any(m.kind in ("asks", "asserts") for m in mentions_by_u.get(u.id, []))]
        tactic_counts = {k: sum(1 for u in inv if rx.search(plain(u.text))) for k, rx in TACTICS.items()}
        out.append({
            "session": s.id, "label": s.label, "recorded": s.recorded, "turns": len(us),
            "first_seq": us[0].seq if us else None, "last_seq": us[-1].seq if us else None,
            "interrogator_share_of_words": round(words(inv) / max(words(us), 1), 3),
            "suspect_words_per_turn": round(words(sus) / max(len(sus), 1), 1),
            "suspect_minimal_answers": sum(1 for u in sus if is_assent(u) or spoken(u.text).lower().strip(".!?, ") in NEGATIVE),
            "questions": len(questions), "questions_that_carry_facts": len(fact_questions),
            "tactics": tactic_counts,
        })
    return out


def audit(case: Case) -> dict:
    """The whole report: fact ledgers, session measures and a headline, all decided by code."""
    facts = [fact_ledger(case, f) for f in case.facts]
    stated = [f for f in facts if f["first_suspect"]]
    counts = {k: sum(1 for f in facts if f["origin"] == k) for k in ORIGINS}
    may = [f for f in facts if f["corroboration"] == "may_corroborate"]
    def n(k: int, one: str, many: str) -> str:
        return f"{k} {one if k == 1 else many}"

    extra = []
    if counts["public"]:
        extra.append(f"{n(counts['public'], 'was', 'were')} already public")
    if counts["planted_adopted"]:
        extra.append(f"{n(counts['planted_adopted'], 'was a false detail', 'were false details')} he went along with")
    if counts["prior_contact"]:
        extra.append(f"{n(counts['prior_contact'], 'may trace', 'may trace')} to an earlier statement with no public transcript")
    headline = (f"Of {n(len(stated), 'key detail', 'key details')} {case.suspect or 'the suspect'} stated or agreed to, investigators said "
                f"{counts['interrogator_first']} first" + "".join(f", {e}" for e in extra) +
                f". {n(len(may), 'detail', 'details')} came from him first in the record.")
    return {"case": {"id": case.id, "title": case.title, "suspect": case.suspect, "notes": case.notes},
            "sessions": [asdict(s) | {"day": s.day.isoformat()} for s in case.sessions],
            "headline": headline, "counts": counts, "facts": facts, "session_stats": session_stats(case),
            "utterance_count": len(case.utterances), "mention_count": sum(1 for m in case.mentions if m.verified is not False)}


MENTION_FIELDS = ("utterance", "fact", "kind", "quote", "by", "verified")
UTT_FIELDS = (("id", ""), ("session", ""), ("seq", 0), ("speaker", ""), ("role", "other"), ("text", ""), ("page", ""), ("sensitive", False))


def load_case(d: dict) -> Case:
    """JSON -> Case (dates as ISO strings)."""
    from app.engine.models import Session

    def day(x):
        return date.fromisoformat(x) if x else None

    return Case(
        id=d["id"], title=d["title"], suspect=d["suspect"], notes=d.get("notes", []),
        sessions=[Session(**(s | {"day": day(s["day"])})) for s in d["sessions"]],
        utterances=[Utterance(**{k: u.get(k, dflt) for k, dflt in UTT_FIELDS}) for u in d["utterances"]],
        facts=[Fact(**(f | {"public_before": day(f.get("public_before"))})) for f in d["facts"]],
        mentions=[Mention(**{k: m[k] for k in MENTION_FIELDS if k in m}) for m in d.get("mentions", [])],
    )

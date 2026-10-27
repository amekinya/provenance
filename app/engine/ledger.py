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

# Investigator lines that point to an earlier conversation we can't read: "I talked to ya the other night", "the jeans Tom
# talked to you about", "you told me last time", "we talked last Monday". General rule: a reference to talking with him
# (talked/spoke to you, you told me/us) tied to an earlier time or place (the other night/day, last time/week/night,
# a weekday, at school/the station), or such a time phrase on its own ("the other night"). A bare "you told us before"
# isn't one: it usually refers to earlier in the same taped interview.
_WHEN = (r"(the other (day|night)|last (time|week|night|(mon|tues|wednes|thurs|fri|satur|sun)day)|(on |last |er )*(mon|tues|wednes|thurs|fri|satur|sun)day"
         r"|at (the|your) (school|station|house)|that night at)")
PRIOR_CONTACT = re.compile(
    rf"\b((talked|talkin'?|spoke|spoken) (to|with) (you|ya|him)\b[^.?!]{{0,60}}\b{_WHEN}|(we|i|tom|mark|they) (talked|spoke)\b[^.?!]{{0,40}}\b{_WHEN}"
    rf"|you (told|said to|mentioned to) (us|me|tom|mark|him|them|the officer)\b[^.?!]{{0,40}}\b{_WHEN}|\bthe other night\b"
    rf"|(talked|spoke) (to|with) you about\b[^.?!]{{0,30}}\b(before|earlier)|off (the )?tape|before (we|the) (turned|started) the (tape|recorder))", re.I)
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


# ---------------------------------------------------------------------------------------------------------------------
# Key terms. Each fact can carry `must`: groups of words, written from the fact's own wording (never from the transcript
# lines they would match). A line carries the fact only if it contains one alternate from every group. A bare 'yeah' or a
# short line (8 words or fewer) is read together with the 2 lines before it in the same session, so 'With the shovel.'
# answering 'How'd he break up the bones?' counts. Code checks this after the model labels and after the verifier, so
# it is the last word.
SHORT_WORDS = 8


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", plain(text).lower())


def _matches(tok: str, alt: str) -> bool:
    if tok == alt:
        return True
    forms = {alt + x for x in ("s", "es", "ed", "d", "ing", "er", "ers")}
    if len(alt) >= 3 and alt[-1] not in "aeiouwxy":
        forms |= {alt + alt[-1] + x for x in ("ed", "ing", "er")}  # stab -> stabbed, cut -> cutting
    if alt.endswith("e"):
        forms.add(alt[:-1] + "ing")                                  # choke -> choking
    if alt.endswith("y"):
        forms |= {alt[:-1] + "ies", alt[:-1] + "ied"}
    if alt.endswith("f"):
        forms.add(alt[:-1] + "ves")                                  # knife -> knives
    if alt.endswith("fe"):
        forms.add(alt[:-2] + "ves")
    return tok in forms


def has_terms(text: str, must: list) -> bool:
    """True if the text has at least one alternate from every key-term group (no groups: always true)."""
    toks = _tok(text)
    joined = " " + " ".join(toks) + " "
    for group in must or []:
        hit = False
        for alt in group:
            a = _tok(alt)
            if len(a) > 1:
                hit = f" {' '.join(a)} " in joined
            elif a:
                hit = any(_matches(t, a[0]) for t in toks)
            if hit:
                break
        if not hit:
            return False
    return True


_GENERIC = set("a an the of to in on at for with and or his her him its their was were is be by from that this it he she they there "
               "then them out into onto over under about after before when what who did does had has have not off near found seen "
               "missing used left put taken given wore wearing got made until some avery avery's".split())


def derive_must(label: str) -> list[list[str]]:
    """Key terms made mechanically from a fact label (for facts nobody wrote key terms for): every content word is its
    own group; plural and simple verb endings are tolerated by the matcher."""
    return [[w] for w in dict.fromkeys(t for t in _tok(label) if t not in _GENERIC and (len(t) > 2 or t.isdigit()))]


def _txt(x) -> str:
    return x.text if hasattr(x, "text") else x["text"]


def _sess(x) -> str:
    return x.session if hasattr(x, "session") else x["session"]


def term_window(order: list, i: int, answers_labeled_question: bool = False) -> str:
    """The text a line is judged by for key terms: the line itself, plus the 2 lines before it in the same session when
    it is a bare agreement or a short line, or when it answers a question labeled with the same detail (the line just
    before, by an investigator)."""
    u = order[i]
    t = _txt(u)
    words = spoken(t).split()
    if len(words) <= SHORT_WORDS or ASSENT.match(spoken(t)) or answers_labeled_question:
        prev = [order[k] for k in range(max(0, i - 2), i) if _sess(order[k]) == _sess(u)]
        t = " ".join([_txt(x) for x in prev] + [t])
    return t


def _spans(text: str) -> list[str]:
    """Single sentences and pairs of consecutive sentences: in a long line, key terms must occur close together, so
    'the vehicle' in one sentence and 'back' three sentences later don't make a match."""
    sents = [x for x in re.split(r"(?<=[.?!])\s+|\.{3,}", plain(text)) if x.strip()] or [text]
    return sents + [sents[k] + " " + sents[k + 1] for k in range(len(sents) - 1)]


def carries(order: list, i: int, must: list, answers_labeled_question: bool = False) -> bool:
    """Does line i carry a detail with these key terms? A bare agreement or a short line (8 words or fewer) is read with
    the 2 lines before it in the same session; an investigator's short line only with investigators' lines (so an
    investigator never gets credit for the suspect's words). A long line must have the terms within one or two
    consecutive sentences; a long suspect line answering a labeled question is read with that question."""
    u = order[i]
    t = _txt(u)
    spoken_t = spoken(t)
    short = len(spoken_t.split()) <= SHORT_WORDS or bool(ASSENT.match(spoken_t))
    is_inv = _role(u) == "interrogator"
    prev = [order[k] for k in range(max(0, i - 2), i) if _sess(order[k]) == _sess(u) and (not is_inv or _role(order[k]) == "interrogator")]
    if short:
        if is_inv and not any(has_terms(t, [g]) for g in must):
            return False  # an investigator's short line may borrow context only if it names part of the detail itself
        ctx = [_txt(x) for x in prev]
        if not is_inv:  # his short answer is also read with the investigators' questions about this detail in the last 6 turns
            ctx = [_txt(x) for x in turns_before(order, i, PART_WINDOW) if _role(x) == "interrogator" and any(has_terms(_txt(x), [g]) for g in must)] + ctx
        return has_terms(" ".join(ctx + [t]), must)
    if answers_labeled_question and not is_inv:
        q = _txt(order[i - 1])
        return any(has_terms(q + " " + sp, must) for sp in _spans(t))
    return any(has_terms(sp, must) for sp in _spans(t))


def _role(x) -> str:
    return x.role if hasattr(x, "role") else x["role"]


def answers_labeled(order: list, i: int, labeled: set) -> bool:
    """Is this a suspect line answering, on the very next line, an investigator line labeled with the same detail?"""
    return (i > 0 and _role(order[i]) == "suspect" and _role(order[i - 1]) == "interrogator" and _sess(order[i - 1]) == _sess(order[i])
            and (order[i - 1].id if hasattr(order[i - 1], "id") else order[i - 1]["id"]) in labeled)


def is_assent(u: Utterance) -> bool:
    return bool(ASSENT.match(spoken(u.text)) or re.fullmatch(r"\W*\(?\s*(\w+\s+)?nods?[^)]*\)?\W*", plain(u.text).strip(), re.I))


PART_WINDOW = 6  # investigator lines this close before his first statement can have supplied part of it


def _event(u: Utterance, kind: str, quote: str, verified, by: str = "model") -> dict:
    return {"utterance": u.id, "seq": u.seq, "session": u.session, "page": u.page, "page_end": u.page_end, "role": u.role, "speaker": u.speaker,
            "kind": kind, "quote": quote or u.text[:200], "sensitive": u.sensitive, "verified": verified, "by": by}


def pages_of(ev) -> str:
    """'p. 469' or 'pp. 469-470' for a turn that continues onto the next printed page."""
    a = ev.get("page") if isinstance(ev, dict) else ev.page
    b = ev.get("page_end") if isinstance(ev, dict) else ev.page_end
    return f"pp. {a}\u2013{b}" if b and b != a else f"p. {a or '?'}"


def _split(x) -> bool:
    return bool(x.split if hasattr(x, "split") and not isinstance(x, dict) else x.get("split"))


def turns_before(order: list, p: int, n: int) -> list:
    """The lines of the last n printed turns before position p, same session. A nod split out of an investigator's
    printed turn isn't a turn of its own, so it doesn't use up the window."""
    out, turns = [], 0
    for k in range(p - 1, -1, -1):
        x = order[k]
        if _sess(x) != _sess(order[p]):
            break
        out.append(x)
        if not _split(x):
            turns += 1
            if turns >= n:
                break
    return out[::-1]


def exchange_before(order: list, p: int, must: list, gap: int = 10) -> list:
    """The stretch of questioning about a detail that leads up to line p: walking back in the same session, the lines
    that name part of it (any key-term group, either side), as long as no more than `gap` printed turns pass without
    one. Always includes the last 6 turns."""
    out, quiet = [], 0
    for k in range(p - 1, -1, -1):
        x = order[k]
        if _sess(x) != _sess(order[p]):
            break
        out.append(x)
        if any(has_terms(_txt(x), [g]) for g in must):
            quiet = 0
        elif not _split(x):
            quiet += 1
            if quiet > gap:
                break
    near = turns_before(order, p, PART_WINDOW)
    keep = {(x.id if hasattr(x, "id") else x["id"]) for x in out} | {(x.id if hasattr(x, "id") else x["id"]) for x in near}
    return [x for x in order[max(0, p - len(out) - len(near)):p] if (x.id if hasattr(x, "id") else x["id"]) in keep]


def _echo_groups(order: list, i: int, must: list) -> list:
    """Groups an investigator line repeats from the suspect's turn just before it (an echo of his words, not a mention)."""
    u = order[i]
    k = i - 1
    while k >= 0 and order[k].split:  # look past a nod split out of this same turn
        k -= 1
    if k < 0 or order[k].session != u.session or order[k].role != "suspect":
        return []
    return [g for g in must if has_terms(u.text, [g]) and has_terms(order[k].text, [g])]


NEARBY_TURNS = 10  # investigator prompts this close to his first statement are shown on the card
LONE_WORDS = {"back", "up", "down", "in", "out", "on", "off"}  # only meaningful inside a phrase (same rule as the key terms)


def _about(order: list, i: int, must: list) -> bool:
    """Is line i about this detail: does it name one of its key-term groups, or is it a short line whose 2 lines before
    it (same session) do?"""
    if any(has_terms(_txt(order[i]), [g]) for g in must):
        return True
    short = len(spoken(_txt(order[i])).split()) <= SHORT_WORDS
    return short and any(any(has_terms(_txt(x), [g]) for g in must) for x in turns_before(order, i, 2))


READ_ALOUD = re.compile(r"^\s*\(read aloud by (\w+) from (\w+)'s written statement; printed speaker: (\w+)\)\s*")


def _requote(ev: dict, text: str, must: list) -> dict:
    """Quote the part of a long line that carries the detail (all key-term groups in one or two consecutive sentences,
    else the first sentence naming part of it), not the start of the line. A line read aloud from his written
    statement keeps its disclosure in `read_aloud`."""
    m = READ_ALOUD.match(text)
    body = text[m.end():] if m else text
    if m:
        ev["read_aloud"] = f"read aloud by {m.group(1).title()} from {m.group(2).title()}'s written statement (printed speaker: {m.group(3).title()})"
    if must and len(spoken(body).split()) > SHORT_WORDS:
        sp = _spans(body)
        best = next((x for x in sp if has_terms(x, must)), None) or next((x for x in sp if any(has_terms(x, [g]) for g in must)), None)
        if best:
            ev["quote"] = best.strip()
    elif m:
        ev["quote"] = plain(body)[:200]
    return ev


def _court_notes(fact: Fact, order: list) -> list[dict]:
    """The court's own observations about how a detail came out, each quote followed by its transcript page (code finds
    the line that prints it)."""
    out = []
    norm = lambda t: re.sub(r"\s+", " ", plain(t).replace("\u2019", "'").replace("\u2018", "'")).strip().lower()  # noqa: E731
    for n in fact.court_notes:
        text, lines = n["text"], []
        for q in n.get("quotes", []):
            u = next((x for x in order if norm(q) in norm(x.text)), None)
            if u:
                lines.append(_cite_dict(_line_ev(u)))
                text = text.replace(f"“{q}”", f"“{q}” ({pages_of(u)})", 1)
        out.append({"page": n["page"], "text": f"The Decision (p. {n['page']}) {text}.", "lines": lines})
    return out


def _cite_dict(e) -> dict:
    return {k: e.get(k) for k in ("utterance", "page", "page_end", "session", "speaker", "quote", "by", "read_aloud") if k in e}


def _line_ev(u: Utterance, **extra) -> dict:
    return {"utterance": u.id, "seq": u.seq, "session": u.session, "page": u.page, "page_end": u.page_end, "role": u.role,
            "speaker": u.speaker, "quote": u.text[:200], "sensitive": u.sensitive} | extra


def turns_after(order: list, p: int, n: int) -> list:
    out, turns = [], 0
    for k in range(p + 1, len(order)):
        x = order[k]
        if _sess(x) != _sess(order[p]):
            break
        out.append(x)
        if not _split(x):
            turns += 1
            if turns >= n:
                break
    return out


def fact_ledger(case: Case, fact: Fact) -> dict:
    """Everything the record says about one fact, in order, and the code's verdict.

    With key terms (`fact.must`) the investigator side is decided by code alone: an investigator line carries the
    detail when it has a word from every key-term group (a short line that names part of it is read with the
    investigators' lines just before it). An investigator line that repeats key words from his turn just before it is
    an echo, not a mention. No model can remove a code match; model labels on investigator lines are extra evidence
    for 'named part of it' only. The suspect side comes from model labels (Ultra) checked against the key terms."""
    by_id = {u.id: u for u in case.utterances}
    sessions = {s.id: s for s in case.sessions}
    order = ordered(case)
    pos = {u.id: i for i, u in enumerate(order)}
    ms = sorted((m for m in case.mentions if m.fact == fact.id and m.utterance in by_id and m.verified is not False),
                key=lambda m: by_id[m.utterance].seq)
    events = []
    inv_full: set[str] = set()      # investigator lines carrying every group (code)
    inv_part: dict[str, list] = {}  # investigator lines carrying some groups (code), echoes removed
    echoes: dict[str, list] = {}    # investigator lines repeating his words from the turn before
    if fact.must:
        for i, u in enumerate(order):
            if u.role != "interrogator":
                continue
            echo = _echo_groups(order, i, fact.must)
            if echo:
                echoes[u.id] = echo
            if carries(order, i, fact.must) and not echo:
                inv_full.add(u.id)
            else:
                gs = [g for g in fact.must if has_terms(u.text, [g]) and g not in echo]
                if gs:
                    inv_part[u.id] = gs
    labeled = {m.utterance for m in ms} | inv_full | set(inv_part)
    model_inv = {m.utterance for m in ms if by_id[m.utterance].role == "interrogator" and m.kind in ("asks", "asserts")}
    model_ev = {}
    for m in ms:
        u = by_id[m.utterance]
        ev = _event(u, m.kind, m.quote, m.verified, m.by)
        if fact.must and u.role == "interrogator":
            if u.id not in inv_full:
                ev["kind_note"] = ("repeats his words from the line before; an echo, not counted" if u.id in echoes
                                   else "the line doesn't contain the detail's key terms; not counted")
                events.append(ev)
            else:
                model_ev[u.id] = ev  # shown on the key-word match below
            continue
        if fact.must and not carries(order, pos[u.id], fact.must, answers_labeled(order, pos[u.id], labeled)):
            ev["kind_note"] = "the line doesn't contain the detail's key terms; not counted"
        events.append(ev)
    for uid in inv_full:
        u = by_id[uid]
        ev = _event(u, "asks" if "?" in u.text else "asserts", "", None, "key words")
        if uid in model_ev:
            ev["model_label"] = model_ev[uid]["kind"]
        events.append(ev)
    events.sort(key=lambda e: e["seq"])

    first_inv = first_sus = None
    inv_before = 0
    inv_pos: list[int] = []  # where investigators put this fact into the record
    for ev in events:
        if ev.get("kind_note"):
            continue
        u = by_id[ev["utterance"]]
        if u.role == "interrogator" and ev["kind"] in ("asserts", "asks"):
            inv_pos.append(pos[u.id])
            if first_inv is None:
                first_inv = ev
            if first_sus is None:
                inv_before += 1
        if u.role == "suspect" and first_sus is None:
            if ev["kind"] == "asserts":
                first_sus = ev
            elif ev["kind"] == "assents":
                # A bare "yeah" adopts a detail only if an investigator just put that detail to him (same session,
                # within the last 3 turns). Otherwise it's not evidence he adopted anything.
                p = pos[u.id]
                if any(p - 3 <= q < p and order[q].session == u.session for q in inv_pos):
                    first_sus = ev
                else:
                    ev["kind_note"] = "agreement without a question about this detail just before it; not counted"

    ref = first_sus or first_inv
    reported = reported_groups(fact, sessions[ref["session"]].day if ref and ref["session"] in sessions else None) if fact.must else set()
    partial = _partly_supplied(fact, order, pos, by_id, inv_part, model_inv, ms, first_inv, first_sus, reported)
    origin, why = _origin(case, fact, first_inv, first_sus, by_id, sessions, partial, reported)
    is_supplied = bool(partial and origin == "interrogator_first" and not (first_inv and first_inv["seq"] < first_sus["seq"]))
    is_part = is_supplied and not partial["all"]
    if is_supplied:
        first_inv = partial["event"]
        if not any(e["utterance"] == first_inv["utterance"] for e in events):
            events.append(first_inv)
            events.sort(key=lambda e: e["seq"])
    if first_inv:
        first_inv["first"] = "interrogator"
    if first_sus:
        first_sus["first"] = "suspect"

    # What the card lists besides the deciding lines (code only).
    upto = first_sus["seq"] if first_sus else float("inf")
    inv_lines = [e for e in events if e["role"] == "interrogator" and e["seq"] < upto and not e.get("kind_note")
                 and (e["kind"] in ("asks", "asserts") or e.get("partial"))]
    if is_supplied:
        inv_lines = sorted({e["utterance"]: e for e in inv_lines + partial["lines"]}.values(), key=lambda e: e["seq"])
    if fact.must:  # quote the part of each long line that carries the detail
        for e in events + (partial["lines"] if partial else []):
            _requote(e, by_id[e["utterance"]].text, fact.must)
    related, nearby, label_words, responses, earlier = [], [], [], [], []
    if fact.must and first_sus:
        for e in events:
            if (e["role"] == "suspect" and e["seq"] < upto and e["kind"] != "denies" and e.get("kind_note", "").startswith("the line doesn't")
                    and any(has_terms(by_id[e["utterance"]].text, [g]) for g in fact.must)):
                related.append(e)
        p = pos[first_sus["utterance"]]
        shown = {e["utterance"] for e in inv_lines[:3]}
        nearby = [_requote(_line_ev(x), x.text, fact.must) for x in turns_before(order, p, NEARBY_TURNS) if x.id in inv_part and x.id not in shown]
        # earlier related investigator mentions: the earliest investigator lines before his first statement that name
        # the detail's most specific part (its key-term group on the fewest lines of the record, if that is at most 2%
        # of them), up to 3, if not already shown
        shown |= {e["utterance"] for e in nearby[:5]}
        counts = [sum(1 for x in order if has_terms(x.text, [g])) for g in fact.must]
        rare = [g for g, c in zip(fact.must, counts) if c == min(counts) and c <= 0.02 * len(order)]
        for x in order:
            if len(earlier) >= 3 or x.seq >= upto:
                break
            if x.id in inv_part and x.id not in shown and any(g in inv_part[x.id] for g in rare):
                earlier.append(_requote(_line_ev(x), x.text, fact.must))
        # words of the detail's label that aren't key terms (degree words like 'unconscious'): who said them first nearby
        covered = {a for g in fact.must for a in g}
        extra_words = [w for w in derive_must(fact.label) if w[0] not in LONE_WORDS
                       and not any(_matches(w[0], t) or _matches(t, w[0]) for a in covered for t in _tok(a))]
        around = turns_before(order, p, NEARBY_TURNS) + [order[p]] + turns_after(order, p, NEARBY_TURNS)
        for (w,) in extra_words:
            # only lines about this detail: a line that names one of its key-term groups, or a short line (8 words or
            # fewer) right after one, read with the 2 lines before it as everywhere else
            said = [x for x in around if has_terms(x.text, [[w]]) and _about(order, pos[x.id], fact.must)]
            if said and said[0].role == "interrogator":
                label_words.append({"word": w, "lines": [_cite_dict(_line_ev(x)) for x in said if x.role == "interrogator"][:3]})
    if fact.must and first_inv and not first_sus:  # never adopted: what he answered when it was put to him
        p = pos[first_inv["utterance"]]
        responses = [_cite_dict(_requote(_line_ev(x), x.text, fact.must)) for x in turns_after(order, p, 4) if x.role == "suspect"][:3]
    adopted_by_assent = bool(first_sus and first_sus["kind"] == "assents")
    return {
        "fact": asdict(fact) | {"public_before": fact.public_before.isoformat() if fact.public_before else None},
        "origin": origin,
        "origin_label": "Investigators named part of it first" if is_part else ORIGINS[origin],
        "corroboration": CORROBORATION[origin], "why": why,
        "partly_supplied": is_part, "supplied_across_lines": is_supplied and partial["all"],
        "partial_words": partial["investigator_words"] if is_supplied else [],
        "first_interrogator": first_inv, "first_suspect": first_sus,
        "investigator_lines_before": [_cite_dict(e) for e in inv_lines[:3]],
        "nearby_prompts": [_cite_dict(e) for e in nearby[:5]],
        "earlier_related_mentions": [_cite_dict(e) for e in earlier],
        "court_notes": _court_notes(fact, order),
        "label_words_from_investigators": label_words,
        "related_statements": [_cite_dict(e) for e in related[:3]],
        "related_statements_total": len(related),
        "his_responses": responses,
        "interrogator_mentions_before_suspect": inv_before if first_sus else None,
        "adopted_by_assent": adopted_by_assent,
        "crossed_sessions": bool(first_inv and first_sus and first_inv["seq"] < first_sus["seq"] and first_inv["session"] != first_sus["session"]),
        "events": events,
    }


def _partly_supplied(fact: Fact, order: list, pos: dict, by_id: dict, inv_part: dict, model_inv: set, ms: list,
                     first_inv: dict | None, first_sus: dict | None, reported: set | None = None) -> dict | None:
    """Did investigators name part of the detail (or every part, across separate lines) before he stated all of it?
    Candidates: investigator lines in the exchange leading up to his first statement (the stretch of questioning that
    keeps naming parts of the detail, with no more than 10 turns between such lines, and at least the last 6 turns)
    that carry some key-term groups (code), plus investigator lines the labeler tied to this detail anywhere earlier
    (extra evidence, which can only add investigator credit). A group counts as theirs unless he had said a word of it
    first, in his lines labeled with this detail or in that exchange. Echoes of his previous turn are already removed.
    Groups a report before the interviews already stated (`reported`) are nobody's: only the rest decide."""
    reported = reported or set()
    if not fact.must or not first_sus or (first_inv and first_inv["seq"] < first_sus["seq"]):
        return None
    p = pos[first_sus["utterance"]]
    window = exchange_before(order, p, fact.must)
    cands = {u.id: u for u in window if u.id in inv_part}
    cands |= {uid: by_id[uid] for uid in model_inv if uid in inv_part and by_id[uid].seq < first_sus["seq"]}
    his = [x for x in window if x.role == "suspect"] + [by_id[m.utterance] for m in ms if by_id[m.utterance].role == "suspect"
                                                       and m.kind != "denies" and by_id[m.utterance].seq < first_sus["seq"]]
    theirs: list[tuple] = []  # (line, group), the earliest investigator line for each group
    for k, g in enumerate(fact.must):
        if k in reported:
            continue
        lines = sorted((u for u in cands.values() if g in inv_part[u.id]), key=lambda u: u.seq)
        for u in lines:
            if not any(x.seq < u.seq and has_terms(x.text, [g]) for x in his):
                theirs.append((u, g))
                break
    if not theirs:
        return None
    their_groups = [g for _, g in theirs]
    lines = sorted({u.id: u for u, _ in theirs}.values(), key=lambda u: u.seq)
    inv_words: list[str] = []
    for u, g in theirs:
        inv_words += [w for w in _said(u.text, g) if w not in inv_words]
    his_lines = [x for x in window if x.role == "suspect"] + [order[p]]
    his_words: list[str] = []
    for k, g in enumerate(fact.must):
        if g in their_groups or k in reported:
            continue
        for x in reversed(his_lines):  # his statement first, then his lines just before it
            ws = _said(x.text, g)
            if ws:
                his_words += [w for w in ws if w not in his_words]
                break
    evs = [_line_ev(u, kind="asks", verified=None, partial=True, by="key words") for u in lines]
    per_line = [(evs[k], [w for u2, g in theirs if u2.id == lines[k].id for w in _said(lines[k].text, g)]) for k in range(len(lines))]
    return {"event": evs[0], "lines": evs, "per_line": per_line, "investigator_words": inv_words, "his_words": his_words,
            "all": len(set(map(tuple, their_groups))) == len(fact.must) - len(reported)}


def _said(text: str, group: list[str]) -> list[str]:
    toks = _tok(text)
    joined = " " + " ".join(toks) + " "
    out = []
    for alt in group:
        a = _tok(alt)
        if len(a) > 1 and f" {' '.join(a)} " in joined:
            out.append(" ".join(a))
        elif len(a) == 1:
            out += [t for t in toks if _matches(t, a[0]) and t not in out]
    return list(dict.fromkeys(out))


def reported_groups(fact: Fact, day) -> set:
    """Indices of the key-term groups that reports before `day` state (the court's list counts as before every recorded
    interview: it says these were reported 'long before' the March 1 confession)."""
    out: set = set()
    for r in fact.reports:
        if r.get("court_list") or (r.get("date") and day and date.fromisoformat(r["date"]) < day):
            out |= set(r.get("covers", []))
    return out


def _report_phrase(r: dict) -> str:
    if r.get("court_list"):
        when = f"before {date.fromisoformat(r['before']):%B %-d, %Y}" if r.get("before") else "before the interviews"
        return f"{r['what']}, reported {when} (Decision p. {r['page']})"
    if r.get("date"):
        return f"{r['what']}, reported {date.fromisoformat(r['date']):%b %-d, %Y} ({r['source']})"
    return f"{r['what']} ({r['source']})"


def _related_phrase(fact: Fact) -> str:
    if not fact.related_reports:
        return ""
    items = "; ".join(f"{r['what']} (p. {r['page']})" if r.get("page") and r.get("source", "").startswith("Decision")
                      else f"{r['what']} ({r['source']}{', ' + format(date.fromisoformat(r['date']), '%b %-d, %Y') if r.get('date') else ''})"
                      for r in fact.related_reports)
    lead = "The Decision lists related reports" if all(r.get("source", "").startswith("Decision") for r in fact.related_reports) else "Related reports"
    return f" {lead}: {items}. A report the detail could be inferred from isn't a report of it, so the transcript decides."


def _cue_sentences(text: str) -> list[str]:
    """The sentences of a line that refer to an earlier conversation."""
    t = plain(text)
    out = []
    for m in PRIOR_CONTACT.finditer(t):
        a = max(t.rfind(c, 0, m.start()) for c in ".?!") + 1
        ends = [k for k in (t.find(c, m.end()) for c in ".?!") if k >= 0]
        out.append(t[a:(min(ends) + 1 if ends else len(t))].strip())
    return list(dict.fromkeys(out))


def _recorded_before(case: Case, fact: Fact, seq: int) -> bool:
    """Does any earlier recorded line, by anyone, carry the detail (code, key terms)?"""
    order = ordered(case)
    return any(u.seq < seq and carries(order, i, fact.must) for i, u in enumerate(order))


def _origin(case: Case, fact: Fact, first_inv: dict | None, first_sus: dict | None, by_id: dict, sessions: dict,
            partial: dict | None = None, reported: set | None = None) -> tuple[str, str]:
    origin, why = _origin_core(case, fact, first_inv, first_sus, by_id, sessions, partial, reported or set())
    if origin != "public" and fact.reports and reported and fact.must:
        rs = [r for r in fact.reports if set(r.get("covers", [])) & reported]
        nots = [g[0] for k, g in enumerate(fact.must) if k not in reported]
        why = (f"Partly public: {'; '.join(_report_phrase(r) for r in rs)}. Not reported: {fact.not_reported or _and([f'“{w}”' for w in nots])}. "
               f"The part that wasn't reported decides it: {why[0].lower()}{why[1:]}")
    if origin != "public" and fact.related_reports:
        why += _related_phrase(fact)
    return origin, why


def _origin_core(case: Case, fact: Fact, first_inv: dict | None, first_sus: dict | None, by_id: dict, sessions: dict,
                 partial: dict | None, reported: set) -> tuple[str, str]:
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
    if fact.reports and fact.must:
        if reported >= set(range(len(fact.must))):
            rs = [r for r in fact.reports if r.get("covers") and (r.get("court_list") or (r.get("date") and date.fromisoformat(r["date"]) < s_day))]
            return "public", (f"Public: {'; '.join(_report_phrase(r) for r in rs)}. That was before the suspect first "
                              f"said it ({_cite(first_sus, sessions)}).{also}")
    elif fact.public_before and s_day and fact.public_before < s_day:
        what = f": {fact.public_what}" if fact.public_what else ""
        return "public", f"It was reported publicly by {fact.public_before:%b %d, %Y}{what}, before the suspect first said it ({_cite(first_sus, sessions)}).{also}"
    first_ev = first_inv if inv_first else first_sus
    if fact.prior_statement and not (fact.must and _recorded_before(case, fact, first_ev["seq"])):
        return "prior_contact", (f"An earlier statement with no public transcript is reported outside the transcripts (we can't read it): {fact.prior_statement}. "
                                 f"In the transcripts it first appears at {_cite(first_ev, sessions)}.")
    if not inv_first and partial:
        named = "; ".join(f"{_and([f'“{w}”' for w in ws])} ({_cite(ev, sessions)})" for ev, ws in partial["per_line"] if ws)
        if partial["all"]:
            part = "every part of it that wasn't reported" if reported else "every part of it"
            return "interrogator_first", (f"Investigators named {part} before he said it, in separate lines: {named}. "
                                          f"He then said it ({_cite(first_sus, sessions)}).")
        his = _and([f"“{w}”" for w in partial["his_words"]])
        his_half = f"he named {his} ({_cite(first_sus, sessions)})" if his else f"he then stated the whole detail ({_cite(first_sus, sessions)})"
        return "interrogator_first", (f"Investigators named {named}; {his_half}. "
                                      "Counted as investigators first, because they named part of the detail before he said all of it.")
    if inv_first:
        u = by_id[first_inv["utterance"]]
        if fact.must:
            # an earlier conversation counts only if the sentence that refers to it names the detail itself, and no
            # earlier recorded line (anyone's) carries it; otherwise the recorded lines decide by order
            cues = _cue_sentences(u.text)
            hit = next((c for c in cues if has_terms(c, fact.must)), None)
            if hit and not _recorded_before(case, fact, first_inv["seq"]):
                return "prior_contact", (f"Investigators first said it ({_cite(first_inv, sessions)}), in a sentence that refers to an earlier "
                                         f"conversation with no public transcript: “{hit}”.")
            how = "by agreeing to a question" if first_sus["kind"] == "assents" else "after that"
            note = ""
            if cues and not hit:
                note = (f" The same line refers to an earlier conversation (“{cues[0]}”), but that sentence doesn't name the detail, "
                        "so it isn't counted as one.")
            elif hit:
                note = " The line refers to an earlier conversation, but an earlier recorded line already carries the detail, so the record decides."
            return "interrogator_first", (f"Investigators first said it ({_cite(first_inv, sessions)}); the suspect first adopted it {how} "
                                          f"({_cite(first_sus, sessions)}).{note}")
        cue = _prior_cue(case, first_inv)
        if cue:
            return "prior_contact", (f"Investigators first said it ({_cite(first_inv, sessions)}), but in words that may refer to something he said "
                                     f"earlier with no public transcript ({pages_of(cue)}: {_cue_quote(cue.text, fact.must)}).")
        how = "by agreeing to a question" if first_sus["kind"] == "assents" else "after that"
        return "interrogator_first", (f"Investigators first said it ({_cite(first_inv, sessions)}); the suspect first adopted it {how} "
                                      f"({_cite(first_sus, sessions)}).")
    cue = _prior_cue(case, first_sus, before=12)
    if cue and fact.must:
        hit = next((c for c in _cue_sentences(cue.text) if has_terms(c, fact.must)), None)
        if hit and not _recorded_before(case, fact, cue.seq):
            return "prior_contact", (f"The suspect said it first in the transcripts ({_cite(first_sus, sessions)}), but just before, an investigator "
                                     f"referred to an earlier conversation with no public transcript in a sentence that names it: “{hit}”.")
    elif cue:
        return "prior_contact", (f"The suspect said it first in the transcripts ({_cite(first_sus, sessions)}), but just before, the investigator "
                                 f"may have referred to an earlier conversation with no public transcript ({pages_of(cue)}: {_cue_quote(cue.text, fact.must)}).")
    return "suspect_first", f"The suspect said it first ({_cite(first_sus, sessions)})."


def _prior_cue(case: Case, ev: dict, before: int = 2) -> Utterance | None:
    """An investigator line at or just before this event that points to an earlier conversation we can't read."""
    window = [u for u in ordered(case) if u.session == ev["session"] and ev["seq"] - before <= u.seq <= ev["seq"]]
    return next((u for u in reversed(window) if u.role == "interrogator" and PRIOR_CONTACT.search(plain(u.text))), None)


def _cue_quote(text: str, must: list | None = None) -> str:
    """The sentence of the investigator's line that refers to the earlier conversation, quoted exactly as printed, and,
    when the detail's key terms are in other sentences of the same line, those sentences as a second quote."""
    t = plain(text)
    m = PRIOR_CONTACT.search(t)
    if not m:
        return f"“{t[:120]}”"
    a = max(t.rfind(c, 0, m.start()) for c in ".?!") + 1
    ends = [k for k in (t.find(c, m.end()) for c in ".?!") if k >= 0]
    cue = t[a:(min(ends) + 1 if ends else len(t))].strip()
    span = next((sp for sp in _spans(t) if must and has_terms(sp, must)), "")
    return f"“{cue}”" + (f" … “{span.strip()}”" if span and cue not in span and span.strip() in t else "")


def _and(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else ""


def _cite(ev: dict, sessions: dict) -> str:
    s = sessions.get(ev["session"])
    return f"{s.label if s else ev['session']}, {pages_of(ev)}"


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
    in_part = sum(1 for f in facts if f.get("partly_supplied"))
    headline = (f"Of {n(len(stated), 'key detail', 'key details')} {case.suspect or 'the suspect'} stated or agreed to, investigators said "
                f"{counts['interrogator_first']} first" + (f" ({in_part} of them in part)" if in_part else "") + "".join(f", {e}" for e in extra) +
                f". {n(len(may), 'detail', 'details')} came from him first in the record.")
    return {"case": {"id": case.id, "title": case.title, "suspect": case.suspect, "notes": case.notes},
            "sessions": [asdict(s) | {"day": s.day.isoformat()} for s in case.sessions],
            "headline": headline, "counts": counts, "facts": facts, "session_stats": session_stats(case),
            "utterance_count": len(case.utterances), "mention_count": sum(1 for m in case.mentions if m.verified is not False)}


MENTION_FIELDS = ("utterance", "fact", "kind", "quote", "by", "verified")
UTT_FIELDS = (("id", ""), ("session", ""), ("seq", 0), ("speaker", ""), ("role", "other"), ("text", ""), ("page", ""), ("sensitive", False),
              ("page_end", ""), ("split", False))


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

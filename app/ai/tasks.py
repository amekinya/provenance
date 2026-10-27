"""Where NVIDIA models do the reading. They never decide who said what first; app/engine/ledger.py does.

- read_page      Nemotron 3.5 Super VL (vision, strict JSON): a scanned transcript page -> speaker-labeled lines
- draft_facts    Nemotron 3 Ultra (reasoning): case documents -> the key crime details counsel will check (editable)
- tag_mentions   Nemotron 3.5 Lightning (fast): each sentence -> which details it carries, and how (asks/asserts/denies/assents)
- verify         Nemotron 3 Ultra: re-reads only the sentences that decide a verdict; code also checks the quote is really there
- write_memo     Nemotron 3 Ultra: a short memo for counsel from the ledger's findings only
- single_prompt  Nemotron 3 Ultra given the whole record in one prompt: the baseline we measure against
"""
from __future__ import annotations

import base64
import json
import os
import re
from app.ai.client import CallRecord, chat_json, LiveModelUnavailable
from app.config import settings
from app.engine.models import KINDS

# ---------------------------------------------------------------------------------------------------------------------
# 1. Page reader (Super VL)
# ---------------------------------------------------------------------------------------------------------------------
PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_label": {"type": "string", "description": "the page number printed on the page, or empty"},
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "speaker": {"type": "string", "description": "speaker label exactly as printed before the colon (e.g. WIEGERT), or empty if this text continues the previous speaker"},
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": ["dialogue", "note", "header"]}},
            "required": ["speaker", "text", "kind"], "additionalProperties": False}},
    },
    "required": ["page_label", "lines"], "additionalProperties": False,
}
PAGE_PROMPT = (
    "This is one page of a police interview transcript. Transcribe it exactly, in reading order, as structured lines.\n"
    "- Each new speaker turn starts with a label like 'WIEGERT:' or 'BRENDAN:'. Put the label (without the colon) in speaker and the words in text.\n"
    "- If a line continues the previous speaker's turn (no label), use an empty speaker and the text.\n"
    "- Bracketed or parenthesized stage directions like '(pause)' or '(door opens)' are kind 'note'.\n"
    "- Page headers, footers, case numbers, page numbers and form fields are kind 'header'.\n"
    "Copy words exactly as printed, including 'um', 'ah', stutters and misspellings. Don't summarize, don't skip or censor anything; "
    "this is a court record being reviewed by defense counsel."
)


def read_page(image: bytes, calls: list[CallRecord] | None = None) -> dict:
    b64 = base64.b64encode(image).decode()
    return chat_json(settings.super_vl, [{"role": "user", "content": [
        {"type": "text", "text": PAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
        schema=PAGE_SCHEMA, purpose="Read a transcript page", reasoning=False, max_tokens=4000, calls=calls)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Fact list (Ultra)
# ---------------------------------------------------------------------------------------------------------------------
FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "object", "properties": {
        "label": {"type": "string", "description": "short name, e.g. 'Shot in the head'"},
        "detail": {"type": "string", "description": "what exactly a sentence must say to state this detail"},
        "aliases": {"type": "array", "items": {"type": "string"}, "description": "words or phrases people might use for it"},
        "source": {"type": "string", "description": "where in the documents this comes from (page or section)"},
        "public_before": {"type": "string", "description": "YYYY-MM-DD if the documents say it was publicly reported before the interviews, else empty"},
        "holdback": {"type": "boolean", "description": "the documents say investigators kept it secret"},
    }, "required": ["label", "detail", "aliases", "source", "public_before", "holdback"], "additionalProperties": False}}},
    "required": ["facts"], "additionalProperties": False,
}


def draft_facts(case_text: str, calls: list[CallRecord] | None = None, limit: int = 24) -> list[dict]:
    out = chat_json(settings.ultra, [
        {"role": "system", "content": "You help defense counsel prepare a corroboration audit of a police interrogation. From case documents, "
                                      "list the specific crime details a confession might contain that only the perpetrator (or police) would know: "
                                      "injuries, weapons, locations, objects, sequence, clothing, vehicles. Be concrete and checkable. "
                                      "Don't judge guilt. Use only the documents."},
        {"role": "user", "content": f"Case documents:\n{case_text[:120000]}\n\nList up to {limit} details."}],
        schema=FACTS_SCHEMA, purpose="Draft the fact list", reasoning=True, max_tokens=8000, calls=calls)
    return out.get("facts", [])[:limit]


# ---------------------------------------------------------------------------------------------------------------------
# 3. Sentence tagger (Lightning) — the model reads; it doesn't decide origins
# ---------------------------------------------------------------------------------------------------------------------
TAG_SCHEMA = {
    "type": "object",
    "properties": {"utterances": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"},
        "graphic": {"type": "boolean", "description": "describes sexual violence or gore in graphic terms"},
        "mentions": {"type": "array", "items": {"type": "object", "properties": {
            "fact": {"type": "string"},
            "kind": {"type": "string", "enum": list(KINDS)},
            "quote": {"type": "string", "description": "the exact words in this sentence that carry the detail"}},
            "required": ["fact", "kind", "quote"], "additionalProperties": False}}},
        "required": ["id", "graphic", "mentions"], "additionalProperties": False}}},
    "required": ["utterances"], "additionalProperties": False,
}
TAG_SYSTEM = (
    "You label sentences from a police interrogation transcript for a defense corroboration audit. For each TARGET sentence, list which "
    "crime details (from the numbered list) it carries, and how:\n"
    "- asks: a question that names or implies the detail (e.g. 'Who shot her in the head?' carries 'shot in the head', even as a question)\n"
    "- asserts: states the detail as true, in any words: partially, in his own words ('the van' for 'a blue cargo van'), or as a short "
    "answer to a question about it ('The bed.' answering 'She was handcuffed to what?')\n"
    "- denies: says the detail itself didn't happen or isn't true, or he doesn't know about it ('I was never at the lake')\n"
    "  If he confirms the detail but denies guilt or explains it innocently ('Sure I scraped my knuckles, but at work'; "
    "'I got home around midnight, I didn't go anywhere'), that is asserts: he stated the detail.\n"
    "- assents: a short agreement ('yeah', 'uh huh') to a question that carried the detail in the line just before\n"
    "A sentence can carry several details or none. Favor recall: label every sentence that refers to a detail, even briefly; a separate "
    "careful check reviews the lines that matter. Skip only sentences that don't refer to the detail at all (a merely similar topic doesn't "
    "count). Quote the exact words. Use the context lines only to understand the targets. Also flag graphic descriptions of sexual violence or gore."
)


def tag_mentions(targets: list[dict], context: list[dict], facts: list[dict], calls: list[CallRecord] | None = None,
                 spec=None) -> list[dict]:
    """targets/context: [{id, speaker, text}]; facts: [{id, label, detail, aliases}]."""
    fact_list = "\n".join(f"{f['id']}: {f['label']} ({f.get('detail', '')}; also said as: {', '.join(f.get('aliases', [])[:6])})" for f in facts)
    ctx = "\n".join(f"  {c['speaker']}: {c['text']}" for c in context)
    tgt = "\n".join(f"[{t['id']}] {t['speaker']}: {t['text']}" for t in targets)
    out = chat_json(spec or settings.lightning, [
        {"role": "system", "content": TAG_SYSTEM},
        {"role": "user", "content": f"Details:\n{fact_list}\n\nContext before the targets:\n{ctx}\n\nTARGET sentences:\n{tgt}\n\nLabel every target id."}],
        schema=TAG_SCHEMA, purpose="Tag details in sentences", reasoning=False, max_tokens=4000, calls=calls)
    known = {t["id"] for t in targets}
    fids = {f["id"] for f in facts}
    rows = []
    for u in out.get("utterances", []):
        if u.get("id") in known:
            u["mentions"] = [m for m in u.get("mentions", []) if m.get("fact") in fids]
            rows.append(u)
    return rows


def quote_in(quote: str, text: str) -> bool:
    """Code check: the model's quote must really be in the sentence, as whole words (spacing, case and curly vs straight
    apostrophes ignored)."""
    def norm(x: str) -> str:
        x = x.translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})).lower()
        return " " + re.sub(r"[^a-z0-9']+", " ", x).strip() + " "
    q = norm(quote).strip()
    return bool(q) and f" {q} " in norm(text)


# ---------------------------------------------------------------------------------------------------------------------
# 4. Verifier (Ultra): only the sentences that decide a verdict get a second, careful read
# ---------------------------------------------------------------------------------------------------------------------
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"checks": {"type": "array", "items": {"type": "object", "properties": {
        # No free-text "why": without reasoning, Ultra sometimes ran on in it until the token limit (about 1 batch in 8 on
        # the Oct 27 benchmark), leaving the batch unchecked.
        "key": {"type": "string"}, "carries_detail": {"type": "boolean"}, "kind": {"type": "string", "enum": list(KINDS)}},
        "required": ["key", "carries_detail", "kind"], "additionalProperties": False}}},
    "required": ["checks"], "additionalProperties": False,
}


VERIFY_REASONING = os.environ.get("VERIFY_REASONING", "1") == "1"  # on: 82% vs 73% on trainval (eval/RESULTS_trainval.md)


def verify(items: list[dict], calls: list[CallRecord] | None = None, reasoning: bool | None = None) -> dict[str, dict]:
    """items: [{key, fact_label, fact_detail, speaker, text, prev}] -> {key: {carries_detail, kind}}.
    Reasoning on or off is chosen on the training writer's interrogations (VERIFY_REASONING; see eval/RESULTS_trainval.md).
    A reasoning call that fails is retried without; a failed batch is retried line by line."""
    reasoning = VERIFY_REASONING if reasoning is None else reasoning
    if reasoning:
        try:
            return _verify(items, calls, True)
        except LiveModelUnavailable as e:
            print(f"  verifier (reasoning on) failed: {str(e)[:140]}; retrying without reasoning")
    try:
        return _verify(items, calls, False)
    except LiveModelUnavailable:
        if len(items) == 1:
            raise
    out: dict[str, dict] = {}  # one bad reply shouldn't leave a whole batch unchecked: retry item by item
    for it in items:
        try:
            out.update(_verify([it], calls, False))
        except LiveModelUnavailable as e:
            print(f"  verifier failed on {it['key']}: {str(e)[:100]}")
    return out


def _verify(items: list[dict], calls: list[CallRecord] | None, reasoning: bool) -> dict[str, dict]:
    def terms(i):
        groups = i.get("must") or []
        return ("\nKey terms (one word from each group must be there; code checks the words): " +
                " + ".join("(" + " | ".join(g) + ")" for g in groups)) if groups else ""
    def stance(i):
        return ("\nQuestion for this item: the words are there; is this the suspect STATING or AGREEING to this detail? Answer "
                "carries_detail=false for a denial, a question, or the words in another sense; give kind asserts or assents when true.") if i.get("stance") else ""

    def part(i):
        return (f"\nQuestion for this item: does the sentence name {', '.join(repr(w) for w in i['part'])} as part of THIS detail (the same "
                "object or event, in the same sense)? It need not convey the whole detail; carries_detail answers this question.") if i.get("part") else ""
    body = "\n\n".join(f"[{i['key']}] Detail: {i['fact_label']} ({i['fact_detail']}){terms(i)}{part(i)}{stance(i)}\nLines before it: {i.get('prev', '')}\n"
                       f"Sentence ({i['speaker']}): {i['text']}" for i in items)
    out = chat_json(settings.ultra, [
        {"role": "system", "content": "You double-check labels for a defense corroboration audit. For each item, decide strictly whether the "
                                      "sentence itself conveys the detail's DISTINCTIVE content: the specific part that would take inside knowledge "
                                      "to say, usually the how, where or what, not the general act or object. For 'stabbed in the stomach', "
                                      "'he stabbed her' doesn't count; 'in the stomach' does. For 'the car was hidden under branches', 'where was "
                                      "her car?' doesn't count; 'branches' or 'covered it' does. Naming only the victim, her car, a general topic or "
                                      "a different aspect doesn't count. When the detail is just an object or a fact ('a knife', 'her hair was cut'), "
                                      "naming it counts. A question that names the distinctive content counts. A short answer counts when, read with "
                                      "the previous line, it supplies the distinctive content, but not when it answers a question about something "
                                      "else. 'Yeah' counts as assent only if the previous line carried the detail. When an item lists key terms, "
                                      "they were written by counsel and define the detail's distinctive content: a line that conveys every key-term "
                                      "group in the detail's sense counts, and degree or extent words in the detail ('until unconscious', 'some of') "
                                      "are NOT required. Reject when the words are used in another sense, about a different event or object, "
                                      "negated, or when the line puts it in a different place or relation than the detail states (near or outside vs "
                                      "inside, before vs after). Give the kind: asks, asserts, denies, assents."},
        {"role": "user", "content": body}],
        schema=VERIFY_SCHEMA, purpose="Verify deciding sentences", reasoning=reasoning, max_tokens=10000 if reasoning else 3000, calls=calls)
    return {c["key"]: c for c in out.get("checks", [])}


STANCE_SCHEMA = {
    "type": "object",
    "properties": {"checks": {"type": "array", "items": {"type": "object", "properties": {
        "key": {"type": "string"}, "stance": {"type": "string", "enum": ["states", "agrees", "denies", "other"]}},
        "required": ["key", "stance"], "additionalProperties": False}}},
    "required": ["checks"], "additionalProperties": False,
}


def stance(items: list[dict], calls: list[CallRecord] | None = None) -> dict[str, str]:
    """The suspect side's only model question. Code has already found the detail's key words in his line (or in the
    question it answers). items: [{key, label, must, before, text}] -> {key: states|agrees|denies|other}. The detail is
    given by its short label and key words only, never the long description (extra clauses caused wrong rejections)."""
    body = "\n\n".join(
        f"[{i['key']}] Detail: {i['label']} (key words: " + " + ".join("(" + " | ".join(g) + ")" for g in i["must"]) + ")"
        f"\nThe lines just before: {i['before'] or '(none)'}\nHis line: {i['text']}" for i in items)
    def ask(reasoning: bool) -> dict:
        return chat_json(settings.ultra, [
            {"role": "system", "content": "You read lines a suspect said in an interrogation. For each item, answer only: in his line, read with "
                                          "the question it answers, is he STATING the detail, AGREEING to it, DENYING it, or is it OTHER "
                                          "(the words are about something else, or he only asks or guesses aloud)? A short answer to a "
                                          "question about the detail states it ('With the shovel.' to 'How did he break the bones?'). "
                                          "Don't judge whether it's true or complete."},
            {"role": "user", "content": body}],
            schema=STANCE_SCHEMA, purpose="Suspect stance", reasoning=reasoning, max_tokens=8000 if reasoning else 2000, calls=calls)
    try:
        out = ask(VERIFY_REASONING)
    except LiveModelUnavailable as e:
        print(f"  stance (reasoning on) failed: {str(e)[:120]}; retrying without reasoning")
        out = ask(False)
    return {c["key"]: c["stance"] for c in out.get("checks", [])}


SENSE_VERSION = 3  # answers saved with an older version of this question are asked again
SENSE_SCHEMA = {
    "type": "object",
    "properties": {"key": {"type": "string"},
                   "about": {"type": "string", "description": "what his line is about, as these lines show: which thing or person, and whose"},
                   "where": {"type": "string", "description": "the place his line puts it, in the lines' own words, or 'not said'"},
                   "stance": {"type": "string", "enum": ["states", "agrees", "denies", "other"]}},
    "required": ["key", "about", "where", "stance"], "additionalProperties": False,
}


def sense(items: list[dict], calls: list[CallRecord] | None = None) -> dict[str, str]:
    """The suspect-side check with the sense of the line, one line per call: items [{key, label, must, before,
    question, between, text}] -> {key: states|agrees|denies|other}. Code has already found the detail's key words in
    his line or in the question it answers. The model first writes what his line is about and where it puts it, as
    the lines show, then compares that with the detail: a different thing or place is OTHER."""
    out: dict[str, str] = {}
    for i in items:
        body = (f"Detail: {i['label']} (key words: " + " + ".join("(" + " | ".join(g) + ")" for g in i["must"]) + ")"
                + (f"\nWhich thing, person and place it is about, as the fact list describes it: {i['detail']}" if i.get("detail") else "")
                f"\nTwo lines before the question: {i['before'] or '(none)'}\nThe question he answers: {i['question'] or '(none)'}"
                + (f"\nLines after the question: {i['between']}" if i.get("between") else "") + f"\nHis line: {i['text']}\nkey: {i['key']}")
        msgs = [{"role": "system", "content": "You read one line a suspect said in an interrogation, with the question it answers and the two lines "
                                              "before that question. First write what his line is about, as these lines show (which thing or person, "
                                              "and whose: a vehicle named in the lines before is the one he means), and where his line puts it. Then "
                                              "compare with the detail and answer: is he STATING this detail, AGREEING to it, DENYING it, or OTHER? Use "
                                              "the fact list's description only to tell which thing, person and place the detail is about; his line "
                                              "doesn't need its other words. "
                                              "Answer OTHER when his line is about a different thing, person or event than the detail (for example "
                                              "Avery's own Suzuki instead of her jeep, or clothes being bloody instead of burned), or puts it "
                                              "somewhere else than the detail says (near, by, beside or outside a place instead of in or inside it), "
                                              "or he denies or says he doesn't know a part the detail names (there was a car, but he doesn't know where "
                                              "it was, for a detail about where the car was), or he only asks or guesses aloud. An answer that accepts "
                                              "what the question takes for granted states it ('He did.' to 'Who cut the rope?' states that the rope "
                                              "was cut). A short answer to a question about the detail states it "
                                              "('With the shovel.' to 'How did he break the bones?'). A line he reads from his own written statement "
                                              "is his statement. Don't judge whether it's true."},
                {"role": "user", "content": body}]
        try:
            r = chat_json(settings.ultra, msgs, schema=SENSE_SCHEMA, purpose="Suspect stance with sense", reasoning=VERIFY_REASONING,
                          max_tokens=6000 if VERIFY_REASONING else 1500, calls=calls)
        except LiveModelUnavailable as e:
            print(f"  sense check failed for {i['key']}: {str(e)[:120]}")
            continue
        if r.get("stance") in ("states", "agrees", "denies", "other"):
            out[i["key"]] = r["stance"]
            out[i["key"] + "#why"] = f"{r.get('about', '')[:160]} | {r.get('where', '')[:80]}"
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 5. Memo (Ultra), from findings only
# ---------------------------------------------------------------------------------------------------------------------
SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"], "additionalProperties": False}
BANNED = re.compile(r"\b(fact[- ]feeding|fed (him|to him)|volunteer\w*|repeated\w*|suppl(y|ied|ies|ying)|(no|nothing) earlier)\b", re.I)
_QUOTE = re.compile(r"[\"\u201c]([^\"\u201d]{4,})[\"\u201d]")


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", t.lower().replace("\u2019", "'")).split())


def memo_problems(text: str, lines: list[str]) -> list[str]:
    """Code checks on model-written memo text: banned wording, and every quotation must be a contiguous part of ONE line
    of the transcript (never words spliced from different lines or pages)."""
    probs = [f"uses '{m.group(0)}'" for m in BANNED.finditer(text)]
    normed = [_norm(x) for x in lines]
    for q in _QUOTE.findall(text):
        nq = _norm(q)
        if nq and not any(nq in x for x in normed):
            probs.append(f"quotation not found in a single line: {q[:60]!r}")
    return probs


def card_text(f: dict) -> str:
    """A detail's card in words, exactly as the page states it: the ledger's explanation, the words of the detail that
    came from investigators near his statement, the court's own observations, and, for a detail he never adopted,
    what he answered."""
    from app.engine.ledger import pages_of, spoken  # noqa: PLC0415

    parts = [f["why"]]
    for w in f.get("label_words_from_investigators") or []:
        who = "; ".join(f"{(l.get('speaker') or 'an investigator').title()} ({pages_of(l)})" for l in w["lines"])
        parts.append(f"“{w['word']}”, a word of the detail, came from investigators near his statement: {who}.")
    parts += [n["text"] for n in f.get("court_notes") or []]
    if f["origin"] == "never_adopted" and f.get("his_responses"):
        said = [f"“{spoken(r['quote'])}” ({pages_of(r)})" for r in f["his_responses"] if spoken(r["quote"])]
        if said:
            parts.append("When it was put to him he answered " + (", ".join(said[:-1]) + " and " + said[-1] if len(said) > 1 else said[0]) + ".")
    return " ".join(parts)


def memo_points(report: dict) -> tuple[list[str], list[str]]:
    """Key points and limits, written by code with a fixed rule so both directions are always covered: the 3 details
    investigators said first with the most investigator mentions before his first statement, then the first 3 details
    that came from him first in the record (in fact-list order), then each public, earlier-statement and never-adopted
    detail. Each point is the detail's card text (card_text), with its page citations and qualifiers."""
    facts = report["facts"]
    inv = sorted((f for f in facts if f["origin"] == "interrogator_first"),
                 key=lambda f: (-(f["interrogator_mentions_before_suspect"] or 0), f["fact"]["id"]))[:3]
    his = [f for f in facts if f["origin"] == "suspect_first"][:3]
    pts = [f"Investigators first: {f['fact']['label']}. {card_text(f)}" for f in inv]
    pts += [f"From him first in the record: {f['fact']['label']}. {card_text(f)}" for f in his]
    pts += [f"Public before the interviews: {f['fact']['label']}. {card_text(f)}" for f in facts if f["origin"] == "public"]
    pts += [f"May trace to an earlier conversation with no public transcript: {f['fact']['label']}. {card_text(f)}"
            for f in facts if f["origin"] == "prior_contact"]
    pts += [f"Never adopted: {f['fact']['label']}. {card_text(f)}" for f in facts if f["origin"] == "never_adopted"]
    limits = ["Draft for counsel's review: investigator lines are matched by key words in code; the suspect's lines were labeled by "
              "Nemotron 3 Ultra, checked by code, and the lines that decide a detail re-checked by Ultra with the question they answer; "
              "borderline calls on his side can change between runs.",
              "Unrecorded contacts can't be traced; a detail counts as possibly from one only when an investigator's sentence about an "
              "earlier conversation names it, or a court record reports such a statement, and nothing is said about what happened in them.",
              "'Public' means the reports the court listed (Decision pp. 69-70) or news we could date state every part of the detail; a "
              "report it could only be inferred from is shown as related, and the transcript decides.",
              "Key terms decide what counts as stating a detail; a reader should check each cited line."]
    return pts, limits


def memo_details(report: dict) -> list[str]:
    """Every detail, in fact-list order, as its card states it."""
    return [f"{f['fact']['label']} ({f['origin_label']}). {card_text(f)}" for f in report["facts"]]


def write_memo(report: dict, calls: list[CallRecord] | None = None, lines: list[str] | None = None) -> dict:
    """Ultra writes the summary only; code checks it (banned wording, quotations from a single line) and writes the key
    points and limits itself, so both directions are always covered and every citation is the ledger's."""
    pts, limits = memo_points(report)
    # details that may trace to an unrecorded conversation go to the model by label only, so the summary can't say what
    # was said in it
    told = [f"May trace to an earlier conversation with no public transcript: {f['fact']['label']}." for f in report["facts"]
            if f["origin"] == "prior_contact"]
    brief = {"headline": report["headline"], "counts": report["counts"],
             "key_points": [x for x in pts if not x.startswith("May trace to an earlier conversation")] + told,
             "sessions": [{k: s[k] for k in ("label", "turns", "interrogator_share_of_words", "questions", "questions_that_carry_facts")}
                          for s in report["session_stats"]]}
    msgs = [{"role": "system", "content": "You write the 3-4 sentence summary of a memo for defense counsel about a corroboration audit of an "
                                          "interrogation. Use ONLY the findings given. Open by saying it is a draft for counsel's review, computed "
                                          "from the record with machine-checked lines. Cover both directions: details investigators said first and "
                                          "details that came from him first in the record. Never claim the confession is false or true, never "
                                          "describe what happened in unrecorded contacts, and don't use the words 'volunteered', 'fact-feeding', "
                                          "'repeated' or 'supplied' (say 'questions that named the detail'). For details that came from him first, "
                                          "say exactly that ('came from him first in the record'); never say nothing earlier supplied them. Call a "
                                          "detail public only if the findings say 'Public:'; a 'Partly public' detail is partly public. Don't quote "
                                          "transcript lines."},
            {"role": "user", "content": f"Findings: {json.dumps(brief)[:40000]}"}]
    summary, probs = "", ["not written"]
    for _ in range(2):
        out = chat_json(settings.ultra, msgs, schema=SUMMARY_SCHEMA, purpose="Write the memo", reasoning=False, max_tokens=1200, calls=calls)
        summary = out.get("summary", "").strip()
        probs = memo_problems(summary, lines or [])
        if not probs:
            break
        msgs.append({"role": "user", "content": f"Rewrite the summary; problems: {probs}"})
    if probs:  # fall back to a summary written by code from the counts
        c = report["counts"]
        summary = (f"Draft for counsel's review, computed from the record with machine-checked lines. {report['headline']} "
                   f"Investigators said {c['interrogator_first']} details first; {c['suspect_first']} came from him first in the record.")
    point_checks = [f"{x[:40]}…: {pr}" for x in pts for pr in memo_problems(x, lines or [])] if lines else []
    details = memo_details(report)
    point_checks += [f"{x[:40]}…: {pr}" for x in details for pr in memo_problems(x, lines or [])] if lines else []
    return {"summary": summary, "key_points": pts, "details": details, "limits": limits,
            "by": "summary: Nemotron 3 Ultra (checked by code); points and details: code", "summary_checks": probs, "point_checks": point_checks}


# ---------------------------------------------------------------------------------------------------------------------
# 6. Baseline: one big prompt, no ledger
# ---------------------------------------------------------------------------------------------------------------------
BASELINE_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "object", "properties": {
        "fact": {"type": "string", "description": "the detail's id exactly as given, e.g. F1"},
        "first_said_by": {"type": "string", "enum": ["interrogator", "suspect", "never"]},
        "utterance_id": {"type": "string", "description": "the id of that first line, e.g. X007-012; empty for never"}},
        "required": ["fact", "first_said_by", "utterance_id"], "additionalProperties": False}}},
    "required": ["facts"], "additionalProperties": False,
}


def single_prompt(utterances: list[dict], facts: list[dict], calls: list[CallRecord] | None = None) -> list[dict]:
    """The baseline: one strong model, one prompt, the same question the ledger answers."""
    record = "\n".join(f"[{u['id']}] {u['speaker']} ({u['role']}): {u['text']}" for u in utterances)
    fact_list = "\n".join(f"{f['id']}: {f['label']}" for f in facts)
    out = chat_json(settings.ultra, [
        {"role": "user", "content": f"Interrogation record:\n{record}\n\nCrime details:\n{fact_list}\n\n"
                                    "For EVERY detail id above, answer:\n"
                                    "1. Did the SUSPECT ever state this detail or agree to it (agreeing includes a 'yeah' to a question that "
                                    "named it)? If not, first_said_by is 'never' and utterance_id is empty.\n"
                                    "2. If he did: who put the detail into the record FIRST, the interrogator (a question naming it counts) or "
                                    "the suspect? utterance_id is the id of that first line: the interrogator's first line naming it if the "
                                    "interrogator was first, otherwise the suspect's first line stating it.\n"
                                    "Return one entry per detail id."}],
        schema=BASELINE_SCHEMA, purpose="Single-prompt baseline", reasoning=True, max_tokens=12000, calls=calls)
    return out.get("facts", [])

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
    "- denies: says the detail itself didn't happen or isn't true, or he doesn't know about it ('I never went to dock 4')\n"
    "  If he confirms the detail but denies guilt or explains it innocently ('I cut my forearm on a fence, not breaking in'; "
    "'I was home by two AM'), that is asserts: he stated the detail.\n"
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


def verify(items: list[dict], calls: list[CallRecord] | None = None, reasoning: bool = False) -> dict[str, dict]:
    """items: [{key, fact_label, fact_detail, speaker, text, prev}] -> {key: {carries_detail, kind, why}}.
    Ultra without reasoning by default: with reasoning on, about one call in three came back without valid JSON on our
    diagnostic runs, which cost more and left lines unchecked. If a reasoning call fails, it's retried without."""
    if not reasoning:
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
    try:
        return _verify(items, calls, True)
    except LiveModelUnavailable as e:
        print(f"  verifier (reasoning on) failed: {str(e)[:140]}; retrying without reasoning")
        return _verify(items, calls, False)


def _verify(items: list[dict], calls: list[CallRecord] | None, reasoning: bool) -> dict[str, dict]:
    body = "\n\n".join(f"[{i['key']}] Detail: {i['fact_label']} ({i['fact_detail']})\nPrevious line: {i.get('prev', '')}\n"
                       f"Sentence ({i['speaker']}): {i['text']}" for i in items)
    out = chat_json(settings.ultra, [
        {"role": "system", "content": "You double-check labels for a defense corroboration audit. For each item, decide strictly whether the "
                                      "sentence itself conveys the detail's DISTINCTIVE content: what someone would need inside knowledge to say. "
                                      "A question that names that content counts. Naming only a general topic, the victim, her car, or a different "
                                      "aspect doesn't (for 'the car was hidden under branches', 'where was her car?' doesn't count; 'branches' or "
                                      "'covered it' does). For a detail with several parts, one distinctive part counts. A short answer counts "
                                      "when, read with the previous line, it supplies the distinctive content. 'Yeah' counts as assent only if the "
                                      "previous line carried the detail. Give the kind: asks, asserts, denies, assents."},
        {"role": "user", "content": body}],
        schema=VERIFY_SCHEMA, purpose="Verify deciding sentences", reasoning=reasoning, max_tokens=10000 if reasoning else 3000, calls=calls)
    return {c["key"]: c for c in out.get("checks", [])}


# ---------------------------------------------------------------------------------------------------------------------
# 5. Memo (Ultra), from findings only
# ---------------------------------------------------------------------------------------------------------------------
MEMO_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "key_points": {"type": "array", "items": {"type": "string"}},
                   "limits": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary", "key_points", "limits"], "additionalProperties": False,
}


def write_memo(report: dict, calls: list[CallRecord] | None = None) -> dict:
    brief = {"headline": report["headline"], "counts": report["counts"],
             "facts": [{"label": f["fact"]["label"], "origin": f["origin_label"], "why": f["why"],
                        "investigator_mentions_before_suspect": f["interrogator_mentions_before_suspect"],
                        "adopted_by_agreeing": f["adopted_by_assent"], "across_sessions": f["crossed_sessions"]} for f in report["facts"]],
             "sessions": report["session_stats"]}
    return chat_json(settings.ultra, [
        {"role": "system", "content": "You write a short internal memo for defense counsel about a corroboration audit of an interrogation. "
                                      "Use ONLY the findings given; cite pages as given; never claim the confession is false or that the suspect is "
                                      "innocent; say what the record shows about where details came from. Plain, precise language."},
        {"role": "user", "content": f"Findings: {json.dumps(brief)[:60000]}\n\nWrite: summary (3-4 sentences), key_points (4-6 bullets, most "
                                    "important first, each with its page cite), limits (2-4 bullets: what this audit can't tell)."}],
        schema=MEMO_SCHEMA, purpose="Write the memo", reasoning=True, max_tokens=5000, calls=calls)


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

from scripts.tag import add_code_assents, candidate_lines, detail_stems


def utt(i, role, text, session="S1"):
    return {"id": f"U{i}", "session": session, "seq": i, "speaker": "DET" if role == "interrogator" else "KID", "role": role, "text": text}


def test_code_assent_follows_an_investigator_line_carrying_the_detail():
    utts = [utt(1, "interrogator", "Did he use a knife?"), utt(2, "suspect", "Yeah."), utt(3, "interrogator", "OK."), utt(4, "suspect", "Uh huh.")]
    mentions = [{"utterance": "U1", "fact": "F1", "kind": "asks", "quote": "a knife", "by": "model", "verified": None}]
    assert add_code_assents(utts, mentions) == 1
    added = mentions[-1]
    assert (added["utterance"], added["fact"], added["kind"], added["by"]) == ("U2", "F1", "assents", "code")


def test_no_code_assent_for_a_statement_or_a_rejected_question():
    utts = [utt(1, "interrogator", "Did he use a knife?"), utt(2, "suspect", "Yeah, I think so, in the garage.")]
    mentions = [{"utterance": "U1", "fact": "F1", "kind": "asks", "quote": "a knife", "by": "model", "verified": None}]
    assert add_code_assents(utts, mentions) == 0
    utts = [utt(1, "interrogator", "Did he use a knife?"), utt(2, "suspect", "Yeah.")]
    mentions = [{"utterance": "U1", "fact": "F1", "kind": "asks", "quote": "a knife", "by": "model", "verified": False}]
    assert add_code_assents(utts, mentions) == 0


def test_candidate_lines_are_earlier_investigator_lines_with_the_details_words():
    fact = {"id": "F1", "label": "Tires and branches were put on the fire", "aliases": ["tires"]}
    assert "tires" in detail_stems(fact) and "branc" in detail_stems(fact)
    utts = [utt(1, "interrogator", "You threw tires and branches on her"), utt(2, "suspect", "We threw some tires on."),
            utt(3, "interrogator", "What happened next?"), utt(4, "interrogator", "More branches?")]
    assert [u["id"] for u in candidate_lines(utts, fact, before_seq=4)] == ["U1"]


def test_verifier_retries_without_reasoning(monkeypatch):
    from app.ai import tasks
    from app.ai.client import LiveModelUnavailable

    def fake(spec, msgs, schema, **kw):
        if kw.get("reasoning"):
            raise LiveModelUnavailable("timeout")
        return {"checks": [{"key": "a", "carries_detail": True, "kind": "asks", "why": ""}]}
    monkeypatch.setattr(tasks, "chat_json", fake)
    out = tasks.verify([{"key": "a", "fact_label": "x", "fact_detail": "", "speaker": "DET", "text": "t", "prev": ""}], reasoning=True)
    assert out["a"]["carries_detail"] is True


def test_search_checks_the_question_right_before_a_short_answer(monkeypatch):
    from scripts import tag

    utts = [utt(1, "interrogator", "Tell us where she was shot?"), utt(2, "suspect", "Outside."),
            utt(3, "interrogator", "No, I mean where, in the garage"), utt(4, "suspect", "In the garage.")]
    facts = [{"id": "F1", "label": "Shot inside the garage", "detail": "", "aliases": []}]
    mentions = [{"utterance": "U4", "fact": "F1", "kind": "asserts", "quote": "In the garage", "by": "model", "verified": True}]
    seen = []

    def fake_verify(items, calls=None, reasoning=False):
        seen.extend(i["key"] for i in items)
        return {i["key"]: {"key": i["key"], "carries_detail": i["key"].startswith("U3"), "kind": "asserts", "why": ""} for i in items}
    monkeypatch.setattr(tag.tasks, "verify", fake_verify)
    cfg = {"id": "t", "title": "t", "suspect": "KID", "sessions": [{"id": "S1", "label": "I", "day": "2020-01-01"}]}
    for u in utts:
        u |= {"page": "", "sensitive": False}
    assert tag.search_investigator_first(cfg, utts, facts, mentions, []) == 1
    assert "U3|F1" in seen
    from app.engine.ledger import audit, load_case
    f = audit(load_case(tag.case_dict(cfg, utts, facts, mentions)))["facts"][0]
    assert f["origin"] == "interrogator_first"


def test_verifier_retries_item_by_item_after_a_bad_batch(monkeypatch):
    from app.ai import tasks
    from app.ai.client import LiveModelUnavailable

    def fake(spec, msgs, schema, **kw):
        body = msgs[-1]["content"]
        if body.count("Detail:") > 1:
            raise LiveModelUnavailable("no valid JSON")
        key = body.split("]")[0].strip("[")
        return {"checks": [{"key": key, "carries_detail": True, "kind": "asks", "why": ""}]}
    monkeypatch.setattr(tasks, "chat_json", fake)
    items = [{"key": k, "fact_label": "x", "fact_detail": "", "speaker": "DET", "text": "t", "prev": ""} for k in ("a", "b")]
    assert set(tasks.verify(items)) == {"a", "b"}

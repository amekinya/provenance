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

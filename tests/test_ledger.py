from datetime import date

from app.engine.ledger import audit, fact_ledger
from app.engine.models import Case, Fact, Mention, Session, Utterance


def mini_case() -> Case:
    s = [Session("A", "Day 1 · school", date(2006, 2, 27)), Session("B", "Day 3 · station", date(2006, 3, 1))]
    u = [
        Utterance("A1", "A", 1, "DET", "interrogator", "Did he say anything about shooting her?", "45"),
        Utterance("A2", "A", 2, "KID", "suspect", "No.", "45"),
        Utterance("A3", "A", 3, "DET", "interrogator", "What happened in the garage?", "46"),
        Utterance("A4", "A", 4, "KID", "suspect", "There was a fire in the pit out back.", "46"),
        Utterance("B1", "B", 5, "DET", "interrogator", "You told us the other night about the car. Where was it?", "3"),
        Utterance("B2", "B", 6, "KID", "suspect", "In the pit by the crusher, the car was covered with branches.", "3"),
        Utterance("B3", "B", 7, "DET", "interrogator", "Who shot her in the head?", "10"),
        Utterance("B4", "B", 8, "KID", "suspect", "Yeah.", "10"),
        Utterance("B5", "B", 9, "DET", "interrogator", "He had a tattoo on his arm, right?", "11"),
        Utterance("B6", "B", 10, "KID", "suspect", "Uh huh.", "11"),
        Utterance("B7", "B", 11, "DET", "interrogator", "Was there a knife?", "12"),
        Utterance("B8", "B", 12, "KID", "suspect", "I don't know.", "12"),
    ]
    f = [Fact("F1", "Shot in the head"), Fact("F2", "Fire in the burn pit"),
         Fact("F3", "Car covered with branches", public_before=date(2006, 2, 1)),
         Fact("F4", "A tattoo on the arm", planted=True), Fact("F5", "A knife was used"),
         Fact("F6", "Car near the crusher"), Fact("F7", "Never mentioned")]
    m = [Mention("A1", "F1", "asks"), Mention("A2", "F1", "denies"), Mention("A4", "F2", "asserts"),
         Mention("B2", "F3", "asserts"), Mention("B2", "F6", "asserts"), Mention("B3", "F1", "asks"), Mention("B4", "F1", "assents"),
         Mention("B5", "F4", "asks"), Mention("B6", "F4", "assents"), Mention("B7", "F5", "asks")]
    return Case("t", "Test", "KID", s, u, f, m)


def by_fact(report):
    return {f["fact"]["id"]: f for f in report["facts"]}


def test_interrogator_first_across_sessions_and_by_assent():
    f = by_fact(audit(mini_case()))["F1"]
    assert f["origin"] == "interrogator_first" and f["corroboration"] == "cannot_corroborate"
    assert f["crossed_sessions"] and f["adopted_by_assent"]
    assert f["interrogator_mentions_before_suspect"] == 2  # asked on day 1 and again on day 3
    assert f["first_interrogator"]["page"] == "45"


def test_suspect_first_may_corroborate():
    f = by_fact(audit(mini_case()))["F2"]
    assert f["origin"] == "suspect_first" and f["corroboration"] == "may_corroborate"


def test_public_planted_never_adopted_and_unmentioned():
    r = by_fact(audit(mini_case()))
    assert r["F3"]["origin"] == "public"
    assert r["F4"]["origin"] == "planted_adopted"
    assert r["F5"]["origin"] == "never_adopted"
    assert r["F7"]["origin"] == "not_mentioned"


def test_reference_to_unrecorded_talk_makes_it_unclear():
    f = by_fact(audit(mini_case()))["F6"]
    assert f["origin"] == "prior_contact" and f["corroboration"] == "unclear"


def test_rejected_mentions_are_ignored():
    c = mini_case()
    for m in c.mentions:
        if m.utterance == "A1":
            m.verified = False  # the verifier found the quote doesn't carry the fact
    f = fact_ledger(c, c.facts[0])
    assert f["first_interrogator"]["utterance"] == "B3"


def test_headline_and_session_stats():
    r = audit(mini_case())
    assert r["headline"].startswith("Of 5 key details KID stated")
    st = {s["session"]: s for s in r["session_stats"]}
    assert st["B"]["questions_that_carry_facts"] == 3 and st["B"]["suspect_minimal_answers"] >= 2


def test_speaker_labels_and_nods():
    from scripts.ingest import _canon, _nod_split
    roster = {"WIEGERT": "interrogator", "FASSBENDER": "interrogator", "BRENDAN": "suspect", "DASSEY": "suspect"}
    assert _canon("WEIGERT", roster) == ("WIEGERT", "interrogator")
    assert _canon("INV. WIEGERT", roster) == ("WIEGERT", "interrogator")
    assert _canon("BRENDAN DASSEY", roster)[1] == "suspect"
    assert _nod_split('Her jeep was in the garage wasn’t it? (Brendan nods “yes”)', {"BRENDAN"})[-1][0] == "suspect"


def test_bare_yes_without_a_question_is_not_adoption():
    from app.engine.ledger import audit, load_case
    d = {"id": "t", "title": "t", "suspect": "Kid", "sessions": [{"id": "S1", "label": "S1", "day": "2020-01-01"}],
         "utterances": [{"id": "u1", "session": "S1", "seq": 1, "speaker": "DET", "role": "interrogator", "text": "Did he have a knife?"},
                        {"id": "u2", "session": "S1", "seq": 2, "speaker": "K", "role": "suspect", "text": "No."},
                        {"id": "u3", "session": "S1", "seq": 3, "speaker": "DET", "role": "interrogator", "text": "What did you eat?"},
                        {"id": "u4", "session": "S1", "seq": 4, "speaker": "DET", "role": "interrogator", "text": "Pizza?"},
                        {"id": "u5", "session": "S1", "seq": 5, "speaker": "DET", "role": "interrogator", "text": "Okay."},
                        {"id": "u6", "session": "S1", "seq": 6, "speaker": "K", "role": "suspect", "text": "Yeah."}],
         "facts": [{"id": "F1", "label": "A knife"}],
         "mentions": [{"utterance": "u1", "fact": "F1", "kind": "asks"}, {"utterance": "u6", "fact": "F1", "kind": "assents"}]}
    f = audit(load_case(d))["facts"][0]
    assert f["origin"] == "never_adopted"


def test_prior_statement_and_curly_apostrophes():
    from app.engine.ledger import TACTICS, audit, load_case, plain
    assert TACTICS["minimization"].search(plain("It’s not your fault"))
    d = {"id": "t", "title": "t", "suspect": "Kid", "sessions": [{"id": "S1", "label": "S1", "day": "2020-01-01"}],
         "utterances": [{"id": "u1", "session": "S1", "seq": 1, "speaker": "DET", "role": "interrogator",
                         "text": "Those jeans that Tom talked to you about the other night with the bleach?"},
                        {"id": "u2", "session": "S1", "seq": 2, "speaker": "K", "role": "suspect", "text": "Yeah, bleach."}],
         "facts": [{"id": "F1", "label": "Bleach on jeans"}],
         "mentions": [{"utterance": "u1", "fact": "F1", "kind": "asks"}, {"utterance": "u2", "fact": "F1", "kind": "asserts"}]}
    assert audit(load_case(d))["facts"][0]["origin"] == "prior_contact"


def test_key_terms_decide_what_counts():
    from app.engine.ledger import derive_must, has_terms
    s = [Session("A", "Day 1", date(2006, 3, 1))]
    u = [
        Utterance("A1", "A", 1, "DET", "interrogator", "Did you see any hair in the fire?", "1"),
        Utterance("A2", "A", 2, "KID", "suspect", "No.", "1"),
        Utterance("A3", "A", 3, "DET", "interrogator", "What else did he do to her head?", "2"),
        Utterance("A4", "A", 4, "KID", "suspect", "That he cut off her hair.", "2"),
        Utterance("A5", "A", 5, "DET", "interrogator", "How'd he break up the bones?", "3"),
        Utterance("A6", "A", 6, "KID", "suspect", "With the shovel.", "3"),
    ]
    f = [Fact("F1", "Her hair was cut", must=[["cut", "chop"], ["hair"]]),
         Fact("F2", "Bones were broken up with a shovel", must=[["bone"], ["break", "broke", "broken"], ["shovel"]])]
    m = [Mention("A4", "F1", "asserts", verified=True), Mention("A6", "F2", "asserts", verified=True)]
    rep = by_fact(audit(Case("t", "T", "KID", s, u, f, m)))
    # Investigator side is code only. 'Did you see any hair in the fire?' names 'hair' (one group, not both), within
    # 6 lines before his statement, so investigators named part of it: counted as investigators first, both halves stated.
    f1 = rep["F1"]
    assert f1["first_suspect"]["utterance"] == "A4" and f1["origin"] == "interrogator_first" and f1["partly_supplied"]
    assert f1["first_interrogator"]["utterance"] == "A1" and f1["first_interrogator"]["by"] == "key words"
    assert f1["why"].startswith("Investigators named “hair”") and "he named “cut”" in f1["why"]
    # a short answer is read with the question before it: the question names the bones and the breaking, he adds the shovel
    f2 = rep["F2"]
    assert f2["first_suspect"]["utterance"] == "A6" and f2["origin"] == "interrogator_first" and f2["partly_supplied"]
    assert f2["partial_words"] == ["bones", "break"] and "he named “shovel”" in f2["why"]
    # a model label can't remove an investigator line that code matched
    u2 = u + [Utterance("A7", "A", 7, "DET", "interrogator", "Did he cut her hair off?", "4"), Utterance("A8", "A", 8, "KID", "suspect", "Yeah.", "4")]
    m2 = [Mention("A7", "F1", "asks", verified=False), Mention("A8", "F1", "assents")]
    f1b = by_fact(audit(Case("t", "T", "KID", s, u2[:2] + u2[6:], f, m2)))["F1"]
    assert f1b["first_interrogator"]["utterance"] == "A7" and f1b["origin"] == "interrogator_first" and not f1b["partly_supplied"]
    assert has_terms("He stabbed her in the belly", [["stomach", "belly"], ["stab"]]) and not has_terms("he stabbed her", [["stomach"], ["stab"]])
    assert derive_must("Red hoodie found near back gate") == [["red"], ["hoodie"], ["back"], ["gate"]]


def test_read_aloud_statement_becomes_the_suspects_line():
    from scripts.ingest import split_read_aloud

    def u(spk, role, text):
        return {"speaker": spk, "role": role, "text": text, "page": "469", "session": "S1"}
    inline = [u("WIEGERT", "interrogator", "OK, then I'm just going to read it and read what you wrote. OK? (Is I got off the bus at 3:45 and "
                "seen her jeep down at the house. He told me that he stabbed her in the stomach and put the knife under the seat. "
                "Is that your statement Brendan?)"), u("BRENDAN", "suspect", "Yes.")]
    out = split_read_aloud(inline, "BRENDAN")
    assert [x["role"] for x in out] == ["interrogator", "suspect", "interrogator", "suspect"]
    assert out[1]["text"].startswith("(read aloud by Wiegert from Brendan's written statement; printed speaker: WIEGERT) Is I got off the bus") and out[1]["page"] == "469"
    assert out[2]["text"] == "Is that your statement Brendan?"
    nextline = [u("WIEGERT", "interrogator", "Now let me read, this would be page two that you added here."), u("WIEGERT", "interrogator", "I'll read this. Is that okay?"),
                u("BRENDAN", "suspect", "Yeah."), u("WIEGERT", "interrogator", "He was angry and that's why he did it. He told me he used rope to tie her up in the jeep.")]
    out = split_read_aloud(nextline, "BRENDAN")
    assert out[3]["role"] == "suspect" and out[3]["text"].startswith("(read aloud by Wiegert from Brendan's written statement; printed speaker: WIEGERT) He was angry")
    miranda = [u("WIEGERT", "interrogator", "I'm going to read this form to you, it's your rights."), u("BRENDAN", "suspect", "OK."),
               u("WIEGERT", "interrogator", "You have the right to remain silent and anything you say can be used against you in court.")]
    assert [x["role"] for x in split_read_aloud(miranda, "BRENDAN")] == ["interrogator", "suspect", "interrogator"]


def test_prior_contact_cues_follow_one_general_rule():
    from app.engine.ledger import PRIOR_CONTACT
    yes = ["I talked to ya the other night and you said nothing about Monte.",
           "those jeans that Tom had talked to you about the other night",
           "We talked last er Monday we talked a little about some things a burn barrel out front",
           "You told me last time that he had a fire going.",
           "when we spoke to you at the school you said",
           "The other night you said it was red."]
    no = ["You told us before that he stabbed her.", "Did you talk to him that night?", "Tell us what you saw in the garage."]
    assert all(PRIOR_CONTACT.search(t) for t in yes), [t for t in yes if not PRIOR_CONTACT.search(t)]
    assert not any(PRIOR_CONTACT.search(t) for t in no), [t for t in no if PRIOR_CONTACT.search(t)]


def test_echo_borrowed_context_and_turn_window():
    s = [Session("A", "Day 1", date(2006, 3, 1))]
    u = [
        Utterance("A1", "A", 1, "KID", "suspect", "Branches, tires, tires, like a garbage bag.", "1"),
        Utterance("A2", "A", 2, "DET", "interrogator", "So you see the branches, the tires, it's all on the fire already?", "1"),
        Utterance("A3", "A", 3, "DET", "interrogator", "We know he hid something under the seat in the truck there.", "2"),
        Utterance("A4", "A", 4, "DET", "interrogator", "It's not your fault.", "2"),
        Utterance("A5", "A", 5, "DET", "interrogator", "Who shot her in the head?", "3"),
        Utterance("A6", "A", 6, "KID", "suspect", "He did.", "3"),
        Utterance("A7", "A", 7, "DET", "interrogator", "Then why didn't you tell us that?", "3"),
        Utterance("A8", "A", 8, "KID", "suspect", "Cuz I couldn't think of it.", "3"),
        Utterance("A9", "A", 9, "DET", "interrogator", "Now you remember it? (Brendan nods)", "3"),
        Utterance("A10", "A", 10, "KID", "suspect", "(nods)", "3", split=True),
        Utterance("A11", "A", 11, "DET", "interrogator", "Tell us about that then.", "3"),
        Utterance("A14", "A", 14, "KID", "suspect", "That he shot her with his .22.", "3"),
    ]
    f = [Fact("F1", "Tires and branches on the fire", must=[["tire", "tires"], ["branch", "branches"], ["fire"]]),
         Fact("F2", "A knife", must=[["knife"]]),
         Fact("F3", "Shot with a .22 rifle", must=[["shot", "shoot"], ["22", "rifle"]])]
    m = [Mention("A14", "F3", "asserts", verified=True)]
    rep = by_fact(audit(Case("t", "T", "KID", s, u, f, m)))
    # the investigator repeats his tires and branches: an echo, so it only adds 'fire'
    f1 = rep["F1"]
    assert f1["first_interrogator"] is None
    assert any(e.get("kind_note", "").startswith("repeats his words") for e in f1["events"]) or not f1["events"]
    # 'It's not your fault.' has no key word of its own, so it can't borrow 'knife' from the lines before (there is none anyway)
    assert rep["F2"]["first_interrogator"] is None
    # 'Who shot her in the head?' is 6 printed turns before his statement once the split nod isn't counted: in part
    f3 = rep["F3"]
    assert f3["origin"] == "interrogator_first" and f3["partly_supplied"] and f3["first_interrogator"]["utterance"] == "A5"
    assert "“shot”" in f3["why"] and "he named “22”" in f3["why"]

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

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


def test_exchange_window_and_every_part_across_lines():
    # investigators name "camera" and later "barrel", 9 quiet turns apart, then he says it: every part, across lines
    s = [Session("A", "Day 1", date(2006, 3, 1))]
    u = [Utterance("A1", "A", 1, "DET", "interrogator", "What happened to her camera?", "1")]
    u += [Utterance(f"A{k}", "A", k, "DET" if k % 2 else "KID", "interrogator" if k % 2 else "suspect",
                    "Tell us." if k % 2 else "I don't know.", "1") for k in range(2, 11)]
    u += [Utterance("A11", "A", 11, "DET", "interrogator", "Was anything put in the barrel?", "2"),
          Utterance("A12", "A", 12, "DET", "interrogator", "Tell us what.", "2"),
          Utterance("A13", "A", 13, "KID", "suspect", "Her camera went in the barrel.", "2"),
          Utterance("A14", "A", 14, "DET", "interrogator", "He backed into the garage, right?", "2"),
          Utterance("A15", "A", 15, "KID", "suspect", "The jeep was in the back.", "2")]
    f = [Fact("F1", "Her camera was put in the barrel", must=[["camera"], ["barrel"]]),
         Fact("F2", "The jeep was in the back", must=[["in the back", "back of"], ["jeep"]])]
    m = [Mention("A13", "F1", "asserts", verified=True), Mention("A15", "F2", "asserts", verified=True)]
    rep = by_fact(audit(Case("t", "T", "KID", s, u, f, m)))
    f1 = rep["F1"]
    assert f1["origin"] == "interrogator_first" and f1["supplied_across_lines"] and not f1["partly_supplied"]
    assert "every part of it" in f1["why"] and "“camera”" in f1["why"] and "“barrel”" in f1["why"]
    # a lone 'back' (in 'backed') is never offered as a label word that came from investigators
    assert all(w["word"] != "back" for w in rep["F2"]["label_words_from_investigators"])


def test_prior_contact_quote_is_the_cue_sentence_and_the_detail_sentence():
    from app.engine.ledger import _cue_quote
    line = ("OK. We know the fire was going. I talked to ya the other night and you said nothing about Monte. "
            "We talked about cleaning somethin' up in that garage. You told me there was blood. What's goin' on?")
    q = _cue_quote(line, [["blood"], ["garage"]])
    assert q.startswith("“I talked to ya the other night and you said nothing about Monte.”")
    assert "“We talked about cleaning somethin' up in that garage. You told me there was blood.”" in q
    # every quoted part is printed in the line as is
    assert all(part in line for part in q.replace("”", "").split(" … “") for part in [part.lstrip("“")])


def _two_sessions():
    return [Session("A", "Day 1", date(2006, 2, 27)), Session("B", "Day 3", date(2006, 3, 1))]


def test_prior_contact_only_when_the_cue_sentence_names_it_and_nothing_recorded_did():
    burn = Fact("F1", "Things burned in the burn barrel", must=[["burn", "burned"], ["barrel"]])
    cue = Utterance("B1", "B", 10, "DET", "interrogator", "OK. We talked last er Monday we talked a little about some things a burn "
                    "barrel out front do you remember anything about that burn barrel?", "619")
    said = Utterance("B2", "B", 11, "KID", "suspect", "He burned stuff in the barrel.", "620")
    m = [Mention("B2", "F1", "asserts", verified=True)]
    rep = by_fact(audit(Case("t", "T", "KID", _two_sessions(), [cue, said], [burn], m)))
    assert rep["F1"]["origin"] == "prior_contact" and "We talked last er Monday" in rep["F1"]["why"]
    # the same cue, but the recorded session already has it: the recorded line decides by order
    early = Utterance("A1", "A", 1, "DET", "interrogator", "Was anything burned in the barrel?", "470")
    rep = by_fact(audit(Case("t", "T", "KID", _two_sessions(), [early, cue, said], [burn], m)))
    assert rep["F1"]["origin"] == "interrogator_first" and rep["F1"]["first_interrogator"]["utterance"] == "A1"


def test_pizza_chatter_and_told_us_last_time_are_not_prior_contact():
    s = _two_sessions()
    rope = Fact("F1", "Tied up with rope", must=[["tie", "tied"], ["rope"]])
    u = [Utterance("B1", "B", 10, "DET", "interrogator", "You ate that whole pizza the other night, and you sure you don't want another sandwich?", "667"),
         Utterance("B2", "B", 11, "KID", "suspect", "He tied her up with rope.", "667")]
    rep = by_fact(audit(Case("t", "T", "KID", s, u, [rope], [Mention("B2", "F1", "asserts", verified=True)])))
    assert rep["F1"]["origin"] == "suspect_first"
    # the cue names something else in another sentence of the line: investigators first, and the card says why it isn't one
    u = [Utterance("B1", "B", 10, "DET", "interrogator", "I talked to ya the other night about school. Did he tie her up with a rope?", "667"),
         Utterance("B2", "B", 11, "KID", "suspect", "Yeah.", "667")]
    rep = by_fact(audit(Case("t", "T", "KID", s, u, [rope], [Mention("B2", "F1", "assents", verified=True)])))
    assert rep["F1"]["origin"] == "interrogator_first" and "doesn't name the detail" in rep["F1"]["why"]
    # "You told us last time ..." when last time was recorded: his recorded line decides by order
    carry = Fact("F2", "He helped carry her out of the bedroom", must=[["carry", "carried"], ["bedroom"]])
    u = [Utterance("A1", "A", 1, "KID", "suspect", "I helped carry her out of the bedroom.", "470"),
         Utterance("B1", "B", 10, "DET", "interrogator", "You told us last time you helped carry her out of the bedroom. Remember that?", "771"),
         Utterance("B2", "B", 11, "KID", "suspect", "Yeah.", "771")]
    rep = by_fact(audit(Case("t", "T", "KID", s, u, [carry], [Mention("A1", "F2", "asserts", verified=True)])))
    assert rep["F2"]["origin"] == "suspect_first" and rep["F2"]["first_suspect"]["utterance"] == "A1"


def test_public_only_when_reports_state_every_part():
    s = _two_sessions()
    u = [Utterance("A1", "A", 1, "DET", "interrogator", "What did you put on the fire?", "442"),
         Utterance("A2", "A", 2, "KID", "suspect", "Some branches and some tires.", "442")]
    court = {"what": "Avery burned tires that night", "source": "Decision and Order", "page": "69", "court_list": True, "before": "2006-03-01"}
    m = [Mention("A2", "F1", "asserts", verified=True)]
    must = [["tire", "tires"], ["branch", "branches"], ["fire", "burn"]]
    # tires and fire reported, branches not: the branches decide, and he named them first
    f = Fact("F1", "Tires and branches on the fire", must=must, reports=[court | {"covers": [0, 2]}], not_reported="the branches")
    r = by_fact(audit(Case("t", "T", "KID", s, u, [f], m)))["F1"]
    assert r["origin"] == "suspect_first" and r["why"].startswith("Partly public: Avery burned tires that night, reported before March 1, 2006 (Decision p. 69)")
    assert "Not reported: the branches" in r["why"] and "2005" not in r["why"]
    # every part reported: public
    f = Fact("F1", "Tires and branches on the fire", must=must, reports=[court | {"covers": [0, 1, 2]}])
    assert by_fact(audit(Case("t", "T", "KID", s, u, [f], m)))["F1"]["origin"] == "public"
    # a report it could only be inferred from: the transcript decides, and the card lists it
    f = Fact("F1", "Tires and branches on the fire", must=must, related_reports=[{"what": "tires were found", "source": "Decision and Order", "page": "69"}])
    r = by_fact(audit(Case("t", "T", "KID", s, u, [f], m)))["F1"]
    assert r["origin"] != "public" and "The Decision lists related reports: tires were found (p. 69)" in r["why"]


def test_long_lines_quote_the_sentence_with_the_detail_and_court_notes_cite_pages():
    s = [Session("A", "Day 1", date(2006, 2, 27))]
    stmt = ("(read aloud by Wiegert from Brendan's written statement; printed speaker: WIEGERT) Is I got off the bus at 3:45 and seen "
            "her jeep down at Steven's house. Then I went in my house and played Playstation. Then he stabbed her in the stomach.")
    u = [Utterance("A1", "A", 1, "DET", "interrogator", "Something with the head. What else?", "584"),
         Utterance("A2", "A", 2, "KID", "suspect", stmt, "470")]
    f = Fact("F1", "Stabbed in the stomach", must=[["stomach"], ["stab", "stabbed"]],
             court_notes=[{"page": "70", "text": "notes the prompt “Something with the head” before he said it", "quotes": ["Something with the head"]}])
    r = by_fact(audit(Case("t", "T", "KID", s, u, [f], [Mention("A2", "F1", "asserts", verified=True)])))["F1"]
    assert r["first_suspect"]["quote"] == "Then he stabbed her in the stomach."
    assert r["first_suspect"]["read_aloud"].startswith("read aloud by Wiegert from Brendan's written statement")
    assert r["court_notes"][0]["text"] == "The Decision (p. 70) notes the prompt “Something with the head” (p. 584) before he said it."


def test_memo_card_text_carries_label_words_and_his_answers():
    from app.ai.tasks import card_text, memo_problems
    f = {"why": "The suspect said it first (Day 1, p. 579).", "origin": "suspect_first", "court_notes": [],
         "label_words_from_investigators": [{"word": "unconscious", "lines": [{"utterance": "A3", "page": "579", "speaker": "WIEGERT"}]}]}
    t = card_text(f)
    assert "“unconscious”, a word of the detail, came from investigators near his statement: Wiegert (p. 579)." in t
    g = {"why": "Investigators raised it (Day 3, p. 662), but the suspect never stated or agreed to it.", "origin": "never_adopted",
         "his_responses": [{"utterance": "B2", "page": "663", "quote": "(shakes head “no”) uh uh"},
                           {"utterance": "B3", "page": "663", "quote": "No but I don't know where it was."}]}
    t = card_text(g)
    assert "he answered “uh uh” (p. 663) and “No but I don't know where it was.” (p. 663)." in t
    assert memo_problems(t, ["(shakes head “no”) uh uh", "No but I don't know where it was."]) == []
    assert memo_problems("Eight came from him first, with nothing earlier supplying them.", []) != []

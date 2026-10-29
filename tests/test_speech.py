"""The recording path's plain code: aligning word times to speaker segments, the speaker summary, the fallback role
proposal, and the upload route's behaviour when the speech endpoint is off. No models run here."""
import io

from fastapi.testclient import TestClient

from app.engine.speech import align, propose_roles_by_code, sentences, speaker_summary


def W(w, start, end):
    return {"w": w, "start": start, "end": end}


WORDS = [W("Where", 0.0, 0.3), W("were", 0.3, 0.5), W("you?", 0.5, 0.9),
         W("At", 1.5, 1.7), W("home.", 1.7, 2.1), W("I", 2.3, 2.4), W("swear.", 2.4, 2.9),
         W("Okay.", 3.6, 3.9), W("Who", 4.0, 4.2), W("saw", 4.2, 4.4), W("you?", 4.4, 4.8)]
SEGS = [{"speaker": 0, "start": 0.0, "end": 1.0}, {"speaker": 1, "start": 1.4, "end": 3.0}, {"speaker": 0, "start": 3.5, "end": 5.0}]


def test_sentences_split_on_end_punctuation():
    assert [" ".join(w["w"] for w in s) for s in sentences(WORDS)] == ["Where were you?", "At home.", "I swear.", "Okay.", "Who saw you?"]


def test_align_merges_sentences_into_turns_by_speaker():
    turns = align(WORDS, SEGS)
    assert [(t["speaker"], t["text"]) for t in turns] == [
        ("SPEAKER 1", "Where were you?"), ("SPEAKER 2", "At home. I swear."), ("SPEAKER 1", "Okay. Who saw you?")]
    assert turns[1]["start"] == 1.5 and turns[1]["end"] == 2.9


def test_a_word_at_a_boundary_does_not_split_a_sentence():
    # "you?" spills 0.2 s into the next speaker's segment; the sentence still belongs to its speaker
    words = [W("Where", 0.0, 0.3), W("were", 0.3, 0.6), W("you?", 0.6, 1.3), W("Home.", 1.5, 1.9)]
    segs = [{"speaker": 0, "start": 0.0, "end": 1.1}, {"speaker": 1, "start": 1.1, "end": 2.0}]
    assert [t["speaker"] for t in align(words, segs)] == ["SPEAKER 1", "SPEAKER 2"]


def test_words_outside_any_segment_take_the_nearest_or_the_previous_speaker():
    words = [W("Hello.", 0.0, 0.4), W("Hi.", 5.0, 5.3)]  # the second word is 3 s from any segment
    assert [t["speaker"] for t in align(words, [{"speaker": 2, "start": 0.0, "end": 0.5}])] == ["SPEAKER 3"]


def test_summary_and_fallback_roles():
    summary = speaker_summary(align(WORDS, SEGS))
    assert [(s["speaker"], s["turns"], s["questions"]) for s in summary] == [("SPEAKER 1", 2, 2), ("SPEAKER 2", 1, 0)]
    roles = propose_roles_by_code(summary + [{"speaker": "SPEAKER 3", "turns": 2, "words": 4, "questions": 1, "samples": []}])
    assert roles["SPEAKER 1"]["role"] == "investigator"
    assert roles["SPEAKER 3"]["role"] == "suspect"  # among speakers with 2+ turns, the one asking the fewest questions
    assert sum(r["role"] == "suspect" for r in roles.values()) == 1


def test_transcribe_says_plainly_when_the_endpoint_is_off(monkeypatch):
    from app import main
    from app.ai import speech

    monkeypatch.setattr(speech, "available", lambda: False)
    r = TestClient(main.app).post("/api/transcribe", files={"file": ("a.mp3", io.BytesIO(b"x" * 100), "audio/mpeg")})
    assert r.status_code == 503 and "endpoint, which is off right now" in r.json()["detail"]


def test_numbered_speaker_names_from_the_recording_step_parse_as_their_roles():
    from app.main import parse_transcript

    utts = parse_transcript("DET 1: Where were you?\nSUSPECT: Home.\nDET 2: Alone?\nOTHER 1: He was with me.\nSUSPECT: Yes.",
                            "SUSPECT", "DET 1, DET 2")
    assert [(u["speaker"], u["role"]) for u in utts] == [
        ("DET 1", "interrogator"), ("SUSPECT", "suspect"), ("DET 2", "interrogator"), ("OTHER 1", "other"), ("SUSPECT", "suspect")]

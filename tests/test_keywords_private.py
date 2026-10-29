from fastapi.testclient import TestClient

from app.ai.tasks import normalize_must


def test_normalize_drops_lone_function_words_and_generic_words():
    assert normalize_must([["Back", "in the back", "rear"], ["the", "Car", "car"]]) == [["in the back", "rear"], ["car"]]
    assert normalize_must([["see", "thing"]]) == []


def test_key_words_line_parsing():
    from app.main import key_words_text, parse_key_words
    m = parse_key_words("cut/chop/hack + hair")
    assert m == [["cut", "chop", "hack"], ["hair"]]
    assert key_words_text(m) == "cut/chop/hack + hair"
    assert parse_key_words("") == []


def test_private_mode_uses_only_the_matcher_and_code(monkeypatch):
    import dataclasses

    from app import main
    from app.ai import matcher, tasks

    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, matcher_url="http://matcher.invalid"))
    monkeypatch.setattr(matcher, "score", lambda model, pairs, calls=None, **kw: [
        0.99 if ("park" in q.lower() and "park" in p.lower()) else 0.01 for q, p in pairs])

    def no_outside(*a, **kw):
        raise AssertionError("private mode must not call an outside model")
    monkeypatch.setattr(tasks, "tag_mentions", no_outside)
    monkeypatch.setattr(tasks, "verify", no_outside)
    c = TestClient(main.app)
    t = "DET: Where were you?\nJORDAN: Home.\nDET: Was it at the park?\nJORDAN: Yeah.\nDET: Anything else?\nJORDAN: No."
    r = c.post("/api/analyze", json={"transcript": t, "facts": "At the park", "suspect": "JORDAN", "interrogators": "DET",
                                     "private": True, "key_words": ["park"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["run"]["private"] is True and d["run"]["cost_usd"] == 0.0
    assert d["facts"][0]["origin"] == "interrogator_first"


def test_private_mode_without_endpoint_says_so(monkeypatch):
    import dataclasses

    from app import main
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, matcher_url=""))
    c = TestClient(main.app)
    t = "DET: Where were you?\nJORDAN: Home.\nDET: Was it at the park?\nJORDAN: Yeah."
    r = c.post("/api/analyze", json={"transcript": t, "facts": "At the park", "suspect": "JORDAN", "interrogators": "DET", "private": True})
    assert r.status_code == 503 and "matcher endpoint" in r.json()["detail"]

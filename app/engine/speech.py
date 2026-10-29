"""Turn a recording's word timestamps (NVIDIA Parakeet) and speaker segments (NVIDIA Nemotron-3-Diarization) into
transcript turns. Plain code: no model decides who said what.

- Each word takes the speaker whose segment overlaps it most (the nearest segment within 1 s if none overlaps).
- Speakers rarely change mid-sentence, so each sentence (ended by . ? !) takes the speaker that holds most of its
  word time; a word misplaced at a turn boundary doesn't split a sentence.
- Consecutive sentences by the same speaker form one turn.
"""
from __future__ import annotations

import re

END = re.compile(r"[.?!]['\")\]]*$")


def _speaker_of(w: dict, segments: list[dict]) -> int | None:
    best, best_ov = None, 0.0
    for s in segments:
        ov = min(w["end"], s["end"]) - max(w["start"], s["start"])
        if ov > best_ov:
            best, best_ov = s["speaker"], ov
    if best is not None:
        return best
    mid = (w["start"] + w["end"]) / 2
    near = min(segments, key=lambda s: min(abs(mid - s["start"]), abs(mid - s["end"])), default=None)
    if near is not None and min(abs(mid - near["start"]), abs(mid - near["end"])) <= 1.0:
        return near["speaker"]
    return None


def sentences(words: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        cur.append(w)
        if END.search(w["w"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def align(words: list[dict], segments: list[dict]) -> list[dict]:
    """-> [{"speaker": "SPEAKER 1", "start", "end", "text"}] in time order."""
    turns: list[dict] = []
    last = 0
    for sent in sentences(words):
        weight: dict[int, float] = {}
        for w in sent:
            spk = _speaker_of(w, segments)
            if spk is not None:
                weight[spk] = weight.get(spk, 0.0) + max(0.01, w["end"] - w["start"])
        spk = max(weight, key=weight.get) if weight else last
        last = spk
        text = " ".join(w["w"] for w in sent)
        if turns and turns[-1]["_spk"] == spk:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = sent[-1]["end"]
        else:
            turns.append({"_spk": spk, "speaker": f"SPEAKER {spk + 1}", "start": sent[0]["start"], "end": sent[-1]["end"], "text": text})
    for t in turns:
        t.pop("_spk")
    return turns


def speaker_summary(turns: list[dict]) -> list[dict]:
    """Per speaker: how many turns and words, how many turns ask a question, and a few sample lines."""
    by: dict[str, dict] = {}
    for t in turns:
        s = by.setdefault(t["speaker"], {"speaker": t["speaker"], "turns": 0, "words": 0, "questions": 0, "samples": []})
        s["turns"] += 1
        s["words"] += len(t["text"].split())
        s["questions"] += "?" in t["text"]
        if len(s["samples"]) < 3 and len(t["text"].split()) >= 4:
            s["samples"].append(t["text"][:160])
    return sorted(by.values(), key=lambda s: int(s["speaker"].split()[-1]))


def propose_roles_by_code(summary: list[dict]) -> dict[str, dict]:
    """Fallback when no model is available: the speaker who asks the largest share of questions is an investigator; the
    one among the rest who asks the fewest is the suspect. Only a proposal: the user confirms."""
    if not summary:
        return {}
    ratio = {s["speaker"]: s["questions"] / max(1, s["turns"]) for s in summary}
    talkers = [s for s in summary if s["turns"] >= 2] or summary
    suspect = min(talkers, key=lambda s: (ratio[s["speaker"]], -s["turns"]))["speaker"] if len(talkers) > 1 else None
    out = {}
    for s in summary:
        if s["speaker"] == suspect:
            out[s["speaker"]] = {"role": "suspect", "why": "asks the fewest questions"}
        elif ratio[s["speaker"]] >= 0.3:
            out[s["speaker"]] = {"role": "investigator", "why": f"asks questions in {round(100 * ratio[s['speaker']])}% of turns"}
        else:
            out[s["speaker"]] = {"role": "other", "why": "few turns or questions"}
    return out

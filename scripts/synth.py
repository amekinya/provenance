"""Synthetic interrogations with known answers, for measuring Provenance and training our own matcher.

Code draws a plan (which details investigators introduce, which the suspect volunteers, which never come up). A model
from a DIFFERENT family than the ones Provenance uses writes a fictional, non-graphic interrogation that follows the plan
and labels every line with the details it carries. Code then keeps only interrogations whose labels reproduce the plan,
so the gold answers don't depend on trusting the writer.
Usage: python -m scripts.synth --n 40 [--generator MODEL_ID] [--split trainval|test] [--append]
  --append adds this writer's interrogations to the split's file (re-running a writer replaces its own).
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.ai.client import CallRecord, LiveModelUnavailable, _get_client, chat_json
from app.config import ModelSpec

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "synth"
PREFERRED = ["deepseek-ai/DeepSeek-V4.1-Flash-1017", "Qwen/Qwen3-30B-A3B-Instruct-2507", "MiniMaxAI/MiniMax-M3", "zai-org/GLM-5.2"]
CRIMES = ["a convenience-store robbery", "a car break-in at a mall", "a house burglary", "an assault outside a bar", "an arson at a garage",
          "a hit-and-run", "a theft from a warehouse", "a shooting that wounded a store clerk", "a stolen motorcycle", "vandalism at a school",
          "a pharmacy break-in", "a mugging at a bus stop", "a stolen delivery van", "a fire at a storage unit", "a jewelry theft"]

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "label": {"type": "string"}, "detail": {"type": "string"}},
            "required": ["id", "label", "detail"], "additionalProperties": False}},
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "speaker": {"type": "string", "enum": ["DET", "SUSPECT"]},
            "text": {"type": "string"},
            "carries": {"type": "array", "items": {"type": "object", "properties": {
                "fact": {"type": "string"}, "kind": {"type": "string", "enum": ["asks", "asserts", "denies", "assents"]}},
                "required": ["fact", "kind"], "additionalProperties": False}}},
            "required": ["speaker", "text", "carries"], "additionalProperties": False}},
    },
    "required": ["facts", "lines"], "additionalProperties": False,
}


def pick_generator(name: str | None) -> ModelSpec:
    ids = {m.id for m in _get_client().models.list().data}
    if name and name not in ids:  # never fall back silently: train and test writers must differ
        raise SystemExit(f"Generator {name} isn't available; have: {sorted(ids)[:40]}")
    for mid in ([name] if name else []) + PREFERRED:
        if mid in ids:
            return ModelSpec("generator", mid, 0.5, 1.5)
    raise SystemExit(f"No generator model available; have: {sorted(ids)[:40]}")


def plan(rng: random.Random) -> list[dict]:
    n = rng.randint(6, 8)
    out = []
    for i in range(1, n + 1):
        origin = rng.choices(["investigator_first", "suspect_first", "never"], [0.5, 0.35, 0.15])[0]
        adopt = rng.choice(["repeats it in his own words", "just agrees ('yeah')"]) if origin == "investigator_first" else ""
        out.append({"id": f"F{i}", "origin": origin, "adoption": adopt})
    return out


def write_one(i: int, gen: ModelSpec, seed: int) -> tuple[dict | None, list[CallRecord]]:
    rng = random.Random(seed + i)
    crime = rng.choice(CRIMES)
    p = plan(rng)
    plan_txt = "\n".join(
        f"- {x['id']}: " + ("the DETECTIVE must be the first to mention it (in a question or statement); later the SUSPECT " + x["adoption"] + "."
                            if x["origin"] == "investigator_first" else
                            "the SUSPECT must be the first to mention it, unprompted; the detective may follow up after." if x["origin"] == "suspect_first" else
                            "the DETECTIVE may mention it, but the SUSPECT never states or agrees to it (he denies or says he doesn't know).")
        for x in p)
    calls: list[CallRecord] = []
    prompt = (
        f"Write a realistic, fictional police interrogation transcript about {crime}. Use only invented names. Keep it non-graphic.\n"
        f"First define {len(p)} concrete crime details (ids {', '.join(x['id'] for x in p)}): specific things like a weapon, a location, an object, "
        "clothing, a vehicle, a time, an injury.\nThen write 30-50 lines between DET and SUSPECT following this plan exactly:\n" + plan_txt + "\n"
        "Rules: no detail may appear before its planned first speaker introduces it. Vary the wording: describe the same detail with different "
        "words in different lines (synonyms, partial references). Include ordinary filler lines that carry no details. For each line, list every "
        "detail it carries and how: asks (question naming it), asserts, denies, assents (a bare 'yeah'/'uh huh' agreeing to a question that carried it)."
    )
    try:
        out = chat_json(gen, [{"role": "user", "content": prompt}], schema=SCHEMA, purpose="Synthetic interrogation", reasoning=False,
                        max_tokens=7000, temperature=0.7, calls=calls, enforce_cap=False)
    except (LiveModelUnavailable, Exception) as e:  # noqa: BLE001
        print(f"  #{i} failed: {str(e)[:120]}")
        return None, calls
    fids = {f["id"] for f in out["facts"]}
    utts = [{"id": f"X{i:03d}-{k + 1:03d}", "session": "S1", "seq": k + 1, "speaker": ln["speaker"],
             "role": "interrogator" if ln["speaker"] == "DET" else "suspect", "text": ln["text"],
             "carries": [c for c in ln["carries"] if c["fact"] in fids]} for k, ln in enumerate(out["lines"])]
    # Consistency filter: the writer's own labels must reproduce the plan, or we drop the interrogation.
    for x in p:
        firsts = [(u["role"], c["kind"]) for u in utts for c in u["carries"] if c["fact"] == x["id"] and
                  ((u["role"] == "interrogator" and c["kind"] in ("asks", "asserts")) or (u["role"] == "suspect" and c["kind"] in ("asserts", "assents")))]
        first_role = firsts[0][0] if firsts else "never"
        sus_adopts = any(r == "suspect" for r, _ in firsts)
        ok = (x["origin"] == "investigator_first" and first_role == "interrogator" and sus_adopts) or \
             (x["origin"] == "suspect_first" and first_role == "suspect") or (x["origin"] == "never" and not sus_adopts)
        if not ok:
            print(f"  #{i} dropped: {x['id']} planned {x['origin']}, labels say first={first_role}")
            return None, calls
    return {"id": f"X{i:03d}", "crime": crime, "generator": gen.model_id, "plan": p, "facts": out["facts"], "utterances": utts}, calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--generator", default=None)
    ap.add_argument("--split", default="trainval", choices=["trainval", "test"])
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    gen = pick_generator(a.generator)
    print("generator:", gen.model_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"interrogations_{a.split}.jsonl"
    kept, calls = [], []
    with ThreadPoolExecutor(8) as ex:
        for rec, c in ex.map(lambda i: write_one(i, gen, a.seed), range(1, a.n + 1)):
            calls += c
            if rec:
                kept.append(rec)
    tag = "" if gen.model_id.startswith("Qwen/") or a.split == "trainval" else gen.model_id.split("/")[-1][:3].upper()
    for r in kept:
        r["split"] = a.split
        if a.split == "test":
            r["id"] = "T" + tag + r["id"]  # unique across test writers
    if a.append and out_file.exists():
        old = [json.loads(x) for x in out_file.read_text().splitlines() if x.strip()]
        old = [r for r in old if r.get("generator") != gen.model_id]
        print(f"keeping {len(old)} interrogations from other writers")
        kept = old + kept
    if not kept:
        raise SystemExit("no interrogations kept; the file was left unchanged")
    out_file.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    print(f"kept {len(kept)}/{a.n} interrogations; {sum(len(r['utterances']) for r in kept)} lines; {len(calls)} calls")


if __name__ == "__main__":
    main()

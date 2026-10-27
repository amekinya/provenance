# Provenance

**Who said it first?** Provenance reads every recorded interview with a suspect and finds, for each key crime detail, the first time it appears in the record and who said it: the investigators or the suspect.

A confession is treated as reliable when the suspect knows things only the culprit could know. But that works only if the investigators didn't say those things first. In 2006, 16-year-old Brendan Dassey "knew" Teresa Halbach was shot in the head. The first person in the record to mention a head shot was an investigator: *“All right, I'm just gonna come out and ask you. Who shot her in the head?”*

Checking this by hand means reading hundreds of transcript pages, sometimes for weeks. Provenance does it in minutes, with a page citation for every claim.

Live demo: _link after deploy_. Built for the Nebius × NVIDIA Global AI Hackathon (Best Apps and Agents).

**What this is not.** It doesn't decide guilt or whether a confession is true. It shows where each detail first appears in the record, with citations, so defense counsel, innocence organizations and courts can check. Dassey's conviction was upheld by the Seventh Circuit en banc (2017), and the U.S. Supreme Court declined review (2018). This project takes no position on the case.

## Why it matters

- 29% of the 375 DNA exonerations tracked by the Innocence Project involved a false confession ([Innocence Project](https://innocenceproject.org/dna-exonerations-in-the-united-states/)).
- False confessions usually contain accurate "inside" details. In 24 of 26 recent false confessions from DNA exonerations, the confession included crime-scene details that matched the scene ([Garrett, *Contaminated Confessions Revisited*, 101 Va. L. Rev. 395 (2015)](https://virginialawreview.org/articles/contaminated-confessions-revisited/)). If the suspect was innocent, those details had to come from somewhere, and the interrogation is the usual source.
- Thirty states and D.C. now record interrogations under statutes, court rules or police policies ([Wilson Center for Science and Justice, Duke, 2024](https://wcsj.law.duke.edu/publications/jurisdictions-that-record-police-interrogations/)). That means many hours of recordings, and a public defender with a full caseload can't trace every detail by hand.

## What it shows

For each key detail, Provenance gives one of these origins:

| Origin | Meaning | Can it corroborate? |
|---|---|---|
| Investigators said it first | An investigator mentioned it before the suspect did | No |
| Already public | It was in the news before the interview | No |
| A false detail he adopted | A test detail investigators knew was false, and he agreed to it | No |
| Unclear: prior contact | It may have come up in an unrecorded contact | Unclear |
| Suspect said it first | It appears in his words first in the record | Possibly |
| Never adopted | Investigators said it; he never stated or agreed to it | — |

The page also has:
- **A map** of every mention across sessions.
- **Session statistics:** talk share, one-word answers, and questions that carry details.
- **Tactic counts** from code, using the categories in Kassin et al. 2010. Examples are "we already know", minimization and "be honest" demands.
- **A short memo** written only from the ledger.

## How it works: models read, code decides

| Step | What runs | Where |
|---|---|---|
| 1. Read scanned transcripts | **Nemotron 3.5 Super VL** turns each page image into speaker-labeled lines. Code merges continuation lines and matches names against the case roster | Nebius Token Factory |
| 2. List key details | **Nemotron 3 Ultra** drafts them from the case documents. We reviewed each one and added a source (2016 federal habeas decision, page cites; public-news dates) | Nebius Token Factory |
| 3. Label every sentence | **Nemotron 3 Ultra** (reasoning off) marks which details each line carries and how: asks, states, denies, or agrees. Code discards any label whose quoted words aren't in the line, and adds a suspect's bare "yeah" that directly answers a question carrying a detail. (Nemotron 3.5 Lightning is cheaper but missed about half of the suspect's adoptions on our training-writer interviews: 57% vs 87% right first speaker) | Nebius Token Factory |
| 3b. Our matcher | NVIDIA **llama-nemotron-rerank-1b-v2**, fine-tuned to score whether a line carries a detail | Trained on a **Nebius Serverless Job**, served on a **Nebius Serverless Endpoint** |
| 4. Decide who said it first | A plain-code ledger walks the record in order. No model decides. A label counts only if the line has the detail's **key terms** (below) | The app |
| 5. Double-check the deciding lines | **Nemotron 3 Ultra** (reasoning on) re-reads only the lines that decide a verdict and keeps a label only if the line conveys the detail's distinctive part ("in the stomach", not just "stabbed her"). Rejected labels are dropped and the ledger re-runs. Reasoning on beat reasoning off on our training-writer interviews (82% vs 73%) | Nebius Token Factory |
| 6. Safety search before crediting the suspect | Before the suspect is credited with any detail, code searches every earlier investigator line for the detail's words, plus the questions just before his answer, and Ultra checks each candidate, so a labeling miss on an investigator's line is much less likely to hand the suspect a detail he was given | Code + Nebius Token Factory |

### Key terms: the code check that has the last word

Each detail in a case's fact list carries `must`: groups of words that make it specific, with alternates in each group. "Her hair was cut" is `[["cut", "chop", "hack", ...], ["hair"]]`; "Stabbed in the stomach" is `[["stomach", "belly", "gut", "abdomen"], ["stab", "stabbed", "knife", "cut"]]`. A line carries the detail only if it has at least one alternate from every group. A bare "yeah" or a short line (8 words or fewer) is read together with the 2 lines before it in the same session, so "With the shovel." answering a question about the bones counts. Code applies this after the model labels and after Ultra's check, and before Ultra even sees a safety-search candidate, so a model can't make a line count that lacks the words.

**Partly supplied.** When investigators named part of a detail before he stated all of it (they asked about her purse, phone and camera; later he said "the burning barrel"), the detail is counted as **investigators first**, the conservative call for counsel, and its card says exactly which words investigators named first, where, and what he added. Code decides this per key-term group: for each group, who first said one of its words, among the lines labeled with the detail (denials and lines Ultra rejected don't count) and the lines his statement is read with. Ultra checks the investigator line that decides it, asked only whether it names those words in this detail's sense. A detail counts as coming from him only if investigators hadn't named any part of it.

How the key terms are written: from each detail's own wording and the court's description only, one group per component that makes the detail specific (what was done, to what, with what, or where), with synonyms and the case's own words for things (Dassey calls the RAV4 "the jeep"). Degree words ("until unconscious") and things every line about the crime shares (her, Avery) aren't groups. They are never written by looking at which transcript lines would match, because that would fit the answer. In the product, counsel writes them with the fact list. For the synthetic benchmark there is no such field, and deriving groups mechanically from each label (every content word its own group) halved accuracy on the training writer's interviews (82% to 41%) because the writers vary the wording on purpose, so the benchmark doesn't use key terms.

## Results

- **Synthetic interrogations with known answers** ([`eval/RESULTS.md`](eval/RESULTS.md)). 29 held-out interrogations (203 details) written by Qwen3 and GLM-5.2, model families different from the training writer; each is kept only if its own labels match the plan we drew in code. Settings (labeler, verifier reasoning) were chosen on the training writer's interviews, never on these.

  | Method | Right first speaker | Right first line | Investigator-supplied details credited to the suspect |
  |---|---|---|---|
  | Keyword match | 56% | 55% | 3 |
  | One Nemotron 3 Ultra prompt per interview | 80% | 77% | 2 |
  | Provenance | 77% | 76% | 1 |

  This is the second run on this test set (the first, before the last two fixes: Provenance 73%, one prompt 82%); every run is listed in `eval/RESULTS.md`. On short 30 to 50 line interviews, a single Ultra prompt is a strong baseline. The real record is 5,755 lines across three recorded sessions, and Provenance cites, and has Ultra re-check, every line a verdict rests on. We tried one Ultra prompt per real session for the 25 Dassey details: for the 3,146-line March 1 session it returned no usable answer (its reasoning ran past 32,000 output tokens twice), so we have no measured comparison on the real record.
- **The Dassey record.** Three recorded interviews (Feb 27, Mar 1 and May 13, 2006; 305 scanned pages). We check our reading against the details the federal court identified as fed to him.

## Honest limits

- Unrecorded contacts can't be traced. The Feb 27 Two Rivers interview has no public transcript, so details it touched are marked "unclear" when referred to later.
- "Already public" relies on the news reports we could find and date. A detail we couldn't date is never marked public.
- The labels come from models and can be wrong. Every claim in the UI links to the exact line so a person can check it.
- Graphic passages are hidden unless the reader chooses to show them.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 7860
```

| What you set | What works |
|---|---|
| Nothing | The sample cases |
| `NEBIUS_API_KEY` | Live analysis of a pasted transcript |
| `ENDPOINT_URL` and `ENDPOINT_TOKEN` | Our Nebius endpoint |

GitHub Actions has two workflows:
- **Run case pipeline:** ingest, facts, tag.
- **Benchmark.**

The Nebius parts are in `train/`.

Sources: Calumet County Sheriff's Department interview transcripts (public court records, as published); *Dassey v. Dittmann*, Decision and Order (E.D. Wis. Aug 12, 2016); CNN transcript, Nov 10, 2005; *The Badger Herald*, Nov 16, 2005.

## License

Apache-2.0.

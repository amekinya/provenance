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
| 3. Label every sentence | **Nemotron 3.5 Lightning** marks which details each line carries and how: asks, states, denies, or agrees. Code discards any label whose quoted words aren't in the line | Nebius Token Factory |
| 3b. Our matcher | NVIDIA **llama-nemotron-rerank-1b-v2**, fine-tuned to score whether a line carries a detail | Trained on a **Nebius Serverless Job**, served on a **Nebius Serverless Endpoint** |
| 4. Decide who said it first | A plain-code ledger walks the record in order. No model decides | The app |
| 5. Double-check the deciding lines | **Nemotron 3 Ultra** re-reads only the lines that decide a verdict. Rejected labels are dropped and the ledger re-runs | Nebius Token Factory |

## Results

_Filled in from `eval/RESULTS.md` after the runs._

- **Synthetic interrogations with known answers.** They were written by a model from a different family, and each one is kept only if its own labels match the plan we drew in code. We compare four methods: keyword match, one Ultra prompt per interview, Provenance, and Provenance with our fine-tuned matcher. We report three measures: the right first speaker, the right first line, and the costly error (an investigator-supplied detail credited to the suspect).
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

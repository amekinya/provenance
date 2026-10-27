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
- **A short memo for counsel, marked as a draft.** Code writes the key points by a fixed rule, so both directions are always covered: the 3 details investigators said first with the most investigator mentions, the first 3 that came from him first, then the public, earlier-statement and never-adopted details, each with the ledger's own page citations. Nemotron 3 Ultra writes only a 3 to 4 sentence summary. Code rejects it if it uses loaded words ("fact-feeding", "volunteered", "repeated"), or quotes anything that isn't a contiguous part of one transcript line. If it fails twice, code writes the summary instead. Details that may trace to an unrecorded conversation reach the model by label only, so the summary can't describe what was said there; the code-written point quotes the court's own account with its page.

## How it works: models read, code decides

| Step | What runs | Where |
|---|---|---|
| 1. Read scanned transcripts | **Nemotron 3.5 Super VL** turns each page image into speaker-labeled lines. Code merges continuation lines and matches names against the case roster | Nebius Token Factory |
| 2. List key details | **Nemotron 3 Ultra** drafts them from the case documents. We reviewed each one and added a source (2016 federal habeas decision, page cites; public-news dates) | Nebius Token Factory |
| 3. Label every sentence | **Nemotron 3 Ultra** (reasoning off) marks which details each line carries and how: asks, states, denies, or agrees. Code discards any label whose quoted words aren't in the line, and adds a suspect's bare "yeah" that directly answers a question carrying a detail. (Nemotron 3.5 Lightning is cheaper but missed about half of the suspect's adoptions on our training-writer interviews: 57% vs 87% right first speaker) | Nebius Token Factory |
| 3b. Our matcher | NVIDIA **llama-nemotron-rerank-1b-v2**, fine-tuned to score whether a line carries a detail | Trained on a **Nebius Serverless Job**, served on a **Nebius Serverless Endpoint** |
| 4. Decide who said it first | A plain-code ledger walks the record in order. No model decides. A label counts only if the line has the detail's **key terms** (below) | The app |
| 5. Double-check the deciding lines | **Nemotron 3 Ultra** (reasoning on) re-reads only the lines that decide a verdict and keeps a label only if the line conveys the detail's distinctive part ("in the stomach", not just "stabbed her"). Rejected labels are dropped and the ledger re-runs. Reasoning on beat reasoning off on our training-writer interviews (82% vs 73%) | Nebius Token Factory |
| 6. Investigator lines by code | For details with key terms, code checks every investigator line itself, so no model can miss or veto one, and the page marks them "matched by key words". For details without key terms (the synthetic benchmark, pasted transcripts), a safety search has Ultra check the earlier investigator lines that share the detail's words before the suspect is credited | Code (+ Nebius Token Factory without key terms) |

### Key terms: the code check that has the last word

Each detail in a case's fact list carries `must`: groups of words that make it specific, with alternates in each group. "Her hair was cut" is `[["cut", "chop", "hack", ...], ["hair"]]`; "Stabbed in the stomach" is `[["stomach", "belly", "gut", "abdomen"], ["stab", "stabbed", "knife", "cut"]]`. A line carries the detail only if it has at least one alternate from every group, within one or two consecutive sentences of a long line. **Investigator lines are decided by code alone**: no model can veto a match, and the page marks them "matched by key words". The suspect's lines are labeled by Nemotron 3 Ultra (temperature 0) and checked by the same code. A bare "yeah" or a short answer (8 words or fewer) is read with the 2 lines before it and with the investigators' questions in the last 6 turns that name part of the detail, so "With the shovel." answering a question about the bones counts. An investigator's short line may borrow the lines before it only if it names at least one group itself, and only from other investigator lines, so an investigator never gets credit for the suspect's words. Code applies this after the model labels and after Ultra's check, and before Ultra even sees a safety-search candidate, so a model can't make a line count that lacks the words.

**Partly supplied.** When investigators named part of a detail before he stated all of it, the detail counts as **investigators first**, the conservative call for counsel, and its card states both halves with pages. Code alone decides this. It looks at the exchange leading up to his first statement: walking back in the same session, the lines that name any key-term group, as long as no more than 10 printed turns pass without one, and always the last 6 turns. It also counts investigator lines the labeler tied to the detail anywhere earlier (these can only add investigator credit). A group counts as theirs unless he said a word of it first, in that exchange or in his own lines about the detail. When they named every group, in separate lines, the card says so. For example, the phone, camera and purse in the burn barrel: "Investigators named every part of it before he said it, in separate lines: 'phone', 'cell', 'camera' and 'purse' (Feb 27, p. 474); 'burn' and 'barrel' (Mar 1, p. 619). He then said it (p. 620)."

**How the ledger reads the record.** These rules are the same for every detail.
- **Echoes don't count.** An investigator line that repeats key-term words from the suspect's turn just before it is shown as an echo, not counted as investigators naming the detail.
- **Turns are printed turns.** A nod split out of an investigator's line (the transcript's '(nods "yes")') is a line for labeling but not a turn, so it doesn't shorten a window.
- **Earlier conversations.** A detail counts as possibly from an earlier, unrecorded conversation when an investigator's line refers to one: talked or spoke to him, or he told or said to them, tied to an earlier time or place ("the other night", "last time", a weekday, "at the school"), or "the other night" on its own, or "off the tape". A bare "you told us before" doesn't count, because it usually refers to earlier in the same recorded interview. The card cites the investigator line and his first statement; it says nothing about what was said off the record.
- **Already public** uses only what the court listed as reported before March 1, 2006 (Decision and Order pp. 69–70, citing the Herald Times Reporter of Nov 16 and Dec 7, 2005) and other news we could date. Each public detail's card says what was reported and where.
- **Page citations** give a range ("pp. 469–470") when a turn continues onto the next printed page.
- **His written statement.** In the Feb 27 transcript, Wiegert reads Brendan's written statement aloud. Those lines are counted as Brendan's, and the card discloses it: "read aloud by Wiegert from Brendan's written statement (printed speaker: Wiegert)".
- **Nearby prompts.** Each card also shows, without changing the verdict, the investigator lines in the 10 turns before his first statement that name part of the detail, and any word of the detail's label that an investigator said first, within 10 turns of his statement, in a line about the detail (one that names a key-term group, or a short line right after one). For example, "unconscious" (Wiegert twice on p. 579) isn't a key term because it's a degree word, so it's shown rather than counted. A lone back, up, down, in, out, on or off is never shown, by the same rule as the key terms.

**Tightened once after review.** After the first full review, one rule was applied to every detail's key terms: a lone function-like word (back, up, down, in, out, on, off) is only allowed inside a phrase. The reason: "going back to Mark's line of questioning ... the car" matched "her body was in the back of the RAV4". Two details changed: F04 ("in the back", "back of", "the back", "rear", "cargo", "trunk") and F16 ("off" only in "took off", "taken off", "take off", "came off", "plates off").

**The suspect's side, searched the same way.** Code also finds the suspect's lines that carry every key-term group but have no label the ledger uses (none, or one the verifier rejected), and Ultra (temperature 0) answers only whether each is him stating or agreeing to it, not a denial or another sense. Those answers are saved, so re-running the audit makes no new calls and gives the same result.

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
- Borderline model calls can change between runs (even at temperature 0), so verdicts that rest on the suspect's lines are for review, not conclusions. The investigator side doesn't move: with key terms, code alone decides which investigator lines carry a detail, and it errs toward crediting investigators (a common word like "back" can match a line about something else).
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

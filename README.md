# Provenance

**Who said it first?** Provenance reads every recorded interview with a suspect and finds, for each key crime detail, the first time it appears in the record and who said it: the investigators or the suspect.

A confession is treated as reliable when the suspect knows things only the culprit could know. But that works only if the investigators didn't say those things first. In 2006, 16-year-old Brendan Dassey "knew" Teresa Halbach was shot in the head. The first person in the record to mention a head shot was an investigator: *“All right, I'm just gonna come out and ask you. Who shot her in the head?”*

Checking this by hand means reading hundreds of transcript pages, sometimes for weeks. Provenance does it in minutes, with a page citation for every claim.

**Live demo: [provenance-2401.onrender.com](https://provenance-2401.onrender.com)** (free hosting: the first visit after a quiet spell can take about a minute to wake up). Built for the Nebius × NVIDIA Global AI Hackathon (Best Apps and Agents).

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
- **A short memo for counsel, marked as a draft.** Code writes the key points by a fixed rule, so both directions are always covered: the 3 details investigators said first with the most investigator mentions, the first 3 that came from him first, then every public, earlier-statement and never-adopted detail. It also lists every detail. Each point is the detail's card in words, with the same qualifiers and page citations: partly public, related reports, words of the detail that came from investigators, the court's observations, and what he answered when a detail was put to him. Nemotron 3 Ultra writes only a 3 to 4 sentence summary. Code rejects it if it uses loaded words ("fact-feeding", "volunteered", "repeated", "supplied", "nothing earlier"), or quotes anything that isn't a contiguous part of one transcript line. If it fails twice, code writes the summary instead. The same quote and wording checks run on the code-written points. Details that may trace to an unrecorded conversation reach the model by label only, so the summary can't describe what was said there.

## How it works: models read, code decides

| Step | What runs | Where |
|---|---|---|
| 1. Read scanned transcripts | **Nemotron 3.5 Super VL** turns each page image into speaker-labeled lines. Code merges continuation lines and matches names against the case roster | Nebius Token Factory |
| 2. List key details | **Nemotron 3 Ultra** drafts them from the case documents. We reviewed each one and added a source (2016 federal habeas decision, page cites; public-news dates) | Nebius Token Factory |
| 3. Label every sentence | **Nemotron 3 Ultra** (reasoning off) marks which details each line carries and how: asks, states, denies, or agrees. Code discards any label whose quoted words aren't in the line, and adds a suspect's bare "yeah" that directly answers a question carrying a detail. (Nemotron 3.5 Lightning is cheaper but missed about half of the suspect's adoptions on our training-writer interviews: 57% vs 87% right first speaker) | Nebius Token Factory |
| 3b. Our matcher (alternative labeler) | NVIDIA **llama-nemotron-rerank-1b-v2**, fine-tuned to score whether a line carries a detail. It can replace step 3: it matches the Ultra labeler's accuracy on the benchmark for about 30% less model cost (results below). The Dassey audit uses the Ultra labels | Trained on a **Nebius Serverless Job**, served on a **Nebius Serverless Endpoint** |
| 4. Decide who said it first | A plain-code ledger walks the record in order. No model decides. A label counts only if the line has the detail's **key terms** (below) | The app |
| 5. Double-check the deciding lines | **Nemotron 3 Ultra** (reasoning on) re-reads the lines that decide a verdict. On the investigator side it never removes a key-term match. On the suspect side, for each detail, it reads his lines that carry the key terms in order, up to his first statement, one line per call, each with the question it answers and the 2 lines before that question. It's given the detail's label and key words only. It first writes what his line is about and where it puts it, then answers states, agrees, denies or other. "Other" covers a line that puts the detail somewhere else ("by the garage" or "outside" for "inside the garage"), is about a thing the lines call by a name outside the key words, or disclaims a part ("I don't know where it was"). Naming who did it states the detail; "I don't know" states nothing. The earliest line it confirms is his first statement. Answers are saved, so re-audits make no calls | Nebius Token Factory |
| 6. Investigator lines by code | For details with key terms, code checks every investigator line itself, so no model can miss or veto one, and the page marks them "matched by key words". For details without key terms (the synthetic benchmark, pasted transcripts), a safety search has Ultra check the earlier investigator lines that share the detail's words before the suspect is credited | Code (+ Nebius Token Factory without key terms) |

### Key terms: the code check that has the last word

Each detail in a case's fact list carries `must`: groups of words that make it specific, with alternates in each group. "Her hair was cut" is `[["cut", "chop", "hack", ...], ["hair"]]`; "Stabbed in the stomach" is `[["stomach", "belly", "gut", "abdomen"], ["stab", "stabbed", "knife", "cut"]]`. A line carries the detail only if it has at least one alternate from every group, within one or two consecutive sentences of a long line. **Investigator lines are decided by code alone**: no model can veto a match, and the page marks them "matched by key words". The suspect's lines are labeled by Nemotron 3 Ultra (temperature 0) and checked by the same code. A bare "yeah" or a short answer (8 words or fewer) is read with the 2 lines before it and with the investigators' questions in the last 6 turns that name part of the detail, so "With the shovel." answering a question about the bones counts. An investigator's short line may borrow the lines before it only if it names at least one group itself, and only from other investigator lines, so an investigator never gets credit for the suspect's words. Code applies this after the model labels and after Ultra's check, and before Ultra even sees a safety-search candidate, so a model can't make a line count that lacks the words.

**Partly supplied.** When investigators named part of a detail before he stated all of it, the detail counts as **investigators first**, the conservative call for counsel, and its card states both halves with pages. Code alone decides this. It looks at the exchange leading up to his first statement: walking back in the same session, the lines that name any key-term group, as long as no more than 10 printed turns pass without one, and always the last 6 turns. It also counts investigator lines the labeler tied to the detail anywhere earlier (these can only add investigator credit). A group counts as theirs unless he said a word of it first, in that exchange or in his own lines about the detail. When they named every group, in separate lines, the card says so. For example, the phone, camera and purse in the burn barrel: "Investigators named every part of it before he said it, in separate lines: 'phone', 'cell', 'camera' and 'purse' (Feb 27, p. 474); 'burn' and 'barrel' (Mar 1, p. 619). He then said it (p. 620)."

**How the ledger reads the record.** These rules are the same for every detail.
- **Echoes don't count.** An investigator line that repeats key-term words from the suspect's turn just before it is shown as an echo, not counted as investigators naming the detail.
- **Places.** When the detail puts something in or on a place ("Shot inside the garage"), an investigator line that puts it by, beside, outside, near or behind that place ("you put 'er by the garage, and Steve shoots her") doesn't name that part. This is the same place rule the suspect-side check applies.
- **Turns are printed turns.** A nod split out of an investigator's line (the transcript's '(nods "yes")') is a line for labeling but not a turn, so it doesn't shorten a window.
- **Earlier conversations.** A detail counts as possibly from an earlier, unrecorded conversation only when the investigator turn that first names it reports one, and no earlier recorded line carries it. A turn reports an earlier conversation when it has a time reference ("the other night", "last time", a weekday, "at the school", "off the tape") and a talk or report verb (talked, spoke, told, said, mentioned). Then any sentence of that turn, or two consecutive sentences, that carries the detail counts. On p. 547 "I talked to ya the other night ..." is followed in the same turn by "We talked about cleaning somethin' up in that garage. You told me that you thought thinking back now there was blood.", so blood on the garage floor counts. "You ate that whole pizza the other night" has no talk verb, so it counts for nothing. "You told us last time" counts for nothing when last time was recorded, because the recorded line decides by order. An earlier recorded line means an investigator line that has the key terms in its own words, or a line of his the ledger counts as a statement. A bare "you told us before" doesn't count, because it usually refers to earlier in the same recorded interview. A statement a court record reports from an unrecorded meeting counts the same way, quoted with the record's page. Nothing is said about what was said off the record.
- **Already public** means every part of the detail was reported before he said it: every key-term group is stated by a report the court listed (Decision and Order pp. 69–70: reported "long before" the March 1 confession, which we read as before all three recorded interviews) or by news we could date ourselves (CNN, Nov 10, 2005; Badger Herald, Nov 16, 2005). The court's "see, e.g." citations are its examples, not report dates, so the card says "reported before March 1, 2006 (Decision p. 69)". When only some parts were reported, the card says "Partly public", names what was and wasn't reported, and the parts that weren't reported decide the verdict through the transcript. Investigators naming a reported part doesn't count as supplying it. A report the detail could only be inferred from (her blood in the cargo area, for "her body was in the back of the RAV4"; rifle casings in the garage, for "shot inside the garage") is listed as related, and the transcript decides.
- **Page citations** give a range ("pp. 469–470") when a turn continues onto the next printed page.
- **His written statement.** In the Feb 27 transcript, Wiegert reads Brendan's written statement aloud. Those lines are counted as Brendan's, and the card discloses it: "read aloud by Wiegert from Brendan's written statement (printed speaker: Wiegert)".
- **Nearby prompts.** Each card also shows, without changing the verdict, the investigator lines in the 10 turns before his first statement that name part of the detail, and any word of the detail's label that an investigator said first, within 10 turns of his statement, in a line about the detail (one that names a key-term group, or a short line right after one). For example, "unconscious" (Wiegert twice on p. 579) isn't a key term because it's a degree word, so it's shown rather than counted. A lone back, up, down, in, out, on or off is never shown, by the same rule as the key terms.
- **The court's observations.** A detail can carry short notes from the Decision about how it came out, each with the Decision page; code adds the transcript page of every line the note quotes. For example, for "Her hair was cut": "The Decision (p. 70) notes that investigators told him again and again that they knew 'something else was done' and 'Something with the head' (p. 584) before he first said Avery cut off her hair." Cards and the memo show them as the court's observation; they don't change the verdict.
- **Earlier related investigator mentions.** Up to 3 investigator lines before his first statement that name the detail's most specific part (its key-term group on the fewest lines of the record, if that's at most 2% of them), for example "stomach area" on p. 452 before his written statement says "stomach". Single words can match lines about other things, so these are shown, not counted.
- **Quotes.** For a long line, the card and table quote the sentence that carries the detail, not the start of the line.

**Tightened once after review.** After the first full review, one rule was applied to every detail's key terms: a lone function-like word (back, up, down, in, out, on, off) is only allowed inside a phrase. The reason: "going back to Mark's line of questioning ... the car" matched "her body was in the back of the RAV4". Two details changed: F04 ("in the back", "back of", "the back", "rear", "cargo", "trunk") and F16 ("off" only in "took off", "taken off", "take off", "came off", "plates off").

**The suspect's side, searched the same way.** Code also finds the suspect's lines that carry every key-term group, whether or not a label or an earlier check covers them. Ultra (temperature 0) reads each with the question it answers and the 2 lines before that question, and says only whether he is stating or agreeing to this detail (step 5). Those answers are saved, so re-running the audit makes no new calls and gives the same result.

How the key terms are written: from each detail's own wording and the court's description only, one group per component that makes the detail specific (what was done, to what, with what, or where), with synonyms and the case's own words for things (Dassey calls the RAV4 "the jeep"). Degree words ("until unconscious") and things every line about the crime shares (her, Avery) aren't groups. They are never written by looking at which transcript lines would match, because that would fit the answer. In the product, counsel writes them with the fact list. For the synthetic benchmark there is no such field, and deriving groups mechanically from each label (every content word its own group) halved accuracy on the training writer's interviews (82% to 41%) because the writers vary the wording on purpose, so the benchmark doesn't use key terms.

**Drafted key words: measured, and switched off.** We tried having Nemotron 3 Ultra draft each detail's key words from its label and description only (never the transcript; [`scripts/draft_key_terms.py`](scripts/draft_key_terms.py)), then running the same key-word pipeline as the Dassey audit. On the dev slice of the training writer's interviews (6 interviews, 44 details, same labels for both) it got **61% right first speaker against 84% without key words** ([`eval/RESULTS_dev.md`](eval/RESULTS_dev.md)). The drafts were too strict for how people actually talk: they asked for every part of a detail ("blue" + "baseball cap" + "frayed brim"; "cut on left hand" as one phrase), while speakers mention a detail in part ("that cap"). So drafting is off by default (`KEYWORD_DRAFTING=1` turns the button on). The Try section still takes key words a person types, one line per detail ("cut/chop + hair"), and then code alone decides the investigator side. The test split wasn't run for this; if drafting is improved, measure it on dev first, then once on test.

### Confidential files: private mode

Defense files can't go to an outside AI service without proper agreements. Private mode keeps every step on infrastructure the user controls:

| Step | Normal mode | Private mode |
|---|---|---|
| Read the lines and label which details each carries | Nemotron 3 Ultra on Nebius Token Factory | **Our fine-tuned matcher** (NVIDIA `llama-nemotron-rerank-1b-v2`, fine-tuned on Nebius), on a Nebius Serverless Endpoint **in the user's own Nebius project**, or self-hosted |
| Bare "yeah" answers, key-word check, who said it first | Code | Code |
| Re-check the deciding lines | Nemotron 3 Ultra | Not run |
| Draft key words | Nemotron 3 Ultra (optional) | **Not available**: type them by hand, or draft them with a model you host yourself |
| Memo | Nemotron 3 Ultra summary + code points | Code only |

Every model in private mode is open: the matcher's weights are ours to deploy anywhere, and the base model is NVIDIA's open `llama-nemotron-rerank-1b-v2`. Token Factory is not called at all. (For normal mode, Nebius says Token Factory offers zero-retention inference in EU or US datacenters and holds SOC 2 Type II, HIPAA, ISO 27001 and ISO 27799 certifications, per its [launch announcement](https://nebius.com/newsroom/nebius-launches-nebius-token-factory-to-deliver-production-ai-inference-at-scale); a firm should still confirm the terms for its own files.)

How accurate it is: recomputed from the saved matcher labels of test run 4 with no new model calls ([`eval/private_from_saved.py`](eval/private_from_saved.py), [`eval/private_mode.json`](eval/private_mode.json)), private mode got **83% right first speaker, 82% right first line, and credited 0 investigator-supplied details to the suspect** on the 29 held-out interrogations. The same labels with Nemotron 3 Ultra's check on top got 71%, because on these synthetic interviews the check rejected more correct matcher labels than wrong ones. That comparison comes from the test set, so we haven't changed the main pipeline because of it; it should be confirmed on the training writer's interviews first. On the Dassey record, private mode hasn't been run.

In the app: the Try section has a **Private mode** switch. It needs `ENDPOINT_URL` (the matcher endpoint); without it the switch explains that and stays off. The public demo server doesn't keep the endpoint running.

## Start from a recording

In **Try it on your own transcript**, "Start from a recording" takes an audio or video file of an interview (up to 30 minutes and 200 MB) and fills the transcript box. Nothing is stored. The upload sits in a temporary file only while the request runs.

| Step | What runs | Where |
|---|---|---|
| 1. Get the sound | ffmpeg takes the audio track from any audio or video file and makes it 16 kHz mono | The app |
| 2. Words with times | NVIDIA **Parakeet TDT 0.6B v3** transcribes in about 16 s chunks, each cut at the quietest moment nearby and padded with half a second of silence. Code then re-transcribes any stretch of clear voice that came back without words (below) | **Nebius Serverless Endpoint** (`/asr` with `timestamps`) |
| 3. Who spoke when | NVIDIA **Nemotron-3-Diarization** (up to 8 speakers, numbered in order of first appearance), run at the same time as step 2 | **Nebius Serverless Endpoint** (`/diarize`) |
| 4. Turns | Plain code gives each word the speaker whose segment overlaps it most. Each sentence goes to the speaker who holds most of its time, so a word at a turn boundary doesn't split it. Consecutive sentences by one speaker become a turn: "SPEAKER 1: ..." | The app |
| 5. Who is who | **Nemotron 3.5 Lightning** proposes which speaker is the suspect and which are investigators, from what they say. Code rejects a proposal that skips a speaker or doesn't name exactly one suspect, and then falls back to a rule: the speaker asking the most questions is an investigator, and the one asking the fewest is the suspect. **The user confirms or changes each speaker** (Investigator / Suspect / Other), seeing sample lines | Nebius Token Factory |
| 6. Audit | "Use this transcript" puts the turns in the form as SUSPECT / DET 1 / OTHER 1. It stays editable, and the usual audit runs | As above |

If the endpoint is off, the page says "Speech recognition needs our Nebius endpoint, which is off right now. You can still paste a transcript." It never shows a made-up transcript. The diarizer (0.4 GB) loads when the endpoint starts, next to Parakeet.

**Measured** ([`eval/audio_test/`](eval/audio_test), `python -m scripts.audio_eval [--name hard]`). The test recordings are scripted interrogations (invented people and events) voiced with Kokoro-82M (`scripts/make_test_audio.py`), so every word, speaker and turn time is known. All runs were on CPU with the exact server code the endpoint runs.

| Recording | Voices | Word error rate | Words given to the right speaker | Script turns kept whole | Roles proposed by the fallback rule |
|---|---|---|---|---|---|
| `interrogation.mp3`, 2.5 min, 28 turns, 300 words | Investigator (female), suspect (male) | 1.0% (all 3 errors are "grey" for "gray") | 100% | 28/28 | Right |
| `interrogation_hard.mp3`, 2.2 min, 28 turns, 300 words | Two investigators and the suspect, all male | 2.7% (3× "grey", "3:15" for "three fifteen", one swapped pair of words) | 100% (3 speakers found) | 28/28 | Right |

On CPU, 2.5 minutes took 21 s to transcribe and 6 s to diarize; a GPU is much faster. Nine minutes of real speech took 12 s to diarize on CPU.

**Same audit from the recording as from the script** (`python -m scripts.audio_audit_compare [--name hard]`, [`audit_compare.json`](eval/audio_test/audit_compare.json)). We ran the live audit twice per recording with the same 6 facts: once on the script, and once on the transcript from the recording, with speakers named the way the "Who is who?" step names them. Every fact got the same origin and corroboration, 6 of 6 on both recordings, with identical counts. The four runs cost $0.32 of Token Factory. Nemotron 3.5 Lightning, given the recording transcripts, named the roles correctly on both, including the two investigators in the 3-voice recording.

**What we found, and what we changed.** On the first runs, Parakeet sometimes returned nothing for several seconds of clear synthetic speech, or for a whole chunk. With the same audio it gave 0, 19 or 36 words depending on half a second of padding. A NeMo-style greedy decoding loop we wrote over the same encoder gave the same output as transformers' `generate`, so this comes from the model's encoder output on these inputs, not from the decoding code. Three changes:
- **Chunk length and padding.** These were chosen on a separate dev recording: Faithful's voiced turns, 3.4 minutes. 16 s chunks had 5.9% word error and 28 s chunks 6.3%.
- **Noise floor.** A noise floor at −80 dBFS is added, because the synthetic audio has exact digital silence between turns (22% of the test clip). Real microphones never produce that, but noise gates and some call recorders do.
- **Re-transcription.** Two safety nets re-transcribe a chunk, or a stretch of voice, that came back without words.

On the dev recording and on 9 minutes of real read speech (LibriSpeech, 73 utterances), word error didn't change (6.1% and 3.7%), and the safety nets never fired. On real speech, no chunk was ever dropped, even without these changes. Before them, the test clip's word error ranged from 3.0% to 18.3% depending on chunk length, all of it from dropped stretches (the real-speech check used the 73 LibriSpeech utterances in `hf-internal-testing/librispeech_asr_dummy`). A 30-second video cut from the test clip (H.264/AAC) still loses its last sentence, which the cut ends mid-word.

**Limits.** These are clean synthetic voices with no noise, an easy case. A real interview room recording (one microphone, people talking over each other, a poor recording) will do worse, and we haven't measured that. Speakers talking at the same time get one speaker per sentence. The role proposal is only a proposal: the user decides.

## Results

- **Synthetic interrogations with known answers** ([`eval/RESULTS.md`](eval/RESULTS.md)). 29 held-out interrogations (203 details) written by Qwen3 and GLM-5.2, model families different from the training writer; each is kept only if its own labels match the plan we drew in code. Settings (labeler, verifier reasoning) were chosen on the training writer's interviews, never on these.

  | Method | Right first speaker | Right first line | Investigator-supplied details credited to the suspect | Model cost (29 interviews) |
  |---|---|---|---|---|
  | Keyword match | 56% | 55% | 3 | $0 |
  | One Nemotron 3 Ultra prompt per interview | 82% | 79% | 0 | $0.23 |
  | Provenance (Ultra labels) | 71% | 70% | 0 | $2.37 |
  | Provenance with our fine-tuned matcher as the labeler | 71% | 69% | 0 | $1.65 |

  This is the fourth run on held-out test data (the earlier ones, with Provenance at 73% and 77% and the one prompt at 80–82%, are all listed in `eval/RESULTS.md`; the code changed between runs as the real record was reviewed). On short 30 to 50 line interviews, a single Ultra prompt is the stronger method, and we say so on the page. Provenance's job is different: the real record is 5,757 lines across three recorded sessions, and every verdict cites lines that code checks and a person can open. We tried one Ultra prompt per real session for the 25 Dassey details: for the 3,146-line March 1 session it returned no usable answer (its reasoning ran past 32,000 output tokens twice), so we have no measured comparison on the real record.
- **Fine-tuning on Nebius** ([`eval/training.json`](eval/training.json)). The matcher was trained on 6,396 sentence-detail pairs from the training writer's interviews, 2 epochs on one L40S in a Nebius Serverless Job (209 seconds). On the held-out test pairs: AUC 0.95 before fine-tuning, 0.99 after; F1 at the dev threshold 0.68 before, 0.83 after.
- **The Dassey record.** Three recorded interviews (Feb 27, Mar 1 and May 13, 2006; 305 scanned pages). The audit reads: "Of 24 key details Brendan Dassey stated or agreed to, investigators said 7 first (1 of them in part), 5 were already public, 2 may trace to an earlier statement with no public transcript. 10 details came from him first in the record." It is a draft for counsel. Before publishing, an independent reviewer checked every verdict and citation against the transcript PDFs and the court's Decision and Order over three rounds; the remaining known issue is listed under Honest limits. Reading quality: on hand-transcribed gold pages, Super VL's word error rate was 0.15% with every speaker right, across 687 words on three pages (Tesseract OCR: 3.05%).

## Honest limits

- Unrecorded contacts can't be traced. The Feb 27 Two Rivers interview has no public transcript, so details it touched are marked "unclear" when referred to later.
- "Already public" relies on the reports the court listed and the news we could date. A detail is public only if they state every part of it.
- The labels come from models and can be wrong. Every claim in the UI links to the exact line so a person can check it.
- Borderline model calls can change between runs (even at temperature 0), so verdicts that rest on the suspect's lines are for review, not conclusions. The investigator side doesn't move: with key terms, code alone decides which investigator lines carry a detail, and it errs toward crediting investigators (a common word like "back" can match a line about something else).
- The suspect-side check reads only the lines around his answer. When the transcript uses one of the detail's own words for a different thing, it can't tell them apart. On Mar 1 he calls Avery's Suzuki "the jeep" (p. 548), and "jeep" is a key word for her RAV4. So "In the garage", said about the Suzuki, is counted as his first statement that her vehicle was in the garage, instead of "Her jeep." a few lines later. Giving the model the fact list's longer description fixed that line, but caused four wrong calls elsewhere, so it isn't used. The verdict (investigators first) is the same either way.
- **Not yet validated by independent false-confession experts.** The [validation protocol](docs/validation_protocol.md) sets out the study: two blind expert coders, Cohen's kappa between them and against Provenance, and the safety metric (investigator-supplied details called "suspect first"). The citation table export already has the blank expert columns.
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
| `ENDPOINT_URL` and `ENDPOINT_TOKEN` | Our Nebius endpoint: the fine-tuned matcher, private mode in the Try section, and starting from a recording (also needs ffmpeg, which the Dockerfile installs) |
| `KEYWORD_DRAFTING=0` | Hides "Draft key words with Nemotron" (hand-written key words still work) |

GitHub Actions has two workflows:
- **Run case pipeline:** ingest, facts, tag.
- **Benchmark.**

The Nebius parts are in `train/`.

Sources: Calumet County Sheriff's Department interview transcripts (public court records, as published); *Dassey v. Dittmann*, Decision and Order (E.D. Wis. Aug 12, 2016); CNN transcript, Nov 10, 2005; *The Badger Herald*, Nov 16, 2005.

## License

Apache-2.0.

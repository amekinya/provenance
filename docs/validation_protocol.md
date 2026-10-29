# Validating Provenance with independent experts

**Status: not done yet.** Provenance has been checked by an independent reviewer against one case (Dassey) and measured on synthetic interrogations with known answers. It has **not** been validated by independent false-confession experts. This is the protocol for that study, ready to run.

## The question

For each key detail a suspect stated or agreed to, does Provenance identify who put it into the record first, and the first line, as well as independent experts do? The categories are:

- investigators first (fully or in part);
- the suspect first;
- already public;
- may trace to an unrecorded contact;
- never adopted.

## Who codes

- **Two independent experts in false confessions.** These are people who have coded interrogations for contamination before, and who had no part in building Provenance.
- **A third expert** settles disagreements.
- **Blind coding.** The coders don't see Provenance's output, or each other's coding, until all coding is finished.

## Materials

- **Cases:** about 10 interrogation records whose full transcripts are public court records, for example from habeas filings or exoneration files.
  - Include disputed and undisputed confessions, short and long interrogations, and several jurisdictions. Don't use only famous cases.
  - Exclude the Dassey case, which was used during development.
- **Key details:** for each case, a fixed list written from the court record (decision, charging documents) **before anyone reads the transcript**, by someone who isn't a coder.
  - Each detail gets its key words, under the README's rule: one group per component that makes the detail specific.
  - The list and key words are frozen before coding starts.
- **Public reports:** for each case, the news reports from before the interrogation, gathered by someone who isn't a coder. Coders may use them only for the "already public" call.

## What each coder records, per detail

Use the citation table exported from the page ("For lawyers" → "Download the citation table"). It has blank columns for Expert A, Expert B and the consensus. For each detail, record:

1. **Did the suspect state or agree to it?** A denial, or "I don't know", is not agreement.
2. **The first line where investigators named it:** session and page, or "none".
3. **The first line where the suspect stated or agreed to it:** session and page.
4. **The category:**
   - investigators first (fully / in part, and which part);
   - suspect first;
   - already public (which report);
   - unrecorded contact (which line refers to it);
   - never adopted.
5. **Notes:** anything ambiguous, such as words used in another sense or a statement read aloud by someone else.

## Measures

- **Agreement between the experts:** Cohen's kappa on the category, and the share of details where both name the same first line.
- **Provenance against the expert consensus:**
  - kappa and percent agreement on the category;
  - exact first-line agreement;
  - the full confusion matrix.
- **The safety metric** (the costly error for counsel): how many details Provenance calls "suspect first" that the consensus calls "investigators first". Report the count and rate with a 95% Wilson interval. The target is zero.
- **The other direction:** details Provenance calls "investigators first" that the consensus calls "suspect first". This error understates what the suspect knew.
- **Time:** minutes per case for an expert with Provenance's citations, against without. Measure it in a separate session, with different cases per condition.
- **Precision:** with about 10 cases of about 20 details each (about 200 decisions) and kappa near 0.8, the 95% interval on kappa is roughly ±0.07.

## What counts as an error

- A **verdict error** is a category that differs from the consensus.
- A **citation error** is a right category with a first line the consensus disagrees with. It only becomes a verdict error if the correct line would change the order.
- A **"public" error** is calling a detail public when the reports don't cover every part of it. The same standard applies to Provenance and to the coders.

## Fixed in advance (pre-registration)

Before any results are seen, write down and don't change:

- the Provenance commit;
- the model settings (labeler, verifier, temperature);
- the matcher threshold;
- the key-word rule;
- the case list and the detail lists.

Report every case, including any where Provenance fails to run.

## Running Provenance for the study

1. Ingest each transcript and run the tag step (`.github/workflows/pipeline.yml`), keeping `run.json`.
2. Export the citation table per case and hand it to the adjudicator only after both experts are done.
3. For confidential files, use private mode: our fine-tuned matcher on an endpoint inside your own Nebius project, plus code. See the README.

## Worksheet template

| Detail | Source in the court record | Key words | Provenance: first said by | Provenance: first investigator line | Provenance: first suspect line | Expert A: first said by | Expert A: first line | Expert A: notes | Expert B: first said by | Expert B: first line | Expert B: notes | Consensus: first said by | Consensus: first line |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e.g. "Her hair was cut" | Decision p. 70 | cut/chop/hack + hair | | | | | | | | | | | |

# Provenance benchmark

29 synthetic interrogations (test split) written by Qwen/Qwen3-30B-A3B-Instruct-2507 and zai-org/GLM-5.2 (model families different from the training writer), 203 details with known first speakers. This is the third run on held-out test data and the second on this 29-interrogation set, after the denial-of-guilt labeling fix and the reasoning verifier chosen on the training writer's interviews.

| Method | Right first speaker | Right first line | Investigator-supplied details wrongly credited to the suspect | Time | Cost |
|---|---|---|---|---|---|
| Keyword match (no model) | 56% | 55% | 3 | 0.1 s | $0.000 |
| Nemotron 3 Ultra, one prompt per interview | 80% | 77% | 2 | 53.9 s | $0.247 |
| Provenance: Nemotron 3 Ultra labels + code ledger + Ultra check + safety search | 77% | 76% | 1 | 541.7 s | $2.264 |

This is test run 3. Earlier runs on the held-out test, newest first:

- 2026-10-27 01:45 UTC (62e80c9): First run on the 29-interrogation test set: Ultra labels, verifier without reasoning.
  - Keyword match (no model): 56% right first speaker, 55% right line, 3 investigator-supplied credited to the suspect
  - Nemotron 3 Ultra, one prompt per interview: 82% right first speaker, 75% right line, 1 investigator-supplied credited to the suspect
  - Provenance: Nemotron 3 Ultra labels + code ledger + Ultra check + safety search: 73% right first speaker, 69% right line, 1 investigator-supplied credited to the suspect
- 2026-10-27 00:50 UTC (a787286): Before the fixes: 6 Qwen-written test interrogations (39 details); the one-prompt baseline then asked a different question than the gold labels.
  - Keyword match (no model): 59% right first speaker, 54% right line, 0 investigator-supplied credited to the suspect
  - Nemotron 3 Ultra, one prompt per interview: 13% right first speaker, 13% right line, 0 investigator-supplied credited to the suspect
  - Provenance: Lightning labels + code ledger + Ultra check: 44% right first speaker, 38% right line, 0 investigator-supplied credited to the suspect

# Provenance benchmark

29 synthetic interrogations (test split) written by Qwen/Qwen3-30B-A3B-Instruct-2507 and zai-org/GLM-5.2 (model families different from the training writer), 203 details with known first speakers.

| Method | Right first speaker | Right first line | Investigator-supplied details wrongly credited to the suspect | Time | Cost |
|---|---|---|---|---|---|
| Keyword match (no model) | 56% | 55% | 3 | 0.1 s | $0.000 |
| Nemotron 3 Ultra, one prompt per interview | 82% | 75% | 1 | 85.1 s | $0.267 |
| Provenance: Nemotron 3 Ultra labels + code ledger + Ultra check + safety search | 73% | 69% | 1 | 343.4 s | $1.980 |

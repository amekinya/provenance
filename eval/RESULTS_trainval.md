# Provenance benchmark

14 synthetic interrogations (trainval split) written by deepseek-ai/DeepSeek-V4.1-Flash-1017 (the training writer; diagnostic split), 95 details with known first speakers.

| Method | Right first speaker | Right first line | Investigator-supplied details wrongly credited to the suspect | Time | Cost |
|---|---|---|---|---|---|
| Keyword match (no model) | 54% | 52% | 4 | 0.0 s | $0.000 |
| Nemotron 3 Ultra, one prompt per interview | 90% | 81% | 0 | 27.3 s | $0.126 |
| Provenance without code agreements or safety search (ablation) | 57% | 53% | 1 | 98.4 s | $0.309 |
| Provenance: Lightning labels + code ledger + Ultra check + safety search | 68% | 65% | 0 | 271.2 s | $0.885 |

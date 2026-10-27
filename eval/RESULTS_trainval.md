# Provenance benchmark

14 synthetic interrogations (trainval split) written by deepseek-ai/DeepSeek-V4.1-Flash-1017 (the training writer; diagnostic split), 95 details with known first speakers.

| Method | Right first speaker | Right first line | Investigator-supplied details wrongly credited to the suspect | Time | Cost |
|---|---|---|---|---|---|
| Provenance: Nemotron 3 Ultra labels + code ledger + Ultra check + safety search | 73% | 72% | 0 | 83.8 s | $0.426 |
| Provenance, verifier with reasoning on (same labels; cost excludes labeling) | 82% | 78% | 0 | 238.2 s | $0.928 |

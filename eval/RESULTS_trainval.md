# Provenance benchmark

14 synthetic interrogations (trainval split) written by deepseek-ai/DeepSeek-V4.1-Flash-1017 (the training writer; diagnostic split), 95 details with known first speakers.

| Method | Right first speaker | Right first line | Investigator-supplied details wrongly credited to the suspect | Time | Cost |
|---|---|---|---|---|---|
| Provenance: Lightning labels + code ledger + Ultra check + safety search | 57% | 56% | 0 | 276.5 s | $0.989 |
| Provenance with Nemotron 3 Ultra as the labeler + code ledger + Ultra check + safety search | 87% | 87% | 0 | 589.4 s | $2.709 |

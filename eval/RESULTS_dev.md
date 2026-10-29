# Provenance benchmark

6 synthetic interrogations (dev split) written by deepseek-ai/DeepSeek-V4.1-Flash-1017 (the training writer; diagnostic split), 44 details with known first speakers.

| Method | Right first speaker | Right first line | Investigator-supplied details wrongly credited to the suspect | Time | Cost |
|---|---|---|---|---|---|
| Provenance: Nemotron 3 Ultra labels + code ledger + Ultra check + safety search | 84% | 80% | 0 | 114.5 s | $0.485 |
| Provenance with key words drafted by Nemotron 3 Ultra (same labels; code decides the investigator side) | 61% | 61% | 0 | 154.5 s | $0.506 |

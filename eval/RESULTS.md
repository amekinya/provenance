# Provenance benchmark

6 synthetic interrogations written by Qwen/Qwen3-30B-A3B-Instruct-2507 (a different model family), 39 details with known first speakers.

| Method | Right first speaker | Right first line | Investigator-supplied details wrongly credited to the suspect | Time | Cost |
|---|---|---|---|---|---|
| Keyword match (no model) | 59% | 54% | 0 | 0.0 s | $0.000 |
| Nemotron 3 Ultra, one prompt per interview | 13% | 13% | 0 | 32.0 s | $0.062 |
| Provenance: Lightning labels + code ledger + Ultra check | 44% | 38% | 0 | 117.2 s | $0.288 |

### LoRA sweep: macro KID x1e3 vs validation by checkpoint (guidance 2, 30 steps)

| Run | 2,000 | 3,000 | 4,000 | 5,000 | 6,000 | 8,000 | 10,000 |
|---|---|---|---|---|---|---|---|
| baseline (rank 8, LR 1e-4, seed 42) | 62.0 | 73.7 | 57.1 | 68.1 | 66.7 | 69.0 | 55.0 |
| baseline, seed 43 | 75.6 | 59.8 | 56.5 | 69.0 | 61.7 | 66.6 | 81.8 |
| rank 4 | - | - | 56.5 | - | 65.5 | 61.1 | 64.3 |
| rank 16 | - | - | 59.0 | - | 58.9 | 53.8 | 65.4 |
| LR 5e-5 | - | - | 55.3 | - | 63.5 | 69.1 | 62.7 |
| LR 2e-4 | - | - | 52.7 | - | 66.2 | 75.0 | 56.5 |
| cosine LR schedule | - | - | 57.7 | - | 72.7 | 69.5 | 68.1 |

Lower is better. '-' = not scored. Same-settings seeds differ by up to 26.8 at 10,000 steps; no rank/LR setting beats that noise. Checkpoint 4,000 is the most consistent across runs.

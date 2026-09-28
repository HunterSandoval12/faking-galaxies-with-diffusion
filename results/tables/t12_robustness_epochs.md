### Robustness check (validation only): classifier training length

| Classifier training | Real-only val. acc. % | Synthetic-only val. acc. % | Gap (pts) | Synthetic as % of real |
|---|---|---|---|---|
| 30 epochs (used) | 86.5 +/- 0.2 | 66.4 +/- 0.2 | 20.1 | 76.8% |
| 50 epochs (check) | 86.1 +/- 0.4 | 66.9 +/- 1.2 | 19.1 | 77.8% |

Seeds 0-2 for both. 50 epochs changes the gap by -0.9 points: the conclusion is not sensitive to the schedule. The test set was not used.

### Generator candidates on validation (30 steps, 100 images/class)

| Candidate | Guidance | Macro KID x1e3 | Macro FID | Class fidelity |
|---|---|---|---|---|
| seed 42 @ 10,000 | 1.5 | 56.6 | 74.7 | 0.436 |
| seed 42 @ 10,000 | 2 | 55.0 | 72.3 | 0.550 |
| seed 42 @ 10,000 | 3 | 57.5 | 72.5 | 0.702 |
| seed 43 @ 4,000 | 1.5 | 60.4 | 81.0 | 0.312 |
| seed 43 @ 4,000 | 2 | 56.5 | 76.8 | 0.397 |
| seed 43 @ 4,000 | 3 | 57.8 | 76.2 | 0.546 |
| seed 42 @ 4,000 | 1.5 | 56.9 | 76.3 | 0.324 |
| seed 42 @ 4,000 | 2 | 57.1 | 74.7 | 0.424 |
| seed 42 @ 4,000 | 3 | 66.8 | 79.2 | 0.568 |

Class fidelity: fraction of generated images a real-trained ResNet-18 assigns to the intended class (mean of 2 classifiers). Chosen: seed 42 @ 10,000 steps, guidance 3 (confirmed by the usefulness pilot).

### Sampling-setting pilots on validation (checkpoint 10,000; 1,000 images per setting)

| Pilot | Guidance | Steps | Macro KID x1e3 | Class fidelity | Synthetic-only val. acc % (2 seeds) | Macro-F1 |
|---|---|---|---|---|---|---|
| guidance | 1.5 | 30 | 56.6 | 0.436 | 50.6 (48.9-52.2) | 0.465 |
| guidance | 2 | 30 | 55.0 | 0.550 | 52.2 (49.3-55.1) | 0.498 |
| guidance | 3 | 30 | 57.5 | 0.702 | 58.8 (58.1-59.5) | 0.554 |
| guidance | 4 | 30 | 61.4 | 0.790 | 53.8 (53.7-53.9) | 0.502 |
| guidance | 6 | 30 | 70.4 | 0.866 | 52.8 (51.8-53.8) | 0.496 |
| steps | 3 | 30 | 57.5 | 0.702 | 58.8 (58.1-59.5) | 0.554 |
| steps | 3 | 50 | 50.2 | 0.687 | 55.5 (55.2-55.9) | 0.517 |
| steps | 3 | 75 | 46.6 | 0.676 | 58.4 (57.8-59.0) | 0.548 |
| steps | 3 | 100 | 45.4 | 0.675 | 58.5 (57.3-59.6) | 0.551 |
| control | real 100/class | - | - | - | 74.4 (73.9-74.8) | 0.728 |

Usefulness = ResNet-18 trained only on the 1,000 generated images, scored on real validation images. Chosen: guidance 3, 30 steps (fewest steps within 2 points of the best).

# Results: figures and tables

Generated 2026-09-27 from saved experiment outputs. Nothing here trains, tunes or selects models.
Regenerate everything with:

```
.\.venv\Scripts\python.exe analysis\compute_extras.py   # GPU pre-computations (cached in results\cache\)
.\.venv\Scripts\python.exe classifier\reproduce_runs.py  # re-runs 14 classifiers with extra logging (fig. 14)
.\.venv\Scripts\python.exe analysis\model_stats.py      # parameter counts + FLOPs (CPU) -> results\model_stats.json
.\.venv\Scripts\python.exe analysis\make_figures.py
.\.venv\Scripts\python.exe analysis\make_tables.py
```

- **Figures:** `figures\*.png` (300 dpi) and `*.pdf` (vector, for the paper). Every chart has a CSV twin with
  exactly the plotted numbers in `figures\data\` (the sample grid, fig. 2, is images only).
- **Tables:** `tables\*.csv` (data), `*.md` (readable), `*.tex` (booktabs, ready to `\input` in the IEEE paper).
- **Statistics:** 95% confidence intervals are paired bootstrap over evaluation images (2,000 resamples), recomputed
  with exactly the method of the pre-registered test evaluation (`classifier\evaluate_test.py`); spot checks match
  the summary written by that one-time evaluation to the last digit (training runs are not included in this repository). "Test" = held-out test set (used once, 2026-09-27).
- **Style:** colour-blind-safe palette (validated: `analysis\validate_palette.py`, a port of the dataviz validator,
  checked against its published values), marker shapes as a second cue, so the figures also work in grayscale print.

## Headline answers

| Question | Answer (held-out test) | Where |
|---|---|---|
| Does a classifier trained on synthetic data perform comparably? | No, not yet: 66.9% vs 86.6% accuracy (77% of real) | Fig. 8, Tab. 6 |
| At what real/synthetic mix does performance peak? | Real only; up to ~10% synthetic costs nothing detectable | Fig. 8, Tab. 6 |
| Does adding synthetic to all real data help? | No (all CIs include 0) | Fig. 10, Tab. 8 |
| When does synthetic help? | When real data is scarce: +2.2 pts [1.2, 3.1] at 10% real | Fig. 9, Tab. 7 |
| Rare vs common class: different optimal ratio? | No measurable effect for either at the class level | Fig. 11, Tab. 9 |
| Why so limited? | Class fidelity: e.g. Cigar-Shaped synthetic images recognised only 28% of the time | Fig. 7, Figs. 12-13, Tab. 11 |
| How well do the classifiers rank each class (ROC AUC)? | Validation only: macro AUC 0.983 real-only, 0.978 50% mix, 0.927 synthetic-only (the test evaluation saved no scores) | Fig. 18, Tab. 14 |

## Figures

| # | File | What it shows |
|---|---|---|
| 1 | `fig01_dataset_splits` | Per-class counts of the working pool / validation / test after cleaning (17,546 images). |
| 2 | `fig02_real_vs_synthetic_samples` | 4 real vs 4 synthetic galaxies per class (synthetic not cherry-picked: sample indices 0-3). |
| 3 | `fig03_diffusion_training` | (a) flat diffusion loss; (b) generator KID by checkpoint for all 7 LoRA runs - large run-to-run swings; chosen generator marked. |
| 4 | `fig04_guidance_pilot` | Choosing guidance: realism (KID) vs class fidelity vs usefulness (synthetic-only classifier) - guidance 3 wins on usefulness. |
| 5 | `fig05_steps_pilot` | Choosing sampling steps: more steps = more realistic, but no more useful; 30 steps chosen (~3.5 h cheaper per set). |
| 6 | `fig06_realism_memorization` | (a) per-class FID of the final set vs real reference; (b) nearest-neighbour similarity: synthetic images are *further* from training images than unseen real ones - no memorization. |
| 7 | `fig07_class_fidelity` | Per class: how often a real-trained classifier recognises synthetic vs real images; Cigar-Shaped (0.28) and Disturbed (0.29) are the weak classes. |
| 8 | `fig08_replacement_curve` | **Primary result**: accuracy and macro-F1 as synthetic replaces real (0-100%), test and validation, 95% CIs. |
| 9 | `fig09_value_of_synthetic_vs_scarcity` | **Key finding**: paired comparison - what synthetic adds grows as real data gets scarcer; clear only at 10% real. |
| 10 | `fig10_augmentation` | Adding 25/50/100% synthetic on top of all real data: no effect. |
| 11 | `fig11_focus_classes` | **Secondary result**: class F1 vs synthetic ratio for Round Smooth, Cigar-Shaped Smooth and the scarcity control - no consistent effect. |
| 12 | `fig12_per_class_f1_heatmap` | Per-class test F1 across the replacement curve - which classes degrade most (Disturbed, Cigar-Shaped). |
| 13 | `fig13_confusion_matrices` | Real-only vs synthetic-only confusion matrices - e.g. synthetic-only confuses Cigar-Shaped with In-between Round (34%) and the two edge-on classes (36%). |
| 14 | `fig14_classifier_learning_curves` | Training vs validation loss and accuracy per epoch for real-only, 50% and synthetic-only (seed means; band = seed range; selected epoch marked). Training accuracy, validation loss and the un-augmented training-set curves come from re-runs with extra logging (`classifier\reproduce_runs.py`), each verified bit-identical to the original run (every logged number and every weight). Validation only. |
| 15 | `fig15_lora_sweep` | Rank / learning-rate / schedule sweep: settings differ less than training seeds do. |
| 16 | `fig16_class_examples` | 3 real example galaxies per class from the training split (seeded random draw), labeled by class - an introductory dataset figure. Made by `analysis\make_class_examples.py` (`--per-class 2` for a smaller version). |
| 17 | `fig17_prediction_examples` | 10 held-out test galaxies (seeded random draw: 2 Cigar-Shaped + 1 from each of 8 other classes, fixed before viewing predictions) with the true class and the real-only / synthetic-only / 50%-mix predictions (majority over seeds, agreement shown), from the saved one-time test predictions. Made by `analysis\make_prediction_examples.py`. |
| 18 | `fig18_roc_curves_val` | One-vs-rest ROC curves per class for real-only, 50% and synthetic-only (seed-mean curves) with per-class AUC, on the **validation** set: the one-time test evaluation saved only predicted classes, not scores, and the test set is not re-run. Probabilities from `analysis\compute_extras.py val_probs` (argmax verified equal to the saved validation predictions). |

## Tables

| # | File | Content |
|---|---|---|
| 1 | `t01_dataset` | Dataset composition: original counts, exact duplicates and overlapping cutouts removed, final splits. |
| 2 | `t02_hyperparameters` | All model, training, sampling and evaluation settings (methods section): optimiser, LR schedule, epochs, batch size, parameter counts (frozen U-Net vs trainable LoRA + class table, VAE, ResNet-18), training / generation time per model, hardware. |
| 3 | `t03_generator_selection` | The three generator candidates x guidance: KID, FID, class fidelity. |
| 4 | `t04_lora_sweep` | KID by checkpoint for all 7 LoRA runs (incl. the 2,000-5,000 scan). |
| 5 | `t05_sampling_pilots` | Guidance and steps pilots: KID, fidelity, synthetic-only usefulness, real control. |
| 6 | `t06_replacement_curve` | Primary result with CIs, differences vs real-only, validation, seeds. |
| 7 | `t07_value_of_synthetic` | Paired real-only vs real+synthetic by amount of real data. |
| 8 | `t08_augmentation` | Augmentation results. |
| 9 | `t09_focus_classes` | Focus-class F1 and differences for all ratios, incl. scarcity control. |
| 10 | `t10_per_class_test` | Per-class precision / recall / F1: real-only, 50%, synthetic-only. |
| 11 | `t11_final_synthetic_set` | Final synthetic set per class: realism, class fidelity, memorization. |
| 12 | `t12_robustness_epochs` | Robustness check (validation only): 50 vs 30 classifier epochs - the gap changes by ~1 point. |
| 13 | `t13_inference_speed` | Inference speed and compute on the RTX 5090: generator 920 ms/image (batch 1) or 479 ms/image (batch 20), 50.8 TFLOPs per image (1.61 TFLOPs per guided U-Net step, 2.51 TFLOPs VAE decode); classifier 1.6 ms (GPU forward) / 1.7 ms (end-to-end) per image at batch 1, 4.74 GFLOPs. Timings: `results\inference_speed.json` (`analysis\benchmark_inference.py`); FLOPs and parameters: `results\model_stats.json` (`analysis\model_stats.py`). |
| 14 | `t14_roc_auc_val` | One-vs-rest ROC AUC per class and macro AUC (validation set) for real-only, 50% and synthetic-only, with 95% paired bootstrap CIs and paired differences. |

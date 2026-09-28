"""Figure: held-out test images with the true label and the predictions of the real-only,
synthetic-only and 50%-mix classifiers.

Usage: python analysis/make_prediction_examples.py [--seed 0]

Uses ONLY the predictions saved by the one-time test evaluation
(runs_cls/exp/<condition>_seed*/test_predictions.npy) - nothing is re-evaluated or selected.
Images: a seeded random draw from the test set, fixed before looking at any prediction -
2 Cigar-Shaped Smooth + 1 image from each of 8 other randomly chosen classes (10 total),
NOT picked by whether predictions were right. Each condition has several seeds; the figure
shows the majority prediction and how many seeds agree.
Output: results/figures/fig17_prediction_examples.png/.pdf + figures/data/ CSV twin.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COL2, CIGAR, H5, INK, INK2, MUTED, SHORT, conditions, preds, save, split_indices, split_labels, style  # noqa: E402

CONDS = [("A_replace0", "Real only"), ("A_replace1", "Synth. only"), ("A_replace0.5", "50% mix")]
TINY = ["Disturbed", "Merging", "Round Smooth", "In-between", "Cigar-Shaped", "Barred Spiral", "Tight Spiral",
        "Loose Spiral", "Edge-on no bulge", "Edge-on w/ bulge"]
GOOD, BAD = "#0ca30c", "#d03b3b"  # status palette (always paired with a glyph + text)


def majority(votes):
    """Most common prediction; ties broken by the lowest-numbered seed's prediction."""
    counts = Counter(votes)
    top = max(counts.values())
    return next(v for v in votes if counts[v] == top), top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    style()

    y = split_labels("test")
    rng = np.random.default_rng(args.seed)
    picks = list(rng.choice(np.flatnonzero(y == CIGAR), 2, replace=False))
    others = rng.choice([c for c in range(10) if c != CIGAR], 8, replace=False)
    picks += [int(rng.choice(np.flatnonzero(y == c))) for c in others]
    picks = sorted(picks, key=lambda p: (y[p], p))  # positions in the test split, ordered by class

    test_idx = split_indices()["test"]
    ds_idx = [int(test_idx[p]) for p in picks]
    order = np.argsort(ds_idx)
    with h5py.File(H5, "r") as f:
        imgs_sorted = f["images"][np.array(ds_idx)[order]]  # h5py needs increasing indices
    imgs = np.empty_like(imgs_sorted)
    imgs[order] = imgs_sorted

    fig, axes = plt.subplots(2, 5, figsize=(COL2, 4.35))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.2, wspace=0.13, hspace=0.62)
    rows = []
    for ax, p, img, di in zip(axes.flat, picks, imgs, ds_idx):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        true = int(y[p])
        ax.text(0.0, -0.05, f"True: {SHORT[true]}", transform=ax.transAxes, fontsize=6.4, color=INK,
                fontweight="bold", va="top")
        row = [p, di, true, SHORT[true]]
        for j, (cond, lab) in enumerate(CONDS):
            votes = [int(preds("test", r)[p]) for r in conditions()[cond]]
            pred, k = majority(votes)
            ok = pred == true
            yy = -0.05 - 0.115 * (j + 1)
            ax.text(0.0, yy, "✓" if ok else "✗", transform=ax.transAxes, fontsize=6.6,
                    color=GOOD if ok else BAD, fontweight="bold", va="top")
            ax.text(0.085, yy, f"{lab}: {TINY[pred]} ({k}/{len(votes)})", transform=ax.transAxes,
                    fontsize=5.9, color=INK2, va="top")
            row += [SHORT[pred], f"{k}/{len(votes)}", ok]
        rows.append(row)
    fig.suptitle("Held-out test galaxies: true class vs. predictions of classifiers trained on real, synthetic and "
                 "mixed data", x=0.005, ha="left", y=0.995, va="top", fontsize=8.5, color=INK, fontweight="semibold")
    fig.text(0.005, 0.945, "Random draw fixed before viewing predictions (2 Cigar-Shaped Smooth + 1 from each of 8 "
             "other classes). Predictions: majority over seeds (agreeing seeds shown), from the saved one-time "
             "test evaluation.", fontsize=6.3, color=MUTED, va="top")
    header = ["test_position", "dataset_index", "true_class", "true_name"]
    for _, lab in CONDS:
        header += [f"{lab} prediction", f"{lab} seeds agreeing", f"{lab} correct"]
    save(fig, "fig17_prediction_examples", rows, header)
    n_ok = {lab: sum(r[4 + 3 * j + 2] for r in rows) for j, (_, lab) in enumerate(CONDS)}
    print(f"saved fig17_prediction_examples: {len(rows)} test images; correct (majority): {n_ok}")


if __name__ == "__main__":
    main()

"""Figure: real example galaxies for each of the 10 Galaxy10 DECaLS classes.

Usage:
    python analysis/make_class_examples.py [--per-class 3] [--seed 0]

Images come from the TRAINING split (working pool) only, chosen by a seeded random draw
(reproducible, not hand-picked). One column per class, labeled with the class name.
Output: results/figures/fig16_class_examples.png + .pdf, and
results/figures/data/fig16_class_examples.csv listing the dataset index of every image shown.
"""

import argparse
import sys
import textwrap
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COL2, H5, INK, INK2, SHORT, labels_all, save, split_indices, style  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=3, choices=[2, 3])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    style()

    train_idx = split_indices()["train"]  # the training split only
    labels = labels_all()
    rng = np.random.default_rng(args.seed)
    picks = {c: np.sort(rng.choice(train_idx[labels[train_idx] == c], args.per_class, replace=False))
             for c in range(len(SHORT))}
    with h5py.File(H5, "r") as f:
        images = {c: f["images"][picks[c]] for c in picks}  # h5py needs increasing indices (sorted above)

    n = args.per_class
    header_in, cell_in = 0.66, 0.70  # space for title + class headers; one square image cell (inches)
    height = cell_in * n + header_in
    fig, axes = plt.subplots(n, len(SHORT), figsize=(COL2, height))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=1 - header_in / height, wspace=0.04, hspace=0.04)
    rows = []
    for c in range(len(SHORT)):
        for k in range(n):
            ax = axes[k, c]
            ax.imshow(images[c][k])
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            rows.append([c, SHORT[c], k + 1, int(picks[c][k])])
        axes[0, c].set_title(f"{c}\n" + textwrap.fill(SHORT[c], 12), fontsize=6.6, color=INK2,
                             fontweight="normal", va="bottom", pad=3)
    fig.suptitle("Galaxy10 DECaLS: real examples of each class (training split, random draw)",
                 x=0.005, ha="left", y=1.0, va="top", fontsize=8.5, color=INK, fontweight="semibold")
    save(fig, "fig16_class_examples", rows, ["class", "name", "example", "dataset_index"])

    # sanity: every image shown is in the training split
    assert all(np.isin(picks[c], train_idx).all() for c in picks)
    print(f"saved fig16_class_examples ({n} per class, seed {args.seed}); all {len(rows)} images from the training split")


if __name__ == "__main__":
    main()

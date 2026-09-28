"""Shared loading, statistics and plotting style for the paper figures and tables.

All statistics are recomputed from saved per-model predictions with EXACTLY the method of
classifier/evaluate_test.py (bootstrap: np.random.default_rng(0), 2,000 resamples of the
evaluation images, the same resample for every model; statistic = mean over seeds;
95% percentile interval), so every number matches runs_cls/exp/summary_<split>.txt.
Nothing here trains, tunes, or selects anything.
"""

import glob
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from prepare_splits import CLASS_NAMES  # noqa: E402

EXP = ROOT / "runs_cls" / "exp"
OUT = ROOT / "results"
FIG, TAB, FIGDATA, CACHE = OUT / "figures", OUT / "tables", OUT / "figures" / "data", OUT / "cache"
H5 = Path.home() / ".astroNN" / "datasets" / "Galaxy10_DECals.h5"
SPLITS = ROOT / "data" / "splits" / "galaxy10_splits.npz"
VAL_SHA1 = "1509d26a5e3a0287"
N_BOOT = 2000
SHORT = ["Disturbed", "Merging", "Round Smooth", "In-between Round", "Cigar-Shaped Smooth", "Barred Spiral",
         "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on w/o Bulge", "Edge-on w/ Bulge"]
ROUND, CIGAR = 2, 4
A_SHARES = ["0", "0.1", "0.25", "0.5", "0.75", "0.9", "1"]
R_SHARES = ["0.1", "0.25", "0.5", "0.75", "0.9"]

for d in (FIG, TAB, FIGDATA, CACHE):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- dataset
@lru_cache(maxsize=None)
def labels_all():
    with h5py.File(H5, "r") as f:
        return f["ans"][:].astype(np.int64)


@lru_cache(maxsize=None)
def split_indices():
    s = np.load(SPLITS)
    return {k: s[k] for k in s.files}


def split_labels(split):
    return labels_all()[split_indices()[split]]


# ---------------------------------------------------------------- classifier runs
@lru_cache(maxsize=None)
def conditions():
    """condition name -> sorted list of run names (e.g. 'A_replace0' -> ['A_replace0_seed0', ...])."""
    out = {}
    for d in sorted(EXP.glob("*_seed*")):
        if (d / "best.pt").is_file():
            out.setdefault(re.sub(r"_seed\d+$", "", d.name), []).append(d.name)
    return out


@lru_cache(maxsize=None)
def preds(split, run):
    return np.load(EXP / run / f"{split}_predictions.npy")


@lru_cache(maxsize=None)
def boot_matrix(split):
    n = len(split_labels(split))
    return np.random.default_rng(0).integers(0, n, size=(N_BOOT, n))


def f1_per_class(y, p):
    k = len(CLASS_NAMES)
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)


@lru_cache(maxsize=None)
def stat(split, cond, kind, cls=None):
    """(point estimate, bootstrap array) of the seed-mean statistic: kind in acc | macro_f1 | cls."""
    y = split_labels(split)
    runs = conditions()[cond]

    def one(idx):
        vals = []
        for r in runs:
            yy, pp = y[idx], preds(split, r)[idx]
            if kind == "acc":
                vals.append((yy == pp).mean())
            else:
                f = f1_per_class(yy, pp)
                vals.append(f.mean() if kind == "macro_f1" else f[cls])
        return float(np.mean(vals))
    return one(np.arange(len(y))), np.array([one(b) for b in boot_matrix(split)])


def summary(split, cond, kind, cls=None):
    """dict(point, lo, hi, seeds, sd) for a condition."""
    p, b = stat(split, cond, kind, cls)
    lo, hi = np.percentile(b, [2.5, 97.5])
    y = split_labels(split)
    per_seed = []
    for r in conditions()[cond]:
        f = f1_per_class(y, preds(split, r))
        per_seed.append((y == preds(split, r)).mean() if kind == "acc" else f.mean() if kind == "macro_f1" else f[cls])
    sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
    return {"point": p, "lo": float(lo), "hi": float(hi), "seeds": len(per_seed), "sd": sd}


def difference(split, a, b, kind, cls=None):
    pa, ba = stat(split, a, kind, cls)
    pb, bb = stat(split, b, kind, cls)
    lo, hi = np.percentile(ba - bb, [2.5, 97.5])
    return {"point": pa - pb, "lo": float(lo), "hi": float(hi)}


def per_class_metrics(split, cond):
    """Seed-mean per-class precision / recall / F1 from saved <split>_metrics.json."""
    ms = [json.loads((EXP / r / f"{split}_metrics.json").read_text()) for r in conditions()[cond]]
    return {k: [float(np.mean([m["per_class"][str(c)][k] for m in ms])) for c in range(len(CLASS_NAMES))]
            for k in ("precision", "recall", "f1")}


def confusion(split, cond, normalize=True):
    cm = sum(np.array(json.loads((EXP / r / f"{split}_metrics.json").read_text())["confusion_matrix"])
             for r in conditions()[cond])
    return cm / cm.sum(1, keepdims=True) if normalize else cm


# ---------------------------------------------------------------- diffusion / FID results
def fid_rows(run):
    p = ROOT / "runs" / run / "fid" / "results.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.is_file() else []


def fid_lookup(run, ckpt, gs, steps, seed=0, n=100):
    m = [r for r in fid_rows(run) if r["checkpoint"] == f"checkpoint-{ckpt}" and r["guidance_scale"] == gs
         and r["sampling_steps"] == steps and r["seed"] == seed and r.get("val_sha1") == VAL_SHA1
         and r["num_per_class"] == n]
    return m[-1] if m else None


def fidelity_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pilot_metrics(pattern):
    ms = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(ROOT / pattern / "metrics.json")))]
    return [m["val"]["accuracy"] for m in ms], [m["val"]["macro_f1"] for m in ms]


# ---------------------------------------------------------------- plotting style (dataviz skill)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated all-pairs, white paper)
MARKERS = ["o", "s", "^"]                              # secondary encoding (aqua is < 3:1 on white)
DEEMPH = "#c3c2b7"                                     # de-emphasis gray for context series
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # sequential blue 100 -> 700
COL1, COL2 = 3.5, 7.16                                 # IEEE single / double column widths (inches)


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
        "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "legend.fontsize": 7.5, "legend.frameon": False, "legend.labelcolor": INK2,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
    })


def dot(ax, x, y, color, marker="o", size=6, label=None, zorder=3):
    """Marker with a 2px white surface ring (dataviz mark spec)."""
    return ax.plot(x, y, linestyle="none", marker=marker, markersize=size, color=color, markeredgecolor="white",
                   markeredgewidth=1.1, label=label, zorder=zorder)


def save(fig, name, data_rows=None, header=None):
    """Save PNG + PDF, and the plotted numbers as a CSV twin (the table view)."""
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    if data_rows is not None:
        import csv
        with open(FIGDATA / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            w.writerows(data_rows)
    import matplotlib.pyplot as plt
    plt.close(fig)

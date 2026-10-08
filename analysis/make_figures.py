"""All paper figures -> results/figures/*.png + *.pdf, each with a CSV twin in results/figures/data/.

Usage: python analysis/make_figures.py   (run analysis/compute_extras.py first)
Style: dataviz skill - validated palette (slots 1-3 blue/orange/aqua, all-pairs on white),
marker shapes as secondary encoding, 95% CI whiskers/bands, one axis per panel,
hairline solid y-grid, text in ink (never series color), legend for >= 2 series.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (A_SHARES, AXIS, BLUE_RAMP, CACHE, CIGAR, COL1, COL2, DEEMPH, INK, INK2, MARKERS, MUTED,  # noqa: E402
                    R_SHARES, ROOT, ROUND, S1, S2, S3, SHORT, difference, dot, fid_lookup, fidelity_rows,
                    pilot_metrics, save, split_indices, split_labels, style, summary, confusion, per_class_metrics,
                    ROC_CONDS, auc_summary, mean_roc)

style()
CMAP = LinearSegmentedColormap.from_list("blue", ["#ffffff"] + BLUE_RAMP)


def ci_err(s):
    return [[s["point"] - s["lo"]], [s["hi"] - s["point"]]]


def panel_label(ax, text):
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=INK, ha="right", va="bottom")


# ------------------------------------------------------------------------------------ 01 dataset
def fig01_dataset():
    idx, y = split_indices(), None
    rows, counts = [], {}
    for s in ("train", "val", "test"):
        counts[s] = np.bincount(split_labels(s), minlength=10)
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    x = np.arange(10)
    w = 0.26
    names = {"train": "Working pool (train)", "val": "Validation", "test": "Held-out test"}
    for i, (s, col) in enumerate(zip(("train", "val", "test"), (S1, S2, S3))):
        ax.bar(x + (i - 1) * w, counts[s], width=w - 0.03, color=col, label=f"{names[s]} ({counts[s].sum():,})", zorder=2)
    for c in range(10):
        ax.text(c - w, counts["train"][c] + 25, f"{counts['train'][c]:,}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        rows.append([c, SHORT[c], counts["train"][c], counts["val"][c], counts["test"][c]])
    ax.set_xticks(x, [s.replace(" ", "\n", 1) for s in SHORT], fontsize=6.2)
    ax.set_ylabel("Images")
    ax.set_title("Galaxy10 DECaLS after cleaning: stratified 70/15/15 split (17,546 images)", loc="left")
    ax.legend(loc="upper right", ncols=3)
    ax.set_ylim(0, counts["train"].max() * 1.18)
    save(fig, "fig01_dataset_splits", rows, ["class", "name", "train", "val", "test"])


# ------------------------------------------------------------------------------------ 02 samples
def fig02_samples():
    z = np.load(CACHE / "sample_grid.npz")
    real, syn = z["real"], z["syn"]
    fig, axes = plt.subplots(10, 9, figsize=(COL2, 8.4), gridspec_kw={"width_ratios": [1] * 4 + [0.12] + [1] * 4,
                                                                         "wspace": 0.04, "hspace": 0.06})
    for c in range(10):
        for j in range(4):
            axes[c, j].imshow(real[c, j]); axes[c, 5 + j].imshow(syn[c, j])
        for j in range(9):
            axes[c, j].set_xticks([]); axes[c, j].set_yticks([]); axes[c, j].grid(False)
            for sp in axes[c, j].spines.values():
                sp.set_visible(False)
        axes[c, 4].axis("off")
        axes[c, 0].set_ylabel(SHORT[c], rotation=0, ha="right", va="center", fontsize=7, color=INK2, labelpad=4)
    axes[0, 1].set_title("Real (working pool)", loc="left", fontsize=8, x=-0.5)
    axes[0, 6].set_title("Synthetic (final generator)", loc="left", fontsize=8, x=-0.5)
    fig.suptitle("Real vs. synthetic galaxies per class (synthetic: guidance 3, 30 steps; not cherry-picked)",
                 x=0.5, y=0.935, fontsize=8.5, color=INK, fontweight="semibold")
    save(fig, "fig02_real_vs_synthetic_samples")


# ------------------------------------------------------------------------------------ 03 diffusion training
def fig03_diffusion_training():
    log = [json.loads(l) for l in open(ROOT / "runs" / "lora_r8" / "train_log.jsonl") if l.strip()]
    runs = ["lora_r8", "lora_r8_seed43", "lora_r4", "lora_r16", "lora_r8_lr5e-5", "lora_r8_lr2e-4", "lora_r8_cosine"]
    cks = [4000, 6000, 8000, 10000]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.5), gridspec_kw={"wspace": 0.28})
    steps = [r["step"] for r in log]; loss = [r["loss"] for r in log]
    a.plot(steps, loss, color=DEEMPH, lw=0.8)
    win = 20
    smooth = np.convolve(loss, np.ones(win) / win, mode="valid")
    a.plot(steps[win - 1:], smooth, color=S1, lw=1.5)
    a.text(steps[-1], 0.228, "blue: 20-point moving mean", color=INK2, fontsize=7, ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    a.set_xlabel("Training step"); a.set_ylabel("Diffusion loss (MSE)")
    a.set_title("Baseline LoRA training loss: flat", loc="left"); panel_label(a, "a")
    rows = []
    M = []
    for r in runs:
        vals = [1e3 * fid_lookup(r, c, 2.0, 30)["macro_kid"] for c in cks]
        M.append(vals)
        b.plot(cks, vals, color=DEEMPH, lw=0.8, zorder=1)
        rows.append([r] + [round(v, 2) for v in vals])
    M = np.array(M)
    b.plot(cks, M.mean(0), color=S1, lw=1.5, zorder=2)
    dot(b, cks, M.mean(0), S1, label="mean of 7 runs")
    b.plot([], [], color=DEEMPH, lw=0.8, label="individual runs (rank / LR / seed / schedule)")
    dot(b, [10000], [1e3 * fid_lookup("lora_r8", 10000, 2.0, 30)["macro_kid"]], S2, marker="s", size=6.5,
        label="chosen generator (seed 42, 10,000 steps)", zorder=4)
    b.set_xticks(cks); b.set_xlabel("Checkpoint (training step)"); b.set_ylabel("Macro KID x10$^3$ vs validation (lower = better)")
    b.set_title("Generator quality by checkpoint: large run-to-run swings", loc="left"); panel_label(b, "b")
    b.set_ylim(50, 97); b.legend(loc="upper left", fontsize=6.8)
    rows.append(["mean"] + [round(v, 2) for v in M.mean(0)])
    save(fig, "fig03_diffusion_training", rows, ["run"] + [f"kid_x1e3_ckpt{c}" for c in cks])


# ------------------------------------------------------------------------------------ 04/05 pilots
def _pilot(fig_name, xs, xlabel, kid, fidel, acc, chosen, title, xtick_labels=None, extra=None):
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2), gridspec_kw={"wspace": 0.42})
    specs = [(kid, "Macro KID x10$^3$ (lower = better)", "Realism"),
             (fidel, "Class fidelity (higher = better)", "Class fidelity"),
             (acc, "Real-validation accuracy (%)", "Usefulness: synthetic-only classifier")]
    for ax, (vals, ylab, t), lab in zip(axes, specs, "abc"):
        ax.axvline(chosen, color=AXIS, lw=0.8, zorder=0)
        ax.plot(xs, [v[0] for v in vals], color=S1, lw=1.5, zorder=2)
        if len(vals[0]) > 1 and vals[0][1] is not None:
            ax.errorbar(xs, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                        fmt="none", ecolor=S1, elinewidth=1, capsize=0, zorder=2)
        dot(ax, xs, [v[0] for v in vals], S1)
        ci = xs.index(chosen)
        dot(ax, [chosen], [vals[ci][0]], S2, marker="s", size=6.5, zorder=4)
        ax.set_title(t, loc="left"); ax.set_ylabel(ylab); ax.set_xlabel(xlabel); panel_label(ax, lab)
        ax.set_xticks(xs, xtick_labels or [f"{x:g}" for x in xs])
    lo, hi = axes[2].get_ylim(); axes[2].text(chosen, lo + 0.03 * (hi - lo), " chosen", color=INK2, fontsize=7, va="bottom")
    if extra:
        extra(axes)
    fig.suptitle(title, x=0.07, ha="left", y=1.06, fontsize=8.5, fontweight="semibold", color=INK)
    return fig


def fig04_guidance():
    gs = [1.5, 2.0, 3.0, 4.0, 6.0]
    fr = fidelity_rows(ROOT / "runs_cls" / "class_fidelity.jsonl")
    kid, fidel, acc, rows = [], [], [], []
    for g in gs:
        k = 1e3 * fid_lookup("lora_r8", 10000, g, 30)["macro_kid"]
        f = float(np.mean([r["macro_fidelity"] for r in fr if r["run"] == "lora_r8" and r["checkpoint"] == "checkpoint-10000"
                           and r["guidance_scale"] == g and r["sampling_steps"] == 30 and r["seed"] == 0]))
        a, _ = pilot_metrics(f"runs_cls/pilot_guidance/synthetic_cfg{g:g}_seed*")
        a = [100 * v for v in a]
        kid.append((k, None)); fidel.append((f, None))
        acc.append((np.mean(a), min(a), max(a)))
        rows.append([g, round(k, 2), round(f, 4), round(np.mean(a), 2), round(min(a), 2), round(max(a), 2)])
    fig = _pilot("fig04", gs, "Guidance scale", kid, fidel, acc, 3.0,
                 "Choosing guidance on validation: realism vs. class fidelity vs. usefulness (whiskers = range of 2 seeds)")
    save(fig, "fig04_guidance_pilot", rows, ["guidance", "kid_x1e3", "class_fidelity", "synthetic_only_val_acc_mean",
                                             "acc_seed_min", "acc_seed_max"])


def fig05_steps():
    steps = [30, 50, 75, 100]
    fr = fidelity_rows(ROOT / "runs_cls" / "class_fidelity_steps_pilot.jsonl")
    hours = {30: 540, 50: 854, 75: 1235, 100: 1594}
    kid, fidel, acc, rows = [], [], [], []
    for s in steps:
        k = 1e3 * fid_lookup("lora_r8", 10000, 3.0, s)["macro_kid"]
        f = float(np.mean([r["macro_fidelity"] for r in fr if r["sampling_steps"] == s and r["guidance_scale"] == 3.0]))
        pat = "runs_cls/pilot_guidance/synthetic_cfg3_seed*" if s == 30 else f"runs_cls/pilot_steps/synthetic_cfg3_steps{s}_seed*"
        a, _ = pilot_metrics(pat)
        a = [100 * v for v in a]
        kid.append((k, None)); fidel.append((f, None)); acc.append((np.mean(a), min(a), max(a)))
        rows.append([s, round(k, 2), round(f, 4), round(np.mean(a), 2), round(min(a), 2), round(max(a), 2),
                     round(hours[s] * 12330 / 1000 / 3600, 2)])
    labels = [f"{s}\n~{hours[s] * 12.33 / 3600:.1f} h" for s in steps]
    fig = _pilot("fig05", steps, "Steps (est. hours for 12,330 images)", kid, fidel, acc, 30,
                 "Choosing sampling steps: more steps look more real but do not teach better (guidance 3)",
                 xtick_labels=labels)
    save(fig, "fig05_steps_pilot", rows, ["steps", "kid_x1e3", "class_fidelity", "synthetic_only_val_acc_mean",
                                          "acc_seed_min", "acc_seed_max", "hours_for_12330_images"])


# ------------------------------------------------------------------------------------ 06 realism + memorization
def fig06_realism_memorization():
    rep = json.load(open(ROOT / "data" / "synthetic" / "main_cfg3_steps30" / "checks" / "report.json"))
    mem = np.load(CACHE / "memorization_sims.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.8), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1]})
    order = list(range(10))
    ys = np.arange(10)[::-1]
    rows = []
    for yy, c in zip(ys, order):
        r = rep["realism"][str(c)]
        a.plot([r["fid_ref"], r["fid_syn"]], [yy, yy], color=AXIS, lw=1, zorder=1)
        rows.append([c, SHORT[c], round(r["fid_syn"], 2), round(r["fid_ref"], 2), round(1e3 * r["kid_syn"], 2)])
    dot(a, [rep["realism"][str(c)]["fid_ref"] for c in order], ys, S1, "o", label="Real working-pool images (reference)")
    dot(a, [rep["realism"][str(c)]["fid_syn"] for c in order], ys, S2, "s", label="Synthetic (final set)")
    a.set_yticks(ys, [SHORT[c] for c in order], fontsize=7); a.grid(axis="x"); a.grid(axis="y", visible=False)
    a.set_xlabel("FID vs validation, same n (lower = more realistic)"); a.set_xlim(0, None)
    a.set_title("Realism per class", loc="left"); panel_label(a, "a")
    a.legend(loc="upper center", fontsize=6.6, bbox_to_anchor=(0.45, -0.17), ncols=2)
    for data, col, lab, mk in ((mem["val"], S1, "Unseen real (validation)", "o"), (mem["syn"], S2, "Synthetic (final set)", "s")):
        xs = np.sort(data); ecdf = np.arange(1, len(xs) + 1) / len(xs)
        b.plot(xs, ecdf, color=col, lw=1.5, label=lab)
        med = np.median(data)
        dot(b, [med], [0.5], col, mk)
    vmax = mem["val"].max()
    b.axvline(vmax, color=MUTED, lw=0.8)
    b.text(vmax, 0.06, f" max real\n {vmax:.3f}", color=INK2, fontsize=6.8, va="bottom")
    b.set_xlabel("Cosine similarity to nearest working-pool image"); b.set_ylabel("Cumulative fraction")
    b.set_title("Memorization check: none", loc="left"); panel_label(b, "b")
    b.legend(loc="upper left", fontsize=6.8)
    b.text(0.02, 0.62, f"0 of {len(mem['syn']):,} synthetic images\nexceed the closest unseen\nreal image",
           transform=b.transAxes, fontsize=6.8, color=INK2, va="top")
    rows.append(["memorization", "synthetic median", round(float(np.median(mem["syn"])), 4), "validation median",
                 round(float(np.median(mem["val"])), 4)])
    save(fig, "fig06_realism_memorization", rows, ["class", "name", "fid_synthetic", "fid_real_reference", "kid_synthetic_x1e3"])


# ------------------------------------------------------------------------------------ 07 class fidelity
def fig07_fidelity():
    d = json.load(open(CACHE / "final_set_fidelity.json"))
    fid, rec = d["fidelity"], d["real_val_recall"]
    order = list(np.argsort(fid))
    ys = np.arange(10)
    fig, ax = plt.subplots(figsize=(COL1 + 0.6, 2.9))
    for yy, c in zip(ys, order):
        ax.plot([fid[c], rec[c]], [yy, yy], color=AXIS, lw=1, zorder=1)
    dot(ax, [rec[c] for c in order], ys, S1, "o", label="Real validation images (classifier recall)")
    dot(ax, [fid[c] for c in order], ys, S2, "s", label="Synthetic images (class fidelity)")
    ax.set_yticks(ys, [SHORT[c] for c in order], fontsize=7); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.02); ax.set_xlabel("Fraction recognized as the intended class")
    ax.set_title("Does a real-trained classifier recognize the class?", loc="left")
    ax.legend(loc="upper left", fontsize=6.6)
    for c in (CIGAR,):
        yy = order.index(c)
        ax.text(fid[c] + 0.03, yy, f"{fid[c]:.2f}", va="center", fontsize=6.8, color=INK2)
    save(fig, "fig07_class_fidelity", [[c, SHORT[c], round(fid[c], 4), round(rec[c], 4)] for c in range(10)],
         ["class", "name", "synthetic_fidelity", "real_val_recall"])


# ------------------------------------------------------------------------------------ 08 replacement curve
def fig08_replacement():
    xs = [100 * float(s) for s in A_SHARES]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6), gridspec_kw={"wspace": 0.28})
    rows = []
    for ax, kind, ylab, lab in ((axes[0], "acc", "Accuracy (%)", "a"), (axes[1], "macro_f1", "Macro-F1", "b")):
        scale = 100 if kind == "acc" else 1
        for split, col, mk, name in (("test", S1, "o", "Held-out test"), ("val", S2, "s", "Validation")):
            ss = [summary(split, f"A_replace{s}", kind) for s in A_SHARES]
            pts = [scale * s["point"] for s in ss]
            ax.fill_between(xs, [scale * s["lo"] for s in ss], [scale * s["hi"] for s in ss], color=col, alpha=0.10, lw=0)
            ax.plot(xs, pts, color=col, lw=1.5)
            dot(ax, xs, pts, col, mk, label=name)
            for s_, x, s in zip(A_SHARES, xs, ss):
                rows.append([split, kind, x, round(scale * s["point"], 3), round(scale * s["lo"], 3), round(scale * s["hi"], 3), s["seeds"]])
        t0 = summary("test", "A_replace0", kind)["point"] * scale
        t1 = summary("test", "A_replace1", kind)["point"] * scale
        ax.annotate(f"{t0:.1f}" if kind == "acc" else f"{t0:.3f}", (0, t0), textcoords="offset points", xytext=(4, 6),
                    fontsize=7, color=INK2)
        ax.annotate(f"{t1:.1f}" if kind == "acc" else f"{t1:.3f}", (100, t1), textcoords="offset points", xytext=(-4, -12),
                    fontsize=7, color=INK2, ha="right")
        ax.set_xticks(xs, [f"{x:g}" for x in xs]); ax.set_xlabel("Synthetic share of training data (%), total fixed at 12,330")
        ax.set_ylabel(ylab); panel_label(ax, lab)
    axes[0].set_title("Replacing real with synthetic: accuracy", loc="left")
    axes[1].set_title("Replacing real with synthetic: macro-F1", loc="left")
    axes[0].legend(loc="lower left")
    fig.text(0.07, -0.08, "Bands: 95% bootstrap CI over evaluation images. Test: 8 seeds at 0%, 3 elsewhere. "
             "Synthetic-only reaches 77% of real-only test accuracy.", fontsize=6.8, color=INK2)
    save(fig, "fig08_replacement_curve", rows, ["split", "metric", "synthetic_share_pct", "value", "ci_lo", "ci_hi", "seeds"])


# ------------------------------------------------------------------------------------ 09 value of synthetic vs scarcity
def fig09_value_of_synthetic():
    shares = sorted(R_SHARES, key=lambda s: -float(s))  # ascending real fraction: 10% ... 90%
    real_frac = [100 * (1 - float(s)) for s in shares] + [100]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.6), gridspec_kw={"wspace": 0.3})
    rows = []
    alone = [summary("test", f"R_realpart{s}", "acc") for s in shares] + [summary("test", "A_replace0", "acc")]
    mix = [summary("test", f"A_replace{s}", "acc") for s in shares] + [summary("test", "A_replace0", "acc")]
    for ss, col, mk, name in ((alone, S1, "o", "Real images only"), (mix, S2, "s", "Same real images + synthetic (to 12,330)")):
        pts = [100 * s["point"] for s in ss]
        a.fill_between(real_frac, [100 * s["lo"] for s in ss], [100 * s["hi"] for s in ss], color=col, alpha=0.10, lw=0)
        a.plot(real_frac, pts, color=col, lw=1.5)
        dot(a, real_frac, pts, col, mk, label=name)
    a.set_xscale("log"); a.set_xticks([10, 25, 50, 75, 100], ["10", "25", "50", "75", "100"]); a.minorticks_off()
    a.set_xlabel("Real data available (% of working pool, log scale)"); a.set_ylabel("Test accuracy (%)")
    a.set_title("Accuracy vs. amount of real data", loc="left"); panel_label(a, "a"); a.legend(loc="upper left")
    diffs = [difference("test", f"A_replace{s}", f"R_realpart{s}", "acc") for s in shares]
    rf = real_frac[:-1]
    b.axhline(0, color=MUTED, lw=0.8, zorder=0)
    for x, d in zip(rf, diffs):
        b.errorbar([x], [100 * d["point"]], yerr=[[100 * (d["point"] - d["lo"])], [100 * (d["hi"] - d["point"])]],
                   fmt="none", ecolor=S2, elinewidth=1.2, capsize=0)
        rows.append([x, round(100 * alone[rf.index(x)]["point"], 3), round(100 * mix[rf.index(x)]["point"], 3),
                     round(100 * d["point"], 3), round(100 * d["lo"], 3), round(100 * d["hi"], 3)])
    dot(b, rf, [100 * d["point"] for d in diffs], S2, "s")
    b.set_xscale("log"); b.set_xticks([10, 25, 50, 75, 90], ["10", "25", "50", "75", "90"]); b.minorticks_off()
    b.set_xlabel("Real data available (% of working pool, log scale)")
    b.set_ylabel("Accuracy gain from synthetic (points)")
    b.set_title("What the synthetic images add (paired, 95% CI)", loc="left"); panel_label(b, "b")
    d10 = diffs[0]  # the 10%-real point
    b.annotate(f"+{100 * d10['point']:.1f} pts\n[{100 * d10['lo']:+.1f}, {100 * d10['hi']:+.1f}]", (10, 100 * d10["point"]),
               textcoords="offset points", xytext=(10, -4), fontsize=7, color=INK2, va="center")
    save(fig, "fig09_value_of_synthetic_vs_scarcity", rows,
         ["real_pct", "real_only_acc", "real_plus_synthetic_acc", "gain_pts", "gain_ci_lo", "gain_ci_hi"])


# ------------------------------------------------------------------------------------ 10 augmentation
def fig10_augmentation():
    conds = [("A_replace0", "Real only\n(12,330)"), ("B_add0.25", "+25%\nsynthetic"), ("B_add0.5", "+50%\nsynthetic"),
             ("B_add1", "+100%\nsynthetic")]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    base = summary("test", "A_replace0", "acc")
    ax.axhspan(100 * base["lo"], 100 * base["hi"], color=S1, alpha=0.08, lw=0)
    ax.axhline(100 * base["point"], color=S1, lw=0.8)
    rows = []
    for i, (c, lab) in enumerate(conds):
        s = summary("test", c, "acc")
        ax.errorbar([i], [100 * s["point"]], yerr=[[100 * (s["point"] - s["lo"])], [100 * (s["hi"] - s["point"])]],
                    fmt="none", ecolor=S1, elinewidth=1.2)
        dot(ax, [i], [100 * s["point"]], S1)
        d = difference("test", c, "A_replace0", "acc") if c != "A_replace0" else {"point": 0, "lo": 0, "hi": 0}
        rows.append([c, round(100 * s["point"], 3), round(100 * s["lo"], 3), round(100 * s["hi"], 3),
                     round(100 * d["point"], 3), round(100 * d["lo"], 3), round(100 * d["hi"], 3)])
    ax.set_xticks(range(4), [l for _, l in conds]); ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(100 * base["point"] - 2.2, 100 * base["point"] + 2.2)
    ax.set_title("Adding synthetic on top of all real data: no effect", loc="left")
    ax.text(-0.45, 100 * base["point"] - 2.1, "band: real-only mean and 95% CI", ha="left", va="bottom", fontsize=6.6, color=INK2)
    save(fig, "fig10_augmentation", rows, ["condition", "acc", "ci_lo", "ci_hi", "diff_vs_real_pts", "diff_lo", "diff_hi"])


# ------------------------------------------------------------------------------------ 11 focus classes
def fig11_focus():
    ks = [0, 0.5, 1, 2, 4]
    panels = [(ROUND, "C_round_x{k}", "A_replace0", "Round Smooth (1,844 real)", S1, "o"),
              (CIGAR, "C_cigar_x{k}", "A_replace0", "Cigar-Shaped Smooth (233 real)", S2, "s"),
              (ROUND, "S_roundscarce_x{k}", "S_roundscarce_x0", "Round Smooth cut to 233 real (control)", S3, "^")]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3), gridspec_kw={"wspace": 0.35})
    rows = []
    for ax, (cls, pat, base, title, col, mk), lab in zip(axes, panels, "abc"):
        names = [base if k == 0 else pat.format(k=f"{k:g}") for k in ks]
        ss = [summary("test", n, "cls", cls) for n in names]
        b0 = ss[0]
        ax.axhspan(b0["lo"], b0["hi"], color=col, alpha=0.08, lw=0)
        ax.axhline(b0["point"], color=col, lw=0.8)
        xpos = list(range(len(ks)))
        ax.errorbar(xpos, [s["point"] for s in ss], yerr=[[s["point"] - s["lo"] for s in ss], [s["hi"] - s["point"] for s in ss]],
                    fmt="none", ecolor=col, elinewidth=1.1)
        dot(ax, xpos, [s["point"] for s in ss], col, mk)
        ax.set_xticks(xpos, [f"{k:g}x" for k in ks]); ax.set_xlabel("Synthetic added (x the class's real count)")
        ax.set_title(title, loc="left", fontsize=7.8); panel_label(ax, lab)
        for k, n, s in zip(ks, names, ss):
            d = difference("test", n, base, "cls", cls) if k != 0 else {"point": 0, "lo": 0, "hi": 0}
            rows.append([title, k, round(s["point"], 4), round(s["lo"], 4), round(s["hi"], 4), s["seeds"],
                         round(d["point"], 4), round(d["lo"], 4), round(d["hi"], 4)])
    axes[0].set_ylabel("Class F1 on held-out test")
    fig.text(0.07, -0.1, "Whiskers: 95% bootstrap CI; band: the no-synthetic baseline's CI. 8 seeds for Cigar-Shaped and the "
             "control, 3 for Round Smooth. No consistent effect in any panel.", fontsize=6.8, color=INK2)
    save(fig, "fig11_focus_classes", rows, ["panel", "k", "f1", "ci_lo", "ci_hi", "seeds", "diff_vs_base", "diff_lo", "diff_hi"])


# ------------------------------------------------------------------------------------ 12 per-class F1 heatmap
def fig12_heatmap():
    M = np.array([per_class_metrics("test", f"A_replace{s}")["f1"] for s in A_SHARES]).T
    fig, ax = plt.subplots(figsize=(COL1 + 0.9, 3.1))
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(10):
        for j in range(len(A_SHARES)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color="white" if v > 0.62 else INK)
    ax.set_xticks(range(len(A_SHARES)), [f"{100 * float(s):g}%" for s in A_SHARES])
    ax.set_yticks(range(10), SHORT, fontsize=7)
    ax.set_xlabel("Synthetic share of training data"); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
    cb.set_label("Class F1 (test)", fontsize=7, color=INK2)
    ax.set_title("Per-class test F1 as synthetic replaces real", loc="left")
    save(fig, "fig12_per_class_f1_heatmap", [[SHORT[i]] + [round(v, 4) for v in M[i]] for i in range(10)],
         ["class"] + [f"synthetic_{s}" for s in A_SHARES])


# ------------------------------------------------------------------------------------ 13 confusion matrices
def fig13_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.4), gridspec_kw={"wspace": 0.08})
    rows = []
    for ax, cond, title in ((axes[0], "A_replace0", "Trained on real only (8 seeds pooled)"),
                            (axes[1], "A_replace1", "Trained on synthetic only (3 seeds pooled)")):
        cm = confusion("test", cond)
        im = ax.imshow(cm, cmap=CMAP, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                            color="white" if cm[i, j] > 0.62 else INK)
                rows.append([cond, SHORT[i], SHORT[j], round(cm[i, j], 4)])
        ax.set_xticks(range(10), [str(c) for c in range(10)]); ax.set_xlabel("Predicted class")
        ax.set_title(title, loc="left"); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(10), [f"{c}  {SHORT[c]}" for c in range(10)], fontsize=6.8); axes[0].set_ylabel("True class")
    axes[1].set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
    cb.set_label("Fraction of true class (row-normalized)", fontsize=7, color=INK2)
    save(fig, "fig13_confusion_matrices", rows, ["condition", "true", "predicted", "fraction"])


# ------------------------------------------------------------------------------------ 14 classifier learning curves
def fig14_learning():
    """Training vs validation loss and accuracy per epoch. The original logs had no validation loss or training
    accuracy, so they come from the reproduced runs (classifier/reproduce_runs.py -> runs_cls/repro/), each verified
    bit-identical to its original run (every logged number and every best.pt tensor)."""
    ver = json.loads((ROOT / "runs_cls" / "repro" / "verification.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.2), sharex=True, sharey="row")
    lines = [("val", "-", 1.6), ("train_eval", "--", 1.2), ("train", ":", 1.3)]
    rows = []
    for col_i, ((cond, lab), color) in enumerate(zip(ROC_CONDS, (S1, S3, S2))):
        runs = sorted(r for r in ver if re.sub(r"_seed\d+$", "", r) == cond)
        assert runs and all(ver[r]["log_identical"] and ver[r]["weights_identical"] for r in runs), cond
        logs = [[json.loads(l) for l in open(ROOT / "runs_cls" / "repro" / r / "train_log.jsonl") if l.strip()]
                for r in runs]
        ep = np.arange(1, len(logs[0]) + 1)

        def get(key, scale=1.0):
            return scale * np.array([[e[key] for e in lg] for lg in logs])
        data = {"val_loss": get("val_loss"), "train_eval_loss": get("train_eval_loss"), "train_loss": get("train_loss"),
                "val_accuracy": get("val_accuracy", 100), "train_eval_accuracy": get("train_eval_accuracy", 100),
                "train_accuracy": get("train_accuracy", 100)}
        selected = np.median([int(np.argmax(a)) + 1 for a in data["val_accuracy"]])  # first best epoch, as in training
        for row_i, metric in enumerate(("loss", "accuracy")):
            ax = axes[row_i, col_i]
            v = data[f"val_{metric}"]
            ax.fill_between(ep, v.min(0), v.max(0), color=color, alpha=0.16, lw=0, zorder=1)
            for key, ls, lw in lines:
                ax.plot(ep, data[f"{key}_{metric}"].mean(0), color=color, ls=ls, lw=lw, zorder=3)
            ax.axvline(selected, color=AXIS, lw=0.9, ls=(0, (1, 2)), zorder=0)
        axes[0, col_i].set_title(f"{lab} ({len(runs)} seeds)", loc="left")
        axes[1, col_i].set_xlabel("Epoch")
        axes[1, col_i].text(selected - 0.6, 0.03, f"selected epoch\n(median {selected:g})", ha="right", va="bottom",
                            fontsize=6.4, color=MUTED, transform=axes[1, col_i].get_xaxis_transform())
        rows += [[cond, int(e)] + [round(float(data[k].mean(0)[i]), 5) for k in data]
                 + [round(float(data["val_loss"].min(0)[i]), 5), round(float(data["val_loss"].max(0)[i]), 5),
                    round(float(data["val_accuracy"].min(0)[i]), 4), round(float(data["val_accuracy"].max(0)[i]), 4),
                    selected] for i, e in enumerate(ep)]
    axes[0, 0].set_ylabel("Cross-entropy loss")
    axes[1, 0].set_ylabel("Accuracy (%)")
    lowest = min(float(r[10]) for r in rows)  # lowest validation accuracy of any seed
    axes[1, 0].set_ylim(10 * np.floor(lowest / 10), 101)
    axes[1, 0].set_xticks([1, 10, 20, 30])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=INK2, ls="-", lw=1.6), Line2D([], [], color=INK2, ls="--", lw=1.2),
               Line2D([], [], color=INK2, ls=":", lw=1.3), Patch(color=INK2, alpha=0.16, lw=0)]
    labels = ["validation", "training set, no augmentation (eval mode)", "training, during the epoch (augmented)",
              "validation, range over seeds"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.955), ncol=4, fontsize=6.8, handlelength=2.6)
    fig.suptitle("Classifier training vs. validation loss and accuracy per epoch (seed means)", x=0.06, ha="left",
                 y=1.0, fontsize=8.5, color=INK, fontweight="semibold")
    fig.subplots_adjust(top=0.84, wspace=0.08, hspace=0.12)
    save(fig, "fig14_classifier_learning_curves", rows,
         ["condition", "epoch"] + [f"{k}_mean" for k in ("val_loss", "train_eval_loss", "train_loss", "val_acc",
                                                          "train_eval_acc", "train_acc")]
         + ["val_loss_min", "val_loss_max", "val_acc_min", "val_acc_max", "median_selected_epoch"])


# ------------------------------------------------------------------------------------ 15 LoRA sweep summary
def fig15_sweep():
    runs = [("lora_r8", "rank 8, LR 1e-4 (baseline)"), ("lora_r8_seed43", "baseline, 2nd seed"), ("lora_r4", "rank 4"),
            ("lora_r16", "rank 16"), ("lora_r8_lr5e-5", "LR 5e-5"), ("lora_r8_lr2e-4", "LR 2e-4"),
            ("lora_r8_cosine", "cosine LR schedule")]
    fig, ax = plt.subplots(figsize=(COL1 + 0.6, 2.6))
    ys = np.arange(len(runs))[::-1]
    rows = []
    for yy, (r, lab) in zip(ys, runs):
        vals = [1e3 * fid_lookup(r, c, 2.0, 30)["macro_kid"] for c in (4000, 6000, 8000, 10000)]
        ax.plot([min(vals), max(vals)], [yy, yy], color=AXIS, lw=1, zorder=1)
        rows.append([r, lab] + [round(v, 2) for v in vals])
    dot(ax, [1e3 * fid_lookup(r, 4000, 2.0, 30)["macro_kid"] for r, _ in runs], ys, S1, "o", label="checkpoint 4,000")
    dot(ax, [1e3 * fid_lookup(r, 10000, 2.0, 30)["macro_kid"] for r, _ in runs], ys, S2, "s", label="checkpoint 10,000")
    ax.set_yticks(ys, [l for _, l in runs], fontsize=7); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Macro KID x10$^3$ vs validation (lower = better); line = range over 4 checkpoints")
    ax.set_title("LoRA sweep: settings differ less than seeds do", loc="left")
    ax.legend(loc="lower right", fontsize=6.8)
    save(fig, "fig15_lora_sweep", rows, ["run", "label", "kid_4000", "kid_6000", "kid_8000", "kid_10000"])


# ------------------------------------------------------------------------------------ 18 ROC curves (validation)
def fig18_roc():
    """One-vs-rest ROC curves per class on the VALIDATION set (seed-mean curves, vertical averaging). The one-time
    test evaluation saved only predicted classes, not scores, so the test set cannot give ROC curves without
    re-running it, which the protocol forbids."""
    from matplotlib.lines import Line2D
    styles = list(zip(ROC_CONDS, (S1, S3, S2), ("o", "^", "s"), ("-", "--", "-.")))
    fig, axes = plt.subplots(2, 5, figsize=(COL2, 3.9), sharex=True, sharey=True)
    rows = []
    for c, ax in enumerate(axes.flat):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8, ls=":", zorder=1)
        ax.text(0.57, 0.39, "AUC", fontsize=6.4, color=MUTED, va="bottom")
        for k, ((cond, _), color, mk, ls) in enumerate(styles):
            fpr, tpr = mean_roc(cond, c)
            ax.plot(fpr, tpr, color=color, ls=ls, lw=1.3, zorder=5 - k)
            y0 = 0.32 - 0.105 * k
            dot(ax, [0.61], [y0], color, mk, size=4.6, zorder=6)
            ax.text(0.67, y0, f"{auc_summary(cond, c)['point']:.3f}", fontsize=6.6, color=INK2, va="center")
            rows += [[SHORT[c], cond, round(float(f), 5), round(float(t), 5)] for f, t in zip(fpr, tpr)]
        ax.set_title(SHORT[c], loc="left", fontsize=7.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.grid(axis="both")
    handles = [Line2D([], [], color=color, ls=ls, lw=1.3, marker=mk, markersize=4.6, markeredgecolor="white")
               for _, color, mk, ls in styles]
    labels = [f"{lab} ({auc_summary(cond, 10)['seeds']} seeds): macro AUC {auc_summary(cond, 10)['point']:.3f}"
              for (cond, lab), *_ in styles]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.05, 0.945), ncol=3, fontsize=7, handlelength=2.6)
    fig.suptitle("ROC curves per class, one-vs-rest, on the validation set (seed-mean curves)", x=0.05, ha="left",
                 y=0.995, fontsize=8.5, color=INK, fontweight="semibold")
    fig.supxlabel("False positive rate", fontsize=8, color=INK2, y=0.02)
    fig.supylabel("True positive rate", fontsize=8, color=INK2, x=0.02)
    fig.subplots_adjust(top=0.84, wspace=0.12, hspace=0.28, left=0.07, bottom=0.1)
    save(fig, "fig18_roc_curves_val", rows, ["class", "condition", "fpr", "tpr_seed_mean"])


FIGS = [fig01_dataset, fig02_samples, fig03_diffusion_training, fig04_guidance, fig05_steps, fig06_realism_memorization,
        fig07_fidelity, fig08_replacement, fig09_value_of_synthetic, fig10_augmentation, fig11_focus, fig12_heatmap,
        fig13_confusion, fig14_learning, fig15_sweep, fig18_roc]

if __name__ == "__main__":
    only = sys.argv[1:]
    for f in FIGS:
        if not only or any(o in f.__name__ for o in only):
            f()
            print("done", f.__name__, flush=True)

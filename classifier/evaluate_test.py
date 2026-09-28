"""Final evaluation of the saved experiment classifiers - the ONLY script that reads the TEST set.

Usage:
    python classifier/evaluate_test.py --split val                      # dry run: must reproduce validation metrics
    python classifier/evaluate_test.py --split test --confirm-test-set  # the one-time test evaluation

Pre-registered (fixed 2026-09-27, before any test data was read):
- Models: every runs_cls/exp/<condition>_seed<k>/best.pt (each = its epoch with the best
  VALIDATION accuracy; nothing is selected or tuned on the test set).
- Per model: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
  -> <run>/<split>_metrics.json, predictions -> <run>/<split>_predictions.npy.
- Per condition: mean +/- SD over seeds.
- 95% confidence intervals: bootstrap over the evaluation images (2,000 resamples; the
  same resample applied to every model, so comparisons are paired), for each condition's
  mean accuracy / macro-F1 and for pre-specified differences:
    A: each synthetic share vs real-only (0%); synthetic-only vs real-only
    R: each mix vs its paired real-only part (same real images, synthetic dropped)
    B: each augmentation vs real-only
    C/S: focus-class F1, each ratio vs its no-synthetic baseline
- Output: runs_cls/exp/summary_<split>.txt and results_<split>.json.
- Interpretation rule (fixed in advance): ~20 differences are tested, so one or two CIs
  excluding 0 are expected by chance alone (the validation dry run showed exactly that in
  the scarcity control, in opposite directions). A single borderline CI is not a finding;
  a claim needs a consistent pattern (e.g. a monotone trend across ratios) or a large effect.
- Seeds: conditions use all seeds present (3, or 8 for the rare-class conditions and the
  real-only baseline).
"""

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_splits import CLASS_NAMES  # noqa: E402
from train_classifier import GALAXY10_H5, build_model, predict, read_images_cached  # noqa: E402

EXP = PROJECT_ROOT / "runs_cls" / "exp"
SPLITS = PROJECT_ROOT / "data" / "splits" / "galaxy10_splits.npz"
ROUND, CIGAR = 2, 4
N_BOOT = 2000


def load_eval_split(key):
    splits = np.load(SPLITS)
    idx = splits[key]
    with h5py.File(GALAXY10_H5, "r") as f:
        labels = f["ans"][:].astype(np.int64)
        if hashlib.sha1(labels.tobytes()).hexdigest() != str(splits["labels_sha1"]):
            raise SystemExit("dataset file does not match the splits")
        images = read_images_cached(f["images"], idx)
    return images, labels[idx]


def metrics(y, pred):
    p, r, f1, n = precision_recall_fscore_support(y, pred, labels=range(len(CLASS_NAMES)), zero_division=0)
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(f1.mean()),
            "per_class": {str(c): {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f1[c]), "n": int(n[c])}
                          for c in range(len(CLASS_NAMES))},
            "confusion_matrix": confusion_matrix(y, pred, labels=range(len(CLASS_NAMES))).tolist()}


def f1_per_class(y, pred):
    """Vectorised per-class F1 for one (y, pred) pair."""
    k = len(CLASS_NAMES)
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--confirm-test-set", action="store_true")
    ap.add_argument("--runs", default=str(EXP / "*_seed*"))
    ap.add_argument("--out-dir", type=Path, default=EXP, help="where summary/results files go")
    args = ap.parse_args()
    if args.split == "test" and not args.confirm_test_set:
        raise SystemExit("refusing to read the TEST set without --confirm-test-set")

    x, y = load_eval_split(args.split)
    device = torch.device("cuda")
    run_dirs = sorted(d for d in glob.glob(args.runs) if (Path(d) / "best.pt").is_file())
    preds, mismatches = {}, []
    for d in run_dirs:
        d = Path(d)
        ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
        model = build_model().to(device).to(memory_format=torch.channels_last)
        model.load_state_dict(ck["state_dict"])
        pred = predict(model, x, device).argmax(1).numpy()
        m = metrics(y, pred)
        (d / f"{args.split}_metrics.json").write_text(json.dumps(m, indent=2))
        np.save(d / f"{args.split}_predictions.npy", pred)
        preds[d.name] = pred
        if args.split == "val":  # dry run: must reproduce the validation metrics saved at training time
            saved = ck["val_metrics"]
            if abs(saved["accuracy"] - m["accuracy"]) > 1e-9 or abs(saved["macro_f1"] - m["macro_f1"]) > 1e-9:
                mismatches.append((d.name, saved["accuracy"], m["accuracy"]))
        del model

    # ---------------------------------------------------------------- aggregation
    cond_runs = defaultdict(list)
    for name in preds:
        cond_runs[re.sub(r"_seed\d+$", "", name)].append(name)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(y), size=(N_BOOT, len(y)))

    memo = {}

    def stat_boot(names, kind, cls=None):
        """(point estimate, bootstrap samples) of the seed-mean statistic for a condition (memoised)."""
        key = (tuple(names), kind, cls)
        if key not in memo:
            memo[key] = _stat_boot(names, kind, cls)
        return memo[key]

    def _stat_boot(names, kind, cls=None):
        def one(idx):
            vals = []
            for nm in names:
                yy, pp = y[idx], preds[nm][idx]
                if kind == "acc":
                    vals.append((yy == pp).mean())
                else:
                    f = f1_per_class(yy, pp)
                    vals.append(f.mean() if kind == "macro_f1" else f[cls])
            return float(np.mean(vals))
        return one(np.arange(len(y))), np.array([one(b) for b in boot])

    lines, results = [], {"split": args.split, "n_images": int(len(y)), "conditions": {}, "differences": []}
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    say(f"{args.split.upper()} RESULTS - {len(preds)} models, {len(y)} images. Mean over seeds; 95% bootstrap CI "
        f"({N_BOOT} resamples of the {args.split} images).")
    if args.split == "val":
        say(f"DRY RUN CHECK - models whose recomputed validation metrics differ from training time: "
            f"{mismatches if mismatches else 'none (all reproduce exactly)'}")

    def cond_line(cond, cls=None):
        names = cond_runs.get(cond)
        if not names:
            return f"  {cond:<24} (missing)"
        acc, acc_b = stat_boot(names, "acc")
        f1, f1_b = stat_boot(names, "macro_f1")
        sd = np.std([(y == preds[n]).mean() for n in names], ddof=1) if len(names) > 1 else 0.0
        entry = {"seeds": len(names), "accuracy": acc, "accuracy_ci": list(np.percentile(acc_b, [2.5, 97.5])),
                 "accuracy_sd_over_seeds": float(sd), "macro_f1": f1, "macro_f1_ci": list(np.percentile(f1_b, [2.5, 97.5]))}
        s = (f"  {cond:<24} acc {100 * acc:5.2f}% [{100 * entry['accuracy_ci'][0]:5.2f}, {100 * entry['accuracy_ci'][1]:5.2f}]"
             f" (seed SD {100 * sd:.2f})  macro-F1 {f1:.4f} [{entry['macro_f1_ci'][0]:.4f}, {entry['macro_f1_ci'][1]:.4f}]"
             f"  n={len(names)}")
        if cls is not None:
            cf, cf_b = stat_boot(names, "cls", cls)
            entry[f"f1_class{cls}"] = cf
            entry[f"f1_class{cls}_ci"] = list(np.percentile(cf_b, [2.5, 97.5]))
            s += f"  {CLASS_NAMES[cls]} F1 {cf:.3f} [{entry[f'f1_class{cls}_ci'][0]:.3f}, {entry[f'f1_class{cls}_ci'][1]:.3f}]"
        results["conditions"][cond] = entry
        return s

    def diff_line(label, a, b, kind, cls=None):
        """Difference a - b of the seed-mean statistic, with a paired bootstrap CI."""
        if a not in cond_runs or b not in cond_runs:
            return f"  {label:<46} (missing)"
        pa, ba = stat_boot(cond_runs[a], kind, cls)
        pb, bb = stat_boot(cond_runs[b], kind, cls)
        d, db = pa - pb, ba - bb
        lo, hi = np.percentile(db, [2.5, 97.5])
        scale, unit = (100, " pts") if kind == "acc" else (1, "")
        sig = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
        results["differences"].append({"label": label, "a": a, "b": b, "metric": kind if cls is None else f"f1_class{cls}",
                                       "difference": float(d), "ci": [float(lo), float(hi)]})
        return f"  {label:<46} {scale * d:+7.3f}{unit} [{scale * lo:+.3f}, {scale * hi:+.3f}]  CI {sig}"

    say("\nA. Replacement curve (fixed size; synthetic share):")
    for f in ["0", "0.1", "0.25", "0.5", "0.75", "0.9", "1"]:
        say(cond_line(f"A_replace{f}"))
    say(" differences vs real-only (accuracy):")
    for f in ["0.1", "0.25", "0.5", "0.75", "0.9", "1"]:
        say(diff_line(f"{float(f):.0%} synthetic - real-only", f"A_replace{f}", "A_replace0", "acc"))
    say("\nR. Paired real-only controls (the mix's real part alone):")
    for f in ["0.1", "0.25", "0.5", "0.75", "0.9"]:
        say(cond_line(f"R_realpart{f}"))
    say(" differences, mix - its real part alone (what the synthetic images add; accuracy):")
    for f in ["0.1", "0.25", "0.5", "0.75", "0.9"]:
        say(diff_line(f"{float(f):.0%}-synthetic mix - its {1 - float(f):.0%} real", f"A_replace{f}", f"R_realpart{f}", "acc"))
    say("\nB. Augmentation (all real + extra synthetic):")
    for a in ["0.25", "0.5", "1"]:
        say(cond_line(f"B_add{a}"))
        say(diff_line(f"+{float(a):.0%} synthetic - real-only", f"B_add{a}", "A_replace0", "acc"))
    for cls, label, base in ((ROUND, "round", "A_replace0"), (CIGAR, "cigar", "A_replace0")):
        say(f"\nC. Focus class {CLASS_NAMES[cls]} (baseline = real-only):")
        say(cond_line(base, cls))
        for k in ["0.5", "1", "2", "4"]:
            say(cond_line(f"C_{label}_x{k}", cls))
            say(diff_line(f"{CLASS_NAMES[cls]} F1: x{k} - baseline", f"C_{label}_x{k}", base, "cls", cls))
    say(f"\nS. Scarcity control: Round Smooth cut to 233 real:")
    say(cond_line("S_roundscarce_x0", ROUND))
    for k in ["0.5", "1", "2", "4"]:
        say(cond_line(f"S_roundscarce_x{k}", ROUND))
        say(diff_line(f"Round Smooth F1: x{k} - x0 (scarce)", f"S_roundscarce_x{k}", "S_roundscarce_x0", "cls", ROUND))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"summary_{args.split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_dir / f"results_{args.split}.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out_dir / f'summary_{args.split}.txt'} and {args.out_dir / f'results_{args.split}.json'}")
    if mismatches:
        raise SystemExit(f"DRY RUN FAILED: {len(mismatches)} model(s) did not reproduce their validation metrics")


if __name__ == "__main__":
    main()

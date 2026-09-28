"""Main classifier experiments (design fixed 2026-09-26), VALIDATION only.

Usage:
    python classifier/run_experiments.py              # run / resume everything
    python classifier/run_experiments.py --summary    # validation tables only
    python classifier/run_experiments.py --smoke DIR  # tiny end-to-end test into DIR

Order (primary question first):
  A  replacement curve: synthetic share 0/10/25/50/75/90/100% at fixed size (main set)
  B  augmentation: all real + 25/50/100% synthetic (main set)
  G  extra focus-class generation (699 Cigar-Shaped, 5,532 Round Smooth; new seeds) + checks
  C  focus class (Round Smooth, Cigar-Shaped) + 0.5/1/2/4 x its real count synthetic
  S  scarcity control: Round Smooth cut to 233 real, + 0/0.5/1/2/4 x synthetic
Every condition: ResNet-18 recipe (LR 1e-3, 30 epochs, best epoch on validation), seeds 0/1/2.
Resumable (finished runs skipped); a failed step is logged and the driver continues.
The TEST set is never read - final test evaluation is a separate, later step.
Log: runs_cls/experiments.log; tables: runs_cls/exp/summary_validation.txt
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
from prepare_splits import CLASS_NAMES  # noqa: E402

PYTHON = sys.executable
MAIN = PROJECT_ROOT / "data" / "synthetic" / "main_cfg3_steps30"
EXTRA_CIGAR = PROJECT_ROOT / "data" / "synthetic" / "focus_extra_cigar"
EXTRA_ROUND = PROJECT_ROOT / "data" / "synthetic" / "focus_extra_round"
ROUND, CIGAR = 2, 4
SEEDS = [0, 1, 2]
A_FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
B_FRACS = [0.25, 0.5, 1.0]
C_RATIOS = [0.5, 1.0, 2.0, 4.0]
S_RATIOS = [0.0, 0.5, 1.0, 2.0, 4.0]
SCARCE_N = 233  # = Cigar-Shaped Smooth's real count
R_FRACS = [0.1, 0.25, 0.5, 0.75, 0.9]  # paired real-only controls (the A mix's real part alone)
EXTRA_SEEDS = [3, 4, 5, 6, 7]          # rare-class analysis + its baseline: 8 seeds in total


class Runner:
    def __init__(self, smoke_dir=None):
        self.smoke = smoke_dir is not None
        self.out = (smoke_dir.resolve() if self.smoke else PROJECT_ROOT / "runs_cls" / "exp")
        self.log_path = (self.out / "experiments.log") if self.smoke else PROJECT_ROOT / "runs_cls" / "experiments.log"
        self.extra = ["--epochs", "1", "--limit", "600"] if self.smoke else []
        self.seeds = [0] if self.smoke else SEEDS
        self.failures = []

    def log(self, msg):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def call(self, cmd, console):
        console.parent.mkdir(parents=True, exist_ok=True)
        with open(console, "a", encoding="utf-8") as out:
            rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            self.log(f"FAILED (exit {rc}): {' '.join(map(str, cmd[1:4]))} ... - see {console}; continuing")
            self.failures.append(str(console))
        return rc == 0

    def train(self, name, mix_args, synthetic_dirs, seeds=None):
        for seed in (seeds if seeds is not None else self.seeds):
            d = self.out / f"{name}_seed{seed}"
            if (d / "metrics.json").is_file():
                continue
            ok = self.call([PYTHON, "classifier/train_classifier.py", "--output-dir", str(d), "--learning-rate", "1e-3",
                            "--seed", str(seed), "--synthetic-dirs", *map(str, synthetic_dirs), *mix_args, *self.extra],
                           d / "console.log")
            if ok:
                m = json.loads((d / "metrics.json").read_text())
                self.log(f"{name} seed {seed}: val acc {m['val']['accuracy']:.4f}, macro-F1 {m['val']['macro_f1']:.4f}")

    def generate_extras(self):
        for d, cls, n, start in ((EXTRA_CIGAR, CIGAR, 699, 233), (EXTRA_ROUND, ROUND, 5532, 1844)):
            if (d / "checks" / "report.json").is_file():
                continue  # generated and checked already
            self.log(f"G. generating {n} extra images of class {cls} -> {d.name}")
            if self.call([PYTHON, "diffusion/generate_synthetic.py", "--output-dir", str(d), "--classes", str(cls),
                          "--per-class", str(n), "--start-index", str(start)], d / "generation.log"):
                self.call([PYTHON, "diffusion/check_synthetic.py", str(d)], d / "check.log")
                self.log(f"G. {d.name}: generation + checks done")

    def main(self):
        self.log(f"=== experiments start ({'SMOKE' if self.smoke else 'full'})")
        a_fracs = [0.0, 1.0] if self.smoke else A_FRACS
        for f in a_fracs:
            self.train(f"A_replace{f:g}", ["--replace-fraction", str(f)], [MAIN])
        self.log("A done")
        for a in ([0.5] if self.smoke else B_FRACS):
            self.train(f"B_add{a:g}", ["--add-fraction", str(a)], [MAIN])
        self.log("B done")
        if self.smoke:
            c_dirs = [MAIN]
        else:
            self.generate_extras()
            c_dirs = [MAIN, EXTRA_CIGAR, EXTRA_ROUND]
        for cls, label in ((ROUND, "round"), (CIGAR, "cigar")):
            for k in ([1.0] if self.smoke else C_RATIOS):
                self.train(f"C_{label}_x{k:g}", ["--focus-class", str(cls), "--focus-ratio", str(k)], c_dirs)
        self.log("C done")
        for k in ([1.0] if self.smoke else S_RATIOS):
            self.train(f"S_roundscarce_x{k:g}", ["--focus-class", str(ROUND), "--focus-ratio", str(k),
                                                 "--focus-real", str(SCARCE_N)], c_dirs)
        self.log("S done")
        # R (added 2026-09-27): each A mix's real part alone - identical real images, synthetic dropped
        for f in ([0.5] if self.smoke else R_FRACS):
            self.train(f"R_realpart{f:g}", ["--replace-fraction", str(f), "--drop-synthetic"], [MAIN])
        self.log("R done")
        # E (added 2026-09-27): extra seeds for the rare-class analysis and its baselines
        extra_seeds = [1] if self.smoke else EXTRA_SEEDS
        self.train("A_replace0", ["--replace-fraction", "0"], [MAIN], extra_seeds)
        for k in ([1.0] if self.smoke else C_RATIOS):
            self.train(f"C_cigar_x{k:g}", ["--focus-class", str(CIGAR), "--focus-ratio", str(k)], c_dirs, extra_seeds)
        for k in ([1.0] if self.smoke else S_RATIOS):
            self.train(f"S_roundscarce_x{k:g}", ["--focus-class", str(ROUND), "--focus-ratio", str(k),
                                                 "--focus-real", str(SCARCE_N)], c_dirs, extra_seeds)
        self.log("E done")
        self.log(f"=== training done; failures: {self.failures if self.failures else 'none'}")
        self.summary()
        # T: the ONE-TIME test evaluation (smoke: dry run on validation instead)
        if self.smoke:
            self.call([PYTHON, "classifier/evaluate_test.py", "--split", "val", "--runs", str(self.out / "*_seed*"),
                       "--out-dir", str(self.out)], self.out / "evaluation.log")
        elif (self.out / "results_test.json").is_file():
            self.log("T. test evaluation already done - NOT repeating it")
        elif self.failures:
            self.log("T. NOT running the test evaluation because training had failures - needs review")
        else:
            self.log("T. one-time TEST evaluation")
            ok = self.call([PYTHON, "classifier/evaluate_test.py", "--split", "test", "--confirm-test-set"],
                           self.out / "evaluation_test.log")
            self.log(f"T. test evaluation {'done - see runs_cls/exp/summary_test.txt' if ok else 'FAILED'}")
        self.log(f"=== experiments done; failures: {self.failures if self.failures else 'none'}")

    # ------------------------------------------------------------------ summary
    def metrics(self, name):
        ms = []
        for p in sorted(self.out.glob(f"{name}_seed*/metrics.json")):  # every seed present
            if p.parent.name.rsplit("_seed", 1)[0] == name:
                ms.append(json.loads(p.read_text())["val"])
        return ms

    def summary(self):
        lines = []
        say = lambda s="": (print(s), lines.append(s))  # noqa: E731

        def row(label, ms, cls=None):
            if not ms:
                return f"  {label:<26} (not run)"
            acc = [m["accuracy"] for m in ms]; f1 = [m["macro_f1"] for m in ms]
            s = (f"  {label:<26} acc {np.mean(acc):.4f} +/- {np.std(acc, ddof=1) if len(acc) > 1 else 0:.4f}   "
                 f"macro-F1 {np.mean(f1):.4f} +/- {np.std(f1, ddof=1) if len(f1) > 1 else 0:.4f}   (n={len(ms)})")
            if cls is not None:
                cf = [m["per_class"][str(cls)]["f1"] for m in ms]
                s += f"   {CLASS_NAMES[cls]} F1 {np.mean(cf):.3f} +/- {np.std(cf, ddof=1) if len(cf) > 1 else 0:.3f}"
            return s

        say("VALIDATION RESULTS (mean +/- SD over seeds). Test set not used.")
        say("\nA. Replacement curve (fixed size; synthetic share):")
        for f in A_FRACS:
            say(row(f"{f:.0%} synthetic", self.metrics(f"A_replace{f:g}")))
        real, syn = self.metrics("A_replace0"), self.metrics("A_replace1")
        if real and syn:
            ra, sa = np.mean([m["accuracy"] for m in real]), np.mean([m["accuracy"] for m in syn])
            rf, sf = np.mean([m["macro_f1"] for m in real]), np.mean([m["macro_f1"] for m in syn])
            say(f"  synthetic-only as % of real-only: accuracy {100 * sa / ra:.1f}%, macro-F1 {100 * sf / rf:.1f}%")
        say("\nB. Augmentation (all real + extra synthetic):")
        say(row("real only (= A 0%)", real))
        for a in B_FRACS:
            say(row(f"+{a:.0%} synthetic", self.metrics(f"B_add{a:g}")))
        for cls, label in ((ROUND, "round"), (CIGAR, "cigar")):
            say(f"\nC. Focus class {CLASS_NAMES[cls]} (+ k x its real count synthetic, that class only):")
            say(row("k = 0 (= A 0%)", real, cls))
            for k in C_RATIOS:
                say(row(f"k = {k:g}", self.metrics(f"C_{label}_x{k:g}"), cls))
        say(f"\nS. Scarcity control: Round Smooth cut to {SCARCE_N} real (+ k x {SCARCE_N} synthetic):")
        for k in S_RATIOS:
            say(row(f"k = {k:g}", self.metrics(f"S_roundscarce_x{k:g}"), ROUND))
        say("\nR. Paired real-only controls (the A mix's real part alone; compare with A):")
        for f in R_FRACS:
            mix, alone = self.metrics(f"A_replace{f:g}"), self.metrics(f"R_realpart{f:g}")
            say(row(f"{1 - f:.0%} real alone", alone))
            if mix and alone:
                d = np.mean([m["accuracy"] for m in mix]) - np.mean([m["accuracy"] for m in alone])
                say(f"      + {f:.0%} synthetic (mix A): {100 * d:+.2f} points accuracy")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "summary_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", action="store_true")
    p.add_argument("--smoke", type=Path, default=None, metavar="DIR")
    args = p.parse_args()
    r = Runner(args.smoke)
    r.summary() if args.summary else r.main()


if __name__ == "__main__":
    main()

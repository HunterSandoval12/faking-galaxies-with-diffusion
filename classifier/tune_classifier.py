"""Overnight driver (2026-09-26): real-only classifier tuning + class fidelity of the candidates.

Usage:
    python classifier/tune_classifier.py [--wait-for-log runs/candidate_evals.log]

1. (optional) wait until the candidate-evaluation queue has finished (GPU memory).
2. Learning-rate search on the VALIDATION set (study design: classifier LR/schedule tuned on
   validation only): ResNet-18 real-only at LR 1e-4, 3e-4, 1e-3 (seed 0).
3. Second seed (seed 1) at the best LR (by best-epoch validation accuracy) - the study design
   requires >= 2 seeds per configuration.
4. Class fidelity of every saved generated set with both seeds' classifiers.
Resumable (finished runs are skipped); a failed step is logged and the driver continues.
It makes NO generator choice (that was a separate review step). Never reads the test set.
Log: runs_cls/classifier_tuning.log; fidelity table: runs_cls/candidate_summary.txt
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
OUT = PROJECT_ROOT / "runs_cls"
LOG = OUT / "classifier_tuning.log"
LRS = ["1e-4", "3e-4", "1e-3"]
SMOKE_TRAIN_ARGS = []   # set by --smoke: tiny, fast runs
SMOKE_GENERATED = None


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, console):
    log("running: " + " ".join(str(c) for c in cmd[1:]))
    with open(console, "a", encoding="utf-8") as out:
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        log(f"FAILED (exit {rc}) - see {console}; continuing")
    return rc == 0


def train(lr, seed):
    d = OUT / f"real_lr{lr}_seed{seed}"
    if (d / "metrics.json").is_file():
        log(f"{d.name}: already trained, skipping")
    else:
        d.mkdir(parents=True, exist_ok=True)
        run([PYTHON, "classifier/train_classifier.py", "--output-dir", str(d), "--learning-rate", lr,
             "--seed", str(seed), *SMOKE_TRAIN_ARGS], d / "console.log")
    if (d / "metrics.json").is_file():
        m = json.loads((d / "metrics.json").read_text())
        log(f"{d.name}: best epoch {m['best_epoch']}, val accuracy {m['val']['accuracy']:.4f}, "
            f"macro-F1 {m['val']['macro_f1']:.4f}")
        return m["val"]["accuracy"]
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-for-log", type=Path, default=None,
                   help="wait until this log contains 'candidate evals done' before using the GPU")
    p.add_argument("--smoke", type=Path, default=None, metavar="DIR", help="tiny end-to-end test into DIR")
    args = p.parse_args()
    global OUT, LOG, SMOKE_TRAIN_ARGS, SMOKE_GENERATED
    if args.smoke:
        OUT, SMOKE_TRAIN_ARGS = args.smoke.resolve(), ["--limit", "600", "--epochs", "1", "--batch-size", "32"]
        LOG = OUT / "classifier_tuning.log"
        SMOKE_GENERATED = str(PROJECT_ROOT / "runs" / "lora_r8" / "fid" / "generated" / "checkpoint-4000_cfg2_steps30_seed0_n100.npz")
    OUT.mkdir(exist_ok=True)
    log("=== classifier driver start")
    if args.wait_for_log:
        log(f"waiting for {args.wait_for_log} to report 'candidate evals done'")
        while not (args.wait_for_log.is_file() and "candidate evals done" in args.wait_for_log.read_text("utf-8-sig")):
            time.sleep(60)
        log("candidate evals finished - starting")

    accs = {lr: train(lr, 0) for lr in LRS}
    done = {lr: a for lr, a in accs.items() if a is not None}
    if not done:
        log("no classifier finished - stopping")
        return
    best_lr = max(done, key=done.get)
    log(f"best LR by validation accuracy: {best_lr} ({done[best_lr]:.4f}); all: "
        + ", ".join(f"{lr}={a:.4f}" for lr, a in done.items()))
    train(best_lr, 1)

    clfs = [OUT / f"real_lr{best_lr}_seed{s}" / "best.pt" for s in (0, 1)]
    clfs = [c for c in clfs if c.is_file()]
    fid_out = OUT / "class_fidelity.jsonl"
    if fid_out.is_file():
        fid_out.rename(OUT / f"class_fidelity_{datetime.datetime.now():%Y%m%d_%H%M%S}.jsonl.bak")
    extra = ["--generated", SMOKE_GENERATED] if SMOKE_GENERATED else []
    if run([PYTHON, "classifier/class_fidelity.py", "--classifiers", *map(str, clfs), "--output", str(fid_out), *extra],
           OUT / "candidate_summary.txt"):
        log(f"class fidelity done with {len(clfs)} classifier(s) - see runs_cls/candidate_summary.txt")
    log("=== classifier driver done (no generator choice made - awaiting user review)")


if __name__ == "__main__":
    main()

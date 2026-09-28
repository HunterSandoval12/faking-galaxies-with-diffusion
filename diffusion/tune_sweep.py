"""Overnight validation-set tuning queue (the study's pre-specified tuning protocol).

Usage:
    python diffusion/tune_sweep.py                 # run / resume the whole queue
    python diffusion/tune_sweep.py --summary       # print the comparison tables only
    python diffusion/tune_sweep.py --dry-run       # list what would run, run nothing
    python diffusion/tune_sweep.py --smoke DIR     # tiny end-to-end test into DIR

Queue, in order:
  1. Re-score the baseline (runs/lora_r8) tuning configurations on the CURRENT
     validation set (after the overlapping-cutout fix), plus checkpoints 6000/8000 at
     guidance 2, plus a generation-seed replicate.
  2. Train and score the sweep runs, one setting changed at a time from the baseline
     (rank 8, LR 1e-4, constant schedule, seed 42): a second training seed first
     (run-to-run noise), then rank 4/16, LR 5e-5/2e-4, cosine LR schedule.
Every run is scored on checkpoints 4000/6000/8000/10000 at guidance 2, 30 steps,
100 images/class. Deciding metric (fixed in advance): macro KID; FID secondary.

Scores are only counted if they were computed on the current validation set
(val_sha1 tag), so pre-fix results are never reused. Resumable: finished training
and scoring are skipped. A failed job is logged and the queue moves on.
Progress: runs/tuning_sweep.log.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUNS = PROJECT_ROOT / "runs"
SPLITS = PROJECT_ROOT / "data" / "splits" / "galaxy10_splits.npz"
BASELINE = "lora_r8"

EVAL_CHECKPOINTS = [4000, 6000, 8000, 10000]
EVAL_GUIDANCE, EVAL_STEPS, EVAL_N = 2.0, 30, 100

# Baseline re-scoring on the cleaned validation set: (checkpoints, guidance scales, steps, seed)
RESCORE_JOBS = [
    ([2000, 4000, 6000, 8000, 10000], [4.0], [30], 0),   # stage 1 (checkpoint scan at guidance 4)
    ([10000], [1.5, 2.0, 3.0, 6.0], [30], 0),            # stage 2 (guidance scan)
    ([4000, 6000, 8000], [2.0], [30], 0),                # stage 2b + remaining sweep checkpoints
    ([10000], [2.0], [20, 50, 75, 100], 0),              # stages 3 / 3b (sampling steps)
    ([10000], [2.0], [30], 1),                           # generation-seed replicate
]

# Sweep runs: name -> extra train_lora.py args (everything else = baseline)
TRAIN_RUNS = {
    "lora_r8_seed43": ["--seed", "43"],                  # training-seed replicate (noise estimate)
    "lora_r4": ["--lora-rank", "4"],
    "lora_r16": ["--lora-rank", "16"],
    "lora_r8_lr5e-5": ["--learning-rate", "5e-5"],
    "lora_r8_lr2e-4": ["--learning-rate", "2e-4"],
    "lora_r8_cosine": ["--lr-scheduler", "cosine"],
}


class Queue:
    def __init__(self, root, smoke):
        self.root = root  # where new runs, fid results and the log go
        self.smoke = smoke
        self.log_path = root / "tuning_sweep.log"
        self.val_sha1 = hashlib.sha1(np.load(SPLITS)["val"].astype(np.int64).tobytes()).hexdigest()[:16]
        self.failures = []
        if smoke:
            self.eval_ckpts, self.eval_n = [10, 20], 2
            self.rescore = [([2000], [2.0], [30], 0)]
            self.train_runs = {"lora_r8_seed43": ["--seed", "43"]}
            self.train_extra = ["--max-train-steps", "20", "--checkpoint-every", "10",
                                "--lr-warmup-steps", "2", "--log-every", "5"]
        else:
            self.eval_ckpts, self.eval_n = EVAL_CHECKPOINTS, EVAL_N
            self.rescore, self.train_runs, self.train_extra = RESCORE_JOBS, TRAIN_RUNS, []

    def log(self, msg):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def ckpt_dir(self, run, step):
        base = RUNS if run == BASELINE else self.root  # smoke mode reads real baseline checkpoints
        return base / run / f"checkpoint-{step}"

    def fid_dir(self, run):
        return self.root / run / "fid"

    def results(self, run):
        path = self.fid_dir(run) / "results.jsonl"
        if not path.is_file():
            return []
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return [r for r in rows if r.get("val_sha1") == self.val_sha1 and r["num_per_class"] == self.eval_n]

    def scored(self, run, step, gs, n_steps, seed):
        return any(r["checkpoint"] == f"checkpoint-{step}" and r["guidance_scale"] == gs
                   and r["sampling_steps"] == n_steps and r["seed"] == seed for r in self.results(run))

    def run(self, cmd, console_path):
        self.log("running: " + " ".join(str(c) for c in cmd))
        console_path.parent.mkdir(parents=True, exist_ok=True)
        with open(console_path, "a", encoding="utf-8") as out:
            rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            self.log(f"FAILED (exit {rc}) - see {console_path}; continuing with the next job")
            self.failures.append(" ".join(str(c) for c in cmd[1:3]))
        return rc == 0

    def score(self, run, steps, guidance, n_steps_list, seed):
        """Score every (checkpoint, guidance, steps) combo not yet scored on the current val set."""
        todo = [(s, g, n) for s in steps for g in guidance for n in n_steps_list
                if not self.scored(run, s, g, n, seed)]
        if not todo:
            self.log(f"{run}: {len(steps)}x{len(guidance)}x{len(n_steps_list)} (seed {seed}) already scored")
            return
        # One call if everything is to do (eval_fid runs the cartesian product); else per (g, n).
        groups = ([(steps, guidance, n_steps_list)] if len(todo) == len(steps) * len(guidance) * len(n_steps_list)
                  else [([s for s, g2, n2 in todo if (g2, n2) == (g, n)], [g], [n])
                        for g, n in dict.fromkeys((g, n) for _, g, n in todo)])
        for ckpts, gs, ns in groups:
            ok = self.run([PYTHON, "diffusion/eval_fid.py", *[str(self.ckpt_dir(run, s)) for s in ckpts],
                           "--guidance-scale", *map(str, gs), "--steps", *map(str, ns),
                           "--num-per-class", str(self.eval_n), "--seed", str(seed),
                           "--output-dir", str(self.fid_dir(run))],
                          self.fid_dir(run) / "sweep_eval.log")
            if ok:
                self.log(f"{run}: scored checkpoints {ckpts} at guidance {gs}, steps {ns}, seed {seed}")

    def main(self):
        self.log(f"=== queue start ({'SMOKE TEST' if self.smoke else 'overnight'}); "
                 f"validation set {self.val_sha1}")
        for ckpts, gs, ns, seed in self.rescore:
            self.score(BASELINE, ckpts, gs, ns, seed)
        for name, extra in self.train_runs.items():
            run_dir = self.root / name
            if not (run_dir / f"checkpoint-{self.eval_ckpts[-1]}" / "lora_unet.safetensors").is_file():
                if run_dir.exists():
                    # A partial run would mix into train_log.jsonl; don't silently resume/overwrite it.
                    self.log(f"{run_dir} exists but is incomplete - skipping (move it aside to retrain)")
                    self.failures.append(f"{name} incomplete")
                    continue
                run_dir.mkdir(parents=True)
                if not self.run([PYTHON, "diffusion/train_lora.py", "--output-dir", str(run_dir),
                                 *extra, *self.train_extra], run_dir / "console.log"):
                    continue
                self.log(f"{name}: training done")
            self.score(name, self.eval_ckpts, [EVAL_GUIDANCE], [EVAL_STEPS], 0)
        self.log(f"=== queue done; failures: {self.failures if self.failures else 'none'}")
        self.summary()

    def summary(self):
        print(f"\nValidation set {self.val_sha1}; guidance {EVAL_GUIDANCE:g}, {EVAL_STEPS} steps, "
              f"{self.eval_n}/class; deciding metric macro KID (lower is better)")
        print(f"  {'run':<18}{'ckpt':>7}{'macro KIDx1e3':>15}{'macro FID':>11}")
        best = {}
        for run in [BASELINE, *self.train_runs]:
            for r in self.results(run):
                if (r["guidance_scale"], r["sampling_steps"], r["seed"]) != (EVAL_GUIDANCE, EVAL_STEPS, 0):
                    continue
                step = int(r["checkpoint"].split("-")[1])
                if step not in self.eval_ckpts:
                    continue
                print(f"  {run:<18}{step:>7}{1e3 * r['macro_kid']:>15.2f}{r['macro_fid']:>11.1f}")
                if run not in best or r["macro_kid"] < best[run][1]:
                    best[run] = (step, r["macro_kid"], r["macro_fid"])
        print("\nBest checkpoint per run (by macro KID):")
        for run, (step, kid, fid) in sorted(best.items(), key=lambda kv: kv[1][1]):
            print(f"  {run:<18} checkpoint-{step:<6} KIDx1e3 {1e3 * kid:6.2f}  FID {fid:6.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", type=Path, default=None, metavar="DIR")
    args = p.parse_args()
    q = Queue(args.smoke.resolve() if args.smoke else RUNS, smoke=args.smoke is not None)
    if args.summary:
        q.summary()
    elif args.dry_run:
        print(f"validation set {q.val_sha1}")
        for ckpts, gs, ns, seed in q.rescore:
            todo = [(s, g, n) for s in ckpts for g in gs for n in ns if not q.scored(BASELINE, s, g, n, seed)]
            print(f"  rescore {BASELINE} ckpts {ckpts} guidance {gs} steps {ns} seed {seed}: {len(todo)} to do")
        for name, extra in q.train_runs.items():
            done = (q.root / name / f"checkpoint-{q.eval_ckpts[-1]}" / "lora_unet.safetensors").is_file()
            print(f"  train {name} {extra}: {'trained' if done else 'to train'}; then score {q.eval_ckpts}")
    else:
        q.root.mkdir(parents=True, exist_ok=True)
        q.main()


if __name__ == "__main__":
    main()

"""Re-run saved classifier training runs with extra per-epoch logging, and verify they are identical.

Usage:
    python classifier/reproduce_runs.py [--runs A_replace0_seed* A_replace0.5_seed* A_replace1_seed*]

The original runs (runs_cls/exp/) logged training loss and validation accuracy / macro-F1 per epoch,
but not training accuracy or validation loss. Training is deterministic (fixed seeds, deterministic
cuDNN), so each run is repeated from its saved config.json into runs_cls/repro/<run>/ with the extra
metrics switched on (train_classifier.py logs training accuracy and validation loss; --log-train-eval
adds the un-augmented training-set loss and accuracy). A reproduction is accepted only if every
originally logged number of every epoch (training loss, validation accuracy and macro-F1, learning
rate) and every tensor of best.pt are bit-identical to the original; the result is written to
runs_cls/repro/verification.json. Only the training and validation sets are read, never the test set.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
EXP, REPRO = ROOT / "runs_cls" / "exp", ROOT / "runs_cls" / "repro"
LOGGED = ("train_loss", "val_accuracy", "val_macro_f1", "lr")  # everything the original logs recorded


def command(cfg, out_dir):
    cmd = [sys.executable, "classifier/train_classifier.py", "--output-dir", str(out_dir), "--splits", cfg["splits"],
           "--epochs", str(cfg["epochs"]), "--batch-size", str(cfg["batch_size"]),
           "--learning-rate", repr(cfg["learning_rate"]), "--weight-decay", repr(cfg["weight_decay"]),
           "--warmup-epochs", repr(cfg["warmup_epochs"]), "--seed", str(cfg["seed"]), "--log-train-eval"]
    if cfg.get("no_augment"):
        cmd.append("--no-augment")
    if cfg.get("synthetic_dirs"):
        cmd += ["--synthetic-dirs", *cfg["synthetic_dirs"]]
    for key in ("replace_fraction", "add_fraction", "focus_class", "focus_ratio", "focus_real", "per_class", "limit"):
        if cfg.get(key) is not None:
            cmd += [f"--{key.replace('_', '-')}", repr(cfg[key]) if isinstance(cfg[key], float) else str(cfg[key])]
    if cfg.get("drop_synthetic"):
        cmd.append("--drop-synthetic")
    if cfg.get("train_npz"):
        raise SystemExit("runs trained on --train-npz sets are not supported")
    return cmd


def verify(run):
    orig = [json.loads(l) for l in open(EXP / run / "train_log.jsonl") if l.strip()]
    new = [json.loads(l) for l in open(REPRO / run / "train_log.jsonl") if l.strip()]
    log_ok = len(orig) == len(new) and all(o[k] == n[k] for o, n in zip(orig, new) for k in LOGGED)
    a = torch.load(EXP / run / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
    b = torch.load(REPRO / run / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
    weights_ok = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    return {"log_identical": log_ok, "weights_identical": weights_ok, "epochs": len(new)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["A_replace0_seed*", "A_replace0.5_seed*", "A_replace1_seed*"])
    args = ap.parse_args()
    runs = sorted({d.name for pat in args.runs for d in EXP.glob(pat) if (d / "config.json").is_file()})
    REPRO.mkdir(parents=True, exist_ok=True)
    ver_path = REPRO / "verification.json"
    results = json.loads(ver_path.read_text()) if ver_path.exists() else {}
    for i, run in enumerate(runs, 1):
        if results.get(run, {}).get("weights_identical") and results[run].get("log_identical"):
            print(f"[{i}/{len(runs)}] {run}: already reproduced and verified", flush=True)
            continue
        cfg = json.loads((EXP / run / "config.json").read_text())
        out = REPRO / run
        t = time.time()
        with open(REPRO / f"{run}.console.log", "w") as console:
            rc = subprocess.run(command(cfg, out), cwd=ROOT, stdout=console, stderr=subprocess.STDOUT).returncode
        res = verify(run) if rc == 0 else {"log_identical": False, "weights_identical": False, "exit_code": rc}
        res["minutes"] = round((time.time() - t) / 60, 2)
        results[run] = res
        ver_path.write_text(json.dumps(results, indent=2))
        print(f"[{i}/{len(runs)}] {run}: {res}", flush=True)
    bad = [r for r in runs if not (results[r]["log_identical"] and results[r]["weights_identical"])]
    print(f"done: {len(runs) - len(bad)}/{len(runs)} runs reproduced bit-identically" + (f"; NOT: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

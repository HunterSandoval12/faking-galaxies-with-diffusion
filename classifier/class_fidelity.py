"""Class fidelity of generated galaxies: does a classifier trained on REAL working-pool
images recognize each generated image as the class it was generated for?

Usage:
    python classifier/class_fidelity.py --classifiers runs_cls/real_lr3e-4_seed0/best.pt runs_cls/real_lr3e-4_seed1/best.pt

Scores every saved generated set (runs/*/fid/generated/*_n100.npz, written by
diffusion/eval_fid.py) with each classifier: per-class fidelity = fraction of that
class's generated images predicted as that class; macro fidelity = mean over classes.
The real-image reference is the same classifier's per-class recall on the VALIDATION
set (a realistic ceiling). Appends to runs_cls/class_fidelity.jsonl and prints a table
joining fidelity with the validation KID/FID of the same images. No test data is read.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_splits import CLASS_NAMES  # noqa: E402
from train_classifier import build_model, predict  # noqa: E402

NAME_RE = re.compile(r"(checkpoint-\d+)_cfg([\d.]+)_steps(\d+)_seed(\d+)_n(\d+)\.npz$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classifiers", type=Path, nargs="+", required=True)
    p.add_argument("--generated", default=str(PROJECT_ROOT / "runs" / "*" / "fid" / "generated" / "*_n100.npz"))
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs_cls" / "class_fidelity.jsonl")
    return p.parse_args()


def kid_lookup(run, checkpoint, gs, steps, seed, val_sha1):
    path = PROJECT_ROOT / "runs" / run / "fid" / "results.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()] if path.is_file() else []
    m = [r for r in rows if r["checkpoint"] == checkpoint and r["guidance_scale"] == gs and r["sampling_steps"] == steps
         and r["seed"] == seed and r.get("val_sha1") == val_sha1 and r["num_per_class"] == 100]
    return m[-1] if m else None


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda")
    files = sorted(glob.glob(args.generated))
    if not files:
        raise SystemExit(f"no generated sets match {args.generated}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    table = {}  # (run, ckpt, gs, steps, seed) -> list of macro fidelity (one per classifier)
    per_cls = {}
    val_sha1 = None
    refs = []
    for clf_path in args.classifiers:
        ck = torch.load(clf_path, map_location="cpu", weights_only=False)
        model = build_model().to(device).to(memory_format=torch.channels_last)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        val_sha1 = ck["config"]["val_sha1"]
        ref = {c: ck["val_metrics"]["per_class"][str(c)]["recall"] for c in range(len(CLASS_NAMES))}
        refs.append(ref)
        for f in files:
            m = NAME_RE.search(f)
            run = Path(f).parents[2].name
            ckpt, gs, steps, seed = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
            z = np.load(f)
            probs = predict(model, z["images"], device)
            pred = probs.argmax(1).numpy()
            labels = z["labels"]
            fid_c = {c: float((pred[labels == c] == c).mean()) for c in range(len(CLASS_NAMES))}
            conf_c = {c: float(probs[labels == c, c].mean()) for c in range(len(CLASS_NAMES))}
            macro = float(np.mean(list(fid_c.values())))
            rec = {"classifier": str(Path(clf_path).parent.name), "run": run, "checkpoint": ckpt,
                   "guidance_scale": gs, "sampling_steps": steps, "seed": seed, "file": Path(f).name,
                   "macro_fidelity": macro, "macro_confidence": float(np.mean(list(conf_c.values()))),
                   "per_class_fidelity": {str(c): v for c, v in fid_c.items()},
                   "per_class_confidence": {str(c): v for c, v in conf_c.items()}}
            with open(args.output, "a") as out:
                out.write(json.dumps(rec) + "\n")
            key = (run, ckpt, gs, steps, seed)
            table.setdefault(key, []).append(macro)
            per_cls.setdefault(key, []).append(fid_c)

    ref_macro = np.mean([np.mean(list(r.values())) for r in refs])
    print(f"\nClass fidelity (mean over {len(args.classifiers)} classifier(s)) joined with validation KID/FID")
    print(f"Real reference: classifiers' macro recall on REAL validation images = {ref_macro:.3f}")
    print(f"  {'run':<16}{'checkpoint':<18}{'cfg':>5}{'steps':>7}{'seed':>6}{'fidelity':>10}{'KIDx1e3':>9}{'FID':>7}")
    for key in sorted(table, key=lambda k: -np.mean(table[k])):
        run, ckpt, gs, steps, seed = key
        r = kid_lookup(run, ckpt, gs, steps, seed, val_sha1)
        kid = f"{1e3 * r['macro_kid']:9.1f}" if r else f"{'-':>9}"
        fid = f"{r['macro_fid']:7.1f}" if r else f"{'-':>7}"
        print(f"  {run:<16}{ckpt:<18}{gs:>5g}{steps:>7}{seed:>6}{np.mean(table[key]):>10.3f}{kid}{fid}")
    print(f"\nAppended {sum(len(v) for v in table.values())} rows to {args.output}")


if __name__ == "__main__":
    main()

"""Galaxy10 DECaLS morphology classifier: ResNet-18 (ImageNet-pretrained), real-only baseline.

Usage:
    python classifier/train_classifier.py --output-dir runs_cls/real_lr3e-4_seed0 --learning-rate 3e-4 --seed 0

Training data: the working pool (train split) only. Model selection: the epoch with the
best VALIDATION accuracy (per-class precision/recall/F1 and macro-F1 are logged too).
The held-out test set is never read by this script.

Design (fixed 2026-09-26): native 256x256 input, ImageNet normalization, random
flips + 90-degree rotations (galaxy morphology is orientation-invariant), no class
weighting (the imbalance is part of the experiment), AdamW with 1 warmup epoch then a
cosine decay to 0, bf16 autocast.

Real/synthetic training mixes (--synthetic-dirs = generate_synthetic.py shard folders),
all defined per class so the class balance never changes:
  --replace-fraction f   keep each class's size; a fraction f of it is synthetic
                         (0 = real only, 1 = synthetic only)
  --add-fraction a       all real images + a x (class size) synthetic images
  --focus-class c --focus-ratio k [--focus-real m]
                         other classes real only; class c = its real images (or m random
                         ones) + k x (that many) synthetic images of class c
Which real/synthetic images are drawn depends on --seed, so seed replicates include
sampling variability.

Outputs in --output-dir: best.pt (weights + config + validation metrics of the best
epoch), train_log.jsonl (one line per epoch: training loss and accuracy as measured during
the epoch, with augmentation; validation loss, accuracy and macro-F1; with --log-train-eval
also the loss and accuracy of the whole training set without augmentation, in eval mode),
metrics.json (best epoch, per-class metrics, confusion matrix), config.json.
"""

import argparse
import glob
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
from prepare_splits import CLASS_NAMES  # noqa: E402

GALAXY10_H5 = Path.home() / ".astroNN" / "datasets" / "Galaxy10_DECals.h5"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"  # decoded split images (see read_images_cached)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data" / "splits" / "galaxy10_splits.npz")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke tests only: use N random train/val images")
    p.add_argument("--train-npz", type=Path, nargs="+", default=None,
                   help="train on these saved GENERATED sets (eval_fid.py npz files) instead of the working pool")
    p.add_argument("--per-class", type=int, default=None,
                   help="train on N random working-pool images per class (equal-size real control)")
    p.add_argument("--synthetic-dirs", type=Path, nargs="+", default=None)
    p.add_argument("--replace-fraction", type=float, default=None)
    p.add_argument("--add-fraction", type=float, default=None)
    p.add_argument("--focus-class", type=int, default=None)
    p.add_argument("--focus-ratio", type=float, default=None)
    p.add_argument("--focus-real", type=int, default=None)
    p.add_argument("--drop-synthetic", action="store_true",
                   help="make exactly the same draws as the mix, then leave the synthetic images out "
                        "(the mix's real part alone - a paired control)")
    p.add_argument("--log-train-eval", action="store_true",
                   help="also evaluate the training set without augmentation after every epoch and log its "
                        "loss/accuracy (extra logging only: training itself is unchanged)")
    args = p.parse_args()
    modes = [args.replace_fraction is not None, args.add_fraction is not None, args.focus_class is not None,
             args.train_npz is not None, args.per_class is not None]
    if sum(modes) > 1:
        p.error("choose one of --replace-fraction / --add-fraction / --focus-class / --train-npz / --per-class")
    if (args.replace_fraction is not None or args.add_fraction is not None or args.focus_class is not None) \
            and not args.synthetic_dirs:
        p.error("mixing options need --synthetic-dirs")
    if args.focus_class is not None and args.focus_ratio is None:
        p.error("--focus-class needs --focus-ratio")
    return args


def load_synthetic_class(dirs, c):
    """All synthetic images of class c from generate_synthetic.py shard folders (in folder order)."""
    parts = []
    for d in dirs:
        for sh in sorted(glob.glob(str(Path(d) / f"class{c}_part*.npz"))):
            if re.search(rf"class{c}_part\d+\.npz$", sh):
                parts.append(np.load(sh)["images"])
    return np.concatenate(parts) if parts else np.empty((0, 256, 256, 3), np.uint8)


def synthetic_class_count(dirs, c):
    """Number of synthetic images of class c (reads only the small label arrays)."""
    return sum(len(np.load(sh)["labels"]) for d in dirs for sh in sorted(glob.glob(str(Path(d) / f"class{c}_part*.npz")))
               if re.search(rf"class{c}_part\d+\.npz$", sh))


def compose_training_set(args, real_x, real_y):
    """Build the real/synthetic mix per class; returns (uint8 images, labels, per-class composition)."""
    rng = np.random.default_rng(1000 + args.seed)  # which images are drawn varies with the seed
    ry = real_y.numpy()
    plan = {}
    for c in range(len(CLASS_NAMES)):
        n = int((ry == c).sum())
        if args.replace_fraction is not None:
            n_syn = int(round(args.replace_fraction * n)); n_real = n - n_syn
        elif args.add_fraction is not None:
            n_real, n_syn = n, int(round(args.add_fraction * n))
        elif c == args.focus_class:
            n_real = min(args.focus_real, n) if args.focus_real else n
            n_syn = int(round(args.focus_ratio * n_real))
        else:
            n_real, n_syn = n, 0
        plan[c] = (n_real, n_syn)
    keep_syn = not args.drop_synthetic
    total = sum(a + (b if keep_syn else 0) for a, b in plan.values())
    x = np.empty((total, 256, 256, 3), dtype=np.uint8)  # filled in place: no duplicate copies
    y = np.empty(total, dtype=np.int64)
    pos = 0
    for c, (n_real, n_syn) in plan.items():
        idx_c = np.flatnonzero(ry == c)
        pick = idx_c if n_real == len(idx_c) else np.sort(rng.choice(idx_c, n_real, replace=False))
        x[pos:pos + n_real] = real_x.numpy()[pick]
        y[pos:pos + n_real] = c
        pos += n_real
        if n_syn and not keep_syn:
            # same random draw as the full mix (keeps the generator state, hence the real
            # subsets of later classes, identical), but the synthetic images are left out
            rng.choice(synthetic_class_count(args.synthetic_dirs, c), n_syn, replace=False)
        elif n_syn:
            pool = load_synthetic_class(args.synthetic_dirs, c)
            if len(pool) < n_syn:
                raise SystemExit(f"class {c}: need {n_syn} synthetic images, only {len(pool)} available")
            x[pos:pos + n_syn] = pool[np.sort(rng.choice(len(pool), n_syn, replace=False))]
            y[pos:pos + n_syn] = c
            pos += n_syn
            del pool
    return torch.from_numpy(x), torch.from_numpy(y), {str(c): {"real": a, "synthetic": b if keep_syn else 0}
                                                      for c, (a, b) in plan.items()}


def load_split(splits_path, key, limit=None, per_class=None):
    """Images + labels for one split ('train' or 'val' only - never 'test')."""
    assert key in ("train", "val"), "this script never reads the test split"
    splits = np.load(splits_path)
    idx = splits[key]
    with h5py.File(GALAXY10_H5, "r") as f:
        labels = f["ans"][:].astype(np.int64)
        if hashlib.sha1(labels.tobytes()).hexdigest() != str(splits["labels_sha1"]):
            raise SystemExit(f"{GALAXY10_H5} does not match the dataset the splits were made from")
        rng = np.random.default_rng(0)  # fixed subset, independent of the training seed
        if per_class:
            idx = np.sort(np.concatenate([rng.choice(idx[labels[idx] == c], per_class, replace=False)
                                          for c in range(len(CLASS_NAMES))]))
        elif limit:  # random subset (the file is sorted by class, so the first N would be one class)
            idx = np.sort(rng.choice(idx, min(limit, len(idx)), replace=False))
        if per_class or limit:
            images = np.empty((len(idx), 256, 256, 3), dtype=np.uint8)
            for s in range(0, len(idx), 1024):
                images[s:s + 1024] = f["images"][idx[s:s + 1024]]
        else:
            images = read_images_cached(f["images"], idx)
    return torch.from_numpy(images), torch.from_numpy(labels[idx])


def read_images_cached(dset, idx):
    """Images at sorted indices `idx`, cached as .npy (keyed by the index hash).

    The HDF5 images are gzip-compressed in tiny chunks (555x16x16x1), so scattered reads
    are ~20x slower than contiguous ones (a full working pool took ~5 min). The first call
    reads the file in contiguous blocks and keeps the wanted rows; later calls load the
    cache in seconds.
    """
    key = hashlib.sha1(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]
    path = CACHE_DIR / f"images_{key}.npy"
    if path.is_file():
        images = np.load(path)
        if images.shape == (len(idx), 256, 256, 3):
            return images
    images = np.empty((len(idx), 256, 256, 3), dtype=np.uint8)
    block = 2220  # multiple of the chunk length along axis 0
    for start in range(0, dset.shape[0], block):
        lo, hi = np.searchsorted(idx, [start, start + block])
        if lo < hi:
            images[lo:hi] = dset[start:start + block][idx[lo:hi] - start]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, images)
    tmp.replace(path)  # atomic: the cache file is either complete or absent
    return images


def to_input(images_u8, augment):
    """uint8 NHWC (on GPU) -> normalized float NCHW; optional random dihedral transform."""
    x = images_u8.permute(0, 3, 1, 2).float().div_(255.0)
    if augment:
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        k = torch.randint(0, 4, (x.shape[0],), device=x.device)
        x = torch.stack([torch.rot90(xi, int(ki), dims=(-2, -1)) for xi, ki in zip(x, k)])
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


@torch.no_grad()
def predict(model, images_u8, device, batch_size=256):
    """Class probabilities for uint8 NHWC images (CPU tensor or numpy)."""
    model.eval()
    out = []
    for s in range(0, len(images_u8), batch_size):
        x = torch.as_tensor(images_u8[s:s + batch_size]).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(to_input(x, augment=False).contiguous(memory_format=torch.channels_last))
        out.append(F.softmax(logits.float(), dim=1).cpu())
    return torch.cat(out)


def evaluate(model, images, labels, device):
    probs = predict(model, images, device)
    pred = probs.argmax(1).numpy()
    y = labels.numpy()
    p, r, f1, n = precision_recall_fscore_support(y, pred, labels=range(len(CLASS_NAMES)), zero_division=0)
    return {
        "accuracy": float((pred == y).mean()),
        "loss": float(F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels).item()),  # cross-entropy
        "macro_f1": float(f1.mean()),
        "per_class": {str(c): {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f1[c]), "n": int(n[c])}
                      for c in range(len(CLASS_NAMES))},
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(CLASS_NAMES))).tolist(),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.train_npz:
        parts = [np.load(p) for p in args.train_npz]
        train_x = torch.from_numpy(np.concatenate([z["images"] for z in parts]))
        train_y = torch.from_numpy(np.concatenate([z["labels"] for z in parts]).astype(np.int64))
        train_source = "generated: " + ", ".join(str(p) for p in args.train_npz)
    else:
        train_x, train_y = load_split(args.splits, "train", args.limit, args.per_class)
        train_source = f"working pool ({args.per_class}/class)" if args.per_class else "working pool"
    composition = None
    if args.replace_fraction is not None or args.add_fraction is not None or args.focus_class is not None:
        real_x, real_y = train_x, train_y
        train_x, train_y, composition = compose_training_set(args, real_x, real_y)
        del real_x, real_y
        mode = (f"replace {args.replace_fraction:g}" if args.replace_fraction is not None else
                f"add {args.add_fraction:g}" if args.add_fraction is not None else
                f"focus class {args.focus_class} ratio {args.focus_ratio:g}"
                + (f", {args.focus_real} real" if args.focus_real else "")) \
            + (", synthetic dropped" if args.drop_synthetic else "")
        n_syn = sum(v["synthetic"] for v in composition.values())
        train_source = f"mix ({mode}): {len(train_y) - n_syn} real + {n_syn} synthetic"
    val_x, val_y = load_split(args.splits, "val", args.limit)
    if len(train_x) <= 13000:  # pinning copies the data; skip it for large mixes to limit RAM use
        try:
            train_x = train_x.pin_memory()
        except RuntimeError as e:
            print(f"warning: could not pin train images ({e}); using pageable memory", flush=True)

    model = build_model().to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_x) / args.batch_size)
    total_steps, warmup_steps = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    def jsonable(v):
        return str(v) if isinstance(v, Path) else [jsonable(x) for x in v] if isinstance(v, list) else v
    config = {**{k: jsonable(v) for k, v in vars(args).items()},
              "model": "resnet18", "pretrained": "IMAGENET1K_V1", "input_size": 256, "train_source": train_source,
              "composition": composition,
              "n_train": len(train_x), "n_val": len(val_x),
              "val_sha1": hashlib.sha1(np.load(args.splits)["val"].astype(np.int64).tobytes()).hexdigest()[:16],
              "train_class_counts": torch.bincount(train_y, minlength=len(CLASS_NAMES)).tolist()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"ResNet-18 (ImageNet-pretrained) | train {len(train_x)} [{train_source}] | val {len(val_x)} | lr {args.learning_rate:g} | "
          f"wd {args.weight_decay:g} | {args.epochs} epochs x {steps_per_epoch} steps | seed {args.seed}", flush=True)

    best, t0, step = None, time.time(), 0
    with open(args.output_dir / "train_log.jsonl", "w") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(len(train_x))
            total_loss, n_seen, n_correct = 0.0, 0, 0
            for s in range(0, len(perm), args.batch_size):
                bi = perm[s:s + args.batch_size]
                x = to_input(train_x[bi].to(device, non_blocking=True), augment=not args.no_augment)
                y = train_y[bi].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x.contiguous(memory_format=torch.channels_last))
                loss = F.cross_entropy(logits.float(), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                total_loss += loss.item() * len(bi)
                n_seen += len(bi)
                n_correct += int((logits.argmax(1) == y).sum())
            m = evaluate(model, val_x, val_y, device)
            rec = {"epoch": epoch, "train_loss": total_loss / n_seen, "val_accuracy": m["accuracy"],
                   "val_macro_f1": m["macro_f1"], "lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1),
                   "train_accuracy": n_correct / n_seen, "val_loss": m["loss"]}
            if args.log_train_eval:  # the whole training set, no augmentation, eval mode
                mt = evaluate(model, train_x, train_y, device)
                rec.update(train_eval_loss=mt["loss"], train_eval_accuracy=mt["accuracy"])
            log.write(json.dumps(rec) + "\n")
            log.flush()
            improved = best is None or m["accuracy"] > best["accuracy"]
            print(f"epoch {epoch:>3}/{args.epochs}  loss {rec['train_loss']:.4f}  val acc {m['accuracy']:.4f}  "
                  f"macro-F1 {m['macro_f1']:.4f}  {rec['elapsed_s']:.0f}s{'  *best' if improved else ''}", flush=True)
            if improved:
                best = {**m, "epoch": epoch}
                torch.save({"state_dict": model.state_dict(), "config": config, "val_metrics": best},
                           args.output_dir / "best.pt")

    (args.output_dir / "metrics.json").write_text(json.dumps({"best_epoch": best["epoch"], "val": best}, indent=2))
    print(f"Done: best epoch {best['epoch']} - val accuracy {best['accuracy']:.4f}, macro-F1 {best['macro_f1']:.4f} "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

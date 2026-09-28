"""Generate the synthetic Galaxy10 training set with the final generator settings.

Usage:
    python diffusion/generate_synthetic.py --output-dir data/synthetic/main_cfg3_steps30

Settings (chosen on the validation set): LoRA checkpoint
runs/lora_r8/checkpoint-10000, guidance 3, 30 DPM-Solver++ steps, generated at 512px and
downsampled (Lanczos) to the native 256px - exactly the pipeline used for all tuning.

Per-class counts default to the working pool's (train split) per-class counts, as the
study design specifies for the main real-vs-synthetic experiment.

Noise seeds are disjoint from every seed used during tuning (those were < 2e6):
image k of class c uses seed SEED_BASE + c * 1_000_000 + k, so the final data is
separate from anything used to choose the settings, and every image is reproducible.

Output: shards <output-dir>/class{c}_part{p:03d}.npz (images uint8 N x 256 x 256 x 3,
labels, seeds, sample_index), written as they complete - an interrupted run resumes by
skipping complete shards - plus manifest.json.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_splits import CLASS_NAMES  # noqa: E402
from sample import GALAXY10_H5, load_lora_checkpoint  # noqa: E402

SEED_BASE = 7_000_000_000
NATIVE_RES = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs" / "lora_r8" / "checkpoint-10000")
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--classes", type=int, nargs="+", default=list(range(len(CLASS_NAMES))))
    p.add_argument("--per-class", type=int, default=None,
                   help="images per class (default: the working pool's per-class counts)")
    p.add_argument("--start-index", type=int, default=0,
                   help="first sample index k (use to extend a class without reusing seeds)")
    p.add_argument("--shard-size", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=20)
    return p.parse_args()


def working_pool_counts(splits_path):
    import h5py
    train_idx = np.load(splits_path)["train"]
    with h5py.File(GALAXY10_H5, "r") as f:
        labels = f["ans"][:].astype(np.int64)
    return np.bincount(labels[train_idx], minlength=len(CLASS_NAMES)).tolist()


def shard_ok(path, n):
    try:
        z = np.load(path)
        return z["images"].shape == (n, NATIVE_RES, NATIVE_RES, 3) and len(z["labels"]) == n
    except Exception:
        return False


@torch.no_grad()
def main():
    args = parse_args()
    ckpt = args.checkpoint.resolve()
    cfg = json.loads((ckpt / "train_config.json").read_text())
    counts = working_pool_counts(Path(cfg["splits"]))
    if args.per_class is not None:
        counts = [args.per_class] * len(CLASS_NAMES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "checkpoint": str(ckpt), "guidance_scale": args.guidance_scale, "sampling_steps": args.steps,
        "sampler": "DPMSolverMultistepScheduler (from the SD 1.5 scheduler config)", "dtype": "float16",
        "generated_resolution": cfg["resolution"], "saved_resolution": NATIVE_RES, "downsample": "PIL LANCZOS",
        "seed_scheme": f"seed = {SEED_BASE} + class * 1_000_000 + k", "start_index": args.start_index,
        "per_class_counts": {str(c): counts[c] for c in args.classes},
        "total": int(sum(counts[c] for c in args.classes)), "shard_size": args.shard_size,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["pretrained_model"], dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    table, cfg = load_lora_checkpoint(pipe.unet, ckpt)
    table = table.to("cuda", torch.float16)
    null_idx, res = cfg["null_class_index"], cfg["resolution"]

    t_start, done = time.time(), 0
    total = manifest["total"]
    for c in args.classes:
        n = counts[c]
        for p, s in enumerate(range(0, n, args.shard_size)):
            ks = list(range(args.start_index + s, args.start_index + min(s + args.shard_size, n)))
            path = args.output_dir / f"class{c}_part{(args.start_index + s) // args.shard_size:03d}.npz"
            if shard_ok(path, len(ks)):
                done += len(ks)
                print(f"class {c} shard {path.name}: exists, skipping", flush=True)
                continue
            imgs = []
            for b in range(0, len(ks), args.batch_size):
                kb = ks[b:b + args.batch_size]
                gens = [torch.Generator("cuda").manual_seed(SEED_BASE + c * 1_000_000 + k) for k in kb]
                out = pipe(
                    prompt_embeds=table[c].unsqueeze(0).expand(len(kb), -1, -1),
                    negative_prompt_embeds=table[null_idx].unsqueeze(0).expand(len(kb), -1, -1),
                    guidance_scale=args.guidance_scale, num_inference_steps=args.steps,
                    height=res, width=res, generator=gens,
                ).images
                imgs += [np.asarray(im.resize((NATIVE_RES, NATIVE_RES), Image.LANCZOS)) for im in out]
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, images=np.stack(imgs), labels=np.full(len(ks), c, dtype=np.int64),
                                seeds=np.array([SEED_BASE + c * 1_000_000 + k for k in ks], dtype=np.int64),
                                sample_index=np.array(ks, dtype=np.int64))
            tmp.replace(path)  # atomic: a shard file is either complete or absent
            done += len(ks)
            el = time.time() - t_start
            print(f"class {c} ({CLASS_NAMES[c]}) shard {path.name}: {len(ks)} images | total {done}/{total} | "
                  f"{el / 60:.1f} min elapsed", flush=True)
    print(f"Done: {done} images in {args.output_dir}")


if __name__ == "__main__":
    main()

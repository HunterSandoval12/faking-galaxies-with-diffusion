"""Per-class FID / KID of generated galaxies against the VALIDATION set, for choosing the
checkpoint and sampling settings (guidance scale, sampling steps).

Usage:
    python diffusion/eval_fid.py runs/lora_r8/checkpoint-{2000,4000,6000,8000,10000} --guidance-scale 4
    python diffusion/eval_fid.py runs/lora_r8/checkpoint-10000 --guidance-scale 1.5 2 3 4 6 --steps 30

Per the study's protocol (fixed in advance), generated images are compared against the validation set
only; the held-out test set is never loaded. A reference row scores the same number of
real working-pool images against validation the same way, giving the best score
achievable at this sample size. For every (checkpoint, guidance scale, steps) configuration this
generates --num-per-class images per class, downsamples them to the native 256x256,
and compares Inception-v3 pool features against the real validation images of the
same class. Results are appended to <output-dir>/results.jsonl.

Notes on interpretation:
- Validation classes are small (50-395 images), so absolute FID values are biased
  upward and not comparable to published FIDs. Compare configurations only at the same
  --num-per-class. KID is unbiased at small sample sizes and is reported alongside.
- Sample k of class c uses the same initial noise in every configuration, which
  reduces noise in between-configuration comparisons.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image
from scipy import linalg
from torchmetrics.image.fid import NoTrainInceptionV3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_splits import CLASS_NAMES  # noqa: E402
from sample import GALAXY10_H5, load_lora_checkpoint  # noqa: E402

NATIVE_RES = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint-<step> directories")
    p.add_argument("--guidance-scale", type=float, nargs="+", default=[4.0])
    p.add_argument("--steps", type=int, nargs="+", default=[30], help="DPM-Solver++ sampling steps")
    p.add_argument("--num-per-class", type=int, default=100)
    p.add_argument("--classes", type=int, nargs="+", default=list(range(len(CLASS_NAMES))))
    p.add_argument("--gen-batch-size", type=int, default=20)
    p.add_argument("--kid-subsets", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None, help="default: <run dir>/fid")
    p.add_argument("--no-save", action="store_true",
                   help="don't save generated images/features to <output-dir>/generated/")
    return p.parse_args()


# ------------------------------------------------------------------------- metrics


def fid_from_features(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():  # standard fallback for near-singular covariances
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kid_from_features(a, b, n_subsets, rng):
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al. 2018). Returns (mean, std)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    m = min(len(a), len(b), 1000)
    d = a.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


class InceptionFeatures:
    def __init__(self, device):
        self.net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images_u8_nhwc, batch_size=100):
        """uint8 NHWC numpy array -> (N, 2048) float32 features."""
        out = []
        for i in range(0, len(images_u8_nhwc), batch_size):
            x = torch.from_numpy(images_u8_nhwc[i:i + batch_size]).permute(0, 3, 1, 2).to(self.device)
            out.append(self.net(x).float().cpu().numpy())
        return np.concatenate(out)


# ---------------------------------------------------------------------------- data


def validation_fingerprint(splits_path):
    """Short hash of the validation indices - identifies which validation set a score used."""
    return hashlib.sha1(np.load(splits_path)["val"].astype(np.int64).tobytes()).hexdigest()[:16]


def validation_features(splits_path, classes, extractor, cache_path):
    """Inception features of the VALIDATION images, per class (cached). Test is never read."""
    splits = np.load(splits_path)
    val_idx = splits["val"]  # this script reads only "val" and "train" (reference), never "test"
    key = str(splits["labels_sha1"])
    if cache_path.is_file():
        cached = np.load(cache_path)
        if str(cached["labels_sha1"]) == key and np.array_equal(cached["val_idx"], val_idx):
            return {c: cached[f"class{c}"] for c in classes}
    with h5py.File(GALAXY10_H5, "r") as f:
        labels = f["ans"][:].astype(np.int64)
        images = f["images"][val_idx]
    feats_all = extractor(images)
    feats = {c: feats_all[labels[val_idx] == c] for c in range(len(CLASS_NAMES))}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, labels_sha1=key, val_idx=val_idx, **{f"class{c}": v for c, v in feats.items()})
    return {c: feats[c] for c in classes}


def working_pool_reference(splits_path, classes, n_per_class, extractor, real, kid_subsets, seed):
    """Score n_per_class random real working-pool (train) images per class against validation."""
    splits = np.load(splits_path)
    train_idx = splits["train"]
    rng = np.random.default_rng(seed)
    out = {}
    with h5py.File(GALAXY10_H5, "r") as f:
        labels = f["ans"][:].astype(np.int64)
        for c in classes:
            pool = train_idx[labels[train_idx] == c]
            pick = np.sort(rng.choice(pool, min(n_per_class, len(pool)), replace=False))
            feats = extractor(f["images"][pick])
            kid_m, kid_s = kid_from_features(feats, real[c], kid_subsets, np.random.default_rng(seed + c))
            out[c] = {"fid": fid_from_features(feats, real[c]), "kid": kid_m, "kid_std": kid_s,
                      "n_gen": len(feats), "n_val": len(real[c])}
    return out


def print_table(title, per_class, rec):
    print(f"\n{title}")
    print(f"  {'cls':>3}  {'name':<24}{'n_val':>6}{'FID':>9}{'KID x1e3':>12}")
    for c, v in per_class.items():
        print(f"  {c:>3}  {CLASS_NAMES[c]:<24}{v['n_val']:>6}{v['fid']:>9.1f}"
              f"{1e3 * v['kid']:>8.2f} ±{1e3 * v['kid_std']:.2f}")
    print(f"  {'':>3}  {'macro average':<24}{'':>6}{rec['macro_fid']:>9.1f}{1e3 * rec['macro_kid']:>8.2f}",
          flush=True)


# ---------------------------------------------------------------------------- main


@torch.no_grad()
def main():
    args = parse_args()
    ckpts = [c.resolve() for c in args.checkpoints]
    for c in ckpts:
        if not (c / "lora_unet.safetensors").is_file():
            raise SystemExit(f"not a LoRA checkpoint directory: {c}")
    first_cfg = json.loads((ckpts[0] / "train_config.json").read_text())
    out_dir = args.output_dir or ckpts[0].parent / "fid"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    extractor = InceptionFeatures(device)
    val_sha1 = validation_fingerprint(Path(first_cfg["splits"]))
    real = validation_features(Path(first_cfg["splits"]), args.classes, extractor, out_dir / "val_features.npz")
    print(f"validation set {val_sha1}; images per class: " + ", ".join(f"{c}:{len(real[c])}" for c in args.classes))

    # Reference floor: the same number of REAL working-pool images scored the same way,
    # i.e. what a perfect generator would get at this sample size.
    ref = working_pool_reference(Path(first_cfg["splits"]), args.classes, args.num_per_class,
                                 extractor, real, args.kid_subsets, args.seed)
    ref_rec = {"run": "reference", "checkpoint": "real-working-pool", "step": None,
               "guidance_scale": None, "sampling_steps": None,
               "num_per_class": args.num_per_class, "seed": args.seed, "val_sha1": val_sha1,
               "macro_fid": float(np.mean([v["fid"] for v in ref.values()])),
               "macro_kid": float(np.mean([v["kid"] for v in ref.values()])),
               "per_class": {str(c): v for c, v in ref.items()}}
    with open(out_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(ref_rec) + "\n")
    print_table(f"REFERENCE: {args.num_per_class} real working-pool images/class vs validation "
                f"(best achievable at this sample size)", ref, ref_rec)

    pipe = StableDiffusionPipeline.from_pretrained(
        first_cfg["pretrained_model"], dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    summary = []
    for ckpt in ckpts:
        table, cfg = load_lora_checkpoint(pipe.unet, ckpt)
        table = table.to(device, torch.float16)
        null_idx, res = cfg["null_class_index"], cfg["resolution"]
        step = cfg.get("step")
        for gs in args.guidance_scale:
            for n_steps in args.steps:
                t0 = time.time()
                per_class = {}
                saved = {"images": [], "features": [], "labels": [], "sample_index": []}
                for c in args.classes:
                    imgs = []
                    for start in range(0, args.num_per_class, args.gen_batch_size):
                        ks = range(start, min(start + args.gen_batch_size, args.num_per_class))
                        gens = [torch.Generator(device).manual_seed(args.seed * 1_000_003 + c * 10_000 + k) for k in ks]
                        n = len(gens)
                        out = pipe(
                            prompt_embeds=table[c].unsqueeze(0).expand(n, -1, -1),
                            negative_prompt_embeds=table[null_idx].unsqueeze(0).expand(n, -1, -1),
                            guidance_scale=gs, num_inference_steps=n_steps,
                            height=res, width=res, generator=gens,
                        ).images
                        imgs += [np.asarray(im.resize((NATIVE_RES, NATIVE_RES), Image.LANCZOS)) for im in out]
                    gen_feats = extractor(np.stack(imgs))
                    kid_m, kid_s = kid_from_features(gen_feats, real[c], args.kid_subsets,
                                                     np.random.default_rng(args.seed + c))
                    per_class[c] = {"fid": fid_from_features(gen_feats, real[c]),
                                    "kid": kid_m, "kid_std": kid_s,
                                    "n_gen": len(gen_feats), "n_val": len(real[c])}
                    if not args.no_save:
                        saved["images"].append(np.stack(imgs))
                        saved["features"].append(gen_feats.astype(np.float32))
                        saved["labels"].append(np.full(len(imgs), c, dtype=np.int64))
                        saved["sample_index"].append(np.arange(len(imgs), dtype=np.int64))

                if not args.no_save:
                    # Kept so the images can be re-scored later (new validation set, class-fidelity
                    # classifier) without regenerating them.
                    gen_dir = out_dir / "generated"
                    gen_dir.mkdir(exist_ok=True)
                    np.savez_compressed(
                        gen_dir / f"{ckpt.name}_cfg{gs:g}_steps{n_steps}_seed{args.seed}_n{args.num_per_class}.npz",
                        **{k: np.concatenate(v) for k, v in saved.items()})

                macro_fid = float(np.mean([v["fid"] for v in per_class.values()]))
                macro_kid = float(np.mean([v["kid"] for v in per_class.values()]))
                rec = {"run": ckpt.parent.name, "checkpoint": ckpt.name, "step": step,
                       "guidance_scale": gs, "sampling_steps": n_steps,
                       "num_per_class": args.num_per_class, "seed": args.seed, "val_sha1": val_sha1,
                       "macro_fid": macro_fid, "macro_kid": macro_kid,
                       "per_class": {str(c): v for c, v in per_class.items()},
                       "elapsed_s": round(time.time() - t0, 1)}
                with open(out_dir / "results.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                summary.append(rec)

                print_table(f"{ckpt.parent.name} step {step} | cfg {gs:g} | {n_steps} steps | "
                            f"{args.num_per_class}/class | {rec['elapsed_s']:.0f} s", per_class, rec)

    if len(summary) > 1:
        print("\nSummary (sorted by macro FID; lower is better):")
        print(f"  {'checkpoint':<18}{'cfg':>5}{'steps':>7}{'macro FID':>11}{'macro KIDx1e3':>15}")
        for r in sorted(summary, key=lambda r: r["macro_fid"]):
            print(f"  {r['checkpoint']:<18}{r['guidance_scale']:>5g}{r['sampling_steps']:>7}"
                  f"{r['macro_fid']:>11.1f}{1e3 * r['macro_kid']:>15.2f}")
        print(f"  {'reference (real)':<18}{'':>5}{'':>7}{ref_rec['macro_fid']:>11.1f}"
              f"{1e3 * ref_rec['macro_kid']:>15.2f}")
    print(f"\nResults appended to {out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()

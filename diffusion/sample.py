"""Generate class-conditional Galaxy10 samples from LoRA checkpoints made by train_lora.py.

Usage:
    python diffusion/sample.py runs/lora_r8/checkpoint-2000 runs/lora_r8/checkpoint-10000
    python diffusion/sample.py runs/lora_r8/checkpoint-10000 --guidance-scale 1 3 5 --num-per-class 6

For every (checkpoint, guidance scale) it writes one grid PNG: one row per class,
starting with --real-per-class real TRAIN images for reference, then generated
samples. Sample k of class c always uses the same initial noise (seed derived
from --seed, c, k), so grids from different checkpoints / guidance scales are
directly comparable.

Classifier-free guidance uses the learned class embedding as the conditional
input and the learned null-class embedding (index 10) as the unconditional one.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
from prepare_splits import CLASS_NAMES  # noqa: E402

GALAXY10_H5 = Path.home() / ".astroNN" / "datasets" / "Galaxy10_DECals.h5"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint-<step> directories")
    p.add_argument("--classes", type=int, nargs="+", default=list(range(len(CLASS_NAMES))))
    p.add_argument("--num-per-class", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, nargs="+", default=[4.0],
                   help="one grid per value; 1.0 = no guidance (conditional only)")
    p.add_argument("--steps", type=int, default=30, help="DPM-Solver++ inference steps")
    p.add_argument("--resolution", type=int, default=None, help="default: the checkpoint's training resolution")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--real-per-class", type=int, default=2, help="real train images per row (0 = none)")
    p.add_argument("--tile", type=int, default=256, help="grid tile size in pixels")
    p.add_argument("--output-dir", type=Path, default=None, help="default: <run dir>/samples")
    p.add_argument("--save-individual", action="store_true", help="also save each sample as its own PNG")
    return p.parse_args()


def load_lora_checkpoint(unet, ckpt_dir):
    """Load LoRA weights into `unet` (adding the adapter on first use). Returns (class_table, config)."""
    cfg = json.loads((ckpt_dir / "train_config.json").read_text())
    if not getattr(unet, "_hf_peft_config_loaded", False):
        unet.add_adapter(LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                                    init_lora_weights="gaussian", target_modules=cfg["lora_target_modules"]))
    state = load_file(ckpt_dir / "lora_unet.safetensors")
    res = set_peft_model_state_dict(unet, state)
    missing = [k for k in res.missing_keys if "lora" in k]
    if res.unexpected_keys or missing:
        raise SystemExit(f"{ckpt_dir}: LoRA keys do not match (unexpected {len(res.unexpected_keys)}, "
                         f"missing {len(missing)})")
    table = load_file(ckpt_dir / "class_embeddings.safetensors")["table"]
    return table, cfg


def real_train_examples(splits_path, classes, per_class, seed):
    """A few real images per class, drawn from the TRAIN split only."""
    if per_class == 0:
        return {}
    splits = np.load(splits_path)
    train_idx = splits["train"]
    with h5py.File(GALAXY10_H5, "r") as f:
        labels = f["ans"][:].astype(np.int64)
        rng = np.random.default_rng(seed)
        out = {}
        for c in classes:
            pick = np.sort(rng.choice(train_idx[labels[train_idx] == c], per_class, replace=False))
            out[c] = [Image.fromarray(im) for im in f["images"][pick]]
    return out


def make_grid(rows, real, classes, tile, title):
    """rows: {class: [PIL images]}; real: {class: [PIL images]}."""
    n_real = len(real.get(classes[0], []))
    n_gen = len(rows[classes[0]])
    label_w, gap, header_h = 190, 12 if n_real else 0, 44
    width = label_w + (n_real + n_gen) * tile + gap
    height = header_h + len(classes) * tile
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    draw.text((6, 6), title, fill="black", font=font)  # own line, above the column labels
    if n_real:
        draw.text((label_w + 4, 26), "real (train)", fill="black", font=font)
    draw.text((label_w + n_real * tile + gap + 4, 26), "generated", fill="black", font=font)
    for r, c in enumerate(classes):
        y = header_h + r * tile
        draw.text((6, y + tile // 2 - 12), f"{c}: {CLASS_NAMES[c]}", fill="black", font=font)
        x = label_w
        for im in real.get(c, []):
            grid.paste(im.resize((tile, tile), Image.LANCZOS), (x, y))
            x += tile
        x += gap
        for im in rows[c]:
            grid.paste(im.resize((tile, tile), Image.LANCZOS), (x, y))
            x += tile
    return grid


@torch.no_grad()
def main():
    args = parse_args()
    ckpts = [c.resolve() for c in args.checkpoints]
    for c in ckpts:
        if not (c / "lora_unet.safetensors").is_file():
            raise SystemExit(f"not a LoRA checkpoint directory: {c}")
    first_cfg = json.loads((ckpts[0] / "train_config.json").read_text())
    for c in args.classes:
        if not 0 <= c < len(CLASS_NAMES):
            raise SystemExit(f"invalid class {c}")

    device = torch.device("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(
        first_cfg["pretrained_model"], dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    # The text encoder is loaded but never called: prompt_embeds come from the learned class table.

    real = real_train_examples(Path(first_cfg["splits"]), args.classes, args.real_per_class, args.seed)

    for ckpt in ckpts:
        table, cfg = load_lora_checkpoint(pipe.unet, ckpt)
        table = table.to(device, torch.float16)
        null_idx = cfg["null_class_index"]
        res = args.resolution or cfg["resolution"]
        out_dir = args.output_dir or ckpt.parent / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        step = cfg.get("step", ckpt.name.split("-")[-1])

        for gs in args.guidance_scale:
            rows = {}
            for c in args.classes:
                n = args.num_per_class
                gens = [torch.Generator(device).manual_seed(args.seed * 1_000_003 + c * 1000 + k) for k in range(n)]
                images = pipe(
                    prompt_embeds=table[c].unsqueeze(0).expand(n, -1, -1),
                    negative_prompt_embeds=table[null_idx].unsqueeze(0).expand(n, -1, -1),
                    guidance_scale=gs,
                    num_inference_steps=args.steps,
                    height=res, width=res,
                    generator=gens,
                ).images
                rows[c] = images
                if args.save_individual:
                    ind = out_dir / f"step{step}_cfg{gs:g}" / f"class{c}"
                    ind.mkdir(parents=True, exist_ok=True)
                    for k, im in enumerate(images):
                        im.save(ind / f"sample{k}.png")

            title = f"{ckpt.parent.name} step {step} | cfg {gs:g} | {args.steps} steps | {res}px | seed {args.seed}"
            grid = make_grid(rows, real, args.classes, args.tile, title)
            path = out_dir / f"grid_step{step}_cfg{gs:g}.png"
            grid.save(path)
            print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()

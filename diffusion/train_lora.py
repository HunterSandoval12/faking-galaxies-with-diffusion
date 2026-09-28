"""LoRA fine-tuning of the Stable Diffusion U-Net, class-conditional on Galaxy10 DECaLS.

Usage:
    python diffusion/train_lora.py [--lora-rank 8] [--resolution 512] [--max-train-steps 10000] ...

Conditioning: text prompts are replaced by a learned class embedding table of
shape (num_classes + 1, seq_len, 768) that is fed to the U-Net's cross-attention
as `encoder_hidden_states`. Row `num_classes` is the "null" (unconditional)
class used for classifier-free guidance. Each row is initialized from the CLIP
text encoding of a class prompt (the null row from the empty prompt, as in SD),
so training starts from conditioning the pretrained U-Net already understands;
the text encoder is then discarded.

Data: only the `train` indices from data/splits/galaxy10_splits.npz are ever
read; val/test images are never loaded.

Checkpoints (every --checkpoint-every steps, and at the end) contain only the
LoRA weights and the class embedding table, never the base model:
    <output-dir>/checkpoint-<step>/lora_unet.safetensors
    <output-dir>/checkpoint-<step>/class_embeddings.safetensors
    <output-dir>/checkpoint-<step>/train_config.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
from prepare_splits import CLASS_NAMES  # noqa: E402

LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]  # attention projections
GALAXY10_H5 = Path.home() / ".astroNN" / "datasets" / "Galaxy10_DECals.h5"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data" / "splits" / "galaxy10_splits.npz")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "lora")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=None, help="default: equal to --lora-rank (scale 1.0)")
    p.add_argument("--cond-dropout", type=float, default=0.1, help="probability of replacing the class with the null class")
    p.add_argument("--resolution", type=int, default=512, help="training resolution; native images are 256x256")
    p.add_argument("--class-embed-init", choices=["clip", "random"], default="clip")
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-train-steps", type=int, default=10000, help="optimizer steps")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", default="constant_with_warmup")
    p.add_argument("--lr-warmup-steps", type=int, default=500)
    p.add_argument("--adam-weight-decay", type=float, default=1e-2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default="bf16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--no-augment", action="store_true", help="disable random flips / 90-degree rotations")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = float(args.lora_rank)
    if not 0.0 <= args.cond_dropout < 1.0:
        p.error("--cond-dropout must be in [0, 1)")
    if args.resolution % 8 != 0:
        p.error("--resolution must be a multiple of 8 (VAE downsampling factor)")
    return args


# ----------------------------------------------------------------------------- data


def load_train_split(splits_path):
    """Load ONLY the train images/labels. Val and test indices are never touched."""
    splits = np.load(splits_path)
    train_idx = splits["train"]  # the only split key this script reads

    if not GALAXY10_H5.is_file():
        from astroNN.datasets import load_galaxy10  # downloads + verifies the checksum
        load_galaxy10()

    with h5py.File(GALAXY10_H5, "r") as f:
        all_labels = f["ans"][:].astype(np.int64)
        if hashlib.sha1(all_labels.tobytes()).hexdigest() != str(splits["labels_sha1"]):
            raise SystemExit(f"{GALAXY10_H5} does not match the dataset the splits were made from")
        assert np.all(np.diff(train_idx) > 0), "train indices must be sorted and unique"
        # h5py fancy indexing needs increasing indices; read in chunks to bound memory.
        images = np.empty((len(train_idx), 256, 256, 3), dtype=np.uint8)
        for start in range(0, len(train_idx), 1024):
            images[start:start + 1024] = f["images"][train_idx[start:start + 1024]]
    return torch.from_numpy(images), torch.from_numpy(all_labels[train_idx]), train_idx


def preprocess(images_u8, resolution, augment):
    """uint8 NHWC 256x256 -> float NCHW in [-1, 1] at `resolution`, on the images' device."""
    x = images_u8.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    if augment:
        # Galaxy morphology is invariant to orientation: random dihedral transform per image.
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        k = torch.randint(0, 4, (x.shape[0],), device=x.device)
        x = torch.stack([torch.rot90(xi, int(ki), dims=(-2, -1)) for xi, ki in zip(x, k)])
    if x.shape[-1] != resolution:
        x = F.interpolate(x, size=(resolution, resolution), mode="bicubic", align_corners=False)
        x = x.clamp_(-1.0, 1.0)
    return x


# ----------------------------------------------------------------------- conditioning


class ClassEmbedding(nn.Module):
    """Learned per-class token sequence used in place of CLIP text embeddings.

    Index `num_classes` is the null class for classifier-free guidance.
    """

    def __init__(self, init_table):
        super().__init__()
        self.num_classes = init_table.shape[0] - 1
        self.table = nn.Parameter(init_table.clone().float())

    @property
    def null_class(self):
        return self.num_classes

    def forward(self, labels):
        return self.table[labels]


def class_prompt(name):
    return f"a telescope image of a galaxy, {name.lower()} morphology"


@torch.no_grad()
def build_class_embedding(model_id, cross_attention_dim, init, device):
    if init == "random":
        # Short sequence, scaled roughly like CLIP hidden states.
        return ClassEmbedding(torch.randn(len(CLASS_NAMES) + 1, 8, cross_attention_dim))

    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    prompts = [class_prompt(n) for n in CLASS_NAMES] + [""]  # last = null class, as in SD CFG
    tokens = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt").input_ids.to(device)
    hidden = text_encoder(tokens)[0].float().cpu()  # (num_classes + 1, 77, 768)
    del text_encoder
    torch.cuda.empty_cache()
    return ClassEmbedding(hidden)


# ------------------------------------------------------------------------ checkpoints


def save_checkpoint(output_dir, step, unet, class_embed, config):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lora_state = {k: v.detach().to("cpu", torch.float32).contiguous()
                  for k, v in get_peft_model_state_dict(unet).items()}
    save_file(lora_state, ckpt_dir / "lora_unet.safetensors")
    save_file({"table": class_embed.table.detach().cpu().contiguous()},
              ckpt_dir / "class_embeddings.safetensors")
    (ckpt_dir / "train_config.json").write_text(json.dumps({**config, "step": step}, indent=2))
    return ckpt_dir


# ----------------------------------------------------------------------------- main


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[args.mixed_precision]

    # --- data (train split only)
    train_images, train_labels, train_idx = load_train_split(args.splits)
    try:  # page-locked memory speeds up host->GPU copies but needs ~2.4 GB of free physical RAM
        train_images = train_images.pin_memory()
    except RuntimeError as e:
        print(f"warning: could not pin train images ({e}); using pageable memory", flush=True)

    # --- models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae", dtype=torch.float32).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet", dtype=weight_dtype).to(device)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()

    unet.add_adapter(LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=LORA_TARGET_MODULES,
    ))
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    for p in lora_params:  # keep trainable weights in fp32 even when the base U-Net is bf16/fp16
        p.data = p.data.float()
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet.train()

    class_embed = build_class_embedding(
        args.pretrained_model, unet.config.cross_attention_dim, args.class_embed_init, device
    ).to(device)

    optimizer = torch.optim.AdamW(
        [{"params": lora_params}, {"params": class_embed.parameters()}],
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
    )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=args.max_train_steps)
    # fp16 needs loss scaling; bf16 and fp32 do not.
    grad_scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision == "fp16")

    # --- config summary
    n_lora = sum(p.numel() for p in lora_params)
    n_embed = sum(p.numel() for p in class_embed.parameters())
    n_unet = sum(p.numel() for p in unet.parameters()) - n_lora
    effective_bs = args.train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_idx) / effective_bs)
    class_counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).tolist()
    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "lora_target_modules": LORA_TARGET_MODULES,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "null_class_index": class_embed.null_class,
        "class_embed_shape": list(class_embed.table.shape),
        "class_prompts": [class_prompt(n) for n in CLASS_NAMES] if args.class_embed_init == "clip" else None,
        "prediction_type": noise_scheduler.config.prediction_type,
        "vae_scaling_factor": vae.config.scaling_factor,
    }

    print("=" * 72)
    print("Galaxy10 class-conditional LoRA fine-tuning")
    print("=" * 72)
    print(f"  base model            {args.pretrained_model}  (U-Net params frozen: {n_unet / 1e6:.1f}M)")
    print(f"  LoRA                  rank {args.lora_rank}, alpha {args.lora_alpha:g}, on {LORA_TARGET_MODULES}")
    print(f"  trainable params      LoRA {n_lora / 1e6:.2f}M + class embeddings {n_embed / 1e6:.2f}M")
    print(f"  class embeddings      {tuple(class_embed.table.shape)} (init: {args.class_embed_init}), "
          f"null class index {class_embed.null_class}")
    print(f"  cond dropout (CFG)    {args.cond_dropout}")
    print(f"  data                  {args.splits.name}: train split only, {len(train_idx)} images")
    print(f"  per-class train count {dict(zip(range(len(CLASS_NAMES)), class_counts))}")
    print(f"  resolution            {args.resolution} (native 256, bicubic resize)"
          f"; augmentation {'off' if args.no_augment else 'random flip + rot90'}")
    print(f"  batch                 {args.train_batch_size} x {args.gradient_accumulation_steps} accum = {effective_bs}")
    print(f"  steps                 {args.max_train_steps} (~{args.max_train_steps / steps_per_epoch:.1f} epochs)")
    print(f"  optimizer             AdamW lr {args.learning_rate:g}, wd {args.adam_weight_decay:g}, "
          f"{args.lr_scheduler} ({args.lr_warmup_steps} warmup)")
    print(f"  precision             {args.mixed_precision}; gradient checkpointing {args.gradient_checkpointing}")
    print(f"  prediction type       {noise_scheduler.config.prediction_type}")
    print(f"  logging / checkpoints every {args.log_every} / {args.checkpoint_every} steps -> {args.output_dir}")
    print(f"  seed                  {args.seed}")
    print("=" * 72, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_config.json").write_text(json.dumps(config, indent=2))
    log_file = open(args.output_dir / "train_log.jsonl", "a")

    # --- training loop
    trainable = lora_params + list(class_embed.parameters())
    perm, perm_pos = torch.randperm(len(train_idx)), 0
    step, running_loss, running_n = 0, 0.0, 0
    t0 = time.time()

    while step < args.max_train_steps:
        for _ in range(args.gradient_accumulation_steps):
            if perm_pos + args.train_batch_size > len(perm):
                perm, perm_pos = torch.randperm(len(train_idx)), 0
            batch_idx = perm[perm_pos:perm_pos + args.train_batch_size]
            perm_pos += args.train_batch_size

            pixels = preprocess(train_images[batch_idx].to(device, non_blocking=True),
                                args.resolution, augment=not args.no_augment)
            labels = train_labels[batch_idx].to(device)

            with torch.no_grad():
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor

            # Classifier-free guidance: replace the class with the null class w.p. cond_dropout.
            drop = torch.rand(labels.shape[0], device=device) < args.cond_dropout
            labels = torch.where(drop, torch.full_like(labels, class_embed.null_class), labels)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            with torch.autocast("cuda", dtype=weight_dtype, enabled=args.mixed_precision != "no"):
                cond = class_embed(labels)
                pred = unet(noisy_latents.to(weight_dtype), timesteps, encoder_hidden_states=cond).sample
            loss = F.mse_loss(pred.float(), target.float())
            grad_scaler.scale(loss / args.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            running_n += 1

        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0 or step == args.max_train_steps:
            avg = running_loss / running_n
            elapsed = time.time() - t0
            rec = {"step": step, "loss": avg, "lr": lr_scheduler.get_last_lr()[0],
                   "grad_norm": float(grad_norm), "epoch": step / steps_per_epoch,
                   "elapsed_s": round(elapsed, 3),
                   "peak_mem_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            eta = elapsed / step * (args.max_train_steps - step)
            print(f"step {step:>7}/{args.max_train_steps}  loss {avg:.5f}  lr {rec['lr']:.2e}  "
                  f"grad {rec['grad_norm']:.3f}  epoch {rec['epoch']:.2f}  "
                  f"{step / elapsed:.2f} it/s  eta {eta / 60:.1f} min  "
                  f"peak mem {rec['peak_mem_alloc_gb']:.1f}/{rec['peak_mem_reserved_gb']:.1f} GB "
                  f"(alloc/reserved)", flush=True)
            running_loss, running_n = 0.0, 0

        if step % args.checkpoint_every == 0 or step == args.max_train_steps:
            ckpt = save_checkpoint(args.output_dir, step, unet, class_embed, config)
            print(f"  saved LoRA checkpoint -> {ckpt}", flush=True)

    log_file.close()
    print(f"Done: {step} steps in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

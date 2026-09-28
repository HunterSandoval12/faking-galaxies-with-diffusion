"""Code shared by Tutorials 5 and 6 (training): precision/memory checks, data readers, the classifier and LoRA
training loops and the generator. Each function appends cells to a notebook (nbcommon.Notebook)."""

from nbcommon import SD_REPO

PAPER_TIMES = {  # measured on the RTX 5090 (results/tables/t02_hyperparameters)
    "lora": "60 min for 10,000 steps at batch 8 (0.36 s/step, bf16)",
    "generation": "99.5 min for the 12,330-image synthetic set (0.48 s/image in batches of 20)",
    "classifier": "1.5 min per run (30 epochs on 12,330 images, bf16, incl. per-epoch validation)",
}


def precision_cells(nb):
    nb.md("## GPU, precision and memory")
    nb.md("""
Training runs in **mixed precision**: the heavy matrix products use 16-bit numbers, the trained weights stay in 32 bits.
There are two 16-bit formats. **bf16** has the same range as 32-bit floats and needs no extra care, but only GPUs with
compute capability 8.0 or newer (Ampere and later, e.g. the A100, L4 or the project's RTX 5090) have bf16 hardware.
Colab's free **T4** is a Turing GPU (compute capability 7.5): it has fast **fp16** tensor cores but no bf16 hardware,
so bf16 would be emulated and slow. `torch.cuda.is_bf16_supported()` can still report `True` on a T4, because the
emulation *works*; the cell therefore checks the compute capability itself.

With fp16, small gradients can underflow to zero, so fp16 training uses a **gradient scaler**
(`torch.amp.GradScaler`): the loss is multiplied by a large factor before backpropagation and the gradients are divided
by it again before the optimizer step; a step whose gradients overflow is skipped and the factor lowered. The project's
own runs used bf16 on the RTX 5090, so numbers from fp16 runs differ slightly.

`PRECISION = "auto"` picks bf16 when the GPU has it and fp16 otherwise; `"fp16"`, `"bf16"` or `"fp32"` force one.
After each training stage the notebook reports the peak GPU memory, to show that it fits the T4's 15 GB.
""")
    nb.code("""
import numpy as np
import torch

PRECISION = "auto"  # "auto" | "bf16" | "fp16" | "fp32"

assert torch.cuda.is_available(), "This tutorial needs a GPU: in Colab choose Runtime > Change runtime type > T4 GPU"
DEVICE = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
NATIVE_BF16 = props.major >= 8  # bf16 tensor cores: Ampere (8.x) and newer
AMP_DTYPE = {"auto": torch.bfloat16 if NATIVE_BF16 else torch.float16, "bf16": torch.bfloat16,
             "fp16": torch.float16, "fp32": torch.float32}[PRECISION]
USE_GRAD_SCALER = AMP_DTYPE == torch.float16  # only fp16 needs loss scaling
GPU_GIB = props.total_memory / 2**30
MEMORY = {}  # peak GPU memory per stage (GiB)


def memory_report(stage):
    \"\"\"Record and print the peak GPU memory since the last report.\"\"\"
    peak = torch.cuda.max_memory_allocated() / 2**30
    MEMORY[stage] = peak
    print(f"[{stage}] peak GPU memory {peak:.1f} GiB of {GPU_GIB:.1f} GiB")
    torch.cuda.reset_peak_memory_stats()


print(f"GPU: {props.name}, compute capability {props.major}.{props.minor}, {GPU_GIB:.1f} GiB")
print(f"native bf16 hardware: {NATIVE_BF16}; torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}")
print(f"training precision: {str(AMP_DTYPE).replace('torch.', '')}; gradient scaler: {USE_GRAD_SCALER}")
""")


def dataset_cells(nb):
    """Readers for Tutorial 1's zip (training/validation/testing directories) and other image zips."""
    nb.code("""
import io
import zipfile

from PIL import Image

CLASS_NAMES = ["Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar-Shaped Smooth",
               "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge",
               "Edge-on with Bulge"]
assert ZIP_PATH.exists(), "Run Tutorial 1 (Dataset Tutorial) first: it saves galaxy10_dataset.zip on your Drive."
dataset_zip = zipfile.ZipFile(ZIP_PATH)


def zip_members(archive, folder):
    \"\"\"[(member, label)] of <root>/<folder>/<label>_<class>/<file>.png in a zip, sorted by class and file name.\"\"\"
    out = []
    for n in sorted(archive.namelist()):
        parts = n.split("/")
        if n.endswith(".png") and len(parts) == 4 and parts[1] == folder:
            out.append((n, int(parts[2].split("_")[0])))
    return out


def stratified(items, fraction, seed=0):
    \"\"\"Fixed-seed stratified subset: the same fraction of every class (at least 1 image per class).\"\"\"
    if fraction >= 1:
        return items
    rng = np.random.default_rng(seed)
    labels = np.array([c for _, c in items])
    keep = []
    for c in range(len(CLASS_NAMES)):
        idx = np.flatnonzero(labels == c)
        keep += rng.choice(idx, max(1, round(fraction * len(idx))), replace=False).tolist()
    return [items[i] for i in sorted(keep)]


def load_images(archive, items):
    \"\"\"uint8 images [N, 256, 256, 3] and int64 labels [N] as (CPU) torch tensors.\"\"\"
    x = np.stack([np.asarray(Image.open(io.BytesIO(archive.read(m))).convert("RGB")) for m, _ in items])
    return torch.from_numpy(x), torch.tensor([c for _, c in items], dtype=torch.int64)
""")


def classifier_code_cell(nb):
    nb.code("""
import copy
import math

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def to_input(images_u8, augment):
    \"\"\"uint8 NHWC on the GPU -> normalised float NCHW; optional random flip + 90-degree rotation (as in the project).\"\"\"
    x = images_u8.permute(0, 3, 1, 2).float().div_(255.0)
    if augment:
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        k = torch.randint(0, 4, (x.shape[0],), device=x.device)
        x = torch.stack([torch.rot90(xi, int(ki), dims=(-2, -1)) for xi, ki in zip(x, k)])
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def build_classifier(pretrained=True):
    \"\"\"torchvision ResNet-18 (ImageNet weights, downloaded on first use) with a new 10-class output layer.\"\"\"
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model.to(DEVICE).to(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate(model, x_u8, y, batch_size=256):
    \"\"\"Mean cross-entropy loss and accuracy of a model on uint8 images.\"\"\"
    model.eval()
    loss, correct = 0.0, 0
    for s in range(0, len(x_u8), batch_size):
        xb = to_input(x_u8[s:s + batch_size].to(DEVICE), augment=False).contiguous(memory_format=torch.channels_last)
        yb = y[s:s + batch_size].to(DEVICE)
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP_DTYPE != torch.float32):
            logits = model(xb).float()
        loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return loss / len(x_u8), correct / len(x_u8)


def train_classifier(train_x, train_y, eval_sets, epochs, seed, augment=True, learning_rate=1e-3,
                     weight_decay=0.05, batch_size=128, warmup_epochs=1.0, verbose=True):
    \"\"\"The project's classifier training (classifier/train_classifier.py): AdamW, linear warm-up for one epoch then
    cosine decay to 0 (per step), cross-entropy, mixed precision. After every epoch it evaluates every set in
    eval_sets ({name: (images, labels)}); the returned model is the epoch with the best VALIDATION accuracy.
    Returns (model, per-epoch history, per-step training losses, selected epoch).\"\"\"
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_x) / batch_size)
    total, warmup = epochs * steps_per_epoch, int(warmup_epochs * steps_per_epoch)

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER)
    history, step_losses, best_acc, best_state, best_epoch = [], [], -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_x))
        loss_sum, correct = 0.0, 0
        for s in range(0, len(perm), batch_size):
            bi = perm[s:s + batch_size]
            x = to_input(train_x[bi].to(DEVICE, non_blocking=True), augment).contiguous(memory_format=torch.channels_last)
            y = train_y[bi].to(DEVICE)
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP_DTYPE != torch.float32):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step_losses.append(loss.item())
            loss_sum += loss.item() * len(bi)
            correct += (logits.argmax(1) == y).sum().item()
        rec = {"epoch": epoch, "train_loss": loss_sum / len(train_x), "train_accuracy": correct / len(train_x),
               "lr": scheduler.get_last_lr()[0]}
        for name, (ex, ey) in eval_sets.items():
            rec[f"{name}_loss"], rec[f"{name}_accuracy"] = evaluate(model, ex, ey)
        rec["elapsed_s"] = time.time() - t0
        history.append(rec)
        if rec["validation_accuracy"] > best_acc:  # model selection on VALIDATION only
            best_acc, best_epoch = rec["validation_accuracy"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {epoch:>2}/{epochs}  train loss {rec['train_loss']:.3f} acc {rec['train_accuracy']:.3f}  "
                  + "  ".join(f"{n} loss {rec[f'{n}_loss']:.3f} acc {rec[f'{n}_accuracy']:.3f}" for n in eval_sets)
                  + f"  ({rec['elapsed_s']:.0f} s)")
    model.load_state_dict(best_state)
    return model, history, step_losses, best_epoch
""")


def lora_code_cell(nb):
    nb.code(f"""
import json

from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

SD_REPO = "{SD_REPO}"
LORA_CONFIG = json.loads((WEIGHTS_DIR / "lora_config.json").read_text())  # the released run's settings


def load_base_generator():
    \"\"\"Stable Diffusion 1.5's VAE and U-Net (16-bit files from Hugging Face, frozen) and its training noise schedule.\"\"\"
    vae = AutoencoderKL.from_pretrained(SD_REPO, subfolder="vae", variant="fp16", torch_dtype=AMP_DTYPE).to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(SD_REPO, subfolder="unet", variant="fp16",
                                                torch_dtype=AMP_DTYPE).to(DEVICE)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    return vae, unet, DDPMScheduler.from_pretrained(SD_REPO, subfolder="scheduler")


def add_lora(unet, cfg=LORA_CONFIG):
    \"\"\"Add LoRA adapters (rank 8 on to_q/to_k/to_v/to_out.0); their weights are kept in fp32.\"\"\"
    unet.add_adapter(LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], init_lora_weights="gaussian",
                                target_modules=cfg["lora_target_modules"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    return params


@torch.no_grad()
def clip_class_table(cfg=LORA_CONFIG):
    \"\"\"Initial class-embedding table: CLIP's encoding of one prompt per class + the empty prompt (the null class).\"\"\"
    tokenizer = CLIPTokenizer.from_pretrained(SD_REPO, subfolder="tokenizer")
    encoder = CLIPTextModel.from_pretrained(SD_REPO, subfolder="text_encoder", variant="fp16",
                                            torch_dtype=torch.float32).to(DEVICE)
    tokens = tokenizer(cfg["class_prompts"] + [""], padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    table = encoder(tokens)[0].float()
    del encoder
    torch.cuda.empty_cache()
    return table  # [11, 77, 768]


def to_vae_input(images_u8, augment, resolution=512):
    \"\"\"uint8 NHWC 256 px -> float NCHW in [-1, 1] at 512 px (preprocess in diffusion/train_lora.py).\"\"\"
    x = images_u8.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    if augment:
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        k = torch.randint(0, 4, (x.shape[0],), device=x.device)
        x = torch.stack([torch.rot90(xi, int(ki), dims=(-2, -1)) for xi, ki in zip(x, k)])
    return F.interpolate(x, size=(resolution, resolution), mode="bicubic", align_corners=False).clamp_(-1.0, 1.0)


def fixed_eval_batch(vae, noise_scheduler, x_u8, y, seed=0):
    \"\"\"A fixed evaluation batch: latents of held-out images with fixed noise and fixed noise levels, so the
    diffusion loss on it changes only when the model changes (the training loss is noisy by design).\"\"\"
    g = torch.Generator(DEVICE).manual_seed(seed)
    with torch.no_grad():
        latents = vae.encode(to_vae_input(x_u8.to(DEVICE), False).to(AMP_DTYPE)).latent_dist.mean.float() \\
            * vae.config.scaling_factor
    noise = torch.randn(latents.shape, generator=g, device=DEVICE)
    t = torch.linspace(50, 950, len(latents), device=DEVICE).long()
    return {{"noisy": noise_scheduler.add_noise(latents, noise, t), "noise": noise, "t": t, "y": y.to(DEVICE)}}


@torch.no_grad()
def eval_diffusion_loss(unet, table, batch):
    with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP_DTYPE != torch.float32):
        pred = unet(batch["noisy"].to(AMP_DTYPE), batch["t"],
                    encoder_hidden_states=table[batch["y"]].to(AMP_DTYPE)).sample
    return F.mse_loss(pred.float(), batch["noise"]).item()


def train_lora(unet, vae, noise_scheduler, lora_params, table, train_x, train_y, steps, batch_size, eval_batch,
               eval_every, learning_rate=1e-4, weight_decay=0.01, warmup_steps=500, cond_dropout=0.1,
               max_grad_norm=1.0, seed=42):
    \"\"\"The project's LoRA training loop (diffusion/train_lora.py): AdamW on the LoRA weights + class table, constant
    learning rate after a linear warm-up, classifier-free-guidance dropout, MSE on the predicted noise, gradient
    clipping. Returns the per-step training losses and the fixed-batch evaluation losses [(step, loss)].\"\"\"
    torch.manual_seed(seed)
    table.requires_grad_(True)
    optimizer = torch.optim.AdamW([{{"params": lora_params}}, {{"params": [table]}}], lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = get_scheduler("constant_with_warmup", optimizer=optimizer, num_warmup_steps=warmup_steps,
                              num_training_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER)
    null = table.shape[0] - 1
    step_losses, eval_losses = [], [(0, eval_diffusion_loss(unet, table, eval_batch))]
    unet.train()
    perm, pos, t0 = torch.randperm(len(train_x)), 0, time.time()
    for step in range(1, steps + 1):
        if pos + batch_size > len(perm):
            perm, pos = torch.randperm(len(train_x)), 0
        bi = perm[pos:pos + batch_size]
        pos += batch_size
        with torch.no_grad():
            pixels = to_vae_input(train_x[bi].to(DEVICE), augment=True).to(AMP_DTYPE)
            latents = vae.encode(pixels).latent_dist.sample().float() * vae.config.scaling_factor
        labels = train_y[bi].to(DEVICE)
        drop = torch.rand(len(labels), device=DEVICE) < cond_dropout  # classifier-free guidance training
        labels = torch.where(drop, torch.full_like(labels, null), labels)
        noise = torch.randn_like(latents)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (len(latents),), device=DEVICE)
        noisy = noise_scheduler.add_noise(latents, noise, t)
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP_DTYPE != torch.float32):
            pred = unet(noisy.to(AMP_DTYPE), t, encoder_hidden_states=table[labels]).sample
        loss = F.mse_loss(pred.float(), noise)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params + [table], max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step_losses.append(loss.item())
        if step % eval_every == 0 or step == steps:
            unet.eval()
            eval_losses.append((step, eval_diffusion_loss(unet, table, eval_batch)))
            unet.train()
            print(f"  step {{step:>5}}/{{steps}}  training loss (mean of the last {{eval_every}}) "
                  f"{{np.mean(step_losses[-eval_every:]):.4f}}  fixed-batch loss {{eval_losses[-1][1]:.4f}}  "
                  f"({{time.time() - t0:.0f}} s)")
    unet.eval()
    table.requires_grad_(False)
    return step_losses, eval_losses


def load_released_lora(unet):
    \"\"\"Put the released LoRA weights into the U-Net's adapters; returns the released class table.\"\"\"
    result = set_peft_model_state_dict(unet, load_file(WEIGHTS_DIR / "lora_unet.safetensors"))
    assert not result.unexpected_keys and not [k for k in result.missing_keys if "lora" in k]
    return load_file(WEIGHTS_DIR / "class_embeddings.safetensors")["table"].to(DEVICE)


@torch.no_grad()
def generate(unet, vae, table, labels, seeds, guidance=3.0, steps=30, resolution=512, batch_size=10):
    \"\"\"Class-conditional images (Tutorial 4's sampling loop, batched): classifier-free guidance with the null class,
    30 DPM-Solver++ steps, VAE decode, Lanczos down to 256 px. One noise generator per image, as in the project.\"\"\"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(SD_REPO, subfolder="scheduler")
    null, images = table.shape[0] - 1, []
    for s in range(0, len(labels), batch_size):
        lab = torch.as_tensor(labels[s:s + batch_size], device=DEVICE)
        latents = torch.cat([torch.randn((1, 4, resolution // 8, resolution // 8), device=DEVICE, dtype=AMP_DTYPE,
                                         generator=torch.Generator(DEVICE).manual_seed(int(sd)))
                             for sd in seeds[s:s + batch_size]])
        scheduler.set_timesteps(steps, device=DEVICE)
        latents = latents * scheduler.init_noise_sigma
        cond = torch.cat([table[torch.full_like(lab, null)], table[lab]]).to(AMP_DTYPE)
        for t in scheduler.timesteps:
            inp = scheduler.scale_model_input(torch.cat([latents] * 2), t)
            eps_null, eps_class = unet(inp, t, encoder_hidden_states=cond).sample.chunk(2)
            latents = scheduler.step(eps_null + guidance * (eps_class - eps_null), t, latents).prev_sample
        x = vae.decode(latents / vae.config.scaling_factor).sample
        x = ((x.float() / 2 + 0.5).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        images += [np.asarray(Image.fromarray(im).resize((256, 256), Image.LANCZOS)) for im in x]
    return images
""")

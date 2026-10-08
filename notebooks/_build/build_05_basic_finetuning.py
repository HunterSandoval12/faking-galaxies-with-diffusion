"""Builds notebooks/05_Basic_Finetuning_Tutorial.ipynb (run from the repository root).

Spec item #5 (Basic fine-tuning): download a pretrained model, set up the optimization loop, show that the loss
decreases, retrain for a few iterations. Both of the project's models are fine-tuned: the ImageNet ResNet-18 into the
galaxy classifier, and Stable Diffusion 1.5 (+ LoRA + class table) into the galaxy generator. FAST mode (default)
keeps it short on a Colab T4; FAST = False runs the paper's full settings. Final results use the released weights.

Usage: python notebooks/_build/build_05_basic_finetuning.py [--times times.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbcommon import REPO_URL, SD_REPO, Notebook, download_cells, drive_cells, install_cells  # noqa: E402
from nbtraining import (PAPER_TIMES, classifier_code_cell, dataset_cells, lora_code_cell,  # noqa: E402
                        precision_cells)

ap = argparse.ArgumentParser()
ap.add_argument("--times", type=Path, help="JSON from a local test run")
args = ap.parse_args()
KEYS = ("__T_TOTAL__", "__T_CLS__", "__T_LORA__", "__T_GEN__", "__T_LOAD__", "__T4_TOTAL__", "__MEM_CLS__",
        "__MEM_LORA__", "__FAST_ACC__", "__LORA_EVAL__")
FILL = {k: "?" for k in KEYS}
if args.times:
    FILL.update(json.loads(args.times.read_text())["fill"])

nb = Notebook()
md, code = nb.md, nb.code

md("# Galaxy10 DECaLS: Basic Fine-tuning Tutorial")
md(f"""
**Project:** *Faking Galaxies with Diffusion: A Real-vs-Synthetic Benchmark for Morphology Classification*
Hunter Sandoval, University of New Mexico. Tutorial 5 of 6.

**Scope.** Both of the project's models are *fine-tuned* versions of pretrained models. This tutorial downloads the
pretrained models, sets up the project's optimization loops, and retrains them for a short while to show the loss going
down:

1. **Classifier**: torchvision's ImageNet-pretrained ResNet-18, fine-tuned into a 10-class galaxy classifier;
2. **Generator**: Stable Diffusion 1.5 from Hugging Face, fine-tuned with LoRA adapters and a learned class-embedding
   table into a class-conditional galaxy generator (a few galaxies are generated before and after).

**FAST mode (on by default)** shrinks the runs so the whole notebook finishes in a few minutes on a free Colab T4: a
fixed-seed 10% stratified subset of the training images, 5 classifier epochs, 200 LoRA steps at batch 4, and 4
generated images. `FAST = False` runs the paper's full settings (table below). The **final results cells** load the
project's released weights, i.e. the models trained with the full settings, so you can compare.
""")
md("## References")
md(f"""
1. K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016, arXiv:1512.03385; torchvision's ImageNet
   weights: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
2. R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022, arXiv:2112.10752;
   weights: https://huggingface.co/{SD_REPO}
3. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022, arXiv:2106.09685.
4. J. Ho, A. Jain, P. Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020, arXiv:2006.11239.
5. J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," arXiv:2207.12598, 2022.
6. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019, arXiv:1711.05101 (AdamW).
7. P. Micikevicius et al., "Mixed Precision Training," ICLR 2018, arXiv:1710.03740 (fp16 with loss scaling).
8. Hugging Face diffusers LoRA training example (the structure the project's `diffusion/train_lora.py` follows):
   https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
9. Project code and model weights: {REPO_URL}
""")

md("# Setup")
install_cells(nb, {"torch": "2.0", "torchvision": "0.15", "diffusers": "0.27", "peft": "0.10", "transformers": "4.30",
                   "safetensors": "0.4", "numpy": "1.23", "pillow": "9.0", "matplotlib": "3.6"}, """
| Python | 3.10 | 3.14.7 | (Colab's default) | |
| PyTorch (`torch`) | 2.0 | 2.14.0 | `pip install "torch>=2.0"` | training |
| torchvision | 0.15 | 0.29.0 | `pip install "torchvision>=0.15"` | ResNet-18 and its ImageNet weights |
| diffusers | 0.27 | 0.40.0 | `pip install "diffusers>=0.27"` | Stable Diffusion's VAE, U-Net, schedulers |
| peft | 0.10 | 0.21.0 | `pip install "peft>=0.10"` | LoRA adapters |
| transformers | 4.30 | 5.17.0 | `pip install "transformers>=4.30"` | the CLIP text encoder (initial class table) |
| safetensors | 0.4 | 0.8.0 | `pip install "safetensors>=0.4"` | reading the released weights |
| NumPy | 1.23 | 2.5.2 | `pip install "numpy>=1.23"` | arrays |
| Pillow | 9.0 | 12.3.0 | `pip install "pillow>=9.0"` | images |
| Matplotlib | 3.6 | 3.11.2 | `pip install "matplotlib>=3.6"` | figures |
""")
md("## Hardware used for running")
md("""
- **GPU required**: in Colab choose *Runtime > Change runtime type > T4 GPU* (the notebook asks for it when opened from
  GitHub). The free T4 has 15 GB of memory and no bf16 hardware, so the notebook trains in fp16 there (next section).
- **Peak GPU memory (FAST mode, fp16):** __MEM_CLS__ for the classifier and __MEM_LORA__ for the LoRA fine-tuning,
  measured with the GPU limited to 15 GB like a T4. Full settings (`FAST = False`): the LoRA training at batch 8
  needs 10.6 GB in fp16 (measured the same way), so it fits a T4 too.
- **Tested on:** a Windows 11 PC with an NVIDIA RTX 5090 (32 GB), AMD Ryzen 7 9800X3D, 32 GB RAM, Python 3.14. Every
  code cell was run there in fp16 with GPU memory capped at 15 GB (to mimic the T4) and in the GPU's native bf16.
""")
md("## Approximate execution times")
md(f"""
| Step | Test PC (FAST, fp16) | Colab T4 (FAST, estimate) | Full settings on the RTX 5090 (the paper's runs) |
|---|---|---|---|
| Load the images from Tutorial 1's zip | __T_LOAD__ | 1-2 min | - |
| Classifier fine-tuning | __T_CLS__ | ~1-2 min | {PAPER_TIMES["classifier"]} |
| LoRA fine-tuning (incl. model download) | __T_LORA__ | ~5-8 min | {PAPER_TIMES["lora"]} |
| Generating 12 images (3 x 4) | __T_GEN__ | ~2 min | - |
| **Whole tutorial** | **__T_TOTAL__** | **__T4_TOTAL__** | - |

The Colab column is an estimate: for these models a T4 is roughly 8-10 times slower than the RTX 5090. With the full
settings (`FAST = False`) a T4 would need about 15 min for the classifier and 8-10 hours for the LoRA fine-tuning.
""")
md("## Expected results")
md("""
- **Classifier**: before fine-tuning (new random output layer) the validation loss is about 2.3 = ln 10 and the
  accuracy ~10%. After 5 FAST epochs on 10% of the data the training loss has dropped steadily and the validation
  accuracy is about __FAST_ACC__. The released classifier (30 epochs, all data) reaches 86.7% validation accuracy.
- **Generator**: the per-step diffusion loss is noisy and nearly flat (by design, see Tutorial 3); the loss on a fixed
  evaluation batch decreases slightly (__LORA_EVAL__). The generated galaxies change from Stable Diffusion's idea of a
  "telescope image of a galaxy" toward the survey's look; the released generator (10,000 steps) produces the images of
  the paper's synthetic set.
""")

drive_cells(nb, "The fine-tuning uses the training and validation images from Tutorial 1.")
dataset_cells(nb)

md("# FAST mode and the full settings")
md(f"""
| Setting | FAST (default) | Full (paper) |
|---|---|---|
| Classifier training images | 10% stratified subset (1,234; fixed seed) | all 12,330 |
| Classifier epochs | 5 | 30 |
| Classifier batch / learning rate | 128 / 1e-3 (1 warm-up epoch + cosine) | 128 / 1e-3 (1 warm-up epoch + cosine) |
| Validation images | all 2,607 | all 2,607 |
| LoRA training images | the same 10% subset | all 12,330 |
| LoRA steps / batch | 200 / 4 | 10,000 / 8 |
| LoRA warm-up steps | 20 | 500 |
| Generated images | 4 classes x (before, after, released) | - |
| Time on the RTX 5090 | minutes | classifier {PAPER_TIMES["classifier"]}; LoRA {PAPER_TIMES["lora"]} |
""")
code("""
FAST = True  # False: the paper's full settings (hours on a T4)

if FAST:
    TRAIN_FRACTION, CLS_EPOCHS = 0.1, 5
    LORA_STEPS, LORA_BATCH, LORA_WARMUP, LORA_EVAL_EVERY = 200, 4, 20, 25
else:
    TRAIN_FRACTION, CLS_EPOCHS = 1.0, 30
    LORA_STEPS, LORA_BATCH, LORA_WARMUP, LORA_EVAL_EVERY = 10000, 8, 500, 500
GEN_CLASSES = [2, 4, 5, 9]  # Round Smooth, Cigar-Shaped Smooth, Barred Spiral, Edge-on with Bulge
GEN_SEEDS = [7_000_000_000 + c * 1_000_000 for c in GEN_CLASSES]  # the paper's seeds for sample 0 of each class
print(f"FAST = {FAST}: {TRAIN_FRACTION:.0%} of the training images, {CLS_EPOCHS} classifier epochs, "
      f"{LORA_STEPS} LoRA steps at batch {LORA_BATCH}")
""")
md("# Download the released weights (for the final results)")
download_cells(nb, ["resnet18_real_only_seed0.safetensors", "lora_unet.safetensors", "class_embeddings.safetensors",
                    "lora_config.json"],
               intro="The fine-tuning below starts from the *public pretrained* models (ImageNet ResNet-18, Stable "
                     "Diffusion 1.5). The project's released weights, trained with the full settings, are downloaded "
                     "here only for the final comparison cells.")
precision_cells(nb)

md("# Part 1: Fine-tuning the classifier")
md("## Load the training and validation images")
md("""
The training images are a fixed-seed **stratified** subset: the same fraction of every class, so the class imbalance
of the full set (e.g. only 233 Cigar-Shaped Smooth images) is kept. All validation images are used for evaluation.
""")
code("""
t = time.time()
train_items = stratified(zip_members(dataset_zip, "training"), TRAIN_FRACTION, seed=0)
train_x, train_y = load_images(dataset_zip, train_items)
val_x, val_y = load_images(dataset_zip, zip_members(dataset_zip, "validation"))
TIMES["load images"] = time.time() - t
print(f"training: {len(train_x)} images, per class {np.bincount(train_y.numpy(), minlength=10).tolist()}")
print(f"validation: {len(val_x)} images; loaded in {TIMES['load images']:.0f} s")
""")
md("## The pretrained model and the optimization loop")
md("""
**Download the pretrained model.** `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` downloads torchvision's
ImageNet-trained weights (45 MB, from PyTorch's servers). Its 1,000-class output layer is replaced by a new, randomly
initialized 10-class layer; everything else starts from the ImageNet features.

**The optimization loop** is the project's `classifier/train_classifier.py`, reproduced below: every step takes a
batch of 128 images, applies random flips and 90-degree rotations, runs the network in mixed precision, computes the
cross-entropy loss, backpropagates (through the gradient scaler when training in fp16), takes an AdamW step (learning
rate 1e-3, weight decay 0.05) and updates the learning rate (one warm-up epoch, then cosine decay). After every epoch
it measures the validation loss and accuracy and remembers the best epoch (Tutorial 3 explains each choice).
""")
classifier_code_cell(nb)
md("## Before fine-tuning")
code("""
torch.manual_seed(0)  # the new output layer starts random; a fixed seed makes this reproducible
model_before = build_classifier()  # downloads the ImageNet weights on first use
loss0, acc0 = evaluate(model_before, val_x, val_y)
print(f"ImageNet ResNet-18 with a new 10-class layer, before any training: validation loss {loss0:.3f} "
      f"(ln 10 = {np.log(10):.3f}), accuracy {acc0:.3f}")
del model_before
""")
md("## Fine-tune")
code("""
t = time.time()
torch.cuda.reset_peak_memory_stats()
classifier, history, step_losses, best_epoch = train_classifier(train_x, train_y, {"validation": (val_x, val_y)},
                                                                 epochs=CLS_EPOCHS, seed=0)
TIMES["classifier fine-tuning"] = time.time() - t
memory_report("classifier fine-tuning")
print(f"kept epoch {best_epoch}; {len(step_losses)} optimization steps in {TIMES['classifier fine-tuning']:.0f} s")
""")
md("## The loss decreases")
md("""
Left: the training loss of every optimization step (gray) and its running mean over 10 steps. Right: the loss and
accuracy after each epoch on the training batches (augmented) and on the validation set.

In FAST mode an epoch is only 10 steps, so the **validation** loss can jump up in the first epochs before it falls: the
network is evaluated in eval mode, which uses batch-norm *running* statistics, and these lag behind weights that change
quickly while the learning rate is at its peak. Once the learning rate decays the two agree again. With the full
settings (97 steps per epoch) this effect is small (Tutorial 3 shows the full runs' curves).
""")
code("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
steps = np.arange(1, len(step_losses) + 1)
axes[0].plot(steps, step_losses, color="#c3c2b7", lw=1, label="each step")
run = np.convolve(step_losses, np.ones(10) / 10, mode="valid")
axes[0].plot(np.arange(10, len(step_losses) + 1), run, color="#2a78d6", lw=2, label="mean of 10 steps")
axes[0].axhline(np.log(10), color="gray", ls=":", lw=1)
axes[0].text(len(step_losses), np.log(10), "ln 10 (random guess)", ha="right", va="bottom", fontsize=8, color="gray")
axes[0].set_xlabel("optimization step")
axes[0].set_ylabel("cross-entropy loss")
axes[0].legend(fontsize=8)
ep = [0] + [h["epoch"] for h in history]
axes[1].plot(ep, [loss0] + [h["validation_loss"] for h in history], "o-", color="#2a78d6", label="validation")
axes[1].plot(ep[1:], [h["train_loss"] for h in history], "s:", color="#2a78d6", label="training (during the epoch)")
axes[1].set_ylabel("cross-entropy loss")
axes[2].plot(ep, [100 * acc0] + [100 * h["validation_accuracy"] for h in history], "o-", color="#2a78d6",
             label="validation")
axes[2].plot(ep[1:], [100 * h["train_accuracy"] for h in history], "s:", color="#2a78d6", label="training (during the epoch)")
axes[2].set_ylabel("accuracy (%)")
for ax in axes[1:]:
    ax.set_xticks(ep)
    ax.set_xlabel("epoch (0 = before fine-tuning)")
    ax.legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("Classifier fine-tuning: the loss decreases", x=0.01, ha="left")
plt.tight_layout()
plt.show()
print(f"first 10 steps: mean loss {np.mean(step_losses[:10]):.3f}; last 10 steps: {np.mean(step_losses[-10:]):.3f}")
""")
md("## Final result: the released classifier")
md("""
The same loop with the full settings (all 12,330 images, 30 epochs) produced the released classifier. Loading it and
scoring the same validation images shows where the fine-tuning ends up.
""")
code("""
from safetensors.torch import load_file

released = build_classifier(pretrained=False)
released.load_state_dict(load_file(WEIGHTS_DIR / "resnet18_real_only_seed0.safetensors"))
fast_loss, fast_acc = evaluate(classifier, val_x, val_y)
rel_loss, rel_acc = evaluate(released, val_x, val_y)
print(f"{'':<34}{'validation loss':>16}{'validation accuracy':>21}")
print(f"{'before fine-tuning':<34}{loss0:>16.3f}{acc0:>21.3f}")
print(f"{'this run (' + ('FAST' if FAST else 'full') + ')':<34}{fast_loss:>16.3f}{fast_acc:>21.3f}")
print(f"{'released (full settings, paper)':<34}{rel_loss:>16.3f}{rel_acc:>21.3f}")
del released
""")

md("# Part 2: Fine-tuning the generator (LoRA)")
md("## The pretrained model")
md(f"""
The generator starts from **Stable Diffusion 1.5** (`{SD_REPO}` on Hugging Face; the VAE and U-Net in 16-bit, 1.9 GB,
downloaded on first use). Its 860-million-parameter U-Net stays frozen. What is trained:

- **LoRA adapters** (rank 8) on the attention projections `to_q`, `to_k`, `to_v`, `to_out.0` of all 16 transformer
  blocks: 1,594,368 values, initialized so that the adapters start as a no-op (their `B` matrices are zero);
- a **class-embedding table** (11 x 77 x 768) that replaces the text prompt: it starts as CLIP's encoding of prompts
  such as "a telescope image of a galaxy, barred spiral morphology", plus the empty prompt as the null class (the CLIP
  text encoder is downloaded once for this and then discarded).

**The optimization loop** is the project's `diffusion/train_lora.py`: encode a batch of training images (flipped and
rotated at random, upsampled to 512 px) into latents with the VAE; replace the class by the null class with
probability 0.1 (so the model also learns unconditional predictions, needed for classifier-free guidance); add noise
at a random level; predict the noise; mean squared error; backpropagate through the frozen U-Net into the LoRA weights
and the table; clip the gradient norm to 1; AdamW step (learning rate 1e-4, weight decay 0.01, linear warm-up then
constant).

Because the training loss mostly measures *which noise level was drawn* (Tutorial 3), it is noisy and nearly flat.
To see the model change, the loop also computes the loss on a **fixed evaluation batch**: one validation image per
class, with fixed noise and fixed noise levels from low to high.
""")
lora_code_cell(nb)
code("""
t = time.time()
torch.cuda.reset_peak_memory_stats()
vae, unet, noise_scheduler = load_base_generator()
lora_params = add_lora(unet)
class_table = clip_class_table()
eval_items = [next(item for item in zip_members(dataset_zip, "validation") if item[1] == c) for c in range(10)]
eval_x, eval_y = load_images(dataset_zip, eval_items)
eval_batch = fixed_eval_batch(vae, noise_scheduler, eval_x, eval_y)
TIMES["load generator"] = time.time() - t
print(f"U-Net {sum(p.numel() for p in unet.parameters()) - sum(p.numel() for p in lora_params):,} frozen parameters; "
      f"LoRA {sum(p.numel() for p in lora_params):,} + class table {class_table.numel():,} trained")
print(f"loaded in {TIMES['load generator']:.0f} s")
""")
md("## Before fine-tuning")
md("""
With the LoRA adapters still a no-op, the model is plain Stable Diffusion 1.5 steered by CLIP's encoding of the class
prompts. The paper's seeds for sample 0 of four classes are used, so the same noise is used before and after training.
""")
code("""
t = time.time()
images_before = generate(unet, vae, class_table, GEN_CLASSES, GEN_SEEDS)
TIMES["generate"] = time.time() - t
print(f"generated {len(images_before)} images in {TIMES['generate']:.0f} s")
""")
md("## Fine-tune")
code("""
t = time.time()
torch.cuda.reset_peak_memory_stats()
lora_losses, eval_losses = train_lora(unet, vae, noise_scheduler, lora_params, class_table, train_x, train_y,
                                      steps=LORA_STEPS, batch_size=LORA_BATCH, eval_batch=eval_batch,
                                      eval_every=LORA_EVAL_EVERY, warmup_steps=LORA_WARMUP)
TIMES["LoRA fine-tuning"] = time.time() - t
memory_report("LoRA fine-tuning")
print(f"{LORA_STEPS} steps in {TIMES['LoRA fine-tuning']:.0f} s ({TIMES['LoRA fine-tuning'] / LORA_STEPS:.2f} s/step)")
""")
md("## The loss decreases")
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
axes[0].plot(np.arange(1, len(lora_losses) + 1), lora_losses, color="#c3c2b7", lw=1, label="each step")
w = min(25, len(lora_losses))
axes[0].plot(np.arange(w, len(lora_losses) + 1), np.convolve(lora_losses, np.ones(w) / w, mode="valid"),
             color="#eb6834", lw=2, label=f"mean of {w} steps")
axes[0].set_xlabel("step")
axes[0].set_ylabel("MSE loss")
axes[0].set_title("training loss (random images, noise levels)", loc="left", fontsize=10)
axes[0].legend(fontsize=8)
axes[1].plot([s for s, _ in eval_losses], [l for _, l in eval_losses], "o-", color="#eb6834")
axes[1].set_xlabel("step")
axes[1].set_ylabel("MSE loss")
axes[1].set_title("fixed evaluation batch (same images, noise, levels)", loc="left", fontsize=10)
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("LoRA fine-tuning of the generator", x=0.01, ha="left")
plt.tight_layout()
plt.show()
print(f"fixed-batch loss: {eval_losses[0][1]:.4f} before -> {eval_losses[-1][1]:.4f} after {LORA_STEPS} steps "
      f"({100 * (eval_losses[-1][1] / eval_losses[0][1] - 1):+.1f}%)")
""")
md("## Final result: before, after, and the released generator")
md("""
The same seeds, generated after the FAST fine-tuning, and with the **released** LoRA weights and class table (10,000
steps at batch 8): the last row reproduces sample 0 of these classes in the paper's synthetic training set.
""")
code("""
t = time.time()
images_after = generate(unet, vae, class_table, GEN_CLASSES, GEN_SEEDS)
released_table = load_released_lora(unet)  # replaces the adapters' weights with the released ones
images_released = generate(unet, vae, released_table, GEN_CLASSES, GEN_SEEDS)
TIMES["generate"] += time.time() - t
memory_report("generation")

rows = [("before fine-tuning", images_before), (f"after {LORA_STEPS} steps ({'FAST' if FAST else 'full'})", images_after),
        ("released (10,000 steps)", images_released)]
fig, axes = plt.subplots(3, len(GEN_CLASSES), figsize=(2.6 * len(GEN_CLASSES), 8.2))
for r, (label, imgs) in enumerate(rows):
    for c, img in enumerate(imgs):
        axes[r, c].imshow(img)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
        if r == 0:
            axes[r, c].set_title(CLASS_NAMES[GEN_CLASSES[c]], fontsize=9)
    axes[r, 0].set_ylabel(label, fontsize=9)
plt.tight_layout()
plt.show()
""")

md("# Summary")
md("""
Both pretrained models were fine-tuned with the project's loops: the classifier's loss fell from ln 10 toward the
level of the released model within a few epochs, and the generator's fixed-batch loss fell while its images moved from
generic Stable Diffusion galaxies toward the survey's look. Tutorial 6 runs the whole pipeline (generator,
synthetic data, classifiers on real / mixed / synthetic data) and follows the losses until they converge.
""")
code("""
TIMES["whole notebook"] = time.time() - T0
for step, seconds in TIMES.items():
    print(f"  {step:<26} {seconds / 60:5.1f} min")
print("peak GPU memory:", {k: f"{v:.1f} GiB" for k, v in MEMORY.items()})
""")

nb.write("05_Basic_Finetuning_Tutorial.ipynb", gpu=True, fill=FILL)

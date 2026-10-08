"""Builds notebooks/03_Model_Optimization_Tutorial.ipynb (run from the repository root).

Spec item #3 (Model Optimization): loss function, optimization algorithm and its parameters, learning rate and its
adjustment, batch size, number of epochs and early stopping, training vs validation loss graphs (gaps, early
stopping, avoiding overfitting). Uses the full runs' training logs (release file training_logs.zip); trains nothing.

Usage: python notebooks/_build/build_03_model_optimization.py [--times times.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbcommon import REPO_URL, Notebook, download_cells, drive_cells, install_cells  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--times", type=Path)
args = ap.parse_args()
FILL = {"__T_TOTAL__": "?"}
if args.times:
    m = json.loads(args.times.read_text())
    FILL["__T_TOTAL__"] = f"~{m['whole notebook']:.0f} s"

nb = Notebook()
md, code = nb.md, nb.code

md("# Galaxy10 DECaLS: Model Optimization Tutorial")
md("""
**Project:** *Faking Galaxies with Diffusion: A Real-vs-Synthetic Benchmark for Morphology Classification*
Hunter Sandoval, University of New Mexico. Tutorial 3 of 6.

**Scope.** How the project's two models were optimized, and why:

1. the **loss functions** (cross-entropy for the ResNet-18 classifier, mean squared error on the predicted noise for
   the diffusion generator);
2. the **optimization algorithm** (AdamW) and its parameters;
3. the **learning rates**, how they were chosen (a search on the validation set) and how they change during training;
4. the **batch sizes**, the **number of epochs / steps**, and the form of **early stopping** used (selecting the best
   epoch or checkpoint on the validation set);
5. the **training vs. validation loss and accuracy curves** of the paper's runs, the gaps between them, and what
   kept overfitting in check.

All curves come from the logs of the project's full training runs (downloaded from the GitHub Release), so this
tutorial trains nothing and runs in seconds on a CPU. Tutorials 5 and 6 run the same optimization loops themselves.
""")
md("## References")
md(f"""
1. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019, arXiv:1711.05101 (AdamW).
2. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015, arXiv:1412.6980.
3. I. Loshchilov and F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017, arXiv:1608.03983
   (cosine learning-rate decay).
4. P. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," arXiv:1706.02677 (learning-rate
   warm-up).
5. J. Ho, A. Jain, P. Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020, arXiv:2006.11239 (the
   diffusion loss).
6. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022, arXiv:2106.09685.
7. K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016, arXiv:1512.03385.
8. Project code, logs and model weights: {REPO_URL} (training code: `classifier/train_classifier.py`,
   `diffusion/train_lora.py`; logs: `results/training_logs/`).
""")

md("# Setup")
install_cells(nb, {"torch": "2.0", "torchvision": "0.15", "diffusers": "0.27", "numpy": "1.23", "matplotlib": "3.6"},
              """
| Python | 3.10 | 3.14.7 | (Colab's default) | |
| PyTorch (`torch`) | 2.0 | 2.14.0 | `pip install "torch>=2.0"` | optimizer and learning-rate scheduler objects |
| torchvision | 0.15 | 0.29.0 | `pip install "torchvision>=0.15"` | the ResNet-18 architecture (no weights) |
| diffusers | 0.27 | 0.40.0 | `pip install "diffusers>=0.27"` | the generator's learning-rate schedule |
| NumPy | 1.23 | 2.5.2 | `pip install "numpy>=1.23"` | arrays |
| Matplotlib | 3.6 | 3.11.2 | `pip install "matplotlib>=3.6"` | figures |
""")
md("## Hardware used for running")
md("""
- **No GPU needed**: nothing is trained, so Colab's standard **CPU runtime** is enough (a GPU runtime works too).
- **Memory:** well under 1 GB. **Disk:** 0.2 MB of logs.
- **Tested on:** a Windows 11 PC (AMD Ryzen 7 9800X3D, 32 GB RAM, Python 3.14). The logs themselves come from the
  project's runs on an NVIDIA RTX 5090 (32 GB): the classifiers took 1.5 min per run (30 epochs), the generator's LoRA
  fine-tuning 60 min (10,000 steps).
""")
md("## Approximate execution times")
md("""
| Step | Test PC | Colab CPU runtime (estimate) |
|---|---|---|
| Download the logs (0.2 MB) | < 1 s | < 5 s |
| **Whole tutorial** | **__T_TOTAL__** | **< 1 min** |
""")
md("## Expected results")
md("""
- Classifier: cross-entropy starts near ln(10) = 2.30 (a random guess among 10 classes) and ends near 0.05 on the
  training set; AdamW with learning rate 1e-3 (best of 1e-4 / 3e-4 / 1e-3 on validation: 85.5% / 86.2% / 86.7%), one
  warm-up epoch then cosine decay; batch 128; 30 epochs; the kept model is the epoch with the best validation accuracy
  (median epoch 26.5 for real-only, 28 for 50% synthetic, 19 for synthetic-only).
- Generator: the diffusion loss stays nearly flat (0.189 -> 0.187 over 10,000 steps); AdamW 1e-4, 500 warm-up steps
  then constant; batch 8; 10,000 steps (6.5 epochs); checkpoint chosen on validation KID and class fidelity.
- Curves: real-only and 50% synthetic reach ~86% / ~84% validation accuracy with validation loss ~0.5-0.6 while the
  training loss falls to ~0.05; synthetic-only overfits the synthetic images (validation loss rising from epoch ~5 to
  ~1.95) while its validation accuracy plateaus at ~64%.
""")

drive_cells(nb, "This tutorial does not read the dataset; the cell keeps the same folders as the other tutorials.")
md("# Download the training logs")
download_cells(nb, ["training_logs.zip"],
               intro="The logs of the paper's runs are one small zip file in the release (the same files are in the "
                     "repository under `results/training_logs/`).")
code("""
import json
import zipfile

import matplotlib.pyplot as plt
import numpy as np

logs = {}
with zipfile.ZipFile(WEIGHTS_DIR / "training_logs.zip") as z:
    for name in z.namelist():
        if name.endswith(".jsonl"):
            key = name.split("/", 1)[1][:-len(".jsonl")]  # e.g. classifier/A_replace0_seed0
            logs[key] = [json.loads(line) for line in z.read(name).decode().splitlines() if line.strip()]
CONDITIONS = [("A_replace0", "Real only", "#2a78d6"), ("A_replace0.5", "50% synthetic", "#1baf7a"),
              ("A_replace1", "Synthetic only", "#eb6834")]
for folder in ("classifier", "classifier_lr_search", "generator"):
    runs = sorted(k.split("/")[1] for k in logs if k.startswith(folder + "/"))
    print(f"{folder:<21} {len(runs):>2} runs: {', '.join(runs)}")
print("one classifier log line:", logs["classifier/A_replace0_seed0"][0])
""")

md("# Loss functions")
md("## Classifier: cross-entropy")
md("""
The classifier outputs 10 logits `z`; softmax turns them into probabilities `p_k = exp(z_k) / sum_j exp(z_j)`. For an
image of true class `y` the **cross-entropy loss** is `L = -log p_y`, averaged over the batch
(`torch.nn.functional.cross_entropy`, which applies the softmax itself). A confident correct answer costs almost 0; a
uniform guess over 10 classes costs `ln 10 = 2.30`; a confident wrong answer costs a lot. It is the standard loss for
single-label classification and the one ImageNet-pretrained ResNets were trained with. The cell evaluates it for three
made-up predictions.
""")
code("""
import torch
import torch.nn.functional as F

logits = torch.tensor([[8.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # confident and correct (true class 0)
                       [0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # no idea: uniform over the 10 classes
                       [0.0, 8, 0, 0, 0, 0, 0, 0, 0, 0]])   # confident and wrong
target = torch.tensor([0, 0, 0])
p_true = F.softmax(logits, dim=1)[:, 0]
losses = F.cross_entropy(logits, target, reduction="none")
for name, p, loss in zip(["confident, correct", "uniform guess", "confident, wrong"], p_true, losses):
    print(f"{name:<20} p(true class) = {p:.4f}   loss = -log p = {loss:.3f}")
print(f"ln(10) = {np.log(10):.3f}")
""")
md("## Generator: mean squared error on the noise")
md("""
The diffusion model is trained to **predict the noise** that was added to a training latent: with a latent `z0`, a
random noise level `t` and Gaussian noise `eps`, the noisy latent is `z_t = sqrt(a_t) z0 + sqrt(1 - a_t) eps`, and the
loss is `L = mean((eps - eps_hat(z_t, t, class))^2)` (reference 5). Most of this loss cannot be removed: at high noise
levels the noise is almost the whole input and at low levels almost none of it, so the loss depends mostly on which
`t` was drawn. The training loss of the LoRA fine-tuning therefore looks **flat** even though the model improves; the
project judged the generator on image quality (KID) and class fidelity on the validation set instead.
""")
code("""
lora = logs["generator/lora_r8"]
steps = np.array([r["step"] for r in lora])
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(steps, [r["loss"] for r in lora], color="#2a78d6", lw=1.2)
ax.set_xlabel("training step")
ax.set_ylabel("MSE loss (mean of 50 steps)")
ax.set_ylim(0, 0.3)
ax.set_title("Generator (LoRA) training loss: nearly flat by design", loc="left", fontsize=10)
ax.grid(alpha=0.3)
plt.show()
print(f"first logged loss {lora[0]['loss']:.4f} (step {lora[0]['step']}), last {lora[-1]['loss']:.4f} "
      f"(step {lora[-1]['step']}); range {min(r['loss'] for r in lora):.3f}-{max(r['loss'] for r in lora):.3f}")
""")

md("# Optimization algorithm")
md("""
Both models are trained with **AdamW** (references 1 and 2): Adam's per-parameter adaptive steps (running averages of
the gradient and of its square, with decay rates `betas = (0.9, 0.999)` and `eps = 1e-8`) plus **decoupled weight
decay**, which shrinks every weight a little at each step independently of the gradient scaling (plain Adam with L2
regularization would weaken the decay for weights with large gradients).

| | Classifier (ResNet-18) | Generator (LoRA) |
|---|---|---|
| Optimizer | `torch.optim.AdamW` | `torch.optim.AdamW` |
| Trained parameters | all 11,181,642 | LoRA 1,594,368 + class table 650,496 (U-Net and VAE frozen) |
| Learning rate | 1e-3 (peak) | 1e-4 |
| betas, eps | (0.9, 0.999), 1e-8 (defaults) | (0.9, 0.999), 1e-8 (defaults) |
| Weight decay | 0.05 | 0.01 |
| Gradient clipping | none | global norm 1.0 |

The cell builds both optimizers exactly as the training scripts do and prints their settings. (The generator's
optimizer is shown on stand-in tensors of the right sizes, to avoid downloading Stable Diffusion here.)
""")
code("""
import torch.nn as nn
from torchvision.models import resnet18

classifier = resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)
opt_classifier = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=0.05)

lora_stand_in = [torch.zeros(1_594_368, requires_grad=True)]      # same number of values as the LoRA weights
table_stand_in = [torch.zeros(11, 77, 768, requires_grad=True)]   # the class-embedding table
opt_generator = torch.optim.AdamW([{"params": lora_stand_in}, {"params": table_stand_in}], lr=1e-4, weight_decay=0.01)

for name, opt in (("classifier", opt_classifier), ("generator", opt_generator)):
    g = opt.param_groups[0]
    n = sum(p.numel() for grp in opt.param_groups for p in grp["params"])
    print(f"{name:<11} {type(opt).__name__}: lr {g['lr']:g}, betas {g['betas']}, eps {g['eps']:g}, "
          f"weight decay {g['weight_decay']:g}, amsgrad {g['amsgrad']}; {n:,} trained values")
""")

md("# Learning rate")
md("## Choosing the initial learning rate")
md("""
The classifier's learning rate was chosen on the **validation set** from three candidates, 1e-4, 3e-4 and 1e-3
(ResNet-18 trained on real images, seed 0; 1e-3 repeated with seed 1). 1e-3 gave the best validation accuracy and is
used for every classifier in the paper. The generator's learning rate 1e-4 is the usual value for LoRA fine-tuning of
Stable Diffusion; a sweep over 5e-5 / 1e-4 / 2e-4 (and ranks 4 / 8 / 16) changed the validation KID less than two
seeds of the same setting did (results/tables/t04_lora_sweep), so the default was kept.
""")
code("""
fig, ax = plt.subplots(figsize=(7, 3))
for name, color in (("real_lr1e-4_seed0", "#9ec5f4"), ("real_lr3e-4_seed0", "#5598e7"),
                    ("real_lr1e-3_seed0", "#184f95"), ("real_lr1e-3_seed1", "#184f95")):
    log = logs[f"classifier_lr_search/{name}"]
    acc = [100 * r["val_accuracy"] for r in log]
    lr = name.split("_")[1][2:]
    ax.plot(range(1, len(acc) + 1), acc, color=color, lw=1.4, ls="--" if name.endswith("seed1") else "-",
            label=f"LR {lr} ({name[-5:]}): best {max(acc):.1f}%")
ax.set_xlabel("epoch")
ax.set_ylabel("validation accuracy (%)")
ax.set_ylim(50, 90)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Learning-rate search on the validation set (real-only ResNet-18)", loc="left", fontsize=10)
ax.grid(alpha=0.3)
plt.show()
""")
md("## Adjusting the learning rate during training")
md("""
- **Classifier: linear warm-up for one epoch, then cosine decay to 0**, updated after every batch. The warm-up keeps the
  first, large gradients of the freshly initialized output layer from damaging the pretrained features (reference 4);
  the cosine decay takes large steps early and ever smaller ones at the end, so the final epochs fine-tune
  (reference 3).
- **Generator: linear warm-up over 500 steps, then constant** (`constant_with_warmup` in diffusers). LoRA fine-tuning is
  run for a fixed budget and judged by checkpoints, so a decaying schedule is not needed; a cosine schedule was tried in
  the sweep and made no measurable difference.

The cell rebuilds both schedules with the same code as the training scripts and checks the classifier's against the
learning rate recorded in its log after every epoch.
""")
code("""
import math

from diffusers.optimization import get_scheduler

EPOCHS, STEPS_PER_EPOCH = 30, math.ceil(12330 / 128)  # 97 batches of 128 per epoch
total, warmup = EPOCHS * STEPS_PER_EPOCH, STEPS_PER_EPOCH


def lr_factor(step):  # classifier/train_classifier.py
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


sched_c = torch.optim.lr_scheduler.LambdaLR(opt_classifier, lr_factor)
lr_c = []
for _ in range(total):
    lr_c.append(sched_c.get_last_lr()[0])
    opt_classifier.step()
    sched_c.step()
sched_g = get_scheduler("constant_with_warmup", optimizer=opt_generator, num_warmup_steps=500, num_training_steps=10000)
lr_g = []
for _ in range(10000):
    lr_g.append(sched_g.get_last_lr()[0])
    opt_generator.step()
    sched_g.step()

logged = [r["lr"] for r in logs["classifier/A_replace0_seed0"]]  # learning rate after each epoch's last step
rebuilt = [lr_c[e * STEPS_PER_EPOCH] if e * STEPS_PER_EPOCH < total else sched_c.get_last_lr()[0] for e in range(1, 31)]
print("classifier schedule rebuilt = logged learning rate after every epoch:", np.allclose(rebuilt, logged))

fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
axes[0].plot(np.arange(total) / STEPS_PER_EPOCH, lr_c, color="#2a78d6")
axes[0].set_xlabel("epoch")
axes[0].set_title("Classifier: 1 warm-up epoch, then cosine to 0", loc="left", fontsize=10)
axes[1].plot(lr_g, color="#eb6834")
axes[1].set_xlabel("step")
axes[1].set_title("Generator: 500 warm-up steps, then constant", loc="left", fontsize=10)
for ax in axes:
    ax.set_ylabel("learning rate")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
""")

md("# Batch size")
md("""
- **Classifier: 128 images per batch** (97 batches per epoch of the 12,330 training images). ResNet-18 at 256 x 256
  pixels in bf16 needs only a few GB at this batch size; 128 gives stable batch-norm statistics and gradients, and the
  learning-rate search was done at this batch size.
- **Generator: 8 images per batch** (1,542 batches per epoch). Each image is upsampled to 512 x 512 and the gradient
  has to flow through the whole 860-million-parameter U-Net (even though only the LoRA weights change), so memory is
  the limit: batch 8 used 10.7 GB on the RTX 5090. On a 15 GB Colab T4, Tutorials 5 and 6 use batch 4 in their FAST
  mode.
""")
code("""
print(f"classifier: {math.ceil(12330 / 128)} steps per epoch at batch 128")
print(f"generator:  {math.ceil(12330 / 8)} steps per epoch at batch 8; 10,000 steps = {10000 / math.ceil(12330 / 8):.1f} "
      f"epochs; peak GPU memory in the run: {lora[-1]['peak_mem_alloc_gb']} GB")
""")

md("# Number of epochs and early stopping")
md("""
- **Classifier: 30 epochs**, with **early stopping in the form of model selection**: training always runs all 30 epochs
  (the cosine schedule needs the full length), and after every epoch the model is scored on the validation set; the
  epoch with the best **validation accuracy** is the one kept (and the one evaluated once on the test set). The cell
  shows which epoch was kept in each run.
- **Generator: 10,000 steps (6.5 epochs)**, saving a checkpoint every 1,000 steps. Checkpoints were compared on the
  validation set by image realism (KID, lower is better) and class fidelity; checkpoint 10,000 was the best on both,
  and a pilot confirmed it gives the most useful synthetic training data. KID of the chosen run by checkpoint
  (results/tables/t04_lora_sweep, guidance 2):

| Checkpoint (steps) | 2,000 | 3,000 | 4,000 | 5,000 | 6,000 | 8,000 | 10,000 |
|---|---|---|---|---|---|---|---|
| Validation KID x1e3 | 62.0 | 73.7 | 57.1 | 68.1 | 66.7 | 69.0 | **55.0** |
""")
code("""
for cond, name, _ in CONDITIONS:
    runs = sorted(k for k in logs if k.startswith(f"classifier/{cond}_seed"))
    kept = [int(np.argmax([r["val_accuracy"] for r in logs[k]])) + 1 for k in runs]
    print(f"{name:<15} {len(runs)} seeds, kept epochs {kept} (median {np.median(kept):g})")
""")

md("# Training vs. validation loss and accuracy")
md("""
The paper's runs, re-run with extra logging (training accuracy and validation loss were not logged originally; the
re-runs reproduce every originally logged number and the final weights bit-for-bit). Three curves per metric:
**validation** (solid), the **training set without augmentation** measured in evaluation mode after each epoch
(dashed), and the **training loss/accuracy as measured during the epoch**, on augmented images while the weights
change (dotted). Seed means; the band is the range of the validation curve over seeds.
""")
code("""
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey="row")
for col, (cond, name, color) in enumerate(CONDITIONS):
    runs = sorted(k for k in logs if k.startswith(f"classifier/{cond}_seed"))
    get = lambda key, s=1.0: s * np.array([[r[key] for r in logs[k]] for k in runs])  # noqa: E731
    ep = np.arange(1, 31)
    kept = np.median([int(np.argmax(v)) + 1 for v in get("val_accuracy")])
    for row, (metric, scale) in enumerate((("loss", 1.0), ("accuracy", 100.0))):
        ax = axes[row, col]
        v = get(f"val_{metric}", scale)
        ax.fill_between(ep, v.min(0), v.max(0), color=color, alpha=0.18, lw=0)
        ax.plot(ep, v.mean(0), color=color, lw=2, label="validation")
        ax.plot(ep, get(f"train_eval_{metric}", scale).mean(0), color=color, ls="--", lw=1.3,
                label="training set, no augmentation")
        ax.plot(ep, get(f"train_{metric}", scale).mean(0), color=color, ls=":", lw=1.5,
                label="training, during the epoch")
        ax.axvline(kept, color="gray", ls=":", lw=1)
        ax.grid(alpha=0.3)
    axes[0, col].set_title(f"{name} ({len(runs)} seeds); kept epoch (median) {kept:g}", loc="left", fontsize=10)
    axes[1, col].set_xlabel("epoch")
axes[0, 0].set_ylabel("cross-entropy loss")
axes[1, 0].set_ylabel("accuracy (%)")
axes[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.show()

for cond, name, _ in CONDITIONS:
    runs = sorted(k for k in logs if k.startswith(f"classifier/{cond}_seed"))
    last = lambda key: np.mean([logs[k][-1][key] for k in runs])  # noqa: E731
    vl = np.mean([[r["val_loss"] for r in logs[k]] for k in runs], 0)
    print(f"{name:<15} epoch 30: train loss {last('train_eval_loss'):.3f} vs validation loss {last('val_loss'):.3f} "
          f"(gap {last('val_loss') - last('train_eval_loss'):.2f}); validation accuracy {100 * last('val_accuracy'):.1f}%; "
          f"lowest validation loss at epoch {int(np.argmin(vl)) + 1}")
""")
md("## Reading the curves: gaps, early stopping and overfitting")
md("""
- **The two training curves differ** because the dotted one is measured *during* the epoch, on randomly flipped and
  rotated images, while the weights are still changing; the dashed one is the whole training set after the epoch,
  without augmentation. Early on the dotted loss is lower because it averages over an epoch in which the model keeps
  improving; later they agree.
- **Real-only and 50% synthetic**: the training loss falls to ~0.05 while the validation loss levels off at ~0.5-0.6.
  That gap is the usual **generalization gap**: the network has nearly memorized its 12,330 training images, but the
  validation loss only drifts up slightly after its minimum (epoch 21) and the validation accuracy keeps creeping up until the learning rate reaches 0,
  so these runs do **not** overfit in the harmful sense. The kept epochs are late (median 26.5 and 28).
- **Synthetic-only** overfits: its training loss goes to ~0.02, but the validation loss is lowest at **epoch 5** and
  then *rises* to ~1.95, while the validation accuracy still improves until about epoch 19 and then plateaus at ~64%.
  The model becomes ever more confident on real galaxies it gets wrong (confident mistakes are expensive in
  cross-entropy), because what it memorizes are properties of the *synthetic* images that real galaxies do not share:
  this is the domain gap that is the paper's main result. Selecting the kept epoch on validation accuracy (median 19)
  stops it where it is most useful.
- **Loss vs. accuracy for early stopping**: the two criteria can disagree (synthetic-only: epoch 5 by loss, 19 by
  accuracy). The project selects on validation accuracy, the metric it reports.
- **What kept overfitting in check**: ImageNet-pretrained features (the network starts from general image features
  instead of memorizing from scratch); random flips and 90-degree rotations (galaxy morphology does not depend on
  orientation, so this multiplies the effective data by 8); weight decay 0.05; the cosine schedule; and selecting the
  epoch on the validation set, which is never trained on. For the generator: only 0.24% of its parameters are trained
  (LoRA rank 8 + class table), and checkpoints were chosen on validation realism and fidelity; a memorization check found
  no synthetic image closer to a training image than unseen real images are.
""")

md("# Summary")
md("""
| | Classifier (ResNet-18) | Generator (Stable Diffusion 1.5 + LoRA) |
|---|---|---|
| Loss | cross-entropy | mean squared error on the predicted noise |
| Optimizer | AdamW, betas (0.9, 0.999), weight decay 0.05 | AdamW, betas (0.9, 0.999), weight decay 0.01; gradient clipping 1.0 |
| Learning rate | 1e-3 (chosen on validation from 1e-4 / 3e-4 / 1e-3); 1 warm-up epoch + cosine to 0 | 1e-4; 500 warm-up steps, then constant |
| Batch size | 128 | 8 |
| Length | 30 epochs | 10,000 steps (6.5 epochs) |
| Early stopping | keep the epoch with the best validation accuracy | keep the checkpoint with the best validation KID / class fidelity |
| Overfitting control | pretrained weights, flips + rotations, weight decay, cosine decay, validation selection | LoRA (0.24% of parameters trained), flips + rotations, validation checkpoint selection |
""")
code("""
TIMES["whole notebook"] = time.time() - T0
print(f"whole notebook: {TIMES['whole notebook']:.0f} s")
""")

nb.write("03_Model_Optimization_Tutorial.ipynb", gpu=False, fill=FILL)

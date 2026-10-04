"""Builds notebooks/06_Full_Training_Tutorial.ipynb (run from the repository root).

Spec item #6 (Full training): run the training loop multiple times and show that the validation and testing losses
converge; if not, add data augmentation. Here the whole pipeline: generator (LoRA) training, generation, and the
paper's classifier experiment (real-only / 50% synthetic / synthetic-only, several seeds, validation and test losses
per epoch), plus an augmentation on/off comparison. FAST mode (default) keeps it short on a Colab T4; the final
results cells use the released weights and the full runs' logs.

Usage: python notebooks/_build/build_06_full_training.py [--times times.json]
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
KEYS = ("__T_TOTAL__", "__T_DOWNLOAD__", "__T_LOAD__", "__T_LORA__", "__T_GEN__", "__T_CLS__", "__T_AUG__",
        "__T4_TOTAL__", "__MEM_LORA__", "__MEM_GEN__", "__MEM_CLS__", "__FAST_RESULTS__")
FILL = {k: "?" for k in KEYS}
if args.times:
    FILL.update(json.loads(args.times.read_text())["fill"])

nb = Notebook()
md, code = nb.md, nb.code

md("# Galaxy10 DECaLS: Full Training Tutorial")
md(f"""
**Project:** *Faking Galaxies with Diffusion: A Real-vs-Synthetic Benchmark for Morphology Classification*
Hunter Sandoval, University of New Mexico. Tutorial 6 of 6.

**Scope.** The project's whole training pipeline, end to end:

1. **train the generator**: Stable Diffusion 1.5 + LoRA + class table on the training images;
2. **generate synthetic galaxies** with it;
3. **train the classifiers of the paper's main experiment**, on real images only, on a 50/50 real/synthetic mix, and on
   synthetic images only, **several times with different seeds**, measuring the **training, validation and test
   loss and accuracy after every epoch** to show that they converge;
4. **data augmentation**: the same real-only training without the random flips and rotations, to show what
   augmentation does to the gap between training and held-out losses.

**FAST mode (on by default)** keeps the notebook to roughly 20-30 minutes on a free Colab T4: 200 LoRA steps at batch 4,
one generated galaxy per class, and classifiers trained for 5 epochs on a fixed-seed 10% stratified subset of the real
and synthetic training images (2 seeds each), evaluated on all validation and test images. The classifiers' synthetic
images come from the paper's own synthetic set (12,330 images, downloaded from the release), so the three training
conditions are the paper's, only smaller. `FAST = False` runs the paper's full settings (table below; the classifier
training sets are then drawn exactly as in the paper). The **final results cells** show the paper's full runs (their
logs) and evaluate the released classifiers, which were trained with the full settings.
""")
md("## References")
md(f"""
1. R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022, arXiv:2112.10752;
   weights: https://huggingface.co/{SD_REPO}
2. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022, arXiv:2106.09685.
3. J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," arXiv:2207.12598, 2022.
4. C. Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models," arXiv:2211.01095.
5. K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016, arXiv:1512.03385.
6. C. Shorten and T. M. Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning," J. Big Data 6, 60, 2019.
7. P. Micikevicius et al., "Mixed Precision Training," ICLR 2018, arXiv:1710.03740.
8. Galaxy10 DECaLS: https://github.com/henrysky/Galaxy10
9. Project code, logs, model weights and synthetic set: {REPO_URL}
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
  GitHub). The T4 has 15 GB of memory and no bf16 hardware, so the notebook trains in fp16 there (next section).
- **Peak GPU memory (FAST mode, fp16, GPU limited to 15 GB like a T4):** LoRA training __MEM_LORA__, generation
  __MEM_GEN__, classifier training __MEM_CLS__. Full settings (`FAST = False`): the LoRA training at batch 8 needs
  10.6 GB in fp16 (measured the same way), so it fits a T4 too.
- **Disk:** about 4 GB on the Colab machine (the 1.75 GB synthetic set, Stable Diffusion's 2 GB).
- **Tested on:** a Windows 11 PC with an NVIDIA RTX 5090 (32 GB), AMD Ryzen 7 9800X3D, 32 GB RAM, Python 3.14. Every
  code cell was run there in fp16 with GPU memory capped at 15 GB (to mimic the T4) and in the GPU's native bf16.
""")
md("## Approximate execution times")
md(f"""
| Step | Test PC (FAST, fp16) | Colab T4 (FAST, estimate) | Full settings on the RTX 5090 (the paper's runs) |
|---|---|---|---|
| Downloads (weights, 1.75 GB synthetic set, Stable Diffusion) | __T_DOWNLOAD__ | 2-4 min | - |
| Load the images | __T_LOAD__ | 1-2 min | - |
| Stage 1: generator (LoRA) training | __T_LORA__ | ~5-8 min | {PAPER_TIMES["lora"]} |
| Stage 2: generation (10 + 10 images) | __T_GEN__ | ~1-2 min | {PAPER_TIMES["generation"]} |
| Stage 3: 6 classifier runs (3 conditions x 2 seeds) | __T_CLS__ | ~8-12 min | {PAPER_TIMES["classifier"]} (x 14 runs in the paper's main curve) |
| Stage 4: augmentation off (2 runs) | __T_AUG__ | ~3-4 min | 1.5 min per run |
| **Whole tutorial** | **__T_TOTAL__** | **__T4_TOTAL__** | - |

The Colab column is an estimate: for these models a T4 is roughly 8-10 times slower than the RTX 5090. With the full
settings (`FAST = False`) a T4 would need about 8-10 hours for the LoRA training and about 15 min per classifier run.
""")
md("## Expected results")
md("""
- **Convergence**: the training loss falls every epoch. The validation and test losses can jump in the first epochs
  of a FAST run (see Stage 3), then fall steadily; FAST runs are short (50 steps), so they are still falling slowly when
  the learning rate reaches 0 at epoch 5. The validation and test curves lie on top of each other (the two held-out sets
  come from the same distribution) and the seeds agree. The full runs (final results) level off over 30 epochs.
- **The paper's result, in small**: synthetic-only training reaches much lower held-out accuracy than real-only; the
  50% mix is close to real-only. FAST mode (10% of the data, 5 epochs): __FAST_RESULTS__.
- **Augmentation**: without flips and rotations the training loss falls faster but the held-out losses stay higher, so
  the gap widens: augmentation is what keeps the held-out losses converging together with the training loss.
- **Released classifiers (full settings)**: validation accuracy 86.7% / 84.0% / 66.2% and test accuracy 87.1% / 84.6% /
  67.2% for real-only / 50% synthetic / synthetic-only (seed 0, as in the paper).
""")

drive_cells(nb, "Training uses the training, validation and testing images from Tutorial 1.")
dataset_cells(nb)

md("# FAST mode and the full settings")
md(f"""
| Setting | FAST (default) | Full (paper) |
|---|---|---|
| Classifier training images per condition | 10% stratified subset (1,234; fixed seed) of the real and of the synthetic images | 12,330 |
| Classifier epochs / seeds per condition | 5 / 2 | 30 / 3 (the paper used 8 for real-only) |
| Classifier batch / learning rate | 128 / 1e-3 (1 warm-up epoch + cosine) | 128 / 1e-3 (1 warm-up epoch + cosine) |
| Validation / test images | all 2,607 / 2,609 | all 2,607 / 2,609 |
| LoRA training images | the 10% subset | all 12,330 |
| LoRA steps / batch / warm-up | 200 / 4 / 20 | 10,000 / 8 / 500 |
| Generated images | 1 per class | the paper's set: 12,330 (`diffusion/generate_synthetic.py`, {PAPER_TIMES["generation"]}) |
| Time on the RTX 5090 | minutes | LoRA {PAPER_TIMES["lora"]}; classifiers {PAPER_TIMES["classifier"]} |

The classifier stage always uses the paper's released synthetic set (the images the full generator produced), so its
three conditions are the paper's; the few images generated in Stage 2 show how that set was made.
""")
code("""
FAST = True  # False: the paper's full settings (hours on a T4)

if FAST:
    FRACTION, CLS_EPOCHS, SEEDS = 0.1, 5, [0, 1]
    LORA_STEPS, LORA_BATCH, LORA_WARMUP, LORA_EVAL_EVERY = 200, 4, 20, 25
else:
    FRACTION, CLS_EPOCHS, SEEDS = 1.0, 30, [0, 1, 2]
    LORA_STEPS, LORA_BATCH, LORA_WARMUP, LORA_EVAL_EVERY = 10000, 8, 500, 500
CONDITIONS = [(0.0, "Real only"), (0.5, "50% synthetic"), (1.0, "Synthetic only")]  # share of synthetic images
COLORS = {"Real only": "#2a78d6", "50% synthetic": "#1baf7a", "Synthetic only": "#eb6834"}
print(f"FAST = {FAST}: {FRACTION:.0%} of the training images, {CLS_EPOCHS} epochs x seeds {SEEDS}; "
      f"LoRA {LORA_STEPS} steps at batch {LORA_BATCH}")
""")
md("# Downloads")
download_cells(nb, ["resnet18_real_only_seed0.safetensors", "resnet18_synthetic_only_seed0.safetensors",
                    "resnet18_mix50_seed0.safetensors", "lora_unet.safetensors", "class_embeddings.safetensors",
                    "lora_config.json", "synthetic_set.zip", "training_logs.zip"],
               intro="The notebook trains its own models starting from the public pretrained ones (ImageNet ResNet-18, "
                     "Stable Diffusion 1.5). It downloads the paper's **synthetic set** (`synthetic_set.zip`, "
                     "12,330 images generated by the project's Stable Diffusion 1.5-based generator, so the CreativeML "
                     "Open RAIL-M license applies to them), the **training logs** of the full runs, and the **released "
                     "weights** for the final results cells.")
precision_cells(nb)

md("# Load the images")
md("""
Real images come from Tutorial 1's zip; synthetic images from `synthetic_set.zip`
(`synthetic_set/<label>_<class>/<sample index>_seed<seed>.png`, in the order the paper drew from). Both training pools
are fixed-seed **stratified** subsets (the same fraction of every class, so the class imbalance is kept); validation and
testing use all images.
""")
code("""
t = time.time()
synthetic_zip = zipfile.ZipFile(WEIGHTS_DIR / "synthetic_set.zip")
synthetic_items = [(n, int(n.split("/")[1].split("_")[0])) for n in sorted(synthetic_zip.namelist()) if n.endswith(".png")]
real_x, real_y = load_images(dataset_zip, stratified(zip_members(dataset_zip, "training"), FRACTION, seed=0))
syn_x, syn_y = load_images(synthetic_zip, stratified(synthetic_items, FRACTION, seed=0))
val_x, val_y = load_images(dataset_zip, zip_members(dataset_zip, "validation"))
test_x, test_y = load_images(dataset_zip, zip_members(dataset_zip, "testing"))
TIMES["load images"] = time.time() - t
print(f"synthetic set: {len(synthetic_items)} images")
for name, y in (("real training pool", real_y), ("synthetic pool", syn_y), ("validation", val_y), ("testing", test_y)):
    print(f"{name:<19} {len(y):>6} images, per class {np.bincount(y.numpy(), minlength=10).tolist()}")
print(f"loaded in {TIMES['load images']:.0f} s")
""")

md("# Stage 1: Train the generator")
md("""
The project's LoRA training loop (Tutorial 5 explains it step by step): Stable Diffusion 1.5 frozen, LoRA adapters on
the attention projections and a CLIP-initialised class-embedding table trained with the diffusion (noise-prediction)
loss. Besides the noisy per-step loss the loop reports the loss on a fixed evaluation batch (one validation image per
class, fixed noise and noise levels), which moves only when the model does.
""")
classifier_code_cell(nb)
lora_code_cell(nb)
code("""
t = time.time()
torch.cuda.reset_peak_memory_stats()
vae, unet, noise_scheduler = load_base_generator()
lora_params = add_lora(unet)
class_table = clip_class_table()
eval_idx = [int(np.flatnonzero(val_y.numpy() == c)[0]) for c in range(10)]
eval_batch = fixed_eval_batch(vae, noise_scheduler, val_x[eval_idx], val_y[eval_idx])
lora_losses, eval_losses = train_lora(unet, vae, noise_scheduler, lora_params, class_table, real_x, real_y,
                                      steps=LORA_STEPS, batch_size=LORA_BATCH, eval_batch=eval_batch,
                                      eval_every=LORA_EVAL_EVERY, warmup_steps=LORA_WARMUP)
TIMES["stage 1: generator"] = time.time() - t
memory_report("stage 1: generator training")
""")
code("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
w = min(25, len(lora_losses))
axes[0].plot(np.arange(1, len(lora_losses) + 1), lora_losses, color="#c3c2b7", lw=1, label="each step")
axes[0].plot(np.arange(w, len(lora_losses) + 1), np.convolve(lora_losses, np.ones(w) / w, mode="valid"),
             color="#eb6834", lw=2, label=f"mean of {w} steps")
axes[0].set_title("training loss", loc="left", fontsize=10)
axes[0].legend(fontsize=8)
axes[1].plot([s for s, _ in eval_losses], [l for _, l in eval_losses], "o-", color="#eb6834")
axes[1].set_title("fixed evaluation batch", loc="left", fontsize=10)
for ax in axes:
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss")
    ax.grid(alpha=0.3)
fig.suptitle("Stage 1: generator (LoRA) training", x=0.01, ha="left")
plt.tight_layout()
plt.show()
print(f"fixed-batch loss {eval_losses[0][1]:.4f} -> {eval_losses[-1][1]:.4f}")
""")

md("# Stage 2: Generate synthetic galaxies")
md("""
One galaxy per class with the generator just trained, using the paper's seeds for sample 0 of each class
(seed = 7,000,000,000 + 1,000,000 x class). The same seeds with the **released** generator (full training) reproduce
sample 0 of every class in the paper's synthetic set, which the classifiers below are trained on; the third row shows
those images from `synthetic_set.zip`.
""")
code("""
t = time.time()
gen_classes = list(range(10))
gen_seeds = [7_000_000_000 + c * 1_000_000 for c in gen_classes]
images_here = generate(unet, vae, class_table, gen_classes, gen_seeds)
released_table = load_released_lora(unet)
images_released = generate(unet, vae, released_table, gen_classes, gen_seeds)
paper_images = [np.asarray(Image.open(io.BytesIO(synthetic_zip.read(next(
    n for n, c in synthetic_items if c == k and n.split("/")[2].startswith("00000_")))))) for k in gen_classes]
TIMES["stage 2: generation"] = time.time() - t
memory_report("stage 2: generation")
corr = [np.corrcoef(a.astype(float).ravel(), b.astype(float).ravel())[0, 1] for a, b in zip(images_released, paper_images)]
print(f"released generator vs the paper's synthetic set, same seeds: pixel correlation {np.min(corr):.4f}-{np.max(corr):.4f}")

rows = [(f"this run ({LORA_STEPS} steps)", images_here), ("released generator", images_released),
        ("paper's synthetic set", paper_images)]
fig, axes = plt.subplots(3, 10, figsize=(16, 5.6))
for r, (label, imgs) in enumerate(rows):
    for c, img in enumerate(imgs):
        axes[r, c].imshow(img)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
        if r == 0:
            axes[r, c].set_title(CLASS_NAMES[c].replace(" ", "\\n", 1), fontsize=7.5)
    axes[r, 0].set_ylabel(label, fontsize=8)
plt.tight_layout()
plt.show()
del unet, vae, lora_params, class_table, released_table
torch.cuda.empty_cache()
""")

md("# Stage 3: Train the classifiers several times")
md("""
The paper's main experiment, per condition: every class keeps its number of training images, and a share of them
(0%, 50% or 100%) is replaced by synthetic images of the same class (the rule of `classifier/train_classifier.py
--replace-fraction`, including its random draws). Each condition is trained with every seed in `SEEDS`, with the
project's training loop (Tutorial 5), and after every epoch the loss and accuracy are measured on the **validation**
and the **test** images.

The test set is only *watched* here, as the tutorial requires: the kept model of every run is still the epoch with the
best **validation** accuracy, and nothing is chosen on test data (in the paper the test set was used exactly once, for
the final evaluation).

In FAST mode an epoch is only 10 steps, so the held-out losses can **jump up in the first epochs** before they fall:
evaluation uses the batch-norm *running* statistics, which lag behind weights that change quickly while the learning
rate is at its peak (Tutorial 5). From epoch 3 on the curves fall steadily; with the full settings (97 steps per epoch)
the effect is small.
""")
code("""
def compose(fraction, seed):
    \"\"\"Training set of one condition: per class, round(fraction x n) synthetic + the rest real (the paper's rule).\"\"\"
    rng = np.random.default_rng(1000 + seed)
    xs, ys = [], []
    for c in range(10):
        real_c = np.flatnonzero(real_y.numpy() == c)
        syn_c = np.flatnonzero(syn_y.numpy() == c)
        n_syn = int(round(fraction * len(real_c)))
        n_real = len(real_c) - n_syn
        pick = real_c if n_real == len(real_c) else np.sort(rng.choice(real_c, n_real, replace=False))
        xs.append(real_x[pick])
        ys += [c] * n_real
        if n_syn:
            xs.append(syn_x[syn_c[np.sort(rng.choice(len(syn_c), n_syn, replace=False))]])
            ys += [c] * n_syn
    return torch.cat(xs), torch.tensor(ys, dtype=torch.int64)


held_out = {"validation": (val_x, val_y), "testing": (test_x, test_y)}
runs = {}  # (condition, seed, augment) -> per-epoch history
t = time.time()
torch.cuda.reset_peak_memory_stats()
for fraction, name in CONDITIONS:
    for seed in SEEDS:
        x, y = compose(fraction, seed)
        print(f"{name}, seed {seed}: {len(x)} training images ({int(round(fraction * 100))}% synthetic)")
        model, history, _, kept = train_classifier(x, y, held_out, epochs=CLS_EPOCHS, seed=seed)
        runs[(name, seed, True)] = history
        print(f"  kept epoch {kept} (best validation accuracy)")
        del model
TIMES["stage 3: classifiers"] = time.time() - t
memory_report("stage 3: classifier training")
""")
md("## The losses converge")
md("""
For each condition: the training loss (measured during each epoch, on augmented images), and the validation and test
losses and accuracies after each epoch; lines are seed means, bands the range over seeds.
""")
code("""
def curves(keys, title):
    \"\"\"2 x n grid: loss (top) and accuracy (bottom) per epoch, one column per key = (condition, augment).\"\"\"
    fig, axes = plt.subplots(2, len(keys), figsize=(4.4 * len(keys), 6.2), sharex=True, sharey="row", squeeze=False)
    for col, (name, augment, label) in enumerate(keys):
        hs = [runs[(name, s, augment)] for s in SEEDS]
        ep = np.arange(1, len(hs[0]) + 1)
        color = COLORS[name]
        for row, (metric, scale) in enumerate((("loss", 1.0), ("accuracy", 100.0))):
            ax = axes[row, col]
            for split, ls, lw in (("train", ":", 1.5), ("validation", "-", 2.0), ("testing", "--", 1.5)):
                v = scale * np.array([[h[f"{split}_{metric}"] for h in hist] for hist in hs])
                ax.plot(ep, v.mean(0), color=color, ls=ls, lw=lw, label=split if split != "train" else "training")
                if split != "train":
                    ax.fill_between(ep, v.min(0), v.max(0), color=color, alpha=0.12, lw=0)
            ax.grid(alpha=0.3)
            ax.set_xticks(ep if len(ep) <= 10 else [1, 10, 20, 30])
        axes[0, col].set_title(f"{label} ({len(SEEDS)} seeds)", loc="left", fontsize=10)
        axes[1, col].set_xlabel("epoch")
    axes[0, 0].set_ylabel("cross-entropy loss")
    axes[1, 0].set_ylabel("accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


curves([(name, True, name) for _, name in CONDITIONS],
       f"Stage 3: training, validation and test curves ({'FAST' if FAST else 'full'} settings)")

print(f"{'condition':<16}{'val loss':>18}{'test loss':>18}{'val acc %':>16}{'test acc %':>16}   last-epoch change of val / test loss")
SUMMARY = {}
for _, name in CONDITIONS:
    hs = [runs[(name, s, True)] for s in SEEDS]
    last = lambda key, scale=1.0: scale * np.array([h[-1][key] for h in hs])  # noqa: E731
    change = [np.mean([h[-1][f"{k}_loss"] - h[-2][f"{k}_loss"] for h in hs]) for k in ("validation", "testing")]
    SUMMARY[name] = {k: float(last(k).mean()) for k in ("validation_loss", "testing_loss", "validation_accuracy",
                                                        "testing_accuracy")}
    print(f"{name:<16}{last('validation_loss').mean():>10.3f} +/- {last('validation_loss').std():.3f}"
          f"{last('testing_loss').mean():>10.3f} +/- {last('testing_loss').std():.3f}"
          f"{last('validation_accuracy', 100).mean():>10.1f} +/- {last('validation_accuracy', 100).std():.1f}"
          f"{last('testing_accuracy', 100).mean():>10.1f} +/- {last('testing_accuracy', 100).std():.1f}"
          f"   {change[0]:+.3f} / {change[1]:+.3f}")
""")

md("# Stage 4: Data augmentation")
md("""
All runs above use the project's augmentation: every training image is randomly flipped and rotated by a multiple of
90 degrees each time it is used (galaxy morphology does not depend on orientation, so this is 8 valid versions of every
image). The cell repeats the real-only runs **without** augmentation and compares the curves. Without it the network
sees the same pixels every epoch: its training loss falls faster, but the validation and test losses stay higher, so
the gap between training and held-out losses widens. Augmentation narrows that gap, which is why the project
uses it in every run.
""")
code("""
t = time.time()
for seed in SEEDS:
    print(f"Real only without augmentation, seed {seed}")
    model, history, _, _ = train_classifier(*compose(0.0, seed), held_out, epochs=CLS_EPOCHS, seed=seed, augment=False)
    runs[("Real only", seed, False)] = history
    del model
TIMES["stage 4: augmentation"] = time.time() - t
curves([("Real only", True, "Real only, with augmentation"), ("Real only", False, "Real only, no augmentation")],
       "Stage 4: the effect of data augmentation")
for augment in (True, False):
    hs = [runs[("Real only", s, augment)] for s in SEEDS]
    gap = np.mean([h[-1]["validation_loss"] - h[-1]["train_loss"] for h in hs])
    print(f"augmentation {'on ' if augment else 'off'}: final training loss {np.mean([h[-1]['train_loss'] for h in hs]):.3f}, "
          f"validation loss {np.mean([h[-1]['validation_loss'] for h in hs]):.3f}, "
          f"test loss {np.mean([h[-1]['testing_loss'] for h in hs]):.3f}; validation - training gap {gap:.3f}")
""")

md("# Final results: the full runs and the released classifiers")
md("## The paper's full runs")
md("""
The same plot for the paper's runs with the full settings (30 epochs, all 12,330 images; 8 / 3 / 3 seeds), from their
training logs. Those runs measured the training and validation curves only: the paper evaluated the test set once, for
the kept models (next cell), not after every epoch. With the full data the validation curves converge smoothly;
synthetic-only is the exception the paper is about: its validation loss rises after epoch ~5 while its accuracy
plateaus at ~64%, because the synthetic images differ systematically from real galaxies.
""")
code("""
logs = {}
with zipfile.ZipFile(WEIGHTS_DIR / "training_logs.zip") as z:
    for member in z.namelist():
        if member.startswith("training_logs/classifier/") and member.endswith(".jsonl"):
            logs[member.split("/")[-1][:-6]] = [json.loads(l) for l in z.read(member).decode().splitlines() if l.strip()]
paper = {"Real only": "A_replace0", "50% synthetic": "A_replace0.5", "Synthetic only": "A_replace1"}
fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.2), sharex=True, sharey="row")
for col, (_, name) in enumerate(CONDITIONS):
    hs = [v for k, v in sorted(logs.items()) if k.startswith(paper[name] + "_seed")]
    ep = np.arange(1, 31)
    for row, (metric, scale) in enumerate((("loss", 1.0), ("accuracy", 100.0))):
        ax = axes[row, col]
        for key, ls, lw, label in ((f"train_{metric}", ":", 1.5, "training"),
                                   (f"val_{metric}", "-", 2.0, "validation")):
            v = scale * np.array([[r[key] for r in h] for h in hs])
            ax.plot(ep, v.mean(0), color=COLORS[name], ls=ls, lw=lw, label=label)
            if key.startswith("val"):
                ax.fill_between(ep, v.min(0), v.max(0), color=COLORS[name], alpha=0.12, lw=0)
        ax.grid(alpha=0.3)
    axes[0, col].set_title(f"{name}: paper, full settings ({len(hs)} seeds)", loc="left", fontsize=10)
    axes[1, col].set_xlabel("epoch")
axes[0, 0].set_ylabel("cross-entropy loss")
axes[1, 0].set_ylabel("accuracy (%)")
axes[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.show()
""")
md("## The released classifiers")
md("""
The released classifiers are seed 0 of each condition, trained with the full settings. Scoring them on all validation
and test images reproduces the paper's numbers (tiny differences in the last digit can come from the GPU and the
16-bit format). This is a re-computation of the paper's one-time test evaluation, not a new selection.
""")
code("""
from safetensors.torch import load_file

PAPER_TEST_ACC = {"Real only": 0.8708, "50% synthetic": 0.8455, "Synthetic only": 0.6719}  # seed 0, one-time evaluation
files = {"Real only": "resnet18_real_only_seed0", "50% synthetic": "resnet18_mix50_seed0",
         "Synthetic only": "resnet18_synthetic_only_seed0"}
print(f"{'released classifier':<22}{'val loss':>10}{'val acc':>10}{'test loss':>11}{'test acc':>10}{'paper test acc':>16}"
      f"{'this run (' + ('FAST' if FAST else 'full') + ') test acc':>27}")
for name, fname in files.items():
    model = build_classifier(pretrained=False)
    model.load_state_dict(load_file(WEIGHTS_DIR / f"{fname}.safetensors"))
    vl, va = evaluate(model, val_x, val_y)
    tl, ta = evaluate(model, test_x, test_y)
    print(f"{name:<22}{vl:>10.3f}{va:>10.3f}{tl:>11.3f}{ta:>10.3f}{PAPER_TEST_ACC[name]:>16.4f}"
          f"{SUMMARY[name]['testing_accuracy']:>27.3f}")
    del model
""")

md("# Summary")
md("""
In every run the training, validation and test losses fell together, the validation and test curves agree, and the
seeds agree; the full-length runs show them levelling off. Augmentation keeps the held-out losses close to the training loss. Trained on real
images the classifier generalises best; replacing half of them with synthetic galaxies costs little, but training on
synthetic images alone leaves a large gap on real galaxies: the paper's main finding.
""")
code("""
TIMES["whole notebook"] = time.time() - T0
for step, seconds in TIMES.items():
    print(f"  {step:<26} {seconds / 60:5.1f} min")
print("peak GPU memory:", {k: f"{v:.1f} GiB" for k, v in MEMORY.items()})
""")

nb.write("06_Full_Training_Tutorial.ipynb", gpu=True, fill=FILL)

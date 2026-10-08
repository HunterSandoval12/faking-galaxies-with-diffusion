# Faking Galaxies with Diffusion: A Real-vs-Synthetic Benchmark for Morphology Classification

Hunter Sandoval, University of New Mexico.

Can a diffusion model, fine-tuned to generate galaxy imagery from scratch, produce training data realistic
enough that a classifier trained on it performs comparably to one trained on real images? A pretrained
Stable Diffusion 1.5 model is fine-tuned with LoRA on Galaxy10 DECaLS to generate class-conditional
galaxies; ResNet-18 classifiers are trained on real, synthetic and mixed data and evaluated once on a
fixed, held-out set of real images.

**Paper:** [Faking Galaxies with Diffusion (PDF)](paper/Faking_Galaxies_with_Diffusion.pdf) - 5-page main text, references and appendix (17 pages).

## Research questions

- **Primary:** Does a classifier trained on synthetic data - instead of real data - perform comparably on held-out
  real images? At what mix of real and synthetic data does performance peak?
- **Secondary:** Does the optimal real-to-synthetic ratio differ between a well-represented class (Round Smooth)
  and a severely underrepresented one (Cigar-Shaped Smooth)?

## Results (held-out test set, evaluated once; 95% bootstrap CIs)

| Question | Answer |
|---|---|
| Comparable? | Not yet: synthetic-only 66.9% vs real-only 86.6% accuracy (77% of real) |
| Where does the mix peak? | Real only; up to ~10% synthetic costs nothing detectable, more costs accuracy |
| Adding synthetic to all real data? | No measurable effect |
| When does synthetic help? | When real data is scarce: +2.2 points [+1.2, +3.1] with only 10% of the real data |
| Rare vs common class? | No measurable effect for either class at the single-class level |

ROC AUC (validation set, one-vs-rest, macro over classes): real-only 0.983, 50% synthetic 0.978, synthetic-only
0.927 (the one-time test evaluation saved predicted classes only, so AUC is reported on validation; Fig. 18, Tab. 14).

Inference speed on one RTX 5090: the generator takes 0.92 s per image at batch size 1 (0.48 s per image at
batch 20; 50.8 TFLOPs per image); the ResNet-18 classifier takes 1.7 ms per image at batch size 1, end-to-end
(4.7 GFLOPs).

Figures, tables and their captions: [`results/README.md`](results/README.md). Real vs. synthetic galaxies for every
class: [`results/figures/fig02_real_vs_synthetic_samples.png`](results/figures/fig02_real_vs_synthetic_samples.png).

## Google Colab tutorials

Step-by-step notebooks that run in Google Colab (index and requirements: [`notebooks/README.md`](notebooks/README.md)):

| # | Tutorial | Open |
|---|---|---|
| 1 | Dataset: download, inputs and labels, the training / validation / testing directories, real vs. synthetic | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HunterSandoval12/faking-galaxies-with-diffusion/blob/main/notebooks/01_Dataset_Tutorial.ipynb) |
| 2 | Model description: the ResNet-18 classifier and the diffusion generator, layer by layer | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HunterSandoval12/faking-galaxies-with-diffusion/blob/main/notebooks/02_Model_Description_Tutorial.ipynb) |
| 3 | Model optimization: losses, AdamW, learning-rate search and schedules, batch sizes, early stopping, training vs. validation curves | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HunterSandoval12/faking-galaxies-with-diffusion/blob/main/notebooks/03_Model_Optimization_Tutorial.ipynb) |
| 4 | Basic testing: classify one test galaxy and generate one synthetic galaxy | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HunterSandoval12/faking-galaxies-with-diffusion/blob/main/notebooks/04_Basic_Testing_Tutorial.ipynb) |
| 5 | Basic fine-tuning: fine-tune the ImageNet ResNet-18 and Stable Diffusion 1.5 (LoRA); the loss decreases (FAST mode by default) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HunterSandoval12/faking-galaxies-with-diffusion/blob/main/notebooks/05_Basic_Finetuning_Tutorial.ipynb) |
| 6 | Full training: generator, generation, real / mixed / synthetic classifiers over several seeds, validation and test convergence, augmentation (FAST mode by default) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HunterSandoval12/faking-galaxies-with-diffusion/blob/main/notebooks/06_Full_Training_Tutorial.ipynb) |

## Pretrained weights and synthetic set (GitHub Release `weights-v1`)

The trained models, the synthetic training set and the training logs are attached to the release
[`weights-v1`](https://github.com/HunterSandoval12/faking-galaxies-with-diffusion/releases/tag/weights-v1) and
download with plain `wget` (no login):

```bash
wget https://github.com/HunterSandoval12/faking-galaxies-with-diffusion/releases/download/weights-v1/resnet18_real_only_seed0.safetensors
wget https://github.com/HunterSandoval12/faking-galaxies-with-diffusion/releases/download/weights-v1/SHA256SUMS.txt
sha256sum -c --ignore-missing SHA256SUMS.txt
```

| File | Contents | Size | SHA-256 |
|---|---|---|---|
| `resnet18_real_only_seed0.safetensors` | ResNet-18 classifier trained on the 12,330 real training images (the baseline) | 42.7 MiB | `ad9a4e605ef50154a6f3e377c23b4f95fae3ca6712b9a5815b4dbfcdc66f40dd` |
| `resnet18_synthetic_only_seed0.safetensors` | ResNet-18 trained on 12,330 synthetic images only | 42.7 MiB | `47c6b1be788acd2ed8283126337f7cd40585352900a38eaeafc2eb2c647d0afd` |
| `resnet18_mix50_seed0.safetensors` | ResNet-18 trained on 50% real + 50% synthetic images | 42.7 MiB | `3049c05fd484655613337b3290adb4c8d2cb013a08c07297679cef745beb24f3` |
| `lora_unet.safetensors` | The generator's LoRA weights (rank 8, U-Net attention layers; checkpoint 10,000) | 6.1 MiB | `9b866502137e2b9ac7783f84b9a8110578a51aeb60c93779e03627bb0faeb20a` |
| `class_embeddings.safetensors` | The generator's learned class-embedding table (11 x 77 x 768; row 10 = null class) | 2.5 MiB | `87a7fb2720dc4a0dedb66c3417265b63d02ff7b829ea86465dfc11ee2468a2cb` |
| `lora_config.json` | The generator's training configuration (rank, alpha, target layers, ...) | 2 KiB | `29dfa782205ee5875677d8cfb358c08c8aab8290442fb1526abcab5ba7b8baf1` |
| `synthetic_examples.zip` | Sample indices 0-3 of every class of the final synthetic set (the synthetic images of Fig. 2) | 5.9 MiB | `eb108eb7029d6856af54899bcb7e54296c798f1cde0bc4ba9ff3bb1e6db6f5d5` |
| `synthetic_set.zip` | The full synthetic training set: 12,330 PNG images (the paper's `main_cfg3_steps30`; per class as many as the real training images) + `manifest.csv` (class, sample index, seed) | 1,878,505,895 bytes (1.75 GiB) | `b2e19570046a86827d821aaf49e7e36353217963b1240fe4e7aa650447f38948` |
| `training_logs.zip` | Per-epoch logs of the classifier runs (incl. the learning-rate search) and the LoRA run (also in `results/training_logs/`) | 193 KiB | `f945a827dae1c68d3e534f92472f17b86ba136bfa541dccdf8b36392583a24d9` |
| `LICENSE-CreativeML-OpenRAIL-M.txt` | The license of the LoRA weights | 14 KiB | `be351ebe7ac01bcdbb018639aadcfd38f136b7dc3f2a3d4d3a24db51d1b210ef` |

The same checksums are in [`weights/SHA256SUMS.txt`](weights/SHA256SUMS.txt). The classifiers were selected on the
validation set (best epoch); they contain weights only (safetensors, no pickled code) and were built from the training
runs by [`tools/make_release_assets.py`](tools/make_release_assets.py).

**Base model.** The generator needs the Stable Diffusion v1.5 base weights, which download from Hugging Face:
[`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
The project's code names the original `runwayml/stable-diffusion-v1-5`, which was taken down in 2024 and now redirects
to that mirror; its U-Net and VAE files are byte-identical (SHA-256) to the ones used in this project, and its fp16
files equal the fp32 weights cast to fp16.

**Licenses of the weights.**
- The **LoRA weights** (`lora_unet.safetensors`, `class_embeddings.safetensors`) are derived from Stable Diffusion
  v1.5 and are distributed under the **CreativeML Open RAIL-M** license
  ([`licenses/CreativeML-OpenRAIL-M.txt`](licenses/CreativeML-OpenRAIL-M.txt)); its use-based restrictions
  (Attachment A) apply to them and to any use of the generator.
- The **synthetic images** (`synthetic_set.zip`, `synthetic_examples.zip`) were generated with the project's Stable
  Diffusion 1.5-based generator, so the CreativeML Open RAIL-M license applies to them as well.
- The **classifiers** were fine-tuned from torchvision's ImageNet-pretrained ResNet-18 weights
  (`ResNet18_Weights.IMAGENET1K_V1`, BSD-3-Clause) on Galaxy10 DECaLS; they are released under the MIT license of
  this repository, and the dataset's own terms apply to the data they were trained on.

## Repository layout

| Path | Contents |
|---|---|
| `data/prepare_splits.py` | Stratified 70/15/15 split; removes exact duplicates and overlapping cutouts; `data/splits/` holds the split indices |
| `diffusion/` | LoRA fine-tuning (`train_lora.py`), sampling (`sample.py`), FID/KID evaluation (`eval_fid.py`), tuning drivers, final generation (`generate_synthetic.py`) and checks (`check_synthetic.py`) |
| `classifier/` | ResNet-18 training with real/synthetic mixing (`train_classifier.py`), class fidelity, experiment driver (`run_experiments.py`), the one-time test evaluation (`evaluate_test.py`), re-runs with extra logging (`reproduce_runs.py`) |
| `paper/` | The paper (PDF) |
| `analysis/` | Figures, tables and statistics for the paper |
| `results/` | 18 figures (PNG + PDF + CSV data), 14 tables (CSV + Markdown + LaTeX) and the training logs (`results/training_logs/`) |
| `notebooks/` | Google Colab tutorials (built by the scripts in `notebooks/_build/`) |
| `tools/` | `make_release_assets.py`: builds the release files and their checksums |
| `licenses/` | License of the LoRA weights (CreativeML Open RAIL-M) |

Training runs, the 18,561 generated images and caches (~23 GB) are not included; they are regenerated by the
pipeline below. The chosen generator and three classifiers are in the release.

## Setup

Tested on Windows 11, Python 3.14, one NVIDIA RTX 5090 (32 GB).

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Galaxy10 DECaLS (2.55 GiB) downloads automatically through astroNN on first use.

## Reproducing the pipeline

Run from the project root, in order (times on an RTX 5090):

1. **Split:** `python data/prepare_splits.py` (~2 min; deterministic, seed 42).
2. **Generator:** `python diffusion/train_lora.py --output-dir runs/lora_r8` (~1 h). Tuning on the validation
   set: `diffusion/eval_fid.py` (checkpoint / guidance / steps scans) and `diffusion/tune_sweep.py`
   (LoRA rank / learning rate / seed sweep).
3. **Classifier learning rate** (validation): `python classifier/tune_classifier.py`.
4. **Sampling pilots** (validation): `classifier/guidance_pilot.ps1`, `classifier/steps_pilot.ps1`.
5. **Synthetic set:** `diffusion/main_generation.ps1` (12,330 images, guidance 3, 30 steps, ~1.7 h, then
   integrity / realism / memorization checks).
6. **Experiments:** `python classifier/run_experiments.py` (all classifiers, validation only), ending in the
   one-time test evaluation `classifier/evaluate_test.py --split test --confirm-test-set`.
7. **Figures and tables:** `python analysis/compute_extras.py`, `python classifier/reproduce_runs.py`
   (learning curves; validation only), `python analysis/model_stats.py` (parameters, FLOPs),
   `python analysis/make_figures.py`, `python analysis/make_tables.py`.

## Methodological safeguards

- The test set is read by one script only (`classifier/evaluate_test.py`, behind `--confirm-test-set`) and was
  evaluated **once**, with the analysis fixed in advance. All tuning and selection used the validation set.
- Galaxy10 DECaLS contains 121 pixel-identical duplicates with conflicting labels (removed before splitting) and
  overlapping cutouts of the same sky across splits (69 validation/test images removed after splitting).
- Generated images are checked for memorization: none of the 18,561 is closer to a training image than the
  closest unseen real image.
- Everything runs with fixed seeds; re-runs reproduce results bit-for-bit.
- Robustness (validation only): training the classifiers for 50 instead of 30 epochs changes the
  real-vs-synthetic gap by about 1 point (20.1 -> 19.1), so the conclusion does not depend on the schedule
  (`results/tables/t12_robustness_epochs`).

## License, code provenance and citations

The code in this repository is released under the [MIT license](LICENSE). The model weights and the data have their
own licenses (see *Licenses of the weights* above and the table below).

All code in this repository was written for this project, with Claude Code (Anthropic) used as an AI-assisted
development tool. The LoRA fine-tuning loop follows the structure of the Hugging Face diffusers example
`examples/text_to_image/train_text_to_image_lora.py` (Apache-2.0). `analysis/validate_palette.py` is a Python port of
a color-blind-safety palette validator supplied with Claude Code's data-visualization guidance.

**Related work.** The closest prior work is GalCatDiff [18], a diffusion model that generates galaxy images per
morphology category (with category embeddings) and evaluates how realistic they are (e.g. their color and size
distributions); this project instead measures whether class-conditional synthetic galaxies can train a classifier.

**Model weights and data (not original to this project):**

| Item | Source | License / terms |
|---|---|---|
| Stable Diffusion v1.5 weights | Released by RunwayML (originally `runwayml/stable-diffusion-v1-5`, now mirrored at `stable-diffusion-v1-5/stable-diffusion-v1-5`), building on the latent diffusion work of CompVis and Stability AI [1] | CreativeML Open RAIL-M |
| CLIP text encoder (used once, to initialize the class embeddings) | Part of Stable Diffusion v1.5 [2] | as above |
| ResNet-18 ImageNet weights (starting point of the classifiers) | torchvision `ResNet18_Weights.IMAGENET1K_V1` [3] | torchvision (BSD-3-Clause) |
| Inception-v3 FID weights | torch-fidelity `weights-inception-2015-12-05` [4, 5] | Apache-2.0 |
| Galaxy10 DECaLS | Leung & Bovy [6], images from the DESI Legacy Imaging Surveys [7], labels from Galaxy Zoo DECaLS [8] | see the dataset source |

**Software** (versions pinned in `requirements.txt`; licenses as declared in the installed package metadata):
PyTorch 2.14 and torchvision 0.29 [9], diffusers 0.40 (Apache-2.0), transformers 5.17 (Apache-2.0) [10],
peft 0.21 (Apache-2.0), accelerate 1.15 (Apache-2.0), safetensors 0.8 (Apache-2.0), torchmetrics 1.9
(Apache-2.0), torch-fidelity 0.4 (Apache-2.0), astroNN 1.1 (MIT) [11], scikit-learn 1.9 (BSD-3-Clause) [12],
NumPy, SciPy, h5py (BSD-3-Clause), Matplotlib, Pillow.

**Methods used:** LoRA [13], classifier-free guidance [14], DPM-Solver++ [15], FID [4], KID [16], color-vision
deficiency simulation [17].

1. R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022.
2. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021.
3. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR 2016.
4. M. Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," NeurIPS 2017.
5. C. Szegedy et al., "Rethinking the Inception Architecture for Computer Vision," CVPR 2016.
6. H. W. Leung and J. Bovy, "Galaxy10 DECaLS," 2019. https://github.com/henrysky/Galaxy10
7. A. Dey et al., "Overview of the DESI Legacy Imaging Surveys," AJ 157, 168, 2019.
8. M. Walmsley et al., "Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314,000 galaxies," MNRAS 509, 2022.
9. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019.
10. T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," EMNLP 2020 (demos).
11. H. W. Leung and J. Bovy, "Deep learning of multi-element abundances from high-resolution spectroscopic data," MNRAS 483, 2019 (astroNN).
12. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR 12, 2011.
13. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
14. J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," NeurIPS 2021 Workshop.
15. C. Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models," arXiv:2211.01095, 2022.
16. M. Bińkowski, D. J. Sutherland, M. Arbel, A. Gretton, "Demystifying MMD GANs," ICLR 2018.
17. G. M. Machado, M. M. Oliveira, L. A. F. Fernandes, "A Physiologically-based Model for Simulation of Color Vision Deficiency," IEEE TVCG 15(6), 2009.
18. X. Fan et al., "Category-based Galaxy Image Generation via Diffusion Models" (GalCatDiff), arXiv:2506.16255, 2025.

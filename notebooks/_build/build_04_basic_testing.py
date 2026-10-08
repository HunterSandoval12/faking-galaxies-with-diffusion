"""Builds notebooks/04_Basic_Testing_Tutorial.ipynb (run from the repository root).

Spec item #4 (Basic Testing): load the pretrained model with wget and run it on one test example.
Here: the ResNet-18 classifiers (real-only, synthetic-only, 50% mix) on one held-out test galaxy from
the paper's Fig. 17, plus one image generated with the LoRA generator (sample 0 of Cigar-Shaped Smooth
of the final synthetic set, regenerated from its seed) and classified.

Usage: python notebooks/_build/build_04_basic_testing.py [--times times.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbcommon import REPO_URL, SD_REPO, Notebook, download_cells, drive_cells, install_cells  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--times", type=Path, help="JSON from a local test run: step times (s), peak memory, results")
args = ap.parse_args()


def fmt(seconds):
    return f"~{seconds:.0f} s" if seconds < 90 else f"~{seconds / 60:.0f} min"


KEYS = ("__T_WEIGHTS__", "__T_CLASSIFY__", "__T_SD__", "__T_GENERATE__", "__T_GENERATE_CPU__", "__T_TOTAL__",
        "__T_TOTAL_CPU__", "__PEAK_RAM__", "__PEAK_VRAM__", "__P_REAL__", "__CORR__", "__GEN_PRED__")
FILL = {k: "?" for k in KEYS}
if args.times:
    m = json.loads(args.times.read_text())
    FILL.update({"__T_WEIGHTS__": fmt(m["download weights"]), "__T_CLASSIFY__": fmt(m["classify"]),
                 "__T_SD__": fmt(m["download + load Stable Diffusion"]), "__T_GENERATE__": fmt(m["generate"]),
                 "__T_GENERATE_CPU__": fmt(m["generate (CPU)"]), "__T_TOTAL__": fmt(m["whole notebook"]),
                 "__T_TOTAL_CPU__": fmt(m["whole notebook (CPU)"]), "__PEAK_RAM__": f"{m['peak_ram_gib']:.1f} GiB",
                 "__PEAK_VRAM__": f"{m['peak_vram_gib']:.1f} GiB", "__P_REAL__": f"{m['p_real']:.2f}",
                 "__CORR__": f"{m['corr']:.4f}", "__GEN_PRED__": m["gen_pred"]})

if args.times:
    FILL.update(m.get("fill", {}))  # hand-worded cells (e.g. download times measured from a local mirror)

nb = Notebook()
md, code = nb.md, nb.code

TEST_INDEX, TEST_LABEL = 7802, 4          # first Cigar-Shaped Smooth galaxy of the paper's Fig. 17
TEST_RA, TEST_DEC, TEST_PX = 233.462311, 39.474945, 0.262
PAPER_PREDICTIONS = {"real only": 4, "synthetic only": 4, "50% mix": 4}  # saved one-time test predictions, seed 0

# ================================================================================ title + scope
md("# Galaxy10 DECaLS: Basic Testing Tutorial")
md("""
**Project:** *Faking Galaxies with Diffusion: A Real-vs-Synthetic Benchmark for Morphology Classification*
Hunter Sandoval, University of New Mexico. Tutorial 4 of 6.

**Scope.** Download the pretrained models with `wget` and run them on **one test example**:

1. the **classifier** (ResNet-18 trained on real images, the paper's baseline) classifies one galaxy from the
   held-out `testing` set, and the two other released classifiers (trained on synthetic images only, and on a 50/50
   mix) classify the same galaxy, reproducing that galaxy's row of the paper's Fig. 17;
2. the **generator** (Stable Diffusion 1.5 + the project's LoRA weights) generates **one synthetic galaxy**, image
   number 0 of the class *Cigar-Shaped Smooth* in the paper's synthetic training set, regenerated from its random
   seed, and the classifier is asked what it sees.

Tutorial 2 describes both models layer by layer; this tutorial only uses them.
""")
md("## References")
md(f"""
1. K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016, arXiv:1512.03385 (ResNet).
2. R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022, arXiv:2112.10752
   (Stable Diffusion); weights: https://huggingface.co/{SD_REPO}
3. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022, arXiv:2106.09685.
4. J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," arXiv:2207.12598, 2022.
5. C. Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models," arXiv:2211.01095,
   2022 (the sampler).
6. Galaxy10 DECaLS: https://github.com/henrysky/Galaxy10 ; images: DESI Legacy Imaging Surveys,
   https://www.legacysurvey.org (A. Dey et al., AJ 157, 168, 2019, arXiv:1804.08657).
7. Project code and model weights: {REPO_URL}
""")

# ================================================================================ setup
md("# Setup")
install_cells(nb, {"torch": "2.0", "torchvision": "0.15", "diffusers": "0.27", "peft": "0.10", "safetensors": "0.4",
                   "numpy": "1.23", "pillow": "9.0", "matplotlib": "3.6"}, """
| Python | 3.10 | 3.14.7 | (Colab's default) | |
| PyTorch (`torch`) | 2.0 | 2.14.0 | `pip install "torch>=2.0"` | running the models |
| torchvision | 0.15 | 0.29.0 | `pip install "torchvision>=0.15"` | the ResNet-18 architecture |
| diffusers | 0.27 | 0.40.0 | `pip install "diffusers>=0.27"` | Stable Diffusion's VAE, U-Net and sampler |
| peft | 0.10 | 0.21.0 | `pip install "peft>=0.10"` | LoRA adapters |
| safetensors | 0.4 | 0.8.0 | `pip install "safetensors>=0.4"` | reading the weight files |
| NumPy | 1.23 | 2.5.2 | `pip install "numpy>=1.23"` | arrays |
| Pillow | 9.0 | 12.3.0 | `pip install "pillow>=9.0"` | images |
| Matplotlib | 3.6 | 3.11.2 | `pip install "matplotlib>=3.6"` | figures |
""")
md("## Hardware used for running")
md("""
- **GPU recommended** for the generator: in Colab choose *Runtime > Change runtime type > T4 GPU* (the notebook asks
  for it when opened from GitHub). It needs about __PEAK_VRAM__ of GPU memory.
- **CPU works too**: the classifier is fast on a CPU, and the generator takes a few minutes instead of seconds. On a
  CPU the random noise is drawn by the CPU's generator, so the generated galaxy differs from the paper's (still a
  Cigar-Shaped Smooth galaxy).
- **Memory (RAM) at peak:** about __PEAK_RAM__.
- **Tested on:** a Windows 11 PC with an NVIDIA RTX 5090 GPU (32 GB), an AMD Ryzen 7 9800X3D CPU and 32 GB of RAM,
  Python 3.14; every code cell was run on the GPU and with the GPU hidden (CPU only).
""")
code("""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32  # the generator's precision (as in the project)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB)")
else:
    print("No GPU found: running on the CPU (the generator is slower). In Colab: Runtime > Change runtime type > T4 GPU.")
""")
md("## Approximate execution times")
md("""
| Step | Test PC (RTX 5090) | Test PC, CPU only | Colab T4 GPU (estimate) |
|---|---|---|---|
| Download the project weights (wget, 143 MiB) | __T_WEIGHTS__ | same | < 1 min |
| Classify the test galaxy (3 classifiers) | __T_CLASSIFY__ | < 5 s | < 5 s |
| Download + load Stable Diffusion (1.9 GB, first run only) | __T_SD__ | same | 1-3 min |
| Generate one galaxy (30 steps) | __T_GENERATE__ | __T_GENERATE_CPU__ | ~10 s |
| **Whole tutorial** | **__T_TOTAL__** | **__T_TOTAL_CPU__** | **~3-5 min** |
""")
md("## Expected results")
md("""
- All downloaded files pass their SHA-256 checks.
- The test galaxy (dataset index 7802, true class *Cigar-Shaped Smooth*) is classified as **Cigar-Shaped Smooth** by
  all three classifiers, as in the paper's saved test predictions; the real-only classifier gives it a probability of
  about __P_REAL__.
- The regenerated synthetic galaxy matches image 0 of the paper's synthetic Cigar-Shaped Smooth set: pixel correlation
  __CORR__ on the test PC's GPU (not exactly 1, because the set was generated 20 images at a time and the GPU's
  arithmetic differs slightly with batch size). A different GPU type (for example Colab's T4) gives a visually
  identical image with a correlation just below that; the CPU gives a different galaxy.
- The real-only classifier labels the regenerated galaxy as __GEN_PRED__.
""")

# ================================================================================ drive + downloads
drive_cells(nb, "The test galaxy comes from Tutorial 1's `testing` directory.")
md("# Download the pretrained models")
download_cells(nb, ["resnet18_real_only_seed0.safetensors", "resnet18_synthetic_only_seed0.safetensors",
                    "resnet18_mix50_seed0.safetensors", "lora_unet.safetensors", "class_embeddings.safetensors",
                    "lora_config.json", "synthetic_examples.zip"],
               intro="The next cell downloads the three classifiers, the generator's trained parts, and the sample of "
                     "the synthetic set (to compare the regenerated galaxy with).")

# ================================================================================ test image
md("# One test example")
md(f"""
The test galaxy is **dataset index {TEST_INDEX}**, the first *Cigar-Shaped Smooth* galaxy of the paper's Fig. 17. That
figure's galaxies were drawn at random from the test set before any prediction was looked at, so the example was not
picked for being easy or hard. Change `TEST_INDEX` and `TEST_LABEL` to try another test galaxy (the file names in
Tutorial 1's directories are the dataset indices).

The image is read from Tutorial 1's zip file on Drive. If that file is missing, the cell downloads the same galaxy
from the Legacy Surveys cutout service instead (Tutorial 1, *Extending the dataset*); that JPEG differs very slightly
from the Galaxy10 pixels.
""")
code(f"""
import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLASS_NAMES = ["Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar-Shaped Smooth",
               "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge",
               "Edge-on with Bulge"]
TEST_INDEX, TEST_LABEL = {TEST_INDEX}, {TEST_LABEL}
TEST_RA, TEST_DEC, TEST_PXSCALE = {TEST_RA}, {TEST_DEC}, {TEST_PX}  # sky position, for the fallback download

member = f"galaxy10/testing/{{TEST_LABEL}}_{{CLASS_NAMES[TEST_LABEL].replace(' ', '_')}}/{{TEST_INDEX:05d}}.png"
if ZIP_PATH.exists():
    with zipfile.ZipFile(ZIP_PATH) as z:
        test_image = np.asarray(Image.open(io.BytesIO(z.read(member))).convert("RGB"))
    source = f"Tutorial 1's testing directory ({{member}})"
else:
    print("Tutorial 1's zip file was not found: downloading the galaxy from the Legacy Surveys instead.")
    url = (f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={{TEST_RA}}&dec={{TEST_DEC}}"
           f"&layer=ls-dr8&pixscale={{TEST_PXSCALE}}&size=256")
    cutout = WORK_ROOT / "test_galaxy.jpg"
    !wget -q -O "{{cutout}}" "{{url}}"
    test_image = np.asarray(Image.open(cutout).convert("RGB"))
    source = "Legacy Surveys cutout (DR8)"
print(f"test image: {{test_image.shape}} {{test_image.dtype}}, true class: {{CLASS_NAMES[TEST_LABEL]}}; from {{source}}")
plt.figure(figsize=(3.2, 3.4))
plt.imshow(test_image)
plt.title(f"test galaxy {{TEST_INDEX}}\\ntrue class: {{CLASS_NAMES[TEST_LABEL]}}", fontsize=9)
plt.axis("off")
plt.show()
""")

# ================================================================================ classify
md("# Classify the test galaxy")
md("""
Loading and preparing the input are the same as in Tutorial 2: torchvision's ResNet-18 with a 10-class last layer,
filled from the downloaded file, and the image scaled to 0-1 and normalized with the ImageNet mean and standard
deviation (`to_input` from `classifier/train_classifier.py`). `model.eval()` and `torch.no_grad()` switch off
training behavior and gradient bookkeeping. The softmax turns the 10 outputs into probabilities.
""")
code(f"""
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torchvision.models import resnet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_classifier(path):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(load_file(path))
    return model.to(DEVICE).eval()


def to_input(images_u8):
    \"\"\"uint8 images [N, 256, 256, 3] -> normalized float tensor [N, 3, 256, 256].\"\"\"
    x = torch.as_tensor(np.array(images_u8)).permute(0, 3, 1, 2).float() / 255.0  # np.array: a writable copy
    return (x - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def class_probabilities(model, image_u8):
    return F.softmax(model(to_input(image_u8[None]).to(DEVICE)).float(), dim=1)[0].cpu()


t = time.time()
classifiers = {{"real only": load_classifier(WEIGHTS_DIR / "resnet18_real_only_seed0.safetensors"),
                "synthetic only": load_classifier(WEIGHTS_DIR / "resnet18_synthetic_only_seed0.safetensors"),
                "50% mix": load_classifier(WEIGHTS_DIR / "resnet18_mix50_seed0.safetensors")}}
PAPER_PREDICTIONS = {PAPER_PREDICTIONS!r}  # the paper's saved test predictions for galaxy {TEST_INDEX} (seed 0 models)
probabilities = {{name: class_probabilities(model, test_image) for name, model in classifiers.items()}}
TIMES["classify"] = time.time() - t

for name, p in probabilities.items():
    top = p.topk(3)
    pred = int(top.indices[0])
    ranking = ", ".join(f"{{CLASS_NAMES[int(i)]}} {{float(v):.3f}}" for v, i in zip(top.values, top.indices))
    same = pred == PAPER_PREDICTIONS[name] if TEST_INDEX == {TEST_INDEX} else None
    print(f"{{name:<15}} -> {{CLASS_NAMES[pred]:<22}} {{'correct' if pred == TEST_LABEL else 'WRONG'}}"
          f"{{'' if same is None else f', same as the paper: {{same}}'}}   top 3: {{ranking}}")
p_real = float(probabilities["real only"][TEST_LABEL])
""")
code("""
fig, ax = plt.subplots(figsize=(8, 3.6))
width = 0.27
for k, (name, p) in enumerate(probabilities.items()):
    ax.bar(np.arange(10) + (k - 1) * width, p.numpy(), width, label=f"trained on {name}")
ax.set_xticks(np.arange(10))
ax.set_xticklabels([n.replace(" ", "\\n", 1) for n in CLASS_NAMES], fontsize=7)
ax.set_ylabel("probability")
ax.set_ylim(0, 1)
ax.set_title(f"Class probabilities for test galaxy {TEST_INDEX} (true class: {CLASS_NAMES[TEST_LABEL]})", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
""")

# ================================================================================ generate
md("# Generate one synthetic galaxy")
md("""
The generator is loaded as in Tutorial 2: the Stable Diffusion v1.5 VAE and U-Net from Hugging Face, plus the
project's LoRA weights and class-embedding table (`load_lora_checkpoint` from `diffusion/sample.py`).

**How we figured this out.** The project generated its synthetic set with `diffusers`' `StableDiffusionPipeline`,
passing a class table row instead of a text prompt. The function `generate` below writes out the same steps by hand:
start from Gaussian noise drawn with a fixed seed, run the DPM-Solver++ sampler for 30 steps, and at every step predict
the noise twice, for the class and for the null class, and combine the two with classifier-free guidance (scale 3);
then decode with the VAE. On the test PC this loop gives exactly the pipeline's pixels. The project's image `k` of
class `c` used the seed `7,000,000,000 + 1,000,000 c + k`, so image 0 of Cigar-Shaped Smooth (class 4) can be
regenerated from seed 7,004,000,000 and compared with the copy in `synthetic_examples.zip`. Hugging Face may warn
that no `HF_TOKEN` is set: no token or account is needed.
""")
code(f"""
import json

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict

SD_REPO = "{SD_REPO}"
t = time.time()
vae = AutoencoderKL.from_pretrained(SD_REPO, subfolder="vae", variant="fp16", torch_dtype=DTYPE).to(DEVICE).eval()
unet = UNet2DConditionModel.from_pretrained(SD_REPO, subfolder="unet", variant="fp16", torch_dtype=DTYPE).to(DEVICE).eval()
scheduler = DPMSolverMultistepScheduler.from_pretrained(SD_REPO, subfolder="scheduler")
lora_config = json.loads((WEIGHTS_DIR / "lora_config.json").read_text())


def load_lora_checkpoint(unet, weights_dir, cfg):
    \"\"\"Add LoRA adapters to `unet` and load the trained weights (from diffusion/sample.py). Returns the class table.\"\"\"
    unet.add_adapter(LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], init_lora_weights="gaussian",
                                target_modules=cfg["lora_target_modules"]))
    result = set_peft_model_state_dict(unet, load_file(weights_dir / "lora_unet.safetensors"))
    missing = [k for k in result.missing_keys if "lora" in k]
    assert not result.unexpected_keys and not missing, "LoRA weights do not match the U-Net"
    return load_file(weights_dir / "class_embeddings.safetensors")["table"]


class_table = load_lora_checkpoint(unet, WEIGHTS_DIR, lora_config).to(DEVICE, DTYPE)
NULL_CLASS = lora_config["null_class_index"]
TIMES["download + load Stable Diffusion"] = time.time() - t
print(f"generator ready in {{TIMES['download + load Stable Diffusion']:.0f}} s")
""")
code("""
@torch.no_grad()
def generate(label, seed, guidance=3.0, steps=30, resolution=512):
    \"\"\"One class-conditional image, the same computation as the project's StableDiffusionPipeline call.\"\"\"
    scheduler.set_timesteps(steps, device=DEVICE)
    noise_gen = torch.Generator(DEVICE).manual_seed(seed)
    latents = torch.randn((1, 4, resolution // 8, resolution // 8), generator=noise_gen, device=DEVICE,
                          dtype=DTYPE) * scheduler.init_noise_sigma
    conditions = torch.cat([class_table[NULL_CLASS][None], class_table[label][None]])  # [null class, class]
    for t in scheduler.timesteps:
        model_input = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        eps_null, eps_class = unet(model_input, t, encoder_hidden_states=conditions).sample.chunk(2)
        eps = eps_null + guidance * (eps_class - eps_null)          # classifier-free guidance
        latents = scheduler.step(eps, t, latents).prev_sample
    x = vae.decode(latents / vae.config.scaling_factor).sample      # latent -> image in [-1, 1]
    x = (x / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((x * 255).round().astype(np.uint8))


GEN_LABEL, GEN_K = 4, 0  # image 0 of Cigar-Shaped Smooth
seed = 7_000_000_000 + GEN_LABEL * 1_000_000 + GEN_K
t = time.time()
generated_512 = generate(GEN_LABEL, seed)
TIMES["generate"] = time.time() - t
generated = np.asarray(generated_512.resize((256, 256), Image.LANCZOS))  # the project's 512 -> 256 step
print(f"generated {generated_512.size} -> {generated.shape} in {TIMES['generate']:.1f} s (seed {seed:,})")
""")
code("""
with zipfile.ZipFile(WEIGHTS_DIR / "synthetic_examples.zip") as z:
    name = next(n for n in z.namelist() if f"/{GEN_LABEL}_" in n and f"/sample{GEN_K}_seed{seed}.png" in n)
    paper_image = np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
corr = float(np.corrcoef(generated.astype(float).ravel(), paper_image.astype(float).ravel())[0, 1])
diff = np.abs(generated.astype(int) - paper_image.astype(int))
print(f"regenerated vs the paper's synthetic image: pixel correlation {corr:.4f}, mean difference {diff.mean():.2f} "
      f"(of 255)")

p_gen = class_probabilities(classifiers["real only"], generated)
gen_pred = CLASS_NAMES[int(p_gen.argmax())]
print(f"real-only classifier on the regenerated galaxy: {gen_pred} ({float(p_gen.max()):.3f}); "
      f"P({CLASS_NAMES[GEN_LABEL]}) = {float(p_gen[GEN_LABEL]):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5))
for ax, img, title in zip(axes, (test_image, paper_image, generated),
                          (f"real test galaxy {TEST_INDEX}\\n({CLASS_NAMES[TEST_LABEL]})",
                           f"paper's synthetic image\\n({CLASS_NAMES[GEN_LABEL]}, sample {GEN_K})",
                           f"regenerated now\\n(pixel correlation {corr:.4f})")):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()
""")
md("""
In the paper, a classifier trained on real images recognizes only 28% of the synthetic Cigar-Shaped Smooth images as
Cigar-Shaped Smooth (the lowest of all classes; many look like the *In-between Round Smooth* or edge-on classes), so do
not be surprised if the classifier disagrees on the generated galaxy. Try other classes (`GEN_LABEL`) and sample numbers
(`GEN_K`; only samples 0-3 have a copy in `synthetic_examples.zip` to compare with).
""")

# ================================================================================ summary
md("# Summary")
md("""
The downloaded classifiers classify a held-out test galaxy exactly as in the paper's evaluation, and the downloaded
generator regenerates an image of the paper's synthetic training set from its seed. Tutorial 5 fine-tunes the
classifier; Tutorial 6 trains it on real, synthetic and mixed data.
""")
code("""
TIMES["whole notebook"] = time.time() - T0
for step, seconds in TIMES.items():
    print(f"  {step:<34} {seconds / 60:5.1f} min")
""")

nb.write("04_Basic_Testing_Tutorial.ipynb", gpu=True, fill=FILL)

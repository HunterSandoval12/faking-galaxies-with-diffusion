"""Builds notebooks/02_Model_Description_Tutorial.ipynb (run from the repository root).

Spec item #2 (Model Description): load a pretrained model (wget from GitHub), save a model, number of
parameters, input layer (compatible with the dataset tutorial), output layer (outputs, activations,
loss), intermediate layers. Both models of the project are described: the ResNet-18 classifier and the
generator (Stable Diffusion 1.5 VAE + U-Net, LoRA adapters, learned class embeddings).

Usage: python notebooks/_build/build_02_model_description.py [--times times.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbcommon import REPO_URL, SD_REPO, Notebook, download_cells, drive_cells, install_cells  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--times", type=Path, help="JSON from a local test run: step times (s) and peak memory")
args = ap.parse_args()


def fmt(seconds):
    return f"~{seconds:.0f} s" if seconds < 90 else f"~{seconds / 60:.0f} min"


FILL = {k: "?" for k in ("__T_WEIGHTS__", "__T_SD__", "__T_TOTAL__", "__T_TOTAL_CPU__", "__PEAK_RAM__",
                         "__PEAK_VRAM__", "__T2_CORRECT__", "__T2_LOSS__")}
if args.times:
    m = json.loads(args.times.read_text())
    FILL.update({"__T_WEIGHTS__": fmt(m["download weights"]), "__T_SD__": fmt(m["download + load Stable Diffusion"]),
                 "__T_TOTAL__": fmt(m["whole notebook"]), "__T_TOTAL_CPU__": fmt(m["whole notebook (CPU)"]),
                 "__PEAK_RAM__": f"{m['peak_ram_gib']:.1f} GiB", "__PEAK_VRAM__": f"{m['peak_vram_gib']:.1f} GiB",
                 "__T2_CORRECT__": str(m["n_correct"]), "__T2_LOSS__": f"{m['loss']:.2f}"})

if args.times:
    FILL.update(m.get("fill", {}))  # hand-worded cells (e.g. download times measured from a local mirror)

nb = Notebook()
md, code = nb.md, nb.code

# ================================================================================ title + scope
md("# Galaxy10 DECaLS: Model Description Tutorial")
md(f"""
**Project:** *Faking Galaxies with Diffusion: A Real-vs-Synthetic Benchmark for Morphology Classification*
Hunter Sandoval, University of New Mexico. Tutorial 2 of 6.

**Scope.** The project uses two models, and this tutorial shows how to access every element of both:

1. the **classifier**: a ResNet-18 convolutional network that sorts a galaxy image into one of 10 morphology classes.
   It is the model the paper evaluates: trained on real, synthetic or mixed images, then tested on real images;
2. the **generator**: Stable Diffusion 1.5 (a variational autoencoder and a U-Net) fine-tuned with LoRA adapters
   and a learned class-embedding table, which makes the synthetic training images.

For each model the tutorial **loads the pretrained weights** (downloaded with `wget` from the project's GitHub
Release; the Stable Diffusion base weights come from Hugging Face), **counts the parameters**, feeds it images from
Tutorial 1 to show the **input layer**, walks through the **intermediate layers**, explains the **output layer**
(outputs, activations and the loss used for training), and **saves** the model. Training itself is covered in
Tutorials 3, 5 and 6.
""")
md("## References")
md(f"""
1. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR 2016, arXiv:1512.03385
   (ResNet).
2. torchvision ResNet-18 and its ImageNet weights: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
3. R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022, arXiv:2112.10752
   (Stable Diffusion). Model card of the weights used: https://huggingface.co/{SD_REPO}
4. O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015,
   arXiv:1505.04597.
5. J. Ho, A. Jain, P. Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020, arXiv:2006.11239 (the training
   loss).
6. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022, arXiv:2106.09685.
7. J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," arXiv:2207.12598, 2022.
8. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021,
   arXiv:2103.00020 (CLIP, which initialized the class embeddings).
9. Hugging Face `diffusers` (https://github.com/huggingface/diffusers) and `peft` (https://github.com/huggingface/peft).
10. Galaxy10 DECaLS dataset: https://github.com/henrysky/Galaxy10
11. Project code and model weights: {REPO_URL}
""")

# ================================================================================ setup
md("# Setup")
install_cells(nb, {"torch": "2.0", "torchvision": "0.15", "diffusers": "0.27", "peft": "0.10", "safetensors": "0.4",
                   "numpy": "1.23", "pillow": "9.0", "matplotlib": "3.6"}, """
| Python | 3.10 | 3.14.7 | (Colab's default) | |
| PyTorch (`torch`) | 2.0 | 2.14.0 | `pip install "torch>=2.0"` | neural networks, GPU computation |
| torchvision | 0.15 | 0.29.0 | `pip install "torchvision>=0.15"` | the ResNet-18 architecture |
| diffusers | 0.27 | 0.40.0 | `pip install "diffusers>=0.27"` | Stable Diffusion's VAE, U-Net and noise schedule |
| peft | 0.10 | 0.21.0 | `pip install "peft>=0.10"` | LoRA adapters |
| safetensors | 0.4 | 0.8.0 | `pip install "safetensors>=0.4"` | reading and writing weight files |
| NumPy | 1.23 | 2.5.2 | `pip install "numpy>=1.23"` | arrays |
| Pillow | 9.0 | 12.3.0 | `pip install "pillow>=9.0"` | reading PNG images |
| Matplotlib | 3.6 | 3.11.2 | `pip install "matplotlib>=3.6"` | figures |
""")
md("## Hardware used for running")
md("""
- **GPU recommended**: in Colab choose *Runtime > Change runtime type > T4 GPU* (the notebook asks for it when opened
  from GitHub). The generator runs in 16-bit floating point on the GPU and needs about __PEAK_VRAM__ of GPU memory.
- **CPU works too**, only slower: the next cell falls back to the CPU (and to 32-bit floats for the generator).
- **Memory (RAM) at peak:** about __PEAK_RAM__.
- **Disk:** about 2 GB for the Stable Diffusion base weights (Hugging Face cache) and 60 MB for the project weights.
- **Tested on:** a Windows 11 PC with an NVIDIA RTX 5090 GPU (32 GB), an AMD Ryzen 7 9800X3D CPU and 32 GB of RAM,
  Python 3.14; every code cell was run there both on the GPU and with the GPU hidden (CPU only).
""")
code("""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32  # the generator's precision (as in the project)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB)")
else:
    print("No GPU found: running on the CPU (slower). In Colab: Runtime > Change runtime type > T4 GPU.")
""")
md("## Approximate execution times")
md("""
| Step | Test PC (RTX 5090) | Colab T4 GPU (estimate) |
|---|---|---|
| Download the project weights (wget, 52 MiB) | __T_WEIGHTS__ | < 30 s |
| Download + load Stable Diffusion (1.9 GB, first run only) | __T_SD__ | 1-3 min |
| Everything else (a few forward passes) | < 1 min | 1-2 min |
| **Whole tutorial** | **__T_TOTAL__** (CPU only: __T_TOTAL_CPU__) | **~3-6 min** |
""")
md("## Expected results")
md("""
- All downloaded files pass their SHA-256 checks.
- **Classifier:** 11,181,642 parameters; a 256 x 256 RGB image becomes a 3 x 256 x 256 tensor, shrinks to 512 feature
  maps of 8 x 8, and ends as 10 class scores. On one validation image per class it gets __T2_CORRECT__ of 10 right,
  with a cross-entropy loss of about __T2_LOSS__; the saved and reloaded model gives identical outputs.
- **Generator:** U-Net 859,520,964 parameters (frozen), LoRA adapters 1,594,368 (trained; 128 adapted attention
  layers), class-embedding table 650,496 (trained), VAE 83,653,863 (frozen). A 512 x 512 image becomes a 4 x 64 x 64
  latent (48x fewer numbers); the U-Net outputs a noise estimate of the same shape; the training loss (mean squared
  error) is small at low noise levels and larger at high ones.
""")

# ================================================================================ drive + downloads
drive_cells(nb, "The tutorial feeds the models real images from the dataset, to show that the inputs fit.")
code("""
import io
import zipfile

import numpy as np
from PIL import Image

CLASS_NAMES = ["Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar-Shaped Smooth",
               "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge",
               "Edge-on with Bulge"]
assert ZIP_PATH.exists(), "Run Tutorial 1 (Dataset Tutorial) first: it saves galaxy10_dataset.zip on your Drive."
dataset_zip = zipfile.ZipFile(ZIP_PATH)
MEMBERS = sorted(n for n in dataset_zip.namelist() if n.endswith(".png"))


def read_png(member):
    \"\"\"One image of Tutorial 1's directories as a uint8 array [256, 256, 3].\"\"\"
    return np.asarray(Image.open(io.BytesIO(dataset_zip.read(member))).convert("RGB"))


def first_per_class(split):
    \"\"\"The first file of every class folder of a split: [(zip member, label)].\"\"\"
    out = {}
    for n in MEMBERS:
        _, s, folder, _ = n.split("/")  # galaxy10/<split>/<label>_<class name>/<index>.png
        if s == split:
            out.setdefault(int(folder.split("_")[0]), n)
    return [(out[c], c) for c in sorted(out)]


print(f"{len(MEMBERS)} images in {ZIP_PATH.name}; e.g. {MEMBERS[0]}")
""")
md("# Download the pretrained weights")
download_cells(nb, ["resnet18_real_only_seed0.safetensors", "lora_unet.safetensors", "class_embeddings.safetensors",
                    "lora_config.json"],
               intro="The next cell downloads the classifier (trained on real images only; the paper's baseline) and "
                     "the generator's trained parts.")
md(f"""
The generator also needs the **Stable Diffusion v1.5 base weights** (about 1.9 GB in 16-bit precision). They come
from Hugging Face, `{SD_REPO}`, and `diffusers` downloads them automatically the first time (Part 2). The original
repository, `runwayml/stable-diffusion-v1-5`, was taken down in 2024 and now redirects there. We checked that its
U-Net and VAE files are byte-identical (SHA-256) to the ones the project used, and that its 16-bit files are exactly
the 32-bit weights rounded to 16 bits, which is what the project ran.
""")

# ================================================================================ part 1: classifier
md("# Part 1: The classifier (ResNet-18)")
md("## Load the model")
md("""
**How we figured this out.** The classifier is torchvision's standard `resnet18` with its last layer replaced (1,000
ImageNet classes -> 10 galaxy classes), exactly as in the project's `classifier/train_classifier.py`. The weight
file's names (`conv1.weight`, `layer1.0.bn1.weight`, ...) are torchvision's layer names, so `load_state_dict` fills
every layer, and it refuses to load if a single layer is missing or has the wrong shape. The file also stores a short
description (metadata), printed below.
""")
code("""
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torchvision.models import resnet18


def build_classifier():
    model = resnet18(weights=None)                                   # the architecture, no weights yet
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))     # 512 features -> 10 classes
    return model


CLASSIFIER_FILE = WEIGHTS_DIR / "resnet18_real_only_seed0.safetensors"
classifier = build_classifier()
classifier.load_state_dict(load_file(CLASSIFIER_FILE))  # strict: every layer must match
classifier = classifier.to(DEVICE).eval()                # eval(): batch norm uses its stored statistics
with safe_open(CLASSIFIER_FILE, "pt") as f:
    for key, value in sorted(f.metadata().items()):
        print(f"{key:<20} {value}")
""")
md("## Number of parameters")
code("""
def n_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


print(f"ResNet-18 classifier: {n_params(classifier):,} parameters ({n_params(classifier, True):,} trainable)")
for name, child in classifier.named_children():
    print(f"  {name:<8} {type(child).__name__:<18} {n_params(child):>12,}")
assert n_params(classifier) == 11_181_642
""")
md("## Input layer")
md("""
The first layer, `conv1`, is a convolution with 3 input channels, so the classifier expects a **4-D float tensor**
`[batch, 3, height, width]`. Tutorial 1's images are uint8 arrays `[256, 256, 3]`, so each image is (1) reordered to
channels-first, (2) scaled to 0-1 and (3) normalized with the ImageNet mean and standard deviation, because the network
started from ImageNet-pretrained weights. This is `to_input` from the project's `classifier/train_classifier.py`
(without its random flips and rotations, which are only used in training). The images keep their native 256 x 256
size. The cell loads one validation image of each class from Tutorial 1's zip file.
""")
code("""
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_input(images_u8):
    \"\"\"uint8 images [N, 256, 256, 3] (Tutorial 1) -> normalized float tensor [N, 3, 256, 256].\"\"\"
    x = torch.as_tensor(np.array(images_u8)).permute(0, 3, 1, 2).float() / 255.0  # np.array: a writable copy
    return (x - IMAGENET_MEAN) / IMAGENET_STD


val_members = first_per_class("validation")
x_u8 = np.stack([read_png(m) for m, _ in val_members])
y = torch.tensor([c for _, c in val_members])
x = to_input(x_u8).to(DEVICE)
print("dataset images:", x_u8.shape, x_u8.dtype, "-> classifier input:", tuple(x.shape), x.dtype)
print("input layer:", classifier.conv1)
with torch.no_grad():
    print("output of the input layer:", tuple(classifier.conv1(x).shape), "(64 feature maps, half the resolution)")
""")
md("## Intermediate layers")
md("""
ResNet-18 is a **convolutional neural network with residual connections** (reference 1). After the *stem*
(`conv1` 7 x 7, batch norm, ReLU, 3 x 3 max-pooling) come four stages, `layer1` to `layer4`, of two *basic blocks*
each. A basic block computes two 3 x 3 convolutions, each followed by batch normalization, and adds the block's input
back to the result (the residual or skip connection) before the final ReLU:

```
x --> conv3x3 --> BN --> ReLU --> conv3x3 --> BN --> (+) --> ReLU --> out
 \\____________________ identity (or 1x1 conv) _______/
```

Each stage after the first halves the resolution and doubles the number of feature maps (the first block uses a 1 x 1
convolution, `downsample`, on the skip path so the shapes match). Global average pooling then turns the 512 feature
maps of 8 x 8 into 512 numbers. The cell records the output shape of every stage for one galaxy (forward hooks), then
prints one block with a downsampling skip connection.
""")
code("""
shapes = []
hooks = [getattr(classifier, name).register_forward_hook(
    lambda module, inputs, output, name=name: shapes.append((name, tuple(output.shape))))
    for name in ["conv1", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]]
with torch.no_grad():
    classifier(x[:1])
for h in hooks:
    h.remove()
print(f"{'input':<8} {tuple(x[:1].shape)}")
for name, shape in shapes:
    print(f"{name:<8} {shape}")
print()
print("layer2[0] (a basic block whose skip path downsamples):")
print(classifier.layer2[0])
""")
md("ResNet-18 is small enough to print completely:")
code("""
print(classifier)
""")
md("## Output layer")
md("""
The last layer, `fc`, is a fully connected (linear) layer from 512 features to **10 outputs, one per class**. It has
**no activation function** inside the model: its outputs are *logits* (unnormalized scores). The **softmax** function
turns them into class probabilities, `p_k = exp(z_k) / sum_j exp(z_j)`, and the predicted class is the largest one.

The classifier was trained with the **cross-entropy loss**, `L = -log p_y` for an image of true class `y`, averaged
over the batch (PyTorch's `F.cross_entropy`, which applies the softmax itself). It is small when the model gives the
true class a high probability. Optimization (AdamW, learning rate schedule, epochs) is the subject of Tutorial 3. The
cell classifies the 10 validation images and computes the loss both with PyTorch and by hand.
""")
code("""
with torch.no_grad():
    logits = classifier(x).float().cpu()
probs = F.softmax(logits, dim=1)
loss = F.cross_entropy(logits, y)
loss_by_hand = -torch.log(probs[torch.arange(len(y)), y]).mean()
print(f"logits: {tuple(logits.shape)} -> probabilities: {tuple(probs.shape)}, each row sums to {probs.sum(1)[0]:.4f}")
print(f"{'true class':<26}{'predicted':<26}{'probability':>11}")
for true, row in zip(y.tolist(), probs):
    pred = int(row.argmax())
    print(f"{CLASS_NAMES[true]:<26}{CLASS_NAMES[pred]:<26}{row[pred]:>11.3f}{'' if pred == true else '   <- wrong'}")
n_correct = int((probs.argmax(1) == y).sum())
print(f"\\n{n_correct} of {len(y)} correct; cross-entropy loss {loss:.4f} (by hand: {loss_by_hand:.4f})")
assert torch.allclose(loss, loss_by_hand, atol=1e-5)
""")
md("## Save the model")
md("""
Saving stores the **state dict**, the dictionary of every weight tensor, not the Python code of the model. The project
uses the `safetensors` format (the released files are `.safetensors`): it stores only tensors plus a text description,
so loading it can never run code, unlike Python pickles. PyTorch's own `torch.save` works too. The cell saves the
classifier both ways, reloads each into a fresh model and checks that the outputs are identical. Copy the files to
`DRIVE_ROOT` if you want to keep them after the session.
""")
code("""
import shutil

SAVE_DIR = WORK_ROOT / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
state = {k: v.detach().cpu().contiguous() for k, v in classifier.state_dict().items()}
save_file(state, SAVE_DIR / "my_classifier.safetensors", metadata={"architecture": "resnet18, fc 512 -> 10"})
torch.save(state, SAVE_DIR / "my_classifier.pt")

for path, loader in ((SAVE_DIR / "my_classifier.safetensors", load_file),
                     (SAVE_DIR / "my_classifier.pt", lambda p: torch.load(p, weights_only=True))):
    reloaded = build_classifier()
    reloaded.load_state_dict(loader(path))
    reloaded = reloaded.to(DEVICE).eval()
    with torch.no_grad():
        same = torch.equal(reloaded(x), classifier(x))
    print(f"{path.name:<26} {path.stat().st_size / 2**20:5.1f} MiB   reloaded outputs identical: {same}")
    assert same
""")

# ================================================================================ part 2: generator
md("# Part 2: The generator (Stable Diffusion 1.5 + LoRA + class embeddings)")
md("""
The generator is a **latent diffusion model** (reference 3). It never works on pixels directly. A **variational
autoencoder (VAE)** compresses a 512 x 512 x 3 image into a 64 x 64 x 4 *latent*, and a **U-Net** learns to remove
noise from latents. To generate, the sampler starts from pure noise and lets the U-Net remove it step by step (30
steps of the DPM-Solver++ sampler); the VAE decoder turns the final latent into an image, which the project shrinks to
Galaxy10's 256 x 256.

Stable Diffusion normally steers the U-Net with a text prompt encoded by CLIP. The project replaces the prompt with a
**learned class-embedding table**: one sequence of 77 x 768 numbers per galaxy class, plus a *null* row (index 10)
for unconditional predictions, used by classifier-free guidance (reference 7). The table started from CLIP's encoding
of prompts like "a telescope image of a galaxy, barred spiral morphology". During fine-tuning only the table and small
**LoRA adapters** (reference 6) in the U-Net's attention layers were trained; the 860 million U-Net weights and the VAE
stayed frozen.

```
class label c --> class-embedding table row c (77 x 768) --------------------------+
                                                                                    | cross-attention
image 512x512x3 --> VAE encoder --> latent 64x64x4 --(+ noise, step t)--> U-Net (+ LoRA) --> predicted noise
generation: noise --(30 denoising steps, guidance 3)--> latent --> VAE decoder --> image 512x512 --> 256x256
```
""")
md("## Load the model")
md(f"""
**How we figured this out.** The release holds only what fine-tuning changed (9 MB), so loading has three steps:
(1) load the base VAE and U-Net from Hugging Face with `diffusers` (`variant="fp16"` picks the 16-bit files, half the
download); (2) add LoRA adapters with the same settings as training, read from `lora_config.json`, and fill them with
the trained weights; (3) load the class-embedding table. Steps 2 and 3 are the function `load_lora_checkpoint` from the
project's `diffusion/sample.py`. It checks that every LoRA weight found its layer. The noise schedule used in training
(`DDPMScheduler`) is loaded too. Hugging Face may warn that no `HF_TOKEN` is set: no token or account is needed.
""")
code(f"""
import json

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

SD_REPO = "{SD_REPO}"
t = time.time()
vae = AutoencoderKL.from_pretrained(SD_REPO, subfolder="vae", variant="fp16", torch_dtype=DTYPE).to(DEVICE).eval()
unet = UNet2DConditionModel.from_pretrained(SD_REPO, subfolder="unet", variant="fp16", torch_dtype=DTYPE).to(DEVICE).eval()
noise_scheduler = DDPMScheduler.from_pretrained(SD_REPO, subfolder="scheduler")
TIMES["download + load Stable Diffusion"] = time.time() - t
lora_config = json.loads((WEIGHTS_DIR / "lora_config.json").read_text())


def load_lora_checkpoint(unet, weights_dir, cfg):
    \"\"\"Add LoRA adapters to `unet` and load the trained weights (from diffusion/sample.py). Returns the class table.\"\"\"
    unet.add_adapter(LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], init_lora_weights="gaussian",
                                target_modules=cfg["lora_target_modules"]))
    result = set_peft_model_state_dict(unet, load_file(weights_dir / "lora_unet.safetensors"))
    missing = [k for k in result.missing_keys if "lora" in k]
    assert not result.unexpected_keys and not missing, "LoRA weights do not match the U-Net"
    return load_file(weights_dir / "class_embeddings.safetensors")["table"]


class_table_fp32 = load_lora_checkpoint(unet, WEIGHTS_DIR, lora_config)
class_table = class_table_fp32.to(DEVICE, DTYPE)
NULL_CLASS = lora_config["null_class_index"]
print(f"loaded in {{TIMES['download + load Stable Diffusion']:.0f}} s; LoRA rank {{lora_config['lora_rank']}}, "
      f"alpha {{lora_config['lora_alpha']}}, target layers {{lora_config['lora_target_modules']}}")
print("class table:", tuple(class_table.shape), f"(rows 0-9 = classes, row {{NULL_CLASS}} = null class)")
""")
md("## Number of parameters")
code("""
n_lora = sum(p.numel() for name, p in unet.named_parameters() if "lora_" in name)
parts = {"VAE encoder": n_params(vae.encoder) + n_params(getattr(vae, "quant_conv", None) or nn.Identity()),
         "VAE decoder": n_params(vae.decoder) + n_params(getattr(vae, "post_quant_conv", None) or nn.Identity()),
         "U-Net (base, frozen)": n_params(unet) - n_lora,
         "LoRA adapters (trained)": n_lora,
         "class-embedding table (trained)": class_table.numel()}
for name, n in parts.items():
    print(f"{name:<32} {n:>13,}")
trained = n_lora + class_table.numel()
print(f"{'total':<32} {sum(parts.values()):>13,}")
print(f"trained during fine-tuning: {trained:,} ({100 * trained / sum(parts.values()):.2f}% of all parameters)")
""")
md("## Input layer")
md("""
The U-Net's input layer, `conv_in`, is a 3 x 3 convolution with **4 input channels**: it takes the VAE's 64 x 64 x 4
latent, not pixels. A Tutorial 1 image reaches it in three steps, as in the project's training script
(`diffusion/train_lora.py`, function `preprocess`): scale the uint8 pixels to [-1, 1], upsample 256 -> 512 (bicubic;
Stable Diffusion works at 512), and encode with the VAE (multiplied by the VAE's scaling factor, 0.18215). The class
enters through a second input, `encoder_hidden_states`: the class table row, 77 x 768. The cell encodes one training
image and decodes it again, to show what the latent keeps.
""")
code("""
import matplotlib.pyplot as plt


def to_vae_input(image_u8, resolution=512):
    \"\"\"uint8 [256, 256, 3] -> float [1, 3, 512, 512] in [-1, 1] (as in diffusion/train_lora.py).\"\"\"
    x = torch.as_tensor(np.array(image_u8))[None].permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return F.interpolate(x, size=(resolution, resolution), mode="bicubic", align_corners=False).clamp(-1, 1)


member, label = first_per_class("training")[2]  # a Round Smooth training galaxy
image_u8 = read_png(member)
pixels = to_vae_input(image_u8).to(DEVICE, DTYPE)
with torch.no_grad():
    latents = vae.encode(pixels).latent_dist.mean * vae.config.scaling_factor  # training draws a sample instead
    reconstruction = vae.decode(latents / vae.config.scaling_factor).sample
print(f"image {image_u8.shape} -> VAE input {tuple(pixels.shape)} -> latent {tuple(latents.shape)} "
      f"({pixels.numel() / latents.numel():.0f}x fewer numbers)")
print("U-Net input layer:", unet.conv_in)
print("class input (encoder_hidden_states):", tuple(class_table[label][None].shape))

rec = ((reconstruction[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
axes[0].imshow(image_u8)
axes[0].set_title(f"training image ({CLASS_NAMES[label]})")
axes[1].imshow(latents[0, :3].float().cpu().permute(1, 2, 0).numpy().clip(-3, 3) / 6 + 0.5)
axes[1].set_title("latent (3 of 4 channels)")
axes[2].imshow(rec)
axes[2].set_title("VAE reconstruction")
for ax in axes:
    ax.axis("off")
plt.show()
""")
md("## Intermediate layers")
md("""
The **U-Net** (reference 4) has an encoder path of 4 *down blocks* (64 -> 32 -> 16 -> 8 pixels, 320 -> 640 -> 1280
-> 1280 channels), a *middle block* at 8 x 8, and a decoder path of 4 *up blocks* that restore the resolution; skip
connections pass each down block's features to the matching up block. The blocks are built from **ResNet blocks**
(convolutions with residual connections, like the classifier's) and **transformer blocks**. Each transformer block has
a self-attention layer (`attn1`) and a **cross-attention** layer (`attn2`) whose keys and values come from
`encoder_hidden_states`: this is where the class embedding steers the image. The noise level `t` enters every ResNet
block through a time embedding.

**LoRA** adds, next to a frozen weight matrix `W` of an attention projection (`to_q`, `to_k`, `to_v`, `to_out.0`),
two small trained matrices `A` (r x in) and `B` (out x r) with rank r = 8: the layer computes
`W x + (alpha / r) B A x`. The next cells list the blocks with their output shapes (forward hooks), show one
cross-attention layer with its LoRA adapters, and check the LoRA formula on one layer.
""")
code("""
block_shapes = []


def record(name):
    def hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        block_shapes.append((name, type(module).__name__, tuple(out.shape)))
    return hook


blocks = [("conv_in", unet.conv_in)] + [(f"down_blocks.{i}", b) for i, b in enumerate(unet.down_blocks)] \\
    + [("mid_block", unet.mid_block)] + [(f"up_blocks.{i}", b) for i, b in enumerate(unet.up_blocks)] \\
    + [("conv_out", unet.conv_out)]
hooks = [module.register_forward_hook(record(name)) for name, module in blocks]
t_mid = torch.tensor([500], device=DEVICE)
with torch.no_grad():
    unet(latents, t_mid, encoder_hidden_states=class_table[label][None])
for h in hooks:
    h.remove()
for (name, module), (_, kind, shape) in zip(blocks, block_shapes):
    print(f"{name:<14} {kind:<24} out {str(shape):<20} {n_params(module):>12,} parameters")
n_adapted = sum(1 for _, m in unet.named_modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0)
print(f"\\nLoRA-adapted layers: {n_adapted} (16 transformer blocks x 2 attention layers x 4 projections)")
print("\\nOne cross-attention layer (down_blocks.0 ... attn2) with its LoRA adapters:")
print(unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2)
""")
code("""
layer = unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_q  # a LoRA-adapted linear layer
A, B = layer.lora_A["default"].weight, layer.lora_B["default"].weight
scale = layer.scaling["default"]
x_in = torch.randn(1, 4096, layer.in_features, device=DEVICE, dtype=DTYPE)
with torch.no_grad():
    y_layer = layer(x_in).float()
    y_formula = layer.base_layer(x_in).float() + scale * (x_in.to(A.dtype) @ A.T @ B.T).float()
print(f"W: {tuple(layer.base_layer.weight.shape)}, A: {tuple(A.shape)}, B: {tuple(B.shape)}, alpha / r = {scale}")
err = (y_layer - y_formula).abs().max().item()
print(f"layer output vs W x + (alpha/r) B A x: max difference {err:.2e} (rounding only; outputs up to "
      f"{y_formula.abs().max().item():.1f})")
assert err < 1e-2 * max(1.0, y_formula.abs().max().item())
""")
md("## Output layer")
md("""
The U-Net's last layer, `conv_out`, is a 3 x 3 convolution back to **4 channels**: the output is the model's estimate
of the **noise** that was added to the latent, with exactly the latent's shape (4 x 64 x 64). There is **no activation
function** on it, because noise values are unbounded (Gaussian).

**Training loss** (reference 5): take a training latent `z0`, draw Gaussian noise `eps` and a random step `t` (0-999),
mix them with the noise schedule into `z_t`, and let the U-Net predict the noise from `z_t`, `t` and the class
embedding. The loss is the **mean squared error** between the true and the predicted noise, `L = mean((eps -
eps_hat)^2)`. In 10% of training steps the class was replaced by the null class, so the same network also learned
unconditional predictions. At sampling time, **classifier-free guidance** combines both: `eps = eps_null + g (eps_class
- eps_null)` with g = 3. The cell computes the training loss of one image at five noise levels.
""")
code("""
print("U-Net output layer:", unet.conv_out)
gen = torch.Generator(DEVICE).manual_seed(0)
noise = torch.randn(latents.shape, generator=gen, device=DEVICE, dtype=DTYPE)
print(f"\\n{'step t':>7} {'noise share':>12} {'MSE loss (class)':>17} {'MSE loss (null class)':>22}")
for t_step in [50, 250, 500, 750, 950]:
    t = torch.tensor([t_step], device=DEVICE)
    z_t = noise_scheduler.add_noise(latents, noise, t)
    with torch.no_grad():
        eps_class = unet(z_t, t, encoder_hidden_states=class_table[label][None]).sample
        eps_null = unet(z_t, t, encoder_hidden_states=class_table[NULL_CLASS][None]).sample
    share = (1 - noise_scheduler.alphas_cumprod[t_step]).sqrt().item()
    loss_c = F.mse_loss(eps_class.float(), noise.float()).item()
    loss_n = F.mse_loss(eps_null.float(), noise.float()).item()
    print(f"{t_step:>7} {share:>12.2f} {loss_c:>17.4f} {loss_n:>22.4f}")
print("output shape:", tuple(eps_class.shape), "= latent shape", tuple(latents.shape))
""")
md("## Save the model")
md("""
Only the trained parts need saving: the LoRA weights (`get_peft_model_state_dict` extracts them from the U-Net) and the
class-embedding table, 9 MB instead of the 2 GB base model. This is exactly how the project's training script wrote its
checkpoints, and the format of the released files. The cell saves both and checks them against the downloaded files.
""")
code("""
lora_state = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in get_peft_model_state_dict(unet).items()}
save_file(lora_state, SAVE_DIR / "my_lora_unet.safetensors")
save_file({"table": class_table_fp32.contiguous()}, SAVE_DIR / "my_class_embeddings.safetensors")

released = load_file(WEIGHTS_DIR / "lora_unet.safetensors")
same_lora = sorted(released) == sorted(lora_state) and all(
    torch.equal(released[k].to(DTYPE), lora_state[k].to(DTYPE)) for k in released)  # compared at the model's precision
same_table = torch.equal(load_file(SAVE_DIR / "my_class_embeddings.safetensors")["table"],
                         load_file(WEIGHTS_DIR / "class_embeddings.safetensors")["table"])
for p in (SAVE_DIR / "my_lora_unet.safetensors", SAVE_DIR / "my_class_embeddings.safetensors"):
    print(f"{p.name:<32} {p.stat().st_size / 2**20:5.2f} MiB")
print(f"{len(lora_state)} LoRA tensors match the released file: {same_lora}; class table identical: {same_table}")
assert same_lora and same_table
""")

# ================================================================================ summary
md("# Summary")
md("""
| | Classifier | Generator |
|---|---|---|
| Architecture | ResNet-18 (CNN with residual connections) | Stable Diffusion 1.5: VAE + U-Net with cross-attention |
| Input | normalized image, 3 x 256 x 256 | latent 4 x 64 x 64 (from a 512 x 512 image) + class embedding 77 x 768 + step t |
| Output | 10 logits -> softmax probabilities | predicted noise, 4 x 64 x 64 (linear output) |
| Training loss | cross-entropy | mean squared error on the noise |
| Trained parameters | all 11.2 million | LoRA adapters + class table: 2.2 million (U-Net and VAE frozen) |
| Saved as | one safetensors file (43 MiB) | LoRA + class table safetensors (9 MB) + config |

Tutorial 4 uses both models: it classifies a test galaxy and generates a synthetic one.
""")
code("""
TIMES["whole notebook"] = time.time() - T0
for step, seconds in TIMES.items():
    print(f"  {step:<34} {seconds / 60:5.1f} min")
""")

nb.write("02_Model_Description_Tutorial.ipynb", gpu=True, fill=FILL)

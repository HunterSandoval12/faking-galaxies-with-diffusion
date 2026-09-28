"""Build the files published in the GitHub Release `weights-v1` (used by the Colab tutorials).

Usage (from the directory that holds the training runs and the synthetic set):
    python tools/make_release_assets.py --project . --out weights

Writes to --out:
  resnet18_real_only_seed0.safetensors       ResNet-18 trained on real images only (the baseline)
  resnet18_synthetic_only_seed0.safetensors  ResNet-18 trained on synthetic images only
  resnet18_mix50_seed0.safetensors           ResNet-18 trained on a 50% real / 50% synthetic mix
  lora_unet.safetensors                      LoRA weights of the chosen generator (checkpoint 10,000)
  class_embeddings.safetensors               the generator's learned class-embedding table (11 x 77 x 768)
  lora_config.json                           the generator's training configuration
  synthetic_examples.zip                     sample indices 0-3 of every class of the synthetic set (PNG)
  LICENSE-CreativeML-OpenRAIL-M.txt          the license of the LoRA weights (derived from Stable Diffusion v1.5)
  SHA256SUMS.txt                             SHA-256 checksum of every file above

The classifier files hold the weights only (no pickled Python objects, no training
configuration), so they are safe to load and contain no local file paths. Each file is
checked: the exported weights must give exactly the original checkpoint's outputs.
"""

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision.models import resnet18

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar-Shaped Smooth",
               "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge",
               "Edge-on with Bulge"]
CLASSIFIERS = {  # release file -> (training run, description of its training data)
    "resnet18_real_only_seed0": ("A_replace0_seed0", "12,330 real training images (0% synthetic)"),
    "resnet18_synthetic_only_seed0": ("A_replace1_seed0", "12,330 synthetic images (100% synthetic)"),
    "resnet18_mix50_seed0": ("A_replace0.5_seed0", "6,165 real + 6,165 synthetic images (50% synthetic)"),
}
N_SYNTHETIC_EXAMPLES = 4  # per class: sample indices 0-3, the same images as the paper's Fig. 2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(2**24):
            h.update(block)
    return h.hexdigest()


def canonicalize_header(path):
    """Rewrite a safetensors header in a fixed order. safetensors writes the metadata from a hash map,
    so its order (and the file's checksum) changes between runs; the tensor data is unaffected."""
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    ordered = {"__metadata__": dict(sorted(header.pop("__metadata__").items()))}
    ordered.update(sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]))
    text = json.dumps(ordered, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)  # the format pads the header to a multiple of 8 bytes with spaces
    path.write_bytes(len(text).to_bytes(8, "little") + text + raw[8 + n:])


def classifier():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model.eval()


def export_classifiers(runs, out):
    x = torch.randn(4, 3, 256, 256, generator=torch.Generator().manual_seed(0))
    for name, (run, data) in CLASSIFIERS.items():
        ckpt = torch.load(runs / run / "best.pt", map_location="cpu", weights_only=True)
        state = {k: v.contiguous() for k, v in ckpt["state_dict"].items()}
        metrics = ckpt["val_metrics"]
        metadata = {
            "architecture": "torchvision.models.resnet18 with fc = Linear(512, 10)",
            "pretraining": "ImageNet (torchvision ResNet18_Weights.IMAGENET1K_V1), then fine-tuned on Galaxy10 DECaLS",
            "training_data": data,
            "input": "RGB 256x256; uint8 / 255, then normalized with ImageNet mean (0.485, 0.456, 0.406) "
                     "and std (0.229, 0.224, 0.225); NCHW",
            "output": "10 logits (apply softmax for class probabilities)",
            "classes": json.dumps(CLASS_NAMES),
            "seed": str(ckpt["config"]["seed"]),
            "selected_epoch": str(metrics["epoch"]),
            "validation_accuracy": f"{metrics['accuracy']:.4f}",
            "validation_macro_f1": f"{metrics['macro_f1']:.4f}",
        }
        path = out / f"{name}.safetensors"
        save_file(state, path, metadata=metadata)
        canonicalize_header(path)

        original, exported = classifier(), classifier()
        original.load_state_dict(ckpt["state_dict"])
        exported.load_state_dict(load_file(path))
        with torch.no_grad():
            assert torch.equal(original(x), exported(x)), name
        print(f"{path.name}: epoch {metrics['epoch']}, val acc {metrics['accuracy']:.4f} - outputs identical")


def export_generator(lora_dir, out):
    for f in ("lora_unet.safetensors", "class_embeddings.safetensors"):
        shutil.copy2(lora_dir / f, out / f)
    cfg = json.loads((lora_dir / "train_config.json").read_text())
    cfg["splits"] = "data/splits/galaxy10_splits.npz"  # the original held a local absolute path
    cfg["output_dir"] = "runs/lora_r8"
    (out / "lora_config.json").write_text(json.dumps(cfg, indent=2) + "\n", newline="\n")
    print("generator files copied; lora_config.json written with repo-relative paths")


def export_synthetic_examples(synthetic_dir, out):
    folders = [f"{c}_{n.replace(' ', '_')}" for c, n in enumerate(CLASS_NAMES)]
    with zipfile.ZipFile(out / "synthetic_examples.zip", "w", compression=zipfile.ZIP_STORED) as z:
        for c in range(len(CLASS_NAMES)):
            found = {}
            for shard in sorted(synthetic_dir.glob(f"class{c}_part*.npz")):
                s = np.load(shard)
                for img, k, seed in zip(s["images"], s["sample_index"], s["seeds"]):
                    if k < N_SYNTHETIC_EXAMPLES:
                        found[int(k)] = (img, int(seed))
                if len(found) == N_SYNTHETIC_EXAMPLES:
                    break
            assert sorted(found) == list(range(N_SYNTHETIC_EXAMPLES)), f"class {c}: {sorted(found)}"
            for k, (img, seed) in sorted(found.items()):
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="PNG")
                info = zipfile.ZipInfo(f"synthetic_examples/{folders[c]}/sample{k}_seed{seed}.png",
                                       date_time=(2026, 9, 27, 0, 0, 0))  # fixed date: reproducible zip
                z.writestr(info, buf.getvalue())
    print(f"synthetic_examples.zip: {N_SYNTHETIC_EXAMPLES} images x {len(CLASS_NAMES)} classes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True, help="directory with runs/, runs_cls/ and data/synthetic/")
    ap.add_argument("--out", type=Path, default=Path("weights"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    export_classifiers(args.project / "runs_cls" / "exp", args.out)
    export_generator(args.project / "runs" / "lora_r8" / "checkpoint-10000", args.out)
    export_synthetic_examples(args.project / "data" / "synthetic" / "main_cfg3_steps30", args.out)
    shutil.copy2(REPO_ROOT / "licenses" / "CreativeML-OpenRAIL-M.txt", args.out / "LICENSE-CreativeML-OpenRAIL-M.txt")
    files = sorted(p for p in args.out.iterdir() if p.name != "SHA256SUMS.txt")
    sums = "".join(f"{sha256(p)}  {p.name}\n" for p in files)
    (args.out / "SHA256SUMS.txt").write_text(sums, newline="\n")  # LF on every OS: stable checksums
    print(sums, end="")
    for p in files:
        print(f"{p.stat().st_size / 2**20:8.2f} MiB  {p.name}")


if __name__ == "__main__":
    main()

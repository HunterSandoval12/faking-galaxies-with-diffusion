"""Parameter counts, FLOPs per inference and the test machine's hardware -> results/model_stats.json.

Usage: python analysis/model_stats.py   (runs on the CPU; no data is read)

FLOPs are counted with PyTorch's FlopCounterMode (torch.utils.flop_counter): matrix multiplications,
convolutions and attention, 2 FLOPs per multiply-accumulate; element-wise operations (normalization,
activations, additions) are not counted, as is usual. Attention is counted through diffusers' classic
matrix-multiply attention processor (the same computation): the counter does not see PyTorch's fused
scaled-dot-product-attention kernel and would silently drop attention (~16% of the U-Net's FLOPs).
Counts depend only on tensor shapes:
  - ResNet-18 classifier: one 3 x 256 x 256 image.
  - U-Net (with the LoRA adapters): one evaluation = one 4 x 64 x 64 latent, one timestep, one
    77 x 768 class embedding; one denoising step with classifier-free guidance = 2 evaluations
    (class + null class; DPM-Solver++ multistep needs one model call per step).
  - VAE decoder: one 4 x 64 x 64 latent -> 512 x 512 image.
  - One generated image = 30 guided denoising steps + one VAE decode.
"""

import json
import platform
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "diffusion"))
sys.path.insert(0, str(ROOT / "classifier"))
sys.path.insert(0, str(ROOT / "data"))
CKPT = ROOT / "runs" / "lora_r8" / "checkpoint-10000"
STEPS = 30


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def flops(fn):
    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            fn()
    return int(counter.get_total_flops())


def hardware():
    info = {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__}
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        info["cpu"] = platform.processor()
    try:
        import psutil
        info["ram_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:  # GPU memory via nvidia-smi (no CUDA context needed)
        import subprocess
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.split()[0]
        info["gpu_memory_gib"] = round(int(mem) / 1024, 1)
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        pass
    speed = ROOT / "results" / "inference_speed.json"  # GPU as recorded by the timing benchmark
    if speed.exists():
        env = json.loads(speed.read_text())["environment"]
        info.update(gpu=env["gpu"], cuda=env["cuda"])
    return info


def main():
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor
    from sample import load_lora_checkpoint
    from train_classifier import build_model

    # classifier
    clf = build_model().eval()
    res = {"resnet18": {"parameters": n_params(clf), "trainable": sum(p.numel() for p in clf.parameters() if p.requires_grad),
                        "flops_per_image": flops(lambda: clf(torch.zeros(1, 3, 256, 256))), "input": "3 x 256 x 256"}}

    # generator
    cfg = json.loads((CKPT / "train_config.json").read_text())
    unet = UNet2DConditionModel.from_pretrained(cfg["pretrained_model"], subfolder="unet").eval()
    vae = AutoencoderKL.from_pretrained(cfg["pretrained_model"], subfolder="vae").eval()
    unet.set_attn_processor(AttnProcessor())  # countable attention (see the module docstring)
    vae.set_attn_processor(AttnProcessor())
    n_unet_base = n_params(unet)
    table, _ = load_lora_checkpoint(unet, CKPT)
    n_lora = n_params(unet) - n_unet_base
    latent = torch.zeros(1, 4, 64, 64)
    t = torch.tensor([500])
    one_eval = flops(lambda: unet(latent, t, encoder_hidden_states=table[:1]))
    guided_step = flops(lambda: unet(torch.cat([latent] * 2), t, encoder_hidden_states=table[:2]))
    decode = flops(lambda: vae.decode(latent / vae.config.scaling_factor))
    from transformers import CLIPTextConfig, CLIPTextModel
    text_encoder = CLIPTextModel(CLIPTextConfig.from_pretrained(cfg["pretrained_model"], subfolder="text_encoder"))
    res["generator"] = {
        "unet_parameters_frozen": n_unet_base, "lora_parameters_trainable": n_lora,
        "class_table_parameters_trainable": int(table.numel()), "lora_rank": cfg["lora_rank"],
        "lora_adapted_layers": sum(1 for _, m in unet.named_modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0),
        "vae_parameters_frozen": n_params(vae), "vae_encoder_parameters": n_params(vae.encoder) + n_params(vae.quant_conv),
        "vae_decoder_parameters": n_params(vae.decoder) + n_params(vae.post_quant_conv),
        "clip_text_encoder_parameters": n_params(text_encoder),
        "unet_flops_per_evaluation": one_eval, "unet_flops_per_guided_step": guided_step,
        "vae_decode_flops": decode, "steps": STEPS, "flops_per_generated_image": STEPS * guided_step + decode,
        "latent": "4 x 64 x 64", "class_embedding": "77 x 768",
    }
    res["counting"] = "torch.utils.flop_counter.FlopCounterMode; 2 FLOPs per multiply-accumulate; matmul/conv/attention only"
    res["hardware"] = hardware()
    (ROOT / "results" / "model_stats.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

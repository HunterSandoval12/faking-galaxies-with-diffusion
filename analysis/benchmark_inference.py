"""Inference-speed benchmark on this machine (RTX 5090): generator and classifier.

Usage: python analysis/benchmark_inference.py

1. Diffusion generator at the final settings: runs/lora_r8/checkpoint-10000, guidance 3,
   30 DPM-Solver++ steps, fp16, 512x512 -> Lanczos 256x256 (the exact generation pipeline of
   diffusion/generate_synthetic.py). Batch size 1 (latency) and 20 (the batch used to generate
   the synthetic set), reported per image.
2. ResNet-18 classifier (runs_cls/exp/A_replace0_seed0/best.pt), batch size 1, bf16 autocast +
   channels_last as in evaluation: (a) GPU forward pass only, (b) end-to-end (uint8 image
   host->GPU, normalisation, forward, softmax back to the host).
Timing: warm-up first, torch.cuda.synchronize() around every timed call, wall clock
(time.perf_counter). Classifier inputs are VALIDATION images (no test data is read).
Writes results/inference_speed.json and table results/tables/t13_inference_speed.*
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diffusion"))
sys.path.insert(0, str(HERE.parent / "classifier"))
from common import OUT, ROOT, split_indices  # noqa: E402

GEN_WARMUP, GEN_TIMED_B1, GEN_TIMED_B20 = 3, 20, 5         # batch-20 runs = 100 images
CLS_WARMUP, CLS_TIMED = 100, 1000


def stats(ms):
    ms = np.asarray(ms)
    return {"mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "min_ms": float(ms.min()), "n": int(len(ms))}


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000, out


@torch.no_grad()
def bench_generator():
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from sample import load_lora_checkpoint
    ckpt = ROOT / "runs" / "lora_r8" / "checkpoint-10000"
    cfg = json.loads((ckpt / "train_config.json").read_text())
    pipe = StableDiffusionPipeline.from_pretrained(cfg["pretrained_model"], dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    table, cfg = load_lora_checkpoint(pipe.unet, ckpt)
    table = table.to("cuda", torch.float16)
    null = cfg["null_class_index"]

    def run(bs, seed):
        c = seed % 10
        gens = [torch.Generator("cuda").manual_seed(9_000_000_000 + seed * 100 + k) for k in range(bs)]
        return pipe(prompt_embeds=table[c].unsqueeze(0).expand(bs, -1, -1),
                    negative_prompt_embeds=table[null].unsqueeze(0).expand(bs, -1, -1),
                    guidance_scale=3.0, num_inference_steps=30, height=512, width=512, generator=gens).images

    for i in range(GEN_WARMUP):
        run(1, i)
    b1, resize = [], []
    for i in range(GEN_TIMED_B1):
        ms, imgs = timed(lambda: run(1, 100 + i))
        b1.append(ms)
        t = time.perf_counter()
        imgs[0].resize((256, 256), Image.LANCZOS)
        resize.append((time.perf_counter() - t) * 1000)
    run(20, 999)  # warm-up at the larger batch
    b20 = [timed(lambda: run(20, 200 + i))[0] / 20 for i in range(GEN_TIMED_B20)]
    peak = torch.cuda.max_memory_allocated() / 2**30
    del pipe
    torch.cuda.empty_cache()
    return {"batch1": stats(b1), "batch20_per_image": stats(b20), "resize_512_to_256_ms": stats(resize),
            "peak_mem_gb": round(peak, 2)}


@torch.no_grad()
def bench_classifier():
    import h5py
    from common import H5
    from train_classifier import build_model, read_images_cached, to_input
    ck = torch.load(ROOT / "runs_cls" / "exp" / "A_replace0_seed0" / "best.pt", map_location="cpu", weights_only=False)
    model = build_model().to("cuda").to(memory_format=torch.channels_last)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with h5py.File(H5, "r") as f:
        val = read_images_cached(f["images"], split_indices()["val"])  # validation images only
    imgs = [torch.from_numpy(np.ascontiguousarray(val[i:i + 1])) for i in range(CLS_WARMUP + CLS_TIMED)]
    gpu_imgs = [im.to("cuda") for im in imgs]

    def forward(x_gpu):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(to_input(x_gpu, augment=False).contiguous(memory_format=torch.channels_last))

    def end_to_end(x_cpu):
        return torch.softmax(forward(x_cpu.to("cuda")).float(), 1).cpu()

    for i in range(CLS_WARMUP):
        forward(gpu_imgs[i]); end_to_end(imgs[i])
    fwd = [timed(lambda: forward(gpu_imgs[CLS_WARMUP + i]))[0] for i in range(CLS_TIMED)]
    e2e = [timed(lambda: end_to_end(imgs[CLS_WARMUP + i]))[0] for i in range(CLS_TIMED)]
    return {"forward_only_batch1": stats(fwd), "end_to_end_batch1": stats(e2e),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}


def main():
    torch.backends.cudnn.benchmark = False      # as used in training/evaluation
    torch.backends.cudnn.deterministic = True
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "python": platform.python_version(), "date": time.strftime("%Y-%m-%d %H:%M")}
    print("environment:", env, flush=True)
    gen = bench_generator()
    print("generator:", json.dumps(gen, indent=1), flush=True)
    cls = bench_classifier()
    print("classifier:", json.dumps(cls, indent=1), flush=True)
    res = {"environment": env, "generator": gen, "classifier": cls,
           "settings": {"generator": "checkpoint-10000, guidance 3, 30 DPM-Solver++ steps, fp16, 512 px (+ resize to 256)",
                        "classifier": "ResNet-18, 256 px, bf16 autocast, channels_last, batch 1"}}
    (OUT / "inference_speed.json").write_text(json.dumps(res, indent=2))

    sys.path.insert(0, str(HERE))
    from make_tables import write
    g1, g20, cf, ce = gen["batch1"], gen["batch20_per_image"], cls["forward_only_batch1"], cls["end_to_end_batch1"]
    fmt = lambda s: f"{s['mean_ms']:.1f}"  # noqa: E731
    rows = [
        ["Diffusion generator", "batch 1 (latency)", fmt(g1), f"{g1['median_ms']:.1f}", f"{g1['p95_ms']:.1f}",
         f"{1000 / g1['mean_ms']:.2f}", g1["n"]],
        ["Diffusion generator", "batch 20 (as used), per image", fmt(g20), f"{g20['median_ms']:.1f}", f"{g20['p95_ms']:.1f}",
         f"{1000 / g20['mean_ms']:.2f}", g20["n"] * 20],
        ["ResNet-18 classifier", "batch 1, GPU forward only", f"{cf['mean_ms']:.2f}", f"{cf['median_ms']:.2f}",
         f"{cf['p95_ms']:.2f}", f"{1000 / cf['mean_ms']:.0f}", cf["n"]],
        ["ResNet-18 classifier", "batch 1, end-to-end", f"{ce['mean_ms']:.2f}", f"{ce['median_ms']:.2f}",
         f"{ce['p95_ms']:.2f}", f"{1000 / ce['mean_ms']:.0f}", ce["n"]],
    ]
    write("t13_inference_speed", f"Inference speed on one {env['gpu']} (ms per image)",
          ["Model", "Mode", "Mean ms", "Median ms", "p95 ms", "Images/s", "Timed images"], rows,
          "Generator: checkpoint-10000, guidance 3, 30 DPM-Solver++ steps, fp16, 512 px incl. VAE decode "
          f"(+ {gen['resize_512_to_256_ms']['mean_ms']:.1f} ms CPU resize to 256 px, not included). "
          "Classifier: ResNet-18 at 256 px, bf16; end-to-end = image upload + normalisation + forward + softmax "
          "download. Warm-up excluded; GPU synchronised around every timed call.")


if __name__ == "__main__":
    main()

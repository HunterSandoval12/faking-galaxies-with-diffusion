"""All paper tables -> results/tables/tNN_*.csv (data), .md (readable), .tex (booktabs, IEEE-ready).

Usage: python analysis/make_tables.py   (run analysis/compute_extras.py first)
Every number is recomputed from saved results with the same methods as the pre-registered
evaluation (see analysis/common.py). CIs are 95% paired bootstrap over evaluation images.
"""

import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (A_SHARES, CACHE, CIGAR, R_SHARES, ROC_CONDS, ROOT, ROUND, SHORT, TAB, auc_difference,  # noqa: E402
                    auc_summary, difference, fid_lookup, fidelity_rows, labels_all, per_class_metrics, pilot_metrics,
                    split_indices, summary)


def pct(s, d=1):
    return f"{100 * s['point']:.{d}f} [{100 * s['lo']:.{d}f}, {100 * s['hi']:.{d}f}]"


def f3(s):
    return f"{s['point']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]"


def dpts(d):
    return f"{100 * d['point']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"


def d3(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def tex_escape(v):
    return str(v).replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("_", "\\_").replace("&", "\\&").replace("#", "\\#")


def write(name, title, header, rows, notes=""):
    with open(TAB / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    if notes:
        md += ["", notes]
    (TAB / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    cols = "l" + "r" * (len(header) - 1)
    tex = ["\\begin{table}[t]", "\\centering", "\\footnotesize", f"\\caption{{{tex_escape(title)}}}",
           f"\\label{{tab:{name}}}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
           " & ".join(tex_escape(h) for h in header) + " \\\\", "\\midrule"]
    tex += [" & ".join(tex_escape(v) for v in r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        tex += [f"\\\\[2pt]\\parbox{{\\linewidth}}{{\\scriptsize {tex_escape(notes)}}}"]
    tex += ["\\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("wrote", name)


def t01_dataset():
    s = np.load(ROOT / "data" / "splits" / "galaxy10_splits.npz")
    lab = labels_all()
    orig = np.bincount(lab, minlength=10)
    dup = np.bincount(lab[s["excluded_indices"]], minlength=10)
    ovl = np.bincount(lab[s["overlap_removed_indices"]], minlength=10)
    rows = []
    for c in range(10):
        tr, va, te = (int((lab[s[k]] == c).sum()) for k in ("train", "val", "test"))
        rows.append([f"{c} {SHORT[c]}", orig[c], dup[c], ovl[c], tr + va + te, tr, va, te])
    tot = [sum(r[i] for r in rows) for i in range(1, 8)]
    rows.append(["Total"] + tot)
    write("t01_dataset", "Dataset composition after cleaning (Galaxy10 DECaLS)",
          ["Class", "Original", "Exact duplicates removed", "Overlapping cutouts removed", "Final", "Working pool (train)",
           "Validation", "Test"], rows,
          "Exact duplicates: pixel-identical images with conflicting labels, removed before the stratified 70/15/15 split. "
          "Overlapping cutouts: validation/test images covering >= 25% of the same sky as an image in another split, "
          "removed after splitting (working pool unchanged).")


def training_minutes(cond):
    """Mean wall-clock minutes of a classifier run (30 epochs incl. per-epoch validation), from its training logs."""
    ends = [[json.loads(l) for l in open(p) if l.strip()][-1]["elapsed_s"]
            for p in glob.glob(str(ROOT / "runs_cls" / "exp" / f"{cond}_seed*" / "train_log.jsonl"))]
    return np.mean(ends) / 60, len(ends)


def generation_minutes():
    """Wall-clock minutes of the final 12,330-image synthetic set, from its generation log (a UTF-16 PowerShell log)."""
    raw = (ROOT / "data" / "synthetic" / "main_cfg3_steps30" / "generation.log").read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    return float(re.findall(r"([\d.]+) min elapsed", text)[-1])


def t02_hyperparameters():
    c = json.load(open(ROOT / "runs" / "lora_r8" / "train_config.json"))
    k = json.load(open(ROOT / "runs_cls" / "exp" / "A_replace0_seed0" / "config.json"))
    ms = json.load(open(ROOT / "results" / "model_stats.json"))
    g, hw = ms["generator"], ms["hardware"]
    lora_log = [json.loads(l) for l in open(ROOT / "runs" / "lora_r8" / "train_log.jsonl") if l.strip()]
    lora_min, lora_epochs = lora_log[-1]["elapsed_s"] / 60, lora_log[-1]["epoch"]
    gen_min = generation_minutes()
    cls_times = [(lab, training_minutes(cond)[0]) for cond, lab in ROC_CONDS]
    rows = [
        ["Diffusion", "Base model", c["pretrained_model"] + " (Stable Diffusion 1.5)"],
        ["Diffusion", "Fine-tuning", f"LoRA rank {c['lora_rank']}, alpha {c['lora_alpha']:g} on to_q/to_k/to_v/to_out.0"],
        ["Diffusion", "Conditioning", "learned 11 x 77 x 768 class table (10 classes + null), CLIP-initialised"],
        ["Diffusion", "Optimiser", f"AdamW, LR {c['learning_rate']:g}, weight decay {c['adam_weight_decay']:g}, "
                                   f"{c['lr_scheduler']} ({c['lr_warmup_steps']} warmup)"],
        ["Diffusion", "Training", f"{c['max_train_steps']:,} steps (= {lora_epochs:.1f} epochs of the 12,330 training "
                                  f"images), batch {c['train_batch_size']}, {c['resolution']}px, {c['mixed_precision']}, "
                                  f"cond. dropout {c['cond_dropout']}, flips + 90-deg rotations, seed {c['seed']}"],
        ["Diffusion", "Parameters", f"U-Net {g['unet_parameters_frozen']:,} (frozen) + LoRA {g['lora_parameters_trainable']:,} "
                                    f"on {g['lora_adapted_layers']} attention projections (trained) + class table "
                                    f"{g['class_table_parameters_trainable']:,} (trained); VAE {g['vae_parameters_frozen']:,} "
                                    f"(frozen); CLIP text encoder {g['clip_text_encoder_parameters']:,} (used once, to "
                                    "initialise the class table)"],
        ["Diffusion", "Training time", f"{lora_min:.0f} min for {c['max_train_steps']:,} steps "
                                       f"({60 * lora_min / c['max_train_steps']:.2f} s/step)"],
        ["Diffusion", "Selected checkpoint", "10,000 steps (chosen on validation KID + class fidelity)"],
        ["Sampling", "Sampler", "DPM-Solver++ (2nd order), 30 steps, classifier-free guidance 3.0"],
        ["Sampling", "Output", "512 px generated, Lanczos-downsampled to native 256 px"],
        ["Sampling", "Generation time", f"{gen_min:.1f} min for the 12,330-image synthetic set "
                                        f"({60 * gen_min / 12330:.2f} s/image, batches of 20)"],
        ["Classifier", "Model", "ResNet-18, ImageNet-pretrained, 256 px input"],
        ["Classifier", "Parameters", f"{ms['resnet18']['parameters']:,} (all trained; ImageNet ResNet-18 with a new "
                                     "10-class output layer)"],
        ["Classifier", "Optimiser", f"AdamW, LR {k['learning_rate']:g} (tuned on validation from 1e-4 / 3e-4 / 1e-3), "
                                    f"weight decay {k['weight_decay']:g}, 1 warmup epoch + cosine"],
        ["Classifier", "Training", f"{k['epochs']} epochs, batch {k['batch_size']}, bf16, flips + 90-deg rotations, "
                                   "best epoch by validation accuracy"],
        ["Classifier", "Seeds", "3 per condition (8 for real-only, Cigar-Shaped and scarcity-control conditions)"],
        ["Classifier", "Training time", "; ".join(f"{lab} {m:.1f} min" for lab, m in cls_times)
         + " per run (30 epochs incl. per-epoch validation; mean over seeds)"],
        ["Evaluation", "Realism", "per-class FID / KID (Inception-v3 pool features) vs validation"],
        ["Evaluation", "Classification", "accuracy, macro-F1, per-class P/R/F1 on the held-out test set (used once); "
                                         "95% paired bootstrap CIs, 2,000 resamples"],
        ["Hardware", "All training and inference", f"1x {hw['gpu']} ({hw['gpu_memory_gib']:.0f} GB); {hw['cpu']}; "
                                                   f"{hw['ram_gib']:.0f} GB RAM; Windows 11; PyTorch {hw['torch']}, "
                                                   f"CUDA {hw['cuda']}"],
    ]
    write("t02_hyperparameters", "Model and training settings", ["Stage", "Setting", "Value"], rows)


def t13_inference_speed():
    """Measured speed (results/inference_speed.json, analysis/benchmark_inference.py) + FLOPs (results/model_stats.json)."""
    sp = json.load(open(ROOT / "results" / "inference_speed.json"))
    ms = json.load(open(ROOT / "results" / "model_stats.json"))
    gen, cls, g, r = sp["generator"], sp["classifier"], ms["generator"], ms["resnet18"]
    g1, g20, cf, ce = gen["batch1"], gen["batch20_per_image"], cls["forward_only_batch1"], cls["end_to_end_batch1"]
    gfl = lambda f: f"{f / 1e9:,.1f}"  # noqa: E731
    rows = [
        ["Diffusion generator", "batch 1 (latency), per image", f"{g1['mean_ms']:.1f}", f"{g1['median_ms']:.1f}",
         f"{g1['p95_ms']:.1f}", f"{1000 / g1['mean_ms']:.2f}", g1["n"], gfl(g["flops_per_generated_image"])],
        ["Diffusion generator", "batch 20 (as used), per image", f"{g20['mean_ms']:.1f}", f"{g20['median_ms']:.1f}",
         f"{g20['p95_ms']:.1f}", f"{1000 / g20['mean_ms']:.2f}", g20["n"] * 20, gfl(g["flops_per_generated_image"])],
        ["- U-Net + LoRA", "one denoising step (guidance: 2 evaluations)", "-", "-", "-", "-", "-",
         gfl(g["unet_flops_per_guided_step"])],
        ["- VAE decoder", "one 512 px image", "-", "-", "-", "-", "-", gfl(g["vae_decode_flops"])],
        ["ResNet-18 classifier", "batch 1, GPU forward only", f"{cf['mean_ms']:.2f}", f"{cf['median_ms']:.2f}",
         f"{cf['p95_ms']:.2f}", f"{1000 / cf['mean_ms']:.0f}", cf["n"], gfl(r["flops_per_image"])],
        ["ResNet-18 classifier", "batch 1, end-to-end", f"{ce['mean_ms']:.2f}", f"{ce['median_ms']:.2f}",
         f"{ce['p95_ms']:.2f}", f"{1000 / ce['mean_ms']:.0f}", ce["n"], gfl(r["flops_per_image"])],
    ]
    write("t13_inference_speed", f"Inference speed and compute on one {sp['environment']['gpu']} (per image)",
          ["Model", "Mode", "Mean ms", "Median ms", "p95 ms", "Images/s", "Timed images", "GFLOPs"], rows,
          f"Generator: checkpoint-10000, guidance 3, {g['steps']} DPM-Solver++ steps, fp16, 512 px incl. VAE decode "
          f"(+ {gen['resize_512_to_256_ms']['mean_ms']:.1f} ms CPU resize to 256 px, not included); one image = "
          f"{g['steps']} guided U-Net steps + one VAE decode; one U-Net evaluation = "
          f"{g['unet_flops_per_evaluation'] / 1e9:,.1f} GFLOPs. Classifier: ResNet-18 at 256 px, bf16; end-to-end = "
          "image upload + normalisation + forward + softmax download. Warm-up excluded; GPU synchronised around every "
          "timed call. FLOPs: PyTorch FlopCounterMode, 2 FLOPs per multiply-accumulate, matrix multiplications, "
          "convolutions and attention (analysis/model_stats.py); '-' = not timed separately.")


def t14_roc_auc():
    """One-vs-rest ROC AUC on the VALIDATION set (the test evaluation saved no scores; the test set was not re-run)."""
    rows = []
    for j in range(11):
        s = {cond: auc_summary(cond, j) for cond, _ in ROC_CONDS}
        rows.append([SHORT[j] if j < 10 else "Macro AUC"] + [f3(s[cond]) for cond, _ in ROC_CONDS]
                    + [d3(auc_difference("A_replace1", "A_replace0", j)), d3(auc_difference("A_replace0.5", "A_replace0", j))])
    seeds = " / ".join(str(auc_summary(cond, 10)["seeds"]) for cond, _ in ROC_CONDS)
    write("t14_roc_auc_val", "One-vs-rest ROC AUC per class on the validation set",
          ["Class"] + [f"{lab} AUC [95% CI]" for _, lab in ROC_CONDS] + ["Synthetic only - real only", "50% - real only"],
          rows, f"Validation set (2,607 images); seed means ({seeds} seeds); 95% paired bootstrap CIs over validation "
                "images (2,000 resamples). Macro AUC = mean of the 10 one-vs-rest AUCs. Computed on validation because "
                "the one-time test evaluation saved only predicted classes, not scores; the test set was not re-run.")


def t03_generator_selection():
    fr = fidelity_rows(ROOT / "runs_cls" / "class_fidelity.jsonl")
    cands = [("lora_r8", 10000, "seed 42 @ 10,000"), ("lora_r8_seed43", 4000, "seed 43 @ 4,000"),
             ("lora_r8", 4000, "seed 42 @ 4,000")]
    rows = []
    for run, ck, lab in cands:
        for g in (1.5, 2.0, 3.0):
            r = fid_lookup(run, ck, g, 30)
            f = [x["macro_fidelity"] for x in fr if x["run"] == run and x["checkpoint"] == f"checkpoint-{ck}"
                 and x["guidance_scale"] == g and x["sampling_steps"] == 30 and x["seed"] == 0]
            rows.append([lab, f"{g:g}", f"{1e3 * r['macro_kid']:.1f}", f"{r['macro_fid']:.1f}", f"{np.mean(f):.3f}"])
    write("t03_generator_selection", "Generator candidates on validation (30 steps, 100 images/class)",
          ["Candidate", "Guidance", "Macro KID x1e3", "Macro FID", "Class fidelity"], rows,
          "Class fidelity: fraction of generated images a real-trained ResNet-18 assigns to the intended class "
          "(mean of 2 classifiers). Chosen: seed 42 @ 10,000 steps, guidance 3 (confirmed by the usefulness pilot).")


def t04_lora_sweep():
    runs = [("lora_r8", "baseline (rank 8, LR 1e-4, seed 42)"), ("lora_r8_seed43", "baseline, seed 43"),
            ("lora_r4", "rank 4"), ("lora_r16", "rank 16"), ("lora_r8_lr5e-5", "LR 5e-5"), ("lora_r8_lr2e-4", "LR 2e-4"),
            ("lora_r8_cosine", "cosine LR schedule")]
    cks = [2000, 3000, 4000, 5000, 6000, 8000, 10000]
    rows = []
    for r, lab in runs:
        vals = []
        for c in cks:
            x = fid_lookup(r, c, 2.0, 30)
            vals.append(f"{1e3 * x['macro_kid']:.1f}" if x else "-")
        rows.append([lab] + vals)
    write("t04_lora_sweep", "LoRA sweep: macro KID x1e3 vs validation by checkpoint (guidance 2, 30 steps)",
          ["Run"] + [f"{c:,}" for c in cks], rows,
          "Lower is better. '-' = not scored. Same-settings seeds differ by up to 26.8 at 10,000 steps; no rank/LR "
          "setting beats that noise. Checkpoint 4,000 is the most consistent across runs.")


def t05_pilots():
    fr_g = fidelity_rows(ROOT / "runs_cls" / "class_fidelity.jsonl")
    fr_s = fidelity_rows(ROOT / "runs_cls" / "class_fidelity_steps_pilot.jsonl")
    rows = []
    for g in (1.5, 2.0, 3.0, 4.0, 6.0):
        r = fid_lookup("lora_r8", 10000, g, 30)
        f = np.mean([x["macro_fidelity"] for x in fr_g if x["run"] == "lora_r8" and x["checkpoint"] == "checkpoint-10000"
                     and x["guidance_scale"] == g and x["sampling_steps"] == 30 and x["seed"] == 0])
        a, f1 = pilot_metrics(f"runs_cls/pilot_guidance/synthetic_cfg{g:g}_seed*")
        rows.append(["guidance", f"{g:g}", 30, f"{1e3 * r['macro_kid']:.1f}", f"{f:.3f}",
                     f"{100 * np.mean(a):.1f} ({100 * min(a):.1f}-{100 * max(a):.1f})", f"{np.mean(f1):.3f}"])
    for s in (30, 50, 75, 100):
        r = fid_lookup("lora_r8", 10000, 3.0, s)
        f = np.mean([x["macro_fidelity"] for x in fr_s if x["sampling_steps"] == s and x["guidance_scale"] == 3.0])
        pat = "runs_cls/pilot_guidance/synthetic_cfg3_seed*" if s == 30 else f"runs_cls/pilot_steps/synthetic_cfg3_steps{s}_seed*"
        a, f1 = pilot_metrics(pat)
        rows.append(["steps", "3", s, f"{1e3 * r['macro_kid']:.1f}", f"{f:.3f}",
                     f"{100 * np.mean(a):.1f} ({100 * min(a):.1f}-{100 * max(a):.1f})", f"{np.mean(f1):.3f}"])
    a, f1 = pilot_metrics("runs_cls/pilot_guidance/real_100perclass_seed*")
    rows.append(["control", "real 100/class", "-", "-", "-", f"{100 * np.mean(a):.1f} ({100 * min(a):.1f}-{100 * max(a):.1f})",
                 f"{np.mean(f1):.3f}"])
    write("t05_sampling_pilots", "Sampling-setting pilots on validation (checkpoint 10,000; 1,000 images per setting)",
          ["Pilot", "Guidance", "Steps", "Macro KID x1e3", "Class fidelity", "Synthetic-only val. acc % (2 seeds)",
           "Macro-F1"], rows,
          "Usefulness = ResNet-18 trained only on the 1,000 generated images, scored on real validation images. "
          "Chosen: guidance 3, 30 steps (fewest steps within 2 points of the best).")


def t06_replacement():
    rows = []
    for s in A_SHARES:
        c = f"A_replace{s}"
        t, f, v = summary("test", c, "acc"), summary("test", c, "macro_f1"), summary("val", c, "acc")
        d = difference("test", c, "A_replace0", "acc") if s != "0" else None
        rows.append([f"{100 * float(s):g}%", pct(t), f3(f), dpts(d) if d else "-", f"{100 * v['point']:.1f}", t["seeds"]])
    write("t06_replacement_curve", "Replacing real with synthetic training data (total fixed at 12,330)",
          ["Synthetic share", "Test accuracy % [95% CI]", "Test macro-F1 [95% CI]", "Delta acc. vs real-only (pts)",
           "Val. accuracy %", "Seeds"], rows,
          "Synthetic-only reaches 77.3% of real-only test accuracy (73.7% of macro-F1).")


def t07_paired_real():
    rows = []
    for s in sorted(R_SHARES, key=lambda x: -float(x)):
        a, r = summary("test", f"A_replace{s}", "acc"), summary("test", f"R_realpart{s}", "acc")
        d = difference("test", f"A_replace{s}", f"R_realpart{s}", "acc")
        n_real = int(round((1 - float(s)) * 12330))
        rows.append([f"{100 * (1 - float(s)):g}% (~{n_real:,})", pct(r), pct(a), dpts(d)])
    write("t07_value_of_synthetic", "What synthetic images add, by amount of real data (paired: identical real images)",
          ["Real data available", "Real only: test acc. %", "+ synthetic (to 12,330): test acc. %", "Gain (pts) [95% CI]"],
          rows, "The gain grows as real data gets scarcer and is clear only at 10% real.")


def t08_augmentation():
    rows = []
    for c, lab in (("A_replace0", "Real only (12,330)"), ("B_add0.25", "+25% synthetic"), ("B_add0.5", "+50% synthetic"),
                   ("B_add1", "+100% synthetic")):
        t, f = summary("test", c, "acc"), summary("test", c, "macro_f1")
        d = difference("test", c, "A_replace0", "acc") if c != "A_replace0" else None
        rows.append([lab, pct(t), f3(f), dpts(d) if d else "-", t["seeds"]])
    write("t08_augmentation", "Adding synthetic images on top of all real data",
          ["Training data", "Test accuracy % [95% CI]", "Test macro-F1 [95% CI]", "Delta acc. vs real-only (pts)", "Seeds"],
          rows, "All differences include 0: no measurable effect.")


def t09_focus():
    rows = []
    panels = [(ROUND, "C_round_x{k}", "A_replace0", "Round Smooth (1,844 real)"),
              (CIGAR, "C_cigar_x{k}", "A_replace0", "Cigar-Shaped Smooth (233 real)"),
              (ROUND, "S_roundscarce_x{k}", "S_roundscarce_x0", "Round Smooth cut to 233 (control)")]
    for cls, pat, base, lab in panels:
        for k in (0, 0.5, 1, 2, 4):
            n = base if k == 0 else pat.format(k=f"{k:g}")
            s = summary("test", n, "cls", cls)
            d = difference("test", n, base, "cls", cls) if k != 0 else None
            rows.append([lab, f"{k:g}x", f3(s), d3(d) if d else "-", f"{100 * summary('test', n, 'acc')['point']:.1f}",
                         s["seeds"]])
    write("t09_focus_classes", "Focus-class analysis: class F1 on the test set when adding synthetic images of that class",
          ["Class (real count)", "Synthetic added", "Class F1 [95% CI]", "Delta F1 vs no synthetic [95% CI]",
           "Overall acc. %", "Seeds"], rows,
          "Pre-registered rule: ~20 differences are tested, so isolated CIs excluding 0 are expected by chance; "
          "no consistent pattern in any class.")


def t10_per_class():
    conds = [("A_replace0", "real only"), ("A_replace0.5", "50% synthetic"), ("A_replace1", "synthetic only")]
    m = {c: per_class_metrics("test", c) for c, _ in conds}
    rows = []
    for c in range(10):
        row = [SHORT[c]]
        for cond, _ in conds:
            row += [f"{m[cond]['precision'][c]:.3f}", f"{m[cond]['recall'][c]:.3f}", f"{m[cond]['f1'][c]:.3f}"]
        rows.append(row)
    header = ["Class"] + [f"{lab}: {k}" for _, lab in conds for k in ("P", "R", "F1")]
    write("t10_per_class_test", "Per-class test precision / recall / F1 (seed means)", header, rows)


def t11_final_set():
    rep = json.load(open(ROOT / "data" / "synthetic" / "main_cfg3_steps30" / "checks" / "report.json"))
    fid = json.load(open(CACHE / "final_set_fidelity.json"))
    rows = []
    for c in range(10):
        r, m = rep["realism"][str(c)], rep["memorization"][str(c)]
        rows.append([SHORT[c], r["n_syn"], f"{r['fid_syn']:.1f}", f"{r['fid_ref']:.1f}", f"{1e3 * r['kid_syn']:.1f}",
                     f"{fid['fidelity'][c]:.3f}", f"{fid['real_val_recall'][c]:.3f}", f"{m['syn_median']:.3f}",
                     f"{m['val_median']:.3f}", m["n_syn_above_val_max"]])
    write("t11_final_synthetic_set", "The final synthetic set (12,330 images): realism, class fidelity, memorization",
          ["Class", "Images", "FID synthetic", "FID real ref.", "KID x1e3", "Class fidelity", "Real val. recall",
           "NN sim. synthetic (median)", "NN sim. validation (median)", "Above max real NN sim."], rows,
          "FID/KID vs validation at equal n. NN sim.: cosine similarity (Inception features) to the nearest working-pool "
          "image; 0 synthetic images are closer than the closest unseen validation image (no memorization).")


def t12_robustness():
    """Validation-only robustness check (2026-09-27): classifier trained 50 instead of 30 epochs."""
    rows = []
    gaps = {}
    for tag, root in (("30 epochs (used)", ROOT / "runs_cls" / "exp"), ("50 epochs (check)", ROOT / "runs_cls" / "robustness" / "epochs50")):
        ra = [json.load(open(root / f"A_replace0_seed{s}" / "metrics.json"))["val"]["accuracy"] for s in (0, 1, 2)]
        sa = [json.load(open(root / f"A_replace1_seed{s}" / "metrics.json"))["val"]["accuracy"] for s in (0, 1, 2)]
        gaps[tag] = [a - b for a, b in zip(ra, sa)]
        rows.append([tag, f"{100 * np.mean(ra):.1f} +/- {100 * np.std(ra, ddof=1):.1f}",
                     f"{100 * np.mean(sa):.1f} +/- {100 * np.std(sa, ddof=1):.1f}", f"{100 * np.mean(gaps[tag]):.1f}",
                     f"{100 * np.mean(sa) / np.mean(ra):.1f}%"])
    write("t12_robustness_epochs", "Robustness check (validation only): classifier training length",
          ["Classifier training", "Real-only val. acc. %", "Synthetic-only val. acc. %", "Gap (pts)", "Synthetic as % of real"],
          rows, "Seeds 0-2 for both. 50 epochs changes the gap by "
                f"{100 * (np.mean(gaps['50 epochs (check)']) - np.mean(gaps['30 epochs (used)'])):+.1f} points: the "
                "conclusion is not sensitive to the schedule. The test set was not used.")


if __name__ == "__main__":
    only = sys.argv[1:]
    for fn in (t01_dataset, t02_hyperparameters, t03_generator_selection, t04_lora_sweep, t05_pilots, t06_replacement,
               t07_paired_real, t08_augmentation, t09_focus, t10_per_class, t11_final_set, t12_robustness,
               t13_inference_speed, t14_roc_auc):
        if not only or any(o in fn.__name__ for o in only):
            fn()

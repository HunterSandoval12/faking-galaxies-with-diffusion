"""Integrity, realism and memorization checks for a generated synthetic set.

Usage:
    python diffusion/check_synthetic.py data/synthetic/main_cfg3_steps30

1. Integrity: shard contents vs manifest counts, unique seeds, image shapes.
2. Realism per class: FID/KID of the synthetic images vs the VALIDATION set, next to a
   reference of REAL working-pool images scored the same way. (For the main set the
   per-class counts equal the working pool's, so the reference uses the same n.)
3. Memorization: cosine similarity (Inception pool features) between each synthetic
   image and its nearest working-pool image, compared with the same statistic for
   unseen real VALIDATION images (what "novel but realistic" looks like). Copies would
   show up as synthetic images unusually close to the working pool. The closest pairs
   are saved as an image grid for visual inspection.
Writes <dir>/checks/report.json, summary.txt, closest_pairs.png. Never reads test data.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_splits import CLASS_NAMES  # noqa: E402
from eval_fid import InceptionFeatures, fid_from_features, kid_from_features  # noqa: E402
from sample import GALAXY10_H5  # noqa: E402

SPLITS = PROJECT_ROOT / "data" / "splits" / "galaxy10_splits.npz"


def real_features(extractor, key, chunk=1024):
    assert key in ("train", "val"), "never reads the test split"
    idx = np.load(SPLITS)[key]
    with h5py.File(GALAXY10_H5, "r") as f:
        labels = f["ans"][:].astype(np.int64)[idx]
        feats = np.concatenate([extractor(f["images"][idx[s:s + chunk]]) for s in range(0, len(idx), chunk)])
    return feats, labels, idx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("synthetic_dir", type=Path)
    p.add_argument("--kid-subsets", type=int, default=100)
    p.add_argument("--grid-pairs", type=int, default=16)
    args = p.parse_args()
    out = args.synthetic_dir / "checks"
    out.mkdir(exist_ok=True)
    manifest = json.loads((args.synthetic_dir / "manifest.json").read_text())
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    # 1. integrity + synthetic features (shard by shard to bound memory)
    extractor = InceptionFeatures(torch.device("cuda"))
    shards = sorted(glob.glob(str(args.synthetic_dir / "class*_part*.npz")))
    s_feats, s_labels, s_seeds, s_where = [], [], [], []
    for sh in shards:
        z = np.load(sh)
        assert z["images"].dtype == np.uint8 and z["images"].shape[1:] == (256, 256, 3), sh
        s_feats.append(extractor(z["images"]))
        s_labels.append(z["labels"])
        s_seeds.append(z["seeds"])
        s_where += [(sh, i) for i in range(len(z["labels"]))]
    s_feats, s_labels, s_seeds = np.concatenate(s_feats), np.concatenate(s_labels), np.concatenate(s_seeds)
    counts = np.bincount(s_labels, minlength=len(CLASS_NAMES))
    expected = {int(c): n for c, n in manifest["per_class_counts"].items()}
    counts_ok = all(counts[c] == n for c, n in expected.items())
    seeds_unique = len(np.unique(s_seeds)) == len(s_seeds)
    say(f"1. INTEGRITY: {len(shards)} shards, {len(s_labels)} images (manifest total {manifest['total']}); "
        f"per-class counts match manifest: {counts_ok}; seeds unique: {seeds_unique}")

    # real features
    t_feats, t_labels, t_idx = real_features(extractor, "train")
    v_feats, v_labels, _ = real_features(extractor, "val")

    # 2. realism per class
    say("\n2. REALISM vs validation (per class; reference = real working-pool images, same scoring)")
    say(f"  {'class':<24}{'n_syn':>6}{'n_ref':>6}{'n_val':>6}{'FID syn':>9}{'FID ref':>9}{'KIDx1e3 syn':>13}{'KIDx1e3 ref':>13}")
    realism = {}
    for c in sorted(expected):
        sf, tf, vf = s_feats[s_labels == c], t_feats[t_labels == c], v_feats[v_labels == c]
        tf = tf[np.random.default_rng(c).choice(len(tf), min(len(tf), len(sf)), replace=False)]  # same n as synthetic
        r = {"n_syn": len(sf), "n_ref": len(tf), "n_val": len(vf),
             "fid_syn": fid_from_features(sf, vf), "fid_ref": fid_from_features(tf, vf),
             "kid_syn": kid_from_features(sf, vf, args.kid_subsets, np.random.default_rng(c))[0],
             "kid_ref": kid_from_features(tf, vf, args.kid_subsets, np.random.default_rng(c))[0]}
        realism[c] = r
        say(f"  {c}: {CLASS_NAMES[c]:<21}{r['n_syn']:>6}{r['n_ref']:>6}{r['n_val']:>6}{r['fid_syn']:>9.1f}{r['fid_ref']:>9.1f}"
            f"{1e3 * r['kid_syn']:>13.2f}{1e3 * r['kid_ref']:>13.2f}")
    say(f"  {'macro average':<24}{'':>18}{np.mean([r['fid_syn'] for r in realism.values()]):>9.1f}"
        f"{np.mean([r['fid_ref'] for r in realism.values()]):>9.1f}"
        f"{1e3 * np.mean([r['kid_syn'] for r in realism.values()]):>13.2f}{1e3 * np.mean([r['kid_ref'] for r in realism.values()]):>13.2f}")

    # 3. memorization: nearest working-pool neighbor (cosine similarity of Inception features)
    dev = torch.device("cuda")
    T = torch.nn.functional.normalize(torch.from_numpy(t_feats).float().to(dev), dim=1)

    def nn_sim(x):
        X = torch.nn.functional.normalize(torch.from_numpy(x).float().to(dev), dim=1)
        sims, idx = [], []
        for s in range(0, len(X), 2048):
            v, i = (X[s:s + 2048] @ T.T).max(1)
            sims.append(v.cpu()); idx.append(i.cpu())
        return torch.cat(sims).numpy(), torch.cat(idx).numpy()

    s_sim, s_nn = nn_sim(s_feats)
    v_sim, _ = nn_sim(v_feats)
    say("\n3. MEMORIZATION: cosine similarity to the NEAREST working-pool image (1.0 = identical features)")
    say("   Baseline = unseen real validation images. Copies would sit far above the baseline.")
    say(f"  {'class':<24}{'syn median':>11}{'val median':>11}{'syn 99th':>10}{'val 99th':>10}{'syn > val max':>15}")
    memo = {}
    for c in sorted(expected):
        ss, vs = s_sim[s_labels == c], v_sim[v_labels == c]
        m = {"syn_median": float(np.median(ss)), "val_median": float(np.median(vs)),
             "syn_p99": float(np.percentile(ss, 99)), "val_p99": float(np.percentile(vs, 99)),
             "val_max": float(vs.max()), "n_syn_above_val_max": int((ss > vs.max()).sum()), "n_syn": int(len(ss))}
        memo[c] = m
        say(f"  {c}: {CLASS_NAMES[c]:<21}{m['syn_median']:>11.3f}{m['val_median']:>11.3f}{m['syn_p99']:>10.3f}"
            f"{m['val_p99']:>10.3f}{m['n_syn_above_val_max']:>9} / {m['n_syn']}")
    thr = float(v_sim.max())
    n_above = int((s_sim > thr).sum())
    say(f"  overall: synthetic median {np.median(s_sim):.3f} vs validation median {np.median(v_sim):.3f}; "
        f"{n_above} of {len(s_sim)} synthetic images are closer to a training image than ANY validation image is ({thr:.3f})")

    # closest pairs grid: synthetic | nearest working-pool image, with pixel correlation
    order = np.argsort(-s_sim)[:args.grid_pairs]
    tile, pairs = 160, []
    grid = Image.new("RGB", (4 * 2 * tile + 3 * 12, ((len(order) + 3) // 4) * (tile + 22)), "white")
    draw = ImageDraw.Draw(grid)
    with h5py.File(GALAXY10_H5, "r") as f:
        for n, i in enumerate(order):
            sh, j = s_where[i]
            syn = np.load(sh)["images"][j]
            real = f["images"][t_idx[s_nn[i]]]
            pc = float(np.corrcoef(syn.astype(float).ravel(), real.astype(float).ravel())[0, 1])
            pairs.append({"shard": Path(sh).name, "row": int(j), "class": int(s_labels[i]), "cosine": float(s_sim[i]),
                          "train_index": int(t_idx[s_nn[i]]), "pixel_corr": pc})
            x, y = (n % 4) * (2 * tile + 12), (n // 4) * (tile + 22)
            grid.paste(Image.fromarray(syn).resize((tile, tile)), (x, y))
            grid.paste(Image.fromarray(real).resize((tile, tile)), (x + tile, y))
            draw.text((x + 2, y + tile + 4), f"c{s_labels[i]} cos {s_sim[i]:.3f} px {pc:.2f}", fill="black")
    grid.save(out / "closest_pairs.png")
    say(f"\n  closest {len(order)} pairs saved to {out / 'closest_pairs.png'} (left = synthetic, right = nearest "
        f"training image); pixel correlation of those pairs: max {max(p['pixel_corr'] for p in pairs):.2f}, "
        f"median {np.median([p['pixel_corr'] for p in pairs]):.2f}")

    report = {"manifest": manifest, "integrity": {"images": int(len(s_labels)), "counts_ok": bool(counts_ok),
              "seeds_unique": bool(seeds_unique), "per_class": counts.tolist()},
              "realism": {str(c): v for c, v in realism.items()}, "memorization": {str(c): v for c, v in memo.items()},
              "memorization_overall": {"syn_median": float(np.median(s_sim)), "val_median": float(np.median(v_sim)),
                                       "val_max": thr, "n_syn_above_val_max": n_above},
              "closest_pairs": pairs}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out / 'report.json'} and {out / 'summary.txt'}")


if __name__ == "__main__":
    main()

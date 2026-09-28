"""GPU pre-computations for the figures (cached in results/cache/). No test data is read.

1. Memorization distributions: cosine similarity (Inception pool features) of every
   synthetic image of the main set, and of every validation image, to its nearest
   working-pool image (same method as diffusion/check_synthetic.py).
2. Class fidelity of the FINAL main synthetic set (12,330 images) with the two real-only
   classifiers (runs_cls/real_lr1e-3_seed0/1), + those classifiers' real validation recall.
3. Sample-grid picks: 4 synthetic (sample_index 0-3) and 4 random working-pool images per class.
4. Validation-set class probabilities of the real-only, 50%-mix and synthetic-only classifiers (all
   seeds), for ROC curves and AUC. The one-time test evaluation saved only predicted classes, not
   scores, so ROC/AUC use the validation set; the test set is not read. Each classifier's argmax must
   equal its saved validation predictions. Run this step alone with: compute_extras.py val_probs
"""

import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classifier"))
from common import CACHE, CLASS_NAMES, EXP, H5, ROOT, split_indices  # noqa: E402
from eval_fid import InceptionFeatures  # noqa: E402
from train_classifier import build_model, predict, read_images_cached  # noqa: E402

MAIN = ROOT / "data" / "synthetic" / "main_cfg3_steps30"


def main():
    dev = torch.device("cuda")
    idx = split_indices()
    with h5py.File(H5, "r") as f:
        labels = f["ans"][:].astype(np.int64)
        train_x = read_images_cached(f["images"], idx["train"])
        val_x = read_images_cached(f["images"], idx["val"])
    train_y, val_y = labels[idx["train"]], labels[idx["val"]]
    shards = sorted(MAIN.glob("class*_part*.npz"))
    syn_x = np.concatenate([np.load(s)["images"] for s in shards])
    syn_y = np.concatenate([np.load(s)["labels"] for s in shards])
    syn_k = np.concatenate([np.load(s)["sample_index"] for s in shards])
    print(f"loaded train {len(train_x)}, val {len(val_x)}, synthetic {len(syn_x)}", flush=True)

    # 1. memorization similarities
    ext = InceptionFeatures(dev)
    feats = {n: ext(x) for n, x in (("train", train_x), ("val", val_x), ("syn", syn_x))}
    T = torch.nn.functional.normalize(torch.from_numpy(feats["train"]).float().to(dev), dim=1)

    def nn(x):
        X = torch.nn.functional.normalize(torch.from_numpy(x).float().to(dev), dim=1)
        return torch.cat([(X[s:s + 2048] @ T.T).max(1).values.cpu() for s in range(0, len(X), 2048)]).numpy()
    np.savez(CACHE / "memorization_sims.npz", syn=nn(feats["syn"]), syn_y=syn_y, val=nn(feats["val"]), val_y=val_y)
    print("1. memorization similarities cached", flush=True)

    # 2. class fidelity of the final set
    fid, rec = [], []
    for s in (0, 1):
        ck = torch.load(ROOT / "runs_cls" / f"real_lr1e-3_seed{s}" / "best.pt", map_location="cpu", weights_only=False)
        m = build_model().to(dev).to(memory_format=torch.channels_last)
        m.load_state_dict(ck["state_dict"])
        p = predict(m, syn_x, dev).argmax(1).numpy()
        fid.append([float((p[syn_y == c] == c).mean()) for c in range(len(CLASS_NAMES))])
        rec.append([ck["val_metrics"]["per_class"][str(c)]["recall"] for c in range(len(CLASS_NAMES))])
    json.dump({"fidelity_per_classifier": fid, "real_val_recall_per_classifier": rec,
               "fidelity": np.mean(fid, 0).tolist(), "real_val_recall": np.mean(rec, 0).tolist()},
              open(CACHE / "final_set_fidelity.json", "w"), indent=2)
    print("2. final-set class fidelity cached:", np.round(np.mean(fid, 0), 3).tolist(), flush=True)

    # 3. sample-grid picks
    rng = np.random.default_rng(0)
    real_pick = {c: np.sort(rng.choice(np.flatnonzero(train_y == c), 4, replace=False)) for c in range(10)}
    syn_pick = {c: np.flatnonzero((syn_y == c) & (syn_k < 4)) for c in range(10)}
    np.savez_compressed(CACHE / "sample_grid.npz",
                        real=np.stack([train_x[real_pick[c]] for c in range(10)]),
                        syn=np.stack([syn_x[syn_pick[c]] for c in range(10)]))
    print("3. sample-grid picks cached", flush=True)


def val_probabilities():
    dev = torch.device("cuda")
    with h5py.File(H5, "r") as f:
        val_x = read_images_cached(f["images"], split_indices()["val"])  # validation images only
    out = {}
    for cond in ("A_replace0", "A_replace0.5", "A_replace1"):
        for d in sorted(EXP.glob(f"{cond}_seed*")):
            ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
            model = build_model().to(dev).to(memory_format=torch.channels_last)
            model.load_state_dict(ck["state_dict"])
            p = predict(model, val_x, dev).numpy().astype(np.float32)
            assert np.array_equal(p.argmax(1), np.load(d / "val_predictions.npy")),                 f"{d.name}: probabilities do not reproduce the saved validation predictions"
            out[d.name] = p
    np.savez_compressed(CACHE / "val_probs.npz", **out)
    print(f"4. validation probabilities cached for {len(out)} classifiers (argmax = saved predictions)", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["val_probs"]:
        val_probabilities()
    else:
        main()
        val_probabilities()

"""Create a reproducible stratified 70/15/15 train/val/test split of Galaxy10 DECaLS.

Usage:
    python data/prepare_splits.py [--seed 42] [--out data/splits/galaxy10_splits.npz]

The split is saved as integer indices into the arrays returned by
astroNN.datasets.load_galaxy10(), so downstream code can do:

    splits = np.load("data/splits/galaxy10_splits.npz")
    x_train, y_train = images[splits["train"]], labels[splits["train"]]

Galaxy10 DECaLS contains some galaxies stored more than once (identical pixels,
conflicting labels). All copies of such galaxies are excluded before splitting;
their indices are saved as `excluded_indices` and appear in no split.

It also contains overlapping cutouts: different entries whose 256x256 cutouts cover
largely the same sky (e.g. each galaxy of a merging pair has its own entry). After
the stratified split, any validation/test image whose cutout overlaps an image in
another split by >= OVERLAP_THRESHOLD of its footprint is removed (for val-test pairs,
the val image), as are overlapping pairs with conflicting labels inside val/test.
Removal happens after splitting so the train split (working pool) is unaffected.
Removed indices are saved as `overlap_removed_indices`.
"""

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # astroNN imports TensorFlow

import h5py
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split

# astroNN's Galaxy10Class dict carries a stale name for class 3 (from Galaxy10 SDSS),
# so use the DECaLS names from the astroNN documentation.
CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
CUTOUT_PX = 256
# >= 25% footprint overlap includes every case where one image's central galaxy lies
# inside the other cutout (worst case: offset of half a width on both axes -> 0.5 x 0.5).
OVERLAP_THRESHOLD = 0.25


def footprint_overlap(ra, dec, px, i, j):
    """Fraction of image i's sky footprint covered by image j (flat-sky approximation)."""
    dx = (ra[j] - ra[i]) * np.cos(np.radians(dec[i])) * 3600
    dy = (dec[j] - dec[i]) * 3600
    wi, wj = CUTOUT_PX * px[i], CUTOUT_PX * px[j]
    ox = max(0.0, min(wi / 2, dx + wj / 2) - max(-wi / 2, dx - wj / 2))
    oy = max(0.0, min(wi / 2, dy + wj / 2) - max(-wi / 2, dy - wj / 2))
    return ox * oy / (wi * wi)


def overlapping_pairs(indices, ra, dec, px):
    """(i, j, overlap) for all pairs among `indices` overlapping >= OVERLAP_THRESHOLD."""
    r, d = np.radians(ra[indices]), np.radians(dec[indices])
    xyz = np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]
    radius = np.radians(CUTOUT_PX * px.max() / 3600)  # beyond this, cutouts can't overlap >= 25%
    out = []
    for a, b in cKDTree(xyz).query_pairs(radius, output_type="ndarray"):
        i, j = int(indices[a]), int(indices[b])
        ov = max(footprint_overlap(ra, dec, px, i, j), footprint_overlap(ra, dec, px, j, i))
        if ov >= OVERLAP_THRESHOLD:
            out.append((i, j, ov))
    return out


def overlap_removals(splits, labels, ra, dec, px):
    """Val/test indices to remove so no cutout overlaps across splits (train is never touched)."""
    split_of = {int(i): s for s in SPLIT_NAMES for i in splits[s]}
    kept = np.sort(np.concatenate([splits[s] for s in SPLIT_NAMES]))
    remove, n_cross, n_conflict = set(), 0, 0
    for i, j, _ in overlapping_pairs(kept, ra, dec, px):
        si, sj = split_of[i], split_of[j]
        if si != sj:
            n_cross += 1
            if "train" in (si, sj):
                remove.add(j if si == "train" else i)  # drop the held-out member
            else:
                remove.add(i if si == "val" else j)  # val-test pair: keep the test image
        elif si != "train" and labels[i] != labels[j]:
            n_conflict += 1
            remove.update((i, j))  # contradictory labels for the same sky in an evaluation set
    return np.array(sorted(remove), dtype=np.int64), n_cross, n_conflict


def duplicate_groups(images):
    """Return groups (lists of indices) of pixel-identical images, only groups of size > 1."""
    by_hash = defaultdict(list)
    for i in range(len(images)):
        by_hash[hashlib.sha1(images[i].tobytes()).hexdigest()].append(i)
    return [g for g in by_hash.values() if len(g) > 1]


def stratified_split(indices, labels, seed):
    """Two-stage stratified split of `indices`: 70 / 30, then the 30 halved into 15 / 15."""
    holdout_frac = SPLIT_FRACS["val"] + SPLIT_FRACS["test"]
    train_idx, holdout_idx = train_test_split(
        indices, test_size=holdout_frac, stratify=labels[indices], random_state=seed
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=SPLIT_FRACS["test"] / holdout_frac,
        stratify=labels[holdout_idx],
        random_state=seed,
    )
    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def print_class_table(labels, kept, excluded, splits):
    n_classes = len(CLASS_NAMES)
    original = np.bincount(labels, minlength=n_classes)
    dropped = np.bincount(labels[excluded], minlength=n_classes)
    total = np.bincount(labels[kept], minlength=n_classes)
    counts = {s: np.bincount(labels[idx], minlength=n_classes) for s, idx in splits.items()}

    header = f"{'cls':>3}  {'name':<24}{'orig':>6}{'dropped':>8}{'kept':>6}" + "".join(
        f"{s:>7}{'(%)':>6}" for s in SPLIT_NAMES
    ) + f"{'max dev':>9}"
    print(header)
    print("-" * len(header))
    for c in range(n_classes):
        row = f"{c:>3}  {CLASS_NAMES[c]:<24}{original[c]:>6}{dropped[c]:>8}{total[c]:>6}"
        devs = []
        for s in SPLIT_NAMES:
            frac = counts[s][c] / total[c]
            row += f"{counts[s][c]:>7}{100 * frac:>6.1f}"
            devs.append(abs(frac - SPLIT_FRACS[s]))
        row += f"{100 * max(devs):>8.2f}%"
        print(row)
    print("-" * len(header))
    row = f"{'':>3}  {'ALL':<24}{original.sum():>6}{dropped.sum():>8}{total.sum():>6}"
    for s in SPLIT_NAMES:
        row += f"{counts[s].sum():>7}{100 * counts[s].sum() / total.sum():>6.1f}"
    print(row)
    print("(split % = share of that class's kept images; max dev = worst |split % - target|)")

    # Class proportions within each split vs. the kept dataset.
    print("\nClass proportions within each split (% of split):")
    print(f"{'cls':>3}  {'kept':>7}" + "".join(f"{s:>8}" for s in SPLIT_NAMES))
    for c in range(n_classes):
        row = f"{c:>3}  {100 * total[c] / total.sum():>7.2f}"
        for s in SPLIT_NAMES:
            row += f"{100 * counts[s][c] / counts[s].sum():>8.2f}"
        print(row)


def check_splits(images, kept, excluded, splits, coords):
    """Verify splits are disjoint (by index, pixel content, and sky footprint), cover exactly
    the kept indices, and contain no excluded index. Recomputes everything independently."""
    ok = True

    # 1. Index level.
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            shared = np.intersect1d(splits[a], splits[b])
            print(f"  index overlap {a:>5} & {b:<5}: {len(shared)}")
            ok &= len(shared) == 0
    all_idx = np.sort(np.concatenate([splits[s] for s in SPLIT_NAMES]))
    covered = np.array_equal(all_idx, kept)
    print(f"  splits cover every kept index exactly once: {covered}")
    ok &= covered
    leaked = np.intersect1d(all_idx, excluded)
    print(f"  excluded indices present in any split: {len(leaked)}")
    ok &= len(leaked) == 0

    # 2. Content level: no pixel-identical image anywhere across (or within) the splits.
    seen = {}
    n_dup = 0
    for s in SPLIT_NAMES:
        for i in splits[s]:
            h = hashlib.sha1(images[i].tobytes()).hexdigest()
            if h in seen:
                n_dup += 1
                if n_dup <= 10:
                    print(f"    duplicate: {seen[h][0]}({seen[h][1]}) == {i}({s})")
            else:
                seen[h] = (i, s)
    print(f"  pixel-identical images across all splits: {n_dup}")
    ok &= n_dup == 0

    # 3. Sky level: no cutouts overlapping across splits; no conflicting labels on
    #    overlapping cutouts within val/test.
    ra, dec, px, labels = coords
    split_of = {int(i): s for s in SPLIT_NAMES for i in splits[s]}
    pairs = overlapping_pairs(all_idx, ra, dec, px)
    cross = sum(split_of[i] != split_of[j] for i, j, _ in pairs)
    conflict = sum(split_of[i] == split_of[j] != "train" and labels[i] != labels[j] for i, j, _ in pairs)
    print(f"  cutouts overlapping >= {OVERLAP_THRESHOLD:.0%} across splits: {cross}")
    print(f"  overlapping cutouts with conflicting labels inside val/test: {conflict}")
    ok &= cross == 0 and conflict == 0
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "splits" / "galaxy10_splits.npz")
    args = parser.parse_args()

    from astroNN.datasets import load_galaxy10  # imported here: pulls in TensorFlow

    images, labels = load_galaxy10()
    labels = labels.astype(np.int64)
    print(f"Loaded Galaxy10 DECaLS: images {images.shape} {images.dtype}, labels {labels.shape}\n")

    dups = duplicate_groups(images)
    excluded = np.sort(np.concatenate(dups)) if dups else np.array([], dtype=np.int64)
    n_conflict = sum(len(set(labels[g])) > 1 for g in dups)
    kept = np.setdiff1d(np.arange(len(labels)), excluded)
    print(f"Found {len(dups)} groups of pixel-identical images ({len(excluded)} images), "
          f"{n_conflict} with conflicting labels; excluding all of them -> {len(kept)} images kept.\n")

    splits = stratified_split(kept, labels, args.seed)

    from astroNN.config import astroNN_CACHE_DIR
    with h5py.File(Path(astroNN_CACHE_DIR) / "datasets" / "Galaxy10_DECals.h5", "r") as f:
        ra, dec, px = (f[k][:].astype(np.float64) for k in ("ra", "dec", "pxscale"))
    overlap_removed, n_cross, n_conflict_pairs = overlap_removals(splits, labels, ra, dec, px)
    by_split = {s: int(np.isin(overlap_removed, splits[s]).sum()) for s in SPLIT_NAMES}
    splits = {s: np.setdiff1d(idx, overlap_removed) for s, idx in splits.items()}
    kept = np.setdiff1d(kept, overlap_removed)
    print(f"Overlapping cutouts (>= {OVERLAP_THRESHOLD:.0%} of footprint): {n_cross} pairs across splits, "
          f"{n_conflict_pairs} conflicting-label pairs inside val/test -> removed {len(overlap_removed)} images "
          f"{by_split}; train split unchanged.")
    print("  removed per class: " + ", ".join(
        f"{c}:{n}" for c, n in enumerate(np.bincount(labels[overlap_removed], minlength=len(CLASS_NAMES))) if n))
    print(f"  -> {len(kept)} images in the final splits.\n")

    all_excluded = np.union1d(excluded, overlap_removed)
    print_class_table(labels, kept, all_excluded, splits)

    print("\nSplit checks:")
    if not check_splits(images, kept, all_excluded, splits, (ra, dec, px, labels)):
        raise SystemExit("FAILED: split integrity check")
    print("  OK: splits are disjoint (index, pixels, sky), exclude all duplicates and overlap removals")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        **splits,
        seed=args.seed,
        n_total=len(labels),
        # All copies of pixel-identical (label-conflicting) images; in no split.
        excluded_indices=excluded,
        # Val/test images removed after splitting for overlapping cutouts; in no split.
        overlap_removed_indices=overlap_removed,
        overlap_threshold=OVERLAP_THRESHOLD,
        # Fingerprint of the label array, so a loader can detect a changed dataset file.
        labels_sha1=hashlib.sha1(labels.tobytes()).hexdigest(),
    )
    print(f"\nSaved split indices to {args.out}")


if __name__ == "__main__":
    main()

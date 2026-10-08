"""Builds notebooks/01_Dataset_Tutorial.ipynb (run from the project root).

Contents:
sections/subsections each in their own text cell, scope + references at the top, versioned
requirements with asserts and a skippable install cell, Google Drive mounting, training /
validation / testing directories with storage requirements, hardware, execution times,
expected results, and the Dataset-tutorial items (Drive setup + zip/unzip, displaying images,
inputs, output labels, extending the dataset, setting up the splits).

Usage: python notebooks/_build/build_01_dataset_tutorial.py [--times times.json]
  --times fills the execution-time table and peak memory from a local test run.
"""

import argparse
import json
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbcommon import RELEASE_URL, REPO_URL, SHA256  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--times", type=Path, help="JSON with measured step times (seconds) from a local run")
args = ap.parse_args()


def fmt(seconds):
    return f"~{seconds:.0f} s" if seconds < 90 else f"~{seconds / 60:.0f} min"


FILL = {"__T_DOWNLOAD__": "?", "__T_VERIFY__": "?", "__T_SPLIT__": "?", "__T_EXPORT__": "?", "__T_UNZIP__": "?",
        "__T_TOTAL__": "?", "__T_TOTAL2__": "?", "__PEAK_RAM__": "?", "__REPO_URL__": REPO_URL}
if args.times:
    m = json.loads(args.times.read_text())
    FILL.update({"__T_DOWNLOAD__": f"{fmt(m['download'])} ({m['download_rate']:.0f} MiB/s)",
                 "__T_VERIFY__": fmt(m["verify + load"]), "__T_SPLIT__": fmt(m["cleaning + split"]),
                 "__T_EXPORT__": fmt(m["write PNG directories"] + m["zip + copy to Drive"]),
                 "__T_UNZIP__": fmt(m["unzip from Drive"]), "__T_TOTAL__": fmt(m["first session"]),
                 "__T_TOTAL2__": fmt(m["later session"]), "__PEAK_RAM__": f"{m['peak_ram_gib']:.1f} GiB"})

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ================================================================================ title + scope
md("# Galaxy10 DECaLS: Dataset Tutorial")
md("""
**Project:** *Faking Galaxies with Diffusion: A Real-vs-Synthetic Benchmark for Morphology Classification*
Hunter Sandoval, University of New Mexico. Tutorial 1 of 6.

**Scope.** This tutorial defines every element of the dataset used in the project, from the download to the
`training` / `validation` / `testing` directories that the other five tutorials read. You will:

1. download **Galaxy10 DECaLS** (17,736 labeled galaxy images, 2.55 GiB) and keep it on Google Drive;
2. learn what the **inputs** are (NumPy arrays holding 256 x 256 RGB images) and what the **output labels** are (10 galaxy morphology classes);
3. **display** images from every class, reproduce the paper's example figures, and compare real galaxies with synthetic ones made by the project's generator;
4. clean the data (remove duplicated and overlapping sky cutouts) and build the paper's **stratified 70/15/15 split**, verified to be *identical* to the split used in the paper;
5. write the split to **`training/`, `validation/` and `testing/` directories** of PNG files, zip them to Google Drive, and unzip them in later sessions;
6. **extend the dataset** by downloading a galaxy image from the DESI Legacy Surveys and adding it to the training directory without leaking held-out data.

This tutorial only prepares data; the models are covered in Tutorials 2-6. In the paper, the `training` directory is
called the *working pool* (the diffusion model and the classifiers are trained on it). `validation` is used for all
tuning and checkpoint selection, and `testing` is the held-out set, used once for the final evaluation.
""")

md("## References")
md("""
1. H. W. Leung and J. Bovy, **Galaxy10 DECaLS** dataset. GitHub: https://github.com/henrysky/Galaxy10 ;
   documentation: https://astronn.readthedocs.io/en/latest/galaxy10.html
2. **astroNN**, the dataset's official loader (we reproduce its download code here): https://github.com/henrysky/astroNN ;
   H. W. Leung and J. Bovy, "Deep learning of multi-element abundances from high-resolution spectroscopic data,"
   MNRAS 483, 3255, 2019, arXiv:1808.04428.
3. M. Walmsley et al., "Galaxy Zoo DECaLS: Detailed Visual Morphology Measurements from Volunteers and Deep Learning
   for 314,000 Galaxies," MNRAS 509, 3966, 2022, arXiv:2102.08414 (source of the class labels).
4. A. Dey et al., "Overview of the DESI Legacy Imaging Surveys," AJ 157, 168, 2019, arXiv:1804.08657 (source of the
   images). Legacy Surveys sky viewer and cutout service: https://www.legacysurvey.org/viewer
5. X. Fan et al., "Category-based Galaxy Image Generation via Diffusion Models" (GalCatDiff), arXiv:2506.16255, 2025
   (the closest prior work to this project).
6. Project code and model weights: __REPO_URL__. `data/prepare_splits.py` is the
   split code reproduced in this notebook.
""")

# ================================================================================ setup
md("# Setup")
md("## Library requirements and installation")
md("""
| Library | Minimum version | Tested version | Install command | Used for |
|---|---|---|---|---|
| Python | 3.10 | 3.14.7 | (Colab's default) | |
| NumPy | 1.23 | 2.5.2 | `pip install "numpy>=1.23"` | arrays |
| h5py | 3.7 | 3.15.1 | `pip install "h5py>=3.7"` | reading the dataset file (HDF5) |
| scikit-learn | 1.1 | 1.9.1 | `pip install "scikit-learn>=1.1"` | stratified split |
| SciPy | 1.9 | 1.18.1 | `pip install "scipy>=1.9"` | sky-coordinate neighbor search |
| Pillow | 9.0 | 12.3.0 | `pip install "pillow>=9.0"` | reading and writing PNG / JPEG images |
| Matplotlib | 3.6 | 3.11.2 | `pip install "matplotlib>=3.6"` | figures |

All of these are **preinstalled on Google Colab**, so the next cell normally installs nothing. It checks each
library and runs `pip` only for a missing or too-old one. On a **second run**, set `SKIP_INSTALL = True` to skip the
check entirely.

**How we figured this out.** The dataset's official loader, `astroNN.datasets.load_galaxy10()`, needs the whole
`astroNN` package, and importing that package also imports TensorFlow and TensorFlow Probability (about 1 GB, and
fragile to install next to Colab's own versions). We printed the loader's source (`import inspect,
astroNN.datasets.galaxy10 as g; print(inspect.getsource(g))`). It turned out to be ~10 lines: download one HDF5 file,
check its SHA-256 checksum, read two datasets with `h5py`. We reproduce those lines below, so only `h5py` is needed.
""")
code("""
SKIP_INSTALL = False  # set to True when re-running the notebook to skip the installation check

import importlib.metadata
import re
import subprocess
import sys

REQUIREMENTS = {"numpy": "1.23", "h5py": "3.7", "scikit-learn": "1.1", "scipy": "1.9", "pillow": "9.0",
                "matplotlib": "3.6"}  # package -> minimum version


def version_tuple(v):
    return tuple(int(x) for x in re.findall(r"\\d+", v)[:3])


def installed_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


if SKIP_INSTALL:
    print("SKIP_INSTALL = True: skipping the installation check.")
else:
    to_install = [f"{pkg}>={minimum}" for pkg, minimum in REQUIREMENTS.items()
                  if installed_version(pkg) is None or version_tuple(installed_version(pkg)) < version_tuple(minimum)]
    if to_install:
        print("Installing:", to_install)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
        print("Done. If a library was upgraded, restart the runtime (Runtime > Restart session) and run again.")
    else:
        print("All required libraries are already installed; nothing to install.")
""")
code("""
# Verify the versions. The notebook stops here with a clear message if a requirement is not met.
assert sys.version_info >= (3, 10), f"Python >= 3.10 required, found {sys.version.split()[0]}"
for pkg, minimum in REQUIREMENTS.items():
    have = installed_version(pkg)
    assert have is not None, f"{pkg} is not installed (pip install '{pkg}>={minimum}')"
    assert version_tuple(have) >= version_tuple(minimum), f"{pkg} {have} is older than the required {minimum}"
    print(f"{pkg:<13} {have:<10} (required >= {minimum})")
print("All version requirements are satisfied.")
""")

md("## Hardware used for running")
md("""
- **No GPU is needed.** Everything in this tutorial runs on the CPU, so the standard (free) Colab **CPU runtime** is
  enough. Tutorials 2-6 train models and need a GPU runtime.
- **Memory:** about __PEAK_RAM__ of RAM at peak, mostly the full dataset held in memory as a 3.25 GiB NumPy array.
  The standard Colab runtime has about 12 GB.
- **Disk:** about 10 GB of the Colab machine's local disk (it has over 100 GB) and about 5 GB of Google Drive.
- **Tested on:** a Windows 11 PC with an AMD Ryzen 7 9800X3D CPU (8 cores) and 32 GB of RAM, running Python 3.14.
  Every code cell was run there, with local folders standing in for Google Drive. The PC's NVIDIA RTX 5090 GPU is
  not used by this tutorial. The project's models were trained on that GPU.
""")

md("## Approximate execution times")
md("""
Measured on the test PC above. Colab CPU runtimes are typically 2-3x slower, and Google Drive reads and writes are
slower than a local disk, so the Colab column is an estimate.

| Step | Test PC | Colab (estimate) |
|---|---|---|
| Installation check | < 1 s | < 1 s (all preinstalled) |
| Download Galaxy10 DECaLS (2.55 GiB) and copy it to Drive (first session only) | __T_DOWNLOAD__ | 2-6 min, depends on the connection |
| Copy the dataset file from Drive (later sessions, instead of the previous row) | < 10 s | 1-2 min |
| Checksum + load into memory | __T_VERIFY__ | 1-2 min |
| Cleaning + split (hashing 17,736 images, neighbor search) | __T_SPLIT__ | < 30 s |
| Write the PNG directories + zip them to Drive (first session only) | __T_EXPORT__ | 5-10 min |
| Unzip from Drive (later sessions, instead of the previous row) | __T_UNZIP__ | 1-3 min |
| Everything else (figures, synthetic-image download, checks, extension demo) | ~10 s | < 1 min |
| **Whole tutorial, first session** | **__T_TOTAL__** | **~10-20 min** |
| **Whole tutorial, later sessions** (dataset and zip already on Drive) | **__T_TOTAL2__** | **~5-10 min** |
""")

md("## Expected results")
md("""
This tutorial's results are counts and checks, not accuracies (those come in Tutorials 4-6). If everything is correct:

- **17,736** images load, as an array of shape `(17736, 256, 256, 3)` and type `uint8`, with **10** classes.
- **121** images are removed as pixel-identical duplicates (60 galaxies stored 2-3 times with conflicting labels), and
  **69** validation/testing images are removed as overlapping sky cutouts, leaving **17,546** images.
- The split is **12,330 training / 2,607 validation / 2,609 testing** (70/15/15, stratified by class). Its SHA-1
  fingerprints are identical to the paper's split (asserted in the notebook).
- The PNG directories reload **bit-for-bit identical** to the arrays (asserted), about **2.4 GiB** in total.
- A galaxy re-downloaded from the Legacy Surveys (DR8) matches its Galaxy10 image with pixel correlation **~0.99**.
- The real vs. synthetic grid shows rows of the paper's Fig. 2 (the seeded draw reproduces its real images; the synthetic
  sample's checksum is verified).
""")

# ================================================================================ drive
md("# Mount Google Drive")
md("""
A Colab machine is wiped when the session ends, so anything worth keeping is stored on your **Google Drive**: the
2.55 GiB dataset file and the zipped directories. Mounting makes your Drive appear as a folder of the Colab machine:

```python
from google.colab import drive
drive.mount("/content/drive")   # asks you to authorize access the first time
```

Your Drive is then the folder `/content/drive/MyDrive/`. This tutorial keeps its files in
`/content/drive/MyDrive/galaxy10_tutorial/` and works on a copy on the Colab machine's own disk (`/content/`), which is
much faster to read than Drive.

When the notebook runs **outside Colab** (for example on your own computer), the next cell skips mounting and uses two
local folders instead: `./galaxy10_tutorial_drive` in place of Drive and `./galaxy10_tutorial_work` in place of
`/content`. You can change them with the environment variables `GALAXY10_TUTORIAL_DRIVE` and `GALAXY10_TUTORIAL_WORK`.
""")
code("""
import os
import time
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
T0 = time.time()
TIMES = {}  # execution time of each step (seconds), reported at the end

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive/galaxy10_tutorial")  # persistent (Google Drive)
    WORK_ROOT = Path("/content")                                      # fast, wiped after the session
else:
    DRIVE_ROOT = Path(os.environ.get("GALAXY10_TUTORIAL_DRIVE", "galaxy10_tutorial_drive")).resolve()
    WORK_ROOT = Path(os.environ.get("GALAXY10_TUTORIAL_WORK", "galaxy10_tutorial_work")).resolve()

RAW_DIR = DRIVE_ROOT / "raw"                     # Galaxy10_DECals.h5 is kept here
ZIP_PATH = DRIVE_ROOT / "galaxy10_dataset.zip"   # the training/validation/testing directories, zipped
H5_PATH = WORK_ROOT / "Galaxy10_DECals.h5"       # working copy of the dataset file
DATASET_DIR = WORK_ROOT / "galaxy10"             # working copy of the directories (unzipped)
for d in (RAW_DIR, WORK_ROOT):
    d.mkdir(parents=True, exist_ok=True)
print("Running in Colab:", IN_COLAB)
print("Persistent storage (Drive):", DRIVE_ROOT)
print("Working copy:              ", WORK_ROOT)
""")

# ================================================================================ directory structure
md("# Dataset requirements and directory structure")
md("## Storage requirements")
md("""
| What | Where | Size | Kept after the session? |
|---|---|---|---|
| `Galaxy10_DECals.h5` (the original dataset, HDF5 format) | `MyDrive/galaxy10_tutorial/raw/` | 2.55 GiB | yes (Drive) |
| `galaxy10_dataset.zip` (the three split directories) | `MyDrive/galaxy10_tutorial/` | ~2.4 GiB | yes (Drive) |
| working copy of `Galaxy10_DECals.h5` | `/content/` | 2.55 GiB | no (copied from Drive) |
| unzipped `galaxy10/` directories | `/content/galaxy10/` | ~2.4 GiB | no (unzipped from Drive) |

**Total on Google Drive: about 5 GB**; the free tier has 15 GB. Training reads 12,330 small files, which is much
faster from the Colab machine's own disk than from Drive, hence the working copies.
""")
md("## Directory structure")
md("""
```
MyDrive/galaxy10_tutorial/                  <- Google Drive (persistent)
    raw/Galaxy10_DECals.h5                  <- original dataset, downloaded once
    galaxy10_dataset.zip                    <- the galaxy10/ folder below, zipped (created once)

/content/                                   <- the Colab machine's disk (wiped after the session)
    Galaxy10_DECals.h5                      <- working copy of the dataset file
    galaxy10/
        training/                           <- 12,330 images (the paper's "working pool")
            0_Disturbed/00517.png           <- one 256 x 256 RGB PNG per galaxy; the file name is the
            1_Merging/...                      galaxy's index in Galaxy10_DECals.h5
            ...
            9_Edge-on_with_Bulge/...
        validation/                         <- 2,607 images, same 10 class folders
        testing/                            <- 2,609 images, same 10 class folders
        metadata/
            training.csv, validation.csv, testing.csv  <- file, dataset index, label, class name, RA, Dec, pixel scale
            class_names.json                <- label number -> class name
            split_indices.npz               <- the split as index arrays (same format as the project repository)
```

The **folder name is the label**: every image of class 4 is in `4_Cigar-Shaped_Smooth/`. Most image-loading
libraries expect this one-folder-per-class layout, and it makes adding new images simple (see *Extending the dataset*).

**Zip and unzip commands.** The notebook does this in Python (so it also runs outside Colab). The equivalent shell
commands in a Colab cell are:

```bash
# first session: zip the directories (from /content) and copy the zip to Drive
!cd /content && zip -r -0 -q galaxy10_dataset.zip galaxy10
!cp /content/galaxy10_dataset.zip /content/drive/MyDrive/galaxy10_tutorial/
# later sessions: unzip from Drive to the Colab machine's disk
!unzip -q /content/drive/MyDrive/galaxy10_tutorial/galaxy10_dataset.zip -d /content
```

`-0` stores the files without compressing them again, because PNG files are already compressed.
""")

# ================================================================================ download
md("# Download and load the dataset")
md("""
**How we figured this out.** The Galaxy10 documentation (reference 1) points to the `astroNN` loader. Its source
(reference 2) shows the file's address, `https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5`,
and its SHA-256 checksum. The next cell does the same: in the first session it downloads the file and checks the
checksum, then keeps a copy on Drive. Later sessions copy the file from Drive instead of downloading it again.
""")
code("""
import hashlib
import shutil
import urllib.request

import numpy as np

URL = "https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5"
SHA256 = "19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571"  # from astroNN's loader
DRIVE_H5 = RAW_DIR / "Galaxy10_DECals.h5"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(2**24):
            h.update(block)
    return h.hexdigest()


def download(url, path):
    \"\"\"Download url to path in 4 MiB chunks, reporting progress every 10%.\"\"\"
    partial = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        total, done, shown = int(response.headers["Content-Length"]), 0, -1
        while chunk := response.read(2**22):
            out.write(chunk)
            done += len(chunk)
            if 10 * done // total != shown:
                shown = 10 * done // total
                print(f"  {10 * shown:3d}% of {total / 2**30:.2f} GiB", flush=True)
    partial.replace(path)


t = time.time()
if not H5_PATH.exists():
    if DRIVE_H5.exists():
        print("Copying the dataset from Drive:", DRIVE_H5)
        shutil.copy(DRIVE_H5, H5_PATH)
    else:
        print("Downloading", URL)
        download(URL, H5_PATH)
        TIMES["download"] = time.time() - t
        print(f"  done in {TIMES['download']:.0f} s")
t = time.time()
if sha256_of(H5_PATH) != SHA256:
    H5_PATH.unlink()
    raise RuntimeError("Checksum mismatch: the file is corrupted and was deleted. Run this cell again.")
if not DRIVE_H5.exists():
    shutil.copy(H5_PATH, DRIVE_H5)  # keep a copy on Drive for later sessions
print(f"{H5_PATH.name}: {H5_PATH.stat().st_size / 2**30:.2f} GiB, checksum OK, copy on Drive: {DRIVE_H5.exists()}")
""")
md("""
The file is in **HDF5** format: a container of named arrays ("datasets"). Listing them with `h5py` shows more than the
images (`images`) and labels (`ans`, short for answer) that the loader returns. Each galaxy's sky position (`ra`, `dec`,
in degrees), the `pxscale` of its image (arcseconds per pixel) and a `redshift` estimate are there too. We use the sky
positions to find overlapping cutouts and to download new images.
""")
code("""
import h5py

with h5py.File(H5_PATH, "r") as f:
    for key in f.keys():
        print(f"{key:<9} shape {str(f[key].shape):<22} dtype {f[key].dtype}")
    images = f["images"][:]                       # reading the whole array at once is the fast way
    labels = f["ans"][:].astype(np.int64)
    ra, dec = f["ra"][:].astype(np.float64), f["dec"][:].astype(np.float64)
    pxscale = f["pxscale"][:].astype(np.float64)
TIMES["verify + load"] = time.time() - t
print(f"loaded {images.shape} {images.dtype} ({images.nbytes / 2**30:.2f} GiB in memory) "
      f"in {TIMES['verify + load']:.0f} s")
""")

# ================================================================================ inputs
md("# The inputs: what is an image?")
md("""
- The dataset is **one 4-D NumPy array** of shape `(17736, 256, 256, 3)` and type `uint8`: 17,736 images of
  256 x 256 pixels with 3 color channels.
- **Each input is therefore a 3-D NumPy array** of shape `(height, width, channels) = (256, 256, 3)` holding integers
  from 0 to 255. The three channels are an RGB color rendering of the survey's g, r and z filters (green, red and
  near-infrared light).
- Each image is a **sky cutout centered on one galaxy**. 16,916 images use 0.262 arcsec per pixel (a 67 x 67 arcsec
  patch of sky) and 820 use 0.524 arcsec per pixel (134 x 134 arcsec), so the same pixel size can mean two different
  sky areas.
- The project's classifier reads the images at this native 256 x 256 size; the diffusion model works at 512 x 512
  (Tutorial 2).
""")
code("""
x = images[0]
print(f"one input: {type(x).__name__}, shape {x.shape}, dtype {x.dtype}, values {x.min()}..{x.max()}")
print("pixel scales (arcsec/pixel) and how many images use each:",
      {float(v): int((pxscale == v).sum()) for v in np.unique(pxscale)})
print("mean pixel value per channel (R, G, B):", images[::50].reshape(-1, 3).mean(0).round(1))
""")

# ================================================================================ labels
md("# The output labels: 10 morphology classes")
md("""
Each image has **one integer label from 0 to 9**: a single-label, 10-class classification problem. The labels come
from Galaxy Zoo volunteers' votes (reference 3). The classes are very unequal in size, and the paper uses this
imbalance as part of the experiment (the rare *Cigar-Shaped Smooth* class vs. the common *Round Smooth* class):

| Label | Class | Images | Label | Class | Images |
|---|---|---|---|---|---|
| 0 | Disturbed | 1,081 | 5 | Barred Spiral | 2,043 |
| 1 | Merging | 1,853 | 6 | Unbarred Tight Spiral | 1,829 |
| 2 | Round Smooth | 2,645 | 7 | Unbarred Loose Spiral | 2,628 |
| 3 | In-between Round Smooth | 2,027 | 8 | Edge-on without Bulge | 1,423 |
| 4 | Cigar-Shaped Smooth | 334 | 9 | Edge-on with Bulge | 1,873 |

**How we figured this out.** astroNN's class-name dictionary (`astroNN.datasets.galaxy10.Galaxy10Class`) gives class 3
a stale name, "Smooth, Cigar shaped", left over from the older Galaxy10 SDSS dataset. The names above come from the
Galaxy10 DECaLS documentation (reference 1), so we define them ourselves.
""")
code("""
CLASS_NAMES = ["Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar-Shaped Smooth",
               "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge",
               "Edge-on with Bulge"]
counts = np.bincount(labels, minlength=10)
for c, (name, n) in enumerate(zip(CLASS_NAMES, counts)):
    print(f"{c}  {name:<24} {n:>5}  ({100 * n / len(labels):4.1f}%)")
assert counts.tolist() == [1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873]
""")

# ================================================================================ display
md("# Displaying images")
md("""
The images are ordinary RGB arrays, so `matplotlib.pyplot.imshow` displays them directly. The helper below shows a few
examples of every class; the cell draws 3 at random (run it again for a different draw).
""")
code("""
import textwrap

import matplotlib.pyplot as plt


def show_examples(indices_per_class, title):
    n = max(len(v) for v in indices_per_class.values())
    fig, axes = plt.subplots(n, 10, figsize=(15, 1.6 * n + 1.2))
    axes = np.atleast_2d(axes)
    for c in range(10):
        for k in range(n):
            axes[k, c].axis("off")
            if k < len(indices_per_class[c]):
                axes[k, c].imshow(images[indices_per_class[c][k]])
        axes[0, c].set_title(f"{c}\\n" + textwrap.fill(CLASS_NAMES[c], 14), fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


rng = np.random.default_rng()
show_examples({c: rng.choice(np.flatnonzero(labels == c), 3, replace=False) for c in range(10)},
              "Random Galaxy10 DECaLS examples (3 per class)")
""")

# ================================================================================ split
md("# Setting up the training, validation and testing sets")
md("""
The paper uses a **stratified 70/15/15 split**, so each class keeps its share in every set. Two cleaning steps, which
we found while checking the data, come around it. The code below is the project's `data/prepare_splits.py`,
reproduced cell by cell. At the end we assert that the result is **identical** to the paper's split.
""")
md("## Removing pixel-identical duplicates")
md("""
**How we figured this out.** After a first split, we hashed every image's pixels to check that no image appeared in two
sets. 60 galaxies turned out to be stored 2-3 times (121 images), always with **conflicting labels**; most often
Disturbed vs. Unbarred Loose Spiral (19 of the 60). Identical sky coordinates confirmed that each group is one galaxy.
The true label is ambiguous, and the copies could land in different sets, so all copies are removed **before** the
split.
""")
code("""
from collections import defaultdict

t = time.time()
by_hash = defaultdict(list)
for i in range(len(images)):
    by_hash[hashlib.sha1(images[i].tobytes()).hexdigest()].append(i)
dup_groups = [g for g in by_hash.values() if len(g) > 1]
excluded = np.sort(np.concatenate(dup_groups))
n_conflict = sum(len(set(labels[g])) > 1 for g in dup_groups)
kept = np.setdiff1d(np.arange(len(labels)), excluded)
print(f"{len(dup_groups)} duplicate groups ({len(excluded)} images), {n_conflict} of them with conflicting labels")
print(f"{len(kept)} images kept")
""")
md("## Stratified 70/15/15 split")
md("""
`sklearn.model_selection.train_test_split` splits in two, so we split twice and stratify by class both times: 70% vs.
30%, then the 30% in half. `random_state=42` makes the split reproducible.
""")
code("""
from sklearn.model_selection import train_test_split

SEED = 42
train_idx, holdout_idx = train_test_split(kept, test_size=0.30, stratify=labels[kept], random_state=SEED)
val_idx, test_idx = train_test_split(holdout_idx, test_size=0.50, stratify=labels[holdout_idx], random_state=SEED)
splits = {"training": np.sort(train_idx), "validation": np.sort(val_idx), "testing": np.sort(test_idx)}
print({k: len(v) for k, v in splits.items()})
""")
md("## Removing overlapping sky cutouts")
md("""
**How we figured this out.** Pixel hashing only finds *identical* copies. We compared every pair of galaxies' sky
coordinates and found pairs whose cutouts cover largely the **same patch of sky**. Typically these are the two
galaxies of a merging pair, each with its own entry, and some pairs sat in different sets. Shifting one image by the
offset between the coordinates lined the pairs up (median pixel correlation 0.96-0.99), which confirmed it.

The fix removes every validation or testing image whose cutout overlaps an image in **another** set by at least 25% of
its area. For a validation-testing pair it removes the validation image, to keep the test set as large as possible.
It also removes overlapping pairs with conflicting labels inside validation or testing. The **training set is never
changed**. The 25% threshold catches every case where one image's central galaxy lies inside the other cutout.
""")
code("""
from scipy.spatial import cKDTree

CUTOUT_PX, OVERLAP_THRESHOLD = 256, 0.25


def cutout_overlap(ra1, dec1, px1, ra2, dec2, px2):
    \"\"\"Fraction of cutout 1's sky area covered by cutout 2 (square cutouts, flat-sky approximation).\"\"\"
    dx = (ra2 - ra1) * np.cos(np.radians(dec1)) * 3600  # offset in arcsec
    dy = (dec2 - dec1) * 3600
    w1, w2 = CUTOUT_PX * px1, CUTOUT_PX * px2            # cutout widths in arcsec
    ox = max(0.0, min(w1 / 2, dx + w2 / 2) - max(-w1 / 2, dx - w2 / 2))
    oy = max(0.0, min(w1 / 2, dy + w2 / 2) - max(-w1 / 2, dy - w2 / 2))
    return ox * oy / (w1 * w1)


def unit_vectors(ra_deg, dec_deg):
    r, d = np.radians(ra_deg), np.radians(dec_deg)
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


def overlapping_pairs(indices):
    \"\"\"(i, j) pairs among `indices` whose cutouts overlap by >= OVERLAP_THRESHOLD.\"\"\"
    tree = cKDTree(unit_vectors(ra[indices], dec[indices]))
    radius = np.radians(CUTOUT_PX * pxscale.max() / 3600)  # cutouts farther apart than this cannot overlap
    pairs = []
    for a, b in tree.query_pairs(radius, output_type="ndarray"):
        i, j = int(indices[a]), int(indices[b])
        overlap = max(cutout_overlap(ra[i], dec[i], pxscale[i], ra[j], dec[j], pxscale[j]),
                      cutout_overlap(ra[j], dec[j], pxscale[j], ra[i], dec[i], pxscale[i]))
        if overlap >= OVERLAP_THRESHOLD:
            pairs.append((i, j))
    return pairs


split_of = {int(i): s for s, idx in splits.items() for i in idx}
remove = set()
for i, j in overlapping_pairs(np.sort(np.concatenate(list(splits.values())))):
    si, sj = split_of[i], split_of[j]
    if si != sj:
        if "training" in (si, sj):
            remove.add(j if si == "training" else i)    # remove the held-out member
        else:
            remove.add(i if si == "validation" else j)  # validation-testing pair: keep the testing image
    elif si != "training" and labels[i] != labels[j]:
        remove.update((i, j))                            # conflicting labels inside validation/testing
overlap_removed = np.array(sorted(remove), dtype=np.int64)
splits = {s: np.setdiff1d(idx, overlap_removed) for s, idx in splits.items()}
TIMES["cleaning + split"] = time.time() - t
print(f"removed {len(overlap_removed)} overlapping validation/testing images")
print({k: len(v) for k, v in splits.items()}, f"total {sum(len(v) for v in splits.values())}")
""")
md("## Verifying the split")
md("""
Three checks: the sizes match the paper, no image is in two sets, and the index arrays have exactly the SHA-1
fingerprints of the split used in the paper, so this notebook reproduces the paper's split bit for bit. When the
notebook runs inside the project repository, it also compares against the saved split file directly.
""")
code("""
EXPECTED_SHA1 = {  # fingerprints of the paper's split (data/splits/galaxy10_splits.npz in the project repository)
    "training": "a8ff1141046adb4a727eadbd843163762acf5237",
    "validation": "1509d26a5e3a028789256e53144324c89eb962b7",
    "testing": "51b6e17c91501b82f5de5b19d57be4b00e4f9378",
    "duplicates removed": "2490960373bab8c5697593edb22b990f1ffd9223",
    "overlaps removed": "532bcafcc7f2e1a3e5868efa3432e79e15d3aafa",
}


def sha1_of(indices):
    return hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


assert {k: len(v) for k, v in splits.items()} == {"training": 12330, "validation": 2607, "testing": 2609}
all_idx = np.concatenate(list(splits.values()))
assert len(np.unique(all_idx)) == len(all_idx), "an image is in more than one set"
got = {**{k: sha1_of(v) for k, v in splits.items()},
       "duplicates removed": sha1_of(excluded), "overlaps removed": sha1_of(overlap_removed)}
for k, h in EXPECTED_SHA1.items():
    assert got[k] == h, f"{k}: fingerprint {got[k]} differs from the paper's {h}"
print("The split is identical to the paper's (all 5 fingerprints match).")

repo_split = Path("data/splits/galaxy10_splits.npz")  # exists only when run inside the project repository
if repo_split.exists():
    saved = np.load(repo_split)
    for ours, theirs in (("training", "train"), ("validation", "val"), ("testing", "test")):
        assert np.array_equal(splits[ours], saved[theirs])
    print("Also identical to", repo_split)

per_class = {s: np.bincount(labels[idx], minlength=10) for s, idx in splits.items()}
print(f"\\n{'class':<26}{'training':>9}{'validation':>11}{'testing':>9}")
for c in range(10):
    print(f"{c} {CLASS_NAMES[c]:<24}{per_class['training'][c]:>9}{per_class['validation'][c]:>11}"
          f"{per_class['testing'][c]:>9}")
""")
md("""
**Reproducing the paper's example figure (Fig. 16).** The paper's class-example figure shows 3 training images per
class, drawn at random with seed 0 (`analysis/make_class_examples.py`). The same draw reproduces it. The figure's
image indices are listed below as well, so the same galaxies are shown even if a future NumPy version draws
differently.
""")
code("""
PAPER_FIG16 = {0: [517, 649, 907], 1: [1116, 1160, 1219], 2: [4228, 4623, 5344], 3: [6675, 6865, 7066],
               4: [7694, 7835, 7878], 5: [8016, 9074, 9687], 6: [10149, 10303, 11545], 7: [12015, 12603, 13219],
               8: [14448, 14492, 15005], 9: [16849, 17073, 17115]}  # dataset indices shown in the paper's Fig. 16
fig16_rng = np.random.default_rng(0)
draw = {c: sorted(fig16_rng.choice(splits["training"][labels[splits["training"]] == c], 3, replace=False).tolist())
        for c in range(10)}
print("seeded draw reproduces the paper's figure:", draw == PAPER_FIG16)
show_examples(PAPER_FIG16, "Paper Fig. 16: real examples of each class (training set)")
""")

# ================================================================================ real vs synthetic
md("# Real vs. synthetic galaxies")
md("""
The project's generator (Stable Diffusion 1.5 fine-tuned with LoRA; Tutorials 2 and 4) made a **synthetic copy of the
training set**: 12,330 images with the same number per class, generated with guidance 3 and 30 sampling steps and
saved at the same 256 x 256 size. The project's GitHub Release holds a small sample of it: the first 4 synthetic images
of every class (sample indices 0-3). They are not hand-picked, and they are the synthetic images of the paper's Fig. 2.

The next cell downloads that sample with `wget` (5.9 MiB) and checks its SHA-256 checksum, the same way Tutorials 2
and 4 download the model weights.
""")
code(f"""
import io
import zipfile

from PIL import Image

RELEASE_URL = "{RELEASE_URL}"
SYNTHETIC_ZIP = WORK_ROOT / "synthetic_examples.zip"
SYNTHETIC_SHA256 = "{SHA256['synthetic_examples.zip']}"

if not (SYNTHETIC_ZIP.exists() and sha256_of(SYNTHETIC_ZIP) == SYNTHETIC_SHA256):
    !wget -q -O "{{SYNTHETIC_ZIP}}" "{{RELEASE_URL}}/synthetic_examples.zip"
assert SYNTHETIC_ZIP.exists() and sha256_of(SYNTHETIC_ZIP) == SYNTHETIC_SHA256, \\
    "download failed or checksum mismatch: delete the file and run this cell again"

synthetic = {{}}  # class -> [uint8 arrays], sample indices 0-3
with zipfile.ZipFile(SYNTHETIC_ZIP) as z:
    for name in sorted(z.namelist()):  # synthetic_examples/<label>_<class>/sample<k>_seed<seed>.png
        c = int(name.split("/")[1].split("_")[0])
        synthetic.setdefault(c, []).append(np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGB")))
print({{CLASS_NAMES[c]: len(v) for c, v in sorted(synthetic.items())}})
print("one synthetic image:", synthetic[0][0].shape, synthetic[0][0].dtype, "(the same format as the real inputs)")
""")
md("""
The grid below puts, for 5 classes, 4 real training images next to 4 synthetic images of the same class. It reuses
the image-arranging function `make_grid` from the project's `diffusion/sample.py` (one row per class with a class
label, the real images, a gap, then the generated images), adapted to show the grid in the notebook instead of saving
it to a file. The real images are the paper's Fig. 2 draw (4 random training images per class, seed 0; indices listed
in the cell), so each row is a row of the paper's Fig. 2. Change `GRID_CLASSES` to see other classes.

Compare the rare *Cigar-Shaped Smooth* row with the others: in the paper, a classifier trained on real images
recognizes only 28% of the synthetic Cigar-Shaped Smooth images as that class, the lowest of all classes.
""")
code("""
from IPython.display import display
from PIL import ImageDraw, ImageFont


def grid_font(size):
    try:
        return ImageFont.load_default(size=size)  # scalable font (Pillow >= 10.1)
    except TypeError:
        return ImageFont.load_default()           # older Pillow: small bitmap font


def make_grid(rows, real, classes, tile, title):
    \"\"\"Adapted from diffusion/sample.py: one row per class - real images, a gap, then generated images.
    rows: {class: [PIL images]} (generated); real: {class: [PIL images]}. Returns one PIL image.\"\"\"
    n_real = len(real.get(classes[0], []))
    n_gen = len(rows[classes[0]])
    label_w, gap, header_h = 250, 12 if n_real else 0, 60
    width = label_w + (n_real + n_gen) * tile + gap
    height = header_h + len(classes) * tile
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    font = grid_font(18)
    draw.text((6, 6), title, fill="black", font=font)  # own line, above the column labels
    if n_real:
        draw.text((label_w + 4, 34), "real (training set)", fill="black", font=font)
    draw.text((label_w + n_real * tile + gap + 4, 34), "synthetic (final generator)", fill="black", font=font)
    for r, c in enumerate(classes):
        y = header_h + r * tile
        draw.text((6, y + tile // 2 - 12), f"{c}: {CLASS_NAMES[c]}", fill="black", font=font)
        x = label_w
        for im in real.get(c, []):
            grid.paste(im.resize((tile, tile), Image.LANCZOS), (x, y))
            x += tile
        x += gap
        for im in rows[c]:
            grid.paste(im.resize((tile, tile), Image.LANCZOS), (x, y))
            x += tile
    return grid


PAPER_FIG2_REAL = {0: [268, 516, 644, 906], 1: [1116, 1397, 2273, 2576], 2: [4335, 4575, 4852, 5497],
                   3: [5583, 6386, 6946, 7235], 4: [7660, 7851, 7860, 7889], 5: [8107, 8540, 8923, 9045],
                   6: [9991, 10000, 10210, 11229], 7: [12816, 13010, 13431, 13816], 8: [14968, 15361, 15411, 15784],
                   9: [16125, 16585, 16949, 17492]}  # dataset indices of the real images in the paper's Fig. 2
fig2_rng = np.random.default_rng(0)
train_labels = labels[splits["training"]]
fig2_draw = {c: splits["training"][np.sort(fig2_rng.choice(np.flatnonzero(train_labels == c), 4, replace=False))]
             .tolist() for c in range(10)}
print("seeded draw reproduces the paper's Fig. 2 real images:", fig2_draw == PAPER_FIG2_REAL)

GRID_CLASSES = [1, 2, 4, 5, 9]  # Merging, Round Smooth, Cigar-Shaped Smooth (rare), Barred Spiral, Edge-on with Bulge
grid = make_grid(rows={c: [Image.fromarray(x) for x in synthetic[c]] for c in GRID_CLASSES},
                 real={c: [Image.fromarray(images[i]) for i in PAPER_FIG2_REAL[c]] for c in GRID_CLASSES},
                 classes=GRID_CLASSES, tile=160,
                 title="Real vs. synthetic galaxies (synthetic: guidance 3, 30 steps; not hand-picked)")
display(grid)
""")

# ================================================================================ directories
md("# Writing the training, validation and testing directories")
md("""
Each image is saved as a PNG file (lossless) at `<split>/<label>_<class name>/<dataset index>.png`, and each split gets
a metadata table. In the first session the next cell writes everything to the Colab machine's disk, zips it and copies
the zip to Drive. In **later sessions it just unzips the zip from Drive**, and if the directories are already in place
it does nothing. Set `REBUILD = True` to rebuild them from the arrays.
""")
code("""
import csv
import json
import zipfile

from PIL import Image

REBUILD = False
FOLDERS = [f"{c}_{name.replace(' ', '_')}" for c, name in enumerate(CLASS_NAMES)]  # e.g. 4_Cigar-Shaped_Smooth
META_FIELDS = ["file", "dataset_index", "label", "class_name", "ra", "dec", "pxscale", "source"]


def write_directories():
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    (DATASET_DIR / "metadata").mkdir(parents=True)
    for split, idx in splits.items():
        for folder in FOLDERS:
            (DATASET_DIR / split / folder).mkdir(parents=True)
        with open(DATASET_DIR / "metadata" / f"{split}.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(META_FIELDS)
            for i in idx:
                rel = f"{split}/{FOLDERS[labels[i]]}/{i:05d}.png"
                Image.fromarray(images[i]).save(DATASET_DIR / rel)
                writer.writerow([rel, i, labels[i], CLASS_NAMES[labels[i]], f"{ra[i]:.6f}", f"{dec[i]:.6f}",
                                 pxscale[i], "Galaxy10_DECals.h5"])
    (DATASET_DIR / "metadata" / "class_names.json").write_text(json.dumps(dict(enumerate(CLASS_NAMES)), indent=2))
    np.savez(DATASET_DIR / "metadata" / "split_indices.npz", train=splits["training"], val=splits["validation"],
             test=splits["testing"], excluded_indices=excluded, overlap_removed_indices=overlap_removed)


def save_zip_to_drive():
    \"\"\"Zip DATASET_DIR on the local disk, then copy the zip to Drive (run again after adding images).\"\"\"
    local_zip = WORK_ROOT / ZIP_PATH.name
    with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_STORED) as z:  # PNGs are already compressed
        for p in sorted(DATASET_DIR.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(WORK_ROOT))
    shutil.copy(local_zip, ZIP_PATH)
    local_zip.unlink()


def directories_complete():
    return DATASET_DIR.exists() and all(len(list((DATASET_DIR / s).glob("*/*.png"))) == len(idx)
                                        for s, idx in splits.items())


t = time.time()
if ZIP_PATH.exists() and not REBUILD:
    if directories_complete():
        print("The directories are already in place; nothing to do.")
    else:
        print("Unzipping", ZIP_PATH)
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(WORK_ROOT)
        TIMES["unzip from Drive"] = time.time() - t
else:
    print("Writing the PNG directories to", DATASET_DIR)
    write_directories()
    TIMES["write PNG directories"] = time.time() - t
    t = time.time()
    print("Zipping them to", ZIP_PATH)
    save_zip_to_drive()
    TIMES["zip + copy to Drive"] = time.time() - t
print(f"zip on Drive: {ZIP_PATH.stat().st_size / 2**30:.2f} GiB; " + ", ".join(f"{k}: {v:.0f} s" for k, v in TIMES.items()))
""")
md("## Checking the directories")
md("""
Count the files in every class folder, then reload a sample of the PNG files and confirm that they are identical to
the original arrays.
""")
code("""
total_bytes = 0
for split, idx in splits.items():
    found = [len(list((DATASET_DIR / split / folder).glob("*.png"))) for folder in FOLDERS]
    assert found == per_class[split].tolist(), f"{split}: file counts differ from the split"
    total_bytes += sum(p.stat().st_size for p in (DATASET_DIR / split).rglob("*.png"))
    print(f"{split:<11} {sum(found):>6} PNG files in {len(FOLDERS)} class folders")
print(f"total {total_bytes / 2**30:.2f} GiB")

check_rng = np.random.default_rng(1)
for i in check_rng.choice(splits["training"], 200, replace=False):
    png = np.asarray(Image.open(DATASET_DIR / "training" / FOLDERS[labels[i]] / f"{i:05d}.png"))
    assert np.array_equal(png, images[i])
print("200 reloaded PNG files are bit-for-bit identical to the arrays (lossless).")
""")
md("## Loading a split from its directory")
md("""
The later tutorials read the data like this. The label is the number at the start of the folder name.
""")
code("""
def load_split(split):
    \"\"\"Load one split directory: images uint8 [N, 256, 256, 3], labels int64 [N], file paths.\"\"\"
    files = sorted((DATASET_DIR / split).glob("*/*.png"))
    x = np.stack([np.asarray(Image.open(p)) for p in files])
    y = np.array([int(p.parent.name.split("_")[0]) for p in files], dtype=np.int64)
    return x, y, files


t = time.time()
x_val, y_val, val_files = load_split("validation")
print(f"validation: images {x_val.shape} {x_val.dtype}, labels {y_val.shape}, loaded in {time.time() - t:.1f} s")
print("images per class:", np.bincount(y_val, minlength=10).tolist())
assert np.array_equal(np.bincount(y_val, minlength=10), per_class["validation"])
del x_val  # free the memory
""")

# ================================================================================ extension
md("# Extending the dataset")
md("""
New galaxies can be added by downloading their images from the same survey. The **DESI Legacy Surveys cutout
service** (reference 4) returns a JPEG image for any position on the sky:

`https://www.legacysurvey.org/viewer/cutout.jpg?ra=<RA>&dec=<Dec>&layer=ls-dr8&pixscale=0.262&size=256`

**How we figured this out.** We downloaded a galaxy that is already in Galaxy10 from three data releases of the survey
and compared each download with the stored image. DR8 (`layer=ls-dr8`) matched with pixel correlation 0.987, DR9 with
0.963 and DR10 with 0.887. Galaxy10 DECaLS was evidently made from DR8, so we use `ls-dr8` to keep new images
consistent with the dataset.
""")
md("## Downloading a galaxy image")
code("""
import io


def download_cutout(ra_deg, dec_deg, pxscale_arcsec=0.262, size=256, layer="ls-dr8"):
    \"\"\"Download a size x size RGB cutout centered on (RA, Dec), in degrees, as a uint8 array.\"\"\"
    url = (f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra_deg:.6f}&dec={dec_deg:.6f}"
           f"&layer={layer}&pixscale={pxscale_arcsec}&size={size}")
    request = urllib.request.Request(url, headers={"User-Agent": "galaxy10-dataset-tutorial"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return np.asarray(Image.open(io.BytesIO(response.read())).convert("RGB")), url


# Example: download a galaxy that is already in the training set and compare it with the stored image.
example = 2942  # a Round Smooth galaxy in the training set
new_img, url = download_cutout(ra[example], dec[example], pxscale[example])
corr = np.corrcoef(new_img.astype(float).ravel(), images[example].astype(float).ravel())[0, 1]
print(url)
print(f"downloaded {new_img.shape} {new_img.dtype}; pixel correlation with the stored image: {corr:.3f}")
assert new_img.shape == (256, 256, 3) and corr > 0.95

fig, axes = plt.subplots(1, 2, figsize=(6, 3.3))
for ax, img, title in zip(axes, (images[example], new_img), ("Galaxy10 (stored)", "downloaded (ls-dr8)")):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
plt.show()
""")
md("## Adding an image to the training directory")
md("""
Adding an image means saving it in the right class folder and recording it in the metadata. One safeguard matters:
**a new training image must not show the same sky as a validation or testing image**, or held-out information leaks
into training (the problem the overlap cleaning removed). `add_to_training` therefore refuses such an image. It only
warns about an overlap with an existing training image, which is harmless but a near-duplicate.

To add a genuinely new galaxy you need its **RA, Dec and label**, for example from the Galaxy Zoo DECaLS catalog
(reference 3), where Galaxy10's labels come from:

```python
img, url = download_cutout(RA, DEC)
add_to_training(img, label=LABEL, ra_deg=RA, dec_deg=DEC, name="my_galaxy")
save_zip_to_drive()   # keep the change on Drive
```

Add new images to `training` only, never to `validation` or `testing`, which must stay the paper's held-out sets.
""")
code("""
all_kept = np.sort(np.concatenate(list(splits.values())))
kept_tree = cKDTree(unit_vectors(ra[all_kept], dec[all_kept]))


def overlaps_with_dataset(ra_deg, dec_deg, pxscale_arcsec=0.262):
    \"\"\"[(dataset index, split, overlap)] for dataset images that overlap a new cutout by >= 25%.\"\"\"
    radius = np.radians(CUTOUT_PX * max(pxscale.max(), pxscale_arcsec) / 3600)
    hits = []
    for a in kept_tree.query_ball_point(unit_vectors([ra_deg], [dec_deg])[0], radius):
        i = int(all_kept[a])
        overlap = max(cutout_overlap(ra_deg, dec_deg, pxscale_arcsec, ra[i], dec[i], pxscale[i]),
                      cutout_overlap(ra[i], dec[i], pxscale[i], ra_deg, dec_deg, pxscale_arcsec))
        if overlap >= OVERLAP_THRESHOLD:
            hits.append((i, split_of[i], round(overlap, 2)))
    return hits


def add_to_training(img, label, ra_deg, dec_deg, name, pxscale_arcsec=0.262, source="Legacy Surveys ls-dr8"):
    \"\"\"Save a new 256 x 256 RGB image in the training directory, refusing held-out overlaps.\"\"\"
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8, "images must be 256 x 256 RGB uint8 arrays"
    assert 0 <= label <= 9, "the label must be 0-9"
    hits = overlaps_with_dataset(ra_deg, dec_deg, pxscale_arcsec)
    held_out = [h for h in hits if h[1] != "training"]
    if held_out:
        raise ValueError(f"refused: the image overlaps validation/testing image(s) {held_out}; "
                         "adding it would leak held-out data into training")
    if hits:
        print(f"warning: overlaps existing training image(s) {hits} (a near-duplicate)")
    rel = f"training/{FOLDERS[label]}/added_{name}.png"
    Image.fromarray(img).save(DATASET_DIR / rel)
    with open(DATASET_DIR / "metadata" / "training.csv", "a", newline="") as fh:
        csv.writer(fh).writerow([rel, -1, label, CLASS_NAMES[label], f"{ra_deg:.6f}", f"{dec_deg:.6f}",
                                 pxscale_arcsec, source])
    print("added", rel)
    return DATASET_DIR / rel


# 1) An image at a validation galaxy's position is refused: it would leak validation data into training.
v = int(splits["validation"][0])
try:
    add_to_training(download_cutout(ra[v], dec[v])[0], int(labels[v]), ra[v], dec[v], name="leak_test")
except ValueError as e:
    print("refused, as expected:", e)

# 2) The training galaxy downloaded above is accepted (with a near-duplicate warning). It is then removed again so
#    that the directories stay identical to the paper's split; set KEEP_ADDED_IMAGES = True to keep your additions.
KEEP_ADDED_IMAGES = False
n_before = len(list((DATASET_DIR / "training").glob("*/*.png")))
added = add_to_training(new_img, int(labels[example]), ra[example], dec[example], name=f"redownload_{example}")
assert len(list((DATASET_DIR / "training").glob("*/*.png"))) == n_before + 1
if not KEEP_ADDED_IMAGES:
    added.unlink()
    meta = DATASET_DIR / "metadata" / "training.csv"
    with open(meta, newline="") as fh:
        rows = [row for row in csv.reader(fh) if "/added_" not in row[0]]
    with open(meta, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("demo image removed again (KEEP_ADDED_IMAGES = False)")
""")

# ================================================================================ summary
md("# Summary")
md("""
The dataset is ready for the other tutorials. The `training/`, `validation/` and `testing/` directories hold 256 x 256
RGB PNG files in 10 class folders. They are identical to the paper's split and saved on Google Drive as one zip file.
The last cell re-checks the expected results and reports the execution times.

Colab copies new files to Google Drive in the background. Before you close the session, wait a minute or run
`drive.flush_and_unmount()`, so that the dataset file and the zip are completely saved.
""")
code("""
TIMES["whole notebook"] = time.time() - T0
found = {s: len(list((DATASET_DIR / s).glob("*/*.png"))) for s in splits}
assert found == {"training": 12330, "validation": 2607, "testing": 2609}, found
assert ZIP_PATH.exists() and DRIVE_H5.exists()
print("Expected results reached: 12,330 / 2,607 / 2,609 images, identical to the paper's split, saved on Drive.")
for step, seconds in TIMES.items():
    print(f"  {step:<24} {seconds / 60:5.1f} min")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
text = nbf.writes(nb)
for key, value in FILL.items():
    text = text.replace(key, value)
out = Path("notebooks/01_Dataset_Tutorial.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells: {sum(c.cell_type == 'markdown' for c in cells)} text, "
      f"{sum(c.cell_type == 'code' for c in cells)} code)")

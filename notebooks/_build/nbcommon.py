"""Shared pieces of the tutorial build scripts: constants, the cell helper and the common setup cells.

The repository name, release tag and checksums live here only, so every tutorial downloads the same
files. Checksums come from weights/SHA256SUMS.txt (written by tools/make_release_assets.py).
"""

import re
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_REPO = "HunterSandoval12/faking-galaxies-with-diffusion"
REPO_URL = f"https://github.com/{GITHUB_REPO}"
RELEASE_TAG = "weights-v1"
RELEASE_URL = f"{REPO_URL}/releases/download/{RELEASE_TAG}"
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SPEC_URL = "https://github.com/pattichis/projects/blob/main/Colab-tutorial-list.md"
SHA256 = {name: digest for digest, name in
          (line.split() for line in (REPO_ROOT / "weights" / "SHA256SUMS.txt").read_text().splitlines())}
CLASS_NAMES = ["Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar-Shaped Smooth",
               "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge",
               "Edge-on with Bulge"]


class Notebook:
    def __init__(self):
        self.cells = []

    def md(self, text):
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(self, text):
        self.cells.append(nbf.v4.new_code_cell(text.strip("\n")))

    def write(self, filename, gpu=False, fill=None):
        nb = nbf.v4.new_notebook()
        nb["cells"] = self.cells
        nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                          "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
        if gpu:  # Colab opens the notebook on a GPU runtime
            nb["metadata"]["accelerator"] = "GPU"
            nb["metadata"]["colab"]["gpuType"] = "T4"
        text = nbf.writes(nb)
        for key, value in (fill or {}).items():
            text = text.replace(key, value)
        left = sorted(set(re.findall(r"__[A-Z][A-Z0-9_]*__", text)))
        assert not left, f"unfilled placeholders: {left}"
        out = REPO_ROOT / "notebooks" / filename
        out.write_text(text, encoding="utf-8", newline="\n")
        n_md = sum(c.cell_type == "markdown" for c in self.cells)
        print(f"wrote {out.relative_to(REPO_ROOT)} ({len(self.cells)} cells: {n_md} text, "
              f"{len(self.cells) - n_md} code)")


def install_cells(nb, requirements, table):
    """requirements: {pip name: minimum version}; table: markdown rows (library table)."""
    nb.md("## Library requirements and installation")
    nb.md(f"""
| Library | Minimum version | Tested version | Install command | Used for |
|---|---|---|---|---|
{table.strip()}

Google Colab preinstalls all of these, so the next cell normally installs nothing. It checks each library and runs
`pip` only for a missing or too-old one. On a **second run**, set `SKIP_INSTALL = True` to skip the check entirely.

**One conflict to remove.** Colab also preinstalls an old version of `torchao` (a quantization library this tutorial
does not use). Recent `peft` versions refuse to create LoRA adapters while a `torchao` older than 0.16 is installed
("ImportError: Found an incompatible version of torchao"), so the cell uninstalls such an old `torchao`
(`pip uninstall -y torchao`). This check runs even when `SKIP_INSTALL = True`.
""")
    reqs = ",\n                ".join(f'"{k}": "{v}"' for k, v in requirements.items())
    nb.code(f"""
SKIP_INSTALL = False  # set to True when re-running the notebook to skip the installation check

import importlib.metadata
import re
import subprocess
import sys

REQUIREMENTS = {{{reqs}}}  # package -> minimum version


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
    to_install = [f"{{pkg}}>={{minimum}}" for pkg, minimum in REQUIREMENTS.items()
                  if installed_version(pkg) is None or version_tuple(installed_version(pkg)) < version_tuple(minimum)]
    if to_install:
        print("Installing:", to_install)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
        print("Done. If a library was upgraded, restart the runtime (Runtime > Restart session) and run again.")
    else:
        print("All required libraries are already installed; nothing to install.")

# peft (LoRA) raises an ImportError while a torchao older than 0.16 is installed; this notebook does not use torchao.
TORCHAO_MIN = "0.16"
old_torchao = installed_version("torchao")
if old_torchao is not None and version_tuple(old_torchao) < version_tuple(TORCHAO_MIN):
    print(f"Removing torchao {{old_torchao}} (incompatible with peft; not used here)")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    import importlib
    importlib.invalidate_caches()  # let this running session see that torchao is gone
""")
    nb.code("""
# Verify the versions. The notebook stops here with a clear message if a requirement is not met.
assert sys.version_info >= (3, 10), f"Python >= 3.10 required, found {sys.version.split()[0]}"
torchao = installed_version("torchao")
assert torchao is None or version_tuple(torchao) >= version_tuple(TORCHAO_MIN), \\
    f"torchao {torchao} is installed and breaks peft: run !pip uninstall -y torchao"
for pkg, minimum in REQUIREMENTS.items():
    have = installed_version(pkg)
    assert have is not None, f"{pkg} is not installed (pip install '{pkg}>={minimum}')"
    assert version_tuple(have) >= version_tuple(minimum), f"{pkg} {have} is older than the required {minimum}"
    print(f"{pkg:<13} {have:<14} (required >= {minimum})")
print("All version requirements are satisfied.")
""")


def drive_cells(nb, purpose):
    nb.md("# Mount Google Drive")
    nb.md(f"""
{purpose} Tutorial 1 (Dataset Tutorial) saved the `training` / `validation` / `testing` directories on your Google
Drive as `MyDrive/galaxy10_tutorial/galaxy10_dataset.zip`. This tutorial reads single images straight out of that zip
file with Python's `zipfile` module, so nothing needs to be unzipped. Mounting makes your Drive a folder of the Colab
machine:

```python
from google.colab import drive
drive.mount("/content/drive")   # asks you to authorise access the first time
```

Outside Colab, the cell uses the same local folders as Tutorial 1 (`./galaxy10_tutorial_drive` and
`./galaxy10_tutorial_work`, or the environment variables `GALAXY10_TUTORIAL_DRIVE` / `GALAXY10_TUTORIAL_WORK`).
""")
    nb.code("""
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
ZIP_PATH = DRIVE_ROOT / "galaxy10_dataset.zip"  # written by Tutorial 1
WEIGHTS_DIR = WORK_ROOT / "weights"             # downloaded model files
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
print("Running in Colab:", IN_COLAB)
print("Dataset zip from Tutorial 1:", ZIP_PATH, "(found)" if ZIP_PATH.exists() else "(NOT FOUND)")
""")


def download_cells(nb, files, intro=""):
    """Markdown + code cell that downloads release files with wget and checks their SHA-256 checksums."""
    listing = "\n".join(f"| `{f}` | `{SHA256[f]}` |" for f in files)
    nb.md(f"""
{intro}
The files are attached to the project's GitHub Release `{RELEASE_TAG}`. A release file has a fixed, public address,
`{RELEASE_URL}/<file name>`, so plain `wget` downloads it; no GitHub account or login is needed. After each download
the cell checks the file's **SHA-256 checksum** against the value published in the repository's README and in the
release notes, so a corrupted or altered file stops the notebook:

| File | SHA-256 |
|---|---|
{listing}

**How we figured this out.** GitHub serves every release file at `.../releases/download/<tag>/<file name>` and
redirects the request to its file storage; `wget` follows the redirect. `hashlib.sha256` computes the checksum in
Python (on Linux, `sha256sum <file>` gives the same value).
""")
    entries = ",\n    ".join(f'"{f}": "{SHA256[f]}"' for f in files)
    nb.code(f"""
import hashlib

RELEASE_URL = "{RELEASE_URL}"
FILES = {{  # file name -> SHA-256 checksum
    {entries},
}}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(2**24):
            h.update(block)
    return h.hexdigest()


t = time.time()
for name, expected in FILES.items():
    path = WEIGHTS_DIR / name
    if not (path.exists() and sha256_of(path) == expected):  # skip files downloaded earlier
        url = f"{{RELEASE_URL}}/{{name}}"
        !wget -q -O "{{path}}" "{{url}}"
    assert path.exists() and sha256_of(path) == expected, \\
        f"{{name}}: download failed or checksum mismatch; delete {{path}} and run this cell again"
    print(f"{{name:<42}} {{path.stat().st_size / 2**20:6.2f}} MiB   SHA-256 OK")
TIMES["download weights"] = time.time() - t
""")

"""Applies SnapMesh's patches to a freshly cloned vendor/TripoSR.

Currently one patch: replace tsr/models/isosurface.py with a scikit-image
marching-cubes implementation, removing the `torchmcubes` compiled dependency
(the hardest part of TripoSR's install, especially on Windows).
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "patches" / "isosurface_skimage.py"
DST = ROOT / "vendor" / "TripoSR" / "tsr" / "models" / "isosurface.py"

if not DST.parent.exists():
    raise SystemExit(
        "vendor/TripoSR not found — clone it first:\n"
        "  git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR"
    )

shutil.copyfile(SRC, DST)
print(f"patched: {DST}")

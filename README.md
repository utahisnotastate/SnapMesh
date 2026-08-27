# SnapMesh 📸➜🧊

**Snap a picture, get a game-ready 3D mesh.**

SnapMesh is a free, local, open-source image-to-3D pipeline for game developers —
with first-class Roblox support. Feed it one concept image (AI-generated art works
great); it hands back a low-poly, vertex-colored, engine-ready mesh. No paid API,
no credits, no cloud: everything runs on your own machine.

```
[concept image]
    → background removal          rembg (U2-Net)
    → image-to-3D reconstruction  TripoSR (MIT, Stability AI / Tripo AI)
    → cleanup + quadric decimation trimesh + fast-simplification
    → exports
        model.glb    vertex-colored, for any engine or Blender
        model.obj    portable geometry
        model.luau   ⭐ Roblox EditableMesh data module
```

### The Roblox superpower: zero uploads

The `.luau` export + the included `roblox/SnapMeshForge.luau` module build your mesh
**in-game from pure code** using `AssetService:CreateEditableMesh()` — per-vertex
colors, smooth normals, everything. No Import 3D, no asset upload, no moderation
queue. Generate an image, run one command, `require()` the result.

## Quick start

```bash
# 1. install uv (https://docs.astral.sh/uv/) then:
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 2. get TripoSR (we patch out its compiled torchmcubes dependency — pure wheels only)
git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR
python patch_triposr.py

# 3. go
.venv/Scripts/python.exe snapmesh.py my_concept.png --tris 3000 --height 8
```

First run downloads the TripoSR weights (~1.6 GB) from Hugging Face automatically.

**Hardware:** any NVIDIA GPU with ≥6 GB VRAM reconstructs in seconds; CPU-only
works too (minutes per image). ~16 GB RAM recommended.

### Tips for good inputs
- One object, centered, on a plain background (SnapMesh removes it anyway, but
  clean inputs segment better). A ¾ view beats a straight-on front view.
- Concept art in a clean, saturated style reconstructs better than photos.
- The model sees one side and imagines the rest — symmetric objects come out best.

### Using the result in Roblox

```lua
local SnapMeshForge = require(ReplicatedStorage.SnapMeshForge)
local data = require(ReplicatedStorage.Meshes.Server_Rack) -- the .luau export
local part = SnapMeshForge.Build(data)
SnapMeshForge.PlaceOnGround(part, Vector3.new(0, 0, 0))
part.Parent = workspace
```

Clones of the built part share mesh content — scatter hundreds cheaply.

## Credits & license

SnapMesh is glue, standing on excellent open work:
[TripoSR](https://github.com/VAST-AI-Research/TripoSR) (MIT) by Tripo AI & Stability AI ·
[rembg](https://github.com/danielgatis/rembg) (MIT) ·
[trimesh](https://trimsh.org) (MIT) ·
[fast-simplification](https://github.com/pyvista/fast-simplification) (MIT) ·
[scikit-image](https://scikit-image.org) (BSD).

SnapMesh itself is MIT licensed. Meshes you generate belong to you (subject to the
TripoSR model card and the rights of your input images).

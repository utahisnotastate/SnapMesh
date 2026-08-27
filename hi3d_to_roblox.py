"""
hi3d_to_roblox.py — bring a high-poly, UV-textured mesh (Hi3D / Hunyuan3D / any
photogrammetry-grade OBJ+MTL+texture) down to a Roblox-ready asset WITHOUT
throwing away what makes it look good.

The two things that make a generated mesh look "cheap" in-engine are:
  1. decimating so hard the silhouette collapses, and
  2. baking appearance into vertex colors instead of keeping the UV texture.

This script avoids both: texture-aware quadric edge collapse (pymeshlab) keeps UV
seams intact while cutting triangle count, and the diffuse map is preserved
(downsampled to Roblox's 1024x1024 ceiling).

Outputs:
    <name>.obj / .mtl / <name>_diffuse.png   -> standard Studio "Import 3D"
    <name>.luau                              -> EditableMesh data module WITH UVs
    <name>.json                              -> same, for the localhost bridge

Usage:
    python hi3d_to_roblox.py "path/to/model.obj" -o outdir --tris 9000 --height 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[hi3d] {msg}", flush=True)


def decimate_with_uvs(obj_path: Path, target_tris: int, out_obj: Path) -> None:
    """Texture-aware quadric edge collapse. Preserves UV parameterization."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    t0 = time.time()
    log(f"loading {obj_path.name} (this can take a minute for large meshes)...")
    ms.load_new_mesh(str(obj_path))
    m = ms.current_mesh()
    log(f"loaded in {time.time() - t0:.1f}s: {m.vertex_number():,} verts / "
        f"{m.face_number():,} tris")

    # Texture-aware collapse refuses to cross UV-island borders, so a mesh with many
    # islands (typical of AI-generated 8K-textured models) plateaus well above target.
    # Escalate permissiveness across passes: keep seams pristine if we can, relax only
    # as far as needed. `extratcoordw` is the UV-distortion weight — lowering it lets
    # the collapser trade a little texture stretch for a lot of triangle reduction.
    passes = [
        dict(qualitythr=0.3, preserveboundary=True, extratcoordw=1.0),
        dict(qualitythr=0.15, preserveboundary=False, extratcoordw=0.5),
        dict(qualitythr=0.05, preserveboundary=False, extratcoordw=0.1),
    ]
    for idx, params in enumerate(passes, start=1):
        if m.face_number() <= target_tris * 1.05:
            break
        t0 = time.time()
        log(f"decimation pass {idx}/{len(passes)} -> {target_tris:,} tris "
            f"(from {m.face_number():,})...")
        ms.meshing_decimation_quadric_edge_collapse_with_texture(
            targetfacenum=target_tris,
            planarquadric=True,      # collapse flat panels aggressively
            preservenormal=True,     # don't let the silhouette flip
            optimalplacement=True,
            **params,
        )
        m = ms.current_mesh()
        log(f"  pass {idx} done in {time.time() - t0:.1f}s: {m.vertex_number():,} verts / "
            f"{m.face_number():,} tris")

    if m.face_number() > target_tris * 1.5:
        log(f"NOTE: plateaued at {m.face_number():,} tris (UV islands limit further "
            "collapse without visible texture stretching)")

    ms.save_current_mesh(
        str(out_obj),
        save_vertex_normal=True,
        save_wedge_texcoord=True,
        save_face_color=False,
    )
    log(f"wrote {out_obj.name}")


def downsample_texture(src: Path, dst: Path, max_size: int = 1024) -> None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # 8K+ source maps trip the decompression-bomb guard
    img = Image.open(src).convert("RGB")
    log(f"texture in: {img.width}x{img.height}")
    if max(img.size) > max_size:
        img = img.resize((max_size, max_size), Image.Resampling.LANCZOS)
    img.save(dst, "PNG", optimize=True)
    log(f"texture out: {img.width}x{img.height} -> {dst.name} "
        f"({dst.stat().st_size // 1024} KB)")


def crease_normals(verts: np.ndarray, faces: np.ndarray,
                   crease_deg: float = 40.0) -> np.ndarray:
    """Per-face-corner normals with hard-edge preservation.

    Averaging every face normal at a vertex ("full smooth shading") is correct for
    organic shapes but ruins hard-surface props: crisp panel bevels turn into a waxy,
    melted look. Instead, a corner only averages in the neighbouring faces whose
    normal lies within `crease_deg` of its own — so flat panels stay flat, bevels stay
    sharp, and genuinely curved regions still shade smoothly.

    Returns (F, 3, 3): a normal per corner of every triangle.
    """
    tri = verts[faces]                                    # (F, 3, 3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(fn, axis=1, keepdims=True)
    unit_fn = np.divide(fn, area, out=np.zeros_like(fn), where=area > 1e-12)

    # vertex -> incident faces (flat CSR-style index, built by sorting)
    n_verts = len(verts)
    corner_v = faces.reshape(-1)                          # (F*3,)
    corner_f = np.repeat(np.arange(len(faces)), 3)        # (F*3,)
    order = np.argsort(corner_v, kind="stable")
    sorted_v, sorted_f = corner_v[order], corner_f[order]
    starts = np.searchsorted(sorted_v, np.arange(n_verts), side="left")
    ends = np.searchsorted(sorted_v, np.arange(n_verts), side="right")

    cos_thr = np.cos(np.radians(crease_deg))
    out = np.zeros((len(faces), 3, 3), dtype=np.float64)

    for fi in range(len(faces)):
        n_self = unit_fn[fi]
        for ci in range(3):
            v = faces[fi, ci]
            neigh = sorted_f[starts[v]:ends[v]]
            nn = unit_fn[neigh]
            keep = nn @ n_self >= cos_thr                 # within the crease angle
            acc = (nn[keep] * area[neigh][keep]).sum(axis=0)  # area-weighted
            mag = np.linalg.norm(acc)
            out[fi, ci] = acc / mag if mag > 1e-12 else n_self
    return out


def normalize_and_export(obj_path: Path, out_dir: Path, name: str,
                         target_height: float, texture_name: str,
                         crease_deg: float = 40.0) -> None:
    """Load the decimated OBJ, normalize scale/position, emit Roblox data files."""
    import trimesh

    mesh = trimesh.load(str(obj_path), process=False, force="mesh")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    uvs = None
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uvs = np.asarray(mesh.visual.uv, dtype=np.float64)
        if len(uvs) != len(verts):
            log(f"WARNING: uv count {len(uvs)} != vertex count {len(verts)}; dropping UVs")
            uvs = None
    if uvs is None:
        log("WARNING: no per-vertex UVs found — texture will not map correctly")

    # Scale so height (Y) == target_height studs; feet at y=0, centered on X/Z.
    height = max(verts[:, 1].max() - verts[:, 1].min(), 1e-9)
    verts *= target_height / height
    verts[:, 0] -= (verts[:, 0].min() + verts[:, 0].max()) / 2.0
    verts[:, 2] -= (verts[:, 2].min() + verts[:, 2].max()) / 2.0
    verts[:, 1] -= verts[:, 1].min()

    log(f"normalized: {len(verts):,} verts / {len(faces):,} tris, "
        f"{target_height} studs tall")

    t0 = time.time()
    log(f"computing crease normals (hard edges preserved above {crease_deg}deg)...")
    corner_normals = crease_normals(verts, faces, crease_deg)
    log(f"  done in {time.time() - t0:.1f}s")

    def fmt(a: np.ndarray, nd: int) -> str:
        return ",".join(f"{v:.{nd}f}".rstrip("0").rstrip(".") or "0"
                        for v in a.reshape(-1))

    vstr = fmt(verts, 4)
    tstr = ",".join(str(i) for i in (faces + 1).reshape(-1))
    ustr = fmt(uvs, 5) if uvs is not None else ""
    nstr = fmt(corner_normals, 4)

    luau = (
        "--!strict\n"
        f"-- SnapMesh/Hi3D export: {name}\n"
        f"-- {len(verts):,} verts / {len(faces):,} tris, UV-mapped.\n"
        f"-- Texture: {texture_name} (upload to Roblox, then set TextureId on the part\n"
        "-- or point a SurfaceAppearance.ColorMap at it).\n"
        "-- Build with SnapMeshForge.Build(require(thisModule)).\n"
        "-- CornerNormals: 9 floats per triangle (3 corners x xyz), crease-angle\n"
        "-- computed so hard panel edges stay sharp.\n"
        "return {\n"
        f'\tName = "{name}",\n'
        f"\tVerts = {{{vstr}}},\n"
        f"\tUVs = {{{ustr}}},\n"
        f"\tTris = {{{tstr}}},\n"
        f"\tCornerNormals = {{{nstr}}},\n"
        "}\n"
    )
    luau_path = out_dir / f"{name}.luau"
    luau_path.write_text(luau, encoding="utf-8")
    log(f"wrote {luau_path.name} ({luau_path.stat().st_size // 1024} KB)")

    payload = {
        "Name": name,
        "Verts": [round(float(v), 4) for v in verts.reshape(-1)],
        "UVs": [round(float(v), 5) for v in uvs.reshape(-1)] if uvs is not None else [],
        "Tris": [int(i) for i in (faces + 1).reshape(-1)],
        "CornerNormals": [round(float(v), 4) for v in corner_normals.reshape(-1)],
    }
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    log(f"wrote {json_path.name} ({json_path.stat().st_size // 1024} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(description="High-poly textured mesh -> Roblox-ready")
    ap.add_argument("obj", type=Path, help="source model.obj (with .mtl + texture)")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("-n", "--name", default=None, help="asset name (default: folder name)")
    ap.add_argument("--tris", type=int, default=9000, help="target triangle budget")
    ap.add_argument("--height", type=float, default=8.0, help="target height in studs")
    ap.add_argument("--texture", type=Path, default=None,
                    help="diffuse map (default: auto-detect in source folder)")
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--crease", type=float, default=40.0,
                    help="crease angle in degrees; edges sharper than this stay hard")
    args = ap.parse_args()

    if not args.obj.exists():
        log(f"ERROR: not found: {args.obj}")
        return 1

    src_dir = args.obj.parent
    name = args.name or "Asset"
    out_dir = args.out or (src_dir / "roblox_ready")
    out_dir.mkdir(parents=True, exist_ok=True)

    texture = args.texture
    if texture is None:
        for pattern in ("diffuse*", "*albedo*", "*basecolor*", "*.jpg", "*.png"):
            hits = sorted(src_dir.glob(pattern))
            if hits:
                texture = hits[0]
                break

    decimated = out_dir / f"{name}.obj"
    decimate_with_uvs(args.obj, args.tris, decimated)

    texture_name = f"{name}_diffuse.png"
    if texture and texture.exists():
        downsample_texture(texture, out_dir / texture_name, args.texture_size)
    else:
        log("WARNING: no diffuse texture found in source folder")
        texture_name = "(none)"

    normalize_and_export(decimated, out_dir, name, args.height, texture_name,
                         crease_deg=args.crease)
    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

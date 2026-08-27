"""
SnapMesh — snap a picture, get a game-ready 3D mesh.

A free, local, open image-to-3D pipeline for game developers (especially Roblox):

    [concept image]
        -> background removal            (rembg, U2-Net)
        -> image-to-3D reconstruction    (TripoSR, MIT — runs on your own GPU/CPU)
        -> cleanup + decimation          (trimesh + fast-simplification, quadric)
        -> exports:
             model.glb    standard, vertex-colored (any engine / Blender)
             model.obj    portable geometry
             model.luau   Roblox EditableMesh data module — builds the mesh
                          IN-GAME from pure code, no asset uploads at all

Usage:
    python snapmesh.py input.png [-o outdir] [--tris 3000] [--height 8]
                       [--mc-res 256] [--no-bg-removal] [--device cuda|cpu]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "TripoSR"
sys.path.insert(0, str(VENDOR_DIR))


def log(msg: str) -> None:
    print(f"[snapmesh] {msg}", flush=True)


def load_and_preprocess(image_path: Path, remove_bg: bool, foreground_ratio: float):
    from PIL import Image
    from tsr.utils import remove_background, resize_foreground

    image = Image.open(image_path).convert("RGBA")

    if remove_bg:
        import rembg

        log("removing background (U2-Net)...")
        session = rembg.new_session("u2net")
        image = remove_background(image, session)

    image = resize_foreground(image, foreground_ratio)

    # Composite onto neutral grey, as TripoSR expects
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def reconstruct(image, device: str, mc_resolution: int, chunk_size: int):
    import torch
    from tsr.system import TSR

    log("loading TripoSR (first run downloads ~1.6GB of weights from Hugging Face)...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(chunk_size)
    model.to(device)

    log(f"reconstructing on {device} ...")
    t0 = time.time()
    with torch.no_grad():
        scene_codes = model([image], device=device)
    log(f"scene encoding done in {time.time() - t0:.1f}s")

    t0 = time.time()
    meshes = model.extract_mesh(
        scene_codes, has_vertex_color=True, resolution=mc_resolution
    )
    log(f"mesh extraction done in {time.time() - t0:.1f}s")
    return meshes[0]


def cleanup_and_decimate(mesh, target_tris: int, target_height: float):
    import trimesh
    from scipy.spatial import cKDTree

    log(f"raw mesh: {len(mesh.vertices)} verts / {len(mesh.faces)} tris")

    # Keep only the largest connected component (drops floating specks)
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        components = sorted(components, key=lambda m: len(m.faces), reverse=True)
        main = components[0]
        kept = [main]
        # keep secondary components that are at least 5% of the main body
        for c in components[1:]:
            if len(c.faces) >= 0.05 * len(main.faces):
                kept.append(c)
        mesh = trimesh.util.concatenate(kept)
        log(f"kept {len(kept)}/{len(components)} components -> {len(mesh.faces)} tris")

    src_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    src_colors = (
        np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3]
        if mesh.visual.kind == "vertex"
        else np.full((len(src_vertices), 3), 200.0)
    )

    if len(mesh.faces) > target_tris:
        import fast_simplification

        reduction = 1.0 - (target_tris / len(mesh.faces))
        log(f"decimating (quadric) by {reduction * 100:.0f}% -> ~{target_tris} tris")
        points, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int64),
            target_reduction=reduction,
        )
        # re-attach colors by nearest original vertex
        tree = cKDTree(src_vertices)
        _, nearest = tree.query(points)
        colors = src_colors[nearest]
        mesh = trimesh.Trimesh(
            vertices=points,
            faces=faces,
            vertex_colors=np.clip(colors, 0, 255).astype(np.uint8),
            process=True,
        )

    # Normalize orientation + scale: TripoSR output is roughly [-0.87, 0.87] cube.
    # Scale uniformly so the mesh's height (Y) equals target_height studs, feet at y=0.
    bounds = mesh.bounds
    height = max(bounds[1][1] - bounds[0][1], 1e-6)
    scale = target_height / height
    mesh.apply_scale(scale)
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation([-center[0], -bounds[0][1], -center[2]])

    log(f"final mesh: {len(mesh.vertices)} verts / {len(mesh.faces)} tris, "
        f"height {target_height} studs")
    return mesh


def export_luau(mesh, out_path: Path, asset_name: str) -> None:
    """Emit a Roblox ModuleScript: { Name, Verts, Tris, Colors } + builder docs.

    Consumed by SnapMeshForge.luau (see repo /roblox folder): builds an
    EditableMesh with per-vertex colors entirely from code — zero uploads.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64) + 1  # 1-based for Luau
    colors = (
        np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
        if mesh.visual.kind == "vertex"
        else np.full((len(verts), 3), 0.78)
    )

    def fmt(a, nd):
        return ",".join(f"{v:.{nd}f}".rstrip("0").rstrip(".") for v in a)

    vstr = fmt(verts.reshape(-1), 3)
    cstr = fmt(colors.reshape(-1), 3)
    tstr = ",".join(str(i) for i in faces.reshape(-1))

    content = (
        "--!strict\n"
        f"-- SnapMesh export: {asset_name}\n"
        "-- Build with SnapMeshForge.Build(require(thisModule)) -- creates a MeshPart\n"
        "-- via AssetService:CreateEditableMesh() with per-vertex colors. No uploads.\n"
        "return {\n"
        f'\tName = "{asset_name}",\n'
        f"\tVerts = {{{vstr}}},\n"
        f"\tColors = {{{cstr}}},\n"
        f"\tTris = {{{tstr}}},\n"
        "}\n"
    )
    out_path.write_text(content, encoding="utf-8")
    log(f"wrote {out_path} ({out_path.stat().st_size // 1024} KB)")

    # JSON twin of the .luau module — lets Roblox Studio pull the mesh straight
    # from a local `python -m http.server` via HttpService (the live-bridge path):
    #   local data = HttpService:JSONDecode(HttpService:GetAsync(url))
    #   SnapMeshForge.Build(data)
    import json

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({
            "Name": asset_name,
            "Verts": [round(float(v), 3) for v in verts.reshape(-1)],
            "Colors": [round(float(c), 3) for c in colors.reshape(-1)],
            "Tris": [int(i) for i in faces.reshape(-1)],
        }),
        encoding="utf-8",
    )
    log(f"wrote {json_path} ({json_path.stat().st_size // 1024} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="SnapMesh: image -> game-ready 3D mesh")
    parser.add_argument("image", type=Path, help="input image (png/jpg)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="output directory")
    parser.add_argument("--tris", type=int, default=3000, help="target triangle budget")
    parser.add_argument("--height", type=float, default=8.0, help="target height in studs")
    parser.add_argument("--mc-res", type=int, default=256, help="marching cubes resolution")
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--no-bg-removal", action="store_true",
                        help="input already has transparent background")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    if not args.image.exists():
        log(f"ERROR: input not found: {args.image}")
        return 1

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out or (args.image.parent / (args.image.stem + "_snapmesh"))
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_name = args.image.stem.replace("_input", "").replace(" ", "_")

    image = load_and_preprocess(args.image, not args.no_bg_removal, args.foreground_ratio)
    image.save(out_dir / "preprocessed.png")

    mesh = reconstruct(image, device, args.mc_res, args.chunk_size)
    mesh = cleanup_and_decimate(mesh, args.tris, args.height)

    mesh.export(out_dir / f"{asset_name}.glb")
    mesh.export(out_dir / f"{asset_name}.obj")
    log(f"wrote {out_dir / (asset_name + '.glb')} and .obj")
    export_luau(mesh, out_dir / f"{asset_name}.luau", asset_name)

    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

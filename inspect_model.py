"""
inspect_model.py — quick health check on a downloaded 3D asset before it enters
the pipeline. Reports geometry counts, UV presence/quality, texture resolution,
scale and watertightness, and flags anything that would degrade the Roblox export.

Usage:
    python inspect_model.py model.glb [--dump-texture out.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--dump-texture", type=Path, default=None)
    args = ap.parse_args()

    log(f"=== {args.model.name} ({args.model.stat().st_size / 1024**2:.1f} MB) ===")
    scene = trimesh.load(str(args.model), process=False)

    if isinstance(scene, trimesh.Scene):
        log(f"scene graph: {len(scene.geometry)} geometry node(s)")
        for name, g in scene.geometry.items():
            log(f"  - {name}: {len(g.vertices):,} verts / {len(g.faces):,} faces")
        mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    else:
        mesh = scene

    log("")
    log(f"TOTAL geometry : {len(mesh.vertices):,} verts / {len(mesh.faces):,} tris")

    ext = mesh.extents
    log(f"bounding box   : {ext[0]:.4f} x {ext[1]:.4f} x {ext[2]:.4f}")
    log(f"tallest axis   : {'XYZ'[int(np.argmax(ext))]} ({ext.max():.4f})")

    # --- UVs -------------------------------------------------------------
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) == 0:
        log("UVs            : *** MISSING *** (texture cannot be mapped)")
    else:
        uv = np.asarray(uv)
        inside = np.mean((uv >= -0.001) & (uv <= 1.001))
        log(f"UVs            : {len(uv):,} coords, "
            f"{inside * 100:.1f}% inside [0,1], "
            f"range u[{uv[:,0].min():.2f},{uv[:,0].max():.2f}] "
            f"v[{uv[:,1].min():.2f},{uv[:,1].max():.2f}]")

    # --- texture ---------------------------------------------------------
    img = None
    mat = getattr(mesh.visual, "material", None)
    if mat is not None:
        for attr in ("baseColorTexture", "image"):
            img = getattr(mat, attr, None)
            if img is not None:
                break
    if img is None:
        log("texture        : *** NONE FOUND ***")
    else:
        log(f"texture        : {img.width}x{img.height} {img.mode}")
        if args.dump_texture:
            img.convert("RGB").save(args.dump_texture)
            log(f"                 dumped -> {args.dump_texture.name}")

    # --- health ----------------------------------------------------------
    log("")
    log(f"watertight     : {mesh.is_watertight}")
    log(f"winding ok     : {mesh.is_winding_consistent}")
    dup = len(mesh.faces) - len(np.unique(np.sort(mesh.faces, axis=1), axis=0))
    log(f"duplicate faces: {dup:,}")
    deg = int((mesh.area_faces < 1e-12).sum())
    log(f"degenerate tris: {deg:,}")

    # --- verdict ---------------------------------------------------------
    log("")
    tri = len(mesh.faces)
    notes = []
    if uv is None or len(uv) == 0:
        notes.append("NO UVs — unusable for textured export")
    if img is None:
        notes.append("NO texture — model will be untextured")
    if tri > 2_000_000:
        notes.append(f"very dense ({tri:,} tris) — decimation will take a minute")
    if deg > tri * 0.01:
        notes.append("many degenerate triangles — may need cleanup")
    log("VERDICT: " + ("OK for pipeline" if not notes else "; ".join(notes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

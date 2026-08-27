"""
blender_bake.py — bake real high-poly detail onto a game-budget mesh.

Runs either as a Blender script or, with no Blender install at all, against the
`bpy` PyPI module (`pip install bpy`) — same Cycles engine, fully headless:

    python blender_bake.py --high high.obj --low low.obj --out outdir --name Asset
    blender --background --python blender_bake.py -- --high ... (equivalent)

Why this beats deriving maps from the albedo (make_pbr.py): a derived normal map
guesses relief from painted brightness. This projects the ACTUAL geometry of the
original multi-million-triangle sculpt onto the decimated mesh, so every bevel,
panel inset and greeble that decimation removed comes back as true shading detail
at zero triangle cost. This is the standard AAA high-to-low workflow.

Bakes:
    <name>_normal_baked.png  — tangent-space normals from the high-poly surface
    <name>_ao_baked.png      — ambient occlusion (contact shadows in the crevices)

Notes:
  * Uses Cycles (the only engine that bakes selected-to-active).
  * The low-poly must already have a UV layout; SnapMesh's decimation preserves
    the source UVs, so this is satisfied automatically.
  * Cage extrusion is derived from the mesh's own size so the ray cast reaches
    the high-poly surface without punching through thin panels.
"""

import argparse
import math
import os
import sys

import bpy


def log(msg: str) -> None:
    print(f"[bake] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    # As a Blender script, our args come after "--"; as a plain python module run,
    # they are just normal argv.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", required=True)
    ap.add_argument("--low", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="Asset")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--extrusion", type=float, default=None,
                    help="cage extrusion in scene units (default: 2%% of mesh size)")
    return ap.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_obj(path: str, name: str):
    """Import an OBJ and return a single joined mesh object named `name`."""
    before = set(bpy.data.objects)
    # Blender 4.x wavefront importer
    bpy.ops.wm.obj_import(filepath=path, forward_axis="NEGATIVE_Z", up_axis="Y")
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        raise RuntimeError(f"no mesh imported from {path}")

    if len(new) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in new:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = new[0]

    obj.name = name
    log(f"imported {name}: {len(obj.data.polygons):,} polys, "
        f"{len(obj.data.uv_layers)} uv layer(s)")
    return obj


def setup_bake_material(low, size: int):
    """Give the low-poly a material whose active node is the target image."""
    img = bpy.data.images.new(f"bake_target", width=size, height=size, alpha=False)
    img.colorspace_settings.name = "Non-Color"  # normals/AO are data, not sRGB

    mat = bpy.data.materials.new("BakeMat")
    mat.use_nodes = True
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    node.select = True
    mat.node_tree.nodes.active = node  # Cycles bakes into the ACTIVE image node

    low.data.materials.clear()
    low.data.materials.append(mat)
    return img, node


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False

    # GPU if available — bakes are embarrassingly parallel
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs:
        cp = prefs.preferences
        for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                cp.compute_device_type = backend
                cp.get_devices()
                if any(d.type == backend for d in cp.devices):
                    for d in cp.devices:
                        d.use = (d.type == backend)
                    scene.cycles.device = "GPU"
                    log(f"using GPU backend: {backend}")
                    break
            except Exception:
                continue
        else:
            log("no GPU backend available; baking on CPU")

    high = import_obj(args.high, "HighPoly")
    low = import_obj(args.low, "LowPoly")

    if not low.data.uv_layers:
        raise RuntimeError("low-poly has no UV map — cannot bake into UV space")

    img, node = setup_bake_material(low, args.size)

    # Extrusion: the ray must start outside the high-poly surface and travel inward.
    # Scale to the model so this works for a 2-stud prop or a 200-stud building.
    dims = max(low.dimensions)
    extrusion = args.extrusion if args.extrusion is not None else max(dims * 0.02, 1e-4)
    log(f"mesh size {dims:.3f}, cage extrusion {extrusion:.4f}")

    bake = scene.render.bake
    bake.use_selected_to_active = True
    bake.cage_extrusion = extrusion
    bake.max_ray_distance = extrusion * 2.0
    bake.use_clear = True
    bake.margin = 16  # bleed past UV island edges to avoid seams under mipmapping

    def run(bake_type: str, out_name: str, colorspace: str):
        node.image = img
        img.colorspace_settings.name = colorspace
        bpy.ops.object.select_all(action="DESELECT")
        high.select_set(True)          # source
        low.select_set(True)
        bpy.context.view_layer.objects.active = low   # target must be ACTIVE
        log(f"baking {bake_type} at {args.size}x{args.size} ...")
        kwargs = dict(type=bake_type, use_selected_to_active=True,
                      cage_extrusion=extrusion, margin=bake.margin, use_clear=True)
        if bake_type == "NORMAL":
            kwargs["normal_space"] = "TANGENT"
        bpy.ops.object.bake(**kwargs)
        path = os.path.join(args.out, out_name)
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        log(f"wrote {out_name}")

    run("NORMAL", f"{args.name}_normal_baked.png", "Non-Color")
    run("AO", f"{args.name}_ao_baked.png", "Non-Color")

    log("done.")


if __name__ == "__main__":
    main()

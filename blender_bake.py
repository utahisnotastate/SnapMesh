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
    ap.add_argument("--ray-distance", type=float, default=None,
                    help="max ray distance (default: just past the extrusion)")
    ap.add_argument("--no-cage", action="store_true",
                    help="use scalar extrusion instead of an explicit cage mesh")
    ap.add_argument("--no-float", action="store_true",
                    help="bake to 8-bit instead of 32-bit float")
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

    # Two prep steps that materially affect bake quality:
    #  1. Normals must face outward. An inverted face bakes a garbage ray hit.
    #  2. The LOW-poly must be smooth-shaded BEFORE baking: its shading defines the
    #     tangent basis the tangent-space normal map is written into. Baking against
    #     flat per-face tangents encodes a discontinuity at every edge, which shows
    #     up as "crinkly" noise across the surface.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.shade_smooth()

    log(f"imported {name}: {len(obj.data.polygons):,} polys, "
        f"{len(obj.data.uv_layers)} uv layer(s), normals fixed + smooth shaded")
    return obj


def make_cage(low, offset: float):
    """Build an explicit cage: a copy of the low-poly pushed out along its normals.

    More robust than a scalar `cage_extrusion`, which offsets along the low-poly's
    interpolated normals only at ray-cast time. A real cage mesh gives every ray a
    well-defined start point and direction, which matters on beveled or concave
    areas where a naive offset can send rays into neighbouring geometry.
    """
    mesh = low.data.copy()
    cage = bpy.data.objects.new("BakeCage", mesh)
    cage.matrix_world = low.matrix_world.copy()
    bpy.context.collection.objects.link(cage)

    disp = cage.modifiers.new("CageOffset", type="DISPLACE")
    disp.direction = "NORMAL"
    disp.mid_level = 0.0
    disp.strength = offset

    bpy.ops.object.select_all(action="DESELECT")
    cage.select_set(True)
    bpy.context.view_layer.objects.active = cage
    bpy.ops.object.modifier_apply(modifier=disp.name)
    return cage


def setup_bake_material(low, size: int, float_buffer: bool = True):
    """Give the low-poly a material whose active node is the target image.

    float_buffer: bake into 32-bit float. An 8-bit target quantises each channel to
    256 steps, and that stepping compounds through the later blend math into visible
    banding. We bake in float and only quantise once, at PNG write.
    """
    img = bpy.data.images.new("bake_target", width=size, height=size, alpha=False,
                              float_buffer=float_buffer, is_data=True)
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

    img, node = setup_bake_material(low, args.size, float_buffer=not args.no_float)

    # Extrusion: the ray must start outside the high-poly surface and travel inward.
    # Scale to the model so this works for a 2-stud prop or a 200-stud building.
    dims = max(low.dimensions)
    extrusion = args.extrusion if args.extrusion is not None else max(dims * 0.02, 1e-4)
    # Keep the ray short — a long max distance lets rays sail past the high-poly surface
    # and latch onto unrelated geometry across the model, which reads as noise.
    ray_distance = args.ray_distance if args.ray_distance is not None else extrusion * 1.05
    log(f"mesh size {dims:.3f}, cage extrusion {extrusion:.4f}, "
        f"max ray distance {ray_distance:.4f}")

    bake = scene.render.bake
    bake.use_selected_to_active = True
    bake.cage_extrusion = extrusion
    bake.max_ray_distance = ray_distance
    bake.use_clear = True
    bake.margin = 16  # bleed past UV island edges to avoid seams under mipmapping

    # Roblox samples tangent-space normal maps with the OpenGL (+Y up) convention,
    # which is Blender's default — set explicitly so a future default change or a
    # DirectX-targeted fork can't silently invert the green channel.
    bake.normal_space = "TANGENT"
    bake.normal_r = "POS_X"
    bake.normal_g = "POS_Y"
    bake.normal_b = "POS_Z"

    cage = None
    if not args.no_cage:
        cage = make_cage(low, extrusion)
        bake.use_cage = True
        bake.cage_object = cage
        log(f"using explicit cage mesh (offset {extrusion:.4f})")

    def run(bake_type: str, out_name: str, colorspace: str):
        node.image = img
        img.colorspace_settings.name = colorspace
        bpy.ops.object.select_all(action="DESELECT")
        high.select_set(True)          # source
        low.select_set(True)
        bpy.context.view_layer.objects.active = low   # target must be ACTIVE
        log(f"baking {bake_type} at {args.size}x{args.size} ...")
        kwargs = dict(type=bake_type, use_selected_to_active=True,
                      cage_extrusion=extrusion, max_ray_distance=ray_distance,
                      margin=bake.margin, use_clear=True)
        if cage is not None:
            kwargs["use_cage"] = True
            kwargs["cage_object"] = cage.name
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

    if cage is not None:
        mesh = cage.data
        bpy.data.objects.remove(cage, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)

    log("done.")


if __name__ == "__main__":
    main()

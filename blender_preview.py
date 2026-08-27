"""
blender_preview.py — render a turntable-style PBR preview of a game-ready mesh.

Useful for two things: verifying that a generated map set actually reads correctly
before it goes in-engine, and producing clean marketing/thumbnail renders of assets.

Usage:
    python blender_preview.py --mesh low.obj --color c.png [--normal n.png]
        [--roughness r.png] [--metalness m.png] --out preview.png [--views 3]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import bpy


def log(msg: str) -> None:
    print(f"[preview] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--color", required=True)
    ap.add_argument("--normal")
    ap.add_argument("--roughness")
    ap.add_argument("--metalness")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--samples", type=int, default=48)
    return ap.parse_args(argv)


def build_material(args) -> bpy.types.Material:
    mat = bpy.data.materials.new("PreviewPBR")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def tex(path, non_color=False):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path)
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        return n

    nt.links.new(tex(args.color).outputs["Color"], bsdf.inputs["Base Color"])

    if args.normal and os.path.exists(args.normal):
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tex(args.normal, True).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if args.roughness and os.path.exists(args.roughness):
        nt.links.new(tex(args.roughness, True).outputs["Color"], bsdf.inputs["Roughness"])
    if args.metalness and os.path.exists(args.metalness):
        nt.links.new(tex(args.metalness, True).outputs["Color"], bsdf.inputs["Metallic"])

    return mat


def three_point(target_z: float, radius: float) -> None:
    """Key / fill / rim — reads hard-surface detail far better than a single lamp."""
    specs = [
        ("Key", (radius * 1.4, -radius * 1.4, radius * 1.7), 900.0),
        ("Fill", (-radius * 1.8, -radius * 0.9, radius * 0.7), 260.0),
        ("Rim", (-radius * 0.6, radius * 2.0, radius * 1.5), 700.0),
    ]
    for name, loc, power in specs:
        d = bpy.data.lights.new(name, type="AREA")
        d.energy = power
        d.size = radius * 1.5
        o = bpy.data.objects.new(name, d)
        o.location = loc
        bpy.context.collection.objects.link(o)
        # aim at the model's mid-height
        dx, dy, dz = -loc[0], -loc[1], target_z - loc[2]
        o.rotation_euler = (math.atan2(math.hypot(dx, dy), -dz), 0,
                            math.atan2(dy, dx) + math.pi / 2)


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.render.film_transparent = False
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res

    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs:
        cp = prefs.preferences
        for backend in ("OPTIX", "CUDA", "HIP"):
            try:
                cp.compute_device_type = backend
                cp.get_devices()
                if any(d.type == backend for d in cp.devices):
                    for d in cp.devices:
                        d.use = (d.type == backend)
                    scene.cycles.device = "GPU"
                    break
            except Exception:
                continue

    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.055, 0.07, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.world = world

    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=args.mesh, forward_axis="NEGATIVE_Z", up_axis="Y")
    meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active or meshes[0]

    obj.data.materials.clear()
    obj.data.materials.append(build_material(args))
    bpy.ops.object.shade_smooth()
    obj.data.use_auto_smooth = True if hasattr(obj.data, "use_auto_smooth") else None

    radius = max(obj.dimensions)
    mid_z = obj.location.z + obj.dimensions.z * 0.5
    three_point(mid_z, radius)

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 60
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam

    base, ext = os.path.splitext(args.out)
    for i in range(args.views):
        ang = math.radians(35 + i * (360.0 / max(args.views, 1)) * 0.28)
        dist = radius * 2.6
        cam.location = (math.sin(ang) * dist, -math.cos(ang) * dist, mid_z + radius * 0.55)
        dx, dy = -cam.location[0], -cam.location[1]
        dz = mid_z - cam.location[2]
        cam.rotation_euler = (math.atan2(math.hypot(dx, dy), -dz), 0,
                              math.atan2(dy, dx) + math.pi / 2)
        path = args.out if args.views == 1 else f"{base}_{i}{ext}"
        scene.render.filepath = path
        log(f"rendering view {i + 1}/{args.views} -> {os.path.basename(path)}")
        bpy.ops.render.render(write_still=True)

    log("done.")


if __name__ == "__main__":
    main()

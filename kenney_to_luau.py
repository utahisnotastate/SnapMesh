"""
kenney_to_luau.py — batch-convert Kenney (or any) OBJ props into Luau data
modules that Roblox can build at runtime via AssetService:CreateEditableMesh().

Why not just import the OBJs: Roblox's Import 3D is a manual, per-file GUI flow
and uploads each mesh as a moderated asset. Emitting the geometry as code means a
whole prop library ships with the place, builds at server boot, and needs no
uploads at all.

Kenney OBJs carry one material group per colour region. We keep those groups
separate so each can be re-materialised independently at build time — that's what
lets a flat-shaded toon prop be recoloured into a different art style without
touching its geometry.

Output per prop:  <Name>.luau  -> { Name, Groups = { {Material, Color, Verts, Tris} } }

Usage:
    python kenney_to_luau.py --manifest manifest.txt -o outdir [--scale 8]
    python kenney_to_luau.py --obj path/to/x.obj -o outdir --name Rock_A
"""

from __future__ import annotations

import argparse
from pathlib import Path


def log(msg: str) -> None:
    print(f"[kenney] {msg}", flush=True)


def parse_mtl(mtl_path: Path) -> dict[str, tuple[float, float, float]]:
    colors: dict[str, tuple[float, float, float]] = {}
    if not mtl_path.exists():
        return colors
    cur = None
    for line in mtl_path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("newmtl "):
            cur = s[7:].strip()
        elif cur and s.startswith("Kd "):
            p = s[3:].split()
            colors[cur] = (float(p[0]), float(p[1]), float(p[2]))
    return colors


def convert(obj_path: Path, scale: float) -> dict:
    """Parse an OBJ into per-material groups of (verts, tris)."""
    lines = obj_path.read_text(errors="ignore").splitlines()

    mtl_colors: dict[str, tuple[float, float, float]] = {}
    for line in lines:
        if line.startswith("mtllib "):
            mtl_colors = parse_mtl(obj_path.parent / line[7:].strip())
            break

    vx: list[float] = []
    vy: list[float] = []
    vz: list[float] = []
    groups: dict[str, dict] = {}
    cur_mtl = "default"

    for line in lines:
        s = line.strip()
        if s.startswith("v "):
            p = s[2:].split()
            vx.append(float(p[0])); vy.append(float(p[1])); vz.append(float(p[2]))
        elif s.startswith("usemtl "):
            cur_mtl = s[7:].strip()
        elif s.startswith("f "):
            g = groups.setdefault(cur_mtl, {"verts": [], "tris": [], "map": {}})
            local = []
            for tok in s[2:].split():
                vi = int(tok.split("/")[0])
                if vi < 0:
                    vi = len(vx) + 1 + vi
                if vi not in g["map"]:
                    g["verts"].extend((vx[vi - 1] * scale,
                                       vy[vi - 1] * scale,
                                       vz[vi - 1] * scale))
                    g["map"][vi] = len(g["verts"]) // 3  # 1-based local index
                local.append(g["map"][vi])
            # fan-triangulate n-gons
            for i in range(1, len(local) - 1):
                g["tris"].extend((local[0], local[i], local[i + 1]))

    out_groups = []
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for name, g in groups.items():
        if not g["tris"]:
            continue
        c = mtl_colors.get(name, (0.5, 0.5, 0.5))
        vs = g["verts"]
        for i in range(0, len(vs), 3):
            for axis in range(3):
                v = vs[i + axis]
                if v < lo[axis]:
                    lo[axis] = v
                if v > hi[axis]:
                    hi[axis] = v
        out_groups.append({"material": name, "color": c,
                           "verts": vs, "tris": g["tris"]})

    # Precomputed bounds matter: Roblox's Model:GetBoundingBox() returns garbage
    # (observed 240 studs for a 4.4-stud prop) when any child MeshPart has a zero
    # extent -- which happens for legitimate flat faces, common in kit props. So we
    # ship exact bounds from the source geometry and never ask the engine at runtime.
    if not out_groups:
        lo = hi = [0.0, 0.0, 0.0]
    return {"groups": out_groups, "min": lo, "max": hi}


def fmt_floats(vals, nd=4) -> str:
    out = []
    for v in vals:
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        out.append(s if s not in ("", "-") else "0")
    return ",".join(out)


def emit_luau(name: str, data: dict, out_path: Path) -> None:
    parts = []
    total_tris = 0
    for g in data["groups"]:
        total_tris += len(g["tris"]) // 3
        cr, cg, cb = g["color"]
        parts.append(
            "\t\t{ Material = \"%s\", Color = {%s,%s,%s}, Verts = {%s}, Tris = {%s} },"
            % (g["material"],
               f"{cr:.4f}".rstrip("0").rstrip("."),
               f"{cg:.4f}".rstrip("0").rstrip("."),
               f"{cb:.4f}".rstrip("0").rstrip("."),
               fmt_floats(g["verts"]),
               ",".join(str(i) for i in g["tris"]))
        )
    body = "\n".join(parts)
    out_path.write_text(
        "--!strict\n"
        f"-- Kenney prop: {name} ({len(data['groups'])} material group(s), "
        f"{total_tris} tris)\n"
        "-- Built at runtime by XUTAHX_KenneyForge. CC0 source geometry (kenney.nl).\n"
        "return {\n"
        f"\tName = \"{name}\",\n"
        f"\tBounds = {{ Min = {{{fmt_floats(data['min'])}}}, "
        f"Max = {{{fmt_floats(data['max'])}}} }},\n"
        "\tGroups = {\n"
        f"{body}\n"
        "\t},\n"
        "}\n",
        encoding="utf-8",
    )
    return total_tris


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    help="text file: <objpath>|<OutputName> per line, # for comments")
    ap.add_argument("--obj", type=Path)
    ap.add_argument("--name")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=8.0,
                    help="Kenney units -> Roblox studs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, str]] = []

    if args.manifest:
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            src, _, nm = s.partition("|")
            src = Path(src.strip())
            jobs.append((src, (nm.strip() or src.stem)))
    elif args.obj:
        jobs.append((args.obj, args.name or args.obj.stem))
    else:
        ap.error("give --manifest or --obj")

    total = 0
    ok = 0
    for src, nm in jobs:
        if not src.exists():
            log(f"MISSING: {src}")
            continue
        data = convert(src, args.scale)
        if not data["groups"]:
            log(f"EMPTY (no faces): {src.name}")
            continue
        tris = emit_luau(nm, data, args.out / f"{nm}.luau")
        size_kb = (args.out / f"{nm}.luau").stat().st_size // 1024
        log(f"{nm:28s} {len(data['groups'])} grp {tris:6d} tris  {size_kb:4d} KB")
        total += tris
        ok += 1

    log(f"converted {ok}/{len(jobs)} props, {total:,} tris total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

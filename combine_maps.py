"""
combine_maps.py — fuse a baked normal map with an albedo-derived one, and bake
ambient occlusion into the color map.

Why fuse: the two normal maps capture genuinely different things.
  * BAKED (high-poly -> low-poly) recovers large-scale GEOMETRY that decimation
    flattened: bevels, panel steps, rounded corners.
  * DERIVED (from the albedo) recovers fine SURFACE detail that was never geometry
    in the first place — painted panel seams, circuit traces, vents.
Neither is redundant, and using only one leaves visible quality on the table.

Blending uses "partial derivative" compositing, which is the mathematically sound
way to add two tangent-space normal maps: convert each to its slope (nx/nz, ny/nz),
sum the slopes, then renormalise. Naively averaging the RGB values instead would
flatten both maps toward neutral.

AO is multiplied into the color map (Roblox's SurfaceAppearance has no AO slot),
applied at partial strength so crevice shadows read without crushing the albedo.

Usage:
    python combine_maps.py --baked N_baked.png --derived N.png --out N_final.png
    python combine_maps.py --ao AO.png --color C.png --out C_ao.png --ao-strength 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def log(msg: str) -> None:
    print(f"[combine] {msg}", flush=True)


def load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size and img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def blend_normals(baked: np.ndarray, derived: np.ndarray,
                  baked_w: float, derived_w: float) -> np.ndarray:
    """Partial-derivative blend of two tangent-space normal maps."""
    def to_slope(n: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
        v = n * 2.0 - 1.0
        nz = np.maximum(np.abs(v[..., 2]), 1e-4)
        return v[..., 0] / nz * w, v[..., 1] / nz * w

    bx, by = to_slope(baked, baked_w)
    dx, dy = to_slope(derived, derived_w)
    sx, sy = bx + dx, by + dy

    nz = np.ones_like(sx)
    inv = 1.0 / np.sqrt(sx * sx + sy * sy + nz * nz)
    out = np.stack([sx * inv, sy * inv, nz * inv], axis=-1)
    return np.clip(out * 0.5 + 0.5, 0.0, 1.0)


def apply_ao(color: np.ndarray, ao: np.ndarray, strength: float,
             normalize: bool = True) -> np.ndarray:
    """Multiply baked AO into the albedo (Roblox SurfaceAppearance has no AO slot).

    The critical step is white-point normalisation. A baked AO map almost never
    reaches pure white on flat, unoccluded surfaces — it plateaus around 0.85-0.95.
    Multiplying that straight into the albedo scales down EVERY pixel, which is what
    reads as a washed-out, desaturated result. Stretching the high end back to 1.0
    (the numpy equivalent of pulling a ColorRamp's white slider inward) means only
    genuinely occluded pixels darken.
    """
    a = ao[..., :1].astype(np.float32)

    if normalize:
        # 99th percentile, not max: immune to a few stray blown-out texels.
        hi = float(np.percentile(a, 99.0))
        lo = float(np.percentile(a, 1.0))
        if hi > lo + 1e-4:
            a = (a - lo) / (hi - lo)
        log(f"AO white point normalised (was {lo:.3f}..{hi:.3f})")

    # Rays that escape geometry entirely return black; without a floor those become
    # holes punched through the albedo.
    a = np.clip(a, 0.25, 1.0)
    a = 1.0 - (1.0 - a) * strength
    return np.clip(color * a, 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baked", type=Path)
    ap.add_argument("--derived", type=Path)
    ap.add_argument("--ao", type=Path)
    ap.add_argument("--color", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--baked-weight", type=float, default=1.0)
    ap.add_argument("--derived-weight", type=float, default=0.7)
    ap.add_argument("--ao-strength", type=float, default=0.55)
    ap.add_argument("--no-ao-normalize", action="store_true",
                    help="skip AO white-point normalisation (not recommended)")
    args = ap.parse_args()

    if args.baked and args.derived:
        b = load_rgb(args.baked, args.size)
        d = load_rgb(args.derived, args.size)
        out = blend_normals(b, d, args.baked_weight, args.derived_weight)
        log(f"blended normals (baked x{args.baked_weight}, derived x{args.derived_weight})")
    elif args.ao and args.color:
        c = load_rgb(args.color, args.size)
        a = load_rgb(args.ao, args.size)
        out = apply_ao(c, a, args.ao_strength, normalize=not args.no_ao_normalize)
        log(f"applied AO to color (strength {args.ao_strength})")
    else:
        ap.error("provide --baked+--derived, or --ao+--color")

    Image.fromarray((out * 255).astype(np.uint8)).save(args.out, "PNG", optimize=True)
    log(f"wrote {args.out.name} ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

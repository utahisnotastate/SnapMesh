"""
make_pbr.py — derive Normal / Roughness / Metalness maps from a diffuse texture.

Why this matters for generated assets: decimating a 2M-triangle model down to a
game budget throws away surface relief. A normal map puts that relief back as
shading detail at zero triangle cost, and roughness/metalness are what stop a
model from reading as flat plastic under Roblox's PBR lighting.

This is a heuristic derivation from the albedo (no high-poly bake required):
  * Normal    — Sobel gradient of a bilaterally-smoothed luminance channel, so
                painted panel seams and circuit traces become real relief while
                broad tonal shifts (lighting baked into the albedo) don't bulge.
  * Roughness — inverse luminance, contrast-shaped: bright/emissive areas read
                glossy, dark recesses read matte.
  * Metalness — low saturation + mid-to-high luminance => metal. Colored, glowing
                regions (holograms, neon) stay dielectric so they don't go grey.

For the highest fidelity, bake maps from the original high-poly mesh instead
(Substance 3D Sampler/Painter, or Blender's bake) and pass them through
--normal/--roughness/--metalness on the ingest side. This gets most of the way
for free.

Usage:
    python make_pbr.py diffuse.png -o outdir [--strength 2.0] [--size 1024]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None


def log(msg: str) -> None:
    print(f"[pbr] {msg}", flush=True)


def luminance(rgb: np.ndarray) -> np.ndarray:
    # Rec. 709 luma
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    pad = np.pad(gray, 1, mode="edge")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            win = pad[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
            gx += win * kx[dy, dx]
            gy += win * ky[dy, dx]
    return gx, gy


def make_normal(rgb: np.ndarray, strength: float) -> np.ndarray:
    gray = luminance(rgb)

    # Blur slightly first: raw albedo noise would otherwise become sparkly relief.
    blurred = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=1.0)
        ),
        dtype=np.float32,
    ) / 255.0

    # Subtract a heavily blurred copy (high-pass): removes baked-in lighting
    # gradients so only local detail — seams, panel lines — becomes relief.
    low = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=24.0)
        ),
        dtype=np.float32,
    ) / 255.0
    detail = np.clip(blurred - low + 0.5, 0.0, 1.0)

    gx, gy = sobel(detail)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    # Pack to tangent-space RGB (OpenGL +Y convention, which Roblox expects)
    out = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
    return np.clip(out, 0, 1)


def make_roughness(rgb: np.ndarray) -> np.ndarray:
    lum = luminance(rgb)
    # Bright => glossier. Compress toward the middle so nothing is a perfect mirror
    # or a total void, both of which look wrong on stylized assets.
    rough = 1.0 - lum
    rough = 0.25 + rough * 0.6
    return np.clip(rough, 0, 1)


def make_metalness(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = np.divide(mx - mn, np.maximum(mx, 1e-6))
    lum = luminance(rgb)
    # Desaturated AND reasonably bright => metal. Saturated glows stay dielectric.
    metal = (1.0 - np.clip(sat / 0.35, 0, 1)) * np.clip((lum - 0.25) / 0.45, 0, 1)
    return np.clip(metal, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive PBR maps from a diffuse texture")
    ap.add_argument("diffuse", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("-n", "--name", default=None)
    ap.add_argument("--strength", type=float, default=2.5, help="normal map strength")
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    out_dir = args.out or args.diffuse.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.diffuse.stem.replace("_diffuse", "")

    img = Image.open(args.diffuse).convert("RGB")
    if max(img.size) > args.size:
        img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    log(f"source {img.width}x{img.height}")

    maps = {
        "normal": (make_normal(rgb, args.strength) * 255).astype(np.uint8),
        "roughness": (np.stack([make_roughness(rgb)] * 3, -1) * 255).astype(np.uint8),
        "metalness": (np.stack([make_metalness(rgb)] * 3, -1) * 255).astype(np.uint8),
    }
    for kind, arr in maps.items():
        p = out_dir / f"{name}_{kind}.png"
        Image.fromarray(arr).save(p, "PNG", optimize=True)
        log(f"wrote {p.name} ({p.stat().st_size // 1024} KB)")

    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

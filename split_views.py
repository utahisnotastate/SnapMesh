"""
split_views.py — cut a multi-view concept sheet into individual view images,
ready to feed a multi-view image-to-3D generator (Hi3D, Tripo, etc).

Those generators want canonical slots — Front / Left / Right / Back — but concept
sheets are usually one wide image containing several panels on a flat background.
This finds the panels by locating the empty background columns between them, crops
each one, and can mirror a side view to synthesise the opposite side (valid for the
bilaterally symmetric hard-surface props these pipelines are usually fed).

Deliberately NOT automatic about slot assignment: a 3/4 view must never be placed
in a Left/Right slot (those expect flat profiles; a diagonal corrupts the
reconstructed silhouette), and only a human can say whether a given side view is
the object's left or right. So this emits panel_0..N plus a contact sheet, and you
assign slots with --assign.

Usage:
    # 1. see how the sheet splits
    python split_views.py sheet.png -o out/ --name AI_Matrix

    # 2. assign slots (skip a panel with '-'), optionally mirroring for the other side
    python split_views.py sheet.png -o out/ --name AI_Matrix \
        --assign front,right,back,- --mirror-right-to-left
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

VALID_SLOTS = {"front", "back", "left", "right", "-"}


def log(msg: str) -> None:
    print(f"[split] {msg}", flush=True)


def find_panels(img: Image.Image, bg_tol: float, min_frac: float) -> list[tuple[int, int]]:
    """Return (x0, x1) spans of content, split on background-only columns."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)

    # Background = the modal corner colour. Concept sheets are flat-backed, so
    # sampling the four corners is more reliable than assuming black or white.
    h, w, _ = rgb.shape
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    bg = np.median(corners, axis=0)

    dist = np.linalg.norm(rgb - bg, axis=2)
    content = dist > (bg_tol * 255.0)
    col_frac = content.mean(axis=0)
    is_content = col_frac > min_frac

    spans, start = [], None
    for x, c in enumerate(is_content):
        if c and start is None:
            start = x
        elif not c and start is not None:
            spans.append((start, x))
            start = None
    if start is not None:
        spans.append((start, len(is_content)))

    # Drop slivers (stray marks, drop shadows) — real panels are a decent slice.
    spans = [(a, b) for a, b in spans if (b - a) > w * 0.04]
    return spans


def crop_panel(img: Image.Image, x0: int, x1: int, bg_tol: float,
               pad_frac: float = 0.04) -> Image.Image:
    """Crop a panel span, then trim vertically to its own content + pad."""
    panel = img.crop((x0, 0, x1, img.height)).convert("RGB")
    a = np.asarray(panel, dtype=np.float32)
    corners = np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]])
    bg = np.median(corners, axis=0)
    content = np.linalg.norm(a - bg, axis=2) > (bg_tol * 255.0)

    rows = np.where(content.any(axis=1))[0]
    y0, y1 = (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, panel.height)

    pad = int(max(panel.width, y1 - y0) * pad_frac)
    box = (max(0, -pad), max(0, y0 - pad),
           min(panel.width, panel.width + pad), min(panel.height, y1 + pad))
    return panel.crop(box)


def contact_sheet(panels: list[Image.Image], out: Path, labels: list[str]) -> None:
    tile = 300
    sheet = Image.new("RGB", (tile * len(panels) + 10 * (len(panels) + 1), tile + 40),
                      (22, 24, 31))
    draw = ImageDraw.Draw(sheet)
    for i, (p, lab) in enumerate(zip(panels, labels)):
        c = p.copy()
        c.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x = 10 + i * (tile + 10) + (tile - c.width) // 2
        sheet.paste(c, (x, 30 + (tile - c.height) // 2))
        draw.text((10 + i * (tile + 10) + 4, 10), lab, fill=(220, 225, 235))
    sheet.save(out)
    log(f"contact sheet -> {out.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("-n", "--name", required=True)
    ap.add_argument("--assign", default=None,
                    help="comma-separated slots per panel, e.g. front,right,back,-")
    ap.add_argument("--mirror-right-to-left", action="store_true",
                    help="synthesise the left view by mirroring the right (symmetric props)")
    ap.add_argument("--mirror-left-to-right", action="store_true")
    ap.add_argument("--bg-tol", type=float, default=0.06,
                    help="how far from background colour counts as content (0-1)")
    ap.add_argument("--min-frac", type=float, default=0.01,
                    help="min fraction of a column that must be content")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    img = Image.open(args.sheet)
    log(f"{args.sheet.name}: {img.width}x{img.height}")

    spans = find_panels(img, args.bg_tol, args.min_frac)
    if not spans:
        log("no panels detected — try raising --bg-tol")
        return 1
    log(f"detected {len(spans)} panel(s): " +
        ", ".join(f"x{a}-{b}" for a, b in spans))

    panels = [crop_panel(img, a, b, args.bg_tol) for a, b in spans]

    if not args.assign:
        for i, p in enumerate(panels):
            path = args.out / f"{args.name}_panel{i}.png"
            p.save(path)
            log(f"  panel{i}: {p.width}x{p.height} -> {path.name}")
        contact_sheet(panels, args.out / f"{args.name}_panels.png",
                      [f"panel{i}" for i in range(len(panels))])
        log("")
        log("Now pick slots and re-run with --assign, e.g.:")
        log(f"  --assign front,right,back,-   (use '-' to skip a panel, "
            f"e.g. a 3/4 view, which must NOT go in a Left/Right slot)")
        return 0

    slots = [s.strip().lower() for s in args.assign.split(",")]
    if len(slots) != len(panels):
        log(f"ERROR: --assign has {len(slots)} entries but {len(panels)} panels detected")
        return 1
    bad = [s for s in slots if s not in VALID_SLOTS]
    if bad:
        log(f"ERROR: invalid slot(s) {bad}; valid: front, back, left, right, -")
        return 1

    written: dict[str, Image.Image] = {}
    for p, slot in zip(panels, slots):
        if slot == "-":
            continue
        path = args.out / f"{args.name}_{slot}.png"
        p.save(path)
        written[slot] = p
        log(f"  {slot:6s} {p.width}x{p.height} -> {path.name}")

    # Mirroring is only valid for bilaterally symmetric props — the left profile of
    # such an object genuinely is the mirror of its right profile.
    for src, dst, flag in (("right", "left", args.mirror_right_to_left),
                           ("left", "right", args.mirror_left_to_right)):
        if flag:
            if src not in written:
                log(f"ERROR: --mirror-{src}-to-{dst} needs a '{src}' panel assigned")
                return 1
            m = written[src].transpose(Image.FLIP_LEFT_RIGHT)
            path = args.out / f"{args.name}_{dst}.png"
            m.save(path)
            written[dst] = m
            log(f"  {dst:6s} {m.width}x{m.height} -> {path.name}  (mirrored from {src})")

    order = [s for s in ("front", "left", "right", "back") if s in written]
    contact_sheet([written[s] for s in order],
                  args.out / f"{args.name}_views.png", order)
    missing = [s for s in ("front", "left", "right", "back") if s not in written]
    log(f"done. slots filled: {', '.join(order)}" +
        (f" | missing: {', '.join(missing)}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

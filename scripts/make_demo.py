#!/usr/bin/env python3
"""Generate a small synthetic unregistered RGB-T tracking dataset.

The point is reproducibility without a private dataset: every sequence is
CLEAN by construction (boxes are drawn from the same trajectory that the
annotation file records), so a correct auditor must return PASS for all of
them. Feed it to audit.py for the regression check, and to benchmark.py to
inject known errors and compare the rules / direct-VLM / layered baselines.

Layout matches the unregistered LUART contract:

    <out>/<Class>_<id>/
      visible.txt            x,y,w,h per line, visible coordinate system
      infrared.txt           x,y,w,h per line, infrared coordinate system
      NotAlign/visible/v001.jpg ...
      NotAlign/infrared/i001.jpg ...

Modalities differ in image size, appearance, and box scale, and are never
compared across coordinates - exactly the property the audit must respect.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw

MODALITIES = {
    # name: (width, height, background, target, prefix)
    "visible": (640, 480, (182, 188, 196), (58, 62, 74), "v"),
    "infrared": (512, 400, (24, 26, 32), (238, 240, 246), "i"),
}


def _trajectory(n_frames, w, h, rng):
    """Straight or gently curved path, plus one category of motion style."""
    x, y = rng.uniform(0.2, 0.5) * w, rng.uniform(0.3, 0.6) * h
    angle = rng.uniform(-0.5, 0.5)
    speed = rng.uniform(0.012, 0.05) * w
    curve = rng.uniform(-0.02, 0.02)
    pts = []
    for _ in range(n_frames):
        pts.append((x, y))
        angle += curve
        x += speed * math.cos(angle)
        y += speed * math.sin(angle) * 0.6
        if x < 0.1 * w or x > 0.9 * w:
            angle = math.pi - angle
        if y < 0.1 * h or y > 0.9 * h:
            angle = -angle
        x = min(max(x, 0.05 * w), 0.95 * w)
        y = min(max(y, 0.05 * h), 0.95 * h)
    return pts


def _render(path_pts, mod, w, h, scale, rng, out_dir):
    boxes = []
    bw, bh = 46 * scale, 64 * scale
    for idx, (cx, cy) in enumerate(path_pts, start=1):
        img = Image.new("RGB", (w, h), mod[2])
        draw = ImageDraw.Draw(img)
        # static ground texture so the frame is not a flat colour
        for gy in range(0, h, 40):
            draw.line((0, gy, w, gy), fill=tuple(c + 8 for c in mod[2]))
        jx, jy = rng.uniform(-2, 2), rng.uniform(-2, 2)
        draw.ellipse([cx - bw / 2 + jx, cy - bh / 2 + jy,
                      cx + bw / 2 + jx, cy + bh / 2 + jy], fill=mod[3])
        img.save(os.path.join(out_dir, f"{mod[4]}{idx:03d}.jpg"), quality=72)
        boxes.append([max(0, round(cx - bw / 2)), max(0, round(cy - bh / 2)),
                      round(bw), round(bh)])
    return boxes


def build(out_root, n_seq, seed):
    rng = random.Random(seed)
    classes = ["Car", "Bus", "Pedestrian", "Bicycle", "Electricbike"]
    made = []
    for i in range(n_seq):
        cls = classes[i % len(classes)]
        name = f"{cls}_{100 + i}"
        d = os.path.join(out_root, name)
        os.makedirs(os.path.join(d, "NotAlign", "visible"), exist_ok=True)
        os.makedirs(os.path.join(d, "NotAlign", "infrared"), exist_ok=True)
        n_frames = rng.choice([28, 32, 36])
        # one geometric truth per frame, expressed in each modality's own
        # coordinate system (different resolution => different pixel scale)
        norm = _trajectory(n_frames, 1.0, 1.0, rng)
        for mname, mod in MODALITIES.items():
            pts = [(nx * mod[0], ny * mod[1]) for nx, ny in norm]
            scale = 1.0 if mname == "visible" else mod[0] / MODALITIES["visible"][0]
            boxes = _render(pts, mod, mod[0], mod[1], scale, rng,
                            os.path.join(d, "NotAlign", mname))
            with open(os.path.join(d, f"{mname}.txt"), "w") as f:
                for b in boxes:
                    f.write(",".join(str(v) for v in b) + "\n")
        made.append(name)
    meta = {"generated_by": "make_demo.py", "seed": seed,
            "sequences": made, "note": "all sequences are clean by construction"}
    with open(os.path.join(out_root, "demo_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"demo dataset: {out_root} ({len(made)} clean sequences, "
          f"{n_frames}-ish frames each, unregistered dual-modality)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="demo dataset root to create")
    ap.add_argument("--n", type=int, default=14, help="number of sequences")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    return build(args.out, args.n, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import math

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_annotation(path: Path):
    rows = []
    if not path.exists():
        return rows
    lines = path.read_text(errors="ignore").splitlines()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        try:
            l, t, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if w <= 1 or h <= 1:
            continue
        rows.append((cls, l, t, w, h))
    return rows


def stable_split(key: str, seed: int):
    h = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < 0.60:
        return "train"
    if x < 0.80:
        return "val"
    return "test"


def expand_box(l, t, w, h, W, H, margin):
    cx = l + w / 2.0
    cy = t + h / 2.0
    nw = w * (1 + 2 * margin)
    nh = h * (1 + 2 * margin)
    x0 = max(0, int(math.floor(cx - nw / 2)))
    y0 = max(0, int(math.floor(cy - nh / 2)))
    x1 = min(W, int(math.ceil(cx + nw / 2)))
    y1 = min(H, int(math.ceil(cy + nh / 2)))
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/ExDark/ExDark")
    ap.add_argument("--target-class", default="Cat")
    ap.add_argument("--output-root", default="data/exdark_object_crops")
    ap.add_argument("--manifest", default="data/exdark_cat_crops.csv")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument(
        "--negative-mode",
        choices=["other_objects", "all_objects"],
        default="other_objects",
        help="other_objects: target-class boxes are positive, every other annotated object box is negative.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    image_root = root
    anno_candidates = [
        root / "ExDark_Annno",
        root / "ExDark_Anno",
        root.parent / "ExDark_Annno",
        root.parent / "ExDark_Anno",
    ]
    anno_root = next((p for p in anno_candidates if p.exists()), None)

    if anno_root is None:
        raise SystemExit(
            "Could not find ExDark annotation folder. Looked for: "
            + ", ".join(str(p) for p in anno_candidates)
        )

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_ann = 0
    images_seen = 0
    boxes_seen = 0

    for class_dir in sorted(image_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("__"):
            continue
        if class_dir.name.lower().startswith("exdark_ann"):
            continue

        anno_class_dir = anno_root / class_dir.name
        if not anno_class_dir.exists():
            continue

        for img_path in sorted(class_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if "__MACOSX" in str(img_path):
                continue

            images_seen += 1
            ann_path = anno_class_dir / f"{img_path.stem}.txt"
            ann = parse_annotation(ann_path)
            if not ann:
                missing_ann += 1
                continue

            # The split is assigned per SOURCE IMAGE so multiple boxes from one image
            # can never leak across train/val/test.
            split = stable_split(str(img_path.relative_to(image_root)), args.seed)

            try:
                with Image.open(img_path) as im0:
                    im = im0.convert("RGB")
                    W, H = im.size

                    for bi, (bbox_class, l, t, w, h) in enumerate(ann):
                        boxes_seen += 1
                        if min(w, h) < args.min_side:
                            continue

                        x0, y0, x1, y1 = expand_box(l, t, w, h, W, H, args.margin)
                        if x1 <= x0 or y1 <= y0:
                            continue

                        crop = im.crop((x0, y0, x1, y1))
                        label = int(bbox_class.lower() == args.target_class.lower())

                        safe_class = bbox_class.replace("/", "_")
                        out_dir = out_root / split / safe_class
                        out_dir.mkdir(parents=True, exist_ok=True)

                        crop_name = f"{img_path.stem}_box{bi:02d}.jpg"
                        crop_path = out_dir / crop_name
                        crop.save(crop_path, quality=92)

                        rows.append({
                            "path": str(crop_path.resolve()),
                            "label": label,
                            "class_name": bbox_class,
                            "split": split,
                            "source": "ExDark_bbox_crop",
                            "original_path": str(img_path.resolve()),
                            "bbox_left": l,
                            "bbox_top": t,
                            "bbox_width": w,
                            "bbox_height": h,
                            "crop_x0": x0,
                            "crop_y0": y0,
                            "crop_x1": x1,
                            "crop_y1": y1,
                        })
            except Exception as e:
                print(f"WARNING: failed {img_path}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No crops produced. Check annotation/image paths.")

    df.to_csv(args.manifest, index=False)

    print(f"Image root:      {image_root}")
    print(f"Annotation root: {anno_root}")
    print(f"Source images seen: {images_seen}")
    print(f"Annotated boxes seen: {boxes_seen}")
    print(f"Missing/empty annotation files: {missing_ann}")
    print(f"Crops written: {len(df)}")
    print(f"Target '{args.target_class}' positives: {int(df.label.sum())}")
    print("\nSplit x label:")
    print(df.groupby(["split", "label"]).size())
    print("\nTop crop classes:")
    print(df.class_name.value_counts().head(20))
    print(f"\nManifest: {args.manifest}")


if __name__ == "__main__":
    main()

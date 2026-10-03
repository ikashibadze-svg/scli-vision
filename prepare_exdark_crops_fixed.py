#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import math
import pandas as pd
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASSES = {
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table"
}

def parse_annotation(path: Path):
    rows = []
    if not path.exists():
        return rows

    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue

        # Standard ExDark/bbGt lines are typically:
        # ClassName x y w h 0 0 ...
        # Official README also mentions annotation-tool prefix data,
        # so accept both the raw line and line[16:].
        candidates = [s.split()]
        if len(s) > 16:
            candidates.append(s[16:].strip().split())

        parsed = None
        for parts in candidates:
            if len(parts) < 5:
                continue
            if parts[0] not in CLASSES:
                continue
            try:
                l, t, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            if w <= 1 or h <= 1:
                continue
            parsed = (parts[0], l, t, w, h)
            break

        if parsed is not None:
            rows.append(parsed)

    return rows

def find_annotation(anno_class_dir: Path, img_path: Path):
    # ExDark archives commonly use "image.jpg.txt".
    candidates = [
        anno_class_dir / f"{img_path.name}.txt",
        anno_class_dir / f"{img_path.stem}.txt",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Last-resort tolerant search.
    matches = list(anno_class_dir.glob(f"{img_path.stem}*.txt"))
    return matches[0] if matches else None

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
    args = ap.parse_args()

    image_root = Path(args.root)
    anno_candidates = [
        image_root / "ExDark_Annno",
        image_root / "ExDark_Anno",
        image_root.parent / "ExDark_Annno",
        image_root.parent / "ExDark_Anno",
    ]
    anno_root = next((x for x in anno_candidates if x.exists()), None)
    if anno_root is None:
        raise SystemExit(
            "Could not find annotation root. Checked:\n  "
            + "\n  ".join(str(x) for x in anno_candidates)
        )

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    images_seen = 0
    ann_files_found = 0
    missing_ann = 0
    parsed_boxes = 0
    too_small = 0
    examples = []

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
            ann_path = find_annotation(anno_class_dir, img_path)
            if ann_path is None:
                missing_ann += 1
                continue

            ann_files_found += 1
            ann = parse_annotation(ann_path)
            if not ann:
                if len(examples) < 5:
                    examples.append(f"UNPARSED: {ann_path}")
                continue

            split = stable_split(str(img_path.relative_to(image_root)), args.seed)

            with Image.open(img_path) as im0:
                im = im0.convert("RGB")
                W, H = im.size

                for bi, (bbox_class, l, t, w, h) in enumerate(ann):
                    parsed_boxes += 1
                    if min(w, h) < args.min_side:
                        too_small += 1
                        continue

                    x0, y0, x1, y1 = expand_box(l, t, w, h, W, H, args.margin)
                    if x1 <= x0 or y1 <= y0:
                        continue

                    crop = im.crop((x0, y0, x1, y1))
                    label = int(bbox_class.lower() == args.target_class.lower())

                    out_dir = out_root / split / bbox_class
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
                        "annotation_path": str(ann_path.resolve()),
                        "bbox_left": l,
                        "bbox_top": t,
                        "bbox_width": w,
                        "bbox_height": h,
                    })

                    if len(examples) < 5:
                        examples.append(
                            f"{img_path.name} -> {ann_path.name} -> "
                            f"{bbox_class} [{l:.0f},{t:.0f},{w:.0f},{h:.0f}]"
                        )

    if not rows:
        print(f"Image root: {image_root}")
        print(f"Annotation root: {anno_root}")
        print(f"Images seen: {images_seen}")
        print(f"Annotation files found: {ann_files_found}")
        print(f"Missing annotation files: {missing_ann}")
        print(f"Parsed boxes: {parsed_boxes}")
        print("Examples:")
        for x in examples:
            print(" ", x)
        raise SystemExit("No crops produced.")

    df = pd.DataFrame(rows)
    df.to_csv(args.manifest, index=False)

    print(f"Image root:              {image_root}")
    print(f"Annotation root:         {anno_root}")
    print(f"Source images seen:      {images_seen}")
    print(f"Annotation files found:  {ann_files_found}")
    print(f"Missing annotation files:{missing_ann}")
    print(f"Parsed boxes:            {parsed_boxes}")
    print(f"Skipped small boxes:     {too_small}")
    print(f"Crops written:           {len(df)}")
    print(f"Target '{args.target_class}' positives: {int(df.label.sum())}")
    print("\nExamples:")
    for x in examples:
        print(" ", x)
    print("\nSplit x label:")
    print(df.groupby(["split", "label"]).size())
    print("\nClasses:")
    print(df.class_name.value_counts())
    print(f"\nManifest: {args.manifest}")

if __name__ == "__main__":
    main()

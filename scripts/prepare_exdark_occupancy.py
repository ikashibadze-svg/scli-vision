#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASSES = {
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table"
}

def stable_split(key: str, seed: int) -> str:
    h = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < 0.60:
        return "train"
    if x < 0.80:
        return "val"
    return "test"

def parse_annotation(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        candidates = [s.split()]
        if len(s) > 16:
            candidates.append(s[16:].strip().split())
        for parts in candidates:
            if len(parts) < 5 or parts[0] not in CLASSES:
                continue
            try:
                l, t, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            if w > 1 and h > 1:
                out.append((parts[0], l, t, w, h))
                break
    return out

def find_annotation(anno_class_dir: Path, img_path: Path):
    for p in [
        anno_class_dir / f"{img_path.name}.txt",
        anno_class_dir / f"{img_path.stem}.txt",
    ]:
        if p.exists():
            return p
    matches = list(anno_class_dir.glob(f"{img_path.stem}*.txt"))
    return matches[0] if matches else None

def expand_box(l, t, w, h, W, H, margin):
    cx, cy = l + w / 2.0, t + h / 2.0
    nw, nh = w * (1 + 2 * margin), h * (1 + 2 * margin)
    return (
        max(0, int(math.floor(cx - nw / 2))),
        max(0, int(math.floor(cy - nh / 2))),
        min(W, int(math.ceil(cx + nw / 2))),
        min(H, int(math.ceil(cy + nh / 2))),
    )

def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    aa = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    bb = max(0, bx1 - bx0) * max(0, by1 - by0)
    return inter / (aa + bb - inter + 1e-9)

def deterministic_rng(seed: int, key: str):
    h = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16) % (2**63 - 1))

def choose_background_box(W, H, pw, ph, forbidden, rng, max_iou, attempts):
    if pw >= W or ph >= H:
        return None
    for _ in range(attempts):
        x0 = int(rng.integers(0, W - pw + 1))
        y0 = int(rng.integers(0, H - ph + 1))
        box = (x0, y0, x0 + pw, y0 + ph)
        if all(iou(box, f) <= max_iou for f in forbidden):
            return box
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/ExDark/ExDark")
    ap.add_argument("--output-root", default="data/exdark_occupancy")
    ap.add_argument("--manifest", default="data/exdark_occupancy.csv")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--max-pos-per-image", type=int, default=1)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--max-negative-iou", type=float, default=0.01)
    ap.add_argument("--attempts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--train-cap-per-label", type=int, default=1800)
    ap.add_argument("--val-cap-per-label", type=int, default=700)
    ap.add_argument("--test-cap-per-label", type=int, default=1800)
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
        raise SystemExit("Could not find ExDark annotation folder.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    source_seen = 0
    bg_failures = 0

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

            ann_path = find_annotation(anno_class_dir, img_path)
            if ann_path is None:
                continue
            ann = parse_annotation(ann_path)
            if not ann:
                continue

            source_seen += 1
            split = stable_split(str(img_path.relative_to(image_root)), args.seed)
            rng = deterministic_rng(args.seed, str(img_path.relative_to(image_root)))

            try:
                with Image.open(img_path) as im0:
                    im = im0.convert("RGB")
                    W, H = im.size

                    raw_boxes = []
                    valid_objects = []
                    for cls, l, t, w, h in ann:
                        raw = (
                            max(0, int(math.floor(l))),
                            max(0, int(math.floor(t))),
                            min(W, int(math.ceil(l + w))),
                            min(H, int(math.ceil(t + h))),
                        )
                        raw_boxes.append(raw)
                        if min(w, h) >= args.min_side:
                            valid_objects.append((cls, l, t, w, h))

                    if not valid_objects:
                        continue

                    # Prefer smaller objects first: closer to the grey-kitten safety case.
                    valid_objects.sort(key=lambda z: z[3] * z[4])
                    chosen = valid_objects[:args.max_pos_per_image]

                    for oi, (cls, l, t, w, h) in enumerate(chosen):
                        pos_box = expand_box(l, t, w, h, W, H, args.margin)
                        x0, y0, x1, y1 = pos_box
                        pw, ph = x1 - x0, y1 - y0
                        if pw < 4 or ph < 4:
                            continue

                        pos_dir = output_root / split / "object"
                        pos_dir.mkdir(parents=True, exist_ok=True)
                        stem = f"{class_dir.name}_{img_path.stem}_obj{oi:02d}"
                        pos_path = pos_dir / f"{stem}.jpg"
                        im.crop(pos_box).save(pos_path, quality=92)

                        pair_id = f"{class_dir.name}/{img_path.name}/obj{oi:02d}"
                        rows.append({
                            "path": str(pos_path.resolve()),
                            "label": 1,
                            "class_name": "OBJECT_PRESENT",
                            "split": split,
                            "source": "ExDark_occupancy",
                            "pair_id": pair_id,
                            "original_class": cls,
                            "original_path": str(img_path.resolve()),
                            "bbox_x0": x0,
                            "bbox_y0": y0,
                            "bbox_x1": x1,
                            "bbox_y1": y1,
                        })

                        for ni in range(args.neg_per_pos):
                            neg_box = choose_background_box(
                                W, H, pw, ph, raw_boxes, rng,
                                args.max_negative_iou, args.attempts
                            )
                            if neg_box is None:
                                bg_failures += 1
                                continue

                            neg_dir = output_root / split / "background"
                            neg_dir.mkdir(parents=True, exist_ok=True)
                            neg_path = neg_dir / f"{stem}_bg{ni:02d}.jpg"
                            im.crop(neg_box).save(neg_path, quality=92)
                            nx0, ny0, nx1, ny1 = neg_box

                            rows.append({
                                "path": str(neg_path.resolve()),
                                "label": 0,
                                "class_name": "BACKGROUND_CLEAR",
                                "split": split,
                                "source": "ExDark_occupancy",
                                "pair_id": pair_id,
                                "original_class": cls,
                                "original_path": str(img_path.resolve()),
                                "bbox_x0": nx0,
                                "bbox_y0": ny0,
                                "bbox_x1": nx1,
                                "bbox_y1": ny1,
                            })
            except Exception as e:
                print(f"WARNING: {img_path}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No occupancy pairs were produced.")

    caps = {
        "train": args.train_cap_per_label,
        "val": args.val_cap_per_label,
        "test": args.test_cap_per_label,
    }
    kept = []
    for split, cap in caps.items():
        d = df[df.split == split]
        for label in [0, 1]:
            x = d[d.label == label].copy()
            if len(x) > cap:
                x = x.sample(n=cap, random_state=args.seed + label + (0 if split=="train" else 10 if split=="val" else 20))
            kept.append(x)
    df = pd.concat(kept, ignore_index=True)

    # Keep pair/source leakage impossible: source split was assigned before sampling.
    df.to_csv(args.manifest, index=False)

    print(f"Source images used: {source_seen}")
    print(f"Background sampling failures: {bg_failures}")
    print(f"Rows written: {len(df)}")
    print("\nSplit x label:")
    print(df.groupby(["split", "label"]).size())
    print("\nLabel names:")
    print(df.class_name.value_counts())
    print("\nOriginal object classes among positives:")
    print(df[df.label == 1].original_class.value_counts())
    print(f"\nManifest: {args.manifest}")

if __name__ == "__main__":
    main()

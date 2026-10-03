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
    rows = []
    if not path.exists():
        return rows
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
                rows.append((parts[0], l, t, w, h))
                break
    return rows


def find_annotation(anno_class_dir: Path, img_path: Path):
    candidates = [
        anno_class_dir / f"{img_path.name}.txt",
        anno_class_dir / f"{img_path.stem}.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(anno_class_dir.glob(f"{img_path.stem}*.txt"))
    return matches[0] if matches else None


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


def expand_candidate(l, t, w, h, W, H, margin):
    cx = l + w / 2.0
    cy = t + h / 2.0
    cw = w * (1 + 2 * margin)
    ch = h * (1 + 2 * margin)
    x0 = int(round(cx - cw / 2))
    y0 = int(round(cy - ch / 2))
    x1 = int(round(cx + cw / 2))
    y1 = int(round(cy + ch / 2))
    return (x0, y0, x1, y1)


def context_box(candidate, factor):
    x0, y0, x1, y1 = candidate
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0) * factor
    h = (y1 - y0) * factor
    return (
        int(round(cx - w / 2)),
        int(round(cy - h / 2)),
        int(round(cx + w / 2)),
        int(round(cy + h / 2)),
    )


def inside(box, W, H):
    x0, y0, x1, y1 = box
    return x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H and x1 > x0 and y1 > y0


def choose_background_candidate(W, H, cw, ch, forbidden, factor, rng, max_iou, attempts):
    # Sample a candidate window whose full context also remains inside the image.
    margin_x = int(math.ceil((factor - 1) * cw / 2))
    margin_y = int(math.ceil((factor - 1) * ch / 2))
    min_x = margin_x
    min_y = margin_y
    max_x0 = W - cw - margin_x
    max_y0 = H - ch - margin_y
    if max_x0 < min_x or max_y0 < min_y:
        return None

    for _ in range(attempts):
        x0 = int(rng.integers(min_x, max_x0 + 1))
        y0 = int(rng.integers(min_y, max_y0 + 1))
        cand = (x0, y0, x0 + cw, y0 + ch)
        if all(iou(cand, f) <= max_iou for f in forbidden):
            ctx = context_box(cand, factor)
            if inside(ctx, W, H):
                return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/ExDark/ExDark")
    ap.add_argument("--output-root", default="data/exdark_context")
    ap.add_argument("--manifest", default="data/exdark_context.csv")
    ap.add_argument("--candidate-margin", type=float, default=0.05)
    ap.add_argument("--context-factor", type=float, default=2.0)
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--max-pos-per-image", type=int, default=1)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--max-negative-iou", type=float, default=0.01)
    ap.add_argument("--attempts", type=int, default=120)
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
    anno_root = next((p for p in anno_candidates if p.exists()), None)
    if anno_root is None:
        raise SystemExit("Could not find ExDark annotation folder.")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    source_seen = 0
    context_edge_skips = 0
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

                    forbidden = []
                    valid = []
                    for cls, l, t, w, h in ann:
                        raw = (
                            max(0, int(math.floor(l))),
                            max(0, int(math.floor(t))),
                            min(W, int(math.ceil(l + w))),
                            min(H, int(math.ceil(t + h))),
                        )
                        forbidden.append(raw)
                        if min(w, h) >= args.min_side:
                            valid.append((cls, l, t, w, h))

                    if not valid:
                        continue

                    # Prefer small objects, closest to the original safety case.
                    valid.sort(key=lambda z: z[3] * z[4])
                    chosen = valid[:args.max_pos_per_image]

                    for oi, (cls, l, t, w, h) in enumerate(chosen):
                        cand = expand_candidate(
                            l, t, w, h, W, H, args.candidate_margin
                        )
                        if not inside(cand, W, H):
                            continue

                        ctx = context_box(cand, args.context_factor)
                        if not inside(ctx, W, H):
                            context_edge_skips += 1
                            continue

                        x0, y0, x1, y1 = cand
                        cw, ch = x1 - x0, y1 - y0

                        pair_id = f"{class_dir.name}/{img_path.name}/obj{oi:02d}"
                        stem = f"{class_dir.name}_{img_path.stem}_obj{oi:02d}"

                        pos_dir = out_root / split / "object"
                        pos_dir.mkdir(parents=True, exist_ok=True)
                        pos_path = pos_dir / f"{stem}.jpg"
                        im.crop(ctx).save(pos_path, quality=92)

                        rows.append({
                            "path": str(pos_path.resolve()),
                            "label": 1,
                            "class_name": "OBJECT_PRESENT",
                            "split": split,
                            "source": "ExDark_context",
                            "pair_id": pair_id,
                            "original_class": cls,
                            "original_path": str(img_path.resolve()),
                            "context_factor": args.context_factor,
                            "candidate_x0": x0,
                            "candidate_y0": y0,
                            "candidate_x1": x1,
                            "candidate_y1": y1,
                        })

                        for ni in range(args.neg_per_pos):
                            neg_cand = choose_background_candidate(
                                W, H, cw, ch, forbidden,
                                args.context_factor, rng,
                                args.max_negative_iou, args.attempts
                            )
                            if neg_cand is None:
                                bg_failures += 1
                                continue
                            neg_ctx = context_box(neg_cand, args.context_factor)

                            neg_dir = out_root / split / "background"
                            neg_dir.mkdir(parents=True, exist_ok=True)
                            neg_path = neg_dir / f"{stem}_bg{ni:02d}.jpg"
                            im.crop(neg_ctx).save(neg_path, quality=92)

                            nx0, ny0, nx1, ny1 = neg_cand
                            rows.append({
                                "path": str(neg_path.resolve()),
                                "label": 0,
                                "class_name": "BACKGROUND_CLEAR",
                                "split": split,
                                "source": "ExDark_context",
                                "pair_id": pair_id,
                                "original_class": cls,
                                "original_path": str(img_path.resolve()),
                                "context_factor": args.context_factor,
                                "candidate_x0": nx0,
                                "candidate_y0": ny0,
                                "candidate_x1": nx1,
                                "candidate_y1": ny1,
                            })
            except Exception as e:
                print(f"WARNING: {img_path}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No context examples were produced.")

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
                x = x.sample(
                    n=cap,
                    random_state=args.seed + label + (0 if split=="train" else 10 if split=="val" else 20),
                )
            kept.append(x)
    df = pd.concat(kept, ignore_index=True)
    df.to_csv(args.manifest, index=False)

    print(f"Source images used: {source_seen}")
    print(f"Context-edge skips: {context_edge_skips}")
    print(f"Background sampling failures: {bg_failures}")
    print(f"Rows written: {len(df)}")
    print("\nSplit x label:")
    print(df.groupby(["split", "label"]).size())
    print("\nPositive original classes:")
    print(df[df.label == 1].original_class.value_counts())
    print(f"\nManifest: {args.manifest}")


if __name__ == "__main__":
    main()

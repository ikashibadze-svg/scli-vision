#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score

from scli_vision.baselines import HOGBaseline
from scli_vision.corruptions import apply_condition
from scli_vision.io import ensure_splits, load_manifest
from scli_vision.metrics import scli_metrics
from scli_vision.model import SCLIVisionBinary


def load_image_small(path: str | Path, working_size: int) -> np.ndarray:
    """Load the ENTIRE prepared context crop at bounded resolution.

    Do not center-crop here: prepare_exdark_context.py encodes candidate geometry
    by placing the candidate in the central half of the context crop. Direct
    resize preserves that normalized geometry.
    """
    path = Path(path)
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((working_size, working_size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def stratified_limit(df: pd.DataFrame, seed: int, negative_ratio: float | None, max_per_class: int | None = None) -> pd.DataFrame:
    """Keep all positives; deterministically cap negatives and/or each class."""
    rng = np.random.default_rng(seed)
    parts = []
    for label in sorted(df.label.unique()):
        part = df[df.label == label].copy()
        if max_per_class is not None and len(part) > max_per_class:
            idx = rng.choice(len(part), size=max_per_class, replace=False)
            part = part.iloc[np.sort(idx)]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)

    if negative_ratio is not None and 1 in set(out.label.unique()) and 0 in set(out.label.unique()):
        pos = out[out.label == 1]
        neg = out[out.label == 0]
        max_neg = int(np.ceil(len(pos) * negative_ratio))
        if len(neg) > max_neg:
            idx = rng.choice(len(neg), size=max_neg, replace=False)
            neg = neg.iloc[np.sort(idx)]
        out = pd.concat([pos, neg], ignore_index=True)

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_subset(df: pd.DataFrame, working_size: int, label: str) -> list[np.ndarray]:
    images = []
    total = len(df)
    for i, p in enumerate(df["path"], 1):
        images.append(load_image_small(p, working_size))
        if i % 250 == 0 or i == total:
            print(f"[{label}] loaded {i}/{total}", flush=True)
    return images


def hog_predict_batches(hog: HOGBaseline, images: list[np.ndarray], batch_size: int) -> np.ndarray:
    out = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        out.append(hog.predict(batch))
    return np.concatenate(out) if out else np.array([], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default="outputs/run")
    ap.add_argument("--working-size", type=int, default=96,
                    help="Decode/crop each source image directly to this square size before RAM storage.")
    ap.add_argument("--negative-ratio", type=float, default=3.0,
                    help="Training negatives per positive. 3.0 keeps ExDark training manageable without discarding positives.")
    ap.add_argument("--max-val-per-class", type=int, default=500,
                    help="Legacy validation cap; retained for compatibility.")
    ap.add_argument("--robust-val-per-class", type=int, default=250,
                    help="Validation images per class used for multi-condition robust gate calibration.")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Test streaming batch size.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg["seed"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = ensure_splits(load_manifest(args.manifest), seed)
    df.to_csv(out / "resolved_manifest.csv", index=False)

    train_full = df[df.split == "train"].reset_index(drop=True)
    val_full = df[df.split == "val"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)

    if len(train_full) == 0 or len(val_full) == 0 or len(test_df) == 0:
        raise SystemExit("Need non-empty train/val/test splits.")

    # Keep all positive training examples and a representative negative sample.
    train_df = stratified_limit(train_full, seed, args.negative_ratio, None)
    val_df = stratified_limit(val_full, seed + 1, None, args.robust_val_per_class)

    print("Dataset sizes:")
    print(f"  train full={len(train_full)} -> working={len(train_df)} "
          f"(pos={int(train_df.label.sum())}, neg={int((train_df.label==0).sum())})")
    print(f"  val   full={len(val_full)} -> working={len(val_df)} "
          f"(pos={int(val_df.label.sum())}, neg={int((val_df.label==0).sum())})")
    print(f"  test  full={len(test_df)} (FULL TEST RETAINED)")
    print(f"  working image size={args.working_size}x{args.working_size}", flush=True)

    train_images = load_subset(train_df, args.working_size, "train")
    y_train = train_df.label.to_numpy(int)
    val_images = load_subset(val_df, args.working_size, "val")
    y_val = val_df.label.to_numpy(int)

    print("Fitting SCLI v8 geometry...", flush=True)
    scli = SCLIVisionBinary(
        image_size=int(cfg["image_size"]),
        n_roles=int(cfg["n_roles"]),
        min_role_spacing=int(cfg["min_role_spacing"]),
        patch_radius=int(cfg["patch_radius"]),
        min_stability=float(cfg["min_stability"]),
        max_constraints=int(cfg["max_constraints"]),
        obs_sigma_factor=float(cfg["obs_sigma_factor"]),
        obs_reference_fraction=float(cfg["obs_reference_fraction"]),
    )
    scli.fit(train_images, y_train)

    print("Fitting HOG baseline...", flush=True)
    hog = HOGBaseline(image_size=int(cfg["image_size"])).fit(train_images, y_train)

    # Training images are no longer needed.
    del train_images
    gc.collect()

    # Robust validation calibration: transformed VALIDATION images only.
    calibration_conditions = cfg.get("calibration_conditions", cfg["conditions"])
    cal_images = []
    cal_labels = []
    cal_groups = []

    print(
        f"Building robust calibration pool: {len(val_images)} validation images "
        f"x {len(calibration_conditions)} conditions...",
        flush=True,
    )
    for ci, condition in enumerate(calibration_conditions):
        rng = np.random.default_rng(seed + 5000 + ci)
        transformed = [apply_condition(im, condition, rng) for im in val_images]
        cal_images.extend(transformed)
        cal_labels.extend(y_val.tolist())
        cal_groups.extend([condition] * len(transformed))
        print(
            f"[calibration:{condition}] {len(transformed)} images",
            flush=True,
        )

    scli.calibrate(
        cal_images,
        np.asarray(cal_labels, dtype=int),
        target_known_accuracy=float(cfg["target_known_accuracy"]),
        min_known_coverage=float(cfg["min_known_coverage"]),
        groups=np.asarray(cal_groups, dtype=object),
        min_group_answer_rate=float(cfg.get("min_group_answer_rate", 0.02)),
    )

    del val_images, cal_images, cal_labels, cal_groups
    gc.collect()

    y_test = test_df.label.to_numpy(int)
    rows = []
    detail_rows = []

    for ci, condition in enumerate(cfg["conditions"]):
        print(f"\n[{condition}] processing {len(test_df)} full held-out images...", flush=True)
        rng = np.random.default_rng(seed + 1000 + ci)

        all_records = []
        all_hog_pred = []

        for start in range(0, len(test_df), args.batch_size):
            batch_df = test_df.iloc[start:start + args.batch_size]
            raw = [load_image_small(p, args.working_size) for p in batch_df.path]
            batch = [apply_condition(im, condition, rng) for im in raw]

            records = scli.predict(batch)
            hp = hog.predict(batch)
            all_records.extend(records)
            all_hog_pred.extend(hp.tolist())

            del raw, batch
            if start == 0 or (start // args.batch_size + 1) % 5 == 0 or start + args.batch_size >= len(test_df):
                done = min(start + args.batch_size, len(test_df))
                print(f"[{condition}] {done}/{len(test_df)}", flush=True)
            gc.collect()

        hog_pred = np.asarray(all_hog_pred, dtype=int)
        metrics = scli_metrics(y_test, all_records)
        metrics["condition"] = condition
        metrics["hog_accuracy"] = float(accuracy_score(y_test, hog_pred))
        metrics["hog_false_clear_rate"] = (
            float(np.mean(hog_pred[y_test == 1] == 0)) if np.any(y_test == 1) else np.nan
        )
        rows.append(metrics)

        for path, y, rec, hp in zip(test_df.path, y_test, all_records, hog_pred):
            detail_rows.append({
                "condition": condition,
                "path": path,
                "truth": int(y),
                **rec,
                "false_clear": bool(y == 1 and rec["known"] and rec["label"] == 0),
                "hog_label": int(hp),
                "hog_false_clear": bool(y == 1 and hp == 0),
            })

        # Write progressively so an interrupted later condition does not lose completed work.
        pd.DataFrame(rows).to_csv(out / "results.partial.csv", index=False)
        pd.DataFrame(detail_rows).to_csv(out / "details.partial.csv", index=False)

    results = pd.DataFrame(rows)
    details = pd.DataFrame(detail_rows)
    results.to_csv(out / "results.csv", index=False)
    details.to_csv(out / "details.csv", index=False)
    details[details.false_clear].to_csv(out / "false_clear_cases.csv", index=False)

    meta = {
        "model": scli.describe(),
        "train_full_n": len(train_full),
        "train_working_n": len(train_df),
        "val_full_n": len(val_full),
        "val_working_n": len(val_df),
        "test_n": len(test_df),
        "positive_train_n": int(y_train.sum()),
        "working_size": args.working_size,
        "negative_ratio": args.negative_ratio,
        "batch_size": args.batch_size,
        "robust_val_per_class": args.robust_val_per_class,
        "calibration_conditions": calibration_conditions,
        "config": cfg,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))

    x = np.arange(len(results))
    plt.figure(figsize=(11, 5.5))
    plt.plot(x, results["hog_accuracy"], marker="o", label="HOG baseline accuracy")
    plt.plot(x, results["scli_raw_accuracy"], marker="o", label="SCLI raw accuracy")
    plt.plot(x, results["known_accuracy"], marker="o", label="SCLI accuracy when KNOWN")
    plt.xticks(x, results["condition"], rotation=30, ha="right")
    plt.ylim(0, 1.02)
    plt.ylabel("accuracy")
    plt.title("SCLI Vision v8 geometry: ExDark robust-certification benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5.5))
    plt.plot(x, results["known_coverage"], marker="o", label="KNOWN coverage")
    plt.plot(x, results["underdetermined_rate"], marker="o", label="UNDERDETERMINED")
    plt.plot(x, results["false_clear_rate"], marker="o", label="SCLI false-clear")
    plt.plot(x, results["hog_false_clear_rate"], marker="o", label="HOG false-clear")
    plt.xticks(x, results["condition"], rotation=30, ha="right")
    plt.ylim(0, 1.02)
    plt.ylabel("fraction")
    plt.title("SCLI Vision v8 geometry safety metrics on full ExDark held-out test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "safety_metrics.png", dpi=180)
    plt.close()

    # Remove partials after successful completion.
    for p in [out / "results.partial.csv", out / "details.partial.csv"]:
        if p.exists():
            p.unlink()

    print("\n" + results.to_string(index=False))
    print("\nSCLI:", json.dumps(scli.describe(), indent=2))
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score

from scli_vision.baselines import HOGBaseline
from scli_vision.corruptions import apply_condition
from scli_vision.io import ensure_splits, load_image, load_manifest
from scli_vision.metrics import scli_metrics
from scli_vision.model import SCLIVisionBinary


def load_subset(df: pd.DataFrame) -> list[np.ndarray]:
    return [load_image(p) for p in df["path"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default="outputs/run")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = ensure_splits(load_manifest(args.manifest), int(cfg["seed"]))
    df.to_csv(out / "resolved_manifest.csv", index=False)
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise SystemExit("Need non-empty train/val/test splits.")

    train_images = load_subset(train_df)
    val_images = load_subset(val_df)
    test_images = load_subset(test_df)
    y_train = train_df.label.to_numpy(int)
    y_val = val_df.label.to_numpy(int)
    y_test = test_df.label.to_numpy(int)

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
    gate = scli.calibrate(
        val_images,
        y_val,
        target_known_accuracy=float(cfg["target_known_accuracy"]),
        min_known_coverage=float(cfg["min_known_coverage"]),
    )

    hog = HOGBaseline(image_size=int(cfg["image_size"])).fit(train_images, y_train)

    rows = []
    detail_rows = []
    for ci, condition in enumerate(cfg["conditions"]):
        rng = np.random.default_rng(int(cfg["seed"]) + 1000 + ci)
        images = [apply_condition(im, condition, rng) for im in test_images]

        records = scli.predict(images)
        metrics = scli_metrics(y_test, records)
        hog_pred = hog.predict(images)
        metrics["condition"] = condition
        metrics["hog_accuracy"] = float(accuracy_score(y_test, hog_pred))
        metrics["hog_false_clear_rate"] = float(np.mean(hog_pred[y_test == 1] == 0)) if np.any(y_test == 1) else np.nan
        rows.append(metrics)

        for path, y, rec in zip(test_df.path, y_test, records):
            detail_rows.append({
                "condition": condition,
                "path": path,
                "truth": int(y),
                **rec,
                "false_clear": bool(y == 1 and rec["known"] and rec["label"] == 0),
            })

    results = pd.DataFrame(rows)
    details = pd.DataFrame(detail_rows)
    results.to_csv(out / "results.csv", index=False)
    details.to_csv(out / "details.csv", index=False)
    details[details.false_clear].to_csv(out / "false_clear_cases.csv", index=False)

    meta = {
        "model": scli.describe(),
        "train_n": len(train_df),
        "val_n": len(val_df),
        "test_n": len(test_df),
        "positive_train_n": int(y_train.sum()),
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
    plt.title("SCLI Vision: real-photo robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5.5))
    plt.plot(x, results["known_coverage"], marker="o", label="KNOWN coverage")
    plt.plot(x, results["underdetermined_rate"], marker="o", label="UNDERDETERMINED")
    plt.plot(x, results["false_clear_rate"], marker="o", label="SCLI false-clear")
    plt.xticks(x, results["condition"], rotation=30, ha="right")
    plt.ylim(0, 1.02)
    plt.ylabel("fraction")
    plt.title("Epistemic behavior and false-clear rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "safety_metrics.png", dpi=180)
    plt.close()

    print(results.to_string(index=False))
    print("\nSCLI:", json.dumps(scli.describe(), indent=2))
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()

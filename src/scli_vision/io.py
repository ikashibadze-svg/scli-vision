from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    required = {"path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    df = df.copy()
    base = path.parent
    df["path"] = df["path"].map(lambda p: str((base / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(Path(str(p)).resolve()))
    df["label"] = df["label"].astype(int)
    bad = [p for p in df["path"] if not Path(p).exists()]
    if bad:
        raise FileNotFoundError(f"{len(bad)} manifest images do not exist. First missing: {bad[0]}")
    return df


def ensure_splits(df: pd.DataFrame, seed: int = 20261003) -> pd.DataFrame:
    if "split" in df.columns and set(df["split"].dropna().str.lower()) >= {"train", "val", "test"}:
        out = df.copy()
        out["split"] = out["split"].str.lower()
        return out

    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.40, stratify=df["label"].to_numpy(), random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=df.iloc[temp_idx]["label"].to_numpy(),
        random_state=seed,
    )
    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    out = df.copy()
    out["split"] = split
    return out


def load_images(paths: Iterable[str | Path]) -> list[np.ndarray]:
    return [load_image(p) for p in paths]

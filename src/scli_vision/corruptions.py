from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_condition(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    x = image.astype(np.float32).copy()
    name = condition.lower()

    if name == "normal":
        return x
    if name == "contrast_10":
        m = x.mean(axis=(0, 1), keepdims=True)
        x = m + 0.10 * (x - m) + rng.normal(0, 0.008, x.shape)
    elif name == "contrast_03":
        m = x.mean(axis=(0, 1), keepdims=True)
        x = m + 0.03 * (x - m) + rng.normal(0, 0.008, x.shape)
    elif name == "dark_15":
        x = 0.15 * x + rng.normal(0, 0.010, x.shape)
    elif name == "blur":
        x = gaussian_filter(x, sigma=(1.6, 1.6, 0))
    elif name == "noise_08":
        x = x + rng.normal(0, 0.08, x.shape)
    elif name == "occlusion_25":
        h, w, _ = x.shape
        side = max(2, int(round(np.sqrt(0.25) * min(h, w))))
        y0 = int(rng.integers(0, max(1, h - side + 1)))
        x0 = int(rng.integers(0, max(1, w - side + 1)))
        x[y0:y0 + side, x0:x0 + side] = x.mean(axis=(0, 1))
    elif name == "lowlight_combo":
        m = x.mean(axis=(0, 1), keepdims=True)
        x = m + 0.08 * (x - m)
        x = 0.18 * x + rng.normal(0, 0.012, x.shape)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return np.clip(np.round(x * 255.0) / 255.0, 0.0, 1.0).astype(np.float32)

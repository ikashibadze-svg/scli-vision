#!/usr/bin/env python3
"""Small no-download smoke test using real images bundled with scikit-image."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from skimage import data, transform

OUT = Path("data/smoke")
POS = OUT / "positive"
NEG = OUT / "negative"
POS.mkdir(parents=True, exist_ok=True)
NEG.mkdir(parents=True, exist_ok=True)

# Create many distinct real face/non-face images from the LFW subset bundled with skimage.
lfw = data.lfw_subset()
for i, arr in enumerate(lfw[:80]):
    p = POS / f"face_{i:03d}.png"
    Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255)).save(p)
for i, arr in enumerate(lfw[100:180]):
    p = NEG / f"nonface_{i:03d}.png"
    Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255)).save(p)

rows = [{"path": str(p.resolve()), "label": 1} for p in sorted(POS.glob("*.png"))]
rows += [{"path": str(p.resolve()), "label": 0} for p in sorted(NEG.glob("*.png"))]
pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
print(OUT / "manifest.csv")

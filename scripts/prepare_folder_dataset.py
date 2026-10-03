#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from scli_vision.io import list_images


def main():
    ap = argparse.ArgumentParser(description="Build a binary manifest from positive/negative image folders.")
    ap.add_argument("--positive-dir", required=True)
    ap.add_argument("--negative-dir", required=True)
    ap.add_argument("--output", default="data/manifest.csv")
    args = ap.parse_args()

    rows = []
    for label, root in [(1, args.positive_dir), (0, args.negative_dir)]:
        for p in list_images(root):
            rows.append({"path": str(p.resolve()), "label": label, "source": Path(root).name})
    if not rows:
        raise SystemExit("No images found.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

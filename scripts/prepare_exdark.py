#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

CLASS_NAMES = {
    1: "Bicycle", 2: "Boat", 3: "Bottle", 4: "Bus", 5: "Car", 6: "Cat",
    7: "Chair", 8: "Cup", 9: "Dog", 10: "Motorbike", 11: "People", 12: "Table"
}
SPLIT_NAMES = {1: "train", 2: "val", 3: "test"}


def main():
    ap = argparse.ArgumentParser(description="Create a target-vs-rest manifest from ExDark imageclasslist.txt")
    ap.add_argument("--image-root", required=True, help="Directory containing ExDark class folders")
    ap.add_argument("--image-class-list", required=True, help="Groundtruth/imageclasslist.txt")
    ap.add_argument("--target-class", default="Cat", choices=list(CLASS_NAMES.values()))
    ap.add_argument("--output", default="data/exdark_target_manifest.csv")
    args = ap.parse_args()

    image_root = Path(args.image_root)
    lines = Path(args.image_class_list).read_text(errors="ignore").splitlines()
    rows = []
    missing = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        filename = parts[0]
        try:
            class_id = int(parts[1]); lighting = int(parts[2]); split_id = int(parts[4])
        except ValueError:
            continue
        cname = CLASS_NAMES[class_id]
        p = image_root / cname / filename
        if not p.exists():
            # Some copies differ in capitalization or layout. Search by name as fallback.
            candidates = list(image_root.rglob(filename))
            if candidates:
                p = candidates[0]
            else:
                missing += 1
                continue
        rows.append({
            "path": str(p.resolve()),
            "label": int(cname == args.target_class),
            "class_name": cname,
            "lighting_type": lighting,
            "split": SPLIT_NAMES.get(split_id, "train"),
            "source": "ExDark",
        })

    if not rows:
        raise SystemExit("No ExDark images matched. Check --image-root and --image-class-list.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} images to {out}; missing files: {missing}")
    print(f"Positive class: {args.target_class}; positives: {sum(r['label'] for r in rows)}")


if __name__ == "__main__":
    main()

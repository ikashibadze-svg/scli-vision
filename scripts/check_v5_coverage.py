#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from scli_vision.model import SCLIVisionBinary


def pattern(n=64):
    y, x = np.mgrid[0:n, 0:n]
    z = 0.40 + 0.015*np.sin(x/4.0)*np.cos(y/5.0)
    obj = (((x-32)/18.0)**2 + ((y-32)/12.0)**2) <= 1.0
    z[obj] += 0.16
    z[24:27, 21:43] -= 0.12
    z[37:40, 24:41] += 0.10
    return np.clip(z, 0, 1).astype(np.float32)


def transform_contrast(z, a, b=0.25):
    return np.clip(a*z+b, 0, 1).astype(np.float32)


def compare(model, a, b, name):
    va, sa, ca, names = model._structural_observables(a)
    vb, sb, cb, _ = model._structural_observables(b)

    valid = (sa > 1e-8) & (sb > 1e-8)
    delta = np.abs(va[valid]-vb[valid])
    ratio = sb[valid]/(sa[valid]+1e-12)

    print(name)
    print(f"  observables: {valid.sum()}/{len(valid)}")
    print(f"  identity median |Δ|: {np.median(delta):.6f}")
    print(f"  identity p95 |Δ|:    {np.percentile(delta,95):.6f}")
    print(f"  raw support median ratio: {np.median(ratio):.4f}")
    print(f"  median noise coefficient: {np.median(cb[valid]):.5f}")
    return np.median(delta), np.percentile(delta,95), np.median(ratio)


def main():
    m = SCLIVisionBinary(image_size=64)
    base = pattern()
    d10, p10, r10 = compare(m, base, transform_contrast(base, .10), "contrast 10%")
    d03, p03, r03 = compare(m, base, transform_contrast(base, .03), "contrast 3%")

    ok = (
        d10 < 0.06
        and d03 < 0.10
        and 0.05 < r10 < 0.16
        and 0.01 < r03 < 0.06
    )
    print("\nPASS" if ok else "\nCHECK")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

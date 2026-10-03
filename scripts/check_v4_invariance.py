#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from scli_vision.model import SCLIVisionBinary


def make_pattern(n=64):
    y, x = np.mgrid[0:n, 0:n]
    z = np.full((n, n), 0.42, dtype=np.float32)

    # A deliberately structured "object": ellipse + internal edges.
    ellipse = (((x - 32) / 17.0) ** 2 + ((y - 32) / 12.0) ** 2) <= 1
    z[ellipse] = 0.58
    z[24:27, 21:43] = 0.30
    z[36:39, 24:40] = 0.72

    # weak background texture
    z += 0.018 * np.sin(x / 4.0) * np.cos(y / 5.0)
    return np.clip(z, 0, 1)


def affine_contrast(z, a, b=0.23):
    return np.clip(a * z + b, 0, 1).astype(np.float32)


def summarize(model, base, transformed, label):
    v0, s0, _, names = model._structural_observables(base)
    v1, s1, _, _ = model._structural_observables(transformed)

    # Ignore coordinates whose support is effectively zero in either image.
    valid = (s0 > 1e-7) & (s1 > 1e-7)
    identity_delta = np.abs(v1[valid] - v0[valid])
    support_ratio = s1[valid] / (s0[valid] + 1e-12)

    print(f"\n{label}")
    print(f"  valid observables: {valid.sum()}/{len(valid)}")
    print(f"  identity median |Δ|: {np.median(identity_delta):.6f}")
    print(f"  identity p95 |Δ|:    {np.percentile(identity_delta,95):.6f}")
    print(f"  raw support median ratio: {np.median(support_ratio):.4f}")
    print(f"  raw support p10/p90: {np.percentile(support_ratio,10):.4f}/{np.percentile(support_ratio,90):.4f}")

    return float(np.median(identity_delta)), float(np.median(support_ratio))


def main():
    model = SCLIVisionBinary(image_size=64)
    base = make_pattern()

    d10, r10 = summarize(
        model, base, affine_contrast(base, 0.10), "10% positive contrast transform"
    )
    d03, r03 = summarize(
        model, base, affine_contrast(base, 0.03), "3% positive contrast transform"
    )

    # v4 design target:
    # identity should move little while raw support approximately scales with contrast.
    passed = (
        d10 < 0.08
        and d03 < 0.12
        and 0.04 < r10 < 0.20
        and 0.005 < r03 < 0.08
    )

    print("\nPASS" if passed else "\nCHECK FAILED")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

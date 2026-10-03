from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage import transform


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return (0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]).astype(np.float32)


@dataclass
class Gate:
    positive_min: float
    negative_max: float
    coverage_min: float
    validation_coverage: float
    validation_known_accuracy: float


class SCLIVisionBinary:
    """Binary observable-constraint identity model.

    label=1 is the target identity (e.g. cat/object-present).
    label=0 is the negative identity.

    The model:
      1. discovers persistent structural roles from positive training images;
      2. builds pairwise role-intensity relations;
      3. keeps relations stable on positives and discriminative vs negatives;
      4. lets a relation vote only when its raw measured magnitude clears
         a raw-unit observability threshold;
      5. can return UNDERDETERMINED instead of forcing a label.
    """

    def __init__(
        self,
        image_size: int = 64,
        n_roles: int = 16,
        min_role_spacing: int = 5,
        patch_radius: int = 3,
        min_stability: float = 0.72,
        max_constraints: int = 220,
        obs_sigma_factor: float = 2.0,
        obs_reference_fraction: float = 0.08,
    ):
        self.image_size = image_size
        self.n_roles = n_roles
        self.min_role_spacing = min_role_spacing
        self.patch_radius = patch_radius
        self.min_stability = min_stability
        self.max_constraints = max_constraints
        self.obs_sigma_factor = obs_sigma_factor
        self.obs_reference_fraction = obs_reference_fraction
        self.roles: np.ndarray | None = None
        self.pairs: np.ndarray | None = None
        self.selected: np.ndarray | None = None
        self.expected_sign: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.reference_magnitude: np.ndarray | None = None
        self.gate: Gate | None = None

    def _canonicalize(self, image: np.ndarray) -> np.ndarray:
        g = transform.resize(_gray(image), (self.image_size, self.image_size), anti_aliasing=True).astype(np.float32)
        q = gaussian_filter(g, 0.8)
        gy, gx = np.gradient(q)
        mag = np.sqrt(gx * gx + gy * gy)
        threshold = np.percentile(mag, 60)
        yy, xx = np.nonzero(mag >= threshold)
        if len(xx) < 8:
            return g
        w = mag[yy, xx] + 1e-6
        cy = float((yy * w).sum() / w.sum())
        cx = float((xx * w).sum() / w.sum())
        sy = float(np.sqrt((((yy - cy) ** 2) * w).sum() / w.sum()) + 1e-6)
        sx = float(np.sqrt((((xx - cx) ** 2) * w).sum() / w.sum()) + 1e-6)
        half_h = int(np.clip(2.1 * sy, self.image_size * 0.30, self.image_size / 2))
        half_w = int(np.clip(2.1 * sx, self.image_size * 0.30, self.image_size / 2))
        y0, y1 = max(0, int(cy - half_h)), min(self.image_size, int(cy + half_h))
        x0, x1 = max(0, int(cx - half_w)), min(self.image_size, int(cx + half_w))
        crop = g[y0:y1, x0:x1]
        if min(crop.shape) < 8:
            crop = g
        return transform.resize(crop, (self.image_size, self.image_size), anti_aliasing=True).astype(np.float32)

    def _discover_roles(self, positive_images: list[np.ndarray]) -> np.ndarray:
        maps = []
        for image in positive_images:
            z = self._canonicalize(image)
            q = gaussian_filter(z, 0.7)
            gy, gx = np.gradient(q)
            mag = np.sqrt(gx * gx + gy * gy)
            maps.append(mag / (np.percentile(mag, 95) + 1e-6))
        consensus = np.mean(maps, axis=0)
        mx = maximum_filter(consensus, size=max(3, self.min_role_spacing), mode="reflect")
        candidate = (consensus >= mx - 1e-9) & (consensus >= np.percentile(consensus, 60))
        yy, xx = np.nonzero(candidate)
        order = np.argsort(consensus[yy, xx])[::-1]
        roles: list[tuple[int, int]] = []
        for oi in order:
            y, x = int(yy[oi]), int(xx[oi])
            if all((y - ry) ** 2 + (x - rx) ** 2 >= self.min_role_spacing ** 2 for ry, rx in roles):
                roles.append((y, x))
            if len(roles) >= self.n_roles:
                break
        if len(roles) < 4:
            raise RuntimeError("Could not discover enough stable roles. Add more/better positive training images.")
        return np.asarray(roles, dtype=int)

    def _role_means(self, image: np.ndarray) -> np.ndarray:
        if self.roles is None:
            raise RuntimeError("Model is not fitted.")
        z = gaussian_filter(self._canonicalize(image), 0.6)
        out = []
        r = self.patch_radius
        for y, x in self.roles:
            ys = slice(max(0, y - r), min(self.image_size, y + r + 1))
            xs = slice(max(0, x - r), min(self.image_size, x + r + 1))
            out.append(float(z[ys, xs].mean()))
        return np.asarray(out, dtype=np.float32)

    def _constraint_values(self, image: np.ndarray) -> np.ndarray:
        means = self._role_means(image)
        return means[self.pairs[:, 0]] - means[self.pairs[:, 1]]

    def _noise_sigma(self, image: np.ndarray) -> float:
        z = self._canonicalize(image)
        low = gaussian_filter(z, 0.8)
        residual = z - low
        mad = np.median(np.abs(residual - np.median(residual)))
        return float(mad / 0.6745 + 1e-5)

    def fit(self, images: list[np.ndarray], labels: np.ndarray) -> "SCLIVisionBinary":
        labels = np.asarray(labels, dtype=int)
        pos = [im for im, y in zip(images, labels) if y == 1]
        neg = [im for im, y in zip(images, labels) if y == 0]
        if len(pos) < 8 or len(neg) < 8:
            raise ValueError("Need at least 8 positive and 8 negative training images.")

        self.roles = self._discover_roles(pos)
        self.pairs = np.asarray(list(combinations(range(len(self.roles)), 2)), dtype=int)

        pos_vals = np.asarray([self._constraint_values(im) for im in pos], dtype=np.float32)
        neg_vals = np.asarray([self._constraint_values(im) for im in neg], dtype=np.float32)

        pos_rate = (pos_vals > 0).mean(axis=0)
        neg_rate = (neg_vals > 0).mean(axis=0)
        stability = np.maximum(pos_rate, 1.0 - pos_rate)
        discrimination = np.abs(pos_rate - neg_rate)
        median_mag = np.median(np.abs(pos_vals), axis=0)
        mag_norm = median_mag / (np.median(median_mag) + 1e-6)
        weight = ((stability - 0.5) / 0.5).clip(0, 1) ** 2 * discrimination * np.sqrt(np.clip(mag_norm, 0, 4) + 1e-6)

        eligible = np.where(stability >= self.min_stability)[0]
        if len(eligible) == 0:
            raise RuntimeError("No stable relational constraints found. Lower min_stability or add data.")
        eligible = eligible[np.argsort(weight[eligible])]
        selected = eligible[-min(self.max_constraints, len(eligible)):]

        self.selected = selected
        self.expected_sign = pos_rate[selected] >= 0.5
        self.weights = weight[selected].astype(np.float32) + 1e-6
        self.reference_magnitude = median_mag[selected].astype(np.float32) + 1e-6
        return self

    def score_one(self, image: np.ndarray) -> dict[str, Any]:
        if self.selected is None:
            raise RuntimeError("Model is not fitted.")
        vals = self._constraint_values(image)[self.selected]
        sigma = self._noise_sigma(image)
        n = (2 * self.patch_radius + 1) ** 2
        sigma_relation = np.sqrt(2.0) * sigma / np.sqrt(n)
        threshold = np.maximum(
            self.obs_sigma_factor * sigma_relation,
            self.obs_reference_fraction * self.reference_magnitude,
        )
        observable = np.abs(vals) >= threshold
        if not observable.any():
            return {"score": 0.5, "coverage": 0.0, "noise_sigma": sigma, "n_observable": 0}

        match = ((vals > 0)[observable] == self.expected_sign[observable]).astype(np.float32)
        w = self.weights[observable]
        score = float((w * match).sum() / (w.sum() + 1e-12))
        coverage = float(w.sum() / (self.weights.sum() + 1e-12))
        return {"score": score, "coverage": coverage, "noise_sigma": sigma, "n_observable": int(observable.sum())}

    def calibrate(
        self,
        images: list[np.ndarray],
        labels: np.ndarray,
        target_known_accuracy: float = 0.90,
        min_known_coverage: float = 0.05,
    ) -> Gate:
        labels = np.asarray(labels, dtype=int)
        rows = [self.score_one(im) for im in images]
        score = np.asarray([r["score"] for r in rows])
        coverage = np.asarray([r["coverage"] for r in rows])

        best = None
        for positive_min in np.arange(0.55, 0.96, 0.02):
            for negative_max in np.arange(0.20, 0.66, 0.02):
                if negative_max >= positive_min:
                    continue
                for coverage_min in np.arange(0.05, 0.91, 0.05):
                    known_pos = (score >= positive_min) & (coverage >= coverage_min)
                    known_neg = (score <= negative_max) & (coverage >= coverage_min)
                    known = known_pos | known_neg
                    cov = float(known.mean())
                    if cov < min_known_coverage:
                        continue
                    pred = np.where(known_pos, 1, 0)
                    acc = float((pred[known] == labels[known]).mean())
                    if acc >= target_known_accuracy and (best is None or cov > best.validation_coverage):
                        best = Gate(float(positive_min), float(negative_max), float(coverage_min), cov, acc)

        if best is None:
            raise RuntimeError(
                "No epistemic gate met the requested target_known_accuracy. "
                "Use more validation data or lower the target for exploration."
            )
        self.gate = best
        return best

    def predict_one(self, image: np.ndarray) -> dict[str, Any]:
        result = self.score_one(image)
        result["raw_label"] = int(result["score"] >= 0.5)
        result["known"] = False
        result["label"] = None
        if self.gate is None:
            return result
        if result["coverage"] >= self.gate.coverage_min:
            if result["score"] >= self.gate.positive_min:
                result["known"] = True
                result["label"] = 1
            elif result["score"] <= self.gate.negative_max:
                result["known"] = True
                result["label"] = 0
        return result

    def predict(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        return [self.predict_one(im) for im in images]

    def describe(self) -> dict[str, Any]:
        return {
            "n_roles": 0 if self.roles is None else int(len(self.roles)),
            "n_constraints": 0 if self.selected is None else int(len(self.selected)),
            "gate": None if self.gate is None else asdict(self.gate),
        }

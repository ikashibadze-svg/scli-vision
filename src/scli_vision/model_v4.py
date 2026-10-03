from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, laplace
from skimage import transform


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    ).astype(np.float32)


@dataclass
class Gate:
    positive_min: float
    negative_max: float
    coverage_min: float
    validation_coverage: float
    validation_known_accuracy: float
    certified: bool = True
    target_known_accuracy: float = 0.90


class SCLIVisionBinary:
    """SCLI Vision v4: invariant identity + raw observability.

    Core separation:
      IDENTITY      = normalized spatial/structural relations.
      OBSERVABILITY = raw signal amplitude compared with raw noise.

    Therefore an affine photometric transform I' = aI+b (a>0) should leave
    the identity coordinates approximately unchanged while reducing raw
    support as a becomes small. When raw support is lost, the model should
    move toward UNDERDETERMINED instead of moving the identity toward
    BACKGROUND_CLEAR.

    The model first attempts the legacy role-relation language. If that
    representation is empirically weak, it expands to the invariant
    structural bank defined below.
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
        min_effect: float = 0.10,
        min_constraints: int = 32,
        expansion_effect_p95: float = 0.12,
    ):
        self.image_size = image_size
        self.n_roles = n_roles
        self.min_role_spacing = min_role_spacing
        self.patch_radius = patch_radius
        self.min_stability = min_stability
        self.max_constraints = max_constraints
        self.obs_sigma_factor = obs_sigma_factor
        self.obs_reference_fraction = obs_reference_fraction
        self.min_effect = min_effect
        self.min_constraints = min_constraints
        self.expansion_effect_p95 = expansion_effect_p95

        self.roles: np.ndarray | None = None
        self.pairs: np.ndarray | None = None
        self.selected: np.ndarray | None = None

        self.pos_center: np.ndarray | None = None
        self.neg_center: np.ndarray | None = None
        self.pos_scale: np.ndarray | None = None
        self.neg_scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None

        self.reference_support: np.ndarray | None = None
        self.noise_coeff: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.representation_mode = "unfitted"
        self.fit_diagnostics: dict[str, Any] = {}
        self.gate: Gate | None = None

    # ------------------------------------------------------------------
    # Image / noise helpers
    # ------------------------------------------------------------------
    def _resize(self, image: np.ndarray) -> np.ndarray:
        return transform.resize(
            _gray(image),
            (self.image_size, self.image_size),
            anti_aliasing=True,
        ).astype(np.float32)

    def _canonicalize(self, image: np.ndarray) -> np.ndarray:
        g = self._resize(image)
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

        return transform.resize(
            crop,
            (self.image_size, self.image_size),
            anti_aliasing=True,
        ).astype(np.float32)

    @staticmethod
    def _noise_sigma_from_gray(z: np.ndarray) -> float:
        low = gaussian_filter(z, 0.8)
        residual = z - low
        mad = np.median(np.abs(residual - np.median(residual)))
        return float(mad / 0.6745 + 1e-5)

    def _noise_sigma(self, image: np.ndarray) -> float:
        return self._noise_sigma_from_gray(self._resize(image))

    # ------------------------------------------------------------------
    # Initial representation: role intensity relations
    # ------------------------------------------------------------------
    def _discover_roles(self, positive_images: list[np.ndarray]) -> np.ndarray:
        maps = []
        for image in positive_images:
            z = self._canonicalize(image)
            q = gaussian_filter(z, 0.7)
            gy, gx = np.gradient(q)
            mag = np.sqrt(gx * gx + gy * gy)
            maps.append(mag / (np.percentile(mag, 95) + 1e-6))

        consensus = np.mean(maps, axis=0)
        mx = maximum_filter(
            consensus,
            size=max(3, self.min_role_spacing),
            mode="reflect",
        )
        candidate = (
            (consensus >= mx - 1e-9)
            & (consensus >= np.percentile(consensus, 60))
        )
        yy, xx = np.nonzero(candidate)
        order = np.argsort(consensus[yy, xx])[::-1]

        roles: list[tuple[int, int]] = []
        for oi in order:
            y, x = int(yy[oi]), int(xx[oi])
            if all(
                (y - ry) ** 2 + (x - rx) ** 2 >= self.min_role_spacing ** 2
                for ry, rx in roles
            ):
                roles.append((y, x))
            if len(roles) >= self.n_roles:
                break

        if len(roles) < 4:
            raise RuntimeError("Could not discover enough stable roles.")
        return np.asarray(roles, dtype=int)

    def _role_means(self, image: np.ndarray) -> np.ndarray:
        if self.roles is None:
            raise RuntimeError("Model is not fitted.")

        z = gaussian_filter(self._canonicalize(image), 0.6)
        r = self.patch_radius
        out = []
        for y, x in self.roles:
            ys = slice(max(0, y-r), min(self.image_size, y+r+1))
            xs = slice(max(0, x-r), min(self.image_size, x+r+1))
            out.append(float(z[ys, xs].mean()))
        return np.asarray(out, dtype=np.float32)

    def _relational_observables(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        means = self._role_means(image)
        raw = means[self.pairs[:, 0]] - means[self.pairs[:, 1]]

        # Identity is normalized by robust global intensity spread.
        z = self._canonicalize(image)
        spread = float(np.percentile(z, 90) - np.percentile(z, 10)) + 1e-6
        identity = raw / spread

        support = np.abs(raw).astype(np.float32)
        n = (2*self.patch_radius + 1) ** 2
        coeff = np.full(
            len(raw),
            np.sqrt(2.0) / np.sqrt(n),
            dtype=np.float32,
        )
        names = [f"role_ratio:{a}-{b}" for a, b in self.pairs]
        return (
            identity.astype(np.float32),
            support,
            coeff,
            names,
        )

    # ------------------------------------------------------------------
    # Expanded representation: invariant structural bank
    # ------------------------------------------------------------------
    def _structural_observables(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Return identity values and separate raw observability supports.

        Every identity coordinate below is dimensionless / contrast-normalized
        (or geometric). Raw amplitude is NEVER used as identity. The matching
        raw support is returned separately and used only by score_one() to
        decide whether that identity coordinate is physically observable.
        """
        z = self._resize(image)
        q = gaussian_filter(z, 0.6)
        q2 = gaussian_filter(z, 1.8)

        gy, gx = np.gradient(q)
        grad = np.sqrt(gx*gx + gy*gy)
        lap_abs = np.abs(laplace(q))
        dog_abs = np.abs(q - q2)
        ori = (np.arctan2(gy, gx) + 2*np.pi) % (2*np.pi)

        # Global raw amplitudes: OBSERVABILITY references only.
        global_std = float(q.std()) + 1e-8
        global_spread = float(np.percentile(q, 90) - np.percentile(q, 10)) + 1e-8
        global_grad = float(grad.mean()) + 1e-8
        global_lap = float(lap_abs.mean()) + 1e-8
        global_dog = float(dog_abs.mean()) + 1e-8
        global_median = float(np.median(q))

        values: list[float] = []
        support: list[float] = []
        coeff: list[float] = []
        names: list[str] = []

        def add(name: str, identity: float, raw_support: float, noise_c: float):
            values.append(float(identity))
            support.append(float(max(raw_support, 0.0)))
            coeff.append(float(max(noise_c, 1e-6)))
            names.append(name)

        grid = 4
        step = self.image_size // grid
        total_grad_sum = float(grad.sum()) + 1e-8
        total_lap_sum = float(lap_abs.sum()) + 1e-8
        total_dog_sum = float(dog_abs.sum()) + 1e-8

        for ry in range(grid):
            for rx in range(grid):
                y0 = ry*step
                y1 = self.image_size if ry == grid-1 else (ry+1)*step
                x0 = rx*step
                x1 = self.image_size if rx == grid-1 else (rx+1)*step
                sl = (slice(y0, y1), slice(x0, x1))

                p = q[sl]
                pg = grad[sl]
                pl = lap_abs[sl]
                pd = dog_abs[sl]
                po = ori[sl]

                pmean = float(p.mean())
                pstd = float(p.std())
                gmean = float(pg.mean())
                lmean = float(pl.mean())
                dmean = float(pd.mean())

                # Local intensity role relative to global image scale.
                raw_mean_delta = abs(pmean - global_median)
                add(
                    f"cell{ry}{rx}:mean_role",
                    (pmean - global_median) / global_spread,
                    raw_mean_delta,
                    1.0,
                )

                # Local contrast/edge/curvature ratios.
                add(
                    f"cell{ry}{rx}:std_ratio",
                    pstd / global_std,
                    pstd,
                    1.0,
                )
                add(
                    f"cell{ry}{rx}:grad_ratio",
                    gmean / global_grad,
                    gmean,
                    np.sqrt(2.0),
                )
                add(
                    f"cell{ry}{rx}:lap_ratio",
                    lmean / global_lap,
                    lmean,
                    2.5,
                )
                add(
                    f"cell{ry}{rx}:dog_ratio",
                    dmean / global_dog,
                    dmean,
                    1.0,
                )

                # Fraction of global structural energy carried by this cell.
                add(
                    f"cell{ry}{rx}:grad_share",
                    float(pg.sum()) / total_grad_sum * (grid*grid),
                    gmean,
                    np.sqrt(2.0),
                )
                add(
                    f"cell{ry}{rx}:lap_share",
                    float(pl.sum()) / total_lap_sum * (grid*grid),
                    lmean,
                    2.5,
                )
                add(
                    f"cell{ry}{rx}:dog_share",
                    float(pd.sum()) / total_dog_sum * (grid*grid),
                    dmean,
                    1.0,
                )

                # Orientation coherence and orientation distribution are
                # already invariant to positive contrast scaling.
                denom = float(pg.sum()) + 1e-8
                cx = float((np.cos(po) * pg).sum())
                cy = float((np.sin(po) * pg).sum())
                coherence = float(np.sqrt(cx*cx + cy*cy) / denom)
                add(
                    f"cell{ry}{rx}:ori_coherence",
                    coherence,
                    gmean,
                    np.sqrt(2.0),
                )

                for b in range(4):
                    lo = b * (2*np.pi/4)
                    hi = (b+1) * (2*np.pi/4)
                    mask = (po >= lo) & (po < hi)
                    h = float(pg[mask].sum() / denom)
                    add(
                        f"cell{ry}{rx}:ori{b}",
                        h,
                        gmean,
                        np.sqrt(2.0),
                    )

        # Center vs border RATIOS only; raw support separate.
        a = self.image_size // 4
        b = self.image_size - a
        center = np.zeros_like(z, dtype=bool)
        center[a:b, a:b] = True
        border = ~center

        for nm, arr, nc in [
            ("grad", grad, np.sqrt(2.0)),
            ("lap", lap_abs, 2.5),
            ("dog", dog_abs, 1.0),
        ]:
            c = float(arr[center].mean())
            o = float(arr[border].mean())
            raw = max(c, o)
            add(
                f"global:center_border_{nm}_logratio",
                float(np.log((c + 1e-6) / (o + 1e-6))),
                raw,
                nc,
            )
            add(
                f"global:center_{nm}_share",
                c / (c + o + 1e-8),
                raw,
                nc,
            )

        # Pure geometry of edge-mass distribution.
        gm = grad + 1e-12
        total = float(gm.sum())
        yy, xx = np.mgrid[0:self.image_size, 0:self.image_size]
        norm = max(self.image_size - 1, 1)
        ex = float((xx * gm).sum() / total) / norm
        ey = float((yy * gm).sum() / total) / norm
        add("global:edge_centroid_x", ex, global_grad, np.sqrt(2.0))
        add("global:edge_centroid_y", ey, global_grad, np.sqrt(2.0))

        xn = (xx - (self.image_size-1)/2) / (self.image_size/2)
        yn = (yy - (self.image_size-1)/2) / (self.image_size/2)
        rr = np.sqrt(xn*xn + yn*yn)
        for radius in (0.30, 0.50, 0.70, 0.90):
            frac = float(gm[rr <= radius].sum() / total)
            add(
                f"global:edge_mass_r{radius:.2f}",
                frac,
                global_grad,
                np.sqrt(2.0),
            )

        # Quadrant edge shares.
        half = self.image_size // 2
        quads = [
            grad[:half, :half],
            grad[:half, half:],
            grad[half:, :half],
            grad[half:, half:],
        ]
        for i, arr in enumerate(quads):
            raw = float(arr.mean())
            add(
                f"global:quadrant_grad_share_{i}",
                float(arr.sum()) / total * 4.0,
                raw,
                np.sqrt(2.0),
            )

        # Global orientation distribution: contrast invariant.
        denom = total_grad_sum
        for b in range(8):
            lo = b * (2*np.pi/8)
            hi = (b+1) * (2*np.pi/8)
            mask = (ori >= lo) & (ori < hi)
            h = float(grad[mask].sum() / denom)
            add(
                f"global:ori{b}",
                h,
                global_grad,
                np.sqrt(2.0),
            )

        # Scale-to-noise invariant is NOT used as identity. We expose no
        # absolute intensity/gradient magnitude feature here by design.

        return (
            np.asarray(values, dtype=np.float32),
            np.asarray(support, dtype=np.float32),
            np.asarray(coeff, dtype=np.float32),
            names,
        )

    # ------------------------------------------------------------------
    # Robust class-conditional distributions
    # ------------------------------------------------------------------
    @staticmethod
    def _robust_center_scale(values: np.ndarray):
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        scale = 1.4826 * mad + 1e-3
        return center.astype(np.float32), scale.astype(np.float32)

    def _fit_bank(
        self,
        pos_values: np.ndarray,
        neg_values: np.ndarray,
        pos_support: np.ndarray,
        neg_support: np.ndarray,
        noise_coeff: np.ndarray,
        names: list[str],
        mode: str,
    ) -> dict[str, Any]:
        pos_center, pos_scale = self._robust_center_scale(pos_values)
        neg_center, neg_scale = self._robust_center_scale(neg_values)

        effect = np.abs(pos_center - neg_center) / (
            pos_scale + neg_scale + 1e-6
        )

        ref_support = np.median(
            np.concatenate([pos_support, neg_support], axis=0),
            axis=0,
        ) + 1e-6

        weight = effect.copy()
        eligible = np.where(effect >= self.min_effect)[0]

        if len(eligible) >= self.min_constraints:
            eligible = eligible[np.argsort(weight[eligible])]
            selected = eligible[-min(self.max_constraints, len(eligible)):]
        else:
            order = np.argsort(weight)
            keep_n = min(
                self.max_constraints,
                max(self.min_constraints, len(eligible)),
                len(order),
            )
            selected = order[-keep_n:]

        self.selected = selected
        self.pos_center = pos_center[selected]
        self.neg_center = neg_center[selected]
        self.pos_scale = pos_scale[selected]
        self.neg_scale = neg_scale[selected]
        self.weights = np.maximum(weight[selected], 1e-4).astype(np.float32)
        self.reference_support = ref_support[selected].astype(np.float32)
        self.noise_coeff = noise_coeff[selected].astype(np.float32)
        self.feature_names = [names[i] for i in selected]
        self.representation_mode = mode

        return {
            "mode": mode,
            "n_observables_total": int(pos_values.shape[1]),
            "n_selected": int(len(selected)),
            "effect_max": float(effect.max()),
            "effect_p95": float(np.percentile(effect, 95)),
            "effect_median": float(np.median(effect)),
            "n_effect_eligible": int((effect >= self.min_effect).sum()),
            "min_effect": float(self.min_effect),
        }

    def fit(self, images: list[np.ndarray], labels: np.ndarray):
        labels = np.asarray(labels, dtype=int)
        pos = [im for im, y in zip(images, labels) if y == 1]
        neg = [im for im, y in zip(images, labels) if y == 0]
        if len(pos) < 8 or len(neg) < 8:
            raise ValueError("Need at least 8 positive and 8 negative images.")

        self.roles = self._discover_roles(pos)
        self.pairs = np.asarray(
            list(combinations(range(len(self.roles)), 2)),
            dtype=int,
        )

        # Initial relation-language diagnostic.
        pvals, nvals, psup, nsup = [], [], [], []
        coeff = names = None
        for im in pos:
            v, s, c, n = self._relational_observables(im)
            pvals.append(v); psup.append(s); coeff = c; names = n
        for im in neg:
            v, s, c, n = self._relational_observables(im)
            nvals.append(v); nsup.append(s); coeff = c; names = n

        pvals = np.asarray(pvals)
        nvals = np.asarray(nvals)
        pc, ps = self._robust_center_scale(pvals)
        nc, ns = self._robust_center_scale(nvals)
        rel_effect = np.abs(pc-nc)/(ps+ns+1e-6)

        relational_diag = {
            "mode": "role_ratios_v4",
            "effect_max": float(rel_effect.max()),
            "effect_p95": float(np.percentile(rel_effect, 95)),
            "effect_median": float(np.median(rel_effect)),
        }

        if relational_diag["effect_p95"] >= self.expansion_effect_p95:
            final_diag = self._fit_bank(
                pvals, nvals,
                np.asarray(psup), np.asarray(nsup),
                coeff, names,
                "role_ratios_v4",
            )
            expanded = False
        else:
            print(
                "SCLI representation incomplete: "
                f"role-ratio effect p95={relational_diag['effect_p95']:.3f} "
                f"< {self.expansion_effect_p95:.3f}. "
                "Expanding to invariant structural bank v4..."
            )

            pvals, nvals, psup, nsup = [], [], [], []
            coeff = names = None
            for im in pos:
                v, s, c, n = self._structural_observables(im)
                pvals.append(v); psup.append(s); coeff = c; names = n
            for im in neg:
                v, s, c, n = self._structural_observables(im)
                nvals.append(v); nsup.append(s); coeff = c; names = n

            final_diag = self._fit_bank(
                np.asarray(pvals), np.asarray(nvals),
                np.asarray(psup), np.asarray(nsup),
                coeff, names,
                "invariant_structural_bank_v4",
            )
            expanded = True

        self.fit_diagnostics = {
            "identity_observability_separated": True,
            "initial_representation": relational_diag,
            "representation_expanded": expanded,
            "final_representation": final_diag,
            "selected_feature_examples": self.feature_names[-16:],
        }

        print(
            "SCLI final representation: "
            f"{self.representation_mode}; "
            f"effect(max/p95/median)="
            f"{final_diag['effect_max']:.3f}/"
            f"{final_diag['effect_p95']:.3f}/"
            f"{final_diag['effect_median']:.3f}; "
            f"eligible={final_diag['n_effect_eligible']}; "
            f"selected={final_diag['n_selected']}"
        )
        return self

    # ------------------------------------------------------------------
    # Observability + inference
    # ------------------------------------------------------------------
    def _selected_observables(self, image: np.ndarray):
        if self.representation_mode == "role_ratios_v4":
            v, s, c, _ = self._relational_observables(image)
        elif self.representation_mode == "invariant_structural_bank_v4":
            v, s, c, _ = self._structural_observables(image)
        else:
            raise RuntimeError("Unknown/unfitted representation.")
        return v[self.selected], s[self.selected], c[self.selected]

    def score_one(self, image: np.ndarray) -> dict[str, Any]:
        if self.selected is None:
            raise RuntimeError("Model is not fitted.")

        values, support, coeff = self._selected_observables(image)
        sigma = self._noise_sigma(image)

        threshold = np.maximum(
            self.obs_sigma_factor * coeff * sigma,
            self.obs_reference_fraction * self.reference_support,
        )
        observable = support >= threshold

        if not observable.any():
            return {
                "score": 0.5,
                "coverage": 0.0,
                "noise_sigma": sigma,
                "n_observable": 0,
            }

        v = values[observable]
        dp = np.abs(v-self.pos_center[observable]) / self.pos_scale[observable]
        dn = np.abs(v-self.neg_center[observable]) / self.neg_scale[observable]
        local_score = dn / (dp+dn+1e-6)

        w = self.weights[observable]
        score = float((w*local_score).sum()/(w.sum()+1e-12))
        coverage = float(w.sum()/(self.weights.sum()+1e-12))

        return {
            "score": score,
            "coverage": coverage,
            "noise_sigma": sigma,
            "n_observable": int(observable.sum()),
        }

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

        best_certified = None
        best_any = None

        for positive_min in np.arange(0.55, 0.96, 0.02):
            for negative_max in np.arange(0.20, 0.66, 0.02):
                if negative_max >= positive_min:
                    continue
                for coverage_min in np.arange(0.05, 0.91, 0.05):
                    kp = (score >= positive_min) & (coverage >= coverage_min)
                    kn = (score <= negative_max) & (coverage >= coverage_min)
                    known = kp | kn
                    cov = float(known.mean())
                    if cov < min_known_coverage:
                        continue
                    pred = np.where(kp, 1, 0)
                    acc = float((pred[known] == labels[known]).mean())

                    cand = Gate(
                        float(positive_min),
                        float(negative_max),
                        float(coverage_min),
                        cov,
                        acc,
                        certified=acc >= target_known_accuracy,
                        target_known_accuracy=float(target_known_accuracy),
                    )

                    if acc >= target_known_accuracy and (
                        best_certified is None
                        or cov > best_certified.validation_coverage
                    ):
                        best_certified = cand

                    if (
                        best_any is None
                        or acc > best_any.validation_known_accuracy + 1e-12
                        or (
                            abs(acc-best_any.validation_known_accuracy) <= 1e-12
                            and cov > best_any.validation_coverage
                        )
                    ):
                        best_any = cand

        if best_certified is not None:
            self.gate = best_certified
            print(
                "SCLI epistemic gate CERTIFIED: "
                f"validation known accuracy="
                f"{best_certified.validation_known_accuracy:.3f}, "
                f"coverage={best_certified.validation_coverage:.3f}"
            )
            return best_certified

        if best_any is None:
            raise RuntimeError("No non-empty epistemic gate could be formed.")

        best_any.certified = False
        self.gate = best_any
        print(
            "WARNING: SCLI epistemic gate is UNCERTIFIED "
            f"at target {target_known_accuracy:.3f}. "
            f"Best validation known accuracy="
            f"{best_any.validation_known_accuracy:.3f}, "
            f"coverage={best_any.validation_coverage:.3f}."
        )
        return best_any

    def predict_one(self, image: np.ndarray) -> dict[str, Any]:
        result = self.score_one(image)
        result["raw_label"] = int(result["score"] >= 0.5)
        result["known"] = False
        result["label"] = None
        result["gate_certified"] = (
            bool(self.gate.certified) if self.gate is not None else False
        )

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
            "representation_mode": self.representation_mode,
            "fit_diagnostics": self.fit_diagnostics,
            "gate": None if self.gate is None else asdict(self.gate),
        }

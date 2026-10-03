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
    """Observable-constraint binary model with representation expansion.

    The first representation is a cross-instance role-relation language.
    If that representation is empirically non-separating, the model marks it
    representation-incomplete and expands to a richer structural observable bank.

    The decision layer remains deterministic and distributional:
      - learn robust admissible distributions per observable for positive/negative;
      - rank observables by robust separation;
      - let an observable vote only when its raw support clears a noise floor;
      - allow UNDERDETERMINED instead of forcing a class.
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
        self.min_stability = min_stability  # diagnostic/backward compatibility
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
        self.representation_mode: str = "unfitted"
        self.fit_diagnostics: dict[str, Any] = {}
        self.gate: Gate | None = None

    # ------------------------------------------------------------------
    # Common image transforms
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

        half_h = int(
            np.clip(
                2.1 * sy,
                self.image_size * 0.30,
                self.image_size / 2,
            )
        )
        half_w = int(
            np.clip(
                2.1 * sx,
                self.image_size * 0.30,
                self.image_size / 2,
            )
        )
        y0, y1 = (
            max(0, int(cy - half_h)),
            min(self.image_size, int(cy + half_h)),
        )
        x0, x1 = (
            max(0, int(cx - half_w)),
            min(self.image_size, int(cx + half_w)),
        )
        crop = g[y0:y1, x0:x1]
        if min(crop.shape) < 8:
            crop = g

        return transform.resize(
            crop,
            (self.image_size, self.image_size),
            anti_aliasing=True,
        ).astype(np.float32)

    def _noise_sigma_from_gray(self, z: np.ndarray) -> float:
        low = gaussian_filter(z, 0.8)
        residual = z - low
        mad = np.median(np.abs(residual - np.median(residual)))
        return float(mad / 0.6745 + 1e-5)

    def _noise_sigma(self, image: np.ndarray) -> float:
        return self._noise_sigma_from_gray(self._resize(image))

    # ------------------------------------------------------------------
    # Representation 1: role-to-role intensity relations
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
                (y - ry) ** 2 + (x - rx) ** 2
                >= self.min_role_spacing ** 2
                for ry, rx in roles
            ):
                roles.append((y, x))
            if len(roles) >= self.n_roles:
                break

        if len(roles) < 4:
            raise RuntimeError(
                "Could not discover enough stable roles."
            )
        return np.asarray(roles, dtype=int)

    def _role_means(self, image: np.ndarray) -> np.ndarray:
        if self.roles is None:
            raise RuntimeError("Model is not fitted.")

        z = gaussian_filter(self._canonicalize(image), 0.6)
        out = []
        r = self.patch_radius
        for y, x in self.roles:
            ys = slice(
                max(0, y - r),
                min(self.image_size, y + r + 1),
            )
            xs = slice(
                max(0, x - r),
                min(self.image_size, x + r + 1),
            )
            out.append(float(z[ys, xs].mean()))
        return np.asarray(out, dtype=np.float32)

    def _relational_values(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = self._role_means(image)
        values = (
            means[self.pairs[:, 0]]
            - means[self.pairs[:, 1]]
        ).astype(np.float32)

        # Support is the actual raw relation magnitude.
        support = np.abs(values).astype(np.float32)
        n = (2 * self.patch_radius + 1) ** 2
        coeff = np.full(
            len(values),
            np.sqrt(2.0) / np.sqrt(n),
            dtype=np.float32,
        )
        return values, support, coeff

    # ------------------------------------------------------------------
    # Representation 2: structural observable bank
    # ------------------------------------------------------------------
    def _structural_observables(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Return values, raw support, noise coefficient, feature names.

        The bank is intentionally low-level and interpretable. It contains:
        local contrast, multi-scale edge energy, Laplacian/DoG structure,
        edge occupancy, orientation coherence/histograms, and global
        center-vs-surround / spatial-distribution observables.
        """
        z = self._resize(image)
        sigma = self._noise_sigma_from_gray(z)

        q = gaussian_filter(z, 0.6)
        q2 = gaussian_filter(z, 1.8)
        gy, gx = np.gradient(q)
        grad = np.sqrt(gx * gx + gy * gy)
        lap = np.abs(laplace(q))
        dog = np.abs(q - q2)
        ori = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)

        values: list[float] = []
        support: list[float] = []
        coeff: list[float] = []
        names: list[str] = []

        grid = 4
        step = self.image_size // grid

        def add(
            name: str,
            value: float,
            raw_support: float,
            noise_c: float,
        ):
            values.append(float(value))
            support.append(float(max(raw_support, 0.0)))
            coeff.append(float(max(noise_c, 1e-6)))
            names.append(name)

        for ry in range(grid):
            for rx in range(grid):
                y0 = ry * step
                y1 = self.image_size if ry == grid - 1 else (ry + 1) * step
                x0 = rx * step
                x1 = self.image_size if rx == grid - 1 else (rx + 1) * step
                sl = (slice(y0, y1), slice(x0, x1))

                p = q[sl]
                pg = grad[sl]
                pl = lap[sl]
                pd = dog[sl]
                po = ori[sl]

                std = float(p.std())
                gmean = float(pg.mean())
                lmean = float(pl.mean())
                dmean = float(pd.mean())

                add(f"cell{ry}{rx}:std", std, std, 1.0)
                add(
                    f"cell{ry}{rx}:grad",
                    gmean,
                    gmean,
                    np.sqrt(2.0),
                )
                add(
                    f"cell{ry}{rx}:lap",
                    lmean,
                    lmean,
                    2.5,
                )
                add(
                    f"cell{ry}{rx}:dog",
                    dmean,
                    dmean,
                    1.0,
                )

                edge_threshold = 2.0 * np.sqrt(2.0) * sigma
                edge_fraction = float(
                    np.mean(pg >= edge_threshold)
                )
                add(
                    f"cell{ry}{rx}:edge_fraction",
                    edge_fraction,
                    gmean,
                    np.sqrt(2.0),
                )

                # Orientation coherence: 1 for aligned gradients, 0 for mixed.
                denom = float(pg.sum()) + 1e-8
                cx = float((np.cos(po) * pg).sum())
                cy = float((np.sin(po) * pg).sum())
                coherence = float(
                    np.sqrt(cx * cx + cy * cy) / denom
                )
                add(
                    f"cell{ry}{rx}:ori_coherence",
                    coherence,
                    gmean,
                    np.sqrt(2.0),
                )

                # Four contrast-normalized orientation bins.
                for b in range(4):
                    lo = b * (2 * np.pi / 4)
                    hi = (b + 1) * (2 * np.pi / 4)
                    mask = (po >= lo) & (po < hi)
                    hist = float(pg[mask].sum() / denom)
                    add(
                        f"cell{ry}{rx}:ori{b}",
                        hist,
                        gmean,
                        np.sqrt(2.0),
                    )

        # Global center-vs-surround structure.
        a = self.image_size // 4
        b = self.image_size - a
        center = np.zeros_like(z, dtype=bool)
        center[a:b, a:b] = True
        border = ~center

        for name, arr, nc in [
            ("grad", grad, np.sqrt(2.0)),
            ("lap", lap, 2.5),
            ("dog", dog, 1.0),
        ]:
            c = float(arr[center].mean())
            o = float(arr[border].mean())
            add(
                f"global:center_minus_border_{name}",
                c - o,
                max(c, o),
                nc,
            )
            add(
                f"global:center_over_border_{name}",
                c / (o + 1e-6),
                max(c, o),
                nc,
            )

        # Intensity spread and multi-scale structural energy.
        spread = float(
            np.percentile(q, 90) - np.percentile(q, 10)
        )
        add("global:intensity_spread", spread, spread, 1.0)

        gmean = float(grad.mean())
        add(
            "global:grad_mean",
            gmean,
            gmean,
            np.sqrt(2.0),
        )

        # Edge-mass spatial concentration / centroid.
        gm = grad + 1e-8
        total = float(gm.sum())
        yy, xx = np.mgrid[
            0:self.image_size,
            0:self.image_size,
        ]
        cx = float((xx * gm).sum() / total)
        cy = float((yy * gm).sum() / total)
        norm = max(self.image_size - 1, 1)

        add(
            "global:edge_centroid_x",
            cx / norm,
            gmean,
            np.sqrt(2.0),
        )
        add(
            "global:edge_centroid_y",
            cy / norm,
            gmean,
            np.sqrt(2.0),
        )

        # Fraction of gradient energy in concentric regions.
        xn = (xx - (self.image_size - 1) / 2) / (
            self.image_size / 2
        )
        yn = (yy - (self.image_size - 1) / 2) / (
            self.image_size / 2
        )
        rr = np.sqrt(xn * xn + yn * yn)
        for radius in [0.35, 0.60, 0.85]:
            frac = float(gm[rr <= radius].sum() / total)
            add(
                f"global:edge_mass_r{radius:.2f}",
                frac,
                gmean,
                np.sqrt(2.0),
            )

        # Global orientation histogram.
        denom = float(grad.sum()) + 1e-8
        for b in range(8):
            lo = b * (2 * np.pi / 8)
            hi = (b + 1) * (2 * np.pi / 8)
            mask = (ori >= lo) & (ori < hi)
            hist = float(grad[mask].sum() / denom)
            add(
                f"global:ori{b}",
                hist,
                gmean,
                np.sqrt(2.0),
            )

        return (
            np.asarray(values, dtype=np.float32),
            np.asarray(support, dtype=np.float32),
            np.asarray(coeff, dtype=np.float32),
            names,
        )

    # ------------------------------------------------------------------
    # Distribution fitting / representation expansion
    # ------------------------------------------------------------------
    @staticmethod
    def _robust_center_scale(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        center = np.median(values, axis=0)
        mad = np.median(
            np.abs(values - center),
            axis=0,
        )
        scale = 1.4826 * mad + 1e-3
        return (
            center.astype(np.float32),
            scale.astype(np.float32),
        )

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
        pos_center, pos_scale = self._robust_center_scale(
            pos_values
        )
        neg_center, neg_scale = self._robust_center_scale(
            neg_values
        )

        effect = np.abs(pos_center - neg_center) / (
            pos_scale + neg_scale + 1e-6
        )

        ref_support = np.median(
            np.concatenate(
                [pos_support, neg_support],
                axis=0,
            ),
            axis=0,
        ) + 1e-6

        support_norm = ref_support / (
            np.median(ref_support) + 1e-6
        )
        weight = (
            effect
            * np.sqrt(
                np.clip(support_norm, 0.05, 8.0)
            )
        )

        eligible = np.where(
            effect >= self.min_effect
        )[0]

        if len(eligible) >= self.min_constraints:
            eligible = eligible[
                np.argsort(weight[eligible])
            ]
            selected = eligible[
                -min(
                    self.max_constraints,
                    len(eligible),
                ):
            ]
        else:
            # Do not invent a success. Keep the strongest observables
            # so the full experiment can run and diagnostics quantify
            # how incomplete the representation remains.
            order = np.argsort(weight)
            keep_n = min(
                self.max_constraints,
                max(
                    self.min_constraints,
                    len(eligible),
                ),
                len(order),
            )
            selected = order[-keep_n:]

        self.selected = selected
        self.pos_center = pos_center[selected]
        self.neg_center = neg_center[selected]
        self.pos_scale = pos_scale[selected]
        self.neg_scale = neg_scale[selected]
        self.weights = np.maximum(
            weight[selected],
            1e-4,
        ).astype(np.float32)
        self.reference_support = (
            ref_support[selected].astype(np.float32)
        )
        self.noise_coeff = (
            noise_coeff[selected].astype(np.float32)
        )
        self.feature_names = [
            names[i] for i in selected
        ]
        self.representation_mode = mode

        diag = {
            "mode": mode,
            "n_observables_total": int(
                pos_values.shape[1]
            ),
            "n_selected": int(len(selected)),
            "effect_max": float(effect.max()),
            "effect_p95": float(
                np.percentile(effect, 95)
            ),
            "effect_median": float(
                np.median(effect)
            ),
            "n_effect_eligible": int(
                (effect >= self.min_effect).sum()
            ),
            "min_effect": float(self.min_effect),
        }
        return diag

    def fit(
        self,
        images: list[np.ndarray],
        labels: np.ndarray,
    ) -> "SCLIVisionBinary":
        labels = np.asarray(labels, dtype=int)
        pos = [
            im for im, y in zip(images, labels)
            if y == 1
        ]
        neg = [
            im for im, y in zip(images, labels)
            if y == 0
        ]

        if len(pos) < 8 or len(neg) < 8:
            raise ValueError(
                "Need at least 8 positive and 8 negative images."
            )

        # ---- Attempt 1: role-relational representation.
        self.roles = self._discover_roles(pos)
        self.pairs = np.asarray(
            list(
                combinations(
                    range(len(self.roles)),
                    2,
                )
            ),
            dtype=int,
        )

        pos_rel = []
        neg_rel = []
        pos_rel_sup = []
        neg_rel_sup = []
        rel_coeff = None

        for im in pos:
            v, s, c = self._relational_values(im)
            pos_rel.append(v)
            pos_rel_sup.append(s)
            rel_coeff = c
        for im in neg:
            v, s, c = self._relational_values(im)
            neg_rel.append(v)
            neg_rel_sup.append(s)
            rel_coeff = c

        pos_rel = np.asarray(pos_rel)
        neg_rel = np.asarray(neg_rel)
        pos_rel_sup = np.asarray(pos_rel_sup)
        neg_rel_sup = np.asarray(neg_rel_sup)

        pos_rate = (pos_rel > 0).mean(axis=0)
        sign_stability = np.maximum(
            pos_rate,
            1.0 - pos_rate,
        )

        pc, ps = self._robust_center_scale(pos_rel)
        nc, ns = self._robust_center_scale(neg_rel)
        rel_effect = np.abs(pc - nc) / (
            ps + ns + 1e-6
        )

        relational_diag = {
            "mode": "role_relations_v2",
            "sign_stability_max": float(
                sign_stability.max()
            ),
            "sign_stability_p95": float(
                np.percentile(sign_stability, 95)
            ),
            "sign_stability_median": float(
                np.median(sign_stability)
            ),
            "effect_max": float(rel_effect.max()),
            "effect_p95": float(
                np.percentile(rel_effect, 95)
            ),
            "effect_median": float(
                np.median(rel_effect)
            ),
        }

        if (
            relational_diag["effect_p95"]
            >= self.expansion_effect_p95
        ):
            names = [
                f"role_relation:{a}-{b}"
                for a, b in self.pairs
            ]
            chosen_diag = self._fit_bank(
                pos_rel,
                neg_rel,
                pos_rel_sup,
                neg_rel_sup,
                rel_coeff,
                names,
                "role_relations_v2",
            )
            expanded = False
        else:
            # ---- Representation expansion.
            print(
                "SCLI representation incomplete: "
                f"role-relation effect p95="
                f"{relational_diag['effect_p95']:.3f} "
                f"< {self.expansion_effect_p95:.3f}. "
                "Expanding to structural observable bank..."
            )

            pvals = []
            nvals = []
            psup = []
            nsup = []
            scoeff = None
            snames = None

            for im in pos:
                v, s, c, names = (
                    self._structural_observables(im)
                )
                pvals.append(v)
                psup.append(s)
                scoeff = c
                snames = names
            for im in neg:
                v, s, c, names = (
                    self._structural_observables(im)
                )
                nvals.append(v)
                nsup.append(s)
                scoeff = c
                snames = names

            chosen_diag = self._fit_bank(
                np.asarray(pvals),
                np.asarray(nvals),
                np.asarray(psup),
                np.asarray(nsup),
                scoeff,
                snames,
                "structural_bank_v3",
            )
            expanded = True

        self.fit_diagnostics = {
            "initial_representation": relational_diag,
            "representation_expanded": expanded,
            "final_representation": chosen_diag,
            "selected_feature_examples": (
                self.feature_names[-12:]
            ),
        }

        print(
            "SCLI final representation: "
            f"{self.representation_mode}; "
            f"effect(max/p95/median)="
            f"{chosen_diag['effect_max']:.3f}/"
            f"{chosen_diag['effect_p95']:.3f}/"
            f"{chosen_diag['effect_median']:.3f}; "
            f"eligible={chosen_diag['n_effect_eligible']}; "
            f"selected={chosen_diag['n_selected']}"
        )
        return self

    # ------------------------------------------------------------------
    # Inference / epistemic gate
    # ------------------------------------------------------------------
    def _selected_observables(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.representation_mode == "role_relations_v2":
            v, s, c = self._relational_values(image)
        elif self.representation_mode == "structural_bank_v3":
            v, s, c, _ = self._structural_observables(image)
        else:
            raise RuntimeError(
                "Unknown/unfitted representation."
            )
        return (
            v[self.selected],
            s[self.selected],
            c[self.selected],
        )

    def score_one(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        if self.selected is None:
            raise RuntimeError("Model is not fitted.")

        values, support, coeff = (
            self._selected_observables(image)
        )
        sigma = self._noise_sigma(image)

        threshold = np.maximum(
            self.obs_sigma_factor * coeff * sigma,
            self.obs_reference_fraction
            * self.reference_support,
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
        dp = np.abs(
            v - self.pos_center[observable]
        ) / self.pos_scale[observable]
        dn = np.abs(
            v - self.neg_center[observable]
        ) / self.neg_scale[observable]

        local_score = dn / (
            dp + dn + 1e-6
        )

        w = self.weights[observable]
        score = float(
            (w * local_score).sum()
            / (w.sum() + 1e-12)
        )
        coverage = float(
            w.sum()
            / (self.weights.sum() + 1e-12)
        )

        return {
            "score": score,
            "coverage": coverage,
            "noise_sigma": sigma,
            "n_observable": int(
                observable.sum()
            ),
        }

    def calibrate(
        self,
        images: list[np.ndarray],
        labels: np.ndarray,
        target_known_accuracy: float = 0.90,
        min_known_coverage: float = 0.05,
    ) -> Gate:
        labels = np.asarray(labels, dtype=int)
        rows = [
            self.score_one(im)
            for im in images
        ]
        score = np.asarray(
            [r["score"] for r in rows]
        )
        coverage = np.asarray(
            [r["coverage"] for r in rows]
        )

        best_certified = None
        best_any = None

        for positive_min in np.arange(
            0.55,
            0.96,
            0.02,
        ):
            for negative_max in np.arange(
                0.20,
                0.66,
                0.02,
            ):
                if negative_max >= positive_min:
                    continue
                for coverage_min in np.arange(
                    0.05,
                    0.91,
                    0.05,
                ):
                    known_pos = (
                        (score >= positive_min)
                        & (coverage >= coverage_min)
                    )
                    known_neg = (
                        (score <= negative_max)
                        & (coverage >= coverage_min)
                    )
                    known = known_pos | known_neg
                    cov = float(known.mean())
                    if cov < min_known_coverage:
                        continue

                    pred = np.where(
                        known_pos,
                        1,
                        0,
                    )
                    acc = float(
                        (
                            pred[known]
                            == labels[known]
                        ).mean()
                    )

                    candidate = Gate(
                        float(positive_min),
                        float(negative_max),
                        float(coverage_min),
                        cov,
                        acc,
                        certified=(
                            acc
                            >= target_known_accuracy
                        ),
                        target_known_accuracy=float(
                            target_known_accuracy
                        ),
                    )

                    if (
                        acc >= target_known_accuracy
                        and (
                            best_certified is None
                            or cov
                            > best_certified.validation_coverage
                        )
                    ):
                        best_certified = candidate

                    if (
                        best_any is None
                        or acc
                        > best_any.validation_known_accuracy
                        + 1e-12
                        or (
                            abs(
                                acc
                                - best_any.validation_known_accuracy
                            )
                            <= 1e-12
                            and cov
                            > best_any.validation_coverage
                        )
                    ):
                        best_any = candidate

        if best_certified is not None:
            self.gate = best_certified
            print(
                "SCLI epistemic gate CERTIFIED: "
                f"validation known accuracy="
                f"{best_certified.validation_known_accuracy:.3f}, "
                f"coverage="
                f"{best_certified.validation_coverage:.3f}"
            )
            return best_certified

        if best_any is None:
            raise RuntimeError(
                "No non-empty epistemic gate could be formed."
            )

        best_any.certified = False
        self.gate = best_any
        print(
            "WARNING: SCLI epistemic gate is UNCERTIFIED "
            f"at target {target_known_accuracy:.3f}. "
            f"Best validation known accuracy="
            f"{best_any.validation_known_accuracy:.3f}, "
            f"coverage={best_any.validation_coverage:.3f}. "
            "Continuing so the held-out test can quantify the gap."
        )
        return best_any

    def predict_one(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        result = self.score_one(image)
        result["raw_label"] = int(
            result["score"] >= 0.5
        )
        result["known"] = False
        result["label"] = None
        result["gate_certified"] = bool(
            self.gate.certified
        ) if self.gate is not None else False

        if self.gate is None:
            return result

        if (
            result["coverage"]
            >= self.gate.coverage_min
        ):
            if (
                result["score"]
                >= self.gate.positive_min
            ):
                result["known"] = True
                result["label"] = 1
            elif (
                result["score"]
                <= self.gate.negative_max
            ):
                result["known"] = True
                result["label"] = 0

        return result

    def predict(
        self,
        images: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        return [
            self.predict_one(im)
            for im in images
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "n_roles": (
                0
                if self.roles is None
                else int(len(self.roles))
            ),
            "n_constraints": (
                0
                if self.selected is None
                else int(len(self.selected))
            ),
            "representation_mode": self.representation_mode,
            "fit_diagnostics": self.fit_diagnostics,
            "gate": (
                None
                if self.gate is None
                else asdict(self.gate)
            ),
        }

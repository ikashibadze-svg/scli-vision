from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .model_v11 import Gate, SCLIVisionBinary as _BaseSCLI


class SCLIVisionBinary(_BaseSCLI):
    """SCLI Vision v12: predictive continuity evidence for CLEAR.

    A candidate region is CLEAR only when the surrounding scene predicts the
    candidate consistently. We fit a small explicit polynomial background
    surface using ONLY the surround ring, then measure whether the center and
    its boundary behave like continuation of that predicted surface.

    This is not a learned vision model: the predictor is a deterministic
    least-squares geometric continuation model recomputed per image.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_constraints"] = max(
            int(kwargs.get("max_constraints", 320)),
            380,
        )
        kwargs["beam_width"] = max(
            int(kwargs.get("beam_width", 300)),
            360,
        )
        kwargs["literal_pool_per_class"] = max(
            int(kwargs.get("literal_pool_per_class", 220)),
            260,
        )
        super().__init__(*args, **kwargs)
        self.predictive_clear_feature_count = 0

    @staticmethod
    def _poly_basis(n: int, degree: int) -> np.ndarray:
        yy, xx = np.mgrid[0:n, 0:n]
        x = (xx.astype(np.float64) - (n - 1) / 2) / max(n / 2, 1)
        y = (yy.astype(np.float64) - (n - 1) / 2) / max(n / 2, 1)

        cols = [
            np.ones_like(x),
            x,
            y,
            x * x,
            x * y,
            y * y,
        ]
        if degree >= 3:
            cols.extend([
                x * x * x,
                x * x * y,
                x * y * y,
                y * y * y,
            ])
        return np.stack(cols, axis=-1)

    @staticmethod
    def _rmse(a: np.ndarray) -> float:
        if a.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(a))))

    @staticmethod
    def _mad_scale(a: np.ndarray) -> float:
        if a.size == 0:
            return 0.0
        med = np.median(a)
        return float(
            1.4826 * np.median(np.abs(a - med)) + 1e-8
        )

    def _predictive_continuity_features(
        self,
        image: np.ndarray,
    ) -> tuple[list[float], list[float], list[float], list[str]]:
        z = self._resize(image)
        n = self.image_size

        a = n // 4
        b = n - a
        center = np.zeros((n, n), dtype=bool)
        center[a:b, a:b] = True
        surround = ~center

        bw = max(2, n // 32)
        boundary = np.zeros((n, n), dtype=bool)
        boundary[a:a+bw, a:b] = True
        boundary[b-bw:b, a:b] = True
        boundary[a:b, a:a+bw] = True
        boundary[a:b, b-bw:b] = True
        boundary[max(0, a-bw):a, a:b] = True
        boundary[b:min(n, b+bw), a:b] = True
        boundary[a:b, max(0, a-bw):a] = True
        boundary[a:b, b:min(n, b+bw)] = True

        vals: list[float] = []
        sups: list[float] = []
        coeffs: list[float] = []
        names: list[str] = []

        def add(
            name: str,
            identity: float,
            raw_support: float,
            noise_coeff: float,
        ):
            vals.append(float(identity))
            sups.append(float(max(raw_support, 0.0)))
            coeffs.append(float(max(noise_coeff, 1e-8)))
            names.append(name)

        basis2 = self._poly_basis(n, 2)
        basis3 = self._poly_basis(n, 3)

        nc = float(center.sum())
        ns = float(surround.sum())
        nb = float(boundary.sum())

        for sigma in (0.6, 1.2, 2.4, 4.0):
            q = gaussian_filter(z, sigma=sigma)

            # Noise estimate comes from high-frequency residual at the same
            # scale. It controls observability, never identity.
            hf = q - gaussian_filter(q, sigma=max(0.8, sigma * 1.5))
            noise = self._mad_scale(hf[surround])

            spread = float(
                np.percentile(q[surround], 90)
                - np.percentile(q[surround], 10)
            ) + 1e-8
            signal_support = max(spread - 2.0 * noise, 0.0)

            for degree, basis in ((2, basis2), (3, basis3)):
                A = basis[surround]
                y = q[surround]

                # Stable tiny system (6 or 10 coefficients).
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                pred = np.tensordot(basis, coef, axes=([-1], [0]))
                residual = q - pred

                rc = residual[center]
                rs = residual[surround]
                rb = residual[boundary]

                center_rmse = self._rmse(rc)
                surround_rmse = self._rmse(rs)
                boundary_rmse = self._rmse(rb)

                eps = 0.05 * spread + 1e-8

                # Positive continuity identities: 1 is best.
                center_ratio = center_rmse / (surround_rmse + eps)
                boundary_ratio = boundary_rmse / (surround_rmse + eps)

                center_similarity = float(
                    1.0 / (1.0 + max(center_ratio - 1.0, 0.0))
                )
                boundary_similarity = float(
                    1.0 / (1.0 + max(boundary_ratio - 1.0, 0.0))
                )

                add(
                    f"clear:predict:d{degree}:s{sigma:.1f}:center_similarity",
                    center_similarity,
                    signal_support,
                    np.sqrt(1 / nc + 1 / ns),
                )
                add(
                    f"clear:predict:d{degree}:s{sigma:.1f}:boundary_similarity",
                    boundary_similarity,
                    signal_support,
                    np.sqrt(1 / nb + 1 / ns),
                )

                # Relative residual ratio retains direction: values >1 mean
                # the center is less explainable than its surround.
                add(
                    f"clear:predict:d{degree}:s{sigma:.1f}:center_residual_ratio",
                    float(np.log((center_rmse + eps) / (surround_rmse + eps))),
                    signal_support,
                    np.sqrt(1 / nc + 1 / ns),
                )
                add(
                    f"clear:predict:d{degree}:s{sigma:.1f}:boundary_residual_ratio",
                    float(np.log((boundary_rmse + eps) / (surround_rmse + eps))),
                    signal_support,
                    np.sqrt(1 / nb + 1 / ns),
                )

                # Model quality itself: CLEAR can be asserted only where the
                # surrounding scene is sufficiently predictable.
                surround_quality = float(
                    np.exp(
                        -surround_rmse
                        / (spread + 1e-8)
                    )
                )
                add(
                    f"clear:predict:d{degree}:s{sigma:.1f}:surround_quality",
                    surround_quality,
                    signal_support,
                    1.0 / np.sqrt(ns),
                )

                # Gradient of prediction residual: object boundaries should
                # create extra residual structure inside/boundary.
                rgy, rgx = np.gradient(residual)
                rg = np.sqrt(rgx * rgx + rgy * rgy)

                gc = float(rg[center].mean())
                gs = float(rg[surround].mean())
                gb = float(rg[boundary].mean())

                grad_center_similarity = float(
                    1.0
                    / (
                        1.0
                        + max(
                            gc / (gs + 0.03 * spread + 1e-8)
                            - 1.0,
                            0.0,
                        )
                    )
                )
                grad_boundary_similarity = float(
                    1.0
                    / (
                        1.0
                        + max(
                            gb / (gs + 0.03 * spread + 1e-8)
                            - 1.0,
                            0.0,
                        )
                    )
                )

                add(
                    f"clear:predict:d{degree}:s{sigma:.1f}:grad_center_similarity",
                    grad_center_similarity,
                    signal_support,
                    np.sqrt(2.0) * np.sqrt(1 / nc + 1 / ns),
                )
                add(
                    f"clear:predict:d{degree}:s{sigma:.1f}:grad_boundary_similarity",
                    grad_boundary_similarity,
                    signal_support,
                    np.sqrt(2.0) * np.sqrt(1 / nb + 1 / ns),
                )

        # Cross-scale persistence: CLEAR evidence should agree across scale.
        # Use the generated center-similarity features directly.
        center_ids = [
            i for i, name in enumerate(names)
            if name.endswith(":center_similarity")
            and ":grad_" not in name
        ]
        boundary_ids = [
            i for i, name in enumerate(names)
            if name.endswith(":boundary_similarity")
            and ":grad_" not in name
        ]

        if center_ids:
            arr = np.asarray([vals[i] for i in center_ids], dtype=float)
            add(
                "clear:predict:multiscale_center_consistency",
                float(1.0 - min(1.0, arr.std())),
                float(np.median([sups[i] for i in center_ids])),
                1.0 / np.sqrt(max(nc, 1.0)),
            )
            add(
                "clear:predict:multiscale_center_min",
                float(arr.min()),
                float(np.median([sups[i] for i in center_ids])),
                1.0 / np.sqrt(max(nc, 1.0)),
            )

        if boundary_ids:
            arr = np.asarray([vals[i] for i in boundary_ids], dtype=float)
            add(
                "clear:predict:multiscale_boundary_consistency",
                float(1.0 - min(1.0, arr.std())),
                float(np.median([sups[i] for i in boundary_ids])),
                1.0 / np.sqrt(max(nb, 1.0)),
            )
            add(
                "clear:predict:multiscale_boundary_min",
                float(arr.min()),
                float(np.median([sups[i] for i in boundary_ids])),
                1.0 / np.sqrt(max(nb, 1.0)),
            )

        return vals, sups, coeffs, names

    def _structural_observables(self, image: np.ndarray):
        values, support, coeff, names = super()._structural_observables(image)
        pv, ps, pc, pn = self._predictive_continuity_features(image)

        self.predictive_clear_feature_count = len(pn)

        return (
            np.concatenate([
                values,
                np.asarray(pv, dtype=np.float32),
            ]),
            np.concatenate([
                support,
                np.asarray(ps, dtype=np.float32),
            ]),
            np.concatenate([
                coeff,
                np.asarray(pc, dtype=np.float32),
            ]),
            list(names) + pn,
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
        diag = super()._fit_bank(
            pos_values,
            neg_values,
            pos_support,
            neg_support,
            noise_coeff,
            names,
            mode,
        )

        self.representation_mode = "predictive_continuity_constraints_v12"
        diag["mode"] = self.representation_mode
        diag["predictive_clear_features_total"] = int(
            sum(name.startswith("clear:predict:") for name in names)
        )
        diag["predictive_clear_features_selected"] = int(
            sum(
                name.startswith("clear:predict:")
                for name in self.feature_names
            )
        )
        diag["clear_evidence"] = (
            "surround_predicts_candidate_and_boundary"
        )
        return diag

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["predictive_clear_feature_count"] = int(
            self.predictive_clear_feature_count
        )
        return d


__all__ = ["Gate", "SCLIVisionBinary"]

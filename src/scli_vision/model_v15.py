from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .model_v13 import Gate, SCLIVisionBinary as _BaseSCLI


class SCLIVisionBinary(_BaseSCLI):
    """SCLI Vision v15: local counterfactual CLEAR reference.

    Train-time paired discovery showed stable OBJECT-vs-CLEAR constraint
    shifts, but v14 lost that advantage at inference because only one candidate
    was available and it had to be compared with a global class midpoint.

    v15 reconstructs a scene-specific CLEAR_0 hypothesis from the surrounding
    ring of every candidate and classifies the LOCAL constraint displacement:

        Delta C_i = C_i(observed) - C_i(counterfactual_clear)

    Counterfactual quality is explicitly certified. A poor surround model
    reduces epistemic coverage rather than forcing a decision.
    """

    def __init__(
        self,
        *args,
        counterfactual_max_constraints: int = 180,
        counterfactual_min_stability: float = 0.58,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.counterfactual_max_constraints = counterfactual_max_constraints
        self.counterfactual_min_stability = counterfactual_min_stability

        self.cf_local_selected: np.ndarray | None = None
        self.cf_direction: np.ndarray | None = None
        self.cf_midpoint: np.ndarray | None = None
        self.cf_scale: np.ndarray | None = None
        self.cf_weights: np.ndarray | None = None
        self.cf_effect: np.ndarray | None = None
        self.cf_pair_stability: np.ndarray | None = None

        self.cf_train_accuracy: float | None = None
        self.cf_train_quality_median: float | None = None
        self.cf_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _robust_scale(x: np.ndarray) -> np.ndarray:
        med = np.median(x, axis=0)
        return (
            1.4826
            * np.median(np.abs(x - med[None, :]), axis=0)
            + 1e-3
        )

    def _counterfactual_clear(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, dict[str, float]]:
        """Construct CLEAR_0 using ONLY the surround ring.

        Two explicit polynomial continuation hypotheses (degree 2 and 3) are
        fitted on the surround. Their average predicts the candidate center.
        Agreement between the two predictors and fit quality on the observed
        surround form a counterfactual-quality certificate.
        """
        z = self._resize(image)
        n = self.image_size

        a = n // 4
        b = n - a
        center = np.zeros((n, n), dtype=bool)
        center[a:b, a:b] = True
        surround = ~center

        # Work on a lightly denoised image so sensor noise is not hallucinated
        # into the counterfactual surface.
        q = gaussian_filter(z, 0.7)

        preds = []
        surround_rmse = []

        for degree in (2, 3):
            basis = self._poly_basis(n, degree)
            A = basis[surround]
            y = q[surround]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = np.tensordot(basis, coef, axes=([-1], [0]))
            preds.append(pred)
            surround_rmse.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                q[surround] - pred[surround]
                            )
                        )
                    )
                )
            )

        pred2, pred3 = preds
        pred = 0.5 * (pred2 + pred3)

        spread = float(
            np.percentile(q[surround], 90)
            - np.percentile(q[surround], 10)
        ) + 1e-6

        fit_rmse = float(np.mean(surround_rmse))
        model_disagreement = float(
            np.sqrt(
                np.mean(
                    np.square(
                        pred2[center] - pred3[center]
                    )
                )
            )
        )

        # Quality in [0,1]. Both surround fit and extrapolation agreement
        # must be good. This is not identity evidence; it only certifies that
        # the local CLEAR_0 reference is meaningful.
        fit_quality = float(
            np.exp(-fit_rmse / (spread + 1e-6))
        )
        agreement_quality = float(
            np.exp(-model_disagreement / (spread + 1e-6))
        )
        quality = float(
            np.sqrt(fit_quality * agreement_quality)
        )

        # Feather replacement inside the central candidate only. This avoids
        # creating an artificial hard edge at the counterfactual boundary.
        mask = center.astype(np.float32)
        soft = gaussian_filter(mask, sigma=2.0)
        soft = soft / (float(soft.max()) + 1e-8)
        soft *= mask

        cf = q * (1.0 - soft) + pred.astype(np.float32) * soft
        cf = np.clip(cf, 0.0, 1.0).astype(np.float32)

        return cf, quality, {
            "fit_quality": fit_quality,
            "agreement_quality": agreement_quality,
            "surround_fit_rmse": fit_rmse,
            "model_disagreement": model_disagreement,
            "surround_spread": spread,
        }

    def _fit_counterfactual_layer(
        self,
        images: list[np.ndarray],
        labels: np.ndarray,
        pair_ids: np.ndarray | None,
    ) -> dict[str, Any]:
        labels = np.asarray(labels, dtype=int)

        deltas = []
        qualities = []

        print(
            "Building v15 local counterfactual training deltas...",
            flush=True,
        )

        for i, image in enumerate(images, 1):
            values, _, _ = self._selected_observables(image)
            cf, quality, _ = self._counterfactual_clear(image)
            cf_values, _, _ = self._selected_observables(cf)

            deltas.append(
                (values - cf_values).astype(np.float32)
            )
            qualities.append(float(quality))

            if i % 500 == 0 or i == len(images):
                print(
                    f"[counterfactual-fit] {i}/{len(images)}",
                    flush=True,
                )

        D = np.asarray(deltas, dtype=np.float32)
        qualities = np.asarray(qualities, dtype=np.float32)

        pos = D[labels == 1]
        neg = D[labels == 0]

        pmed = np.median(pos, axis=0)
        nmed = np.median(neg, axis=0)
        diff = pmed - nmed
        direction = np.where(diff >= 0, 1.0, -1.0).astype(
            np.float32
        )

        po = pos * direction[None, :]
        no = neg * direction[None, :]
        pmed_o = np.median(po, axis=0)
        nmed_o = np.median(no, axis=0)

        midpoint = 0.5 * (pmed_o + nmed_o)
        scale = 0.5 * (
            self._robust_scale(po)
            + self._robust_scale(no)
        )
        gap = np.maximum(pmed_o - nmed_o, 1e-4)
        scale = np.maximum(scale, 0.20 * gap)
        scale = np.maximum(scale, 1e-3)

        effect = gap / scale

        # Matched-pair stability of LOCAL counterfactual displacement.
        stability = np.full(
            D.shape[1],
            0.5,
            dtype=np.float32,
        )

        matched_pairs = 0
        if pair_ids is not None:
            pair_ids = np.asarray(pair_ids, dtype=object)
            pos_map = {
                str(pid): i
                for i, (pid, y) in enumerate(
                    zip(pair_ids, labels)
                )
                if y == 1
            }
            neg_map = {
                str(pid): i
                for i, (pid, y) in enumerate(
                    zip(pair_ids, labels)
                )
                if y == 0
            }
            common = sorted(set(pos_map) & set(neg_map))
            matched_pairs = len(common)

            if matched_pairs:
                pi = np.asarray(
                    [pos_map[p] for p in common],
                    dtype=int,
                )
                ni = np.asarray(
                    [neg_map[p] for p in common],
                    dtype=int,
                )
                pdiff = (
                    D[pi] - D[ni]
                ) * direction[None, :]
                stability = np.mean(
                    pdiff > 0,
                    axis=0,
                ).astype(np.float32)

        stability_conf = np.clip(
            2.0 * (stability - 0.5),
            0.0,
            1.0,
        )
        rank_score = (
            effect
            * (0.20 + 0.80 * stability_conf)
        )

        eligible = np.where(
            stability
            >= self.counterfactual_min_stability
        )[0]

        order = np.argsort(rank_score)[::-1]

        if len(eligible) >= 24:
            elig_order = eligible[
                np.argsort(rank_score[eligible])[::-1]
            ]
            chosen = elig_order[
                : min(
                    self.counterfactual_max_constraints,
                    len(elig_order),
                )
            ]
        else:
            chosen = order[
                : min(
                    self.counterfactual_max_constraints,
                    len(order),
                )
            ]

        chosen = np.asarray(chosen, dtype=int)

        weights = np.maximum(
            rank_score[chosen],
            1e-6,
        )
        weights = weights / (weights.sum() + 1e-12)

        self.cf_local_selected = chosen
        self.cf_direction = direction[chosen]
        self.cf_midpoint = midpoint[chosen].astype(
            np.float32
        )
        self.cf_scale = scale[chosen].astype(np.float32)
        self.cf_weights = weights.astype(np.float32)
        self.cf_effect = effect[chosen].astype(np.float32)
        self.cf_pair_stability = stability[chosen].astype(
            np.float32
        )

        def score_batch(X):
            xo = (
                X[:, chosen]
                * self.cf_direction[None, :]
            )
            margin = np.clip(
                (
                    xo
                    - self.cf_midpoint[None, :]
                )
                / self.cf_scale[None, :],
                -6.0,
                6.0,
            )
            local = np.tanh(margin)
            return local @ self.cf_weights

        ps = score_batch(D[labels == 1])
        ns = score_batch(D[labels == 0])
        train_acc = float(
            (
                np.sum(ps > 0)
                + np.sum(ns < 0)
            )
            / (len(ps) + len(ns))
        )

        self.cf_train_accuracy = train_acc
        self.cf_train_quality_median = float(
            np.median(qualities)
        )

        self.cf_diagnostics = {
            "matched_pairs": int(matched_pairs),
            "selected_constraints": int(len(chosen)),
            "effect_max": float(
                np.max(effect[chosen])
            ),
            "effect_p95": float(
                np.percentile(effect[chosen], 95)
            ),
            "effect_median": float(
                np.median(effect[chosen])
            ),
            "pair_stability_max": float(
                np.max(stability[chosen])
            ),
            "pair_stability_p95": float(
                np.percentile(stability[chosen], 95)
            ),
            "pair_stability_median": float(
                np.median(stability[chosen])
            ),
            "train_accuracy_full_observability": train_acc,
            "counterfactual_quality_median": float(
                self.cf_train_quality_median
            ),
            "quality_p10": float(
                np.percentile(qualities, 10)
            ),
            "quality_p90": float(
                np.percentile(qualities, 90)
            ),
        }

        print(
            "SCLI v15 counterfactual delta layer: "
            f"constraints={len(chosen)}, "
            f"train_acc={train_acc:.3f}, "
            f"pair_stability_p95="
            f"{self.cf_diagnostics['pair_stability_p95']:.3f}, "
            f"cf_quality_med="
            f"{self.cf_train_quality_median:.3f}"
        )

        return self.cf_diagnostics

    def fit(
        self,
        images: list[np.ndarray],
        labels: np.ndarray,
        pair_ids: list[str] | np.ndarray | None = None,
    ):
        labels = np.asarray(labels, dtype=int)

        # v13 discovers the strong explicit contextual/predictive constraints.
        super().fit(
            images,
            labels,
            pair_ids=pair_ids,
        )

        cf_diag = self._fit_counterfactual_layer(
            images,
            labels,
            None
            if pair_ids is None
            else np.asarray(pair_ids, dtype=object),
        )

        self.representation_mode = (
            "local_counterfactual_clear_v15"
        )

        final = self.fit_diagnostics.get(
            "final_representation",
            {},
        )
        final["mode"] = self.representation_mode
        final["counterfactual_delta_layer"] = cf_diag
        self.fit_diagnostics[
            "final_representation"
        ] = final
        self.fit_diagnostics[
            "local_counterfactual_reference"
        ] = True

        return self

    def score_one(self, image: np.ndarray) -> dict[str, Any]:
        if (
            self.cf_local_selected is None
            or self.cf_direction is None
            or self.cf_midpoint is None
            or self.cf_scale is None
            or self.cf_weights is None
        ):
            return super().score_one(image)

        values, support, coeff = self._selected_observables(
            image
        )

        cf, cf_quality, cf_meta = (
            self._counterfactual_clear(image)
        )
        cf_values, _, _ = self._selected_observables(
            cf
        )

        local_idx = self.cf_local_selected

        delta = (
            values[local_idx]
            - cf_values[local_idx]
        )

        sigma = self._noise_sigma(image)
        structural_snr = self._structural_snr(image)

        threshold = np.maximum(
            self.obs_sigma_factor
            * coeff[local_idx]
            * sigma,
            self.obs_reference_fraction
            * self.reference_support[local_idx],
        )

        # A local delta is meaningful only where the original measurement
        # itself was physically observable.
        observable = (
            support[local_idx]
            >= threshold
        )

        if not observable.any():
            return {
                "score": 0.5,
                "coverage": 0.0,
                "counterfactual_quality": cf_quality,
                "constraint_agreement": 0.0,
                "evidence_strength": 0.0,
                "noise_sigma": sigma,
                "n_observable": 0,
                "structural_snr": structural_snr,
                **cf_meta,
            }

        oriented = (
            delta
            * self.cf_direction
        )
        margin = np.clip(
            (
                oriented
                - self.cf_midpoint
            )
            / self.cf_scale,
            -6.0,
            6.0,
        )
        local = np.tanh(margin)

        w = self.cf_weights
        ow = w[observable]
        ol = local[observable]

        observable_mass = float(ow.sum())

        consensus = float(
            np.sum(ow * ol)
            / (observable_mass + 1e-12)
        )

        signed_mass = float(
            np.sum(
                ow * np.sign(ol)
            )
        )
        agreement = float(
            abs(signed_mass)
            / (observable_mass + 1e-12)
        )

        strength = float(
            np.sum(
                ow * np.abs(ol)
            )
            / (observable_mass + 1e-12)
        )

        # Counterfactual quality is part of epistemic coverage, never of
        # identity score. Poor local CLEAR_0 construction -> abstention.
        coverage = float(
            np.clip(
                observable_mass
                * agreement
                * strength
                * cf_quality,
                0.0,
                1.0,
            )
        )

        score = float(
            0.5 * (consensus + 1.0)
        )

        return {
            "score": score,
            "coverage": coverage,
            "counterfactual_quality": cf_quality,
            "constraint_agreement": agreement,
            "evidence_strength": strength,
            "observable_weight_mass": observable_mass,
            "noise_sigma": sigma,
            "n_observable": int(
                observable.sum()
            ),
            "structural_snr": structural_snr,
            **cf_meta,
        }

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({
            "counterfactual_train_accuracy": (
                self.cf_train_accuracy
            ),
            "counterfactual_train_quality_median": (
                self.cf_train_quality_median
            ),
            "counterfactual_diagnostics": (
                self.cf_diagnostics
            ),
        })
        return d


__all__ = ["Gate", "SCLIVisionBinary"]

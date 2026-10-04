from __future__ import annotations

from typing import Any

import numpy as np

from .model_v13 import Gate, SCLIVisionBinary as _BaseSCLI


class SCLIVisionBinary(_BaseSCLI):
    """SCLI Vision v14: paired monotonic invariant consensus.

    v13 established that many explicit constraints move in a stable direction
    inside matched OBJECT/CLEAR scene pairs. v14 uses that discovery directly.

    For each selected constraint i:
      direction_i = sign(median(C_object - C_clear))
      midpoint_i  = midpoint of oriented OBJECT/CLEAR medians
      scale_i     = robust within-class scale

    At inference, only observable constraints vote. Their oriented margins are
    converted into bounded evidence and combined with paired-stability weights.
    This removes the brittle conjunction-rule bottleneck while preserving
    explicit constraints, observability, and abstention.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monotonic_direction: np.ndarray | None = None
        self.monotonic_midpoint: np.ndarray | None = None
        self.monotonic_scale: np.ndarray | None = None
        self.monotonic_weights: np.ndarray | None = None
        self.monotonic_train_accuracy: float | None = None
        self.monotonic_train_coverage: float | None = None
        self.monotonic_pair_stability_median: float | None = None

    @staticmethod
    def _mad(x: np.ndarray, axis=0) -> np.ndarray:
        med = np.median(x, axis=axis)
        return (
            1.4826
            * np.median(
                np.abs(x - np.expand_dims(med, axis=axis)),
                axis=axis,
            )
            + 1e-3
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
        # v13 performs paired feature selection. Its rule layer is retained
        # only as a diagnostic; v14 does not use those rules for inference.
        diag = super()._fit_bank(
            pos_values,
            neg_values,
            pos_support,
            neg_support,
            noise_coeff,
            names,
            mode,
        )

        sel = np.asarray(self.selected, dtype=int)
        p = pos_values[:, sel]
        n = neg_values[:, sel]

        paired = self._paired_feature_scores(
            pos_values,
            neg_values,
            names,
        )

        if paired is not None:
            direction = paired["direction"][sel].astype(np.float32)
            stability = paired["direction_stability"][sel].astype(np.float32)
            pair_score = paired["score"][sel].astype(np.float32)
        else:
            d = np.median(p, axis=0) - np.median(n, axis=0)
            direction = np.where(d >= 0, 1.0, -1.0).astype(np.float32)
            stability = np.full(len(sel), 0.5, dtype=np.float32)
            pair_score = np.abs(d).astype(np.float32)

        po = p * direction[None, :]
        no = n * direction[None, :]

        pmed = np.median(po, axis=0)
        nmed = np.median(no, axis=0)

        # Ensure positive class is above negative after orientation.
        flip = pmed < nmed
        if np.any(flip):
            direction[flip] *= -1.0
            po[:, flip] *= -1.0
            no[:, flip] *= -1.0
            pmed = np.median(po, axis=0)
            nmed = np.median(no, axis=0)

        midpoint = 0.5 * (pmed + nmed)

        pscale = self._mad(po, axis=0)
        nscale = self._mad(no, axis=0)
        scale = 0.5 * (pscale + nscale)

        # Also respect the actual class-median gap so extremely tiny scales
        # cannot turn numerical noise into huge evidence.
        gap = np.maximum(pmed - nmed, 1e-4)
        scale = np.maximum(scale, 0.20 * gap)
        scale = np.maximum(scale, 1e-3)

        stability_conf = np.clip(
            2.0 * (stability - 0.5),
            0.0,
            1.0,
        )
        weights = (
            np.maximum(pair_score, 1e-6)
            * (0.25 + 0.75 * stability_conf)
        )
        weights = weights / (weights.sum() + 1e-12)

        self.monotonic_direction = direction.astype(np.float32)
        self.monotonic_midpoint = midpoint.astype(np.float32)
        self.monotonic_scale = scale.astype(np.float32)
        self.monotonic_weights = weights.astype(np.float32)
        self.monotonic_pair_stability_median = float(
            np.median(stability)
        )

        # Training diagnostic using all constraints as observable. This is
        # not certification; robust validation remains authoritative.
        def batch_score(X):
            oriented = X * direction[None, :]
            margin = np.clip(
                (oriented - midpoint[None, :])
                / scale[None, :],
                -6.0,
                6.0,
            )
            local = np.tanh(margin)
            consensus = local @ weights
            return consensus

        ps = batch_score(p)
        ns = batch_score(n)
        train_acc = (
            np.sum(ps > 0.0) + np.sum(ns < 0.0)
        ) / (len(ps) + len(ns))

        self.monotonic_train_accuracy = float(train_acc)
        self.monotonic_train_coverage = 1.0

        self.representation_mode = "paired_monotonic_consensus_v14"
        diag["mode"] = self.representation_mode
        diag["paired_monotonic_layer"] = {
            "n_constraints": int(len(sel)),
            "train_accuracy_full_observability": float(train_acc),
            "paired_stability_median": float(
                self.monotonic_pair_stability_median
            ),
            "weight_max": float(weights.max()),
            "weight_p95": float(np.percentile(weights, 95)),
            "median_class_gap_in_scales": float(
                np.median(gap / scale)
            ),
        }

        print(
            "SCLI v14 paired monotonic consensus: "
            f"constraints={len(sel)}, "
            f"train_acc={train_acc:.3f}, "
            f"stability_median="
            f"{self.monotonic_pair_stability_median:.3f}"
        )
        return diag

    def score_one(self, image: np.ndarray) -> dict[str, Any]:
        if (
            self.monotonic_direction is None
            or self.monotonic_midpoint is None
            or self.monotonic_scale is None
            or self.monotonic_weights is None
        ):
            return super().score_one(image)

        values, support, coeff = self._selected_observables(image)
        sigma = self._noise_sigma(image)
        structural_snr = self._structural_snr(image)

        threshold = np.maximum(
            self.obs_sigma_factor * coeff * sigma,
            self.obs_reference_fraction * self.reference_support,
        )
        observable = support >= threshold

        if not observable.any():
            return {
                "score": 0.5,
                "coverage": 0.0,
                "constraint_agreement": 0.0,
                "evidence_strength": 0.0,
                "noise_sigma": sigma,
                "n_observable": 0,
                "structural_snr": structural_snr,
            }

        oriented = values * self.monotonic_direction
        margin = np.clip(
            (oriented - self.monotonic_midpoint)
            / self.monotonic_scale,
            -6.0,
            6.0,
        )
        local = np.tanh(margin)

        w = self.monotonic_weights
        ow = w[observable]
        ol = local[observable]

        observable_mass = float(ow.sum())
        if observable_mass <= 1e-12:
            return {
                "score": 0.5,
                "coverage": 0.0,
                "constraint_agreement": 0.0,
                "evidence_strength": 0.0,
                "noise_sigma": sigma,
                "n_observable": int(observable.sum()),
                "structural_snr": structural_snr,
            }

        # Normalize among observable evidence for identity estimate.
        consensus = float(
            np.sum(ow * ol) / (observable_mass + 1e-12)
        )

        # Agreement penalizes mixed-sign evidence. It is 1 when all
        # observable constraints point the same way and approaches 0 when
        # their weighted votes cancel.
        signed_mass = float(np.sum(ow * np.sign(ol)))
        agreement = float(
            abs(signed_mass) / (observable_mass + 1e-12)
        )

        # Evidence strength asks whether margins are materially away from the
        # learned decision midpoints, not merely barely on one side.
        strength = float(
            np.sum(ow * np.abs(ol))
            / (observable_mass + 1e-12)
        )

        # Epistemic coverage combines physical observability with consistency
        # and actual margin strength. Missing evidence cannot be amplified.
        coverage = float(
            np.clip(
                observable_mass * agreement * strength,
                0.0,
                1.0,
            )
        )

        # Convert bounded consensus [-1,1] to probability-like score [0,1].
        score = float(0.5 * (consensus + 1.0))

        return {
            "score": score,
            "coverage": coverage,
            "constraint_agreement": agreement,
            "evidence_strength": strength,
            "observable_weight_mass": observable_mass,
            "noise_sigma": sigma,
            "n_observable": int(observable.sum()),
            "structural_snr": structural_snr,
        }

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({
            "monotonic_train_accuracy": self.monotonic_train_accuracy,
            "monotonic_pair_stability_median": (
                self.monotonic_pair_stability_median
            ),
        })
        return d


__all__ = ["Gate", "SCLIVisionBinary"]

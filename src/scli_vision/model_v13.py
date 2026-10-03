from __future__ import annotations

from typing import Any

import numpy as np

from .model_v12 import Gate, SCLIVisionBinary as _BaseSCLI


class SCLIVisionBinary(_BaseSCLI):
    """SCLI Vision v13: paired invariant discovery.

    Training examples in the contextual ExDark benchmark carry pair_id:
    OBJECT_PRESENT and BACKGROUND_CLEAR samples from the same original scene.

    v13 uses those matched pairs to discover constraints whose change is
    consistent within-scene:

        delta C_i(pair) = C_i(object) - C_i(clear)

    This suppresses scene/camera/illumination nuisance during representation
    selection. The downstream rule system still receives only explicit
    observable constraints at inference time; pair_id is TRAIN-ONLY metadata.
    """

    def __init__(
        self,
        *args,
        paired_min_direction_stability: float = 0.58,
        paired_candidate_cap: int = 420,
        paired_candidate_floor: int = 140,
        **kwargs,
    ):
        kwargs["max_constraints"] = max(
            int(kwargs.get("max_constraints", 380)),
            340,
        )
        super().__init__(*args, **kwargs)

        self.paired_min_direction_stability = paired_min_direction_stability
        self.paired_candidate_cap = paired_candidate_cap
        self.paired_candidate_floor = paired_candidate_floor

        self._pending_pos_pair_ids: list[str] | None = None
        self._pending_neg_pair_ids: list[str] | None = None
        self.paired_diagnostics: dict[str, Any] = {}
        self.paired_feature_original_indices: np.ndarray | None = None

    def fit(
        self,
        images: list[np.ndarray],
        labels: np.ndarray,
        pair_ids: list[str] | np.ndarray | None = None,
    ):
        labels = np.asarray(labels, dtype=int)

        if pair_ids is None:
            print(
                "WARNING: v13 received no pair_ids; falling back to v12 "
                "unpaired representation discovery."
            )
            return super().fit(images, labels)

        pair_ids = np.asarray(pair_ids, dtype=object)
        if len(pair_ids) != len(labels):
            raise ValueError("pair_ids must match labels length.")

        self._pending_pos_pair_ids = [
            str(pid)
            for pid, y in zip(pair_ids, labels)
            if y == 1
        ]
        self._pending_neg_pair_ids = [
            str(pid)
            for pid, y in zip(pair_ids, labels)
            if y == 0
        ]

        result = super().fit(images, labels)

        # Pair metadata is no longer needed after training.
        self._pending_pos_pair_ids = None
        self._pending_neg_pair_ids = None
        return result

    @staticmethod
    def _robust_scale(x: np.ndarray, axis=0) -> np.ndarray:
        med = np.median(x, axis=axis)
        mad = np.median(
            np.abs(x - np.expand_dims(med, axis=axis)),
            axis=axis,
        )
        return 1.4826 * mad + 1e-3

    def _paired_feature_scores(
        self,
        pos_values: np.ndarray,
        neg_values: np.ndarray,
        names: list[str],
    ):
        if (
            self._pending_pos_pair_ids is None
            or self._pending_neg_pair_ids is None
        ):
            return None

        pos_map = {
            pid: i
            for i, pid in enumerate(self._pending_pos_pair_ids)
        }
        neg_map = {
            pid: i
            for i, pid in enumerate(self._pending_neg_pair_ids)
        }
        common = sorted(set(pos_map) & set(neg_map))

        if len(common) < 40:
            return None

        pi = np.asarray([pos_map[p] for p in common], dtype=int)
        ni = np.asarray([neg_map[p] for p in common], dtype=int)

        p = pos_values[pi]
        n = neg_values[ni]
        delta = p - n

        pos_fraction = np.mean(delta > 0, axis=0)
        neg_fraction = np.mean(delta < 0, axis=0)
        direction_stability = np.maximum(
            pos_fraction,
            neg_fraction,
        )
        direction = np.where(
            pos_fraction >= neg_fraction,
            1.0,
            -1.0,
        )

        median_delta = np.median(delta, axis=0)

        # Normalize paired shift by robust within-class feature scale. This
        # prevents tiny numerically stable changes from ranking as important.
        p_scale = self._robust_scale(p, axis=0)
        n_scale = self._robust_scale(n, axis=0)
        pooled_scale = 0.5 * (p_scale + n_scale) + 1e-3
        normalized_shift = np.abs(median_delta) / pooled_scale

        # How far direction consistency is above chance.
        direction_conf = np.clip(
            2.0 * (direction_stability - 0.5),
            0.0,
            1.0,
        )

        # A constraint is useful if its object-vs-clear change is both large
        # and directionally consistent across matched scenes.
        score = normalized_shift * direction_conf

        # Extra paired rank statistic: how often object exceeds its matched
        # clear counterpart in the learned direction.
        signed_delta = direction[None, :] * delta
        margin_positive = np.mean(signed_delta > 0, axis=0)

        return {
            "common_pairs": common,
            "direction_stability": direction_stability,
            "direction": direction,
            "median_delta": median_delta,
            "normalized_shift": normalized_shift,
            "direction_conf": direction_conf,
            "score": score,
            "margin_positive": margin_positive,
            "names": names,
        }

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
        paired = self._paired_feature_scores(
            pos_values,
            neg_values,
            names,
        )

        if paired is None:
            diag = super()._fit_bank(
                pos_values,
                neg_values,
                pos_support,
                neg_support,
                noise_coeff,
                names,
                mode,
            )
            self.representation_mode = (
                "paired_invariant_constraints_v13_fallback"
            )
            diag["mode"] = self.representation_mode
            diag["paired_training_used"] = False
            return diag

        stability = paired["direction_stability"]
        score = paired["score"]

        eligible = np.where(
            stability >= self.paired_min_direction_stability
        )[0]

        # Rank by matched-pair effect, not by global class separation.
        rank = np.argsort(score)[::-1]
        strong_rank = [
            int(i)
            for i in rank
            if stability[i] >= self.paired_min_direction_stability
            and score[i] > 0
        ]

        keep_n = min(
            self.paired_candidate_cap,
            max(
                self.paired_candidate_floor,
                min(len(strong_rank), self.paired_candidate_cap),
            ),
            len(rank),
        )

        if len(strong_rank) >= self.paired_candidate_floor:
            candidate = np.asarray(
                strong_rank[:keep_n],
                dtype=int,
            )
        else:
            # Preserve scientific continuity: if few features meet the
            # stability rule, fill the candidate bank with the strongest
            # paired scores, but record this explicitly.
            candidate = rank[:keep_n].astype(int)

        self.paired_feature_original_indices = candidate.copy()

        psub = pos_values[:, candidate]
        nsub = neg_values[:, candidate]
        pssub = pos_support[:, candidate]
        nssub = neg_support[:, candidate]
        csub = noise_coeff[candidate]
        nmsub = [names[i] for i in candidate]

        diag = super()._fit_bank(
            psub,
            nsub,
            pssub,
            nssub,
            csub,
            nmsub,
            mode,
        )

        # super() selected indices are relative to candidate. Remap them to
        # the full observable vector so inference extracts the same features.
        local_selected = np.asarray(self.selected, dtype=int)
        full_selected = candidate[local_selected]
        self.selected = full_selected

        selected_pair_score = score[full_selected]
        selected_stability = stability[full_selected]

        self.paired_diagnostics = {
            "paired_training_used": True,
            "matched_pairs": int(len(paired["common_pairs"])),
            "features_total": int(len(names)),
            "paired_stability_eligible": int(len(eligible)),
            "candidate_features": int(len(candidate)),
            "paired_score_max": float(np.max(score)),
            "paired_score_p95": float(np.percentile(score, 95)),
            "paired_score_median": float(np.median(score)),
            "direction_stability_max": float(np.max(stability)),
            "direction_stability_p95": float(
                np.percentile(stability, 95)
            ),
            "direction_stability_median": float(
                np.median(stability)
            ),
            "selected_pair_score_median": float(
                np.median(selected_pair_score)
            ),
            "selected_direction_stability_median": float(
                np.median(selected_stability)
            ),
            "selected_examples": [
                {
                    "name": names[int(i)],
                    "score": float(score[int(i)]),
                    "direction_stability": float(stability[int(i)]),
                    "direction": (
                        "OBJECT>CLEAR"
                        if paired["direction"][int(i)] > 0
                        else "OBJECT<CLEAR"
                    ),
                }
                for i in full_selected[
                    np.argsort(score[full_selected])[-12:]
                ][::-1]
            ],
        }

        self.representation_mode = "paired_invariant_constraints_v13"
        diag["mode"] = self.representation_mode
        diag["paired_training"] = self.paired_diagnostics

        print(
            "SCLI v13 paired discovery: "
            f"pairs={self.paired_diagnostics['matched_pairs']}, "
            f"eligible={self.paired_diagnostics['paired_stability_eligible']}, "
            f"pair_score_p95={self.paired_diagnostics['paired_score_p95']:.3f}, "
            f"stability_p95="
            f"{self.paired_diagnostics['direction_stability_p95']:.3f}"
        )
        return diag

    def _selected_observables(self, image: np.ndarray):
        # model_v10 uses rule_source_mode to choose the underlying observable
        # generator. v13 public representation_mode is new, but the rules were
        # still built from the contextual structural bank.
        v, sup, coeff, _ = self._structural_observables(image)
        return (
            v[self.selected],
            sup[self.selected],
            coeff[self.selected],
        )

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["paired_diagnostics"] = self.paired_diagnostics
        return d


__all__ = ["Gate", "SCLIVisionBinary"]

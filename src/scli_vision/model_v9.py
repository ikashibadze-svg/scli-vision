from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .model_v7 import Gate, SCLIVisionBinary as _BaseSCLI


class SCLIVisionBinary(_BaseSCLI):
    """SCLI Vision v9: explicit conjunctions over contextual constraints.

    v8 showed that a single linear direction through the constraint space is
    too weak. v9 keeps the strong v7 contextual representation and mines
    explicit high-precision rules:

        (C_i op t_i) AND (C_j op t_j) [AND (C_k op t_k)] -> OBJECT/CLEAR

    A rule can vote only when every constraint appearing in that rule is
    physically observable. Missing evidence therefore removes the rule rather
    than being imputed or amplified.
    """

    def __init__(
        self,
        *args,
        max_rules_per_class: int = 120,
        atomic_pool_per_class: int = 70,
        pair_pool_per_class: int = 160,
        min_atomic_precision: float = 0.58,
        min_pair_precision: float = 0.82,
        min_triple_precision: float = 0.90,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_rules_per_class = max_rules_per_class
        self.atomic_pool_per_class = atomic_pool_per_class
        self.pair_pool_per_class = pair_pool_per_class
        self.min_atomic_precision = min_atomic_precision
        self.min_pair_precision = min_pair_precision
        self.min_triple_precision = min_triple_precision

        self.rules: list[dict[str, Any]] = []
        self.rule_source_mode: str | None = None
        self.rule_total_weight: float = 0.0
        self.rule_train_accuracy: float | None = None
        self.rule_train_activation: float | None = None
        self.rule_activation_reference: float = 1.0
        self.rule_activation_p90: float = 1.0

    @staticmethod
    def _atom_mask(X: np.ndarray, atom: dict[str, Any]) -> np.ndarray:
        j = atom["feature"]
        if atom["op"] == ">=":
            return X[:, j] >= atom["threshold"]
        return X[:, j] <= atom["threshold"]

    @staticmethod
    def _rule_mask_from_atom_masks(
        atom_masks: list[np.ndarray],
        atom_ids: tuple[int, ...],
    ) -> np.ndarray:
        m = atom_masks[atom_ids[0]].copy()
        for ai in atom_ids[1:]:
            m &= atom_masks[ai]
        return m

    def _mine_rules(
        self,
        pos_values: np.ndarray,
        neg_values: np.ndarray,
    ) -> dict[str, Any]:
        X = np.concatenate([pos_values, neg_values], axis=0).astype(np.float32)
        y = np.concatenate([
            np.ones(len(pos_values), dtype=np.int8),
            np.zeros(len(neg_values), dtype=np.int8),
        ])

        quantiles = np.asarray(
            [0.12, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.88],
            dtype=float,
        )

        all_rules: list[dict[str, Any]] = []
        class_diags = {}

        for target in (0, 1):
            n_target = int(np.sum(y == target))
            min_atom_target = max(24, int(round(0.025 * n_target)))
            min_pair_target = max(18, int(round(0.018 * n_target)))
            min_triple_target = max(12, int(round(0.012 * n_target)))

            # ----------------------------------------------------------
            # 1. Atomic literal pool. Keep only the best threshold for a
            #    feature/direction/target so later conjunctions are diverse.
            # ----------------------------------------------------------
            atoms: list[dict[str, Any]] = []
            atom_masks: list[np.ndarray] = []

            for j in range(X.shape[1]):
                ts = np.unique(np.quantile(X[:, j], quantiles))
                for op in (">=", "<="):
                    best = None
                    for t in ts:
                        m = X[:, j] >= t if op == ">=" else X[:, j] <= t
                        total = int(m.sum())
                        if total == 0:
                            continue
                        tp = int(np.sum(m & (y == target)))
                        if tp < min_atom_target:
                            continue
                        precision = tp / total
                        if precision < self.min_atomic_precision:
                            continue
                        support = tp / n_target
                        quality = (precision - 0.5) * np.sqrt(tp) * (0.7 + support)
                        cand = {
                            "feature": int(j),
                            "op": op,
                            "threshold": float(t),
                            "precision": float(precision),
                            "target_support": int(tp),
                            "total_support": int(total),
                            "quality": float(quality),
                        }
                        if best is None or cand["quality"] > best["quality"]:
                            best = cand
                    if best is not None:
                        atoms.append(best)

            atoms.sort(key=lambda a: a["quality"], reverse=True)
            atoms = atoms[: self.atomic_pool_per_class]
            atom_masks = [self._atom_mask(X, a) for a in atoms]

            # High-precision atomic rules can survive directly.
            candidates: list[dict[str, Any]] = []
            for ai, a in enumerate(atoms):
                if a["precision"] >= max(self.min_triple_precision, 0.92):
                    candidates.append({
                        "target": int(target),
                        "atom_ids": (ai,),
                        "atoms": [a],
                        "precision": a["precision"],
                        "target_support": a["target_support"],
                        "total_support": a["total_support"],
                        "quality": a["quality"],
                        "order": 1,
                    })

            # ----------------------------------------------------------
            # 2. Pair conjunctions.
            # ----------------------------------------------------------
            pairs: list[dict[str, Any]] = []
            for a in range(len(atoms)):
                for b in range(a + 1, len(atoms)):
                    if atoms[a]["feature"] == atoms[b]["feature"]:
                        continue
                    m = atom_masks[a] & atom_masks[b]
                    total = int(m.sum())
                    if total == 0:
                        continue
                    tp = int(np.sum(m & (y == target)))
                    if tp < min_pair_target:
                        continue
                    precision = tp / total
                    if precision < self.min_pair_precision:
                        continue
                    support = tp / n_target
                    quality = (precision - 0.5) * np.sqrt(tp) * (1.0 + support)
                    pairs.append({
                        "target": int(target),
                        "atom_ids": (a, b),
                        "atoms": [atoms[a], atoms[b]],
                        "precision": float(precision),
                        "target_support": int(tp),
                        "total_support": int(total),
                        "quality": float(quality),
                        "order": 2,
                        "_mask": m,
                    })

            pairs.sort(key=lambda r: r["quality"], reverse=True)
            pairs = pairs[: self.pair_pool_per_class]

            # Pairs with strong precision are valid final rules.
            candidates.extend([
                {k: v for k, v in r.items() if k != "_mask"}
                for r in pairs
                if r["precision"] >= 0.88
            ])

            # ----------------------------------------------------------
            # 3. Triple conjunctions. Expand only the strongest pairs.
            # ----------------------------------------------------------
            triples: list[dict[str, Any]] = []
            for pr in pairs[: min(60, len(pairs))]:
                used_features = {x["feature"] for x in pr["atoms"]}
                for c in range(min(50, len(atoms))):
                    if atoms[c]["feature"] in used_features:
                        continue
                    m = pr["_mask"] & atom_masks[c]
                    total = int(m.sum())
                    if total == 0:
                        continue
                    tp = int(np.sum(m & (y == target)))
                    if tp < min_triple_target:
                        continue
                    precision = tp / total
                    if precision < self.min_triple_precision:
                        continue
                    support = tp / n_target
                    quality = (
                        (precision - 0.5)
                        * np.sqrt(tp)
                        * (1.15 + support)
                    )
                    triples.append({
                        "target": int(target),
                        "atoms": pr["atoms"] + [atoms[c]],
                        "precision": float(precision),
                        "target_support": int(tp),
                        "total_support": int(total),
                        "quality": float(quality),
                        "order": 3,
                        "_mask": m,
                    })

            triples.sort(key=lambda r: r["quality"], reverse=True)
            candidates.extend([
                {k: v for k, v in r.items() if k != "_mask"}
                for r in triples[: self.max_rules_per_class]
            ])

            # ----------------------------------------------------------
            # Greedy diversity: avoid storing many near-identical rules.
            # Recompute masks, reject Jaccard > .92 against stronger rules.
            # ----------------------------------------------------------
            candidates.sort(key=lambda r: r["quality"], reverse=True)
            kept = []
            kept_masks: list[np.ndarray] = []

            for rule in candidates:
                m = np.ones(len(X), dtype=bool)
                for atom in rule["atoms"]:
                    m &= self._atom_mask(X, atom)

                duplicate = False
                for km in kept_masks:
                    union = int(np.sum(m | km))
                    if union == 0:
                        continue
                    jac = float(np.sum(m & km) / union)
                    if jac > 0.92:
                        duplicate = True
                        break
                if duplicate:
                    continue

                # Precision-derived mass; support prevents tiny perfect rules
                # from dominating.
                precision_margin = max(rule["precision"] - 0.5, 1e-3)
                support_factor = np.log1p(rule["target_support"])
                rule["weight"] = float(precision_margin * support_factor)
                kept.append(rule)
                kept_masks.append(m)

                if len(kept) >= self.max_rules_per_class:
                    break

            all_rules.extend(kept)

            class_diags[str(target)] = {
                "atoms": int(len(atoms)),
                "pairs": int(len(pairs)),
                "triples": int(len(triples)),
                "rules_kept": int(len(kept)),
                "best_rule_precision": float(
                    max((r["precision"] for r in kept), default=np.nan)
                ),
                "median_rule_precision": float(
                    np.median([r["precision"] for r in kept])
                    if kept else np.nan
                ),
            }

        # Balance OBJECT and CLEAR rule families so rule count/support cannot
        # create an implicit class prior.
        for target in (0, 1):
            mass = sum(
                r["weight"] for r in all_rules
                if r["target"] == target
            )
            if mass > 0:
                for r in all_rules:
                    if r["target"] == target:
                        r["weight"] = float(r["weight"] / mass)

        self.rules = all_rules
        self.rule_total_weight = float(
            sum(r["weight"] for r in self.rules)
        )

        # --------------------------------------------------------------
        # Training diagnostic of conjunction decision layer.
        # --------------------------------------------------------------
        pos_mass = np.zeros(len(X), dtype=np.float64)
        neg_mass = np.zeros(len(X), dtype=np.float64)

        for rule in self.rules:
            m = np.ones(len(X), dtype=bool)
            for atom in rule["atoms"]:
                m &= self._atom_mask(X, atom)
            if rule["target"] == 1:
                pos_mass[m] += rule["weight"]
            else:
                neg_mass[m] += rule["weight"]

        total_mass = pos_mass + neg_mass
        active = total_mass > 0
        pred = (pos_mass >= neg_mass).astype(np.int8)

        self.rule_train_activation = float(active.mean())
        self.rule_train_accuracy = float(
            np.mean(pred[active] == y[active])
        ) if active.any() else np.nan

        if active.any():
            active_mass = total_mass[active]
            # Typical evidence mass, not total library mass, defines 1.0
            # epistemic support. Median is robust to a few samples firing
            # very many rules.
            self.rule_activation_reference = float(
                max(np.median(active_mass), 1e-6)
            )
            self.rule_activation_p90 = float(
                max(np.percentile(active_mass, 90), 1e-6)
            )
        else:
            self.rule_activation_reference = 1.0
            self.rule_activation_p90 = 1.0

        return {
            "n_rules": int(len(self.rules)),
            "n_positive_rules": int(
                sum(r["target"] == 1 for r in self.rules)
            ),
            "n_negative_rules": int(
                sum(r["target"] == 0 for r in self.rules)
            ),
            "train_activation": float(self.rule_train_activation),
            "train_accuracy_when_activated": float(self.rule_train_accuracy),
            "activation_reference_median": float(
                self.rule_activation_reference
            ),
            "activation_reference_p90": float(
                self.rule_activation_p90
            ),
            "by_class": class_diags,
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
        diag = super()._fit_bank(
            pos_values,
            neg_values,
            pos_support,
            neg_support,
            noise_coeff,
            names,
            mode,
        )

        # Mine rules only in the already selected explicit constraint space.
        psel = pos_values[:, self.selected]
        nsel = neg_values[:, self.selected]
        rule_diag = self._mine_rules(psel, nsel)

        print(
            "SCLI v9 conjunction rules: "
            f"rules={rule_diag['n_rules']}, "
            f"train_activation={rule_diag['train_activation']:.3f}, "
            f"train_accuracy_when_activated="
            f"{rule_diag['train_accuracy_when_activated']:.3f}, "
            f"activation_reference="
            f"{rule_diag['activation_reference_median']:.5f}"
        )

        self.rule_source_mode = mode
        self.representation_mode = "contextual_conjunction_rules_v9"
        diag["source_mode"] = self.rule_source_mode
        diag["mode"] = self.representation_mode
        diag["rule_layer"] = rule_diag
        return diag

    def _selected_observables(self, image: np.ndarray):
        """Return the selected coordinates from the representation that
        generated the v9 rules, even though public representation_mode is v9.
        """
        if self.rule_source_mode == "role_ratios_v4":
            v, sup, coeff, _ = self._relational_observables(image)
        else:
            # ExDark contextual experiments arrive here.
            v, sup, coeff, _ = self._structural_observables(image)
        return (
            v[self.selected],
            sup[self.selected],
            coeff[self.selected],
        )

    @staticmethod
    def _atom_satisfied(value: float, atom: dict[str, Any]) -> bool:
        if atom["op"] == ">=":
            return value >= atom["threshold"]
        return value <= atom["threshold"]

    def score_one(self, image: np.ndarray) -> dict[str, Any]:
        if self.selected is None:
            raise RuntimeError("Model is not fitted.")

        values, support, coeff = self._selected_observables(image)
        sigma = self._noise_sigma(image)
        structural_snr = self._structural_snr(image)

        threshold = np.maximum(
            self.obs_sigma_factor * coeff * sigma,
            self.obs_reference_fraction * self.reference_support,
        )
        observable = support >= threshold

        if not self.rules:
            return super().score_one(image)

        pos_mass = 0.0
        neg_mass = 0.0
        observable_rule_mass = 0.0
        activated_rule_mass = 0.0
        n_observable_rules = 0
        n_activated_rules = 0

        for rule in self.rules:
            atom_features = [a["feature"] for a in rule["atoms"]]
            if not all(observable[j] for j in atom_features):
                continue

            n_observable_rules += 1
            observable_rule_mass += rule["weight"]

            satisfied = True
            for atom in rule["atoms"]:
                if not self._atom_satisfied(
                    float(values[atom["feature"]]),
                    atom,
                ):
                    satisfied = False
                    break

            if not satisfied:
                continue

            n_activated_rules += 1
            activated_rule_mass += rule["weight"]
            if rule["target"] == 1:
                pos_mass += rule["weight"]
            else:
                neg_mass += rule["weight"]

        # Score is pure rule evidence. No active conjunction -> unknown/neutral.
        total_vote = pos_mass + neg_mass
        if total_vote <= 1e-12:
            score = 0.5
        else:
            score = float(pos_mass / total_vote)

        observable_rule_coverage = float(
            observable_rule_mass / (self.rule_total_weight + 1e-12)
        )

        # v9.1 evidence calibration:
        # normalize active rule mass by the TYPICAL non-zero train activation,
        # not by the entire rule library. Sparse high-precision conjunctions
        # are supposed to activate only a small part of the library.
        activation_strength = float(
            min(
                1.0,
                activated_rule_mass
                / (self.rule_activation_reference + 1e-12),
            )
        )

        # Conflicting OBJECT and CLEAR evidence is epistemically weak even if
        # many rules fired. Agreement=1 for one-sided evidence, 0 for a tie.
        if total_vote <= 1e-12:
            agreement = 0.0
        else:
            agreement = float(
                abs(pos_mass - neg_mass) / (total_vote + 1e-12)
            )

        activation = activation_strength
        coverage = float(activation_strength * agreement)

        return {
            "score": score,
            "coverage": coverage,
            "rule_activation": activation,
            "rule_agreement": agreement,
            "activated_rule_mass": float(activated_rule_mass),
            "observable_rule_coverage": observable_rule_coverage,
            "noise_sigma": sigma,
            "n_observable": int(observable.sum()),
            "n_observable_rules": int(n_observable_rules),
            "n_activated_rules": int(n_activated_rules),
            "structural_snr": structural_snr,
        }

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({
            "rule_count": int(len(self.rules)),
            "rule_train_activation": self.rule_train_activation,
            "rule_train_accuracy": self.rule_train_accuracy,
            "rule_activation_reference": self.rule_activation_reference,
            "rule_activation_p90": self.rule_activation_p90,
        })
        return d


__all__ = ["Gate", "SCLIVisionBinary"]

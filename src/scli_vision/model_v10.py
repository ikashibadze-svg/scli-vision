from __future__ import annotations

from typing import Any

import numpy as np

from .model_v7 import Gate, SCLIVisionBinary as _BaseSCLI


class SCLIVisionBinary(_BaseSCLI):
    """SCLI Vision v10: self-constructing higher-order constraint rules.

    The representation is the explicit contextual constraint graph from v7.
    The decision layer constructs conjunctions up to order 4 with beam search.

    Important safeguards:
      * weak atoms are allowed into search; only the conjunction must be strong;
      * every candidate rule is verified on an internal holdout split of TRAIN;
      * a rule can vote only when every atom is physically observable;
      * no rule -> UNDERDETERMINED;
      * conflicting OBJECT/CLEAR rule mass reduces epistemic coverage;
      * if robust validation cannot certify a gate, the experiment continues
        with an explicit reject-all UNCERTIFIED gate instead of crashing.
    """

    def __init__(
        self,
        *args,
        beam_width: int = 180,
        literal_pool_per_class: int = 140,
        max_rule_depth: int = 4,
        max_rules_per_class: int = 100,
        internal_verify_fraction: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beam_width = beam_width
        self.literal_pool_per_class = literal_pool_per_class
        self.max_rule_depth = max_rule_depth
        self.max_rules_per_class = max_rules_per_class
        self.internal_verify_fraction = internal_verify_fraction

        self.rules: list[dict[str, Any]] = []
        self.rule_source_mode: str | None = None
        self.rule_total_weight = 0.0
        self.rule_activation_reference = 1.0
        self.rule_activation_p90 = 1.0
        self.rule_train_activation: float | None = None
        self.rule_train_accuracy: float | None = None
        self.rule_verify_accuracy: float | None = None

    @staticmethod
    def _literal_mask(X: np.ndarray, lit: dict[str, Any]) -> np.ndarray:
        j = lit["feature"]
        if lit["op"] == ">=":
            return X[:, j] >= lit["threshold"]
        return X[:, j] <= lit["threshold"]

    @staticmethod
    def _safe_precision(tp: int, total: int) -> float:
        return float(tp / total) if total > 0 else 0.0

    @staticmethod
    def _wilson_lower_bound(tp: int, total: int, z: float = 1.28) -> float:
        """Approx. 80% one-sided Wilson lower bound, used only for ranking."""
        if total <= 0:
            return 0.0
        p = tp / total
        z2 = z * z
        den = 1.0 + z2 / total
        center = p + z2 / (2.0 * total)
        rad = z * np.sqrt(
            (p * (1.0 - p) + z2 / (4.0 * total)) / total
        )
        return float((center - rad) / den)

    def _internal_split(self, y: np.ndarray):
        rng = np.random.default_rng(20261004)
        discover = []
        verify = []
        for target in (0, 1):
            idx = np.where(y == target)[0].copy()
            rng.shuffle(idx)
            nv = max(1, int(round(len(idx) * self.internal_verify_fraction)))
            verify.extend(idx[:nv].tolist())
            discover.extend(idx[nv:].tolist())
        return (
            np.asarray(sorted(discover), dtype=int),
            np.asarray(sorted(verify), dtype=int),
        )

    def _rule_mask(
        self,
        X: np.ndarray,
        literals: list[dict[str, Any]],
    ) -> np.ndarray:
        m = np.ones(len(X), dtype=bool)
        for lit in literals:
            m &= self._literal_mask(X, lit)
        return m

    def _build_literal_pool(
        self,
        Xd: np.ndarray,
        yd: np.ndarray,
        target: int,
    ) -> list[dict[str, Any]]:
        """Build a broad literal pool.

        Unlike v9, a literal does NOT need to be strongly predictive alone.
        It only needs a small directional lift and sufficient support, because
        higher-order conjunction may create the actual distinction.
        """
        base = float(np.mean(yd == target))
        n_target = int(np.sum(yd == target))
        min_target_hits = max(10, int(round(0.008 * n_target)))

        qs = np.asarray(
            [0.08, 0.14, 0.20, 0.28, 0.36, 0.45, 0.55, 0.64, 0.72, 0.80, 0.86, 0.92],
            dtype=float,
        )

        out = []
        for j in range(Xd.shape[1]):
            thresholds = np.unique(np.quantile(Xd[:, j], qs))
            for op in (">=", "<="):
                best = None
                for t in thresholds:
                    m = Xd[:, j] >= t if op == ">=" else Xd[:, j] <= t
                    total = int(m.sum())
                    if total == 0:
                        continue
                    tp = int(np.sum(m & (yd == target)))
                    if tp < min_target_hits:
                        continue
                    precision = tp / total
                    lift = precision - base

                    # Permit weak atoms. We only remove atoms that point
                    # meaningfully in the wrong direction.
                    if lift < 0.005:
                        continue

                    target_support = tp / max(n_target, 1)
                    quality = (
                        max(lift, 1e-4)
                        * np.sqrt(tp)
                        * (0.6 + target_support)
                    )
                    cand = {
                        "feature": int(j),
                        "op": op,
                        "threshold": float(t),
                        "precision": float(precision),
                        "target_hits": int(tp),
                        "total_hits": int(total),
                        "lift": float(lift),
                        "quality": float(quality),
                    }
                    if best is None or cand["quality"] > best["quality"]:
                        best = cand

                if best is not None:
                    out.append(best)

        out.sort(key=lambda a: a["quality"], reverse=True)

        # Diversity: cap how many literals one feature can contribute.
        kept = []
        per_feature = {}
        for lit in out:
            j = lit["feature"]
            if per_feature.get(j, 0) >= 2:
                continue
            kept.append(lit)
            per_feature[j] = per_feature.get(j, 0) + 1
            if len(kept) >= self.literal_pool_per_class:
                break
        return kept

    def _evaluate_rule(
        self,
        X: np.ndarray,
        y: np.ndarray,
        literals: list[dict[str, Any]],
        target: int,
    ) -> dict[str, Any]:
        m = self._rule_mask(X, literals)
        total = int(m.sum())
        tp = int(np.sum(m & (y == target)))
        precision = self._safe_precision(tp, total)
        target_n = max(int(np.sum(y == target)), 1)
        support = tp / target_n
        return {
            "mask": m,
            "total": total,
            "tp": tp,
            "precision": precision,
            "target_support": float(support),
            "wilson": self._wilson_lower_bound(tp, total),
        }

    def _search_class_rules(
        self,
        Xd: np.ndarray,
        yd: np.ndarray,
        Xv: np.ndarray,
        yv: np.ndarray,
        target: int,
    ):
        literals = self._build_literal_pool(Xd, yd, target)
        if not literals:
            return [], {
                "literal_pool": 0,
                "accepted": 0,
                "by_depth": {},
            }

        n_target_d = int(np.sum(yd == target))
        n_target_v = int(np.sum(yv == target))

        # Search constraints. These are deliberately not final safety
        # thresholds; robust validation remains the certification layer.
        min_disc_tp = {
            1: max(18, int(0.012 * n_target_d)),
            2: max(14, int(0.010 * n_target_d)),
            3: max(10, int(0.007 * n_target_d)),
            4: max(8, int(0.005 * n_target_d)),
        }
        min_disc_precision = {
            1: 0.54,
            2: 0.60,
            3: 0.66,
            4: 0.70,
        }
        accept_verify_precision = {
            1: 0.76,
            2: 0.80,
            3: 0.84,
            4: 0.86,
        }
        min_verify_tp = {
            1: max(8, int(0.010 * n_target_v)),
            2: max(7, int(0.008 * n_target_v)),
            3: max(6, int(0.006 * n_target_v)),
            4: max(5, int(0.005 * n_target_v)),
        }

        # Beam members contain literal indices and discovery mask/score.
        beam = []
        accepted = []
        by_depth = {}

        # Seed with all broad literals.
        for li, lit in enumerate(literals):
            ev = self._evaluate_rule(Xd, yd, [lit], target)
            if (
                ev["tp"] >= min_disc_tp[1]
                and ev["precision"] >= min_disc_precision[1]
            ):
                score = (
                    (ev["precision"] - 0.5)
                    * np.sqrt(ev["tp"])
                    * (0.8 + ev["target_support"])
                )
                beam.append({
                    "ids": (li,),
                    "literals": [lit],
                    "mask": ev["mask"],
                    "score": float(score),
                    "disc": ev,
                })

        beam.sort(key=lambda r: r["score"], reverse=True)
        beam = beam[: self.beam_width]

        for depth in range(1, self.max_rule_depth + 1):
            if depth > 1:
                next_beam = []
                seen = set()

                # Expand by the top broad literal pool. Feature uniqueness
                # prevents trivial multiple-threshold conjunctions.
                for rule in beam:
                    used_features = {
                        x["feature"] for x in rule["literals"]
                    }
                    last_id = rule["ids"][-1]

                    for li, lit in enumerate(literals):
                        if li <= last_id:
                            continue
                        if lit["feature"] in used_features:
                            continue

                        ids = rule["ids"] + (li,)
                        if ids in seen:
                            continue
                        seen.add(ids)

                        m = rule["mask"] & self._literal_mask(Xd, lit)
                        total = int(m.sum())
                        if total == 0:
                            continue
                        tp = int(np.sum(m & (yd == target)))
                        if tp < min_disc_tp[depth]:
                            continue

                        precision = tp / total
                        if precision < min_disc_precision[depth]:
                            continue

                        support = tp / max(n_target_d, 1)
                        wilson = self._wilson_lower_bound(tp, total)
                        score = (
                            max(wilson - 0.5, 1e-4)
                            * np.sqrt(tp)
                            * (1.0 + 0.4 * depth + support)
                        )

                        next_beam.append({
                            "ids": ids,
                            "literals": rule["literals"] + [lit],
                            "mask": m,
                            "score": float(score),
                            "disc": {
                                "mask": m,
                                "total": total,
                                "tp": tp,
                                "precision": float(precision),
                                "target_support": float(support),
                                "wilson": float(wilson),
                            },
                        })

                next_beam.sort(
                    key=lambda r: r["score"],
                    reverse=True,
                )
                beam = next_beam[: self.beam_width]

            # Verify current beam on the internal holdout.
            depth_accepted = 0
            for rule in beam:
                vv = self._evaluate_rule(
                    Xv, yv, rule["literals"], target
                )
                if (
                    vv["tp"] < min_verify_tp[depth]
                    or vv["precision"] < accept_verify_precision[depth]
                ):
                    continue

                # Verification lower bound influences rule weight. This
                # discourages tiny lucky conjunctions.
                verification_quality = max(vv["wilson"] - 0.5, 1e-4)
                weight = (
                    verification_quality
                    * np.log1p(vv["tp"])
                    * (1.0 + 0.15 * depth)
                )
                accepted.append({
                    "target": int(target),
                    "literals": rule["literals"],
                    "order": int(depth),
                    "discover_precision": float(rule["disc"]["precision"]),
                    "verify_precision": float(vv["precision"]),
                    "verify_wilson": float(vv["wilson"]),
                    "verify_target_hits": int(vv["tp"]),
                    "verify_total_hits": int(vv["total"]),
                    "weight": float(weight),
                })
                depth_accepted += 1

            by_depth[str(depth)] = {
                "beam_size": int(len(beam)),
                "accepted": int(depth_accepted),
            }

            if not beam:
                break

        # Greedy de-duplication on verification activation patterns.
        accepted.sort(
            key=lambda r: (
                r["verify_wilson"],
                r["verify_precision"],
                r["verify_target_hits"],
            ),
            reverse=True,
        )

        kept = []
        masks = []
        for rule in accepted:
            m = self._rule_mask(Xv, rule["literals"])
            duplicate = False
            for km in masks:
                union = int(np.sum(m | km))
                if union == 0:
                    continue
                jaccard = float(np.sum(m & km) / union)
                if jaccard > 0.90:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(rule)
            masks.append(m)
            if len(kept) >= self.max_rules_per_class:
                break

        diag = {
            "literal_pool": int(len(literals)),
            "accepted_before_dedup": int(len(accepted)),
            "accepted": int(len(kept)),
            "by_depth": by_depth,
            "best_verify_precision": float(
                max((r["verify_precision"] for r in kept), default=np.nan)
            ),
            "median_verify_precision": float(
                np.median([r["verify_precision"] for r in kept])
                if kept else np.nan
            ),
        }
        return kept, diag

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

        di, vi = self._internal_split(y)
        Xd, yd = X[di], y[di]
        Xv, yv = X[vi], y[vi]

        rules = []
        by_class = {}
        for target in (0, 1):
            rr, dd = self._search_class_rules(
                Xd, yd, Xv, yv, target
            )
            rules.extend(rr)
            by_class[str(target)] = dd

        # Balance class-family total rule mass.
        for target in (0, 1):
            mass = sum(
                r["weight"] for r in rules
                if r["target"] == target
            )
            if mass > 0:
                for r in rules:
                    if r["target"] == target:
                        r["weight"] = float(r["weight"] / mass)

        self.rules = rules
        self.rule_total_weight = float(
            sum(r["weight"] for r in rules)
        )

        # Diagnostics on all TRAIN data.
        pos_mass = np.zeros(len(X), dtype=np.float64)
        neg_mass = np.zeros(len(X), dtype=np.float64)

        for rule in self.rules:
            m = self._rule_mask(X, rule["literals"])
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

        # Separate internal VERIFY diagnostic.
        pv = np.zeros(len(Xv), dtype=np.float64)
        nv = np.zeros(len(Xv), dtype=np.float64)
        for rule in self.rules:
            m = self._rule_mask(Xv, rule["literals"])
            if rule["target"] == 1:
                pv[m] += rule["weight"]
            else:
                nv[m] += rule["weight"]
        av = (pv + nv) > 0
        predv = (pv >= nv).astype(np.int8)
        self.rule_verify_accuracy = float(
            np.mean(predv[av] == yv[av])
        ) if av.any() else np.nan
        verify_activation = float(av.mean())

        if active.any():
            active_mass = total_mass[active]
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
            "verify_activation": float(verify_activation),
            "verify_accuracy_when_activated": float(self.rule_verify_accuracy),
            "activation_reference_median": float(
                self.rule_activation_reference
            ),
            "activation_reference_p90": float(
                self.rule_activation_p90
            ),
            "by_class": by_class,
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

        psel = pos_values[:, self.selected]
        nsel = neg_values[:, self.selected]
        rule_diag = self._mine_rules(psel, nsel)

        print(
            "SCLI v10 self-constructed rules: "
            f"rules={rule_diag['n_rules']}, "
            f"train_activation={rule_diag['train_activation']:.3f}, "
            f"train_acc={rule_diag['train_accuracy_when_activated']:.3f}, "
            f"verify_activation={rule_diag['verify_activation']:.3f}, "
            f"verify_acc={rule_diag['verify_accuracy_when_activated']:.3f}"
        )

        self.rule_source_mode = mode
        self.representation_mode = "self_constructed_constraints_v10"
        diag["source_mode"] = mode
        diag["mode"] = self.representation_mode
        diag["self_constructing_rule_layer"] = rule_diag
        return diag

    def _selected_observables(self, image: np.ndarray):
        if self.rule_source_mode == "role_ratios_v4":
            v, sup, coeff, _ = self._relational_observables(image)
        else:
            v, sup, coeff, _ = self._structural_observables(image)
        return (
            v[self.selected],
            sup[self.selected],
            coeff[self.selected],
        )

    @staticmethod
    def _literal_satisfied(
        value: float,
        lit: dict[str, Any],
    ) -> bool:
        if lit["op"] == ">=":
            return value >= lit["threshold"]
        return value <= lit["threshold"]

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
            return {
                "score": 0.5,
                "coverage": 0.0,
                "rule_activation": 0.0,
                "rule_agreement": 0.0,
                "noise_sigma": sigma,
                "n_observable": int(observable.sum()),
                "n_observable_rules": 0,
                "n_activated_rules": 0,
                "structural_snr": structural_snr,
            }

        pos_mass = 0.0
        neg_mass = 0.0
        activated_mass = 0.0
        observable_mass = 0.0
        n_observable_rules = 0
        n_activated_rules = 0

        for rule in self.rules:
            feats = [lit["feature"] for lit in rule["literals"]]
            if not all(observable[j] for j in feats):
                continue

            n_observable_rules += 1
            observable_mass += rule["weight"]

            satisfied = True
            for lit in rule["literals"]:
                if not self._literal_satisfied(
                    float(values[lit["feature"]]),
                    lit,
                ):
                    satisfied = False
                    break
            if not satisfied:
                continue

            n_activated_rules += 1
            activated_mass += rule["weight"]
            if rule["target"] == 1:
                pos_mass += rule["weight"]
            else:
                neg_mass += rule["weight"]

        total_vote = pos_mass + neg_mass
        if total_vote <= 1e-12:
            score = 0.5
            agreement = 0.0
        else:
            score = float(pos_mass / total_vote)
            agreement = float(
                abs(pos_mass - neg_mass) / (total_vote + 1e-12)
            )

        activation_strength = float(
            min(
                1.0,
                activated_mass
                / (self.rule_activation_reference + 1e-12),
            )
        )
        coverage = float(activation_strength * agreement)

        return {
            "score": score,
            "coverage": coverage,
            "rule_activation": activation_strength,
            "rule_agreement": agreement,
            "activated_rule_mass": float(activated_mass),
            "observable_rule_coverage": float(
                observable_mass / (self.rule_total_weight + 1e-12)
            ),
            "noise_sigma": sigma,
            "n_observable": int(observable.sum()),
            "n_observable_rules": int(n_observable_rules),
            "n_activated_rules": int(n_activated_rules),
            "structural_snr": structural_snr,
        }

    def calibrate(self, *args, **kwargs):
        """Robust calibration that never crashes the benchmark."""
        try:
            return super().calibrate(*args, **kwargs)
        except RuntimeError as e:
            if "No non-empty epistemic gate" not in str(e):
                raise

            target = float(kwargs.get("target_known_accuracy", 0.90))
            self.gate = Gate(
                positive_min=1.01,
                negative_max=-0.01,
                coverage_min=1.01,
                snr_min=float("inf"),
                validation_coverage=0.0,
                validation_known_accuracy=float("nan"),
                certified=False,
                target_known_accuracy=target,
            )
            print(
                "WARNING: no non-empty robust gate exists. "
                "Using explicit reject-all UNCERTIFIED gate so held-out "
                "evaluation can continue."
            )
            return self.gate

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({
            "rule_count": int(len(self.rules)),
            "rule_train_activation": self.rule_train_activation,
            "rule_train_accuracy": self.rule_train_accuracy,
            "rule_verify_accuracy": self.rule_verify_accuracy,
            "rule_activation_reference": self.rule_activation_reference,
            "rule_activation_p90": self.rule_activation_p90,
        })
        return d


__all__ = ["Gate", "SCLIVisionBinary"]

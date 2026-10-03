from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, laplace

from .model_v10 import Gate, SCLIVisionBinary as _BaseSCLI


class SCLIVisionBinary(_BaseSCLI):
    """SCLI Vision v11: dual-evidence object / clear reasoning.

    OBJECT and CLEAR are not complements.

    OBJECT evidence:
        explicit contextual discontinuity / structure constraints.

    CLEAR evidence:
        explicit positive continuity constraints showing that the candidate
        region behaves like a continuation of its own surrounding scene.

    This directly implements:
        absence(object evidence) != evidence(clear).
    """

    def __init__(self, *args, **kwargs):
        # v11 adds a small bank of dedicated CLEAR constraints. Keep enough
        # selected coordinates so those constraints are not squeezed out by
        # the already-strong object-oriented contextual bank.
        kwargs["max_constraints"] = max(
            int(kwargs.get("max_constraints", 220)),
            320,
        )
        kwargs["beam_width"] = max(
            int(kwargs.get("beam_width", 180)),
            300,
        )
        kwargs["literal_pool_per_class"] = max(
            int(kwargs.get("literal_pool_per_class", 140)),
            220,
        )
        kwargs["max_rules_per_class"] = max(
            int(kwargs.get("max_rules_per_class", 100)),
            140,
        )
        super().__init__(*args, **kwargs)
        self.clear_feature_count = 0
        self.selected_clear_feature_count = 0

    @staticmethod
    def _similarity_ratio(a: float, b: float, eps: float) -> float:
        """1 at equality, approaches 0 as positive quantities diverge."""
        return float(np.exp(-abs(np.log((a + eps) / (b + eps)))))

    @staticmethod
    def _relative_similarity(a: float, b: float, eps: float) -> float:
        """Bounded [0,1] similarity based on relative absolute difference."""
        return float(
            1.0 - min(1.0, abs(a - b) / (abs(a) + abs(b) + eps))
        )

    def _structural_observables(self, image: np.ndarray):
        # Start from the full v7/v10 contextual bank.
        values, support, coeff, names = super()._structural_observables(image)

        z = self._resize(image)
        q = gaussian_filter(z, 0.8)
        residual = z - q

        gy, gx = np.gradient(q)
        grad = np.sqrt(gx * gx + gy * gy)
        lap_abs = np.abs(laplace(q))
        dog_abs = np.abs(q - gaussian_filter(q, 2.0))
        ori = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)

        rgy, rgx = np.gradient(residual)
        rgrad = np.sqrt(rgx * rgx + rgy * rgy)
        rlap = np.abs(laplace(residual))
        rdog = np.abs(residual - gaussian_filter(residual, 1.2))

        vals = list(values.astype(float))
        sups = list(support.astype(float))
        coefs = list(coeff.astype(float))
        nms = list(names)

        def add(
            name: str,
            identity: float,
            evidence_support: float,
            noise_coeff: float,
        ):
            vals.append(float(identity))
            sups.append(float(max(evidence_support, 0.0)))
            coefs.append(float(max(noise_coeff, 1e-8)))
            nms.append(name)

        n = self.image_size
        a = n // 4
        b = n - a

        center = np.zeros((n, n), dtype=bool)
        center[a:b, a:b] = True
        surround = ~center

        nc = float(center.sum())
        ns = float(surround.sum())

        # --------------------------------------------------------------
        # 1. Positive center-surround CONTINUITY evidence.
        # --------------------------------------------------------------
        c_std = float(q[center].std())
        s_std = float(q[surround].std())
        cn_std = float(residual[center].std())
        sn_std = float(residual[surround].std())
        std_ex = min(
            max(c_std - cn_std, 0.0),
            max(s_std - sn_std, 0.0),
        )

        spread = float(np.percentile(q, 90) - np.percentile(q, 10)) + 1e-8
        cm = float(q[center].mean())
        sm = float(q[surround].mean())
        mean_noise = 0.5 * (cn_std + sn_std)
        mean_support = max(spread - 2.0 * mean_noise, 0.0)

        add(
            "clear:mean_continuity",
            float(np.exp(-abs(cm - sm) / spread)),
            mean_support,
            np.sqrt(1 / nc + 1 / ns),
        )
        add(
            "clear:std_ratio_similarity",
            self._similarity_ratio(c_std, s_std, 0.03 * (c_std + s_std) + 1e-8),
            std_ex,
            np.sqrt(1 / nc + 1 / ns),
        )
        add(
            "clear:std_relative_similarity",
            self._relative_similarity(c_std, s_std, 1e-8),
            std_ex,
            np.sqrt(1 / nc + 1 / ns),
        )

        metrics = [
            ("grad", grad, rgrad, np.sqrt(2.0)),
            ("lap", lap_abs, rlap, 2.5),
            ("dog", dog_abs, rdog, 1.0),
        ]

        for nm, arr, narr, base_c in metrics:
            cv = float(arr[center].mean())
            sv = float(arr[surround].mean())
            cn = float(narr[center].mean())
            sn = float(narr[surround].mean())
            ex = min(max(cv - cn, 0.0), max(sv - sn, 0.0))
            uc = base_c * np.sqrt(1 / nc + 1 / ns)

            add(
                f"clear:{nm}_ratio_similarity",
                self._similarity_ratio(
                    cv, sv, 0.03 * (cv + sv) + 1e-8
                ),
                ex,
                uc,
            )
            add(
                f"clear:{nm}_relative_similarity",
                self._relative_similarity(cv, sv, 1e-8),
                ex,
                uc,
            )

        # --------------------------------------------------------------
        # 2. Candidate-boundary continuity.
        # A true obstacle frequently inserts excess structural energy on
        # the candidate boundary. CLEAR requires the boundary to look like
        # a continuation of nearby scene structure.
        # --------------------------------------------------------------
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

        nearby = np.zeros((n, n), dtype=bool)
        w2 = max(2 * bw, 4)
        nearby[max(0, a-w2):min(n, b+w2), max(0, a-w2):min(n, b+w2)] = True
        nearby &= ~boundary

        nb = float(boundary.sum())
        nn = float(nearby.sum())

        for nm, arr, narr, base_c in metrics:
            bv = float(arr[boundary].mean())
            nv = float(arr[nearby].mean())
            bn = float(narr[boundary].mean())
            nnoise = float(narr[nearby].mean())
            ex = min(max(bv - bn, 0.0), max(nv - nnoise, 0.0))

            add(
                f"clear:boundary_{nm}_continuity",
                self._similarity_ratio(
                    bv, nv, 0.03 * (bv + nv) + 1e-8
                ),
                ex,
                base_c * np.sqrt(1 / nb + 1 / nn),
            )

            # Explicit no-boundary-excess identity: 1 is continuous.
            excess = max(bv - nv, 0.0)
            add(
                f"clear:no_boundary_{nm}_excess",
                float(1.0 / (1.0 + excess / (nv + 1e-6))),
                ex,
                base_c * np.sqrt(1 / nb + 1 / nn),
            )

        # --------------------------------------------------------------
        # 3. Orientation continuity.
        # --------------------------------------------------------------
        ch = []
        sh = []
        cden = float(grad[center].sum()) + 1e-8
        sden = float(grad[surround].sum()) + 1e-8

        for k in range(8):
            lo = k * (2 * np.pi / 8)
            hi = (k + 1) * (2 * np.pi / 8)
            cmask = center & (ori >= lo) & (ori < hi)
            smask = surround & (ori >= lo) & (ori < hi)
            ch.append(float(grad[cmask].sum() / cden))
            sh.append(float(grad[smask].sum() / sden))

        ch = np.asarray(ch, dtype=np.float64)
        sh = np.asarray(sh, dtype=np.float64)
        l1 = float(0.5 * np.abs(ch - sh).sum())
        cosine = float(
            np.dot(ch, sh)
            / (np.linalg.norm(ch) * np.linalg.norm(sh) + 1e-8)
        )

        cg_ex = max(
            float(grad[center].mean()) - float(rgrad[center].mean()),
            0.0,
        )
        sg_ex = max(
            float(grad[surround].mean()) - float(rgrad[surround].mean()),
            0.0,
        )
        o_sup = min(cg_ex, sg_ex)
        o_c = np.sqrt(2.0) * np.sqrt(1 / nc + 1 / ns)

        add(
            "clear:orientation_l1_similarity",
            1.0 - min(1.0, l1),
            o_sup,
            o_c,
        )
        add(
            "clear:orientation_cosine",
            cosine,
            o_sup,
            o_c,
        )

        # Jensen-Shannon orientation similarity in [0,1].
        m = 0.5 * (ch + sh)
        eps = 1e-10
        js = 0.5 * (
            np.sum(ch * np.log((ch + eps) / (m + eps)))
            + np.sum(sh * np.log((sh + eps) / (m + eps)))
        )
        js_norm = float(min(1.0, js / np.log(2.0)))
        add(
            "clear:orientation_js_similarity",
            1.0 - js_norm,
            o_sup,
            o_c,
        )

        # --------------------------------------------------------------
        # 4. Directional continuation.
        # Compare each side of candidate with its immediately adjacent
        # exterior strip. This is a local continuity test rather than a
        # category appearance feature.
        # --------------------------------------------------------------
        strip = max(3, n // 16)
        regions = {
            "top": (
                (slice(a, a + strip), slice(a, b)),
                (slice(max(0, a - strip), a), slice(a, b)),
            ),
            "bottom": (
                (slice(b - strip, b), slice(a, b)),
                (slice(b, min(n, b + strip)), slice(a, b)),
            ),
            "left": (
                (slice(a, b), slice(a, a + strip)),
                (slice(a, b), slice(max(0, a - strip), a)),
            ),
            "right": (
                (slice(a, b), slice(b - strip, b)),
                (slice(a, b), slice(b, min(n, b + strip))),
            ),
        }

        for side, (inside_sl, outside_sl) in regions.items():
            ni = float(q[inside_sl].size)
            no = float(q[outside_sl].size)

            ist = float(q[inside_sl].std())
            ost = float(q[outside_sl].std())
            inst = float(residual[inside_sl].std())
            onst = float(residual[outside_sl].std())
            sup = min(max(ist - inst, 0.0), max(ost - onst, 0.0))

            add(
                f"clear:{side}_texture_continuity",
                self._similarity_ratio(
                    ist, ost, 0.03 * (ist + ost) + 1e-8
                ),
                sup,
                np.sqrt(1 / max(ni, 1) + 1 / max(no, 1)),
            )

            ig = float(grad[inside_sl].mean())
            og = float(grad[outside_sl].mean())
            ing = float(rgrad[inside_sl].mean())
            ong = float(rgrad[outside_sl].mean())
            gsup = min(max(ig - ing, 0.0), max(og - ong, 0.0))

            add(
                f"clear:{side}_edge_continuity",
                self._similarity_ratio(
                    ig, og, 0.03 * (ig + og) + 1e-8
                ),
                gsup,
                np.sqrt(2.0)
                * np.sqrt(1 / max(ni, 1) + 1 / max(no, 1)),
            )

        # --------------------------------------------------------------
        # 5. Multi-scale continuity: genuine scene continuation should
        # persist across scale; noise-only coincidences should not.
        # --------------------------------------------------------------
        for sigma in (1.2, 2.2, 3.6):
            qs = gaussian_filter(z, sigma)
            sgy, sgx = np.gradient(qs)
            sg = np.sqrt(sgx * sgx + sgy * sgy)

            cv = float(sg[center].mean())
            sv = float(sg[surround].mean())

            # Use v11's existing noise-debiased gradient support as the
            # observability witness; the identity is the multi-scale ratio.
            sup = min(cg_ex, sg_ex)
            add(
                f"clear:multiscale_grad_similarity_{sigma:.1f}",
                self._similarity_ratio(
                    cv, sv, 0.03 * (cv + sv) + 1e-8
                ),
                sup,
                np.sqrt(2.0) * np.sqrt(1 / nc + 1 / ns),
            )

        return (
            np.asarray(vals, dtype=np.float32),
            np.asarray(sups, dtype=np.float32),
            np.asarray(coefs, dtype=np.float32),
            nms,
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
        self.clear_feature_count = int(
            sum(name.startswith("clear:") for name in names)
        )

        diag = super()._fit_bank(
            pos_values,
            neg_values,
            pos_support,
            neg_support,
            noise_coeff,
            names,
            mode,
        )

        self.selected_clear_feature_count = int(
            sum(
                self.feature_names[i].startswith("clear:")
                for i in range(len(self.feature_names))
            )
        )

        self.representation_mode = "dual_evidence_constraints_v11"
        diag["mode"] = self.representation_mode
        diag["clear_features_total"] = int(self.clear_feature_count)
        diag["clear_features_selected"] = int(
            self.selected_clear_feature_count
        )
        diag["dual_evidence_principle"] = (
            "absence_object_evidence_is_not_clear_evidence"
        )
        return diag

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({
            "clear_feature_count": int(self.clear_feature_count),
            "selected_clear_feature_count": int(
                self.selected_clear_feature_count
            ),
        })
        return d


__all__ = ["Gate", "SCLIVisionBinary"]

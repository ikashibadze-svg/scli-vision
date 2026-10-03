from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, laplace
from skimage import transform
from sklearn.covariance import LedoitWolf


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
    snr_min: float
    validation_coverage: float
    validation_known_accuracy: float
    certified: bool = True
    target_known_accuracy: float = 0.90


class SCLIVisionBinary:
    """SCLI Vision v5: invariant identity + raw observability.

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

        # v8 joint constraint geometry.
        self.geometry_center: np.ndarray | None = None
        self.geometry_scale: np.ndarray | None = None
        self.geometry_weights: np.ndarray | None = None
        self.geometry_projection_scale: float = 1.0
        self.geometry_train_accuracy: float | None = None

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
        """v6 invariant identity with noise-debiased observability support.

        Identity coordinates remain dimensionless and contrast-normalized.

        Crucially, support is no longer total gradient/std/Laplacian energy.
        Sensor noise itself creates those energies. We estimate the structural
        energy of the high-frequency residual and subtract that expected
        noise contribution before a constraint is allowed to vote.
        """
        z = self._resize(image)

        # Signal image and empirical noise residual.
        q = gaussian_filter(z, 0.8)
        q2 = gaussian_filter(z, 2.0)
        residual = z - q

        gy, gx = np.gradient(q)
        grad = np.sqrt(gx*gx + gy*gy)
        lap_abs = np.abs(laplace(q))
        dog_abs = np.abs(q - q2)
        ori = (np.arctan2(gy, gx) + 2*np.pi) % (2*np.pi)

        rgy, rgx = np.gradient(residual)
        rgrad = np.sqrt(rgx*rgx + rgy*rgy)
        rlap = np.abs(laplace(residual))
        rdog = np.abs(residual - gaussian_filter(residual, 1.2))

        global_std = float(q.std()) + 1e-8
        global_spread = float(np.percentile(q,90)-np.percentile(q,10)) + 1e-8
        global_grad = float(grad.mean()) + 1e-8
        global_lap = float(lap_abs.mean()) + 1e-8
        global_dog = float(dog_abs.mean()) + 1e-8
        global_median = float(np.median(q))

        # Noise-bias estimates in the same units as the structural supports.
        noise_std = float(residual.std())
        noise_grad = float(rgrad.mean())
        noise_lap = float(rlap.mean())
        noise_dog = float(rdog.mean())

        global_std_excess = max(global_std - noise_std, 0.0)
        global_grad_excess = max(global_grad - noise_grad, 0.0)
        global_lap_excess = max(global_lap - noise_lap, 0.0)
        global_dog_excess = max(global_dog - noise_dog, 0.0)

        values: list[float] = []
        support: list[float] = []
        coeff: list[float] = []
        names: list[str] = []

        def add(name: str, identity: float, excess_support: float, uncertainty_c: float):
            values.append(float(identity))
            support.append(float(max(excess_support, 0.0)))
            coeff.append(float(max(uncertainty_c, 1e-8)))
            names.append(name)

        grid = 4
        step = self.image_size // grid
        n_cell = float(step*step)
        n_global = float(self.image_size*self.image_size)

        # After subtracting the noise bias, only uncertainty of the estimated
        # aggregate remains and does scale ~1/sqrt(N_eff).
        cell_mean_c = 1.0/np.sqrt(n_cell)
        cell_grad_c = np.sqrt(2.0)/np.sqrt(n_cell)
        cell_lap_c = 2.5/np.sqrt(n_cell)
        cell_dog_c = 1.0/np.sqrt(n_cell)

        global_grad_c = np.sqrt(2.0)/np.sqrt(n_global)
        global_lap_c = 2.5/np.sqrt(n_global)
        global_dog_c = 1.0/np.sqrt(n_global)

        total_grad_sum = float(grad.sum()) + 1e-8
        total_lap_sum = float(lap_abs.sum()) + 1e-8
        total_dog_sum = float(dog_abs.sum()) + 1e-8

        cell_std = np.zeros((grid,grid), dtype=np.float32)
        cell_grad = np.zeros((grid,grid), dtype=np.float32)
        cell_lap = np.zeros((grid,grid), dtype=np.float32)
        cell_dog = np.zeros((grid,grid), dtype=np.float32)
        cell_grad_excess = np.zeros((grid,grid), dtype=np.float32)
        cell_lap_excess = np.zeros((grid,grid), dtype=np.float32)
        cell_dog_excess = np.zeros((grid,grid), dtype=np.float32)
        cell_std_excess = np.zeros((grid,grid), dtype=np.float32)
        cell_edge_fraction = np.zeros((grid,grid), dtype=np.float32)

        # Relative edge topology comes from the signal image, but its ability
        # to vote is governed by excess gradient support.
        edge_q = float(np.percentile(grad, 75))
        edge_map = grad >= edge_q

        for ry in range(grid):
            for rx in range(grid):
                y0 = ry*step
                y1 = self.image_size if ry == grid-1 else (ry+1)*step
                x0 = rx*step
                x1 = self.image_size if rx == grid-1 else (rx+1)*step
                sl = (slice(y0,y1), slice(x0,x1))

                p = q[sl]
                pg = grad[sl]
                pl = lap_abs[sl]
                pd = dog_abs[sl]
                po = ori[sl]
                pe = edge_map[sl]

                pr = residual[sl]
                prg = rgrad[sl]
                prl = rlap[sl]
                prd = rdog[sl]

                pmean = float(p.mean())
                pstd = float(p.std())
                gmean = float(pg.mean())
                lmean = float(pl.mean())
                dmean = float(pd.mean())

                nstd = float(pr.std())
                ng = float(prg.mean())
                nl = float(prl.mean())
                nd = float(prd.mean())

                std_ex = max(pstd - nstd, 0.0)
                grad_ex = max(gmean - ng, 0.0)
                lap_ex = max(lmean - nl, 0.0)
                dog_ex = max(dmean - nd, 0.0)

                cell_std[ry,rx] = pstd
                cell_grad[ry,rx] = gmean
                cell_lap[ry,rx] = lmean
                cell_dog[ry,rx] = dmean
                cell_std_excess[ry,rx] = std_ex
                cell_grad_excess[ry,rx] = grad_ex
                cell_lap_excess[ry,rx] = lap_ex
                cell_dog_excess[ry,rx] = dog_ex
                cell_edge_fraction[ry,rx] = float(pe.mean())

                raw_mean_delta = abs(pmean-global_median)
                mean_noise = nstd/np.sqrt(n_cell)
                mean_ex = max(raw_mean_delta - 2.0*mean_noise, 0.0)

                add(
                    f"cell{ry}{rx}:mean_role",
                    (pmean-global_median)/global_spread,
                    mean_ex,
                    cell_mean_c,
                )
                add(
                    f"cell{ry}{rx}:std_ratio",
                    pstd/global_std,
                    std_ex,
                    cell_mean_c,
                )
                add(
                    f"cell{ry}{rx}:grad_ratio",
                    gmean/global_grad,
                    grad_ex,
                    cell_grad_c,
                )
                add(
                    f"cell{ry}{rx}:lap_ratio",
                    lmean/global_lap,
                    lap_ex,
                    cell_lap_c,
                )
                add(
                    f"cell{ry}{rx}:dog_ratio",
                    dmean/global_dog,
                    dog_ex,
                    cell_dog_c,
                )

                add(
                    f"cell{ry}{rx}:grad_share",
                    float(pg.sum())/total_grad_sum*(grid*grid),
                    grad_ex,
                    cell_grad_c,
                )
                add(
                    f"cell{ry}{rx}:lap_share",
                    float(pl.sum())/total_lap_sum*(grid*grid),
                    lap_ex,
                    cell_lap_c,
                )
                add(
                    f"cell{ry}{rx}:dog_share",
                    float(pd.sum())/total_dog_sum*(grid*grid),
                    dog_ex,
                    cell_dog_c,
                )

                efrac = float(pe.mean())
                add(
                    f"cell{ry}{rx}:edge_occupancy",
                    efrac,
                    grad_ex,
                    cell_grad_c,
                )

                denom = float(pg.sum()) + 1e-8
                cx = float((np.cos(po)*pg).sum())
                cy = float((np.sin(po)*pg).sum())
                coherence = float(np.sqrt(cx*cx+cy*cy)/denom)
                add(
                    f"cell{ry}{rx}:ori_coherence",
                    coherence,
                    grad_ex,
                    cell_grad_c,
                )

                for b in range(4):
                    lo = b*(2*np.pi/4)
                    hi = (b+1)*(2*np.pi/4)
                    mask = (po >= lo) & (po < hi)
                    h = float(pg[mask].sum()/denom)
                    add(
                        f"cell{ry}{rx}:ori{b}",
                        h,
                        grad_ex,
                        cell_grad_c,
                    )

        # Pairwise topology/ratio features; support is the weaker excess
        # evidence of the two participating regions.
        metrics = [
            ("std", cell_std, cell_std_excess, cell_mean_c),
            ("grad", cell_grad, cell_grad_excess, cell_grad_c),
            ("lap", cell_lap, cell_lap_excess, cell_lap_c),
            ("dog", cell_dog, cell_dog_excess, cell_dog_c),
        ]
        coords = [(r,c) for r in range(grid) for c in range(grid)]
        pair_set = set()
        for r,c in coords:
            i = r*grid+c
            if c+1 < grid:
                pair_set.add((i,r*grid+c+1))
            if r+1 < grid:
                pair_set.add((i,(r+1)*grid+c))
            j = r*grid+(grid-1-c)
            if i < j:
                pair_set.add((i,j))
            j = (grid-1-r)*grid+c
            if i < j:
                pair_set.add((i,j))

        for mname, mat, exmat, base_c in metrics:
            flat = mat.ravel()
            exflat = exmat.ravel()
            global_ref = float(np.median(flat)) + 1e-8
            for i,j in sorted(pair_set):
                a = float(flat[i])
                b = float(flat[j])
                eps = 0.05*global_ref + 1e-8
                pair_support = min(float(exflat[i]), float(exflat[j]))

                add(
                    f"pair:{mname}:{i}-{j}:logratio",
                    float(np.log((a+eps)/(b+eps))),
                    pair_support,
                    np.sqrt(2.0)*base_c,
                )
                add(
                    f"pair:{mname}:{i}-{j}:reldiff",
                    (a-b)/(a+b+2*eps),
                    pair_support,
                    np.sqrt(2.0)*base_c,
                )

        # Symmetry relations.
        for mname, mat, exmat, base_c in metrics:
            left = mat[:,:grid//2]
            right = np.fliplr(mat[:,grid-grid//2:])
            top = mat[:grid//2,:]
            bottom = np.flipud(mat[grid-grid//2:,:])

            exleft = exmat[:,:grid//2]
            exright = np.fliplr(exmat[:,grid-grid//2:])
            extop = exmat[:grid//2,:]
            exbottom = np.flipud(exmat[grid-grid//2:,:])

            lr_num = float(np.mean(np.abs(left-right)))
            lr_den = float(np.mean(left+right)) + 1e-8
            tb_num = float(np.mean(np.abs(top-bottom)))
            tb_den = float(np.mean(top+bottom)) + 1e-8

            add(
                f"global:{mname}:lr_asymmetry",
                lr_num/lr_den,
                float(np.mean(np.minimum(exleft,exright))),
                base_c/np.sqrt(max(left.size,1)),
            )
            add(
                f"global:{mname}:tb_asymmetry",
                tb_num/tb_den,
                float(np.mean(np.minimum(extop,exbottom))),
                base_c/np.sqrt(max(top.size,1)),
            )

        # Center/border ratios with debiased support.
        a = self.image_size//4
        b = self.image_size-a
        center = np.zeros_like(z, dtype=bool)
        center[a:b,a:b] = True
        border = ~center
        n_center = float(center.sum())
        n_border = float(border.sum())

        for nm, arr, narr, base_c in [
            ("grad", grad, rgrad, np.sqrt(2.0)),
            ("lap", lap_abs, rlap, 2.5),
            ("dog", dog_abs, rdog, 1.0),
        ]:
            c = float(arr[center].mean())
            o = float(arr[border].mean())
            cn = float(narr[center].mean())
            on = float(narr[border].mean())
            ex = min(max(c-cn,0.0), max(o-on,0.0))

            add(
                f"global:center_border_{nm}_logratio",
                float(np.log((c+1e-6)/(o+1e-6))),
                ex,
                base_c*np.sqrt(1/n_center+1/n_border),
            )
            add(
                f"global:center_{nm}_share",
                c/(c+o+1e-8),
                ex,
                base_c*np.sqrt(1/n_center+1/n_border),
            )

        # Edge mass geometry; all use global excess gradient as support.
        gm = grad + 1e-12
        total = float(gm.sum())
        yy, xx = np.mgrid[0:self.image_size,0:self.image_size]
        norm = max(self.image_size-1,1)

        add(
            "global:edge_centroid_x",
            float((xx*gm).sum()/total)/norm,
            global_grad_excess,
            global_grad_c,
        )
        add(
            "global:edge_centroid_y",
            float((yy*gm).sum()/total)/norm,
            global_grad_excess,
            global_grad_c,
        )

        xn = (xx-(self.image_size-1)/2)/(self.image_size/2)
        yn = (yy-(self.image_size-1)/2)/(self.image_size/2)
        rr = np.sqrt(xn*xn+yn*yn)
        for radius in (0.25,0.40,0.55,0.70,0.85):
            add(
                f"global:edge_mass_r{radius:.2f}",
                float(gm[rr <= radius].sum()/total),
                global_grad_excess,
                global_grad_c,
            )

        row_mass = grad.sum(axis=1)
        col_mass = grad.sum(axis=0)
        row_mass = row_mass/(row_mass.sum()+1e-8)
        col_mass = col_mass/(col_mass.sum()+1e-8)
        bins = 8
        bs = self.image_size//bins
        for i in range(bins):
            x0 = i*bs
            x1 = self.image_size if i == bins-1 else (i+1)*bs
            add(
                f"global:row_edge_share_{i}",
                float(row_mass[x0:x1].sum())*bins,
                global_grad_excess,
                global_grad_c,
            )
            add(
                f"global:col_edge_share_{i}",
                float(col_mass[x0:x1].sum())*bins,
                global_grad_excess,
                global_grad_c,
            )

        ef = cell_edge_fraction
        ef_mean = float(ef.mean()) + 1e-8
        for r in range(grid):
            for c in range(grid):
                add(
                    f"topology:edge_occ_ratio_{r}{c}",
                    float(ef[r,c]/ef_mean),
                    float(cell_grad_excess[r,c]),
                    cell_grad_c,
                )

        denom = total_grad_sum
        for b in range(8):
            lo = b*(2*np.pi/8)
            hi = (b+1)*(2*np.pi/8)
            mask = (ori >= lo) & (ori < hi)
            add(
                f"global:ori{b}",
                float(grad[mask].sum()/denom),
                global_grad_excess,
                global_grad_c,
            )


        # --------------------------------------------------------------
        # v7 CONTEXTUAL CONSTRAINTS
        # The context dataset is built so the candidate occupies the
        # central half of the crop. The surrounding ring comes from the
        # same original image, cancelling camera/illumination nuisance.
        # --------------------------------------------------------------
        ca = self.image_size // 4
        cb = self.image_size - ca
        candidate = np.zeros_like(z, dtype=bool)
        candidate[ca:cb, ca:cb] = True
        surround = ~candidate

        n_candidate = float(candidate.sum())
        n_surround = float(surround.sum())

        # Raw center/surround signal and noise estimates.
        c_std = float(q[candidate].std())
        s_std = float(q[surround].std())
        cn_std = float(residual[candidate].std())
        sn_std = float(residual[surround].std())
        c_std_ex = max(c_std - cn_std, 0.0)
        s_std_ex = max(s_std - sn_std, 0.0)

        c_spread = float(
            np.percentile(q[candidate], 90)
            - np.percentile(q[candidate], 10)
        )
        s_spread = float(
            np.percentile(q[surround], 90)
            - np.percentile(q[surround], 10)
        )

        add(
            "context:std_logratio",
            float(np.log((c_std + 1e-6) / (s_std + 1e-6))),
            min(c_std_ex, s_std_ex),
            np.sqrt(1/n_candidate + 1/n_surround),
        )
        add(
            "context:spread_logratio",
            float(np.log((c_spread + 1e-6) / (s_spread + 1e-6))),
            min(c_std_ex, s_std_ex),
            np.sqrt(1/n_candidate + 1/n_surround),
        )
        add(
            "context:std_share",
            c_std / (c_std + s_std + 1e-8),
            min(c_std_ex, s_std_ex),
            np.sqrt(1/n_candidate + 1/n_surround),
        )

        context_metrics = [
            ("grad", grad, rgrad, np.sqrt(2.0)),
            ("lap", lap_abs, rlap, 2.5),
            ("dog", dog_abs, rdog, 1.0),
        ]

        for nm, arr, narr, base_c in context_metrics:
            c_raw = float(arr[candidate].mean())
            s_raw = float(arr[surround].mean())
            c_noise = float(narr[candidate].mean())
            s_noise = float(narr[surround].mean())
            c_ex = max(c_raw - c_noise, 0.0)
            s_ex = max(s_raw - s_noise, 0.0)
            pair_ex = min(c_ex, s_ex)
            uc = base_c * np.sqrt(1/n_candidate + 1/n_surround)

            add(
                f"context:{nm}_logratio",
                float(np.log((c_raw + 1e-6)/(s_raw + 1e-6))),
                pair_ex,
                uc,
            )
            add(
                f"context:{nm}_reldiff",
                (c_raw - s_raw)/(c_raw + s_raw + 1e-8),
                pair_ex,
                uc,
            )
            add(
                f"context:{nm}_share",
                c_raw/(c_raw + s_raw + 1e-8),
                pair_ex,
                uc,
            )

        # Candidate boundary band: physical objects often create a local
        # discontinuity against their immediate surround even when albedo
        # is similar. Use only geometry/ratios as identity.
        bw = max(2, self.image_size // 32)
        boundary = np.zeros_like(z, dtype=bool)
        # inside strips
        boundary[ca:ca+bw, ca:cb] = True
        boundary[cb-bw:cb, ca:cb] = True
        boundary[ca:cb, ca:ca+bw] = True
        boundary[ca:cb, cb-bw:cb] = True
        # immediately outside strips
        boundary[max(0,ca-bw):ca, ca:cb] = True
        boundary[cb:min(self.image_size,cb+bw), ca:cb] = True
        boundary[ca:cb, max(0,ca-bw):ca] = True
        boundary[ca:cb, cb:min(self.image_size,cb+bw)] = True

        outer_ring = surround & (~boundary)
        n_boundary = float(boundary.sum())
        n_outer = float(outer_ring.sum())

        for nm, arr, narr, base_c in context_metrics:
            b_raw = float(arr[boundary].mean())
            o_raw = float(arr[outer_ring].mean()) if outer_ring.any() else 0.0
            b_noise = float(narr[boundary].mean())
            o_noise = float(narr[outer_ring].mean()) if outer_ring.any() else 0.0
            b_ex = max(b_raw - b_noise, 0.0)
            o_ex = max(o_raw - o_noise, 0.0)

            add(
                f"context:boundary_{nm}_logratio",
                float(np.log((b_raw + 1e-6)/(o_raw + 1e-6))),
                min(b_ex, o_ex) if outer_ring.any() else b_ex,
                base_c * np.sqrt(
                    1/max(n_boundary,1.0) + 1/max(n_outer,1.0)
                ),
            )
            add(
                f"context:boundary_{nm}_share",
                b_raw/(b_raw + o_raw + 1e-8),
                min(b_ex, o_ex) if outer_ring.any() else b_ex,
                base_c * np.sqrt(
                    1/max(n_boundary,1.0) + 1/max(n_outer,1.0)
                ),
            )

        # Orientation-distribution mismatch between candidate and surround.
        # This is contrast invariant and directly measures structural
        # departure from the local scene.
        c_denom = float(grad[candidate].sum()) + 1e-8
        s_denom = float(grad[surround].sum()) + 1e-8
        ch = []
        sh = []
        for b in range(8):
            lo = b*(2*np.pi/8)
            hi = (b+1)*(2*np.pi/8)
            cm = candidate & (ori >= lo) & (ori < hi)
            sm = surround & (ori >= lo) & (ori < hi)
            cv = float(grad[cm].sum()/c_denom)
            sv = float(grad[sm].sum()/s_denom)
            ch.append(cv)
            sh.append(sv)

            c_ex = max(
                float(grad[candidate].mean())
                - float(rgrad[candidate].mean()),
                0.0,
            )
            s_ex = max(
                float(grad[surround].mean())
                - float(rgrad[surround].mean()),
                0.0,
            )
            add(
                f"context:ori_delta_{b}",
                cv - sv,
                min(c_ex, s_ex),
                np.sqrt(2.0) * np.sqrt(
                    1/n_candidate + 1/n_surround
                ),
            )

        ch = np.asarray(ch, dtype=np.float32)
        sh = np.asarray(sh, dtype=np.float32)
        ori_l1 = float(0.5*np.abs(ch-sh).sum())
        ori_cos = float(
            np.dot(ch, sh) /
            (np.linalg.norm(ch)*np.linalg.norm(sh) + 1e-8)
        )
        c_ex = max(
            float(grad[candidate].mean())
            - float(rgrad[candidate].mean()),
            0.0,
        )
        s_ex = max(
            float(grad[surround].mean())
            - float(rgrad[surround].mean()),
            0.0,
        )
        add(
            "context:orientation_l1",
            ori_l1,
            min(c_ex, s_ex),
            np.sqrt(2.0) * np.sqrt(
                1/n_candidate + 1/n_surround
            ),
        )
        add(
            "context:orientation_cosine",
            ori_cos,
            min(c_ex, s_ex),
            np.sqrt(2.0) * np.sqrt(
                1/n_candidate + 1/n_surround
            ),
        )

        # Edge topology: how much relative edge mass lives inside the
        # candidate, on its boundary, and in its surround.
        edge_total = float(grad.sum()) + 1e-8
        candidate_edge_share = float(grad[candidate].sum()/edge_total)
        boundary_edge_share = float(grad[boundary].sum()/edge_total)
        surround_edge_share = float(grad[surround].sum()/edge_total)

        global_grad_ex = max(
            float(grad.mean()) - float(rgrad.mean()),
            0.0,
        )
        add(
            "context:candidate_edge_mass",
            candidate_edge_share,
            global_grad_ex,
            np.sqrt(2.0)/np.sqrt(max(n_global,1.0)),
        )
        add(
            "context:boundary_edge_mass",
            boundary_edge_share,
            global_grad_ex,
            np.sqrt(2.0)/np.sqrt(max(n_global,1.0)),
        )
        add(
            "context:surround_edge_mass",
            surround_edge_share,
            global_grad_ex,
            np.sqrt(2.0)/np.sqrt(max(n_global,1.0)),
        )
        add(
            "context:candidate_vs_surround_edge_mass",
            candidate_edge_share/(surround_edge_share + 1e-8),
            global_grad_ex,
            np.sqrt(2.0)/np.sqrt(max(n_global,1.0)),
        )

        return (
            np.asarray(values,dtype=np.float32),
            np.asarray(support,dtype=np.float32),
            np.asarray(coeff,dtype=np.float32),
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

        # --------------------------------------------------------------
        # v8 JOINT CONSTRAINT GEOMETRY
        # Independent constraint votes discard correlations. Here the
        # selected invariant constraints define a geometry. We whiten their
        # joint covariance with Ledoit-Wolf shrinkage and solve the Fisher
        # direction that maximally separates OBJECT_PRESENT from CLEAR.
        #
        # This is still a deterministic constraint system: there is no
        # learned hidden representation. The coordinates are the explicit
        # constraints above; only their joint metric is estimated.
        # --------------------------------------------------------------
        psel = pos_values[:, selected]
        nsel = neg_values[:, selected]

        geom_center = 0.5 * (
            np.median(psel, axis=0) + np.median(nsel, axis=0)
        )
        p_mad = 1.4826 * np.median(
            np.abs(psel - np.median(psel, axis=0)),
            axis=0,
        )
        n_mad = 1.4826 * np.median(
            np.abs(nsel - np.median(nsel, axis=0)),
            axis=0,
        )
        geom_scale = 0.5 * (p_mad + n_mad) + 1e-3

        pz = np.clip((psel - geom_center) / geom_scale, -8.0, 8.0)
        nz = np.clip((nsel - geom_center) / geom_scale, -8.0, 8.0)
        zall = np.concatenate([pz, nz], axis=0)

        cov = LedoitWolf(assume_centered=False).fit(zall).covariance_
        d = pz.mean(axis=0) - nz.mean(axis=0)

        try:
            gw = np.linalg.solve(
                cov + 1e-4 * np.eye(cov.shape[0], dtype=np.float64),
                d.astype(np.float64),
            )
        except np.linalg.LinAlgError:
            gw = d.astype(np.float64)

        # Remove numerically irrelevant coordinates and normalize L1 mass
        # so observability coverage has a direct interpretation.
        gw[np.abs(gw) < 1e-8] = 0.0
        if np.all(gw == 0):
            gw = d.astype(np.float64)
        gw = gw / (np.sum(np.abs(gw)) + 1e-12)

        pp = pz @ gw
        npj = nz @ gw
        if np.median(pp) < np.median(npj):
            gw = -gw
            pp = -pp
            npj = -npj

        # Center the projection midpoint at zero using a small global offset
        # encoded by shifting every feature center along the discriminant
        # direction. This keeps "missing = neutral" approximately valid.
        pmed = float(np.median(pp))
        nmed = float(np.median(npj))
        gap = max(pmed - nmed, 1e-4)
        proj_mid = 0.5 * (pmed + nmed)

        # Adjust center minimally along w so w·z midpoint becomes zero.
        denom = float(np.sum(gw * gw)) + 1e-12
        center_shift_z = (proj_mid / denom) * gw
        geom_center = geom_center + center_shift_z * geom_scale

        # Recompute train projections after recentering.
        pz2 = np.clip((psel - geom_center) / geom_scale, -8.0, 8.0)
        nz2 = np.clip((nsel - geom_center) / geom_scale, -8.0, 8.0)
        pp2 = pz2 @ gw
        np2 = nz2 @ gw

        pmed2 = float(np.median(pp2))
        nmed2 = float(np.median(np2))
        proj_scale = max(0.5 * (pmed2 - nmed2), 1e-3)

        train_pred_pos = pp2 >= 0.0
        train_pred_neg = np2 < 0.0
        geom_train_acc = float(
            (train_pred_pos.sum() + train_pred_neg.sum())
            / (len(pp2) + len(np2))
        )

        self.geometry_center = geom_center.astype(np.float32)
        self.geometry_scale = geom_scale.astype(np.float32)
        self.geometry_weights = gw.astype(np.float32)
        self.geometry_projection_scale = float(proj_scale)
        self.geometry_train_accuracy = geom_train_acc

        # For v8 observability coverage, use discriminant mass rather than
        # univariate effect mass.
        self.weights = np.maximum(
            np.abs(self.geometry_weights),
            1e-8,
        ).astype(np.float32)

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
            "geometry_train_accuracy": float(self.geometry_train_accuracy),
            "geometry_projection_scale": float(self.geometry_projection_scale),
            "geometry_nonzero_weights": int(
                np.sum(np.abs(self.geometry_weights) > 1e-8)
            ),
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
                "Expanding to contextual constraint geometry v8..."
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
                "contextual_constraint_geometry_v8",
            )
            expanded = True

        self.fit_diagnostics = {
            "identity_observability_separated": True,
            "aggregated_noise_propagation": True,
            "noise_bias_subtraction": True,
            "robust_group_calibration_supported": True,
            "contextual_center_surround_constraints": True,
            "joint_constraint_geometry": True,
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
        elif self.representation_mode == "contextual_constraint_geometry_v8":
            v, s, c, _ = self._structural_observables(image)
        else:
            raise RuntimeError("Unknown/unfitted representation.")
        return v[self.selected], s[self.selected], c[self.selected]

    def _structural_snr(self, image: np.ndarray) -> float:
        z = self._resize(image)
        q = gaussian_filter(z, 1.2)
        residual = z - gaussian_filter(z, 0.8)
        signal = float(np.percentile(q,90)-np.percentile(q,10))
        noise = float(
            np.median(np.abs(residual-np.median(residual))) / 0.6745
            + 1e-6
        )
        return signal / noise

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

        if not observable.any():
            return {
                "score": 0.5,
                "coverage": 0.0,
                "noise_sigma": sigma,
                "n_observable": 0,
                "structural_snr": structural_snr,
            }

        if (
            self.geometry_center is not None
            and self.geometry_scale is not None
            and self.geometry_weights is not None
        ):
            z = np.clip(
                (values - self.geometry_center)
                / self.geometry_scale,
                -8.0,
                8.0,
            )

            # Missing constraints contribute exactly zero in centered
            # geometry. Observable ones contribute along the learned joint
            # discriminant direction.
            contrib = self.geometry_weights * z
            projection = float(contrib[observable].sum())

            # Do NOT amplify sparse observations to full projection. Missing
            # evidence should naturally pull the score toward 0.5.
            scale = max(self.geometry_projection_scale, 1e-6)
            logit = np.clip(2.0 * projection / scale, -12.0, 12.0)
            score = float(1.0 / (1.0 + np.exp(-logit)))

            mass = np.abs(self.geometry_weights)
            coverage = float(
                mass[observable].sum()
                / (mass.sum() + 1e-12)
            )
        else:
            v = values[observable]
            dp = np.abs(
                v-self.pos_center[observable]
            ) / self.pos_scale[observable]
            dn = np.abs(
                v-self.neg_center[observable]
            ) / self.neg_scale[observable]
            local_score = dn/(dp+dn+1e-6)

            w = self.weights[observable]
            score = float(
                (w*local_score).sum()
                / (w.sum()+1e-12)
            )
            coverage = float(
                w.sum()
                / (self.weights.sum()+1e-12)
            )

        return {
            "score": score,
            "coverage": coverage,
            "noise_sigma": sigma,
            "n_observable": int(observable.sum()),
            "structural_snr": structural_snr,
        }

    def calibrate(
        self,
        images: list[np.ndarray],
        labels: np.ndarray,
        target_known_accuracy: float = 0.90,
        min_known_coverage: float = 0.05,
        groups: np.ndarray | None = None,
        min_group_answer_rate: float = 0.02,
    ) -> Gate:
        """Calibrate a robust epistemic gate.

        If groups are supplied (e.g. normal, contrast_10, dark_15...), a gate
        can be certified only if every group on which it answers often enough
        also satisfies the target KNOWN accuracy. This prevents an easy
        regime from masking unsafe behavior in a low-observability regime.
        """
        labels = np.asarray(labels,dtype=int)
        if groups is None:
            groups = np.asarray(["all"]*len(labels), dtype=object)
        else:
            groups = np.asarray(groups,dtype=object)

        rows = [self.score_one(im) for im in images]
        score = np.asarray([r["score"] for r in rows])
        coverage = np.asarray([r["coverage"] for r in rows])
        snr = np.asarray([r["structural_snr"] for r in rows])

        best_certified = None
        best_any = None
        unique_groups = np.unique(groups)

        # Candidate SNR floors from validation quantiles plus simple anchors.
        qs = np.unique(np.quantile(snr, [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70]))
        snr_candidates = np.unique(np.r_[0.0, qs, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

        for positive_min in np.arange(0.55,0.96,0.02):
            for negative_max in np.arange(0.20,0.66,0.02):
                if negative_max >= positive_min:
                    continue
                for coverage_min in np.arange(0.05,0.91,0.05):
                    for snr_min in snr_candidates:
                        kp = (
                            (score >= positive_min)
                            & (coverage >= coverage_min)
                            & (snr >= snr_min)
                        )
                        kn = (
                            (score <= negative_max)
                            & (coverage >= coverage_min)
                            & (snr >= snr_min)
                        )
                        known = kp | kn
                        cov = float(known.mean())
                        if cov < min_known_coverage:
                            continue

                        pred = np.where(kp,1,0)
                        acc = float((pred[known] == labels[known]).mean())

                        group_safe = True
                        group_stats = {}
                        for g in unique_groups:
                            gm = groups == g
                            kg = known & gm
                            answer_rate = float(kg.sum()/max(gm.sum(),1))
                            if kg.any():
                                gacc = float((pred[kg] == labels[kg]).mean())
                            else:
                                gacc = float("nan")
                            group_stats[str(g)] = {
                                "answer_rate": answer_rate,
                                "known_accuracy": gacc,
                            }
                            if (
                                answer_rate >= min_group_answer_rate
                                and (not np.isfinite(gacc) or gacc < target_known_accuracy)
                            ):
                                group_safe = False
                                break

                        certified = (
                            acc >= target_known_accuracy
                            and group_safe
                        )

                        cand = Gate(
                            float(positive_min),
                            float(negative_max),
                            float(coverage_min),
                            float(snr_min),
                            cov,
                            acc,
                            certified=certified,
                            target_known_accuracy=float(target_known_accuracy),
                        )

                        if certified and (
                            best_certified is None
                            or cov > best_certified.validation_coverage
                        ):
                            best_certified = cand

                        # fallback prioritizes accuracy first, then coverage
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
                "SCLI robust epistemic gate CERTIFIED: "
                f"validation known accuracy={best_certified.validation_known_accuracy:.3f}, "
                f"coverage={best_certified.validation_coverage:.3f}, "
                f"snr_min={best_certified.snr_min:.3f}"
            )
            return best_certified

        if best_any is None:
            raise RuntimeError("No non-empty epistemic gate could be formed.")

        best_any.certified = False
        self.gate = best_any
        print(
            "WARNING: robust SCLI gate is UNCERTIFIED. "
            f"Best pooled validation known accuracy="
            f"{best_any.validation_known_accuracy:.3f}, "
            f"coverage={best_any.validation_coverage:.3f}, "
            f"snr_min={best_any.snr_min:.3f}."
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

        if (
            result["coverage"] >= self.gate.coverage_min
            and result["structural_snr"] >= self.gate.snr_min
        ):
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
            "geometry_train_accuracy": self.geometry_train_accuracy,
            "fit_diagnostics": self.fit_diagnostics,
            "gate": None if self.gate is None else asdict(self.gate),
        }

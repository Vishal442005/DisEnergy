"""Box-free topological mask extraction and cubical persistent homology features."""

from __future__ import annotations

from dataclasses import dataclass

import gudhi
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed


@dataclass
class TopologicalFeatures:
    """Summary topological descriptors extracted from a 2D/3D DisEnergy manifold."""

    diagrams: dict[int, np.ndarray]
    betti_curves: np.ndarray
    persistence_entropy: np.ndarray
    top_lifetimes: np.ndarray
    feature_vector: np.ndarray


def _persistence_entropy_1d(lifetimes: np.ndarray) -> float:
    pos = lifetimes[lifetimes > 1e-9]
    if len(pos) == 0:
        return 0.0
    p = pos / pos.sum()
    return float(-(p * np.log2(p)).sum())


def extract_cubical_persistence(
    manifold: np.ndarray,
    filtration: str = "superlevel",
    betti_bins: int = 32,
    top_k: int = 8,
) -> TopologicalFeatures:
    """Compute cubical persistent homology (H0, H1[, H2]) on a 2D or 3D manifold in [0, 1]."""
    arr = np.asarray(manifold, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D manifold, got shape {arr.shape}")

    # GUDHI CubicalComplex uses sublevel filtration; negate for superlevel filtration
    filt_arr = (1.0 - arr) if filtration == "superlevel" else arr
    cc = gudhi.CubicalComplex(top_dimensional_cells=filt_arr)
    cc.persistence(homology_coeff_field=2, min_persistence=1e-4)

    max_dim = arr.ndim - 1  # H0, H1 for 2D; H0, H1, H2 for 3D
    diagrams: dict[int, np.ndarray] = {}
    t_grid = np.linspace(0.0, 1.0, betti_bins, dtype=np.float32)

    betti_list: list[np.ndarray] = []
    ent_list: list[float] = []
    top_life_list: list[np.ndarray] = []
    summary_stats: list[float] = []

    for dim in range(max_dim + 1):
        pairs = cc.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            dgm = np.zeros((0, 2), dtype=np.float32)
        else:
            dgm = np.asarray(pairs, dtype=np.float32)
            # Cap infinite essential component death at 1.0 for bounded filtration [0, 1]
            dgm = np.where(np.isinf(dgm), 1.0, dgm)

        diagrams[dim] = dgm
        lifetimes = np.maximum(dgm[:, 1] - dgm[:, 0], 0.0) if len(dgm) > 0 else np.zeros(0, dtype=np.float32)

        # Betti curve over normalized filtration grid
        if len(dgm) > 0:
            active = (dgm[:, 0:1] <= t_grid[None, :]) & (t_grid[None, :] < dgm[:, 1:2])
            betti = active.sum(axis=0).astype(np.float32)
        else:
            betti = np.zeros(betti_bins, dtype=np.float32)
        betti_list.append(betti)
        ent_list.append(_persistence_entropy_1d(lifetimes))

        sorted_life = np.sort(lifetimes)[::-1]
        padded_top = np.zeros(top_k, dtype=np.float32)
        n_copy = min(top_k, len(sorted_life))
        if n_copy > 0:
            padded_top[:n_copy] = sorted_life[:n_copy]
        top_life_list.append(padded_top)

        total_life = float(lifetimes.sum())
        mean_life = float(lifetimes.mean()) if len(lifetimes) > 0 else 0.0
        max_life = float(lifetimes.max()) if len(lifetimes) > 0 else 0.0
        count = float(len(lifetimes))
        summary_stats.extend([count, total_life, mean_life, max_life])

    betti_curves = np.concatenate(betti_list, axis=0)
    pers_entropy = np.asarray(ent_list, dtype=np.float32)
    top_lifetimes = np.concatenate(top_life_list, axis=0)
    stats_arr = np.asarray(summary_stats, dtype=np.float32)
    feature_vector = np.concatenate([betti_curves, pers_entropy, top_lifetimes, stats_arr], axis=0)

    return TopologicalFeatures(
        diagrams=diagrams,
        betti_curves=betti_curves,
        persistence_entropy=pers_entropy,
        top_lifetimes=top_lifetimes,
        feature_vector=feature_vector,
    )


def _border_background_mask(shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(shape) == 2:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    elif len(shape) == 3:
        mask[0, :, :] = True
        mask[-1, :, :] = True
        mask[:, 0, :] = True
        mask[:, -1, :] = True
        mask[:, :, 0] = True
        mask[:, :, -1] = True
    return mask


def extract_box_free_mask(
    manifold: np.ndarray,
    fine_manifold: np.ndarray | None = None,
    method: str = "watershed_filled",
    min_persistence_ratio: float = 0.08,
    fill_enclosed_basins: bool = True,
    use_border_bg_prior: bool = True,
) -> np.ndarray:
    """Extract a box-free binary foreground mask directly from the DisEnergy manifold.

    Parameters
    ----------
    manifold : np.ndarray
        Normalized fused manifold in [0, 1] of shape (H, W) or (D, H, W).
    fine_manifold : np.ndarray, optional
        Fine-scale normalized manifold in [0, 1] used as a sharp boundary crest
        surface for topological watershed snapping.
    method : str
        - 'otsu_filled': Otsu superlevel thresholding + enclosed-basin hole filling.
        - 'watershed_filled': Crest-snapping topological watershed using coarse
          superlevel basin markers flooded against the fine/fused boundary crest.
    """
    m = np.squeeze(np.asarray(manifold, dtype=np.float32))
    if np.ptp(m) < 1e-8:
        return np.zeros(m.shape, dtype=bool)

    tau = float(threshold_otsu(m))
    superlevel = m >= tau
    filled = ndi.binary_fill_holes(superlevel) if fill_enclosed_basins else superlevel.copy()

    if method == "otsu_filled":
        return filled.astype(bool)

    if method != "watershed_filled":
        raise ValueError(f"Unknown extraction method '{method}'.")

    if not np.any(filled) or np.all(filled):
        return filled.astype(bool)

    if fine_manifold is not None:
        mf = np.squeeze(np.asarray(fine_manifold, dtype=np.float32))
        crest = (0.7 * mf + 0.3 * m).astype(np.float32)
    else:
        crest = m

    # Background seed: strictly outside dilated coarse basin (and border if prior enabled)
    bg_seed = ~ndi.binary_dilation(filled, iterations=2)
    if use_border_bg_prior:
        border = _border_background_mask(m.shape)
        bg_seed = bg_seed | (border & ~superlevel)

    # Foreground seed: deep interior core of each significant connected component
    fg_seed = np.zeros(m.shape, dtype=bool)
    lbls, n_cc = ndi.label(filled)
    min_area = max(4, int(0.0005 * m.size))
    for idx in range(1, n_cc + 1):
        comp = lbls == idx
        if int(comp.sum()) < min_area and n_cc > 1:
            continue
        dist = ndi.distance_transform_edt(comp)
        dmax = float(dist.max())
        core_thresh = max(1.0, (0.55 - 0.5 * min_persistence_ratio) * dmax)
        core = (dist >= core_thresh) & comp
        if not np.any(core):
            core = (dist == dmax) & comp
        fg_seed |= core

    markers = np.zeros(m.shape, dtype=np.int32)
    markers[bg_seed] = 1
    markers[fg_seed] = 2

    if not (np.any(markers == 1) and np.any(markers == 2)):
        return filled.astype(bool)

    ws = watershed(crest, markers=markers)
    fg_mask = ws == 2
    if fill_enclosed_basins:
        fg_mask = ndi.binary_fill_holes(fg_mask)
    return fg_mask.astype(bool)

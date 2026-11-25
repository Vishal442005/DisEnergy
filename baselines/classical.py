"""Classical mathematical segmentation & anomaly-map baselines."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, graph, segmentation
from skimage.feature import graycomatrix, graycoprops


def _to_gray_2d(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[:, :, 0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return color.rgb2gray(arr).astype(np.float32)
    return arr.mean(axis=-1).astype(np.float32)


def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _border_aware_polarity(mask: np.ndarray) -> np.ndarray:
    """Ensure background touches borders so mask polarity stays consistent."""
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    if float(mask[border].mean()) > 0.5:
        return ~mask
    return mask


def chan_vese_baseline(image: np.ndarray, max_num_iter: int = 150) -> np.ndarray:
    """Unsupervised Chan-Vese active contours."""
    gray = _to_gray_2d(image)
    mask = segmentation.chan_vese(gray, mu=0.25, lambda1=1.0, lambda2=1.0, max_num_iter=max_num_iter)
    return _border_aware_polarity(mask.astype(bool))


def slic_graphcut_baseline(image: np.ndarray, n_segments: int = 150) -> np.ndarray:
    """SLIC superpixels + normalized graph cuts."""
    arr = np.asarray(image, dtype=np.float32)
    rgb = np.repeat(arr[..., None], 3, axis=-1) if arr.ndim == 2 else arr[..., :3]
    labels = segmentation.slic(rgb, n_segments=n_segments, compactness=10.0, start_label=0)
    rag = graph.rag_mean_color(rgb, labels, mode="similarity")
    cut_labels = graph.cut_normalized(labels, rag)
    # Pick non-border dominant superpixel cluster as foreground
    border = np.zeros_like(cut_labels, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    b_vals, b_cnts = np.unique(cut_labels[border], return_counts=True)
    bg_ids = {int(v) for v, c in zip(b_vals, b_cnts) if c >= 0.15 * float(border.sum())}
    fg = ~np.isin(cut_labels, list(bg_ids))
    if not np.any(fg):
        vals, counts = np.unique(cut_labels, return_counts=True)
        fg = cut_labels == vals[np.argmin(counts)]
    return fg.astype(bool)


def gabor_energy_baseline(image: np.ndarray) -> np.ndarray:
    """Multi-scale, multi-orientation Gabor filter energy + Otsu mask."""
    gray = _to_gray_2d(image)
    energy = np.zeros_like(gray, dtype=np.float32)
    for freq in (0.1, 0.2, 0.35):
        for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
            real, imag = filters.gabor(gray, frequency=freq, theta=theta)
            energy += np.sqrt(real**2 + imag**2).astype(np.float32)
    norm_e = _norm01(ndi.gaussian_filter(energy, sigma=2.0))
    tau = float(filters.threshold_otsu(norm_e)) if np.ptp(norm_e) > 1e-8 else 0.5
    return ndi.binary_fill_holes(norm_e >= tau).astype(bool)


def glcm_entropy_baseline(image: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """Patch-level GLCM dissimilarity/entropy anomaly map + Otsu mask."""
    gray = (_norm01(_to_gray_2d(image)) * 15).astype(np.uint8)
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            if patch.size < 4:
                continue
            glcm = graycomatrix(patch, distances=[1], angles=[0, np.pi / 2], levels=16, symmetric=True, normed=True)
            val = float(graycoprops(glcm, "contrast").mean())
            out[y : y + patch_size, x : x + patch_size] = val
    out = _norm01(ndi.gaussian_filter(out, sigma=patch_size * 0.25))
    tau = float(filters.threshold_otsu(out)) if np.ptp(out) > 1e-8 else 0.5
    return (out >= tau).astype(bool)


def wavelet_energy_baseline(image: np.ndarray) -> np.ndarray:
    """Difference-of-Gaussians multi-band Haar-like energy map + Otsu mask."""
    gray = _to_gray_2d(image)
    energy = np.zeros_like(gray, dtype=np.float32)
    for s in (1.0, 2.0, 4.0):
        band = ndi.gaussian_filter(gray, s) - ndi.gaussian_filter(gray, 2.0 * s)
        energy += np.abs(band)
    norm_e = _norm01(ndi.gaussian_filter(energy, sigma=2.0))
    tau = float(filters.threshold_otsu(norm_e)) if np.ptp(norm_e) > 1e-8 else 0.5
    return ndi.binary_fill_holes(norm_e >= tau).astype(bool)


def felzenszwalb_baseline(image: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """Felzenszwalb graph-based segmentation with border background removal."""
    arr = np.asarray(image, dtype=np.float32)
    rgb = np.repeat(arr[..., None], 3, axis=-1) if arr.ndim == 2 else arr[..., :3]
    labels = segmentation.felzenszwalb(rgb, scale=scale, sigma=0.8, min_size=50)
    border = np.zeros_like(labels, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bg_ids = set(np.unique(labels[border]))
    fg = ~np.isin(labels, list(bg_ids))
    return ndi.binary_fill_holes(fg).astype(bool)


def spectral_residual_baseline(image: np.ndarray) -> np.ndarray:
    """Hou & Zhang spectral residual saliency map + Otsu threshold."""
    gray = _to_gray_2d(image)
    fft = np.fft.fft2(gray)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    avg_log = ndi.uniform_filter(log_amp, size=3)
    residual = log_amp - avg_log
    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    saliency = _norm01(ndi.gaussian_filter(saliency.astype(np.float32), sigma=3.0))
    tau = float(filters.threshold_otsu(saliency)) if np.ptp(saliency) > 1e-8 else 0.5
    return ndi.binary_fill_holes(saliency >= tau).astype(bool)

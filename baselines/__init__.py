"""Classical mathematical and lightweight deep-learning baselines."""

from .classical import (
    chan_vese_baseline,
    felzenszwalb_baseline,
    gabor_energy_baseline,
    glcm_entropy_baseline,
    slic_graphcut_baseline,
    spectral_residual_baseline,
    wavelet_energy_baseline,
)

__all__ = [
    "chan_vese_baseline",
    "felzenszwalb_baseline",
    "gabor_energy_baseline",
    "glcm_entropy_baseline",
    "slic_graphcut_baseline",
    "spectral_residual_baseline",
    "wavelet_energy_baseline",
]

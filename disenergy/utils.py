"""Utility functions for deterministic seeding, noise injection, and profiling."""

from __future__ import annotations

import os
import random
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve 'auto' device string safely to CUDA if available, else CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def inject_noise(
    image: np.ndarray,
    noise_type: str,
    snr_db: float,
    rng: np.random.Generator | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Inject calibrated Gaussian, Rician (MRI), or Speckle (ultrasound) noise at target SNR (dB).

    Parameters
    ----------
    image : np.ndarray
        Clean image array in [0, 1] of any shape (H, W), (H, W, C), or (D, H, W, C).
    noise_type : str
        One of {'gaussian', 'rician', 'speckle'}.
    snr_db : float
        Signal-to-noise ratio in decibels: 10 * log10(P_signal / P_noise).
    rng : np.random.Generator, optional
        Random generator for reproducibility.
    clip : bool
        Whether to clamp output to [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    img = np.asarray(image, dtype=np.float32)
    signal_power = float(np.mean(img**2))
    if signal_power <= 1e-12:
        signal_power = 1e-4
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = float(np.sqrt(target_noise_power))

    kind = noise_type.lower()
    if kind == "gaussian":
        noisy = img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    elif kind == "rician":
        # Two-channel complex Gaussian magnitude model: |I + n1 + i*n2|, each with std = sigma / sqrt(2)
        comp_sigma = sigma / np.sqrt(2.0)
        n_real = rng.normal(0.0, comp_sigma, size=img.shape).astype(np.float32)
        n_imag = rng.normal(0.0, comp_sigma, size=img.shape).astype(np.float32)
        noisy = np.sqrt((img + n_real) ** 2 + n_imag**2)
    elif kind == "speckle":
        # Multiplicative noise I * (1 + eta) where E[(I * eta)^2] = sigma^2 -> std(eta) = sigma / rms(I)
        rms = float(np.sqrt(signal_power))
        rel_sigma = sigma / max(rms, 1e-6)
        mult = rng.normal(0.0, rel_sigma, size=img.shape).astype(np.float32)
        noisy = img * (1.0 + mult)
    else:
        raise ValueError(f"Unsupported noise_type '{noise_type}'. Use gaussian, rician, or speckle.")

    if clip:
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.astype(np.float32)


@dataclass
class ProfileResult:
    """Latency and peak memory measurements for a single callable."""

    mean_ms: float
    std_ms: float
    peak_ram_mb: float
    peak_vram_mb: float


def profile_callable(
    fn: Callable[[], object],
    warmup: int = 3,
    repeats: int = 20,
) -> ProfileResult:
    """Measure mean/std wall-clock latency (ms) and peak RAM/VRAM usage (MB)."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_vram = (
        float(torch.cuda.max_memory_allocated()) / (1024.0**2)
        if torch.cuda.is_available()
        else 0.0
    )
    return ProfileResult(
        mean_ms=float(np.mean(times_ms)),
        std_ms=float(np.std(times_ms)),
        peak_ram_mb=float(peak_bytes) / (1024.0**2),
        peak_vram_mb=peak_vram,
    )

"""Numerical verification suite for Theorems 1-5 of the DisEnergy framework."""

from __future__ import annotations

import time

import gudhi
import numpy as np
import torch

from disenergy.core import DisEnergyConfig, DisEnergyOperator
from disenergy.topology import extract_cubical_persistence


def _make_smooth_test_image(h: int = 128, w: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    fg = 0.35 * np.exp(-((r / 24.0) ** 4))
    bg = 0.20 + 0.05 * np.sin(xx / 16.0) * np.cos(yy / 16.0)
    img = np.stack([bg + fg, bg + 0.7 * fg, bg + 0.4 * fg], axis=-1)
    return img.astype(np.float32)


def test_theorem1_affine_photometric_invariance() -> None:
    """Theorem 1: D_s(a*I + c) = a * D_s(I) and normalized M_s(a*I + c) = M_s(I)."""
    img = _make_smooth_test_image(96, 96)
    a = 1.75
    c = np.array([0.4, -0.15, 0.25], dtype=np.float32)
    img_shift = img + c[None, None, :]
    img_affine = a * img + c[None, None, :]

    cfg = DisEnergyConfig(
        block_sizes=(2, 8, 16),
        energy_mode="homogeneous",
        w2_mode="diagonal",
        delta_zero=1e-14,
        epsilon=1e-4,
    )
    op = DisEnergyOperator(cfg)

    out_base = op(img)
    out_shift = op(img_shift)
    out_affine = op(img_affine)

    for w_s in cfg.block_sizes:
        d_base = out_base.scale_fields[w_s]
        d_shift = out_shift.scale_fields[w_s]
        d_affine = out_affine.scale_fields[w_s]
        assert torch.allclose(d_shift, d_base, atol=1e-5, rtol=1e-4)
        assert torch.allclose(d_affine, a * d_base, atol=1e-4, rtol=1e-3)

        m_base = out_base.scale_manifolds[w_s]
        m_affine = out_affine.scale_manifolds[w_s]
        assert torch.allclose(m_affine, m_base, atol=1e-4, rtol=1e-3)

    assert torch.allclose(out_affine.fused_manifold, out_base.fused_manifold, atol=1e-3)


def test_theorem2_scale_dependent_noise_attenuation() -> None:
    """Theorem 2: Stochastic noise fluctuation Std_xi[D_s(I + xi)] decays as O(1 / w_s)."""
    rng = np.random.default_rng(42)
    img = _make_smooth_test_image(128, 128)
    scales = (1, 2, 4, 8, 16, 32)
    cfg = DisEnergyConfig(block_sizes=scales, energy_mode="homogeneous", w2_mode="diagonal")
    op = DisEnergyOperator(cfg)

    sigma = 0.08
    n_trials = 10
    trials = {s: [] for s in scales}

    for _ in range(n_trials):
        noisy = img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        out_noisy = op(noisy)
        for s in scales:
            trials[s].append(out_noisy.scale_fields[s])

    mean_stds = {s: float(torch.stack(trials[s], dim=0).std(dim=0).mean().item()) for s in scales}

    # Verify strictly monotone noise variance attenuation from fine to large scales
    assert (
        mean_stds[1]
        > mean_stds[2]
        > mean_stds[4]
        > mean_stds[8]
        > mean_stds[16]
        > mean_stds[32]
    )
    # Check O(1/w_s) attenuation factor from w_s=1 to w_s=32
    assert mean_stds[1] / max(mean_stds[32], 1e-8) > 10.0


def test_theorem3_linear_complexity_scaling() -> None:
    """Theorem 3: Runtime grows roughly linearly with pixel count N (not quadratically N^2)."""
    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(4, 16), radius=2))
    timings = {}
    for size in (128, 256, 512):
        x = torch.rand((1, 3, size, size), dtype=torch.float32)
        _ = op(x)  # warmup
        t0 = time.perf_counter()
        for _ in range(5):
            _ = op(x)
        timings[size] = (time.perf_counter() - t0) / 5.0

    # 512x512 has 16x pixels of 128x128; O(N^2) would be ~256x, O(N) should be << 40x
    ratio = timings[512] / max(timings[128], 1e-6)
    assert ratio < 40.0


def test_theorem4_topological_bottleneck_stability() -> None:
    """Theorem 4: Bottleneck distance d_B(Dgm(M_pert), Dgm(M)) <= ||M_pert - M||_inf."""
    img = _make_smooth_test_image(96, 96)
    rng = np.random.default_rng(7)
    pert = img + rng.normal(0.0, 0.03, size=img.shape).astype(np.float32)

    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(4, 16, 32)))
    m_clean = op(img).fused_manifold.squeeze().numpy().astype(np.float64)
    m_pert = op(pert).fused_manifold.squeeze().numpy().astype(np.float64)

    linf = float(np.max(np.abs(m_pert - m_clean)))
    t_clean = extract_cubical_persistence(m_clean)
    t_pert = extract_cubical_persistence(m_pert)

    for dim in (0, 1):
        dgm1 = t_clean.diagrams[dim].astype(np.float64)
        dgm2 = t_pert.diagrams[dim].astype(np.float64)
        db = float(gudhi.bottleneck_distance(dgm1, dgm2))
        assert db <= linf + 1e-3


def test_theorem5_iso_mean_texture_separability() -> None:
    """Theorem 5: Pure texture contrast (equal mean, unequal variance) pops out on DisEnergy."""
    rng = np.random.default_rng(123)
    h, w = 128, 128
    # Background: zero-variance constant 0.5; Foreground disk: mean=0.5, high texture variance
    img = np.full((h, w, 1), 0.5, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    disk = ((yy - 64) ** 2 + (xx - 64) ** 2) <= 24**2
    tex = rng.normal(0.5, 0.18, size=(h, w)).clip(0.0, 1.0).astype(np.float32)
    # Match exact empirical mean of disk to 0.5
    tex[disk] += 0.5 - float(tex[disk].mean())
    img[disk, 0] = tex[disk]

    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(2, 8, 16), eta=1.0))
    fused = op(img).fused_manifold.squeeze().numpy()

    fg_mean = float(fused[disk].mean())
    bg_mean = float(fused[~disk].mean())
    assert fg_mean > bg_mean + 0.25

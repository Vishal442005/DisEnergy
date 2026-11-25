"""Core functional, 3D volumetric, TDA, hybrid, and classical baseline tests."""

from __future__ import annotations

import numpy as np
import torch

from baselines.classical import (
    chan_vese_baseline,
    felzenszwalb_baseline,
    gabor_energy_baseline,
    glcm_entropy_baseline,
    slic_graphcut_baseline,
    spectral_residual_baseline,
    wavelet_energy_baseline,
)
from disenergy.classifiers import build_tabular_classifier, train_topological_mlp
from disenergy.core import DisEnergyConfig, DisEnergyOperator, compute_moran_i, shannon_entropy
from disenergy.hybrid import DisEnergyNet
from disenergy.topology import extract_box_free_mask, extract_cubical_persistence
from disenergy.utils import inject_noise, set_seed


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2.0 * inter / denom) if denom > 0 else 1.0


def test_single_pixel_limit_and_bures_w2() -> None:
    """Ensure w_s=1 limit is finite/non-degenerate and Bures W2 matches diagonal on 1-channel."""
    set_seed(42)
    img1 = np.random.default_rng(1).uniform(0.1, 0.9, size=(64, 64, 1)).astype(np.float32)
    op_diag = DisEnergyOperator(DisEnergyConfig(block_sizes=(1, 4), w2_mode="diagonal"))
    op_bures = DisEnergyOperator(DisEnergyConfig(block_sizes=(1, 4), w2_mode="bures"))

    out_d = op_diag(img1)
    out_b = op_bures(img1)
    assert torch.isfinite(out_d.fused_manifold).all()
    assert torch.isfinite(out_b.fused_manifold).all()
    assert torch.allclose(out_d.scale_fields[4], out_b.scale_fields[4], atol=2e-2)

    # Multi-channel Bures check
    img3 = np.random.default_rng(2).uniform(0.1, 0.9, size=(64, 64, 3)).astype(np.float32)
    out_b3 = op_bures(img3)
    assert out_b3.fused_manifold.shape == (1, 1, 64, 64)
    assert torch.isfinite(out_b3.fused_manifold).all()


def test_moran_entropy_suppresses_white_noise_scale() -> None:
    """Moran's I regularized fusion should down-weight pure uncorrelated white noise maps."""
    rng = np.random.default_rng(9)
    coherent = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    coherent[:, :, 16:48, 16:48] = 1.0
    noise = torch.from_numpy(rng.uniform(0.0, 1.0, size=(1, 1, 64, 64)).astype(np.float32))

    assert float(compute_moran_i(coherent).item()) > 0.8
    assert abs(float(compute_moran_i(noise).item())) < 0.1
    assert float(shannon_entropy(coherent).item()) < float(shannon_entropy(noise).item())


def test_smooth_disk_and_ring_segmentation() -> None:
    """Verify ring-free mask extraction on both solid disk and hollow ring phantoms."""
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2)
    gt_disk = r <= 28

    img = np.full((h, w, 3), 0.15, dtype=np.float32)
    img[gt_disk] = np.array([0.85, 0.55, 0.35], dtype=np.float32)

    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(2, 8, 32)))
    out = op(img)
    fused = out.fused_manifold.squeeze().numpy()
    fine = out.scale_manifolds[2].squeeze().numpy()

    mask_ws = extract_box_free_mask(fused, fine_manifold=fine, method="watershed_filled")
    mask_otsu = extract_box_free_mask(fused, method="otsu_filled")
    assert _dice(mask_ws, gt_disk) > 0.95
    assert _dice(mask_otsu, gt_disk) > 0.80

    # Ring phantom should yield non-trivial H1 loop persistence
    ring = (r >= 18) & (r <= 28)
    img_ring = np.full((h, w, 3), 0.1, dtype=np.float32)
    img_ring[ring] = 0.9
    fused_ring = op(img_ring).fused_manifold.squeeze().numpy()
    tda = extract_cubical_persistence(fused_ring)
    assert tda.diagrams[1].shape[0] >= 1
    assert float(tda.feature_vector.shape[0]) > 20


def test_3d_volumetric_disenergy_and_tda() -> None:
    """Verify 3D volumetric DisEnergy (e.g., BraTS cubes) + 3D cubical PH."""
    zz, yy, xx = np.mgrid[0:32, 0:32, 0:32]
    sphere = ((zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) <= 8**2
    vol = np.full((32, 32, 32, 2), 0.1, dtype=np.float32)
    vol[sphere, 0] = 0.85
    vol[sphere, 1] = 0.60

    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(2, 4, 8), radius=1))
    out3d = op(vol, spatial_dims=3)
    assert out3d.fused_manifold.shape == (1, 1, 32, 32, 32)
    fused3d = out3d.fused_manifold.squeeze().numpy()
    fine3d = out3d.scale_manifolds[2].squeeze().numpy()
    mask3d = extract_box_free_mask(fused3d, fine_manifold=fine3d, method="watershed_filled")
    assert _dice(mask3d, sphere) > 0.88

    tda3d = extract_cubical_persistence(fused3d)
    assert 2 in tda3d.diagrams


def test_noise_injectors_classifiers_hybrid_and_baselines() -> None:
    """Smoke test noise injectors, TDA classifiers, DisEnergyNet, and classical baselines."""
    set_seed(42)
    img = np.full((64, 64, 3), 0.2, dtype=np.float32)
    img[20:44, 20:44] = 0.8

    for ntype in ("gaussian", "rician", "speckle"):
        noisy = inject_noise(img, ntype, snr_db=15.0)
        assert noisy.shape == img.shape
        assert 0.0 <= float(noisy.min()) and float(noisy.max()) <= 1.0

    # Topological classifiers
    x_feat = np.random.default_rng(42).normal(size=(24, 96)).astype(np.float32)
    y_lbl = np.array([0] * 12 + [1] * 12, dtype=np.int64)
    svm = build_tabular_classifier("rbf_svm", seed=42)
    svm.fit(x_feat, y_lbl)
    assert svm.predict_proba(x_feat).shape == (24, 2)

    rf = build_tabular_classifier("random_forest", seed=42)
    rf.fit(x_feat, y_lbl)
    assert rf.predict(x_feat).shape == (24,)

    mlp, scaler = train_topological_mlp(x_feat, y_lbl, num_classes=2, epochs=10)
    with torch.no_grad():
        logits = mlp(torch.from_numpy(scaler.transform(x_feat).astype(np.float32)))
    assert logits.shape == (24, 2)

    # DisEnergyNet hybrid forward
    net = DisEnergyNet(in_channels=3, num_classes=1, base_ch=8)
    inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    out = net(inp)
    assert out.shape == (1, 1, 64, 64)

    # Classical baselines smoke test
    for fn in (
        chan_vese_baseline,
        slic_graphcut_baseline,
        gabor_energy_baseline,
        glcm_entropy_baseline,
        wavelet_energy_baseline,
        felzenszwalb_baseline,
        spectral_residual_baseline,
    ):
        pred = fn(img)
        assert pred.shape == (64, 64) and pred.dtype == bool

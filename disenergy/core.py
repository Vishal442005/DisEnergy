"""Core vectorized 2D & 3D DisEnergy operator, multi-channel Wasserstein-2, and fusion."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DisEnergyConfig:
    """Configuration for the multi-scale DisEnergy operator."""

    block_sizes: tuple[int, ...] = (2, 8, 32)
    radius: int = 2
    gamma: float = 0.5
    beta: float = 1.0
    eta: float = 1.0
    epsilon: float = 1e-4
    gradient_filter: str = "scharr"  # "scharr" | "sobel" | "central"
    energy_mode: str = "homogeneous"  # "homogeneous" | "quadratic"
    w2_mode: str = "diagonal"  # "diagonal" | "bures"
    norm_mode: str = "quantile"  # "quantile" | "minmax"
    quantile_clip: tuple[float, float] = (0.005, 0.995)
    fusion_mode: str = "moran_entropy"  # "moran_entropy" | "inverse_entropy" | "uniform"
    entropy_bins: int = 64
    entropy_tau: float = 0.05
    delta_zero: float = 1e-12


@dataclass
class DisEnergyOutput:
    """Structured output containing per-scale manifolds, fused 3D surface, and weights."""

    scale_fields: dict[int, torch.Tensor] = field(default_factory=dict)
    scale_manifolds: dict[int, torch.Tensor] = field(default_factory=dict)
    fused_manifold: torch.Tensor | None = None
    fusion_weights: dict[int, torch.Tensor] = field(default_factory=dict)


def to_channel_first_tensor(
    image: np.ndarray | torch.Tensor,
    spatial_dims: int = 2,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert 2D/3D NumPy or Torch image to float32 tensor of shape (B, C, *spatial)."""
    if isinstance(image, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(image))
    else:
        t = image
    t = t.to(device=device, dtype=torch.float32)

    if spatial_dims == 2:
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            # Heuristic: (H, W, C) if last dim <= 8 else (B, H, W)
            if t.shape[-1] <= 8 and t.shape[0] > 8:
                t = t.permute(2, 0, 1).unsqueeze(0)
            else:
                t = t.unsqueeze(1)
        elif t.ndim == 4:
            if t.shape[-1] <= 8 and t.shape[1] > 8:
                t = t.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected 2D/3D/4D tensor for spatial_dims=2, got shape {tuple(t.shape)}")
    elif spatial_dims == 3:
        if t.ndim == 3:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 4:
            if t.shape[-1] <= 8 and t.shape[0] > 8:
                t = t.permute(3, 0, 1, 2).unsqueeze(0)
            else:
                t = t.unsqueeze(1)
        elif t.ndim == 5:
            if t.shape[-1] <= 8 and t.shape[1] > 8:
                t = t.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError(f"Expected 3D/4D/5D tensor for spatial_dims=3, got shape {tuple(t.shape)}")
    else:
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
    return t


def _pad_to_multiple(x: torch.Tensor, w_s: int, spatial_dims: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pad spatial dimensions cleanly so each axis is divisible by w_s."""
    spatial = x.shape[2:]
    pads_right = [(w_s - (dim % w_s)) % w_s for dim in spatial]
    if all(p == 0 for p in pads_right):
        return x, spatial

    # F.pad expects reverse order: (w_left, w_right, h_left, h_right, [d_left, d_right])
    pad_tuple: list[int] = []
    for p in reversed(pads_right):
        pad_tuple.extend([0, p])

    can_reflect = all(p < dim for p, dim in zip(pads_right, spatial))
    mode = "reflect" if (can_reflect and spatial_dims == 2) else "replicate"
    return F.pad(x, pad_tuple, mode=mode), spatial


def shannon_entropy(x: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Compute normalized histogram Shannon entropy in [0, 1] per batch element.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (B, 1, *spatial) with values in [0, 1].
    """
    b = x.shape[0]
    flat = x.reshape(b, -1).clamp(0.0, 1.0)
    entropies = []
    log_bins = np.log2(bins)
    for i in range(b):
        hist = torch.histc(flat[i], bins=bins, min=0.0, max=1.0)
        probs = hist / hist.sum().clamp_min(1.0)
        mask = probs > 0
        h = -(probs[mask] * torch.log2(probs[mask])).sum() / log_bins
        entropies.append(h)
    return torch.stack(entropies, dim=0)


def compute_moran_i(x: torch.Tensor) -> torch.Tensor:
    """Compute 4/6-neighbor spatial autocorrelation Moran's I per batch element."""
    b = x.shape[0]
    spatial_dims = x.ndim - 2
    flat = x.reshape(b, -1)
    mean = flat.mean(dim=1, keepdim=True)
    z = (x - mean.view(b, *([1] * (x.ndim - 1))))
    denom = (z * z).reshape(b, -1).sum(dim=1).clamp_min(1e-12)

    n_pixels = flat.shape[1]
    num = torch.zeros(b, device=x.device, dtype=x.dtype)
    edge_count = 0

    for axis in range(2, 2 + spatial_dims):
        dim_len = x.shape[axis]
        if dim_len < 2:
            continue
        sl1 = [slice(None)] * x.ndim
        sl2 = [slice(None)] * x.ndim
        sl1[axis] = slice(0, dim_len - 1)
        sl2[axis] = slice(1, dim_len)
        prod = (z[tuple(sl1)] * z[tuple(sl2)]).reshape(b, -1).sum(dim=1)
        num = num + 2.0 * prod
        other_prod = int(np.prod([x.shape[a] for a in range(2, 2 + spatial_dims) if a != axis]))
        edge_count += 2 * (dim_len - 1) * other_prod

    if edge_count == 0:
        return torch.zeros(b, device=x.device, dtype=x.dtype)
    return (float(n_pixels) / float(edge_count)) * (num / denom)


def normalize_manifold(
    field: torch.Tensor,
    norm_mode: str = "quantile",
    quantile_clip: tuple[float, float] = (0.005, 0.995),
) -> torch.Tensor:
    """Normalize per-image energy divergence field into [0, 1]."""
    b = field.shape[0]
    flat = field.reshape(b, -1)
    if norm_mode == "quantile":
        q_low, q_high = quantile_clip
        lo = torch.quantile(flat, q_low, dim=1, keepdim=True)
        hi = torch.quantile(flat, q_high, dim=1, keepdim=True)
    elif norm_mode == "minmax":
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
    else:
        raise ValueError(f"Unknown norm_mode '{norm_mode}'.")

    span = (hi - lo).clamp_min(1e-12)
    normed = ((flat - lo) / span).clamp(0.0, 1.0)
    return normed.reshape_as(field)


class DisEnergyOperator(nn.Module):
    """Vectorized multi-scale Dirichlet-Wasserstein 3D energy manifold operator."""

    def __init__(self, config: DisEnergyConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or DisEnergyConfig()
        self.register_buffer("_scharr_x", self._make_2d_kernel("scharr")[0], persistent=False)
        self.register_buffer("_scharr_y", self._make_2d_kernel("scharr")[1], persistent=False)
        self.register_buffer("_sobel_x", self._make_2d_kernel("sobel")[0], persistent=False)
        self.register_buffer("_sobel_y", self._make_2d_kernel("sobel")[1], persistent=False)
        self.register_buffer("_central_x", self._make_2d_kernel("central")[0], persistent=False)
        self.register_buffer("_central_y", self._make_2d_kernel("central")[1], persistent=False)

    @staticmethod
    def _make_2d_kernel(kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        if kind == "scharr":
            kx = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]) / 32.0
        elif kind == "sobel":
            kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        elif kind == "central":
            kx = torch.tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]])
        else:
            raise ValueError(f"Unknown gradient filter '{kind}'.")
        ky = kx.t().contiguous()
        return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)

    def _gradient_energy_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sum_c ||grad I_c(p)||_2^2 with reflect padding -> (B, 1, H, W)."""
        c = x.shape[1]
        kind = self.cfg.gradient_filter
        kx = getattr(self, f"_{kind}_x").to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        ky = getattr(self, f"_{kind}_y").to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)

        pad_mode = "reflect" if (x.shape[-2] > 1 and x.shape[-1] > 1) else "replicate"
        xp = F.pad(x, (1, 1, 1, 1), mode=pad_mode)
        gx = F.conv2d(xp, kx, groups=c)
        gy = F.conv2d(xp, ky, groups=c)
        return (gx * gx + gy * gy).sum(dim=1, keepdim=True)

    def _gradient_energy_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 3D central-difference gradient energy sum_c ||grad I_c||_2^2 -> (B, 1, D, H, W)."""
        pad_mode = "replicate"
        xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode=pad_mode)
        gz = 0.5 * (xp[:, :, 2:, 1:-1, 1:-1] - xp[:, :, :-2, 1:-1, 1:-1])
        gy = 0.5 * (xp[:, :, 1:-1, 2:, 1:-1] - xp[:, :, 1:-1, :-2, 1:-1])
        gx = 0.5 * (xp[:, :, 1:-1, 1:-1, 2:] - xp[:, :, 1:-1, 1:-1, :-2])
        return (gz * gz + gy * gy + gx * gx).sum(dim=1, keepdim=True)

    @staticmethod
    def _symmetric_sqrt(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute matrix square root of symmetric PSD matrices (..., C, C)."""
        eye = torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)
        sym = 0.5 * (mat + mat.transpose(-1, -2)) + eps * eye
        eigvals, eigvecs = torch.linalg.eigh(sym)
        sqrt_vals = torch.sqrt(eigvals.clamp_min(0.0))
        return (eigvecs * sqrt_vals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)

    @staticmethod
    def _bures_term(sqrt_cov_k: torch.Tensor, cov_j: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute Tr(Cov_k + Cov_j - 2*(Cov_k^{1/2} Cov_j Cov_k^{1/2})^{1/2})."""
        prod = sqrt_cov_k @ cov_j @ sqrt_cov_k
        prod = 0.5 * (prod + prod.transpose(-1, -2))
        eye = torch.eye(prod.shape[-1], device=prod.device, dtype=prod.dtype)
        eigvals = torch.linalg.eigvalsh(prod + eps * eye).clamp_min(0.0)
        tr_cross = torch.sqrt(eigvals).sum(dim=-1)
        tr_k = torch.diagonal(sqrt_cov_k @ sqrt_cov_k, dim1=-2, dim2=-1).sum(dim=-1)
        tr_j = torch.diagonal(cov_j, dim1=-2, dim2=-1).sum(dim=-1)
        return (tr_k + tr_j - 2.0 * tr_cross).clamp_min(0.0)

    def compute_single_scale_2d(self, x: torch.Tensor, w_s: int) -> torch.Tensor:
        """Compute raw D_s block field for 2D tensor x of shape (B, C, H, W)."""
        x_pad, _ = _pad_to_multiple(x, w_s, spatial_dims=2)
        grad_sq = self._gradient_energy_2d(x_pad)

        if w_s == 1:
            mean_b = x_pad
            grad_b = grad_sq
            var_b = torch.zeros_like(mean_b)
            centered = torch.zeros_like(x_pad)
        else:
            mean_b = F.avg_pool2d(x_pad, kernel_size=w_s, stride=w_s)
            grad_b = F.avg_pool2d(grad_sq, kernel_size=w_s, stride=w_s)
            mean_up = mean_b.repeat_interleave(w_s, dim=-2).repeat_interleave(w_s, dim=-1)
            centered = x_pad - mean_up
            var_b = F.avg_pool2d(centered * centered, kernel_size=w_s, stride=w_s)

        b, c, hb, wb = mean_b.shape
        tr_var = var_b.sum(dim=1, keepdim=True)

        if self.cfg.energy_mode == "homogeneous":
            e_loc = torch.sqrt((grad_b + self.cfg.gamma * tr_var).clamp_min(0.0) + self.cfg.delta_zero)
        elif self.cfg.energy_mode == "quadratic":
            e_loc = grad_b + self.cfg.gamma * tr_var
        else:
            raise ValueError(f"Unknown energy_mode '{self.cfg.energy_mode}'.")

        r = self.cfg.radius
        k_size = 2 * r + 1
        n_win = k_size * k_size
        center_idx = n_win // 2

        # Spatial decay weights across (2r+1)x(2r+1) window, excluding center self-pair
        dy = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        dx = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(dy, dx, indexing="ij")
        dist = torch.sqrt(yy * yy + xx * xx).reshape(-1)
        keep_mask = torch.arange(n_win, device=x.device) != center_idx
        dist_nbr = dist[keep_mask]
        weights = 1.0 / (dist_nbr.pow(self.cfg.beta) + self.cfg.epsilon)  # (K^2 - 1,)

        # Valid neighbor mask to normalize border blocks accurately
        ones = torch.ones((1, 1, hb, wb), device=x.device, dtype=x.dtype)
        valid_nbr = F.unfold(ones, kernel_size=k_size, padding=r).squeeze(0)[keep_mask]  # (K^2-1, Hb*Wb)
        z_k = (valid_nbr * weights.unsqueeze(-1)).sum(dim=0).clamp_min(1e-12)  # (Hb*Wb,)

        # Unfold e_loc and mean_b
        e_nbr = F.unfold(e_loc, kernel_size=k_size, padding=r)[:, keep_mask, :]  # (B, K^2-1, L)
        e_ctr = e_loc.reshape(b, 1, hb * wb)
        de = torch.abs(e_ctr - e_nbr)  # (B, K^2-1, L)

        mean_nbr = F.unfold(mean_b, kernel_size=k_size, padding=r).view(b, c, n_win, hb * wb)[:, :, keep_mask, :]
        mean_ctr = mean_b.reshape(b, c, 1, hb * wb)
        mean_diff_sq = ((mean_ctr - mean_nbr) ** 2).sum(dim=1)  # (B, K^2-1, L)

        if self.cfg.w2_mode == "diagonal" or w_s == 1:
            std_b = torch.sqrt(var_b + 1e-12) if w_s > 1 else torch.zeros_like(var_b)
            std_nbr = F.unfold(std_b, kernel_size=k_size, padding=r).view(b, c, n_win, hb * wb)[:, :, keep_mask, :]
            std_ctr = std_b.reshape(b, c, 1, hb * wb)
            std_diff_sq = ((std_ctr - std_nbr) ** 2).sum(dim=1)
            w2 = torch.sqrt((mean_diff_sq + std_diff_sq).clamp_min(0.0) + 1e-12)
        elif self.cfg.w2_mode == "bures":
            # Full C x C covariance per block via centered outer product (zero cancellation)
            outer_pix = (centered.unsqueeze(2) * centered.unsqueeze(1)).reshape(
                b, c * c, x_pad.shape[-2], x_pad.shape[-1]
            )
            cov_b = F.avg_pool2d(outer_pix, kernel_size=w_s, stride=w_s).reshape(b, c, c, hb, wb)
            cov_flat = cov_b.reshape(b, c * c, hb, wb)
            cov_nbr = F.unfold(cov_flat, kernel_size=k_size, padding=r).view(b, c, c, n_win, hb * wb)
            cov_nbr = cov_nbr[:, :, :, keep_mask, :].permute(0, 3, 4, 1, 2)  # (B, K^2-1, L, C, C)
            cov_ctr = cov_b.reshape(b, c, c, hb * wb).permute(0, 3, 1, 2)  # (B, L, C, C)
            sqrt_ctr = self._symmetric_sqrt(cov_ctr).unsqueeze(1)  # (B, 1, L, C, C)
            bures = self._bures_term(sqrt_ctr, cov_nbr)  # (B, K^2-1, L)
            w2 = torch.sqrt((mean_diff_sq + bures).clamp_min(0.0) + 1e-12)
        else:
            raise ValueError(f"Unknown w2_mode '{self.cfg.w2_mode}'.")

        contrib = (de + self.cfg.eta * w2) * weights.view(1, -1, 1) * valid_nbr.unsqueeze(0)
        d_s = contrib.sum(dim=1) / z_k.unsqueeze(0)
        return d_s.reshape(b, 1, hb, wb)

    def compute_single_scale_3d(self, x: torch.Tensor, w_s: int) -> torch.Tensor:
        """Memory-safe vectorized 3D D_s block field for tensor (B, C, D, H, W)."""
        x_pad, _ = _pad_to_multiple(x, w_s, spatial_dims=3)
        grad_sq = self._gradient_energy_3d(x_pad)

        if w_s == 1:
            mean_b = x_pad
            grad_b = grad_sq
            var_b = torch.zeros_like(mean_b)
        else:
            mean_b = F.avg_pool3d(x_pad, kernel_size=w_s, stride=w_s)
            grad_b = F.avg_pool3d(grad_sq, kernel_size=w_s, stride=w_s)
            mean_up = (
                mean_b.repeat_interleave(w_s, dim=-3)
                .repeat_interleave(w_s, dim=-2)
                .repeat_interleave(w_s, dim=-1)
            )
            centered = x_pad - mean_up
            var_b = F.avg_pool3d(centered * centered, kernel_size=w_s, stride=w_s)

        std_b = torch.sqrt(var_b + 1e-12) if w_s > 1 else torch.zeros_like(var_b)
        tr_var = var_b.sum(dim=1, keepdim=True)
        if self.cfg.energy_mode == "homogeneous":
            e_loc = torch.sqrt((grad_b + self.cfg.gamma * tr_var).clamp_min(0.0) + self.cfg.delta_zero)
        else:
            e_loc = grad_b + self.cfg.gamma * tr_var

        b, _, db, hb, wb = mean_b.shape
        num = torch.zeros((b, 1, db, hb, wb), device=x.device, dtype=x.dtype)
        z_k = torch.zeros((1, 1, db, hb, wb), device=x.device, dtype=x.dtype)
        r = self.cfg.radius

        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    dist = float(np.sqrt(dz * dz + dy * dy + dx * dx))
                    weight = 1.0 / (dist**self.cfg.beta + self.cfg.epsilon)

                    z_src = slice(max(0, dz), db + min(0, dz))
                    y_src = slice(max(0, dy), hb + min(0, dy))
                    x_src = slice(max(0, dx), wb + min(0, dx))

                    z_dst = slice(max(0, -dz), db + min(0, -dz))
                    y_dst = slice(max(0, -dy), hb + min(0, -dy))
                    x_dst = slice(max(0, -dx), wb + min(0, -dx))

                    de = torch.abs(
                        e_loc[:, :, z_dst, y_dst, x_dst] - e_loc[:, :, z_src, y_src, x_src]
                    )
                    dm = ((mean_b[:, :, z_dst, y_dst, x_dst] - mean_b[:, :, z_src, y_src, x_src]) ** 2).sum(
                        dim=1, keepdim=True
                    )
                    ds = ((std_b[:, :, z_dst, y_dst, x_dst] - std_b[:, :, z_src, y_src, x_src]) ** 2).sum(
                        dim=1, keepdim=True
                    )
                    w2 = torch.sqrt((dm + ds).clamp_min(0.0) + 1e-12)

                    num[:, :, z_dst, y_dst, x_dst] += weight * (de + self.cfg.eta * w2)
                    z_k[:, :, z_dst, y_dst, x_dst] += weight

        return num / z_k.clamp_min(1e-12)

    def _compute_fusion_weights(
        self,
        manifolds: dict[int, torch.Tensor],
        scales: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        """Compute per-image scale fusion weights omega_s (B, 1, ...)."""
        b = next(iter(manifolds.values())).shape[0]
        device = next(iter(manifolds.values())).device
        dtype = next(iter(manifolds.values())).dtype

        if self.cfg.fusion_mode == "uniform":
            w = torch.full((b,), 1.0 / len(scales), device=device, dtype=dtype)
            return {s: w for s in scales}

        raw_scores: list[torch.Tensor] = []
        for s in scales:
            m = manifolds[s]
            h = shannon_entropy(m, bins=self.cfg.entropy_bins)
            if self.cfg.fusion_mode == "inverse_entropy":
                score = 1.0 / (h + self.cfg.entropy_tau)
            elif self.cfg.fusion_mode == "moran_entropy":
                moran = compute_moran_i(m).clamp_min(0.0)
                score = moran / (h + self.cfg.entropy_tau)
            else:
                raise ValueError(f"Unknown fusion_mode '{self.cfg.fusion_mode}'.")
            raw_scores.append(score)

        stacked = torch.stack(raw_scores, dim=0)  # (S, B)
        denom = stacked.sum(dim=0, keepdim=True)
        # Fallback to uniform for flat/zero-signal inputs
        uniform = torch.full_like(stacked, 1.0 / len(scales))
        weights = torch.where(denom > 1e-8, stacked / denom.clamp_min(1e-8), uniform)
        return {s: weights[idx] for idx, s in enumerate(scales)}

    @torch.no_grad()
    def forward(
        self,
        image: np.ndarray | torch.Tensor,
        spatial_dims: int = 2,
        block_sizes: Iterable[int] | None = None,
    ) -> DisEnergyOutput:
        """Compute multi-scale DisEnergy fields, normalized 3D manifolds, and fused surface."""
        x = to_channel_first_tensor(image, spatial_dims=spatial_dims, device=self._scharr_x.device)
        scales = tuple(block_sizes) if block_sizes is not None else tuple(self.cfg.block_sizes)
        orig_spatial = x.shape[2:]
        interp_mode = "bilinear" if spatial_dims == 2 else "trilinear"

        scale_fields: dict[int, torch.Tensor] = {}
        scale_manifolds: dict[int, torch.Tensor] = {}

        for w_s in scales:
            if spatial_dims == 2:
                d_s = self.compute_single_scale_2d(x, w_s=w_s)
            else:
                d_s = self.compute_single_scale_3d(x, w_s=w_s)
            scale_fields[w_s] = d_s

            # Pad target size matches _pad_to_multiple output, then crop to orig_spatial
            padded_shape = tuple(
                dim + ((w_s - (dim % w_s)) % w_s) for dim in orig_spatial
            )
            up = F.interpolate(d_s, size=padded_shape, mode=interp_mode, align_corners=False)
            slices = (slice(None), slice(None), *(slice(0, dim) for dim in orig_spatial))
            up = up[slices]
            scale_manifolds[w_s] = normalize_manifold(
                up, norm_mode=self.cfg.norm_mode, quantile_clip=self.cfg.quantile_clip
            )

        weights = self._compute_fusion_weights(scale_manifolds, scales)
        fused = torch.zeros_like(scale_manifolds[scales[0]])
        view_shape = (x.shape[0], 1, *([1] * spatial_dims))
        for s in scales:
            fused = fused + weights[s].view(*view_shape) * scale_manifolds[s]
        fused = normalize_manifold(fused, norm_mode="minmax")

        return DisEnergyOutput(
            scale_fields=scale_fields,
            scale_manifolds=scale_manifolds,
            fused_manifold=fused,
            fusion_weights=weights,
        )

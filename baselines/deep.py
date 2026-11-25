"""Lightweight deep-learning baselines (4 GB VRAM / FP16 AMP safe)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from disenergy.hybrid import ConvBlock, LightweightUNet


class AttentionGate(nn.Module):
    """Additive attention gate for Attention U-Net."""

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.wx = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(F.relu(self.wg(g) + self.wx(x), inplace=True))
        return x * alpha


class AttentionUNet(nn.Module):
    """Lightweight Attention U-Net baseline."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.att3 = AttentionGate(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.att2 = AttentionGate(base_ch * 2, base_ch * 2, base_ch)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.att1 = AttentionGate(base_ch, base_ch, max(8, base_ch // 2))
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        g3 = self.up3(b)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], dim=1))
        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))
        return self.head(d1)


class UNetPlusPlus(nn.Module):
    """3-level nested UNet++ baseline safe for 4 GB VRAM."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_ch: int = 24) -> None:
        super().__init__()
        nb = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.x0_0 = ConvBlock(in_channels, nb[0])
        self.x1_0 = ConvBlock(nb[0], nb[1])
        self.x2_0 = ConvBlock(nb[1], nb[2])
        self.x3_0 = ConvBlock(nb[2], nb[3])

        self.x0_1 = ConvBlock(nb[0] + nb[1], nb[0])
        self.x1_1 = ConvBlock(nb[1] + nb[2], nb[1])
        self.x2_1 = ConvBlock(nb[2] + nb[3], nb[2])

        self.x0_2 = ConvBlock(nb[0] * 2 + nb[1], nb[0])
        self.x1_2 = ConvBlock(nb[1] * 2 + nb[2], nb[1])

        self.x0_3 = ConvBlock(nb[0] * 3 + nb[1], nb[0])
        self.head = nn.Conv2d(nb[0], num_classes, 1)

    @staticmethod
    def _up(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(F.max_pool2d(x0_0, 2))
        x0_1 = self.x0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))

        x2_0 = self.x2_0(F.max_pool2d(x1_0, 2))
        x1_1 = self.x1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))

        x3_0 = self.x3_0(F.max_pool2d(x2_0, 2))
        x2_1 = self.x2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        return self.head(x0_3)


class PatchAttentionBlock(nn.Module):
    """Lightweight factorized self-attention block for SegFormer-style encoder."""

    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.norm1(x.flatten(2).transpose(1, 2))
        kv_spatial = self.sr(x)
        kv = self.norm1(kv_spatial.flatten(2).transpose(1, 2))
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        tokens = x.flatten(2).transpose(1, 2) + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class LightweightSegFormer(nn.Module):
    """SegFormer-B0-style hierarchical transformer segmentation baseline."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, dims: tuple[int, int, int] = (32, 64, 128)) -> None:
        super().__init__()
        d1, d2, d3 = dims
        self.stem1 = nn.Sequential(nn.Conv2d(in_channels, d1, 7, stride=4, padding=3), nn.BatchNorm2d(d1), nn.GELU())
        self.blk1 = PatchAttentionBlock(d1, num_heads=2, sr_ratio=4)

        self.stem2 = nn.Sequential(nn.Conv2d(d1, d2, 3, stride=2, padding=1), nn.BatchNorm2d(d2), nn.GELU())
        self.blk2 = PatchAttentionBlock(d2, num_heads=4, sr_ratio=2)

        self.stem3 = nn.Sequential(nn.Conv2d(d2, d3, 3, stride=2, padding=1), nn.BatchNorm2d(d3), nn.GELU())
        self.blk3 = PatchAttentionBlock(d3, num_heads=4, sr_ratio=1)

        self.proj = nn.ModuleList([nn.Conv2d(d, d1, 1) for d in dims])
        self.fuse = nn.Sequential(nn.Conv2d(d1 * 3, d1, 1), nn.BatchNorm2d(d1), nn.GELU(), nn.Conv2d(d1, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.blk1(self.stem1(x))
        f2 = self.blk2(self.stem2(f1))
        f3 = self.blk3(self.stem3(f2))

        target_sz = f1.shape[2:]
        p1 = self.proj[0](f1)
        p2 = F.interpolate(self.proj[1](f2), size=target_sz, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj[2](f3), size=target_sz, mode="bilinear", align_corners=False)
        out = self.fuse(torch.cat([p1, p2, p3], dim=1))
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


def build_dl_segmentation_model(name: str, in_channels: int = 3, num_classes: int = 1) -> nn.Module:
    """Factory for 4 GB VRAM-safe segmentation baselines."""
    key = name.lower()
    if key == "unet":
        return LightweightUNet(in_channels=in_channels, num_classes=num_classes, base_ch=32)
    if key in ("attention_unet", "attn_unet"):
        return AttentionUNet(in_channels=in_channels, num_classes=num_classes, base_ch=32)
    if key in ("unet++", "unetpp"):
        return UNetPlusPlus(in_channels=in_channels, num_classes=num_classes, base_ch=24)
    if key in ("segformer", "segformer_b0"):
        return LightweightSegFormer(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown DL baseline '{name}'.")


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Balanced Dice + BCE loss for binary segmentation."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + smooth) / (denom + smooth)).mean()
    return 0.5 * bce + 0.5 * dice


def train_segmentation_baseline(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 1e-3,
    grad_accum: int = 2,
    device: str = "auto",
    seed: int = 42,
) -> nn.Module:
    """Train a segmentation model with optional CUDA FP16 AMP and gradient accumulation."""
    torch.manual_seed(seed)
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    use_amp = dev.type == "cuda"
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ds_x = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    if ds_x.ndim == 4 and ds_x.shape[-1] <= 8:
        ds_x = ds_x.permute(0, 3, 1, 2)
    ds_y = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    if ds_y.ndim == 3:
        ds_y = ds_y.unsqueeze(1)

    n = ds_x.shape[0]
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        opt.zero_grad()
        for step, start in enumerate(range(0, n, batch_size)):
            idx = perm[start : start + batch_size]
            bx = ds_x[idx].to(dev)
            by = ds_y[idx].to(dev)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = dice_bce_loss(model(bx), by) / float(grad_accum)
            scaler.scale(loss).backward()
            if ((step + 1) % grad_accum == 0) or (start + batch_size >= n):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
    model.eval()
    return model

"""DisEnergy-Net: Plug-and-play manifold channel injection for neural backbones."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .core import DisEnergyConfig, DisEnergyOperator


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightUNet(nn.Module):
    """4-GB-VRAM-friendly U-Net backbone configurable for RGB/MRI or DisEnergy-Net channels."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class DisEnergyNet(nn.Module):
    """Augments any segmentation backbone with multi-scale DisEnergy manifold priors."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_ch: int = 32,
        include_per_scale: bool = True,
        operator_cfg: DisEnergyConfig | None = None,
    ) -> None:
        super().__init__()
        self.operator = DisEnergyOperator(operator_cfg)
        self.include_per_scale = include_per_scale
        n_extra = (len(self.operator.cfg.block_sizes) + 1) if include_per_scale else 1
        self.backbone = LightweightUNet(
            in_channels=in_channels + n_extra,
            num_classes=num_classes,
            base_ch=base_ch,
        )

    def build_augmented_input(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.operator(x, spatial_dims=2)
            priors = []
            if self.include_per_scale:
                for s in self.operator.cfg.block_sizes:
                    priors.append(out.scale_manifolds[s])
            priors.append(out.fused_manifold)
        return torch.cat([x, *priors], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aug = self.build_augmented_input(x)
        return self.backbone(aug)

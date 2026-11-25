"""Profile trainable parameters, peak RAM/VRAM, and per-image latency (256x256)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from baselines.deep import build_dl_segmentation_model
from disenergy.core import DisEnergyConfig, DisEnergyOperator
from disenergy.hybrid import DisEnergyNet
from disenergy.utils import profile_callable, resolve_device


def run_profiling(out_dir: Path, image_size: int = 256, device: str = "auto") -> pd.DataFrame:
    """Measure efficiency metrics for DisEnergy vs lightweight DL baselines."""
    dev = resolve_device(device)
    x = torch.rand((1, 3, image_size, image_size), dtype=torch.float32, device=dev)

    models: dict[str, torch.nn.Module] = {
        "DisEnergy_Operator": DisEnergyOperator(DisEnergyConfig(block_sizes=(2, 8, 32))).to(dev),
        "UNet": build_dl_segmentation_model("unet", in_channels=3).to(dev),
        "Attention_UNet": build_dl_segmentation_model("attention_unet", in_channels=3).to(dev),
        "UNet++": build_dl_segmentation_model("unet++", in_channels=3).to(dev),
        "SegFormer_B0": build_dl_segmentation_model("segformer", in_channels=3).to(dev),
        "DisEnergyNet": DisEnergyNet(in_channels=3, base_ch=32).to(dev),
    }

    rows = []
    for name, mod in models.items():
        mod.eval()
        trainable_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)

        def _step(m: torch.nn.Module = mod) -> object:
            with torch.no_grad():
                return m(x)

        prof = profile_callable(_step, warmup=2, repeats=10)
        rows.append(
            {
                "model": name,
                "device": str(dev),
                "image_size": image_size,
                "trainable_params": int(trainable_params),
                "mean_ms": round(prof.mean_ms, 3),
                "std_ms": round(prof.std_ms, 3),
                "peak_ram_mb": round(prof.peak_ram_mb, 2),
                "peak_vram_mb": round(prof.peak_vram_mb, 2),
            }
        )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "compute_profile.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DisEnergy and DL baselines.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/profiling"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    df = run_profiling(args.out_dir, image_size=args.image_size, device=args.device)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

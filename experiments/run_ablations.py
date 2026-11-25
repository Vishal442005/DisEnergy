"""Inner-fold ablation runner covering mathematical, fusion, normalization, and grid axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.metrics import compute_segmentation_metrics
from disenergy.core import DisEnergyConfig, DisEnergyOperator
from disenergy.topology import extract_box_free_mask
from disenergy.utils import set_seed


def run_ablations(split_json: Path, out_dir: Path, fold_idx: int = 0, max_val_samples: int = 50) -> pd.DataFrame:
    """Evaluate all 7 ablation dimensions strictly on the inner validation fold."""
    set_seed(42)
    manifest = json.loads(split_json.read_text())
    id_to_rec = {r["sample_id"]: r for r in manifest["records"]}
    val_ids = manifest["folds"][fold_idx]["val_ids"][:max_val_samples]
    samples = [(sid, np.load(id_to_rec[sid]["image_path"]), np.load(id_to_rec[sid]["mask_path"]).astype(bool)) for sid in val_ids]

    variants: list[tuple[str, DisEnergyConfig, str]] = [
        ("Full_DisEnergy_Default", DisEnergyConfig(), "watershed_filled"),
        ("Abl_Quadratic_Energy", DisEnergyConfig(energy_mode="quadratic"), "watershed_filled"),
        ("Abl_Dirichlet_Only_eta0", DisEnergyConfig(eta=0.0), "watershed_filled"),
        ("Abl_W2_Only_HighEta", DisEnergyConfig(eta=10.0), "watershed_filled"),
        ("Abl_Bures_FullW2", DisEnergyConfig(block_sizes=(4, 16, 32), w2_mode="bures"), "watershed_filled"),
        ("Abl_MinMax_Norm", DisEnergyConfig(norm_mode="minmax"), "watershed_filled"),
        ("Abl_InverseEntropy_Fusion", DisEnergyConfig(fusion_mode="inverse_entropy"), "watershed_filled"),
        ("Abl_Uniform_Fusion", DisEnergyConfig(fusion_mode="uniform"), "watershed_filled"),
        ("Abl_Otsu_Extractor", DisEnergyConfig(), "otsu_filled"),
    ]

    # Hyperparameter sensitivity grid on inner validation fold
    for beta in (0.5, 1.0, 2.0):
        for r in (1, 2, 3):
            variants.append((f"Grid_beta{beta}_r{r}", DisEnergyConfig(beta=beta, radius=r), "watershed_filled"))

    rows = []
    for name, cfg, ext_method in variants:
        op = DisEnergyOperator(cfg)
        for sid, img, gt in samples:
            out = op(img)
            fused = out.fused_manifold.squeeze().numpy()
            fine = out.scale_manifolds[cfg.block_sizes[0]].squeeze().numpy()
            pred = extract_box_free_mask(fused, fine_manifold=fine if ext_method == "watershed_filled" else None, method=ext_method)
            m = compute_segmentation_metrics(pred, gt)
            rows.append({"variant": name, "sample_id": sid, **m})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inner-fold DisEnergy ablations.")
    parser.add_argument("--split-json", type=Path, default=Path("data/splits/synthetic.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--max-val-samples", type=int, default=50)
    args = parser.parse_args()

    df = run_ablations(args.split_json, args.out_dir, max_val_samples=args.max_val_samples)
    print(df.groupby("variant")[["dsc", "iou", "hd95", "boundary_f1"]].mean().round(4))


if __name__ == "__main__":
    main()

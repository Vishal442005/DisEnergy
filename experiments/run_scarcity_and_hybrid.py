"""Regime 2: Label scarcity (1%-100%) and DisEnergyNet plug-in synergy benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.make_figures import plot_curves
from analysis.metrics import compute_segmentation_metrics
from baselines.deep import build_dl_segmentation_model, train_segmentation_baseline
from disenergy.hybrid import DisEnergyNet
from disenergy.utils import resolve_device, set_seed


def run_scarcity_suite(
    split_json: Path,
    out_dir: Path,
    fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
    epochs: int = 10,
    device: str = "auto",
    seed: int = 42,
) -> pd.DataFrame:
    """Compare UNet, AttentionUNet, SegFormer, and DisEnergyNet across label fractions."""
    set_seed(seed)
    dev = resolve_device(device)
    manifest = json.loads(split_json.read_text())
    id_to_rec = {r["sample_id"]: r for r in manifest["records"]}
    fold0 = manifest["folds"][0]

    train_ids = fold0["train_ids"] + fold0["val_ids"]
    test_ids = fold0["test_ids"]

    x_tr_all = np.stack([np.load(id_to_rec[i]["image_path"]) for i in train_ids], axis=0)
    y_tr_all = np.stack([np.load(id_to_rec[i]["mask_path"]).astype(np.float32) for i in train_ids], axis=0)
    x_te = np.stack([np.load(id_to_rec[i]["image_path"]) for i in test_ids], axis=0)
    y_te = np.stack([np.load(id_to_rec[i]["mask_path"]).astype(bool) for i in test_ids], axis=0)

    in_ch = x_tr_all.shape[-1] if x_tr_all.ndim == 4 else 1
    x_te_t = torch.from_numpy(x_te.transpose(0, 3, 1, 2) if x_te.ndim == 4 else x_te[:, None]).to(dev)

    rows = []
    rng = np.random.default_rng(seed)
    for frac in fractions:
        k = max(2, int(np.ceil(frac * len(x_tr_all))))
        idx = rng.choice(len(x_tr_all), size=k, replace=False)

        candidates: dict[str, torch.nn.Module] = {
            "UNet": build_dl_segmentation_model("unet", in_channels=in_ch),
            "Attention_UNet": build_dl_segmentation_model("attention_unet", in_channels=in_ch),
            "SegFormer_B0": build_dl_segmentation_model("segformer", in_channels=in_ch),
            "DisEnergyNet": DisEnergyNet(in_channels=in_ch, base_ch=32),
        }

        for mname, model in candidates.items():
            trained = train_segmentation_baseline(
                model,
                x_tr_all[idx],
                y_tr_all[idx],
                epochs=epochs,
                batch_size=min(4, k),
                device=str(dev),
                seed=seed,
            )
            with torch.no_grad():
                probs_list = []
                for start in range(0, len(x_te_t), 8):
                    batch = x_te_t[start : start + 8]
                    probs_list.append(torch.sigmoid(trained(batch)).squeeze(1).cpu().numpy())
                probs = np.concatenate(probs_list, axis=0)
            for j, sid in enumerate(test_ids):
                pred = probs[j] >= 0.5
                m = compute_segmentation_metrics(pred, y_te[j])
                rows.append({"sample_id": sid, "fraction": frac, "method": mname, **m})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "regime2_scarcity.csv", index=False)

    series = {}
    for mname in ["DisEnergyNet", "UNet", "Attention_UNet", "SegFormer_B0"]:
        sub = df[df["method"] == mname].groupby("fraction")["dsc"]
        means = [float(sub.mean().loc[f]) for f in fractions]
        stds = [float(sub.std().loc[f]) for f in fractions]
        series[mname] = (means, stds)

    plot_curves(
        [f * 100.0 for f in fractions],
        series,
        xlabel="Labeled Training Data (%)",
        ylabel="Test Dice Similarity Coefficient (DSC)",
        title="Label-Scarcity Learning Curves",
        out_path=out_dir / "fig_scarcity_curves.png",
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Regime 2 label-scarcity and DisEnergyNet suite.")
    parser.add_argument("--split-json", type=Path, default=Path("data/splits/synthetic.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scarcity"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    df = run_scarcity_suite(args.split_json, args.out_dir, epochs=args.epochs, device=args.device)
    print(df.groupby(["fraction", "method"])["dsc"].mean().unstack("method").round(4))


if __name__ == "__main__":
    main()

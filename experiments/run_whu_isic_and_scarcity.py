"""Benchmark ISIC 2018 (dermoscopy), WHU Building (remote sensing), and CPU-fast Regime 2 Scarcity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.transform import resize

from analysis.export_latex import format_benchmark_table
from analysis.make_figures import plot_critical_difference, plot_curves
from analysis.metrics import compute_segmentation_metrics
from analysis.stats_tests import friedman_nemenyi
from baselines.classical import (
    chan_vese_baseline,
    felzenszwalb_baseline,
    gabor_energy_baseline,
    spectral_residual_baseline,
    wavelet_energy_baseline,
)
from baselines.deep import build_dl_segmentation_model, train_segmentation_baseline
from disenergy.core import DisEnergyConfig, DisEnergyOperator
from disenergy.hybrid import DisEnergyNet
from disenergy.topology import extract_box_free_mask
from disenergy.utils import set_seed


def run_unsupervised_dataset(split_json: Path, out_dir: Path, name: str, caption: str, label: str) -> pd.DataFrame:
    """Evaluate DisEnergy vs classical baselines on any segmented dataset."""
    recs = json.loads(split_json.read_text())["records"]
    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(2, 8, 32), beta=2.0, radius=1))
    rows = []
    for r in recs:
        img = np.load(r["image_path"])
        gt = np.load(r["mask_path"]).astype(bool)
        if not np.any(gt):
            continue
        out = op(img)
        pred_de = extract_box_free_mask(
            out.fused_manifold.squeeze().numpy(),
            fine_manifold=out.scale_manifolds[2].squeeze().numpy(),
            method="watershed_filled",
        )
        rows.append({"sample_id": r["sample_id"], "method": "DisEnergy", **compute_segmentation_metrics(pred_de, gt)})
        for bname, fn in [
            ("Chan_Vese", chan_vese_baseline),
            ("Felzenszwalb", felzenszwalb_baseline),
            ("Gabor_Energy", gabor_energy_baseline),
            ("Wavelet_Energy", wavelet_energy_baseline),
            ("Spectral_Residual", spectral_residual_baseline),
        ]:
            rows.append({"sample_id": r["sample_id"], "method": bname, **compute_segmentation_metrics(fn(img), gt)})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}_unsupervised.csv", index=False)
    tex = format_benchmark_table(
        df,
        metric_cols=["dsc", "iou", "hd95", "boundary_f1", "specificity"],
        caption=caption,
        label=label,
    )
    (out_dir / f"{name}_table.tex").write_text(tex)
    return df


def run_cpu_scarcity_128(
    split_json: Path,
    out_dir: Path,
    name: str,
    fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
    epochs: int = 12,
) -> pd.DataFrame:
    """Run CPU-feasible 128x128 Regime 2 scarcity curves (UNet, AttentionUNet, SegFormer, DisEnergyNet)."""
    set_seed(42)
    m = json.loads(split_json.read_text())
    id2r = {r["sample_id"]: r for r in m["records"]}
    fold0 = m["folds"][0]
    tr_ids = fold0["train_ids"] + fold0["val_ids"]
    te_ids = fold0["test_ids"]

    def _load_128(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for sid in ids:
            img = resize(np.load(id2r[sid]["image_path"]), (128, 128, 3), order=1, preserve_range=True).astype(np.float32)
            msk = (resize(np.load(id2r[sid]["mask_path"]).astype(np.float32), (128, 128), order=0, preserve_range=True) > 0.5)
            xs.append(img)
            ys.append(msk.astype(np.float32))
        return np.stack(xs), np.stack(ys)

    x_tr_all, y_tr_all = _load_128(tr_ids)
    x_te, y_te_f = _load_128(te_ids)
    y_te = y_te_f > 0.5
    x_te_t = torch.from_numpy(x_te.transpose(0, 3, 1, 2))

    rng = np.random.default_rng(42)
    rows = []
    for frac in fractions:
        k = max(2, int(np.ceil(frac * len(x_tr_all))))
        idx = rng.choice(len(x_tr_all), size=k, replace=False)

        models = {
            "UNet": build_dl_segmentation_model("unet", in_channels=3),
            "Attention_UNet": build_dl_segmentation_model("attention_unet", in_channels=3),
            "SegFormer_B0": build_dl_segmentation_model("segformer", in_channels=3),
            "DisEnergyNet": DisEnergyNet(in_channels=3, base_ch=16),
        }
        # Shrink UNet / Attention_UNet base_ch to 16 for fair parameter-matched CPU comparison
        from baselines.deep import AttentionUNet
        from disenergy.hybrid import LightweightUNet

        models["UNet"] = LightweightUNet(in_channels=3, base_ch=16)
        models["Attention_UNet"] = AttentionUNet(in_channels=3, base_ch=16)

        for mname, mod in models.items():
            trained = train_segmentation_baseline(
                mod,
                x_tr_all[idx],
                y_tr_all[idx],
                epochs=epochs,
                batch_size=min(4, k),
                device="cpu",
                seed=42,
            )
            with torch.no_grad():
                probs = torch.sigmoid(trained(x_te_t)).squeeze(1).numpy()
            for j, sid in enumerate(te_ids):
                rows.append({"sample_id": sid, "fraction": frac, "method": mname, **compute_segmentation_metrics(probs[j] >= 0.5, y_te[j])})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}_scarcity.csv", index=False)

    series = {}
    for mname in ["DisEnergyNet", "UNet", "Attention_UNet", "SegFormer_B0"]:
        sub = df[df["method"] == mname].groupby("fraction")["dsc"]
        series[mname] = ([float(sub.mean().loc[f]) for f in fractions], [float(sub.std().loc[f]) for f in fractions])

    plot_curves(
        [f * 100.0 for f in fractions],
        series,
        xlabel="Labeled Training Data (%)",
        ylabel="Test DSC",
        title=f"Label-Scarcity Synergy on {name.upper()} (128x128)",
        out_path=out_dir / f"fig5_{name}_scarcity.png",
    )
    return df


def main() -> None:
    set_seed(42)
    out_root = Path("outputs/full_local")

    if Path("data/splits/whu.json").exists():
        df_whu = run_unsupervised_dataset(
            Path("data/splits/whu.json"),
            out_root / "whu",
            "whu",
            "Zero-Shot Building Extraction on WHU Building Dataset (n=150).",
            "tab:whu",
        )
        print("=== WHU Building Dataset ===")
        print(df_whu.groupby("method")[["dsc", "iou", "hd95", "boundary_f1", "specificity"]].mean().round(4))

    if Path("data/splits/isic2018.json").exists():
        df_isic = run_unsupervised_dataset(
            Path("data/splits/isic2018.json"),
            out_root / "isic2018",
            "isic2018",
            "Zero-Shot Dermoscopy Lesion Segmentation on ISIC 2018 (n=100).",
            "tab:isic",
        )
        print("=== ISIC 2018 Dermoscopy ===")
        print(df_isic.groupby("method")[["dsc", "iou", "hd95", "boundary_f1", "specificity"]].mean().round(4))

    # Regime 2 real scarcity curves on DRIVE (40 scans) & WHU Building (150 scans)
    df_drv_sc = run_cpu_scarcity_128(Path("data/splits/drive.json"), out_root / "scarcity", "drive", epochs=12)
    print("=== DRIVE Regime 2 Scarcity (Mean DSC) ===")
    print(df_drv_sc.groupby(["fraction", "method"])["dsc"].mean().unstack("method").round(4))

    if Path("data/splits/whu.json").exists():
        df_whu_sc = run_cpu_scarcity_128(Path("data/splits/whu.json"), out_root / "scarcity", "whu", epochs=10)
        print("=== WHU Regime 2 Scarcity (Mean DSC) ===")
        print(df_whu_sc.groupby(["fraction", "method"])["dsc"].mean().unstack("method").round(4))

    # Updated 5-dataset Friedman-Nemenyi CD Diagram (ECSSD, DRIVE, MVTec, WHU, ISIC2018)
    piv_frames = []
    for tag, csv_p in [
        ("ECSSD", out_root / "ecssd/ecssd_affine_stress.csv"),
        ("DRIVE", out_root / "drive/drive_full40.csv"),
        ("MVTec_AD", out_root / "mvtec/mvtec_benchmark.csv"),
        ("WHU_Building", out_root / "whu/whu_unsupervised.csv"),
        ("ISIC_2018", out_root / "isic2018/isic2018_unsupervised.csv"),
    ]:
        if csv_p.exists():
            sub = pd.read_csv(csv_p)
            sub["method"] = sub["method"].replace({"DisEnergy_Green": "DisEnergy"})
            means = sub.groupby("method")["dsc"].mean()
            means.name = tag
            piv_frames.append(means)
    if len(piv_frames) >= 3:
        mat = pd.DataFrame(piv_frames).dropna(axis=1)
        if mat.shape[1] >= 3:
            fn = friedman_nemenyi(mat)
            plot_critical_difference(
                fn["mean_ranks"],
                fn["critical_difference_0.05"],
                out_root / "fig7_cross_domain_cd.png",
                title="Cross-Domain Friedman-Nemenyi CD Diagram (5 Real Benchmarks)",
            )
            print("5-Benchmark Cross-Domain CD ranks:", fn["mean_ranks"], "Friedman p:", fn["p_value"])


if __name__ == "__main__":
    main()

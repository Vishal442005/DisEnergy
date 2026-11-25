"""Unified cross-domain benchmark runner across unsupervised, scarcity, and noise-stress regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.export_latex import format_benchmark_table
from analysis.make_figures import plot_critical_difference, plot_curves
from analysis.metrics import compute_segmentation_metrics, worst_decile_score
from analysis.stats_tests import friedman_nemenyi, paired_wilcoxon_holm, shapiro_levene_audit
from baselines.classical import (
    chan_vese_baseline,
    felzenszwalb_baseline,
    gabor_energy_baseline,
    glcm_entropy_baseline,
    slic_graphcut_baseline,
    spectral_residual_baseline,
    wavelet_energy_baseline,
)
from disenergy.core import DisEnergyConfig, DisEnergyOperator
from disenergy.topology import extract_box_free_mask
from disenergy.utils import inject_noise, set_seed

CLASSICAL_METHODS = {
    "Chan_Vese": chan_vese_baseline,
    "SLIC_GraphCut": slic_graphcut_baseline,
    "Gabor_Energy": gabor_energy_baseline,
    "GLCM_Entropy": glcm_entropy_baseline,
    "Wavelet_Energy": wavelet_energy_baseline,
    "Felzenszwalb": felzenszwalb_baseline,
    "Spectral_Residual": spectral_residual_baseline,
}


def run_unsupervised_and_noise_suite(
    split_json: Path,
    out_dir: Path,
    max_test_samples: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Regime 1 (zero-training) and Regime 3 (zero-shot noise stress)."""
    set_seed(42)
    manifest = json.loads(split_json.read_text())
    id_to_rec = {r["sample_id"]: r for r in manifest["records"]}
    test_ids = manifest["folds"][0]["test_ids"][:max_test_samples]
    samples = [(sid, np.load(id_to_rec[sid]["image_path"]), np.load(id_to_rec[sid]["mask_path"]).astype(bool)) for sid in test_ids]

    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(2, 8, 32), radius=2))

    # Regime 1: Zero-training clean comparison
    reg1_rows = []
    for sid, img, gt in samples:
        out = op(img)
        pred_de = extract_box_free_mask(
            out.fused_manifold.squeeze().numpy(),
            fine_manifold=out.scale_manifolds[2].squeeze().numpy(),
            method="watershed_filled",
        )
        reg1_rows.append({"sample_id": sid, "method": "DisEnergy", **compute_segmentation_metrics(pred_de, gt)})
        for mname, fn in CLASSICAL_METHODS.items():
            pred_m = fn(img)
            reg1_rows.append({"sample_id": sid, "method": mname, **compute_segmentation_metrics(pred_m, gt)})

    df_reg1 = pd.DataFrame(reg1_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_reg1.to_csv(out_dir / "regime1_unsupervised.csv", index=False)

    # Statistical audit & Wilcoxon + Holm-Bonferroni
    piv = df_reg1.pivot(index="sample_id", columns="method", values="dsc")
    audit = shapiro_levene_audit({col: piv[col].to_numpy() for col in piv.columns})
    (out_dir / "normality_levene_audit.json").write_text(json.dumps(audit, indent=2))

    wilcoxon_df = paired_wilcoxon_holm(
        piv["DisEnergy"].to_numpy(),
        {col: piv[col].to_numpy() for col in piv.columns if col != "DisEnergy"},
    )
    wilcoxon_df.to_csv(out_dir / "wilcoxon_holm.csv", index=False)

    # Worst-decile table
    worst10 = (
        df_reg1.groupby("method")["dsc"]
        .agg(mean_dsc="mean", p10_dsc=lambda s: worst_decile_score(s.to_numpy(), lower_is_better=False))
        .reset_index()
    )
    worst10.to_csv(out_dir / "worst_decile_dsc.csv", index=False)

    # Friedman + CD diagram across samples (treating samples/blocks as repeated observations)
    fn_res = friedman_nemenyi(piv)
    plot_critical_difference(
        fn_res["mean_ranks"],
        fn_res["critical_difference_0.05"],
        out_dir / "fig_cd_diagram.png",
    )

    tex = format_benchmark_table(
        df_reg1,
        metric_cols=["dsc", "iou", "hd95", "boundary_f1", "specificity"],
        caption="Zero-Training Unsupervised Comparison.",
        label="tab:regime1",
    )
    (out_dir / "regime1_table.tex").write_text(tex)

    # Regime 3: Zero-shot Noise Stress across SNR = [25, 20, 15, 10] dB
    rng = np.random.default_rng(42)
    snr_levels = [25.0, 20.0, 15.0, 10.0]
    noise_rows = []
    stress_methods = {
        "Chan_Vese": chan_vese_baseline,
        "Gabor_Energy": gabor_energy_baseline,
        "Spectral_Residual": spectral_residual_baseline,
    }

    for ntype in ("gaussian", "rician", "speckle"):
        for snr in snr_levels:
            for sid, img, gt in samples[:10]:
                noisy = inject_noise(img, ntype, snr_db=snr, rng=rng)
                out = op(noisy)
                pred_de = extract_box_free_mask(
                    out.fused_manifold.squeeze().numpy(),
                    fine_manifold=out.scale_manifolds[2].squeeze().numpy(),
                    method="watershed_filled",
                )
                noise_rows.append(
                    {
                        "sample_id": sid,
                        "noise_type": ntype,
                        "snr_db": snr,
                        "method": "DisEnergy",
                        **compute_segmentation_metrics(pred_de, gt),
                    }
                )
                for mname, fn in stress_methods.items():
                    noise_rows.append(
                        {
                            "sample_id": sid,
                            "noise_type": ntype,
                            "snr_db": snr,
                            "method": mname,
                            **compute_segmentation_metrics(fn(noisy), gt),
                        }
                    )

    df_noise = pd.DataFrame(noise_rows)
    df_noise.to_csv(out_dir / "regime3_noise_stress.csv", index=False)

    for ntype in ("gaussian", "rician", "speckle"):
        sub = df_noise[df_noise["noise_type"] == ntype]
        series = {}
        for meth in ["DisEnergy", *stress_methods.keys()]:
            msub = sub[sub["method"] == meth].groupby("snr_db")["dsc"]
            means = [float(msub.mean().loc[s]) for s in snr_levels]
            stds = [float(msub.std().loc[s]) for s in snr_levels]
            series[meth] = (means, stds)
        plot_curves(
            snr_levels,
            series,
            xlabel="SNR (dB)",
            ylabel="Dice Similarity Coefficient (DSC)",
            title=f"Zero-Shot Robustness Under {ntype.capitalize()} Noise",
            out_path=out_dir / f"fig_noise_{ntype}.png",
        )

    return df_reg1, df_noise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified DisEnergy benchmarks.")
    parser.add_argument("--split-json", type=Path, default=Path("data/splits/synthetic.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--max-test-samples", type=int, default=30)
    args = parser.parse_args()

    df1, df3 = run_unsupervised_and_noise_suite(args.split_json, args.out_dir, max_test_samples=args.max_test_samples)
    print("=== Regime 1 (Unsupervised) ===")
    print(df1.groupby("method")[["dsc", "iou", "hd95", "boundary_f1", "specificity"]].mean().round(4))
    print("=== Regime 3 (Noise Stress Mean DSC) ===")
    print(df3.groupby(["noise_type", "snr_db", "method"])["dsc"].mean().unstack("method").round(4))


if __name__ == "__main__":
    main()

"""Full local CPU research suite: MVTec AD, PH2, full EuroSAT/DRIVE/ECSSD, affine stress, and Figs 3/4/8."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analysis.export_latex import format_benchmark_table
from analysis.make_figures import (
    plot_critical_difference,
    plot_multiscale_diagnostics,
    plot_persistence_and_watershed,
    plot_qualitative_and_failures,
)
from analysis.metrics import compute_classification_metrics, compute_segmentation_metrics
from analysis.stats_tests import friedman_nemenyi
from baselines.classical import (
    chan_vese_baseline,
    gabor_energy_baseline,
    glcm_entropy_baseline,
    spectral_residual_baseline,
    wavelet_energy_baseline,
)
from disenergy.classifiers import train_topological_mlp
from disenergy.core import DisEnergyConfig, DisEnergyOperator
from disenergy.topology import extract_box_free_mask, extract_cubical_persistence
from disenergy.utils import set_seed


def run_mvtec_suite(out_dir: Path) -> pd.DataFrame:
    """Evaluate zero-shot DisEnergy vs classical baselines on MVTec AD surface subsets."""
    split_p = Path("data/splits/mvtec.json")
    if not split_p.exists():
        return pd.DataFrame()
    recs = json.loads(split_p.read_text())["records"]
    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(2, 8, 32), beta=2.0, radius=1))

    rows = []
    img_scores: list[tuple[int, float]] = []
    for r in recs:
        img = np.load(r["image_path"])
        gt = np.load(r["mask_path"]).astype(bool)
        lbl = int(r["label"])
        out = op(img)
        fused = out.fused_manifold.squeeze().numpy()
        fine = out.scale_manifolds[2].squeeze().numpy()
        img_scores.append((lbl, float(np.percentile(fused, 99.5))))

        if lbl == 1 and np.any(gt):
            pix_auc = float(roc_auc_score(gt.ravel(), fused.ravel()))
            pred_de = extract_box_free_mask(fused, fine_manifold=fine, method="watershed_filled")
            rows.append(
                {
                    "sample_id": r["sample_id"],
                    "method": "DisEnergy",
                    "pixel_auroc": pix_auc,
                    **compute_segmentation_metrics(pred_de, gt),
                }
            )
            for bname, fn in [
                ("GLCM_Entropy", glcm_entropy_baseline),
                ("Wavelet_Energy", wavelet_energy_baseline),
                ("Gabor_Energy", gabor_energy_baseline),
                ("Spectral_Residual", spectral_residual_baseline),
            ]:
                rows.append(
                    {
                        "sample_id": r["sample_id"],
                        "method": bname,
                        "pixel_auroc": np.nan,
                        **compute_segmentation_metrics(fn(img), gt),
                    }
                )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "mvtec_benchmark.csv", index=False)

    y_img = np.array([x[0] for x in img_scores])
    s_img = np.array([x[1] for x in img_scores])
    img_auc = float(roc_auc_score(y_img, s_img)) if len(np.unique(y_img)) > 1 else 0.5
    (out_dir / "mvtec_image_auroc.json").write_text(
        json.dumps({"image_level_anomaly_auroc": img_auc, "n_images": len(y_img)}, indent=2)
    )
    tex = format_benchmark_table(
        df,
        metric_cols=["dsc", "iou", "hd95", "boundary_f1", "specificity"],
        caption="Zero-Shot Industrial Surface Defect Localization on MVTec AD (Tile, Grid, Carpet, Leather).",
        label="tab:mvtec",
    )
    (out_dir / "mvtec_table.tex").write_text(tex)
    return df


def run_tda_classification_suite(
    split_json: Path,
    out_dir: Path,
    name: str,
    max_samples: int = 1000,
) -> pd.DataFrame:
    """Run 5-fold CV comparing DisEnergy TDA (RF, SVM, MLP) vs HOG+SVM."""
    recs = json.loads(split_json.read_text())["records"][:max_samples]
    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(2, 8, 32), beta=2.0, radius=1))
    x_tda, x_hog, ys = [], [], []
    for r in recs:
        img = np.load(r["image_path"])
        fused = op(img).fused_manifold.squeeze().numpy()
        x_tda.append(extract_cubical_persistence(fused).feature_vector)
        x_hog.append(hog(img.mean(axis=-1), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2)))
        ys.append(int(r["label"]))

    x_tda_arr = np.vstack(x_tda)
    x_hog_arr = np.vstack(x_hog)
    y_arr = np.asarray(ys, dtype=np.int64)
    n_classes = len(np.unique(y_arr))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for mname, feat, clf in [
        (
            "DisEnergy_TDA_RF",
            x_tda_arr,
            Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))]),
        ),
        (
            "DisEnergy_TDA_SVM",
            x_tda_arr,
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("pca", PCA(n_components=min(32, x_tda_arr.shape[1]), random_state=42)),
                    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
                ]
            ),
        ),
        (
            "HOG_SVM",
            x_hog_arr,
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("pca", PCA(n_components=min(32, x_hog_arr.shape[1]), random_state=42)),
                    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
                ]
            ),
        ),
    ]:
        probs = cross_val_predict(clf, feat, y_arr, cv=cv, method="predict_proba")
        met = compute_classification_metrics(y_arr, probs)
        rows.append({"dataset": name, "method": mname, **{k: round(v, 4) for k, v in met.items()}})

    # Also evaluate 5-fold TopologicalMLP
    import torch

    oof_probs = np.zeros((len(y_arr), n_classes), dtype=np.float32)
    for tr_idx, te_idx in cv.split(x_tda_arr, y_arr):
        mlp, scaler = train_topological_mlp(x_tda_arr[tr_idx], y_arr[tr_idx], num_classes=n_classes, epochs=60, seed=42)
        with torch.no_grad():
            logits = mlp(torch.from_numpy(scaler.transform(x_tda_arr[te_idx]).astype(np.float32)))
            oof_probs[te_idx] = torch.softmax(logits, dim=-1).numpy()
    met_mlp = compute_classification_metrics(y_arr, oof_probs)
    rows.append({"dataset": name, "method": "DisEnergy_TDA_MLP", **{k: round(v, 4) for k, v in met_mlp.items()}})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}_classification.csv", index=False)
    return df


def run_ecssd_full_and_affine_stress(out_dir: Path) -> None:
    """Run full-dataset 5-fold ECSSD evaluation + Regime 3b affine contrast/brightness stress."""
    m = json.loads(Path("data/splits/ecssd.json").read_text())
    recs = m["records"]
    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(2, 8, 32), beta=2.0, radius=1))

    # Full 1,000-image DisEnergy evaluation + Figs 3 & 8 data collection
    rows = []
    sample_records = []
    for r in recs:
        img = np.load(r["image_path"])
        gt = np.load(r["mask_path"]).astype(bool)
        out = op(img)
        fused = out.fused_manifold.squeeze().numpy()
        fine = out.scale_manifolds[2].squeeze().numpy()
        pred = extract_box_free_mask(fused, fine_manifold=fine, method="watershed_filled")
        met = compute_segmentation_metrics(pred, gt)
        rows.append({"sample_id": r["sample_id"], **met})
        if len(sample_records) < 100:
            sample_records.append((met["dsc"], r["sample_id"], img, gt, fused, pred))

    df_full = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(out_dir / "ecssd_full1000_disenergy.csv", index=False)

    # Fig 3: Persistence diagram + Betti curves + watershed overlay on best sample
    sample_records.sort(key=lambda t: t[0])
    worst_case = sample_records[0]
    median_case = sample_records[len(sample_records) // 2]
    best_case = sample_records[-1]

    tda_best = extract_cubical_persistence(best_case[4], betti_bins=32)
    plot_persistence_and_watershed(
        best_case[2],
        best_case[4],
        best_case[5],
        best_case[3],
        tda_best.diagrams,
        tda_best.betti_curves,
        32,
        out_dir / "fig3_persistence_watershed.png",
    )

    # Fig 4: Multi-scale diagnostic bars from ecssd_pilot
    pilot_csv = Path("outputs/ecssd_pilot/pilot_segmentation.csv")
    if pilot_csv.exists():
        plot_multiscale_diagnostics(pilot_csv, out_dir / "fig4_multiscale_diagnostics.png")

    # Fig 8: Best, Median, and Worst-decile failure mode gallery
    plot_qualitative_and_failures(
        [
            ("Best", best_case[2], best_case[3], best_case[4], best_case[5], best_case[0]),
            ("Median", median_case[2], median_case[3], median_case[4], median_case[5], median_case[0]),
            ("Failure (Worst)", worst_case[2], worst_case[3], worst_case[4], worst_case[5], worst_case[0]),
        ],
        out_dir / "fig8_qualitative_and_failures.png",
    )

    # Regime 3b: Affine photometric shift/contrast stress (a*I + c) on 30 ECSSD images
    affine_rows = []
    transforms = [("clean", 1.0, 0.0), ("dim_shift", 0.5, -0.15), ("bright_scale", 1.5, 0.2)]
    for sid_idx, r in enumerate(recs[:30]):
        img = np.load(r["image_path"])
        gt = np.load(r["mask_path"]).astype(bool)
        for tname, a, c in transforms:
            pert = np.clip(a * img + c, 0.0, 1.0).astype(np.float32)
            out = op(pert)
            pred_de = extract_box_free_mask(
                out.fused_manifold.squeeze().numpy(),
                fine_manifold=out.scale_manifolds[2].squeeze().numpy(),
                method="watershed_filled",
            )
            affine_rows.append({"sample_id": r["sample_id"], "shift": tname, "method": "DisEnergy", **compute_segmentation_metrics(pred_de, gt)})
            for bname, fn in [("Chan_Vese", chan_vese_baseline), ("Gabor_Energy", gabor_energy_baseline), ("Spectral_Residual", spectral_residual_baseline)]:
                affine_rows.append({"sample_id": r["sample_id"], "shift": tname, "method": bname, **compute_segmentation_metrics(fn(pert), gt)})
    pd.DataFrame(affine_rows).to_csv(out_dir / "ecssd_affine_stress.csv", index=False)


def run_drive_full40(out_dir: Path) -> None:
    """Run all 40 DRIVE retinal scans including green-channel vs RGB ablation."""
    recs = json.loads(Path("data/splits/drive.json").read_text())["records"]
    op = DisEnergyOperator(DisEnergyConfig(block_sizes=(1, 4, 16), beta=2.0, radius=1))
    rows = []
    for r in recs:
        img = np.load(r["image_path"])
        gt = np.load(r["mask_path"]).astype(bool)
        fov = img.mean(axis=-1) > 0.06
        out_rgb = op(img)
        p_rgb = extract_box_free_mask(out_rgb.fused_manifold.squeeze().numpy() * fov, method="otsu_filled", fill_enclosed_basins=False) & fov
        rows.append({"sample_id": r["sample_id"], "method": "DisEnergy_RGB", **compute_segmentation_metrics(p_rgb, gt)})

        # Green channel carries strongest hemoglobin vessel contrast
        out_g = op(img[:, :, 1:2])
        p_g = extract_box_free_mask(out_g.fused_manifold.squeeze().numpy() * fov, method="otsu_filled", fill_enclosed_basins=False) & fov
        rows.append({"sample_id": r["sample_id"], "method": "DisEnergy_Green", **compute_segmentation_metrics(p_g, gt)})

        for bname, fn in [("Chan_Vese", chan_vese_baseline), ("Gabor_Energy", gabor_energy_baseline), ("Wavelet_Energy", wavelet_energy_baseline), ("Spectral_Residual", spectral_residual_baseline)]:
            rows.append({"sample_id": r["sample_id"], "method": bname, **compute_segmentation_metrics(fn(img) & fov, gt)})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "drive_full40.csv", index=False)
    tex = format_benchmark_table(
        df,
        metric_cols=["dsc", "iou", "hd95", "boundary_f1", "specificity"],
        caption="Full 40-Scan Retinal Vessel Extraction on DRIVE (RGB vs. Green-Channel DisEnergy).",
        label="tab:drive_full",
    )
    (out_dir / "drive_full40_table.tex").write_text(tex)


def main() -> None:
    set_seed(42)
    out_root = Path("outputs/full_local")
    out_root.mkdir(parents=True, exist_ok=True)

    df_mv = run_mvtec_suite(out_root / "mvtec")
    if not df_mv.empty:
        print("=== MVTec AD Defect Localization ===")
        print(df_mv.groupby("method")[["dsc", "iou", "hd95", "boundary_f1", "specificity"]].mean().round(4))
        print(Path(out_root / "mvtec/mvtec_image_auroc.json").read_text())

    if Path("data/splits/ph2_ood.json").exists():
        df_ph2 = run_tda_classification_suite(Path("data/splits/ph2_ood.json"), out_root / "ph2", "ph2_ood", max_samples=200)
        print("=== PH2 Dermoscopy 3-Class Classification (n=200, 5-fold CV) ===")
        print(df_ph2.to_string(index=False))

    df_es = run_tda_classification_suite(Path("data/splits/eurosat.json"), out_root / "eurosat", "eurosat_1000", max_samples=1000)
    print("=== EuroSAT Full 1,000-Patch 10-Class Classification (5-fold CV) ===")
    print(df_es.to_string(index=False))

    run_drive_full40(out_root / "drive")
    df_d = pd.read_csv(out_root / "drive/drive_full40.csv")
    print("=== DRIVE Full 40 Scans ===")
    print(df_d.groupby("method")[["dsc", "iou", "hd95", "boundary_f1", "specificity"]].mean().round(4))

    run_ecssd_full_and_affine_stress(out_root / "ecssd")
    df_ec = pd.read_csv(out_root / "ecssd/ecssd_full1000_disenergy.csv")
    print("=== ECSSD Full 1,000 Images (DisEnergy beta=2.0, r=1) ===")
    print(df_ec[["dsc", "iou", "hd95", "boundary_f1", "specificity"]].mean().round(4))
    df_af = pd.read_csv(out_root / "ecssd/ecssd_affine_stress.csv")
    print("=== ECSSD Regime 3b Affine Stress (Mean DSC) ===")
    print(df_af.groupby(["shift", "method"])["dsc"].mean().unstack("method").round(4))

    # Cross-dataset Friedman-Nemenyi CD across ECSSD (30), DRIVE (40), MVTec (defects)
    piv_frames = []
    for tag, csv_p in [
        ("ECSSD_Affine", out_root / "ecssd/ecssd_affine_stress.csv"),
        ("DRIVE_40", out_root / "drive/drive_full40.csv"),
        ("MVTec_AD", out_root / "mvtec/mvtec_benchmark.csv"),
    ]:
        if csv_p.exists():
            sub = pd.read_csv(csv_p)
            sub["method"] = sub["method"].replace({"DisEnergy_Green": "DisEnergy"})
            means = sub.groupby("method")["dsc"].mean()
            means.name = tag
            piv_frames.append(means)
    if len(piv_frames) >= 2:
        mat = pd.DataFrame(piv_frames).dropna(axis=1)
        if mat.shape[1] >= 3:
            fn = friedman_nemenyi(mat)
            plot_critical_difference(fn["mean_ranks"], fn["critical_difference_0.05"], out_root / "fig7_cross_domain_cd.png")
            print("Cross-domain CD ranks:", fn["mean_ranks"])


if __name__ == "__main__":
    main()

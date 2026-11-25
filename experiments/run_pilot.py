"""Pilot evaluation, multi-scale diagnostics, and 3D manifold visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.export_latex import format_benchmark_table
from analysis.make_figures import generate_graphical_abstract, plot_3d_manifolds
from analysis.metrics import compute_classification_metrics, compute_segmentation_metrics
from disenergy.classifiers import build_tabular_classifier
from disenergy.core import DisEnergyConfig, DisEnergyOperator
from disenergy.topology import extract_box_free_mask, extract_cubical_persistence
from disenergy.utils import set_seed


def run_pilot(split_json: Path, out_dir: Path, max_samples: int = 60) -> pd.DataFrame:
    """Run pilot diagnostic evaluation across Fine, Mid, Large, and Fused manifolds."""
    set_seed(42)
    manifest = json.loads(split_json.read_text())
    records = manifest["records"][:max_samples]

    cfg = DisEnergyConfig(block_sizes=(2, 8, 32), radius=2)
    op = DisEnergyOperator(cfg)

    rows = []
    tda_features = []
    labels = []
    first_visual = None

    for idx, rec in enumerate(records):
        img = np.load(rec["image_path"])
        gt = np.load(rec["mask_path"]).astype(bool)
        out = op(img)

        fine = out.scale_manifolds[2].squeeze().numpy()
        mid = out.scale_manifolds[8].squeeze().numpy()
        large = out.scale_manifolds[32].squeeze().numpy()
        fused = out.fused_manifold.squeeze().numpy()

        for scale_name, man in [("Fine_2", fine), ("Mid_8", mid), ("Large_32", large), ("Fused_Otsu", fused)]:
            mask = extract_box_free_mask(man, method="otsu_filled")
            m = compute_segmentation_metrics(mask, gt)
            rows.append({"sample_id": rec["sample_id"], "method": scale_name, **m})

        mask_ws = extract_box_free_mask(fused, fine_manifold=fine, method="watershed_filled")
        m_ws = compute_segmentation_metrics(mask_ws, gt)
        rows.append({"sample_id": rec["sample_id"], "method": "DisEnergy_Watershed", **m_ws})

        tda = extract_cubical_persistence(fused)
        tda_features.append(tda.feature_vector)
        labels.append(int(rec["label"]))

        if idx == 0:
            first_visual = (img, {"Fine (w=2)": fine, "Mid (w=8)": mid, "Large (w=32)": large, "Fused": fused}, mask_ws)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pilot_segmentation.csv", index=False)

    # Fold-0 topological classification check
    fold0 = manifest["folds"][0]
    id_to_idx = {r["sample_id"]: i for i, r in enumerate(records)}
    tr_idx = [id_to_idx[i] for i in fold0["train_ids"] + fold0["val_ids"] if i in id_to_idx]
    te_idx = [id_to_idx[i] for i in fold0["test_ids"] if i in id_to_idx]
    x_arr = np.vstack(tda_features)
    y_arr = np.asarray(labels, dtype=np.int64)

    if len(np.unique(y_arr[tr_idx])) > 1 and len(te_idx) > 0:
        clf = build_tabular_classifier("rbf_svm", seed=42)
        clf.fit(x_arr[tr_idx], y_arr[tr_idx])
        probs = clf.predict_proba(x_arr[te_idx])
        cls_metrics = compute_classification_metrics(y_arr[te_idx], probs)
        (out_dir / "pilot_classification.json").write_text(json.dumps(cls_metrics, indent=2))
    else:
        (out_dir / "pilot_classification.json").write_text(
            json.dumps({"note": "single-class segmentation dataset; classification skipped"}, indent=2)
        )

    # Export pilot LaTeX table & 300-dpi figures
    tex = format_benchmark_table(
        df,
        metric_cols=["dsc", "iou", "hd95", "boundary_f1", "specificity"],
        caption="Pilot Multi-Scale Diagnostic Comparison.",
        label="tab:pilot_scales",
    )
    (out_dir / "pilot_table.tex").write_text(tex)

    if first_visual is not None:
        img0, mans0, mask0 = first_visual
        plot_3d_manifolds(img0, mans0, out_dir / "fig_pilot_3d_manifolds.png")
        generate_graphical_abstract(img0, mans0["Fused"], mask0, out_dir / "fig_graphical_abstract.png")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DisEnergy pilot diagnostics.")
    parser.add_argument("--split-json", type=Path, default=Path("data/splits/synthetic.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/pilot"))
    parser.add_argument("--max-samples", type=int, default=60)
    args = parser.parse_args()

    df = run_pilot(args.split_json, args.out_dir, max_samples=args.max_samples)
    print(df.groupby("method")[["dsc", "iou", "hd95", "boundary_f1", "specificity"]].mean().round(4))


if __name__ == "__main__":
    main()

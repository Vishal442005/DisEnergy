"""Verify strict zero patient/subject leakage across train/val/test folds."""

from __future__ import annotations

from pathlib import Path

from data.prepare_datasets import build_leak_free_nested_splits, create_synthetic_benchmark_dataset


def test_zero_subject_leakage_across_all_folds(tmp_path: Path) -> None:
    records = create_synthetic_benchmark_dataset(tmp_path / "syn", n_samples=50, image_size=64, seed=42)
    manifest = build_leak_free_nested_splits(records, n_folds=5, seed=42)
    id_to_subj = {r["sample_id"]: r["subject_id"] for r in manifest["records"]}

    for fold in manifest["folds"]:
        tr_subj = {id_to_subj[i] for i in fold["train_ids"]}
        va_subj = {id_to_subj[i] for i in fold["val_ids"]}
        te_subj = {id_to_subj[i] for i in fold["test_ids"]}

        assert tr_subj.isdisjoint(va_subj)
        assert tr_subj.isdisjoint(te_subj)
        assert va_subj.isdisjoint(te_subj)
        assert len(fold["train_ids"]) + len(fold["val_ids"]) + len(fold["test_ids"]) == len(records)

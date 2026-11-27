"""Dataset ingestion, 256x256 / 3D-ROI shard builder, and leak-free patient-level splits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass
class SampleRecord:
    """Metadata entry for a single preprocessed image/volume sample."""

    sample_id: str
    subject_id: str
    dataset: str
    image_path: str
    mask_path: str | None
    label: int
    domain: str


SUPPORTED_DATASETS = (
    "synthetic",
    "brats",
    "isic2018",
    "busi",
    "drive",
    "mvtec",
    "whu",
    "eurosat",
    "ecssd",
    "ph2_ood",
    "stare_ood",
)


def preprocess_image_mask_pair(
    image: np.ndarray,
    mask: np.ndarray | None,
    target_size: int = 256,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resize image (bilinear) and mask (nearest) into float32 [0,1] and binary uint8."""
    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 2:
        img = img[..., None]
    c = img.shape[-1]
    img_r = resize(img, (target_size, target_size, c), order=1, preserve_range=True, anti_aliasing=True).astype(
        np.float32
    )
    img_r = np.clip(img_r, 0.0, 1.0)
    if mask is None:
        return img_r, None
    msk_r = resize(mask.astype(np.float32), (target_size, target_size), order=0, preserve_range=True)
    return img_r, (msk_r > 0.5).astype(np.uint8)


def _binarize_mask(mask_arr: np.ndarray) -> np.ndarray:
    """Binarize uint8/float masks whether encoded in {0, 1} or {0, 255}."""
    arr = np.asarray(mask_arr)
    thresh = 127.0 if float(arr.max()) > 1.5 else 0.5
    return arr > thresh


def _save_pair(
    out_dir: Path,
    sample_id: str,
    img: np.ndarray,
    mask: np.ndarray | None,
    image_size: int,
) -> tuple[str, str | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_r, msk_r = preprocess_image_mask_pair(img, mask, target_size=image_size)
    img_p = out_dir / f"{sample_id}_img.npy"
    np.save(img_p, img_r.astype(np.float32))
    msk_p_str: str | None = None
    if msk_r is not None:
        msk_p = out_dir / f"{sample_id}_msk.npy"
        np.save(msk_p, msk_r.astype(np.uint8))
        msk_p_str = str(msk_p)
    return str(img_p), msk_p_str


def create_synthetic_benchmark_dataset(
    out_dir: Path,
    n_samples: int = 60,
    image_size: int = 128,
    seed: int = 42,
) -> list[SampleRecord]:
    """Generate a reproducible synthetic multi-subject dataset for CI and pipeline verification."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[SampleRecord] = []

    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    n_subjects = max(10, n_samples // 3)

    for idx in range(n_samples):
        subj_id = f"subj_{idx % n_subjects:03d}"
        cls_label = int(idx % 2)
        cy = float(rng.uniform(0.32 * image_size, 0.68 * image_size))
        cx = float(rng.uniform(0.32 * image_size, 0.68 * image_size))
        rad = float(rng.uniform(0.14 * image_size, 0.24 * image_size))
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        bg = 0.18 + 0.04 * np.sin(xx / 18.0) * np.cos(yy / 18.0)
        img = np.stack([bg, bg * 0.95, bg * 1.05], axis=-1).astype(np.float32)

        if cls_label == 0:
            mask = dist <= rad
            img[mask] = np.array([0.78, 0.55, 0.42], dtype=np.float32)
        else:
            theta = np.arctan2(yy - cy, xx - cx)
            r_mod = rad * (1.0 + 0.18 * np.sin(3.0 * theta))
            mask = dist <= r_mod
            cavity = dist <= 0.38 * rad
            tex = rng.normal(0.55, 0.14, size=(image_size, image_size)).clip(0.1, 0.95).astype(np.float32)
            for c in range(3):
                img[:, :, c] = np.where(mask & ~cavity, tex * (0.9 + 0.1 * c), img[:, :, c])
            mask = mask & ~cavity

        img = np.clip(img + rng.normal(0.0, 0.02, size=img.shape).astype(np.float32), 0.0, 1.0)
        sample_id = f"syn_{idx:04d}"
        img_p = out_dir / f"{sample_id}_img.npy"
        msk_p = out_dir / f"{sample_id}_msk.npy"
        np.save(img_p, img.astype(np.float32))
        np.save(msk_p, mask.astype(np.uint8))

        records.append(
            SampleRecord(
                sample_id=sample_id,
                subject_id=subj_id,
                dataset="synthetic",
                image_path=str(img_p),
                mask_path=str(msk_p),
                label=cls_label,
                domain="synthetic",
            )
        )
    return records


def prepare_ecssd(raw_dir: Path, out_dir: Path, image_size: int = 256, max_samples: int | None = None) -> list[SampleRecord]:
    """Extract and preprocess CUHK ECSSD images + ground_truth_mask archives."""
    for archive_name in ("images.zip", "ground_truth_mask.zip"):
        arc = raw_dir / archive_name
        if arc.exists():
            with zipfile.ZipFile(arc) as zf:
                zf.extractall(raw_dir)

    img_files = sorted(
        list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.jpeg")),
        key=lambda p: p.stem,
    )
    msk_map = {p.stem: p for p in raw_dir.rglob("*.png")}
    records: list[SampleRecord] = []

    for img_path in img_files:
        if img_path.stem not in msk_map:
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        msk = np.asarray(Image.open(msk_map[img_path.stem]).convert("L"), dtype=np.float32) > 127.5
        sid = f"ecssd_{img_path.stem}"
        img_p, msk_p = _save_pair(out_dir, sid, img, msk, image_size=image_size)
        records.append(
            SampleRecord(
                sample_id=sid,
                subject_id=sid,
                dataset="ecssd",
                image_path=img_p,
                mask_path=msk_p,
                label=1,
                domain="natural_scene",
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    if not records:
        raise FileNotFoundError(f"No paired ECSSD jpg/png files found under {raw_dir}")
    return records


def prepare_busi(raw_dir: Path, out_dir: Path, image_size: int = 256, max_samples: int | None = None) -> list[SampleRecord]:
    """Preprocess BUSI (folders, busi-whu-seg, or HuggingFace parquet files)."""
    records: list[SampleRecord] = []
    # Layout A: Parquet files under raw_dir or sibling busi_parquet/
    pq_dir = raw_dir if list(raw_dir.glob("*.parquet")) else (raw_dir.parent / "busi_parquet")
    pq_files = sorted(pq_dir.glob("*.parquet")) if pq_dir.exists() else []
    if pq_files:
        import pandas as pd

        idx = 0
        for pf in pq_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                img_b = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
                msk_b = row["mask"]["bytes"] if isinstance(row["mask"], dict) else row["mask"]
                img = np.asarray(Image.open(io.BytesIO(img_b)).convert("RGB"), dtype=np.float32) / 255.0
                msk = _binarize_mask(np.asarray(Image.open(io.BytesIO(msk_b)).convert("L")))
                sid = f"busi_{pf.stem}_{idx:04d}"
                img_p, msk_p = _save_pair(out_dir, sid, img, msk, image_size=image_size)
                records.append(
                    SampleRecord(
                        sample_id=sid,
                        subject_id=sid,
                        dataset="busi",
                        image_path=img_p,
                        mask_path=msk_p,
                        label=1 if np.any(msk) else 0,
                        domain="biomedical_ultrasound",
                    )
                )
                idx += 1
                if max_samples is not None and len(records) >= max_samples:
                    return records
        return records

    # Layout B: standard Mendeley/Kaggle folders benign/, malignant/, normal/
    class_dirs = {"normal": 0, "benign": 1, "malignant": 2}
    found_standard = any((raw_dir / k).exists() for k in class_dirs)

    if found_standard:
        for cname, label in class_dirs.items():
            cdir = raw_dir / cname
            if not cdir.exists():
                continue
            imgs = sorted(p for p in cdir.glob("*.png") if "_mask" not in p.stem)
            for img_path in imgs:
                mask_files = sorted(cdir.glob(f"{img_path.stem}_mask*.png"))
                if not mask_files:
                    continue
                img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
                combined = np.zeros(img.shape[:2], dtype=bool)
                for mf in mask_files:
                    combined |= _binarize_mask(np.asarray(Image.open(mf).convert("L")))
                sid = f"busi_{cname}_{img_path.stem.replace(' ', '_')}"
                img_p, msk_p = _save_pair(out_dir, sid, img, combined, image_size=image_size)
                records.append(
                    SampleRecord(
                        sample_id=sid,
                        subject_id=sid,
                        dataset="busi",
                        image_path=img_p,
                        mask_path=msk_p,
                        label=label,
                        domain="biomedical_ultrasound",
                    )
                )
                if max_samples is not None and len(records) >= max_samples:
                    return records
        return records

    # Layout C: busi-whu-seg (split/{images, masks, metadata.csv})
    for split in ("train", "validation", "test"):
        meta_path = raw_dir / split / "metadata.csv"
        if meta_path.exists():
            import csv

            with meta_path.open(encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    img_path = raw_dir / split / row["image_file_name"].strip()
                    msk_path = raw_dir / split / row["mask_file_name"].strip()
                    if not (img_path.exists() and msk_path.exists()):
                        continue
                    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
                    msk = _binarize_mask(np.asarray(Image.open(msk_path).convert("L")))
                    lbl_str = row.get("label", "").strip().lower()
                    lbl = 1 if "malignant" in lbl_str else 0
                    sid = f"busi_{split}_{img_path.stem}"
                    img_p, msk_p = _save_pair(out_dir, sid, img, msk, image_size=image_size)
                    records.append(
                        SampleRecord(
                            sample_id=sid,
                            subject_id=sid,
                            dataset="busi",
                            image_path=img_p,
                            mask_path=msk_p,
                            label=lbl,
                            domain="biomedical_ultrasound",
                        )
                    )
                    if max_samples is not None and len(records) >= max_samples:
                        return records
            continue

        img_dir = raw_dir / split / "images"
        msk_dir = raw_dir / split / "masks"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            candidates = [
                msk_dir / img_path.name,
                msk_dir / f"{img_path.stem}.png",
                msk_dir / f"{img_path.stem}_mask{img_path.suffix}",
                msk_dir / f"{img_path.stem}_mask.png",
            ]
            msk_path = next((c for c in candidates if c.exists()), None)
            if msk_path is None:
                continue
            img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
            msk = _binarize_mask(np.asarray(Image.open(msk_path).convert("L")))
            lbl = 1 if "malignant" in img_path.stem.lower() else 0
            sid = f"busi_{split}_{img_path.stem}"
            img_p, msk_p = _save_pair(out_dir, sid, img, msk, image_size=image_size)
            records.append(
                SampleRecord(
                    sample_id=sid,
                    subject_id=sid,
                    dataset="busi",
                    image_path=img_p,
                    mask_path=msk_p,
                    label=lbl,
                    domain="biomedical_ultrasound",
                )
            )
            if max_samples is not None and len(records) >= max_samples:
                return records
    if not records:
        raise FileNotFoundError(f"No BUSI image/mask pairs found under {raw_dir}")
    return records


def prepare_isic2018(raw_dir: Path, out_dir: Path, image_size: int = 256, max_samples: int | None = None) -> list[SampleRecord]:
    """Preprocess ISIC 2018 Task 1 dermoscopy lesion images & masks."""
    img_files = sorted(raw_dir.rglob("ISIC_*.jpg"))
    msk_map = {p.stem.replace("_segmentation", ""): p for p in raw_dir.rglob("ISIC_*_segmentation.png")}
    records: list[SampleRecord] = []
    for img_path in img_files:
        if img_path.stem not in msk_map:
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        msk = _binarize_mask(np.asarray(Image.open(msk_map[img_path.stem]).convert("L")))
        sid = img_path.stem
        img_p, msk_p = _save_pair(out_dir, sid, img, msk, image_size=image_size)
        records.append(
            SampleRecord(
                sample_id=sid,
                subject_id=sid,
                dataset="isic2018",
                image_path=img_p,
                mask_path=msk_p,
                label=1,
                domain="biomedical_dermoscopy",
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    if not records:
        raise FileNotFoundError(f"No ISIC 2018 pairs found under {raw_dir}")
    return records


def prepare_mvtec(
    raw_dir: Path,
    out_dir: Path,
    image_size: int = 256,
    categories: tuple[str, ...] = ("tile", "grid", "carpet", "leather"),
    max_samples: int | None = None,
) -> list[SampleRecord]:
    """Preprocess MVTec AD surface anomaly subsets (defects with GT masks + good test samples)."""
    records: list[SampleRecord] = []
    for cat in categories:
        cat_dir = raw_dir / cat
        if not cat_dir.exists():
            continue
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            is_good = defect_dir.name == "good"
            for img_path in sorted(defect_dir.glob("*.png")):
                img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
                if is_good:
                    msk = np.zeros(img.shape[:2], dtype=bool)
                else:
                    gt_path = gt_dir / defect_dir.name / f"{img_path.stem}_mask.png"
                    if not gt_path.exists():
                        continue
                    msk = _binarize_mask(np.asarray(Image.open(gt_path).convert("L")))
                sid = f"mvtec_{cat}_{defect_dir.name}_{img_path.stem}"
                img_p, msk_p = _save_pair(out_dir, sid, img, msk, image_size=image_size)
                records.append(
                    SampleRecord(
                        sample_id=sid,
                        subject_id=sid,
                        dataset="mvtec",
                        image_path=img_p,
                        mask_path=msk_p,
                        label=0 if is_good else 1,
                        domain="industrial",
                    )
                )
                if max_samples is not None and len(records) >= max_samples:
                    return records
    if not records:
        raise FileNotFoundError(f"No MVTec categories {categories} found under {raw_dir}")
    return records


def prepare_brats_2d(
    raw_dir: Path,
    out_dir: Path,
    image_size: int = 256,
    slices_per_subject: int = 3,
    max_samples: int | None = None,
) -> list[SampleRecord]:
    """Extract patient-grouped 4-modality axial slices from BraTS NIfTI subject folders."""
    import nibabel as nib

    records: list[SampleRecord] = []
    subj_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    for sdir in subj_dirs:
        seg_files = list(sdir.glob("*seg*.nii*"))
        if not seg_files:
            continue
        seg = np.asarray(nib.load(seg_files[0]).get_fdata(), dtype=np.float32)
        mods = []
        for tag in ("flair", "t1ce", "t2", "t1"):
            cand = list(sdir.glob(f"*{tag}*.nii*"))
            if not cand:
                break
            vol = np.asarray(nib.load(cand[0]).get_fdata(), dtype=np.float32)
            nz = vol[vol > 0]
            if len(nz) > 0:
                hi = float(np.percentile(nz, 99.5))
                vol = np.clip(vol / max(hi, 1e-6), 0.0, 1.0)
            mods.append(vol)
        if len(mods) != 4:
            continue
        vol4 = np.stack(mods, axis=-1)  # (H, W, D, 4)
        tumor_areas = (seg > 0).sum(axis=(0, 1))
        pos_z = np.where(tumor_areas > 150)[0]
        if len(pos_z) == 0:
            continue
        chosen = np.unique(np.linspace(pos_z[0], pos_z[-1], num=min(slices_per_subject, len(pos_z)), dtype=int))
        for z in chosen:
            sid = f"{sdir.name}_z{int(z):03d}"
            img_p, msk_p = _save_pair(out_dir, sid, vol4[:, :, z, :], seg[:, :, z] > 0, image_size=image_size)
            records.append(
                SampleRecord(
                    sample_id=sid,
                    subject_id=sdir.name,
                    dataset="brats",
                    image_path=img_p,
                    mask_path=msk_p,
                    label=1,
                    domain="biomedical_mri",
                )
            )
            if max_samples is not None and len(records) >= max_samples:
                return records
    if not records:
        raise FileNotFoundError(f"No valid BraTS NIfTI subject folders found under {raw_dir}")
    return records


def prepare_drive(raw_dir: Path, out_dir: Path, image_size: int = 256, max_samples: int | None = None) -> list[SampleRecord]:
    """Preprocess DRIVE retinal vessel images + manual1 ground truth masks."""
    img_files = sorted(list(raw_dir.rglob("*_training.tif")) + list(raw_dir.rglob("*_test.tif")))
    gt_map = {}
    for p in list(raw_dir.rglob("*_manual1.tif")) + list(raw_dir.rglob("*_manual1.gif")):
        stem_id = p.stem.split("_")[0]
        gt_map[stem_id] = p

    records: list[SampleRecord] = []
    seen: set[str] = set()
    for img_path in img_files:
        stem_id = img_path.stem.split("_")[0]
        if stem_id not in gt_map:
            continue
        sid = f"drive_{img_path.stem}"
        if sid in seen:
            continue
        seen.add(sid)
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        msk = _binarize_mask(np.asarray(Image.open(gt_map[stem_id]).convert("L")))
        img_p, msk_p = _save_pair(out_dir, sid, img, msk, image_size=image_size)
        records.append(
            SampleRecord(
                sample_id=sid,
                subject_id=sid,
                dataset="drive",
                image_path=img_p,
                mask_path=msk_p,
                label=1,
                domain="biomedical_retinal",
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    if not records:
        raise FileNotFoundError(f"No DRIVE image/manual1 pairs found under {raw_dir}")
    return records


def prepare_eurosat(raw_dir: Path, out_dir: Path, image_size: int = 128, max_samples: int | None = 1000) -> list[SampleRecord]:
    """Preprocess EuroSAT RGB parquet shards into 10-class scene classification records."""
    import pandas as pd

    pq_files = sorted(raw_dir.glob("*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"No EuroSAT parquet files found under {raw_dir}")

    records: list[SampleRecord] = []
    per_class_limit = (max_samples // 10) if max_samples is not None else None
    class_counts: dict[int, int] = {}

    for pf in pq_files:
        df = pd.read_parquet(pf)
        for idx, row in df.iterrows():
            lbl = int(row["label"])
            if per_class_limit is not None and class_counts.get(lbl, 0) >= per_class_limit:
                continue
            img_b = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            img = np.asarray(Image.open(io.BytesIO(img_b)).convert("RGB"), dtype=np.float32) / 255.0
            sid = f"eurosat_{pf.stem}_{idx:05d}"
            img_p, _ = _save_pair(out_dir, sid, img, None, image_size=image_size)
            records.append(
                SampleRecord(
                    sample_id=sid,
                    subject_id=sid,
                    dataset="eurosat",
                    image_path=img_p,
                    mask_path=None,
                    label=lbl,
                    domain="remote_sensing",
                )
            )
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
            if max_samples is not None and len(records) >= max_samples:
                return records
    return records


def prepare_ph2(raw_dir: Path, out_dir: Path, image_size: int = 256, max_samples: int | None = None) -> list[SampleRecord]:
    """Preprocess PH2 dermoscopy dataset (200 lesions: Common Nevus, Atypical Nevus, Melanoma)."""
    import csv

    csv_path = raw_dir / "PH2_simple_dataset.csv"
    img_dir = raw_dir / "images"
    if not (csv_path.exists() and img_dir.exists()):
        raise FileNotFoundError(f"Expected PH2_simple_dataset.csv and images/ under {raw_dir}")

    label_map = {"common nevus": 0, "atypical nevus": 1, "melanoma": 2}
    records: list[SampleRecord] = []
    with csv_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = row["image_name"].strip()
            diag = row["diagnosis"].strip().lower()
            img_path = img_dir / f"{name}.jpg"
            if not img_path.exists() or diag not in label_map:
                continue
            img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
            sid = f"ph2_{name}"
            img_p, _ = _save_pair(out_dir, sid, img, None, image_size=image_size)
            records.append(
                SampleRecord(
                    sample_id=sid,
                    subject_id=sid,
                    dataset="ph2_ood",
                    image_path=img_p,
                    mask_path=None,
                    label=label_map[diag],
                    domain="biomedical_dermoscopy",
                )
            )
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def prepare_whu(raw_dir: Path, out_dir: Path, image_size: int = 256, max_samples: int | None = None) -> list[SampleRecord]:
    """Preprocess WHU Building remote sensing images + building masks."""
    records: list[SampleRecord] = []
    for img_path in sorted(raw_dir.rglob("*.png")):
        if "Mask" in img_path.parts or "mask" in img_path.parts:
            continue
        msk_path = Path(str(img_path).replace("/Image/", "/Mask/"))
        if not msk_path.exists():
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        msk = _binarize_mask(np.asarray(Image.open(msk_path).convert("L")))
        sid = f"whu_{img_path.stem}"
        img_p, msk_p = _save_pair(out_dir, sid, img, msk, image_size=image_size)
        records.append(
            SampleRecord(
                sample_id=sid,
                subject_id=sid,
                dataset="whu",
                image_path=img_p,
                mask_path=msk_p,
                label=1 if np.any(msk) else 0,
                domain="remote_sensing",
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    if not records:
        raise FileNotFoundError(f"No WHU Image/Mask pairs found under {raw_dir}")
    return records


def build_leak_free_nested_splits(
    records: Sequence[SampleRecord],
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Generate patient/subject-level stratified 5-fold CV splits with inner val folds."""
    y = np.asarray([r.label for r in records], dtype=np.int64)
    groups = np.asarray([r.subject_id for r in records])
    unique_groups = np.unique(groups)
    use_group = len(unique_groups) < len(records) and len(unique_groups) >= n_folds
    n_classes = len(np.unique(y))

    if use_group and n_classes > 1:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = list(splitter.split(np.zeros(len(records)), y, groups))
    elif n_classes > 1:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = list(splitter.split(np.zeros(len(records)), y))
    else:
        # Single-class segmentation datasets (e.g., ECSSD / DRIVE): deterministic K-fold
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = list(kf.split(np.zeros(len(records))))

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(fold_iter):
        train_val_subjects = {records[i].subject_id for i in train_val_idx}
        test_subjects = {records[i].subject_id for i in test_idx}
        if use_group:
            assert train_val_subjects.isdisjoint(test_subjects), (
                f"Subject leakage detected in fold {fold_idx}!"
            )

        tv_groups = np.asarray([records[i].subject_id for i in train_val_idx])
        tv_y = y[train_val_idx]
        if use_group and len(np.unique(tv_y)) > 1:
            inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + fold_idx)
            inner_tr_rel, inner_val_rel = next(inner_splitter.split(np.zeros(len(train_val_idx)), tv_y, tv_groups))
        elif len(np.unique(tv_y)) > 1:
            inner_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold_idx)
            inner_tr_rel, inner_val_rel = next(inner_splitter.split(np.zeros(len(train_val_idx)), tv_y))
        else:
            from sklearn.model_selection import KFold

            kf_in = KFold(n_splits=5, shuffle=True, random_state=seed + fold_idx)
            inner_tr_rel, inner_val_rel = next(kf_in.split(np.zeros(len(train_val_idx))))

        train_idx = train_val_idx[inner_tr_rel]
        val_idx = train_val_idx[inner_val_rel]

        folds.append(
            {
                "fold": fold_idx,
                "train_ids": [records[i].sample_id for i in train_idx],
                "val_ids": [records[i].sample_id for i in val_idx],
                "test_ids": [records[i].sample_id for i in test_idx],
            }
        )

    manifest = {
        "n_folds": n_folds,
        "seed": seed,
        "n_samples": len(records),
        "n_subjects": len(unique_groups),
        "records": [asdict(r) for r in records],
        "folds": folds,
    }
    payload_hash = hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()[:16]
    manifest["splits_sha256"] = payload_hash
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare standardized DisEnergy dataset shards and splits.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="synthetic")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--n-samples", type=int, default=60)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds_raw = args.raw_dir / args.dataset
    ds_out = args.out_dir / args.dataset

    if args.dataset == "synthetic":
        records = create_synthetic_benchmark_dataset(
            ds_out, n_samples=args.n_samples, image_size=args.image_size, seed=args.seed
        )
    elif args.dataset == "ecssd":
        records = prepare_ecssd(ds_raw, ds_out, image_size=args.image_size, max_samples=args.max_samples)
    elif args.dataset == "busi":
        records = prepare_busi(ds_raw, ds_out, image_size=args.image_size, max_samples=args.max_samples)
    elif args.dataset == "isic2018":
        records = prepare_isic2018(ds_raw, ds_out, image_size=args.image_size, max_samples=args.max_samples)
    elif args.dataset == "mvtec":
        records = prepare_mvtec(ds_raw, ds_out, image_size=args.image_size, max_samples=args.max_samples)
    elif args.dataset == "brats":
        records = prepare_brats_2d(ds_raw, ds_out, image_size=args.image_size, max_samples=args.max_samples)
    elif args.dataset == "drive":
        records = prepare_drive(ds_raw, ds_out, image_size=args.image_size, max_samples=args.max_samples)
    elif args.dataset == "eurosat":
        records = prepare_eurosat(ds_raw, ds_out, image_size=args.image_size, max_samples=args.max_samples or 1000)
    elif args.dataset == "ph2_ood":
        records = prepare_ph2(ds_raw, ds_out, image_size=args.image_size, max_samples=args.max_samples)
    elif args.dataset == "whu":
        records = prepare_whu(ds_raw, ds_out, image_size=args.image_size, max_samples=args.max_samples)
    else:
        raise NotImplementedError(
            f"Place raw '{args.dataset}' archive under {ds_raw} and wire its folder adapter."
        )

    manifest = build_leak_free_nested_splits(records, n_folds=5, seed=args.seed)
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.splits_dir / f"{args.dataset}.json"
    split_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(records)} samples and leak-free 5-fold manifest to {split_path} (sha256={manifest['splits_sha256']})")


if __name__ == "__main__":
    main()

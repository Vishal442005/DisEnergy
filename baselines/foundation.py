"""Optional zero-shot foundation model wrappers (SAM, MedSAM, DINOv2)."""

from __future__ import annotations

from typing import Any

import numpy as np


class SAMZeroShotBaseline:
    """Lazy-loaded SAM / MedSAM automatic grid-prompted mask extractor."""

    def __init__(self, checkpoint: str, model_type: str = "vit_b", device: str = "cuda") -> None:
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment_anything is required for SAM/MedSAM baselines. "
                "Install via: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        self.generator = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.80)

    def predict_mask(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        """Select highest-area non-border-spanning SAM mask as foreground hypothesis."""
        masks: list[dict[str, Any]] = self.generator.generate(image_rgb_u8)
        h, w = image_rgb_u8.shape[:2]
        best_mask = np.zeros((h, w), dtype=bool)
        best_score = -1.0
        for m in masks:
            seg = m["segmentation"]
            area_frac = float(seg.mean())
            if 0.005 <= area_frac <= 0.75:
                score = float(m.get("predicted_iou", 0.5)) * float(m.get("stability_score", 0.5))
                if score > best_score:
                    best_score = score
                    best_mask = seg.astype(bool)
        return best_mask


class DINOv2FeatureExtractor:
    """Lazy-loaded DINOv2 backbone for zero-shot / kNN patch classification."""

    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required for DINOv2 baselines.") from exc

        self._torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def extract_cls_embedding(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image_rgb_u8, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
            cls_vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return cls_vec.astype(np.float32)

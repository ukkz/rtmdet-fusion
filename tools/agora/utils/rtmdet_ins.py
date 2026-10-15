#!/usr/bin/env python3
"""
RTMDet-Ins inference wrapper.

Lightweight inference class for person instance segmentation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

# Add the project root to the import path (configs.default, rtmdet_fusion)
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from configs.default import (
    RTMDET_INS_S_CFG,
    RTMDET_INS_S_WEIGHTS,
    IMG_SIZE,
    PERSON_CLASS_ID,
)


class RTMDetInsInference:
    """RTMDet-Ins inference"""

    def __init__(self, weights_path: str, device: str = None):
        """
        Args:
            weights_path: model weights file
            device: inference device (None = auto)
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device

        self.model = self._load_model()
        self._initialized = True

    def _load_model(self) -> nn.Module:
        """Build the model and load the weights"""
        # Import mmdet lazily
        from mmengine.config import ConfigDict
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules

        register_all_modules()

        # NMS / InstanceData workaround for Apple Silicon (MPS)
        if self.device == 'mps':
            from rtmdet_fusion.utils.mps_workaround import patch_nms
            patch_nms()

        cfg = ConfigDict(RTMDET_INS_S_CFG)
        model = MODELS.build(cfg)

        ckpt = torch.load(str(self.weights_path), map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def infer(self, image: np.ndarray) -> List[Dict]:
        """
        Detect persons in an image.

        Args:
            image: BGR image

        Returns:
            list of detections: [{'bbox': [x1,y1,x2,y2], 'score': float, 'mask': np.ndarray}, ...]
        """
        from mmdet.structures import DetDataSample

        orig_h, orig_w = image.shape[:2]

        # Resize and normalize
        img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        img_norm = (img_resized.astype(np.float32) - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # Backbone and neck
        backbone_feats = self.model.backbone(img_tensor)
        pafpn_feats = self.model.neck(backbone_feats)

        # Detection head
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': (IMG_SIZE, IMG_SIZE),
            'ori_shape': (IMG_SIZE, IMG_SIZE),
            'scale_factor': (1.0, 1.0),
            'pad_shape': (IMG_SIZE, IMG_SIZE),
        })

        results_list = self.model.bbox_head.predict(
            pafpn_feats, [data_sample], rescale=False
        )

        detections = []
        if len(results_list) > 0:
            result = results_list[0]
            bboxes = result.bboxes
            labels = result.labels
            scores = result.scores
            masks = result.masks if hasattr(result, 'masks') else None

            # Person class only
            person_mask = labels == PERSON_CLASS_ID
            person_indices = person_mask.nonzero(as_tuple=True)[0]

            for idx in person_indices:
                bbox = bboxes[idx].cpu().numpy()
                score = scores[idx].item()
                mask = masks[idx].cpu().numpy() if masks is not None else None

                # Scale to the input image size
                scale_x = orig_w / IMG_SIZE
                scale_y = orig_h / IMG_SIZE
                bbox_scaled = np.array([
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                ])

                # Resize the mask
                if mask is not None:
                    mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h))

                detections.append({
                    'bbox': bbox_scaled,
                    'score': score,
                    'mask': mask
                })

        return detections


# Singleton (lazy)
_inference_instance: Optional[RTMDetInsInference] = None


def get_rtmdet_ins_inference(weights_path: str = None) -> Optional[RTMDetInsInference]:
    """
    Return the shared RTMDet-Ins inference instance (singleton).

    Args:
        weights_path: weights file (first call only; default weights/rtmdet-ins_s.pth)

    Returns:
        RTMDetInsInference, or None if initialization failed
    """
    global _inference_instance

    if _inference_instance is not None:
        return _inference_instance

    if weights_path is None and RTMDET_INS_S_WEIGHTS.exists():
        weights_path = str(RTMDET_INS_S_WEIGHTS)

    if weights_path is None:
        print("Warning: RTMDet-Ins weights not found")
        return None

    try:
        _inference_instance = RTMDetInsInference(weights_path)
        print(f"RTMDet-Ins loaded from: {weights_path}")
        return _inference_instance
    except Exception as e:
        print(f"Warning: Failed to load RTMDet-Ins: {e}")
        return None

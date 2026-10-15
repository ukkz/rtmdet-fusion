#!/usr/bin/env python3
"""RTMDet-Fusion inference viewer

Runs a checkpoint trained with tools/train.py (PyTorch .pth) on an image and shows
detections, head/body keypoints, part masks and depth.

Keypoint coordinates (relative to the reference bbox):
- x: horizontal position in the bbox [0, 1]
- y: depth [0, 1] (center = 0.5)
- z: vertical position in the bbox [0, 1] (image Y flipped, up is positive)

Usage:
    uv run python tools/viewer.py --model work_dirs/<run>/best_model_s.pth
    uv run python tools/viewer.py --model work_dirs/<run>/best_model_s.pth --image path/to/image.jpg

Keys:
    O        open an image
    V        cycle view mode (both / head / body / parts / depth)
    R        toggle 2-pass reference bbox refit
    Q / Esc  quit
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QSizePolicy, QStatusBar, QFileDialog, QPushButton, QShortcut,
    QScrollArea, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QPainter, QPen, QBrush, QColor, QFont
from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer

from rtmdet_fusion.utils.mps_workaround import patch_nms
patch_nms()

from mmengine.config import ConfigDict
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from configs.default import (
    NUM_KEYPOINTS, MAX_INSTANCES, IMG_SIZE,
    PERSON_CLASS_ID, KEYPOINT_NAMES, SKELETON,
    KEYPOINT_COLORS_BGR, PERSON_COLORS_BGR,
    GRID_SIZE, NUM_BINS,
    NUM_BODY_KEYPOINTS, NUM_BINS_BODY, BIN_MARGIN,
    BODY_BBOX_RANGE_SCALE, BODY_REF_BBOX_EXPAND, BODY_KEYPOINT_NAMES, BODY_SKELETON,
    BODY_KEYPOINT_COLORS_BGR, BODY_KEYPOINT_COLORS_RGB,
    get_rtmdet_config, get_rtmdet_weights, get_rtmdet_in_channels,
)

from rtmdet_fusion.models.fusion_model import (
    BboxFeatureExtractor, HeadBboxHead, DCCHead, MaskDepthBranch
)
from rtmdet_fusion.models.encoders import HybridEncoder
from rtmdet_fusion.utils.versioning import check_checkpoint_version

# Part mask colors (BGR), keyed by MaskDepthBranch output channel
VIEWER_PART_COLORS_BGR = {
    2: (0, 0, 255),     # head: red
    3: (0, 200, 0),     # torso: green
    4: (255, 128, 0),   # limbs: blue
    5: (0, 255, 255),   # hands: yellow
}

KEYPOINT_COLORS_RGB = [
    (0, 1, 0),       # center - green
    (0, 0, 1),       # top - blue
    (1, 0, 1),       # left_ear - magenta
    (0.8, 0, 0.8),   # right_ear
    (0, 1, 1),       # left_eye - cyan
    (0, 0.8, 0.8),   # right_eye
    (1, 0, 0),       # nose - red
    (1, 0.5, 0),     # chin - orange
    (0.5, 0.5, 0.5), # occiput - gray
    (1, 1, 0),       # vertex - yellow
    (0, 0.8, 1),     # neck - light blue
]

# Checkpoint entries every model needs (attribute name = key without '_state_dict')
REQUIRED_CHECKPOINT_KEYS = (
    'hybrid_encoder_state_dict',
    'bbox_feat_extractor_state_dict',
    'head_bbox_head_state_dict',
    'dcc_head_state_dict',
    'body_bbox_head_state_dict',
    'body_kpt_head_state_dict',
)


# ============================================================
# Inference model
# ============================================================

class RTMDetFusionForInference(nn.Module):
    """RTMDet-Fusion for inference"""

    def __init__(self, rtmdet_cfg: ConfigDict, in_channels: int = 128):
        """
        Args:
            rtmdet_cfg: RTMDet-Ins config
            in_channels: RTMDet-Ins backbone C3 channels (tiny: 96, s: 128)
        """
        super().__init__()

        self.rtmdet = MODELS.build(rtmdet_cfg)

        # Backbone output channels (C3, C4, C5)
        backbone_channels = [in_channels, in_channels * 2, in_channels * 4]

        self.hybrid_encoder = HybridEncoder(
            in_channels=backbone_channels,
            hidden_dim=256,
            out_channels=256,
            num_csp_blocks=3,
            widen_factor=0.5
        )

        self.bbox_feat_extractor = BboxFeatureExtractor(
            in_channels=256,
            out_channels=256,
            grid_size=GRID_SIZE
        )

        self.head_bbox_head = HeadBboxHead(
            in_channels=256,
            hidden_dim=256,
            num_layers=2
        )

        self.dcc_head = DCCHead(
            in_channels=256,
            num_keypoints=NUM_KEYPOINTS,
            num_bins=NUM_BINS,
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            target_sigma=0.025,
            bin_margin=BIN_MARGIN
        )

        # Body reference bbox head + body keypoint head
        self.body_bbox_head = HeadBboxHead(
            in_channels=256,
            hidden_dim=256,
            num_layers=2,
            range_scale=BODY_BBOX_RANGE_SCALE
        )
        self.body_kpt_head = DCCHead(
            in_channels=256,
            num_keypoints=NUM_BODY_KEYPOINTS,
            num_bins=NUM_BINS_BODY,
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            target_sigma=0.025,
            bin_margin=BIN_MARGIN
        )

        # Mask/depth branch (created by load_model only if the checkpoint has it)
        self.mask_depth_branch = None
        self.has_mask = False

    def extract_backbone_feat(self, images: torch.Tensor) -> List[torch.Tensor]:
        return self.rtmdet.backbone(images)

    def extract_pafpn_feat(self, backbone_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.rtmdet.neck(backbone_feats)

    def extract_hybrid_feat(self, backbone_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.hybrid_encoder(backbone_feats)


def detect_variant_from_checkpoint(checkpoint: dict) -> str:
    """Detect the RTMDet variant from the checkpoint's tensor shapes.

    hybrid_encoder input_proj.0.0.weight is [256, C3_channels, 1, 1]:
    - [256, 96, 1, 1] -> tiny
    - [256, 128, 1, 1] -> s
    """
    in_channels = checkpoint['hybrid_encoder_state_dict']['input_proj.0.0.weight'].shape[1]
    for variant in ('tiny', 's'):
        if get_rtmdet_in_channels(variant) == in_channels:
            return variant
    raise ValueError(f"Unknown Hybrid Encoder input channels: {in_channels}")


def load_model(checkpoint_path: str, rtmdet_weights: Optional[str], device: str) -> RTMDetFusionForInference:
    """Load a model.

    Args:
        checkpoint_path: checkpoint written by tools/train.py
        rtmdet_weights: RTMDet-Ins weights; None selects the weights of the detected variant
        device: torch device
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in checkpoint]
    if missing:
        raise ValueError(
            f"{checkpoint_path} is not a v2 checkpoint (missing: {', '.join(missing)})"
        )

    check_checkpoint_version(checkpoint, tool_name='viewer')

    variant = detect_variant_from_checkpoint(checkpoint)
    print(f"Detected RTMDet variant: {variant}")

    rtmdet_cfg = ConfigDict(get_rtmdet_config(variant))
    in_channels = get_rtmdet_in_channels(variant)
    model = RTMDetFusionForInference(rtmdet_cfg, in_channels=in_channels)

    weights_path = Path(rtmdet_weights) if rtmdet_weights else get_rtmdet_weights(variant)
    if not weights_path.exists():
        raise FileNotFoundError(f"RTMDet-Ins weights not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.rtmdet.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    print(f"Loaded RTMDet weights: {weights_path}")

    for key in REQUIRED_CHECKPOINT_KEYS:
        getattr(model, key.removesuffix('_state_dict')).load_state_dict(checkpoint[key])

    # Absent in stage-1 checkpoints (trained with the mask/depth branch disabled)
    if 'mask_depth_branch_state_dict' in checkpoint:
        model.mask_depth_branch = MaskDepthBranch()
        model.mask_depth_branch.load_state_dict(checkpoint['mask_depth_branch_state_dict'])
        model.has_mask = True
        print("  Loaded mask/depth branch")
    else:
        print("  No mask/depth branch in checkpoint (parts/depth views disabled)")

    print(f"Loaded checkpoint: {checkpoint_path}")

    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    metrics = checkpoint.get('val_metrics', {})
    shown = [f"{k}: {metrics[k]:.4f}" for k in ('loss', 'error_unified', 'body_error_unified')
             if k in metrics]
    if shown:
        print(f"  Val {', '.join(shown)}")

    model = model.to(device)
    model.eval()
    return model


# ============================================================
# Inference
# ============================================================

def refit_bbox_from_keypoints(
    kpts_rel: torch.Tensor,
    ref_bboxes: torch.Tensor,
    expand_ratio: float = 1.2,
    square: bool = True,
) -> torch.Tensor:
    """Rebuild reference bboxes from predicted keypoints with the ground-truth rule (2-pass refit).

    Ground-truth reference bbox rules:
        head: bounding box of all keypoints -> square on the long side (center kept) -> x1.2
        body: bounding rectangle of all keypoints -> x BODY_REF_BBOX_EXPAND (not squared)

    Args:
        kpts_rel: (N, K, 3) first-pass relative coordinates [rel_x, rel_y, rel_z]
        ref_bboxes: (N, 4) first-pass reference bboxes [x1, y1, x2, y2]
        expand_ratio: expansion ratio
        square: whether to make the bbox square (head=True, body=False)

    Returns:
        refit_bboxes: (N, 4) refitted reference bboxes
    """
    x1 = ref_bboxes[:, 0:1]
    y1 = ref_bboxes[:, 1:2]
    w = (ref_bboxes[:, 2] - ref_bboxes[:, 0]).unsqueeze(1)
    h = (ref_bboxes[:, 3] - ref_bboxes[:, 1]).unsqueeze(1)

    # Relative -> image coordinates
    img_x = kpts_rel[:, :, 0] * w + x1              # (N, K)
    img_y = (1 - kpts_rel[:, :, 2]) * h + y1        # (N, K)

    x_min = img_x.min(dim=1).values
    x_max = img_x.max(dim=1).values
    y_min = img_y.min(dim=1).values
    y_max = img_y.max(dim=1).values

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    if square:
        half_w = half_h = torch.maximum(x_max - x_min, y_max - y_min) / 2 * expand_ratio
    else:
        half_w = (x_max - x_min) / 2 * expand_ratio
        half_h = (y_max - y_min) / 2 * expand_ratio

    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)


def predict_keypoints(model: RTMDetFusionForInference, kpt_head: DCCHead,
                      hybrid_feats: List[torch.Tensor], ref_bboxes: torch.Tensor,
                      priors: torch.Tensor, valid_mask: torch.Tensor):
    """Sample features in the reference bboxes and decode a keypoint head.

    Returns:
        rel_coords (N, K, 3), confidences (N, K), visibilities (N, K), in_frame (N, K)
    """
    grid_features, pos_encoding = model.bbox_feat_extractor(
        hybrid_feats, ref_bboxes, priors, valid_mask, IMG_SIZE
    )
    output = kpt_head(grid_features, pos_encoding)
    rel_coords, confidences, visibilities = kpt_head.decode_relative(
        output['heatmap_x'],
        output['heatmap_y'],
        output['heatmap_z'],
        output['visibility_logits']
    )
    return rel_coords, confidences, visibilities, torch.sigmoid(output['in_frame_logits'])


def rel_to_image(rel_coords: np.ndarray, ref_bbox: np.ndarray) -> np.ndarray:
    """Relative keypoints (K, 3) -> image coordinates (K, 2). z is up-positive, so it is flipped."""
    x1, y1, x2, y2 = ref_bbox
    img_x = rel_coords[:, 0] * (x2 - x1) + x1
    img_y = (1 - rel_coords[:, 2]) * (y2 - y1) + y1
    return np.stack([img_x, img_y], axis=-1)


@torch.no_grad()
def run_inference(model: RTMDetFusionForInference, image: np.ndarray, device: str,
                  refit: bool = False) -> List[Dict]:
    """Run inference on a BGR image.

    Args:
        refit: if True, run the 2-pass reference bbox refit: rebuild the reference bboxes
               from the first-pass keypoints with the ground-truth rule and decode again.
               This removes the parallel keypoint offset caused by systematic error in
               the predicted reference bbox.
    """
    orig_h, orig_w = image.shape[:2]
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    img_norm = (img_resized.astype(np.float32) - mean) / std
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    backbone_feats = model.extract_backbone_feat(img_tensor)
    pafpn_feats = model.extract_pafpn_feat(backbone_feats)
    hybrid_feats = model.extract_hybrid_feat(backbone_feats)

    # Person detection (RTMDet-Ins)
    data_sample = DetDataSample()
    data_sample.set_metainfo({
        'img_shape': (IMG_SIZE, IMG_SIZE),
        'ori_shape': (IMG_SIZE, IMG_SIZE),
        'scale_factor': (1.0, 1.0),
        'pad_shape': (IMG_SIZE, IMG_SIZE),
    })
    result = model.rtmdet.bbox_head.predict(pafpn_feats, [data_sample], rescale=False)[0]

    person_indices = (result.labels == PERSON_CLASS_ID).nonzero(as_tuple=True)[0]
    if len(person_indices) == 0:
        return []
    if len(person_indices) > MAX_INSTANCES:
        sorted_order = result.scores[person_indices].argsort(descending=True)[:MAX_INSTANCES]
        person_indices = person_indices[sorted_order]

    batch_bboxes = result.bboxes[person_indices]
    batch_scores = result.scores[person_indices]
    batch_masks = result.masks[person_indices].cpu().numpy()
    batch_priors = result.priors[person_indices]
    N = len(person_indices)
    valid_mask = torch.ones(N, dtype=torch.bool, device=batch_bboxes.device)

    # Instance features in the person bbox -> head bbox
    grid_features, _ = model.bbox_feat_extractor(
        hybrid_feats, batch_bboxes, batch_priors, valid_mask, IMG_SIZE
    )
    head_bboxes = model.head_bbox_head(grid_features, batch_bboxes)
    head_bboxes_expanded = model.dcc_head.expand_bbox(head_bboxes)

    # Head keypoints (relative to head_bboxes_expanded)
    head_kpts = predict_keypoints(
        model, model.dcc_head, hybrid_feats, head_bboxes_expanded, batch_priors, valid_mask
    )
    if refit:
        head_bboxes_expanded = refit_bbox_from_keypoints(
            head_kpts[0], head_bboxes_expanded, expand_ratio=1.2
        )
        head_bboxes = head_bboxes_expanded  # display the refitted box
        head_kpts = predict_keypoints(
            model, model.dcc_head, hybrid_feats, head_bboxes_expanded, batch_priors, valid_mask
        )

    # Body reference bbox + body keypoints
    body_bboxes_expanded = model.body_kpt_head.expand_bbox(
        model.body_bbox_head(grid_features, batch_bboxes)
    )
    body_kpts = predict_keypoints(
        model, model.body_kpt_head, hybrid_feats, body_bboxes_expanded, batch_priors, valid_mask
    )
    if refit:
        body_bboxes_expanded = refit_bbox_from_keypoints(
            body_kpts[0], body_bboxes_expanded,
            expand_ratio=BODY_REF_BBOX_EXPAND, square=False
        )
        body_kpts = predict_keypoints(
            model, model.body_kpt_head, hybrid_feats, body_bboxes_expanded, batch_priors, valid_mask
        )

    # Part masks + depth (stride 8, upsampled to the input size)
    mask_pred_batch = None
    if model.has_mask:
        inst_feats = grid_features.mean(dim=1)  # (N, 256)
        mask_out = model.mask_depth_branch(hybrid_feats, inst_feats, batch_priors)  # (N, 8, H8, W8)
        mask_pred_batch = F.interpolate(
            mask_out, size=(IMG_SIZE, IMG_SIZE),
            mode='bilinear', align_corners=False
        )

    head_rel, head_conf, head_vis, head_in_frame = (t.cpu().numpy() for t in head_kpts)
    body_rel, body_conf, body_vis, body_in_frame = (t.cpu().numpy() for t in body_kpts)
    head_ref = head_bboxes_expanded.cpu().numpy()
    body_ref = body_bboxes_expanded.cpu().numpy()

    detections = []
    for i in range(N):
        det = {
            'bbox': batch_bboxes[i].cpu().numpy(),
            'head_bbox': head_bboxes[i].cpu().numpy(),
            'score': batch_scores[i].item(),
            'mask': batch_masks[i],
            'keypoints_2d': rel_to_image(head_rel[i], head_ref[i]),
            'keypoints_rel': head_rel[i],         # [0, 1] relative coordinates
            'confidences': head_conf[i],
            'visibilities': head_vis[i],          # visibility probability [0, 1]
            'in_frame': head_in_frame[i],         # in-frame probability [0, 1]
            'body_bbox': body_ref[i],
            'body_keypoints_2d': rel_to_image(body_rel[i], body_ref[i]),
            'body_keypoints_rel': body_rel[i],
            'body_confidences': body_conf[i],
            'body_visibilities': body_vis[i],
            'body_in_frame': body_in_frame[i],
        }
        if mask_pred_batch is not None:
            det['pred_masks'] = torch.sigmoid(mask_pred_batch[i, :7]).cpu().numpy()  # (7, H, W) probabilities
            det['pred_depth'] = mask_pred_batch[i, 7].cpu().numpy()  # rel_y-equivalent depth
        detections.append(det)

    # Scale to the original image size
    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE
    bbox_scale = np.array([scale_x, scale_y, scale_x, scale_y])
    for det in detections:
        for key in ('bbox', 'head_bbox', 'body_bbox'):
            det[key] = (det[key] * bbox_scale).astype(np.float32)
        for key in ('keypoints_2d', 'body_keypoints_2d'):
            det[key][:, 0] *= scale_x
            det[key][:, 1] *= scale_y
        if 'pred_masks' in det:
            pm = det['pred_masks'].transpose(1, 2, 0)  # (H, W, 7)
            det['pred_masks'] = cv2.resize(
                pm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)
            det['pred_depth'] = cv2.resize(
                det['pred_depth'], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
        # Boolean instance mask -> uint8 [0, 255]
        det['mask'] = cv2.resize(det['mask'].astype(np.uint8) * 255, (orig_w, orig_h))

    return detections


# ============================================================
# GUI widgets
# ============================================================

class Person3DWidget(FigureCanvas):
    """3D view of one person's keypoints ([0, 1] relative coordinates)"""

    def __init__(self, parent=None, width=4, height=3):
        self.fig = Figure(figsize=(width, height), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.elev = 20
        self.azim = -90
        self._dragging = False
        self._last_pos = None

        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)

    def _on_press(self, event):
        if event.button == 1:
            self._dragging = True
            self._last_pos = (event.x, event.y)

    def _on_release(self, event):
        self._dragging = False
        self._last_pos = None

    def _on_motion(self, event):
        if self._dragging and self._last_pos and event.x and event.y:
            dx = event.x - self._last_pos[0]
            self.azim += dx * 0.5
            self._last_pos = (event.x, event.y)
            self._update_view()

    def _update_view(self):
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.draw()

    def plot_keypoints(
        self,
        keypoints_rel: np.ndarray,
        person_idx: int,
        use_body: bool = False,
    ):
        """Plot keypoints in [0, 1] relative coordinates.

        Args:
            keypoints_rel: (K, 3) [rel_x, rel_y, rel_z], all in [0, 1]
            use_body: True for the 31 body keypoints (no name labels, body skeleton and colors)
        """
        self.ax.clear()

        if use_body:
            names = BODY_KEYPOINT_NAMES
            skeleton = BODY_SKELETON
            colors = BODY_KEYPOINT_COLORS_RGB
        else:
            names = KEYPOINT_NAMES
            skeleton = SKELETON
            colors = KEYPOINT_COLORS_RGB

        # Axes: X = x (right), Y = y (depth, 0.5 = center), Z = z (up)
        points = np.asarray(keypoints_rel)

        for i, (x, y, z) in enumerate(points):
            color = colors[i % len(colors)]
            self.ax.scatter([x], [y], [z], c=[color], s=60 if use_body else 100)
            if not use_body:
                label = names[i] if i < len(names) else f"kpt{i}"
                self.ax.text(x, y, z, f" {label}", fontsize=12, color=color)

        for s1, s2 in skeleton:
            if s1 < len(points) and s2 < len(points):
                x1, y1, z1 = points[s1]
                x2, y2, z2 = points[s2]
                self.ax.plot([x1, x2], [y1, y2], [z1, z2], 'gray', linewidth=1, alpha=0.7)

        # Fixed [0, 1] range
        margin = 0.05
        self.ax.set_xlim(-margin, 1 + margin)
        self.ax.set_ylim(-margin, 1 + margin)
        self.ax.set_zlim(-margin, 1 + margin)

        # Grid on the z=0 plane
        grid_x = np.linspace(0, 1, 5)
        grid_y = np.linspace(0, 1, 5)
        for gx in grid_x:
            self.ax.plot([gx, gx], [0, 1], [0, 0], 'lightgray', linewidth=0.5, alpha=0.5)
        for gy in grid_y:
            self.ax.plot([0, 1], [gy, gy], [0, 0], 'lightgray', linewidth=0.5, alpha=0.5)

        # Reference line at center = 0.5
        self.ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)

        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X (left/right)', fontsize=12)
        self.ax.set_ylabel('Y (depth, 0.5=center)', fontsize=12)
        self.ax.set_zlabel('Z (up)', fontsize=12)
        system = 'Body 31pts' if use_body else 'Head'
        self.ax.set_title(f'Person #{person_idx} ({system}, [0,1])', fontsize=16, fontweight='bold', y=0.95)
        self.ax.tick_params(labelsize=10)
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)
        self.draw()


class ImageWidget(QWidget):
    """Image view with detection overlays"""

    FONT_SIZE_LABEL = 40
    FONT_SIZE_KPT = 30
    FONT_SIZE_INFO = 30
    CIRCLE_RADIUS = 10
    CIRCLE_OUTLINE = 12
    CIRCLE_RADIUS_BODY = 7
    LINE_WIDTH_BBOX = 4
    LINE_WIDTH_HEAD_BBOX = 2
    LINE_WIDTH_SKELETON = 2

    # View modes cycled with V (parts/depth need a checkpoint with the mask/depth branch)
    VIEW_MODES = ['both', 'head', 'body', 'parts', 'depth']

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)
        self._pixmap = None
        self._image_size = (0, 0)
        self._detections = []
        self._info_text = ""
        self._view_mode = 0

    def cycle_view_mode(self) -> str:
        """Switch to the next view mode and return its name"""
        self._view_mode = (self._view_mode + 1) % len(self.VIEW_MODES)
        self.update()
        return self.VIEW_MODES[self._view_mode]

    def current_view_mode(self) -> str:
        return self.VIEW_MODES[self._view_mode]

    def set_image(self, image: np.ndarray):
        h, w = image.shape[:2]
        self._image_size = (w, h)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def set_detections(self, detections: List[Dict], info_text: str = ""):
        self._detections = detections
        self._info_text = info_text
        self.update()

    def _get_image_rect(self) -> QRectF:
        if not self._pixmap:
            return QRectF()

        widget_w, widget_h = self.width(), self.height()
        img_w, img_h = self._image_size

        scale = min(widget_w / img_w, widget_h / img_h)
        scaled_w = img_w * scale
        scaled_h = img_h * scale

        x = (widget_w - scaled_w) / 2
        y = (widget_h - scaled_h) / 2

        return QRectF(x, y, scaled_w, scaled_h)

    def _img_to_widget(self, img_x: float, img_y: float) -> QPointF:
        rect = self._get_image_rect()
        if rect.isEmpty() or self._image_size[0] == 0:
            return QPointF(0, 0)

        scale_x = rect.width() / self._image_size[0]
        scale_y = rect.height() / self._image_size[1]

        return QPointF(rect.x() + img_x * scale_x, rect.y() + img_y * scale_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if not self._pixmap:
            return

        rect = self._get_image_rect()
        painter.drawPixmap(rect.toRect(), self._pixmap)

        self._draw_annotations(painter)

    def _draw_annotations(self, painter: QPainter):
        if not self._detections:
            if self._info_text:
                self._draw_info_text(painter)
            return

        mode = self.VIEW_MODES[self._view_mode]
        show_head = mode in ('head', 'both')
        show_body = mode in ('body', 'both')
        # In parts/depth modes the overlay is composited into the image (_display_results);
        # only person bboxes and labels are drawn here

        for i, det in enumerate(self._detections):
            color_bgr = PERSON_COLORS_BGR[i % len(PERSON_COLORS_BGR)]
            color = QColor(color_bgr[2], color_bgr[1], color_bgr[0])

            bbox = det['bbox']
            head_bbox = det['head_bbox']
            score = det['score']
            kpts_2d = det['keypoints_2d']
            confs = det['confidences']
            vis = det['visibilities']
            in_frame = det['in_frame']

            p1 = self._img_to_widget(bbox[0], bbox[1])
            p2 = self._img_to_widget(bbox[2], bbox[3])
            pen = QPen(color, self.LINE_WIDTH_BBOX)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(p1, p2))

            if show_head:
                hp1 = self._img_to_widget(head_bbox[0], head_bbox[1])
                hp2 = self._img_to_widget(head_bbox[2], head_bbox[3])
                pen = QPen(QColor(0, 255, 0), self.LINE_WIDTH_HEAD_BBOX)
                painter.setPen(pen)
                painter.drawRect(QRectF(hp1, hp2))

            # Body reference bbox (orange dashed)
            if show_body:
                body_bbox = det['body_bbox']
                bp1 = self._img_to_widget(body_bbox[0], body_bbox[1])
                bp2 = self._img_to_widget(body_bbox[2], body_bbox[3])
                pen = QPen(QColor(255, 165, 0), self.LINE_WIDTH_HEAD_BBOX)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(QRectF(bp1, bp2))

            label = f"#{i} ({score:.2f})"
            font = QFont()
            font.setPixelSize(self.FONT_SIZE_LABEL)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(color))
            painter.drawText(p1 + QPointF(2, -5), label)

            # Body keypoints (points + skeleton, no labels)
            if show_body:
                body_kpts_2d = det['body_keypoints_2d']
                body_in_frame = det['body_in_frame']
                body_pts = [
                    self._img_to_widget(body_kpts_2d[j, 0], body_kpts_2d[j, 1])
                    for j in range(len(body_kpts_2d))
                ]
                pen = QPen(QColor(200, 200, 200, 180), self.LINE_WIDTH_SKELETON)
                painter.setPen(pen)
                for s1, s2 in BODY_SKELETON:
                    if s1 < len(body_pts) and s2 < len(body_pts):
                        painter.drawLine(body_pts[s1], body_pts[s2])
                for j in range(len(body_pts)):
                    c = BODY_KEYPOINT_COLORS_BGR[j % len(BODY_KEYPOINT_COLORS_BGR)]
                    kc = QColor(c[2], c[1], c[0])
                    pt = body_pts[j]
                    if body_in_frame[j] < 0.5:
                        # Predicted out of frame: draw a cross
                        r = self.CIRCLE_RADIUS_BODY
                        painter.setPen(QPen(kc, 2))
                        painter.drawLine(pt + QPointF(-r, -r), pt + QPointF(r, r))
                        painter.drawLine(pt + QPointF(-r, r), pt + QPointF(r, -r))
                    else:
                        painter.setPen(QPen(QColor(255, 255, 255), 1))
                        painter.setBrush(QBrush(kc))
                        painter.drawEllipse(pt, self.CIRCLE_RADIUS_BODY, self.CIRCLE_RADIUS_BODY)

            # Head keypoints
            if show_head:
                pen = QPen(QColor(180, 180, 180), self.LINE_WIDTH_SKELETON)
                painter.setPen(pen)
                widget_pts = []
                for j in range(NUM_KEYPOINTS):
                    kx, ky = kpts_2d[j]
                    widget_pts.append(self._img_to_widget(kx, ky))

                for s1, s2 in SKELETON:
                    if s1 < len(widget_pts) and s2 < len(widget_pts):
                        painter.drawLine(widget_pts[s1], widget_pts[s2])

                for j in range(NUM_KEYPOINTS):
                    conf = confs[j]
                    kpt_color_bgr = KEYPOINT_COLORS_BGR[j % len(KEYPOINT_COLORS_BGR)]

                    alpha = int(max(0.7, conf) * 255)
                    kpt_color = QColor(kpt_color_bgr[2], kpt_color_bgr[1], kpt_color_bgr[0], alpha)

                    pt = widget_pts[j]

                    if in_frame[j] < 0.5:
                        # Predicted out of frame: draw a cross
                        r = self.CIRCLE_RADIUS
                        painter.setPen(QPen(kpt_color, 3))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawLine(pt + QPointF(-r, -r), pt + QPointF(r, r))
                        painter.drawLine(pt + QPointF(-r, r), pt + QPointF(r, -r))
                    else:
                        painter.setPen(QPen(QColor(255, 255, 255, alpha), 1))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawEllipse(pt, self.CIRCLE_OUTLINE, self.CIRCLE_OUTLINE)

                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(kpt_color))
                        painter.drawEllipse(pt, self.CIRCLE_RADIUS, self.CIRCLE_RADIUS)

                    kpt_name = KEYPOINT_NAMES[j] if j < len(KEYPOINT_NAMES) else f'{j}'
                    font = QFont()
                    font.setPixelSize(self.FONT_SIZE_KPT)
                    painter.setFont(font)
                    painter.setPen(QPen(kpt_color))
                    painter.drawText(pt + QPointF(8, -3), f"{kpt_name}({conf:.2f},{vis[j]:.2f})")

        # View mode indicator
        font = QFont()
        font.setPixelSize(24)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 0)))
        painter.drawText(QPointF(self.width() - 220, 60), f"[V: {mode.upper()}]")

        self._draw_info_text(painter)

    def _draw_info_text(self, painter: QPainter):
        if not self._info_text:
            return

        font = QFont()
        font.setPixelSize(self.FONT_SIZE_INFO)
        font.setBold(True)
        painter.setFont(font)

        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(QPointF(12, 27), self._info_text)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(QPointF(10, 25), self._info_text)


class MainWindow(QMainWindow):
    """Main window"""

    KEY_HELP = "O: open image | V: cycle view mode | R: toggle 2-pass refit | Q/Esc: quit"

    def __init__(self, model: RTMDetFusionForInference, device: str, image_path: Optional[str] = None):
        super().__init__()
        self.model = model
        self.device = device
        self.current_image = None
        self.current_detections = []
        self.current_image_path = None
        self.refit = False  # 2-pass reference bbox refit (toggled with R)
        self._pending_image_path = image_path
        self._shown = False

        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("RTMDet-Fusion Viewer")

        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(50, 50, screen.width() - 100, screen.height() - 100)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout(main_widget)

        button_layout = QHBoxLayout()
        self.load_btn = QPushButton("Open Image (O)")
        self.load_btn.clicked.connect(self._on_load_image)
        button_layout.addWidget(self.load_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        image_layout = QHBoxLayout()

        self.image_widget_2d = ImageWidget()
        image_layout.addWidget(self.image_widget_2d, stretch=3)

        self.scroll_area_3d = QScrollArea()
        self.scroll_area_3d.setWidgetResizable(True)
        self.scroll_area_3d.setStyleSheet("background-color: #333;")
        self.scroll_area_3d.setFrameStyle(QFrame.NoFrame)

        self.container_3d = QWidget()
        self.layout_3d = QVBoxLayout(self.container_3d)
        self.layout_3d.setContentsMargins(5, 5, 5, 5)
        self.layout_3d.setSpacing(5)

        self.placeholder_3d = QLabel("3D View [0,1]\n(after detection)")
        self.placeholder_3d.setAlignment(Qt.AlignCenter)
        self.placeholder_3d.setStyleSheet("color: #888; font-size: 14px;")
        self.layout_3d.addWidget(self.placeholder_3d)
        self.layout_3d.addStretch()

        self.scroll_area_3d.setWidget(self.container_3d)
        image_layout.addWidget(self.scroll_area_3d, stretch=2)

        self.person_3d_widgets: List[Person3DWidget] = []

        layout.addLayout(image_layout)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self.KEY_HELP)

        QShortcut(QKeySequence(Qt.Key_O), self, self._on_load_image)
        QShortcut(QKeySequence(Qt.Key_V), self, self._on_cycle_view_mode)
        QShortcut(QKeySequence(Qt.Key_R), self, self._on_toggle_refit)
        QShortcut(QKeySequence(Qt.Key_Q), self, self.close)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.close)

    def _on_cycle_view_mode(self):
        """Cycle view modes (both -> head -> body -> parts -> depth)"""
        mode = self.image_widget_2d.cycle_view_mode()
        # parts/depth change the composited image, so redraw (skip if no image is loaded)
        if self.current_image is not None and self.current_image_path:
            self._display_results(self.current_image_path)
        else:
            self._update_3d_views(self.current_detections)
        note = ""
        if mode in ('parts', 'depth') and not self.model.has_mask:
            note = " - this checkpoint has no mask/depth branch (stage-2 checkpoint required)"
        self.status_bar.showMessage(f"View mode: {mode.upper()} (V to cycle){note}")

    def _on_toggle_refit(self):
        """Toggle the 2-pass reference bbox refit and re-run inference on the current image"""
        self.refit = not self.refit
        state = "ON" if self.refit else "OFF"
        self.status_bar.showMessage(f"2-pass refit: {state} (R to toggle)")
        if self.current_image_path:
            self._load_and_infer(self.current_image_path)

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if path:
            self._load_and_infer(path)

    def _load_and_infer(self, image_path):
        self.status_bar.showMessage(f"Processing: {image_path}")
        QApplication.processEvents()

        image = cv2.imread(image_path)
        if image is None:
            self.status_bar.showMessage(f"Failed to load: {image_path}")
            return

        self.current_image = image.copy()
        self.current_image_path = image_path

        try:
            self.current_detections = run_inference(
                self.model, image, self.device, refit=self.refit
            )
        except Exception as e:
            self.status_bar.showMessage(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return

        self._display_results(image_path)

    def _display_results(self, image_path):
        image = self.current_image.copy()
        detections = self.current_detections
        mode = self.image_widget_2d.current_view_mode()

        if mode == 'parts':
            # Part masks (head=red / torso=green / limbs=blue / hands=yellow)
            for det in detections:
                pm = det.get('pred_masks')
                if pm is None:
                    continue
                for ch, color in VIEWER_PART_COLORS_BGR.items():
                    sel = pm[ch] > 0.5
                    image[sel] = (
                        image[sel] * 0.45 + np.array(color) * 0.55
                    ).astype(np.uint8)
        elif mode == 'depth':
            # Depth, normalized per person symmetrically around the pelvis (0.5); red=near, blue=far
            for det in detections:
                pm = det.get('pred_masks')
                dp = det.get('pred_depth')
                if pm is None or dp is None:
                    continue
                sel = pm[0] > 0.5  # inside the clothed full-body mask only
                if not sel.any():
                    continue
                vals = dp[sel] - 0.5
                radius = max(abs(float(vals.min())), abs(float(vals.max())), 1e-3)
                norm = np.zeros_like(dp)
                norm[sel] = np.clip((dp[sel] - 0.5 + radius) / (2 * radius), 0, 1)
                colored = cv2.applyColorMap(
                    (255 - norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO
                )
                image[sel] = (
                    image[sel] * 0.25 + colored[sel] * 0.75
                ).astype(np.uint8)
        else:
            # RTMDet-Ins instance masks in person colors
            for i, det in enumerate(detections):
                color = PERSON_COLORS_BGR[i % len(PERSON_COLORS_BGR)]
                mask_bool = det['mask'] > 127  # uint8 [0, 255]
                image[mask_bool] = (
                    image[mask_bool] * 0.5 + np.array(color) * 0.5
                ).astype(np.uint8)

        self.image_widget_2d.set_image(image)

        refit_text = " | Refit: ON" if self.refit else ""
        info = f"File: {Path(image_path).name} | Detections: {len(detections)}{refit_text}"
        self.image_widget_2d.set_detections(detections, info)

        self._update_3d_views(detections)

        self.status_bar.showMessage(f"{len(detections)} persons detected - {image_path}")

    def _update_3d_views(self, detections: List[Dict]):
        for widget in self.person_3d_widgets:
            self.layout_3d.removeWidget(widget)
            widget.deleteLater()
        self.person_3d_widgets.clear()

        if detections:
            self.placeholder_3d.hide()
        else:
            self.placeholder_3d.show()
            return

        for i in range(self.layout_3d.count() - 1, -1, -1):
            item = self.layout_3d.itemAt(i)
            if item.spacerItem():
                self.layout_3d.removeItem(item)

        viewport_height = self.scroll_area_3d.viewport().height()
        widget_height = max(300, int(viewport_height * 0.5))

        # Body mode shows the 31 body keypoints; every other mode shows the head
        use_body = self.image_widget_2d.current_view_mode() == 'body'

        for i, det in enumerate(detections):
            kpts_rel = det['body_keypoints_rel'] if use_body else det['keypoints_rel']
            widget = Person3DWidget(parent=self.container_3d, width=5, height=4)
            widget.setFixedHeight(widget_height)
            widget.plot_keypoints(kpts_rel, i, use_body=use_body)
            self.layout_3d.addWidget(widget)
            self.person_3d_widgets.append(widget)

    def _resize_3d_widgets(self):
        if not self.person_3d_widgets:
            return

        viewport_height = self.scroll_area_3d.viewport().height()
        widget_height = max(300, int(viewport_height * 0.5))

        for widget in self.person_3d_widgets:
            widget.setFixedHeight(widget_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_3d_widgets()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._shown and self._pending_image_path:
            self._shown = True
            QTimer.singleShot(0, lambda: self._load_and_infer(self._pending_image_path))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RTMDet-Fusion inference viewer")
    parser.add_argument('--model', type=str, required=True,
                        help="Checkpoint (.pth) written by tools/train.py")
    parser.add_argument('--rtmdet-weights', type=str, default=None,
                        help="RTMDet-Ins weights (default: weights/rtmdet-ins_{variant}.pth "
                             "for the variant detected from the checkpoint)")
    parser.add_argument('--image', type=str, default=None,
                        help="Image to run on at startup")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    register_all_modules()
    model = load_model(args.model, args.rtmdet_weights, device)

    app = QApplication(sys.argv)
    window = MainWindow(model, device, args.image)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

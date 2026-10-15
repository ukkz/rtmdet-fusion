#!/usr/bin/env python3
"""Train RTMDet-Fusion.

The frozen RTMDet-Ins detector (tiny and/or s) supplies person boxes, masks and backbone
features. On top of it the script trains the Hybrid Encoder, the head/body bbox heads, the head
(11 points) and body (31 points) DCC keypoint heads with in-frame prediction and, in stage 2,
the mask/depth branch. Detections are matched to the ground truth by Hungarian matching on masks.

Keypoints use the unified coordinate system (x, y, z) as in RTMW3D, relative to the expanded
reference bbox:
- x: image X direction, [0, 1] inside the bbox
- y: depth, center = 0.5
- z: image Y direction (up is positive), [0, 1] inside the bbox
Margin bins extend x/z to [-BIN_MARGIN, 1 + BIN_MARGIN], so points slightly outside the bbox
(or the frame) can still be represented.

Two-stage recipe (--epochs is the total epoch count; a resumed run continues at the
checkpoint's epoch + 1):
    # Stage 1: keypoints (the mask/depth branch is not created)
    uv run python tools/train.py --epochs 100 --batch-size 48 --amp
    # Stage 2: fine-tune the best stage-1 checkpoint with the mask/depth branch
    uv run python tools/train.py --resume best --mask-weight 4.0 --epochs <total> --amp

Main options:
    --variant tiny|s|both   RTMDet-Ins variant(s); 'both' trains the two models on the same batches
    --resume best|final     continue from {work_dir}/{name}_model_{variant}.pth
    --no-multiscale         train at --img-size only instead of random 256-640 resolutions
    --experiment            allow training from an untagged or modified commit

Release models are trained from a clean commit that carries its version tag
(`git describe --tags --dirty` must be exactly vX.Y.Z); otherwise the script refuses to start
unless --experiment is given.
"""

import argparse
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the project root to sys.path (for configs etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Reduce CUDA allocator fragmentation (must be set before importing torch).
# Multi-scale training (variable 256-640 shapes) fragments the caching allocator steadily within
# an epoch; once dedicated VRAM is used up, Windows (WDDM) spills to shared memory and training
# slows down 10-20x. expandable_segments prevents the fragmentation.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rtmdet_fusion.utils.device import get_device, setup_device_optimizations


# ============================================================
# Random seed
# ============================================================

RANDOM_SEED = 42


def setup_random_seed():
    """Seed the random number generators.

    Makes data augmentation reproducible while favoring speed.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # Speed over determinism (cuDNN deterministic mode stays off)
    torch.backends.cudnn.benchmark = True

DEVICE_TYPE = get_device()

if DEVICE_TYPE == 'mps':
    from rtmdet_fusion.utils.mps_workaround import patch_nms
    patch_nms()

setup_device_optimizations(DEVICE_TYPE)

from mmengine.config import ConfigDict
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from mmdet.structures import DetDataSample

from configs.default import (
    NUM_KEYPOINTS, MAX_INSTANCES, IMG_SIZE,
    PERSON_CLASS_ID, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    GRID_SIZE, NUM_BINS, MULTISCALE_SIZES,
    WARMUP_EPOCHS, WEIGHT_DECAY,
    # Body keypoints / off-frame handling
    NUM_BODY_KEYPOINTS, NUM_BINS_BODY, BIN_MARGIN,
    BODY_BBOX_RANGE_SCALE, BODY_BONE_LINKS,
    OOF_COORD_WEIGHT, TRAIN_INS_SCORE_THR,
    # Paths
    TRAIN_ANN_PATH, VAL_ANN_PATH, TRAIN_IMG_DIR, VAL_IMG_DIR,
    WORK_DIR,
    # Matching thresholds
    MATCHING_IOU_THRESHOLD, MATCHING_SCORE_THRESHOLD, MATCHING_MIN_COVERAGE,
    # RTMDet-Ins variants
    get_rtmdet_config, get_rtmdet_weights, get_rtmdet_in_channels,
)

from rtmdet_fusion.models.fusion_model import (
    BboxFeatureExtractor, DCCHead, HeadBboxHead, MaskDepthBranch
)
from rtmdet_fusion.models.encoders import HybridEncoder
from rtmdet_fusion.models.losses import (
    SimCCLoss, HeadBboxIoULoss, HeadBboxL1Loss, BoneLoss, BoneDirectionLoss,
    CoordL2Loss, ImageCoordL2Loss, BboxJitterConsistencyLoss, VisibilityBCELoss,
    InFrameBCELoss, MaskBCEDiceLoss, MaskedDepthL1Loss,
)
from rtmdet_fusion.utils.bbox import expand_bbox
from rtmdet_fusion.utils.ema import ModelEMA
from rtmdet_fusion.utils.versioning import get_code_version, is_release_version
from rtmdet_fusion.datasets.keypoint_dataset import KeypointDataset, collate_fn

# Version of the training code (git describe --tags --dirty), embedded in checkpoints
CODE_VERSION = get_code_version()


# ============================================================
# Per-variant training state
# ============================================================

@dataclass
class VariantContext:
    """Training state of one RTMDet-Ins variant."""
    name: str  # 'tiny' or 's'
    model: nn.Module = None
    optimizer: torch.optim.Optimizer = None
    warmup_scheduler: Any = None
    plateau_scheduler: Any = None  # cosine or plateau scheduler (see scheduler_type)
    scheduler_type: str = 'cosine'
    scaler: torch.amp.GradScaler = None
    best_val_loss: float = float('inf')

    # Head losses (None = disabled; kept per variant)
    simcc_loss_fn: nn.Module = None
    head_bbox_loss_fn: nn.Module = None
    head_bbox_l1_loss_fn: nn.Module = None
    bone_loss_fn: nn.Module = None
    bone_direction_loss_fn: nn.Module = None
    coord_loss_fn: nn.Module = None
    image_coord_loss_fn: nn.Module = None
    consistency_loss_fn: nn.Module = None
    visibility_loss_fn: nn.Module = None

    # Body losses
    body_bbox_loss_fn: nn.Module = None
    body_bbox_l1_loss_fn: nn.Module = None
    body_simcc_loss_fn: nn.Module = None
    body_bone_loss_fn: nn.Module = None
    body_coord_loss_fn: nn.Module = None
    body_visibility_loss_fn: nn.Module = None
    body_bone_direction_loss_fn: nn.Module = None
    body_image_coord_loss_fn: nn.Module = None
    body_consistency_loss_fn: nn.Module = None

    # Stage 2: mask/depth losses
    mask_loss_fn: nn.Module = None
    depth_loss_fn: nn.Module = None
    # In-frame BCE (shared by head and body)
    in_frame_loss_fn: nn.Module = None

    # Exponential moving average of the weights (used for validation and saving)
    ema: Any = None  # ModelEMA or None


# ============================================================
# Multi-scale training
# ============================================================

def sample_multiscale_size(sizes: List[int] = MULTISCALE_SIZES) -> int:
    """Pick a random training resolution."""
    return random.choice(sizes)


def resize_batch(images: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize a batch of images to target_size x target_size."""
    return F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)


# ============================================================
# Training model
# ============================================================

class RTMDetFusionForTraining(nn.Module):
    """RTMDet-Fusion for training.

    Holds the frozen RTMDet-Ins model and the trainable modules. The DCC heads output unified
    (x, y, z) coordinates.
    """

    def __init__(self, rtmdet_cfg: ConfigDict, in_channels: int = 128,
                 with_mask_branch: bool = False):
        """
        Args:
            rtmdet_cfg: RTMDet-Ins config
            in_channels: RTMDet-Ins backbone output channels (tiny: 96, s: 128)
            with_mask_branch: create the mask/depth branch (stage 2)
        """
        super().__init__()

        self.rtmdet = MODELS.build(rtmdet_cfg)

        # Backbone output channels (C3, C4, C5) = in_channels x [1, 2, 4]
        backbone_channels = [in_channels, in_channels * 2, in_channels * 4]

        self.hybrid_encoder = HybridEncoder(
            in_channels=backbone_channels,
            hidden_dim=256,
            out_channels=256,
            num_csp_blocks=3,
            widen_factor=0.5
        )

        # Keypoint modules (batched)
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

        # Head DCCHead (unified coordinates; margin bins also cover points outside the bbox).
        # target_sigma=0.03: a Gaussian about 4 bins wide at 64 bins
        self.dcc_head = DCCHead(
            in_channels=256,
            num_keypoints=NUM_KEYPOINTS,
            num_bins=NUM_BINS,
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            target_sigma=0.03,
            bin_margin=BIN_MARGIN
        )

        # Body reference bbox head (HeadBboxHead with a wider prediction range)
        self.body_bbox_head = HeadBboxHead(
            in_channels=256,
            hidden_dim=256,
            num_layers=2,
            range_scale=BODY_BBOX_RANGE_SCALE
        )

        # Body keypoint head (31 points, 128 bins)
        self.body_kpt_head = DCCHead(
            in_channels=256,
            num_keypoints=NUM_BODY_KEYPOINTS,
            num_bins=NUM_BINS_BODY,
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            target_sigma=0.03,
            bin_margin=BIN_MARGIN
        )

        # Stage 2: mask/depth branch, created only when --mask-weight > 0. When None it is left
        # out of the optimizer, EMA and checkpoint, so stage-1 checkpoints stay compatible.
        self.mask_depth_branch = MaskDepthBranch() if with_mask_branch else None

    def extract_backbone_feat(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Extract CSPNeXt backbone features."""
        return self.rtmdet.backbone(images)

    def extract_pafpn_feat(self, backbone_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run the PAFPN neck."""
        return self.rtmdet.neck(backbone_feats)

    def extract_hybrid_feat(self, backbone_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run the Hybrid Encoder."""
        return self.hybrid_encoder(backbone_feats)

    def forward_head_bbox(
        self,
        hybrid_feats: List[torch.Tensor],
        ins_bboxes: torch.Tensor,
        priors: torch.Tensor,
        valid_mask: torch.Tensor,
        img_size: int = IMG_SIZE,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict head bboxes from Ins bboxes (batched)."""
        grid_features, _ = self.bbox_feat_extractor(
            hybrid_feats, ins_bboxes, priors, valid_mask, img_size,
            batch_indices=batch_indices
        )
        pred_head_bboxes = self.head_bbox_head(grid_features, ins_bboxes)
        return pred_head_bboxes, grid_features

    def forward_dcc(
        self,
        hybrid_feats: List[torch.Tensor],
        bboxes: torch.Tensor,
        priors: torch.Tensor,
        valid_mask: torch.Tensor,
        img_size: int = IMG_SIZE,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict head keypoints inside the given head bboxes (batched)."""
        grid_features, pos_encoding = self.bbox_feat_extractor(
            hybrid_feats, bboxes, priors, valid_mask, img_size,
            batch_indices=batch_indices
        )
        return self.dcc_head(grid_features, pos_encoding)

    def forward_body_kpt(
        self,
        hybrid_feats: List[torch.Tensor],
        bboxes: torch.Tensor,
        priors: torch.Tensor,
        valid_mask: torch.Tensor,
        img_size: int = IMG_SIZE,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict body keypoints inside the given body reference bboxes (batched)."""
        grid_features, pos_encoding = self.bbox_feat_extractor(
            hybrid_feats, bboxes, priors, valid_mask, img_size,
            batch_indices=batch_indices
        )
        return self.body_kpt_head(grid_features, pos_encoding)


def trainable_module_names(model: 'RTMDetFusionForTraining') -> List[str]:
    """Names of the trainable submodules (they match the checkpoint key names).

    Do not change the order: it pairs with raw_state_dicts and the EMA state.
    mask_depth_branch (stage 2) is appended only when it exists.
    """
    names = [
        'hybrid_encoder',
        'bbox_feat_extractor',
        'head_bbox_head',
        'dcc_head',
        'body_bbox_head',
        'body_kpt_head',
    ]
    if getattr(model, 'mask_depth_branch', None) is not None:
        names.append('mask_depth_branch')
    return names


def trainable_modules(model: 'RTMDetFusionForTraining') -> List[nn.Module]:
    """Trainable submodules (shared by grad clipping, EMA, optimizer and checkpoints)."""
    return [getattr(model, n) for n in trainable_module_names(model)]


# ============================================================
# Utilities
# ============================================================

def select_top_person_instances(
    bboxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    priors: torch.Tensor,
    max_instances: int = MAX_INSTANCES,
    person_class_id: int = PERSON_CLASS_ID
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Select the top-scoring person detections into fixed-size, zero-padded tensors.

    Args:
        bboxes: (N, 4) detection bboxes
        labels: (N,) detection labels (most likely class)
        scores: (N,) detection scores
        priors: (N, 4) priors
        max_instances: maximum number of instances
        person_class_id: person class ID

    Returns:
        bboxes: (max_instances, 4), zero padded
        labels: (max_instances,), zero padded
        scores: (max_instances,), zero padded
        priors: (max_instances, 4), zero padded
        valid_count: number of valid instances
    """
    device = bboxes.device
    N = bboxes.shape[0]

    # Fixed-size outputs, zero filled
    out_bboxes = torch.zeros((max_instances, 4), device=device)
    out_labels = torch.zeros((max_instances,), dtype=torch.long, device=device)
    out_scores = torch.zeros((max_instances,), device=device)
    out_priors = torch.zeros((max_instances, 4), device=device)

    if N == 0:
        return out_bboxes, out_labels, out_scores, out_priors, 0

    # Person detections only (labels hold the most likely class)
    person_mask = labels == person_class_id
    person_indices = torch.where(person_mask)[0]

    if len(person_indices) == 0:
        return out_bboxes, out_labels, out_scores, out_priors, 0

    person_bboxes = bboxes[person_indices]
    person_labels = labels[person_indices]
    person_scores = scores[person_indices]
    person_priors = priors[person_indices]

    # Sort by score
    sorted_order = person_scores.argsort(descending=True)
    person_bboxes = person_bboxes[sorted_order]
    person_labels = person_labels[sorted_order]
    person_scores = person_scores[sorted_order]
    person_priors = person_priors[sorted_order]

    # Keep the top max_instances
    valid_count = min(len(person_indices), max_instances)
    out_bboxes[:valid_count] = person_bboxes[:valid_count]
    out_labels[:valid_count] = person_labels[:valid_count]
    out_scores[:valid_count] = person_scores[:valid_count]
    out_priors[:valid_count] = person_priors[:valid_count]

    return out_bboxes, out_labels, out_scores, out_priors, valid_count


# ============================================================
# Hungarian mask matching
# ============================================================

def hungarian_mask_matching(
    gt_masks,
    ins_masks,
    device: str,
    iou_threshold: float = MATCHING_IOU_THRESHOLD,
    score_threshold: float = MATCHING_SCORE_THRESHOLD,
    min_coverage: float = MATCHING_MIN_COVERAGE,
) -> torch.Tensor:
    """Match GT masks to Ins masks with the Hungarian algorithm.

    The IoU / coverage / recall matrices are computed at once with a matmul on the device; only
    the small N_gt x N_ins cost matrix is moved to the CPU for scipy's linear_sum_assignment.
    Same algorithm as match_masks() in tools/agora/utils/matching.py.

    A pair is a match candidate if either holds:
    1. IoU >= iou_threshold
    2. coverage >= min_coverage and combined score >= score_threshold

    Args:
        gt_masks: GT masks (N_gt, H, W), torch.Tensor or numpy.ndarray
        ins_masks: Ins masks (N_ins, H, W), torch.Tensor or numpy.ndarray
        device: device of the result
        iou_threshold: IoU threshold
        score_threshold: combined score threshold
        min_coverage: minimum coverage

    Returns:
        matched_ins_indices: (N_gt,) index of the matched Ins mask per GT, -1 if unmatched
    """
    # numpy -> tensor
    if isinstance(gt_masks, np.ndarray):
        gt_masks = torch.from_numpy(gt_masks)
    if isinstance(ins_masks, np.ndarray):
        ins_masks = torch.from_numpy(ins_masks)

    n_gt = gt_masks.shape[0]
    n_ins = ins_masks.shape[0]

    if n_gt == 0:
        return torch.zeros((0,), dtype=torch.long, device=device)
    if n_ins == 0:
        return torch.full((n_gt,), -1, dtype=torch.long, device=device)

    # Move both to the compute device
    compute_device = torch.device(device) if isinstance(device, str) else device
    gt_t = gt_masks.to(compute_device, non_blocking=True)
    ins_t = ins_masks.to(compute_device, non_blocking=True)

    # Binarize and flatten
    gt_flat = (gt_t > 0).float().reshape(n_gt, -1)      # (N_gt, H*W)
    ins_flat = (ins_t > 0).float().reshape(n_ins, -1)   # (N_ins, H*W)

    # IoU / coverage / recall matrices from a single matmul
    intersection = gt_flat @ ins_flat.T                  # (N_gt, N_ins)
    gt_area = gt_flat.sum(dim=1, keepdim=True)           # (N_gt, 1)
    ins_area = ins_flat.sum(dim=1, keepdim=True).T       # (1, N_ins)
    union = gt_area + ins_area - intersection

    iou_mat = intersection / union.clamp(min=1e-8)
    cov_mat = intersection / ins_area.clamp(min=1e-8)
    rec_mat = intersection / gt_area.clamp(min=1e-8)

    # Combined score: 0.4*iou + 0.4*coverage + 0.2*gt_recall
    score_mat = iou_mat * 0.4 + cov_mat * 0.4 + rec_mat * 0.2

    # Candidates: IoU >= threshold OR (coverage >= min_coverage AND score >= threshold)
    is_candidate = (iou_mat >= iou_threshold) | (
        (cov_mat >= min_coverage) & (score_mat >= score_threshold)
    )

    # Cost: -score for candidates (minimized), 1.0 otherwise
    cost_matrix = torch.where(is_candidate, -score_mat, torch.ones_like(score_mat))

    # Only the small matrix (N_gt x N_ins, typically about 5x5) goes to the CPU
    cost_np = cost_matrix.detach().cpu().numpy().astype(np.float64)

    row_ind, col_ind = linear_sum_assignment(cost_np)

    matched_ins_indices = torch.full((n_gt,), -1, dtype=torch.long, device=device)
    for assignment_idx in range(len(row_ind)):
        gt_idx = int(row_ind[assignment_idx])
        ins_idx = int(col_ind[assignment_idx])
        # cost < 1.0 marks a candidate pair (its cost is -score)
        if cost_np[gt_idx, ins_idx] < 1.0:
            matched_ins_indices[gt_idx] = ins_idx

    return matched_ins_indices


# ============================================================
# Keypoint re-referencing (GT bbox -> any bbox)
# ============================================================

def recompute_keypoints_relative(
    gt_kpts_unified: torch.Tensor,
    gt_head_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-express GT keypoints relative to target_bboxes instead of the GT bboxes.

    No clamping: keypoints outside the margin-bin range are reported through valid_mask=False,
    and the caller drops them from the loss via target_weight.

    Args:
        gt_kpts_unified: (N, K, 3) [rel_x, rel_y, rel_z] relative to the GT bboxes
        gt_head_bboxes: (N, 4) GT bboxes [x1, y1, x2, y2]
        target_bboxes: (N, 4) target bboxes [x1, y1, x2, y2] (GT or predicted)

    Returns:
        new_kpts: (N, K, 3) [rel_x, rel_y, rel_z] relative to target_bboxes (unclamped)
        valid_mask: (N, K) True where rel_x and rel_z lie in [-BIN_MARGIN, 1 + BIN_MARGIN]
    """
    # GT bbox
    gt_x1 = gt_head_bboxes[:, 0:1]  # (N, 1)
    gt_y1 = gt_head_bboxes[:, 1:2]
    gt_x2 = gt_head_bboxes[:, 2:3]
    gt_y2 = gt_head_bboxes[:, 3:4]
    gt_w = gt_x2 - gt_x1
    gt_h = gt_y2 - gt_y1
    gt_bbox_size = (gt_w + gt_h) / 2

    # GT-relative -> absolute coordinates
    rel_x = gt_kpts_unified[:, :, 0]  # (N, K)
    rel_y = gt_kpts_unified[:, :, 1]  # (N, K)
    rel_z = gt_kpts_unified[:, :, 2]  # (N, K)

    abs_x = rel_x * gt_w + gt_x1                             # image X
    abs_depth = (rel_y - 0.5) * gt_bbox_size                  # depth (center = 0)
    abs_img_y = (1 - rel_z) * gt_h + gt_y1                    # image Y

    # Target bbox
    tgt_x1 = target_bboxes[:, 0:1]
    tgt_y1 = target_bboxes[:, 1:2]
    tgt_x2 = target_bboxes[:, 2:3]
    tgt_y2 = target_bboxes[:, 3:4]
    tgt_w = tgt_x2 - tgt_x1
    tgt_h = tgt_y2 - tgt_y1
    tgt_bbox_size = (tgt_w + tgt_h) / 2

    # Absolute -> target-relative coordinates
    new_rel_x = (abs_x - tgt_x1) / tgt_w.clamp(min=1e-6)
    new_rel_y = abs_depth / tgt_bbox_size.clamp(min=1e-6) + 0.5
    new_rel_z = 1 - (abs_img_y - tgt_y1) / tgt_h.clamp(min=1e-6)

    # Flag keypoints outside the margin-bin range [-BIN_MARGIN, 1 + BIN_MARGIN] (inside it SimCC
    # can represent them even outside the bbox, so they stay supervised). rel_y is depth and
    # depends on the bbox size, so it is not masked here.
    lo, hi = -BIN_MARGIN, 1.0 + BIN_MARGIN
    in_x = (new_rel_x >= lo) & (new_rel_x <= hi)
    in_z = (new_rel_z >= lo) & (new_rel_z <= hi)
    valid_mask = in_x & in_z  # (N, K)

    new_kpts = torch.stack([new_rel_x, new_rel_y, new_rel_z], dim=-1)
    return new_kpts, valid_mask


# ============================================================
# Bbox jittering
# ============================================================

def apply_bbox_jittering(
    bboxes: torch.Tensor,
    gt_keypoints: torch.Tensor,
    gt_vis: torch.Tensor,
    max_shift_ratio: float = 0.25,
    max_scale_ratio: float = 0.15,
    jitter_prob: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly perturb bboxes and recompute the GT keypoints relative to them.

    Makes the keypoint heads follow visual features rather than the bbox position. Shift and
    scale are sampled independently for x and y. No clamping: keypoints that leave the
    margin-bin range get valid_mask=0.

    Args:
        bboxes: (N, 4) [x1, y1, x2, y2] bboxes (absolute coordinates)
        gt_keypoints: (N, K, 3) [rel_x, rel_y, rel_z] relative to bboxes
        gt_vis: (N, K) visibility (currently unused)
        max_shift_ratio: maximum shift as a fraction of the bbox width/height
        max_scale_ratio: maximum scale change (±)
        jitter_prob: probability of jittering a sample

    Returns:
        jittered_bboxes: (N, 4) perturbed bboxes
        new_gt_keypoints: (N, K, 3) keypoints relative to the perturbed bboxes (unclamped)
        valid_mask: (N, K) 0 for keypoints outside the margin-bin range
    """
    N, K, _ = gt_keypoints.shape
    device = bboxes.device

    # Samples to jitter
    apply_mask = torch.rand(N, device=device) < jitter_prob

    # Original bbox
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Random shift (independent x/y, ±max_shift_ratio of the width/height)
    shift_x = (torch.rand(N, device=device) * 2 - 1) * max_shift_ratio * w
    shift_y = (torch.rand(N, device=device) * 2 - 1) * max_shift_ratio * h

    # Random scale (independent x/y, 1 ± max_scale_ratio)
    scale_x = 1.0 + (torch.rand(N, device=device) * 2 - 1) * max_scale_ratio
    scale_y = 1.0 + (torch.rand(N, device=device) * 2 - 1) * max_scale_ratio

    # New center and size
    new_cx = cx + shift_x
    new_cy = cy + shift_y
    new_w = w * scale_x
    new_h = h * scale_y

    # New corners
    new_x1 = new_cx - new_w / 2
    new_y1 = new_cy - new_h / 2
    new_x2 = new_cx + new_w / 2
    new_y2 = new_cy + new_h / 2

    # Keep the original bbox for samples that are not jittered
    jittered_x1 = torch.where(apply_mask, new_x1, x1)
    jittered_y1 = torch.where(apply_mask, new_y1, y1)
    jittered_x2 = torch.where(apply_mask, new_x2, x2)
    jittered_y2 = torch.where(apply_mask, new_y2, y2)

    jittered_bboxes = torch.stack([jittered_x1, jittered_y1, jittered_x2, jittered_y2], dim=1)

    # Absolute GT keypoint coordinates (original bbox)
    rel_x = gt_keypoints[:, :, 0]  # (N, K)
    rel_y = gt_keypoints[:, :, 1]  # (N, K) depth
    rel_z = gt_keypoints[:, :, 2]  # (N, K)

    abs_x = rel_x * w.unsqueeze(1) + x1.unsqueeze(1)  # (N, K)
    abs_img_y = (1 - rel_z) * h.unsqueeze(1) + y1.unsqueeze(1)  # (N, K)
    bbox_size = (w + h) / 2
    abs_depth = (rel_y - 0.5) * bbox_size.unsqueeze(1)  # (N, K)

    # Relative coordinates w.r.t. the jittered bbox
    jittered_w = jittered_x2 - jittered_x1
    jittered_h = jittered_y2 - jittered_y1
    jittered_bbox_size = (jittered_w + jittered_h) / 2

    new_rel_x = (abs_x - jittered_x1.unsqueeze(1)) / jittered_w.unsqueeze(1).clamp(min=1e-6)
    new_rel_z = 1 - (abs_img_y - jittered_y1.unsqueeze(1)) / jittered_h.unsqueeze(1).clamp(min=1e-6)
    new_rel_y = abs_depth / jittered_bbox_size.unsqueeze(1).clamp(min=1e-6) + 0.5

    # No clamping; keypoints outside the margin-bin range are dropped through valid_mask
    lo, hi = -BIN_MARGIN, 1.0 + BIN_MARGIN
    in_x = (new_rel_x >= lo) & (new_rel_x <= hi)
    in_z = (new_rel_z >= lo) & (new_rel_z <= hi)
    jitter_valid = in_x & in_z  # (N, K)

    new_gt_keypoints = torch.stack([new_rel_x, new_rel_y, new_rel_z], dim=-1)

    # Samples that are not jittered keep their keypoints and an all-True valid mask
    apply_mask_expanded = apply_mask.view(N, 1, 1).expand(-1, K, 3)
    final_gt_keypoints = torch.where(apply_mask_expanded, new_gt_keypoints, gt_keypoints)

    apply_mask_kpt = apply_mask.view(N, 1).expand(-1, K)  # (N, K)
    valid_mask = torch.where(
        apply_mask_kpt,
        jitter_valid,
        torch.ones_like(jitter_valid),
    )

    return jittered_bboxes, final_gt_keypoints, valid_mask


# ============================================================
# Training loop
# ============================================================


def _worker_init_fn(worker_id: int):
    """Prevent periodic GC stalls in DataLoader workers.

    Each worker holds about 870k annotation dicts, and full passes of the generational GC cause
    periodic stalls of tens of seconds. gc.freeze() removes the loaded objects from GC tracking.

    NOTE: keep this at module level. With the Windows spawn start method it is pickled, and a
    local function in main() fails to start with AttributeError.
    """
    import gc
    gc.collect()
    gc.freeze()


def compute_gt_bbox_prob(epoch: int, warmup_epochs: int = 3, steepness: float = 1.0) -> float:
    """Scheduled sampling: probability of using the GT bbox instead of the predicted one.

    Smooth sigmoid transition:
    - epoch < warmup_epochs - 2: almost always the GT bbox
    - epoch = warmup_epochs: 50%
    - epoch > warmup_epochs + 2: almost always the predicted bbox

    Args:
        epoch: current epoch
        warmup_epochs: center of the transition (50% at this epoch)
        steepness: width of the transition (larger = more gradual)

    Returns:
        probability of using the GT bbox, in [0, 1]
    """
    import math
    return 1.0 / (1.0 + math.exp((epoch - warmup_epochs) / steepness))


def train_one_epoch_dual(
    contexts: List[VariantContext],
    dataloader: DataLoader,
    device: str,
    epoch: int,
    device_type: str = 'cuda',
    jitter_prob: float = 0.5,
    jitter_shift: float = 0.15,
    jitter_scale: float = 0.15,
    use_multiscale: bool = True,
    base_img_size: int = IMG_SIZE
) -> Dict[str, Dict[str, float]]:
    """Train one epoch.

    All variants in `contexts` are trained on the same batches, one after another (sequential
    to save GPU memory).

    Args:
        contexts: variant contexts
        dataloader: shared data loader
        device: device
        epoch: current epoch
        device_type: device type
        jitter_prob: bbox jittering probability
        jitter_shift: bbox jittering maximum shift ratio
        jitter_scale: bbox jittering maximum scale change ratio
        use_multiscale: use multi-scale training
        base_img_size: base image size

    Returns:
        metrics per variant name
    """
    # Trainable modules in train mode, the frozen detector in eval mode
    for ctx in contexts:
        for mod in trainable_modules(ctx.model):
            mod.train()
        ctx.model.rtmdet.eval()

    # Per-variant metrics
    metrics = {}
    for ctx in contexts:
        metrics[ctx.name] = {
            'total_loss': 0.0,
            'total_head_bbox_loss': 0.0,
            'total_head_bbox_l1_loss': 0.0,
            'total_simcc_loss': 0.0,
            'total_simcc_x': 0.0,
            'total_simcc_y': 0.0,
            'total_simcc_z': 0.0,
            'total_bone_loss': 0.0,
            'total_bone_direction_loss': 0.0,
            'total_coord_loss': 0.0,
            'total_image_coord_loss': 0.0,
            'total_consistency_loss': 0.0,
            'total_visibility_loss': 0.0,
            'total_matched': 0,
            'num_batches': 0,
            'hungarian_match_count': 0,
            # Body
            'total_body_bbox_loss': 0.0,
            'total_body_simcc_loss': 0.0,
            'total_body_bone_loss': 0.0,
            'total_body_coord_loss': 0.0,
            'total_body_visibility_loss': 0.0,
            'total_in_frame_loss': 0.0,
            'total_body_matched': 0,
            # Stage 2: mask/depth
            'total_mask_loss': 0.0,
            'total_depth_loss': 0.0,
        }

    current_img_size = base_img_size
    variant_names = '/'.join([ctx.name for ctx in contexts])

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{variant_names}]")
    for batch in pbar:
        # Move the batch to the device (shared by all variants)
        images_original = batch['images'].to(device, non_blocking=True)
        gt_keypoints_unified_list = batch['gt_keypoints_unified']
        gt_visibilities_list = batch['gt_visibilities']
        gt_head_bboxes_list = batch['gt_head_bboxes']
        gt_masks_list = batch['gt_masks']
        gt_ignore_flags_list = batch['gt_ignore_flags']
        # Body
        gt_body_ref_bboxes_list = batch['gt_body_ref_bboxes']
        gt_body_keypoints_list = batch['gt_body_keypoints_unified']
        gt_body_visibilities_list = batch['gt_body_visibilities']
        gt_body_ignore_flags_list = batch['gt_body_ignore_flags']
        img_shapes = batch['img_shapes']
        # Stage 2: mask/depth teachers (present only with with_mask_teachers)
        _none_list = [None] * len(img_shapes)
        gt_mask_teachers_list = batch.get('gt_mask_teachers', _none_list)
        gt_mask_cv_list = batch.get('gt_mask_channel_valid', _none_list)
        gt_depth_teachers_list = batch.get('gt_depth_teachers', _none_list)
        gt_depth_valid_list = batch.get('gt_depth_valid', _none_list)

        B = images_original.shape[0]

        # Multi-scale resize (all variants use the same scale)
        if use_multiscale:
            current_img_size = sample_multiscale_size()
            images = resize_batch(images_original, current_img_size)
            scale_factor = current_img_size / base_img_size
        else:
            current_img_size = base_img_size
            images = images_original
            scale_factor = 1.0

        batch_data_samples = []
        for i in range(B):
            data_sample = DetDataSample()
            if use_multiscale:
                data_sample.set_metainfo({
                    'img_shape': (current_img_size, current_img_size),
                    'ori_shape': img_shapes[i],
                    'scale_factor': (scale_factor, scale_factor),
                    'pad_shape': (current_img_size, current_img_size),
                })
            else:
                data_sample.set_metainfo({
                    'img_shape': img_shapes[i],
                    'ori_shape': img_shapes[i],
                    'scale_factor': (1.0, 1.0),
                    'pad_shape': img_shapes[i],
                })
            batch_data_samples.append(data_sample)

        # Process each variant in turn
        batch_losses = {}
        for ctx in contexts:
            model = ctx.model
            optimizer = ctx.optimizer
            scaler = ctx.scaler
            use_amp = scaler is not None
            m = metrics[ctx.name]

            # RTMDet-Ins inference (per variant)
            with torch.no_grad():
                backbone_feats = model.extract_backbone_feat(images)
                pafpn_feats = model.extract_pafpn_feat(backbone_feats)

                try:
                    ins_results_list = model.rtmdet.bbox_head.predict(
                        pafpn_feats, batch_data_samples, rescale=False
                    )
                except RuntimeError as e:
                    if device_type == 'mps' and "Placeholder tensor is empty" in str(e):
                        continue
                    raise

            with torch.amp.autocast('cuda', enabled=use_amp):
                hybrid_feats = model.extract_hybrid_feat(backbone_feats)

            # === Step 1: Hungarian matching per image ===
            # Matches of all images are collected first so the heads run in one batched forward
            collected_ins_bboxes = []
            collected_priors = []
            collected_gt_head_bboxes = []
            collected_gt_kpts = []
            collected_gt_vis = []
            collected_target_weights = []
            collected_batch_indices = []  # image index of each instance
            # Body
            collected_gt_body_bboxes = []
            collected_gt_body_kpts = []
            collected_gt_body_vis = []
            collected_body_valid = []  # per instance: body losses apply
            collected_head_valid = []  # per instance: head losses apply
            # Stage 2: mask/depth teachers
            mask_enabled = ctx.mask_loss_fn is not None
            collected_mask_teachers = []
            collected_mask_cv = []
            collected_depth_teachers = []
            collected_depth_valid = []

            for b in range(B):
                ins_results = ins_results_list[b]
                gt_kpts_unified = gt_keypoints_unified_list[b].to(device, non_blocking=True)
                gt_vis = gt_visibilities_list[b].to(device, non_blocking=True)
                gt_head_bboxes = gt_head_bboxes_list[b].to(device, non_blocking=True)
                gt_masks = gt_masks_list[b].to(device, non_blocking=True)
                gt_ignore_flags = gt_ignore_flags_list[b].to(device, non_blocking=True)
                gt_body_bboxes = gt_body_ref_bboxes_list[b].to(device, non_blocking=True)
                gt_body_kpts = gt_body_keypoints_list[b].to(device, non_blocking=True)
                gt_body_vis = gt_body_visibilities_list[b].to(device, non_blocking=True)
                gt_body_ignore = gt_body_ignore_flags_list[b].to(device, non_blocking=True)

                if len(ins_results.bboxes) == 0 or len(gt_kpts_unified) == 0:
                    continue
                if not hasattr(ins_results, 'priors') or ins_results.priors is None:
                    continue
                if not hasattr(ins_results, 'masks') or ins_results.masks is None:
                    continue

                bboxes_det, labels, scores_det, priors_det, valid_count = select_top_person_instances(
                    ins_results.bboxes, ins_results.labels, ins_results.scores,
                    ins_results.priors, max_instances=MAX_INSTANCES
                )
                if valid_count == 0:
                    continue

                bboxes_det = bboxes_det[:valid_count]
                priors_det = priors_det[:valid_count]
                ins_masks = ins_results.masks[:valid_count]

                if use_multiscale and scale_factor != 1.0:
                    gt_masks_resized = F.interpolate(
                        gt_masks.float().unsqueeze(0),
                        size=(current_img_size, current_img_size), mode='nearest'
                    ).squeeze(0)
                    ins_masks_resized = F.interpolate(
                        ins_masks.float().unsqueeze(0),
                        size=(current_img_size, current_img_size), mode='nearest'
                    ).squeeze(0)
                    matched_ins_indices = hungarian_mask_matching(
                        gt_masks_resized, ins_masks_resized, device
                    )
                    gt_head_bboxes_s = gt_head_bboxes * scale_factor
                    gt_body_bboxes_s = gt_body_bboxes * scale_factor
                else:
                    matched_ins_indices = hungarian_mask_matching(
                        gt_masks, ins_masks, device
                    )
                    gt_head_bboxes_s = gt_head_bboxes
                    gt_body_bboxes_s = gt_body_bboxes

                m['hungarian_match_count'] += 1

                # Keep instances whose head or body annotation is usable
                valid_gt_mask = (matched_ins_indices >= 0) & (
                    (~gt_ignore_flags) | (~gt_body_ignore)
                )
                if not valid_gt_mask.any():
                    continue

                valid_gt_indices = torch.where(valid_gt_mask)[0]
                matched_ins_idx = matched_ins_indices[valid_gt_mask]

                collected_ins_bboxes.append(bboxes_det[matched_ins_idx])
                collected_priors.append(priors_det[matched_ins_idx])
                collected_gt_head_bboxes.append(gt_head_bboxes_s[valid_gt_indices])
                collected_gt_kpts.append(gt_kpts_unified[valid_gt_indices])
                collected_gt_vis.append(gt_vis[valid_gt_indices])
                # Off-frame (v=0) keypoints are supervised with a lower coordinate weight;
                # instances whose head annotation is ignored get head weight 0
                sel_vis = gt_vis[valid_gt_indices]
                head_w = torch.where(
                    sel_vis > 0,
                    torch.ones_like(sel_vis),
                    torch.full_like(sel_vis, OOF_COORD_WEIGHT),
                )
                head_w = head_w * (~gt_ignore_flags[valid_gt_indices]).float().unsqueeze(1)
                collected_target_weights.append(head_w)
                collected_batch_indices.append(
                    torch.full((len(valid_gt_indices),), b, dtype=torch.long, device=device)
                )
                # Body
                collected_gt_body_bboxes.append(gt_body_bboxes_s[valid_gt_indices])
                collected_gt_body_kpts.append(gt_body_kpts[valid_gt_indices])
                collected_gt_body_vis.append(gt_body_vis[valid_gt_indices])
                collected_body_valid.append(~gt_body_ignore[valid_gt_indices])
                collected_head_valid.append(~gt_ignore_flags[valid_gt_indices])
                # Stage 2: mask/depth teachers (same order as the annotations)
                if mask_enabled and gt_mask_teachers_list[b] is not None:
                    mt = gt_mask_teachers_list[b].to(device, non_blocking=True)
                    collected_mask_teachers.append(mt[valid_gt_indices])
                    collected_mask_cv.append(
                        gt_mask_cv_list[b].to(device, non_blocking=True)[valid_gt_indices]
                    )
                    collected_depth_teachers.append(
                        gt_depth_teachers_list[b].to(device, non_blocking=True)[valid_gt_indices]
                    )
                    collected_depth_valid.append(
                        gt_depth_valid_list[b].to(device, non_blocking=True)[valid_gt_indices]
                    )

            # Nothing matched: no loss for this variant
            batch_loss = torch.tensor(0.0, device=device)
            batch_count = 0

            if len(collected_ins_bboxes) == 0:
                batch_losses[ctx.name] = 0.0
                # The optimizer step below is skipped because batch_count == 0
            else:
                # === Step 2: concatenate all matches ===
                cat_ins_bboxes = torch.cat(collected_ins_bboxes)
                cat_priors = torch.cat(collected_priors)
                cat_gt_head_bboxes = torch.cat(collected_gt_head_bboxes)
                cat_gt_kpts = torch.cat(collected_gt_kpts)
                cat_gt_vis = torch.cat(collected_gt_vis)
                cat_target_weights = torch.cat(collected_target_weights)
                cat_batch_indices = torch.cat(collected_batch_indices)
                # Body
                cat_gt_body_bboxes = torch.cat(collected_gt_body_bboxes)
                cat_gt_body_kpts = torch.cat(collected_gt_body_kpts)
                cat_gt_body_vis = torch.cat(collected_gt_body_vis)
                cat_body_valid = torch.cat(collected_body_valid)
                cat_head_valid = torch.cat(collected_head_valid)
                # Stage 2: mask/depth teachers
                cat_mask_teachers = None
                if mask_enabled and collected_mask_teachers:
                    cat_mask_teachers = torch.cat(collected_mask_teachers)
                    cat_mask_cv = torch.cat(collected_mask_cv)
                    cat_depth_teachers = torch.cat(collected_depth_teachers)
                    cat_depth_valid = torch.cat(collected_depth_valid)
                total_N = cat_ins_bboxes.shape[0]
                cat_valid_mask = torch.ones(total_N, dtype=torch.bool, device=device)

                # === Step 3: batched forward and losses ===
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Head bbox prediction for all instances (keeps the Ins grid features)
                    pred_head_bboxes, grid_features_ins = model.forward_head_bbox(
                        hybrid_feats, cat_ins_bboxes, cat_priors, cat_valid_mask,
                        current_img_size, batch_indices=cat_batch_indices
                    )

                    # Head bbox loss only for instances with a usable head annotation
                    head_bbox_loss = torch.tensor(0.0, device=device)
                    head_bbox_l1_loss = torch.tensor(0.0, device=device)
                    if cat_head_valid.any():
                        hsel = cat_head_valid
                        head_bbox_loss = ctx.head_bbox_loss_fn(
                            pred_head_bboxes[hsel], cat_gt_head_bboxes[hsel]
                        )
                        if ctx.head_bbox_l1_loss_fn is not None:
                            head_bbox_l1_loss = ctx.head_bbox_l1_loss_fn(
                                pred_head_bboxes[hsel], cat_gt_head_bboxes[hsel]
                            )
                            head_bbox_loss = head_bbox_loss + head_bbox_l1_loss

                    # Scheduled sampling (all instances at once)
                    gt_prob = compute_gt_bbox_prob(epoch, warmup_epochs=3, steepness=1.0)
                    use_gt_bbox = torch.rand(total_N, device=device) < gt_prob

                    gt_expanded = expand_bbox(cat_gt_head_bboxes)
                    pred_expanded = expand_bbox(pred_head_bboxes.detach())

                    selected_bboxes = torch.where(
                        use_gt_bbox.unsqueeze(1).expand(-1, 4),
                        gt_expanded, pred_expanded
                    )

                    # Re-reference the keypoints and jitter (all instances at once)
                    selected_gt_kpts, recompute_valid = recompute_keypoints_relative(
                        cat_gt_kpts, cat_gt_head_bboxes, selected_bboxes
                    )

                    jittered_bboxes, jittered_gt_kpts, jitter_valid = apply_bbox_jittering(
                        selected_bboxes, selected_gt_kpts, cat_gt_vis,
                        max_shift_ratio=jitter_shift,
                        max_scale_ratio=jitter_scale,
                        jitter_prob=jitter_prob
                    )
                    # Drop keypoints that left the bbox range via target_weight
                    cat_target_weights = cat_target_weights * recompute_valid.float() * jitter_valid.float()

                    # Keypoint prediction (all instances at once)
                    dcc_output = model.forward_dcc(
                        hybrid_feats, jittered_bboxes, cat_priors, cat_valid_mask,
                        current_img_size, batch_indices=cat_batch_indices
                    )

                    heatmap_x = dcc_output['heatmap_x']
                    heatmap_y = dcc_output['heatmap_y']
                    heatmap_z = dcc_output['heatmap_z']
                    visibility_logits = dcc_output['visibility_logits']

                    target_x, target_y, target_z = model.dcc_head.generate_target(
                        jittered_gt_kpts, cat_target_weights
                    )

                    simcc_loss, simcc_x, simcc_y, simcc_z = ctx.simcc_loss_fn(
                        heatmap_x, heatmap_y, heatmap_z,
                        target_x, target_y, target_z,
                        cat_target_weights,
                        return_per_axis=True
                    )

                    loss = head_bbox_loss + simcc_loss

                    bone_loss = torch.tensor(0.0, device=device)
                    bone_direction_loss = torch.tensor(0.0, device=device)
                    coord_loss = torch.tensor(0.0, device=device)
                    image_coord_loss = torch.tensor(0.0, device=device)
                    pred_coords = None
                    needs_decode = (
                        ctx.bone_loss_fn is not None
                        or ctx.bone_direction_loss_fn is not None
                        or ctx.coord_loss_fn is not None
                        or ctx.image_coord_loss_fn is not None
                    )
                    if needs_decode:
                        pred_coords, _, _ = model.dcc_head.decode_relative(
                            heatmap_x, heatmap_y, heatmap_z
                        )
                        if ctx.bone_loss_fn is not None:
                            bone_loss = ctx.bone_loss_fn(
                                pred_coords, jittered_gt_kpts, cat_target_weights
                            )
                            loss = loss + bone_loss
                        if ctx.bone_direction_loss_fn is not None:
                            bone_direction_loss = ctx.bone_direction_loss_fn(
                                pred_coords, jittered_gt_kpts, cat_target_weights
                            )
                            loss = loss + bone_direction_loss
                        if ctx.coord_loss_fn is not None:
                            coord_loss = ctx.coord_loss_fn(
                                pred_coords, jittered_gt_kpts, cat_target_weights
                            )
                            loss = loss + coord_loss
                        if ctx.image_coord_loss_fn is not None:
                            # Image-coordinate L2 (pred and target both mapped through the bbox)
                            image_coord_loss = ctx.image_coord_loss_fn(
                                pred_coords, jittered_gt_kpts,
                                jittered_bboxes, jittered_bboxes,
                                cat_target_weights,
                            )
                            loss = loss + image_coord_loss

                    # Bbox-jitter consistency: run the head again on a second jitter of the same
                    # instances and enforce agreement in image coordinates (needs pred_coords).
                    consistency_loss = torch.tensor(0.0, device=device)
                    if ctx.consistency_loss_fn is not None and pred_coords is not None:
                        jittered_bboxes_b, _, _ = apply_bbox_jittering(
                            selected_bboxes, selected_gt_kpts, cat_gt_vis,
                            max_shift_ratio=jitter_shift,
                            max_scale_ratio=jitter_scale,
                            jitter_prob=jitter_prob,
                        )
                        dcc_output_b = model.forward_dcc(
                            hybrid_feats, jittered_bboxes_b, cat_priors, cat_valid_mask,
                            current_img_size, batch_indices=cat_batch_indices,
                        )
                        pred_coords_b, _, _ = model.dcc_head.decode_relative(
                            dcc_output_b['heatmap_x'],
                            dcc_output_b['heatmap_y'],
                            dcc_output_b['heatmap_z'],
                        )
                        consistency_loss = ctx.consistency_loss_fn(
                            pred_coords, pred_coords_b,
                            jittered_bboxes, jittered_bboxes_b,
                            cat_target_weights,
                        )
                        loss = loss + consistency_loss

                    visibility_loss = torch.tensor(0.0, device=device)
                    if ctx.visibility_loss_fn is not None:
                        # Exclude instances whose head annotation is ignored
                        vis_include = (cat_gt_vis > 0) & cat_head_valid.unsqueeze(1)
                        visibility_loss = ctx.visibility_loss_fn(
                            visibility_logits, cat_gt_vis, include_mask=vis_include
                        )
                        loss = loss + visibility_loss

                    # In-frame prediction (head)
                    in_frame_loss = torch.tensor(0.0, device=device)
                    if ctx.in_frame_loss_fn is not None:
                        in_frame_loss = ctx.in_frame_loss_fn(
                            dcc_output['in_frame_logits'], cat_gt_vis,
                            include_mask=cat_head_valid.unsqueeze(1).expand_as(cat_gt_vis),
                        )
                        loss = loss + in_frame_loss

                    # ==================================================
                    # Body (reference bbox + 31 keypoints)
                    # ==================================================
                    body_bbox_loss = torch.tensor(0.0, device=device)
                    body_simcc_loss = torch.tensor(0.0, device=device)
                    body_bone_loss = torch.tensor(0.0, device=device)
                    body_coord_loss = torch.tensor(0.0, device=device)
                    body_visibility_loss = torch.tensor(0.0, device=device)
                    n_body = 0
                    if ctx.body_simcc_loss_fn is not None and cat_body_valid.any():
                        bsel = cat_body_valid
                        n_body = int(bsel.sum().item())

                        # Body reference bbox prediction (shares the Ins grid features)
                        pred_body_bboxes = model.body_bbox_head(
                            grid_features_ins, cat_ins_bboxes
                        )
                        pb = pred_body_bboxes[bsel]
                        gb = cat_gt_body_bboxes[bsel]

                        body_bbox_loss = ctx.body_bbox_loss_fn(pb, gb)
                        if ctx.body_bbox_l1_loss_fn is not None:
                            body_bbox_loss = body_bbox_loss + ctx.body_bbox_l1_loss_fn(pb, gb)

                        # Scheduled sampling (same schedule as the head)
                        use_gt_body = torch.rand(n_body, device=device) < gt_prob
                        body_selected = torch.where(
                            use_gt_body.unsqueeze(1).expand(-1, 4),
                            model.body_kpt_head.expand_bbox(gb),
                            model.body_kpt_head.expand_bbox(pb.detach()),
                        )

                        body_gt_kpts_sel, body_recompute_valid = recompute_keypoints_relative(
                            cat_gt_body_kpts[bsel], gb, body_selected
                        )
                        body_jbboxes, body_jkpts, body_jvalid = apply_bbox_jittering(
                            body_selected, body_gt_kpts_sel, cat_gt_body_vis[bsel],
                            max_shift_ratio=jitter_shift,
                            max_scale_ratio=jitter_scale,
                            jitter_prob=jitter_prob,
                        )

                        body_vis_sel = cat_gt_body_vis[bsel]
                        body_tw = torch.where(
                            body_vis_sel > 0,
                            torch.ones_like(body_vis_sel),
                            torch.full_like(body_vis_sel, OOF_COORD_WEIGHT),
                        )
                        body_tw = body_tw * body_recompute_valid.float() * body_jvalid.float()

                        body_out = model.forward_body_kpt(
                            hybrid_feats, body_jbboxes,
                            cat_priors[bsel], cat_valid_mask[bsel],
                            current_img_size, batch_indices=cat_batch_indices[bsel],
                        )

                        b_target_x, b_target_y, b_target_z = model.body_kpt_head.generate_target(
                            body_jkpts, body_tw
                        )
                        body_simcc_loss, _, _, _ = ctx.body_simcc_loss_fn(
                            body_out['heatmap_x'], body_out['heatmap_y'], body_out['heatmap_z'],
                            b_target_x, b_target_y, b_target_z,
                            body_tw, return_per_axis=True
                        )
                        loss = loss + body_bbox_loss + body_simcc_loss

                        body_needs_decode = (
                            ctx.body_bone_loss_fn is not None
                            or ctx.body_coord_loss_fn is not None
                            or ctx.body_bone_direction_loss_fn is not None
                            or ctx.body_image_coord_loss_fn is not None
                            or ctx.body_consistency_loss_fn is not None
                        )
                        if body_needs_decode:
                            body_pred_coords, _, _ = model.body_kpt_head.decode_relative(
                                body_out['heatmap_x'], body_out['heatmap_y'], body_out['heatmap_z']
                            )
                            if ctx.body_bone_loss_fn is not None:
                                body_bone_loss = ctx.body_bone_loss_fn(
                                    body_pred_coords, body_jkpts, body_tw
                                )
                                loss = loss + body_bone_loss
                            if ctx.body_coord_loss_fn is not None:
                                body_coord_loss = ctx.body_coord_loss_fn(
                                    body_pred_coords, body_jkpts, body_tw
                                )
                                loss = loss + body_coord_loss
                            # Same parallel-shift countermeasures as the head
                            if ctx.body_bone_direction_loss_fn is not None:
                                loss = loss + ctx.body_bone_direction_loss_fn(
                                    body_pred_coords, body_jkpts, body_tw
                                )
                            if ctx.body_image_coord_loss_fn is not None:
                                loss = loss + ctx.body_image_coord_loss_fn(
                                    body_pred_coords, body_jkpts,
                                    body_jbboxes, body_jbboxes, body_tw,
                                )
                            if ctx.body_consistency_loss_fn is not None:
                                # Second jitter pass, agreement in image coordinates
                                body_jbboxes_b, _, _ = apply_bbox_jittering(
                                    body_selected, body_gt_kpts_sel, cat_gt_body_vis[bsel],
                                    max_shift_ratio=jitter_shift,
                                    max_scale_ratio=jitter_scale,
                                    jitter_prob=jitter_prob,
                                )
                                body_out_b = model.forward_body_kpt(
                                    hybrid_feats, body_jbboxes_b,
                                    cat_priors[bsel], cat_valid_mask[bsel],
                                    current_img_size, batch_indices=cat_batch_indices[bsel],
                                )
                                body_pred_coords_b, _, _ = model.body_kpt_head.decode_relative(
                                    body_out_b['heatmap_x'],
                                    body_out_b['heatmap_y'],
                                    body_out_b['heatmap_z'],
                                )
                                loss = loss + ctx.body_consistency_loss_fn(
                                    body_pred_coords, body_pred_coords_b,
                                    body_jbboxes, body_jbboxes_b, body_tw,
                                )

                        if ctx.body_visibility_loss_fn is not None:
                            body_visibility_loss = ctx.body_visibility_loss_fn(
                                body_out['visibility_logits'], body_vis_sel
                            )
                            loss = loss + body_visibility_loss

                        if ctx.in_frame_loss_fn is not None:
                            body_in_frame_loss = ctx.in_frame_loss_fn(
                                body_out['in_frame_logits'], body_vis_sel
                            )
                            in_frame_loss = in_frame_loss + body_in_frame_loss
                            loss = loss + body_in_frame_loss

                    # ==================================================
                    # Stage 2: mask/depth branch
                    # ==================================================
                    mask_loss = torch.tensor(0.0, device=device)
                    depth_loss = torch.tensor(0.0, device=device)
                    if ctx.mask_loss_fn is not None and cat_mask_teachers is not None:
                        inst_feats = grid_features_ins.mean(dim=1)  # (N, 256)
                        mask_out = model.mask_depth_branch(
                            hybrid_feats, inst_feats, cat_priors,
                            batch_indices=cat_batch_indices
                        )
                        sup = cat_mask_teachers.shape[-1]
                        pred_up = F.interpolate(
                            mask_out, size=(sup, sup),
                            mode='bilinear', align_corners=False
                        )
                        mask_loss = ctx.mask_loss_fn(
                            pred_up[:, :7], cat_mask_teachers, cat_mask_cv
                        )
                        depth_loss = ctx.depth_loss_fn(
                            pred_up[:, 7], cat_depth_teachers, cat_depth_valid
                        )
                        loss = loss + mask_loss + depth_loss

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    batch_loss = loss
                    batch_count = 1
                    m['total_matched'] += total_N
                    m['total_body_matched'] += n_body

                    m['total_head_bbox_loss'] += head_bbox_loss.detach().item()
                    if ctx.head_bbox_l1_loss_fn is not None:
                        m['total_head_bbox_l1_loss'] += head_bbox_l1_loss.detach().item()
                    m['total_simcc_loss'] += simcc_loss.detach().item()
                    m['total_simcc_x'] += simcc_x.detach().item()
                    m['total_simcc_y'] += simcc_y.detach().item()
                    m['total_simcc_z'] += simcc_z.detach().item()
                    if ctx.bone_loss_fn is not None:
                        m['total_bone_loss'] += bone_loss.detach().item()
                    if ctx.bone_direction_loss_fn is not None:
                        m['total_bone_direction_loss'] += bone_direction_loss.detach().item()
                    if ctx.coord_loss_fn is not None:
                        m['total_coord_loss'] += coord_loss.detach().item()
                    if ctx.image_coord_loss_fn is not None:
                        m['total_image_coord_loss'] += image_coord_loss.detach().item()
                    if ctx.consistency_loss_fn is not None:
                        m['total_consistency_loss'] += consistency_loss.detach().item()
                    if ctx.visibility_loss_fn is not None:
                        m['total_visibility_loss'] += visibility_loss.detach().item()
                    # Body / in-frame
                    if ctx.in_frame_loss_fn is not None:
                        m['total_in_frame_loss'] += in_frame_loss.detach().item()
                    if ctx.body_simcc_loss_fn is not None and n_body > 0:
                        m['total_body_bbox_loss'] += body_bbox_loss.detach().item()
                        m['total_body_simcc_loss'] += body_simcc_loss.detach().item()
                        m['total_body_bone_loss'] += body_bone_loss.detach().item()
                        m['total_body_coord_loss'] += body_coord_loss.detach().item()
                        m['total_body_visibility_loss'] += body_visibility_loss.detach().item()
                    if ctx.mask_loss_fn is not None:
                        m['total_mask_loss'] += mask_loss.detach().item()
                        m['total_depth_loss'] += depth_loss.detach().item()

            # Optimizer step per variant
            if batch_count > 0:
                batch_loss = batch_loss / batch_count

                if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                    optimizer.zero_grad()

                    if scaler is not None:
                        scaler.scale(batch_loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for mod in trainable_modules(model) for p in mod.parameters()],
                            max_norm=1.0
                        )
                        stepped = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
                        if stepped:
                            scaler.step(optimizer)
                        scaler.update()
                    else:
                        batch_loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for mod in trainable_modules(model) for p in mod.parameters()],
                            max_norm=1.0
                        )
                        stepped = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
                        if stepped:
                            optimizer.step()

                    # Update the EMA only when the optimizer step happened
                    if stepped and ctx.ema is not None:
                        ctx.ema.update(trainable_modules(model))

                    m['total_loss'] += batch_loss.item()
                    m['num_batches'] += 1
                    batch_losses[ctx.name] = batch_loss.item()

        # Progress bar
        loss_str = ', '.join([f"{n}:{batch_losses.get(n, 0):.3f}" for n in [c.name for c in contexts]])
        pbar.set_postfix({'loss': loss_str, 'scale': current_img_size})

    # Results
    results = {}
    gt_prob = compute_gt_bbox_prob(epoch, warmup_epochs=3, steepness=1.0)

    for ctx in contexts:
        m = metrics[ctx.name]
        # Loss components are averaged per batch, as in validation
        n_batches = max(m['num_batches'], 1)
        results[ctx.name] = {
            'loss': m['total_loss'] / n_batches,
            'matched': m['total_matched'],
            'head_bbox_loss': m['total_head_bbox_loss'] / n_batches,
            'head_bbox_l1_loss': m['total_head_bbox_l1_loss'] / n_batches,
            'simcc_loss': m['total_simcc_loss'] / n_batches,
            'simcc_x': m['total_simcc_x'] / n_batches,
            'simcc_y': m['total_simcc_y'] / n_batches,
            'simcc_z': m['total_simcc_z'] / n_batches,
            'bone_loss': m['total_bone_loss'] / n_batches,
            'bone_direction_loss': m['total_bone_direction_loss'] / n_batches,
            'coord_loss': m['total_coord_loss'] / n_batches,
            'image_coord_loss': m['total_image_coord_loss'] / n_batches,
            'consistency_loss': m['total_consistency_loss'] / n_batches,
            'visibility_loss': m['total_visibility_loss'] / n_batches,
            # Body
            'body_bbox_loss': m['total_body_bbox_loss'] / n_batches,
            'body_simcc_loss': m['total_body_simcc_loss'] / n_batches,
            'body_bone_loss': m['total_body_bone_loss'] / n_batches,
            'body_coord_loss': m['total_body_coord_loss'] / n_batches,
            'body_visibility_loss': m['total_body_visibility_loss'] / n_batches,
            'in_frame_loss': m['total_in_frame_loss'] / n_batches,
            'body_matched': m['total_body_matched'],
            # Stage 2
            'mask_loss': m['total_mask_loss'] / n_batches,
            'depth_loss': m['total_depth_loss'] / n_batches,
            'hungarian_match': m['hungarian_match_count'],
            'gt_bbox_prob': gt_prob,
            'scale': current_img_size if use_multiscale else base_img_size,
        }

    return results


@torch.no_grad()
def validate_dual(
    contexts: List[VariantContext],
    dataloader: DataLoader,
    device: str,
    device_type: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """Validate all variants.

    Keypoints are predicted inside the predicted head/body bboxes (no scheduled sampling,
    jittering or multi-scale).

    Args:
        contexts: variant contexts
        dataloader: shared data loader
        device: device
        device_type: device type

    Returns:
        metrics per variant name
    """
    for ctx in contexts:
        ctx.model.eval()

    # Per-variant metrics
    metrics = {}
    for ctx in contexts:
        metrics[ctx.name] = {
            'total_loss': 0.0,
            'total_head_bbox_loss': 0.0,
            'total_head_bbox_l1_loss': 0.0,
            'total_simcc_loss': 0.0,
            'total_simcc_x': 0.0,
            'total_simcc_y': 0.0,
            'total_simcc_z': 0.0,
            'total_bone_loss': 0.0,
            'total_coord_loss': 0.0,
            'total_visibility_loss': 0.0,
            'total_matched': 0,
            'total_error_unified': 0.0,
            'total_confidence': 0.0,
            'num_batches': 0,
            # Body
            'total_body_bbox_loss': 0.0,
            'total_body_simcc_loss': 0.0,
            'total_body_bone_loss': 0.0,
            'total_body_coord_loss': 0.0,
            'total_body_error_unified': 0.0,
            'total_in_frame_loss': 0.0,
            'total_body_matched': 0,
            # Stage 2
            'total_mask_loss': 0.0,
            'total_depth_loss': 0.0,
        }

    variant_names = '/'.join([ctx.name for ctx in contexts])

    for batch in tqdm(dataloader, desc=f"Validation [{variant_names}]"):
        images = batch['images'].to(device, non_blocking=True)
        gt_keypoints_unified_list = batch['gt_keypoints_unified']
        gt_visibilities_list = batch['gt_visibilities']
        gt_head_bboxes_list = batch['gt_head_bboxes']
        gt_masks_list = batch['gt_masks']
        gt_ignore_flags_list = batch['gt_ignore_flags']
        # Body
        gt_body_ref_bboxes_list = batch['gt_body_ref_bboxes']
        gt_body_keypoints_list = batch['gt_body_keypoints_unified']
        gt_body_visibilities_list = batch['gt_body_visibilities']
        gt_body_ignore_flags_list = batch['gt_body_ignore_flags']
        img_shapes = batch['img_shapes']
        # Stage 2: mask/depth teachers
        _none_list = [None] * len(img_shapes)
        gt_mask_teachers_list = batch.get('gt_mask_teachers', _none_list)
        gt_mask_cv_list = batch.get('gt_mask_channel_valid', _none_list)
        gt_depth_teachers_list = batch.get('gt_depth_teachers', _none_list)
        gt_depth_valid_list = batch.get('gt_depth_valid', _none_list)

        B = images.shape[0]

        batch_data_samples = []
        for i in range(B):
            data_sample = DetDataSample()
            data_sample.set_metainfo({
                'img_shape': img_shapes[i],
                'ori_shape': img_shapes[i],
                'scale_factor': (1.0, 1.0),
                'pad_shape': img_shapes[i],
            })
            batch_data_samples.append(data_sample)

        # Process each variant in turn
        for ctx in contexts:
            model = ctx.model
            m = metrics[ctx.name]

            backbone_feats = model.extract_backbone_feat(images)
            pafpn_feats = model.extract_pafpn_feat(backbone_feats)

            try:
                ins_results_list = model.rtmdet.bbox_head.predict(
                    pafpn_feats, batch_data_samples, rescale=False
                )
            except RuntimeError as e:
                if device_type == 'mps' and "Placeholder tensor is empty" in str(e):
                    continue
                raise

            hybrid_feats = model.extract_hybrid_feat(backbone_feats)

            for b in range(B):
                ins_results = ins_results_list[b]
                gt_kpts_unified = gt_keypoints_unified_list[b].to(device, non_blocking=True)
                gt_vis = gt_visibilities_list[b].to(device, non_blocking=True)
                gt_head_bboxes = gt_head_bboxes_list[b].to(device, non_blocking=True)
                gt_masks = gt_masks_list[b].to(device, non_blocking=True)
                gt_ignore_flags = gt_ignore_flags_list[b].to(device, non_blocking=True)
                gt_body_bboxes_all = gt_body_ref_bboxes_list[b].to(device, non_blocking=True)
                gt_body_kpts_all = gt_body_keypoints_list[b].to(device, non_blocking=True)
                gt_body_vis_all = gt_body_visibilities_list[b].to(device, non_blocking=True)
                gt_body_ignore = gt_body_ignore_flags_list[b].to(device, non_blocking=True)

                if len(ins_results.bboxes) == 0 or len(gt_kpts_unified) == 0:
                    continue

                if not hasattr(ins_results, 'priors') or ins_results.priors is None:
                    continue

                if not hasattr(ins_results, 'masks') or ins_results.masks is None:
                    continue

                bboxes, labels, scores, priors, valid_count = select_top_person_instances(
                    ins_results.bboxes,
                    ins_results.labels,
                    ins_results.scores,
                    ins_results.priors,
                    max_instances=MAX_INSTANCES
                )

                if valid_count == 0:
                    continue

                bboxes = bboxes[:valid_count]
                priors = priors[:valid_count]
                ins_masks = ins_results.masks[:valid_count]

                matched_ins_indices = hungarian_mask_matching(
                    gt_masks, ins_masks, device
                )

                # Keep instances whose head or body annotation is usable
                valid_gt_mask = (matched_ins_indices >= 0) & (
                    (~gt_ignore_flags) | (~gt_body_ignore)
                )
                if not valid_gt_mask.any():
                    continue

                valid_gt_indices = torch.where(valid_gt_mask)[0]
                matched_gt_kpts_unified = gt_kpts_unified[valid_gt_indices]
                matched_gt_vis = gt_vis[valid_gt_indices]
                matched_gt_head_bboxes = gt_head_bboxes[valid_gt_indices]
                matched_head_valid = ~gt_ignore_flags[valid_gt_indices]
                matched_gt_body_bboxes = gt_body_bboxes_all[valid_gt_indices]
                matched_gt_body_kpts = gt_body_kpts_all[valid_gt_indices]
                matched_gt_body_vis = gt_body_vis_all[valid_gt_indices]
                matched_body_valid = ~gt_body_ignore[valid_gt_indices]

                matched_ins_idx = matched_ins_indices[valid_gt_mask]
                matched_ins_bboxes = bboxes[matched_ins_idx]
                matched_priors = priors[matched_ins_idx]

                N_matched = len(matched_gt_head_bboxes)
                valid_mask = torch.ones(N_matched, dtype=torch.bool, device=device)
                matched_target_weight = torch.where(
                    matched_gt_vis > 0,
                    torch.ones_like(matched_gt_vis),
                    torch.full_like(matched_gt_vis, OOF_COORD_WEIGHT),
                )
                matched_target_weight = (
                    matched_target_weight * matched_head_valid.float().unsqueeze(1)
                )

                single_hybrid_feats = [f[b:b+1] for f in hybrid_feats]

                pred_head_bboxes, grid_features_ins = model.forward_head_bbox(
                    single_hybrid_feats, matched_ins_bboxes, matched_priors, valid_mask, IMG_SIZE
                )

                head_bbox_loss = torch.tensor(0.0, device=device)
                head_bbox_l1_loss = torch.tensor(0.0, device=device)
                if matched_head_valid.any():
                    hsel = matched_head_valid
                    head_bbox_loss = ctx.head_bbox_loss_fn(
                        pred_head_bboxes[hsel], matched_gt_head_bboxes[hsel]
                    )
                    if ctx.head_bbox_l1_loss_fn is not None:
                        head_bbox_l1_loss = ctx.head_bbox_l1_loss_fn(
                            pred_head_bboxes[hsel], matched_gt_head_bboxes[hsel]
                        )
                        head_bbox_loss = head_bbox_loss + head_bbox_l1_loss

                pred_head_bboxes_expanded = model.dcc_head.expand_bbox(pred_head_bboxes)

                # Re-reference the GT keypoints to the predicted bbox (the bbox the features
                # are sampled from)
                pred_bbox_kpts, recompute_valid = recompute_keypoints_relative(
                    matched_gt_kpts_unified,
                    matched_gt_head_bboxes,
                    pred_head_bboxes_expanded
                )
                # Drop keypoints outside the bbox range via target_weight
                matched_target_weight = matched_target_weight * recompute_valid.float()

                dcc_output = model.forward_dcc(
                    single_hybrid_feats, pred_head_bboxes_expanded, matched_priors, valid_mask, IMG_SIZE
                )

                heatmap_x = dcc_output['heatmap_x']
                heatmap_y = dcc_output['heatmap_y']
                heatmap_z = dcc_output['heatmap_z']
                visibility_logits = dcc_output['visibility_logits']

                target_x, target_y, target_z = model.dcc_head.generate_target(
                    pred_bbox_kpts, matched_target_weight
                )

                simcc_loss, simcc_x, simcc_y, simcc_z = ctx.simcc_loss_fn(
                    heatmap_x, heatmap_y, heatmap_z,
                    target_x, target_y, target_z,
                    matched_target_weight,
                    return_per_axis=True
                )

                loss = head_bbox_loss + simcc_loss

                pred_rel_coords, confidences, _ = model.dcc_head.decode_relative(
                    heatmap_x, heatmap_y, heatmap_z
                )

                bone_loss = torch.tensor(0.0, device=device)
                if ctx.bone_loss_fn is not None:
                    bone_loss = ctx.bone_loss_fn(
                        pred_rel_coords, pred_bbox_kpts, matched_target_weight
                    )
                    loss = loss + bone_loss

                coord_loss = torch.tensor(0.0, device=device)
                if ctx.coord_loss_fn is not None:
                    coord_loss = ctx.coord_loss_fn(
                        pred_rel_coords, pred_bbox_kpts, matched_target_weight
                    )
                    loss = loss + coord_loss

                visibility_loss = torch.tensor(0.0, device=device)
                if ctx.visibility_loss_fn is not None:
                    vis_include = (matched_gt_vis > 0) & matched_head_valid.unsqueeze(1)
                    visibility_loss = ctx.visibility_loss_fn(
                        visibility_logits, matched_gt_vis, include_mask=vis_include
                    )
                    loss = loss + visibility_loss

                # In-frame prediction (head)
                in_frame_loss = torch.tensor(0.0, device=device)
                if ctx.in_frame_loss_fn is not None:
                    in_frame_loss = ctx.in_frame_loss_fn(
                        dcc_output['in_frame_logits'], matched_gt_vis,
                        include_mask=matched_head_valid.unsqueeze(1).expand_as(matched_gt_vis),
                    )
                    loss = loss + in_frame_loss

                # ==================================================
                # Body (reference bbox + 31 keypoints)
                # ==================================================
                body_bbox_loss = torch.tensor(0.0, device=device)
                body_simcc_loss = torch.tensor(0.0, device=device)
                body_bone_loss = torch.tensor(0.0, device=device)
                body_coord_loss = torch.tensor(0.0, device=device)
                body_error_unified = None
                n_body = 0
                if ctx.body_simcc_loss_fn is not None and matched_body_valid.any():
                    bsel = matched_body_valid
                    n_body = int(bsel.sum().item())

                    pred_body_bboxes = model.body_bbox_head(
                        grid_features_ins, matched_ins_bboxes
                    )
                    pb = pred_body_bboxes[bsel]
                    gb = matched_gt_body_bboxes[bsel]

                    body_bbox_loss = ctx.body_bbox_loss_fn(pb, gb)
                    if ctx.body_bbox_l1_loss_fn is not None:
                        body_bbox_loss = body_bbox_loss + ctx.body_bbox_l1_loss_fn(pb, gb)

                    pred_body_expanded = model.body_kpt_head.expand_bbox(pb)
                    body_gt_kpts_sel, body_recompute_valid = recompute_keypoints_relative(
                        matched_gt_body_kpts[bsel], gb, pred_body_expanded
                    )
                    body_vis_sel = matched_gt_body_vis[bsel]
                    body_tw = torch.where(
                        body_vis_sel > 0,
                        torch.ones_like(body_vis_sel),
                        torch.full_like(body_vis_sel, OOF_COORD_WEIGHT),
                    )
                    body_tw = body_tw * body_recompute_valid.float()

                    body_out = model.forward_body_kpt(
                        single_hybrid_feats, pred_body_expanded,
                        matched_priors[bsel], valid_mask[bsel], IMG_SIZE
                    )

                    b_tx, b_ty, b_tz = model.body_kpt_head.generate_target(
                        body_gt_kpts_sel, body_tw
                    )
                    body_simcc_loss, _, _, _ = ctx.body_simcc_loss_fn(
                        body_out['heatmap_x'], body_out['heatmap_y'], body_out['heatmap_z'],
                        b_tx, b_ty, b_tz, body_tw, return_per_axis=True
                    )
                    loss = loss + body_bbox_loss + body_simcc_loss

                    body_pred_coords, _, _ = model.body_kpt_head.decode_relative(
                        body_out['heatmap_x'], body_out['heatmap_y'], body_out['heatmap_z']
                    )
                    if ctx.body_bone_loss_fn is not None:
                        body_bone_loss = ctx.body_bone_loss_fn(
                            body_pred_coords, body_gt_kpts_sel, body_tw
                        )
                        loss = loss + body_bone_loss
                    if ctx.body_coord_loss_fn is not None:
                        body_coord_loss = ctx.body_coord_loss_fn(
                            body_pred_coords, body_gt_kpts_sel, body_tw
                        )
                        loss = loss + body_coord_loss

                    if ctx.body_visibility_loss_fn is not None:
                        loss = loss + ctx.body_visibility_loss_fn(
                            body_out['visibility_logits'], body_vis_sel
                        )
                    if ctx.in_frame_loss_fn is not None:
                        loss = loss + ctx.in_frame_loss_fn(
                            body_out['in_frame_logits'], body_vis_sel
                        )

                    body_err = torch.sqrt(
                        ((body_pred_coords - body_gt_kpts_sel) ** 2).sum(dim=-1)
                    ) * body_tw
                    if body_tw.sum() > 0:
                        body_error_unified = (body_err.sum() / body_tw.sum()).item()

                # Stage 2: mask/depth
                mask_loss_v = torch.tensor(0.0, device=device)
                depth_loss_v = torch.tensor(0.0, device=device)
                if (ctx.mask_loss_fn is not None
                        and gt_mask_teachers_list[b] is not None):
                    mt = gt_mask_teachers_list[b].to(device)[valid_gt_indices]
                    mcv = gt_mask_cv_list[b].to(device)[valid_gt_indices]
                    dt = gt_depth_teachers_list[b].to(device)[valid_gt_indices]
                    dv = gt_depth_valid_list[b].to(device)[valid_gt_indices]
                    inst_feats = grid_features_ins.mean(dim=1)
                    mask_out = model.mask_depth_branch(
                        single_hybrid_feats, inst_feats, matched_priors
                    )
                    sup = mt.shape[-1]
                    pred_up = F.interpolate(
                        mask_out, size=(sup, sup), mode='bilinear', align_corners=False
                    )
                    mask_loss_v = ctx.mask_loss_fn(pred_up[:, :7], mt, mcv)
                    depth_loss_v = ctx.depth_loss_fn(pred_up[:, 7], dt, dv)
                    loss = loss + mask_loss_v + depth_loss_v

                m['total_loss'] += loss.item()
                m['total_head_bbox_loss'] += head_bbox_loss.item()
                if ctx.head_bbox_l1_loss_fn is not None:
                    m['total_head_bbox_l1_loss'] += head_bbox_l1_loss.item()
                m['total_simcc_loss'] += simcc_loss.item()
                m['total_simcc_x'] += simcc_x.item()
                m['total_simcc_y'] += simcc_y.item()
                m['total_simcc_z'] += simcc_z.item()
                if ctx.bone_loss_fn is not None:
                    m['total_bone_loss'] += bone_loss.item()
                if ctx.coord_loss_fn is not None:
                    m['total_coord_loss'] += coord_loss.item()
                if ctx.visibility_loss_fn is not None:
                    m['total_visibility_loss'] += visibility_loss.item()
                # Body
                if ctx.in_frame_loss_fn is not None:
                    m['total_in_frame_loss'] += in_frame_loss.item()
                if n_body > 0:
                    m['total_body_bbox_loss'] += body_bbox_loss.item()
                    m['total_body_simcc_loss'] += body_simcc_loss.item()
                    m['total_body_bone_loss'] += body_bone_loss.item()
                    m['total_body_coord_loss'] += body_coord_loss.item()
                    if body_error_unified is not None:
                        m['total_body_error_unified'] += body_error_unified
                    m['total_body_matched'] += n_body
                if ctx.mask_loss_fn is not None:
                    m['total_mask_loss'] += mask_loss_v.item()
                    m['total_depth_loss'] += depth_loss_v.item()
                m['total_matched'] += len(valid_gt_indices)
                m['num_batches'] += 1

                error_unified = torch.sqrt(
                    ((pred_rel_coords - pred_bbox_kpts) ** 2).sum(dim=-1)
                )
                error_unified_masked = error_unified * matched_target_weight
                if matched_target_weight.sum() > 0:
                    m['total_error_unified'] += (error_unified_masked.sum() / matched_target_weight.sum()).item()

                conf_masked = confidences * matched_target_weight
                if matched_target_weight.sum() > 0:
                    m['total_confidence'] += (conf_masked.sum() / matched_target_weight.sum()).item()

    # Results
    results = {}
    for ctx in contexts:
        m = metrics[ctx.name]
        n_batches = max(m['num_batches'], 1)
        results[ctx.name] = {
            'loss': m['total_loss'] / n_batches,
            'head_bbox_loss': m['total_head_bbox_loss'] / n_batches,
            'head_bbox_l1_loss': m['total_head_bbox_l1_loss'] / n_batches,
            'simcc_loss': m['total_simcc_loss'] / n_batches,
            'simcc_x': m['total_simcc_x'] / n_batches,
            'simcc_y': m['total_simcc_y'] / n_batches,
            'simcc_z': m['total_simcc_z'] / n_batches,
            'bone_loss': m['total_bone_loss'] / n_batches,
            'coord_loss': m['total_coord_loss'] / n_batches,
            'visibility_loss': m['total_visibility_loss'] / n_batches,
            # Body
            'body_bbox_loss': m['total_body_bbox_loss'] / n_batches,
            'body_simcc_loss': m['total_body_simcc_loss'] / n_batches,
            'body_bone_loss': m['total_body_bone_loss'] / n_batches,
            'body_coord_loss': m['total_body_coord_loss'] / n_batches,
            'body_error_unified': m['total_body_error_unified'] / n_batches,
            'in_frame_loss': m['total_in_frame_loss'] / n_batches,
            'body_matched': m['total_body_matched'],
            'mask_loss': m['total_mask_loss'] / n_batches,
            'depth_loss': m['total_depth_loss'] / n_batches,
            'matched': m['total_matched'],
            'error_unified': m['total_error_unified'] / n_batches,
            'confidence': m['total_confidence'] / n_batches,
        }

    return results


# ============================================================
# Variant context, resume and checkpoints
# ============================================================

def create_variant_context(
    variant_name: str,
    device: str,
    args,
    use_amp: bool,
    work_dir: Path,
    warmup_epochs: int = WARMUP_EPOCHS
) -> VariantContext:
    """Build the model, optimizer, schedulers, losses and EMA of one variant.

    With --resume, the state is restored from {work_dir}/{args.resume}_model_{variant}.pth.

    Args:
        variant_name: 'tiny' or 's'
        device: device
        args: command-line arguments
        use_amp: enable AMP
        work_dir: work directory
        warmup_epochs: LR warmup epochs

    Returns:
        the initialized VariantContext
    """
    # Model
    rtmdet_cfg = ConfigDict(get_rtmdet_config(variant_name))
    rtmdet_in_channels = get_rtmdet_in_channels(variant_name)
    mask_weight = getattr(args, 'mask_weight', 0.0)
    model = RTMDetFusionForTraining(
        rtmdet_cfg, in_channels=rtmdet_in_channels,
        with_mask_branch=(mask_weight > 0)
    )

    # RTMDet-Ins weights
    weights_path = str(get_rtmdet_weights(variant_name))
    if Path(weights_path).exists():
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.rtmdet.load_state_dict(state_dict, strict=False)
        print(f"  [{variant_name}] Loaded RTMDet weights: {weights_path}")
    else:
        print(f"  [{variant_name}] Warning: RTMDet weights not found: {weights_path}")

    model = model.to(device)

    # Freeze RTMDet-Ins
    for param in model.rtmdet.parameters():
        param.requires_grad = False

    # Lower the RTMDet-Ins score threshold for training-time matching: heavily occluded people
    # far in the back become candidates too, and Hungarian matching acts as the quality gate.
    # (The exporters and the viewer set their own thresholds.)
    try:
        model.rtmdet.bbox_head.test_cfg.score_thr = TRAIN_INS_SCORE_THR
        print(f"  [{variant_name}] Train matching score_thr: {TRAIN_INS_SCORE_THR}")
    except AttributeError:
        print(f"  [{variant_name}] Warning: could not set train score_thr")

    trainable_params = [
        p for mod in trainable_modules(model)
        for p in mod.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr, weight_decay=WEIGHT_DECAY
    )

    # LR schedule: linear warmup, then cosine (default) or plateau. Cosine is the default because
    # with long epochs the val loss tends to improve slightly every epoch, so a plateau scheduler
    # may never lower the LR.
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    scheduler_type = getattr(args, 'scheduler', 'cosine')
    min_lr = getattr(args, 'min_lr', 1e-5)
    if scheduler_type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - warmup_epochs),
            eta_min=min_lr,
        )
    else:  # plateau
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,   # epochs are long (thousands of iterations), so react early
            min_lr=min_lr,
        )

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Losses
    simcc_loss_fn = SimCCLoss(
        beta=10.0,
        label_softmax=False,
        use_target_weight=True,
        loss_weight=1.0
    ).to(device)

    # The head bbox losses are weighted so that they keep about 20% of the head loss weight;
    # with a smaller share the predicted head bboxes tend to be stretched.
    head_bbox_loss_fn = HeadBboxIoULoss(
        loss_weight=2.0,
        use_giou=True
    ).to(device)

    head_bbox_l1_loss_fn = HeadBboxL1Loss(
        loss_weight=12.0,
        img_size=args.img_size
    ).to(device)

    bone_loss_fn = BoneLoss(
        loss_weight=10.0,
        use_target_weight=True
    ).to(device)

    # Bone direction loss (cosine; catches keypoint groups shifted in parallel)
    bone_direction_loss_fn = None
    if args.bone_direction_weight > 0:
        bone_direction_loss_fn = BoneDirectionLoss(
            loss_weight=args.bone_direction_weight,
            use_target_weight=True,
        ).to(device)

    coord_loss_fn = CoordL2Loss(
        loss_weight=15.0,
        use_target_weight=True
    ).to(device)

    # Image-coordinate L2 loss (supervises absolute positions, not bbox-relative ones)
    image_coord_loss_fn = None
    if args.image_coord_weight > 0:
        image_coord_loss_fn = ImageCoordL2Loss(
            loss_weight=args.image_coord_weight,
            use_target_weight=True,
        ).to(device)

    # Bbox-jitter consistency loss
    consistency_loss_fn = None
    if args.consistency_weight > 0:
        consistency_loss_fn = BboxJitterConsistencyLoss(
            loss_weight=args.consistency_weight,
            use_target_weight=True,
        ).to(device)

    visibility_loss_fn = VisibilityBCELoss(
        loss_weight=0.5,
        label_smoothing=0.0
    ).to(device)

    # ==================================================
    # Body losses (--body-weight scales the whole group; 0 disables it)
    # ==================================================
    body_scale = getattr(args, 'body_weight', 1.0)
    body_bbox_loss_fn = None
    body_bbox_l1_loss_fn = None
    body_simcc_loss_fn = None
    body_bone_loss_fn = None
    body_coord_loss_fn = None
    body_visibility_loss_fn = None
    in_frame_loss_fn = None
    body_bone_direction_loss_fn = None
    body_image_coord_loss_fn = None
    body_consistency_loss_fn = None
    if body_scale > 0:
        body_bbox_loss_fn = HeadBboxIoULoss(
            loss_weight=2.0 * body_scale, use_giou=True
        ).to(device)
        body_bbox_l1_loss_fn = HeadBboxL1Loss(
            loss_weight=12.0 * body_scale, img_size=args.img_size
        ).to(device)
        body_simcc_loss_fn = SimCCLoss(
            beta=10.0, label_softmax=False,
            use_target_weight=True, loss_weight=1.0 * body_scale
        ).to(device)
        body_bone_loss_fn = BoneLoss(
            loss_weight=10.0 * body_scale,
            use_target_weight=True,
            bone_links=BODY_BONE_LINKS,
        ).to(device)
        body_coord_loss_fn = CoordL2Loss(
            loss_weight=15.0 * body_scale, use_target_weight=True
        ).to(device)
        body_visibility_loss_fn = VisibilityBCELoss(
            loss_weight=0.5 * body_scale, label_smoothing=0.0
        ).to(device)
        # In-frame prediction (shared by head and body)
        in_frame_loss_fn = InFrameBCELoss(loss_weight=0.5).to(device)

        # Same parallel-shift countermeasures as the head, with the same weights
        if args.bone_direction_weight > 0:
            body_bone_direction_loss_fn = BoneDirectionLoss(
                loss_weight=args.bone_direction_weight * body_scale,
                use_target_weight=True,
                bone_links=BODY_BONE_LINKS,
            ).to(device)
        body_image_coord_loss_fn = None
        if args.image_coord_weight > 0:
            body_image_coord_loss_fn = ImageCoordL2Loss(
                loss_weight=args.image_coord_weight * body_scale,
                use_target_weight=True,
            ).to(device)
        if args.consistency_weight > 0:
            body_consistency_loss_fn = BboxJitterConsistencyLoss(
                loss_weight=args.consistency_weight * body_scale,
                use_target_weight=True,
            ).to(device)

    # Stage 2: mask/depth losses (enabled by --mask-weight > 0)
    mask_loss_fn = None
    depth_loss_fn = None
    if mask_weight > 0:
        mask_loss_fn = MaskBCEDiceLoss(loss_weight=mask_weight).to(device)
        depth_loss_fn = MaskedDepthL1Loss(
            loss_weight=getattr(args, 'depth_weight', 5.0)
        ).to(device)
        print(f"  [{variant_name}] Mask/Depth branch enabled "
              f"(mask_w={mask_weight}, depth_w={getattr(args, 'depth_weight', 5.0)})")

    # EMA of the trainable submodules
    ema = None
    if not args.no_ema:
        ema = ModelEMA(
            trainable_modules(model),
            decay=args.ema_decay,
        )

    ctx = VariantContext(
        name=variant_name,
        model=model,
        optimizer=optimizer,
        warmup_scheduler=warmup_scheduler,
        plateau_scheduler=main_scheduler,
        scheduler_type=scheduler_type,
        scaler=scaler,
        best_val_loss=float('inf'),
        simcc_loss_fn=simcc_loss_fn,
        head_bbox_loss_fn=head_bbox_loss_fn,
        head_bbox_l1_loss_fn=head_bbox_l1_loss_fn,
        bone_loss_fn=bone_loss_fn,
        bone_direction_loss_fn=bone_direction_loss_fn,
        coord_loss_fn=coord_loss_fn,
        image_coord_loss_fn=image_coord_loss_fn,
        consistency_loss_fn=consistency_loss_fn,
        visibility_loss_fn=visibility_loss_fn,
        body_bbox_loss_fn=body_bbox_loss_fn,
        body_bbox_l1_loss_fn=body_bbox_l1_loss_fn,
        body_simcc_loss_fn=body_simcc_loss_fn,
        body_bone_loss_fn=body_bone_loss_fn,
        body_coord_loss_fn=body_coord_loss_fn,
        body_visibility_loss_fn=body_visibility_loss_fn,
        body_bone_direction_loss_fn=body_bone_direction_loss_fn,
        body_image_coord_loss_fn=body_image_coord_loss_fn,
        body_consistency_loss_fn=body_consistency_loss_fn,
        mask_loss_fn=mask_loss_fn,
        depth_loss_fn=depth_loss_fn,
        in_frame_loss_fn=in_frame_loss_fn,
        ema=ema,
    )

    if args.resume:
        resume_path = work_dir / f'{args.resume}_model_{variant_name}.pth'
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
            resume_variant_context(ctx, ckpt, resume_path)
        else:
            print(f"  [{variant_name}] Warning: Resume checkpoint not found: {resume_path}")

    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"  [{variant_name}] Trainable params: {trainable_count:,}")

    return ctx


def resume_variant_context(ctx: VariantContext, ckpt: dict, resume_path: Path) -> None:
    """Restore weights, EMA, optimizer state and best_val_loss from a checkpoint.

    With EMA enabled during training, the '{name}_state_dict' slots hold the EMA weights and
    'raw_state_dicts' the raw training weights; checkpoints trained with --no-ema only have the
    '{name}_state_dict' slots. Present modules are loaded strictly.

    Starting stage 2 from a stage-1 checkpoint (no mask/depth branch) creates the branch fresh:
    its EMA starts from its initial weights, the optimizer state starts fresh, and best_val_loss
    is reset because the loss composition changes.
    """
    model = ctx.model
    names = trainable_module_names(model)
    raw_sds = ckpt.get('raw_state_dicts')

    missing = [n for n in names if f'{n}_state_dict' not in ckpt]
    new_modules = [n for n in missing if n == 'mask_depth_branch']
    unsupported = [n for n in missing if n not in new_modules]
    if unsupported:
        raise RuntimeError(
            f"Cannot resume from {resume_path}: it has no weights for {unsupported}. "
            f"Only checkpoints of the current model generation can be resumed."
        )

    # Training weights: raw weights if present, otherwise the main slots
    for n in names:
        if n in new_modules:
            continue
        sd = raw_sds[n] if raw_sds is not None else ckpt[f'{n}_state_dict']
        getattr(model, n).load_state_dict(sd)

    # The EMA continues from the checkpoint's EMA weights when it has them; otherwise
    # (--no-ema checkpoint or a newly created module) it starts from the current weights.
    if ctx.ema is not None:
        ctx.ema.load_state_dicts([
            ckpt[f'{n}_state_dict'] if raw_sds is not None and n not in new_modules
            else getattr(model, n).state_dict()
            for n in names
        ])
        print(f"  [{ctx.name}] Resumed EMA state")

    if new_modules:
        ctx.best_val_loss = float('inf')
        print(f"  [{ctx.name}] {resume_path.name} has no {new_modules}: created fresh "
              f"(stage 2). Optimizer state starts fresh and best_val_loss is reset.")
    else:
        try:
            ctx.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError as e:
            print(f"  [{ctx.name}] Warning: Could not load optimizer: {e}")
        ctx.best_val_loss = ckpt.get('best_val_loss', float('inf'))

    print(f"  [{ctx.name}] Resumed from {resume_path}")


def checkpoint_config(
    args,
    jitter_prob: float,
    jitter_shift: float,
    jitter_scale: float,
    use_multiscale: bool
) -> dict:
    """The 'config' entry of a checkpoint: model hyperparameters and all CLI arguments."""
    return {
        'num_keypoints': NUM_KEYPOINTS,
        'num_bins': NUM_BINS,
        'num_body_keypoints': NUM_BODY_KEYPOINTS,
        'num_bins_body': NUM_BINS_BODY,
        'bin_margin': BIN_MARGIN,
        'body_bbox_range_scale': BODY_BBOX_RANGE_SCALE,
        'max_instances': MAX_INSTANCES,
        'img_size': args.img_size,
        'jitter_prob': jitter_prob,
        'jitter_shift': jitter_shift,
        'jitter_scale': jitter_scale,
        'multiscale': use_multiscale,
        'train_args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }


def save_checkpoint(
    ctx: VariantContext,
    work_dir: Path,
    epoch: int,
    args,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    is_best: bool,
    jitter_prob: float,
    jitter_shift: float,
    jitter_scale: float,
    use_multiscale: bool
) -> None:
    """Save the checkpoint of a variant when it is the best so far.

    With EMA enabled the main '{name}_state_dict' slots hold the EMA weights, so inference always
    loads the weights that were validated. The raw training weights go to 'raw_state_dicts' for
    resuming.
    """
    module_names = trainable_module_names(ctx.model)
    if ctx.ema is not None:
        main_sds = ctx.ema.state_dicts()
    else:
        main_sds = [getattr(ctx.model, n).state_dict() for n in module_names]

    checkpoint_data = {
        'version': CODE_VERSION,
        'epoch': epoch,
        'optimizer_state_dict': ctx.optimizer.state_dict(),
        'best_val_loss': ctx.best_val_loss,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': checkpoint_config(args, jitter_prob, jitter_shift, jitter_scale, use_multiscale),
    }
    for name, sd in zip(module_names, main_sds):
        checkpoint_data[f'{name}_state_dict'] = sd

    # With EMA, the raw training weights are stored as well (for resuming)
    if ctx.ema is not None:
        checkpoint_data['raw_state_dicts'] = {
            n: getattr(ctx.model, n).state_dict() for n in module_names
        }

    if is_best:
        save_path = work_dir / f'best_model_{ctx.name}.pth'
        torch.save(checkpoint_data, save_path)
        print(f"  [{ctx.name}] Saved best model: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train RTMDet-Fusion.")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Total number of epochs (a resumed run continues at the '
                             'checkpoint epoch + 1)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--img-size', type=int, default=IMG_SIZE)
    parser.add_argument('--work-dir', type=str, default=str(WORK_DIR))
    parser.add_argument('--variant', type=str, default='both', choices=['tiny', 's', 'both'],
                        help='RTMDet-Ins variant (tiny/s/both). "both" trains the two variants '
                             'on the same batches')
    parser.add_argument('--train-ann', type=str,
                        default=str(TRAIN_ANN_PATH))
    parser.add_argument('--val-ann', type=str,
                        default=str(VAL_ANN_PATH))
    parser.add_argument('--img-dir', type=str,
                        default=str(TRAIN_IMG_DIR),
                        help='Training image directory')
    parser.add_argument('--val-img-dir', type=str,
                        default=str(VAL_IMG_DIR),
                        help='Validation image directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from (best, final or another name; loads '
                             '{work_dir}/{name}_model_{variant}.pth)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable AMP (mixed precision, CUDA only)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader workers (default: 4 on CUDA, otherwise 0)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='TensorBoard run name suffix (default: timestamp only)')
    parser.add_argument('--start-epoch', type=int, default=None,
                        help='Start epoch (default: restored from the checkpoint)')
    parser.add_argument('--jitter-prob', type=float, default=0.8,
                        help='Bbox jittering probability (default: 0.8)')
    parser.add_argument('--jitter-shift', type=float, default=0.25,
                        help='Bbox jittering maximum shift ratio, independent for x/y '
                             '(default: 0.25)')
    parser.add_argument('--jitter-scale', type=float, default=0.15,
                        help='Bbox jittering maximum scale change ratio, independent for x/y '
                             '(default: 0.15)')
    parser.add_argument('--no-jitter', action='store_true',
                        help='Disable bbox jittering')
    parser.add_argument('--no-multiscale', action='store_true',
                        help='Disable multi-scale training (train at --img-size only)')
    parser.add_argument('--no-ema', action='store_true',
                        help='Disable the EMA (exponential moving average) of the weights')
    parser.add_argument('--ema-decay', type=float, default=0.9998,
                        help='EMA decay (default: 0.9998)')
    parser.add_argument('--consistency-weight', type=float, default=5.0,
                        help='Weight of the bbox-jitter consistency loss (0 disables it, '
                             'default: 5.0)')
    parser.add_argument('--image-coord-weight', type=float, default=10.0,
                        help='Weight of the image-coordinate L2 loss (0 disables it, '
                             'default: 10.0)')
    parser.add_argument('--bone-direction-weight', type=float, default=5.0,
                        help='Weight of the bone direction loss (0 disables it, default: 5.0)')
    parser.add_argument('--body-weight', type=float, default=1.0,
                        help='Scale of the whole body loss group (0 disables the body losses, '
                             'default: 1.0)')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='LR warmup epochs (default: 2 = half LR in epoch 0, full LR from '
                             'epoch 1)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau'],
                        help='LR schedule after the warmup (default: cosine; plateau halves the '
                             'LR when the val loss stops improving)')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                        help='Final LR (cosine eta_min / plateau min_lr, default: 1e-5)')
    parser.add_argument('--mask-weight', type=float, default=0.0,
                        help='Weight of the mask loss (BCE + Dice). 0 (default, stage 1) does not '
                             'create the mask/depth branch. Stage 2 fine-tunes the stage-1 '
                             'checkpoint with --resume best --mask-weight 4.0')
    parser.add_argument('--depth-weight', type=float, default=5.0,
                        help='Weight of the depth L1 loss (only with --mask-weight > 0)')
    parser.add_argument('--experiment', action='store_true',
                        help='Allow training from an untagged or modified commit. The checkpoint '
                             'records the git describe string and is not a release.')
    args = parser.parse_args()

    if not is_release_version(CODE_VERSION) and not args.experiment:
        print(f"Error: the code version is '{CODE_VERSION or 'unknown (git unavailable)'}', "
              f"not a release tag.", file=sys.stderr)
        print("  Release training must start from a clean commit that carries its version tag:\n"
              "    git tag -a vX.Y.Z -m \"...\"   (before training; push the tag only after the "
              "model is adopted)\n"
              "  To train from an untagged or modified tree anyway, pass --experiment; the "
              "checkpoint then records the git describe string and is not a release.",
              file=sys.stderr)
        sys.exit(1)

    device = DEVICE_TYPE
    print(f"Device: {device}")
    print(f"Code version: {CODE_VERSION or 'unknown (git unavailable)'}")
    if not is_release_version(CODE_VERSION):
        print("  Experiment run (--experiment): the checkpoint is not a release.")

    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

    # Random seed (reproducible data augmentation)
    setup_random_seed()
    print(f"  Random seed: {RANDOM_SEED}, cudnn.benchmark: enabled")

    use_amp = args.amp and device == 'cuda'
    if args.amp and device != 'cuda':
        print("Warning: AMP is only supported on CUDA, disabled.")
    if use_amp:
        print("  AMP: enabled")

    register_all_modules()

    use_multiscale = not args.no_multiscale
    base_img_size = args.img_size

    # Variants to train
    if args.variant == 'both':
        variant_list = ['tiny', 's']
    else:
        variant_list = [args.variant]

    print("Train config:")
    print(f"  RTMDet variant: {args.variant} ({', '.join(variant_list)})")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Image size: {args.img_size}")
    print(f"  Max instances: {MAX_INSTANCES}")
    print(f"  DCC bins: {NUM_BINS} (all axes)")
    print(f"  Multi-scale: {'enabled' if use_multiscale else 'disabled'}")
    if use_multiscale:
        print(f"    Scales: {MULTISCALE_SIZES}")
    print(f"  LR warmup: {args.warmup_epochs} epochs, scheduler: {args.scheduler} "
          f"(min_lr={args.min_lr:.0e})")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("Creating variant contexts...")
    contexts: List[VariantContext] = []
    for variant_name in variant_list:
        ctx = create_variant_context(
            variant_name=variant_name,
            device=device,
            args=args,
            use_amp=use_amp,
            work_dir=work_dir,
            warmup_epochs=args.warmup_epochs
        )
        contexts.append(ctx)

    val_img_dir = args.val_img_dir if args.val_img_dir else args.img_dir

    # Datasets (stage 2 also loads the mask/depth teachers)
    with_mask_teachers = args.mask_weight > 0
    train_dataset = KeypointDataset(
        ann_file=args.train_ann,
        img_dir=args.img_dir,
        img_size=args.img_size,
        is_train=True,
        with_mask_teachers=with_mask_teachers
    )
    val_dataset = KeypointDataset(
        ann_file=args.val_ann,
        img_dir=val_img_dir,
        img_size=args.img_size,
        is_train=False,
        with_mask_teachers=with_mask_teachers
    )

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 4 if device == 'cuda' else 0

    pin_memory = (device == 'cuda')
    # prefetch=2 keeps the page-locked host memory small (workers x prefetch batches are pinned);
    # GC stalls are handled by gc.freeze() in _worker_init_fn
    prefetch = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory, persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch, worker_init_fn=_worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory, persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch, worker_init_fn=_worker_init_fn
    )

    # Likewise exclude the loaded datasets from GC tracking in the main process
    import gc
    gc.collect()
    gc.freeze()

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # TensorBoard run name (timestamp + optional name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.run_name:
        run_name = f"{timestamp}_{args.run_name}"
    else:
        run_name = timestamp
    log_dir = work_dir / 'tensorboard' / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard log dir: {log_dir}")

    # Start epoch
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif args.resume:
        # Continue after the checkpoint's epoch (taken from the first variant)
        resume_path = work_dir / f'{args.resume}_model_{variant_list[0]}.pth'
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resuming from epoch {start_epoch}")

    # Fast-forward the LR schedulers to start_epoch (cosine applies the decay of the epochs
    # already done; plateau keeps no per-epoch state)
    if start_epoch > 0:
        for ctx in contexts:
            for e in range(start_epoch):
                if e < args.warmup_epochs:
                    ctx.warmup_scheduler.step()
                elif ctx.scheduler_type == 'cosine':
                    ctx.plateau_scheduler.step()
        for ctx in contexts:
            print(f"  [{ctx.name}] Resumed LR: {ctx.optimizer.param_groups[0]['lr']:.2e}")

    # Bbox jittering
    jitter_prob = 0.0 if args.no_jitter else args.jitter_prob
    jitter_shift = args.jitter_shift
    jitter_scale = args.jitter_scale
    print(f"  Bbox Jittering: prob={jitter_prob:.0%}, shift={jitter_shift:.0%}, scale={jitter_scale:.0%}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics_dict = train_one_epoch_dual(
            contexts=contexts,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            device_type=device,
            jitter_prob=jitter_prob,
            jitter_shift=jitter_shift,
            jitter_scale=jitter_scale,
            use_multiscale=use_multiscale,
            base_img_size=base_img_size
        )

        # Print and log the training metrics of each variant
        for ctx in contexts:
            v = ctx.name
            m = train_metrics_dict[v]
            current_lr = ctx.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch} [{v}] train: loss={m['loss']:.4f}, "
                  f"simcc={m['simcc_loss']:.4f}, "
                  f"bone={m['bone_loss']:.4f}, "
                  f"coord={m['coord_loss']:.4f}, "
                  f"body_simcc={m['body_simcc_loss']:.4f}, "
                  f"body_bbox={m['body_bbox_loss']:.4f}, "
                  f"lr={current_lr:.2e}, "
                  f"gt_bbox={m['gt_bbox_prob']:.0%}, "
                  f"matched={m['matched']}/{m['body_matched']}")

            writer.add_scalar(f'Loss/{v}/train_total', m['loss'], epoch)
            writer.add_scalars(f'HeadBbox/{v}/train', {
                'giou': m['head_bbox_loss'],
                'l1': m['head_bbox_l1_loss']
            }, epoch)
            writer.add_scalar(f'Loss/{v}/train_simcc', m['simcc_loss'], epoch)
            writer.add_scalars(f'SimCC_XYZ/{v}/train', {
                'x': m['simcc_x'],
                'y': m['simcc_y'],
                'z': m['simcc_z']
            }, epoch)
            writer.add_scalar(f'Loss/{v}/train_bone', m['bone_loss'], epoch)
            writer.add_scalar(f'Loss/{v}/train_bone_direction', m['bone_direction_loss'], epoch)
            writer.add_scalar(f'Loss/{v}/train_coord', m['coord_loss'], epoch)
            writer.add_scalar(f'Loss/{v}/train_image_coord', m['image_coord_loss'], epoch)
            writer.add_scalar(f'Loss/{v}/train_consistency', m['consistency_loss'], epoch)
            writer.add_scalar(f'Loss/{v}/train_visibility', m['visibility_loss'], epoch)
            writer.add_scalar(f'Metrics/{v}/train_matched', m['matched'], epoch)
            writer.add_scalar(f'Hyperparams/{v}/learning_rate', current_lr, epoch)
            writer.add_scalar(f'Hyperparams/{v}/gt_bbox_prob', m['gt_bbox_prob'], epoch)
            writer.add_scalar(f'Matching/{v}/hungarian', m['hungarian_match'], epoch)
            # Body
            writer.add_scalar(f'Body/{v}/train_bbox', m['body_bbox_loss'], epoch)
            writer.add_scalar(f'Body/{v}/train_simcc', m['body_simcc_loss'], epoch)
            writer.add_scalar(f'Body/{v}/train_bone', m['body_bone_loss'], epoch)
            writer.add_scalar(f'Body/{v}/train_coord', m['body_coord_loss'], epoch)
            writer.add_scalar(f'Body/{v}/train_visibility', m['body_visibility_loss'], epoch)
            writer.add_scalar(f'Loss/{v}/train_in_frame', m['in_frame_loss'], epoch)
            writer.add_scalar(f'Metrics/{v}/train_body_matched', m['body_matched'], epoch)
            # Stage 2
            writer.add_scalar(f'Mask/{v}/train_mask', m['mask_loss'], epoch)
            writer.add_scalar(f'Mask/{v}/train_depth', m['depth_loss'], epoch)

        writer.flush()

        # Validate (with the EMA weights swapped in when EMA is enabled)
        ema_applied = []
        for ctx in contexts:
            if ctx.ema is not None:
                ctx.ema.apply_to(trainable_modules(ctx.model))
                ema_applied.append(ctx)
        try:
            val_metrics_dict = validate_dual(
                contexts=contexts,
                dataloader=val_loader,
                device=device,
                device_type=device
            )
        finally:
            for ctx in ema_applied:
                ctx.ema.restore(trainable_modules(ctx.model))

        # Print and log the validation metrics, step the schedulers and save checkpoints
        for ctx in contexts:
            v = ctx.name
            vm = val_metrics_dict[v]
            tm = train_metrics_dict[v]

            print(f"Epoch {epoch} [{v}] val: loss={vm['loss']:.4f}, "
                  f"simcc={vm['simcc_loss']:.4f}, "
                  f"error_unified={vm['error_unified']:.4f}, "
                  f"body_err={vm['body_error_unified']:.4f}, "
                  f"conf={vm['confidence']:.3f}, "
                  f"matched={vm['matched']}/{vm['body_matched']}")

            writer.add_scalar(f'Loss/{v}/val_total', vm['loss'], epoch)
            writer.add_scalars(f'HeadBbox/{v}/val', {
                'giou': vm['head_bbox_loss'],
                'l1': vm['head_bbox_l1_loss']
            }, epoch)
            writer.add_scalar(f'Loss/{v}/val_simcc', vm['simcc_loss'], epoch)
            writer.add_scalars(f'SimCC_XYZ/{v}/val', {
                'x': vm['simcc_x'],
                'y': vm['simcc_y'],
                'z': vm['simcc_z']
            }, epoch)
            writer.add_scalar(f'Loss/{v}/val_bone', vm['bone_loss'], epoch)
            writer.add_scalar(f'Loss/{v}/val_coord', vm['coord_loss'], epoch)
            writer.add_scalar(f'Loss/{v}/val_visibility', vm['visibility_loss'], epoch)
            writer.add_scalar(f'Metrics/{v}/val_error_unified', vm['error_unified'], epoch)
            writer.add_scalar(f'Metrics/{v}/val_confidence', vm['confidence'], epoch)
            writer.add_scalar(f'Metrics/{v}/val_matched', vm['matched'], epoch)
            # Body
            writer.add_scalar(f'Body/{v}/val_bbox', vm['body_bbox_loss'], epoch)
            writer.add_scalar(f'Body/{v}/val_simcc', vm['body_simcc_loss'], epoch)
            writer.add_scalar(f'Body/{v}/val_bone', vm['body_bone_loss'], epoch)
            writer.add_scalar(f'Body/{v}/val_coord', vm['body_coord_loss'], epoch)
            writer.add_scalar(f'Metrics/{v}/val_body_error_unified', vm['body_error_unified'], epoch)
            writer.add_scalar(f'Loss/{v}/val_in_frame', vm['in_frame_loss'], epoch)
            writer.add_scalar(f'Metrics/{v}/val_body_matched', vm['body_matched'], epoch)
            # Stage 2
            writer.add_scalar(f'Mask/{v}/val_mask', vm['mask_loss'], epoch)
            writer.add_scalar(f'Mask/{v}/val_depth', vm['depth_loss'], epoch)

            # LR schedulers
            if epoch < args.warmup_epochs:
                ctx.warmup_scheduler.step()
            elif ctx.scheduler_type == 'cosine':
                ctx.plateau_scheduler.step()
            else:
                ctx.plateau_scheduler.step(vm['loss'])

            # Best model
            is_best = vm['loss'] < ctx.best_val_loss
            if is_best:
                ctx.best_val_loss = vm['loss']

            save_checkpoint(
                ctx=ctx,
                work_dir=work_dir,
                epoch=epoch,
                args=args,
                train_metrics=tm,
                val_metrics=vm,
                is_best=is_best,
                jitter_prob=jitter_prob,
                jitter_shift=jitter_shift,
                jitter_scale=jitter_scale,
                use_multiscale=use_multiscale
            )

        writer.flush()

    # Final model (EMA weights in the main slots when EMA is enabled)
    for ctx in contexts:
        final_path = work_dir / f'final_model_{ctx.name}.pth'
        module_names = trainable_module_names(ctx.model)
        if ctx.ema is not None:
            main_sds = ctx.ema.state_dicts()
        else:
            main_sds = [getattr(ctx.model, n).state_dict() for n in module_names]
        final_ckpt = {
            'version': CODE_VERSION,
            'epoch': args.epochs - 1,
            'optimizer_state_dict': ctx.optimizer.state_dict(),
            'best_val_loss': ctx.best_val_loss,
            'config': checkpoint_config(args, jitter_prob, jitter_shift, jitter_scale, use_multiscale),
        }
        for name, sd in zip(module_names, main_sds):
            final_ckpt[f'{name}_state_dict'] = sd
        if ctx.ema is not None:
            final_ckpt['raw_state_dicts'] = {
                n: getattr(ctx.model, n).state_dict() for n in module_names
            }
        torch.save(final_ckpt, final_path)
        print(f"  [{ctx.name}] Saved final model: {final_path}")

    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"  View with: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()

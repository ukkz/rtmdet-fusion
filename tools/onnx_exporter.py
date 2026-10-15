#!/usr/bin/env python3
"""RTMDet-Fusion ONNX Exporter

Exports the model to ONNX as a single end-to-end graph: image in, every output
(bboxes, scores, masks, keypoints, part masks, depth, ...) out.

Output layout:
- Fixed MAX_INSTANCES slots per output; `valid_count` gives the number of valid detections
- Confidences, visibilities, probabilities and depth of unused slots are 0.0
- Names and order follow export_contract.OUTPUT_NAMES (see docs/output-contract.md)
- Requires a stage-2 checkpoint (body heads and mask/depth branch)

Usage:
    # All multi-scale sizes (256, 320, 416, 512, 640)
    uv run python tools/onnx_exporter.py --checkpoint work_dirs/main/best_model_s.pth

    # A single size
    uv run python tools/onnx_exporter.py --checkpoint work_dirs/main/best_model_s.pth --img-size 416

    # With verification against PyTorch
    uv run python tools/onnx_exporter.py --checkpoint work_dirs/main/best_model_s.pth --verify
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

ONNX_MODELS_DIR = Path(__file__).resolve().parent.parent / 'onnx_models'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.config import ConfigDict
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

from configs.default import (
    NUM_KEYPOINTS, MAX_INSTANCES, IMG_SIZE, MULTISCALE_SIZES,
    PERSON_CLASS_ID, FPN_STRIDES, GRID_SIZE, NUM_BODY_KEYPOINTS,
    get_rtmdet_config, get_rtmdet_weights, get_rtmdet_in_channels,
)

from rtmdet_fusion.models.fusion_model import (
    BboxFeatureExtractor, HeadBboxHead, DCCHead, MaskDepthBranch,
    compute_head_quaternion, compute_appearance_features, compute_pose_features
)
from rtmdet_fusion.models.encoders import HybridEncoder
from rtmdet_fusion.utils.versioning import (
    check_checkpoint_version, version_filename_part,
)
from rtmdet_fusion.utils.export_contract import (
    OUTPUT_NAMES, PART_MASK_OUTPUT_DIVISOR, EXPORT_SCORE_THR, export_settings_metadata,
    v2_head_config,
)

# Number of candidates kept before the static NMS (TopK)
NMS_PRE_TOPK = 100


# ============================================================
# End-to-end model for ONNX
# ============================================================

class RTMDetFusionONNX(nn.Module):
    """End-to-end model for ONNX export.

    Input: image (1, 3, img_size, img_size), any multi-scale size
    Output: fixed MAX_INSTANCES slots of detections (see forward)
    """

    def __init__(
        self,
        rtmdet_cfg: ConfigDict,
        head_cfg: Dict,
        num_classes: int = 80,
        score_threshold: float = EXPORT_SCORE_THR,
        iou_threshold: float = 0.6,
        max_instances: int = MAX_INSTANCES,
        img_size: int = IMG_SIZE,
        in_channels: int = 128,
    ):
        """
        Args:
            rtmdet_cfg: RTMDet-Ins config
            head_cfg: return value of v2_head_config()
            in_channels: RTMDet-Ins backbone output channels (tiny: 96, s: 128)
        """
        super().__init__()

        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_instances = max_instances
        self.img_size = img_size

        # RTMDet-Ins
        self.rtmdet = MODELS.build(rtmdet_cfg)

        # Backbone output channels (C3, C4, C5)
        backbone_channels = [in_channels, in_channels * 2, in_channels * 4]

        # Hybrid Encoder
        self.hybrid_encoder = HybridEncoder(
            in_channels=backbone_channels,
            hidden_dim=256,
            out_channels=256,
            num_csp_blocks=3,
            widen_factor=0.5
        )

        # Bbox feature extraction. Uses the matrix-multiplication sampling: numerically
        # equivalent to F.grid_sample but needs no GridSample / gather op, which converts
        # more cleanly and runs faster in TensorFlow.js
        self.bbox_feat_extractor = BboxFeatureExtractor(
            in_channels=256,
            out_channels=256,
            grid_size=GRID_SIZE
        )
        self.bbox_feat_extractor.matmul_sampling = True

        self.head_bbox_head = HeadBboxHead(
            in_channels=256,
            hidden_dim=256,
            num_layers=2
        )

        # Head keypoints (unified coordinates, margin bins)
        self.dcc_head = DCCHead(
            in_channels=256,
            num_keypoints=NUM_KEYPOINTS,
            num_bins=head_cfg['num_bins'],
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            target_sigma=0.025,
            bin_margin=head_cfg['bin_margin'],
        )

        # Body reference bbox head + body keypoint head
        self.body_bbox_head = HeadBboxHead(
            in_channels=256,
            hidden_dim=256,
            num_layers=2,
            range_scale=head_cfg['body_bbox_range_scale'],
        )
        self.body_kpt_head = DCCHead(
            in_channels=256,
            num_keypoints=NUM_BODY_KEYPOINTS,
            num_bins=head_cfg['num_bins_body'],
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            target_sigma=0.025,
            bin_margin=head_cfg['bin_margin'],
        )

        # Part mask / depth branch (output resolution img_size / PART_MASK_OUTPUT_DIVISOR)
        self.mask_depth_branch = MaskDepthBranch()
        self.part_mask_size = img_size // PART_MASK_OUTPUT_DIVISOR

        # FPN strides for prior generation
        self.strides = FPN_STRIDES

        # Register image normalization parameters
        self.register_buffer('mean', torch.tensor([103.53, 116.28, 123.675]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([57.375, 57.12, 58.395]).view(1, 3, 1, 1))

    def _normalize_image(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize the image (BGR input)."""
        return (images - self.mean) / self.std

    def _decode_bboxes(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        kernel_preds: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode the RTMDet bbox predictions.

        Returns:
            bboxes: (N, 4) [x1, y1, x2, y2]
            scores: (N,)
            priors: (N, 4) [cx, cy, stride, stride]
            kernels: (N, num_gen_params) mask generation kernels
        """
        device = cls_scores[0].device
        all_bboxes = []
        all_scores = []
        all_priors = []
        all_kernels = []

        for level_idx, stride in enumerate(self.strides):
            cls_score = cls_scores[level_idx][0]  # (num_classes, H, W)
            bbox_pred = bbox_preds[level_idx][0]  # (4, H, W)
            kernel_pred = kernel_preds[level_idx][0]  # (num_gen_params, H, W)

            H, W = cls_score.shape[1], cls_score.shape[2]

            # Priors (RTMDet uses offset=0)
            shifts_x = torch.arange(0, W, device=device) * stride
            shifts_y = torch.arange(0, H, device=device) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # Priors: [cx, cy, stride, stride] (as float)
            priors = torch.stack([
                shift_x.float(), shift_y.float(),
                torch.full_like(shift_x, stride, dtype=torch.float32),
                torch.full_like(shift_x, stride, dtype=torch.float32)
            ], dim=-1)

            # Flatten predictions
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            kernel_pred = kernel_pred.permute(1, 2, 0).reshape(-1, kernel_pred.shape[0])

            # Person class score
            person_scores = cls_score[:, PERSON_CLASS_ID].sigmoid()

            # Bbox decoding (RTMDet outputs are already in pixels)
            x1 = shift_x - bbox_pred[:, 0]
            y1 = shift_y - bbox_pred[:, 1]
            x2 = shift_x + bbox_pred[:, 2]
            y2 = shift_y + bbox_pred[:, 3]
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            all_bboxes.append(bboxes)
            all_scores.append(person_scores)
            all_priors.append(priors)
            all_kernels.append(kernel_pred)

        return (
            torch.cat(all_bboxes, dim=0),
            torch.cat(all_scores, dim=0),
            torch.cat(all_priors, dim=0),
            torch.cat(all_kernels, dim=0)
        )

    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """IoU with tensor ops only.

        Args:
            boxes1: (N, 4) x1, y1, x2, y2
            boxes2: (M, 4) x1, y1, x2, y2

        Returns:
            iou: (N, M) IoU matrix
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

        # Intersection
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

        # IoU
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-6)
        return iou

    def _nms(
        self,
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        priors: torch.Tensor,
        kernels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Static NMS with a fixed-size output (no ONNX NonMaxSuppression / ScatterND op).

        Greedily selects max_instances boxes:
        1. Keep the top NMS_PRE_TOPK candidates by score
        2. Select the highest-scoring box (as a one-hot mask)
        3. Suppress boxes whose IoU with it exceeds the threshold
        4. Repeat 2-3 max_instances times
        5. Gather the results by matrix products with the one-hot masks

        Always returns (max_instances, ...) tensors; zeros when nothing is detected.
        """
        device = bboxes.device
        MAX_PRE_NMS = NMS_PRE_TOPK

        # TopK (descending score)
        topk_scores, topk_indices = torch.topk(scores, MAX_PRE_NMS, sorted=True)
        topk_bboxes = bboxes[topk_indices]
        topk_priors = priors[topk_indices]
        topk_kernels = kernels[topk_indices]

        # Valid mask (score >= threshold)
        valid_mask = (topk_scores >= self.score_threshold).float()

        # IoU matrix
        iou_matrix = self._box_iou(topk_bboxes, topk_bboxes)  # (MAX_PRE_NMS, MAX_PRE_NMS)

        # One-hot selection vector of each iteration
        selection_masks = []

        # Accumulated suppression mask
        suppressed = torch.zeros(MAX_PRE_NMS, device=device)

        # Greedy selection (fixed number of iterations)
        for k in range(self.max_instances):
            # Available score = score x valid mask x (1 - suppressed)
            available_scores = topk_scores * valid_mask * (1.0 - suppressed)

            # Index of the best score
            best_idx = torch.argmax(available_scores)

            # One-hot mask (avoids ScatterND).
            # ONNX OneHot requires rank >= 1, hence reshape(1)
            one_hot = F.one_hot(best_idx.reshape(1), MAX_PRE_NMS).float().reshape(-1)  # (MAX_PRE_NMS,)

            # Selected score (without indexing)
            best_score = (one_hot * available_scores).sum()

            # Zero the one-hot vector when no valid box is left,
            # so that an empty selection does not affect the outputs
            score_valid = (best_score > 0).float()
            one_hot = one_hot * score_valid

            selection_masks.append(one_hot)

            # IoU with the selected box by matrix product (iou_matrix[best_idx] without indexing)
            iou_with_best = torch.matmul(one_hot.reshape(1, -1), iou_matrix).reshape(-1)  # (MAX_PRE_NMS,)

            # Suppress the selected box itself and boxes with IoU > threshold (the selected box is
            # suppressed explicitly: its self-IoU may not exceed the threshold, e.g. threshold >= 1)
            new_suppressed = (iou_with_best > self.iou_threshold).float()
            suppressed = torch.clamp(suppressed + new_suppressed + one_hot, 0.0, 1.0)

        # Stack the selections: (max_instances, MAX_PRE_NMS)
        selection_matrix = torch.stack(selection_masks, dim=0)

        # Gather by matrix products (avoids ScatterND):
        # out[k] = sum_i(selection_matrix[k, i] * topk[i]).
        # topk_scores is made 2D explicitly; some runtimes do not support MatMul(2D, 1D)
        out_bboxes = torch.matmul(selection_matrix, topk_bboxes)  # (max_instances, 4)
        out_scores = torch.matmul(selection_matrix, topk_scores.reshape(MAX_PRE_NMS, 1)).reshape(self.max_instances)  # (max_instances,)
        out_priors = torch.matmul(selection_matrix, topk_priors)  # (max_instances, 4)
        out_kernels = torch.matmul(selection_matrix, topk_kernels)  # (max_instances, kernel_dim)

        return out_bboxes, out_scores, out_priors, out_kernels

    def _generate_masks(
        self,
        mask_feat: torch.Tensor,
        kernels: torch.Tensor,
        priors: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Instance mask generation (follows MMDet rtmdet_ins_head.py).

        Based on MMDet _mask_predict_by_feat_single. Points to note:
        - mask_feat has num_prototypes=8 channels (not 256)
        - relative coordinates are (prior - coord) / (stride * 8)
        - kernel split order is [w1, w2, w3, b1, b2, b3]
        """
        device = mask_feat.device
        B, num_prototypes, H, W = mask_feat.shape  # num_prototypes=8
        N = kernels.shape[0]  # always max_instances (the NMS output is fixed-size)

        # Dynamic conv settings (RTMDet-Ins)
        dyconv_channels = 8  # mid_channels

        # Kernel parameter sizes
        # weight_nums = [(num_prototypes + 2) * dyconv_channels, dyconv_channels * dyconv_channels, dyconv_channels * 1]
        # bias_nums = [dyconv_channels, dyconv_channels, 1]
        in_channels = num_prototypes + 2  # 8 + 2 = 10
        w1_size = in_channels * dyconv_channels      # 10 * 8 = 80
        w2_size = dyconv_channels * dyconv_channels  # 8 * 8 = 64
        w3_size = dyconv_channels * 1                # 8 * 1 = 8
        b1_size = dyconv_channels                    # 8
        b2_size = dyconv_channels                    # 8
        b3_size = 1                                  # 1

        # Kernel split (MMDet: all weights first, then all biases)
        # Order: [w1, w2, w3, b1, b2, b3]
        idx = 0
        w1 = kernels[:, idx:idx+w1_size]
        idx += w1_size
        w2 = kernels[:, idx:idx+w2_size]
        idx += w2_size
        w3 = kernels[:, idx:idx+w3_size]
        idx += w3_size
        b1 = kernels[:, idx:idx+b1_size]
        idx += b1_size
        b2 = kernels[:, idx:idx+b2_size]
        idx += b2_size
        b3 = kernels[:, idx:idx+b3_size]

        # Coordinate grid at the mask_feat scale
        # (MMDet uses single_level_grid_priors)
        stride = self.strides[0]  # mask_feat corresponds to stride 8
        y_coords = torch.arange(H, device=device).float() * stride
        x_coords = torch.arange(W, device=device).float() * stride
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coord = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
        coord = coord.view(1, H * W, 2)  # (1, H*W, 2)

        # Priors
        points = priors[:, :2].view(N, 1, 2)      # (N, 1, 2) prior center [x, y]
        # (N, 1, 1) stride. Unused slots have prior=0; clamp to 1 to avoid 0-division NaN
        strides = priors[:, 2:3].view(N, 1, 1).clamp(min=1.0)

        # Relative coordinates (MMDet: (prior - coord) / (stride * 8))
        # points: (N, 1, 2), coord: (1, H*W, 2) -> (N, H*W, 2)
        relative_coord = (points - coord) / (strides * 8)  # (N, H*W, 2)
        coord_flat = relative_coord.permute(0, 2, 1)       # (N, 2, H*W)
        feat_flat = mask_feat[0:1].expand(N, -1, -1, -1).reshape(N, num_prototypes, H * W)

        # Dynamic convolution with bmm (equivalent to F.conv2d with groups=N).
        # MMDet concatenates [relative_coord 2ch, mask_feat 8ch]; here the weights are split
        # and two bmm results are summed instead (mathematically equivalent). onnx2tf converts
        # a concat of a conv output and a computed tensor with mismatched NHWC layouts and fails
        # on it

        # Layer 1: (N, 2+8, H*W) -> (N, 8, H*W)
        w1 = w1.view(N, dyconv_channels, in_channels)  # (N, 8, 10)
        b1 = b1.view(N, dyconv_channels, 1)            # (N, 8, 1)
        x = torch.bmm(w1[:, :, :2], coord_flat) + torch.bmm(w1[:, :, 2:], feat_flat) + b1
        x = F.relu(x)

        # Layer 2: (N, 8, H*W) -> (N, 8, H*W)
        w2 = w2.view(N, dyconv_channels, dyconv_channels)  # (N, 8, 8)
        b2 = b2.view(N, dyconv_channels, 1)                # (N, 8, 1)
        x = torch.bmm(w2, x) + b2
        x = F.relu(x)

        # Layer 3: (N, 8, H*W) -> (N, 1, H*W)
        w3 = w3.view(N, 1, dyconv_channels)  # (N, 1, 8)
        b3 = b3.view(N, 1, 1)                # (N, 1, 1)
        x = torch.bmm(w3, x) + b3

        # Reshape to masks
        x = x.view(N, H, W)

        # Resize to the input image size (x stride)
        x = F.interpolate(
            x.reshape(N, 1, H, W),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=True
        ).reshape(N, self.img_size, self.img_size)

        # Smoothstep: saturates to 0/1 at +-8, smooth near-linear edge around the center
        t = torch.clamp((x + 8.0) / 16.0, 0, 1)
        masks = t * t * (3 - 2 * t)

        # Zero masks for invalid instances
        masks = masks * valid_mask.view(N, 1, 1).float()

        return masks

    def _process_instances(
        self,
        hybrid_feats: List[torch.Tensor],
        bboxes: torch.Tensor,
        priors: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Per-instance heads for all max_instances slots (same flow as run_inference in tools/viewer.py).

        Args:
            hybrid_feats: Hybrid Encoder outputs
            bboxes: (N, 4) detected bboxes
            priors: (N, 4) priors
            valid_mask: (N,) valid mask

        Returns: dict with the keys below (unused slots are zeroed in forward)
            head_bboxes: (N, 4) head bbox (expanded; reference of the keypoint coordinates),
                may extend outside the body bbox or the image (negative coordinates)
            keypoints: (N, NUM_KEYPOINTS, 3) unified-coordinate keypoints [rel_x, rel_y, rel_z]
                rel_x/rel_z are relative to head_bboxes ([-0.125, 1.125] with margin bins),
                rel_y is depth (center=0.5)
            confidences: (N,) keypoint confidence (mean over the keypoints)
            visibilities / in_frame: (N, NUM_KEYPOINTS) visibility / in-frame probability [0,1]
            head_quaternions: (N, 4) head pose quaternion [w, x, y, z]
            appearance_features / pose_features: (N, 256) L2-normalized features
            body_bboxes: (N, 4) body reference bbox (expanded; reference of the body keypoints)
            body_keypoints: (N, NUM_BODY_KEYPOINTS, 3) body unified coordinates [rel_x, rel_y, rel_z]
            body_confidences / body_visibilities / body_in_frame: (N, NUM_BODY_KEYPOINTS)
            part_masks: (N, 7, S/4, S/4) mask probabilities (clothed / bare body / 4 parts / occlusion)
            depth: (N, S/4, S/4) depth (rel_y scale: 0.5 = pelvis, normalized by the body reference bbox size)
        """
        # Head bbox
        grid_features, pos_encoding = self.bbox_feat_extractor(
            hybrid_feats, bboxes, priors, valid_mask, self.img_size
        )
        head_bboxes = self.head_bbox_head(grid_features, bboxes)

        # Expanded head bbox; keypoints are predicted relative to it
        head_bboxes_expanded = self.dcc_head.expand_bbox(head_bboxes)

        # Head keypoints
        head_grid_features, head_pos_encoding = self.bbox_feat_extractor(
            hybrid_feats, head_bboxes_expanded, priors, valid_mask, self.img_size
        )
        dcc_output = self.dcc_head(head_grid_features, head_pos_encoding)

        # Decode keypoints (relative coordinates)
        keypoints_rel, confidences_per_kpt, visibilities = self.dcc_head.decode_relative(
            dcc_output['heatmap_x'],
            dcc_output['heatmap_y'],
            dcc_output['heatmap_z'],
            dcc_output['visibility_logits']
        )

        # Mean confidence over the keypoints
        confidences = confidences_per_kpt.mean(dim=-1)  # (N,)

        # === Tracking features ===
        # Head pose quaternion
        head_quaternions = compute_head_quaternion(keypoints_rel)

        # Appearance feature (mean-pooled head bbox grid features + L2 normalization)
        appearance_features = compute_appearance_features(head_grid_features)

        # Pose feature (mean-pooled keypoint features + L2 normalization)
        pose_features = compute_pose_features(dcc_output['kpt_features'])

        # === Body reference bbox + body keypoints ===
        body_bboxes = self.body_bbox_head(grid_features, bboxes)
        body_bboxes_expanded = self.body_kpt_head.expand_bbox(body_bboxes)
        body_grid_features, body_pos_encoding = self.bbox_feat_extractor(
            hybrid_feats, body_bboxes_expanded, priors, valid_mask, self.img_size
        )
        body_output = self.body_kpt_head(body_grid_features, body_pos_encoding)
        body_keypoints, body_confidences, body_visibilities = self.body_kpt_head.decode_relative(
            body_output['heatmap_x'],
            body_output['heatmap_y'],
            body_output['heatmap_z'],
            body_output['visibility_logits']
        )

        # === Part masks / depth (stride 8 -> S/4, same interpolation as the training targets) ===
        inst_feats = grid_features.mean(dim=1)  # (N, 256)
        mask_out = self.mask_depth_branch(hybrid_feats, inst_feats, priors)  # (N, 8, S/8, S/8)
        mask_out = F.interpolate(
            mask_out, size=(self.part_mask_size, self.part_mask_size),
            mode='bilinear', align_corners=False
        )

        # Image coordinates from head_bboxes / body_bboxes:
        #   img_x = rel_x * (x2 - x1) + x1
        #   img_y = (1 - rel_z) * (y2 - y1) + y1  # z points up, hence the flip
        return {
            'head_bboxes': head_bboxes_expanded,
            'keypoints': keypoints_rel,
            'confidences': confidences,
            'visibilities': visibilities,
            'in_frame': torch.sigmoid(dcc_output['in_frame_logits']),
            'head_quaternions': head_quaternions,
            'appearance_features': appearance_features,
            'pose_features': pose_features,
            'body_bboxes': body_bboxes_expanded,
            'body_keypoints': body_keypoints,
            'body_confidences': body_confidences,
            'body_visibilities': body_visibilities,
            'body_in_frame': torch.sigmoid(body_output['in_frame_logits']),
            'part_masks': torch.sigmoid(mask_out[:, :MaskDepthBranch.NUM_MASK_CHANNELS]),
            'depth': mask_out[:, MaskDepthBranch.NUM_MASK_CHANNELS],
        }

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """End-to-end inference.

        Args:
            images: (1, 3, H, W) BGR, not normalized

        Returns: tuple in OUTPUT_NAMES order. See _process_instances for the per-instance outputs
            (unused slots: probabilities, features and depth are 0.0)
            bboxes: (max_instances, 4) person bbox [x1, y1, x2, y2]
            scores: (max_instances,) detection score
            masks: (max_instances, img_size, img_size) instance mask
            head_bboxes: (max_instances, 4) head bbox (expanded; reference of the keypoints),
                may extend outside the body bbox or the image (negative coordinates)
            keypoints: (max_instances, NUM_KEYPOINTS, 3) unified-coordinate keypoints [rel_x, rel_y, rel_z]
                rel_x/rel_z are relative to head_bboxes ([-0.125, 1.125] with margin bins)
            confidences: (max_instances,) keypoint confidence (0.0 for unused slots)
            visibilities: (max_instances, NUM_KEYPOINTS) visibility (0.0 for unused slots)
            valid_count: (1,) number of valid detections
            head_quaternions: (max_instances, 4) head pose quaternion [w, x, y, z]
                reference pose (q=[1,0,0,0]): front=-Y, up=+Z, right=-X;
                [0, 0, 0, 0] for unused slots
            appearance_features: (max_instances, 256) appearance feature (L2-normalized),
                zero vector for unused slots
            pose_features: (max_instances, 256) pose feature (L2-normalized),
                zero vector for unused slots
        """
        # ONNX export assumes batch_size=1

        # === Step 1: Image normalization ===
        images_norm = self._normalize_image(images)

        # === Step 2: Backbone ===
        backbone_feats = self.rtmdet.backbone(images_norm)

        # === Step 3: PAFPN ===
        pafpn_feats = self.rtmdet.neck(backbone_feats)

        # === Step 4: RTMDet-Ins head ===
        cls_scores, bbox_preds, kernel_preds, mask_feat = self.rtmdet.bbox_head(pafpn_feats)

        # === Step 5: Bbox decoding + NMS ===
        all_bboxes, all_scores, all_priors, all_kernels = self._decode_bboxes(
            cls_scores, bbox_preds, kernel_preds
        )
        bboxes, scores, priors, kernels = self._nms(
            all_bboxes, all_scores, all_priors, all_kernels
        )

        # Clip bboxes to the image
        bboxes = torch.clamp(bboxes, min=0, max=self.img_size)

        # Valid mask: score > threshold
        valid_mask = scores > self.score_threshold
        valid_count = valid_mask.sum()

        # === Step 6: Instance masks ===
        masks = self._generate_masks(mask_feat, kernels, priors, valid_mask)

        # === Step 7: Hybrid Encoder ===
        hybrid_feats = self.hybrid_encoder(backbone_feats)

        # === Step 8: Per-instance heads (always all max_instances slots) ===
        inst = self._process_instances(hybrid_feats, bboxes, priors, valid_mask)

        # === Step 9: Zero the unused slots ===
        # Unused slots still produce nonzero confidences (softmax over a flat distribution gives
        # ~0.008), so zero them explicitly. Coordinates (*_bboxes / *keypoints) are left as is;
        # use valid_count to tell valid slots
        vm = valid_mask.float()
        inst['confidences'] = inst['confidences'] * vm
        for key in ('visibilities', 'in_frame', 'head_quaternions', 'appearance_features',
                    'pose_features', 'body_confidences', 'body_visibilities', 'body_in_frame'):
            inst[key] = inst[key] * vm.reshape(-1, 1)
        inst['part_masks'] = inst['part_masks'] * vm.reshape(-1, 1, 1, 1)
        inst['depth'] = inst['depth'] * vm.reshape(-1, 1, 1)

        outputs = {
            'bboxes': bboxes,
            'scores': scores,
            'masks': masks,
            'valid_count': valid_count.reshape(1).to(torch.int32),
            **inst,
        }
        return tuple(outputs[name] for name in OUTPUT_NAMES)


# ============================================================
# Variant detection and weight loading
# ============================================================

# (model attribute, checkpoint key) of the trained modules
TRAINED_MODULES = (
    ('hybrid_encoder', 'hybrid_encoder_state_dict'),
    ('bbox_feat_extractor', 'bbox_feat_extractor_state_dict'),
    ('head_bbox_head', 'head_bbox_head_state_dict'),
    ('dcc_head', 'dcc_head_state_dict'),
    ('body_bbox_head', 'body_bbox_head_state_dict'),
    ('body_kpt_head', 'body_kpt_head_state_dict'),
    ('mask_depth_branch', 'mask_depth_branch_state_dict'),
)


def detect_variant_from_checkpoint(checkpoint: dict) -> str:
    """Detect the RTMDet variant from the checkpoint tensor shapes.

    Uses the shape of hybrid_encoder input_proj.0.0.weight:
    - [256, 96, 1, 1] -> tiny
    - [256, 128, 1, 1] -> s
    """
    state_dict = checkpoint.get('hybrid_encoder_state_dict', {})
    if 'input_proj.0.0.weight' not in state_dict:
        raise ValueError("Cannot detect the RTMDet variant: "
                         "hybrid_encoder_state_dict['input_proj.0.0.weight'] is missing")

    # input_proj.0.0.weight: [out_channels, in_channels, 1, 1]
    in_channels = state_dict['input_proj.0.0.weight'].shape[1]
    variants = {96: 'tiny', 128: 's'}
    if in_channels not in variants:
        raise ValueError(f"Cannot detect the RTMDet variant: unknown in_channels={in_channels}")
    return variants[in_channels]


def load_weights(
    model: RTMDetFusionONNX,
    rtmdet_weights: str,
    checkpoint: dict
) -> None:
    """Load the weights (the checkpoint is already loaded)."""
    missing = [key for _, key in TRAINED_MODULES if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing modules: {missing}")

    # RTMDet-Ins weights (strict=False: the MMDet checkpoint also holds non-model entries)
    if not Path(rtmdet_weights).exists():
        raise FileNotFoundError(f"RTMDet-Ins weights not found: {rtmdet_weights}")
    ckpt = torch.load(rtmdet_weights, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.rtmdet.load_state_dict(state_dict, strict=False)
    print(f"Loaded RTMDet weights: {rtmdet_weights}")

    # Trained modules are loaded strictly: silently ignoring mismatched keys would export
    # untrained weights
    for attr, key in TRAINED_MODULES:
        getattr(model, attr).load_state_dict(checkpoint[key])
    print(f"Loaded {', '.join(attr for attr, _ in TRAINED_MODULES)} weights")


# ============================================================
# ONNX export
# ============================================================

def embed_onnx_metadata(output_path: str, version: Optional[str],
                        settings: Optional[Dict[str, str]] = None) -> None:
    """Embed the version and export settings into the ONNX metadata_props.

    Call this last, after every step that rewrites the file (e.g. optimization).
    """
    props = dict(settings or {})
    if version:
        props['rtmdet_fusion_version'] = version
    if not props:
        return
    try:
        import onnx
        model_onnx = onnx.load(output_path)
        for key, value in props.items():
            entry = model_onnx.metadata_props.add()
            entry.key = key
            entry.value = value
        onnx.save(model_onnx, output_path)
        print(f"  Embedded metadata: {props}")
    except Exception as e:
        print(f"  Warning: Failed to embed version metadata: {e}")


def _assert_onnxscript_dynamo_ready() -> None:
    """The dynamo exporter requires onnxscript>=0.6.0.

    onnxscript<0.6.0 has a name-collision bug (pytorch#168211, `Duplicate definition of name`)
    and **silently writes an invalid ONNX** with dynamo=True (only --verify reveals it).
    pyproject.toml cannot require 0.6 because of the uv.lock, so it is enforced here.
    """
    try:
        import onnxscript
    except ImportError:
        raise RuntimeError(
            "The dynamo ONNX export requires onnxscript>=0.6.0, but onnxscript is not installed. "
            "Run `uv pip install \"onnxscript>=0.6.0\"`."
        )
    parts = onnxscript.__version__.split(".")
    major_minor = tuple(int(p) for p in parts[:2])
    if major_minor < (0, 6):
        raise RuntimeError(
            f"The dynamo ONNX export requires onnxscript>=0.6.0 "
            f"(installed: {onnxscript.__version__}); older versions produce an invalid model "
            f"because of the name-collision bug pytorch#168211. Run "
            f"`uv pip install \"onnxscript>=0.6.0\"`, then `uv run --no-sync` if the automatic "
            f"environment sync reverts it."
        )


def export_onnx(
    model: RTMDetFusionONNX,
    output_path: str,
    opset_version: int = 18,
    optimize: str = 'basic',
    version: Optional[str] = None,
    settings: Optional[Dict[str, str]] = None,
) -> None:
    """Export to ONNX.

    Args:
        model: model to export
        output_path: output path
        opset_version: ONNX opset version
        optimize: ONNX Runtime optimization level
            - 'none': no optimization
            - 'basic': constant folding and redundant node removal (default)
            - 'extended': also node fusions (use with care)
        version: version string embedded into the metadata
        settings: export settings embedded into the metadata (export_settings_metadata)
    """
    model.eval()
    device = next(model.parameters()).device
    img_size = model.img_size

    # Dummy input (BGR, not normalized, 0-255)
    dummy_input = torch.randint(0, 256, (1, 3, img_size, img_size), dtype=torch.float32, device=device)

    input_names = ['images']
    output_names = OUTPUT_NAMES

    # Check the dynamo exporter prerequisite (onnxscript>=0.6.0) before exporting
    _assert_onnxscript_dynamo_ready()

    print("Exporting to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Image size: {img_size}")
    print(f"  Opset version: {opset_version}")
    print(f"  Max instances: {model.max_instances}")
    print(f"  Num keypoints: {NUM_KEYPOINTS}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,  # fixed shapes
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
        dynamo=True,
        external_data=False,  # inline the weights into a single .onnx (no .data sidecar)
    )

    print(f"Exported to: {output_path}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    # ONNX Runtime optimization
    if optimize != 'none':
        try:
            import onnxruntime as ort

            print(f"Optimizing with ONNX Runtime (level: {optimize})...")

            opt_level_map = {
                'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            }
            opt_level = opt_level_map.get(optimize, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)

            # Save the optimized model through the session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = opt_level
            sess_options.optimized_model_filepath = output_path

            # The optimization runs and is saved when the session is created
            _ = ort.InferenceSession(output_path, sess_options, providers=['CPUExecutionProvider'])

            print(f"  Optimized successfully (level: {optimize})")

        except ImportError:
            print("  Warning: onnxruntime not installed, skipping optimization")
            print("  Install with: uv pip install onnxruntime")
        except Exception as e:
            print(f"  Warning: Optimization failed: {e}")
            print("  Keeping original model")

    # Embed the metadata (after every step that rewrites the file)
    embed_onnx_metadata(output_path, version, settings)


# ============================================================
# Verification
# ============================================================

def verify_onnx(
    onnx_path: str,
    pytorch_model: RTMDetFusionONNX,
    device: str = 'cpu'
) -> bool:
    """Compare the ONNX model outputs with PyTorch."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnx/onnxruntime not installed, skipping verification")
        return True

    print("Verifying ONNX model...")

    # ONNX model check
    try:
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print("  ONNX model check: OK")
    except Exception as e:
        print(f"  ONNX model check failed: {e}")
        return False

    # PyTorch inference
    pytorch_model.eval()
    pytorch_model.to(device)
    img_size = pytorch_model.img_size

    # Dummy input (BGR, not normalized, 0-255)
    dummy_input = torch.randint(0, 256, (1, 3, img_size, img_size), dtype=torch.float32, device=device)

    with torch.no_grad():
        pytorch_outputs = pytorch_model(dummy_input)

    # ONNX Runtime inference
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    ort_inputs = {'images': dummy_input.cpu().numpy()}
    ort_outputs = sess.run(None, ort_inputs)

    output_names = OUTPUT_NAMES

    print("  ONNX Runtime inference: OK")
    print("  Output shapes:")
    for name, output in zip(output_names, ort_outputs):
        print(f"    {name}: {output.shape}")

    all_close = True
    for i, name in enumerate(output_names):
        pytorch_out = pytorch_outputs[i].cpu().numpy()
        ort_out = ort_outputs[i]

        # equal_nan=True: masks of unused slots (beyond valid_count) can be NaN at the same
        # positions in both PyTorch and ONNX. NaNs at the same positions count as a match;
        # only shifted NaN positions or value differences are mismatches.
        if not np.allclose(pytorch_out, ort_out, rtol=1e-3, atol=1e-5, equal_nan=True):
            diff = np.abs(pytorch_out - ort_out)
            max_diff = float(np.nanmax(diff)) if diff.size else 0.0
            nan_pos_mismatch = int(np.logical_xor(np.isnan(pytorch_out), np.isnan(ort_out)).sum())
            print(f"  {name}: MISMATCH (max_diff={max_diff:.6f}, nan_pos_mismatch={nan_pos_mismatch})")
            all_close = False
        else:
            print(f"  {name}: Match OK")

    if all_close:
        print("Verification PASSED!")
    else:
        print("Verification FAILED!")

    return all_close


# ============================================================
# Main
# ============================================================

def export_single_size(
    img_size: int,
    variant: str,
    checkpoint: dict,
    rtmdet_weights: str,
    output_path: str,
    args,
    head_cfg: Dict,
    version: Optional[str] = None,
):
    """Export a single input size."""
    print(f"\n{'='*60}")
    print(f"Exporting: {output_path}")
    print(f"{'='*60}")

    rtmdet_cfg = ConfigDict(get_rtmdet_config(variant))
    in_channels = get_rtmdet_in_channels(variant)
    model = RTMDetFusionONNX(
        rtmdet_cfg=rtmdet_cfg,
        head_cfg=head_cfg,
        num_classes=80,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_instances=MAX_INSTANCES,
        img_size=img_size,
        in_channels=in_channels,
    )

    print("  Loading weights...")
    load_weights(model, rtmdet_weights, checkpoint)

    model.to(args.device)
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    export_onnx(
        model,
        output_path,
        opset_version=args.opset,
        optimize=args.optimize,
        version=version,
        settings=export_settings_metadata(
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            max_instances=model.max_instances,
            topk=NMS_PRE_TOPK,
            builtin_nms=True,
            grid_sample='matmul',
        ),
    )

    if args.verify:
        verify_onnx(output_path, model, args.device)

    # Free the model (saves memory when exporting several sizes)
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="RTMDet-Fusion ONNX Exporter")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint path (stage 2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file (only used with a single size; default: '
                             'onnx_models/rtmdet_fusion_{variant}_{size}_{version}.onnx)')
    parser.add_argument('--img-size', type=int, default=None,
                        choices=MULTISCALE_SIZES,
                        help='Input image size (default: export every multi-scale size)')
    parser.add_argument('--rtmdet-weights', type=str, default=None,
                        help='RTMDet-Ins weights (default: chosen by the detected variant)')
    parser.add_argument('--opset', type=int, default=18,
                        help='ONNX opset version (dynamo cannot emit 17, hence the default 18)')
    parser.add_argument('--verify', action='store_true',
                        help='Compare the exported model with PyTorch')
    parser.add_argument('--optimize', type=str, default='basic',
                        choices=['none', 'basic', 'extended'],
                        help='ONNX Runtime optimization level (default: basic)')
    parser.add_argument('--score-threshold', type=float, default=EXPORT_SCORE_THR,
                        help=f'Detection score cut-off (default: {EXPORT_SCORE_THR})')
    parser.add_argument('--iou-threshold', type=float, default=0.6,
                        help='NMS IoU threshold (default: 0.6)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Version check (artifacts inherit the version of the code that produced the weights)
    ckpt_version = check_checkpoint_version(checkpoint, tool_name='onnx_exporter')

    # Head config (stops on non-v2 / stage-1 checkpoints) and RTMDet variant
    try:
        head_cfg = v2_head_config(checkpoint)
        variant = detect_variant_from_checkpoint(checkpoint)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Detected RTMDet variant: {variant}")
    print(f"Head config: {head_cfg}")

    rtmdet_weights = args.rtmdet_weights or str(get_rtmdet_weights(variant))

    # Register MMDetection modules
    register_all_modules()

    if args.img_size is not None:
        sizes_to_export = [args.img_size]
    else:
        sizes_to_export = MULTISCALE_SIZES
        print(f"Multi-scale export mode: {sizes_to_export}")

    exported_files = []
    for img_size in sizes_to_export:
        if args.output is not None and len(sizes_to_export) == 1:
            output_path = args.output
        else:
            # Generated name with the version embedded in the checkpoint, in onnx_models/
            # (where tools/tf-converter and examples/web-bench look for ONNX models)
            version_part = version_filename_part(ckpt_version)
            ONNX_MODELS_DIR.mkdir(exist_ok=True)
            output_path = str(ONNX_MODELS_DIR / f'rtmdet_fusion_{variant}_{img_size}{version_part}.onnx')

        export_single_size(
            img_size=img_size,
            variant=variant,
            checkpoint=checkpoint,
            rtmdet_weights=rtmdet_weights,
            output_path=output_path,
            args=args,
            head_cfg=head_cfg,
            version=ckpt_version,
        )
        exported_files.append(output_path)

    if len(exported_files) > 1:
        print(f"\n{'='*60}")
        print(f"Export complete! {len(exported_files)} files created:")
        for f in exported_files:
            print(f"  - {f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""RTMDet-Fusion CoreML exporter

Exports a trained checkpoint (tools/train.py, stage 2 with the mask/depth branch) to a CoreML
ML Program (.mlpackage, FP16) with the 19 outputs of the output contract
(docs/output-contract.md; CoreML names are export_contract.COREML_OUTPUT_NAMES).

- The model is built from the trace-friendly modules in rtmdet_fusion.models.fusion_model_coreml
  (same state_dict keys as the training model) and converted with torch.jit.trace + coremltools.
- Built-in NMS: the static NMS selects --max-instances output slots (default 5) from the
  top --topk candidates (default 100). Heads and masks run only for the selected slots.
- bbox features are sampled by matrix multiplication (numerically equivalent to F.grid_sample).
- Run the model on CPU+GPU (MLComputeUnits.cpuAndGPU). The Neural Engine produces broken
  numerics for this model (convolutions are less precise there and the error accumulates
  through the attention layers), and on Apple Silicon Macs the GPU is also faster.

Tested environment:
- torch 2.10.0, torchvision 0.25.0, coremltools 9.0, mmcv 2.1.0, mmdet 3.3.0, scipy 1.13.1
- numpy 1.26.x (with numpy 2.x the conversion of 'int' ops fails with
  "only 0-dimensional arrays can be converted to Python scalars")
- On macOS, mmcv has to be built from source with MMCV_WITH_OPS=1 and setuptools<81
  (newer setuptools lacks pkg_resources)

Usage:
    # macOS (default deployment target: macOS15)
    uv run python tools/coreml_exporter_native.py \\
        --checkpoint work_dirs/main/best_model_s.pth

    # iOS
    uv run python tools/coreml_exporter_native.py \\
        --checkpoint work_dirs/main/best_model_tiny.pth --img-size 256 \\
        --deployment-target iOS17
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.config import ConfigDict
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

from configs.default import (
    NUM_KEYPOINTS, NUM_BODY_KEYPOINTS, IMG_SIZE, MULTISCALE_SIZES,
    FPN_STRIDES, GRID_SIZE, PERSON_CLASS_ID,
    get_rtmdet_config, get_rtmdet_weights, get_rtmdet_in_channels,
)

from rtmdet_fusion.models.fusion_model_coreml import (
    BboxFeatureExtractorCoreML,
    HeadBboxHeadCoreML,
    DCCHeadCoreML,
    MaskDepthBranchCoreML,
    compute_head_quaternion,
)
from rtmdet_fusion.models.encoders import HybridEncoder
from rtmdet_fusion.utils.versioning import (
    check_checkpoint_version, version_filename_part,
)
from rtmdet_fusion.utils.export_contract import (
    OUTPUT_NAMES, COREML_OUTPUT_NAMES, PART_MASK_OUTPUT_DIVISOR, EXPORT_SCORE_THR,
    export_settings_metadata, v2_head_config,
)


# ============================================================
# End-to-end CoreML model
# ============================================================

class RTMDetFusionCoreML(nn.Module):
    """End-to-end CoreML model

    Differences from the ONNX export model (RTMDetFusionONNX in tools/onnx_exporter.py):
    - Sizes come from constants: no int casts of shape[0], no N==0 guards
    - Float-only ops: float one-hot, no int/long casts
    - all inputs and outputs are Float16 (FLOAT16 compute precision with an iOS17/macOS15+ target);
      valid_count is a Float16 count

    NMS is built in. The raw TopK candidates are dominated by duplicates of the most prominent
    person, so emitting only a few raw candidates and running NMS afterwards would drop the other
    people before NMS could see them.
    """

    def __init__(
        self,
        rtmdet_cfg: ConfigDict,
        head_cfg: dict,
        in_channels: int = 128,
        topk: int = 100,
        max_instances: int = 5,
        iou_threshold: float = 0.7,
        img_size: int = IMG_SIZE,
        score_threshold: float = EXPORT_SCORE_THR,
    ):
        super().__init__()
        self.topk = topk  # candidates before NMS
        self.max_instances = max_instances  # output slots; every per-instance stage runs this many times
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.score_threshold = score_threshold

        # RTMDet-Ins (frozen)
        self.rtmdet = MODELS.build(rtmdet_cfg)

        backbone_channels = [in_channels, in_channels * 2, in_channels * 4]

        self.hybrid_encoder = HybridEncoder(
            in_channels=backbone_channels,
            hidden_dim=256,
            out_channels=256,
            num_csp_blocks=3,
            widen_factor=0.5
        )

        self.bbox_feat_extractor = BboxFeatureExtractorCoreML(
            in_channels=256,
            out_channels=256,
            grid_size=GRID_SIZE,
        )

        self.head_bbox_head = HeadBboxHeadCoreML(
            in_channels=256,
            hidden_dim=256,
            num_layers=2
        )

        self.dcc_head = DCCHeadCoreML(
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
        self.body_bbox_head = HeadBboxHeadCoreML(
            in_channels=256,
            hidden_dim=256,
            num_layers=2,
            range_scale=head_cfg['body_bbox_range_scale'],
        )
        self.body_kpt_head = DCCHeadCoreML(
            in_channels=256,
            num_keypoints=NUM_BODY_KEYPOINTS,
            num_bins=head_cfg['num_bins_body'],
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            target_sigma=0.025,
            bin_margin=head_cfg['bin_margin'],
        )

        # Part mask / depth branch (output size img_size / PART_MASK_OUTPUT_DIVISOR)
        self.mask_depth_branch = MaskDepthBranchCoreML()
        self.part_mask_size = img_size // PART_MASK_OUTPUT_DIVISOR

        # Image normalization (BGR)
        self.register_buffer('mean', torch.tensor([103.53, 116.28, 123.675]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([57.375, 57.12, 58.395]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """End-to-end inference

        Args:
            images: (1, 3, H, W) BGR, not normalized (0-255)

        Returns:
            Tuple in export_contract.OUTPUT_NAMES order, with the same meaning and shapes as the
            ONNX export (docs/output-contract.md). The leading dimension is max_instances.
        """
        # Step 1: normalize
        images_norm = (images - self.mean) / self.std

        # Step 2-4: backbone -> PAFPN -> detection head
        backbone_feats = self.rtmdet.backbone(images_norm)
        pafpn_feats = self.rtmdet.neck(backbone_feats)
        cls_scores, bbox_preds, kernel_preds, mask_feat = self.rtmdet.bbox_head(pafpn_feats)

        # Step 5: bbox decode -> NMS
        all_bboxes, all_scores, all_priors, all_kernels = self._decode_bboxes(
            cls_scores, bbox_preds, kernel_preds, images.device
        )
        bboxes, scores, priors, kernels = self._nms(
            all_bboxes, all_scores, all_priors, all_kernels
        )

        bboxes = torch.clamp(bboxes, min=0.0, max=float(self.img_size))
        valid_mask = scores > self.score_threshold

        # Step 6: instance masks (dynamic conv)
        masks = self._generate_masks(mask_feat, kernels, priors, valid_mask)

        # Step 7: Hybrid Encoder
        hybrid_feats = self.hybrid_encoder(backbone_feats)

        # Step 8: head bbox
        grid_features, pos_enc = self.bbox_feat_extractor(
            hybrid_feats, bboxes, priors, valid_mask, self.img_size
        )
        head_bboxes = self.head_bbox_head(grid_features, bboxes)
        head_bboxes_expanded = self.dcc_head.expand_bbox(head_bboxes)

        # Step 9: head keypoints
        head_grid_features, head_pos_enc = self.bbox_feat_extractor(
            hybrid_feats, head_bboxes_expanded, priors, valid_mask, self.img_size
        )
        dcc_output = self.dcc_head(head_grid_features, head_pos_enc)

        keypoints_rel, confidences_per_kpt, visibilities = self.dcc_head.decode_relative(
            dcc_output['heatmap_x'],
            dcc_output['heatmap_y'],
            dcc_output['heatmap_z'],
            dcc_output['visibility_logits'],
        )
        confidences = confidences_per_kpt.mean(dim=-1)

        # Step 10: tracking features
        head_quaternions = compute_head_quaternion(keypoints_rel)
        appearance_features = F.normalize(head_grid_features.mean(dim=1), dim=-1)
        pose_features = F.normalize(dcc_output['kpt_features'].mean(dim=1), dim=-1)

        # Step 11: body reference bbox + body keypoints
        body_bboxes = self.body_bbox_head(grid_features, bboxes)
        body_bboxes_expanded = self.body_kpt_head.expand_bbox(body_bboxes)
        body_grid_features, body_pos_enc = self.bbox_feat_extractor(
            hybrid_feats, body_bboxes_expanded, priors, valid_mask, self.img_size
        )
        body_output = self.body_kpt_head(body_grid_features, body_pos_enc)
        body_keypoints, body_confidences, body_visibilities = self.body_kpt_head.decode_relative(
            body_output['heatmap_x'],
            body_output['heatmap_y'],
            body_output['heatmap_z'],
            body_output['visibility_logits'],
        )

        # Step 12: part masks / depth (stride 8 -> S/4, same interpolation as the training target)
        mask_out = self.mask_depth_branch(
            hybrid_feats, grid_features.mean(dim=1), priors, self.img_size, self.max_instances
        )
        mask_out = F.interpolate(
            mask_out, size=(self.part_mask_size, self.part_mask_size),
            mode='bilinear', align_corners=False,
        )
        num_mask_ch = MaskDepthBranchCoreML.NUM_MASK_CHANNELS

        # Step 13: zero the empty slots (valid_mask=False); coordinate outputs are left as is
        vm = valid_mask.float()
        vm2 = vm.unsqueeze(1)
        outputs = {
            'bboxes': bboxes,
            'scores': scores,
            'masks': masks,
            'head_bboxes': head_bboxes_expanded,
            'keypoints': keypoints_rel,
            'confidences': confidences * vm,
            'visibilities': visibilities * vm2,
            'valid_count': valid_mask.sum().float().reshape(1),
            'head_quaternions': head_quaternions * vm2,
            'appearance_features': appearance_features * vm2,
            'pose_features': pose_features * vm2,
            'in_frame': torch.sigmoid(dcc_output['in_frame_logits']) * vm2,
            'body_bboxes': body_bboxes_expanded,
            'body_keypoints': body_keypoints,
            'body_confidences': body_confidences * vm2,
            'body_visibilities': body_visibilities * vm2,
            'body_in_frame': torch.sigmoid(body_output['in_frame_logits']) * vm2,
            'part_masks': torch.sigmoid(mask_out[:, :num_mask_ch]) * vm.reshape(-1, 1, 1, 1),
            'depth': mask_out[:, num_mask_ch] * vm.reshape(-1, 1, 1),
        }
        return tuple(outputs[name] for name in OUTPUT_NAMES)

    def _decode_bboxes(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        kernel_preds: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode bboxes of all levels (no shape[0] access, float32 throughout)"""
        num_classes = 80
        kernel_dim = kernel_preds[0].shape[1]  # channel dim (a constant in the trace)

        all_bboxes, all_scores, all_priors, all_kernels = [], [], [], []

        for level_idx, stride in enumerate(FPN_STRIDES):
            cls_score = cls_scores[level_idx][0]    # (num_classes, H, W)
            bbox_pred = bbox_preds[level_idx][0]    # (4, H, W)
            kernel_pred = kernel_preds[level_idx][0]  # (kernel_dim, H, W)

            # H, W from constants (no shape access)
            H = self.img_size // stride
            W = H
            HW = H * W

            # Priors (float32)
            shifts_x = torch.arange(0, W, dtype=torch.float32, device=device) * stride
            shifts_y = torch.arange(0, H, dtype=torch.float32, device=device) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            stride_f = torch.full_like(shift_x, float(stride))
            prior = torch.stack([shift_x, shift_y, stride_f, stride_f], dim=-1)

            cls_score = cls_score.permute(1, 2, 0).reshape(HW, num_classes)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(HW, 4)
            kernel_pred = kernel_pred.permute(1, 2, 0).reshape(HW, kernel_dim)

            person_scores = cls_score[:, PERSON_CLASS_ID].sigmoid()

            x1 = shift_x - bbox_pred[:, 0]
            y1 = shift_y - bbox_pred[:, 1]
            x2 = shift_x + bbox_pred[:, 2]
            y2 = shift_y + bbox_pred[:, 3]
            bbox = torch.stack([x1, y1, x2, y2], dim=-1)

            all_bboxes.append(bbox)
            all_scores.append(person_scores)
            all_priors.append(prior)
            all_kernels.append(kernel_pred)

        return (
            torch.cat(all_bboxes, dim=0),
            torch.cat(all_scores, dim=0),
            torch.cat(all_priors, dim=0),
            torch.cat(all_kernels, dim=0),
        )

    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """IoU matrix (tensor ops only)

        Coordinates are divided by img_size before computing areas: the model runs in fp16, and
        in 640 px space an area of 640x640 = 409600 exceeds the fp16 maximum 65504, becomes inf
        and turns IoU into NaN. IoU is a ratio, so normalization does not change it.
        """
        b1 = boxes1 / float(self.img_size)
        b2 = boxes2 / float(self.img_size)
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])  # (N,)
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])  # (M,)

        lt = torch.max(b1[:, None, :2], b2[None, :, :2])  # (N, M, 2)
        rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])  # (N, M, 2)
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter
        # eps on the normalized scale (1e-6 would be subnormal in fp16)
        return inter / (union + 1e-4)

    def _nms(
        self,
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        priors: torch.Tensor,
        kernels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Static NMS (greedy, fixed-size output)

        Same scheme as RTMDetFusionONNX._nms: from the top `topk` candidates, pick max_instances
        times in descending score order and suppress candidates whose IoU with the pick exceeds
        the threshold. The output is always (max_instances, ...); unfilled slots are zero.

        No index assignment and no int cast of argmax: the selection is a matmul with a one-hot
        matrix, and the one-hot itself is built from float ops only (mark the positions equal to
        the maximum, then keep the first one with cumsum).
        """
        pre_n = self.topk
        topk_scores, topk_indices = torch.topk(scores, pre_n, sorted=True)
        topk_bboxes = bboxes[topk_indices]
        topk_priors = priors[topk_indices]
        topk_kernels = kernels[topk_indices]

        # Candidates below the score threshold are never selected
        valid_mask = (topk_scores >= self.score_threshold).float()
        iou_matrix = self._box_iou(topk_bboxes, topk_bboxes)  # (pre_n, pre_n)

        selection_masks = []
        suppressed = torch.zeros(pre_n, device=bboxes.device)
        for _ in range(self.max_instances):
            available = topk_scores * valid_mask * (1.0 - suppressed)
            best = available.max()
            hit = (available >= best).float()
            # Keep only the first of tied maxima (same choice as argmax)
            one_hot = hit * (torch.cumsum(hit, dim=0) <= 1.0).float()
            # No candidates left -> empty slot (a zero-score slot in the output)
            one_hot = one_hot * (best > 0.0).float()
            selection_masks.append(one_hot)

            # iou_matrix[selected row] via matmul (no indexing). The selected box itself is
            # suppressed explicitly: its self-IoU falls below the threshold for tiny boxes
            # (normalized coordinates + eps) or when the threshold is >= 1
            iou_with_best = torch.matmul(one_hot.reshape(1, -1), iou_matrix).reshape(-1)
            suppressed = torch.clamp(
                suppressed + (iou_with_best > self.iou_threshold).float() + one_hot, 0.0, 1.0)

        selection_matrix = torch.stack(selection_masks, dim=0)  # (max_instances, pre_n)
        out_bboxes = torch.matmul(selection_matrix, topk_bboxes)
        out_scores = torch.matmul(
            selection_matrix, topk_scores.reshape(pre_n, 1)).reshape(self.max_instances)
        out_priors = torch.matmul(selection_matrix, topk_priors)
        out_kernels = torch.matmul(selection_matrix, topk_kernels)
        return out_bboxes, out_scores, out_priors, out_kernels

    def _generate_masks(
        self,
        mask_feat: torch.Tensor,
        kernels: torch.Tensor,
        priors: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Instance masks (RTMDet-Ins dynamic conv with bmm)"""
        device = mask_feat.device
        K = self.max_instances
        num_prototypes = 8
        stride = FPN_STRIDES[0]  # mask_feat is stride 8
        H = self.img_size // stride
        W = H
        HW = H * W
        dyconv_channels = 8
        in_channels = num_prototypes + 2

        # Split the kernels
        w1_size = in_channels * dyconv_channels
        w2_size = dyconv_channels * dyconv_channels
        w3_size = dyconv_channels
        b1_size = dyconv_channels
        b2_size = dyconv_channels

        idx = 0
        w1 = kernels[:, idx:idx + w1_size]; idx += w1_size
        w2 = kernels[:, idx:idx + w2_size]; idx += w2_size
        w3 = kernels[:, idx:idx + w3_size]; idx += w3_size
        b1 = kernels[:, idx:idx + b1_size]; idx += b1_size
        b2 = kernels[:, idx:idx + b2_size]; idx += b2_size
        b3 = kernels[:, idx:idx + 1]

        # Coordinate grid
        y_coords = torch.arange(H, dtype=torch.float32, device=device) * stride
        x_coords = torch.arange(W, dtype=torch.float32, device=device) * stride
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coord = torch.stack([x_grid, y_grid], dim=-1).reshape(1, HW, 2)

        points = priors[:, :2].unsqueeze(1)        # (K, 1, 2)
        # Empty slots have a zero prior; clamp to avoid 0/0 (NaN) in their masks
        strides_val = priors[:, 2:3].unsqueeze(1).clamp(min=1.0)  # (K, 1, 1)

        relative_coord = (points - coord) / (strides_val * 8)
        relative_coord = relative_coord.permute(0, 2, 1).reshape(K, 2, H, W)

        mask_feat_expanded = mask_feat[0:1].expand(K, -1, -1, -1)
        input_feat = torch.cat([relative_coord, mask_feat_expanded], dim=1)
        x = input_feat.reshape(K, in_channels, HW)

        # Dynamic conv (3 layers, bmm)
        w1 = w1.reshape(K, dyconv_channels, in_channels)
        b1 = b1.reshape(K, dyconv_channels, 1)
        x = F.relu(torch.bmm(w1, x) + b1)

        w2 = w2.reshape(K, dyconv_channels, dyconv_channels)
        b2 = b2.reshape(K, dyconv_channels, 1)
        x = F.relu(torch.bmm(w2, x) + b2)

        w3 = w3.reshape(K, 1, dyconv_channels)
        b3 = b3.reshape(K, 1, 1)
        x = torch.bmm(w3, x) + b3

        # Resize + smoothstep
        x = F.interpolate(
            x.reshape(K, 1, H, W),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=True,
        ).reshape(K, self.img_size, self.img_size)

        t = torch.clamp((x + 8.0) / 16.0, 0, 1)
        masks = t * t * (3 - 2 * t)
        masks = masks * valid_mask.float().unsqueeze(1).unsqueeze(2)
        return masks


# ============================================================
# Checkpoint and weights
# ============================================================

# Trained modules: (attribute of RTMDetFusionCoreML, checkpoint key)
FUSION_MODULES = (
    ('hybrid_encoder', 'hybrid_encoder_state_dict'),
    ('bbox_feat_extractor', 'bbox_feat_extractor_state_dict'),
    ('head_bbox_head', 'head_bbox_head_state_dict'),
    ('dcc_head', 'dcc_head_state_dict'),
    ('body_bbox_head', 'body_bbox_head_state_dict'),
    ('body_kpt_head', 'body_kpt_head_state_dict'),
    ('mask_depth_branch', 'mask_depth_branch_state_dict'),
)


def detect_variant_from_checkpoint(checkpoint: dict) -> str:
    """Detect the RTMDet-Ins variant from the Hybrid Encoder input projection width"""
    try:
        in_ch = checkpoint['hybrid_encoder_state_dict']['input_proj.0.0.weight'].shape[1]
    except KeyError:
        raise ValueError(
            "Checkpoint has no hybrid_encoder_state_dict['input_proj.0.0.weight']") from None
    for variant in ('tiny', 's'):
        if get_rtmdet_in_channels(variant) == in_ch:
            return variant
    raise ValueError(f"Unknown backbone width in the checkpoint: {in_ch} channels")


def check_fusion_modules(checkpoint: dict) -> None:
    """Raise ValueError if a trained module is missing from the checkpoint"""
    missing = [key for _, key in FUSION_MODULES if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing trained modules: {missing}")


def load_weights(model: RTMDetFusionCoreML, rtmdet_weights: Path, checkpoint: dict) -> None:
    """Load the RTMDet-Ins weights and the trained modules

    The CoreML modules have the same state_dict keys as the training model, so the trained
    modules are loaded strictly (a key mismatch would otherwise export untrained weights).
    """
    ckpt = torch.load(rtmdet_weights, map_location='cpu', weights_only=False)
    model.rtmdet.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    print(f"  RTMDet-Ins: {rtmdet_weights}")

    for attr, key in FUSION_MODULES:
        getattr(model, attr).load_state_dict(checkpoint[key])
    print(f"  Trained modules: {', '.join(attr for attr, _ in FUSION_MODULES)}")


# ============================================================
# Export
# ============================================================

DEPLOYMENT_TARGETS = ('iOS17', 'iOS18', 'macOS15')


def _rename_outputs(mlmodel):
    """Rename the outputs by position to export_contract.COREML_OUTPUT_NAMES

    coremltools' automatic output names differ between variants, so applications could not
    find outputs by name without this.
    """
    import coremltools as ct

    spec = mlmodel.get_spec()
    outputs = list(spec.description.output)
    if len(outputs) != len(COREML_OUTPUT_NAMES):
        raise RuntimeError(
            f"Converted model has {len(outputs)} outputs, expected {len(COREML_OUTPUT_NAMES)}")

    for out, name in zip(outputs, COREML_OUTPUT_NAMES):
        if out.name != name:
            ct.utils.rename_feature(spec, out.name, name)
    return ct.models.MLModel(
        spec, weights_dir=mlmodel.weights_dir, compute_units=ct.ComputeUnit.CPU_AND_GPU)


def export_coreml(
    model: RTMDetFusionCoreML,
    img_size: int,
    output_path: str,
    deployment_target: str,
    metadata: dict,
) -> None:
    """torch.jit.trace -> coremltools (ML Program, FP16) -> .mlpackage"""
    import coremltools as ct

    print("Tracing (torch.jit.trace)...")
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.inference_mode():
        traced = torch.jit.trace(model, dummy)
        traced = torch.jit.freeze(traced)

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='images', shape=(1, 3, img_size, img_size))],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=getattr(ct.target, deployment_target),
        convert_to='mlprogram',
    )
    mlmodel = _rename_outputs(mlmodel)
    for key, value in metadata.items():
        mlmodel.user_defined_metadata[key] = value
    print(f"  Metadata: {metadata}")
    mlmodel.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="RTMDet-Fusion CoreML exporter")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint path (.pth, trained through stage 2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .mlpackage path '
                             '(default: rtmdet_fusion_coreml_{variant}_{size}_{version}.mlpackage)')
    parser.add_argument('--img-size', type=int, default=640, choices=MULTISCALE_SIZES,
                        help='Input resolution (default: 640)')
    parser.add_argument('--rtmdet-weights', type=str, default=None,
                        help='RTMDet-Ins weights (default: chosen from the detected variant)')
    parser.add_argument('--topk', type=int, default=100,
                        help='Number of candidates before NMS (default: 100)')
    parser.add_argument('--max-instances', type=int, default=5,
                        help='Number of output slots selected by NMS (default: 5)')
    parser.add_argument('--iou-threshold', type=float, default=0.7,
                        help='NMS IoU threshold on person bboxes (default: 0.7)')
    parser.add_argument('--score-threshold', type=float, default=EXPORT_SCORE_THR,
                        help='Detection score cut-off for valid_count and NMS candidates '
                             f'(default: {EXPORT_SCORE_THR}). Keep it low when the application '
                             'applies its own threshold')
    parser.add_argument('--deployment-target', type=str, default='macOS15',
                        choices=DEPLOYMENT_TARGETS,
                        help='Minimum deployment target (default: macOS15; iOS17/iOS18 for iOS)')
    args = parser.parse_args()

    if args.max_instances > args.topk:
        parser.error('--max-instances must not exceed --topk')
    if not Path(args.checkpoint).exists():
        sys.exit(f"Error: Checkpoint not found: {args.checkpoint}")

    register_all_modules()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    try:
        variant = detect_variant_from_checkpoint(checkpoint)
        head_cfg = v2_head_config(checkpoint)
        check_fusion_modules(checkpoint)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    # Artifacts inherit the version of the code that produced the weights
    ckpt_version = check_checkpoint_version(checkpoint, tool_name='coreml_exporter_native')

    rtmdet_weights = Path(args.rtmdet_weights or get_rtmdet_weights(variant))
    if not rtmdet_weights.exists():
        sys.exit(f"Error: RTMDet-Ins weights not found: {rtmdet_weights}")

    output_path = args.output or (
        f'rtmdet_fusion_coreml_{variant}_{args.img_size}'
        f'{version_filename_part(ckpt_version)}.mlpackage'
    )

    print(f"\n{'='*60}")
    print("CoreML export")
    print(f"  Variant: {variant}")
    print(f"  Image size: {args.img_size}")
    print(f"  NMS: top {args.topk} candidates -> {args.max_instances} slots, "
          f"IoU > {args.iou_threshold}")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Head config: {head_cfg}")
    print(f"  Deployment target: {args.deployment_target}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Version and export settings embedded in user_defined_metadata
    metadata = {}
    if ckpt_version:
        metadata['rtmdet_fusion_version'] = ckpt_version
    metadata.update(export_settings_metadata(
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_instances=args.max_instances,
        topk=args.topk,
        builtin_nms=True,
        grid_sample='matmul',
    ))

    model = RTMDetFusionCoreML(
        rtmdet_cfg=ConfigDict(get_rtmdet_config(variant)),
        head_cfg=head_cfg,
        in_channels=get_rtmdet_in_channels(variant),
        topk=args.topk,
        max_instances=args.max_instances,
        iou_threshold=args.iou_threshold,
        img_size=args.img_size,
        score_threshold=args.score_threshold,
    )

    print("Loading weights...")
    load_weights(model, rtmdet_weights, checkpoint)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    export_coreml(model, args.img_size, output_path, args.deployment_target, metadata)
    print(f"\nExport complete: {output_path}")


if __name__ == '__main__':
    main()

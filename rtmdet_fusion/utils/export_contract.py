"""Output contract of the exported models (ONNX / CoreML, and tfjs converted from them).

Output names, order, part-mask resolution and the head configuration are defined here once and
shared by every exporter, so the implementations cannot drift apart. Changing output names or
order breaks applications and is a MAJOR change.
"""

from typing import Dict

from configs.default import (
    NUM_BINS, NUM_BINS_BODY, BIN_MARGIN, BODY_BBOX_RANGE_SCALE,
)


# Output names (must match the return order of each exporter's forward)
OUTPUT_NAMES = [
    # Detection, head keypoints and tracking features
    'bboxes', 'scores', 'masks', 'head_bboxes',
    'keypoints', 'confidences', 'visibilities', 'valid_count',
    'head_quaternions', 'appearance_features', 'pose_features',
    # In-frame probability, body keypoints, part masks and depth
    'in_frame', 'body_bboxes', 'body_keypoints', 'body_confidences',
    'body_visibilities', 'body_in_frame', 'part_masks', 'depth',
]

# CoreML output names. The out_ prefix avoids collisions with var names inside the traced graph.
# 'confidences' is named out_kpt_confidences (part of the contract).
COREML_OUTPUT_NAMES = [
    'out_kpt_confidences' if name == 'confidences' else f'out_{name}'
    for name in OUTPUT_NAMES
]

# Part mask / depth output resolution = input resolution / 4 (same as the training target img_size/4)
PART_MASK_OUTPUT_DIVISOR = 4

# Detection score cut-off baked into exports (lower bound for valid_count and NMS candidates).
# Kept low because applications are expected to apply their own, higher threshold on `scores`.
# Baking configs.INS_SCORE_THR (0.3, used by the viewer and training-time matching) would make
# any application threshold below 0.3 meaningless.
EXPORT_SCORE_THR = 0.05


def export_settings_metadata(
    score_threshold: float,
    iou_threshold: float,
    max_instances: int,
    topk: int,
    builtin_nms: bool,
    grid_sample: str,
) -> Dict[str, str]:
    """Export settings embedded in artifact metadata (ONNX metadata_props / CoreML
    user_defined_metadata / tfjs model.json userDefinedMetadata; values are strings).

    Re-exports that differ only in e.g. the score cut-off have identical weights and file names,
    so the settings are recorded in the artifact itself.
    topk is the number of candidates before NMS (the number of output slots when builtin_nms=false).
    """
    return {
        'rtmdet_fusion_score_threshold': f'{score_threshold:g}',
        'rtmdet_fusion_iou_threshold': f'{iou_threshold:g}',
        'rtmdet_fusion_max_instances': str(max_instances),
        'rtmdet_fusion_topk': str(topk),
        'rtmdet_fusion_builtin_nms': 'true' if builtin_nms else 'false',
        'rtmdet_fusion_grid_sample': grid_sample,
    }


# Checkpoint modules required for export
V2_REQUIRED_STATE_DICTS = (
    'body_bbox_head_state_dict', 'body_kpt_head_state_dict', 'mask_depth_branch_state_dict',
)


def v2_head_config(checkpoint: dict) -> Dict:
    """Read the head construction parameters from a checkpoint.

    Raises ValueError when the checkpoint cannot satisfy the output contract: body heads missing
    (not a v2 checkpoint) or mask/depth branch missing (stage-1 checkpoint).
    """
    missing = [k for k in V2_REQUIRED_STATE_DICTS if k not in checkpoint]
    if any(k != 'mask_depth_branch_state_dict' for k in missing):
        raise ValueError(f"Not a v2 checkpoint (missing: {missing})")
    if missing:
        raise ValueError(
            "The checkpoint has no mask/depth branch: it is from training stage 1. "
            "Run stage 2 before exporting: "
            "`tools/train.py --resume best --mask-weight 4.0`"
        )
    cfg = checkpoint.get('config', {})
    return {
        'num_bins': cfg.get('num_bins', NUM_BINS),
        'num_bins_body': cfg.get('num_bins_body', NUM_BINS_BODY),
        'bin_margin': cfg.get('bin_margin', BIN_MARGIN),
        'body_bbox_range_scale': cfg.get('body_bbox_range_scale', BODY_BBOX_RANGE_SCALE),
    }

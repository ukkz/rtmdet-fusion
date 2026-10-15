"""Bounding box utilities"""

import torch

# Default expansion ratio (same value as configs/default.py BBOX_EXPAND_RATIO,
# defined here to avoid an import cycle)
DEFAULT_BBOX_EXPAND_RATIO = 1.0


def expand_bbox(bboxes: torch.Tensor, expand_ratio: float = DEFAULT_BBOX_EXPAND_RATIO) -> torch.Tensor:
    """Expand bboxes around their center (as in RTMO)

    Args:
        bboxes: (N, 4) [x1, y1, x2, y2]
        expand_ratio: expansion ratio (default: BBOX_EXPAND_RATIO)

    Returns:
        expanded_bboxes: (N, 4) [x1, y1, x2, y2]
    """
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]

    w_new = w * expand_ratio
    h_new = h * expand_ratio

    x1_new = cx - w_new / 2
    y1_new = cy - h_new / 2
    x2_new = cx + w_new / 2
    y2_new = cy + h_new / 2

    return torch.stack([x1_new, y1_new, x2_new, y2_new], dim=1)

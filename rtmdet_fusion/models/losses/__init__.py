"""RTMDet-Fusion losses

Keypoint, bbox and mask/depth losses used by tools/train.py.
"""

from .keypoint import (
    KLDiscretLoss, SimCCLoss, BoneLoss, BoneDirectionLoss,
    CoordL2Loss, ImageCoordL2Loss, BboxJitterConsistencyLoss,
    VisibilityBCELoss,
    InFrameBCELoss,
)
from .bbox import HeadBboxIoULoss, HeadBboxL1Loss
from .mask import MaskBCEDiceLoss, MaskedDepthL1Loss

__all__ = [
    # Keypoint losses
    'KLDiscretLoss',
    'SimCCLoss',
    'BoneLoss',
    'BoneDirectionLoss',
    'CoordL2Loss',
    'ImageCoordL2Loss',
    'BboxJitterConsistencyLoss',
    'VisibilityBCELoss',
    'InFrameBCELoss',
    # Mask / depth losses
    'MaskBCEDiceLoss',
    'MaskedDepthL1Loss',
    # Bbox losses
    'HeadBboxIoULoss',
    'HeadBboxL1Loss',
]

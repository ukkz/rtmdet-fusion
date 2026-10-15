"""Bbox losses

GIoU and L1 losses for the reference bbox heads (head bbox, body reference bbox).
Follows the DETR family, which weights L1 : GIoU = 5 : 2.
"""

from typing import Optional

import torch
import torch.nn as nn


class HeadBboxL1Loss(nn.Module):
    """Bbox L1 loss (as in DETR)

    L1 loss on coordinates normalized by the image size.
    DETR uses it with L1 : GIoU = 5 : 2.
    """

    def __init__(self, loss_weight: float = 5.0, img_size: int = 640):
        """
        Args:
            loss_weight: loss weight (5.0 as in DETR)
            img_size: image size used for the normalization
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.img_size = img_size

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the L1 loss

        Args:
            pred_bboxes: (N, 4) predicted bbox [x1, y1, x2, y2] (pixels)
            target_bboxes: (N, 4) GT bbox [x1, y1, x2, y2] (pixels)
            weight: (N,) optional weights

        Returns:
            loss: scalar
        """
        N = pred_bboxes.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)

        # Normalize by the image size (to roughly 0-1)
        pred_norm = pred_bboxes / self.img_size
        target_norm = target_bboxes / self.img_size

        # L1 loss (absolute error per coordinate)
        l1_loss = torch.abs(pred_norm - target_norm)  # (N, 4)
        l1_per_box = l1_loss.sum(dim=-1)  # (N,) sum of the 4 coordinates

        # Weighting
        if weight is not None:
            total_weight = weight.sum() + 1e-8
            loss = (l1_per_box * weight).sum() / total_weight
        else:
            loss = l1_per_box.mean()

        return loss * self.loss_weight


class HeadBboxIoULoss(nn.Module):
    """Bbox IoU loss (as in RTMO)

    IoU loss following RTMO's bbox supervision.
    Uses GIoU (Generalized IoU) for stable training.
    """

    def __init__(self, loss_weight: float = 1.0, use_giou: bool = True):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_giou = use_giou

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the IoU loss

        Args:
            pred_bboxes: (N, 4) predicted bbox [x1, y1, x2, y2]
            target_bboxes: (N, 4) GT bbox [x1, y1, x2, y2]
            weight: (N,) optional weights

        Returns:
            loss: scalar
        """
        N = pred_bboxes.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)

        # Intersection
        inter_x1 = torch.max(pred_bboxes[:, 0], target_bboxes[:, 0])
        inter_y1 = torch.max(pred_bboxes[:, 1], target_bboxes[:, 1])
        inter_x2 = torch.min(pred_bboxes[:, 2], target_bboxes[:, 2])
        inter_y2 = torch.min(pred_bboxes[:, 3], target_bboxes[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Areas
        pred_area = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * \
                    (pred_bboxes[:, 3] - pred_bboxes[:, 1])
        target_area = (target_bboxes[:, 2] - target_bboxes[:, 0]) * \
                      (target_bboxes[:, 3] - target_bboxes[:, 1])

        # Union
        union_area = pred_area + target_area - inter_area + 1e-8

        # IoU
        iou = inter_area / union_area

        if self.use_giou:
            # GIoU: IoU - (C - U) / C
            # C: area of the smallest enclosing box
            enclose_x1 = torch.min(pred_bboxes[:, 0], target_bboxes[:, 0])
            enclose_y1 = torch.min(pred_bboxes[:, 1], target_bboxes[:, 1])
            enclose_x2 = torch.max(pred_bboxes[:, 2], target_bboxes[:, 2])
            enclose_y2 = torch.max(pred_bboxes[:, 3], target_bboxes[:, 3])

            enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-8
            giou = iou - (enclose_area - union_area) / enclose_area

            loss = 1.0 - giou
        else:
            loss = 1.0 - iou

        # Weighting
        if weight is not None:
            total_weight = weight.sum() + 1e-8
            loss = (loss * weight).sum() / total_weight
        else:
            loss = loss.mean()

        return loss * self.loss_weight

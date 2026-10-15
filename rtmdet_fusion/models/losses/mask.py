"""Mask / depth losses

Losses on the MaskDepthBranch outputs:
- MaskBCEDiceLoss: BCE + Dice on the 7 mask channels (clothed / bare body / 4 parts / occlusion)
- MaskedDepthL1Loss: masked L1 on the depth channel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskBCEDiceLoss(nn.Module):
    """BCE + Dice loss for the mask channels

    channel_valid excludes individual (instance, channel) pairs:
    - occlusion channel: instances with occlusion_valid=False are excluded
    - part channels: instances without a teacher (SMPL-X rendering) are excluded

    Args:
        loss_weight: overall weight
        bce_weight: weight of the BCE term (relative to Dice)
        dice_weight: weight of the Dice term
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        channel_valid: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: (N, C, H, W) mask logits
            targets: (N, C, H, W) teachers {0, 1}
            channel_valid: (N, C) channels included in the loss (None: all)

        Returns:
            loss: scalar
        """
        N, C, H, W = pred_logits.shape
        device = pred_logits.device
        if N == 0:
            return torch.tensor(0.0, device=device)

        if channel_valid is None:
            channel_valid = torch.ones(N, C, dtype=torch.bool, device=device)
        cv = channel_valid.float()  # (N, C)
        n_valid = cv.sum().clamp(min=1.0)

        # BCE (mean per channel, then averaged over the valid channels)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, targets, reduction='none'
        ).mean(dim=(2, 3))  # (N, C)
        bce_loss = (bce * cv).sum() / n_valid

        # Dice
        prob = torch.sigmoid(pred_logits)
        inter = (prob * targets).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - (2 * inter + 1.0) / (union + 1.0)  # (N, C)
        dice_loss = (dice * cv).sum() / n_valid

        return (self.bce_weight * bce_loss + self.dice_weight * dice_loss) * self.loss_weight


class MaskedDepthL1Loss(nn.Module):
    """Masked L1 loss for the depth channel

    L1 is computed only inside the teacher support (clothed mask or bare body, pixels with
    depth > 0). Depth is rel_y equivalent (0.5 = pelvis, normalized by the body reference
    bbox size).

    Args:
        loss_weight: overall weight
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_depth: (N, H, W) predicted depth (rel_y equivalent)
            target_depth: (N, H, W) teacher depth
            valid_mask: (N, H, W) teacher support (bool)

        Returns:
            loss: scalar
        """
        device = pred_depth.device
        if pred_depth.shape[0] == 0:
            return torch.tensor(0.0, device=device)

        vm = valid_mask.float()
        n = vm.sum().clamp(min=1.0)
        l1 = (pred_depth - target_depth).abs() * vm
        return l1.sum() / n * self.loss_weight

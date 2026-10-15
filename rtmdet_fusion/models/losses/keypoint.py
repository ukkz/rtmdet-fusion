"""Keypoint losses

Losses for the unified-coordinate (x, y, z) SimCC keypoint heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDiscretLoss(nn.Module):
    """KL divergence loss for SimCC (after RTMW3D's KLDiscretLoss)

    Plain KL divergence without learnable variance parameters.

    Args:
        beta: loss coefficient (default 10.0, as in RTMW3D)
        label_softmax: whether to apply softmax to the target
            - True: the target is logits (raw values)
            - False: the target is already a probability distribution (generate_target)
            Note: applying softmax to a probability distribution flattens it and makes the
            predictions collapse toward the center.
        use_target_weight: whether to use the visibility weights
        loss_weight: overall loss weight
    """

    def __init__(
        self,
        beta: float = 10.0,
        label_softmax: bool = True,
        use_target_weight: bool = True,
        loss_weight: float = 1.0
    ):
        super().__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        # Element-wise KL divergence
        self.criterion = nn.KLDivLoss(reduction='none')

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred: (N, K, num_bins) predicted logits
            target: (N, K, num_bins) target distribution (Gaussian)
            target_weight: (N, K) visibility weight (optional)

        Returns:
            loss: scalar loss
        """
        N, K, num_bins = pred.shape
        device = pred.device

        # NaN guard on the inputs
        if torch.isnan(pred).any() or torch.isnan(target).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # log_softmax of the prediction
        log_pred = F.log_softmax(pred, dim=-1)  # (N, K, num_bins)

        # Target
        if self.label_softmax:
            target = F.softmax(target, dim=-1)
        else:
            # Already a distribution: normalize only
            target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)

        # KL(target || pred) = sum(target * (log(target) - log(pred)))
        # nn.KLDivLoss takes log_pred and target
        kl_loss = self.criterion(log_pred, target)  # (N, K, num_bins)

        # Sum over the bins
        kl_loss = kl_loss.sum(dim=-1)  # (N, K)

        # Weighting
        if self.use_target_weight and target_weight is not None:
            kl_loss = kl_loss * target_weight  # (N, K)
            total_weight = target_weight.sum() + 1e-8
            loss = kl_loss.sum() / total_weight
        else:
            loss = kl_loss.mean()

        # Apply beta and loss_weight
        loss = loss * self.beta * self.loss_weight

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss


class SimCCLoss(nn.Module):
    """SimCC loss for unified coordinates

    KL divergence on each of the three axes (x, y, z), using KLDiscretLoss as in RTMW3D.
    """

    def __init__(
        self,
        beta: float = 10.0,
        label_softmax: bool = False,  # generate_target already returns distributions
        use_target_weight: bool = True,
        loss_weight: float = 1.0
    ):
        super().__init__()
        self.kl_loss = KLDiscretLoss(
            beta=beta,
            label_softmax=label_softmax,
            use_target_weight=use_target_weight,
            loss_weight=1.0  # 1.0 internally; loss_weight is applied outside
        )
        self.loss_weight = loss_weight

    def forward(
        self,
        pred_x: torch.Tensor,
        pred_y: torch.Tensor,
        pred_z: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_weight: torch.Tensor = None,
        return_per_axis: bool = False
    ) -> torch.Tensor:
        """
        Args:
            pred_x, pred_y, pred_z: (N, K, num_bins) predicted logits per axis
            target_x, target_y, target_z: (N, K, num_bins) target distributions per axis
            target_weight: (N, K) visibility weight
            return_per_axis: also return the per-axis losses

        Returns:
            return_per_axis=False: loss (scalar)
            return_per_axis=True: (loss, loss_x, loss_y, loss_z) scalars
        """
        loss_x = self.kl_loss(pred_x, target_x, target_weight)
        loss_y = self.kl_loss(pred_y, target_y, target_weight)
        loss_z = self.kl_loss(pred_z, target_z, target_weight)

        # Mean of the three axes * loss_weight
        loss = (loss_x + loss_y + loss_z) / 3.0 * self.loss_weight

        if return_per_axis:
            # Per-axis losses, also scaled by loss_weight
            return loss, loss_x * self.loss_weight, loss_y * self.loss_weight, loss_z * self.loss_weight

        return loss


class BoneLoss(nn.Module):
    """Bone length loss (3D structural consistency)

    Constrains the 3D distances between connected keypoints (bone lengths) to match the GT.
    After RTMW3D's BoneLoss, with a head skeleton by default.

    Head bones (13):
    - center(0) - nose(6)
    - center(0) - top(1)
    - center(0) - vertex(9)
    - center(0) - occiput(8)
    - center(0) - left_eye(4)
    - center(0) - right_eye(5)
    - center(0) - left_ear(2)
    - center(0) - right_ear(3)
    - center(0) - neck(10)
    - left_ear(2) - right_ear(3)
    - left_eye(4) - right_eye(5)
    - top(1) - vertex(9)
    - vertex(9) - occiput(8)

    Args:
        loss_weight: loss weight (default 5.0)
        use_target_weight: whether to weight by visibility
    """

    # Bones: (start_idx, end_idx)
    BONE_LINKS = [
        (0, 6),   # center - nose
        (0, 1),   # center - top
        (0, 9),   # center - vertex
        (0, 8),   # center - occiput
        (0, 4),   # center - left_eye
        (0, 5),   # center - right_eye
        (0, 2),   # center - left_ear
        (0, 3),   # center - right_ear
        (0, 10),  # center - neck
        (2, 3),   # left_ear - right_ear
        (4, 5),   # left_eye - right_eye
        (1, 9),   # top - vertex
        (9, 8),   # vertex - occiput
    ]

    def __init__(
        self,
        loss_weight: float = 5.0,
        use_target_weight: bool = True,
        bone_links: list = None
    ):
        """
        Args:
            bone_links: bones [(start_idx, end_idx), ...]. None uses the head skeleton
                        (BONE_LINKS); the body head passes configs.default.BODY_BONE_LINKS.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight
        self.bone_links = bone_links if bone_links is not None else self.BONE_LINKS

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute the bone length loss

        Args:
            pred: predicted keypoints (N, K, 3) [x, y, z]
            target: GT keypoints (N, K, 3)
            target_weight: visibility weight (N, K) or None

        Returns:
            loss: scalar loss
        """
        N, K, _ = pred.shape
        device = pred.device

        if N == 0:
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0, device=device)
        valid_bones = 0

        for start_idx, end_idx in self.bone_links:
            # Predicted bone length
            pred_start = pred[:, start_idx]  # (N, 3)
            pred_end = pred[:, end_idx]      # (N, 3)
            pred_bone_vec = pred_end - pred_start
            pred_bone_len = torch.norm(pred_bone_vec, dim=-1)  # (N,)

            # GT bone length
            target_start = target[:, start_idx]
            target_end = target[:, end_idx]
            target_bone_vec = target_end - target_start
            target_bone_len = torch.norm(target_bone_vec, dim=-1)  # (N,)

            # L1 loss
            bone_loss = torch.abs(pred_bone_len - target_bone_len)  # (N,)

            # Visibility weighting
            if self.use_target_weight and target_weight is not None:
                # Only when both end points are valid
                weight = target_weight[:, start_idx] * target_weight[:, end_idx]
                bone_loss = bone_loss * weight

            total_loss = total_loss + bone_loss.mean()
            valid_bones += 1

        # Normalize by the number of bones
        if valid_bones > 0:
            total_loss = total_loss / valid_bones

        return total_loss * self.loss_weight


class BoneDirectionLoss(nn.Module):
    """Bone direction loss (cosine similarity)

    BoneLoss only constrains bone lengths, so a uniform translation of all keypoints costs
    nothing. This loss constrains the bone directions (unit vectors) and directly penalizes
    such shifts.

    Args:
        loss_weight: loss weight (default 5.0)
        use_target_weight: whether to weight by visibility
    """

    # Same bones as BoneLoss
    BONE_LINKS = BoneLoss.BONE_LINKS

    def __init__(
        self,
        loss_weight: float = 5.0,
        use_target_weight: bool = True,
        bone_links: list = None,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight
        self.bone_links = bone_links if bone_links is not None else self.BONE_LINKS

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the bone direction loss

        Args:
            pred: predicted keypoints (N, K, 3) [x, y, z]
            target: GT keypoints (N, K, 3)
            target_weight: visibility weight (N, K) or None

        Returns:
            loss: scalar loss
        """
        N, K, _ = pred.shape
        device = pred.device

        if N == 0:
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0, device=device)
        valid_bones = 0

        for start_idx, end_idx in self.bone_links:
            pred_vec = pred[:, end_idx] - pred[:, start_idx]  # (N, 3)
            target_vec = target[:, end_idx] - target[:, start_idx]  # (N, 3)

            pred_dir = F.normalize(pred_vec, dim=-1, eps=1e-6)
            target_dir = F.normalize(target_vec, dim=-1, eps=1e-6)

            # 1 - cosine similarity: 0 = identical, 2 = opposite
            cos_sim = (pred_dir * target_dir).sum(dim=-1)  # (N,)
            bone_loss = 1.0 - cos_sim  # (N,)

            if self.use_target_weight and target_weight is not None:
                weight = target_weight[:, start_idx] * target_weight[:, end_idx]
                bone_loss = bone_loss * weight

            total_loss = total_loss + bone_loss.mean()
            valid_bones += 1

        if valid_bones > 0:
            total_loss = total_loss / valid_bones

        return total_loss * self.loss_weight


class CoordL2Loss(nn.Module):
    """Direct L2 loss on the unified coordinates

    SimCCLoss learns the shape of the distributions with KL divergence but does not supervise
    the final coordinate values. This loss compares the soft-argmax coordinates with the GT
    directly (all axes x, y, z) to improve the coordinate accuracy.

    Args:
        loss_weight: loss weight (default 1.0)
        use_target_weight: whether to weight by visibility
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        use_target_weight: bool = True
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute the coordinate L2 loss

        Args:
            pred: predicted keypoints (N, K, 3) [x, y, z] normalized coordinates
            target: GT keypoints (N, K, 3)
            target_weight: visibility weight (N, K) or None

        Returns:
            loss: scalar loss
        """
        N, K, _ = pred.shape
        device = pred.device

        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # NaN guard on the inputs
        if torch.isnan(pred).any() or torch.isnan(target).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Squared error per coordinate
        l2_loss = (pred - target) ** 2  # (N, K, 3)

        # Sum over the coordinates (x + y + z)
        l2_per_kpt = l2_loss.sum(dim=-1)  # (N, K)

        # Visibility weighting
        if self.use_target_weight and target_weight is not None:
            l2_per_kpt = l2_per_kpt * target_weight  # (N, K)
            total_weight = target_weight.sum().clamp(min=1.0)
            loss = l2_per_kpt.sum() / total_weight
        else:
            loss = l2_per_kpt.mean()

        return loss * self.loss_weight


class ImageCoordL2Loss(nn.Module):
    """L2 loss in image coordinates

    CoordL2Loss works on bbox-relative coordinates, so it depends on how the bbox is placed
    and sized. This loss maps both prediction and GT to image coordinates first, directly
    supervising where in the image each keypoint should be.

    Normalized by the bbox size so that the loss scale does not depend on the bbox size.

    Args:
        loss_weight: loss weight (default 10.0)
        use_target_weight: whether to weight by visibility
    """

    def __init__(
        self,
        loss_weight: float = 10.0,
        use_target_weight: bool = True,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight

    def forward(
        self,
        pred_rel: torch.Tensor,
        target_rel: torch.Tensor,
        pred_bbox: torch.Tensor,
        target_bbox: torch.Tensor,
        target_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the image-coordinate L2 loss

        Args:
            pred_rel: predicted relative coordinates (N, K, 3) [rel_x, rel_y, rel_z]
            target_rel: GT relative coordinates (N, K, 3)
            pred_bbox: reference bbox of the prediction (N, 4) [x1, y1, x2, y2]
            target_bbox: reference bbox of the GT (N, 4)
            target_weight: visibility weight (N, K) or None

        Returns:
            loss: scalar loss
        """
        N, K, _ = pred_rel.shape
        device = pred_rel.device

        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # pred_rel -> image coordinates
        p_x1 = pred_bbox[:, 0:1]  # (N, 1)
        p_y1 = pred_bbox[:, 1:2]
        p_x2 = pred_bbox[:, 2:3]
        p_y2 = pred_bbox[:, 3:4]
        p_w = p_x2 - p_x1
        p_h = p_y2 - p_y1

        pred_img_x = pred_rel[:, :, 0] * p_w + p_x1
        pred_img_y = (1.0 - pred_rel[:, :, 2]) * p_h + p_y1  # z -> image Y is flipped

        # target_rel -> image coordinates
        t_x1 = target_bbox[:, 0:1]
        t_y1 = target_bbox[:, 1:2]
        t_x2 = target_bbox[:, 2:3]
        t_y2 = target_bbox[:, 3:4]
        t_w = t_x2 - t_x1
        t_h = t_y2 - t_y1

        target_img_x = target_rel[:, :, 0] * t_w + t_x1
        target_img_y = (1.0 - target_rel[:, :, 2]) * t_h + t_y1

        # Normalize by the mean bbox size of prediction and target
        norm = ((p_w + p_h + t_w + t_h) / 4.0).clamp(min=1e-6)  # (N, 1)

        dx = (pred_img_x - target_img_x) / norm
        dy = (pred_img_y - target_img_y) / norm
        l2_per_kpt = dx ** 2 + dy ** 2  # (N, K)

        if self.use_target_weight and target_weight is not None:
            l2_per_kpt = l2_per_kpt * target_weight
            total_weight = target_weight.sum().clamp(min=1.0)
            loss = l2_per_kpt.sum() / total_weight
        else:
            loss = l2_per_kpt.mean()

        return loss * self.loss_weight


class BboxJitterConsistencyLoss(nn.Module):
    """Bbox-jitter consistency loss

    The same instance is predicted twice with two different bbox jitters; both predictions
    are mapped to image coordinates and forced to agree with an L2 loss.

    This builds the invariant of feature-based prediction directly into the objective:
    the same object must map to the same image coordinates regardless of how the bbox is cut.

    Args:
        loss_weight: loss weight (default 5.0)
    """

    def __init__(
        self,
        loss_weight: float = 5.0,
        use_target_weight: bool = True,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight

    def forward(
        self,
        pred_rel_a: torch.Tensor,
        pred_rel_b: torch.Tensor,
        bbox_a: torch.Tensor,
        bbox_b: torch.Tensor,
        target_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Image-coordinate agreement loss between two jittered predictions

        Args:
            pred_rel_a: (N, K, 3) relative coordinates of the first prediction
            pred_rel_b: (N, K, 3) relative coordinates of the second prediction
                        (same instances, different jitter)
            bbox_a: (N, 4) first bbox
            bbox_b: (N, 4) second bbox
            target_weight: (N, K) visibility weight (optional)

        Returns:
            loss: scalar loss
        """
        N, K, _ = pred_rel_a.shape
        device = pred_rel_a.device

        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # bbox a
        a_x1 = bbox_a[:, 0:1]
        a_y1 = bbox_a[:, 1:2]
        a_x2 = bbox_a[:, 2:3]
        a_y2 = bbox_a[:, 3:4]
        a_w = a_x2 - a_x1
        a_h = a_y2 - a_y1

        # bbox b
        b_x1 = bbox_b[:, 0:1]
        b_y1 = bbox_b[:, 1:2]
        b_x2 = bbox_b[:, 2:3]
        b_y2 = bbox_b[:, 3:4]
        b_w = b_x2 - b_x1
        b_h = b_y2 - b_y1

        # Map each prediction to image coordinates
        img_xa = pred_rel_a[:, :, 0] * a_w + a_x1
        img_ya = (1.0 - pred_rel_a[:, :, 2]) * a_h + a_y1
        img_xb = pred_rel_b[:, :, 0] * b_w + b_x1
        img_yb = (1.0 - pred_rel_b[:, :, 2]) * b_h + b_y1

        # Normalize by the mean bbox size (loss scale independent of the bbox size)
        norm = ((a_w + a_h + b_w + b_h) / 4.0).clamp(min=1e-6)

        dx = (img_xa - img_xb) / norm
        dy = (img_ya - img_yb) / norm
        l2 = dx ** 2 + dy ** 2  # (N, K)

        if self.use_target_weight and target_weight is not None:
            l2 = l2 * target_weight
            total_weight = target_weight.sum().clamp(min=1.0)
            loss = l2.sum() / total_weight
        else:
            loss = l2.mean()

        return loss * self.loss_weight


class VisibilityBCELoss(nn.Module):
    """BCE loss for the visibility prediction (as in RTMO)

    Learns keypoint visibility as binary classification.
    COCO visibility:
      - 0: not labeled / out of frame (excluded)
      - 1: labeled but occluded (target=0)
      - 2: labeled and visible (target=1)

    Args:
        loss_weight: loss weight (default 0.5; auxiliary to the coordinate losses.
            RTMO uses no explicit weight, i.e. 1.0)
        label_smoothing: label smoothing factor (default 0.0)
    """

    def __init__(
        self,
        loss_weight: float = 0.5,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.label_smoothing = label_smoothing
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_visibility: torch.Tensor,
        include_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute the visibility BCE loss

        Args:
            pred_logits: (N, K) visibility logits
            target_visibility: (N, K) GT visibility (COCO 0/1/2)
            include_mask: (N, K) keypoints to include (optional;
                          None uses only keypoints with visibility > 0)

        Returns:
            loss: scalar loss
        """
        N, K = pred_logits.shape
        device = pred_logits.device

        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # COCO visibility -> binary target
        # 0: not labeled (excluded)
        # 1: occluded -> target = 0
        # 2: visible -> target = 1
        target_binary = (target_visibility == 2).float()  # (N, K)

        # Keypoints included in the loss: visibility > 0 (labeled only)
        if include_mask is None:
            include_mask = (target_visibility > 0).float()
        else:
            include_mask = include_mask.float()

        # Label smoothing
        if self.label_smoothing > 0:
            target_binary = target_binary * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Element-wise BCE
        bce_loss = self.criterion(pred_logits, target_binary)  # (N, K)

        # Masked mean
        masked_loss = bce_loss * include_mask
        total_weight = include_mask.sum().clamp(min=1.0)
        loss = masked_loss.sum() / total_weight

        return loss * self.loss_weight


class InFrameBCELoss(nn.Module):
    """BCE loss for the in-frame / out-of-frame prediction

    Learns whether a keypoint lies inside the image frame (visibility > 0) as binary
    classification. Unlike VisibilityBCELoss (visible vs occluded, v=0 excluded), all
    keypoints are included.

    COCO visibility:
      - 0: out of frame -> target = 0
      - 1: occluded (in frame) -> target = 1
      - 2: visible (in frame) -> target = 1

    Args:
        loss_weight: loss weight (default 0.5)
    """

    def __init__(self, loss_weight: float = 0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_visibility: torch.Tensor,
        include_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: (N, K) in-frame logits
            target_visibility: (N, K) GT visibility (COCO 0/1/2)
            include_mask: (N, K) keypoints to include (optional, default all)

        Returns:
            loss: scalar loss
        """
        N, K = pred_logits.shape
        device = pred_logits.device

        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        target_binary = (target_visibility > 0).float()  # (N, K)

        if include_mask is None:
            include_mask = torch.ones_like(target_binary)
        else:
            include_mask = include_mask.float()

        bce_loss = self.criterion(pred_logits, target_binary)  # (N, K)
        masked_loss = bce_loss * include_mask
        total_weight = include_mask.sum().clamp(min=1.0)
        loss = masked_loss.sum() / total_weight

        return loss * self.loss_weight

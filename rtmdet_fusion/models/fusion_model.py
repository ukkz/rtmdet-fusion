"""RTMDet-Fusion heads

Trainable heads on top of the frozen RTMDet-Ins detector:
- BboxFeatureExtractor: samples a grid of Hybrid Encoder features inside each bbox
- HeadBboxHead: regresses a reference bbox from the Ins bbox (head bbox, body reference bbox)
- DCCHead: SimCC keypoint head with unified coordinates (x, y, z) relative to the reference bbox:
  - x: horizontal position in the bbox [0, 1] (image X direction)
  - y: pseudo-depth [0, 1] (reference point = 0.5)
  - z: vertical position in the bbox [0, 1] (image Y flipped, up is positive)
- MaskDepthBranch: per-instance part masks and depth
- Helpers for head orientation and tracking features (export outputs)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.default import (
    NUM_KEYPOINTS, FPN_STRIDES, BBOX_EXPAND_RATIO
)

from rtmdet_fusion.models.encoders import DynamicBinEncoding, SinePositionalEncoding2D


# ==============================================================================
# Attention modules
# ==============================================================================

class KeypointQueryAttention(nn.Module):
    """Keypoint Query Cross-Attention

    Learnable keypoint queries gather information from the grid features via cross-attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_keypoints: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_keypoints = num_keypoints
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.keypoint_queries = nn.Parameter(
            torch.randn(num_keypoints, embed_dim) * 0.02
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        grid_features: torch.Tensor,
        pos_encoding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            grid_features: (N, G, D) where G = grid_size^2
            pos_encoding: (G, D) optional positional encoding for grid

        Returns:
            keypoint_features: (N, K, D)
        """
        N, G, D = grid_features.shape
        K = self.num_keypoints

        if pos_encoding is not None:
            grid_features = grid_features + pos_encoding.unsqueeze(0)

        grid_features = self.norm_kv(grid_features)

        queries = self.keypoint_queries.unsqueeze(0).expand(N, -1, -1)
        queries = self.norm_q(queries)

        q = self.q_proj(queries)
        k = self.k_proj(grid_features)
        v = self.v_proj(grid_features)

        q = q.view(N, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, G, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, G, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, K, D)
        out = self.out_proj(out)
        out = out + self.keypoint_queries.unsqueeze(0)

        return out


class KeypointSelfAttention(nn.Module):
    """Keypoint Self-Attention"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, K, D = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv_proj(x).view(N, K, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, K, D)
        out = self.out_proj(out)

        return out + residual


class FFN(nn.Module):
    """Feed-Forward Network"""

    def __init__(self, embed_dim: int, expansion_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * expansion_factor)
        self.fc2 = nn.Linear(embed_dim * expansion_factor, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual


# ==============================================================================
# Batched bbox feature extractor
# ==============================================================================

def separable_bilinear_sample(
    feat: torch.Tensor,
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
) -> torch.Tensor:
    """Bilinear sampling of an axis-aligned grid via matrix products

    Exactly equivalent to F.grid_sample(mode='bilinear', padding_mode='zeros',
    align_corners=True) when the grid varies only along columns in x and only along rows in y.
    Builds tent-weight matrices Wx, Wy with max(0, 1-|p-j|) and computes Wy · F · Wxᵀ.
    No gather / GridSample is used, so onnx2tf needs no transpose fix-ups and CoreML can
    run it on the ANE (at the cost of more compute).

    Args:
        feat: (1, C, H, W) feature map
        x_norm: (N, G) normalized column coordinates [-1, 1]
        y_norm: (N, G) normalized row coordinates [-1, 1]
    Returns:
        (N, C, G, G) sampled features ([n, c, gy, gx])
    """
    _, C, H, W = feat.shape
    N, G = x_norm.shape
    px = (x_norm + 1.0) * 0.5 * (W - 1)  # (N, G) pixel coordinates
    py = (y_norm + 1.0) * 0.5 * (H - 1)
    jx = torch.arange(W, dtype=feat.dtype, device=feat.device).reshape(1, 1, W)
    iy = torch.arange(H, dtype=feat.dtype, device=feat.device).reshape(1, 1, H)
    wx = torch.relu(1.0 - (px.unsqueeze(-1) - jx).abs())  # (N, G, W)
    wy = torch.relu(1.0 - (py.unsqueeze(-1) - iy).abs())  # (N, G, H)
    t = torch.matmul(feat.reshape(1, C * H, W), wx.transpose(1, 2))  # (N, C*H, G)
    t = t.reshape(N, C, H, G)
    # Multiplying with (N, 1, G, H) directly makes tfjs BatchMatMul flatten the batch
    # dimensions and pair the partial broadcast (1 -> C) incorrectly. Expand explicitly.
    wy = wy.unsqueeze(1).expand(N, C, G, H)
    return torch.matmul(wy, t)  # (N, C, G, H) @ (N, C, H, G) -> (N, C, G, G)


class BboxFeatureExtractor(nn.Module):
    """Extracts grid features from bbox regions

    Samples every FPN level for all bboxes at once (no per-instance loop) and selects the
    level of each instance with a one-hot weighted sum. ONNX-exportable.

    With matmul_sampling=True (set by the exporters), separable_bilinear_sample is used
    instead of F.grid_sample (numerically equivalent, same state_dict).
    Training and the viewer use F.grid_sample.
    """

    matmul_sampling = False

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        grid_size: int = 14  # 14x14 = 196 points (makes use of the P3/P4 resolution)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        # FPN strides
        self.strides = FPN_STRIDES  # [8, 16, 32]

        # 2D Positional Encoding
        self.pos_encoding = SinePositionalEncoding2D(embed_dim=in_channels)

        # Feature projection
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(inplace=True),
        )

        # Positional encoding projection (without SiLU)
        self.pos_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
        )

    def _create_grids_batched(
        self,
        bboxes: torch.Tensor,
        img_size: int
    ) -> torch.Tensor:
        """Build the sampling grids of all bboxes at once

        Args:
            bboxes: (N, 4) [x1, y1, x2, y2]
            img_size: image size

        Returns:
            grids: (N, grid_size, grid_size, 2) normalized coordinates [-1, 1]
        """
        N = bboxes.shape[0]
        device = bboxes.device
        G = self.grid_size

        # Normalize bbox coordinates to [-1, 1]
        x1 = (bboxes[:, 0] / img_size) * 2 - 1  # (N,)
        y1 = (bboxes[:, 1] / img_size) * 2 - 1
        x2 = (bboxes[:, 2] / img_size) * 2 - 1
        y2 = (bboxes[:, 3] / img_size) * 2 - 1

        # Bbox width and height
        w = x2 - x1  # (N,)
        h = y2 - y1

        # Base grid [0, 1] -> (G, G)
        base_x = torch.linspace(0, 1, G, device=device)
        base_y = torch.linspace(0, 1, G, device=device)
        base_grid_y, base_grid_x = torch.meshgrid(base_y, base_x, indexing='ij')
        # (G, G) -> (1, G, G)
        base_grid_x = base_grid_x.unsqueeze(0)
        base_grid_y = base_grid_y.unsqueeze(0)

        # Scale and shift into each bbox
        # (N, 1, 1) * (1, G, G) -> (N, G, G)
        grid_x = x1.view(N, 1, 1) + base_grid_x * w.view(N, 1, 1)
        grid_y = y1.view(N, 1, 1) + base_grid_y * h.view(N, 1, 1)

        # Stack: (N, G, G, 2)
        grids = torch.stack([grid_x, grid_y], dim=-1)

        return grids

    def _create_axes_batched(
        self,
        bboxes: torch.Tensor,
        img_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """1D x / y axes of _create_grids_batched (same formula, for matmul sampling)

        Returns:
            axis_x: (N, grid_size) normalized column coordinates [-1, 1]
            axis_y: (N, grid_size) normalized row coordinates [-1, 1]
        """
        G = self.grid_size
        x1 = (bboxes[:, 0:1] / img_size) * 2 - 1  # (N, 1)
        y1 = (bboxes[:, 1:2] / img_size) * 2 - 1
        x2 = (bboxes[:, 2:3] / img_size) * 2 - 1
        y2 = (bboxes[:, 3:4] / img_size) * 2 - 1
        base = torch.linspace(0, 1, G, device=bboxes.device).reshape(1, G)
        return x1 + base * (x2 - x1), y1 + base * (y2 - y1)

    def _stride_to_level(self, strides: torch.Tensor) -> torch.Tensor:
        """Map strides to FPN levels (tensor ops only)

        Args:
            strides: (N,) stride values

        Returns:
            levels: (N,) FPN level (0, 1, 2)
        """
        # stride: 8 -> level 0, 16 -> level 1, 32 -> level 2 (threshold based)
        levels = torch.zeros_like(strides, dtype=torch.long)
        levels = torch.where(strides > 12, torch.ones_like(levels), levels)
        levels = torch.where(strides > 24, 2 * torch.ones_like(levels), levels)
        return levels

    def forward(
        self,
        fpn_feats: List[torch.Tensor],
        bboxes: torch.Tensor,
        priors: torch.Tensor,
        valid_mask: torch.Tensor,
        img_size: int,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fpn_feats: list of FPN feature maps [(B, C, H0, W0), ...]
                       with batch_indices=None, B=1 and the image is shared by all instances
            bboxes: bbox coordinates (N, 4) [x1, y1, x2, y2]
            priors: prior info (N, 4) [cx, cy, stride_w, stride_h]
            valid_mask: (N,) mask of valid bboxes
            img_size: image size
            batch_indices: (N,) batch index of the image each instance belongs to
                           (None: expand fpn_feats[i] to all instances)

        Returns:
            grid_features: (N, grid_size^2, out_channels)
            pos_encoding: (grid_size^2, out_channels)
        """
        device = bboxes.device
        N = bboxes.shape[0]
        C = self.in_channels
        G = self.grid_size
        G2 = G * G

        if N == 0:
            # No instances
            grid_features = torch.zeros(0, G2, self.out_channels, device=device)
            pos_enc = self.pos_encoding(G, device)
            pos_enc_proj = self.pos_proj(pos_enc)
            return grid_features, pos_enc_proj

        use_matmul = self.matmul_sampling and batch_indices is None

        # === Step 1: build the grids of all bboxes ===
        if use_matmul:
            # The matmul version (export) only needs the 1D x / y coordinates
            # (computed directly: slicing them out of the 4D grid makes onnx2tf mix up axes)
            axis_x, axis_y = self._create_axes_batched(bboxes, img_size)  # (N, G) x2
        else:
            grids = self._create_grids_batched(bboxes, img_size)  # (N, G, G, 2)

        # === Step 2: sample every FPN level ===
        all_sampled = []
        for level, feat_map in enumerate(fpn_feats):
            if use_matmul:
                all_sampled.append(separable_bilinear_sample(feat_map, axis_x, axis_y))
                continue
            if batch_indices is not None:
                # Batched: take the features of the image each instance belongs to
                feat_selected = feat_map[batch_indices]  # (N, C, H, W)
            else:
                # Single image shared by all instances
                feat_selected = feat_map.expand(N, -1, -1, -1)
            sampled = F.grid_sample(
                feat_selected, grids,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )  # (N, C, G, G)
            all_sampled.append(sampled)

        # Stack: (N, 3, C, G, G)
        stacked = torch.stack(all_sampled, dim=1)

        # === Step 3: select the level from the stride ===
        strides = priors[:, 2]  # (N,)
        levels = self._stride_to_level(strides)  # (N,)

        # One-hot weighted sum (avoids GatherElements, WebGPU compatible)
        # stacked: (N, 3, C, G, G), levels: (N,) -> selected: (N, C, G, G)
        num_levels = len(self.strides)
        one_hot = F.one_hot(levels, num_classes=num_levels).float()  # (N, 3)
        one_hot = one_hot.view(N, num_levels, 1, 1, 1)  # (N, 3, 1, 1, 1)
        selected = (stacked * one_hot).sum(dim=1)  # (N, C, G, G)

        # === Step 4: zero out invalid slots ===
        selected = selected * valid_mask.view(N, 1, 1, 1).float()

        # Flatten: (N, C, G, G) -> (N, G*G, C)
        grid_features = selected.view(N, C, G2).transpose(1, 2)  # (N, G2, C)

        # === Step 5: Projection ===
        grid_features = self.proj(grid_features)  # (N, G2, out_channels)

        # Positional encoding
        pos_enc = self.pos_encoding(G, device)  # (G2, in_channels)
        pos_enc_proj = self.pos_proj(pos_enc)  # (G2, out_channels)

        return grid_features, pos_enc_proj


class HeadBboxHead(nn.Module):
    """Reference bbox regression head

    Input: grid features (N, G, D)
    Output: bbox (N, 4) [x1, y1, x2, y2]

    The prediction range is range_scale times the Ins bbox around its center, so the
    predicted box can lie outside the Ins bbox (and outside the image).
    Used for the head bbox (default range) and the body reference bbox
    (range_scale=BODY_BBOX_RANGE_SCALE).
    """

    # Default prediction range relative to the Ins bbox (head bbox)
    BBOX_RANGE_SCALE = 1.25

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        range_scale: float = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.range_scale = range_scale if range_scale is not None else self.BBOX_RANGE_SCALE

        layers = []
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(inplace=True),
            ])
        layers.append(nn.Linear(hidden_dim, 4))
        self.reg_head = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        grid_features: torch.Tensor,
        ins_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """Predict the bbox

        Args:
            grid_features: (N, G, D) grid features
            ins_bboxes: (N, 4) Ins detection bbox [x1, y1, x2, y2]

        Returns:
            head_bboxes: (N, 4) predicted bbox [x1, y1, x2, y2] (absolute coordinates,
                         may be negative / outside the image)
        """
        N = grid_features.shape[0]
        device = grid_features.device

        if N == 0:
            return torch.zeros((0, 4), device=device)

        pooled_features = grid_features.mean(dim=1)  # (N, D)
        offsets = self.reg_head(pooled_features)  # (N, 4)
        offsets = torch.sigmoid(offsets)  # map to [0, 1]

        # Ins bbox center and prediction range
        ins_x1, ins_y1 = ins_bboxes[:, 0], ins_bboxes[:, 1]
        ins_x2, ins_y2 = ins_bboxes[:, 2], ins_bboxes[:, 3]
        ins_w = ins_x2 - ins_x1
        ins_h = ins_y2 - ins_y1

        # Ins bbox center
        center_x = (ins_x1 + ins_x2) / 2
        center_y = (ins_y1 + ins_y2) / 2

        # Range: center +/- range_scale/2 * ins_size
        range_x = ins_w * self.range_scale / 2
        range_y = ins_h * self.range_scale / 2

        # Map offset [0, 1] to [-range, +range]
        # offset=0.5 is the center, 0 the left/top end, 1 the right/bottom end
        head_x1 = center_x + (offsets[:, 0] - 0.5) * 2 * range_x
        head_y1 = center_y + (offsets[:, 1] - 0.5) * 2 * range_y
        head_x2 = center_x + (offsets[:, 2] - 0.5) * 2 * range_x
        head_y2 = center_y + (offsets[:, 3] - 0.5) * 2 * range_y

        head_x1_final = torch.min(head_x1, head_x2)
        head_x2_final = torch.max(head_x1, head_x2)
        head_y1_final = torch.min(head_y1, head_y2)
        head_y2_final = torch.max(head_y1, head_y2)

        head_bboxes = torch.stack([
            head_x1_final, head_y1_final, head_x2_final, head_y2_final
        ], dim=1)

        return head_bboxes


class DCCHead(nn.Module):
    """Dynamic Coordinate Classifier head (unified coordinates)

    Outputs unified coordinates (x, y, z) relative to the reference bbox:
    - x: relative X in the bbox [0, 1]
    - y: relative pseudo-depth [0, 1] (reference point = 0.5)
    - z: relative Z in the bbox [0, 1] (image Y flipped, up is positive)

    All axes share the same scale. With bin_margin > 0 the bins cover
    [-bin_margin, 1 + bin_margin] so points slightly outside the bbox can be represented.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_keypoints: int = NUM_KEYPOINTS,
        num_bins: int = 128,  # shared by all axes
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        bbox_expand_ratio: float = BBOX_EXPAND_RATIO,
        target_sigma: float = 0.04,  # SimCC Gaussian sigma
        dropout: float = 0.2,  # regularization
        bin_margin: float = 0.0  # bin range [-margin, 1 + margin] (points outside the bbox)
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.bbox_expand_ratio = bbox_expand_ratio
        self.target_sigma = target_sigma
        self.bin_margin = bin_margin
        # Bin range [bin_min, bin_max] ([0, 1] when margin=0)
        self.bin_min = -bin_margin
        self.bin_max = 1.0 + bin_margin
        # Constant for the entropy computation (avoids an ONNX TracerWarning)
        self.log_num_bins = math.log(num_bins)
        # Reference entropy for the confidence normalization (a constant in ONNX/CoreML too).
        # confidence = 0 at H_max = log(num_bins) (uniform) and 1.0 at H_target (the ideal
        # Gaussian with target_sigma). Note: training builds the heads with target_sigma=0.03,
        # while the viewer and exporters use 0.025, so at inference a keypoint that reproduces
        # the training target exactly scores about 0.93.
        _ideal_bins = torch.linspace(self.bin_min, self.bin_max, num_bins)
        _ideal = torch.exp(-((_ideal_bins - 0.5) ** 2) / (2 * target_sigma ** 2))
        _ideal = _ideal / _ideal.sum()
        self.target_entropy = float(-(_ideal * torch.log(_ideal + 1e-8)).sum())

        # Attention layers
        self.cross_attn = KeypointQueryAttention(
            embed_dim=hidden_dim,
            num_keypoints=num_keypoints,
            num_heads=num_heads,
            dropout=dropout
        )

        self.self_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.self_attn_layers.append(
                KeypointSelfAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            )
            self.ffn_layers.append(
                FFN(embed_dim=hidden_dim, expansion_factor=4, dropout=dropout)
            )

        # DynamicBinEncoding per axis (x, y, z)
        self.dbe_x = DynamicBinEncoding(embed_dim=hidden_dim, num_bins=num_bins)
        self.dbe_y = DynamicBinEncoding(embed_dim=hidden_dim, num_bins=num_bins)
        self.dbe_z = DynamicBinEncoding(embed_dim=hidden_dim, num_bins=num_bins)

        # Visibility head (as in RTMO)
        # kpt_features (N, K, hidden_dim) -> visibility_logits (N, K)
        self.visibility_head = nn.Linear(hidden_dim, 1)

        # In-frame / out-of-frame head (a binary target independent of visibility)
        self.in_frame_head = nn.Linear(hidden_dim, 1)

        # Bin coordinates over [bin_min, bin_max] ([0, 1] when margin=0)
        self.register_buffer('bin_coords', torch.linspace(self.bin_min, self.bin_max, num_bins))

    def forward(
        self,
        grid_features: torch.Tensor,
        pos_encoding: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            grid_features: (N, G, hidden_dim) G = grid_size^2
            pos_encoding: (G, hidden_dim)

        Returns:
            Dict with heatmap_x, heatmap_y, heatmap_z, visibility_logits, in_frame_logits,
            kpt_features
        """
        device = grid_features.device

        # Cross-attention: grid features -> keypoint features
        kpt_features = self.cross_attn(grid_features, pos_encoding)

        # Self-attention layers
        for self_attn, ffn in zip(self.self_attn_layers, self.ffn_layers):
            kpt_features = self_attn(kpt_features)
            kpt_features = ffn(kpt_features)

        # Heatmaps of the unified coordinates
        bin_enc_x = self.dbe_x.get_bin_encodings(self.num_bins, device)
        bin_enc_y = self.dbe_y.get_bin_encodings(self.num_bins, device)
        bin_enc_z = self.dbe_z.get_bin_encodings(self.num_bins, device)

        heatmap_x = torch.matmul(kpt_features, bin_enc_x.transpose(0, 1))
        heatmap_y = torch.matmul(kpt_features, bin_enc_y.transpose(0, 1))
        heatmap_z = torch.matmul(kpt_features, bin_enc_z.transpose(0, 1))

        # Clamp for numerical stability
        logit_clamp = 50.0
        heatmap_x = heatmap_x.clamp(-logit_clamp, logit_clamp)
        heatmap_y = heatmap_y.clamp(-logit_clamp, logit_clamp)
        heatmap_z = heatmap_z.clamp(-logit_clamp, logit_clamp)

        # Visibility prediction
        # (N, K, hidden_dim) -> (N, K, 1) -> (N, K)
        visibility_logits = self.visibility_head(kpt_features).squeeze(-1)
        visibility_logits = visibility_logits.clamp(-logit_clamp, logit_clamp)

        # In-frame prediction
        in_frame_logits = self.in_frame_head(kpt_features).squeeze(-1)
        in_frame_logits = in_frame_logits.clamp(-logit_clamp, logit_clamp)

        return {
            'heatmap_x': heatmap_x,  # (N, K, num_bins)
            'heatmap_y': heatmap_y,  # (N, K, num_bins)
            'heatmap_z': heatmap_z,  # (N, K, num_bins)
            'visibility_logits': visibility_logits,  # (N, K)
            'in_frame_logits': in_frame_logits,  # (N, K)
            'kpt_features': kpt_features,  # (N, K, hidden_dim)
        }

    def _compute_entropy_confidence(self, prob: torch.Tensor) -> torch.Tensor:
        """Entropy-based confidence, normalized to [0, 1] against the target distribution

        Lower entropy (a peakier distribution) means higher confidence:
        - uniform distribution (H = log(num_bins)): 0.0
        - ideal Gaussian with target_sigma (H = target_entropy): 1.0
        - anything sharper: clamped to 1.0

        SimCC learns a Gaussian target with KL divergence, so the best distribution the model
        can realistically reach is the target itself. Scaling it to 1.0 lets the value be read
        as a 0.00-1.00 score without post-processing.

        Args:
            prob: (N, K, num_bins) probability distribution

        Returns:
            confidence: (N, K) confidence [0, 1]
        """
        # Entropy: H = -sum(p * log(p))
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)
        # 0 at H_max (uniform), 1.0 at H_target (ideal Gaussian)
        denom = max(self.log_num_bins - self.target_entropy, 1e-6)
        confidence = (self.log_num_bins - entropy) / denom
        return confidence.clamp(0, 1)

    def decode_relative(
        self,
        heatmap_x: torch.Tensor,
        heatmap_y: torch.Tensor,
        heatmap_z: torch.Tensor,
        visibility_logits: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode relative coordinates from the heatmaps (soft-argmax, no bbox conversion)

        Args:
            heatmap_x: (N, K, num_bins) X heatmap
            heatmap_y: (N, K, num_bins) Y (depth) heatmap
            heatmap_z: (N, K, num_bins) Z heatmap
            visibility_logits: (N, K) visibility logits (optional)

        Returns:
            rel_coords: (N, K, 3) relative coordinates [rel_x, rel_y, rel_z]
            confidences: (N, K) entropy-based confidence
            visibilities: (N, K) visibility probability [0, 1] (1.0 without logits)
        """
        prob_x = F.softmax(heatmap_x, dim=-1)
        prob_y = F.softmax(heatmap_y, dim=-1)
        prob_z = F.softmax(heatmap_z, dim=-1)

        rel_x = (prob_x * self.bin_coords).sum(dim=-1)
        rel_y = (prob_y * self.bin_coords).sum(dim=-1)
        rel_z = (prob_z * self.bin_coords).sum(dim=-1)

        rel_coords = torch.stack([rel_x, rel_y, rel_z], dim=-1)

        # Entropy-based confidence: higher for peakier distributions
        conf_x = self._compute_entropy_confidence(prob_x)
        conf_y = self._compute_entropy_confidence(prob_y)
        conf_z = self._compute_entropy_confidence(prob_z)
        confidences = (conf_x + conf_y + conf_z) / 3

        # Visibility: sigmoid of logits, or 1.0 if not provided
        if visibility_logits is not None:
            visibilities = torch.sigmoid(visibility_logits)
        else:
            visibilities = torch.ones_like(confidences)

        return rel_coords, confidences, visibilities

    def expand_bbox(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Expand bboxes around their center by bbox_expand_ratio (as in RTMO)"""
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]

        new_w = w * self.bbox_expand_ratio
        new_h = h * self.bbox_expand_ratio

        new_bboxes = torch.stack([
            cx - new_w / 2,
            cy - new_h / 2,
            cx + new_w / 2,
            cy + new_h / 2
        ], dim=-1)

        return new_bboxes

    def generate_target(
        self,
        target_coords: torch.Tensor,
        target_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the SimCC training targets

        Args:
            target_coords: (N, K, 3) normalized coordinates [rel_x, rel_y, rel_z]
            target_weight: (N, K) visibility weight

        Returns:
            target_x: (N, K, num_bins) Gaussian target of the X axis
            target_y: (N, K, num_bins) Gaussian target of the Y axis
            target_z: (N, K, num_bins) Gaussian target of the Z axis
        """
        N, K, _ = target_coords.shape

        # Bin coordinates
        bins = self.bin_coords.view(1, 1, -1)  # (1, 1, num_bins)

        # Target coordinate per axis
        rel_x = target_coords[:, :, 0:1]  # (N, K, 1)
        rel_y = target_coords[:, :, 1:2]  # (N, K, 1)
        rel_z = target_coords[:, :, 2:3]  # (N, K, 1)

        # Gaussian labels
        sigma = self.target_sigma
        target_x = torch.exp(-((bins - rel_x) ** 2) / (2 * sigma ** 2))
        target_y = torch.exp(-((bins - rel_y) ** 2) / (2 * sigma ** 2))
        target_z = torch.exp(-((bins - rel_z) ** 2) / (2 * sigma ** 2))

        # Normalize to probability distributions
        target_x = target_x / (target_x.sum(dim=-1, keepdim=True) + 1e-8)
        target_y = target_y / (target_y.sum(dim=-1, keepdim=True) + 1e-8)
        target_z = target_z / (target_z.sum(dim=-1, keepdim=True) + 1e-8)

        return target_x, target_y, target_z


# ==============================================================================
# Part mask / depth branch
# ==============================================================================

class MaskDepthBranch(nn.Module):
    """Part mask + depth branch

    Fuses the three Hybrid Encoder levels at stride 8 into prototype features and generates
    per-instance mask and depth maps with an instance-specific dynamic conv (bmm based like
    RTMDet-Ins; ONNX/tfjs friendly).

    Output channels (8 ch, stride 8):
        0: clothed full-body mask (amodal)   logit
        1: bare-body full mask (amodal)      logit
        2-5: bare-body parts head/torso/limbs/hands  logit
        6: occluded region (of ch0)          logit
        7: depth (rel_y equivalent: 0.5 = pelvis, pseudo-depth normalized by the body
           reference bbox size)
    """

    NUM_MASK_CHANNELS = 7
    NUM_OUT_CHANNELS = 8  # 7 masks + 1 depth

    def __init__(
        self,
        in_channels: int = 256,
        feat_channels: int = 128,
        num_prototypes: int = 8,
        dyconv_channels: int = 8,
        num_stacked_convs: int = 4,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dyconv_channels = dyconv_channels

        # Prototype features (like RTMDet-Ins MaskFeatModule; Hybrid P3/P4/P5 -> stride 8)
        self.fusion_conv = nn.Conv2d(in_channels * 3, feat_channels, 1)
        convs = []
        for _ in range(num_stacked_convs):
            convs.extend([
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True),
            ])
        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(feat_channels, num_prototypes, 1)

        # Number of dynamic conv parameters
        # layer1: (proto+2) -> C, layer2: C -> C, layer3: C -> NUM_OUT_CHANNELS
        c = dyconv_channels
        self.w1 = (num_prototypes + 2) * c
        self.w2 = c * c
        self.w3 = c * self.NUM_OUT_CHANNELS
        self.b1 = c
        self.b2 = c
        self.b3 = self.NUM_OUT_CHANNELS
        num_gen_params = self.w1 + self.w2 + self.w3 + self.b1 + self.b2 + self.b3

        # Generates the kernels from the instance feature (pooled Ins grid features, 256-d)
        self.kernel_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels, num_gen_params),
        )

    def compute_prototypes(self, hybrid_feats: List[torch.Tensor]) -> torch.Tensor:
        """Build prototype features from the three Hybrid levels

        Returns:
            mask_feat: (B, num_prototypes, H8, W8)  stride 8
        """
        p3 = hybrid_feats[0]
        size = p3.shape[-2:]
        feats = [p3] + [
            F.interpolate(f, size=size, mode='bilinear', align_corners=False)
            for f in hybrid_feats[1:]
        ]
        x = self.fusion_conv(torch.cat(feats, dim=1))
        x = self.stacked_convs(x)
        return self.projection(x)

    def forward(
        self,
        hybrid_feats: List[torch.Tensor],
        instance_feats: torch.Tensor,
        priors: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        stride: int = 8,
    ) -> torch.Tensor:
        """
        Args:
            hybrid_feats: Hybrid Encoder outputs [P3, P4, P5]
            instance_feats: (N, 256) instance features (mean-pooled Ins grid features)
            priors: (N, 4) [cx, cy, stride_w, stride_h]
            batch_indices: (N,) image of each instance (None: all from image 0)

        Returns:
            out: (N, 8, H8, W8) 7 mask logits + 1 depth (stride 8)
        """
        N = instance_feats.shape[0]
        device = instance_feats.device
        mask_feat = self.compute_prototypes(hybrid_feats)  # (B, P, H, W)
        B, P, H, W = mask_feat.shape

        if N == 0:
            return torch.zeros((0, self.NUM_OUT_CHANNELS, H, W), device=device)

        if batch_indices is None:
            feats = mask_feat[0:1].expand(N, -1, -1, -1)
        else:
            feats = mask_feat[batch_indices]  # (N, P, H, W)

        # Relative coordinate features (relative to the prior center, as in the exporters' mask head)
        ys = torch.arange(H, device=device, dtype=torch.float32) * stride
        xs = torch.arange(W, device=device, dtype=torch.float32) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
        rel_x = (priors[:, 0].view(N, 1, 1) - grid_x.unsqueeze(0)) / (stride * 8)
        rel_y = (priors[:, 1].view(N, 1, 1) - grid_y.unsqueeze(0)) / (stride * 8)
        rel_coord = torch.stack([rel_x, rel_y], dim=1)  # (N, 2, H, W)

        feats_flat = feats.reshape(N, P, H * W)
        coord_flat = rel_coord.reshape(N, 2, H * W)

        # Per-instance dynamic conv parameters
        params = self.kernel_head(instance_feats)  # (N, num_gen_params)
        c = self.dyconv_channels
        o = self.NUM_OUT_CHANNELS
        idx = 0
        w1 = params[:, idx:idx + self.w1].view(N, c, P + 2); idx += self.w1
        w2 = params[:, idx:idx + self.w2].view(N, c, c); idx += self.w2
        w3 = params[:, idx:idx + self.w3].view(N, o, c); idx += self.w3
        b1 = params[:, idx:idx + self.b1]; idx += self.b1
        b2 = params[:, idx:idx + self.b2]; idx += self.b2
        b3 = params[:, idx:idx + self.b3]

        # Dynamic conv via bmm (equivalent to 1x1 convs).
        # The first layer's input is the concatenation [prototype features P ch, relative
        # coordinates 2 ch], but it is computed as the sum of two bmm with split weights
        # instead (mathematically equivalent): concatenating a conv output with a computed
        # tensor makes onnx2tf's NHWC conversion disagree on the two input layouts and fail.
        x = (torch.bmm(w1[:, :, :P], feats_flat) + torch.bmm(w1[:, :, P:], coord_flat)
             + b1.unsqueeze(2))
        x = F.relu(x)
        x = torch.bmm(w2, x) + b2.unsqueeze(2)
        x = F.relu(x)
        x = torch.bmm(w3, x) + b3.unsqueeze(2)  # (N, 8, H*W)

        return x.view(N, o, H, W)


# ==============================================================================
# Head orientation and tracking feature helpers (inference / export)
# ==============================================================================

def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 norm over the last dimension (keepdim)

    torch.norm / F.normalize become ReduceL2 with the axes given as an input tensor in
    ONNX (opset 18), which onnx2tf 1.28.8 cannot convert to TF (axis type error in tf.norm).
    Written as sum + sqrt so that no ReduceL2 is generated (used by the inference-only helpers).
    """
    return torch.sqrt((x * x).sum(dim=-1, keepdim=True))


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Same as F.normalize(x, dim=-1), without generating ReduceL2"""
    return x / _l2_norm(x).clamp(min=eps)


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cross product of (N, 3) vectors (same as torch.cross)

    torch.cross expands in ONNX into selections on a 3D tensor whose axes onnx2tf mixes up,
    so only 2D component slices are used.
    """
    ax, ay, az = a[:, 0:1], a[:, 1:2], a[:, 2:3]
    bx, by, bz = b[:, 0:1], b[:, 1:2], b[:, 2:3]
    return torch.cat([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], dim=-1)


def compute_head_quaternion(keypoints_rel: torch.Tensor) -> torch.Tensor:
    """Head orientation quaternion from the relative keypoint coordinates

    Reference orientation (q = [1, 0, 0, 0]):
    - forward (center->nose): -Y axis (facing the camera)
    - up (center->top): +Z axis
    - right: -X axis

    Coordinate system:
    - X: image right (positive = right)
    - Y: depth (positive = into the screen)
    - Z: image up (positive = up)

    Args:
        keypoints_rel: (N, 11, 3) relative coordinates [rel_x, rel_y, rel_z]
            - rel_x: X in the head bbox [0, 1]
            - rel_y: depth [0, 1] (center=0.5)
            - rel_z: Y in the head bbox [0, 1] (up is 1)

    Returns:
        quaternion: (N, 4) [w, x, y, z], normalized
    """
    N = keypoints_rel.shape[0]
    device = keypoints_rel.device

    if N == 0:
        return torch.zeros((0, 4), device=device)

    # Take the keypoints from a 2D-flattened tensor (selecting along axis 1 of a 3D tensor
    # makes onnx2tf misread NWC/NCW and pick from a different axis)
    kpts_flat = keypoints_rel.reshape(N, -1)  # (N, K*3)
    center = kpts_flat[:, 0:3]    # (N, 3) center (index 0)
    top = kpts_flat[:, 3:6]       # (N, 3) top (index 1)
    nose = kpts_flat[:, 18:21]    # (N, 3) nose (index 6)

    # Current orientation vectors
    forward = nose - center       # forward (direction of the nose)
    up_raw = top - center         # up (raw)

    # Normalize (guard against zero vectors)
    forward_norm = _l2_norm(forward).clamp(min=1e-6)
    forward = forward / forward_norm

    # Gram-Schmidt orthogonalization
    # right = forward x up_raw
    right = _cross(forward, up_raw)
    right_norm = _l2_norm(right).clamp(min=1e-6)
    right = right / right_norm

    # Recompute up (guarantees orthogonality)
    up = _cross(right, forward)

    # Rotation matrix
    # Reference: forward_ref = [0, -1, 0], up_ref = [0, 0, 1], right_ref = [-1, 0, 0]
    # R @ ref_vec = current_vec:
    # R @ [-1, 0, 0]^T = right   => R[:, 0] = -right
    # R @ [0, -1, 0]^T = forward => R[:, 1] = -forward
    # R @ [0, 0, 1]^T = up       => R[:, 2] = up
    # so R = [-right | -forward | up] (R[i][j] = component i of column vector j).
    # Assembling R as a 3D tensor and picking elements makes onnx2tf mix up axes, so the
    # needed elements are computed directly from the (N, 1) vector components.
    rx, ry, rz = right[:, 0:1], right[:, 1:2], right[:, 2:3]
    fx, fy, fz = forward[:, 0:1], forward[:, 1:2], forward[:, 2:3]
    ux, uy, uz = up[:, 0:1], up[:, 1:2], up[:, 2:3]

    # Rotation matrix -> quaternion (simplified Shepperd method).
    # Typical head poses (within +/-90 deg) mostly have trace > 0, where this is stable.
    trace = -rx - fy + uz  # R00 + R11 + R22

    # w = 0.5 * sqrt(1 + trace), clamped for stability
    w = 0.5 * torch.sqrt(torch.clamp(1.0 + trace, min=1e-8))

    # Division by 4w (guard against small w)
    w_clamped = w.clamp(min=1e-4)
    inv_4w = 0.25 / w_clamped

    x = (-fz - uy) * inv_4w   # R21 - R12
    y = (ux + rz) * inv_4w    # R02 - R20
    z = (-ry + fx) * inv_4w   # R10 - R01

    # Concatenate and normalize
    quaternion = torch.cat([w, x, y, z], dim=-1)
    quaternion = _l2_normalize(quaternion)

    return quaternion


def compute_appearance_features(grid_features: torch.Tensor) -> torch.Tensor:
    """Appearance feature from the grid features

    Args:
        grid_features: (N, G, D) grid features

    Returns:
        appearance_features: (N, D) L2-normalized appearance features
    """
    if grid_features.shape[0] == 0:
        return torch.zeros((0, grid_features.shape[-1]), device=grid_features.device)

    # Mean pooling
    features = grid_features.mean(dim=1)  # (N, D)

    # L2 normalization
    features = _l2_normalize(features)

    return features


def compute_pose_features(kpt_features: torch.Tensor) -> torch.Tensor:
    """Pose feature from the keypoint features

    Args:
        kpt_features: (N, K, D) keypoint features

    Returns:
        pose_features: (N, D) L2-normalized pose features
    """
    if kpt_features.shape[0] == 0:
        return torch.zeros((0, kpt_features.shape[-1]), device=kpt_features.device)

    # Mean pooling
    features = kpt_features.mean(dim=1)  # (N, D)

    # L2 normalization
    features = _l2_normalize(features)

    return features

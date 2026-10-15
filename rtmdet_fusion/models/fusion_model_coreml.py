"""CoreML versions of the model components

They use the same weights as the main model (rtmdet_fusion.models.fusion_model): the state_dict
keys are identical, so a checkpoint from tools/train.py loads directly. The forward passes are
written for jit.trace + coremltools conversion:
1. No shape-dependent control flow (sizes come from constants, not from shape[0])
2. bbox feature sampling by matrix multiplication (no grid_sample / gather)
3. No N==0 guards
4. No int/long casts (float ops throughout)
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
from rtmdet_fusion.models.fusion_model import MaskDepthBranch, separable_bilinear_sample


# ==============================================================================
# Attention modules
# ==============================================================================

class KeypointQueryAttentionCoreML(nn.Module):
    """Keypoint Query Cross-Attention (CoreML version)

    Differences from the main model:
    - No tuple unpacking of the input shape
    - No N==0 guard
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
            grid_features: (N, G, D)
            pos_encoding: (G, D)
        Returns:
            keypoint_features: (N, K, D)
        """
        H = self.num_heads
        HD = self.head_dim

        if pos_encoding is not None:
            grid_features = grid_features + pos_encoding.unsqueeze(0)

        grid_features = self.norm_kv(grid_features)

        queries = self.keypoint_queries.unsqueeze(0).expand(
            grid_features.shape[0], -1, -1
        )
        queries = self.norm_q(queries)

        q = self.q_proj(queries)
        k = self.k_proj(grid_features)
        v = self.v_proj(grid_features)

        # Split only the last dimension (N stays symbolic)
        q = q.unflatten(-1, (H, HD)).transpose(1, 2)   # (N, H, K, HD)
        k = k.unflatten(-1, (H, HD)).transpose(1, 2)   # (N, H, G, HD)
        v = v.unflatten(-1, (H, HD)).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                     # (N, H, K, HD)
        out = out.transpose(1, 2).flatten(-2)            # (N, K, D)
        out = self.out_proj(out)
        return out + self.keypoint_queries.unsqueeze(0)


class KeypointSelfAttentionCoreML(nn.Module):
    """Keypoint Self-Attention (CoreML version)"""

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
        H = self.num_heads
        HD = self.head_dim
        residual = x

        x = self.norm(x)
        # unflatten + indexing instead of tuple unpacking
        qkv = self.qkv_proj(x).unflatten(-1, (3, H, HD))  # (N, K, 3, H, HD)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, N, H, K, HD)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).flatten(-2)  # (N, K, D)
        out = self.out_proj(out)
        return out + residual


class FFNCoreML(nn.Module):
    """Feed-Forward Network (CoreML version, same structure as the main model)"""

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
# Bbox Feature Extractor
# ==============================================================================

class BboxFeatureExtractorCoreML(nn.Module):
    """Extract grid features from bbox regions (CoreML version)

    Differences from the main model:
    - Sampling by matrix multiplication (separable_bilinear_sample) instead of F.grid_sample;
      numerically equivalent, no GridSample / gather ops
    - FPN level selection by float weights instead of F.one_hot (no int casts)
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        grid_size: int = 14,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.strides = FPN_STRIDES

        self.pos_encoding = SinePositionalEncoding2D(embed_dim=in_channels)

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(inplace=True),
        )

        self.pos_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(
        self,
        fpn_feats: List[torch.Tensor],
        bboxes: torch.Tensor,
        priors: torch.Tensor,
        valid_mask: torch.Tensor,
        img_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fpn_feats: [(1,C,H,W), ...] FPN feature maps
            bboxes: (N, 4) [x1, y1, x2, y2]
            priors: (N, 4) [cx, cy, stride_w, stride_h]
            valid_mask: (N,)
            img_size: input image size
        Returns:
            grid_features: (N, G^2, out_channels)
            pos_encoding: (G^2, out_channels)
        """
        device = bboxes.device
        G = self.grid_size
        G2 = G * G

        # Sample every FPN level along the bbox x / y axes
        # (same formula as the main model's _create_axes_batched)
        x1 = (bboxes[:, 0:1] / img_size) * 2 - 1
        y1 = (bboxes[:, 1:2] / img_size) * 2 - 1
        x2 = (bboxes[:, 2:3] / img_size) * 2 - 1
        y2 = (bboxes[:, 3:4] / img_size) * 2 - 1
        base = torch.linspace(0, 1, G, device=device).reshape(1, G)
        axis_x = x1 + base * (x2 - x1)
        axis_y = y1 + base * (y2 - y1)
        all_sampled = [separable_bilinear_sample(f, axis_x, axis_y) for f in fpn_feats]

        stacked = torch.stack(all_sampled, dim=1)  # (N, 3, C, G, G)

        # FPN level selection from the prior stride (float ops only)
        strides = priors[:, 2]
        level_weights = _stride_to_weights(strides)  # (N, 3)
        level_weights = level_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (N, 3, 1, 1, 1)
        selected = (stacked * level_weights).sum(dim=1)  # (N, C, G, G)

        selected = selected * valid_mask.float().unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # Flatten + projection
        C = selected.shape[1]
        grid_features = selected.reshape(selected.shape[0], C, G2).transpose(1, 2)
        grid_features = self.proj(grid_features)

        pos_enc = self.pos_encoding(G, device)
        pos_enc_proj = self.pos_proj(pos_enc)

        return grid_features, pos_enc_proj


def _stride_to_weights(strides: torch.Tensor) -> torch.Tensor:
    """Stride -> FPN level weights (float one-hot, no int/long casts)

    stride 8 -> [1,0,0], stride 16 -> [0,1,0], stride 32 -> [0,0,1]
    """
    is_level0 = (strides <= 12.0).float()
    is_level2 = (strides > 24.0).float()
    is_level1 = 1.0 - is_level0 - is_level2
    return torch.stack([is_level0, is_level1, is_level2], dim=-1)


# ==============================================================================
# HeadBboxHead
# ==============================================================================

class HeadBboxHeadCoreML(nn.Module):
    """Head bbox regression head (CoreML version, no N==0 guard)

    range_scale is configurable so the same head also serves as the body reference bbox head
    (same specification as the main model).
    """

    BBOX_RANGE_SCALE = 1.25

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        range_scale: Optional[float] = None,
    ):
        super().__init__()
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
        """
        Args:
            grid_features: (N, G, D)
            ins_bboxes: (N, 4) [x1, y1, x2, y2]
        Returns:
            head_bboxes: (N, 4) [x1, y1, x2, y2]
        """
        pooled = grid_features.mean(dim=1)
        offsets = torch.sigmoid(self.reg_head(pooled))

        ins_x1, ins_y1 = ins_bboxes[:, 0], ins_bboxes[:, 1]
        ins_x2, ins_y2 = ins_bboxes[:, 2], ins_bboxes[:, 3]
        cx = (ins_x1 + ins_x2) * 0.5
        cy = (ins_y1 + ins_y2) * 0.5
        range_x = (ins_x2 - ins_x1) * self.range_scale * 0.5
        range_y = (ins_y2 - ins_y1) * self.range_scale * 0.5

        raw_x1 = cx + (offsets[:, 0] - 0.5) * 2 * range_x
        raw_y1 = cy + (offsets[:, 1] - 0.5) * 2 * range_y
        raw_x2 = cx + (offsets[:, 2] - 0.5) * 2 * range_x
        raw_y2 = cy + (offsets[:, 3] - 0.5) * 2 * range_y

        x1 = torch.min(raw_x1, raw_x2)
        x2 = torch.max(raw_x1, raw_x2)
        y1 = torch.min(raw_y1, raw_y2)
        y2 = torch.max(raw_y1, raw_y2)

        return torch.stack([x1, y1, x2, y2], dim=-1)


# ==============================================================================
# DCCHead
# ==============================================================================

class DCCHeadCoreML(nn.Module):
    """Dynamic Coordinate Classifier Head (CoreML version)

    Differences from the main model:
    - Uses the CoreML attention modules
    - No shape unpacking
    - No N==0 guard
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_keypoints: int = NUM_KEYPOINTS,
        num_bins: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        bbox_expand_ratio: float = BBOX_EXPAND_RATIO,
        target_sigma: float = 0.04,
        dropout: float = 0.2,
        bin_margin: float = 0.0,  # bins span [-margin, 1+margin] (points outside the bbox)
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.bbox_expand_ratio = bbox_expand_ratio
        self.target_sigma = target_sigma
        self.bin_min = -bin_margin
        self.bin_max = 1.0 + bin_margin
        self.log_num_bins = math.log(num_bins)
        # Reference entropy for confidence normalization (same logic as the main model):
        # confidence = 0 at H_max = log(num_bins), 1.0 at H_target
        _ideal_bins = torch.linspace(self.bin_min, self.bin_max, num_bins)
        _ideal = torch.exp(-((_ideal_bins - 0.5) ** 2) / (2 * target_sigma ** 2))
        _ideal = _ideal / _ideal.sum()
        self.target_entropy = float(-(_ideal * torch.log(_ideal + 1e-8)).sum())

        self.cross_attn = KeypointQueryAttentionCoreML(
            embed_dim=hidden_dim,
            num_keypoints=num_keypoints,
            num_heads=num_heads,
            dropout=dropout
        )

        self.self_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.self_attn_layers.append(
                KeypointSelfAttentionCoreML(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            )
            self.ffn_layers.append(
                FFNCoreML(embed_dim=hidden_dim, expansion_factor=4, dropout=dropout)
            )

        self.dbe_x = DynamicBinEncoding(embed_dim=hidden_dim, num_bins=num_bins)
        self.dbe_y = DynamicBinEncoding(embed_dim=hidden_dim, num_bins=num_bins)
        self.dbe_z = DynamicBinEncoding(embed_dim=hidden_dim, num_bins=num_bins)

        self.visibility_head = nn.Linear(hidden_dim, 1)
        # In-frame / out-of-frame head (binary, independent of visibility)
        self.in_frame_head = nn.Linear(hidden_dim, 1)

        self.register_buffer('bin_coords', torch.linspace(self.bin_min, self.bin_max, num_bins))

    def forward(
        self,
        grid_features: torch.Tensor,
        pos_encoding: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        device = grid_features.device

        kpt_features = self.cross_attn(grid_features, pos_encoding)

        for self_attn, ffn in zip(self.self_attn_layers, self.ffn_layers):
            kpt_features = self_attn(kpt_features)
            kpt_features = ffn(kpt_features)

        bin_enc_x = self.dbe_x.get_bin_encodings(self.num_bins, device)
        bin_enc_y = self.dbe_y.get_bin_encodings(self.num_bins, device)
        bin_enc_z = self.dbe_z.get_bin_encodings(self.num_bins, device)

        heatmap_x = torch.matmul(kpt_features, bin_enc_x.transpose(0, 1))
        heatmap_y = torch.matmul(kpt_features, bin_enc_y.transpose(0, 1))
        heatmap_z = torch.matmul(kpt_features, bin_enc_z.transpose(0, 1))

        logit_clamp = 50.0
        heatmap_x = heatmap_x.clamp(-logit_clamp, logit_clamp)
        heatmap_y = heatmap_y.clamp(-logit_clamp, logit_clamp)
        heatmap_z = heatmap_z.clamp(-logit_clamp, logit_clamp)

        visibility_logits = self.visibility_head(kpt_features).squeeze(-1)
        visibility_logits = visibility_logits.clamp(-logit_clamp, logit_clamp)

        in_frame_logits = self.in_frame_head(kpt_features).squeeze(-1)
        in_frame_logits = in_frame_logits.clamp(-logit_clamp, logit_clamp)

        return {
            'heatmap_x': heatmap_x,
            'heatmap_y': heatmap_y,
            'heatmap_z': heatmap_z,
            'visibility_logits': visibility_logits,
            'in_frame_logits': in_frame_logits,
            'kpt_features': kpt_features,
        }

    def decode_relative(
        self,
        heatmap_x: torch.Tensor,
        heatmap_y: torch.Tensor,
        heatmap_z: torch.Tensor,
        visibility_logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode bbox-relative coordinates from the heatmaps"""
        prob_x = F.softmax(heatmap_x, dim=-1)
        prob_y = F.softmax(heatmap_y, dim=-1)
        prob_z = F.softmax(heatmap_z, dim=-1)

        rel_x = (prob_x * self.bin_coords).sum(dim=-1)
        rel_y = (prob_y * self.bin_coords).sum(dim=-1)
        rel_z = (prob_z * self.bin_coords).sum(dim=-1)

        rel_coords = torch.stack([rel_x, rel_y, rel_z], dim=-1)

        # Entropy-based confidence
        conf_x = self._entropy_confidence(prob_x)
        conf_y = self._entropy_confidence(prob_y)
        conf_z = self._entropy_confidence(prob_z)
        confidences = (conf_x + conf_y + conf_z) / 3.0

        if visibility_logits is not None:
            visibilities = torch.sigmoid(visibility_logits)
        else:
            visibilities = torch.ones_like(confidences)

        return rel_coords, confidences, visibilities

    def _entropy_confidence(self, prob: torch.Tensor) -> torch.Tensor:
        """Entropy-based confidence, normalized to [0, 1] against the target distribution

        0 at H_max (uniform distribution), 1.0 at H_target (ideal Gaussian).
        Same specification as the main model's DCCHead._compute_entropy_confidence.
        """
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)
        denom = max(self.log_num_bins - self.target_entropy, 1e-6)
        return ((self.log_num_bins - entropy) / denom).clamp(0, 1)

    def expand_bbox(self, bboxes: torch.Tensor) -> torch.Tensor:
        cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        new_w = w * self.bbox_expand_ratio
        new_h = h * self.bbox_expand_ratio
        return torch.stack([
            cx - new_w * 0.5, cy - new_h * 0.5,
            cx + new_w * 0.5, cy + new_h * 0.5
        ], dim=-1)


# ==============================================================================
# MaskDepthBranch
# ==============================================================================

class MaskDepthBranchCoreML(MaskDepthBranch):
    """Part mask + depth branch (CoreML version; state_dict identical to the main model)

    Differences from the main model:
    - Feature map size and instance count come from constant arguments (no int ops from shape access)
    - No N==0 guard and no batch_indices (single image)
    """

    def forward(
        self,
        hybrid_feats: List[torch.Tensor],
        instance_feats: torch.Tensor,
        priors: torch.Tensor,
        img_size: int,
        num_instances: int,
    ) -> torch.Tensor:
        """
        Args:
            hybrid_feats: Hybrid Encoder outputs [P3, P4, P5] (batch 1)
            instance_feats: (N, 256) instance features (mean-pooled ins grid features)
            priors: (N, 4) [cx, cy, stride_w, stride_h]
            img_size: input resolution
            num_instances: N (constant)
        Returns:
            out: (N, 8, img_size/8, img_size/8) 7 mask logit channels + 1 depth channel
        """
        device = instance_feats.device
        stride = FPN_STRIDES[0]
        N = num_instances
        H = img_size // stride
        W = H
        P = self.num_prototypes
        c = self.dyconv_channels
        o = self.NUM_OUT_CHANNELS

        # Prototype features (Hybrid P3/P4/P5 -> stride 8, same as the main model's compute_prototypes)
        feats = [hybrid_feats[0]] + [
            F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False)
            for f in hybrid_feats[1:]
        ]
        x = self.fusion_conv(torch.cat(feats, dim=1))
        x = self.stacked_convs(x)
        mask_feat = self.projection(x).expand(N, -1, -1, -1)  # (N, P, H, W)

        # Relative coordinate features (relative to the prior center)
        ys = torch.arange(H, dtype=torch.float32, device=device) * stride
        xs = torch.arange(W, dtype=torch.float32, device=device) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        rel_x = (priors[:, 0].reshape(N, 1, 1) - grid_x.unsqueeze(0)) / (stride * 8)
        rel_y = (priors[:, 1].reshape(N, 1, 1) - grid_y.unsqueeze(0)) / (stride * 8)
        rel_coord = torch.stack([rel_x, rel_y], dim=1)  # (N, 2, H, W)

        feats_flat = mask_feat.reshape(N, P, H * W)
        coord_flat = rel_coord.reshape(N, 2, H * W)

        # Per-instance dynamic conv parameters (same split order as the main model)
        params = self.kernel_head(instance_feats)
        idx = 0
        w1 = params[:, idx:idx + self.w1].reshape(N, c, P + 2); idx += self.w1
        w2 = params[:, idx:idx + self.w2].reshape(N, c, c); idx += self.w2
        w3 = params[:, idx:idx + self.w3].reshape(N, o, c); idx += self.w3
        b1 = params[:, idx:idx + self.b1]; idx += self.b1
        b2 = params[:, idx:idx + self.b2]; idx += self.b2
        b3 = params[:, idx:idx + self.b3]

        # First layer as a sum of bmm over the split weights instead of concatenating the inputs
        # (same formula as the main model)
        x = F.relu(torch.bmm(w1[:, :, :P], feats_flat) + torch.bmm(w1[:, :, P:], coord_flat)
                   + b1.unsqueeze(2))
        x = F.relu(torch.bmm(w2, x) + b2.unsqueeze(2))
        x = torch.bmm(w3, x) + b3.unsqueeze(2)  # (N, 8, H*W)
        return x.reshape(N, o, H, W)


# ==============================================================================
# Utilities
# ==============================================================================

def compute_head_quaternion(keypoints_rel: torch.Tensor) -> torch.Tensor:
    """Head orientation quaternion from the keypoints (no N==0 guard)"""
    center = keypoints_rel[:, 0]
    top = keypoints_rel[:, 1]
    nose = keypoints_rel[:, 6]

    fwd = nose - center
    up_raw = top - center

    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    right = torch.cross(fwd, up_raw, dim=-1)
    right = right / right.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    up = torch.cross(right, fwd, dim=-1)

    R = torch.stack([-right, -fwd, up], dim=-1)  # (N, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    s = torch.sqrt((trace + 1.0).clamp(min=1e-8)) * 2.0
    w = s * 0.25
    x = (R[:, 2, 1] - R[:, 1, 2]) / s.clamp(min=1e-4)
    y = (R[:, 0, 2] - R[:, 2, 0]) / s.clamp(min=1e-4)
    z = (R[:, 1, 0] - R[:, 0, 1]) / s.clamp(min=1e-4)

    quat = torch.stack([w, x, y, z], dim=-1)
    return F.normalize(quat, dim=-1)

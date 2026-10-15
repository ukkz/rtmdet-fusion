"""HybridEncoder (after the RTMO paper)

Takes the CSPNeXt C3, C4, C5 features and produces P3, P4, P5 with FPN + PAN.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CSPRepLayer


class HybridEncoder(nn.Module):
    """Hybrid Encoder (after the RTMO paper)

    Takes the CSPNeXt C3, C4, C5 features and produces P3, P4, P5 with FPN + PAN.
    A trainable encoder independent of the (frozen) RTMDet-Ins PAFPN.

    Architecture:
    C3 (stride 8)  ─┐
    C4 (stride 16) ─┼─→ Channel Projection ─→ FPN (top-down) ─→ PAN (bottom-up) ─→ P3, P4, P5
    C5 (stride 32) ─┘

    Args:
        in_channels: input channels [C3_ch, C4_ch, C5_ch]
        hidden_dim: internal channels (default 256)
        out_channels: output channels (default 256)
        num_csp_blocks: blocks per CSPRepLayer (default 3)
        widen_factor: CSPRepLayer width factor (default 0.5)
    """

    def __init__(
        self,
        in_channels: List[int] = [256, 256, 256],  # RTMDet-S: [128, 256, 512] or adjusted
        hidden_dim: int = 256,
        out_channels: int = 256,
        num_csp_blocks: int = 3,
        widen_factor: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # Channel projection (every level to hidden_dim)
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ) for ch in in_channels
        ])

        # FPN: Top-down pathway (C5 → C4 → C3)
        # P5 = proj(C5)
        # P4 = CSPRep(concat(upsample(P5), proj(C4)))
        # P3 = CSPRep(concat(upsample(P4), proj(C3)))
        self.fpn_lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ) for _ in range(2)  # for C4, C3
        ])

        self.fpn_csp_layers = nn.ModuleList([
            CSPRepLayer(hidden_dim * 2, hidden_dim, num_csp_blocks, widen_factor)
            for _ in range(2)  # for P4, P3
        ])

        # PAN: Bottom-up pathway (P3 → P4 → P5)
        # N3 = P3
        # N4 = CSPRep(concat(downsample(N3), P4))
        # N5 = CSPRep(concat(downsample(N4), P5))
        self.pan_downsample_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ) for _ in range(2)  # for N3->N4, N4->N5
        ])

        self.pan_csp_layers = nn.ModuleList([
            CSPRepLayer(hidden_dim * 2, hidden_dim, num_csp_blocks, widen_factor)
            for _ in range(2)  # for N4, N5
        ])

        # Output projection (P3, P4, P5 -> out_channels)
        self.output_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ) for _ in range(3)  # for P3, P4, P5
        ])

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            features: [C3, C4, C5] backbone features
                C3: (B, C3_ch, H/8, W/8)
                C4: (B, C4_ch, H/16, W/16)
                C5: (B, C5_ch, H/32, W/32)

        Returns:
            [P3, P4, P5]: encoded features
                P3: (B, out_channels, H/8, W/8)
                P4: (B, out_channels, H/16, W/16)
                P5: (B, out_channels, H/32, W/32)
        """
        assert len(features) == 3, f"Expected 3 features, got {len(features)}"

        # Channel projection
        c3 = self.input_proj[0](features[0])  # (B, hidden_dim, H/8, W/8)
        c4 = self.input_proj[1](features[1])  # (B, hidden_dim, H/16, W/16)
        c5 = self.input_proj[2](features[2])  # (B, hidden_dim, H/32, W/32)

        # FPN: Top-down
        p5 = c5  # top level as is

        # P4 = CSPRep(concat(upsample(P5), lateral(C4)))
        p5_up = F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p4_lateral = self.fpn_lateral_convs[0](c4)
        p4 = self.fpn_csp_layers[0](torch.cat([p5_up, p4_lateral], dim=1))

        # P3 = CSPRep(concat(upsample(P4), lateral(C3)))
        p4_up = F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p3_lateral = self.fpn_lateral_convs[1](c3)
        p3 = self.fpn_csp_layers[1](torch.cat([p4_up, p3_lateral], dim=1))

        # PAN: Bottom-up
        n3 = p3  # bottom level as is

        # N4 = CSPRep(concat(downsample(N3), P4))
        n3_down = self.pan_downsample_convs[0](n3)
        n4 = self.pan_csp_layers[0](torch.cat([n3_down, p4], dim=1))

        # N5 = CSPRep(concat(downsample(N4), P5))
        n4_down = self.pan_downsample_convs[1](n4)
        n5 = self.pan_csp_layers[1](torch.cat([n4_down, p5], dim=1))

        # Output projection (P3, P4, P5)
        out_p3 = self.output_proj[0](n3)  # (B, out_channels, H/8, W/8)
        out_p4 = self.output_proj[1](n4)  # (B, out_channels, H/16, W/16)
        out_p5 = self.output_proj[2](n5)  # (B, out_channels, H/32, W/32)

        return [out_p3, out_p4, out_p5]

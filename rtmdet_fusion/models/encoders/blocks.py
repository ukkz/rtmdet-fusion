"""Building blocks

RepVGGBlock and CSPRepLayer used by the HybridEncoder.
"""

import torch
import torch.nn as nn


class RepVGGBlock(nn.Module):
    """RepVGG block (re-parameterizable for inference)

    Training: 3x3 conv + 1x1 conv + identity (each with BN)
    Inference: can be fused into a single 3x3 conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups

        # 3x3 branch
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 1x1 branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Identity branch (only when in_channels == out_channels and stride == 1)
        self.identity = nn.BatchNorm2d(in_channels) if (
            in_channels == out_channels and stride == 1
        ) else None

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)


class CSPRepLayer(nn.Module):
    """CSP (Cross Stage Partial) + RepVGG layer for the RTMO-style Hybrid Encoder

    Splits the input into two branches, runs RepVGG blocks on one and concatenates them.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        widen_factor: float = 1.0
    ):
        super().__init__()
        hidden_channels = int(out_channels * widen_factor)

        # Two input branches
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )

        # RepVGG blocks
        self.blocks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels)
            for _ in range(num_blocks)
        ])

        # Output fusion
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two branches
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        # RepVGG blocks on one branch
        x1 = self.blocks(x1)

        # Concatenate
        out = torch.cat([x1, x2], dim=1)
        return self.conv3(out)

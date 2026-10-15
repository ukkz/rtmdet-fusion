"""Positional encodings

DynamicBinEncoding (DBE) and SinePositionalEncoding2D.
"""

import math

import torch
import torch.nn as nn


class DynamicBinEncoding(nn.Module):
    """Dynamic Bin Encoding (DBE), after the RTMO paper

    Encoding from the RTMO paper:
    PE(x_i)_c = sin(x_i / t^(c/C)) for even c
    PE(x_i)_c = cos(x_i / t^((c-1)/C)) for odd c

    followed by a learnable linear transform φ:
    p̂_k(x_i) = softmax(f_k · φ(PE(x_i)))

    Args:
        embed_dim: encoding dimension
        num_bins: number of bins
        temperature: temperature (default 10000.0)
    """

    def __init__(
        self,
        embed_dim: int,
        num_bins: int,
        temperature: float = 10000.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bins = num_bins
        self.temperature = temperature

        # Learnable linear transform φ (RTMO)
        self.linear_transform = nn.Linear(embed_dim, embed_dim)

        # Initialization
        nn.init.xavier_uniform_(self.linear_transform.weight)
        nn.init.zeros_(self.linear_transform.bias)

    def forward(self, bin_coords: torch.Tensor) -> torch.Tensor:
        """Positional encoding of bin coordinates

        Args:
            bin_coords: (B,) normalized bin coordinates in [0, 1]

        Returns:
            encoding: (B, embed_dim) positional encoding
        """
        B = bin_coords.shape[0]
        device = bin_coords.device

        # Dimension indices
        dim_indices = torch.arange(0, self.embed_dim, device=device).float()

        # Denominator as in the RTMO paper
        # t^(c/C) where t=temperature, c=dim_index, C=embed_dim
        div_term = torch.pow(
            self.temperature,
            (dim_indices // 2 * 2) / self.embed_dim
        )

        # Encoding
        # bin_coords: (B,) -> (B, 1)
        # div_term: (D,) -> (1, D)
        angles = bin_coords.unsqueeze(1) / div_term.unsqueeze(0)  # (B, D)

        # Alternating sin/cos
        pe = torch.zeros(B, self.embed_dim, device=device)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])

        # Learnable linear transform
        pe = self.linear_transform(pe)

        return pe

    def get_bin_encodings(self, num_bins: int, device: torch.device) -> torch.Tensor:
        """Positional encodings of all bins

        The encoding always spans [0, 1] over the bins; the head's bin_coords (which may
        include a margin) only affect decoding.

        Args:
            num_bins: number of bins
            device: device

        Returns:
            encodings: (num_bins, embed_dim)
        """
        # Normalized coordinates in [0, 1]
        bin_coords = torch.linspace(0, 1, num_bins, device=device)
        return self.forward(bin_coords)


class SinePositionalEncoding2D(nn.Module):
    """2D Sine Positional Encoding for grid features

    Adds an X/Y coordinate encoding to every grid position.
    """

    def __init__(self, embed_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Half of embed_dim for X, half for Y
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        self.dim_per_axis = embed_dim // 2

    def forward(self, grid_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            grid_size: grid size (H = W = grid_size)
            device: device

        Returns:
            pos_encoding: (grid_size * grid_size, embed_dim)
        """
        # Normalized coordinates [0, 1]
        y_coords = torch.linspace(0, 1, grid_size, device=device)
        x_coords = torch.linspace(0, 1, grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # flatten to (H*W,)
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        # Sinusoidal encoding
        dim = torch.arange(0, self.dim_per_axis, 2, device=device).float()
        div_term = torch.exp(dim * (-math.log(self.temperature) / self.dim_per_axis))

        # X encoding
        pe_x = torch.zeros(grid_size * grid_size, self.dim_per_axis, device=device)
        pe_x[:, 0::2] = torch.sin(grid_x.unsqueeze(1) * div_term * math.pi)
        pe_x[:, 1::2] = torch.cos(grid_x.unsqueeze(1) * div_term * math.pi)

        # Y encoding
        pe_y = torch.zeros(grid_size * grid_size, self.dim_per_axis, device=device)
        pe_y[:, 0::2] = torch.sin(grid_y.unsqueeze(1) * div_term * math.pi)
        pe_y[:, 1::2] = torch.cos(grid_y.unsqueeze(1) * div_term * math.pi)

        # Concatenate
        pos_encoding = torch.cat([pe_x, pe_y], dim=-1)  # (H*W, embed_dim)

        return pos_encoding

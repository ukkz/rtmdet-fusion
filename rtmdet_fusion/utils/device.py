"""Device utilities

Detection and setup of the PyTorch device.
"""

import torch


def get_device() -> str:
    """Return the best available device

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        device name ('cuda', 'mps', 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def setup_device_optimizations(device: str) -> None:
    """Apply device-specific optimizations

    Args:
        device: device name ('cuda', 'mps', 'cpu')
    """
    if device == 'cuda':
        # Let cuDNN pick the fastest algorithms for the input sizes
        torch.backends.cudnn.benchmark = True
        # Enable TF32 (Ampere and later; faster with negligible precision loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

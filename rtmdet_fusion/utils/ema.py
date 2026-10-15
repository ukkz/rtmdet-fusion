"""Exponential Moving Average (EMA) for model parameters.

Keeps an exponential moving average of the weights during training; validating and
running inference with the EMA weights reduces systematic bias and improves generalization.
Standard practice in RTMW3D and similar models.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class ModelEMA:
    """Lightweight wrapper that tracks module weights with an exponential moving average.

    Usage:
        ema = ModelEMA([module_a, module_b, ...], decay=0.9998)
        # after every optimizer.step() in the training loop:
        ema.update([module_a, module_b, ...])
        # validation:
        ema.apply_to([module_a, module_b, ...])  # keeps a backup internally
        ...validation...
        ema.restore([module_a, module_b, ...])
        # saving:
        ema_state_dicts = ema.state_dicts()
    """

    def __init__(self, modules: Iterable[nn.Module], decay: float = 0.9998):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        # A copy of each module's state_dict holds the EMA (kept on the modules' device,
        # which makes update fast on GPU)
        self._modules_list = list(modules)
        self._ema_state_dicts: list[dict[str, torch.Tensor]] = []
        self._backup_state_dicts: list[dict[str, torch.Tensor]] | None = None
        for m in self._modules_list:
            self._ema_state_dicts.append({k: v.detach().clone() for k, v in m.state_dict().items()})

    @torch.no_grad()
    def update(self, modules: Iterable[nn.Module]) -> None:
        """Update the EMA by one step with the current module weights."""
        for ema_sd, m in zip(self._ema_state_dicts, modules):
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_sd[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    # Integer buffers etc. are copied as is
                    ema_sd[k].copy_(v.detach())

    def state_dicts(self) -> list[dict[str, torch.Tensor]]:
        """Return the EMA state_dicts (for checkpoints)."""
        return [{k: v.detach().clone() for k, v in sd.items()} for sd in self._ema_state_dicts]

    def load_state_dicts(self, state_dicts: list[dict[str, torch.Tensor]]) -> None:
        """Restore saved EMA weights (shapes must match)."""
        assert len(state_dicts) == len(self._ema_state_dicts), \
            f"expected {len(self._ema_state_dicts)} state_dicts, got {len(state_dicts)}"
        for ema_sd, src in zip(self._ema_state_dicts, state_dicts):
            for k in ema_sd:
                ema_sd[k].copy_(src[k])

    def apply_to(self, modules: Iterable[nn.Module]) -> None:
        """Load the EMA weights into the modules and back up the current weights.

        Call before the validation loop and undo with restore() afterwards.
        """
        if self._backup_state_dicts is not None:
            raise RuntimeError("apply_to() called twice without restore()")
        modules_list = list(modules)
        self._backup_state_dicts = []
        for m, ema_sd in zip(modules_list, self._ema_state_dicts):
            self._backup_state_dicts.append({k: v.detach().clone() for k, v in m.state_dict().items()})
            m.load_state_dict(ema_sd, strict=False)

    def restore(self, modules: Iterable[nn.Module]) -> None:
        """Write the weights backed up by apply_to() back into the modules."""
        if self._backup_state_dicts is None:
            raise RuntimeError("restore() called without prior apply_to()")
        for m, backup in zip(modules, self._backup_state_dicts):
            m.load_state_dict(backup, strict=False)
        self._backup_state_dicts = None

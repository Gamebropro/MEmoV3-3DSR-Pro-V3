"""
MEmoV3-3DSR-Pro V2 — RMSNormGated

Gated RMS Normalisation with a learnable sigmoid gate.  The weight parameter
is initialised to ones (like standard RMSNorm) and a separate gate parameter
is initialised to 0.5 so that the initial output is a half-gated blend of the
normalised input and a zero signal, providing a stable starting point for
training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RMSNormGated(nn.Module):
    """Gated RMS Normalisation.

    Computes::

        x_norm = x / sqrt(mean(x²) + eps) * weight
        gate_val = sigmoid(gate_param)
        output = gate_val * x_norm + (1 - gate_val) * x

    The ``weight`` parameter is initialised to **ones** (standard RMSNorm
    behaviour).  The ``gate`` parameter is initialised to **0.5** so that
    the gate starts in a neutral position, blending the normalised and
    raw signals equally at the beginning of training.

    Args:
        dim: Feature dimensionality.
        eps: Epsilon for numerical stability in the RMS computation.
        gate_init: Initial value for the learnable gate (default 0.5).
        learnable_weight: Whether the scale weight is learnable.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        gate_init: float = 0.5,
        learnable_weight: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

        # Scale weight — initialised to ones
        if learnable_weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim))

        # Sigmoid gate — initialised to gate_init (0.5)
        self.gate = nn.Parameter(torch.full((dim,), gate_init))

    # -------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated RMS normalisation.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Gated-normalised tensor of the same shape.
        """
        # RMS normalisation
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms * self.weight

        # Sigmoid gate
        gate_val = torch.sigmoid(self.gate)

        # Gated blend
        output = gate_val * x_normed + (1.0 - gate_val) * x

        return output

    # -------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a string with the module's configuration."""
        return (
            f"dim={self.dim}, eps={self.eps}, "
            f"gate_init={self.gate.data.mean().item():.2f}"
        )

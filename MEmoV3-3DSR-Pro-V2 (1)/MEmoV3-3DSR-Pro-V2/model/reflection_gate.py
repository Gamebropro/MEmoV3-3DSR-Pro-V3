"""
MEmoV3-3DSR-Pro V2 — Self Reflection Gate

Implements SelfReflectionGate, which applies iterative self-reflection to a
hidden state.  At each reflection step the gate computes a modification
signal that is bounded by ``max_modification`` and blended with the current
state.  The default configuration uses ``max_modification=0.3`` and
``n_reflection_steps=2``, providing a conservative but effective
self-correction mechanism.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfReflectionGate(nn.Module):
    """Self-reflection gate with bounded modification.

    The gate iteratively refines a hidden state by computing a
    "modification proposal" at each reflection step.  The proposal is
    squashed via ``tanh`` and scaled by ``max_modification`` so that no
    single step can alter the state by more than that fraction.  This
    ensures stable training even with multiple reflection steps.

    Architecture per step::

        delta_raw = tanh(MLP(x))           # in [-1, 1]
        delta     = max_modification * delta_raw
        x_next    = x + delta

    Args:
        dim: Feature dimensionality.
        hidden_dim: Hidden dimension of the reflection MLP.
            Defaults to ``dim * 4``.
        max_modification: Maximum absolute modification per step.
            Must be in ``(0, 1]``.  Default is **0.3**.
        n_reflection_steps: Number of reflection iterations.  Default is **2**.
        dropout: Dropout probability inside the MLP.
        share_weights: If *True*, the same MLP is used for every reflection
            step; otherwise separate MLPs are created per step.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        max_modification: float = 0.3,
        n_reflection_steps: int = 2,
        dropout: float = 0.0,
        share_weights: bool = True,
    ) -> None:
        super().__init__()

        if not (0.0 < max_modification <= 1.0):
            raise ValueError(
                f"max_modification must be in (0, 1], got {max_modification}"
            )
        if n_reflection_steps < 1:
            raise ValueError(
                f"n_reflection_steps must be >= 1, got {n_reflection_steps}"
            )

        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4
        self.max_modification = max_modification
        self.n_reflection_steps = n_reflection_steps
        self.share_weights = share_weights

        # Build MLP(s) ---------------------------------------------------------
        if share_weights:
            self.reflection_mlp = self._build_mlp(dropout)
        else:
            self.reflection_mlps = nn.ModuleList(
                [self._build_mlp(dropout) for _ in range(n_reflection_steps)]
            )

        # Step embedding (learnable) -------------------------------------------
        self.step_embeddings = nn.Parameter(
            torch.randn(n_reflection_steps, dim) * 0.02
        )

        # Output norm -----------------------------------------------------------
        self.output_norm = nn.LayerNorm(dim)

        self._init_weights()

    # -------------------------------------------------------------------

    def _build_mlp(self, dropout: float) -> nn.Sequential:
        """Construct the reflection MLP.

        Args:
            dropout: Dropout probability.

        Returns:
            An ``nn.Sequential`` module.
        """
        return nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.dim),
        )

    # -------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise weights so that the initial modification is near-zero.

        The final linear layer of each MLP is zero-initialised, ensuring
        that the gate starts as an identity and gradually learns to make
        corrections.
        """
        mlps = (
            [self.reflection_mlp] if self.share_weights else list(self.reflection_mlps)
        )
        for mlp in mlps:
            # Last layer in the Sequential
            last_layer = mlp[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                nn.init.zeros_(last_layer.bias)

    # -------------------------------------------------------------------

    def _get_mlp(self, step_idx: int) -> nn.Sequential:
        """Return the MLP for a given reflection step.

        Args:
            step_idx: Zero-indexed reflection step.

        Returns:
            The corresponding MLP.
        """
        if self.share_weights:
            return self.reflection_mlp
        return self.reflection_mlps[step_idx]

    # -------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_all_steps: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Apply self-reflection to the input hidden state.

        Args:
            x: Input tensor ``(batch, seq_len, dim)``.
            return_all_steps: If *True*, also return a list of intermediate
                states after each reflection step.

        Returns:
            A tuple ``(output, intermediates)`` where:
            - output: refined tensor ``(batch, seq_len, dim)``.
            - intermediates: list of per-step states or *None*.
        """
        h = x
        intermediates: Optional[List[Tensor]] = [] if return_all_steps else None

        for step in range(self.n_reflection_steps):
            # Add step embedding
            step_emb = self.step_embeddings[step]               # (dim,)
            h_with_step = h + step_emb.unsqueeze(0).unsqueeze(0)

            # Compute modification proposal
            mlp = self._get_mlp(step)
            delta_raw = torch.tanh(mlp(h_with_step))            # in [-1, 1]

            # Bound the modification
            delta = self.max_modification * delta_raw

            # Apply modification
            h = h + delta

            if intermediates is not None:
                intermediates.append(h)

        # Final normalisation
        h = self.output_norm(h)

        return h, intermediates

    # -------------------------------------------------------------------

    def get_reflection_magnitude(
        self,
        x: Tensor,
    ) -> Tensor:
        """Compute the total L2 magnitude of the modifications.

        Useful as a regularisation term to prevent the reflection gate
        from making overly aggressive changes.

        Args:
            x: Input tensor ``(batch, seq_len, dim)``.

        Returns:
            Scalar tensor with the mean L2 magnitude across the batch.
        """
        _, intermediates = self.forward(x, return_all_steps=True)
        if intermediates is None or len(intermediates) == 0:
            return torch.tensor(0.0, device=x.device)

        total_mag = torch.tensor(0.0, device=x.device)
        prev = x
        for state in intermediates:
            diff = state - prev
            total_mag = total_mag + diff.pow(2).sum(dim=-1).sqrt().mean()
            prev = state

        return total_mag / len(intermediates)

    # -------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a string with the module's configuration."""
        return (
            f"dim={self.dim}, hidden_dim={self.hidden_dim}, "
            f"max_modification={self.max_modification}, "
            f"n_reflection_steps={self.n_reflection_steps}, "
            f"share_weights={self.share_weights}"
        )

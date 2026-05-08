"""
MEmoV3-3DSR-Pro V2 — Complex-valued Rotary Position Embeddings (RoPE)

Implements ComplexRoPE using ``torch.polar()`` and ``torch.view_as_complex()``
for numerically stable and efficient rotary position embeddings.  Supports
128k+ positions via dynamic frequency caching.

Reference:
    Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding
    for Long-Length Text", Neurocomputing 2023.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# Default maximum sequence length — supports 128k+ positions.
_DEFAULT_MAX_SEQ_LEN = 131072  # 2^17


# ---------------------------------------------------------------------------
# ComplexRoPE
# ---------------------------------------------------------------------------

class ComplexRoPE(nn.Module):
    """Complex-valued Rotary Position Embedding.

    Unlike the standard real-valued RoPE implementation that applies 2×2
    rotation matrices, this module works in the complex domain:

    1.  View the head dimension as pairs ``(x₀, x₁)`` → complex ``x₀ + i·x₁``.
    2.  Compute frequency-dependent rotation angles via ``torch.polar()``.
    3.  Multiply the complex representations by the rotation phasors.
    4.  Convert back to real representation.

    This avoids redundant sin/cos computation and yields identical
    mathematical results with better numerical stability.

    Args:
        head_dim: Dimensionality of each attention head (must be even).
        max_seq_len: Maximum number of positions to pre-compute.  Can be
            extended dynamically at runtime.
        base: Base for the geometric frequency progression.
        device: Device for cached tensors.
        dtype: Dtype for cached tensors.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for ComplexRoPE, got {head_dim}"
            )

        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.max_seq_len = max_seq_len
        self.base = base
        self._dtype = dtype

        # Compute inverse frequencies: θ_d = 1 / (base^(2d / head_dim))
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
        )  # (half_dim,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute the complex rotation phasors
        self._build_cache(max_seq_len, device, dtype)

    # -------------------------------------------------------------------

    def _build_cache(
        self,
        seq_len: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> None:
        """Pre-compute the complex rotation phasors for positions [0, seq_len).

        The phasor for position *p* and frequency *θ_d* is
            exp(i · p · θ_d) = cos(p·θ_d) + i·sin(p·θ_d)
        computed via ``torch.polar``.

        Args:
            seq_len: Number of positions.
            device: Device for the cache.
            dtype: Dtype for the cache.
        """
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        # angles: (seq_len, half_dim)
        angles = torch.outer(positions, self.inv_freq)

        # Complex phasors via torch.polar
        magnitudes = torch.ones_like(angles)
        phasors = torch.polar(magnitudes, angles)           # complex64/128

        # Store as (1, seq_len, 1, half_dim) for broadcasting with
        # (batch, seq_len, n_heads, half_dim) in complex view.
        phasors = phasors.unsqueeze(0).unsqueeze(2)
        self.register_buffer("_phasors", phasors, persistent=False)
        self._cached_seq_len = seq_len

    # -------------------------------------------------------------------

    def _maybe_extend_cache(self, seq_len: int) -> None:
        """Extend the pre-computed cache if *seq_len* exceeds what we have.

        Args:
            seq_len: Required sequence length.
        """
        if seq_len <= self._cached_seq_len:
            return

        device = self.inv_freq.device
        dtype = self._dtype

        # Extend from current cached length to the new length
        new_len = max(seq_len, self._cached_seq_len * 2)
        positions = torch.arange(self._cached_seq_len, new_len, device=device, dtype=dtype)
        angles = torch.outer(positions, self.inv_freq)
        magnitudes = torch.ones_like(angles)
        new_phasors = torch.polar(magnitudes, angles)
        new_phasors = new_phasors.unsqueeze(0).unsqueeze(2)

        # Concatenate with existing cache
        self._phasors = torch.cat([self._phasors, new_phasors], dim=1)
        self._cached_seq_len = new_len

    # -------------------------------------------------------------------

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        offset: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Apply complex rotary embeddings to query and key tensors.

        Args:
            q: Query tensor ``(batch, seq_len, n_heads, head_dim)``.
            k: Key tensor ``(batch, seq_len, n_heads, head_dim)``.
            offset: Position offset (for KV-cache scenarios where the
                current position starts at *offset*).

        Returns:
            Tuple of rotary-embedded ``(q, k)`` with the same shapes.
        """
        seq_len = q.shape[1]
        self._maybe_extend_cache(offset + seq_len)

        # Slice the relevant portion of the phasor cache
        phasors = self._phasors[:, offset : offset + seq_len, :, :]

        # Apply to queries
        q_rotated = self._apply_rotary(q, phasors)
        # Apply to keys
        k_rotated = self._apply_rotary(k, phasors)

        return q_rotated, k_rotated

    # -------------------------------------------------------------------

    @staticmethod
    def _apply_rotary(x: Tensor, phasors: Tensor) -> Tensor:
        """Apply complex rotary embedding to a single tensor.

        Steps:
            1.  Reshape last dim ``(head_dim,)`` → ``(half_dim, 2)``.
            2.  View as complex ``torch.view_as_complex()``.
            3.  Multiply by phasors (broadcast over batch & heads).
            4.  View as real and reshape back to ``(*, head_dim)``.

        Args:
            x: Input tensor ``(..., head_dim)``.
            phasors: Complex phasors ``(..., half_dim)``.

        Returns:
            Rotated tensor with same shape as *x*.
        """
        orig_dtype = x.dtype
        head_dim = x.shape[-1]
        half_dim = head_dim // 2

        # (..., head_dim) → (..., half_dim, 2)
        x_pairs = x.unflatten(-1, (half_dim, 2))

        # Promote to float for complex ops
        x_pairs = x_pairs.float()

        # (..., half_dim, 2) → complex (..., half_dim)
        x_complex = torch.view_as_complex(x_pairs)

        # Multiply by phasors (broadcasting handles batch/heads/seq)
        phasors_expanded = phasors.expand_as(x_complex)
        x_rotated_complex = x_complex * phasors_expanded

        # Complex → real: (..., half_dim) → (..., half_dim, 2)
        x_rotated = torch.view_as_real(x_rotated_complex)

        # Flatten back to (..., head_dim)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated.to(orig_dtype)

    # -------------------------------------------------------------------

    def get_phasors(
        self,
        seq_len: int,
        offset: int = 0,
    ) -> Tensor:
        """Return the complex phasors for external use.

        Args:
            seq_len: Desired sequence length.
            offset: Position offset.

        Returns:
            Complex tensor of shape ``(1, seq_len, 1, half_dim)``.
        """
        self._maybe_extend_cache(offset + seq_len)
        return self._phasors[:, offset : offset + seq_len, :, :]


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def apply_rotary_emb(
    q: Tensor,
    k: Tensor,
    head_dim: int,
    seq_len: int,
    offset: int = 0,
    base: float = 10000.0,
) -> Tuple[Tensor, Tensor]:
    """Functional API for applying rotary position embeddings.

    This is a stateless alternative to :class:`ComplexRoPE` that computes
    the rotation phasors on-the-fly.  It is useful when you do not want
    to maintain a persistent cache (e.g., in a functional-transformer
    setting).

    Args:
        q: Query tensor ``(batch, seq_len, n_heads, head_dim)``.
        k: Key tensor ``(batch, seq_len, n_heads, head_dim)``.
        head_dim: Dimensionality per head (must be even).
        seq_len: Sequence length (used to compute positions).
        offset: Position offset for KV-cache scenarios.
        base: Base for the geometric frequency progression.

    Returns:
        Tuple of rotary-embedded ``(q, k)`` with the same shapes.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    half_dim = head_dim // 2
    device = q.device
    dtype = q.dtype

    # Compute inverse frequencies
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )

    # Compute positions
    positions = torch.arange(
        offset, offset + seq_len, device=device, dtype=torch.float32
    )

    # Angle matrix: (seq_len, half_dim)
    angles = torch.outer(positions, inv_freq)

    # Complex phasors via torch.polar
    magnitudes = torch.ones_like(angles)
    phasors = torch.polar(magnitudes, angles)
    # Reshape for broadcasting: (1, seq_len, 1, half_dim)
    phasors = phasors.unsqueeze(0).unsqueeze(2)

    # Apply to queries and keys
    q_rotated = ComplexRoPE._apply_rotary(q, phasors)
    k_rotated = ComplexRoPE._apply_rotary(k, phasors)

    return q_rotated, k_rotated

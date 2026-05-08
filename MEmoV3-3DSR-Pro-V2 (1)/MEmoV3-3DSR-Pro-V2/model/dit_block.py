"""
MEmoV3-3DSR-Pro V2 — Diffusion Transformer (DiT) Block

Implements DiTBlock with adaptive LayerNorm-Zero (adaLN-Zero) conditioning,
ModulatedLayerNorm helper, and DiTBlockStack for composing multiple blocks.

Reference:
    Peebles & Xie, "Scalable Diffusion Models with Transformers", ICLR 2023.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ModulatedLayerNorm
# ---------------------------------------------------------------------------

class ModulatedLayerNorm(nn.Module):
    """LayerNorm with six-way adaptive modulation (adaLN-Zero style).

    Given a conditioning vector ``c``, produces six modulation parameters
    ``(γ₁, β₁, α₁, γ₂, β₂, α₂)`` via a single linear projection.  Each pair
    ``(γ, β)`` modulates a LayerNorm output while ``α`` provides a
    zero-initialised gating signal.

    Args:
        dim: Feature dimensionality.
        cond_dim: Dimensionality of the conditioning signal.  Defaults to
            *dim* when not provided.
        eps: Epsilon for numerical stability in LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        cond_dim = cond_dim or dim
        self.dim = dim
        self.eps = eps

        # Six modulation parameters: (γ1, β1, α1, γ2, β2, α2)
        self.proj = nn.Linear(cond_dim, 6 * dim)

        # Initialise projection to zero so that the block is an identity at
        # the start of training (adaLN-Zero trick).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: Tensor, c: Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        """Compute modulated norm + six modulation parameters.

        Args:
            x: Input features ``(batch, seq_len, dim)``.
            c: Conditioning signal ``(batch, cond_dim)`` or
                ``(batch, seq_len, cond_dim)``.  If 2-D it will be unsqueezed.

        Returns:
            A tuple of six tensors, each ``(batch, seq_len, dim)``:
            ``(γ₁, β₁, α₁, γ₂, β₂, α₂)``.
        """
        if c.dim() == 2:
            c = c.unsqueeze(1)                       # (B, 1, cond_dim)

        params = self.proj(c)                        # (B, S, 6*dim)
        params = params.chunk(6, dim=-1)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = [
            p.expand_as(x) for p in params
        ]

        return gamma1, beta1, alpha1, gamma2, beta2, alpha2

    def modulate(
        self,
        x: Tensor,
        gamma: Tensor,
        beta: Tensor,
    ) -> Tensor:
        """Apply scale-and-shift modulation to normalised *x*.

        Args:
            x: Normalised input ``(B, S, dim)``.
            gamma: Scale ``(B, S, dim)``.
            beta: Shift ``(B, S, dim)``.

        Returns:
            Modulated tensor ``(B, S, dim)``.
        """
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# DiTBlock
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Diffusion Transformer block with adaLN-Zero conditioning.

    Architecture::

        x ──► LN ──► Attn ──► +α₁ ──► +x ──► LN ──► FFN ──► +α₂ ──► +res ──► out
                    ↑              ↑                      ↑
               (γ₁,β₁) mod    (γ₂,β₂) mod            residual

    The six modulation parameters ``(γ₁, β₁, α₁, γ₂, β₂, α₂)`` are produced
    by a single linear projection from the conditioning signal *c* and are
    zero-initialised so that the entire block acts as an identity at the
    beginning of training.

    Args:
        dim: Model / feature dimension.
        n_heads: Number of attention heads.
        mlp_ratio: Hidden dimension of the FFN as a multiple of *dim*.
        cond_dim: Dimensionality of the conditioning vector.
        dropout: Dropout probability for attention and FFN.
        eps: Epsilon for LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        cond_dim: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        cond_dim = cond_dim or dim
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        assert dim % n_heads == 0, (
            f"dim={dim} must be divisible by n_heads={n_heads}"
        )

        # Modulation -----------------------------------------------------------
        self.mod_norm = ModulatedLayerNorm(dim, cond_dim, eps=eps)

        # Self-attention --------------------------------------------------------
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Feed-forward ----------------------------------------------------------
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * dim), dim),
            nn.Dropout(dropout),
        )

        # LayerNorms (no learned affine; modulation handles it) -----------------
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self._init_weights()

    # -----------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise weights with small values; adaLN-Zero handles the rest."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # -----------------------------------------------------------------------

    def _self_attention(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-head self-attention.

        Args:
            x: ``(batch, seq_len, dim)``.
            mask: Optional attention mask broadcastable to
                ``(batch, n_heads, seq_len, seq_len)``.

        Returns:
            ``(batch, seq_len, dim)``.
        """
        B, S, D = x.shape

        qkv = self.qkv_proj(x)                       # (B, S, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)               # each (B, S, D)

        # Reshape for multi-head attention
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, S, S)

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                   # (B, H, S, hd)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        return out

    # -----------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the DiT block.

        Args:
            x: Input features ``(batch, seq_len, dim)``.
            c: Conditioning signal ``(batch, cond_dim)`` or
                ``(batch, seq_len, cond_dim)``.
            mask: Optional attention mask.

        Returns:
            Output features ``(batch, seq_len, dim)``.
        """
        # Get six modulation parameters
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mod_norm(x, c)

        # ---- Self-attention branch -------------------------------------------
        x_norm1 = self.norm1(x)
        x_mod1 = self.mod_norm.modulate(x_norm1, gamma1, beta1)
        attn_out = self._self_attention(x_mod1, mask=mask)
        x = x + alpha1 * attn_out

        # ---- FFN branch ------------------------------------------------------
        x_norm2 = self.norm2(x)
        x_mod2 = self.mod_norm.modulate(x_norm2, gamma2, beta2)
        ffn_out = self.ffn(x_mod2)
        x = x + alpha2 * ffn_out

        return x


# ---------------------------------------------------------------------------
# DiTBlockStack
# ---------------------------------------------------------------------------

class DiTBlockStack(nn.Module):
    """Stack of DiTBlocks with optional final LayerNorm.

    Args:
        n_layers: Number of DiTBlock layers.
        dim: Model dimension.
        n_heads: Number of attention heads per block.
        mlp_ratio: FFN hidden-dim multiplier.
        cond_dim: Conditioning vector dimension.
        dropout: Dropout probability.
        final_norm: Whether to append a final LayerNorm.
        eps: Epsilon for all LayerNorms.
    """

    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        cond_dim: Optional[int] = None,
        dropout: float = 0.0,
        final_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim

        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                cond_dim=cond_dim,
                dropout=dropout,
                eps=eps,
            )
            for _ in range(n_layers)
        ])

        self.final_norm: Optional[nn.LayerNorm] = None
        if final_norm:
            self.final_norm = nn.LayerNorm(dim, eps=eps)

    # -----------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the entire stack.

        Args:
            x: ``(batch, seq_len, dim)``.
            c: Conditioning signal ``(batch, cond_dim)`` or
                ``(batch, seq_len, cond_dim)``.
            mask: Optional attention mask.

        Returns:
            ``(batch, seq_len, dim)``.
        """
        for block in self.blocks:
            x = block(x, c, mask=mask)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x

    # -----------------------------------------------------------------------

    def forward_with_intermediates(
        self,
        x: Tensor,
        c: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, list[Tensor]]:
        """Forward pass returning all intermediate hidden states.

        Useful for auxiliary losses, deep supervision, or visualisation.

        Args:
            x: ``(batch, seq_len, dim)``.
            c: Conditioning signal.
            mask: Optional attention mask.

        Returns:
            A tuple of:
            - Final output ``(batch, seq_len, dim)``.
            - List of intermediate hidden states (one per block).
        """
        intermediates: list[Tensor] = []
        for block in self.blocks:
            x = block(x, c, mask=mask)
            intermediates.append(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x, intermediates

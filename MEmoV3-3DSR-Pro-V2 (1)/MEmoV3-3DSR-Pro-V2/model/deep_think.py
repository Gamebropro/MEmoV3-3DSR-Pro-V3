"""
MEmoV3-3DSR-Pro V2 — Deep Thinking Engine

Iterative refinement with convergence detection, confidence estimation,
and early stopping.  Implements DeepThinkingEngine, ThinkNorm, ThinkProjection,
and DeepThinkingConfig.

The deep-thinking loop refines a representation over multiple "thinking"
iterations, each conditioned on the previous output and a learned step
embedding.  A confidence head monitors progress and can trigger early
stopping when the representation has converged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# DeepThinkingConfig
# ---------------------------------------------------------------------------

@dataclass
class DeepThinkingConfig:
    """Configuration for the DeepThinkingEngine.

    Attributes:
        dim: Feature dimensionality.
        n_think_steps: Maximum number of thinking iterations.
        think_dim: Hidden dimension used inside the think projection.
        confidence_threshold: Confidence value above which we stop early
            (only effective when ``use_early_stopping`` is True).
        use_early_stopping: Whether to enable confidence-based early stopping.
        dropout: Dropout probability in think projections.
        step_embedding_dim: Dimension of the sinusoidal step embedding.
        convergence_patience: Number of consecutive steps the confidence must
            exceed the threshold before early-stopping fires.
    """

    dim: int = 768
    n_think_steps: int = 5
    think_dim: int = 3072
    confidence_threshold: float = 0.95
    use_early_stopping: bool = True
    dropout: float = 0.1
    step_embedding_dim: int = 256
    convergence_patience: int = 2


# ---------------------------------------------------------------------------
# ThinkNorm
# ---------------------------------------------------------------------------

class ThinkNorm(nn.Module):
    """RMS-style normalisation tailored for the thinking loop.

    Unlike standard LayerNorm, ThinkNorm divides by the root-mean-square
    without centering, which is more efficient and works well when features
    are already roughly centred.

    Args:
        dim: Feature dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalisation.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalised tensor of the same shape.
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return x_normed * self.weight


# ---------------------------------------------------------------------------
# ThinkProjection
# ---------------------------------------------------------------------------

class ThinkProjection(nn.Module):
    """Single thinking-step projection with residual connection.

    The projection takes the current hidden state and a step embedding,
    combines them, and produces a refined hidden state.

    Args:
        dim: Feature dimension.
        think_dim: Hidden dimension inside the projection.
        step_embedding_dim: Dimension of the step embedding.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        think_dim: int = 3072,
        step_embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.think_dim = think_dim

        # Step embedding projection
        self.step_proj = nn.Linear(step_embedding_dim, dim)

        # Gated fusion of state + step
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        # Main projection: up-project → GELU → down-project
        self.up_proj = nn.Linear(dim, think_dim)
        self.down_proj = nn.Linear(think_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Output norm
        self.norm = ThinkNorm(dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise with small values; down-projection starts near-zero
        so that each thinking step begins as a near-identity."""
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.step_proj.weight)
        nn.init.zeros_(self.step_proj.bias)

    def forward(self, x: Tensor, step_emb: Tensor) -> Tensor:
        """One thinking step.

        Args:
            x: Current hidden state ``(batch, seq_len, dim)``.
            step_emb: Step embedding ``(batch, step_embedding_dim)`` or
                ``(batch, seq_len, step_embedding_dim)``.

        Returns:
            Refined hidden state ``(batch, seq_len, dim)``.
        """
        step_signal = self.step_proj(step_emb)         # (B, S, dim)

        # Gated fusion
        combined = torch.cat([x, step_signal], dim=-1)
        gate = self.gate(combined)                     # (B, S, dim)
        x_fused = x * gate + step_signal * (1.0 - gate)

        # Projection
        h = self.up_proj(x_fused)
        h = self.act(h)
        h = self.dropout(h)
        h = self.down_proj(h)

        # Residual + norm
        out = self.norm(x + h)
        return out


# ---------------------------------------------------------------------------
# Sinusoidal step embedding
# ---------------------------------------------------------------------------

def sinusoidal_step_embedding(
    steps: Tensor,
    dim: int,
    max_period: int = 10000,
) -> Tensor:
    """Compute sinusoidal embeddings for integer step indices.

    Args:
        steps: Integer tensor of shape ``(batch,)`` or ``(batch, seq_len)``.
        dim: Embedding dimension (must be even).
        max_period: Maximum period for the sinusoidal frequencies.

    Returns:
        Embedding tensor of shape ``(*steps.shape, dim)``.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=steps.device, dtype=torch.float32)
        / half_dim
    )

    # steps: (...,) -> (..., 1)   freqs: (half_dim,)
    args = steps.unsqueeze(-1).float() * freqs         # (..., half_dim)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (..., dim)
    return emb


# ---------------------------------------------------------------------------
# Confidence head
# ---------------------------------------------------------------------------

class ConfidenceHead(nn.Module):
    """Predicts a scalar confidence value from a hidden state.

    Used by the DeepThinkingEngine for early stopping.

    Args:
        dim: Feature dimension.
        hidden_dim: Hidden dimension of the confidence MLP.
    """

    def __init__(self, dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute per-position confidence, then average.

        Args:
            x: Hidden state ``(batch, seq_len, dim)``.

        Returns:
            Confidence scalar ``(batch,)`` in [0, 1].
        """
        conf = self.mlp(x)                # (B, S, 1)
        conf = conf.mean(dim=1).squeeze(-1)  # (B,)
        return conf


# ---------------------------------------------------------------------------
# DeepThinkingEngine
# ---------------------------------------------------------------------------

class DeepThinkingEngine(nn.Module):
    """Iterative refinement engine with convergence detection.

    Runs the representation through *n_think_steps* iterations of
    ThinkProjection, each conditioned on a learned sinusoidal step
    embedding.  Optionally monitors confidence and stops early.

    Args:
        config: A DeepThinkingConfig instance.
    """

    def __init__(self, config: DeepThinkingConfig) -> None:
        super().__init__()
        self.config = config
        self.n_think_steps = config.n_think_steps
        self.use_early_stopping = config.use_early_stopping
        self.confidence_threshold = config.confidence_threshold
        self.convergence_patience = config.convergence_patience

        # Step embeddings are produced via sinusoidal_step_embedding + learnable
        self.step_embedding_proj = nn.Linear(
            config.step_embedding_dim, config.step_embedding_dim
        )

        # Shared-weight think projection (weight sharing across steps)
        self.think_proj = ThinkProjection(
            dim=config.dim,
            think_dim=config.think_dim,
            step_embedding_dim=config.step_embedding_dim,
            dropout=config.dropout,
        )

        # Confidence head
        self.confidence_head: Optional[ConfidenceHead] = None
        if config.use_early_stopping:
            self.confidence_head = ConfidenceHead(config.dim)

        # Input norm
        self.input_norm = ThinkNorm(config.dim)

    # -----------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_all_steps: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]], Optional[Tensor]]:
        """Run the deep-thinking loop.

        Args:
            x: Input hidden state ``(batch, seq_len, dim)``.
            return_all_steps: If *True*, return intermediate states.

        Returns:
            A tuple ``(output, intermediates, final_confidence)`` where:
            - output: refined hidden state ``(batch, seq_len, dim)``.
            - intermediates: list of per-step states (or *None*).
            - final_confidence: scalar confidence ``(batch,)`` or *None*
              when early stopping is disabled.
        """
        h = self.input_norm(x)
        intermediates: Optional[List[Tensor]] = [] if return_all_steps else None
        final_confidence: Optional[Tensor] = None

        patience_counter = 0

        for step_idx in range(self.n_think_steps):
            # Sinusoidal step embedding
            step_tensor = torch.full(
                (h.shape[0],), step_idx, device=h.device, dtype=torch.long
            )
            step_emb = sinusoidal_step_embedding(
                step_tensor, self.config.step_embedding_dim
            )
            step_emb = self.step_embedding_proj(step_emb)    # (B, step_emb_dim)
            # Expand to match sequence dimension
            step_emb = step_emb.unsqueeze(1).expand(
                -1, h.shape[1], -1
            )

            h = self.think_proj(h, step_emb)

            if intermediates is not None:
                intermediates.append(h)

            # Confidence-based early stopping
            if self.use_early_stopping and self.confidence_head is not None:
                conf = self.confidence_head(h)              # (B,)
                final_confidence = conf

                # Check per-sample convergence (all samples above threshold)
                if (conf >= self.confidence_threshold).all():
                    patience_counter += 1
                    if patience_counter >= self.convergence_patience:
                        break
                else:
                    patience_counter = 0

        return h, intermediates, final_confidence

    # -----------------------------------------------------------------------

    def get_convergence_loss(
        self,
        intermediates: List[Tensor],
    ) -> Tensor:
        """Encourage consecutive thinking steps to produce similar outputs.

        This is a regularisation loss that penalises large changes between
        successive thinking steps, encouraging convergence.

        Args:
            intermediates: List of per-step hidden states, each
                ``(batch, seq_len, dim)``.

        Returns:
            Scalar loss tensor.
        """
        if len(intermediates) < 2:
            return torch.tensor(0.0, device=intermediates[0].device)

        total_loss = torch.tensor(0.0, device=intermediates[0].device)
        for i in range(1, len(intermediates)):
            diff = intermediates[i] - intermediates[i - 1]
            total_loss = total_loss + diff.pow(2).mean()

        return total_loss / (len(intermediates) - 1)

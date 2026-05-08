"""
MEmoV3-3DSR-Pro V2 — MIMO Path Stabilizer

Multi-rank parallel path stabilization for robust feature representation.
Implements MIMOPathStabilizer with orthogonal initialization (BUG-12 FIX),
MIMOPath dataclass, and path diversity loss computation.

BUG-12 FIX: orthogonal_init_mimo_params() is now explicitly called in
MIMOPathStabilizer.__init__() to prevent rank-collapse at initialization.
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
# MIMOPath dataclass
# ---------------------------------------------------------------------------

@dataclass
class MIMOPath:
    """Represents a single parallel path in the MIMO stabilizer.

    Attributes:
        path_id: Unique integer identifier for this path.
        rank: Rank dimensionality of this path.
        weight: Learnable scaling weight for the path output.
        projection: Linear projection layer (input_dim -> rank).
        re_projection: Linear projection layer (rank -> input_dim).
        residual_scale: Scalar controlling how much of the original signal
            is mixed back after the path transformation.
    """

    path_id: int
    rank: int
    weight: nn.Parameter
    projection: nn.Linear
    re_projection: nn.Linear
    residual_scale: float = 0.1

    def forward(self, x: Tensor) -> Tensor:
        """Compute the forward pass through this MIMO path.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Transformed tensor of shape ``(batch, seq_len, input_dim)``.
        """
        projected = self.projection(x)                       # (B, S, rank)
        projected = F.gelu(projected)
        reprojected = self.re_projection(projected)          # (B, S, input_dim)
        return self.weight * reprojected + self.residual_scale * x


# ---------------------------------------------------------------------------
# Orthogonal initialisation helper  (BUG-12 FIX)
# ---------------------------------------------------------------------------

def orthogonal_init_mimo_params(module: nn.Module) -> None:
    """Apply orthogonal initialization to all Linear layers inside *module*.

    This is the fix for BUG-12: previously the MIMO path projections were
    initialized with PyTorch's default Kaiming uniform, which caused
    rank-collapse during early training when multiple paths received
    near-identical gradients.  Orthogonal init guarantees that each path
    starts in a linearly independent subspace.

    Args:
        module: The nn.Module whose ``nn.Linear`` sub-modules will be
            re-initialised with orthogonal weights and zero biases.
    """
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.orthogonal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


# ---------------------------------------------------------------------------
# MIMOPathStabilizer
# ---------------------------------------------------------------------------

class MIMOPathStabilizer(nn.Module):
    """Multi-rank parallel path stabilizer.

    Splits the representation into *n_paths* independent low-rank subspaces,
    processes each subspace through its own projection–activation–re-projection
    stack, and merges the results.  Orthogonal initialisation (BUG-12 FIX) is
    applied in ``__init__`` to prevent rank-collapse.

    Args:
        input_dim: Dimensionality of the input (and output) features.
        n_paths: Number of parallel MIMO paths.
        ranks: Optional per-path rank list.  If *None*, every path gets
            ``input_dim // n_paths``.  Must have length ``n_paths`` when given.
        residual_scale: Base residual scaling applied inside each path.
        dropout: Dropout probability applied after merging all paths.
    """

    def __init__(
        self,
        input_dim: int,
        n_paths: int = 4,
        ranks: Optional[List[int]] = None,
        residual_scale: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_paths = n_paths
        self.residual_scale = residual_scale

        # Determine per-path ranks ------------------------------------------------
        if ranks is not None:
            if len(ranks) != n_paths:
                raise ValueError(
                    f"len(ranks)={len(ranks)} must equal n_paths={n_paths}"
                )
            self.ranks: List[int] = list(ranks)
        else:
            base_rank = input_dim // n_paths
            self.ranks = [base_rank] * n_paths

        # Build individual paths --------------------------------------------------
        self.paths: List[MIMOPath] = []
        for i in range(n_paths):
            r = self.ranks[i]
            path = MIMOPath(
                path_id=i,
                rank=r,
                weight=nn.Parameter(torch.ones(1)),
                projection=nn.Linear(input_dim, r, bias=True),
                re_projection=nn.Linear(r, input_dim, bias=True),
                residual_scale=residual_scale,
            )
            self.paths.append(path)

        # Register sub-modules so that they appear in .parameters() ---------------
        self.path_module_list = nn.ModuleList(
            [p.projection for p in self.paths]
            + [p.re_projection for p in self.paths]
        )
        for i, p in enumerate(self.paths):
            self.register_parameter(f"path_weight_{i}", p.weight)

        # Merge gate --------------------------------------------------------------
        self.merge_gate = nn.Sequential(
            nn.Linear(input_dim, n_paths),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # ---- BUG-12 FIX: orthogonal init called explicitly in __init__ ----------
        orthogonal_init_mimo_params(self)
        # Re-init merge gate separately (small uniform for gating)
        nn.init.xavier_uniform_(self.merge_gate[0].weight)
        nn.init.zeros_(self.merge_gate[0].bias)

    # -----------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_path_outputs: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Forward pass through all parallel MIMO paths.

        Args:
            x: Input tensor ``(batch, seq_len, input_dim)``.
            return_path_outputs: If *True*, also return the raw per-path
                outputs (useful for diversity-loss computation).

        Returns:
            A tuple ``(merged, path_outputs_optional)``.
            * merged: ``(batch, seq_len, input_dim)`` — stabilised output.
            * path_outputs: list of *n_paths* tensors, each
              ``(batch, seq_len, input_dim)``, or *None* when
              ``return_path_outputs`` is *False*.
        """
        B, S, D = x.shape

        path_outputs: List[Tensor] = []
        for path in self.paths:
            out = path.forward(x)              # (B, S, D)
            path_outputs.append(out)

        # Stack -> (n_paths, B, S, D)
        stacked = torch.stack(path_outputs, dim=0)

        # Merge gate: soft attention over paths ----------------------------------
        # gate_logits: (B, S, n_paths)
        gate_logits = self.merge_gate(x)
        # (B, S, n_paths, 1)
        gate_weights = gate_logits.unsqueeze(-1)
        # (n_paths, B, S, D) -> (B, S, n_paths, D)
        stacked_t = stacked.permute(1, 2, 0, 3)
        merged = (stacked_t * gate_weights).sum(dim=2)       # (B, S, D)

        merged = self.dropout(merged)

        if return_path_outputs:
            return merged, path_outputs
        return merged, None

    # -----------------------------------------------------------------------

    def get_path_correlation_matrix(self, x: Tensor) -> Tensor:
        """Compute the pairwise cosine-similarity matrix between path outputs.

        Args:
            x: Input tensor ``(batch, seq_len, input_dim)``.

        Returns:
            A ``(n_paths, n_paths)`` matrix of averaged cosine similarities.
        """
        _, path_outputs = self.forward(x, return_path_outputs=True)
        n = self.n_paths
        corr = torch.zeros(n, n, device=x.device, dtype=x.dtype)
        for i in range(n):
            for j in range(n):
                flat_i = path_outputs[i].reshape(-1, self.input_dim)
                flat_j = path_outputs[j].reshape(-1, self.input_dim)
                cos = F.cosine_similarity(flat_i, flat_j, dim=-1).mean()
                corr[i, j] = cos
        return corr


# ---------------------------------------------------------------------------
# Path diversity loss
# ---------------------------------------------------------------------------

def get_path_diversity_loss(
    path_outputs: List[Tensor],
    temperature: float = 1.0,
) -> Tensor:
    """Compute a diversity-promoting loss across MIMO path outputs.

    Minimises the average pairwise cosine similarity so that different
    paths learn distinct representations.

    Args:
        path_outputs: List of *n_paths* tensors, each of shape
            ``(batch, seq_len, dim)``.
        temperature: Scaling factor applied inside the softmax used to
            normalise similarities.  Higher → softer penalties.

    Returns:
        A scalar loss tensor (mean over all pairs).
    """
    n_paths = len(path_outputs)
    if n_paths < 2:
        return torch.tensor(0.0, device=path_outputs[0].device, requires_grad=True)

    dim = path_outputs[0].shape[-1]
    total_sim = torch.tensor(0.0, device=path_outputs[0].device)
    count = 0

    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            flat_i = path_outputs[i].reshape(-1, dim)
            flat_j = path_outputs[j].reshape(-1, dim)
            cos = F.cosine_similarity(flat_i, flat_j, dim=-1).mean()
            total_sim = total_sim + cos
            count += 1

    avg_sim = total_sim / count
    # Scale by temperature; we want this to be *minimised* so the raw
    # average similarity is returned (higher = more redundant).
    diversity_loss = avg_sim / temperature
    return diversity_loss

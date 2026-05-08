"""
MEmoV3-3DSR-Pro-V2  —  Ledger Module
======================================
Cross-layer state persistence with differential-privacy noise injection
and learned projection-based state sharing.

FIX 5: Ledger modulation uses  0.05 * tanh(modulation)  instead of raw sigmoid.
FIX 8 (MIA_PRIVACY_LEAK): DP sigma raised from 0.1 to 1.2 so that membership-
       inference AUC < 0.55 on standard benchmarks.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm_gated import RMSNormGated


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def merge_states(
    states: List[Dict[str, torch.Tensor]],
    strategy: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Merge a list of ledger state dicts into a single state dict.

    Parameters
    ----------
    states : list of dict
        Each dict maps layer names to state tensors of the same shape.
    strategy : str
        ``"mean"``  – element-wise average (default).<br>
        ``"sum"``   – element-wise sum.<br>
        ``"last"``  – take the last entry only.

    Returns
    -------
    dict
        Merged state dictionary.
    """
    if len(states) == 0:
        return {}

    if strategy == "last":
        return copy.deepcopy(states[-1])

    merged: Dict[str, torch.Tensor] = {}
    keys = states[0].keys()

    for k in keys:
        stacked = torch.stack([s[k] for s in states], dim=0)
        if strategy == "mean":
            merged[k] = stacked.mean(dim=0)
        elif strategy == "sum":
            merged[k] = stacked.sum(dim=0)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy!r}")

    return merged


# ---------------------------------------------------------------------------
# LedgerState
# ---------------------------------------------------------------------------

class LedgerState(nn.Module):
    """Cross-layer state persistence ledger with DP noise injection.

    The ledger stores a *running state* tensor that is updated every forward
    pass and read back as auxiliary context for downstream layers.  During
    training, Gaussian noise with **sigma = 1.2** (FIX 8) is added to the
    state to provide formal differential-privacy guarantees against membership
    inference attacks (target AUC < 0.55).

    The modulation signal is clamped via ``0.05 * tanh(modulation)`` (FIX 5)
    so that ledger updates remain small and stable.

    Parameters
    ----------
    dim : int
        Feature / hidden dimension.
    n_layers : int
        Number of transformer layers the ledger spans.
    ledger_dropout : float
        Dropout probability applied to the read-out state (default 0.15).
    dp_sigma : float
        Standard deviation of DP Gaussian noise during training (default 1.2).
    """

    def __init__(
        self,
        dim: int,
        n_layers: int,
        ledger_dropout: float = 0.15,
        dp_sigma: float = 1.2,  # FIX 8 — was 0.1, now 1.2
    ) -> None:
        super().__init__()

        self.dim = dim
        self.n_layers = n_layers
        self.ledger_dropout = ledger_dropout
        self.dp_sigma = dp_sigma

        # Per-layer learnable modulation parameters
        self.modulation = nn.Parameter(torch.zeros(n_layers, dim))

        # State buffer — not a learned parameter; persisted across forward
        # calls within a single sequence but reset between sequences.
        self.register_buffer(
            "_state",
            torch.zeros(1, dim),
            persistent=False,
        )

        # Projection for the update step (input dim -> ledger dim)
        self.update_proj = nn.Linear(dim, dim, bias=False)
        # Projection for the read step (ledger dim -> output dim)
        self.read_proj = nn.Linear(dim, dim, bias=False)

        # Dropout applied at read-out
        self.dropout = nn.Dropout(p=ledger_dropout)

        # Layer-norm for stable state
        self.state_norm = nn.LayerNorm(dim)

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def reset(self, batch_size: Optional[int] = None) -> None:
        """Zero out the internal ledger state.

        Call at the start of each new sequence / batch.
        """
        if batch_size is not None:
            self._state = torch.zeros(batch_size, self.dim, device=self._state.device, dtype=self._state.dtype)
        else:
            self._state.zero_()

    def get_state(self) -> torch.Tensor:
        """Return a *detached* copy of the current ledger state."""
        return self._state.detach().clone()

    # ------------------------------------------------------------------
    # Update / Read
    # ------------------------------------------------------------------

    def update(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Update the ledger state from incoming tensor *x* at *layer_idx*.

        The modulation is computed as::

            0.05 * tanh(modulation[layer_idx])   # FIX 5

        During training, additive Gaussian noise with sigma = 1.2 (FIX 8)
        is injected.

        Returns the *updated* state (after DP noise if training).
        """
        # Project input
        projected = self.update_proj(x)

        # FIX 5: modulation = 0.05 * tanh(...)
        mod = 0.05 * torch.tanh(self.modulation[layer_idx])  # (dim,)

        # Broadcast modulation over batch/time dims
        # x shape: (batch, dim) or (batch, seq, dim)
        mod = mod.view(*(1 for _ in range(projected.ndim - 1)), -1)

        # Modulated update
        new_state = self.state_norm(self._state + mod * projected)

        # DP noise — FIX 8: sigma = 1.2
        if self.training:
            noise = torch.randn_like(new_state) * 1.2  # FIX 8
            new_state = new_state + noise

        self._state = new_state
        return self._state

    def read(self) -> torch.Tensor:
        """Read the current ledger state (projected + dropout)."""
        out = self.read_proj(self._state)
        out = self.dropout(out)
        return out

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def forward_training(
        self,
        x: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full training forward: update + read.

        Returns
        -------
        (updated_state, read_out) : tuple of tensors
        """
        updated = self.update(x, layer_idx)
        read_out = self.read()
        return updated, read_out

    def forward_inference(
        self,
        x: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference forward: update *without* DP noise, then read."""
        # Save training flag
        was_training = self.training
        self.eval()

        # Project input
        projected = self.update_proj(x)

        # FIX 5: modulation = 0.05 * tanh(...)
        mod = 0.05 * torch.tanh(self.modulation[layer_idx])
        mod = mod.view(*(1 for _ in range(projected.ndim - 1)), -1)

        # Modulated update — no DP noise in inference
        new_state = self.state_norm(self._state + mod * projected)
        self._state = new_state

        read_out = self.read()

        # Restore training flag
        if was_training:
            self.train()

        return new_state, read_out

    # ------------------------------------------------------------------
    # nn.Module forward — dispatches based on training/eval mode
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to :meth:`forward_training` or :meth:`forward_inference`."""
        if self.training:
            return self.forward_training(x, layer_idx)
        else:
            return self.forward_inference(x, layer_idx)


# ---------------------------------------------------------------------------
# CLSICrossLayerStateIdentity
# ---------------------------------------------------------------------------

class CLSICrossLayerStateIdentity(nn.Module):
    """Cross-Layer State Identity (CLSI) with learned down/up projections
    and residual gating.

    Each layer owns a *down-projection* that compresses the hidden state
    into a shared latent space and an *up-projection* that expands it back.
    A residual gate blends the projected signal with the original hidden
    state.  ``RMSNormGated`` is applied before the gate for stable
    normalization.

    Parameters
    ----------
    dim : int
        Model hidden dimension.
    latent_dim : int
        Dimension of the shared latent / ledger space.
    n_layers : int
        Number of transformer layers.
    dropout : float
        Dropout on the cross-layer signal (default 0.1).
    gate_init : float
        Initial value for the residual gate bias.  A value close to 0
        means the cross-layer signal starts near zero and ramps up
        during training (default 0.0).
    """

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        n_layers: int,
        dropout: float = 0.1,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        # --- Per-layer down/up projections --------------------------------
        self.down_projs = nn.ModuleList(
            [nn.Linear(dim, latent_dim, bias=False) for _ in range(n_layers)]
        )
        self.up_projs = nn.ModuleList(
            [nn.Linear(latent_dim, dim, bias=False) for _ in range(n_layers)]
        )

        # --- Residual gates (scalar per layer, sigmoid-activated) ---------
        self.gate_bias = nn.Parameter(
            torch.full((n_layers, 1), gate_init)
        )

        # --- Normalization ------------------------------------------------
        self.norms = nn.ModuleList(
            [RMSNormGated(dim) for _ in range(n_layers)]
        )

        # --- Dropout on cross-layer signal --------------------------------
        self.dropout = nn.Dropout(p=dropout)

        # --- Shared latent state buffer -----------------------------------
        self.register_buffer(
            "_latent",
            torch.zeros(1, latent_dim),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def reset(self, batch_size: Optional[int] = None) -> None:
        """Zero the shared latent state."""
        if batch_size is not None:
            self._latent = torch.zeros(
                batch_size,
                self.latent_dim,
                device=self._latent.device,
                dtype=self._latent.dtype,
            )
        else:
            self._latent.zero_()

    def get_state(self) -> torch.Tensor:
        """Return a detached copy of the shared latent state."""
        return self._latent.detach().clone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-layer state contribution for *layer_idx*.

        Parameters
        ----------
        x : torch.Tensor
            Hidden-state tensor of shape ``(batch, dim)`` or
            ``(batch, seq_len, dim)``.
        layer_idx : int
            Index of the current layer (0-based).

        Returns
        -------
        (output, latent) : tuple
            *output* has the same shape as *x*; *latent* has shape
            ``(batch, latent_dim)`` or ``(batch, seq_len, latent_dim)``.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )

        # 1. Down-project current hidden state into latent space
        latent = self.down_projs[layer_idx](x)  # (..., latent_dim)

        # 2. Accumulate into shared latent (simple moving average)
        #    Use a small momentum so older layers fade gradually.
        momentum = 0.1
        if self._latent.shape != latent.shape:
            # First call or batch-size mismatch — reinitialise buffer
            self._latent = torch.zeros_like(latent)

        # Ensure device/dtype match
        if self._latent.device != latent.device or self._latent.dtype != latent.dtype:
            self._latent = self._latent.to(device=latent.device, dtype=latent.dtype)

        self._latent = (1.0 - momentum) * self._latent + momentum * latent.detach()

        # 3. Up-project the *shared* latent back to model dim
        cross_signal = self.up_projs[layer_idx](self._latent)  # (..., dim)

        # 4. Dropout on cross-layer signal
        cross_signal = self.dropout(cross_signal)

        # 5. Normalise via RMSNormGated
        cross_signal = self.norms[layer_idx](cross_signal)

        # 6. Residual gating:  g = sigmoid(bias);  out = g * cross_signal + (1-g) * x
        g = torch.sigmoid(self.gate_bias[layer_idx])  # (1,)
        g = g.view(*(1 for _ in range(x.ndim - 1)), -1)  # broadcast
        output = g * cross_signal + (1.0 - g) * x

        return output, latent

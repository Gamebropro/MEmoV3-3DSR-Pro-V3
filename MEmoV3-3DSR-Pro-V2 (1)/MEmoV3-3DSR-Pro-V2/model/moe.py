"""
MEmoV3-3DSR-Pro-V2  Mixture-of-Experts (MoE) Layer
====================================================
Top-2 gating with jitter noise, SwiGLU expert FFN,
and FIX-11 entropy-regularised load-balance loss.

CPU & GPU compatible.  Full type hints, no pseudocode.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standalone load-balance loss (with entropy regularisation – FIX 11)
# ---------------------------------------------------------------------------

def load_balance_loss(
    expert_indices: torch.Tensor,   # (batch, top_k)  int64
    expert_weights: torch.Tensor,   # (batch, top_k)  float
    probs: torch.Tensor,            # (batch, n_experts)  float – full gating probs
    n_experts: int,
    entropy_coeff: float = 0.01,
) -> torch.Tensor:
    """
    Compute the load-balance loss with entropy regularisation.

    Standard term:  L_bal = n_experts * sum_i( f_i * P_i )
        f_i = fraction of tokens routed to expert i
        P_i = mean gating probability for expert i

    FIX 11 – entropy regularisation to reduce expert CV:
        L_ent = -entropy_coeff * sum( probs * log(probs + 1e-8) )

    Total:  L_bal + L_ent
    """
    # ---- fraction of tokens per expert (f_i) ----
    # expert_indices: (batch, top_k) → flatten then one-hot
    flat_indices = expert_indices.reshape(-1)                        # (batch*top_k,)
    one_hot = F.one_hot(flat_indices, num_classes=n_experts).float() # (batch*top_k, n_experts)
    f = one_hot.mean(dim=0)                                          # (n_experts,)

    # ---- mean gating probability per expert (P_i) ----
    P = probs.mean(dim=0)  # (n_experts,)

    # ---- standard load-balance term ----
    bal = n_experts * (f * P).sum()

    # ---- FIX 11: entropy regularisation ----
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    ent = entropy_coeff * entropy

    return bal + ent


# ---------------------------------------------------------------------------
# MoERouter – Top-2 gating with jitter noise
# ---------------------------------------------------------------------------

class MoERouter(nn.Module):
    """
    Top-2 expert router with jitter noise during training.

    Returns
    -------
    expert_indices : (B, 2)   int64
    expert_weights : (B, 2)   float  (softmax-normalised, top-2 only)
    full_logits    : (B, E)   float  (raw gating logits, for aux loss)
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, d_model)

        Returns
        -------
        expert_indices, expert_weights, full_logits
        """
        logits = self.gate(x)  # (batch, n_experts)

        # Jitter noise (training only)
        if self.training and self.jitter_noise > 0.0:
            noise = torch.empty_like(logits).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
            logits = logits * noise

        # Full probability distribution (used by load-balance loss)
        probs = F.softmax(logits, dim=-1)  # (batch, n_experts)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # (batch, top_k)

        # Renormalise the top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        return top_k_indices, top_k_weights, logits

    def get_load_balance_loss(
        self,
        expert_indices: torch.Tensor,
        probs: torch.Tensor,
        n_experts: int,
        entropy_coeff: float = 0.01,
    ) -> torch.Tensor:
        """
        Convenience wrapper that also returns the full probability tensor
        so the caller does not need to recompute it.

        FIX 11: includes entropy regularisation.
        """
        return load_balance_loss(
            expert_indices=expert_indices,
            expert_weights=None,  # not needed inside load_balance_loss
            probs=probs,
            n_experts=n_experts,
            entropy_coeff=entropy_coeff,
        )


# ---------------------------------------------------------------------------
# MoEExpert – SwiGLU FFN
# ---------------------------------------------------------------------------

class MoEExpert(nn.Module):
    """
    Single expert: SwiGLU feed-forward network.

        output = w2( SiLU(w1(x)) * w3(x) )
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (..., d_model)

        Returns
        -------
        (..., d_model)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# MoELayer – Full Mixture-of-Experts layer
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer with Top-2 routing, SwiGLU experts,
    and FIX-11 entropy-regularised load-balance loss.

    Parameters
    ----------
    d_model : int
        Model / hidden dimension.
    n_experts : int
        Number of experts.
    d_ff : int | None
        Expert FFN inner dimension (default: 4 * d_model).
    top_k : int
        Number of experts to route each token to (default: 2).
    jitter_noise : float
        Magnitude of jitter noise injected during training (default: 0.1).
    entropy_coeff : float
        Coefficient for entropy regularisation in the load-balance loss
        (FIX 11, default: 0.01).
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        d_ff: int | None = None,
        top_k: int = 2,
        jitter_noise: float = 0.1,
        entropy_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.entropy_coeff = entropy_coeff

        self.router = MoERouter(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
        )

        self.experts = nn.ModuleList(
            [MoEExpert(d_model=d_model, d_ff=d_ff) for _ in range(n_experts)]
        )

    # ------------------------------------------------------------------ #
    #  Core forward
    # ------------------------------------------------------------------ #

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)  or  (batch, d_model)

        Returns
        -------
        output : (batch, seq_len, d_model)  or  (batch, d_model)
        load_balance_loss : scalar tensor
        """
        input_shape = x.shape
        squeezed = False

        # Allow 2-D input (batch, d_model)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, d_model)
            squeezed = True

        batch, seq_len, d_model = x.shape
        tokens = x.reshape(batch * seq_len, d_model)  # (N, d_model)

        # ---- routing ----
        expert_indices, expert_weights, full_logits = self.router(tokens)
        # expert_indices : (N, top_k)
        # expert_weights : (N, top_k)
        # full_logits    : (N, n_experts)

        # Full softmax probs for load-balance loss
        probs = F.softmax(full_logits, dim=-1)  # (N, n_experts)

        # ---- dispatch & compute ----
        # We iterate over each expert and gather the tokens routed to it.
        output = torch.zeros_like(tokens)  # (N, d_model)

        for k in range(self.top_k):
            # indices / weights for the k-th selected expert per token
            k_indices = expert_indices[:, k]   # (N,)
            k_weights = expert_weights[:, k]   # (N,)

            for e in range(self.n_experts):
                # Mask of tokens whose k-th choice is expert e
                mask = (k_indices == e)
                if not mask.any():
                    continue

                # Gather the masked tokens
                expert_input = tokens[mask]  # (n_e, d_model)
                expert_output = self.experts[e](expert_input)  # (n_e, d_model)

                # Weighted scatter-add
                weight = k_weights[mask].unsqueeze(-1)  # (n_e, 1)
                output[mask] += weight * expert_output

        # ---- reshape back ----
        output = output.reshape(batch, seq_len, d_model)
        if squeezed:
            output = output.squeeze(1)

        # ---- load-balance loss (FIX 11) ----
        lb_loss = load_balance_loss(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            probs=probs,
            n_experts=self.n_experts,
            entropy_coeff=self.entropy_coeff,
        )

        return output, lb_loss


# ---------------------------------------------------------------------------
# Quick self-test (run: python moe.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    d_model = 64
    n_experts = 8
    batch = 4
    seq_len = 16

    layer = MoELayer(
        d_model=d_model,
        n_experts=n_experts,
        d_ff=256,
        top_k=2,
        jitter_noise=0.1,
        entropy_coeff=0.01,
    )

    # --- CPU test ---
    x = torch.randn(batch, seq_len, d_model)
    out, lb = layer(x)
    print(f"[CPU]  output shape: {out.shape}  load_balance_loss: {lb.item():.4f}")
    assert out.shape == (batch, seq_len, d_model)
    assert lb.dim() == 0  # scalar

    # --- 2-D input test ---
    x2 = torch.randn(batch, d_model)
    out2, lb2 = layer(x2)
    print(f"[CPU 2D] output shape: {out2.shape}  load_balance_loss: {lb2.item():.4f}")
    assert out2.shape == (batch, d_model)

    # --- GPU test (if available) ---
    if torch.cuda.is_available():
        layer_gpu = layer.cuda()
        x_gpu = x.cuda()
        out_gpu, lb_gpu = layer_gpu(x_gpu)
        print(f"[GPU]  output shape: {out_gpu.shape}  load_balance_loss: {lb_gpu.item():.4f}")
        assert out_gpu.shape == (batch, seq_len, d_model)
    else:
        print("[GPU]  CUDA not available, skipping GPU test.")

    # --- Verify entropy term is present ---
    # When entropy_coeff > 0, the loss should differ from the pure
    # load-balance term.
    with torch.no_grad():
        logits_test = torch.randn(batch * seq_len, n_experts)
        probs_test = F.softmax(logits_test, dim=-1)
        indices_test = torch.randint(0, n_experts, (batch * seq_len, 2))
        weights_test = torch.ones(batch * seq_len, 2) / 2.0

        loss_with_ent = load_balance_loss(
            indices_test, weights_test, probs_test, n_experts, entropy_coeff=0.01
        )
        loss_no_ent = load_balance_loss(
            indices_test, weights_test, probs_test, n_experts, entropy_coeff=0.0
        )
        print(f"Loss w/ entropy: {loss_with_ent.item():.4f}  |  w/o entropy: {loss_no_ent.item():.4f}")
        assert not torch.isclose(loss_with_ent, loss_no_ent, atol=1e-6), \
            "Entropy term should change the loss value."

    print("\nAll self-tests passed!")

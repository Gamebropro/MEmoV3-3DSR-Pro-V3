"""
MEmoV3-3DSR-Pro V2 — Rectified Flow Sampler

Implements RectifiedFlowSampler for generative sampling via rectified flow
(Euler integration).  Supports three noise schedules:
    - **cosine**: Cosine schedule (Nichol & Dhariwal, 2021).
    - **linear**: Linear interpolation from noise to signal.
    - **EDM**: Euler Discretisation Method schedule with noise-dependent
      scaling (Karras et al., 2022).

The sampler integrates the learned velocity field from pure noise to the
data distribution using Euler's method.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Noise schedule enum
# ---------------------------------------------------------------------------

class NoiseSchedule(str, Enum):
    """Supported noise schedules for rectified flow sampling."""

    COSINE = "cosine"
    LINEAR = "linear"
    EDM = "edm"


# ---------------------------------------------------------------------------
# Schedule functions
# ---------------------------------------------------------------------------

def cosine_schedule(t: Tensor) -> Tensor:
    """Cosine noise schedule.

    Maps time ``t ∈ [0, 1]`` to a signal scale using the cosine function,
    providing smoother transitions near the endpoints compared to linear
    schedules.

    Args:
        t: Normalised time steps ``(batch,)`` or scalar, in [0, 1].

    Returns:
        Signal scale ``α(t)`` of the same shape.
    """
    # α(t) = cos(π/2 · t)  — goes from 1 → 0 as t goes 0 → 1
    return torch.cos(math.pi / 2.0 * t)


def linear_schedule(t: Tensor) -> Tensor:
    """Linear noise schedule.

    Simple linear interpolation: ``α(t) = 1 - t``.

    Args:
        t: Normalised time steps in [0, 1].

    Returns:
        Signal scale ``α(t)`` of the same shape.
    """
    return 1.0 - t


def edm_schedule(
    t: Tensor,
    sigma_data: float = 0.5,
    sigma_min: float = 1e-4,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> Tensor:
    """EDM (Karras et al., 2022) noise schedule.

    Uses a power-law mapping of normalised time to noise level, then
    converts to a signal scale.

    Args:
        t: Normalised time steps in [0, 1].
        sigma_data: Standard deviation of the data distribution.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        rho: Steepness of the schedule.

    Returns:
        Signal scale ``α(t)`` of the same shape.
    """
    # Map t → sigma via the EDM power schedule
    sigma = (
        sigma_min ** (1.0 / rho)
        + t * (sigma_max ** (1.0 / rho) - sigma_min ** (1.0 / rho))
    ) ** rho

    # Signal scale: α = sigma_data / sqrt(sigma² + sigma_data²)
    alpha = sigma_data / torch.sqrt(sigma.pow(2) + sigma_data ** 2)
    return alpha


# Map from enum to function
_SCHEDULE_FN = {
    NoiseSchedule.COSINE: cosine_schedule,
    NoiseSchedule.LINEAR: linear_schedule,
    NoiseSchedule.EDM: edm_schedule,
}


# ---------------------------------------------------------------------------
# RectifiedFlowSampler
# ---------------------------------------------------------------------------

class RectifiedFlowSampler(nn.Module):
    """Rectified flow sampler using Euler integration.

    Given a velocity model ``v_θ(x, t)`` that predicts the derivative of
    the signal with respect to time, the sampler integrates the ODE::

        dx/dt = v_θ(x, t)

    from ``t = 0`` (pure noise) to ``t = 1`` (data) using Euler steps.

    The coupling between noise and signal is::

        x_t = α(t) · x_1 + (1 - α(t)) · ε

    where ``ε ~ N(0, I)`` and ``α(t)`` is determined by the chosen schedule.

    Args:
        velocity_model: A callable ``v_θ(x, t) → velocity`` that takes a
            batch of noisy samples and a batch of time steps and returns
            the predicted velocity field.
        schedule: Noise schedule to use.  One of ``"cosine"``, ``"linear"``,
            or ``"edm"``.
        n_steps: Number of Euler integration steps.
        edm_sigma_data: Data standard deviation (only used with EDM schedule).
        edm_sigma_min: Minimum noise level (only used with EDM schedule).
        edm_sigma_max: Maximum noise level (only used with EDM schedule).
        edm_rho: Schedule steepness (only used with EDM schedule).
        clip_denoised: Whether to clip the final denoised output to [-1, 1].
    """

    def __init__(
        self,
        velocity_model: Callable[[Tensor, Tensor], Tensor],
        schedule: str = "cosine",
        n_steps: int = 50,
        edm_sigma_data: float = 0.5,
        edm_sigma_min: float = 1e-4,
        edm_sigma_max: float = 80.0,
        edm_rho: float = 7.0,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()

        self.velocity_model = velocity_model
        self.n_steps = n_steps
        self.clip_denoised = clip_denoised

        # Schedule
        try:
            self.schedule = NoiseSchedule(schedule)
        except ValueError:
            raise ValueError(
                f"Unknown schedule '{schedule}'. "
                f"Must be one of: {list(NoiseSchedule)}"
            )

        # EDM-specific parameters
        self.edm_sigma_data = edm_sigma_data
        self.edm_sigma_min = edm_sigma_min
        self.edm_sigma_max = edm_sigma_max
        self.edm_rho = edm_rho

    # -------------------------------------------------------------------

    def _alpha(self, t: Tensor) -> Tensor:
        """Compute the signal scale ``α(t)`` using the chosen schedule.

        Args:
            t: Normalised time in [0, 1].

        Returns:
            Signal scale of the same shape.
        """
        schedule_fn = _SCHEDULE_FN[self.schedule]
        if self.schedule == NoiseSchedule.EDM:
            return schedule_fn(
                t,
                sigma_data=self.edm_sigma_data,
                sigma_min=self.edm_sigma_min,
                sigma_max=self.edm_sigma_max,
                rho=self.edm_rho,
            )
        return schedule_fn(t)

    # -------------------------------------------------------------------

    def sample(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        return_trajectory: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Generate samples via rectified flow (Euler integration).

        Args:
            shape: Shape of the samples to generate, e.g.
                ``(batch, channels, height, width)`` or
                ``(batch, seq_len, dim)``.
            device: Device for the samples.
            dtype: Dtype for the samples.
            return_trajectory: If *True*, also return the trajectory of
                intermediate samples at each integration step.

        Returns:
            A tuple ``(samples, trajectory)`` where:
            - samples: Generated samples of the requested shape.
            - trajectory: Optional list of intermediate samples (one per
              Euler step) or *None*.
        """
        if device is None:
            device = next(
                (p.device for p in self.velocity_model.parameters()
                 if isinstance(self.velocity_model, nn.Module)),
                torch.device("cpu"),
            )

        # Start from pure noise (t = 0)
        x = torch.randn(shape, device=device, dtype=dtype)

        trajectory: Optional[List[Tensor]] = [] if return_trajectory else None

        dt = 1.0 / self.n_steps

        for step in range(self.n_steps):
            t_val = step / self.n_steps                          # current time
            t = torch.full((shape[0],), t_val, device=device, dtype=dtype)

            # Predict velocity
            v = self.velocity_model(x, t)

            # Euler step: x_{t+dt} = x_t + v(x_t, t) · dt
            x = x + v * dt

            if trajectory is not None:
                trajectory.append(x.clone())

        # Optional clipping
        if self.clip_denoised:
            x = x.clamp(-1.0, 1.0)

        return x, trajectory

    # -------------------------------------------------------------------

    def compute_loss(
        self,
        x_1: Tensor,
    ) -> Tensor:
        """Compute the rectified flow training loss.

        The loss is the simple MSE between the predicted velocity and the
        target velocity ``x_1 - ε``, derived from the linear interpolation
        coupling.

        Args:
            x_1: Clean data samples ``(batch, ...)``.

        Returns:
            Scalar loss tensor.
        """
        batch_size = x_1.shape[0]
        device = x_1.device
        dtype = x_1.dtype

        # Sample random t ∈ [0, 1]
        t = torch.rand(batch_size, device=device, dtype=dtype)

        # Sample noise
        epsilon = torch.randn_like(x_1)

        # Compute α(t)
        alpha_t = self._alpha(t)

        # Reshape alpha for broadcasting
        # alpha_t: (batch,) → (batch, 1, 1, ...) or (batch, 1, ...)
        n_extra_dims = x_1.dim() - 1
        alpha_t_expanded = alpha_t.view(
            batch_size, *([1] * n_extra_dims)
        )

        # Noisy sample: x_t = α(t) · x_1 + (1 - α(t)) · ε
        x_t = alpha_t_expanded * x_1 + (1.0 - alpha_t_expanded) * epsilon

        # Target velocity: dx/dt = x_1 - ε  (for the standard coupling)
        target_v = x_1 - epsilon

        # Predicted velocity
        predicted_v = self.velocity_model(x_t, t)

        # MSE loss
        loss = F.mse_loss(predicted_v, target_v)

        return loss

    # -------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a string with the module's configuration."""
        return (
            f"schedule={self.schedule.value}, n_steps={self.n_steps}, "
            f"clip_denoised={self.clip_denoised}"
        )

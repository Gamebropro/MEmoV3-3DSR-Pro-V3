#!/usr/bin/env python3
"""
MEmoV3-3DSR-Pro-V2  —  Training Script
========================================

Training entry-point for the Mamba3RP architecture with all critical fixes:

  FIX  3 (NO_DATA_PIPELINE)    : WebDataset for real data pipeline
  FIX  4 (WRONG_LOSS_FUNCTION) : RectifiedFlowLoss  (NOT CrossEntropy)
  FIX  6 (CHECKPOINT_CORRUPTION): Atomic checkpoint save via os.replace
  FIX  9 (SRS_GRADIENT_EXPLOSION): clip_grad_value_ on sr_scale parameters
  FIX 12 (NO_GRADIENT_CLIPPING): clip_grad_norm_ on all parameters

Additional features:
  - Full argparse CLI
  - Config loading from YAML
  - Gradient accumulation (8 steps)
  - Mixed precision (bf16)
  - Logging (loss, lr, grad norm, memory)
  - CPU + GPU auto-detect
  - Checkpoint resume support
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import webdataset as wds

# ---------------------------------------------------------------------------
# YAML support — graceful fallback
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Model import — supports both package-relative and direct imports
# ---------------------------------------------------------------------------
try:
    from model.mamba3_rp import Mamba3RP, Mamba3RPConfig
except ImportError:
    # Fallback: add parent directory to sys.path so we can import as a module
    _PROJECT_ROOT = Path(__file__).resolve().parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from model.mamba3_rp import Mamba3RP, Mamba3RPConfig


# =====================================================================
# Logging setup
# =====================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure the root logger with a clean format."""
    logger = logging.getLogger("MEmoV3-Train")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# =====================================================================
# FIX 3: Real data pipeline using WebDataset
# =====================================================================

def get_dataloader(
    urls: str = "data/{00000..00999}.tar",
    batch_size: int = 2,
    shuffle_buffer: int = 1000,
    num_workers: int = 4,
) -> wds.DataPipeline:
    """FIX 3: Real data pipeline using WebDataset.

    Parameters
    ----------
    urls           : str   — WebDataset shard pattern or comma-separated URLs
    batch_size     : int   — per-GPU batch size
    shuffle_buffer : int   — shuffle buffer size
    num_workers    : int   — DataLoader worker count

    Returns
    -------
    wds.DataPipeline  — iterable yielding (input_tensor, target_tensor) pairs
    """
    dataset = (
        wds.WebDataset(urls, resampled=True)
        .shuffle(shuffle_buffer)
        .decode("torch")
        .to_tuple("input.py", "target.py")
        .batched(batch_size)
    )
    return dataset


# =====================================================================
# FIX 4: Correct loss function for this architecture
# =====================================================================

class RectifiedFlowLoss(nn.Module):
    """FIX 4: Rectified Flow loss — the correct loss for this architecture.

    Instead of CrossEntropy (which is wrong for a rectified-flow / diffusion
    model), we use the simple MSE-style matching loss between the model
    prediction and the target flow vector.

    Reference: Liu et al., "Flow Straight and Fast: Learning to Generate
    and Transfer Data with Rectified Flow" (ICLR 2023).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute rectified flow loss.

        Parameters
        ----------
        pred   : (B, ...)  — model prediction
        target : (B, ...)  — ground-truth target

        Returns
        -------
        scalar loss
        """
        loss = (pred - target) ** 2
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


# =====================================================================
# FIX 6: Atomic checkpoint save
# =====================================================================

def save_ckpt(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """FIX 6: Atomic checkpoint save.

    Writes to a temporary file first, then atomically renames it to the
    target path.  This prevents corrupted checkpoints if the process is
    killed mid-write.

    Parameters
    ----------
    model     : nn.Module — the model to save
    path      : str       — destination file path (e.g. ``ckpt_1000.pt``)
    optimizer : optional optimizer to include in checkpoint
    scheduler : optional LR scheduler to include
    step      : current training step
    extras    : any additional key-value data to store
    """
    tmp_path = path + ".tmp"
    checkpoint: Dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extras is not None:
        checkpoint["extras"] = extras

    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX


def load_ckpt(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> int:
    """Load a checkpoint, restoring model / optimizer / scheduler state.

    Parameters
    ----------
    path      : str          — path to the checkpoint file
    model     : nn.Module    — model to load weights into
    optimizer : optional optimizer to restore
    scheduler : optional LR scheduler to restore
    device    : optional torch device to map tensors to

    Returns
    -------
    int  — the training step stored in the checkpoint (0 if absent)
    """
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt.get("step", 0)


# =====================================================================
# Config helpers
# =====================================================================

def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    yaml_path : str — path to the YAML file

    Returns
    -------
    dict — parsed configuration

    Raises
    ------
    RuntimeError — if PyYAML is not installed
    FileNotFoundError — if the file does not exist
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for loading YAML configs. "
            "Install it with: pip install pyyaml"
        )
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def build_config(cli_args: argparse.Namespace) -> Mamba3RPConfig:
    """Build Mamba3RPConfig from CLI args and optional YAML file.

    Priority: CLI flag > YAML value > default in dataclass.

    Parameters
    ----------
    cli_args : parsed argparse namespace

    Returns
    -------
    Mamba3RPConfig
    """
    overrides: Dict[str, Any] = {}

    # Load YAML if provided
    if cli_args.config and cli_args.config.strip():
        overrides.update(load_config_from_yaml(cli_args.config))

    # CLI flags override YAML
    _INT_FIELDS = {
        "d_model": "d_model",
        "n_layer": "n_layer",
        "d_state": "d_state",
        "d_conv": "d_conv",
        "expand": "expand",
        "rbf_num_centers": "rbf_num_centers",
        "n_mimo_paths": "n_mimo_paths",
        "n_experts": "n_experts",
        "n_active_experts": "n_active_experts",
        "context_window": "context_window",
        "block_size": "block_size",
        "vocab_size": "vocab_size",
    }
    _FLOAT_FIELDS = {
        "sr_scale": "sr_scale",
        "rbf_beta": "rbf_beta",
        "privacy_sigma": "privacy_sigma",
        "ledger_dropout": "ledger_dropout",
        "dropout": "dropout",
        "dt_min": "dt_min",
        "dt_max": "dt_max",
        "dt_init_floor": "dt_init_floor",
    }
    _BOOL_FIELDS = {
        "use_attnres": "use_attnres",
        "use_gradient_checkpointing": "use_gradient_checkpointing",
        "tie_embeddings": "tie_embeddings",
        "use_bias": "use_bias",
    }

    for cli_attr, cfg_attr in _INT_FIELDS.items():
        val = getattr(cli_args, cli_attr, None)
        if val is not None:
            overrides[cfg_attr] = val

    for cli_attr, cfg_attr in _FLOAT_FIELDS.items():
        val = getattr(cli_args, cli_attr, None)
        if val is not None:
            overrides[cfg_attr] = val

    for cli_attr, cfg_attr in _BOOL_FIELDS.items():
        val = getattr(cli_args, cli_attr, None)
        if val is not None:
            overrides[cfg_attr] = val

    return Mamba3RPConfig(**overrides)


# =====================================================================
# Device detection
# =====================================================================

def auto_detect_device() -> torch.device:
    """Auto-detect the best available compute device.

    Returns
    -------
    torch.device — cuda, mps, or cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# =====================================================================
# Gradient diagnostics
# =====================================================================

def compute_grad_norm(model: nn.Module) -> float:
    """Compute the total L2 gradient norm across all parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    float — total gradient norm
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm_sq)


# =====================================================================
# CLI argument parser
# =====================================================================

def build_argparser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        description="MEmoV3-3DSR-Pro-V2 Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Config file ---
    parser.add_argument(
        "--config", type=str, default="",
        help="Path to YAML config file (CLI flags take priority)",
    )

    # --- Data ---
    parser.add_argument(
        "--data-urls", type=str, default="data/{00000..00999}.tar",
        help="WebDataset shard pattern or comma-separated URLs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device batch size",
    )

    # --- Optimiser ---
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=8,
        help="Gradient accumulation steps before an optimizer step",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="FIX 12: Max gradient norm for clip_grad_norm_",
    )
    parser.add_argument(
        "--sr-scale-clip-value", type=float, default=5.0,
        help="FIX 9: Max value for clip_grad_value_ on sr_scale params",
    )

    # --- LR Schedule ---
    parser.add_argument(
        "--sched-t0", type=int, default=500,
        help="CosineAnnealingWarmRestarts T_0",
    )
    parser.add_argument(
        "--sched-t-mult", type=int, default=2,
        help="CosineAnnealingWarmRestarts T_mult",
    )

    # --- Training length ---
    parser.add_argument(
        "--max-steps", type=int, default=100_000,
        help="Total training steps",
    )

    # --- Checkpointing ---
    parser.add_argument(
        "--ckpt-dir", type=str, default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every", type=int, default=100,
        help="Save a checkpoint every N steps",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to checkpoint to resume from",
    )

    # --- Precision ---
    parser.add_argument(
        "--no-amp", action="store_true", default=False,
        help="Disable automatic mixed precision (bf16)",
    )

    # --- Logging ---
    parser.add_argument(
        "--log-every", type=int, default=10,
        help="Log training metrics every N steps",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level",
    )

    # --- Model architecture overrides ---
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--d-state", type=int, default=None)
    parser.add_argument("--d-conv", type=int, default=None)
    parser.add_argument("--expand", type=int, default=None)
    parser.add_argument("--sr-scale", type=float, default=None)
    parser.add_argument("--rbf-num-centers", type=int, default=None)
    parser.add_argument("--rbf-beta", type=float, default=None)
    parser.add_argument("--n-mimo-paths", type=int, default=None)
    parser.add_argument("--n-experts", type=int, default=None)
    parser.add_argument("--n-active-experts", type=int, default=None)
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--use-attnres", type=bool, default=None)
    parser.add_argument("--use-gradient-checkpointing", type=bool, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--privacy-sigma", type=float, default=None)
    parser.add_argument("--ledger-dropout", type=float, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--tie-embeddings", type=bool, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dt-min", type=float, default=None)
    parser.add_argument("--dt-max", type=float, default=None)
    parser.add_argument("--dt-init-floor", type=float, default=None)
    parser.add_argument("--use-bias", type=bool, default=None)

    # --- Device ---
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device: 'auto', 'cuda', 'mps', or 'cpu'",
    )

    return parser


# =====================================================================
# Main training loop
# =====================================================================

def train(cli_args: Optional[argparse.Namespace] = None) -> None:
    """Main training entry-point.

    Parameters
    ----------
    cli_args : optional pre-parsed namespace (if None, parses from sys.argv)
    """
    # ------------------------------------------------------------------
    # 1. Parse CLI & build config
    # ------------------------------------------------------------------
    if cli_args is None:
        cli_args = build_argparser().parse_args()

    logger = setup_logging(cli_args.log_level)
    logger.info("=" * 60)
    logger.info("MEmoV3-3DSR-Pro-V2 — Training")
    logger.info("=" * 60)

    config = build_config(cli_args)
    logger.info("Model config: %s", config)

    # ------------------------------------------------------------------
    # 2. Device
    # ------------------------------------------------------------------
    if cli_args.device == "auto":
        device = auto_detect_device()
    else:
        device = torch.device(cli_args.device)
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    model = Mamba3RP(config)
    model = model.to(device)

    n_params = model.get_num_params(non_embedding=True)
    n_params_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model parameters: %.2fM total, %.2fM non-embedding",
        n_params_total / 1e6,
        n_params / 1e6,
    )

    # ------------------------------------------------------------------
    # 4. Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cli_args.lr,
        weight_decay=cli_args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cli_args.sched_t0,
        T_mult=cli_args.sched_t_mult,
    )

    # ------------------------------------------------------------------
    # 5. Loss — FIX 4: RectifiedFlowLoss
    # ------------------------------------------------------------------
    criterion = RectifiedFlowLoss(reduction="mean")
    logger.info("Loss function: RectifiedFlowLoss  [FIX 4]")

    # ------------------------------------------------------------------
    # 6. DataLoader — FIX 3: WebDataset
    # ------------------------------------------------------------------
    dataloader = get_dataloader(
        urls=cli_args.data_urls,
        batch_size=cli_args.batch_size,
    )
    logger.info(
        "DataLoader: WebDataset  [FIX 3]  urls=%s  batch_size=%d",
        cli_args.data_urls,
        cli_args.batch_size,
    )

    # ------------------------------------------------------------------
    # 7. Mixed precision — bf16
    # ------------------------------------------------------------------
    use_amp = (not cli_args.no_amp) and (device.type == "cuda")
    scaler: Optional[torch.amp.GradScaler] = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    logger.info("Mixed precision: %s (amp=%s)", amp_dtype, use_amp)

    # ------------------------------------------------------------------
    # 8. Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    if cli_args.resume and cli_args.resume.strip():
        if os.path.isfile(cli_args.resume):
            start_step = load_ckpt(
                cli_args.resume,
                model,
                optimizer,
                scheduler,
                device=device,
            )
            logger.info("Resumed from %s at step %d", cli_args.resume, start_step)
        else:
            logger.warning("Resume checkpoint not found: %s — starting fresh", cli_args.resume)

    # ------------------------------------------------------------------
    # 9. Checkpoint directory
    # ------------------------------------------------------------------
    ckpt_dir = Path(cli_args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoint directory: %s", ckpt_dir)

    # ------------------------------------------------------------------
    # 10. Training loop
    # ------------------------------------------------------------------
    grad_accum_steps: int = cli_args.grad_accum_steps
    max_grad_norm: float = cli_args.max_grad_norm
    sr_scale_clip_value: float = cli_args.sr_scale_clip_value

    model.train()
    data_iter = iter(dataloader)

    logger.info(
        "Starting training: max_steps=%d  grad_accum=%d  start_step=%d",
        cli_args.max_steps,
        grad_accum_steps,
        start_step,
    )
    logger.info("FIX  6: Atomic checkpoint save enabled")
    logger.info("FIX  9: clip_grad_value_(sr_scale, %.1f)", sr_scale_clip_value)
    logger.info("FIX 12: clip_grad_norm_(all, %.1f)", max_grad_norm)

    running_loss: float = 0.0
    step_time: float = time.time()

    for step in range(start_step, cli_args.max_steps):
        # --------------------------------------------------------------
        # Accumulate gradients over micro-batches
        # --------------------------------------------------------------
        accumulated_loss: float = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(grad_accum_steps):
            # Fetch next batch (re-iterate dataset if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # batch is a tuple: (input_tensor, target_tensor)
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device, non_blocking=True)
                targets = batch[1].to(device, non_blocking=True)
            else:
                # Single-tensor WebDataset — use as both input and target
                inputs = batch.to(device, non_blocking=True)
                targets = inputs.clone()

            # Forward with mixed precision
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(inputs)
                # model.forward returns dict with "logits" key
                if isinstance(outputs, dict):
                    pred = outputs["logits"]
                else:
                    pred = outputs
                loss = criterion(pred, targets)
                loss = loss / grad_accum_steps  # scale for accumulation

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item() * grad_accum_steps

        # --------------------------------------------------------------
        # Gradient clipping — FIX 9 & FIX 12
        # --------------------------------------------------------------

        # Un-scale before clipping if using AMP
        if scaler is not None:
            scaler.unscale_(optimizer)

        # FIX 12: clip_grad_norm_ on ALL parameters
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        # FIX 9: clip_grad_value_ on sr_scale parameters specifically
        for name, param in model.named_parameters():
            if "sr_scale" in name and param.grad is not None:
                torch.nn.utils.clip_grad_value_(param, sr_scale_clip_value)

        # Optimiser step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------
        running_loss += accumulated_loss
        elapsed = time.time() - step_time

        if (step + 1) % cli_args.log_every == 0:
            avg_loss = running_loss / cli_args.log_every
            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else cli_args.lr
            grad_norm_val = total_grad_norm if isinstance(total_grad_norm, float) else total_grad_norm.item()

            # GPU memory (if available)
            mem_alloc_mb = 0.0
            mem_reserved_mb = 0.0
            if device.type == "cuda":
                mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                mem_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)

            tokens_per_sec = 0.0
            if elapsed > 0:
                # Approximate: batch_size * seq_len * grad_accum / elapsed
                tokens_per_sec = cli_args.batch_size * 1024 * grad_accum_steps / elapsed

            logger.info(
                "step=%6d | loss=%.6f | lr=%.2e | grad_norm=%.4f | "
                "mem_alloc=%.0fMB | mem_reserved=%.0fMB | "
                "tok/s=%.0f | dt=%.2fs",
                step + 1,
                avg_loss,
                current_lr,
                grad_norm_val,
                mem_alloc_mb,
                mem_reserved_mb,
                tokens_per_sec,
                elapsed,
            )

            running_loss = 0.0
            step_time = time.time()

        # --------------------------------------------------------------
        # Checkpointing — FIX 6: atomic save
        # --------------------------------------------------------------
        if (step + 1) % cli_args.save_every == 0:
            ckpt_path = str(ckpt_dir / f"ckpt_{step + 1}.pt")
            save_ckpt(
                model=model,
                path=ckpt_path,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step + 1,
            )
            logger.info("Checkpoint saved: %s  [FIX 6: atomic]", ckpt_path)

            # Also save a "latest" symlink / copy for easy resume
            latest_path = str(ckpt_dir / "latest.pt")
            save_ckpt(
                model=model,
                path=latest_path,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step + 1,
            )

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    final_path = str(ckpt_dir / "final.pt")
    save_ckpt(
        model=model,
        path=final_path,
        optimizer=optimizer,
        scheduler=scheduler,
        step=cli_args.max_steps,
    )
    logger.info("Training complete. Final checkpoint: %s", final_path)


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    train()

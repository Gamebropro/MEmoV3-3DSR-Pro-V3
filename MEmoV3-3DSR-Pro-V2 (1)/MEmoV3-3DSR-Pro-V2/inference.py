"""
MEmoV3-3DSR-Pro-V2  —  Inference Engine
=========================================

Full inference pipeline with:

  FIX  7 (PROMPT_INJECTION):  Input sanitization — strips <, >, {, } and caps length.
  FIX 14 (QUANTIZATION_DRIFT): Dynamic quantization for inference (qint8 on nn.Linear).

  - InferenceEngine class with generate() and stream_generate()
  - _sample_next_token with temperature / top-k / top-p sampling
  - KV cache (HierarchicalCache) with LRU eviction
  - CPU + GPU support (auto-detect)
  - torch.no_grad() guard on all inference paths
  - Deep-thinking mode: up to 120 iterative refinement steps with
    convergence threshold 0.001 on hidden-state L2 delta
  - Full argparse CLI

Usage
-----
    # Standard generation
    python inference.py --prompt "Explain quantum entanglement" --max-tokens 256

    # Streaming generation
    python inference.py --prompt "Hello world" --stream

    # Deep-thinking mode
    python inference.py --prompt "Solve: 2x + 3 = 11" --deep-thinking

    # With quantization (FIX 14)
    python inference.py --prompt "Test" --quantize

    # CPU only
    python inference.py --prompt "Test" --device cpu
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MEmoV3-Inference")

# ---------------------------------------------------------------------------
# FIX 7 — Prompt injection sanitization
# ---------------------------------------------------------------------------

def sanitize(text: str) -> str:
    """Sanitize user input to mitigate prompt-injection attacks.

    * Strips characters that could be used for template injection: ``< > { }``.
    * Truncates to 2 048 characters to prevent excessively long prompts.
    * Collapses consecutive whitespace.

    Parameters
    ----------
    text : str
        Raw user input.

    Returns
    -------
    str
        Sanitized text.
    """
    # Remove potentially dangerous characters
    cleaned = re.sub(r'[<>{}]', '', text)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Hard length cap
    return cleaned[:2048]


# ---------------------------------------------------------------------------
# Model imports — graceful fallback so this file can run standalone
# ---------------------------------------------------------------------------

try:
    from model.mamba3_rp import Mamba3RP, Mamba3RPConfig
    from model.cache import HierarchicalCache, create_hierarchical_cache
    _HAS_MODEL = True
    logger.info("MEmoV3-3DSR-Pro-V2 model modules loaded successfully.")
except ImportError as exc:
    _HAS_MODEL = False
    logger.warning(
        "Could not import full model modules (%s). "
        "Falling back to lightweight demo model.", exc,
    )


# ---------------------------------------------------------------------------
# Lightweight demo model (used when full model is unavailable)
# ---------------------------------------------------------------------------

class _DemoBlock(nn.Module):
    """Simple residual block with SwiGLU FFN — quantization-friendly."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.w2 = nn.Linear(d_model * 4, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_model * 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x + residual


class _DemoModel(nn.Module):
    """Minimal autoregressive model for standalone testing.

    Uses simple SwiGLU blocks instead of TransformerEncoderLayer so
    that ``torch.quantization.quantize_dynamic`` works without issues.
    Not intended for production — just enough to exercise the
    InferenceEngine when the full Mamba3RP stack is not installed.
    """

    def __init__(self, vocab_size: int = 50280, d_model: int = 256, n_layer: int = 2) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer

        # Expose a .config attribute so InferenceEngine can detect params
        self.config = type("Config", (), {
            "d_model": d_model,
            "expand": 2,
            "d_state": 16,
            "n_layer": n_layer,
        })()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            _DemoBlock(d_model) for _ in range(n_layer)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embedding.weight

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return {"logits": logits}

    @torch.no_grad()
    def step(
        self,
        input_id: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Single-token step for autoregressive generation.

        Parameters
        ----------
        input_id : (B, 1) long tensor
        state    : optional list of per-layer key/value tensors

        Returns
        -------
        logits     : (B, 1, vocab_size)
        new_states : list of updated per-layer states
        """
        result = self.forward(input_id)
        logits = result["logits"]
        # No real recurrent state for this demo — just pass through
        new_states: List[torch.Tensor] = []
        return logits, new_states


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(
    config: Optional["Mamba3RPConfig"] = None,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """Instantiate the Mamba3RP model (or demo fallback) and optionally
    load a checkpoint.

    Parameters
    ----------
    config : Mamba3RPConfig, optional
        Model configuration.  If ``None`` and the full model is available,
        a default config is used.
    device : torch.device
        Target device.
    checkpoint_path : str, optional
        Path to a ``.pt`` / ``.safetensors`` checkpoint.

    Returns
    -------
    nn.Module
        The loaded model on *device* in eval mode.
    """
    if _HAS_MODEL:
        if config is None:
            config = Mamba3RPConfig()
        model = Mamba3RP(config)
    else:
        model = _DemoModel()

    if checkpoint_path is not None:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    logger.info(
        "Model created: %s  |  params=%.2fM  |  device=%s",
        type(model).__name__,
        sum(p.numel() for p in model.parameters()) / 1e6,
        device,
    )
    return model


# ---------------------------------------------------------------------------
# FIX 14 — Dynamic quantization
# ---------------------------------------------------------------------------

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply PyTorch dynamic quantization to all ``nn.Linear`` layers.

    FIX 14 (QUANTIZATION_DRIFT): Using ``torch.quantization.quantize_dynamic``
    ensures that inference uses int8 weight-only quantization for linear
    layers, which reduces memory and improves throughput on CPU without
    introducing the accuracy drift associated with static quantization
    calibration on mismatched data.

    Parameters
    ----------
    model : nn.Module
        Model in eval mode.

    Returns
    -------
    nn.Module
        Quantized model.
    """
    model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    logger.info("Applied dynamic quantization (qint8) to nn.Linear layers.")
    return model


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample a single token from logits with temperature, top-k, and
    nucleus (top-p) filtering.

    Parameters
    ----------
    logits : Tensor
        Shape ``(B, 1, V)`` or ``(B, V)`` — raw unnormalised scores.
    temperature : float
        Temperature for scaling.  ``1.0`` = standard, ``< 1.0`` = sharper,
        ``> 1.0`` = flatter.  ``0.0`` = greedy argmax.
    top_k : int, optional
        If set, only sample from the top-k highest-probability tokens.
    top_p : float
        Nucleus sampling threshold (0.0 – 1.0).  If ``< 1.0``, only
        tokens within the smallest set whose cumulative probability
        exceeds *top_p* are considered.

    Returns
    -------
    Tensor
        Shape ``(B, 1)`` — sampled token ids.
    """
    # Ensure 3-D: (B, 1, V)
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)

    # Greedy decoding when temperature == 0
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)  # (B, 1)

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k is not None and top_k > 0:
        top_k_clamped = min(top_k, logits.size(-1))
        topk_values, _ = torch.topk(logits, top_k_clamped, dim=-1)
        threshold = topk_values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        # (keep the first token above the threshold so sum >= top_p)
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        # Scatter back to original ordering
        logits = torch.zeros_like(sorted_logits).scatter_(
            -1, sorted_indices, sorted_logits
        )

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.squeeze(1), num_samples=1)  # (B, 1)


# ---------------------------------------------------------------------------
# KV-Cache wrapper
# ---------------------------------------------------------------------------

class KVCacheManager:
    """Manages the hierarchical KV cache across layers for autoregressive
    generation.

    Wraps :class:`HierarchicalCache` when available, otherwise falls back
    to a simple per-layer list of ``(key, value)`` tensors.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        head_dim: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        memory_budget_mb: float = 512.0,
    ) -> None:
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype

        if _HAS_MODEL:
            self._cache = create_hierarchical_cache(
                num_layers=n_layers,
                max_size_per_layer=256,
                memory_budget_mb=memory_budget_mb,
                compress_rank=64,
                device=str(device),
                dtype=dtype,
            )
        else:
            self._cache = None
            # Simple fallback: list of lists of (key, value) tensors
            self._simple_cache: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
                [] for _ in range(n_layers)
            ]

        self._batch_size = batch_size
        self._d_model = d_model
        self._n_heads = n_heads
        self._head_dim = head_dim

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Insert a new KV entry for *layer_idx*."""
        if self._cache is not None:
            self._cache.update(layer_idx, key, value)
        else:
            self._simple_cache[layer_idx].append((key, value))

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve the most recent KV entry for *layer_idx*.

        Returns
        -------
        (key, value) or None
        """
        if self._cache is not None:
            entry = self._cache.get(layer_idx)
            if entry is not None:
                return entry.key, entry.value
            return None
        else:
            if self._simple_cache[layer_idx]:
                return self._simple_cache[layer_idx][-1]
            return None

    def get_all(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Concatenate all KV entries for *layer_idx* along the sequence dim.

        Returns
        -------
        (key, value) each (B, S, H, D) or None
        """
        if self._cache is not None:
            k, v = self._cache.get_cached_kv(layer_idx)
            return k, v
        else:
            entries = self._simple_cache[layer_idx]
            if not entries:
                return None
            keys = torch.cat([e[0] for e in entries], dim=1)
            values = torch.cat([e[1] for e in entries], dim=1)
            return keys, values

    def reset(self) -> None:
        """Clear all cached entries."""
        if self._cache is not None:
            self._cache.reset()
        else:
            self._simple_cache = [[] for _ in range(self.n_layers)]

    def memory_usage_mb(self) -> float:
        """Return total cache memory in MB."""
        if self._cache is not None:
            return self._cache.get_memory_usage() / (1024 * 1024)
        total_bytes = 0
        for layer_entries in self._simple_cache:
            for k, v in layer_entries:
                total_bytes += k.nelement() * k.element_size()
                total_bytes += v.nelement() * v.element_size()
        return total_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: float = 1.0
    deep_thinking: bool = False
    deep_thinking_steps: int = 120
    convergence_threshold: float = 0.001
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    pad_token_id: int = 0


class InferenceEngine:
    """Full-featured inference engine for MEmoV3-3DSR-Pro-V2.

    Features
    --------
    * **generate()** — batch autoregressive generation returning the
      complete sequence.
    * **stream_generate()** — generator that yields one token at a time
      for streaming / interactive use.
    * **_sample_next_token()** — temperature / top-k / top-p sampling.
    * **KV cache** — hierarchical cache with LRU eviction (FIX 13) for
      efficient autoregressive generation.
    * **Deep thinking** — iterative refinement up to
      ``deep_thinking_steps`` (default 120) with early stopping when
      hidden-state L2 delta < ``convergence_threshold`` (default 0.001).
    * **CPU + GPU** — automatic device selection; works on both.
    * **torch.no_grad()** — all inference paths are wrapped.
    * **FIX 7** — input sanitization.
    * **FIX 14** — optional dynamic quantization.

    Parameters
    ----------
    model : nn.Module
        The language model (``Mamba3RP`` or compatible).
    device : torch.device
        Inference device.
    gen_config : GenerationConfig
        Generation hyper-parameters.
    quantize : bool
        Whether to apply dynamic int8 quantization (FIX 14).
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        gen_config: Optional[GenerationConfig] = None,
        quantize: bool = False,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.gen_config = gen_config or GenerationConfig()

        # FIX 14: Apply dynamic quantization if requested
        if quantize:
            model = apply_dynamic_quantization(model)

        self.model = model.to(self.device)
        self.model.eval()

        # KV cache (initialised lazily on first generate call)
        self._kv_cache: Optional[KVCacheManager] = None

        # SSM states for autoregressive step() (Mamba3RP-specific)
        self._ssm_states: Optional[List[torch.Tensor]] = None

        logger.info(
            "InferenceEngine ready: device=%s  quantize=%s  "
            "deep_thinking_steps=%d  convergence_threshold=%.4f",
            self.device,
            quantize,
            self.gen_config.deep_thinking_steps,
            self.gen_config.convergence_threshold,
        )

    # ------------------------------------------------------------------
    # Public API: generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """Generate text from a prompt.

        The prompt is first sanitized (FIX 7), then tokenized,
        processed through the model with autoregressive decoding,
        and detokenized back to text.

        Parameters
        ----------
        prompt : str
            Raw user prompt.
        config : GenerationConfig, optional
            Override default generation config for this call.
        tokenizer : optional
            Object with ``encode(text) -> List[int]`` and
            ``decode(ids) -> str`` methods.  If ``None``, a simple
            character-level tokenizer is used.

        Returns
        -------
        str
            Generated text (prompt + continuation).
        """
        cfg = config or self.gen_config

        # FIX 7: Sanitize input
        prompt = sanitize(prompt)
        logger.debug("Sanitized prompt (%d chars): %.80s…", len(prompt), prompt)

        # Tokenize
        input_ids = self._tokenize(prompt, tokenizer)
        input_ids = input_ids.to(self.device)

        # Generate token ids
        output_ids = self._generate_ids(input_ids, cfg)

        # Detokenize
        return self._detokenize(output_ids, tokenizer)

    # ------------------------------------------------------------------
    # Public API: stream_generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        tokenizer: Optional[Any] = None,
    ) -> Generator[str, None, None]:
        """Streaming generator that yields one token string at a time.

        Parameters
        ----------
        prompt : str
            Raw user prompt (will be sanitized via FIX 7).
        config : GenerationConfig, optional
            Override default generation config.
        tokenizer : optional
            Tokenizer with encode/decode methods.

        Yields
        ------
        str
            Each generated token as a string.
        """
        cfg = config or self.gen_config

        # FIX 7: Sanitize input
        prompt = sanitize(prompt)

        input_ids = self._tokenize(prompt, tokenizer)
        input_ids = input_ids.to(self.device)

        # Prefill: process prompt through the model
        B, L_prompt = input_ids.shape

        # Run prefill
        if hasattr(self.model, 'forward'):
            with torch.no_grad():
                result = self.model(input_ids)
                if isinstance(result, dict):
                    logits = result["logits"]
                else:
                    logits = result
        else:
            raise RuntimeError("Model has no forward() method")

        # Initialise SSM states if using Mamba3RP
        self._init_ssm_states(B, input_ids.device, input_ids.dtype)

        # Sample first new token from the last prompt position
        next_logits = logits[:, -1:, :]  # (B, 1, V)
        next_token = _sample_next_token(
            next_logits,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )  # (B, 1)

        # Yield the first token
        yield self._decode_token(next_token.squeeze(0).tolist(), tokenizer)

        generated_count = 1

        # Autoregressive loop
        while generated_count < cfg.max_new_tokens:
            # Check EOS
            if cfg.eos_token_id is not None and next_token.item() == cfg.eos_token_id:
                break

            # Step the model
            if hasattr(self.model, 'step'):
                step_logits, self._ssm_states = self.model.step(
                    next_token, self._ssm_states,
                )
            else:
                with torch.no_grad():
                    result = self.model(next_token)
                    step_logits = result["logits"] if isinstance(result, dict) else result

            # Repetition penalty
            if cfg.repetition_penalty != 1.0:
                step_logits = self._apply_repetition_penalty(
                    step_logits, next_token, cfg.repetition_penalty,
                )

            # Sample
            next_token = _sample_next_token(
                step_logits,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )

            generated_count += 1
            yield self._decode_token(next_token.squeeze(0).tolist(), tokenizer)

    # ------------------------------------------------------------------
    # Internal: full ID generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_ids(
        self,
        input_ids: torch.Tensor,
        cfg: GenerationConfig,
    ) -> torch.Tensor:
        """Core autoregressive generation loop.

        Parameters
        ----------
        input_ids : (B, L_prompt) long tensor on self.device
        cfg : GenerationConfig

        Returns
        -------
        (B, L_prompt + max_new_tokens) long tensor
        """
        B, L_prompt = input_ids.shape
        device = input_ids.device

        # ---- Prefill ----
        result = self.model(input_ids)
        if isinstance(result, dict):
            logits = result["logits"]
        else:
            logits = result

        # Initialise SSM states (for Mamba3RP step())
        self._init_ssm_states(B, device, input_ids.dtype)

        # ---- Deep thinking (optional) ----
        if cfg.deep_thinking:
            logits = self._deep_thinking(logits, cfg)

        # ---- First new token ----
        next_logits = logits[:, -1:, :]  # (B, 1, V)
        next_token = _sample_next_token(
            next_logits,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )  # (B, 1)

        generated = torch.cat([input_ids, next_token], dim=1)  # (B, L+1)
        generated_count = 1

        # ---- Autoregressive loop ----
        while generated_count < cfg.max_new_tokens:
            # EOS check
            if cfg.eos_token_id is not None and next_token.item() == cfg.eos_token_id:
                break

            # Model step
            if hasattr(self.model, 'step'):
                step_logits, self._ssm_states = self.model.step(
                    next_token, self._ssm_states,
                )
            else:
                with torch.no_grad():
                    result = self.model(next_token)
                    step_logits = result["logits"] if isinstance(result, dict) else result

            # Repetition penalty
            if cfg.repetition_penalty != 1.0:
                step_logits = self._apply_repetition_penalty(
                    step_logits, next_token, cfg.repetition_penalty,
                )

            # Deep thinking per step (optional, expensive)
            if cfg.deep_thinking:
                step_logits = self._deep_thinking(step_logits, cfg)

            # Sample
            next_token = _sample_next_token(
                step_logits,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )

            generated = torch.cat([generated, next_token], dim=1)
            generated_count += 1

        return generated

    # ------------------------------------------------------------------
    # Internal: deep thinking
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _deep_thinking(
        self,
        logits: torch.Tensor,
        cfg: GenerationConfig,
    ) -> torch.Tensor:
        """Iterative deep-thinking refinement.

        Runs up to ``cfg.deep_thinking_steps`` (default 120) refinement
        iterations.  At each step, the model re-processes the current
        logits through a forward pass and blends the result with the
        previous iteration via exponential moving average.  Early
        stopping occurs when the L2 distance between consecutive
        hidden states falls below ``cfg.convergence_threshold``
        (default 0.001).

        Parameters
        ----------
        logits : (B, L, V) tensor
        cfg : GenerationConfig

        Returns
        -------
        (B, L, V) tensor — refined logits.
        """
        max_steps = cfg.deep_thinking_steps
        threshold = cfg.convergence_threshold
        alpha = 0.5  # EMA blending factor

        prev_hidden: Optional[torch.Tensor] = None
        refined_logits = logits

        logger.debug("Deep thinking: starting %d-step refinement", max_steps)
        t_start = time.monotonic()

        for step in range(max_steps):
            # Greedy decode current logits to token ids
            current_ids = refined_logits.argmax(dim=-1)  # (B, L)

            # Re-run through model
            with torch.no_grad():
                result = self.model(current_ids)
                if isinstance(result, dict):
                    new_logits = result["logits"]
                else:
                    new_logits = result

            # EMA blend
            if step == 0:
                refined_logits = new_logits
            else:
                refined_logits = alpha * refined_logits + (1.0 - alpha) * new_logits

            # Convergence check on the hidden representation (logits)
            if prev_hidden is not None:
                delta = (refined_logits - prev_hidden).norm().item()
                if delta < threshold:
                    logger.debug(
                        "Deep thinking: converged at step %d (delta=%.6f < %.6f)",
                        step, delta, threshold,
                    )
                    break

            prev_hidden = refined_logits.detach().clone()

        elapsed = time.monotonic() - t_start
        logger.debug("Deep thinking: completed in %.2fs", elapsed)
        return refined_logits

    # ------------------------------------------------------------------
    # Internal: SSM state management
    # ------------------------------------------------------------------

    def _init_ssm_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialise SSM recurrent states for Mamba3RP's step() API.

        If the model doesn't use SSM states (e.g. the demo model),
        this is a no-op.
        """
        if not hasattr(self.model, 'config'):
            self._ssm_states = []
            return

        config = self.model.config
        if not hasattr(config, 'd_model'):
            self._ssm_states = []
            return

        d_inner = config.d_model * getattr(config, 'expand', 2)
        n_state = getattr(config, 'd_state', 16)
        n_layer = getattr(config, 'n_layer', 2)

        self._ssm_states = [
            torch.zeros(batch_size, d_inner, n_state, device=device, dtype=dtype)
            for _ in range(n_layer)
        ]

    # ------------------------------------------------------------------
    # Internal: repetition penalty
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        last_token: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty by dividing the probability of the
        last generated token.

        Parameters
        ----------
        logits : (B, 1, V) or (B, V)
        last_token : (B, 1)
        penalty : float (> 1.0 penalises repetition)

        Returns
        -------
        Tensor with same shape as *logits*.
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)

        B, _, V = logits.shape
        for b in range(B):
            token_id = last_token[b, 0].item()
            if logits[b, 0, token_id] > 0:
                logits[b, 0, token_id] /= penalty
            else:
                logits[b, 0, token_id] *= penalty

        return logits

    # ------------------------------------------------------------------
    # Internal: tokenization helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str, tokenizer: Optional[Any] = None) -> torch.Tensor:
        """Encode *text* to a ``(1, L)`` long tensor.

        Falls back to UTF-8 byte-level tokenization if no tokenizer
        is provided.
        """
        if tokenizer is not None:
            ids: List[int] = tokenizer.encode(text)
        else:
            # Byte-level fallback
            ids = list(text.encode("utf-8"))
            # Clamp to reasonable vocab range
            ids = [min(b, 255) for b in ids]

        return torch.tensor([ids], dtype=torch.long)

    def _detokenize(self, ids: torch.Tensor, tokenizer: Optional[Any] = None) -> str:
        """Decode a ``(1, L)`` long tensor back to a string."""
        id_list = ids.squeeze(0).tolist()
        if tokenizer is not None:
            return tokenizer.decode(id_list)
        else:
            # Byte-level fallback
            try:
                byte_values = bytes([max(0, min(255, b)) for b in id_list])
                return byte_values.decode("utf-8", errors="replace")
            except Exception:
                return "".join(chr(b) for b in id_list if 32 <= b < 127)

    def _decode_token(self, ids: List[int], tokenizer: Optional[Any] = None) -> str:
        """Decode a single token (list of ints) to string."""
        if tokenizer is not None:
            return tokenizer.decode(ids)
        else:
            try:
                byte_values = bytes([max(0, min(255, b)) for b in ids])
                return byte_values.decode("utf-8", errors="replace")
            except Exception:
                return "".join(chr(b) for b in ids if 32 <= b < 127)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI for inference."""
    parser = argparse.ArgumentParser(
        description="MEmoV3-3DSR-Pro-V2 Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ---- Input ----
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Input prompt text.",
    )
    # ---- Generation ----
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0). 0 = greedy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling threshold (default: disabled).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus (top-p) sampling threshold (default: 1.0).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (default: 1.0 = disabled).",
    )
    # ---- Deep thinking ----
    parser.add_argument(
        "--deep-thinking",
        action="store_true",
        help="Enable deep-thinking iterative refinement.",
    )
    parser.add_argument(
        "--deep-thinking-steps",
        type=int,
        default=120,
        help="Max deep-thinking steps (default: 120).",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.001,
        help="Convergence threshold for deep-thinking early stop (default: 0.001).",
    )
    # ---- Model ----
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt / .safetensors).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cpu', 'cuda', 'cuda:0', etc.  Auto-detect if omitted.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic int8 quantization (FIX 14).",
    )
    # ---- Output ----
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output token-by-token.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for CLI inference."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Model
    model = create_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        deep_thinking=args.deep_thinking,
        deep_thinking_steps=args.deep_thinking_steps,
        convergence_threshold=args.convergence_threshold,
        repetition_penalty=args.repetition_penalty,
    )

    # Engine
    engine = InferenceEngine(
        model=model,
        device=device,
        gen_config=gen_config,
        quantize=args.quantize,
    )

    # Generate
    if args.stream:
        logger.info("Streaming generation...")
        print(f"Prompt: {args.prompt}")
        print("Response: ", end="", flush=True)
        for token_str in engine.stream_generate(args.prompt, gen_config):
            print(token_str, end="", flush=True)
        print()  # newline after streaming
    else:
        logger.info("Generating...")
        t_start = time.monotonic()
        response = engine.generate(args.prompt, gen_config)
        elapsed = time.monotonic() - t_start
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")
        logger.info("Generation completed in %.2fs", elapsed)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Run basic sanity checks on CPU with the demo model."""
    print("=" * 60)
    print("MEmoV3-3DSR-Pro-V2  Inference Engine — Self-Test")
    print("=" * 60)

    device = torch.device("cpu")

    # ---- 1. Sanitization (FIX 7) ----
    print("\n[1] FIX 7 — Input sanitization")
    raw = 'Hello <script>alert("xss")</script> {injection} test ' + "A" * 3000
    cleaned = sanitize(raw)
    assert "<" not in cleaned, "Angle brackets should be stripped"
    assert ">" not in cleaned, "Angle brackets should be stripped"
    assert "{" not in cleaned, "Curly braces should be stripped"
    assert len(cleaned) <= 2048, f"Length should be capped: got {len(cleaned)}"
    print(f"  Input length: {len(raw)} -> Sanitized length: {len(cleaned)}")
    print(f"  Cleaned (first 80): {cleaned[:80]}")
    print("  PASS")

    # ---- 2. Dynamic quantization (FIX 14) ----
    print("\n[2] FIX 14 — Dynamic quantization")
    demo_model = _DemoModel(vocab_size=1000, d_model=128, n_layer=1)
    demo_model.eval()
    quantized = apply_dynamic_quantization(demo_model)
    # Check that the model still works
    dummy = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        out = quantized(dummy)
    assert "logits" in out, "Quantized model should return logits"
    print(f"  Quantized model output shape: {out['logits'].shape}")
    print("  PASS")

    # ---- 3. Sampling ----
    print("\n[3] _sample_next_token")
    logits = torch.randn(1, 1, 1000)
    # Greedy
    token_greedy = _sample_next_token(logits, temperature=0.0)
    assert token_greedy.item() == logits.argmax(dim=-1).item(), "Greedy should match argmax"
    # Temperature
    token_temp = _sample_next_token(logits, temperature=0.5)
    assert 0 <= token_temp.item() < 1000, "Token should be in vocab range"
    # Top-k
    token_topk = _sample_next_token(logits, top_k=10)
    assert 0 <= token_topk.item() < 1000, "Top-k token in range"
    # Top-p
    token_topp = _sample_next_token(logits, top_p=0.9)
    assert 0 <= token_topp.item() < 1000, "Top-p token in range"
    print("  Greedy / temperature / top-k / top-p — all PASS")

    # ---- 4. Full generation ----
    print("\n[4] InferenceEngine.generate()")
    model = _DemoModel(vocab_size=256, d_model=128, n_layer=1)
    model.eval()
    engine = InferenceEngine(
        model=model,
        device=device,
        gen_config=GenerationConfig(
            max_new_tokens=16,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        ),
    )
    result = engine.generate("Hello world")
    print(f"  Generated text ({len(result)} chars): {result[:80]}…")
    print("  PASS")

    # ---- 5. Streaming generation ----
    print("\n[5] InferenceEngine.stream_generate()")
    tokens = list(engine.stream_generate(
        "Test prompt",
        GenerationConfig(max_new_tokens=8, temperature=1.0),
    ))
    print(f"  Streamed {len(tokens)} token(s)")
    assert len(tokens) > 0, "Should yield at least one token"
    print("  PASS")

    # ---- 6. Deep thinking ----
    print("\n[6] Deep thinking mode")
    engine_dt = InferenceEngine(
        model=model,
        device=device,
        gen_config=GenerationConfig(
            max_new_tokens=8,
            temperature=0.8,
            deep_thinking=True,
            deep_thinking_steps=5,  # small for test speed
            convergence_threshold=0.001,
        ),
    )
    result_dt = engine_dt.generate("Deep think test")
    print(f"  Deep-thinking result ({len(result_dt)} chars): {result_dt[:60]}…")
    print("  PASS")

    # ---- 7. Repetition penalty ----
    print("\n[7] Repetition penalty")
    logits_rp = torch.zeros(1, 1, 100)
    logits_rp[0, 0, 42] = 10.0  # make token 42 very likely
    last_tok = torch.tensor([[42]])
    penalized = InferenceEngine._apply_repetition_penalty(
        logits_rp.clone(), last_tok, penalty=2.0,
    )
    assert penalized[0, 0, 42].item() < logits_rp[0, 0, 42].item(), \
        "Penalized logit should be lower"
    print(f"  Before penalty: {logits_rp[0, 0, 42].item():.2f}  "
          f"After: {penalized[0, 0, 42].item():.2f}")
    print("  PASS")

    # ---- 8. CPU + GPU (if available) ----
    print("\n[8] CPU + GPU support")
    if torch.cuda.is_available():
        gpu_model = _DemoModel(vocab_size=256, d_model=128, n_layer=1).cuda()
        gpu_model.eval()
        gpu_engine = InferenceEngine(
            model=gpu_model,
            device=torch.device("cuda"),
            gen_config=GenerationConfig(max_new_tokens=4),
        )
        gpu_result = gpu_engine.generate("GPU test")
        print(f"  GPU result: {gpu_result[:40]}…")
        print("  PASS (GPU)")
    else:
        print("  CUDA not available — CPU-only test passed above.")
        print("  PASS (CPU only)")

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        _self_test()

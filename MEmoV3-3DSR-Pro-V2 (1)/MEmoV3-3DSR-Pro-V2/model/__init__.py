"""
MEmoV3-3DSR-Pro V2 — Model Package

Exports all public model classes and utilities.
"""

from model.stabilizer import (
    MIMOPathStabilizer,
    MIMOPath,
    get_path_diversity_loss,
    orthogonal_init_mimo_params,
)

from model.dit_block import (
    DiTBlock,
    ModulatedLayerNorm,
    DiTBlockStack,
)

from model.deep_think import (
    DeepThinkingEngine,
    ThinkNorm,
    ThinkProjection,
    DeepThinkingConfig,
    ConfidenceHead,
    sinusoidal_step_embedding,
)

from model.rope import (
    ComplexRoPE,
    apply_rotary_emb,
)

from model.rmsnorm_gated import (
    RMSNormGated,
)

from model.reflection_gate import (
    SelfReflectionGate,
)

from model.rectified_flow import (
    RectifiedFlowSampler,
    NoiseSchedule,
    cosine_schedule,
    linear_schedule,
    edm_schedule,
)

# ---------------------------------------------------------------------------
# Conditional imports — these modules may not be present yet during
# incremental development.  They are listed in the package's public API
# but are imported lazily so that the package does not crash if a
# dependency is still under construction.
# ---------------------------------------------------------------------------

_import_errors: list[str] = []

try:
    from model.mamba3_rp import Mamba3RP, Mamba3RPConfig, Mamba3RPBlock
except ImportError as _e:
    _import_errors.append(f"model.mamba3_rp: {_e}")
    Mamba3RP = None  # type: ignore[assignment, misc]
    Mamba3RPConfig = None  # type: ignore[assignment, misc]
    Mamba3RPBlock = None  # type: ignore[assignment, misc]

try:
    from model.ledger import LedgerState, CLSICrossLayerStateIdentity, merge_states
except ImportError as _e:
    _import_errors.append(f"model.ledger: {_e}")
    LedgerState = None  # type: ignore[assignment, misc]
    CLSICrossLayerStateIdentity = None  # type: ignore[assignment, misc]
    merge_states = None  # type: ignore[assignment, misc]

try:
    from model.attnres_kimi_triton import KimiAttentionResiduals
except ImportError as _e:
    _import_errors.append(f"model.attnres_kimi_triton: {_e}")
    KimiAttentionResiduals = None  # type: ignore[assignment, misc]

try:
    from model.context_window import ContextWindowManager, SlidingWindowAttention
except ImportError as _e:
    _import_errors.append(f"model.context_window: {_e}")
    ContextWindowManager = None  # type: ignore[assignment, misc]
    SlidingWindowAttention = None  # type: ignore[assignment, misc]

try:
    from model.moe import MoERouter, MoEExpert, MoELayer, load_balance_loss
except ImportError as _e:
    _import_errors.append(f"model.moe: {_e}")
    MoERouter = None  # type: ignore[assignment, misc]
    MoEExpert = None  # type: ignore[assignment, misc]
    MoELayer = None  # type: ignore[assignment, misc]
    load_balance_loss = None  # type: ignore[assignment, misc]

try:
    from model.cache import HierarchicalCache, HierarchicalCacheLayer, KVCacheEntry
except ImportError as _e:
    _import_errors.append(f"model.cache: {_e}")
    HierarchicalCache = None  # type: ignore[assignment, misc]
    HierarchicalCacheLayer = None  # type: ignore[assignment, misc]
    KVCacheEntry = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # stabilizer
    "MIMOPathStabilizer",
    "MIMOPath",
    "get_path_diversity_loss",
    "orthogonal_init_mimo_params",
    # dit_block
    "DiTBlock",
    "ModulatedLayerNorm",
    "DiTBlockStack",
    # deep_think
    "DeepThinkingEngine",
    "ThinkNorm",
    "ThinkProjection",
    "DeepThinkingConfig",
    "ConfidenceHead",
    "sinusoidal_step_embedding",
    # rope
    "ComplexRoPE",
    "apply_rotary_emb",
    # rmsnorm_gated
    "RMSNormGated",
    # reflection_gate
    "SelfReflectionGate",
    # rectified_flow
    "RectifiedFlowSampler",
    "NoiseSchedule",
    "cosine_schedule",
    "linear_schedule",
    "edm_schedule",
    # mamba3_rp (conditional)
    "Mamba3RP",
    "Mamba3RPConfig",
    "Mamba3RPBlock",
    # ledger (conditional)
    "LedgerState",
    "CLSICrossLayerStateIdentity",
    "merge_states",
    # attnres_kimi_triton (conditional)
    "KimiAttentionResiduals",
    # context_window (conditional)
    "ContextWindowManager",
    "SlidingWindowAttention",
    # moe (conditional)
    "MoERouter",
    "MoEExpert",
    "MoELayer",
    "load_balance_loss",
    # cache (conditional)
    "HierarchicalCache",
    "HierarchicalCacheLayer",
    "KVCacheEntry",
]

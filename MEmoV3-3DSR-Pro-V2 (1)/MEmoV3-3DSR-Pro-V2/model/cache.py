"""
MEmoV3-3DSR-Pro-V2 Hierarchical KV Cache with LRU Eviction (FIX 13)

Implements a multi-layer key-value cache with SVD-based compression
and memory-bounded LRU eviction to prevent memory leaks.

Components:
    - KVCacheEntry: Dataclass holding key/value tensors for a single entry.
    - HierarchicalCacheLayer: Per-layer KV cache with SVD compression + LRU eviction.
    - HierarchicalCache: Multi-layer cache coordinator with budget tracking + LRU eviction.
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KVCacheEntry
# ---------------------------------------------------------------------------

@dataclass
class KVCacheEntry:
    """A single entry in the KV cache storing key and value tensors.

    Attributes:
        key:   Key tensor of shape ``(seq_len, num_heads, head_dim)`` or
               ``(batch, seq_len, num_heads, head_dim)``.
        value: Value tensor of same shape as *key*.
        timestamp: Monotonically increasing timestamp used for LRU ordering.
    """

    key: torch.Tensor
    value: torch.Tensor
    timestamp: float = field(default_factory=time.monotonic)

    # -- helpers -------------------------------------------------------------

    def memory_bytes(self) -> int:
        """Return total memory consumed by key + value tensors in bytes."""
        return self.key.nelement() * self.key.element_size() + self.value.nelement() * self.value.element_size()

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "KVCacheEntry":
        """Move tensors to *device* (and optionally cast *dtype*)."""
        key = self.key.to(device=device, dtype=dtype)
        value = self.value.to(device=device, dtype=dtype)
        return KVCacheEntry(key=key, value=value, timestamp=self.timestamp)

    def pin_memory(self) -> "KVCacheEntry":
        """Pin underlying tensors for faster async CPU-GPU transfer."""
        return KVCacheEntry(
            key=self.key.pin_memory(),
            value=self.value.pin_memory(),
            timestamp=self.timestamp,
        )


# ---------------------------------------------------------------------------
# HierarchicalCacheLayer
# ---------------------------------------------------------------------------

class HierarchicalCacheLayer:
    """Per-layer KV cache with optional SVD compression and LRU eviction.

    FIX 13 (MEMORY_LEAK_CACHE): When the number of stored entries exceeds
    *max_size*, the oldest (least-recently-used) entry is evicted, keeping
    memory bounded.

    Args:
        layer_idx:    Index of the transformer layer this cache belongs to.
        max_size:     Maximum number of entries before LRU eviction kicks in.
        compress_rank: Rank used for truncated SVD compression (0 = disabled).
        device:       Torch device for stored tensors.
        dtype:        Torch dtype for stored tensors.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        max_size: int = 256,
        compress_rank: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.layer_idx = layer_idx
        self.max_size = max_size
        self.compress_rank = compress_rank
        self.device = device
        self.dtype = dtype

        # OrderedDict preserves insertion order – we use it for LRU.
        # The "oldest" entry is the first one inserted that has not been
        # re-accessed/promoted.
        self.cache: Dict[str, KVCacheEntry] = {}
        self._access_order: List[str] = []  # front = oldest, back = newest

        # Compression bookkeeping
        self._compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._total_entries_ever: int = 0
        self._eviction_count: int = 0

    # -- LRU helpers --------------------------------------------------------

    def _touch(self, entry_id: str) -> None:
        """Mark *entry_id* as recently used (move to end of LRU list)."""
        if entry_id in self._access_order:
            self._access_order.remove(entry_id)
        self._access_order.append(entry_id)

    def _evict_oldest(self) -> Optional[str]:
        """Evict the least-recently-used entry.  Returns evicted key or None."""
        if not self._access_order:
            return None
        oldest_key = self._access_order.pop(0)
        self.cache.pop(oldest_key, None)
        self._compressed.pop(oldest_key, None)
        self._eviction_count += 1
        logger.debug(
            "Layer %d LRU evicted entry '%s' (total evictions=%d)",
            self.layer_idx,
            oldest_key,
            self._eviction_count,
        )
        return oldest_key

    def _enforce_lru_limit(self) -> None:
        """FIX 13: Evict entries until cache size <= max_size."""
        while len(self.cache) > self.max_size:
            self._evict_oldest()

    # -- Public API ---------------------------------------------------------

    def update(self, key: torch.Tensor, value: torch.Tensor, entry_id: Optional[str] = None) -> str:
        """Insert or replace a KV entry.

        If *entry_id* is ``None`` a new auto-incremented id is generated.

        FIX 13: After insertion, if ``len(self.cache) > max_size``, the oldest
        entry is evicted via ``pop(0)`` on the LRU ordering list.
        """
        if entry_id is None:
            entry_id = f"entry_{self._total_entries_ever}"
        self._total_entries_ever += 1

        # Move to target device/dtype
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)

        entry = KVCacheEntry(key=key, value=value)

        # If key already exists, overwrite and promote in LRU
        if entry_id in self.cache:
            self._access_order.remove(entry_id)

        self.cache[entry_id] = entry
        self._access_order.append(entry_id)

        # FIX 13 – LRU eviction when cache exceeds max_size
        self._enforce_lru_limit()

        return entry_id

    def compress(self, entry_id: Optional[str] = None) -> None:
        """Apply truncated SVD compression to cached entries.

        If *entry_id* is given, compress only that entry; otherwise compress
        all entries whose rank is applicable.

        Compression replaces the stored ``(key, value)`` pair with their
        low-rank factors ``U_k, S_k, Vh_k`` and ``U_v, S_v, Vh_v`` which
        can reconstruct the original matrices approximately via
        ``U @ diag(S) @ Vh``.

        Only applies when ``compress_rank > 0`` and the tensor has enough
        columns for the requested rank.
        """
        if self.compress_rank <= 0:
            return

        target_ids = [entry_id] if entry_id is not None else list(self.cache.keys())

        for eid in target_ids:
            if eid not in self.cache:
                continue

            entry = self.cache[eid]
            k_compressed = self._svd_compress_tensor(entry.key, self.compress_rank)
            v_compressed = self._svd_compress_tensor(entry.value, self.compress_rank)
            if k_compressed is not None and v_compressed is not None:
                self._compressed[eid] = (*k_compressed, *v_compressed)

    def _svd_compress_tensor(
        self, tensor: torch.Tensor, rank: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compress a 2-D+ tensor via truncated SVD along the last two dims.

        Returns ``(U, S, Vh)`` or ``None`` if the tensor is too small.
        """
        original_shape = tensor.shape
        # Reshape to 2-D for SVD: (..., M, N) -> (M, N) by merging leading dims
        if tensor.dim() < 2:
            return None
        M, N = original_shape[-2], original_shape[-1]
        mat = tensor.reshape(-1, M, N)  # (batch_dims, M, N)
        # Process each batch slice; we average the singular vectors for a single
        # compressed representation.
        # For simplicity we SVD the mean across batch dims.
        mat_mean = mat.mean(dim=0)  # (M, N)
        if min(M, N) <= rank:
            return None
        try:
            U, S, Vh = torch.linalg.svd(mat_mean, full_matrices=False)
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            return (U.to(device=self.device, dtype=self.dtype),
                    S.to(device=self.device, dtype=self.dtype),
                    Vh.to(device=self.device, dtype=self.dtype))
        except Exception as exc:
            logger.warning("SVD compression failed for layer %d entry: %s", self.layer_idx, exc)
            return None

    def get(self, entry_id: Optional[str] = None) -> Optional[KVCacheEntry]:
        """Retrieve a cached KV entry.

        If *entry_id* is ``None``, returns the most-recently-used entry.
        Accessing an entry promotes it in the LRU ordering.
        """
        if not self.cache:
            return None

        if entry_id is None:
            # Return MRU entry
            entry_id = self._access_order[-1]

        entry = self.cache.get(entry_id, None)
        if entry is not None:
            # Promote in LRU
            self._touch(entry_id)
        return entry

    def get_compressed(
        self, entry_id: Optional[str] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return the compressed SVD factors for an entry.

        Returns ``(U_k, S_k, Vh_k, U_v, S_v, Vh_v)`` or ``None``.
        """
        if not self._compressed:
            return None
        if entry_id is None:
            entry_id = self._access_order[-1]
        return self._compressed.get(entry_id, None)

    def reset(self) -> None:
        """Clear all cached entries and compression data."""
        self.cache.clear()
        self._access_order.clear()
        self._compressed.clear()
        self._eviction_count = 0
        # Keep _total_entries_ever for ID uniqueness across resets

    # -- Stats --------------------------------------------------------------

    def memory_usage(self) -> int:
        """Total bytes used by cached key/value tensors."""
        total = 0
        for entry in self.cache.values():
            total += entry.memory_bytes()
        for factors in self._compressed.values():
            for f in factors:
                total += f.nelement() * f.element_size()
        return total

    def entry_count(self) -> int:
        return len(self.cache)

    def eviction_count(self) -> int:
        return self._eviction_count

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return (
            f"HierarchicalCacheLayer(layer={self.layer_idx}, entries={len(self.cache)}, "
            f"max_size={self.max_size}, evictions={self._eviction_count}, "
            f"memory={self.memory_usage() / 1024:.1f}KB)"
        )


# ---------------------------------------------------------------------------
# HierarchicalCache  (multi-layer coordinator)
# ---------------------------------------------------------------------------

class HierarchicalCache:
    """Multi-layer KV cache coordinator with memory budget tracking and LRU eviction.

    FIX 13 (MEMORY_LEAK_CACHE): When total memory across all layers exceeds
    *memory_budget_bytes*, the coordinator evicts the least-recently-used
    entry from the layer with the most entries, guaranteeing bounded memory.

    Args:
        num_layers:     Number of transformer layers.
        max_size_per_layer: Maximum entries per layer before per-layer LRU.
        memory_budget_bytes: Hard memory budget across all layers (0 = unlimited).
        compress_rank:  SVD compression rank (0 = disabled).
        device:         Default torch device.
        dtype:          Default torch dtype.
    """

    def __init__(
        self,
        num_layers: int = 12,
        max_size_per_layer: int = 256,
        memory_budget_bytes: int = 0,
        compress_rank: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_layers = num_layers
        self.max_size_per_layer = max_size_per_layer
        self.memory_budget_bytes = memory_budget_bytes
        self.compress_rank = compress_rank
        self.device = device
        self.dtype = dtype

        self.layers: Dict[int, HierarchicalCacheLayer] = {}
        for i in range(num_layers):
            self.layers[i] = HierarchicalCacheLayer(
                layer_idx=i,
                max_size=max_size_per_layer,
                compress_rank=compress_rank,
                device=device,
                dtype=dtype,
            )

        self._global_evictions: int = 0

    # -- Public API ---------------------------------------------------------

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, entry_id: Optional[str] = None) -> str:
        """Insert/replace a KV entry for *layer_idx*.

        Triggers:
            1. Per-layer LRU eviction if that layer exceeds its *max_size*.
            2. Global budget LRU eviction if total memory exceeds *memory_budget_bytes*.

        Returns the entry_id used.
        """
        if layer_idx not in self.layers:
            raise IndexError(f"Layer {layer_idx} not in cache (num_layers={self.num_layers})")

        entry_id = self.layers[layer_idx].update(key, value, entry_id)

        # FIX 13 – global memory budget LRU eviction
        if self.memory_budget_bytes > 0:
            self._enforce_global_budget()

        return entry_id

    def get(self, layer_idx: int, entry_id: Optional[str] = None) -> Optional[KVCacheEntry]:
        """Retrieve a KV entry from *layer_idx*.

        If *entry_id* is ``None``, returns the MRU entry for that layer.
        """
        if layer_idx not in self.layers:
            return None
        return self.layers[layer_idx].get(entry_id)

    def get_memory_usage(self) -> int:
        """Total memory consumed across all layers in bytes."""
        return sum(layer.memory_usage() for layer in self.layers.values())

    def get_compression_stats(self) -> Dict[str, Any]:
        """Return a dict of compression and cache statistics."""
        stats: Dict[str, Any] = {
            "total_memory_bytes": self.get_memory_usage(),
            "total_memory_mb": self.get_memory_usage() / (1024 * 1024),
            "memory_budget_bytes": self.memory_budget_bytes,
            "budget_utilization": (
                self.get_memory_usage() / self.memory_budget_bytes
                if self.memory_budget_bytes > 0
                else 0.0
            ),
            "global_evictions": self._global_evictions,
            "per_layer": {},
        }
        for idx, layer in self.layers.items():
            stats["per_layer"][idx] = {
                "entries": layer.entry_count(),
                "max_size": layer.max_size,
                "evictions": layer.eviction_count(),
                "memory_bytes": layer.memory_usage(),
                "memory_kb": layer.memory_usage() / 1024,
                "compressed_entries": len(layer._compressed),
            }
        return stats

    def compress_all(self, layer_idx: Optional[int] = None) -> None:
        """Run SVD compression on the specified layer or all layers."""
        if layer_idx is not None:
            if layer_idx in self.layers:
                self.layers[layer_idx].compress()
        else:
            for layer in self.layers.values():
                layer.compress()

    def reset(self, layer_idx: Optional[int] = None) -> None:
        """Reset a specific layer or all layers."""
        if layer_idx is not None:
            if layer_idx in self.layers:
                self.layers[layer_idx].reset()
        else:
            for layer in self.layers.values():
                layer.reset()
            self._global_evictions = 0

    # -- Global budget enforcement (FIX 13) ---------------------------------

    def _enforce_global_budget(self) -> None:
        """Evict LRU entries across layers until memory is within budget."""
        while self.get_memory_usage() > self.memory_budget_bytes and self._any_entries():
            # Find layer with the most entries – evict from there first
            max_layer_idx = max(self.layers.keys(), key=lambda i: len(self.layers[i].cache))
            evicted = self.layers[max_layer_idx]._evict_oldest()
            if evicted is not None:
                self._global_evictions += 1
            else:
                # Nothing left to evict
                break

    def _any_entries(self) -> bool:
        return any(len(l.cache) > 0 for l in self.layers.values())

    # -- Dunder methods -----------------------------------------------------

    def __len__(self) -> int:
        return sum(len(l) for l in self.layers.values())

    def __repr__(self) -> str:
        total = self.get_memory_usage()
        return (
            f"HierarchicalCache(layers={self.num_layers}, total_entries={len(self)}, "
            f"total_memory={total / 1024:.1f}KB, budget={self.memory_budget_bytes}, "
            f"global_evictions={self._global_evictions})"
        )


# ---------------------------------------------------------------------------
# Utility: create cache for CPU+GPU hybrid operation
# ---------------------------------------------------------------------------

def create_hierarchical_cache(
    num_layers: int = 12,
    max_size_per_layer: int = 256,
    memory_budget_mb: float = 512.0,
    compress_rank: int = 64,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> HierarchicalCache:
    """Factory function to build a :class:`HierarchicalCache`.

    Args:
        num_layers:         Number of transformer layers.
        max_size_per_layer: Max entries per layer before LRU eviction.
        memory_budget_mb:   Hard memory budget in **megabytes** (0 = unlimited).
        compress_rank:      SVD compression rank (0 = disabled).
        device:             ``"cpu"`` or ``"cuda"`` (or any valid torch device).
        dtype:              Tensor dtype for cached entries.

    Returns:
        A fully initialised :class:`HierarchicalCache`.
    """
    torch_device = torch.device(device)
    # Auto-select CUDA if available and requested device is generic "cuda"
    if device == "cuda" and torch.cuda.is_available():
        torch_device = torch.device("cuda")
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available – falling back to CPU")
        torch_device = torch.device("cpu")

    memory_budget_bytes = int(memory_budget_mb * 1024 * 1024) if memory_budget_mb > 0 else 0

    return HierarchicalCache(
        num_layers=num_layers,
        max_size_per_layer=max_size_per_layer,
        memory_budget_bytes=memory_budget_bytes,
        compress_rank=compress_rank,
        device=torch_device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Self-test (run with: python -m model.cache)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("MEmoV3-3DSR-Pro-V2  KV Cache – Self-Test")
    print("=" * 60)

    # --- 1. Basic layer test with LRU eviction ---
    print("\n[1] HierarchicalCacheLayer – LRU eviction test")
    layer = HierarchicalCacheLayer(layer_idx=0, max_size=3, device=torch.device("cpu"))
    for i in range(5):
        k = torch.randn(1, 4, 8)
        v = torch.randn(1, 4, 8)
        eid = layer.update(k, v)
        print(f"  Inserted {eid}  |  cache size={len(layer)}  |  {layer}")

    assert len(layer) == 3, f"Expected 3 entries after LRU, got {len(layer)}"
    print(f"  >> LRU eviction working: {layer.eviction_count()} evictions")

    # --- 2. Access promotion ---
    print("\n[2] LRU access promotion test")
    layer2 = HierarchicalCacheLayer(layer_idx=1, max_size=3)
    ids = []
    for i in range(3):
        k = torch.randn(1, 2, 4)
        v = torch.randn(1, 2, 4)
        ids.append(layer2.update(k, v))

    # Access the oldest entry to promote it
    _ = layer2.get(ids[0])
    # Insert two more – ids[1] and ids[2] should be evicted before ids[0]
    for i in range(2):
        k = torch.randn(1, 2, 4)
        v = torch.randn(1, 2, 4)
        layer2.update(k, v)

    assert ids[0] in layer2.cache, "Promoted entry should survive"
    assert ids[1] not in layer2.cache, "Non-promoted entry should be evicted"
    print(f"  >> Access promotion works: ids[0] survives, ids[1] evicted")

    # --- 3. SVD compression ---
    print("\n[3] SVD compression test")
    layer3 = HierarchicalCacheLayer(layer_idx=2, max_size=10, compress_rank=4)
    k = torch.randn(16, 32)
    v = torch.randn(16, 32)
    eid = layer3.update(k, v)
    layer3.compress(eid)
    factors = layer3.get_compressed(eid)
    assert factors is not None, "Compressed factors should exist"
    U_k, S_k, Vh_k, U_v, S_v, Vh_v = factors
    print(f"  >> SVD factors: U_k={U_k.shape}, S_k={S_k.shape}, Vh_k={Vh_k.shape}")

    # --- 4. Multi-layer HierarchicalCache ---
    print("\n[4] HierarchicalCache – multi-layer with budget")
    cache = create_hierarchical_cache(
        num_layers=4,
        max_size_per_layer=10,
        memory_budget_mb=0.01,  # ~10KB budget – very tight to force global eviction
        compress_rank=0,
    )
    for layer_idx in range(4):
        for i in range(10):
            k = torch.randn(8, 16)
            v = torch.randn(8, 16)
            cache.update(layer_idx, k, v)

    stats = cache.get_compression_stats()
    print(f"  >> Total entries: {len(cache)}")
    print(f"  >> Memory used: {stats['total_memory_mb']:.4f} MB")
    print(f"  >> Budget: {stats['memory_budget_bytes']} bytes")
    print(f"  >> Global evictions: {stats['global_evictions']}")
    assert cache.get_memory_usage() <= cache.memory_budget_bytes or cache.memory_budget_bytes == 0
    print(f"  >> Memory budget enforcement working")

    # --- 5. Reset ---
    print("\n[5] Reset test")
    cache.reset()
    assert len(cache) == 0, "Cache should be empty after reset"
    print(f"  >> Cache reset successful: {cache}")

    # --- 6. CPU + GPU test ---
    print("\n[6] CPU+GPU hybrid test")
    cache_gpu = create_hierarchical_cache(
        num_layers=2,
        max_size_per_layer=5,
        memory_budget_mb=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)
    eid = cache_gpu.update(0, k, v)
    entry = cache_gpu.get(0, eid)
    assert entry is not None
    print(f"  >> Device: {entry.key.device}  |  dtype: {entry.key.dtype}")

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)

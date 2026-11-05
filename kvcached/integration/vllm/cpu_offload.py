# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Minimal CPU retention for cached prefix blocks evicted by a shrink.

When a lowered memory limit forces kvcached to evict cached prefixes, the
block pool copies each block's KV data to CPU memory first. A later prefix
lookup that misses on the GPU copies the block back into a fresh GPU block
instead of recomputing it.

This is deliberately simple: saves are synchronous, every shrink-evicted block
is kept, the CPU store is an unbounded dict of pinned buffers, and only
attention KV groups are supported. It requires the
block pool and the KV tensors to live in one process (a single local worker).
Enable it with ``KVCACHED_CPU_OFFLOAD=1``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()

ENABLED = os.getenv("KVCACHED_CPU_OFFLOAD", "0") == "1"


@dataclass
class _GroupViews:
    tensors: List[torch.Tensor]  # one per layer, as handed to vLLM
    blocks_dim: int
    ratio: int  # kernel blocks per kvcached block


@dataclass
class OffloadStats:
    saved_blocks: int = 0
    saved_bytes: int = 0
    save_s: float = 0.0
    restored_blocks: int = 0
    restore_s: float = 0.0
    restore_failures: int = 0
    restore_no_free_block: int = 0
    restore_batches: int = 0
    lookup_cpu_misses: int = 0  # GPU miss and not in the CPU store either
    by_reason: Dict[str, int] = field(default_factory=dict)


_views: Dict[int, _GroupViews] = {}
_store: Dict[Any, List[torch.Tensor]] = {}  # cache key -> per-layer CPU copy
stats = OffloadStats()


def register_kv_tensors(group_id: int, tensors: List[torch.Tensor],
                        blocks_dim: int, ratio: int) -> None:
    if ENABLED:
        _views[group_id] = _GroupViews(list(tensors), blocks_dim, ratio)
        logger.info("CPU offload: registered %d KV tensors for group %d",
                    len(tensors), group_id)
        _restore_before_lookup()


def _restore_before_lookup() -> None:
    """Restore a request's saved prefix blocks in one batch before vLLM looks
    them up, so the lookup finds them on the GPU. Restoring them one at a time
    during the lookup costs about 16 small copies per block."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager.get_computed_blocks, "_cpu_offload", False):
        return
    original = KVCacheManager.get_computed_blocks

    def get_computed_blocks(self, request):
        restore_prefix = getattr(self.block_pool, "restore_prefix", None)
        if (_store and restore_prefix is not None and self.num_kv_cache_groups == 1
                and self.prefix_cache_lookup_enabled(request)):
            spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
            if getattr(self.block_pool, "hash_block_size", spec.block_size) == spec.block_size:
                restore_prefix(request.block_hashes, (request.num_tokens - 1) // spec.block_size)
        return original(self, request)

    get_computed_blocks._cpu_offload = True
    KVCacheManager.get_computed_blocks = get_computed_blocks


def ready(group_id: int) -> bool:
    return ENABLED and group_id in _views


def _block_slices(group_id: int, block_id: int) -> List[torch.Tensor]:
    v = _views[group_id]
    return [t.narrow(v.blocks_dim, block_id * v.ratio, v.ratio) for t in v.tensors]


_SAVE_CHUNK = 256  # blocks gathered per GPU copy, bounding temporary GPU memory


def save(blocks: List[tuple[Any, int, int]], reason: str) -> None:
    """Copy (cache key, group id, block id) triples to CPU before they are freed."""
    blocks = [b for b in blocks if ready(b[1])]
    if not blocks:
        return
    start = time.perf_counter()
    nbytes = 0
    for group_id in {g for _, g, _ in blocks}:
        v = _views[group_id]
        group = [(k, b) for k, g, b in blocks if g == group_id]
        for i in range(0, len(group), _SAVE_CHUNK):
            chunk = group[i:i + _SAVE_CHUNK]
            # Kernel-block indices for each kvcached block, in chunk order.
            idx = torch.tensor([b * v.ratio + r for _, b in chunk for r in range(v.ratio)],
                               device=v.tensors[0].device)
            # Pinned host memory lets restores copy asynchronously over PCIe.
            per_layer = []
            for t in v.tensors:
                gathered = t.index_select(v.blocks_dim, idx)
                host = torch.empty(gathered.shape, dtype=gathered.dtype,
                                   pin_memory=torch.cuda.is_available())
                host.copy_(gathered)
                per_layer.append(host)
            for j, (key, _) in enumerate(chunk):
                copies = [c.narrow(v.blocks_dim, j * v.ratio, v.ratio) for c in per_layer]
                _store[key] = copies
            nbytes += sum(c.numel() * c.element_size() for c in per_layer)
    elapsed = time.perf_counter() - start
    stats.saved_blocks += len(blocks)
    stats.saved_bytes += nbytes
    stats.save_s += elapsed
    stats.by_reason[reason] = stats.by_reason.get(reason, 0) + len(blocks)
    logger.info(
        "CPU offload: saved %d blocks (%.1f MiB) in %.1f ms; store holds %d blocks",
        len(blocks), nbytes / 2**20, elapsed * 1e3, len(_store))


def has(key: Any) -> bool:
    found = key in _store
    if not found and _store:
        stats.lookup_cpu_misses += 1
        if stats.lookup_cpu_misses in (1, 10, 100, 1000) or stats.lookup_cpu_misses % 5000 == 0:
            logger.info("CPU offload: %d lookups missed both GPU and CPU store "
                        "(%d restored so far)", stats.lookup_cpu_misses, stats.restored_blocks)
    return found


def stored(key: Any) -> bool:
    """Like has(), without counting a miss."""
    return key in _store


def restore_many(keys: List[Any], group_id: int, block_ids: List[int]) -> bool:
    """Copy saved blocks into fresh GPU blocks with one copy per layer."""
    if not keys or not ready(group_id) or any(k not in _store for k in keys):
        return False
    v = _views[group_id]
    start = time.perf_counter()
    try:
        idx = torch.tensor([b * v.ratio + r for b in block_ids for r in range(v.ratio)],
                           device=v.tensors[0].device)
        for layer, dst in enumerate(v.tensors):
            src = torch.cat([_store[k][layer] for k in keys], dim=v.blocks_dim)
            dst.index_copy_(v.blocks_dim, idx, src.to(dst.device))
    except Exception:
        stats.restore_failures += 1
        logger.exception("CPU offload: batched restore of %d blocks failed", len(keys))
        return False
    stats.restored_blocks += len(keys)
    stats.restore_s += time.perf_counter() - start
    stats.restore_batches += 1
    if stats.restore_batches in (1, 10) or stats.restore_batches % 50 == 0:
        logger.info("CPU offload: restored %d blocks in %d batches, %.1f ms total; "
                    "%d not restored for lack of a free GPU block", stats.restored_blocks,
                    stats.restore_batches, stats.restore_s * 1e3, stats.restore_no_free_block)
    return True


def note_no_free_block() -> None:
    """A saved block could not be restored because no GPU block was free."""
    stats.restore_no_free_block += 1
    if stats.restore_no_free_block in (1, 10, 100) or stats.restore_no_free_block % 1000 == 0:
        logger.info("CPU offload: %d restores failed for lack of a free GPU block "
                    "(%d restored)", stats.restore_no_free_block, stats.restored_blocks)


def restore(key: Any, group_id: int, block_id: int) -> bool:
    """Copy a saved block into a freshly allocated GPU block.

    The copies are queued on the current stream, ahead of any model step that
    reads the block. The CPU copy stays in the store, so its pinned memory is
    never freed while a copy from it may still be in flight.
    """
    copies = _store.get(key)
    if copies is None or not ready(group_id):
        return False
    start = time.perf_counter()
    try:
        for dst, src in zip(_block_slices(group_id, block_id), copies):
            dst.copy_(src, non_blocking=True)
    except Exception:
        stats.restore_failures += 1
        logger.exception("CPU offload: restore failed for block %d", block_id)
        return False
    stats.restored_blocks += 1
    stats.restore_s += time.perf_counter() - start
    if stats.restored_blocks % 64 == 1:
        logger.info(
            "CPU offload: restored %d blocks, %.1f ms total to queue copies; "
            "%d not restored for lack of a free GPU block",
            stats.restored_blocks, stats.restore_s * 1e3, stats.restore_no_free_block)
    return True

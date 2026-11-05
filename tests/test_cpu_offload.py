# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU retention of shrink-evicted prefix blocks (cpu_offload.py)."""

import pytest

torch = pytest.importorskip("torch")

from kvcached.integration.vllm import cpu_offload  # noqa: E402
from test_prefix_cache import pool_factory  # noqa: E402,F401

NUM_BLOCKS = 16
# FlashAttention layout: (K/V, blocks, tokens per block, heads, head dim).
SHAPE = (2, NUM_BLOCKS, 4, 2, 8)


@pytest.fixture
def offload(monkeypatch):
    monkeypatch.setattr(cpu_offload, "ENABLED", True)
    monkeypatch.setattr(cpu_offload, "_views", {})
    monkeypatch.setattr(cpu_offload, "_store", {})
    monkeypatch.setattr(cpu_offload, "stats", cpu_offload.OffloadStats())
    torch.manual_seed(0)
    layers = [torch.randn(SHAPE) for _ in range(3)]
    cpu_offload.register_kv_tensors(0, layers, blocks_dim=1, ratio=1)
    return layers


def _block(layers, block_id):
    return [t[:, block_id].clone() for t in layers]


def _key(name: bytes, group_id: int = 0) -> bytes:
    return name + group_id.to_bytes(4, "big")


def test_disabled_by_default_registers_nothing(monkeypatch):
    monkeypatch.setattr(cpu_offload, "ENABLED", False)
    monkeypatch.setattr(cpu_offload, "_views", {})
    cpu_offload.register_kv_tensors(0, [torch.zeros(SHAPE)], blocks_dim=1, ratio=1)
    assert not cpu_offload.ready(0)


def test_saved_block_restores_into_a_different_block(offload):
    original = _block(offload, 3)
    cpu_offload.save([(_key(b"a"), 0, 3), (_key(b"b"), 0, 5)], "shrink")
    for t in offload:
        t[:, 3].zero_()  # the evicted page is reused by another engine

    assert cpu_offload.restore(_key(b"a"), 0, 9)
    for got, want in zip(_block(offload, 9), original):
        assert torch.equal(got, want)
    # The CPU copy is kept so its memory outlives any in-flight async copy.
    assert cpu_offload.has(_key(b"a"))
    assert cpu_offload.has(_key(b"b"))
    assert cpu_offload.stats.saved_blocks == 2
    assert cpu_offload.stats.restored_blocks == 1


def test_save_copies_rather_than_aliasing_gpu_memory(offload):
    original = _block(offload, 2)
    cpu_offload.save([(_key(b"a"), 0, 2)], "shrink")
    for t in offload:
        t[:, 2].fill_(7.0)
    assert cpu_offload.restore(_key(b"a"), 0, 2)
    for got, want in zip(_block(offload, 2), original):
        assert torch.equal(got, want)


def test_ratio_saves_every_kernel_block(monkeypatch):
    monkeypatch.setattr(cpu_offload, "ENABLED", True)
    monkeypatch.setattr(cpu_offload, "_views", {})
    monkeypatch.setattr(cpu_offload, "_store", {})
    layer = torch.arange(2 * 8 * 3, dtype=torch.float32).view(2, 8, 3)  # 4 blocks x 2 kernel blocks
    cpu_offload.register_kv_tensors(0, [layer], blocks_dim=1, ratio=2)
    want = layer[:, 2:4].clone()  # kvcached block 1
    cpu_offload.save([(_key(b"a"), 0, 1)], "shrink")
    assert cpu_offload.restore(_key(b"a"), 0, 3)
    assert torch.equal(layer[:, 6:8], want)


def test_batched_restore_places_each_block(offload):
    want = {b: _block(offload, b) for b in (1, 4, 6)}
    keys = [_key(bytes([b])) for b in (1, 4, 6)]
    cpu_offload.save([(k, 0, b) for k, b in zip(keys, (1, 4, 6))], "shrink")
    for t in offload:
        t.zero_()

    assert cpu_offload.restore_many(keys, 0, [10, 2, 13])  # arbitrary new homes
    for src, dst in ((1, 10), (4, 2), (6, 13)):
        for got, expected in zip(_block(offload, dst), want[src]):
            assert torch.equal(got, expected)
    assert torch.count_nonzero(offload[0][:, 1]) == 0  # nothing written elsewhere


def test_batched_restore_refuses_unknown_keys(offload):
    cpu_offload.save([(_key(b"a"), 0, 1)], "shrink")
    assert not cpu_offload.restore_many([_key(b"a"), _key(b"missing")], 0, [2, 3])


def test_restore_unknown_key_is_a_miss(offload):
    assert not cpu_offload.restore(_key(b"missing"), 0, 1)


def test_pool_restores_a_shrink_evicted_prefix_on_lookup(pool_factory, offload):  # noqa: F811
    pool, _ = pool_factory(num_blocks=NUM_BLOCKS)
    (block,) = pool.get_new_blocks(1)
    key = _key(b"prefix")
    pool._index_restored_block(key, block)  # cache it as a finished request would
    pool.free_blocks([block])
    original = _block(offload, block.block_id)

    pool._evict_block_ids([block.block_id], reason="shrink")
    assert pool.get_cached_block(b"prefix") is not None  # restored, not a miss
    restored = pool.get_cached_block(b"prefix")
    for got, want in zip(_block(offload, restored.block_id), original):
        assert torch.equal(got, want)
    assert restored.block_id in pool._evictable_blocks


def test_pool_does_not_retain_non_shrink_evictions(pool_factory, offload):  # noqa: F811
    pool, _ = pool_factory(num_blocks=NUM_BLOCKS)
    (block,) = pool.get_new_blocks(1)
    pool._index_restored_block(_key(b"prefix"), block)
    pool.free_blocks([block])

    pool._evict_block_ids([block.block_id], reason="shortage")
    assert pool.get_cached_block(b"prefix") is None
    assert cpu_offload.stats.saved_blocks == 0

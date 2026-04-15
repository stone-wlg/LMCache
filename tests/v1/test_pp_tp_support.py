# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TP>1 and PP>1 support in LMCache.

Background — GLM-5.1 requires 16 H100 GPUs, typically deployed as
TP=8 PP=2 or TP=4 PP=4.  The save_only_first_rank MLA optimisation
previously treated global rank 0 as the sole "first rank", causing
all PP stages beyond stage-0 to behave as passive receivers and never
store their (different) KV layers.  The fixes:
  - LMCacheMetadata.is_first_rank() is now PP-stage-aware when tp_size>1:
      returns True for every worker where global_rank % tp_size == 0,
      i.e. tp_rank == 0 within each PP stage.
  - TokenDatabase._make_key_by_hash() uses pp_rank (= worker_id // tp_size)
    as the key worker_id under save_only_first_rank so that each PP stage
    gets a unique, stable cache key.
  - The broadcast sender now uses local_worker_id (local GPU device index)
    instead of the global rank to avoid OOB device indices on multi-node.
  - RemoteBackend._mla_worker_id_as0_mode now uses `not is_first_rank()`
    instead of `worker_id != 0` so that the first TP rank of each PP stage
    (e.g., global rank 8 in TP=8 PP=2) correctly stores its KV layers to
    the FS backend rather than silently dropping all writes.
"""

# Standard
import torch
import pytest
from unittest import mock

# First Party
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase


def _make_metadata(
    global_rank: int,
    global_world_size: int,
    tp_size: int,
    pp_size: int,
    local_worker_id: int | None = None,
) -> LMCacheMetadata:
    """Helper: build LMCacheMetadata mimicking vllm_service_factory output."""
    if local_worker_id is None:
        local_worker_id = global_rank % tp_size  # = tp_rank
    local_world_size = tp_size
    return LMCacheMetadata(
        model_name="test_model",
        world_size=global_world_size,
        local_world_size=local_world_size,
        worker_id=global_rank,
        local_worker_id=local_worker_id,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 16, 8, 128),
        use_mla=True,
        tp_size=tp_size,
    )


class TestIsFirstRankPPAware:
    """LMCacheMetadata.is_first_rank() must be PP-stage-aware when tp_size>1."""

    def test_tp1_pp1_legacy_only_rank0_is_first(self):
        """tp_size=1 → legacy path, only global rank 0 is first."""
        meta0 = _make_metadata(0, 1, tp_size=1, pp_size=1)
        assert meta0.is_first_rank() is True

    def test_tp4_pp1_only_rank0_is_first(self):
        """TP=4 PP=1: only global rank 0 is first (tp_rank==0 in the sole PP stage)."""
        for rank in range(4):
            meta = _make_metadata(rank, 4, tp_size=4, pp_size=1)
            assert meta.is_first_rank() == (rank == 0), (
                f"rank={rank}: expected is_first_rank={rank == 0}"
            )

    def test_tp4_pp2_each_pp_stage_has_first_rank(self):
        """TP=4 PP=2 (8 total): ranks 0 and 4 are first ranks (tp_rank==0)."""
        # PP stage 0: global ranks 0,1,2,3
        # PP stage 1: global ranks 4,5,6,7
        expected = {0: True, 1: False, 2: False, 3: False,
                    4: True, 5: False, 6: False, 7: False}
        for rank, exp in expected.items():
            meta = _make_metadata(rank, 8, tp_size=4, pp_size=2)
            assert meta.is_first_rank() == exp, (
                f"rank={rank}: expected is_first_rank={exp}, got {meta.is_first_rank()}"
            )

    def test_tp8_pp2_16_gpus(self):
        """TP=8 PP=2 (16 total, GLM-5.1 scenario): ranks 0 and 8 are first."""
        first_ranks = {0, 8}
        for rank in range(16):
            meta = _make_metadata(rank, 16, tp_size=8, pp_size=2)
            assert meta.is_first_rank() == (rank in first_ranks), (
                f"rank={rank}: expected is_first_rank={rank in first_ranks}"
            )

    def test_tp4_pp4_16_gpus(self):
        """TP=4 PP=4 (16 total): ranks 0,4,8,12 are first (one per PP stage)."""
        first_ranks = {0, 4, 8, 12}
        for rank in range(16):
            meta = _make_metadata(rank, 16, tp_size=4, pp_size=4)
            assert meta.is_first_rank() == (rank in first_ranks), (
                f"rank={rank}: expected is_first_rank={rank in first_ranks}"
            )

    def test_tp1_default_legacy_behaviour(self):
        """Default tp_size=1 (not explicitly set) preserves legacy is_first_rank."""
        # Without tp_size, is_first_rank() = worker_id == first_rank (== 0).
        meta_rank0 = LMCacheMetadata(
            model_name="m", world_size=4, local_world_size=4,
            worker_id=0, local_worker_id=0,
            kv_dtype=torch.bfloat16, kv_shape=(28, 2, 16, 8, 128), use_mla=True,
        )
        meta_rank1 = LMCacheMetadata(
            model_name="m", world_size=4, local_world_size=4,
            worker_id=1, local_worker_id=1,
            kv_dtype=torch.bfloat16, kv_shape=(28, 2, 16, 8, 128), use_mla=True,
        )
        assert meta_rank0.is_first_rank() is True
        assert meta_rank1.is_first_rank() is False


class TestCacheKeyPPStageUniqueness:
    """TokenDatabase must generate unique keys per PP stage under save_only_first_rank."""

    def _make_db(self, global_rank: int, world_size: int, tp_size: int) -> ChunkedTokenDatabase:
        meta = _make_metadata(global_rank, world_size, tp_size, pp_size=world_size // tp_size)
        config = LMCacheEngineConfig.from_legacy(chunk_size=16)
        db = ChunkedTokenDatabase(config, meta)
        db.save_only_first_rank = True
        return db

    def test_pp1_tp4_all_ranks_same_key(self):
        """PP=1, TP=4: first rank and all other ranks produce same pp_rank=0 key."""
        # save_only_first_rank: world_size=1, worker_id=pp_rank=0 for all
        dbs = [self._make_db(rank, 4, tp_size=4) for rank in range(4)]
        keys = [db._make_key_by_hash(chunk_hash=42) for db in dbs]
        # All keys should have the same worker_id (pp_rank=0) and world_size=1
        for key in keys:
            assert key.world_size == 1, f"Expected world_size=1, got {key.world_size}"
            assert key.worker_id == 0, f"Expected worker_id=0 (pp_rank), got {key.worker_id}"

    def test_pp2_tp4_each_stage_unique_key(self):
        """PP=2, TP=4 (8 total): PP stage 0 gets pp_rank=0, stage 1 gets pp_rank=1."""
        # First rank of each PP stage
        db_pp0 = self._make_db(global_rank=0, world_size=8, tp_size=4)
        db_pp1 = self._make_db(global_rank=4, world_size=8, tp_size=4)
        key_pp0 = db_pp0._make_key_by_hash(chunk_hash=99)
        key_pp1 = db_pp1._make_key_by_hash(chunk_hash=99)
        # Same chunk, same world_size=1, but different pp_rank → different keys
        assert key_pp0.world_size == 1
        assert key_pp1.world_size == 1
        assert key_pp0.worker_id == 0, f"PP stage 0 key worker_id should be 0 (pp_rank)"
        assert key_pp1.worker_id == 1, f"PP stage 1 key worker_id should be 1 (pp_rank)"
        assert key_pp0 != key_pp1

    def test_pp4_tp4_16gpus_unique_keys(self):
        """TP=4 PP=4 (16 total): each PP stage gets a unique pp_rank key."""
        first_ranks = [0, 4, 8, 12]
        keys = []
        for rank in first_ranks:
            db = self._make_db(global_rank=rank, world_size=16, tp_size=4)
            keys.append(db._make_key_by_hash(chunk_hash=77))
        # All keys must be unique
        assert len(set(str(k) for k in keys)) == 4, "Expected 4 distinct keys for 4 PP stages"
        # worker_ids must be consecutive pp_ranks: 0,1,2,3
        worker_ids = sorted(k.worker_id for k in keys)
        assert worker_ids == [0, 1, 2, 3]

    def test_key_stable_across_tp_size_changes(self):
        """pp_rank-based key is stable: PP stage 1 yields worker_id=1 for TP=4 or TP=8."""
        # TP=4, PP=2: PP stage 1 first rank = global rank 4
        db_tp4 = self._make_db(global_rank=4, world_size=8, tp_size=4)
        key_tp4 = db_tp4._make_key_by_hash(chunk_hash=55)
        # TP=8, PP=2: PP stage 1 first rank = global rank 8
        db_tp8 = self._make_db(global_rank=8, world_size=16, tp_size=8)
        key_tp8 = db_tp8._make_key_by_hash(chunk_hash=55)
        # Both should have worker_id=1 (pp_rank=1) regardless of tp_size
        assert key_tp4.worker_id == 1, f"TP=4: expected pp_rank=1, got {key_tp4.worker_id}"
        assert key_tp8.worker_id == 1, f"TP=8: expected pp_rank=1, got {key_tp8.worker_id}"

    def test_non_first_ranks_produce_same_key_as_first_rank(self):
        """Passive TP ranks in a PP stage produce the same cache key as the first rank."""
        # PP=2, TP=4: PP stage 1 first rank=4, passive ranks=5,6,7
        # All should produce the same key (pp_rank=1, world_size=1)
        db_first = self._make_db(global_rank=4, world_size=8, tp_size=4)
        db_passive = self._make_db(global_rank=5, world_size=8, tp_size=4)
        key_first = db_first._make_key_by_hash(chunk_hash=33)
        key_passive = db_passive._make_key_by_hash(chunk_hash=33)
        assert key_first == key_passive, (
            "First and passive ranks in the same PP stage should share the same cache key"
        )


class TestIsPassivePPBehaviour:
    """_is_passive() must be False for first TP rank in every PP stage."""

    def _make_engine_like_passive_check(
        self, global_rank: int, world_size: int, tp_size: int
    ) -> bool:
        """Replicate cache_engine._is_passive() logic without a full engine."""
        meta = _make_metadata(global_rank, world_size, tp_size, pp_size=world_size // tp_size)
        # _is_passive = save_only_first_rank and not is_first_rank
        return not meta.is_first_rank()

    def test_tp8_pp2_first_ranks_not_passive(self):
        """TP=8 PP=2: ranks 0 and 8 (first TP rank of each PP stage) must NOT be passive."""
        assert not self._make_engine_like_passive_check(0, 16, 8), "rank 0 should not be passive"
        assert not self._make_engine_like_passive_check(8, 16, 8), "rank 8 (PP stage 1) should not be passive"

    def test_tp8_pp2_non_first_ranks_are_passive(self):
        """TP=8 PP=2: ranks 1-7 and 9-15 (non-first in their PP stage) must be passive."""
        for r in [1, 2, 3, 4, 5, 6, 7]:
            assert self._make_engine_like_passive_check(r, 16, 8), f"rank {r} should be passive"
        for r in [9, 10, 11, 12, 13, 14, 15]:
            assert self._make_engine_like_passive_check(r, 16, 8), f"rank {r} should be passive"


class TestRemoteBackendMlaWorkerIdAs0ModePP:
    """RemoteBackend._mla_worker_id_as0_mode must use is_first_rank() (PP-aware)
    rather than worker_id != 0, so that the first TP rank of each PP stage
    (e.g., global rank 8 in TP=8 PP=2) is treated as an active writer.

    Regression test for: PP-stage-1 first rank silently skipping all FS writes
    and reading stale PP-stage-0 data on vLLM restart.
    """

    def _make_backend_mla_mode(
        self,
        global_rank: int,
        world_size: int,
        tp_size: int,
        save_only_first_rank: bool = True,
    ) -> bool:
        """Return _mla_worker_id_as0_mode for the given rank config."""
        import asyncio
        import threading
        from lmcache.v1.storage_backend.remote_backend import RemoteBackend
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
        from lmcache.v1.memory_management import AdHocMemoryAllocator

        config = LMCacheEngineConfig(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=1.0,
            local_disk=None,
            max_local_disk_size=0.0,
            remote_url="lm://localhost:65432",
            remote_serde="naive",
            use_layerwise=False,
            save_decode_cache=False,
            enable_blending=False,
            extra_config={"save_only_first_rank": save_only_first_rank},
        )
        meta = _make_metadata(global_rank, world_size, tp_size, pp_size=world_size // tp_size)
        allocator = AdHocMemoryAllocator()
        cpu_backend = LocalCPUBackend(config, meta, memory_allocator=allocator)
        loop = asyncio.new_event_loop()

        with mock.patch("torch.cuda.Stream"):
            backend = RemoteBackend(config=config, metadata=meta, loop=loop,
                                    local_cpu_backend=cpu_backend)

        mode = backend._mla_worker_id_as0_mode
        loop.close()
        return mode

    def test_tp8_pp2_rank0_not_in_mla_mode(self):
        """Global rank 0 (PP0 first rank) must NOT be in _mla_worker_id_as0_mode."""
        assert not self._make_backend_mla_mode(0, 16, 8), (
            "rank 0 (PP0 first rank) should never be in _mla_worker_id_as0_mode"
        )

    def test_tp8_pp2_rank8_not_in_mla_mode(self):
        """Global rank 8 (PP1 first rank, TP8 PP2) must NOT be in _mla_worker_id_as0_mode.

        This is the key regression: with worker_id!=0, rank 8 was incorrectly
        placed in _mla_worker_id_as0_mode, silently dropping all its FS writes
        and causing garbled output after vLLM restart.
        """
        assert not self._make_backend_mla_mode(8, 16, 8), (
            "rank 8 (PP1 first rank) must NOT be in _mla_worker_id_as0_mode — "
            "it needs to independently store its PP-stage-1 KV layers"
        )

    def test_tp8_pp2_passive_ranks_are_in_mla_mode(self):
        """Non-first-rank workers in each PP stage are passive and use _mla_worker_id_as0_mode."""
        for r in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]:
            assert self._make_backend_mla_mode(r, 16, 8), (
                f"rank {r} (passive worker) should be in _mla_worker_id_as0_mode"
            )

    def test_tp4_pp4_first_ranks_not_in_mla_mode(self):
        """TP=4 PP=4 (16 total): ranks 0,4,8,12 (first rank of each PP stage) must NOT be in mode."""
        for r in [0, 4, 8, 12]:
            assert not self._make_backend_mla_mode(r, 16, 4), (
                f"rank {r} (PP first rank) must NOT be in _mla_worker_id_as0_mode"
            )

    def test_save_only_first_rank_false_disables_mode_for_all(self):
        """save_only_first_rank=False: no worker should be in _mla_worker_id_as0_mode."""
        for r in range(8):
            assert not self._make_backend_mla_mode(r, 8, 4, save_only_first_rank=False), (
                f"rank {r}: save_only_first_rank=False must disable _mla_worker_id_as0_mode"
            )

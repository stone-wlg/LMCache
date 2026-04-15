# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List

# Add import for mock
from unittest import mock
import asyncio
import threading

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class MockConnector(RemoteConnector):
    def __init__(self):
        self.storage = {}

    async def exists(self, key):
        return key in self.storage

    def exists_sync(self, key):
        return key in self.storage

    async def put(self, key, value):
        self.storage[key] = value

    async def get(self, key):
        return self.storage.get(key)

    async def close(self):
        pass

    async def list(self) -> List[str]:
        return []


# Mock the entire torch.cuda.Stream class
@mock.patch("torch.cuda.Stream")
def test_remote_mla_worker_id_as0(mock_stream):
    # Create configuration
    config = LMCacheEngineConfig(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=5.0,
        local_disk=None,
        max_local_disk_size=0.0,
        remote_url="lm://localhost:65432",
        remote_serde="naive",
        use_layerwise=False,
        save_decode_cache=False,
        enable_blending=False,
        extra_config={"remote_enable_mla_worker_id_as0": True},
    )

    metadata = LMCacheMetadata(
        model_name="test-model",
        kv_dtype=torch.float16,
        kv_shape=(32, 1, 256, 64, 128),
        use_mla=True,
        world_size=4,
        local_world_size=4,
        worker_id=2,
        local_worker_id=2,
    )
    metadata0 = LMCacheMetadata(
        model_name="test-model",
        kv_dtype=torch.float16,
        kv_shape=(32, 1, 256, 64, 128),
        use_mla=True,
        world_size=4,
        local_world_size=4,
        worker_id=0,
        local_worker_id=0,
    )

    # Create memory allocator and local backend
    # First Party
    from lmcache.v1.memory_management import AdHocMemoryAllocator

    pin_allocator = AdHocMemoryAllocator()
    local_cpu_backend = LocalCPUBackend(
        config, metadata, memory_allocator=pin_allocator
    )

    loop = asyncio.new_event_loop()
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
    )
    backend.connection = MockConnector()

    # Start the event loop in a separate thread
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    # Create key
    key = CacheEngineKey(
        model_name="test-model",
        world_size=4,
        worker_id=2,
        chunk_hash="test_hash",
        dtype=torch.float32,
    )

    local_cpu_backend0 = LocalCPUBackend(
        config, metadata0, memory_allocator=pin_allocator
    )
    backend0 = RemoteBackend(
        config=config,
        metadata=metadata0,
        loop=loop,
        local_cpu_backend=local_cpu_backend0,
    )
    backend0.connection = backend.connection
    # Create key
    key0 = CacheEngineKey(
        model_name="test-model",
        world_size=4,
        worker_id=0,
        chunk_hash="test_hash",
        dtype=torch.float32,
    )

    # Test not contains before adding data
    assert not backend.contains(key)
    assert not backend0.contains(key0)

    # Test submit_put_task
    memory_obj = local_cpu_backend.allocate(torch.Size([10, 10]), torch.float32)
    future = backend.submit_put_task(key, memory_obj)
    # Wait for put task to complete
    if future is not None:
        future.result()

    # Test not contains after adding data since worker_id 2 skipped put
    assert not backend.contains(key)

    future = backend0.submit_put_task(key0, memory_obj)
    # Wait for put task to complete
    if future is not None:
        future.result()

    # Test contains after adding data since worker_id 0 should put
    assert backend0.contains(key0)
    # Test contains after adding data since we use worker_id 0 instead
    assert backend.contains(key)

    # Test get_blocking
    retrieved = backend.get_blocking(key)
    assert retrieved is not None
    assert retrieved.get_shape() == torch.Size([10, 10])

    # Cleanup
    async def shutdown():
        # Get all tasks
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        # Wait for all tasks to be cancelled (or completed)
        await asyncio.gather(*tasks, return_exceptions=True)
        # Then stop the loop
        loop.stop()

    # Schedule the shutdown coroutine in the event loop thread
    future = asyncio.run_coroutine_threadsafe(shutdown(), loop)
    try:
        # Wait for the shutdown to complete, but with a timeout
        future.result(timeout=10)
    except Exception as e:
        print(f"Error during shutdown: {e}")
    finally:
        # Wait for the loop thread to finish
        loop_thread.join(timeout=1.0)
        loop.close()


def _make_loop_and_backend(
    extra_config: dict,
    worker_id: int,
    world_size: int = 4,
):
    """Helper: build a RemoteBackend with a MockConnector for the given config."""
    config = LMCacheEngineConfig(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=5.0,
        local_disk=None,
        max_local_disk_size=0.0,
        remote_url="lm://localhost:65432",
        remote_serde="naive",
        use_layerwise=False,
        save_decode_cache=False,
        enable_blending=False,
        extra_config=extra_config,
    )
    metadata = LMCacheMetadata(
        model_name="test-model",
        kv_dtype=torch.float16,
        kv_shape=(32, 1, 256, 64, 128),
        use_mla=True,
        world_size=world_size,
        local_world_size=world_size,
        worker_id=worker_id,
        local_worker_id=worker_id,
    )
    # First Party
    from lmcache.v1.memory_management import AdHocMemoryAllocator

    allocator = AdHocMemoryAllocator()
    local_cpu_backend = LocalCPUBackend(config, metadata, memory_allocator=allocator)
    loop = asyncio.new_event_loop()
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
    )
    backend.connection = MockConnector()
    return loop, backend, local_cpu_backend


def _run_loop(loop):
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    return loop_thread


def _stop_loop(loop, loop_thread):
    async def _shutdown():
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
    try:
        future.result(timeout=10)
    except Exception:
        pass
    loop_thread.join(timeout=1.0)
    loop.close()


@mock.patch("torch.cuda.Stream")
def test_save_only_first_rank_false_disables_mla_worker_id_as0_mode(mock_stream):
    """When save_only_first_rank=False, non-zero workers must NOT activate
    _mla_worker_id_as0_mode.  Each worker should independently write its own
    data to the remote backend (identified by its own worker_id) and retrieve
    it back under its own key — not under worker_id=0.

    This is the regression test for the garbled-output-after-vLLM-restart bug
    where the default _mla_worker_id_as0_mode=use_mla caused workers 1-N to
    skip writing and read the wrong (worker 0's) KV shard from the FS backend.
    """
    # --- worker 1 with save_only_first_rank=False ---
    extra_cfg = {"save_only_first_rank": False}
    loop, backend1, cpu1 = _make_loop_and_backend(extra_cfg, worker_id=1)
    loop_thread = _run_loop(loop)

    # _mla_worker_id_as0_mode must be False
    assert not backend1._mla_worker_id_as0_mode, (
        "Expected _mla_worker_id_as0_mode=False when save_only_first_rank=False"
    )

    key1 = CacheEngineKey(
        model_name="test-model",
        world_size=4,
        worker_id=1,
        chunk_hash="hash_w1",
        dtype=torch.float32,
    )

    # Worker 1 should be able to write its own data
    mem_obj = cpu1.allocate(torch.Size([10, 10]), torch.float32)
    future = backend1.submit_put_task(key1, mem_obj)
    if future is not None:
        future.result(timeout=5)

    # Worker 1's data should be present under its own key (not worker_id=0)
    assert backend1.contains(key1), (
        "Worker 1 should have written its own data when save_only_first_rank=False"
    )

    # And worker 1 should retrieve its own data
    retrieved = backend1.get_blocking(key1)
    assert retrieved is not None, (
        "Worker 1 should retrieve its own cached data when save_only_first_rank=False"
    )

    _stop_loop(loop, loop_thread)


@mock.patch("torch.cuda.Stream")
def test_save_only_first_rank_true_enables_mla_worker_id_as0_mode(mock_stream):
    """When save_only_first_rank=True (the default for MLA models), non-zero
    workers SHOULD activate _mla_worker_id_as0_mode: skip writes and look up
    data under worker_id=0.
    """
    extra_cfg = {"save_only_first_rank": True}
    loop, backend2, cpu2 = _make_loop_and_backend(extra_cfg, worker_id=2)
    loop0, backend0, cpu0 = _make_loop_and_backend(extra_cfg, worker_id=0)
    # Share the same MockConnector so writes by worker 0 are visible to worker 2
    backend2.connection = backend0.connection

    loop_thread = _run_loop(loop)
    loop0_thread = _run_loop(loop0)

    # Worker 2 should be in _mla_worker_id_as0_mode
    assert backend2._mla_worker_id_as0_mode, (
        "Expected _mla_worker_id_as0_mode=True for non-zero worker when "
        "save_only_first_rank=True"
    )
    # Worker 0 should NOT be in _mla_worker_id_as0_mode (it is the writer)
    assert not backend0._mla_worker_id_as0_mode

    key0 = CacheEngineKey(
        model_name="test-model",
        world_size=4,
        worker_id=0,
        chunk_hash="hash_mla",
        dtype=torch.float32,
    )

    # Worker 0 stores data
    mem_obj = cpu0.allocate(torch.Size([8, 8]), torch.float32)
    future = backend0.submit_put_task(key0, mem_obj)
    if future is not None:
        future.result(timeout=5)

    # Worker 2 should find it (via worker_id=0 rewrite)
    key2 = CacheEngineKey(
        model_name="test-model",
        world_size=4,
        worker_id=2,
        chunk_hash="hash_mla",
        dtype=torch.float32,
    )
    assert backend2.contains(key2), (
        "Worker 2 should find worker 0's data via _mla_worker_id_as0_mode"
    )

    _stop_loop(loop, loop_thread)
    _stop_loop(loop0, loop0_thread)


@mock.patch("torch.cuda.Stream")
def test_default_mla_model_uses_save_only_first_rank_as_default(mock_stream):
    """Without explicit extra_config, default save_only_first_rank=use_mla=True
    for MLA models, so the _mla_worker_id_as0_mode default also follows that.
    Non-zero workers in MLA world_size>1 with no explicit config should have
    _mla_worker_id_as0_mode=True.
    """
    # No extra_config at all → defaults apply
    _, backend_default, _ = _make_loop_and_backend(
        extra_config={}, worker_id=3, world_size=4
    )
    # Default: save_only_first_rank = use_mla = True → _mla_worker_id_as0_mode = True
    assert backend_default._mla_worker_id_as0_mode, (
        "Default MLA model with world_size>1 and worker_id!=0 should have "
        "_mla_worker_id_as0_mode=True (backward-compatible default)"
    )

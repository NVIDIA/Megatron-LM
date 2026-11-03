# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for shared checkpoint metadata reads."""

import gc
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.distributed.checkpoint import BytesStorageMetadata, FileSystemReader, Metadata

from megatron.core.dist_checkpointing import (
    ShardedTensor,
    load,
    load_common_state_dict,
    load_tensors_metadata,
    save,
)
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
from megatron.core.dist_checkpointing.strategies.cached_metadata_filesystem_reader import (
    CachedMetadataFileSystemReader,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    _get_filesystem_reader,
)
from megatron.core.msc_utils import MultiStorageClientFeature
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def clear_reader_cache():
    """Keep test files and metadata references isolated between tests."""
    CachedMetadataFileSystemReader.clear_metadata_cache()
    yield
    CachedMetadataFileSystemReader.clear_metadata_cache()


def _write_metadata(directory, label="first", filename=".metadata"):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = Metadata(
        state_dict_metadata={label: BytesStorageMetadata()}, planner_data={}, storage_data={}
    )
    metadata.all_local_plans = []
    path = directory / filename
    path.write_bytes(pickle.dumps(metadata))
    return path


@pytest.fixture
def base_reads():
    """Count actual deserializations without replacing the filesystem implementation."""
    with mock.patch.object(
        FileSystemReader, "read_metadata", autospec=True, side_effect=FileSystemReader.read_metadata
    ) as reads:
        yield reads


def test_metadata_copies_preserve_reader_identity(tmp_path, base_reads):
    """Reader IDs and save-side attribute replacement cannot change other readers' metadata."""
    _write_metadata(tmp_path)
    first_reader = _get_filesystem_reader(tmp_path, cache_metadata=False)
    second_reader = _get_filesystem_reader(str(tmp_path), cache_metadata=True)
    first = first_reader.read_metadata()
    second = second_reader.read_metadata()
    assert base_reads.call_count == 1
    assert first is not second
    assert first.state_dict_metadata is second.state_dict_metadata
    assert first.all_local_plans is second.all_local_plans
    assert first.storage_meta.load_id == first_reader.load_id
    assert second.storage_meta.load_id == second_reader.load_id
    first.storage_data = {"replacement": None}
    first.storage_meta.load_id = "modified"
    again = second_reader.read_metadata()
    assert again.storage_data == {}
    assert again.storage_meta.load_id == second_reader.load_id


def test_replacement_with_same_size_and_mtime(tmp_path, base_reads):
    """Atomic replacement must invalidate even when size and mtime have been preserved."""
    path = _write_metadata(tmp_path, "first")
    reader = _get_filesystem_reader(tmp_path)
    assert "first" in reader.read_metadata().state_dict_metadata
    original_stat = path.stat()
    replacement = _write_metadata(tmp_path, "other", filename=".metadata.new")
    assert replacement.stat().st_size == original_stat.st_size
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replacement.replace(path)
    assert "other" in reader.read_metadata().state_dict_metadata
    assert base_reads.call_count == 2


def test_file_changed_during_read_is_not_cached(tmp_path):
    """Do not associate a read with an identity that changed during deserialization."""
    _write_metadata(tmp_path, "first")
    original = FileSystemReader.read_metadata

    def read_and_replace(reader):
        metadata = original(reader)
        # Use atomic replacement so invalidation does not depend on timestamp resolution.
        _write_metadata(tmp_path, "other", filename=".metadata.new").replace(tmp_path / ".metadata")
        return metadata

    with mock.patch.object(FileSystemReader, "read_metadata", read_and_replace):
        assert "first" in _get_filesystem_reader(tmp_path).read_metadata().state_dict_metadata
    assert "other" in _get_filesystem_reader(tmp_path).read_metadata().state_dict_metadata


def test_cache_is_bounded_and_can_be_cleared_by_directory(tmp_path, base_reads):
    """Switching checkpoints evicts the old entry; clearing another directory is harmless."""
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    _write_metadata(first_dir)
    _write_metadata(second_dir)
    first, second = map(_get_filesystem_reader, (first_dir, second_dir))
    first.read_metadata()
    CachedMetadataFileSystemReader.clear_metadata_cache(second_dir)
    first.read_metadata()
    assert base_reads.call_count == 1
    second.read_metadata()
    first.read_metadata()
    assert base_reads.call_count == 3
    CachedMetadataFileSystemReader.clear_metadata_cache(first_dir)
    first.read_metadata()
    assert base_reads.call_count == 4
    first.reset(second_dir)
    first.read_metadata()
    assert base_reads.call_count == 5


@pytest.mark.skipif(
    not hasattr(FileSystemReader, "_get_metadata_path"), reason="Requires rank-local metadata API"
)
def test_rank_local_metadata(tmp_path, base_reads):
    """Different rank-local metadata files must not alias the directory's cache entry."""
    _write_metadata(tmp_path, "first", "__0.metadata")
    _write_metadata(tmp_path, "other", "__1.metadata")
    reader = _get_filesystem_reader(tmp_path)
    assert "first" in reader.read_metadata(rank=0).state_dict_metadata
    assert "first" in reader.read_metadata(rank=0).state_dict_metadata
    assert base_reads.call_count == 1
    assert "other" in reader.read_metadata(rank=1).state_dict_metadata
    CachedMetadataFileSystemReader.clear_metadata_cache(tmp_path)
    reader.read_metadata(rank=1)
    assert base_reads.call_count == 3


def test_concurrent_readers(tmp_path, base_reads):
    """Concurrent readers deserialize once and retain separate load identities."""
    _write_metadata(tmp_path)
    readers = [_get_filesystem_reader(tmp_path) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        metadatas = list(pool.map(lambda reader: reader.read_metadata(), readers))
    assert base_reads.call_count == 1
    for reader, metadata in zip(readers, metadatas):
        assert metadata.storage_meta.load_id == reader.load_id


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_gc_restored_on_success_and_failure(tmp_path, enabled, fail):
    """Restore the caller's GC setting, including when unpickling raises."""
    _write_metadata(tmp_path)
    original = FileSystemReader.read_metadata

    def read(reader):
        assert not gc.isenabled()
        if fail:
            raise pickle.UnpicklingError("invalid metadata")
        return original(reader)

    previous = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        with mock.patch.object(FileSystemReader, "read_metadata", read):
            if fail:
                with pytest.raises(pickle.UnpicklingError):
                    _get_filesystem_reader(tmp_path).read_metadata()
            else:
                _get_filesystem_reader(tmp_path).read_metadata()
        assert gc.isenabled() == enabled
    finally:
        (gc.enable if previous else gc.disable)()


@pytest.mark.parametrize("retain", [False, True])
def test_training_load_failure_releases_transient_metadata(
    tmp_path, monkeypatch, retain, base_reads
):
    """Even a failure before the tensor-load phase respects the cache-retention setting."""
    from megatron.training import checkpointing

    _write_metadata(tmp_path)

    def fail_after_read():
        _get_filesystem_reader(tmp_path).read_metadata()
        raise RuntimeError("load interrupted")

    monkeypatch.setattr(
        checkpointing, "get_args", lambda: SimpleNamespace(ckpt_assume_constant_structure=retain)
    )
    monkeypatch.setattr(checkpointing, "get_run_config", fail_after_read)
    with pytest.raises(RuntimeError, match="load interrupted"):
        checkpointing.load_checkpoint([], None, None)
    _get_filesystem_reader(tmp_path).read_metadata()
    assert base_reads.call_count == (1 if retain else 2)


def test_worker_reset_releases_metadata(tmp_path, monkeypatch, base_reads):
    """Worker reset also releases the metadata cache owned by MCore."""
    pytest.importorskip(
        "nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader"
    )
    from megatron.training import async_utils

    _write_metadata(tmp_path)
    _get_filesystem_reader(tmp_path).read_metadata()
    monkeypatch.setattr(async_utils, "_async_calls_queue", None)
    monkeypatch.setattr(async_utils, "_results_queue", None)
    async_utils.reset_persistent_async_worker()
    _get_filesystem_reader(tmp_path).read_metadata()
    assert base_reads.call_count == 2


def test_removing_shards_invalidates_metadata(tmp_path, base_reads):
    """Metadata rewrites performed by the removal API invalidate the shared entry."""
    _write_metadata(tmp_path, "remove")
    (tmp_path / "remove_0.distcp").touch()
    _get_filesystem_reader(tmp_path).read_metadata()
    TorchDistLoadShardedStrategy().remove_sharded_tensors(str(tmp_path), "remove")
    assert _get_filesystem_reader(tmp_path).read_metadata().state_dict_metadata == {}
    assert base_reads.call_count == 3  # cached read, removal's direct read, read after invalidation


def test_multi_storage_reader_is_unchanged(monkeypatch):
    """The local-filesystem cache must not intercept remote-storage paths."""
    factory = mock.Mock()
    monkeypatch.setattr(MultiStorageClientFeature, "is_enabled", lambda: True)
    monkeypatch.setattr(
        MultiStorageClientFeature,
        "import_package",
        lambda: SimpleNamespace(torch=SimpleNamespace(MultiStorageFileSystemReader=factory)),
    )
    assert _get_filesystem_reader("msc://checkpoint") is factory.return_value
    factory.assert_called_once_with("msc://checkpoint", thread_count=2)


class TestMetadataCacheLoad:
    """Validate the read sequence against real distributed checkpoints."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def state_dict(fill):
        """A tensor shard and common state, with distinct values on each rank."""
        return {
            "weight": ShardedTensor.from_rank_offsets(
                "weight",
                torch.full((4, 3), fill + Utils.rank, device="cuda"),
                (0, Utils.rank, Utils.world_size),
            ),
            "common": {"iteration": 42},
        }

    @pytest.mark.parametrize("cache_metadata", [False, True])
    @pytest.mark.parametrize("fully_parallel", [False, True])
    def test_load_phases_share_one_read(
        self, tmp_path_dist_ckpt, base_reads, cache_metadata, fully_parallel
    ):
        """Common-state, metadata queries, and tensor loading share a single deserialization."""
        with TempNamedDir(tmp_path_dist_ckpt / "shared_metadata") as checkpoint_dir:
            save(self.state_dict(3.0), checkpoint_dir)
            strategy = TorchDistLoadShardedStrategy(cache_metadata=cache_metadata)
            load_strategy = (
                FullyParallelLoadStrategyWrapper(strategy, per_rank_object_load=True)
                if fully_parallel
                else strategy
            )
            assert load_common_state_dict(checkpoint_dir)["common"] == {"iteration": 42}
            assert load_tensors_metadata(checkpoint_dir)["weight"].global_shape == (
                4 * Utils.world_size,
                3,
            )
            assert "weight" in load_sharded_metadata(checkpoint_dir)
            loaded = load(self.state_dict(0.0), checkpoint_dir, load_strategy)
            assert torch.equal(loaded["weight"], self.state_dict(3.0)["weight"].data)
            assert loaded["common"] == {"iteration": 42}
            assert base_reads.call_count == 1
            if cache_metadata:
                strategy.cached_global_metadata.storage_data = {}
                # Save-side metadata replacement must not corrupt subsequent loads.
                loaded = load(self.state_dict(0.0), checkpoint_dir, load_strategy)
                assert torch.equal(loaded["weight"], self.state_dict(3.0)["weight"].data)
                assert base_reads.call_count == 1

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for shared checkpoint metadata reads."""

import pickle
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
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    CachedMetadataFileSystemReader,
    TorchDistLoadShardedStrategy,
    _get_filesystem_reader,
)
from megatron.core.msc_utils import MultiStorageClientFeature
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def clear_reader_cache():
    """Keep test files and metadata references isolated between tests."""
    if CachedMetadataFileSystemReader is not None:
        CachedMetadataFileSystemReader.clear_metadata_cache()
    yield
    if CachedMetadataFileSystemReader is not None:
        CachedMetadataFileSystemReader.clear_metadata_cache()


def _write_metadata(directory, label="first", filename=".metadata"):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = Metadata(
        state_dict_metadata={label: BytesStorageMetadata()}, planner_data={}, storage_data={}
    )
    path = directory / filename
    with path.open("wb") as metadata_file:
        pickle.dump(metadata, metadata_file)
    return path


@pytest.fixture
def base_reads():
    """Count actual deserializations without replacing the filesystem implementation."""
    with mock.patch.object(
        FileSystemReader, "read_metadata", autospec=True, side_effect=FileSystemReader.read_metadata
    ) as reads:
        yield reads


def test_reader_without_nvrx(tmp_path, monkeypatch):
    """Synchronous reads remain supported when the optional dependency is unavailable."""
    from megatron.core.dist_checkpointing.strategies import torch as strategy_module

    _write_metadata(tmp_path)
    monkeypatch.setattr(strategy_module, "CachedMetadataFileSystemReader", None)
    reader = _get_filesystem_reader(tmp_path)
    assert type(reader) is FileSystemReader
    assert "first" in reader.read_metadata().state_dict_metadata


@pytest.mark.skipif(CachedMetadataFileSystemReader is None, reason="Requires NVRx")
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("outcome", ["error", "empty"])
def test_training_load_releases_transient_metadata(
    tmp_path, monkeypatch, retain, outcome, base_reads
):
    """Exceptions and early returns after metadata reads respect cache retention."""
    from megatron.training import checkpointing

    _write_metadata(tmp_path)

    def read_checkpoint(*args, **kwargs):
        _get_filesystem_reader(tmp_path).read_metadata()
        if outcome == "error":
            raise RuntimeError("load interrupted")
        return None, "", False, None

    monkeypatch.setattr(
        checkpointing,
        "get_args",
        lambda: SimpleNamespace(
            ckpt_assume_constant_structure=retain,
            load=str(tmp_path),
            ckpt_format="torch_dist",
            auto_detect_ckpt_format=True,
        ),
    )
    monkeypatch.setattr(checkpointing, "_load_base_checkpoint", read_checkpoint)
    if outcome == "error":
        with pytest.raises(RuntimeError, match="load interrupted"):
            checkpointing.load_checkpoint([], None, None)
    else:
        assert checkpointing.load_checkpoint([], None, None) == (0, 0)
    _get_filesystem_reader(tmp_path).read_metadata()
    assert base_reads.call_count == (1 if retain else 2)


def test_worker_reset_releases_metadata(tmp_path, monkeypatch, base_reads):
    """Worker reset releases the same NVRx cache used by load phases."""
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


@pytest.mark.skipif(CachedMetadataFileSystemReader is None, reason="Requires NVRx")
def test_removing_shards_invalidates_metadata(tmp_path, base_reads):
    """Metadata rewrites performed by the removal API invalidate the shared entry."""
    _write_metadata(tmp_path, "remove")
    (tmp_path / "remove_0.distcp").touch()
    _get_filesystem_reader(tmp_path).read_metadata()
    TorchDistLoadShardedStrategy().remove_sharded_tensors(str(tmp_path), "remove")
    assert _get_filesystem_reader(tmp_path).read_metadata().state_dict_metadata == {}
    assert base_reads.call_count == 3  # cached read, removal's direct read, read after invalidation


def test_multi_storage_backend_is_preserved(monkeypatch):
    """MSC keeps its own reader for path resolution and storage access."""
    factory = mock.Mock()
    monkeypatch.setattr(MultiStorageClientFeature, "is_enabled", lambda: True)
    monkeypatch.setattr(
        MultiStorageClientFeature,
        "import_package",
        lambda: SimpleNamespace(torch=SimpleNamespace(MultiStorageFileSystemReader=factory)),
    )
    assert _get_filesystem_reader("msc://checkpoint") is factory.return_value
    factory.assert_called_once_with("msc://checkpoint", thread_count=2)


@pytest.mark.skipif(CachedMetadataFileSystemReader is None, reason="Requires NVRx")
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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
import io
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import pytest
import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemBase

from megatron.core.dist_checkpointing import (
    ShardedTensor,
    check_is_distributed_checkpoint,
    load,
    load_common_state_dict,
    load_content_metadata,
    load_plain_tensors,
    load_tensors_metadata,
    remove_sharded_tensors,
    save,
)
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.filesystem import (
    make_filesystem_reader,
    make_filesystem_writer,
)
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
from megatron.core.dist_checkpointing.strategies import torch as torch_strategy
from megatron.core.dist_checkpointing.strategies.common import save_common
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from megatron.core.dist_checkpointing.validation import (
    save_integrity_manifest,
    verify_integrity_manifest,
)
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.utils import grant_shape_mismatch_for_gtp_padding
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class _NoFilenoStream:
    def __init__(self, stream, fail_close=False):
        self.stream = stream
        self.fail_close = fail_close

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def fileno(self):
        raise io.UnsupportedOperation("This checkpoint stream has no file descriptor")

    def close(self):
        if not self.stream.closed:
            self.stream.close()
            if self.fail_close:
                raise OSError("injected close failure")


class URIFileSystem(FileSystemBase):
    """Map URI paths to shared test storage while rejecting local-path escapes."""

    prefix = "test://checkpoints"

    def __init__(self, root, failure=None, failure_suffix=".distcp"):
        self.root = Path(root)
        self.failure = failure
        self.failure_suffix = failure_suffix
        self.accesses = []
        self.worker_failed = None

    def _path(self, path):
        assert isinstance(path, str) and path.startswith(self.prefix + "/"), path
        self.accesses.append(path)
        return self.root / path[len(self.prefix) + 1 :]

    def init_path(self, path):
        self._path(path)
        return path

    def concat_path(self, path, suffix):
        assert not PurePosixPath(suffix).is_absolute() and ".." not in suffix.split("/")
        return path.rstrip("/") + "/" + suffix

    @contextmanager
    def create_stream(self, path, mode):
        local_path = self._path(path)
        failing = mode == "wb" and path.endswith(self.failure_suffix)
        if failing and self.failure == "thread":
            if threading.current_thread() is not threading.main_thread():
                self.worker_failed.set()
                raise OSError("injected writer thread failure")
            assert self.worker_failed.wait(10), "DCP did not start its second I/O thread"
        stream = _NoFilenoStream(
            local_path.open(mode), fail_close=failing and self.failure == "close"
        )
        try:
            yield stream
        finally:
            stream.close()

    def mkdir(self, path):
        self._path(path).mkdir(parents=True, exist_ok=True)

    def stat(self, path):
        local_path = self._path(path)
        try:
            local_path.stat()
        except FileNotFoundError:
            return "missing"
        return "directory" if local_path.is_dir() else "file"

    def exists(self, path):
        return self.stat(path) != "missing"

    def listdir(self, path):
        if self.failure == "list":
            raise OSError("injected listing failure")
        return os.listdir(self._path(path))

    def rename(self, path, new_path):
        self._path(path).rename(self._path(new_path))

    def rm_file(self, path):
        self._path(path).unlink()

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id):
        return isinstance(checkpoint_id, str) and checkpoint_id.startswith(cls.prefix + "/")

    def sync(self, stream):
        stream.flush()
        if self.failure == "sync" and str(stream.name).endswith(self.failure_suffix):
            raise OSError("injected sync failure")


@pytest.fixture
def distributed_checkpoint():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _state(value=3):
    return {
        "weight": ShardedTensor.from_rank_offsets(
            "weight", torch.full((2, 3), value), replica_id=Utils.rank
        ),
        "step": 7,
    }


@pytest.mark.parametrize("fully_parallel", [False, True])
def test_injected_filesystem_roundtrip(
    tmp_path_dist_ckpt, distributed_checkpoint, monkeypatch, fully_parallel
):
    monkeypatch.setattr(MultiStorageClientFeature, "is_enabled", lambda: True)

    def unexpected_msc():
        raise AssertionError("An explicit filesystem was bypassed by MSC")

    monkeypatch.setattr(MultiStorageClientFeature, "import_package", unexpected_msc)
    with TempNamedDir(tmp_path_dist_ckpt / "injected_roundtrip") as root:
        filesystem = URIFileSystem(root)
        path = "test://checkpoints/step_7"
        save_strategy = TorchDistSaveShardedStrategy(thread_count=2)
        load_strategy = TorchDistLoadShardedStrategy()
        if fully_parallel:
            save_strategy = FullyParallelSaveStrategyWrapper(save_strategy)
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy)
        save(
            _state(),
            path,
            save_strategy,
            filesystem=filesystem,
            content_metadata={"format": "example"},
            verify_integrity=True,
        )
        restored = load(
            _state(0), path, load_strategy, filesystem=filesystem, verify_integrity=True
        )
        assert torch.equal(restored["weight"], torch.full((2, 3), 3))
        assert restored["step"] == 7
        assert check_is_distributed_checkpoint(path, filesystem=filesystem)
        assert set(load_tensors_metadata(path, filesystem=filesystem)) == {"weight"}
        assert "common_state/shard_0_1" in load_sharded_metadata(path, filesystem=filesystem)
        assert load_content_metadata(path, filesystem=filesystem) == {"format": "example"}
        assert load_common_state_dict(path, filesystem=filesystem)["step"] == 7
        assert torch.equal(
            load_plain_tensors(path, filesystem=filesystem)["weight"], torch.full((2, 3), 3)
        )
        padded = ShardedTensor.from_rank_offsets("weight", torch.zeros(4, 3), replica_id=Utils.rank)
        padded.gtp_pad_length = 2
        grant_shape_mismatch_for_gtp_padding(
            {"weight": padded}, path, 1, load_strategy, filesystem=filesystem
        )
        assert padded.allow_shape_mismatch

        torch.distributed.barrier()
        save_common({"legacy": 11}, path, filesystem=filesystem)
        torch.distributed.barrier()
        assert load_common_state_dict(path, filesystem=filesystem) == {"legacy": 11}
        if Utils.rank == 0:
            save_integrity_manifest(path, filesystem=filesystem)
        torch.distributed.barrier()
        verify_integrity_manifest(path, filesystem=filesystem)
        with filesystem.create_stream(path + "/integrity.json", "rb") as stream:
            manifest = json.load(stream)
        assert {"common.pt", "metadata.json", ".metadata"} <= manifest["files"].keys()
        torch.distributed.barrier()
        if Utils.rank == 0:
            with filesystem.create_stream(path + "/common.pt", "wb") as stream:
                stream.write(b"corrupted")
        torch.distributed.barrier()
        with pytest.raises(CheckpointingException, match="hash mismatch"):
            verify_integrity_manifest(path, filesystem=filesystem)


@pytest.mark.parametrize("failure", ["sync", "close", "thread"])
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_dcp_stream_failures_propagate(tmp_path_dist_ckpt, distributed_checkpoint, failure):
    with TempNamedDir(tmp_path_dist_ckpt / "stream_failure") as root:
        filesystem = URIFileSystem(root, failure=failure)
        if failure == "thread":
            filesystem.worker_failed = threading.Event()
        path = "test://checkpoints/failed"
        writer = make_filesystem_writer(path, filesystem, thread_count=2)
        state = {f"weight_{i}": torch.ones(3) for i in range(4)}
        with pytest.raises(dcp.CheckpointException, match="injected"):
            dcp.save(state, storage_writer=writer)
        assert not filesystem.exists(path + "/.metadata")


def test_sync_false_still_closes_streams(tmp_path_dist_ckpt, distributed_checkpoint):
    with TempNamedDir(tmp_path_dist_ckpt / "no_sync") as root:
        filesystem = URIFileSystem(root, failure="sync", failure_suffix="")
        path = "test://checkpoints/no_sync"
        writer = make_filesystem_writer(path, filesystem, sync_files=False)
        dcp.save({"weight": torch.arange(3)}, storage_writer=writer)
        restored = {"weight": torch.zeros(3, dtype=torch.int64)}
        dcp.load(restored, storage_reader=make_filesystem_reader(path, filesystem))
        assert torch.equal(restored["weight"], torch.arange(3))


def test_rank_zero_sidecar_failure_propagates(tmp_path_dist_ckpt, distributed_checkpoint):
    with TempNamedDir(tmp_path_dist_ckpt / "sidecar_failure") as root:
        filesystem = URIFileSystem(root, failure="close", failure_suffix="metadata.json")
        with pytest.raises(CheckpointingException, match="injected close failure"):
            save(_state(), "test://checkpoints/failed", filesystem=filesystem)


def test_directory_failure_precedes_writes(tmp_path_dist_ckpt, distributed_checkpoint):
    with TempNamedDir(tmp_path_dist_ckpt / "directory_failure") as root:
        filesystem = URIFileSystem(root, failure="list")
        path = "test://checkpoints/failed"
        filesystem.mkdir(path)
        state = _state()
        with pytest.raises(CheckpointingException, match="injected listing failure"):
            save(state, path, filesystem=filesystem)
        assert not list((root / "failed").iterdir())
        assert set(state) == {"weight", "step"}


def test_incompatible_injection_fails_before_io(tmp_path, monkeypatch):
    filesystem = URIFileSystem(tmp_path)
    state = {"step": 1}
    with pytest.raises(TypeError, match="torch_dist"):
        save(state, "test://checkpoints/failed", object(), filesystem=filesystem)
    monkeypatch.setattr(torch_strategy, "HAVE_NVRX", False)
    with pytest.raises(RuntimeError, match="newer nvidia-resiliency-ext"):
        save(state, "test://checkpoints/failed", filesystem=filesystem, async_sharded_save=True)
    wrapper = FullyParallelSaveStrategyWrapper(TorchDistSaveShardedStrategy())
    with pytest.raises(RuntimeError, match="newer nvidia-resiliency-ext"):
        wrapper.async_save(state, "test://checkpoints/failed", filesystem=filesystem)
    assert filesystem.accesses == []
    assert state == {"step": 1}


def test_default_sync_without_nvrx(tmp_path_dist_ckpt, distributed_checkpoint, monkeypatch):
    monkeypatch.setattr(torch_strategy, "HAVE_NVRX", False)
    with TempNamedDir(tmp_path_dist_ckpt / "default_without_nvrx") as path:
        save(_state(), path)
        restored = load(_state(0), path)
        assert torch.equal(restored["weight"], torch.full((2, 3), 3))
        assert {"metadata.json", ".metadata"} <= set(os.listdir(path))


def test_legacy_strategy_signatures(tmp_path_dist_ckpt, distributed_checkpoint):
    class LegacySaver(TorchDistSaveShardedStrategy):
        def save(self, state, path):
            return super().save(state, path)

    class LegacyLoader(TorchDistLoadShardedStrategy):
        def load(self, state, path):
            return super().load(state, path)

        def load_tensors_metadata(self, path, metadata=None):
            return super().load_tensors_metadata(path, metadata)

        def load_sharded_metadata(self, path):
            return super().load_sharded_metadata(path)

    with TempNamedDir(tmp_path_dist_ckpt / "legacy_strategy") as path:
        saver = FullyParallelSaveStrategyWrapper(LegacySaver())
        loader = FullyParallelLoadStrategyWrapper(LegacyLoader())
        save(_state(), path, saver)
        restored = load(_state(0), path, loader)
        assert torch.equal(restored["weight"], torch.full((2, 3), 3))
        assert set(load_tensors_metadata(path, loader)) == {"weight"}
        assert "weight" in load_sharded_metadata(path, loader)
        filesystem = URIFileSystem(path)
        for operation in (load_tensors_metadata, load_sharded_metadata):
            with pytest.raises(TypeError, match="does not accept filesystem"):
                operation("test://checkpoints/unread", loader, filesystem=filesystem)
        with pytest.raises(TypeError, match="does not accept filesystem"):
            load(_state(0), "test://checkpoints/unread", loader, filesystem=filesystem)
        assert filesystem.accesses == []


def _require_injected_nvrx():
    module = pytest.importorskip("nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async")
    if "filesystem" not in inspect.signature(module.FileSystemWriterAsync.__init__).parameters:
        pytest.skip("Installed NVRx does not support filesystem injection")


def test_removal_invalidates_cached_metadata(tmp_path_dist_ckpt, distributed_checkpoint):
    _require_injected_nvrx()
    from nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader import (
        CachedMetadataFileSystemReader,
    )

    with TempNamedDir(tmp_path_dist_ckpt / "remove_injected") as root:
        filesystem = URIFileSystem(root)
        path = "test://checkpoints/remove"
        state = _state()
        state["drop_weight"] = ShardedTensor.from_rank_offsets(
            "drop_weight", torch.ones(2), replica_id=Utils.rank
        )
        strategy = TorchDistSaveShardedStrategy(thread_count=2, separation_hint="drop")
        request = save(state, path, strategy, filesystem=filesystem, async_sharded_save=True)
        request.execute_sync()
        reader = CachedMetadataFileSystemReader(path, filesystem=filesystem)
        try:
            if Utils.rank == 0:
                assert "drop_weight" in reader.read_metadata().state_dict_metadata
                remove_sharded_tensors(path, "drop", filesystem=filesystem)
            torch.distributed.barrier()
            assert "drop_weight" not in reader.read_metadata().state_dict_metadata
            assert not any(name.startswith("drop") for name in filesystem.listdir(path))
            assert torch.equal(
                load(_state(0), path, filesystem=filesystem)["weight"], torch.full((2, 3), 3)
            )
        finally:
            CachedMetadataFileSystemReader.clear_metadata_cache()


def test_async_filesystem_capture(tmp_path_dist_ckpt, distributed_checkpoint, monkeypatch):
    _require_injected_nvrx()
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue

    with TempNamedDir(tmp_path_dist_ckpt / "async_injected") as root:
        filesystem = URIFileSystem(root)
        queue = AsyncCallsQueue(persistent=True)
        strategy = TorchDistSaveShardedStrategy(cached_metadata=True)
        try:
            for step in (3, 5):
                path = f"test://checkpoints/step_{step}"
                request = save(
                    _state(step),
                    path,
                    strategy,
                    filesystem=filesystem,
                    async_sharded_save=True,
                    verify_integrity=True,
                )
                monkeypatch.setattr(MultiStorageClientFeature, "is_enabled", lambda: True)
                queue.schedule_async_request(request)
                queue.maybe_finalize_async_calls(blocking=True)
                restored = load(_state(0), path, filesystem=filesystem, verify_integrity=True)
                assert torch.equal(restored["weight"], torch.full((2, 3), step))
        finally:
            queue.close()

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import signal
from types import SimpleNamespace

import pytest
import torch

from megatron.training import async_utils, dgrad_logging, dist_signal_handler, wandb_utils


@pytest.fixture(autouse=True)
def _reset_module_state():
    original_async_queue = async_utils._async_calls_queue
    original_results_queue = async_utils._results_queue
    original_dgrad_logger = dgrad_logging._LOGGER
    yield
    async_utils._async_calls_queue = original_async_queue
    async_utils._results_queue = original_results_queue
    dgrad_logging._LOGGER = original_dgrad_logger


def test_dist_signal_handler_helpers_without_initialized_distributed(monkeypatch):
    monkeypatch.setattr(dist_signal_handler.torch.distributed, "is_available", lambda: False)
    monkeypatch.setattr(dist_signal_handler.torch.distributed, "is_initialized", lambda: False)

    assert dist_signal_handler.get_world_size() == 1
    assert dist_signal_handler.all_gather_item(7, dtype=torch.int32) == [7]


def test_dist_signal_handler_device_selection(monkeypatch):
    monkeypatch.setattr(dist_signal_handler.torch.distributed, "get_backend", lambda: "gloo")
    assert dist_signal_handler.get_device() == torch.device("cpu")

    monkeypatch.setattr(dist_signal_handler.torch.distributed, "get_backend", lambda: "nccl")
    assert dist_signal_handler.get_device(local_rank=3) == torch.device("cuda:3")

    monkeypatch.setattr(dist_signal_handler.torch.distributed, "get_backend", lambda: "unknown")
    with pytest.raises(RuntimeError):
        dist_signal_handler.get_device()


def test_distributed_signal_handler_records_and_restores_signal():
    handler = dist_signal_handler.DistributedSignalHandler(signal.SIGUSR1)
    original = signal.getsignal(signal.SIGUSR1)

    with handler as active:
        signal.getsignal(signal.SIGUSR1)(signal.SIGUSR1, None)
        assert active._signal_received is True

    assert handler.released is True
    assert signal.getsignal(signal.SIGUSR1) == original
    assert handler.release() is False


def test_async_calls_queue_is_lazily_created_and_reused(monkeypatch):
    created = []

    class FakeAsyncCallsQueue:
        def __init__(self, persistent=False):
            self.persistent = persistent
            self.requests = []
            created.append(self)

        def schedule_async_request(self, request):
            self.requests.append(request)

        def get_num_unfinalized_calls(self):
            return len(self.requests)

    monkeypatch.setattr(
        async_utils,
        "get_args",
        lambda: SimpleNamespace(async_strategy="mcore", use_persistent_ckpt_worker=True),
    )
    monkeypatch.setattr(
        async_utils,
        "get_async_strategy",
        lambda strategy, *args: (strategy, {"AsyncCallsQueue": FakeAsyncCallsQueue}),
    )

    queue = async_utils._get_async_calls_queue()
    async_utils.schedule_async_save("request")

    assert async_utils._get_async_calls_queue() is queue
    assert queue.persistent is True
    assert queue.requests == ["request"]
    assert created == [queue]
    assert not async_utils.is_empty_async_queue()


def test_maybe_finalize_async_save_handles_disabled_and_enabled_paths(monkeypatch):
    calls = []

    class FakeQueue:
        def get_num_unfinalized_calls(self):
            return 1

        def maybe_finalize_async_calls(self, blocking, no_dist=False):
            calls.append(("finalize", blocking, no_dist))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(async_utils, "get_args", lambda: SimpleNamespace(async_save=False))
    async_utils._async_calls_queue = FakeQueue()
    async_utils.maybe_finalize_async_save(blocking=True)
    assert calls == []

    monkeypatch.setattr(async_utils, "get_args", lambda: SimpleNamespace(async_save=True))
    monkeypatch.setattr(async_utils, "print_rank_0", lambda message: calls.append(("print", message)))
    monkeypatch.setattr(
        "megatron.training.checkpointing.finalize_deletion_processes",
        lambda blocking=False: calls.append(("delete", blocking)),
    )

    async_utils.maybe_finalize_async_save(blocking=True, terminate=True)

    assert ("finalize", True, False) in calls
    assert ("delete", True) in calls
    assert ("close",) in calls


def test_reset_persistent_async_worker_closes_queue_and_clears_metadata(monkeypatch):
    calls = []

    class FakeQueue:
        def close(self, abort=False):
            calls.append(("queue_close", abort))

    class FakeManager:
        def shutdown(self):
            calls.append(("manager_shutdown",))

    class FakeResultsQueue:
        _manager = FakeManager()

    class FakeReader:
        @staticmethod
        def clear_metadata_cache():
            calls.append(("clear_cache",))

    async_utils._async_calls_queue = FakeQueue()
    async_utils._results_queue = FakeResultsQueue()
    monkeypatch.setattr(async_utils, "get_async_strategy", lambda *args: ("mcore", FakeReader))

    async_utils.reset_persistent_async_worker("mcore")

    assert calls == [("queue_close", True), ("manager_shutdown",), ("clear_cache",)]
    assert async_utils._async_calls_queue is None
    assert async_utils._results_queue is None


def test_dgrad_logger_captures_and_saves_backward_grads(monkeypatch, tmp_path):
    saved = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    logger = dgrad_logging.DataGradLogger(str(tmp_path))
    monkeypatch.setattr(
        dgrad_logging,
        "save_grads",
        lambda save_dir, state_dict, iteration, tag: saved.append(
            (save_dir, dict(state_dict), iteration, tag)
        ),
    )

    logger.register_hooks([model])
    model(torch.ones(1, 2, requires_grad=True)).sum().backward()
    logger.save(iteration=3)
    logger.remove_hooks()

    assert saved
    save_dir, state_dict, iteration, tag = saved[0]
    assert save_dir == str(tmp_path)
    assert iteration == 3
    assert tag == "dgrads"
    assert "model_chunk0" in state_dict
    assert logger._dgrads_state_dict == {}
    assert logger._hooks == []


def test_dgrad_global_helpers_reuse_logger(monkeypatch, tmp_path):
    calls = []

    class FakeLogger:
        def __init__(self, save_dir):
            self.save_dir = save_dir

        def register_hooks(self, model):
            calls.append(("register", model, self.save_dir))

        def remove_hooks(self):
            calls.append(("remove",))

        def save(self, iteration):
            calls.append(("save", iteration))

    monkeypatch.setattr(dgrad_logging, "DataGradLogger", FakeLogger)
    model = [object()]

    dgrad_logging.enable_dgrad_logging(model, str(tmp_path))
    dgrad_logging.disable_dgrad_logging()
    dgrad_logging.save_dgrads(5)

    assert calls == [
        ("register", model, str(tmp_path)),
        ("remove",),
        ("save", 5),
    ]


def test_wandb_checkpoint_success_logs_artifact_and_tracker(monkeypatch, tmp_path):
    artifact_calls = []

    class FakeArtifact:
        def __init__(self, name, type, metadata):
            artifact_calls.append(("create", name, type, metadata))

        def add_reference(self, reference, checksum=False):
            artifact_calls.append(("reference", reference, checksum))

        def add_file(self, filename):
            artifact_calls.append(("file", filename))

    class FakeRun:
        entity = "entity"
        project = "project"

        def log_artifact(self, artifact, aliases):
            artifact_calls.append(("log", aliases))

    writer = SimpleNamespace(Artifact=FakeArtifact, run=FakeRun())
    save_dir = tmp_path / "run"
    checkpoint = save_dir / "iter_0000001"
    tracker = tmp_path / "tracker.txt"
    save_dir.mkdir()
    checkpoint.mkdir()
    tracker.write_text("latest", encoding="utf-8")
    monkeypatch.setattr(wandb_utils, "get_wandb_writer", lambda: writer)

    wandb_utils.on_save_checkpoint_success(
        checkpoint_path=str(checkpoint),
        tracker_filename=str(tracker),
        save_dir=str(save_dir),
        iteration=1,
    )

    assert artifact_calls[0] == ("create", "run", "model", {"iteration": 1})
    assert ("file", str(tracker)) in artifact_calls
    assert ("log", ["iter_0000001"]) in artifact_calls
    assert (save_dir / "latest_wandb_artifact_path.txt").read_text() == "entity/project"


def test_wandb_checkpoint_load_uses_recorded_artifact_path(monkeypatch, tmp_path):
    used = []

    class FakeRun:
        def use_artifact(self, name):
            used.append(name)

    writer = SimpleNamespace(run=FakeRun())
    load_dir = tmp_path / "run"
    checkpoint = load_dir / "iter_0000007"
    load_dir.mkdir()
    checkpoint.mkdir()
    (load_dir / "latest_wandb_artifact_path.txt").write_text("entity/project", encoding="utf-8")
    monkeypatch.setattr(wandb_utils, "get_wandb_writer", lambda: writer)

    wandb_utils.on_load_checkpoint_success(str(checkpoint), str(load_dir))

    assert used == ["entity/project/run:iter_0000007"]

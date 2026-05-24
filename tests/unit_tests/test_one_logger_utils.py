# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

from megatron.training import one_logger_utils


class _FakeOneLogger:
    def __init__(self):
        self.store = {}
        self.metrics = []
        self.app_tags = []
        self.finished = False

    def get_context_manager(self):
        return nullcontext()

    def store_set(self, key, value):
        self.store[key] = value

    def store_get(self, key):
        return self.store[key]

    def store_has_key(self, key):
        return key in self.store

    def store_pop(self, key):
        return self.store.pop(key)

    def log_metrics(self, metrics):
        self.metrics.append(metrics)

    def log_app_tag(self, tag):
        self.app_tags.append(tag)

    def finish(self):
        self.finished = True


def _base_metrics():
    return {
        "iteration": 5,
        "train_duration": 4.0,
        "eval_duration": 2.0,
        "eval_iterations": 2,
        "total_flops_since_current_train_start": 8 * 10**12,
        "num_floating_point_operations_so_far": 12 * 10**12,
        "consumed_train_samples": 24,
        "world_size": 2,
        "seq_length": 8,
    }


def test_on_train_start_initializes_store_and_logs_metrics(monkeypatch):
    logger = _FakeOneLogger()
    monkeypatch.setattr(one_logger_utils, "get_one_logger", lambda: logger)
    monkeypatch.setattr(
        one_logger_utils,
        "get_args",
        lambda: SimpleNamespace(global_batch_size=4),
    )

    one_logger_utils.on_train_start(
        iteration=2,
        consumed_train_samples=8,
        train_samples=None,
        seq_length=16,
        train_iters=10,
        save="/tmp/ckpt",
        async_save=True,
        log_throughput=True,
        num_floating_point_operations_so_far=2 * 10**12,
    )

    assert logger.store["iteration_start"] == 2
    assert logger.store["train_samples_start"] == 8
    assert logger.metrics[-1]["train_samples_target"] == 40
    assert logger.metrics[-1]["train_tokens_target"] == 640
    assert logger.metrics[-1]["save_checkpoint_strategy"] == "async"
    assert logger.metrics[-1]["train_tflop_start"] == 2.0


def test_produce_e2e_metrics_tracks_iteration_minima_and_throughput(monkeypatch):
    logger = _FakeOneLogger()
    logger.store.update(
        {
            "get_e2e_base_metrics": _base_metrics,
            "iteration_start": 1,
            "train_samples_start": 8,
            "train_iterations_time_msecs_total": 1000.0,
            "tracked_train_iterations": 3,
            "validation_iterations_time_msecs_total": 500.0,
            "tracked_validation_iterations": 1,
            "train_throughput_per_gpu_max": 0.5,
        }
    )
    monkeypatch.setattr(one_logger_utils, "get_one_logger", lambda: logger)

    metrics = one_logger_utils._produce_e2e_metrics(log_throughput=True, throughput=1.5)

    assert metrics["train_iterations"] == 4
    assert metrics["train_samples"] == 16
    assert metrics["train_tokens"] == 128
    assert metrics["validation_iterations_time_msecs_avg"] == 1000.0
    assert metrics["train_throughput_per_gpu"] == 1.0
    assert metrics["train_throughput_per_gpu_max"] == 1.5
    assert logger.store["train_iterations_time_msecs_min"] == 1500.0
    assert logger.store["validation_iterations_time_msecs_min"] == 1500.0


def test_track_e2e_metrics_logs_produced_metrics(monkeypatch):
    logger = _FakeOneLogger()
    monkeypatch.setattr(one_logger_utils, "get_one_logger", lambda: logger)
    monkeypatch.setattr(
        one_logger_utils,
        "_produce_e2e_metrics",
        lambda log_throughput=False, throughput=None: {"throughput": throughput},
    )

    one_logger_utils.track_e2e_metrics(log_throughput=True, throughput=7)

    assert logger.metrics == [{"throughput": 7}]


def test_checkpoint_lifecycle_logs_productive_metrics(monkeypatch):
    logger = _FakeOneLogger()
    logger.store.update(
        {
            "get_e2e_base_metrics": _base_metrics,
            "save_checkpoint_count": 0,
            "save_checkpoint_sync_time_total": 0.0,
            "app_train_loop_start_time": 123,
        }
    )
    monkeypatch.setattr(one_logger_utils, "get_one_logger", lambda: logger)

    productive = one_logger_utils.on_save_checkpoint_start(async_save=True)
    one_logger_utils.on_save_checkpoint_end(
        save_checkpoint_duration=3.5,
        current_iteration=productive["train_iterations_productive_end"],
        async_save=True,
    )
    one_logger_utils.on_save_checkpoint_success(productive, async_save=True)

    assert logger.store["save_checkpoint_count"] == 1
    assert logger.store["save_checkpoint_sync_time_max"] == 3.5
    assert logger.store["save_checkpoint_sync_time_min"] == 3.5
    assert logger.store["iters_prod_max"] == 5
    assert logger.metrics[-1]["save_checkpoint_sync_time_total_productive"] == 3.5


def test_on_pretrain_start_tracks_application_metadata(monkeypatch):
    logger = _FakeOneLogger()
    args = SimpleNamespace(
        app_tag_run_name="run",
        app_tag_run_version="v1",
        data_parallel_size=2,
        context_parallel_size=1,
        global_batch_size=8,
        micro_batch_size=2,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        world_size=2,
        seq_length=16,
        log_throughput=True,
    )
    monkeypatch.setattr(one_logger_utils, "get_one_logger", lambda: logger)
    monkeypatch.setattr(one_logger_utils, "get_args", lambda: args)

    one_logger_utils.on_pretrain_start()

    assert logger.store["app_tag_run_name"] == "run"
    assert logger.store["app_tag_run_version"] == "v1"
    assert logger.metrics[-1]["app_run_type"] == "training"
    assert logger.metrics[-1]["one_logger_utils_version"] == "1.2.0-mlm"


def test_track_config_flags_app_tag_and_finish(monkeypatch):
    logger = _FakeOneLogger()
    logger.store["app_tag_run_name"] = "run"
    logger.store["app_tag_run_version"] = "v1"
    monkeypatch.setattr(one_logger_utils, "get_one_logger", lambda: logger)

    one_logger_utils.track_config_flags(
        train_iters=10,
        skip_train=False,
        do_train=True,
        do_valid=False,
        do_test=True,
        dataloader_type="single",
    )
    one_logger_utils.track_app_tag(batch_size=8, world_size=2, seq_length=16)
    one_logger_utils.finish()

    assert logger.metrics[-1]["is_train_iterations_enabled"]
    assert logger.metrics[-1]["is_validation_iterations_enabled"] is False
    assert logger.metrics[-1]["is_test_iterations_enabled"]
    assert logger.app_tags == ["run_v1_8_2_16"]
    assert logger.finished

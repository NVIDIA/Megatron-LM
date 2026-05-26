# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.training.checkpointing import save_grads
from megatron.training.global_vars import set_args
from megatron.training import training
from megatron.training.training import (
    build_train_valid_test_data_loaders,
    build_train_valid_test_data_iterators,
    checkpoint_and_decide_exit,
    compute_throughputs_and_append_to_progress_log,
    get_model,
    get_start_time_from_progress_log,
    get_train_valid_test_num_samples,
    get_megatron_optimizer_config,
    get_optimizer_param_scheduler,
    dummy_train_step,
    num_floating_point_operations,
    post_training_step_callbacks,
    preprocess_common_state_dict,
    should_disable_forward_pre_hook,
    setup_model_and_optimizer,
    training_log,
    update_train_iters,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
    return iter([1]), iter([2]), iter([3])


def create_test_args():
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = 1
    args.train_iters = 1
    args.eval_interval = 1
    args.eval_iters = 1
    args.global_batch_size = 1
    args.consumed_train_samples = 1
    args.consumed_valid_samples = 1
    args.dataloader_type = "external"
    args.skip_train = False
    args.full_validation = False
    args.multiple_validation_sets = False
    args.perform_rl_step = False
    args.phase_transition_iterations = None

    return args


def create_flop_args(**overrides):
    args = SimpleNamespace(
        group_query_attention=True,
        num_query_groups=2,
        num_attention_heads=4,
        num_layers=4,
        num_experts=None,
        moe_layer_freq=None,
        moe_router_topk=1,
        mtp_num_layers=None,
        moe_ffn_hidden_size=None,
        ffn_hidden_size=32,
        moe_latent_size=None,
        moe_shared_expert_intermediate_size=None,
        swiglu=False,
        multi_latent_attention=False,
        q_lora_rank=None,
        qk_head_dim=8,
        qk_pos_emb_head_dim=4,
        kv_lora_rank=4,
        v_head_dim=8,
        hidden_size=16,
        kv_channels=4,
        seq_length=8,
        attention_output_gate=False,
        experimental_attention_variant=None,
        linear_attention_freq=None,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=3,
        padded_vocab_size=128,
        hybrid_layer_pattern=None,
        mamba_state_dim=8,
        mamba_head_dim=4,
        mamba_num_groups=2,
        mamba_num_heads=4,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_num_floating_point_operations_dense_transformer_paths():
    dense = create_flop_args(num_experts=None, group_query_attention=False)
    gated = create_flop_args(attention_output_gate=True, swiglu=True, mtp_num_layers=1)

    assert num_floating_point_operations(dense, batch_size=2) > 0
    assert dense.num_query_groups == dense.num_attention_heads
    assert num_floating_point_operations(gated, batch_size=2) > num_floating_point_operations(
        create_flop_args(), batch_size=2
    )


def test_num_floating_point_operations_moe_and_linear_attention_paths():
    moe = create_flop_args(
        num_experts=4,
        moe_layer_freq=[1, 0, 1, 0],
        moe_router_topk=2,
        moe_ffn_hidden_size=24,
        moe_shared_expert_intermediate_size=8,
        moe_latent_size=6,
        mtp_num_layers=1,
    )
    linear = create_flop_args(
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=2,
    )

    assert num_floating_point_operations(moe, batch_size=1) > 0
    assert num_floating_point_operations(linear, batch_size=1) > 0


def test_num_floating_point_operations_validates_attention_patterns():
    with pytest.raises(RuntimeError, match="moe-layer-freq"):
        num_floating_point_operations(create_flop_args(num_experts=2, moe_layer_freq="bad"), 1)

    with pytest.raises(ValueError, match="linear_attention_freq is None"):
        num_floating_point_operations(
            create_flop_args(experimental_attention_variant="gated_delta_net"),
            1,
        )

    with pytest.raises(ValueError, match="Invalid linear_attention_freq"):
        num_floating_point_operations(
            create_flop_args(
                experimental_attention_variant="gated_delta_net",
                linear_attention_freq=object(),
            ),
            1,
        )


def test_get_start_time_from_progress_log_handles_async_and_world_size_reset(monkeypatch, tmp_path):
    progress = tmp_path / "progress.txt"
    progress.write_text(
        "\n".join(
            [
                "2026-05-25 10:00:00\tJob ID: old\t# GPUs: 2\tStarting job",
                "2026-05-25 10:01:00\tJob ID: old\t# GPUs: 2\tSaved checkpoint\t"
                "Iteration: 1\tJob throughput: 1\tCumulative throughput: 1\t"
                "Floating-point operations: 1.00e+03\tTokens: 1",
                "2026-05-25 10:02:00\tJob ID: new\t# GPUs: 4\tStarting job",
                "2026-05-25 10:03:00\tJob ID: new\t# GPUs: 4\tSaving async checkpoint\t"
                "Iteration: 2\tJob throughput: 1\tCumulative throughput: 1\t"
                "Floating-point operations: 2.00e+03\tTokens: 1",
                "2026-05-25 10:04:00\tJob ID: new\t# GPUs: 4\tSaved async checkpoint\t"
                "Iteration: 2\tJob throughput: 1\tCumulative throughput: 1\t"
                "Floating-point operations: 0.00e+00\tTokens: 1",
                "2026-05-25 10:05:00\tJob ID: same\t# GPUs: 4\tStarting job",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(training, "get_args", lambda: SimpleNamespace(save=str(tmp_path), world_size=4))
    monkeypatch.setattr(training, "print_rank_0", lambda *args, **kwargs: None)

    start_time, start_flops = get_start_time_from_progress_log()

    assert start_time.strftime("%Y-%m-%d %H:%M:%S") == "2026-05-25 10:02:00"
    assert start_flops == 1000.0


def test_preprocess_common_state_dict_strips_rank_and_sorts_optimizer_groups():
    common_state = {
        "args": SimpleNamespace(
            rank=3,
            local_rank=1,
            use_distributed_optimizer=True,
        ),
        "optimizer": {
            "param_state": {},
            "optimizer": {
                "param_groups": [
                    {
                        "wd_mult": 1.0,
                        "lr_mult": 2.0,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                        "is_vision_model_param": False,
                        "is_engram_parallel": False,
                    },
                    {
                        "wd_mult": 1.0,
                        "lr_mult": 1.0,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                        "is_vision_model_param": False,
                        "is_engram_parallel": False,
                    },
                ]
            },
        },
    }

    processed = preprocess_common_state_dict(common_state)

    assert "rank" not in processed["args"]
    assert "local_rank" not in processed["args"]
    assert "param_state" not in processed["optimizer"]
    assert [group["lr_mult"] for group in processed["optimizer"]["optimizer"]["param_groups"]] == [1.0, 2.0]
    assert "rank" in vars(common_state["args"])


def test_preprocess_common_state_dict_handles_chained_optimizers():
    optimizer_state = {
        0: {
            "param_state": {},
            "optimizer": {
                "param_groups": [
                    {
                        "wd_mult": 2.0,
                        "lr_mult": 1.0,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                        "is_vision_model_param": False,
                        "is_engram_parallel": False,
                    },
                    {
                        "wd_mult": 1.0,
                        "lr_mult": 1.0,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                        "is_vision_model_param": False,
                        "is_engram_parallel": False,
                    },
                ]
            },
        },
        2: {"optimizer": {"param_groups": []}},
    }
    common_state = {
        "args": SimpleNamespace(use_distributed_optimizer=True, rank=7),
        "optimizer": optimizer_state,
    }

    processed = preprocess_common_state_dict(common_state)

    assert "rank" not in processed["args"]
    assert "param_state" not in processed["optimizer"][0]
    assert [group["wd_mult"] for group in processed["optimizer"][0]["optimizer"]["param_groups"]] == [
        1.0,
        2.0,
    ]


def test_get_train_valid_test_num_samples_iteration_sample_and_phase_paths(monkeypatch):
    args = SimpleNamespace(
        train_samples=None,
        train_iters=9,
        global_batch_size=4,
        full_validation=False,
        skip_train=False,
        eval_interval=3,
        eval_iters=2,
        phase_transition_iterations=None,
        iteration=0,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)

    assert get_train_valid_test_num_samples() == (36, 16, 8)

    args.train_samples = 100
    args.full_validation = True
    assert get_train_valid_test_num_samples() == (100, None, 8)

    args.full_validation = False
    args.skip_train = True
    assert get_train_valid_test_num_samples() == (100, 8, 8)

    args.skip_train = False
    args.phase_transition_iterations = [3, 6]
    args.iteration = 4
    assert get_train_valid_test_num_samples()[0] == 12


def test_cyclic_iter_restarts_after_exhausting_iterable():
    iterator = training.cyclic_iter([1, 2])

    assert [next(iterator), next(iterator), next(iterator), next(iterator)] == [1, 2, 1, 2]


def test_update_train_iters_constant_and_rampup(monkeypatch):
    constant = SimpleNamespace(
        train_iters=None,
        rampup_batch_size=None,
        train_samples=100,
        global_batch_size=8,
    )
    update_train_iters(constant)
    assert constant.train_iters == 12

    calls = []
    rampup = SimpleNamespace(
        train_iters=None,
        rampup_batch_size=(2, 2, 4),
        train_samples=12,
        global_batch_size=4,
    )
    monkeypatch.setattr(training, "update_num_microbatches", lambda consumed, consistency_check=False: calls.append(consumed))
    monkeypatch.setattr(training, "get_current_global_batch_size", lambda: 2)
    update_train_iters(rampup)

    assert rampup.train_iters == 4
    assert calls == [0, 2, 4, 0]


def test_checkpoint_and_decide_exit_save_and_iteration_paths(monkeypatch):
    calls = []
    args = SimpleNamespace(
        exit_signal_handler=False,
        save="/tmp/checkpoints",
        save_interval=5,
        non_persistent_save_interval=None,
        exit_duration_in_mins=None,
        exit_interval=None,
        phase_transition_iterations=None,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: object())
    monkeypatch.setattr(training, "save_checkpoint_and_time", lambda *items, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(training, "print_datetime", lambda *items, **kwargs: calls.append(("print", items)))

    assert checkpoint_and_decide_exit(None, None, None, 10, 0.0, {}, None) is False
    assert calls == [{"train_data_iterator": None}]

    args.save_interval = None
    args.exit_interval = 3
    assert checkpoint_and_decide_exit(None, None, None, 6, 0.0, {}, None) is True
    assert ("print", ("exiting program at iteration 6",)) in calls


def test_checkpoint_and_decide_exit_signal_and_non_persistent_paths(monkeypatch):
    calls = []
    args = SimpleNamespace(
        exit_signal_handler=True,
        save="/tmp/checkpoints",
        save_interval=None,
        non_persistent_save_interval=4,
        exit_duration_in_mins=None,
        exit_interval=None,
        phase_transition_iterations=None,
    )
    signal_handler = SimpleNamespace(signals_received=lambda: [15])
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: object())
    monkeypatch.setattr(training, "get_signal_handler", lambda: signal_handler)
    monkeypatch.setattr(training, "save_checkpoint_and_time", lambda *items, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(training, "print_datetime", lambda *items, **kwargs: calls.append(("print", items)))

    assert checkpoint_and_decide_exit("m", "o", "s", 4, 123.0, {"ctx": True}, "iter") is True
    assert calls[0] == {"train_data_iterator": "iter"}
    assert ("print", ("exiting program after receiving SIGTERM.",)) in calls

    calls.clear()
    args.exit_signal_handler = False
    assert checkpoint_and_decide_exit("m", "o", "s", 8, 123.0, {}, "iter") is False
    assert calls == [{"non_persistent_ckpt": True, "train_data_iterator": "iter"}]


def test_compute_throughputs_and_append_to_progress_log_formats_checkpoint_line(monkeypatch):
    calls = []
    args = SimpleNamespace(
        save="/tmp/checkpoints",
        num_floating_point_operations_so_far=1.0e12,
        world_size=2,
        consumed_train_samples=500,
        seq_length=1024,
        async_save=True,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "_TRAIN_START_TIME", 90.0)
    monkeypatch.setattr(training.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        training,
        "get_start_time_from_progress_log",
        lambda: (training.datetime.fromtimestamp(80), 0.5e12),
    )
    monkeypatch.setattr(training, "append_to_progress_log", lambda message: calls.append(message))

    compute_throughputs_and_append_to_progress_log(7, 3.0e12)

    assert len(calls) == 1
    assert calls[0].startswith("Saving async checkpoint\tIteration: 7")
    assert "Floating-point operations: 3.00e+12" in calls[0]
    assert "Tokens (in billions): 0.00" in calls[0]

    args.save = None
    calls.clear()
    compute_throughputs_and_append_to_progress_log(8, 4.0e12)
    assert calls == []


def test_build_train_valid_test_data_loaders_regular_and_rl_paths(monkeypatch):
    real_tensor = torch.tensor
    args = SimpleNamespace(
        iteration=2,
        train_samples=None,
        train_iters=4,
        eval_interval=2,
        eval_iters=1,
        global_batch_size=8,
        consumed_train_samples=0,
        consumed_valid_samples=0,
        phase_transition_iterations=[1],
        perform_rl_step=False,
        skip_train=False,
        full_validation=True,
        multiple_validation_sets=True,
    )
    loader_calls = []
    provider_calls = []

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    def provider(samples):
        provider_calls.append(samples)
        return "train", ["valid-a", "valid-b"], "test"

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "print_rank_0", lambda *items, **kwargs: None)
    monkeypatch.setattr(training.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training.torch.distributed, "broadcast", lambda tensor, src: None)
    monkeypatch.setattr(
        training,
        "build_pretraining_data_loader",
        lambda dataset, consumed: loader_calls.append((dataset, consumed)) or (dataset, consumed),
    )

    train_loader, valid_loaders, test_loader = build_train_valid_test_data_loaders(provider)

    assert provider_calls == [(24, None, 8)]
    assert train_loader == ("train", 8)
    assert valid_loaders == [("valid-a", 0), ("valid-b", 0)]
    assert test_loader == ("test", 0)
    assert args.do_train and args.do_valid and args.do_test

    args.perform_rl_step = True
    args.train_iters = 0
    args.eval_iters = 0
    args.full_validation = False
    args.do_train = args.do_valid = args.do_test = False
    train_loader, valid_loaders, test_loader = build_train_valid_test_data_loaders(provider)
    assert (train_loader, valid_loaders, test_loader) == (None, None, None)
    assert not args.do_train and not args.do_valid and not args.do_test


def test_build_train_valid_test_data_iterators_wraps_multiple_validation_loaders(monkeypatch):
    args = SimpleNamespace(
        dataloader_type="external",
        full_validation=True,
        multiple_validation_sets=True,
        eval_iters=1,
    )
    wrapped = []
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(
        training,
        "build_train_valid_test_data_loaders",
        lambda provider: ("train-loader", ["valid-a", "valid-b"], "test-loader"),
    )
    monkeypatch.setattr(training, "RerunDataIterator", lambda iterator: wrapped.append(iterator) or ("wrapped", iterator))

    train_iter, valid_iters, test_iter = build_train_valid_test_data_iterators(lambda _: None)

    assert train_iter == ("wrapped", "train-loader")
    assert valid_iters == [("wrapped", "valid-a"), ("wrapped", "valid-b")]
    assert test_iter == ("wrapped", "test-loader")
    assert args.eval_iters == [7, 7]


def test_build_train_valid_test_data_iterators_single_and_cyclic_modes(monkeypatch):
    wrapped = []
    args = SimpleNamespace(
        dataloader_type="single",
        full_validation=False,
        multiple_validation_sets=False,
        eval_iters=1,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(
        training,
        "build_train_valid_test_data_loaders",
        lambda provider: ([1], [[2]], [3]),
    )
    monkeypatch.setattr(training, "RerunDataIterator", lambda iterator: wrapped.append(iterator) or iterator)

    train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(lambda _: None)

    assert next(train_iter) == 1
    assert next(valid_iter) == 2
    assert next(test_iter) == 3

    args.dataloader_type = "cyclic"
    train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(lambda _: None)
    assert [next(train_iter), next(train_iter)] == [1, 1]
    assert [next(valid_iter), next(valid_iter)] == [2, 2]
    assert [next(test_iter), next(test_iter)] == [3, 3]


def test_should_disable_forward_pre_hook_requires_dist_optimizer_and_overlap():
    args = SimpleNamespace(
        use_megatron_fsdp=False,
        use_distributed_optimizer=True,
        optimizer="adam",
        overlap_param_gather=True,
    )
    assert should_disable_forward_pre_hook(args) is True

    args.use_megatron_fsdp = True
    assert should_disable_forward_pre_hook(args) is False

    args.use_megatron_fsdp = False
    args.use_distributed_optimizer = False
    args.optimizer = "adam"
    assert should_disable_forward_pre_hook(args) is False

    args.optimizer = "distributed_adam"
    args.overlap_param_gather = False
    assert should_disable_forward_pre_hook(args) is False


def test_dummy_train_step_consumes_microbatches_until_rerun_stops(monkeypatch):
    calls = []

    class FakeRerunStateMachine:
        def __init__(self):
            self.remaining = 2

        def should_run_forward_backward(self, data_iterator):
            self.remaining -= 1
            return self.remaining >= 0

    monkeypatch.setattr(training, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: FakeRerunStateMachine())
    monkeypatch.setattr(
        training,
        "get_batch_on_this_tp_rank",
        lambda iterator: calls.append(("tp", iterator)) or "batch",
    )
    monkeypatch.setattr(
        training,
        "get_batch_on_this_cp_rank",
        lambda batch: calls.append(("cp", batch)) or "cp-batch",
    )

    dummy_train_step("iterator")

    assert calls == [
        ("tp", "iterator"),
        ("cp", "batch"),
        ("tp", "iterator"),
        ("cp", "batch"),
    ]


def test_training_log_updates_accumulators_and_writers(monkeypatch):
    calls = []
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*args, **kwargs)

    class FakeTimer:
        def __init__(self, name):
            self.name = name

        def elapsed(self, barrier=False, reset=True):
            calls.append(("elapsed", self.name, barrier, reset))
            return 2.0

    class FakeTimers:
        def __call__(self, name, log_level=None):
            return FakeTimer(name)

        def write(self, names, writer, iteration, normalizer=None, reset=False):
            calls.append(("write", tuple(names), iteration, normalizer, reset, writer is not None))

        def log(self, names, normalizer=None, reset=True):
            calls.append(("log", tuple(names), normalizer, reset))

    class FakeWriter:
        def add_scalar(self, name, value, iteration):
            calls.append(("scalar", name, iteration))

    class FakeWandb:
        def log(self, payload, iteration=None):
            calls.append(("wandb", tuple(sorted(payload)), iteration))

    args = SimpleNamespace(
        timing_log_level=2,
        perform_rl_step=False,
        rl_use_sequence_packing=False,
        micro_batch_size=2,
        data_parallel_size=3,
        world_size=6,
        seq_length=8,
        tensorboard_log_interval=1,
        consumed_train_samples=48,
        skipped_train_samples=0,
        log_loss_scale_to_tensorboard=True,
        log_world_size_to_tensorboard=True,
        log_memory_to_tensorboard=False,
        log_max_attention_logit=True,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=None,
        log_interval=1,
        train_iters=10,
        log_throughput=True,
        log_timers_to_tensorboard=True,
        log_energy=False,
        record_memory_history=False,
        memory_snapshot_path="unused",
        log_memory_interval=None,
    )
    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: FakeWriter())
    monkeypatch.setattr(training, "get_wandb_writer", lambda: FakeWandb())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_energy_monitor", lambda: SimpleNamespace(lap=lambda: 0.0))
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", lambda value: value)
    monkeypatch.setattr(training.one_logger_utils, "track_app_tag", lambda *items: calls.append(("tag", items)))
    monkeypatch.setattr(training.one_logger_utils, "track_e2e_metrics", lambda *items: calls.append(("e2e", items)))
    monkeypatch.setattr(training, "num_floating_point_operations", lambda args, batch_size: 12e12)
    monkeypatch.setattr(training, "print_rank_last", lambda message: calls.append(("print_last", message)))

    total_loss = {}
    report_memory = training_log(
        {"lm loss": torch.tensor([2.0])},
        total_loss,
        learning_rate=0.001,
        iteration=2,
        loss_scale=128.0,
        report_memory_flag=False,
        skipped_iter=0,
        grad_norm=1.5,
        params_norm=2.5,
        num_zeros_in_grad=3,
        max_attention_logit=4.0,
    )

    assert report_memory is False
    assert total_loss["advanced iterations"] == 0
    assert total_loss["skipped iterations"] == 0
    assert total_loss["nan iterations"] == 0
    assert any(item[:2] == ("scalar", "learning-rate") for item in calls)
    assert any(item[:2] == ("scalar", "throughput") for item in calls)
    assert any(item[0] == "print_last" and "lm loss" in item[1] for item in calls)


def test_post_training_step_callbacks_runs_optional_hooks(monkeypatch):
    calls = []

    class FakeCudaRuntime:
        def cudaProfilerStop(self):
            calls.append("profiler-stop")
            return 0

    class FakeNvtxContext:
        def __exit__(self, exc_type, exc, tb):
            calls.append("nvtx-exit")

    args = SimpleNamespace(
        train_sync_interval=2,
        log_interval=2,
        log_straggler=True,
        check_weight_hash_across_dp_replicas_interval=2,
        adlr_autoresume=True,
        adlr_autoresume_interval=2,
        profile=True,
        profile_step_end=2,
        profile_ranks=[],
        use_pytorch_profiler=False,
        manual_gc=True,
        manual_gc_interval=2,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training.torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(training.stimer, "report", lambda flops, interval: calls.append(("straggler", flops, interval)))
    monkeypatch.setattr(training, "should_disable_forward_pre_hook", lambda args: True)
    monkeypatch.setattr(training, "disable_forward_pre_hook", lambda model: calls.append("disable-hook"))
    monkeypatch.setattr(training, "enable_forward_pre_hook", lambda model: calls.append("enable-hook"))
    monkeypatch.setattr(training, "check_param_hashes_across_dp_replicas", lambda model, cross_check=True: True)
    monkeypatch.setattr(training.torch.distributed, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(training, "print_rank_0", lambda message: calls.append(("print", message)))
    monkeypatch.setattr(training, "check_adlr_autoresume_termination", lambda *items: calls.append("autoresume"))
    monkeypatch.setattr(training.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(training.torch.cuda, "cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(training.torch.cuda, "check_error", lambda value: calls.append(("check-error", value)))
    monkeypatch.setattr(training.gc, "collect", lambda: calls.append("gc"))

    remaining_flops = post_training_step_callbacks(
        model=[object()],
        optimizer=object(),
        opt_param_scheduler=object(),
        iteration=2,
        prof=None,
        num_floating_point_operations_since_last_log_event=123.0,
        nsys_nvtx_context=FakeNvtxContext(),
    )

    assert remaining_flops == 0.0
    assert "sync" in calls
    assert ("straggler", 123.0, 2) in calls
    assert "disable-hook" in calls and "enable-hook" in calls
    assert "autoresume" in calls
    assert "nvtx-exit" in calls
    assert "gc" in calls


def test_get_optimizer_param_scheduler_iteration_and_sample_modes(monkeypatch):
    created = []

    class FakeScheduler:
        def __init__(self, optimizer, **kwargs):
            self.optimizer = optimizer
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(training, "OptimizerParamScheduler", FakeScheduler)
    iteration_args = SimpleNamespace(
        train_iters=10,
        lr_decay_iters=None,
        global_batch_size=4,
        lr_wsd_decay_iters=2,
        lr_warmup_fraction=0.1,
        lr_warmup_iters=0,
        train_samples=None,
        lr_decay_samples=None,
        lr_wsd_decay_samples=None,
        lr_warmup_samples=0,
        lr_warmup_init=0.0,
        lr=0.01,
        min_lr=0.001,
        lr_decay_style="linear",
        start_weight_decay=0.0,
        end_weight_decay=0.1,
        weight_decay_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        lr_wsd_decay_style="exponential",
    )
    monkeypatch.setattr(training, "get_args", lambda: iteration_args)

    scheduler = get_optimizer_param_scheduler("optimizer")

    assert scheduler.kwargs["lr_decay_steps"] == 40
    assert scheduler.kwargs["lr_warmup_steps"] == 4
    assert scheduler.kwargs["wsd_decay_steps"] == 8

    sample_args = SimpleNamespace(**vars(iteration_args))
    sample_args.train_iters = None
    sample_args.train_samples = 100
    sample_args.lr_decay_samples = None
    sample_args.lr_wsd_decay_samples = 20
    sample_args.lr_warmup_fraction = None
    sample_args.lr_warmup_samples = 5
    monkeypatch.setattr(training, "get_args", lambda: sample_args)
    monkeypatch.setattr(training, "update_train_iters", lambda args: setattr(args, "train_iters", 25))

    scheduler = get_optimizer_param_scheduler("optimizer")

    assert scheduler.kwargs["lr_decay_steps"] == 100
    assert scheduler.kwargs["lr_warmup_steps"] == 5
    assert scheduler.kwargs["wsd_decay_steps"] == 20


def test_get_megatron_optimizer_config_selects_supported_optimizers():
    adam_config, adam_overrides = get_megatron_optimizer_config(SimpleNamespace(optimizer="adam"))
    muon_config, _ = get_megatron_optimizer_config(SimpleNamespace(optimizer="muon"))
    sgd_config, sgd_overrides = get_megatron_optimizer_config(SimpleNamespace(optimizer="sgd"))

    assert adam_config.optimizer == "adam"
    assert muon_config.optimizer == "muon"
    assert sgd_config.optimizer == "sgd"
    assert isinstance(adam_overrides, dict)
    assert isinstance(sgd_overrides, dict)
    with pytest.raises(ValueError, match="Invalid optimizer type"):
        get_megatron_optimizer_config(SimpleNamespace(optimizer="rmsprop"))


def test_get_model_builds_single_chunk_without_cuda_or_ddp(monkeypatch):
    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self, **metadata):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1))
            self.metadata = metadata

    args = SimpleNamespace(
        virtual_pipeline_model_parallel_size=None,
        init_model_with_meta_device=False,
        load=None,
        export_kd_teacher_load=None,
        use_torch_fsdp2=True,
        use_cpu_initialization=True,
        use_megatron_fsdp=False,
        fp16=False,
        bf16=False,
    )
    pg_collection = SimpleNamespace(pp="pp", dp="dp", cp="cp", tp="tp")
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training.ProcessGroupCollection, "use_mpu_process_groups", lambda: pg_collection)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(training, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(training, "is_pp_first_stage", lambda pp: True)
    monkeypatch.setattr(training, "is_pp_last_stage", lambda pp: True)
    monkeypatch.setattr(
        training.tensor_parallel,
        "set_defaults_if_not_set_tensor_model_parallel_attributes",
        lambda param: calls.append(("set-defaults", tuple(param.shape))),
    )
    monkeypatch.setattr(training, "correct_amax_history_if_needed", lambda model: calls.append(("amax", len(model))))

    model = get_model(lambda **kwargs: FakeModel(**kwargs), wrap_with_ddp=False)

    assert len(model) == 1
    assert model[0].metadata["pre_process"] is True
    assert model[0].metadata["post_process"] is True
    assert model[0].model_type is training.ModelType.encoder_or_decoder
    assert ("set-defaults", (1,)) in calls
    assert ("amax", 1) in calls


def test_get_model_builds_virtual_pipeline_chunks(monkeypatch):
    class FakeModel(torch.nn.Module):
        def __init__(self, **metadata):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1))
            self.metadata = metadata

    args = SimpleNamespace(
        virtual_pipeline_model_parallel_size=3,
        init_model_with_meta_device=False,
        load=None,
        export_kd_teacher_load=None,
        use_torch_fsdp2=True,
        use_cpu_initialization=True,
        use_megatron_fsdp=False,
        fp16=False,
        bf16=False,
    )
    pg_collection = SimpleNamespace(pp="pp", dp="dp", cp="cp", tp="tp")
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training.ProcessGroupCollection, "use_mpu_process_groups", lambda: pg_collection)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 2 if group == "pp" else 1)
    monkeypatch.setattr(training, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(training, "is_pp_first_stage", lambda pp: True)
    monkeypatch.setattr(training, "is_pp_last_stage", lambda pp: True)
    monkeypatch.setattr(training, "is_vp_first_stage", lambda vp_stage, vp_size: vp_stage == 0)
    monkeypatch.setattr(training, "is_vp_last_stage", lambda vp_stage, vp_size: vp_stage == vp_size - 1)
    monkeypatch.setattr(
        training.tensor_parallel,
        "set_defaults_if_not_set_tensor_model_parallel_attributes",
        lambda param: None,
    )
    monkeypatch.setattr(training, "correct_amax_history_if_needed", lambda model: None)

    model = get_model(lambda **kwargs: FakeModel(**kwargs), wrap_with_ddp=False)

    assert [chunk.vp_stage for chunk in model] == [0, 1, 2]
    assert [chunk.metadata["pre_process"] for chunk in model] == [True, False, False]
    assert [chunk.metadata["post_process"] for chunk in model] == [False, False, True]


def test_setup_model_and_optimizer_skips_optimizer_for_inference_only_rl(monkeypatch):
    calls = []
    args = SimpleNamespace(
        skip_train=True,
        perform_rl_step=True,
        no_load_optim=True,
        moe_use_upcycling=False,
        load=None,
        pretrained_checkpoint=None,
        ckpt_convert_format=None,
        fp16=False,
        bf16=False,
    )
    model = [SimpleNamespace()]
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: object())
    monkeypatch.setattr(
        training,
        "get_one_logger",
        lambda: SimpleNamespace(log_metrics=lambda metrics: calls.append(("metrics", tuple(sorted(metrics))))),
    )
    monkeypatch.setattr(training.one_logger_utils, "get_timestamp_in_ms", lambda: 123)
    monkeypatch.setattr(training, "get_model", lambda *items, **kwargs: model)
    monkeypatch.setattr(training, "unwrap_model", lambda model: model)
    monkeypatch.setattr(training, "update_train_iters", lambda item: calls.append("update-iters"))

    returned_model, optimizer, scheduler = setup_model_and_optimizer(
        lambda: None,
        training.ModelType.encoder_or_decoder,
    )

    assert returned_model is model
    assert optimizer is None
    assert scheduler is None
    assert args.iteration == 0
    assert args.num_floating_point_operations_so_far == 0
    assert "update-iters" in calls


class TestTraining:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)

    def test_build_train_valid_test_data_iterators(self):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )
        train_data = next(train_iter)
        valid_data = next(valid_iter)
        test_data = next(test_iter)
        assert (train_data, valid_data, test_data) == (1, 2, 3)

    def test_closed_formula_vocab_size_with_padding(self):
        def old_round_impl(after, multiple):
            while (after % multiple) != 0:
                after += 1
            return after

        args = SimpleNamespace()
        args.rank = 0
        args.tensor_model_parallel_size = 1

        for vocab in range(1, 600000, 1000):
            for mult in [1, 17, 32, 64, 128]:
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

        for vocab in range(1, 10_000, 500):
            for mult in range(1, 1024 + 1):
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestSaveGrads:
    """Tests for the save_grads function."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_save_grads(self, tmp_path_dist_ckpt):
        """Test that save_grads creates the correct directory structure and saves
        state_dict correctly.

        With TP=1, PP=1 on 8 GPUs, we have 8 DP ranks. Only the rank with
        expert_data_parallel_rank==0 should save. All ranks verify the result.
        """
        save_dir = str(tmp_path_dist_ckpt / "test_save_grads")

        with TempNamedDir(save_dir, sync=True) as save_dir:
            # Create a mock state_dict with gradients (use deterministic values for reproducibility).
            state_dict = defaultdict(dict)
            state_dict["model_chunk0"]["layer.weight"] = torch.arange(16).reshape(4, 4).float()
            state_dict["model_chunk0"]["layer.bias"] = torch.arange(4).float()

            iteration = 100
            grad_label = "wgrads"

            # All ranks call save_grads, but only expert_data_parallel_rank==0 actually saves.
            save_grads(save_dir, dict(state_dict), iteration, grad_label)

            # Synchronize before checking results since only rank 0 saves.
            torch.distributed.barrier()

            # All ranks verify the file was created by rank 0.
            expected_dir = Path(save_dir) / grad_label / f"iter_{iteration:07d}"
            assert expected_dir.exists(), f"Expected directory {expected_dir} to exist"

            expected_file = expected_dir / "mp_rank_00.pth"
            assert expected_file.exists(), f"Expected file {expected_file} to exist"

            # Verify saved content.
            loaded = torch.load(expected_file)
            assert "model_chunk0" in loaded
            assert "layer.weight" in loaded["model_chunk0"]
            assert "layer.bias" in loaded["model_chunk0"]
            assert torch.equal(
                loaded["model_chunk0"]["layer.weight"], state_dict["model_chunk0"]["layer.weight"]
            )
            assert torch.equal(
                loaded["model_chunk0"]["layer.bias"], state_dict["model_chunk0"]["layer.bias"]
            )

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for config construction before training runtime initialization."""

import logging
import runpy
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer import TransformerConfig
from megatron.training import arguments, global_vars, initialize
from megatron.training.argument_utils import (
    gpt_config_from_args,
    hybrid_config_from_args,
    inference_cfg_container_from_args,
    model_config_from_args,
    pretrain_cfg_container_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.config import DistributedInitConfig
from megatron.training.config.training_config import TokenizerConfig, ValidationConfig
from megatron.training.models import (
    BertModelConfig,
    GPTModelConfig,
    LLaVAModelConfig,
    MimoModelConfig,
    T5ModelConfig,
)


@pytest.fixture
def isolated_globals(monkeypatch):
    """Avoid changing services owned by the distributed test harness."""
    for name in ("_GLOBAL_ARGS", "_GLOBAL_RUN_CONFIG", "_GLOBAL_TOKENIZER", "_GLOBAL_TRAIN_STATE"):
        monkeypatch.setattr(global_vars, name, None)


def _runtime_args():
    return Namespace(
        rank=0,
        world_size=2,
        global_batch_size=8,
        micro_batch_size=2,
        data_parallel_size=2,
        gtp_weight_remat_size=1,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        seq_length=32,
        padded_vocab_size=None,
        enable_experimental=False,
        exit_signal_handler=False,
        exit_signal_handler_for_training=False,
        disable_jit_fuser=False,
    )


@pytest.mark.parametrize(
    "source,expected_world_size",
    [("process_group", 12), ("environment", 8), ("slurm", 16), ("default", 1)],
)
def test_batch_finalization_uses_core_world_size(
    monkeypatch, run_config, source, expected_world_size
):
    # The container already exists; launch metadata is resolved at runtime, not captured in it.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: source == "process_group")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 12)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    if source in ("slurm", "default"):
        monkeypatch.delenv("WORLD_SIZE")
    if source == "default":
        monkeypatch.delenv("SLURM_NTASKS")
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    run_config.train.micro_batch_size = 2

    warning = (
        pytest.warns(UserWarning, match="Defaulting to world size 1")
        if source == "default"
        else nullcontext()
    )
    with warning:
        run_config.finalize()
    assert run_config.train.global_batch_size == 2 * expected_world_size
    assert run_config.validation.eval_global_batch_size == 2 * expected_world_size


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize(
    "overrides", [None, "overlap.yaml", {"qkv_fprop": {"method": "ring_exchange"}}]
)
def test_tp_user_buffer_overrides_reach_model_config(adapter, overrides):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.tp_comm_overlap_cfg = overrides
    config = adapter(
        args, config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)
    )
    assert config.tp_comm_overlap_cfg == overrides


@pytest.mark.parametrize("te_version", ["1.8.0", "1.9.0", "2.7.0"])
@pytest.mark.parametrize("fp8,bf16_layers", [(None, 0), ("hybrid", 0), ("hybrid", 1)])
@pytest.mark.parametrize("override_source", ["none", "dict", "file"])
def test_tp_user_buffers_use_config_and_explicit_batch_shape(
    monkeypatch, tmp_path, te_version, fp8, bf16_layers, override_source
):
    from transformer_engine.pytorch import module as te_module

    config = TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=8,
        tensor_model_parallel_size=2,
        context_parallel_size=2,
        fp8=fp8,
        fp8_recipe="tensorwise",
        first_last_layers_bf16=bool(bf16_layers),
        num_layers_at_start_in_bf16=bf16_layers,
        tp_comm_bootstrap_backend="gloo",
    )
    overrides = {"qkv_fprop": {"method": "ring_exchange"}}
    source = overrides if override_source == "dict" else None
    if override_source == "file":
        path = tmp_path / "overlap.yaml"
        path.write_text("qkv_fprop:\n  method: ring_exchange\n")
        source = str(path)
    initialize_ub, create_group = Mock(), Mock()
    modes = SimpleNamespace(NONE="none", FP8="fp8")
    monkeypatch.setattr(te_module.base, "initialize_ub", initialize_ub)
    monkeypatch.setattr(te_module.base, "UserBufferQuantizationMode", modes, raising=False)
    monkeypatch.setattr(initialize, "create_group", create_group)
    monkeypatch.setattr(
        initialize, "get_args", Mock(side_effect=AssertionError("global args read"))
    )
    monkeypatch.setattr(initialize, "get_te_version", lambda: te_version)
    monkeypatch.setattr(
        initialize,
        "is_te_min_version",
        lambda minimum: tuple(map(int, te_version.split(".")))
        >= tuple(map(int, minimum.split("."))),
    )
    warning = (
        pytest.warns(UserWarning, match="supports only MPI")
        if te_version == "1.8.0"
        else nullcontext()
    )
    with warning:
        initialize._initialize_tp_communicators(
            config, sequence_length=96, micro_batch_size=3, tp_comm_overlap_cfg=source
        )
    expected = dict(shape=[144, 128], tp_size=2, ub_cfgs={} if source is None else overrides)
    if te_version == "2.7.0":
        expected["quantization_modes"] = [modes.FP8 if fp8 else modes.NONE]
        if bf16_layers:
            expected["quantization_modes"].append(modes.NONE)
    else:
        expected["use_fp8"] = fp8 is not None
    if te_version == "1.8.0":
        create_group.assert_called_once_with(backend="mpi", group_desc="TP_BOOTSTRAP_GROUP_MPI")
    else:
        expected["bootstrap_backend"] = "gloo"
        create_group.assert_not_called()
    initialize_ub.assert_called_once_with(**expected)


def test_distributed_initialization_prefers_config_to_args(monkeypatch, run_config):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank, args.world_size = 0, 999
    # These topology values are normally resolved by validate_args.
    args.gtp_weight_remat_size = args.expert_gtp_weight_remat_size = 1
    args.gtp_remat_num_sequence_shards = 1
    args.virtual_pipeline_model_parallel_size = None
    args.expert_tensor_parallel_size = 1
    original_args = vars(args).copy()
    # Keep the legacy defaults different so an accidental args read is detected.
    monkeypatch.setenv("WORLD_SIZE", "4")
    run_config.dist = DistributedInitConfig(
        distributed_backend="gloo",
        distributed_timeout_minutes=37,
        local_rank=1,
        use_sharp=True,
        sharp_enabled_group="dp_replica",
        use_gloo_process_groups=False,
        high_priority_stream_groups=["tp"],
        use_tp_pp_dp_mapping=True,
        nccl_communicator_config_path="owned.yaml",
    )
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "print_rank_0", Mock())
    monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 2)
    set_device = Mock()
    monkeypatch.setattr(initialize.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(initialize.torch.distributed, "is_initialized", lambda: False)
    init_pg, init_mpu = Mock(), Mock()
    monkeypatch.setattr(initialize.torch.distributed, "init_process_group", init_pg)
    monkeypatch.setattr(initialize.inprocess_restart, "maybe_force_nccl_backend_init", Mock())
    monkeypatch.setattr(initialize.mpu, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(initialize.mpu, "initialize_model_parallel", init_mpu)
    for name in (
        "get_tensor_model_parallel_world_size",
        "get_pipeline_model_parallel_world_size",
        "get_gtp_weight_remat_world_size",
    ):
        monkeypatch.setattr(initialize.mpu, name, lambda: 1)
    for name in ("TORCH_FR_DUMP_TEMP_FILE", "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"):
        monkeypatch.delenv(name, raising=False)
    store = object()

    initialize._initialize_distributed(None, None, store)

    set_device.assert_called_once_with(1)
    init_pg.assert_called_once_with(
        backend="gloo", store=store, world_size=4, rank=0, timeout=timedelta(minutes=37)
    )
    kwargs = init_mpu.call_args.kwargs
    assert kwargs["distributed_timeout_minutes"] == 37
    assert kwargs["nccl_communicator_config_path"] == "owned.yaml"
    assert kwargs["order"] == "tp-cp-ep-pp-dp"
    assert kwargs["high_priority_stream_groups"] == ["tp"]
    assert kwargs["use_sharp"] is True
    assert kwargs["gtp_remat_num_sequence_shards"] == 1
    assert kwargs["sharp_enabled_group"] == "dp_replica"
    assert kwargs["create_gloo_process_groups"] is False
    assert vars(args) == original_args


@pytest.mark.parametrize("experimental", [False, True])
def test_parse_only_prepares_args(monkeypatch, isolated_globals, experimental):
    args = Namespace(use_checkpoint_args=False, yaml_cfg=None, enable_experimental=experimental)
    monkeypatch.setattr(arguments, "parse_args", Mock(return_value=args))
    monkeypatch.setattr(arguments, "validate_args", Mock())
    services = Mock()
    experimental_flag = Mock()
    monkeypatch.setattr(global_vars, "initialize_runtime_services", services)
    monkeypatch.setattr(arguments, "set_experimental_flag", experimental_flag)

    assert arguments.parse_and_validate_args() is args
    assert global_vars.get_args() is args
    assert global_vars._GLOBAL_TRAIN_STATE is None
    services.assert_not_called()
    if experimental:
        experimental_flag.assert_called_once_with(True)
    else:
        experimental_flag.assert_not_called()

    replacement_args = deepcopy(args)
    monkeypatch.setattr(arguments, "parse_args", Mock(return_value=replacement_args))
    with pytest.raises(AssertionError, match="args is already initialized"):
        arguments.parse_and_validate_args()
    assert global_vars.get_args() is args
    services.assert_not_called()


@pytest.mark.parametrize("aliases", [None, False, True])
def test_explicit_inference_bootstrap_preserves_sparse_args(monkeypatch, isolated_globals, aliases):
    args = _runtime_args()
    # Sparse legacy namespaces must not acquire new checkpoint requirements.
    if aliases is not None:
        for name in ("save_optim", "save_rng", "load_optim", "load_rng"):
            setattr(args, f"no_{name}", aliases)
        args.ckpt_fully_parallel_save = aliases
        args.ckpt_fully_parallel_load = aliases
        args.profile = aliases
    initialize = Mock()
    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    global_vars.set_global_variables(args, cfg)
    assert global_vars.get_args() is args
    assert global_vars.get_run_config() is cfg
    assert cfg.model is None
    assert cfg.checkpoint.save_optim is (not aliases)
    assert cfg.checkpoint.save_rng is (not aliases)
    assert cfg.checkpoint.load_optim is (not aliases)
    assert cfg.checkpoint.load_rng is (not aliases)
    assert cfg.checkpoint.fully_parallel_save is (True if aliases is None else aliases)
    assert cfg.checkpoint.fully_parallel_load is bool(aliases)
    assert cfg.profiling.use_nsys_profiler is bool(aliases)
    initialize.assert_called_once_with(args, build_tokenizer=True, training=False)
    with pytest.raises(AssertionError, match="already initialized"):
        global_vars.set_global_variables(args, cfg)


def test_explicit_training_bootstrap_preserves_config(monkeypatch, isolated_globals, run_config):
    args = _runtime_args()
    cfg = run_config
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    cfg.train.train_iters = 7
    cfg.train.micro_batch_size = 2
    cfg.logger.log_interval = 3
    initialize = Mock()
    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)

    global_vars.set_global_variables(args, cfg, build_tokenizer=False)

    assert global_vars.get_args() is args
    assert global_vars.get_run_config() is cfg
    assert global_vars.get_run_config().train.train_iters == 7
    assert global_vars.get_run_config().logger.log_interval == 3
    initialize.assert_called_once_with(args, build_tokenizer=False, training=True)


def test_bootstrap_rejects_registered_config_before_setting_args(monkeypatch, isolated_globals):
    args = _runtime_args()
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    global_vars.set_run_config(cfg)
    initialize = Mock()
    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)

    with pytest.raises(AssertionError, match="run config is already initialized"):
        global_vars.set_global_variables(args, cfg)

    assert global_vars._GLOBAL_ARGS is None
    assert global_vars.get_run_config() is cfg
    initialize.assert_not_called()


def test_parse_restores_checkpoint_args_before_validation_and_config(monkeypatch, isolated_globals):
    from megatron.training import checkpointing

    args = Namespace(
        use_checkpoint_args=True,
        yaml_cfg=None,
        enable_experimental=False,
        load="resume",
        pretrained_checkpoint="pretrained",
        non_persistent_ckpt_type=None,
        padded_vocab_size=None,
    )
    events = []
    monkeypatch.setattr(arguments, "parse_args", Mock(return_value=args))

    def restore(received_args, load_arg="load"):
        assert received_args is args
        events.append(load_arg)
        if load_arg == "load":
            received_args.padded_vocab_size = 512

    def validate(received_args, defaults):
        assert received_args.padded_vocab_size == 512
        events.append("validate")

    monkeypatch.setattr(checkpointing, "load_args_from_checkpoint", restore)
    monkeypatch.setattr(arguments, "validate_args", validate)
    runtime = Mock()
    monkeypatch.setattr(global_vars, "initialize_runtime_services", runtime)

    assert arguments.parse_and_validate_args() is args
    assert events == ["pretrained_checkpoint", "load", "validate"]
    assert global_vars.get_args().padded_vocab_size == 512
    runtime.assert_not_called()


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("checkpoint_vocab", [None, 256])
def test_vocabulary_resolves_after_config_construction(
    monkeypatch, isolated_globals, adapter, checkpoint_vocab
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.padded_vocab_size = checkpoint_vocab
    args.vocab_size = None
    model_cfg = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size_from_tokenizer=True,
    )
    assert model_cfg.vocab_size == checkpoint_vocab
    cfg = SimpleNamespace(model=model_cfg, tokenizer=TokenizerConfig())
    global_vars.set_args(args)

    def initialize_services(received_args):
        # Config construction has finished without creating a tokenizer.
        assert cfg.model.vocab_size == checkpoint_vocab
        assert received_args is args
        if received_args.padded_vocab_size is None:
            received_args.padded_vocab_size = 128

    initialize = Mock(side_effect=initialize_services)
    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)
    global_vars.initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)

    assert model_cfg.vocab_size == (128 if checkpoint_vocab is None else checkpoint_vocab)
    assert model_cfg.should_pad_vocab is False
    assert cfg.tokenizer.padded_vocab_size == model_cfg.vocab_size
    initialize.assert_called_once_with(args)


@pytest.mark.parametrize("use_bootstrap", [False, True])
def test_runtime_service_order_and_microbatch_inputs(
    monkeypatch, isolated_globals, use_bootstrap, run_config
):
    args = _runtime_args()
    cfg = run_config
    monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 2)
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 2
    # Runtime services must use the container even when legacy args disagree.
    args.global_batch_size = 99
    args.micro_batch_size = 99
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    cfg.model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
        seq_length=32,
        vocab_size=128,
    )
    if not use_bootstrap:
        global_vars.set_args(args)
        global_vars.set_run_config(cfg)
    calls = []
    microbatches = Mock(side_effect=lambda **kwargs: calls.append("microbatches"))
    monkeypatch.setattr(global_vars, "init_num_microbatches_calculator", microbatches)

    for name in (
        "_build_tokenizer",
        "_set_tensorboard_writer",
        "_set_wandb_writer",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_timers",
        "_set_energy_monitor",
        "_set_telemetry",
    ):

        def record(received_args, service=name, **kwargs):
            assert received_args is args
            expected = {
                "_set_telemetry": {"include_training": True},
                "_build_tokenizer": {
                    "tensor_model_parallel_size": 1,
                    "make_vocab_size_divisible_by": 128,
                },
            }
            assert kwargs == expected.get(service, {})
            calls.append(service)

        monkeypatch.setattr(global_vars, name, record)

    initialize_train_state = global_vars._set_train_state

    def record_train_state():
        assert global_vars.get_args() is args
        calls.append("_set_train_state")
        initialize_train_state()

    monkeypatch.setattr(global_vars, "_set_train_state", record_train_state)
    assert global_vars._GLOBAL_TRAIN_STATE is None
    if use_bootstrap:
        global_vars.set_global_variables(args, cfg)
    else:
        global_vars.initialize_runtime_services(args, training=True)
    assert isinstance(global_vars.get_train_state(), global_vars.TrainState)
    assert calls == [
        "_build_tokenizer",
        "_set_wandb_writer",
        "_set_telemetry",
        "microbatches",
        "_set_tensorboard_writer",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_timers",
        "_set_energy_monitor",
        "_set_train_state",
    ]
    microbatches.assert_called_once_with(
        rank=0,
        global_batch_size=8,
        micro_batch_size=2,
        data_parallel_size=2,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        seq_length=32,
    )


def test_custom_model_config_does_not_receive_vocabulary(monkeypatch, isolated_globals):
    custom_model = SimpleNamespace()
    cfg = SimpleNamespace(model=custom_model, tokenizer=TokenizerConfig())
    resolve_tokenizer_vocab_size(cfg, 128)
    assert vars(custom_model) == {}
    assert cfg.tokenizer.padded_vocab_size == 128


@pytest.mark.parametrize(
    "world_size, mimo_llm_world_size, parallel_kwargs, expected_dp, expected_batch",
    [
        (8, None, {"tensor_model_parallel_size": 2}, 4, 4),
        (16, 8, {"tensor_model_parallel_size": 2}, 4, 4),
        (
            32,
            None,
            {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 2,
                "pipeline_dtype": torch.float32,
                "context_parallel_size": 2,
            },
            4,
            4,
        ),
        (
            16,
            None,
            {
                "tensor_model_parallel_size": 2,
                "tensor_parallel_num_weight_shards": 8,
                "sequence_parallel": True,
                "tensor_parallel_num_sequence_shards": 4,
                "context_parallel_size": 4,
            },
            1,
            2,
        ),
        (
            64,
            32,
            {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 2,
                "pipeline_dtype": torch.float32,
                "tensor_parallel_num_weight_shards": 8,
                "sequence_parallel": True,
                "tensor_parallel_num_sequence_shards": 4,
                "context_parallel_size": 4,
            },
            1,
            2,
        ),
    ],
)
def test_training_batch_topology_uses_model_config(
    monkeypatch,
    run_config,
    world_size,
    mimo_llm_world_size,
    parallel_kwargs,
    expected_dp,
    expected_batch,
):
    from examples.mimo.training.builder import MimoBuildConfig

    args = _runtime_args()
    args.world_size = args.mimo_llm_world_size = 999
    args.data_parallel_size = args.batch_parallel_size = 999
    original_args = vars(args).copy()
    monkeypatch.setattr(
        "megatron.training.config.container.safe_get_world_size", lambda: world_size
    )
    monkeypatch.setattr(global_vars, "safe_get_world_size", lambda: world_size)
    run_config.train.micro_batch_size = 2
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    run_config.model = MimoBuildConfig(
        transformer=TransformerConfig(
            num_layers=2, hidden_size=32, num_attention_heads=4, **parallel_kwargs
        ),
        seq_length=32,
        vocab_size=128,
        mimo_llm_world_size=mimo_llm_world_size,
    )
    microbatches = Mock()
    monkeypatch.setattr(global_vars, "init_num_microbatches_calculator", microbatches)
    for name in (
        "_set_wandb_writer",
        "_set_telemetry",
        "_set_tensorboard_writer",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_timers",
        "_set_energy_monitor",
        "_set_train_state",
    ):
        monkeypatch.setattr(global_vars, name, Mock())
    monkeypatch.setattr(global_vars, "_detect_gpu_identity", Mock(return_value={}))
    telemetry = []
    monkeypatch.setattr(
        global_vars,
        "_set_telemetry",
        lambda args, **kwargs: telemetry.append(global_vars.build_telemetry_resource_attrs(args)),
    )
    run_config.finalize()
    global_vars.initialize_runtime_services(args, build_tokenizer=False, training=True)
    assert vars(args) == original_args
    assert run_config.get_model_world_size() == (mimo_llm_world_size or world_size)
    assert run_config.get_data_parallel_size() == expected_dp
    assert run_config.get_batch_parallel_size() == expected_batch
    assert microbatches.call_args.kwargs['data_parallel_size'] == expected_batch
    assert telemetry[0]['dl.data_parallel.size'] == expected_dp
    assert telemetry[0]['dl.tensor_parallel.size'] == 2
    assert telemetry[0]['megatron.hidden_size'] == 32
    assert telemetry[0] == global_vars.build_telemetry_resource_attrs(args)


@pytest.mark.parametrize("known_vocab", [None, 512])
@pytest.mark.parametrize(
    "model_cls", [BertModelConfig, T5ModelConfig, LLaVAModelConfig, MimoModelConfig]
)
def test_callback_model_config_owns_shared_inputs(
    monkeypatch, isolated_globals, run_config, known_vocab, model_cls
):
    """Callback applications keep a typed model without selecting a GPT/Hybrid builder."""
    import megatron.training.argument_utils as adapters
    from megatron.training.config import PretrainConfigContainer

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.max_position_embeddings = 2048
    args.mimo_llm_world_size = 8
    args.encoder_num_layers = 3
    args.padded_vocab_size = known_vocab
    transformer = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)
    convert = Mock(return_value=transformer)
    monkeypatch.setattr(adapters, "core_transformer_config_from_args", convert)
    monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 16)

    model = model_config_from_args(args, model_config_cls=model_cls)
    convert.assert_called_once_with(args)
    cfg = run_config
    cfg.model = model
    resolve_tokenizer_vocab_size(cfg, 256)
    args.padded_vocab_size = 999
    args.max_position_embeddings = 32

    assert type(cfg.model) is model_cls
    assert cfg.model.transformer is transformer
    assert cfg.model.seq_length == 2048
    if model_cls is MimoModelConfig:
        assert cfg.model.mimo_llm_world_size == 8
        assert cfg.get_model_world_size() == 8
    else:
        assert not hasattr(cfg.model, "mimo_llm_world_size")
        assert cfg.get_model_world_size() == 16
    if model_cls is BertModelConfig:
        assert cfg.model.bert_binary_head == args.bert_binary_head
        assert not hasattr(cfg.model, "hybrid_layer_pattern")
        assert not hasattr(cfg.model, "rotary_percent")
    if model_cls is T5ModelConfig:
        assert cfg.model.encoder_num_layers == 3
        assert not hasattr(cfg.model, "hybrid_layer_pattern")
    assert cfg.model.vocab_size == (256 if known_vocab is None else known_vocab)
    restored = PretrainConfigContainer.from_dict(cfg.to_dict())
    assert type(restored.model) is model_cls
    assert restored.to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    "section, name, value, error",
    [
        ("checkpoint", "ckpt_format", "torch", "requires torch_dist"),
        ("dist", "use_torch_fsdp2", True, "not FSDP"),
        ("ddp", "use_megatron_fsdp", True, "not FSDP"),
        ("dist", "use_tp_pp_dp_mapping", True, "TP-PP-DP mapping"),
    ],
)
def test_native_gtp_sequence_sharding_validates_cross_config_inputs(
    run_config, section, name, value, error
):
    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            tensor_parallel_num_weight_shards=8,
            sequence_parallel=True,
            tensor_parallel_num_sequence_shards=4,
            context_parallel_size=4,
        ),
        seq_length=32,
    )
    run_config.validate()
    owner = getattr(run_config, section)
    assert hasattr(owner, name)
    setattr(owner, name, value)
    with pytest.raises(ValueError, match=error):
        run_config.validate()


def test_callback_upcycling_uses_temporary_dense_config(monkeypatch, run_config):
    from megatron.training import training

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.skip_train = True
    run_config.validation.skip_train = True
    args.moe_use_upcycling = True
    args.moe_upcycling_granularity = 2
    args.num_experts = 2
    args.expert_model_parallel_size = 1
    args.ffn_hidden_size = 64
    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            ffn_hidden_size=64,
            num_moe_experts=2,
        ),
        seq_length=16,
        vocab_size=128,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "unwrap_model", lambda model: model)
    monkeypatch.setattr(training, "checkpoint_exists", lambda path: False)
    monkeypatch.setattr(torch.distributed, "barrier", Mock())
    built_configs = []

    class ReachedDenseModel(Exception):
        pass

    def build(*unused, config, **kwargs):
        built_configs.append(config)
        if len(built_configs) == 2:
            raise ReachedDenseModel
        return [SimpleNamespace()]

    monkeypatch.setattr(training, "get_model", build)
    with pytest.raises(ReachedDenseModel):
        training.setup_model_and_optimizer(
            Mock(), model_provider_func=Mock(), cfg_container=run_config
        )
    assert built_configs[0] is run_config.model.transformer
    assert built_configs[1].num_moe_experts is None
    assert built_configs[1].expert_model_parallel_size == 1
    assert built_configs[1].ffn_hidden_size == 128
    assert run_config.model.transformer.num_moe_experts == 2
    assert run_config.model.transformer.ffn_hidden_size == 64


@pytest.mark.parametrize("family", ["bert", "t5"])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("binary_head", [False, True])
def test_callbacks_construct_from_registered_model_config(
    monkeypatch, isolated_globals, family, inference, binary_head
):
    """Direct callbacks, including converters that bypass get_model, use the same owner."""
    import importlib

    provider = importlib.import_module(f"examples.{family}.pretrain_{family}")
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.num_layers, args.hidden_size, args.num_attention_heads = 2, 32, 4
    args.max_position_embeddings, args.padded_vocab_size = 64, 128
    args.params_dtype = torch.float32
    args.main_grads_dtype = args.main_params_dtype = torch.float32
    args.exp_avg_dtype = args.exp_avg_sq_dtype = torch.float32
    args.encoder_num_layers = 2
    args.relative_attention_num_buckets = 32
    args.relative_attention_max_distance = 128
    args.bert_binary_head = binary_head
    model_cls = BertModelConfig if family == "bert" else T5ModelConfig
    model = model_config_from_args(args, model_config_cls=model_cls)
    cfg = (
        inference_cfg_container_from_args(args, model)
        if inference
        else pretrain_cfg_container_from_args(args, model)
    )
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    monkeypatch.setattr(provider, "get_args", lambda: args)
    build = Mock()
    monkeypatch.setattr(provider, "BertModel" if family == "bert" else "T5Model", build)
    for name in (
        "hidden_size",
        "num_layers",
        "padded_vocab_size",
        "max_position_embeddings",
        "untie_embeddings_and_output_weights",
        "fp16_lm_cross_entropy",
        "position_embedding_type",
        "rotary_percent",
        "bert_binary_head",
        "encoder_num_layers",
        "relative_attention_num_buckets",
        "relative_attention_max_distance",
    ):
        delattr(args, name)

    provider.model_provider()

    assert build.call_args.kwargs["config"] is cfg.model.transformer
    assert build.call_args.kwargs["vocab_size"] == 128
    assert build.call_args.kwargs["max_sequence_length"] == 64
    assert build.call_args.kwargs["share_embeddings_and_output_weights"] is True
    if family == "bert":
        assert build.call_args.kwargs["add_binary_head"] is binary_head
        assert build.call_args.kwargs["num_tokentypes"] == (2 if binary_head else 0)
        batch = tuple(object() for _ in range(6))
        monkeypatch.setattr(provider, "get_batch", lambda _: batch)
        monkeypatch.setattr(provider, "get_timers", Mock())
        forward = Mock()
        provider.forward_step(None, forward)
        assert forward.call_args.kwargs["tokentype_ids"] is (batch[1] if binary_head else None)

        dataset_config = Mock()
        monkeypatch.setattr(provider, "BERTMaskedWordPieceDatasetConfig", dataset_config)
        monkeypatch.setattr(provider, "build_tokenizer", Mock())
        dataset_builder = Mock()
        dataset_builder.return_value.build.return_value = (None, None, None)
        monkeypatch.setattr(provider, "BlendedMegatronDatasetBuilder", dataset_builder)
        provider.train_valid_test_datasets_provider([1, 1, 1])
        assert dataset_config.call_args.kwargs["classification_head"] is binary_head
    else:
        assert build.call_args.kwargs["encoder_config"].num_layers == 2
        assert build.call_args.kwargs["relative_attention_num_buckets"] == 32
        assert build.call_args.kwargs["relative_attention_max_distance"] == 128


def test_model_precision_is_derived_centrally(monkeypatch, run_config):
    from megatron.core.optimizer import OptimizerConfig

    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 1)
    run_config.train.micro_batch_size = 1

    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
        ),
        seq_length=64,
    )
    run_config.finalize()
    runtime_optimizer = OptimizerConfig()
    runtime_optimizer = run_config.finalize_optimizer_config(runtime_optimizer)
    for config in (run_config.optimizer, runtime_optimizer):
        assert config.bf16 and not config.fp16
        assert config.params_dtype is torch.bfloat16

    # The recipe also changes an optimizer-derived precision flag, not just metadata.
    run_config.model.transformer.fp8_recipe = "tensorwise"
    runtime_optimizer = OptimizerConfig(use_precision_aware_optimizer=True, fp8_recipe="delayed")
    assert runtime_optimizer.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    runtime_optimizer = run_config.finalize_optimizer_config(runtime_optimizer)
    assert runtime_optimizer.fp8_recipe == "tensorwise"
    assert not runtime_optimizer.use_precision_aware_optimizer_no_fp8_or_ds_fp8


def test_llava_callback_consumes_its_model_config(monkeypatch, isolated_globals, run_config):
    import pretrain_vlm

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    pretrain_vlm.add_vlm_extra_args(parser)
    args = parser.parse_args([])
    args.num_layers, args.hidden_size, args.num_attention_heads = 2, 32, 4
    args.encoder_num_layers = 1
    args.max_position_embeddings, args.padded_vocab_size = 64, 128
    args.seq_length, args.decoder_seq_length = 16, 32
    args.params_dtype = torch.float32
    args.ckpt_format = "torch"
    cfg = run_config
    cfg.model = model_config_from_args(args, model_config_cls=LLaVAModelConfig)
    cfg.model.rotary_base = 12345
    cfg.model.rope_scaling = True
    monkeypatch.setattr(pretrain_vlm, "get_args", lambda: args)
    monkeypatch.setattr(pretrain_vlm, "get_num_image_embeddings", lambda *a, **kw: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(pretrain_vlm.context_parallel, "get_padding", lambda *a: 0)
    monkeypatch.setattr(pretrain_vlm.parallel_state, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(pretrain_vlm.parallel_state, "is_pipeline_last_stage", lambda: True)
    build = Mock()
    monkeypatch.setattr(pretrain_vlm, "LLaVAModel", build)
    args.rotary_base = 999
    args.use_rope_scaling = False
    args.padded_vocab_size = 999

    pretrain_vlm.model_provider()

    assert build.call_args.kwargs["language_vocab_size"] == 128
    assert build.call_args.kwargs["language_rotary_base"] == 12345
    assert build.call_args.kwargs["language_rope_scaling"] is True
    assert build.call_args.kwargs["language_transformer_config"].hidden_size == 32
    assert build.call_args.kwargs["vision_transformer_config"].num_layers == 1
    assert cfg.model.transformer.num_layers == 2


@pytest.mark.parametrize("eval_batch_sizes", [None, (16, 4)])
@pytest.mark.parametrize("schedule", [None, "0:8 32:16"])
@pytest.mark.parametrize("rl", [False, True])
def test_native_batch_defaults_preserve_explicit_eval_sizes(
    monkeypatch, run_config, eval_batch_sizes, schedule, rl
):
    run_config.train.micro_batch_size = 2
    run_config.train.step_batch_size_schedule = schedule
    run_config.rl.perform_rl_step = rl
    if eval_batch_sizes is not None:
        (
            run_config.validation.eval_global_batch_size,
            run_config.validation.eval_micro_batch_size,
        ) = eval_batch_sizes
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    before = deepcopy(run_config.to_dict())
    run_config.validate()
    run_config.validate()
    assert run_config.to_dict() == before
    monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 2)
    run_config.finalize()
    # Runtime re-entry must not apply derivations a second time.
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    run_config.finalize()
    assert run_config.train.global_batch_size == 4
    assert (
        run_config.validation.eval_global_batch_size,
        run_config.validation.eval_micro_batch_size,
    ) == (eval_batch_sizes or (4, 2))


@pytest.mark.parametrize("mode", ["constant", "decrease", "schedule"])
def test_training_batch_divisibility_preserves_calculator_modes(monkeypatch, run_config, mode):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 2)
    run_config.train.micro_batch_size = 2
    run_config.train.global_batch_size = 10
    run_config.train.decrease_batch_size_if_needed = mode == "decrease"
    run_config.train.step_batch_size_schedule = "0:10 32:16" if mode == "schedule" else None
    run_config.validation.eval_global_batch_size = 8
    before = deepcopy(run_config.to_dict())
    if mode == "constant":
        with pytest.raises(AssertionError, match="Training global batch size must be divisible"):
            run_config.finalize()
        assert not run_config._finalized
        assert run_config.to_dict() == before
    else:
        run_config.finalize()
        assert run_config.train.global_batch_size == 10
        assert run_config.validation.eval_micro_batch_size == 2


@pytest.mark.parametrize("invalid_policy", ["eval", "rl"])
def test_derived_batch_validation_precedes_config_mutation(monkeypatch, run_config, invalid_policy):
    monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 2)
    run_config.train.micro_batch_size = 3
    if invalid_policy == "eval":
        run_config.validation.eval_micro_batch_size = 4
        error, message = AssertionError, "Evaluation global batch size must be divisible"
    else:
        run_config.rl.perform_rl_step = True
        error, message = ValueError, "divisible by global_batch_size"
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    before = deepcopy(run_config.to_dict())
    run_config.validate()
    assert run_config.to_dict() == before
    with pytest.raises(error, match=message):
        run_config.finalize()
    assert run_config.to_dict() == before
    assert not run_config._finalized


@pytest.mark.parametrize("world_size, mimo_llm_world_size", [(0, None), (-1, None), (4, 8), (4, 0)])
def test_training_topology_rejects_invalid_rank_counts(
    monkeypatch, run_config, world_size, mimo_llm_world_size
):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    monkeypatch.setattr(
        "megatron.training.config.container.safe_get_world_size", lambda: world_size
    )
    run_config.model = MimoModelConfig(
        transformer=run_config.model.transformer,
        seq_length=run_config.model.seq_length,
        mimo_llm_world_size=mimo_llm_world_size,
    )
    run_config.train.micro_batch_size = 1
    before = deepcopy(run_config.to_dict())
    with pytest.raises(ValueError, match="Training model world size must fit"):
        run_config.finalize()
    assert run_config.to_dict() == before
    assert not run_config._finalized


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"tensor_model_parallel_size": 0}, "Parallel sizes must be positive"),
        ({"pipeline_model_parallel_size": 0}, "Parallel sizes must be positive"),
        ({"context_parallel_size": 0}, "Parallel sizes must be positive"),
        ({"gtp_weight_remat_size": 0}, "Parallel sizes must be positive"),
        ({"tensor_parallel_num_sequence_shards": 0}, "must be positive"),
        (
            {"tensor_parallel_num_sequence_shards": 2, "gtp_weight_remat_size": 2},
            "Runtime context-parallel size",
        ),
        (
            {
                "context_parallel_size": 2,
                "gtp_weight_remat_size": 3,
                "tensor_parallel_num_sequence_shards": 2,
            },
            "must divide",
        ),
        ({"tensor_model_parallel_size": 3}, "world_size .* must be divisible"),
    ],
)
def test_parallel_size_helpers_reject_invalid_topology(monkeypatch, run_config, overrides, error):
    monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 8)
    for name, value in overrides.items():
        monkeypatch.setattr(run_config.model.transformer, name, value)
    before = deepcopy(run_config.to_dict())
    for helper in (run_config.get_data_parallel_size, run_config.get_batch_parallel_size):
        with pytest.raises(ValueError, match=error):
            helper()
    assert run_config.to_dict() == before


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("pad_vocab,checkpoint_vocab", [(True, None), (True, 512), (False, None)])
def test_config_first_matches_legacy_tokenizer_and_model_inputs(
    monkeypatch, isolated_globals, adapter, pad_vocab, checkpoint_vocab
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 133
    args.pad_vocab_size = pad_vocab
    args.padded_vocab_size = checkpoint_vocab
    args.tensor_model_parallel_size = 2
    transformer = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)

    legacy_args = deepcopy(args)
    legacy_tokenizer = build_tokenizer(legacy_args)
    legacy_model = adapter(legacy_args, config=deepcopy(transformer))

    model = adapter(args, config=deepcopy(transformer), vocab_size_from_tokenizer=True)
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    global_vars.set_args(args)

    def initialize(received_args):
        global_vars._build_tokenizer(received_args)

    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)
    global_vars.initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)
    assert model.as_dict() == legacy_model.as_dict()
    assert cfg.tokenizer.padded_vocab_size == legacy_args.padded_vocab_size
    assert global_vars.get_tokenizer().vocab_size == legacy_tokenizer.vocab_size


@pytest.mark.parametrize(
    "adapter", [gpt_config_from_args, hybrid_config_from_args, BertModelConfig, LLaVAModelConfig]
)
@pytest.mark.parametrize("pad_vocab", [False, True])
@pytest.mark.parametrize("padded_vocab", [None, 768])
def test_training_tokenizer_padding_uses_model_inputs(
    monkeypatch, isolated_globals, adapter, pad_vocab, padded_vocab
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank, args.world_size = 0, 2
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 129
    args.pad_vocab_size = pad_vocab
    args.padded_vocab_size = padded_vocab
    args.max_position_embeddings = 64
    args.tensor_model_parallel_size = 1
    args.make_vocab_size_divisible_by = 128
    transformer = TransformerConfig(
        num_layers=2, hidden_size=128, num_attention_heads=4, tensor_model_parallel_size=2
    )
    if adapter in (BertModelConfig, LLaVAModelConfig):
        model = adapter(
            transformer=transformer,
            seq_length=64,
            vocab_size=padded_vocab if padded_vocab is not None or pad_vocab else args.vocab_size,
        )
    else:
        model = adapter(args, config=transformer)
    model.make_vocab_size_divisible_by = 256
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig(), dist=DistributedInitConfig())
    global_vars.set_args(args)
    global_vars.set_run_config(cfg)
    for service in ("_set_wandb_writer", "_set_telemetry", "initialize_training_runtime_services"):
        monkeypatch.setattr(global_vars, service, Mock())

    global_vars.initialize_runtime_services(args, training=True)
    resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)

    expected = padded_vocab if padded_vocab is not None else (512 if pad_vocab else None)
    assert args.padded_vocab_size == cfg.tokenizer.padded_vocab_size == expected
    assert cfg.model.vocab_size == (129 if expected is None else expected)
    assert global_vars.get_tokenizer().vocab_size == 129
    assert args.tensor_model_parallel_size == 1
    assert args.make_vocab_size_divisible_by == 128


@pytest.mark.parametrize("share_embeddings", [False, True])
def test_gpt_embedding_groups_use_registered_model(monkeypatch, isolated_globals, share_embeddings):
    import pretrain_gpt

    cfg = SimpleNamespace(
        model=GPTModelConfig(
            transformer=TransformerConfig(
                num_layers=3,
                hidden_size=128,
                num_attention_heads=4,
                pipeline_model_parallel_size=3,
                pipeline_dtype=torch.float32,
                pipeline_model_parallel_layout="Et*3|m|L",
                mtp_num_layers=1,
            ),
            seq_length=64,
            share_embeddings_and_output_weights=share_embeddings,
        )
    )
    global_vars.set_run_config(cfg)
    global_vars.set_args(Namespace(untie_embeddings_and_output_weights=share_embeddings))

    assert pretrain_gpt.get_embedding_ranks([0, 2, 4]) == (
        [0, 2, 4] if share_embeddings else [0, 2]
    )
    assert pretrain_gpt.get_embedding_ranks([7]) == [7]


@pytest.mark.parametrize("module_name", ["pretrain_gpt", "pretrain_hybrid"])
@pytest.mark.parametrize("sequence_packing_scheduler", [None, "dp_balanced"])
def test_dataset_helpers_use_registered_model(
    monkeypatch, isolated_globals, run_config, module_name, sequence_packing_scheduler
):
    import importlib

    entrypoint = importlib.import_module(module_name)
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.seq_length = 32
    args.data_parallel_size = 999
    args.sequence_packing_scheduler = None if sequence_packing_scheduler else "dp_balanced"
    transformer = TransformerConfig(
        num_layers=3,
        hidden_size=128,
        num_attention_heads=4,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        context_parallel_size=2,
        hybrid_context_parallel=True,
        pipeline_model_parallel_size=3,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_layout="Et*3|m|L",
        mtp_num_layers=1,
    )
    transformer.sequence_packing_scheduler = sequence_packing_scheduler
    cfg = run_config
    cfg.model = GPTModelConfig(transformer=transformer, seq_length=64)
    # TP=2, PP=3, CP=2 gives DP=2 independently of the stale CLI topology.
    monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 24)
    global_vars.set_args(args)
    monkeypatch.setattr(entrypoint, "build_tokenizer", Mock())
    monkeypatch.setattr(entrypoint, "get_blend_and_blend_per_split", lambda args: (None, None))
    monkeypatch.setattr(entrypoint, "GPTDatasetConfig", SimpleNamespace)

    dataset = entrypoint.core_gpt_dataset_config_from_args(args)
    assert dataset.context_parallel_size == 2
    assert dataset.sequence_parallel_size == 2
    assert dataset.hybrid_context_parallel is True
    assert dataset.data_parallel_size == 2
    assert dataset.sequence_length == 32  # Actual data length, not position capacity.

    monkeypatch.setattr(entrypoint.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(entrypoint.mpu, "get_pipeline_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        entrypoint.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None
    )
    monkeypatch.setattr(entrypoint, "is_first_or_last_pipeline_stage", lambda stage: False)
    assert entrypoint.is_dataset_built_on_rank() is (sequence_packing_scheduler is None)
    assert entrypoint.is_dataset_built_on_rank(is_packed_sequence=True) is (
        sequence_packing_scheduler is None
    )
    assert args.data_parallel_size == 999
    assert args.context_parallel_size == args.tensor_model_parallel_size == 1
    assert args.mtp_num_layers is None


def test_explicit_model_vocabulary_is_not_overwritten(monkeypatch, isolated_globals):
    from megatron.training.models import GPTModelConfig

    args = _runtime_args()
    args.padded_vocab_size = 128
    global_vars.set_args(args)
    monkeypatch.setattr(global_vars, "initialize_runtime_services", Mock())
    model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size=512,
        should_pad_vocab=False,
    )
    resolve_tokenizer_vocab_size(
        SimpleNamespace(model=model, tokenizer=TokenizerConfig()), args.padded_vocab_size
    )
    assert model.vocab_size == 512
    assert model.should_pad_vocab is False


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
def test_missing_tokenizer_vocabulary_fails_before_model_construction(adapter):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    model = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size_from_tokenizer=True,
    )
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    with pytest.raises(ValueError, match="vocabulary must be resolved"):
        resolve_tokenizer_vocab_size(cfg, None)


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("raw_vocab", [None, 133])
@pytest.mark.parametrize("padded_vocab_present", [False, True])
@pytest.mark.parametrize("vocab_kwargs", [{}, {"vocab_size_from_tokenizer": True}])
def test_tokenizer_vocabulary_is_authoritative_when_padding(
    monkeypatch, isolated_globals, adapter, raw_vocab, padded_vocab_present, vocab_kwargs
):
    """Tokenizer metadata/special tokens may differ from an optional CLI size."""
    from megatron.core.tokenizers import MegatronTokenizer

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.tokenizer_type = "HuggingFaceTokenizer"
    args.vocab_size = raw_vocab
    args.tensor_model_parallel_size = 2
    args.make_vocab_size_divisible_by = 128
    if not padded_vocab_present:
        # YAML namespaces do not inherit argparse's None default for this field.
        del args.padded_vocab_size
    tokenizer = SimpleNamespace(vocab_size=261)
    factory = Mock(return_value=tokenizer)
    monkeypatch.setattr(MegatronTokenizer, "from_pretrained", factory)
    model = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        **vocab_kwargs,
    )
    assert model.vocab_size is None
    factory.assert_not_called()
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    global_vars.set_args(args)
    monkeypatch.setattr(global_vars, "initialize_runtime_services", global_vars._build_tokenizer)

    global_vars.initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)
    assert model.vocab_size == cfg.tokenizer.padded_vocab_size == 512
    assert model.should_pad_vocab is False
    assert global_vars.get_tokenizer() is tokenizer
    factory.assert_called_once()


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("cli_vocab,expected", [(None, 512), (768, 768)])
def test_checkpoint_vocabulary_precedence_survives_runtime_setup(
    monkeypatch, isolated_globals, adapter, cli_vocab, expected
):
    """Exercise real checkpoint-argument restoration, not a prefilled namespace."""
    from megatron.training import checkpointing

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.load = "checkpoint"
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 133
    args.padded_vocab_size = cli_vocab
    args.tensor_model_parallel_size = 2
    checkpoint_args = Namespace(padded_vocab_size=512)
    state = {"args": checkpoint_args, "checkpoint_version": 3.0}
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        Mock(return_value=(state, "checkpoint", False, None)),
    )

    checkpointing.load_args_from_checkpoint(args)
    model = adapter(
        args, config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)
    )
    assert model.vocab_size == expected
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    global_vars.set_args(args)
    monkeypatch.setattr(global_vars, "initialize_runtime_services", global_vars._build_tokenizer)
    global_vars.initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)

    assert model.vocab_size == cfg.tokenizer.padded_vocab_size == expected
    assert model.should_pad_vocab is False
    assert global_vars.get_tokenizer().vocab_size == 133


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("padded_vocab_present", [False, True])
@pytest.mark.parametrize(
    "pad_vocab,vocab_kwargs", [(False, {}), (True, {"vocab_size_from_tokenizer": False})]
)
def test_known_unpadded_vocabulary_needs_no_tokenizer_for_config(
    adapter, padded_vocab_present, pad_vocab, vocab_kwargs
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.vocab_size = 133
    args.pad_vocab_size = pad_vocab
    if not padded_vocab_present:
        del args.padded_vocab_size
    model = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        **vocab_kwargs,
    )
    assert model.vocab_size == 133
    # Preserve the existing model-builder padding policy for raw vocabulary.
    assert model.should_pad_vocab is True


def test_vocabulary_resolution_neither_reads_globals_nor_initializes_services(monkeypatch):
    monkeypatch.setattr(
        global_vars, "get_args", Mock(side_effect=AssertionError("global args read"))
    )
    initialize = Mock(side_effect=AssertionError("runtime initialization"))
    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)
    cfg = SimpleNamespace(model=None, tokenizer=TokenizerConfig())
    resolve_tokenizer_vocab_size(cfg, 256)
    assert cfg.tokenizer.padded_vocab_size == 256
    initialize.assert_not_called()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("vocabulary", ["tokenizer", "checkpoint", "explicit", "custom"])
def test_initialize_megatron_finalizes_before_services_and_resolves_vocabulary(
    monkeypatch, isolated_globals, training, vocabulary
):
    """The shared startup owns validation, tokenization, and runtime derivation."""
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 133
    args.padded_vocab_size = 512 if vocabulary == "checkpoint" else None
    args.batch_parallel_size = 2
    args.world_size = 2
    monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 2)
    args.num_layers, args.hidden_size, args.num_attention_heads = 2, 128, 4
    args.params_dtype = torch.float32
    args.micro_batch_size = 2
    args.main_grads_dtype = torch.float32
    args.main_params_dtype = torch.float32
    args.exp_avg_dtype = torch.float32
    args.exp_avg_sq_dtype = torch.float32
    model = gpt_config_from_args(
        args, config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)
    )
    if vocabulary == "explicit":
        model.vocab_size = 768
    elif vocabulary == "custom":
        model = None
        args.pad_vocab_size = False
    cfg = (
        pretrain_cfg_container_from_args(args, model)
        if training
        else inference_cfg_container_from_args(args, build_model_config=False)
    )
    if model is not None or not training:
        cfg.model = model
    global_vars.set_args(args)
    global_vars.set_run_config(cfg)
    events = []
    finalize = cfg.finalize

    def finalize_inputs():
        finalize()
        # Stand in for a centralized derivation: services must see its result.
        cfg.logger.log_interval = 17
        events.append("finalize")

    monkeypatch.setattr(cfg, "finalize", finalize_inputs)
    build = global_vars._build_tokenizer

    def tokenize(received_args, **kwargs):
        assert events == ["finalize"]
        assert cfg.logger.log_interval == 17
        events.append("tokenizer")
        return build(received_args, **kwargs)

    monkeypatch.setattr(global_vars, "_build_tokenizer", tokenize)
    expected_vocab = {"tokenizer": 256, "checkpoint": 512, "explicit": 768}

    def writer(received_args):
        assert received_args is args
        assert cfg.logger.log_interval == 17
        if training:
            assert cfg.train.global_batch_size == cfg.validation.eval_global_batch_size == 4
            assert cfg.validation.eval_micro_batch_size == 2
        if model is not None:
            assert model.vocab_size == expected_vocab[vocabulary]
        assert cfg.tokenizer.padded_vocab_size == args.padded_vocab_size
        events.append("writer")

    monkeypatch.setattr(global_vars, "_set_wandb_writer", writer)
    monkeypatch.setattr(global_vars, "_set_telemetry", Mock())
    training_services = Mock(side_effect=lambda args: events.append("training"))
    monkeypatch.setattr(global_vars, "initialize_training_runtime_services", training_services)
    monkeypatch.setattr(initialize, "setup_logging", Mock())
    monkeypatch.setattr(initialize, "initialize_rerun_state_machine", Mock())
    monkeypatch.setattr(initialize.tuning, "install", Mock())

    initialize.initialize_megatron(
        allow_no_cuda=True,
        skip_mpu_initialization=True,
        training=training,
        build_tokenizer=vocabulary != "custom",
    )

    assert events == (
        ["finalize"]
        + (["tokenizer"] if vocabulary != "custom" else [])
        + ["writer"]
        + (["training"] if training else [])
    )
    assert training_services.call_count == int(training)
    assert (global_vars._GLOBAL_TOKENIZER is None) == (vocabulary == "custom")


@pytest.mark.parametrize("bootstrap", [False, True])
def test_invalid_config_stops_before_runtime_services(monkeypatch, isolated_globals, bootstrap):
    args = _runtime_args()
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    cfg.profiling.use_nsys_profiler = True
    cfg.profiling.use_pytorch_profiler = True
    services = Mock(side_effect=AssertionError("runtime services must not start"))
    monkeypatch.setattr(global_vars, "initialize_runtime_services", services)
    monkeypatch.setattr(initialize, "initialize_runtime_services", services)
    with pytest.raises(ValueError, match="mutually exclusive"):
        if bootstrap:
            global_vars.set_global_variables(args, cfg)
        else:
            global_vars.set_args(args)
            global_vars.set_run_config(cfg)
            initialize.initialize_megatron(allow_no_cuda=True)
    services.assert_not_called()


def test_rl_entrypoint_leaves_runtime_setup_to_pretrain(monkeypatch, isolated_globals, run_config):
    import megatron.training as training_package
    from megatron.rl.callbacks import RLCheckpointCallback, RLLoggingCallback
    from megatron.training import argument_utils, utils

    args = Namespace(reset_attention_mask=False, batch_invariant_mode=False)
    model = Mock()
    # The real RL callback derives its rollout interval from these batch values.
    cfg = run_config
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 8
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    monkeypatch.setattr(logging, "basicConfig", Mock())
    monkeypatch.setattr(arguments, "parse_and_validate_args", Mock(return_value=args))
    monkeypatch.setattr(utils, "is_hybrid_model", lambda args: False)
    monkeypatch.setattr(argument_utils, "gpt_config_from_args", Mock(return_value=model))
    monkeypatch.setattr(argument_utils, "pretrain_cfg_container_from_args", Mock(return_value=cfg))
    services = Mock(side_effect=AssertionError("entrypoint initialized services before pretrain"))
    monkeypatch.setattr(global_vars, "initialize_runtime_services", services)
    pretrain = Mock()
    monkeypatch.setattr(training_package, "pretrain", pretrain)

    runpy.run_path(str(Path(__file__).resolve().parents[3] / "train_rl.py"), run_name="__main__")

    pretrain.assert_called_once()
    assert pretrain.call_args.args[0] is cfg
    callbacks = pretrain.call_args.kwargs["callbacks"]
    assert [type(callback) for callback in callbacks] == [RLLoggingCallback, RLCheckpointCallback]
    assert global_vars.get_run_config() is cfg
    assert global_vars._GLOBAL_TOKENIZER is None
    services.assert_not_called()

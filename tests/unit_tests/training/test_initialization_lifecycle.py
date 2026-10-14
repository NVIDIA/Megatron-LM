# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for config construction before training runtime initialization."""

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer import TransformerConfig
from megatron.training import arguments, global_vars
from megatron.training.argument_utils import (
    gpt_config_from_args,
    hybrid_config_from_args,
    inference_cfg_container_from_args,
    model_provider_config_from_args,
    pretrain_cfg_container_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.config.training_config import TokenizerConfig
from megatron.training.models import ModelProviderConfig


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
    initialize.assert_called_once_with(args, build_tokenizer=True, training=True)
    with pytest.raises(AssertionError, match="already initialized"):
        global_vars.set_global_variables(args, cfg)


def test_explicit_training_bootstrap_preserves_config(monkeypatch, isolated_globals, run_config):
    args = _runtime_args()
    cfg = run_config
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    cfg.train.train_iters = 7
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
def test_runtime_service_order_and_microbatch_inputs(monkeypatch, isolated_globals, use_bootstrap):
    args = _runtime_args()
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    cfg.model = ModelProviderConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
        seq_length=32,
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
    "world_size, training_world_size, parallel_kwargs, expected_dp, expected_batch",
    [
        (8, None, {"tensor_model_parallel_size": 2}, 4, 4),
        (16, 8, {"tensor_model_parallel_size": 2}, 4, 4),
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
    ],
)
def test_training_batch_topology_uses_model_config(
    monkeypatch,
    run_config,
    world_size,
    training_world_size,
    parallel_kwargs,
    expected_dp,
    expected_batch,
):
    args = _runtime_args()
    args.world_size = world_size
    args.data_parallel_size = args.batch_parallel_size = 999
    run_config.model = ModelProviderConfig(
        transformer=TransformerConfig(
            num_layers=2, hidden_size=32, num_attention_heads=4, **parallel_kwargs
        ),
        seq_length=32,
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
    global_vars.initialize_runtime_services(
        args, build_tokenizer=False, training=True, training_world_size=training_world_size
    )
    assert args.world_size == world_size
    assert args.data_parallel_size == expected_dp
    assert args.batch_parallel_size == expected_batch
    assert microbatches.call_args.kwargs['data_parallel_size'] == expected_batch
    assert telemetry[0]['dl.data_parallel.size'] == expected_dp
    assert telemetry[0]['dl.tensor_parallel.size'] == 2
    assert telemetry[0]['megatron.hidden_size'] == 32
    assert telemetry[0] == global_vars.build_telemetry_resource_attrs(args)


@pytest.mark.parametrize("known_vocab", [None, 512])
def test_callback_model_config_owns_shared_inputs(
    monkeypatch, isolated_globals, run_config, known_vocab
):
    """Callback applications keep a typed model without selecting a GPT/Hybrid builder."""
    import megatron.training.argument_utils as adapters
    from megatron.training.config import PretrainConfigContainer

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.max_position_embeddings = 2048
    args.padded_vocab_size = known_vocab
    transformer = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)
    convert = Mock(return_value=transformer)
    monkeypatch.setattr(adapters, "core_transformer_config_from_args", convert)

    model = model_provider_config_from_args(args)
    convert.assert_called_once_with(args)
    cfg = run_config
    cfg.model = model
    resolve_tokenizer_vocab_size(cfg, 256)
    args.padded_vocab_size = 999
    args.max_position_embeddings = 32

    assert isinstance(cfg.model, ModelProviderConfig)
    assert cfg.model.builder is None
    assert cfg.model.transformer is transformer
    assert cfg.model.seq_length == 2048
    assert cfg.model.vocab_size == (256 if known_vocab is None else known_vocab)
    assert ModelProviderConfig.from_dict(model.as_dict()).as_dict() == model.as_dict()
    restored = PretrainConfigContainer.from_dict(cfg.to_dict())
    assert isinstance(restored.model, ModelProviderConfig)
    assert restored.to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    "section, name, value, error",
    [
        ("checkpoint", "ckpt_format", "torch", "requires torch_dist"),
        ("dist", "use_torch_fsdp2", True, "not FSDP"),
        ("dist", "use_megatron_fsdp", True, "not FSDP"),
        ("dist", "use_tp_pp_dp_mapping", True, "TP-PP-DP mapping"),
    ],
)
def test_native_gtp_sequence_sharding_validates_cross_config_inputs(
    run_config, section, name, value, error
):
    run_config.model = ModelProviderConfig(
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
    setattr(getattr(run_config, section), name, value)
    with pytest.raises(ValueError, match=error):
        run_config.validate()


def test_callback_upcycling_uses_temporary_dense_config(monkeypatch, run_config):
    from megatron.training import training

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.skip_train = True
    args.moe_use_upcycling = True
    args.moe_upcycling_granularity = 2
    args.num_experts = 2
    args.expert_model_parallel_size = 1
    args.ffn_hidden_size = 64
    run_config.model = ModelProviderConfig(
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
def test_callbacks_construct_from_registered_model_config(
    monkeypatch, isolated_globals, family, inference
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
    cfg = (
        inference_cfg_container_from_args(args, build_model_config=False)
        if inference
        else pretrain_cfg_container_from_args(args)
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
    ):
        delattr(args, name)

    provider.model_provider()

    assert build.call_args.kwargs["config"] is cfg.model.transformer
    assert build.call_args.kwargs["vocab_size"] == 128
    assert build.call_args.kwargs["max_sequence_length"] == 64
    assert build.call_args.kwargs["share_embeddings_and_output_weights"] is True


def test_model_precision_is_derived_centrally(run_config):
    from megatron.core.optimizer import OptimizerConfig

    run_config.model = ModelProviderConfig(
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
    "adapter", [gpt_config_from_args, hybrid_config_from_args, ModelProviderConfig]
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
    if adapter is ModelProviderConfig:
        model = ModelProviderConfig(
            transformer=transformer,
            seq_length=64,
            vocab_size=padded_vocab if padded_vocab is not None or pad_vocab else args.vocab_size,
        )
    else:
        model = adapter(args, config=transformer)
    model.make_vocab_size_divisible_by = 256
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
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
        model=ModelProviderConfig(
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
def test_dataset_helpers_use_registered_model(monkeypatch, isolated_globals, module_name):
    import importlib

    entrypoint = importlib.import_module(module_name)
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.seq_length = 32
    args.data_parallel_size = 2
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
    cfg = SimpleNamespace(
        model=ModelProviderConfig(transformer=transformer, seq_length=64),
        rng=SimpleNamespace(seed=1234),
    )
    global_vars.set_args(args)
    global_vars.set_run_config(cfg)
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
    assert entrypoint.is_dataset_built_on_rank()
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
    state = {"args": checkpoint_args, "iteration": 17, "checkpoint_version": 3.0}
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        Mock(return_value=(state, "checkpoint", False, None)),
    )

    checkpointing.load_args_from_checkpoint(args)
    assert args.iteration == 17
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

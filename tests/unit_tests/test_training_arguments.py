# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from pathlib import Path
import sys

from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars
from megatron.training.yaml_arguments import (
    core_transformer_config_from_yaml,
    load_yaml,
    validate_yaml,
)
from tests.unit_tests.dist_checkpointing import init_basic_mock_args

GPT_YAML_CONFIG = Path(__file__).resolve().parents[2] / "examples" / "gpt3" / "gpt_config.yaml"


def _reset_test_state():
    destroy_global_vars()
    destroy_num_microbatches_calculator()


def _make_cli_args(**overrides):
    _reset_test_state()

    original_argv = sys.argv
    sys.argv = ['test_training_arguments.py']
    try:
        args = parse_args(ignore_unknown_args=True)
    finally:
        sys.argv = original_argv

    init_basic_mock_args(args, tp=1, pp=4, bf16=True)

    args.num_layers = 4
    args.hidden_size = 256
    args.num_attention_heads = 8
    args.max_position_embeddings = 512
    args.world_size = 4
    args.rank = 0
    args.use_legacy_models = False
    args.micro_batch_size = 2
    args.global_batch_size = 8
    args._is_global_batch_size_explicitly_specified = True
    args.seq_length = 128
    args.train_iters = 10
    args.train_samples = None
    args.lr = 3e-5
    args.create_attention_mask_in_dataloader = True
    args.bf16 = True
    args.add_bias_linear = False
    args.swiglu = True
    args.use_distributed_optimizer = True
    args.attention_backend = "unfused"
    args.position_embedding_type = "rope"
    args.rotary_percent = 1.0
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.overlap_p2p_comm = True
    args.align_param_gather = False
    args.overlap_grad_reduce = False
    args.overlap_param_gather = False
    args.overlap_param_gather_with_optimizer_step = False
    args.fp8_param_gather = False
    args.fp4_param_gather = False
    args.use_ring_exchange_p2p = False
    args.batch_p2p_comm = False
    args.overlap_p2p_comm_warmup_flush = False
    args.variable_seq_lengths = False
    args.num_layers_per_virtual_pipeline_stage = None
    args.num_virtual_stages_per_pipeline_rank = None
    args.hybrid_layer_pattern = None
    args.decoder_first_pipeline_num_layers = None
    args.decoder_last_pipeline_num_layers = None
    args.account_for_embedding_in_pipeline_split = False
    args.account_for_loss_in_pipeline_split = False

    for key, value in overrides.items():
        assert hasattr(args, key)
        setattr(args, key, value)

    return args


def _make_yaml_args(**overrides):
    _reset_test_state()

    args = load_yaml(str(GPT_YAML_CONFIG))
    args.world_size = 4
    args.rank = 0
    args.train_iters = 10
    args.train_samples = None
    args.lr_decay_samples = None
    args.lr_warmup_samples = 0
    args.lr_warmup_iters = 0
    args.micro_batch_size = 2
    args.global_batch_size = 8
    args._is_global_batch_size_explicitly_specified = True
    args.step_batch_size_schedule = None
    args.seq_length = 128
    args.max_position_embeddings = 512
    args.use_distributed_optimizer = True
    args.overlap_param_gather = False
    args.overlap_grad_reduce = False
    args.align_param_gather = False
    args.model_parallel.tensor_model_parallel_size = 1
    args.model_parallel.context_parallel_size = 1
    args.model_parallel.pipeline_model_parallel_size = 4
    args.model_parallel.virtual_pipeline_model_parallel_size = None
    args.model_parallel.overlap_p2p_comm = True
    args.model_parallel.variable_seq_lengths = False
    args.model_parallel.tp_comm_overlap = False
    args.model_parallel.sequence_parallel = False

    for key, value in overrides.items():
        if key.startswith("model_parallel__"):
            setattr(args.model_parallel, key.split("__", 1)[1], value)
        elif key.startswith("language_model__"):
            setattr(args.language_model, key.split("__", 1)[1], value)
        else:
            assert hasattr(args, key)
            setattr(args, key, value)

    return args


def test_cli_non_interleaved_overlap_not_force_disabled():
    args = _make_cli_args(overlap_p2p_comm=True)

    validated = validate_args(args)

    assert validated.overlap_p2p_comm is True
    assert validated.virtual_pipeline_model_parallel_size is None


def test_cli_non_interleaved_align_param_gather_still_handled():
    args = _make_cli_args(overlap_p2p_comm=True, align_param_gather=True)

    validated = validate_args(args)

    assert validated.overlap_p2p_comm is True
    assert validated.align_param_gather is False


def test_yaml_non_interleaved_overlap_not_force_disabled():
    args = _make_yaml_args(model_parallel__overlap_p2p_comm=True)

    validated = validate_yaml(args)

    assert validated.overlap_p2p_comm is True
    assert validated.model_parallel.overlap_p2p_comm is True
    assert validated.virtual_pipeline_model_parallel_size is None


def test_yaml_non_interleaved_align_param_gather_still_handled():
    args = _make_yaml_args(align_param_gather=True, model_parallel__overlap_p2p_comm=True)

    validated = validate_yaml(args)

    assert validated.overlap_p2p_comm is True
    assert validated.model_parallel.overlap_p2p_comm is True
    assert validated.align_param_gather is False


def test_core_transformer_config_keeps_batch_p2p_mapping():
    cli_args = _make_cli_args(overlap_p2p_comm=True)
    validated_cli_args = validate_args(cli_args)
    cli_config = core_transformer_config_from_args(validated_cli_args)

    yaml_args = _make_yaml_args(model_parallel__overlap_p2p_comm=True)
    validated_yaml_args = validate_yaml(yaml_args)
    yaml_config = core_transformer_config_from_yaml(validated_yaml_args)

    assert cli_config.overlap_p2p_comm is True
    assert cli_config.batch_p2p_comm is False
    assert yaml_config.overlap_p2p_comm is True
    assert yaml_config.batch_p2p_comm is False

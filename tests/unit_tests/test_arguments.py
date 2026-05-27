# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from megatron.training import arguments


def _minimal_training_argv(extra_args=None):
    argv = [
        "program",
        "--num-layers",
        "2",
        "--hidden-size",
        "16",
        "--num-attention-heads",
        "4",
        "--seq-length",
        "8",
        "--max-position-embeddings",
        "8",
        "--micro-batch-size",
        "2",
        "--train-iters",
        "4",
        "--lr",
        "0.001",
        "--min-lr",
        "0.0",
        "--bf16",
        "--mock-data",
    ]
    if extra_args:
        argv.extend(extra_args)
    return argv


def _parse_minimal_training_args(monkeypatch, extra_args=None):
    monkeypatch.setattr("sys.argv", _minimal_training_argv(extra_args))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return arguments.parse_args()


def _patch_validate_environment(monkeypatch):
    monkeypatch.setattr(arguments, "get_device_arch_version", lambda: 10)
    monkeypatch.setattr(arguments, "is_flashinfer_min_version", lambda version: True)
    monkeypatch.setattr(arguments, "is_te_min_version", lambda version: True)
    monkeypatch.setattr(arguments, "is_torch_min_version", lambda version: True)


def test_add_megatron_arguments_registers_training_parser_groups():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    returned_parser = arguments.add_megatron_arguments(parser)
    group_titles = {group.title for group in parser._action_groups}

    assert returned_parser is parser
    assert "network size" in group_titles
    assert "training" in group_titles
    assert "learning rate and weight decay" in group_titles
    assert "checkpointing" in group_titles
    assert "distributed init" in group_titles
    assert "validation" in group_titles
    assert "data and dataloader" in group_titles
    assert "tokenizer" in group_titles


def test_parser_accepts_representative_training_arguments():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    arguments.add_megatron_arguments(parser)

    parsed = parser.parse_args(
        [
            "--num-layers",
            "2",
            "--hidden-size",
            "16",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "8",
            "--max-position-embeddings",
            "8",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "1",
            "--lr",
            "0.001",
            "--min-lr",
            "0.0001",
            "--lr-decay-style",
            "cosine",
            "--dataloader-type",
            "single",
            "--tokenizer-type",
            "NullTokenizer",
            "--bf16",
            "--use-distributed-optimizer",
        ]
    )

    assert parsed.num_layers == 2
    assert parsed.hidden_size == 16
    assert parsed.num_attention_heads == 4
    assert parsed.micro_batch_size == 1
    assert parsed.global_batch_size == 1
    assert parsed.lr == 0.001
    assert parsed.min_lr == 0.0001
    assert parsed.lr_decay_style == "cosine"
    assert parsed.dataloader_type == "single"
    assert parsed.tokenizer_type == "NullTokenizer"
    assert parsed.bf16
    assert parsed.use_distributed_optimizer


def test_parse_args_sets_rank_and_world_size_from_environment(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "--num-layers",
            "2",
            "--hidden-size",
            "16",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "8",
            "--max-position-embeddings",
            "8",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "1",
        ],
    )
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    parsed = arguments.parse_args()

    assert parsed.rank == 3
    assert parsed.world_size == 8
    assert parsed.enable_msc


def test_parse_args_allows_extra_provider_and_unknown_args(monkeypatch):
    def extra_provider(parser):
        parser.add_argument("--custom-flag", type=int, default=0)
        return parser

    monkeypatch.setattr("sys.argv", ["program", "--custom-flag", "7", "--unknown-flag"])

    parsed = arguments.parse_args(extra_args_provider=extra_provider, ignore_unknown_args=True)

    assert parsed.custom_flag == 7


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("[0,1]*2", [0, 1, 0, 1]),
        ("([1]+[0])*2", [1, 0, 1, 0]),
        ("[1,0,0]", [1, 0, 0]),
    ],
)
def test_eval_pattern_accepts_safe_list_expressions(pattern, expected):
    assert arguments._eval_pattern(pattern) == expected


def test_eval_pattern_rejects_unsafe_expression():
    with pytest.raises(ValueError, match="Invalid pattern"):
        arguments._eval_pattern("[import('os').system('echo unsafe')]")


def test_frequency_and_tuple_helpers():
    assert arguments.no_rope_freq_type(None) is None
    assert arguments.no_rope_freq_type(2) == 2
    assert arguments.no_rope_freq_type("2") == 2
    assert arguments.no_rope_freq_type("[1,0]") == [1, 0]

    assert arguments.moe_freq_type(3) == 3
    assert arguments.moe_freq_type("3") == 3
    assert arguments.moe_freq_type("[1,0,1]") == [1, 0, 1]

    assert arguments.la_freq_type(None) is None
    assert arguments.la_freq_type(4) == 4
    assert arguments.la_freq_type("4") == 4
    assert arguments.la_freq_type("[1,1,0]") == [1, 1, 0]

    assert arguments.tuple_type(None) is None
    assert arguments.tuple_type((1, 2)) == (1, 2)
    assert arguments.tuple_type("1,2,3") == (1, 2, 3)
    assert arguments.tuple_type("(4,5)") == (4, 5)


def test_validate_args_derives_basic_training_defaults(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch)

    validated = arguments.validate_args(args)

    assert validated is args
    assert args.data_parallel_size == 1
    assert args.global_batch_size == 2
    assert args.dataloader_type == "single"
    assert args.encoder_num_layers == 2
    assert args.encoder_seq_length == 8
    assert args.ffn_hidden_size == 64
    assert args.kv_channels == 4
    assert args.params_dtype == torch.bfloat16
    assert args.accumulate_allreduce_grads_in_fp32
    assert args.use_dist_ckpt == (args.ckpt_format != "torch")
    assert args.start_weight_decay == args.weight_decay
    assert args.end_weight_decay == args.weight_decay


def test_validate_args_handles_data_path_split_and_phase_transitions(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(
        monkeypatch,
        [
            "--data-path",
            "1.0",
            "train",
            "--phase-transition-iterations",
            "8, 2, 5",
        ],
    )
    args.mock_data = False

    arguments.validate_args(args)

    assert args.split == "969, 30, 1"
    assert args.phase_transition_iterations == [2, 5, 8]


def test_validate_args_updates_deprecated_cuda_graph_flag(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--enable-cuda-graph"])

    arguments.validate_args(args)

    assert args.cuda_graph_impl == "local"
    assert not hasattr(args, "enable_cuda_graph")


def test_validate_args_resolves_rl_parallel_generation_alias(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--global-batch-size", "6"])
    args.perform_rl_step = True
    args.grpo_group_size = 3
    args.grpo_prompts_per_step = 2
    args.grpo_iterations = 1
    args.rl_parallel_generation_tasks = 2
    args.rl_num_parallel_generations = None
    args.rl_num_parallel_generation_batches = None
    args.rl_generation_batch_size = None
    args.rl_partial_rollouts = True
    args.rl_use_sequence_packing = False

    arguments.validate_args(args)

    assert args.rl_num_parallel_generations == 6
    assert args.rl_parallel_generation_tasks == 2
    assert args.rl_generation_batch_size == 1
    assert args.grpo_samples_per_iteration == 6
    assert not args.rl_enforce_generation_order


def test_validate_args_resolves_rl_generation_batches(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--global-batch-size", "4"])
    args.perform_rl_step = True
    args.grpo_group_size = 2
    args.grpo_prompts_per_step = 2
    args.grpo_iterations = 1
    args.rl_parallel_generation_tasks = None
    args.rl_num_parallel_generations = None
    args.rl_num_parallel_generation_batches = 2
    args.rl_generation_batch_size = None
    args.rl_partial_rollouts = True
    args.rl_use_sequence_packing = False

    arguments.validate_args(args)

    assert args.rl_generation_batch_size == 2
    assert args.rl_parallel_generation_tasks == 4
    assert args.rl_enforce_generation_order
    assert args.grpo_samples_per_iteration == 4


def test_validate_args_moe_deprecated_and_tokenizer_paths(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch)
    args.num_experts = 2
    args.moe_ffn_hidden_size = None
    args.moe_router_load_balancing_type = ["aux_loss"]
    args.moe_aux_loss_coeff = [0.1]
    args.no_weight_decay_cond_type = "apply_wd_to_qk_layernorm"
    args.apply_wd_to_qk_layernorm = False
    args.tiktoken_special_tokens = {"<extra>": 1}
    args.tokenizer_special_tokens = None
    args.tokenizer_hf_use_fast = True
    args.tokenizer_hf_include_special_tokens = True

    arguments.validate_args(args)

    assert args.moe_ffn_hidden_size == args.ffn_hidden_size
    assert args.moe_router_load_balancing_type == "aux_loss"
    assert args.moe_aux_loss_coeff == 0.1
    assert args.apply_wd_to_qk_layernorm
    assert args.no_weight_decay_cond_type is None
    assert args.tokenizer_special_tokens == {"<extra>": 1}


def test_validate_args_skip_train_and_async_save_paths(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--ckpt-format", "torch_dist"])
    args.skip_train = True
    args.perform_rl_step = False
    args.no_load_optim = False
    args.async_save = True
    args.use_persistent_ckpt_worker = False

    arguments.validate_args(args)

    assert args.no_load_optim
    assert not args.async_save


def test_validate_args_dtype_and_precision_guard_paths(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch)
    args.sequence_parallel = True
    args.tensor_model_parallel_size = 1
    args.main_params_dtype = "bf16"
    args.exp_avg_dtype = "fp32"
    args.exp_avg_sq_dtype = "fp16"
    args.mamba_inference_conv_states_dtype = "auto"
    args.mamba_inference_ssm_states_dtype = None
    args.grad_reduce_in_bf16 = True
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.bias_gelu_fusion = True

    arguments.validate_args(args)

    assert args.sequence_parallel is False
    assert args.main_params_dtype is torch.bfloat16
    assert args.exp_avg_dtype is torch.float32
    assert args.exp_avg_sq_dtype is torch.float16
    assert args.mamba_inference_conv_states_dtype is None
    assert args.accumulate_allreduce_grads_in_fp32 is False
    assert args.bias_gelu_fusion is False


def test_validate_model_config_args_from_heterogeneous_config_accepts_matching_args():
    config = {
        "hidden_act": "silu",
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "tie_word_embeddings": False,
        "rope_theta": 10000,
        "rope_scaling": {"factor": 2.0},
        "block_configs": [
            {"attention": {"n_heads_in_group": 2}},
            {"attention": {"n_heads_in_group": 2}},
        ],
    }
    args = SimpleNamespace(
        heterogeneous_layers_config_path=None,
        heterogeneous_layers_config_encoded_json=json.dumps(config),
        swiglu=True,
        normalization="RMSNorm",
        group_query_attention=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        use_rope_scaling=True,
        use_rotary_position_embeddings=True,
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        untie_embeddings_and_output_weights=True,
        rotary_base=10000,
        rope_scaling_factor=2.0,
        num_query_groups=2,
    )

    arguments.validate_model_config_args_from_heterogeneous_config(args)


def test_validate_model_config_args_from_heterogeneous_config_rejects_mismatch():
    config = {
        "hidden_act": "silu",
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "tie_word_embeddings": True,
        "rope_theta": 10000,
        "rope_scaling": {"factor": 1.0},
        "block_configs": [{"attention": {"n_heads_in_group": 2}}],
    }
    args = SimpleNamespace(
        heterogeneous_layers_config_path=None,
        heterogeneous_layers_config_encoded_json=json.dumps(config),
        swiglu=False,
        normalization="LayerNorm",
        group_query_attention=False,
        position_embedding_type="learned_absolute",
        rotary_percent=0.5,
        use_rope_scaling=False,
        use_rotary_position_embeddings=False,
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        untie_embeddings_and_output_weights=True,
        rotary_base=5000,
        rope_scaling_factor=2.0,
        num_query_groups=1,
    )

    with pytest.raises(ValueError, match="Arguments differ from heterogeneous config"):
        arguments.validate_model_config_args_from_heterogeneous_config(args)


@dataclasses.dataclass(init=False)
class _FakeTransformerConfig:
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_core_transformer_config_from_args_maps_validated_args(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--swiglu", "--group-query-attention"])
    arguments.validate_args(args)

    config = arguments.core_transformer_config_from_args(args, config_class=_FakeTransformerConfig)

    assert config.kwargs["num_layers"] == 2
    assert config.kwargs["hidden_size"] == 16
    assert config.kwargs["num_attention_heads"] == 4
    assert config.kwargs["pipeline_dtype"] == torch.bfloat16
    assert config.kwargs["deallocate_pipeline_outputs"] is True
    assert config.kwargs["batch_p2p_comm"] is True
    assert config.kwargs["gated_linear_unit"] is True
    assert config.kwargs["activation_func"] is torch.nn.functional.silu
    assert config.kwargs["num_query_groups"] == args.num_query_groups

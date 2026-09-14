# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from argparse import ArgumentParser
from copy import deepcopy
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from examples.hybrid.puzzle import (
    add_puzzle_args,
    build_puzzle_layer_config_list,
    build_puzzle_model_config,
)
from megatron.core.activations import squared_relu
from megatron.core.models.hybrid import MTPSplit
from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_layer_config_list
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import add_megatron_arguments, parse_args, validate_args
from megatron.training.models.hybrid import HybridModelBuilder
from tests.test_utils.python_scripts.recipe_parser import load_and_flatten

REPO_ROOT = Path(__file__).resolve().parents[3]
PUZZLE_CASE = "nemotron3_puzzle_75b_nightly_tp1_pp1_cp1_ep8_dgx_gb200"
LIGHTNING_CASE = "nemotron3_5_lightning_nightly_tp1_pp1_cp1_ep8_dgx_gb200"

# Independent transcription of config.json at Hugging Face revision 7cd7fa0.
# Indices are zero-based decoder positions, not positions in a pattern string.
ATTENTION_INDICES = (7, 16, 25, 36, 47, 58, 69, 78)
MOE_LAYERS = (
    (1, 1280, 4),
    (3, 1280, 8),
    (5, 1280, 10),
    (8, 1280, 8),
    (10, 1280, 8),
    (12, 1280, 8),
    (14, 1280, 12),
    (17, 1280, 8),
    (19, 1280, 10),
    (21, 1280, 8),
    (23, 2688, 12),
    (26, 1536, 14),
    (28, 2688, 12),
    (30, 1536, 12),
    (32, 1536, 12),
    (34, 2688, 12),
    (37, 2688, 12),
    (39, 2688, 12),
    (41, 1536, 10),
    (43, 2688, 12),
    (45, 2688, 12),
    (48, 1792, 12),
    (50, 1792, 14),
    (52, 1280, 10),
    (54, 1280, 10),
    (56, 1280, 12),
    (59, 1280, 8),
    (61, 1280, 12),
    (63, 1280, 10),
    (65, 1280, 8),
    (67, 1280, 8),
    (70, 1280, 10),
    (72, 1280, 10),
    (74, 1280, 10),
    (76, 1280, 12),
    (79, 1280, 12),
    (81, 1280, 14),
    (83, 1280, 16),
    (85, 1792, 18),
    (87, 2048, 18),
)


@pytest.fixture
def puzzle_args():
    parser = ArgumentParser()
    add_megatron_arguments(parser)
    assert add_puzzle_args(parser) is parser
    args = parser.parse_args([])
    # This scalar is normally derived by validate_args, not the parser.
    args.params_dtype = torch.float32
    return args


@pytest.fixture
def puzzle_config(puzzle_args):
    return build_puzzle_model_config(puzzle_args)


@pytest.fixture
def puzzle_recipe():
    case_dir = REPO_ROOT / "tests/functional_tests/test_cases/nemotron" / PUZZLE_CASE
    with (case_dir / "model_config.yaml").open() as stream:
        return yaml.safe_load(stream)


def test_puzzle_scalar_defaults_reach_model_config(puzzle_args):
    initialized_as_hybrid = []
    original_post_init = TransformerConfig.__post_init__

    def check_post_init(config):
        initialized_as_hybrid.append(config.is_hybrid_model)
        original_post_init(config)

    with patch.object(TransformerConfig, "__post_init__", check_post_init):
        model_config = build_puzzle_model_config(puzzle_args)

    assert initialized_as_hybrid and all(initialized_as_hybrid)
    assert puzzle_args.is_hybrid_model is True
    assert puzzle_args.hybrid_layer_pattern is None
    assert not hasattr(puzzle_args, "hybrid_layer_config_list")
    assert puzzle_args.mtp_num_layers is None
    assert model_config.hybrid_layer_pattern is None
    assert model_config.mtp_num_layers is None
    assert model_config.num_layers == 88
    assert model_config.hidden_size == 4096
    assert model_config.vocab_size == 131072
    assert model_config.position_embedding_type == "none"
    assert model_config.share_embeddings_and_output_weights is False


def test_puzzle_complete_published_decoder_architecture(puzzle_config):
    parsed = parse_hybrid_layer_config_list(
        puzzle_config.hybrid_layer_config_list, expected_num_layers=88
    )
    decoder = parsed.main_layer_config_list
    expected_types = [MambaLayerConfig] * 88
    for index in ATTENTION_INDICES:
        expected_types[index] = AttentionLayerConfig
    for index, _, _ in MOE_LAYERS:
        expected_types[index] = MoELayerConfig

    assert [type(layer) for layer in decoder] == expected_types
    assert expected_types.count(MambaLayerConfig) == 40
    assert expected_types.count(MoELayerConfig) == 40
    assert expected_types.count(AttentionLayerConfig) == 8
    assert (
        tuple(
            (index, layer.moe_ffn_hidden_size, layer.moe_router_topk)
            for index, layer in enumerate(decoder)
            if type(layer) is MoELayerConfig
        )
        == MOE_LAYERS
    )


def test_puzzle_common_dimensions_and_prediction_template(puzzle_config):
    layers = puzzle_config.hybrid_layer_config_list
    assert len(layers) == 91
    assert layers[88] is MTPSplit
    assert type(layers[89]) is AttentionLayerConfig
    assert type(layers[90]) is MoELayerConfig
    assert layers[90].moe_ffn_hidden_size == 2688
    assert layers[90].moe_router_topk == 22

    for layer in layers:
        if layer is MTPSplit:
            continue
        assert layer.num_layers == 88
        assert layer.hidden_size == 4096
        assert layer.is_hybrid_model is True
        assert layer.mtp_num_layers is None
        assert layer.normalization == "RMSNorm"
        assert layer.layernorm_epsilon == 1e-5
        assert layer.add_bias_linear is False
        assert layer.activation_func is squared_relu
        assert layer.init_method_std == 0.02
        if type(layer) is MambaLayerConfig:
            assert layer.mamba_state_dim == 96
            assert layer.mamba_num_heads == 128
            assert layer.mamba_head_dim == 64
            assert layer.mamba_num_groups == 8
        elif type(layer) is AttentionLayerConfig:
            assert layer.num_attention_heads == 32
            assert layer.num_query_groups == 2
            assert layer.kv_channels == 128
        else:
            assert layer.num_moe_experts == 512
            assert layer.moe_latent_size == 1024
            assert layer.moe_shared_expert_intermediate_size == 5376
            assert layer.moe_router_topk_scaling_factor == 5.0


def test_puzzle_factory_builds_independent_configs(puzzle_config):
    source = puzzle_config.transformer

    def source_state():
        # A deepcopy of functools.partial has a new identity even if unchanged.
        # Snapshot both its identity and contents, including mutable keyword args.
        return deepcopy(
            {
                name: (
                    (id(value), value.func, value.args, value.keywords)
                    if isinstance(value, partial)
                    else value
                )
                for name, value in vars(source).items()
            }
        )

    source_before = source_state()
    first = build_puzzle_layer_config_list(source)
    second = build_puzzle_layer_config_list(source)

    assert source_state() == source_before
    assert first is not second
    for first_layer, second_layer in zip(first, second, strict=True):
        if first_layer is MTPSplit:
            assert second_layer is MTPSplit
        else:
            assert first_layer is not second_layer
            assert first_layer is not source
    assert len({id(layer) for layer in first if layer is not MTPSplit}) == 90

    first[1].moe_ffn_hidden_size = 256
    assert first[3].moe_ffn_hidden_size == 1280
    assert second[1].moe_ffn_hidden_size == 1280
    assert source_state() == source_before


def test_puzzle_builder_infers_mtp_depth_without_mutating_layer_configs(puzzle_args, puzzle_config):
    source_layers = puzzle_config.hybrid_layer_config_list
    assert puzzle_config.mtp_num_layers is None

    # The real builder/parser run, but no 75B parameters or process groups are allocated.
    with (
        patch("megatron.training.models.hybrid.get_args", return_value=puzzle_args),
        patch("megatron.training.models.hybrid.HybridModel") as model_class,
    ):
        HybridModelBuilder(puzzle_config).build_model(Mock(), pre_process=True, post_process=True)

    assert puzzle_config.mtp_num_layers == 1
    assert puzzle_args.mtp_num_layers == 1
    assert model_class.call_args.kwargs["hybrid_layer_config_list"] is source_layers
    assert model_class.call_args.kwargs["hybrid_layer_pattern"] is None
    assert not hasattr(puzzle_args, "hybrid_layer_config_list")
    assert all(layer.mtp_num_layers is None for layer in source_layers if layer is not MTPSplit)


@pytest.mark.parametrize(
    ("test_case", "entrypoint"),
    [(PUZZLE_CASE, "examples/hybrid/puzzle.py"), (LIGHTNING_CASE, "pretrain_hybrid.py")],
)
def test_nemotron_nightly_recipe_entrypoints(test_case, entrypoint):
    workloads = load_and_flatten(REPO_ROOT / "tests/test_utils/recipes/gb200/nemotron.yaml")
    matches = [workload.spec for workload in workloads if workload.spec["test_case"] == test_case]
    assert len(matches) == 1
    spec = matches[0]
    assert spec["environment"] == "dev"
    assert spec["scope"] == "L2"
    assert spec["cadence"] == ["nightly"]
    assert spec["platforms"] == "dgx_gb200"
    assert (spec["nodes"], spec["gpus"], spec["n_repeat"]) == (2, 4, 5)
    script = spec["script"].format(
        **spec, assets_dir="/test/assets", artifacts_dir="/test/artifacts"
    )
    assert f'"TRAINING_SCRIPT_PATH={entrypoint}"' in script
    assert f"/nemotron/{test_case}/model_config.yaml" in script


def test_puzzle_recipe_covers_full_size_training_and_resume(puzzle_recipe):
    recipe = puzzle_recipe
    args = recipe["MODEL_ARGS"]
    assert recipe["TEST_TYPE"] == "ckpt-resume"
    assert args["--train-iters"] == 20
    assert args["--save-interval"] == 10
    assert args["--micro-batch-size"] == 1
    assert args["--global-batch-size"] == 32
    assert args["--seq-length"] == 2048
    assert args["--expert-model-parallel-size"] == 8
    for dimension in ("tensor-model", "pipeline-model", "context", "expert-tensor"):
        assert args[f"--{dimension}-parallel-size"] == 1
    assert args.get("--no-load-optim", False) is False
    assert args.get("--no-load-rng", False) is False
    assert args.get("--moe-shared-expert-overlap", False) is False
    assert "--hybrid-layer-pattern" not in args
    assert "--mtp-num-layers" not in args
    assert set(recipe["METRICS"]) == {
        "iteration-time",
        "lm loss",
        "mtp_1 loss",
        "num-zeros",
        "mem-allocated-bytes",
        "mem-max-allocated-bytes",
    }


def test_puzzle_recipe_validates_and_builds_training_config(monkeypatch, puzzle_recipe):
    argv = ["examples/hybrid/puzzle.py"]
    for flag, value in puzzle_recipe["MODEL_ARGS"].items():
        if value is False:
            continue
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv(
        "CUDA_DEVICE_MAX_CONNECTIONS", str(puzzle_recipe["ENV_VARS"]["CUDA_DEVICE_MAX_CONNECTIONS"])
    )

    # Validate the GB200 configuration without requiring eight local GPUs or
    # allocating a model. Data/checkpoint paths are not accessed by these APIs.
    with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
        args = validate_args(parse_args(extra_args_provider=add_puzzle_args))
        model_config = build_puzzle_model_config(args)
        train_config = pretrain_cfg_container_from_args(args, model_config)

    assert model_config.num_layers == 88
    assert model_config.hidden_size == 4096
    assert model_config.vocab_size == 131072
    assert model_config.expert_model_parallel_size == 8
    assert args.data_parallel_size == 8
    assert model_config.bf16 is True
    assert model_config.params_dtype is torch.bfloat16
    assert model_config.recompute_granularity == "full"
    assert model_config.recompute_method == "uniform"
    assert model_config.recompute_num_layers == 1
    assert model_config.moe_flex_dispatcher_backend == "hybridep"
    assert model_config.moe_grouped_gemm is True
    assert model_config.use_fused_weighted_squared_relu is True
    assert model_config.moe_shared_expert_overlap is False
    assert model_config.mtp_num_layers is None
    assert model_config.hybrid_layer_pattern is None
    assert not hasattr(args, "hybrid_layer_config_list")
    assert len(model_config.hybrid_layer_config_list) == 91

    assert train_config.model is model_config
    assert train_config.optimizer.optimizer == "adam"
    assert train_config.optimizer.use_distributed_optimizer is True
    assert train_config.optimizer.use_precision_aware_optimizer is True
    assert train_config.optimizer.store_param_remainders is True
    for dtype_field in (
        "main_grads_dtype",
        "main_params_dtype",
        "exp_avg_dtype",
        "exp_avg_sq_dtype",
    ):
        assert getattr(train_config.optimizer, dtype_field) is torch.float32
    assert train_config.ddp.grad_reduce_in_fp32 is False
    assert train_config.checkpoint.load_optim is True
    assert train_config.checkpoint.load_rng is True
    assert train_config.checkpoint.save_optim is True
    assert train_config.checkpoint.save_rng is True

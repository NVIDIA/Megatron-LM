# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the rollout engine's inference container composed by the RL entry point."""

import importlib.util
import warnings
from unittest.mock import patch

import pytest

from megatron.inference.utils import add_inference_args
from megatron.rl.config import (
    add_rl_args,
    rollout_inference_container_from_args,
    rollout_model_is_separate,
    validate_rl_run,
)
from megatron.training.argument_utils import (
    _default_config_from_args,
    gpt_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.config import (
    InferenceConfigContainer,
    RLConfig,
    RunConfigRecord,
    TokenizerConfig,
)

BASE_ARGV = (
    "test_rollout_inference_config.py --num-layers 2 --hidden-size 128 --num-attention-heads 8 "
    "--micro-batch-size 1 --global-batch-size 4 --seq-length 32 --max-position-embeddings 32 "
    "--no-rope-fusion --vocab-size 128 --tokenizer-type NullTokenizer"
).split()


def _containers(*extra):
    with patch("sys.argv", BASE_ARGV + list(extra)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # train_rl.py's provider: the inference server flags plus the RL and rollout-inference groups.
        args = validate_args(
            parse_args(extra_args_provider=lambda parser: add_rl_args(add_inference_args(parser)))
        )
        pretrain = pretrain_cfg_container_from_args(args, gpt_config_from_args(args))
    return args, pretrain, rollout_inference_container_from_args(args, pretrain)


def test_policy_and_log_probs_come_from_the_section():
    args, _, deployment = _containers(
        "--inference-kv-cache-management-mode",
        "recompute",
        "--inference-static-kv-memory-pointers",
        "--inference-dynamic-batching-unified-memory-level",
        "1",
    )
    section = deployment.inference
    assert (section.kv_cache_management_mode, section.static_kv_memory_pointers) == (
        "recompute",
        True,
    )
    # RL always wants generated-token log probs and never prompt log probs, whatever the CLI said.
    assert (section.return_log_probs, section.skip_prompt_log_probs) == (True, True)
    assert (args.return_log_probs, args.skip_prompt_log_probs) == (False, False)


def test_shared_weights_when_parallelism_matches():
    _, pretrain, deployment = _containers()
    assert deployment.model is pretrain.model
    assert not rollout_model_is_separate(deployment, pretrain.model)
    assert deployment.checkpoint.load is None
    # The deployment shares the training run's process-level sections.
    for name in ("dist", "rng", "tokenizer", "logger", "profiling"):
        assert getattr(deployment, name) is getattr(pretrain, name)


@pytest.mark.parametrize(
    "extra, expected",
    [
        (["--rl-inference-tensor-model-parallel-size", "1"], {"tensor_model_parallel_size": 1}),
        (
            [
                "--rl-inference-pipeline-model-parallel-size",
                "1",
                "--rl-inference-expert-model-parallel-size",
                "1",
            ],
            {"pipeline_model_parallel_size": 1, "expert_model_parallel_size": 1},
        ),
    ],
)
def test_refit_copy_carries_only_the_overrides(extra, expected):
    _, pretrain, deployment = _containers(*extra)
    assert deployment.model is not pretrain.model
    assert rollout_model_is_separate(deployment, pretrain.model)
    for name, value in expected.items():
        assert getattr(deployment.model.transformer, name) == value
    # Everything else is the training model's, and the training model is untouched.
    assert deployment.model.transformer.num_layers == pretrain.model.transformer.num_layers
    assert pretrain.model.transformer.tensor_model_parallel_size == 1
    assert (
        deployment.model.transformer.activation_func is pretrain.model.transformer.activation_func
    )


def test_context_parallel_training_forces_a_cp1_refit_copy():
    args, pretrain, _ = _containers()
    pretrain.model.transformer.context_parallel_size = 2  # as --context-parallel-size 2 would
    deployment = rollout_inference_container_from_args(args, pretrain)
    assert rollout_model_is_separate(deployment, pretrain.model)
    assert deployment.model.transformer.context_parallel_size == 1
    assert pretrain.model.transformer.context_parallel_size == 2


def test_weight_placement_is_the_rl_sections():
    """The rollout model's placement (UVM, offload while idle) is RL loop policy: it stays on the
    RL section, and the RL run config checks it against the deployment."""
    args, pretrain, deployment = _containers(
        "--rl-inference-model-unified-memory-level", "1", "--rl-offload-inference-model-weights"
    )
    rl = _default_config_from_args(RLConfig, args)
    assert (rl.rl_inference_model_unified_memory_level, rl.rl_offload_inference_model_weights) == (
        1,
        True,
    )
    run = RunConfigRecord(pretrain=pretrain, inference=deployment, rl=rl)
    with pytest.raises(ValueError, match="requires a separate inference model"):
        validate_rl_run(run)
    args, pretrain, deployment = _containers(
        "--rl-inference-model-unified-memory-level",
        "1",
        "--rl-offload-inference-model-weights",
        "--rl-inference-tensor-model-parallel-size",
        "1",
    )
    validate_rl_run(
        RunConfigRecord(
            pretrain=pretrain, inference=deployment, rl=_default_config_from_args(RLConfig, args)
        )
    )


def test_finalize_applies_the_training_rng_policy_to_the_rollout_model():
    """``finalize()`` runs the run's RNG finalizer on the container's model (#7591, #6925). The
    deployment shares the training run's RNG section, so a separately built rollout model gets the
    training run's tracker policy rather than the section defaults."""
    _, pretrain, deployment = _containers(
        "--rl-inference-tensor-model-parallel-size",
        "1",
        "--te-rng-tracker",
        "--inference-rng-tracker",
    )
    assert deployment.rng is pretrain.rng
    assert deployment.model is not pretrain.model
    deployment.finalize()
    transformer = deployment.model.transformer
    assert (transformer.use_te_rng_tracker, transformer.inference_rng_tracker) == (True, True)


@pytest.mark.parametrize(
    "flag",
    [
        "--rl-inference-tensor-model-parallel-size",
        "--rl-inference-pipeline-model-parallel-size",
        "--rl-inference-expert-model-parallel-size",
        "--rl-inference-expert-tensor-model-parallel-size",
    ],
)
def test_disaggregated_shards_take_their_parallelism_from_the_spec(flag):
    """With roles in --inference-shards every rank's layout comes from the shard specification:
    the deployment refits the training model into the shards and rejects the --rl-inference-*
    overrides."""
    shards = ["--inference-shards", "tp=1,role=prefill+tp=1,role=decode"]
    _, pretrain, deployment = _containers(*shards)
    assert deployment.model is pretrain.model
    assert rollout_model_is_separate(deployment, pretrain.model)
    with pytest.raises(ValueError, match=flag):
        _containers(*shards, flag, "1")


@pytest.mark.skipif(importlib.util.find_spec("omegaconf") is None, reason="omegaconf not installed")
@pytest.mark.parametrize("training_tp", [1, 2])
def test_deployment_round_trips_through_the_container_serialization(training_tp, tmp_path):
    """The record written next to a checkpoint reads back as itself, also when the rollout tensor
    parallelism differs from training's: the weight-shard counts, which ModelParallelConfig
    derives the GTP remat sizes from, follow the deployment's parallelism."""
    args, pretrain, _ = _containers("--rl-inference-tensor-model-parallel-size", "1")
    training = pretrain.model.transformer
    # As --tensor-model-parallel-size would set them: one weight shard per TP rank (no GTP).
    training.tensor_model_parallel_size = training.tensor_parallel_num_weight_shards = training_tp
    deployment = rollout_inference_container_from_args(args, pretrain)
    serialized = deployment.to_dict()
    assert serialized["_target_"].endswith("InferenceConfigContainer")
    deployment.to_yaml(str(tmp_path / "rollout_inference_config.yaml"))
    rebuilt = InferenceConfigContainer.from_yaml(str(tmp_path / "rollout_inference_config.yaml"))
    assert rebuilt.to_dict() == serialized
    rollout = rebuilt.model.transformer
    assert (
        rollout.tensor_model_parallel_size,
        rollout.tensor_parallel_num_weight_shards,
        rollout.gtp_weight_remat_size,
    ) == (1, 1, 1)
    assert training.tensor_parallel_num_weight_shards == training_tp
    assert isinstance(rebuilt.tokenizer, TokenizerConfig)

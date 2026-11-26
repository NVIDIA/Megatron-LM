# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from argparse import ArgumentParser, Namespace

from megatron.core.inference.shards_spec import spec_declares_disaggregation
from megatron.training.argument_utils import ArgumentGroupFactory, _default_config_from_args
from megatron.training.config import CheckpointConfig, RLConfig, RootConfigContainer
from megatron.training.config.container import (
    ROLLOUT_LAYOUT_FIELDS,
    InferenceConfigContainer,
    PretrainConfigContainer,
)
from megatron.training.config.inference_config import InferenceSetupConfig
from megatron.training.global_vars import (
    get_root_config_container,
    set_root_config_container,
    unset_root_config_container,
)
from megatron.training.models.base import ModelConfig

# The flag that overrides each rollout layout field. CP has none: context-parallel training
# forces the rollout's to 1.
_DEPLOYMENT_ARGS = {
    "tensor_model_parallel_size": "rl_inference_tensor_model_parallel_size",
    "pipeline_model_parallel_size": "rl_inference_pipeline_model_parallel_size",
    "expert_model_parallel_size": "rl_inference_expert_model_parallel_size",
    "expert_tensor_parallel_size": "rl_inference_expert_tensor_model_parallel_size",
}
# What a separately built rollout model takes from its deployment:
# the layout, plus the fields `rollout_inference_container_from_args` derives with it.
ROLLOUT_MODEL_LAYOUT_FIELDS = (
    *ROLLOUT_LAYOUT_FIELDS,
    "sequence_parallel",
    "tensor_parallel_num_sequence_shards",
    "tensor_parallel_num_weight_shards",
    "expert_tensor_parallel_num_weight_shards",
)


def rollout_inference_container_from_args(
    args: Namespace, pretrain: PretrainConfigContainer, *, model_cfg: ModelConfig | None = None
) -> InferenceConfigContainer:
    """Compose the rollout engine's deployment from the pretrain container and the RL arguments."""
    if model_cfg is None:
        model_cfg = pretrain.model
    overrides = {
        name: getattr(args, arg)
        for name, arg in _DEPLOYMENT_ARGS.items()
        if getattr(args, arg) is not None
    }
    if spec_declares_disaggregation(args.inference_shards):
        for name, arg in _DEPLOYMENT_ARGS.items():
            if name in overrides:
                raise ValueError(
                    f"--{arg.replace('_', '-')} cannot be combined with disaggregated "
                    "--inference-shards; configure parallelism in each shard specification"
                )
        # Each rank builds its shard from the specification (`build_rollout_inference_model`);
        # the training weights arrive by refit.
        inference_model_cfg = model_cfg
    else:
        if model_cfg.transformer.context_parallel_size > 1:
            # Rollout inference does not support CP; CP ranks become extra data-parallel replicas.
            overrides["context_parallel_size"] = 1
        # An override equal to the training layout changes nothing: rollouts share the model.
        overrides = {
            name: value
            for name, value in overrides.items()
            if getattr(model_cfg.transformer, name) != value
        }
        if overrides:
            inference_model_cfg = copy.deepcopy(model_cfg)
            transformer = inference_model_cfg.transformer
            for name, value in overrides.items():
                setattr(transformer, name, value)
            # Fields derived from the tensor parallelism follow the rollout layout, so the section
            # loads again: sequence parallelism needs TP > 1, and with CP 1 the rollout shards a
            # sequence only across its sequence-parallel ranks.
            transformer.sequence_parallel = (
                transformer.sequence_parallel and transformer.tensor_model_parallel_size > 1
            )
            transformer.tensor_parallel_num_sequence_shards = (
                transformer.tensor_model_parallel_size if transformer.sequence_parallel else 1
            )
            # The weight-shard counts are derived with the tensor parallelism.
            transformer.tensor_parallel_num_weight_shards = (
                transformer.tensor_model_parallel_size * transformer.gtp_weight_remat_size
            )
            transformer.expert_tensor_parallel_num_weight_shards = (
                transformer.expert_tensor_parallel_size * transformer.expert_gtp_weight_remat_size
            )
        else:
            inference_model_cfg = model_cfg

    inference = _default_config_from_args(InferenceSetupConfig, args)
    # RL needs the log probs of generated tokens, never prompt log probs.
    inference.return_log_probs = True
    inference.skip_prompt_log_probs = True

    return InferenceConfigContainer(
        model=inference_model_cfg,
        checkpoint=CheckpointConfig(),  # load=None: the weights come from the training model
        inference=inference,
        dist=pretrain.dist,
        rng=pretrain.rng,
        tokenizer=copy.deepcopy(pretrain.tokenizer),
        logger=pretrain.logger,
        profiling=pretrain.profiling,
    )


def add_rl_args(parser: ArgumentParser) -> ArgumentParser:
    """Add the `rl` argument group generated from `RLConfig`, then the rollout inference group."""
    ArgumentGroupFactory(RLConfig).build_group(parser, "rl")
    return add_rollout_inference_args(parser)


def add_rollout_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """The rollout engine's deployment, read by `rollout_inference_container_from_args`."""
    group = parser.add_argument_group(title="rollout inference")
    group.add_argument(
        "--rl-inference-tensor-model-parallel-size",
        type=int,
        default=None,
        help="Tensor model parallelism of the rollout engine (default: training).",
    )
    group.add_argument(
        "--rl-inference-pipeline-model-parallel-size",
        type=int,
        default=None,
        help="Pipeline model parallelism of the rollout engine (default: training).",
    )
    group.add_argument(
        "--rl-inference-expert-model-parallel-size",
        type=int,
        default=None,
        help="Expert model parallelism of the rollout engine (default: training).",
    )
    group.add_argument(
        "--rl-inference-expert-tensor-model-parallel-size",
        type=int,
        default=None,
        help="Expert tensor parallelism of the rollout engine (default: training).",
    )
    return parser


def set_rl_run_config(root: RootConfigContainer) -> None:
    """Register an RL run's root container."""
    assert (
        root.inference is not None and root.algorithm is not None
    ), "an RL run's root container carries its inference and algorithm members"
    assert root.algorithm.rl is not None, "an RL run's algorithm container carries its rl section"
    set_root_config_container(root)


def get_rl_run_config() -> RootConfigContainer:
    """Return the registered RL run's root container."""
    root = get_root_config_container()
    assert (
        root.algorithm is not None and root.algorithm.rl is not None
    ), "RL run config is not initialized."
    return root


def get_rl_config() -> RLConfig:
    """Return the registered RL run's RL section."""
    return get_rl_run_config().algorithm.rl


def unset_rl_run_config() -> None:
    """Forget the run's root container (tests and multi-run processes)."""
    unset_root_config_container()

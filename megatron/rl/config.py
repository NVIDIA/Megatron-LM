# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from argparse import ArgumentParser, Namespace

from megatron.core.inference.shards_spec import spec_declares_disaggregation
from megatron.training.argument_utils import ArgumentGroupFactory, _default_config_from_args
from megatron.training.config import CheckpointConfig, RLConfig, RootConfigContainer
from megatron.training.config.container import InferenceConfigContainer, PretrainConfigContainer
from megatron.training.config.inference_config import InferenceSetupConfig
from megatron.training.global_vars import (
    get_root_config_container,
    set_root_config_container,
    unset_root_config_container,
)
from megatron.training.models.base import ModelConfig

_DEPLOYMENT_ARGS = {
    "tensor_model_parallel_size": "rl_inference_tensor_model_parallel_size",
    "pipeline_model_parallel_size": "rl_inference_pipeline_model_parallel_size",
    "expert_model_parallel_size": "rl_inference_expert_model_parallel_size",
    "expert_tensor_parallel_size": "rl_inference_expert_tensor_model_parallel_size",
}
# Transformer fields in which a rollout deployment's layout can differ from training's:
# the overridable parallelism, plus the CP that context-parallel training forces to 1.
ROLLOUT_LAYOUT_FIELDS = (*_DEPLOYMENT_ARGS, "context_parallel_size")
# What a separately built rollout model takes from its deployment:
# the layout, plus the weight-shard counts derived with it.
ROLLOUT_MODEL_LAYOUT_FIELDS = (
    *ROLLOUT_LAYOUT_FIELDS,
    "tensor_parallel_num_weight_shards",
    "expert_tensor_parallel_num_weight_shards",
)


def rollout_model_is_separate(
    inference_container: InferenceConfigContainer, training_model_cfg: ModelConfig | None
) -> bool:
    """Whether rollouts need their own model instead of the training model.

    True for disaggregated shards, or for a deployment layout that differs from the training
    model's in TP, PP, EP, expert TP or CP. When either side has no transformer section, a
    distinct model config object counts as separate.
    """
    if spec_declares_disaggregation(inference_container.inference.inference_shards):
        return True
    deployment, training = inference_container.model, training_model_cfg
    layouts = (getattr(deployment, "transformer", None), getattr(training, "transformer", None))
    if None in layouts:
        return deployment is not training
    return any(
        getattr(layouts[0], name) != getattr(layouts[1], name) for name in ROLLOUT_LAYOUT_FIELDS
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
        tokenizer=pretrain.tokenizer,
        logger=pretrain.logger,
        profiling=pretrain.profiling,
    )


def validate_rl_run(root: RootConfigContainer, *, global_batch_size: int | None = None) -> None:
    """Validate an RL run's root container."""
    training, inference, rl = root.training, root.inference, root.algorithm.rl
    if global_batch_size is None:
        global_batch_size = training.train.global_batch_size
    # The offload `finalize_rl_run` will request, checked like the offload the DDP section
    # requests (`PretrainConfigContainer._validate_ddp_config`).
    if (
        rl.requires_grad_buffer_offload
        and training.optimizer.reuse_grad_buf_for_mxfp8_param_ag
        and not training.ddp.use_megatron_fsdp
    ):
        raise ValueError(
            "Gradient-buffer offload is incompatible with "
            "--reuse-grad-buf-for-mxfp8-param-ag: "
            "parameters share the gradient-buffer storage that offload releases."
        )
    root.validate()
    transformer = getattr(training.model, "transformer", None)
    cuda_graph_impl = None if transformer is None else transformer.cuda_graph_impl
    rl.validate_run(
        cuda_graph_impl=cuda_graph_impl,
        micro_batch_size=training.train.micro_batch_size,
        global_batch_size=global_batch_size,
        train_iters=training.train.train_iters,
        train_samples=training.train.train_samples,
        skip_train=training.validation.skip_train,
        load_optim=training.checkpoint.load_optim,
        save_interval=training.checkpoint.save_interval,
        exit_interval=training.train.exit_interval,
    )
    if rl.rl_offload_inference_model_weights and not rollout_model_is_separate(
        inference, training.model
    ):
        raise ValueError(
            "--rl-offload-inference-model-weights requires a separate inference model. "
            "This flag is only useful when doing refit since the weights are shared with "
            "the training model."
        )


def finalize_rl_run(root: RootConfigContainer) -> None:
    """Validate the root container, then finalize the deployment."""
    global_batch_size = root.training.train.global_batch_size
    if global_batch_size is None:
        # The training container derives an unset global batch size when it finalizes,
        # after this call; check the RL rules against the size it will derive.
        global_batch_size, _, _ = root.training._resolve_batch_sizes(
            root.training.get_batch_parallel_size()
        )
    validate_rl_run(root, global_batch_size=global_batch_size)
    root.inference.finalize()
    if root.algorithm.rl.requires_grad_buffer_offload:
        # Applied before the training container finalizes its DDP section.
        root.training.ddp.grad_buffer_offload = True


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
        root is not None and root.algorithm is not None and root.algorithm.rl is not None
    ), "RL run config is not initialized."
    return root


def get_rl_config() -> RLConfig:
    """Return the registered RL run's RL section."""
    return get_rl_run_config().algorithm.rl


def unset_rl_run_config() -> None:
    """Forget the registered root container (tests and multi-run processes)."""
    unset_root_config_container()

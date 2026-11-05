# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from argparse import ArgumentParser, Namespace

from megatron.core.inference.shards_spec import spec_declares_disaggregation
from megatron.training.argument_utils import ArgumentGroupFactory, _default_config_from_args
from megatron.training.config import RLConfig, RunConfigRecord
from megatron.training.config.container import InferenceConfigContainer, PretrainConfigContainer
from megatron.training.config.inference_config import InferenceSetupConfig
from megatron.training.global_vars import (
    get_run_config_record,
    set_run_config_record,
    unset_run_config_record,
)

_DEPLOYMENT_ARGS = {
    "tensor_model_parallel_size": "rl_inference_tensor_model_parallel_size",
    "pipeline_model_parallel_size": "rl_inference_pipeline_model_parallel_size",
    "expert_model_parallel_size": "rl_inference_expert_model_parallel_size",
    "expert_tensor_parallel_size": "rl_inference_expert_tensor_model_parallel_size",
}

def rollout_model_is_separate(inference_container, training_model_cfg) -> bool:
    """Whether rollouts run on their own copy of the model instead of the training model itself:
    disaggregated shards (each built from the shard specification), or a deployment layout that
    differs from the training model's (the CP forced to 1 under context-parallel training counts).
    Without a model config on both sides, distinct model objects are a separate build."""
    if spec_declares_disaggregation(inference_container.inference.inference_shards):
        return True
    deployment, training = inference_container.model, training_model_cfg
    layouts = (getattr(deployment, "transformer", None), getattr(training, "transformer", None))
    if None in layouts:
        return deployment is not training
    # The overridable parallelism, plus the CP that context-parallel training forces to 1.
    return any(
        getattr(layouts[0], name) != getattr(layouts[1], name)
        for name in (*_DEPLOYMENT_ARGS, "context_parallel_size")
    )


def rollout_inference_container_from_args(
    args: Namespace, pretrain: PretrainConfigContainer, *, model_cfg=None
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
            # CP ranks become extra data-parallel replicas of the rollout model.
            overrides["context_parallel_size"] = 1
        if overrides:
            inference_model_cfg = copy.deepcopy(model_cfg)
            transformer = inference_model_cfg.transformer
            for name, value in overrides.items():
                setattr(transformer, name, value)
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
        checkpoint=None,
        inference=inference,
        dist=pretrain.dist,
        rng=pretrain.rng,
        tokenizer=pretrain.tokenizer,
        logger=pretrain.logger,
        profiling=pretrain.profiling,
    )


def validate_rl_run(record: RunConfigRecord) -> None:
    """Validate an RL run's record."""
    pretrain, inference, rl = record.pretrain, record.inference, record.rl
    record.validate()
    rl.validate_ddp_config(pretrain.ddp)
    transformer = getattr(pretrain.model, "transformer", None)
    cuda_graph_impl = None if transformer is None else transformer.cuda_graph_impl
    rl.validate_run(
        cuda_graph_impl=cuda_graph_impl,
        micro_batch_size=pretrain.train.micro_batch_size,
        global_batch_size=pretrain.train.global_batch_size,
        train_iters=pretrain.train.train_iters,
        train_samples=pretrain.train.train_samples,
        skip_train=pretrain.validation.skip_train,
        load_optim=pretrain.checkpoint.load_optim,
        save_interval=pretrain.checkpoint.save_interval,
        exit_interval=pretrain.train.exit_interval,
    )
    if rl.rl_offload_inference_model_weights and not rollout_model_is_separate(
        inference, pretrain.model
    ):
        raise ValueError(
            "--rl-offload-inference-model-weights requires a separate inference model. "
            "This flag is only useful when doing refit since the weights are shared with "
            "the training model."
        )


def finalize_rl_run(record: RunConfigRecord) -> None:
    """Validate the record, then finalize the deployment."""
    validate_rl_run(record)
    record.inference.finalize()
    if record.rl.perform_rl_step:
        record.rl.finalize_ddp_config(record.pretrain.ddp)


def add_rl_args(parser: ArgumentParser) -> ArgumentParser:
    """Add the `rl` argument group generated from `RLConfig`."""
    ArgumentGroupFactory(RLConfig).build_group(parser, "rl")
    return add_rollout_inference_args(parser)


def add_rollout_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """The rollout engine's deployment, read by `rollout_inference_container_from_args`."""
    group = parser.add_argument_group(title='rollout inference')
    group.add_argument('--rl-inference-tensor-model-parallel-size', type=int, default=None,
                       help='Tensor model parallelism of the rollout engine (default: training).')
    group.add_argument('--rl-inference-pipeline-model-parallel-size', type=int, default=None,
                       help='Pipeline model parallelism of the rollout engine (default: training).')
    group.add_argument('--rl-inference-expert-model-parallel-size', type=int, default=None,
                       help='Expert model parallelism of the rollout engine (default: training).')
    group.add_argument('--rl-inference-expert-tensor-model-parallel-size', type=int, default=None,
                       help='Expert tensor parallelism of the rollout engine (default: training).')
    return parser


def set_rl_run_config(record: RunConfigRecord) -> None:
    """Register an RL run's record."""
    assert record.inference is not None and record.rl is not None, (
        "an RL run's record carries its inference and rl members"
    )
    set_run_config_record(record)


def get_rl_run_config() -> RunConfigRecord:
    """Return the registered RL run's record."""
    record = get_run_config_record()
    assert record is not None and record.rl is not None, "RL run config is not initialized."
    return record


def unset_rl_run_config() -> None:
    """Forget the registered record (tests and multi-run processes)."""
    unset_run_config_record()

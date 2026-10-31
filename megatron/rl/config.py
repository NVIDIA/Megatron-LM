# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field, fields

from megatron.core.inference.shards_spec import spec_declares_disaggregation
from megatron.training.argument_utils import (
    ArgumentGroupFactory,
    _default_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.config import CheckpointConfig, RLConfig
from megatron.training.config.container import InferenceConfigContainer, PretrainConfigContainer
from megatron.training.config.inference_config import InferenceSetupConfig

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
        checkpoint=CheckpointConfig(),  # load=None: the weights come from the training model
        inference=inference,
        dist=pretrain.dist,
        rng=pretrain.rng,
        tokenizer=pretrain.tokenizer,
        logger=pretrain.logger,
        profiling=pretrain.profiling,
    )


@dataclass(kw_only=True)
class RLRunConfigContainer(PretrainConfigContainer):
    """The RL run's configuration, subclassing `PretrainConfigContainer`."""

    inference: InferenceConfigContainer
    rl: RLConfig = field(default_factory=RLConfig)

    @classmethod
    def from_pretrain(
        cls, pretrain: PretrainConfigContainer, *, inference: InferenceConfigContainer, rl: RLConfig
    ) -> "RLRunConfigContainer":
        """Build the RL run config around an existing pretrain container's sections."""
        sections = {
            f.name: getattr(pretrain, f.name) for f in fields(PretrainConfigContainer) if f.init
        }
        return cls(**sections, inference=inference, rl=rl)

    def validate(self) -> None:
        """Validate the sections, then the RL section against the run it drives."""
        # The offload `_finalize` will request, checked like the offload the DDP section
        # requests (`PretrainConfigContainer._validate_ddp_config`).
        if (
            self.rl.requires_grad_buffer_offload
            and self.optimizer.reuse_grad_buf_for_mxfp8_param_ag
            and not self.ddp.use_megatron_fsdp
        ):
            raise ValueError(
                "Gradient-buffer offload is incompatible with "
                "--reuse-grad-buf-for-mxfp8-param-ag: "
                "parameters share the gradient-buffer storage that offload releases."
            )
        super().validate()
        transformer = getattr(self.model, "transformer", None)
        cuda_graph_impl = None if transformer is None else transformer.cuda_graph_impl
        self.rl.validate_run(
            cuda_graph_impl=cuda_graph_impl,
            micro_batch_size=self.train.micro_batch_size,
            global_batch_size=self.train.global_batch_size,
            train_iters=self.train.train_iters,
            train_samples=self.train.train_samples,
            skip_train=self.validation.skip_train,
            load_optim=self.checkpoint.load_optim,
            save_interval=self.checkpoint.save_interval,
            exit_interval=self.train.exit_interval,
        )
        if self.rl.rl_offload_inference_model_weights and not rollout_model_is_separate(
            self.inference, self.model
        ):
            raise ValueError(
                "--rl-offload-inference-model-weights requires a separate inference model. "
                "This flag is only useful when doing refit since the weights are shared with "
                "the training model."
            )

    def _finalize(self) -> None:
        """The pretrain derivations, then the deployment's, then the RL layout on the DDP config."""
        super()._finalize()
        self.inference.finalize()
        if self.rl.requires_grad_buffer_offload:
            self.ddp.grad_buffer_offload = True


def rl_run_config_container_from_args(
    args: Namespace, model_cfg=None, *, rollout_model_cfg=None
) -> RLRunConfigContainer:
    """Compose the RL run config from the entry point's arguments."""
    pretrain = pretrain_cfg_container_from_args(args, model_cfg)
    return RLRunConfigContainer.from_pretrain(
        pretrain,
        inference=rollout_inference_container_from_args(
            args, pretrain, model_cfg=rollout_model_cfg
        ),
        rl=_default_config_from_args(RLConfig, args),
    )


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


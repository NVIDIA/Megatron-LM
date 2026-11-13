# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from argparse import Namespace
from typing import Any, Callable, Optional

import torch

from megatron.core import mpu
from megatron.core._rank_utils import safe_get_world_size
from megatron.core.distributed import TorchFullyShardedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.inference.shards import build_inference_pg_collection
from megatron.core.inference.shards_spec import spec_declares_disaggregation
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    get_num_microbatches,
    init_num_microbatches_calculator,
    update_num_microbatches,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_model_config
from megatron.rl.config import ROLLOUT_MODEL_LAYOUT_FIELDS, get_rl_config, rollout_model_is_separate
from megatron.rl.inference.disagg import build_disagg_inference_model
from megatron.rl.rl_utils import inference_model_alloc_context
from megatron.training.callbacks import Callback, CallbackContext
from megatron.training.checkpointing import load_checkpoint
from megatron.training.config import InferenceConfigContainer
from megatron.training.global_vars import get_args, get_run_config, get_train_state
from megatron.training.training import get_model
from megatron.training.utils import print_rank_0

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP  # noqa: F401

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False


def build_rollout_inference_model(
    inference_container: InferenceConfigContainer,
    args: Namespace,
    config: TransformerConfig,
    model_provider: Callable[..., torch.nn.Module],
    model_type: ModelType,
) -> Optional[list[torch.nn.Module]]:
    """Build the rollout engine's separately parallelized model, if the deployment needs one."""
    if spec_declares_disaggregation(inference_container.inference.inference_shards):
        # Disaggregated rollouts build this rank's prefill/decode model on
        # the shard process groups maintained by the per-pool refit path.
        return build_disagg_inference_model(
            args,
            model_type,
            config,
            rl_config=get_rl_config(),
            inference_container=inference_container,
            model_alloc_ctx=inference_model_alloc_context(),
        )
    if not rollout_model_is_separate(inference_container, get_run_config().model):
        return None

    deployment = inference_container.model.transformer
    print_rank_0(
        "Building separate RL inference model with custom parallelism: "
        f"TP={deployment.tensor_model_parallel_size}, "
        f"PP={deployment.pipeline_model_parallel_size}, "
        f"CP={deployment.context_parallel_size}, "
        f"EP={deployment.expert_model_parallel_size}, "
        f"ExptTP={deployment.expert_tensor_parallel_size}"
    )
    inference_pg_collection = build_inference_pg_collection(
        safe_get_world_size(),
        tp_size=deployment.tensor_model_parallel_size,
        pp_size=deployment.pipeline_model_parallel_size,
        cp_size=deployment.context_parallel_size,
        ep_size=deployment.expert_model_parallel_size,
        expt_tp_size=deployment.expert_tensor_parallel_size,
        use_tp_pp_dp_mapping=get_run_config().dist.use_tp_pp_dp_mapping,
    )

    # Build an isolated inference config so the training config remains unchanged.
    inference_config = copy.deepcopy(config)
    for name in ROLLOUT_MODEL_LAYOUT_FIELDS:
        setattr(inference_config, name, getattr(deployment, name))

    with inference_model_alloc_context():
        inference_model = get_model(
            model_provider,
            model_type,
            wrap_with_ddp=False,
            pg_collection=inference_pg_collection,
            config=inference_config,
        )
    inference_model[0].eval()
    return inference_model


def load_reference_state_dict(
    model: list, checkpointing_context: dict, args: Any, *, rollout_only: bool
) -> dict:
    """Load the pretrained checkpoint's weights as the RL reference state dict."""
    cfg = get_run_config()
    if rollout_only:
        print_rank_0("> RL rollout-only: using current weights as reference.")
        return {k: (v.cpu() if v is not None else v) for k, v in model[0].state_dict().items()}

    print_rank_0("> Loading pretrained checkpoint for reference weights in RL training...")
    skip_load_to_model_and_opt = (
        HAVE_FSDP2 and isinstance(cfg.ddp, TorchFullyShardedDataParallelConfig) and cfg.checkpoint.ckpt_format == "torch_dist"
    )
    # The reference weights come from --pretrained-checkpoint, weights-only: the load directory
    # is cleared so the pretrained fallback applies, for this load only (the run's checkpoint
    # policy and the arguments are not written).
    load_checkpoint(
        model,
        None,  # Don't load optimizer state
        None,  # Don't load scheduler state
        checkpointing_context=checkpointing_context,
        restore_training_state=True,
        skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        load_dir=None,
        finetune=True,
        load_optim=False,
    )
    ref_state_dict = {
        k: (v.cpu() if v is not None else v) for k, v in model[0].state_dict().items()
    }

    # Reload RL training checkpoint weights with the run's policy, without the optimizer.
    print_rank_0("> Reloading RL training checkpoint...")
    load_checkpoint(
        model,
        None,
        None,
        checkpointing_context=checkpointing_context,
        restore_training_state=True,
        skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        load_optim=False,
    )
    update_num_microbatches(consumed_samples=get_train_state().consumed_train_samples, verbose=True)
    return ref_state_dict


def reinitialize_grpo_microbatch_calculator(args: Any, data_parallel_size: int) -> None:
    """Reinitialize the microbatch calculator with the GRPO configuration."""
    cfg = get_run_config()
    # Rebuild the calculator on the loop's plain MPU data-parallel size;
    # startup initialized it with the batch-parallel degree (DP x gtp weight-remat sequence shards).
    print_rank_0("> Reinitializing microbatch calculator for GRPO training...")
    destroy_num_microbatches_calculator()
    init_num_microbatches_calculator(
        rank=args.rank,
        global_batch_size=cfg.train.global_batch_size,
        micro_batch_size=cfg.train.micro_batch_size,
        data_parallel_size=data_parallel_size,
        decrease_batch_size_if_needed=cfg.train.decrease_batch_size_if_needed,
        step_batch_size_schedule=cfg.train.step_batch_size_schedule,
        seq_length=args.seq_length,
    )
    print_rank_0(f"> GRPO training: num_microbatches set to {get_num_microbatches()}")


class RLSetupCallback(Callback):
    """Builds the RL loop's state at `on_data_init_start` and keeps it for the rollout hooks."""

    def __init__(
        self,
        inference_container: InferenceConfigContainer,
        *,
        model_provider: Callable[..., torch.nn.Module],
        model_type: ModelType,
    ) -> None:
        self._inference_container = inference_container
        self._model_provider = model_provider
        self._model_type = model_type
        self.model: Optional[list] = None
        self.optimizer: Any = None
        self.inference_model: Optional[list] = None
        self.ref_state_dict: Optional[dict] = None

    def on_data_init_start(self, context: CallbackContext) -> None:
        args = get_args()
        cfg = get_run_config()
        self.model = context.model
        self.optimizer = context.optimizer
        self.inference_model = build_rollout_inference_model(
            self._inference_container,
            args,
            get_model_config(self.model[0]),
            self._model_provider,
            self._model_type,
        )
        if not cfg.validation.skip_train:
            # GRPO loop state; eval-only runs (--skip-train) evaluate the loaded weights directly.
            self.ref_state_dict = load_reference_state_dict(
                self.model,
                context.checkpointing_context,
                args,
                rollout_only=get_rl_config().rl_rollout_only,
            )
            # RL runs on the plain MPU groups: the loop's data-parallel size for them.
            reinitialize_grpo_microbatch_calculator(args, mpu.get_data_parallel_world_size())

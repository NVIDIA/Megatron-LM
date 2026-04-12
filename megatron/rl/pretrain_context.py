# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from contextlib import nullcontext
from typing import Any, List, Optional

import torch

from megatron.core import mpu
from megatron.core.inference.contexts.dynamic_context import HAVE_TORCH_MEMORY_SAVER
from megatron.core.inference.unified_memory import create_unified_mempool
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.resharding.refit import swap_model_weights
from megatron.rl import rl_utils
from megatron.rl.parallel_utils import build_inference_pg_collection
from megatron.training.pretrain_context import PretrainContext
from megatron.training.utils import print_rank_0, unwrap_model

if HAVE_TORCH_MEMORY_SAVER:
    from torch_memory_saver import torch_memory_saver

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as _torch_FSDP  # noqa: F401

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False


# Canonical list of RL timer names included in `timers_to_log` when RL is
# active. Matches the list previously hard-coded in `training.py`. When the
# profiling branch lands this will be imported from `rl_profiling` instead.
RL_LOGGABLE_TIMER_NAMES = [
    # Top-level RL phases
    'rl/rollout-collection',
    'rl/prepare-data-for-update',
    # Rollout collection breakdown
    'rl/inference-setup',
    'rl/collect-rollouts',
    'rl/sync-rollouts',
    'rl/suspend-engine',
    # Optimizer offload/restore
    'rl/offload-optimizer-before-inference',
    'rl/restore-optimizer-after-inference',
    'rl/offload-kv-cache-after-inference',
    'rl/restore-kv-cache-before-inference',
    # Fine-grained offload/restore breakdown
    'rl/restore/grad-buffers',
    'rl/restore/optimizer-state',
    'rl/restore/wait-for-transfers',
    'rl/offload/grad-buffers',
    'rl/offload/optimizer-state',
    # Weight prefetching
    'rl/prefetch-weights-to-gpu',
    'rl/prefetch-weights-to-cpu',
    # Data preparation
    'rl/compute-group-stats',
    'rl/prepare-advantages',
    'rl/prepare-trajectories',
    'rl/get-ltor-masks',
    'rl/create-dataloader',
    'rl/sequence-packing',
    'rl/align-inference-logprobs',
    'rl/log-wandb-tb',
    'rl/pack-sequences',
    'rl/regather-trajectories',
    # Logprobs computation
    'rl/compute-logprobs',
    'rl/compute-old-logprobs',
    'rl/compute-ref-logprobs',
    'rl/get-logprobs',
    'rl/forward-pass',
    'rl/log-softmax',
    # Inference / cuda graphs
    'rl/build-cuda-graphs',
    'rl/wait-for-decode-only',
]


class RLPretrainContext(PretrainContext):
    """Pretrain context for GRPO-style RL training."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.args.perform_rl_step:
            raise ValueError("RLPretrainContext requires --perform-rl-step to be set.")
        self.inference_model = None
        self.ref_state_dict: Optional[dict] = None
        self.buffered_rollouts = None

    def should_build_optimizer(self) -> bool:
        # RL inference-only mode (skip_train + perform_rl_step) still needs
        # the optimizer for --rl-offload-optimizer-during-inference, unless
        # the user explicitly passes --no-load-optim.
        return not (self.args.skip_train and self.args.no_load_optim)

    def should_build_standard_dataloaders(self) -> bool:
        return False

    def attach_training_state(self, **kwargs: Any) -> None:
        super().attach_training_state(**kwargs)
        self.inference_model = self._build_inference_model()
        if (
            self.args.rl_offload_inference_model_weights_when_idle
            and self.inference_model is None
        ):
            raise ValueError(
                "--rl-offload-inference-model-weights-when-idle requires a separate "
                "inference model. This flag is only useful when doing refit since "
                "the weights are shared with the training model."
            )

    def _custom_inference_parallelism_requested(self) -> bool:
        args = self.args
        return (
            args.rl_inference_tensor_model_parallel_size is not None
            or args.rl_inference_pipeline_model_parallel_size is not None
            or args.rl_inference_expert_model_parallel_size is not None
            or args.rl_inference_expert_tensor_model_parallel_size is not None
        )

    def _build_inference_model(self):
        """Build a separate inference model with custom parallelism, if requested."""
        if not self._custom_inference_parallelism_requested():
            return None

        args = self.args
        print_rank_0(
            "Building separate RL inference model with custom parallelism: "
            f"TP={args.rl_inference_tensor_model_parallel_size}, "
            f"PP={args.rl_inference_pipeline_model_parallel_size}, "
            f"EP={args.rl_inference_expert_model_parallel_size}, "
            f"ExptTP={args.rl_inference_expert_tensor_model_parallel_size}"
        )
        inference_pg_collection = build_inference_pg_collection(
            args.world_size,
            tp_size=args.rl_inference_tensor_model_parallel_size,
            pp_size=args.rl_inference_pipeline_model_parallel_size,
            ep_size=args.rl_inference_expert_model_parallel_size,
            expt_tp_size=args.rl_inference_expert_tensor_model_parallel_size,
            use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
        )

        inference_config = copy.deepcopy(self.config)
        if args.rl_inference_tensor_model_parallel_size is not None:
            inference_config.tensor_model_parallel_size = (
                args.rl_inference_tensor_model_parallel_size
            )
        if args.rl_inference_pipeline_model_parallel_size is not None:
            inference_config.pipeline_model_parallel_size = (
                args.rl_inference_pipeline_model_parallel_size
            )
        if args.rl_inference_expert_model_parallel_size is not None:
            inference_config.expert_model_parallel_size = (
                args.rl_inference_expert_model_parallel_size
            )
        if args.rl_inference_expert_tensor_model_parallel_size is not None:
            inference_config.expert_tensor_parallel_size = (
                args.rl_inference_expert_tensor_model_parallel_size
            )

        # Optionally allocate the RL inference model weights from a unified virtual
        # memory (UVM) mempool so we can prefetch weights to CPU when idle while
        # keeping CUDA-graph-safe pointers. Alternatively, use torch_memory_saver
        # to offload the weights to CPU when idle.
        uvm_mempool = None
        uvm_level = args.rl_inference_model_unified_memory_level
        if uvm_level and uvm_level > 0:
            uvm_mempool = create_unified_mempool()

        use_torch_saver_for_inference_model = (
            args.rl_offload_inference_model_weights_when_idle
            and uvm_level == 0
            and HAVE_TORCH_MEMORY_SAVER
        )
        if use_torch_saver_for_inference_model:
            model_alloc_ctx = torch_memory_saver.region(
                tag="rl_inference_model", enable_cpu_backup=True
            )
        elif uvm_mempool is not None:
            model_alloc_ctx = torch.cuda.use_mem_pool(uvm_mempool)
        else:
            model_alloc_ctx = nullcontext()

        from megatron.training.training import get_model

        with model_alloc_ctx:
            inference_model = get_model(
                self.model_provider,
                self.model_type,
                wrap_with_ddp=False,
                pg_collection=inference_pg_collection,
                config=inference_config,
            )
        inference_model[0].eval()
        return inference_model
    # ------------------------------------------------------------------

    def should_run_train_loop(self) -> bool:
        # RL inference-only mode (--skip-train --perform-rl-step) still enters the training loop.
        return True

    def before_train_loop(self) -> None:
        self._load_reference_weights()
        self._reinitialize_microbatch_calculator()

    def _load_reference_weights(self) -> None:
        from megatron.training.checkpointing import load_checkpoint

        args = self.args
        if args.skip_train:
            print_rank_0("> RL inference-only: using current weights as reference.")
            self.ref_state_dict = {
                k: (v.cpu() if v is not None else v)
                for k, v in self.model[0].state_dict().items()
            }
            return

        print_rank_0(
            "> Loading pretrained checkpoint for reference weights in RL training..."
        )
        load, finetune, no_load_optim = args.load, args.finetune, args.no_load_optim
        args.no_load_optim = True
        args.load = None
        args.finetune = True
        load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context=self.checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
            and getattr(args, "use_torch_fsdp2", False)
            and args.ckpt_format == "torch_dist",
        )
        self.ref_state_dict = {
            k: (v.cpu() if v is not None else v)
            for k, v in self.model[0].state_dict().items()
        }

        args.load = load
        args.finetune = finetune
        print_rank_0("> Reloading RL training checkpoint...")
        load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context=self.checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
            and getattr(args, "use_torch_fsdp2", False)
            and args.ckpt_format == "torch_dist",
        )
        args.no_load_optim = no_load_optim

    def _reinitialize_microbatch_calculator(self) -> None:
        # IMPORTANT FIX: For RL training, reinitialize the microbatch calculator with
        # the correct configuration.
        # TODO: Understand and document why this IMPORTANT FIX is necessary.
        from megatron.core.num_microbatches_calculator import (
            destroy_num_microbatches_calculator,
            init_num_microbatches_calculator,
        )

        args = self.args
        print_rank_0("> Reinitializing microbatch calculator for GRPO training...")
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            args.rank,
            args.rampup_batch_size,
            args.global_batch_size,
            args.micro_batch_size,
            mpu.get_data_parallel_world_size(),
            args.decrease_batch_size_if_needed,
        )
        print_rank_0(
            f"> GRPO training: num_microbatches set to {get_num_microbatches()}"
        )

    def begin_iteration(self, iteration: int) -> None:
        args = self.args
        if self.optimizer is None:
            # Release stale CUDA cached memory before inference.
            torch.cuda.empty_cache()
        with torch.no_grad():
            # Buffered rollouts are reused across GRPO epochs; get_grpo_data_iterator
            # rebuilds them when the current collection has been consumed.
            self.buffered_rollouts = rl_utils.get_grpo_data_iterator(
                self.model,
                self.inference_model,
                self.optimizer,
                iteration,
                self.ref_state_dict,
                grpo_iterations=args.grpo_iterations,
                grpo_prompts_per_step=args.grpo_prompts_per_step,
                grpo_group_size=args.grpo_group_size,
                global_batch_size=args.global_batch_size,
                sequence_packing=args.rl_use_sequence_packing,
                buffered_rollouts=self.buffered_rollouts,
                is_correction=args.rl_inference_logprobs_is_correction,
            )
        self.train_data_iterator = self.buffered_rollouts

    def iteration_sequence_count(self) -> int:
        if self.args.rl_use_sequence_packing:
            return rl_utils.get_iteration_sequence_count(self.args)
        return super().iteration_sequence_count()

    def iteration_bin_count(self) -> Optional[int]:
        if self.args.rl_use_sequence_packing:
            return (
                mpu.get_data_parallel_world_size()
                * self.args.micro_batch_size
                * get_num_microbatches()
            )
        return None

    def on_microbatch_change(
        self, iteration: int, old_num_microbatches: int, new_num_microbatches: int
    ) -> bool:
        if self.args.rl_use_sequence_packing:
            print_rank_0(
                f"[Sequence Packing] Skipping automatic checkpoint at iteration "
                f"{iteration} (microbatch change: {old_num_microbatches} -> "
                f"{new_num_microbatches})"
            )
            return False
        return True

    def run_eval(
        self,
        *,
        prefix: str,
        iteration: int,
        verbose: bool,
        write_to_tensorboard: bool,
    ) -> None:
        eval_model = self.model
        training_model = None
        if self.inference_model is not None:
            # Swap the training weights back onto the inference model before eval.
            inf_core = unwrap_model(self.inference_model[0])
            rl_utils._maybe_prefetch_separate_inference_model_weights(
                inf_core, to_cpu=False
            )
            swap_model_weights(self.model, self.inference_model, self.args.refit_method)
            eval_model = self.inference_model
            training_model = self.model
        rl_utils.evaluate_and_print_results_rl(
            self.valid_data_iterator,
            eval_model,
            self.optimizer,
            iteration,
            write_to_tensorboard=write_to_tensorboard,
            training_model=training_model,
        )

    def extra_log_timers(self) -> List[str]:
        return list(RL_LOGGABLE_TIMER_NAMES)

    def log_tensorboard_metrics(self, writer, wandb_writer, iteration: int) -> None:
        args = self.args
        if args.rl_use_sequence_packing:
            packing_metrics = rl_utils.get_sequence_packing_tensorboard_metrics(args)
            for metric_name, metric_value in packing_metrics.items():
                writer.add_scalar(metric_name, metric_value, iteration)
            if wandb_writer and packing_metrics:
                wandb_writer.log(packing_metrics, iteration)
        samples_per_collection = (
            args.grpo_iterations * (args.grpo_samples_per_iteration // args.global_batch_size)
        )
        grpo_collection_iteration = iteration // samples_per_collection
        writer.add_scalar(
            'grpo_collection_iteration', grpo_collection_iteration, iteration
        )
        if wandb_writer:
            wandb_writer.log(
                {'grpo_collection_iteration': grpo_collection_iteration}, iteration
            )

    def extra_log_string(self) -> str:
        if self.args.rl_use_sequence_packing:
            return rl_utils.get_sequence_packing_log_info(self.args)
        return ""

    def shutdown(self) -> None:
        rl_utils.rl_inference_interface_shutdown()


def rl_pretrain_context_factory(**kwargs: Any) -> RLPretrainContext:
    """RL-specific factory: construct an RLPretrainContext."""
    return RLPretrainContext(**kwargs)

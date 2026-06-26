# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous MIMO (Nemotron6-MoE VLM) training entry point.

Builds the per-module model and process groups for the disjoint vision/language
grids, then drives training through ``pretrain()`` with model/optimizer and
data-iterator hooks.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.mimo.model_providers.nemotron_moe_vlm import add_model_provider_args
from examples.mimo.training.args import add_hetero_grid_args, validate_hetero_grid_args
from examples.mimo.training.bootstrap import MimoRuntime, build_mimo_runtime, mimo_model_provider
from examples.mimo.training.data import add_data_args
from examples.mimo.training.distributed import initialize_distributed, shutdown_distributed
from examples.mimo.training.step import mimo_forward_step
from megatron.core.config import set_experimental_flag
from megatron.core.enums import ModelType
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.training.training import get_optimizer_param_scheduler, pretrain


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the model-provider, hetero-grid, and data arg groups."""
    parser = add_model_provider_args(parser)
    parser = add_hetero_grid_args(parser)
    parser = add_data_args(parser)
    return parser


def _parse_and_validate() -> argparse.Namespace:
    """Parse/validate args; architecture flags come from the CLI (run script)."""
    args = parse_args(extra_args_provider)
    validate_args(args, {})
    validate_hetero_grid_args(args, int(os.environ.get("WORLD_SIZE", args.world_size)))

    args.data_parallel_size = args.llm_dp
    if getattr(args, "mtp_num_layers", None) is None:
        args.mtp_num_layers = 0
    if getattr(args, "padded_vocab_size", None) is None:
        # No tokenizer in the mock run; pad the vocab to the language TP shard.
        multiple = args.make_vocab_size_divisible_by * args.llm_tp
        args.padded_vocab_size = int(math.ceil(args.vocab_size / multiple) * multiple)
    args.dataloader_type = "external"  # per-rank iterator passed through
    # Mock data is train-only; disable eval and keep eval_interval positive (modulo guard).
    args.eval_iters = 0
    if getattr(args, "eval_interval", None) is None:
        args.eval_interval = args.train_iters or 1
    return args


def _setup_globals(args: argparse.Namespace) -> None:
    """Set global state; the microbatch calculator keys on the language data-parallel size."""
    set_global_variables(args, build_tokenizer=False)
    if args.enable_experimental:
        set_experimental_flag(True)


def _build_optimizer(args: argparse.Namespace, model):
    """Build the MimoOptimizer from a shared OptimizerConfig."""
    return get_mimo_optimizer(
        model[0],
        OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            clip_grad=args.clip_grad,
            bf16=args.bf16,
            fp16=args.fp16,
            use_distributed_optimizer=True,
            log_num_zeros_in_grad=args.log_num_zeros_in_grad,
        ),
    )


def _mimo_branch_name(topology) -> str:
    """Return this rank's MIMO branch ("language" or the encoder grid name)."""
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

    if topology.grids[MIMO_LANGUAGE_MODULE_KEY].is_current_rank_in_grid():
        return "language"
    return next(
        (
            name
            for name, grid in topology.grids.items()
            if name != MIMO_LANGUAGE_MODULE_KEY and grid.is_current_rank_in_grid()
        ),
        "language",
    )


class MimoSetup:
    """pretrain setup_model_and_optimizer_func: return the eagerly-built per-submodule-DDP
    MimoModel + MimoOptimizer + scheduler, and drive resume load with per-module groups.
    """

    def __init__(self, runtime: MimoRuntime, args: argparse.Namespace):
        self.runtime = runtime
        self.args = args

    def __call__(self, model_provider, model_type, checkpointing_context=None):
        runtime, args = self.runtime, self.args
        optimizer = _build_optimizer(args, runtime.model)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        # Per-grid rng key namespace on the model (read by stock save/load); torch_dist only.
        runtime.model[0].rng_state_key_prefix = f"mimo.{_mimo_branch_name(runtime.topology)}."
        args.use_distributed_optimizer = True
        if args.load:
            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                runtime.model,
                optimizer,
                opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                tp_group=runtime.pg_collection.tp,
                pp_group=runtime.pg_collection.pp,
                dp_cp_group=runtime.pg_collection.dp_cp,
                rng_state_key_prefix=runtime.model[0].rng_state_key_prefix,
            )
        else:
            args.iteration = 0
            args.num_floating_point_operations_so_far = 0
        return runtime.model, optimizer, opt_param_scheduler


def main() -> None:
    """Run heterogeneous MIMO training."""
    args = _parse_and_validate()

    _setup_globals(args)
    initialize_distributed()

    # Per-rank runtime: per-submodule-DDP MimoModel, topology, comms, data iterator, grad sync.
    # build_mimo_runtime seeds per-role RNG (host + CUDA) via the stock _set_random_seed.
    rt = build_mimo_runtime(args)
    assert rt.model[0].config.finalize_model_grads_func is not None, (
        "build_mimo_runtime must install a finalize_model_grads_func on the language config"
    )

    cfg = pretrain_cfg_container_from_args(args, rt.model[0].config)

    def _dataset_provider(train_val_test_num_samples):
        return rt.data_iterator, None, None

    _dataset_provider.is_distributed = True

    try:
        pretrain(
            cfg,
            _dataset_provider,
            functools.partial(mimo_model_provider, topology=rt.topology, args=args),
            ModelType.encoder_or_decoder,
            mimo_forward_step,
            skip_model_parallel_init=True,
            setup_model_and_optimizer_func=MimoSetup(rt, args),
            p2p_communicator=rt.communicator,
            schedule_pg_collection=rt.topology.schedule_pg_collection,
        )
    finally:
        if hasattr(rt.model[0], "destroy"):
            rt.model[0].destroy()
        rt.topology.destroy()
        shutdown_distributed()


if __name__ == "__main__":
    main()

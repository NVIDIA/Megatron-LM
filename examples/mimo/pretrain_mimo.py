# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous MIMO (Nemotron6-MoE VLM) training entry point.

Builds the per-module model and process groups for the disjoint vision/language
grids, then drives training through ``pretrain()`` with model/optimizer and
data-iterator hooks.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.mimo.model_providers.nemotron_moe_vlm import (
    add_model_provider_args,
    prepare_model_provider_args,
    validate_model_provider_args,
)
from examples.mimo.training.args import add_hetero_grid_args, validate_hetero_grid_args
from examples.mimo.training.bootstrap import build_mimo_runtime
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
    """Parse and validate args with the model-provider preset and grid checks."""
    args = parse_args(extra_args_provider)

    # Apply the model-provider preset before validation so preset-derived sizes flow in.
    prepare_model_provider_args(args)

    validate_args(args, {})  # no dataset path / tokenizer for the mock run

    validate_model_provider_args(args)
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    validate_hetero_grid_args(args, world_size)

    # Data-parallel size follows the language grid.
    args.data_parallel_size = args.llm_dp

    # The preset leaves mtp_num_layers None; FLOPs/throughput accounting needs an int.
    if getattr(args, "mtp_num_layers", None) is None:
        args.mtp_num_layers = 0

    # Mock runs build no tokenizer; derive the padded vocab from the configured vocab.
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = args.vocab_size

    return args


def _setup_globals(args: argparse.Namespace) -> None:
    """Set global state; the microbatch calculator keys on the language data-parallel size."""
    set_global_variables(args, build_tokenizer=False)
    if args.enable_experimental:
        set_experimental_flag(True)


def _seed_everything(args: argparse.Namespace) -> None:
    """Seed host RNG; per-module CUDA RNG is seeded inside build_mimo_runtime."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


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


def _noop_provider(*_args, **_kwargs):
    """Placeholder for pretrain's model/dataset providers (the hooks replace them)."""
    return None


def main() -> None:
    """Run heterogeneous MIMO training."""
    args = _parse_and_validate()

    _setup_globals(args)
    initialize_distributed()
    _seed_everything(args)

    # Per-rank runtime: per-submodule-DDP MimoModel, topology, comms, data iterator, grad sync.
    rt = build_mimo_runtime(args)
    assert rt.model[0].config.finalize_model_grads_func is not None, (
        "build_mimo_runtime must install a finalize_model_grads_func on the language config"
    )

    cfg = pretrain_cfg_container_from_args(args, rt.model[0].config)

    def _setup_model_and_optimizer(model_provider, model_type, checkpointing_context=None):
        # Build the optimizer/scheduler, drive resume load, and set args.iteration.
        optimizer = _build_optimizer(args, rt.model)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        # Per-branch RNG key namespace + distributed-optimizer sharding for checkpoints.
        args.rng_state_key_prefix = f"mimo.{_mimo_branch_name(rt.topology)}."
        args.use_distributed_optimizer = True
        if args.load:
            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                rt.model,
                optimizer,
                opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                tp_group=rt.pg_collection.tp,
                pp_group=rt.pg_collection.pp,
                dp_cp_group=rt.pg_collection.dp_cp,
            )
        else:
            args.iteration = 0
            args.num_floating_point_operations_so_far = 0
        return rt.model, optimizer, opt_param_scheduler

    def _build_data_iterators():
        from megatron.core.rerun_state_machine import RerunDataIterator

        args.do_train = True
        args.do_valid = False
        args.do_test = False
        train_iter = rt.data_iterator
        if train_iter is not None and not isinstance(train_iter, RerunDataIterator):
            train_iter = RerunDataIterator(train_iter)
        return train_iter, None, None

    try:
        pretrain(
            cfg,
            _noop_provider,
            _noop_provider,
            ModelType.encoder_or_decoder,
            mimo_forward_step,
            skip_model_parallel_init=True,
            setup_model_and_optimizer_func=_setup_model_and_optimizer,
            build_data_iterators_func=_build_data_iterators,
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

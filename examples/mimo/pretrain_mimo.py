# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hetero MIMO (NMFW-516 PR-E5) on the stock Megatron ``train()`` loop.

Drives the Nemotron6-MoE VLM (20L) over the non-colocated path (encoder/language
grids on disjoint rank spans) with mock data: replicates the non-MPU pieces of
``initialize_megatron``, feeds E4 ``build_mimo_runtime``'s model to ``train()``,
re-keys ``data_parallel_size`` to ``llm_dp``, and pins ``parallel_state`` to this
rank's own grid so the stock checkpoint/FLOPs/logging paths work without MPU init.
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

import megatron.training.training as training_module
from examples.mimo.model_providers.nemotron_moe_vlm import (
    add_model_provider_args,
    prepare_model_provider_args,
    validate_model_provider_args,
)
from examples.mimo.training.args import add_hetero_grid_args, validate_hetero_grid_args
from examples.mimo.training.bootstrap import build_mimo_runtime
from examples.mimo.training.data import add_data_args
from examples.mimo.training.distributed import (
    initialize_distributed,
    print_rank_0,
    shutdown_distributed,
)
from examples.mimo.training.step import mimo_forward_step
from megatron.core import parallel_state
from megatron.core.config import set_experimental_flag
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.global_vars import set_global_variables as _stock_set_global_variables
from megatron.training.training import get_optimizer_param_scheduler


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the model-provider, hetero-grid, and data arg groups."""
    parser = add_model_provider_args(parser)
    parser = add_hetero_grid_args(parser)
    parser = add_data_args(parser)
    return parser


def _parse_and_validate() -> argparse.Namespace:
    """Stock arg pipeline plus the MIMO preset/validation hooks and the dp_size fix."""
    args = parse_args(extra_args_provider)

    # Apply the Nemotron preset BEFORE stock validation so preset-derived sizes flow in.
    prepare_model_provider_args(args)

    validate_args(args, {})  # no dataset path / tokenizer for the mock run

    validate_model_provider_args(args)
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    validate_hetero_grid_args(args, world_size)

    # Re-key DP on the language grid (stock keyed it on the world-spanning flags).
    args.data_parallel_size = args.llm_dp

    # Nemotron preset leaves mtp_num_layers None; stock FLOPs/throughput needs an int.
    if getattr(args, "mtp_num_layers", None) is None:
        args.mtp_num_layers = 0

    return args


def _setup_globals(args: argparse.Namespace) -> None:
    """Set non-MPU global state; the calculator is keyed on the already-fixed llm_dp."""
    _stock_set_global_variables(args, build_tokenizer=False)
    if args.enable_experimental:
        set_experimental_flag(True)


def _seed_everything(args: argparse.Namespace) -> None:
    """Seed host (python/numpy/torch) RNG; E4 configure_module_rng owns the CUDA tracker."""
    # TODO(reuse): stock _set_random_seed would clobber E4's per-module CUDA RNG offsets (#5285).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def _build_optimizer(args: argparse.Namespace, model):
    """Build the MimoOptimizer (one shared OptimizerConfig, distributed, bf16 unless --fp32)."""
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
            bf16=not args.fp32,
            use_distributed_optimizer=True,
            log_num_zeros_in_grad=args.log_num_zeros_in_grad,
        ),
    )


def _set_mpu_data_parallel_world_size(args: argparse.Namespace) -> None:
    """Pin the MPU DP world size to llm_dp so train()'s sample accounting needs no MPU init."""
    # Bootstrap/MPU-materialization compatibility point (see CLAUDE.md).
    parallel_state._MPU_DATA_PARALLEL_WORLD_SIZE = args.llm_dp


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


def main() -> None:
    """Program entrypoint: stock-args MIMO run on the stock train() loop."""
    args = _parse_and_validate()

    # Non-MPU global setup, then torch.distributed bring-up (no MPU init).
    _setup_globals(args)
    initialize_distributed()
    _seed_everything(args)
    _set_mpu_data_parallel_world_size(args)

    # Per-rank runtime: per-submodule-DDP MimoModel + topology + communicator +
    # role-aware data iterator + grad-sync hook (E4).
    rt = build_mimo_runtime(args)

    optimizer = _build_optimizer(args, rt.model)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    # E4 configure_grad_sync installs the MIMO dual grad-finalizer on model.config;
    # stock train() preserves a caller-installed finalize_model_grads_func, so it
    # survives without a monkeypatch.
    assert rt.model[0].config.finalize_model_grads_func is not None, (
        "expected build_mimo_runtime -> configure_grad_sync to install a MIMO "
        "finalize_model_grads_func on the language config"
    )

    # TODO(reuse): stock num_floating_point_operations computes the language model's
    # FLOPs on every rank and the throughput print divides by the full world_size
    # (not the LLM world size), so per-GPU TFLOP/s is cosmetic for the hetero layout.
    # Proper per-module hetero FLOPs accounting is deferred (NMFW-516).

    # Pin parallel_state to this rank's per-module groups so stock-path reads
    # (training_log, report_memory, checkpoint shard/RNG) work without full mpu init.
    _pgc = rt.pg_collection
    if getattr(_pgc, "mp", None) is not None:
        parallel_state._MODEL_PARALLEL_GROUP = _pgc.mp
    if getattr(_pgc, "tp", None) is not None:
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = _pgc.tp
    if getattr(_pgc, "pp", None) is not None:
        parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = _pgc.pp
    if getattr(_pgc, "dp", None) is not None:
        parallel_state._DATA_PARALLEL_GROUP = _pgc.dp
        if getattr(_pgc, "dp_cp", None) is not None:
            parallel_state._DATA_PARALLEL_GROUP_WITH_CP = _pgc.dp_cp

    # Bookkeeping fields stock train() reads that setup_model_and_optimizer /
    # the dataset builder would normally set.
    args.iteration = 0
    args.num_floating_point_operations_so_far = 0
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.do_train = True
    args.do_valid = False
    args.do_test = False

    # Every MIMO submodule optimizer is a DistributedOptimizer, but the top-level
    # ``args`` flag defaults False because we bypass setup_model_and_optimizer. Pin
    # it True so stock checkpointing selects the 'dp_reshardable' optimizer sharding
    # format (the format that round-trips cleanly through torch_dist save/load).
    args.use_distributed_optimizer = True

    # Reuse stock save_checkpoint/load_checkpoint. The hetero deadlock came from the
    # default fully-parallel save wrapper; the run script passes
    # --no-ckpt-fully-parallel-save so stock uses the plain (world-coordinated)
    # torch_dist strategy. Namespace the RNG ShardedObject key per branch so the
    # encoder/LLM grids (which can share a (pp,tp) factorization) don't collide.
    args.rng_state_key_prefix = f"mimo.{_mimo_branch_name(rt.topology)}."
    checkpointing_context = {}

    # Resume LOAD: stock load_checkpoint normally runs inside the bypassed
    # setup_model_and_optimizer, so we drive it here (after the parallel_state pins +
    # use_distributed_optimizer so the load request's sharded keys match the save).
    if args.load:
        resume_iter, resume_nfpo = load_checkpoint(
            rt.model, optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context
        )
        args.iteration = resume_iter
        args.num_floating_point_operations_so_far = resume_nfpo

    config = rt.model[0].config

    print_rank_0(
        "Starting hetero MIMO training on stock train(): "
        f"world_size={torch.distributed.get_world_size()}, "
        f"llm_dp={args.llm_dp}, "
        f"global_batch_size={args.global_batch_size}, train_iters={args.train_iters}"
    )

    try:
        training_module.train(
            forward_step_func=mimo_forward_step,
            model=rt.model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            train_data_iterator=rt.data_iterator,
            valid_data_iterator=None,
            process_non_loss_data_func=None,
            config=config,
            checkpointing_context=checkpointing_context,
            non_loss_data_func=None,
            p2p_communicator=rt.communicator,
            schedule_pg_collection=rt.topology.schedule_pg_collection,
        )
        torch.distributed.barrier()
        print_rank_0("Hetero MIMO training (stock loop) completed")
    finally:
        if hasattr(rt.model[0], "destroy"):
            rt.model[0].destroy()
        rt.topology.destroy()
        shutdown_distributed()


if __name__ == "__main__":
    main()

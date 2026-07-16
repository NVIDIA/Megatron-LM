# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hetero grid/topology CLI args + validation for the MIMO example."""

from __future__ import annotations

import argparse
from typing import List

from examples.mimo.training.topology import ModuleGridSpec
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

MIMO_LAYOUT_NON_COLOCATED = "non-colocated"
MIMO_LAYOUT_COLOCATED = "colocated"


def add_hetero_grid_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register hetero parallelism args for the single-encoder MIMO example."""
    grid = parser.add_argument_group("hetero module grids")
    grid.add_argument(
        "--mimo-layout",
        choices=(MIMO_LAYOUT_NON_COLOCATED, MIMO_LAYOUT_COLOCATED),
        default=MIMO_LAYOUT_NON_COLOCATED,
        help="Place encoder and language grids on disjoint ranks or colocate both on every rank.",
    )

    # Single encoder grid; CP/PP stay fixed at 1.
    grid.add_argument("--encoder-tp", type=int, default=2,
                      help="Encoder tensor-model-parallel size.")
    grid.add_argument("--encoder-dp", type=int, default=2,
                      help="Encoder data-parallel size.")

    # Language grid placement + factorization.
    grid.add_argument("--llm-offset", type=int, default=4,
                      help="First global rank of the language grid span.")
    grid.add_argument("--llm-tp", type=int, default=2,
                      help="Language tensor-model-parallel size.")
    grid.add_argument("--llm-cp", type=int, default=1,
                      help="Language context-parallel size (CP=1 only for now).")
    grid.add_argument("--llm-pp", type=int, default=1,
                      help="Language pipeline-model-parallel size.")
    grid.add_argument("--llm-dp", type=int, default=2,
                      help="Language data-parallel size. Global batch is keyed on this.")
    # MoE expert parallelism for the language grid.
    grid.add_argument("--llm-ep", type=int, default=1,
                      help="Language expert-model-parallel size (MoE).")
    grid.add_argument("--llm-expt-tp", type=int, default=None,
                      help="Language expert tensor-parallel size; defaults to 1 when unset "
                           "(experts default to TP=1; the 20L MoE recipe passes --llm-expt-tp 1).")

    grid.add_argument(
        "--llm-only",
        action="store_true",
        help=(
            "Run only the MIMO language module on the LLM grid. Keeps the MIMO "
            "training/data path but creates no encoder ranks or bridge communicators; "
            "requires --llm-offset 0 so the language grid covers WORLD_SIZE."
        ),
    )
    return parser


def validate_hetero_grid_args(args: argparse.Namespace, world_size: int) -> tuple[int, int]:
    """Validate the hetero grid layout; returns ``(encoder_size, llm_size)``."""
    if args.llm_cp != 1:
        raise ValueError("hetero MIMO training currently supports CP=1 only")

    # MoE expert count must divide evenly across the language grid's expert parallelism.
    num_experts = _num_experts(args)
    if num_experts and num_experts % args.llm_ep != 0:
        raise ValueError(
            f"--num-experts ({num_experts}) must be divisible by --llm-ep ({args.llm_ep})"
        )

    llm_size = args.llm_tp * args.llm_cp * args.llm_pp * args.llm_dp
    layout = args.mimo_layout

    if args.llm_only:
        if layout == MIMO_LAYOUT_COLOCATED:
            raise ValueError("colocated MIMO requires an encoder module")
        if args.llm_offset != 0:
            raise ValueError(
                "--llm-only requires --llm-offset 0 so language ranks cover WORLD_SIZE"
            )
        llm_ranks = set(range(args.llm_offset, args.llm_offset + llm_size))
        all_ranks = set(range(world_size))
        if llm_ranks != all_ranks:
            raise ValueError(
                "--llm-only requires the language grid to cover every torchrun rank exactly "
                f"once; covered={sorted(llm_ranks)}, world={sorted(all_ranks)}"
            )
        return 0, llm_size

    encoder_size = args.encoder_tp * args.encoder_dp
    if layout == MIMO_LAYOUT_COLOCATED:
        if args.llm_pp != 1 or args.llm_offset != 0:
            raise ValueError("colocated MIMO requires --llm-pp 1 and --llm-offset 0")
        if encoder_size != world_size or llm_size != world_size:
            raise ValueError(
                "colocated encoder and language grids must each cover WORLD_SIZE; "
                f"encoder={encoder_size}, language={llm_size}, world={world_size}"
            )
        smaller_dp, larger_dp = sorted((args.encoder_dp, args.llm_dp))
        if larger_dp % smaller_dp:
            raise ValueError("colocated encoder and language DP sizes must divide evenly")
        if (args.micro_batch_size * args.llm_dp) % args.encoder_dp:
            raise ValueError("colocated encoder micro-batch size must be integral")
        return encoder_size, llm_size
    if layout != MIMO_LAYOUT_NON_COLOCATED:
        raise ValueError(f"unsupported MIMO layout: {layout}")

    # Fan-out divisibility: the bridge splits (mbs * llm_dp) LLM lanes across
    # encoder_dp encoder lanes; the split must be exact.
    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError(
            "--micro-batch-size * --llm-dp must be divisible by --encoder-dp "
            f"(got {args.micro_batch_size} * {args.llm_dp} % {args.encoder_dp} != 0)"
        )

    encoder_ranks = set(range(encoder_size))  # encoder span always starts at rank 0
    llm_ranks = set(range(args.llm_offset, args.llm_offset + llm_size))
    all_ranks = set(range(world_size))

    if not encoder_ranks.isdisjoint(llm_ranks):
        raise ValueError(
            "hetero MIMO expects disjoint module rank spans; "
            f"spans overlap at {sorted(encoder_ranks & llm_ranks)}"
        )
    if encoder_ranks | llm_ranks != all_ranks:
        raise ValueError(
            "The non-colocated module grids must cover every torchrun rank exactly once; "
            f"covered={sorted(encoder_ranks | llm_ranks)}, world={sorted(all_ranks)}"
        )

    return encoder_size, llm_size


def validate_colocated_runtime_args(args: argparse.Namespace) -> None:
    """Reject runtime modes not supported by the initial colocated path."""
    if args.mimo_layout != MIMO_LAYOUT_COLOCATED:
        return

    unsupported = {
        "activation recompute": args.recompute_granularity is not None,
        "async checkpoint save": args.async_save,
        "rerun (pass --rerun-mode disabled)": args.rerun_mode != "disabled",
        "TE RNG tracker": args.te_rng_tracker,
        "parameter-norm logging": args.log_params_norm,
        "zero-gradient logging": args.log_num_zeros_in_grad,
        "FP16": args.fp16,
        "FSDP": args.use_megatron_fsdp or args.use_torch_fsdp2,
        "DDP overlap": bool(
            args.overlap_grad_reduce
            or args.overlap_param_gather
            or args.overlap_param_gather_with_optimizer_step
        ),
        "model upcycling": args.moe_use_upcycling,
        "meta-device initialization": args.init_model_with_meta_device,
        "CUDA graphs": args.cuda_graph_impl != "none",
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError("colocated MIMO does not support: " + ", ".join(enabled))
    if args.llm_expt_tp not in (None, 1):
        raise ValueError("colocated MIMO requires --llm-expt-tp 1")
    if args.eval_micro_batch_size != args.micro_batch_size:
        raise ValueError("colocated train and eval micro-batch sizes must match")

    checkpoint_requested = bool(args.save or args.load or args.pretrained_checkpoint)
    if checkpoint_requested and not args.data_parallel_random_init:
        raise ValueError("colocated checkpointing requires --data-parallel-random-init")
    if checkpoint_requested and args.ckpt_format != "torch_dist":
        raise ValueError("colocated checkpointing requires --ckpt-format torch_dist")
    if checkpoint_requested and (args.ckpt_fully_parallel_save or args.ckpt_fully_parallel_load):
        raise ValueError("colocated MIMO does not support fully parallel checkpointing")


def build_module_grid_specs(
    args: argparse.Namespace, world_size: int, encoder_module_name: str
) -> List[ModuleGridSpec]:
    """Map grid args to the ModuleGridSpec list create_topology consumes."""
    encoder_size, llm_size = validate_hetero_grid_args(args, world_size)

    language_grid_spec = ModuleGridSpec(
        name=MIMO_LANGUAGE_MODULE_KEY,
        num_ranks=llm_size,
        tp=args.llm_tp,
        cp=args.llm_cp,
        pp=args.llm_pp,
        ep=args.llm_ep,
        rank_offset=args.llm_offset,
        expt_tp=args.llm_expt_tp or 1,
    )

    if args.llm_only:
        return [language_grid_spec]

    encoder_grid_spec = ModuleGridSpec(
        name=encoder_module_name,
        num_ranks=encoder_size,
        tp=args.encoder_tp,
        cp=1,
        pp=1,
        ep=1,
        rank_offset=0,
        expt_tp=1,
    )
    return [encoder_grid_spec, language_grid_spec]


def _num_experts(args: argparse.Namespace) -> int:
    """Resolve MoE expert count from the stock --num-experts arg."""
    value = getattr(args, "num_experts", None)
    return int(value) if value else 0

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hetero grid/topology CLI args + validation for the MIMO example."""

from __future__ import annotations

import argparse
import math
from typing import List

from examples.mimo.training.topology import ModuleGridSpec
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

# Default encoder grid module name (single-sourced with radio_encoder.py in #5397).
DEFAULT_ENCODER_MODULE_NAME = "radio_encoder"


def add_hetero_grid_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the hetero per-module parallelism args (--encoder-*/--llm-*)."""
    grid = parser.add_argument_group("hetero module grids")

    # Encoder grid factorization (the encoder span always starts at rank 0).
    grid.add_argument("--encoder-tp", type=int, default=2,
                      help="Encoder tensor-model-parallel size.")
    grid.add_argument("--encoder-cp", type=int, default=1,
                      help="Encoder context-parallel size (CP=1 only for now).")
    grid.add_argument("--encoder-pp", type=int, default=1,
                      help="Encoder pipeline-model-parallel size.")
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
    grid.add_argument("--llm-expt-dp", type=int, default=None,
                      help="Language expert data-parallel size; derived from the grid when unset.")

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
    """Validate the disjoint hetero grid layout; returns ``(encoder_size, llm_size)``."""
    if args.encoder_cp != 1 or args.llm_cp != 1:
        raise ValueError("hetero MIMO training currently supports CP=1 only")

    # MoE expert count must divide evenly across the language grid's expert parallelism.
    num_experts = _num_experts(args)
    if num_experts and num_experts % args.llm_ep != 0:
        raise ValueError(
            f"--num-experts ({num_experts}) must be divisible by --llm-ep ({args.llm_ep})"
        )

    # Sample-based scheduler resolution: derive --train-iters from --train-samples
    # using the llm_dp-keyed global batch size.
    if getattr(args, "train_samples", None) is not None:
        derived_gbs = args.micro_batch_size * _num_microbatches(args) * args.llm_dp
        gbs = args.global_batch_size if getattr(args, "global_batch_size", None) else derived_gbs
        if gbs <= 0:
            raise ValueError(
                "--train-samples requires a positive derived/explicit --global-batch-size"
            )
        args.train_iters = math.ceil(args.train_samples / gbs)

    llm_size = args.llm_tp * args.llm_cp * args.llm_pp * args.llm_dp

    if args.llm_only:
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

    # Fan-out divisibility: the bridge splits (mbs * llm_dp) LLM lanes across
    # encoder_dp encoder lanes; the split must be exact.
    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError(
            "--micro-batch-size * --llm-dp must be divisible by --encoder-dp "
            f"(got {args.micro_batch_size} * {args.llm_dp} % {args.encoder_dp} != 0)"
        )

    encoder_size = args.encoder_tp * args.encoder_cp * args.encoder_pp * args.encoder_dp
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


def build_module_grid_specs(
    args: argparse.Namespace, world_size: int
) -> List[ModuleGridSpec]:
    """Map parsed grid args onto the ModuleGridSpec list create_topology consumes.

    Returns ``[encoder_grid_spec, language_grid_spec]`` (or just the language spec
    when ``--llm-only``). ``num_ranks`` is the ground truth ModuleGridSpec field;
    ``dp`` / ``expt_dp`` are derived in ``ModuleGridSpec.__post_init__`` from the
    tp/cp/pp/ep/expt_tp factorization, so we pass ``num_ranks = tp*cp*pp*dp`` and
    let the dataclass re-derive dp (matching the supplied value) and expt_dp.

    Must be called after :func:`validate_hetero_grid_args` so the spans are known
    to tile ``[0, world_size)``.
    """
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

    encoder_name = getattr(args, "vision_encoder_key", None) or DEFAULT_ENCODER_MODULE_NAME
    encoder_grid_spec = ModuleGridSpec(
        name=encoder_name,
        num_ranks=encoder_size,
        tp=args.encoder_tp,
        cp=args.encoder_cp,
        pp=args.encoder_pp,
        ep=1,
        rank_offset=0,
        expt_tp=1,
    )
    return [encoder_grid_spec, language_grid_spec]


def _num_experts(args: argparse.Namespace) -> int:
    """Resolve MoE expert count from the stock --num-experts arg."""
    value = getattr(args, "num_experts", None)
    return int(value) if value else 0


def _num_microbatches(args: argparse.Namespace) -> int:
    """Resolve the per-step microbatch count from whichever arg the loop exposes."""
    value = getattr(args, "num_microbatches", None)
    if value:
        return int(value)
    return 1

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hetero grid/topology CLI args + validation for the MIMO example."""

from __future__ import annotations

import argparse
from typing import List

from examples.mimo.training.topology import ModuleGridSpec
from examples.mimo.utils.hetero import get_language_batch_parallel_size
from megatron.core.model_parallel_config import resolve_tensor_parallel_weight_shards
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.tensor_parallel.gtp_parallel_layout import (
    resolve_tensor_parallel_sequence_shards,
)


def add_mimo_dcp_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register execution choices for adapters using packed MIMO training."""
    group = parser.add_argument_group("packed MIMO")
    group.add_argument(
        "--mimo-dcp-encoder-schedule",
        choices=("whole-step", "per-round"),
        default="whole-step",
        help="Encode the source step together, or encode on first decoder use and "
        "release each graph after its last consumer. Per-round may trade throughput for memory.",
    )
    return parser


def add_hetero_grid_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register hetero parallelism args for the single-encoder MIMO example."""
    grid = parser.add_argument_group("hetero module grids")
    grid.add_argument(
        "--mimo-packed",
        action="store_true",
        help="Route source-step media into THD decoder packs on disjoint encoder/decoder grids.",
    )

    # Single encoder grid; CP/PP stay fixed at 1.
    grid.add_argument(
        "--mimo-encoder-tp", type=int, default=2, help="Encoder tensor-model-parallel size."
    )
    grid.add_argument("--mimo-encoder-dp", type=int, default=2, help="Encoder data-parallel size.")

    # Language grid placement + factorization.
    grid.add_argument(
        "--mimo-llm-offset",
        type=int,
        default=4,
        help="First global rank of the language grid span.",
    )
    grid.add_argument(
        "--mimo-llm-tp", type=int, default=2, help="Language tensor-model-parallel size."
    )
    grid.add_argument(
        "--mimo-llm-cp",
        type=int,
        default=1,
        help="Language context-parallel size; encoder CP remains 1.",
    )
    grid.add_argument(
        "--mimo-llm-pp", type=int, default=1, help="Language pipeline-model-parallel size."
    )
    grid.add_argument(
        "--mimo-llm-dp",
        type=int,
        default=2,
        help="Language data-parallel size. Global batch is keyed on this.",
    )
    # MoE expert parallelism for the language grid.
    grid.add_argument(
        "--mimo-llm-ep", type=int, default=1, help="Language expert-model-parallel size (MoE)."
    )
    grid.add_argument(
        "--mimo-llm-expt-tp",
        type=int,
        default=None,
        help="Language expert tensor-parallel size; defaults to 1 when unset "
        "(experts default to TP=1; the 20L MoE recipe passes "
        "--mimo-llm-expt-tp 1).",
    )

    grid.add_argument(
        "--mimo-llm-only",
        action="store_true",
        help=(
            "Run only the MIMO language module on the LLM grid. Keeps the MIMO "
            "training/data path but creates no encoder ranks or bridge communicators; "
            "requires --mimo-llm-offset 0 so the language grid covers WORLD_SIZE."
        ),
    )
    grid.add_argument(
        "--mimo-encoder-ddp-overlap",
        action="store_true",
        help=(
            "Apply the global grad-reduce and param-gather overlap settings to encoder DDP. "
            "Requires every encoder DP rank to execute encoder backward on every microbatch."
        ),
    )
    grid.add_argument(
        "--mimo-run-input-projections-on-llm-ranks",
        action="store_true",
        help=(
            "Build modality input projections on LLM ranks. Use --no-load-optim "
            "when loading a checkpoint saved with the other placement."
        ),
    )
    grid.add_argument(
        "--mimo-bridge-skip-shape-exchange",
        action="store_true",
        help=(
            "Derive MIMO fan-out receive shapes from language-batch metadata and skip "
            "bridge shape exchange."
        ),
    )
    return parser


def validate_hetero_grid_args(args: argparse.Namespace, world_size: int) -> tuple[int, int]:
    """Validate the disjoint hetero grid layout; returns ``(encoder_size, llm_size)``."""
    packed = getattr(args, "mimo_packed", False)
    if packed:
        _validate_packed_args(args)
    if args.mimo_llm_cp < 1:
        raise ValueError("--mimo-llm-cp must be positive")
    llm_data_parallel_size = get_language_batch_parallel_size(args)

    if getattr(args, "mimo_encoder_ddp_overlap", False) and not getattr(
        args, "overlap_grad_reduce", False
    ):
        raise ValueError("--mimo-encoder-ddp-overlap requires --overlap-grad-reduce")

    if args.mimo_run_input_projections_on_llm_ranks and args.mimo_llm_only:
        raise ValueError(
            "--mimo-run-input-projections-on-llm-ranks cannot be used with --mimo-llm-only"
        )

    # MoE expert count must divide evenly across the language grid's expert parallelism.
    num_experts = _num_experts(args)
    if num_experts and num_experts % args.mimo_llm_ep != 0:
        raise ValueError(
            f"--num-experts ({num_experts}) must be divisible by "
            f"--mimo-llm-ep ({args.mimo_llm_ep})"
        )

    gtp_weight_remat_size, _ = resolve_hetero_gtp_degrees(args)
    llm_size = (
        args.mimo_llm_tp
        * args.mimo_llm_cp
        * args.mimo_llm_pp
        * args.mimo_llm_dp
        * gtp_weight_remat_size
    )

    if args.mimo_llm_only:
        if getattr(args, "mimo_encoder_ddp_overlap", False):
            raise ValueError("--mimo-encoder-ddp-overlap cannot be used with --mimo-llm-only")
        if args.mimo_llm_offset != 0:
            raise ValueError(
                "--mimo-llm-only requires --mimo-llm-offset 0 so language ranks cover WORLD_SIZE"
            )
        llm_ranks = set(range(args.mimo_llm_offset, args.mimo_llm_offset + llm_size))
        all_ranks = set(range(world_size))
        if llm_ranks != all_ranks:
            raise ValueError(
                "--mimo-llm-only requires the language grid to cover every torchrun rank "
                f"exactly once; covered={sorted(llm_ranks)}, world={sorted(all_ranks)}"
            )
        args.mimo_llm_world_size = llm_size
        return 0, llm_size

    # Fan-out divisibility: the bridge splits every LLM data lane across
    # mimo_encoder_dp encoder lanes; the split must be exact.
    if not packed and (args.micro_batch_size * llm_data_parallel_size) % args.mimo_encoder_dp != 0:
        raise ValueError(
            "--micro-batch-size * language sample lanes must be divisible by "
            "--mimo-encoder-dp "
            f"(got {args.micro_batch_size} * {llm_data_parallel_size} "
            f"% {args.mimo_encoder_dp} != 0)"
        )

    encoder_size = args.mimo_encoder_tp * args.mimo_encoder_dp
    encoder_ranks = set(range(encoder_size))  # encoder span always starts at rank 0
    llm_ranks = set(range(args.mimo_llm_offset, args.mimo_llm_offset + llm_size))
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

    args.mimo_llm_world_size = llm_size
    return encoder_size, llm_size


def _validate_packed_args(args: argparse.Namespace) -> None:
    """Keep the first packed disaggregated path on its tested sequential contract."""
    if (
        args.mimo_encoder_tp != 1
        or args.mimo_llm_tp != 1
        or args.mimo_llm_pp != 1
        or (args.mimo_llm_expt_tp or 1) != 1
        or resolve_hetero_gtp_degrees(args) != (1, 1)
    ):
        raise ValueError(
            "--mimo-packed requires encoder/decoder TP=PP=1 and no GTP rematerialization"
        )
    if (
        not getattr(args, "bf16", False)
        or getattr(args, "fp16", False)
        or getattr(args, "fp8", None)
    ):
        raise ValueError("--mimo-packed requires BF16")
    if (
        not getattr(args, "use_distributed_optimizer", False)
        or getattr(args, "optimizer", "adam") != "adam"
    ):
        raise ValueError("--mimo-packed requires the distributed Adam optimizer")
    if (
        not getattr(args, "calculate_per_token_loss", False)
        or getattr(args, "dataloader_type", None) != "external"
        or not getattr(args, "sequence_packing_scheduler", None)
    ):
        raise ValueError(
            "--mimo-packed requires per-token loss, external data and a packing scheduler"
        )
    unsupported = (
        "mimo_llm_only",
        "mimo_run_input_projections_on_llm_ranks",
        "mimo_encoder_prefetch",
        "mimo_encoder_ddp_overlap",
        "overlap_grad_reduce",
        "overlap_param_gather",
        "overlap_param_gather_with_optimizer_step",
        "overlap_moe_expert_parallel_comm",
        "correct_encoder_grad_for_partial_participation",
        "ddp_average_in_collective",
        "use_layer_wise_distributed_optimizer",
        "freeze_all_layers",
    )
    enabled = [name for name in unsupported if getattr(args, name, False)]
    if enabled:
        raise ValueError(f"--mimo-packed does not support {', '.join(enabled)}")
    if getattr(args, "rerun_mode", "disabled") != "disabled":
        raise ValueError("--mimo-packed requires --rerun-mode disabled")


def build_module_grid_specs(
    args: argparse.Namespace, world_size: int, encoder_module_name: str
) -> List[ModuleGridSpec]:
    """Map grid args to the ModuleGridSpec list create_topology consumes."""
    encoder_size, llm_size = validate_hetero_grid_args(args, world_size)
    gtp_weight_remat_size, expert_gtp_weight_remat_size = resolve_hetero_gtp_degrees(args)
    _, gtp_remat_num_sequence_shards = resolve_tensor_parallel_sequence_shards(
        args.mimo_llm_tp,
        getattr(args, "tensor_parallel_num_sequence_shards", None),
        gtp_weight_remat_size,
        getattr(args, "sequence_parallel", False),
    )

    language_grid_spec = ModuleGridSpec(
        name=MIMO_LANGUAGE_MODULE_KEY,
        num_ranks=llm_size,
        tp=args.mimo_llm_tp,
        cp=args.mimo_llm_cp,
        pp=args.mimo_llm_pp,
        ep=args.mimo_llm_ep,
        gtp_remat=gtp_weight_remat_size,
        gtp_remat_num_sequence_shards=gtp_remat_num_sequence_shards,
        rank_offset=args.mimo_llm_offset,
        expt_tp=args.mimo_llm_expt_tp or 1,
        expt_gtp_remat=expert_gtp_weight_remat_size,
    )

    if args.mimo_llm_only:
        return [language_grid_spec]

    encoder_grid_spec = ModuleGridSpec(
        name=encoder_module_name,
        num_ranks=encoder_size,
        tp=args.mimo_encoder_tp,
        cp=1,
        pp=1,
        ep=1,
        rank_offset=0,
        expt_tp=1,
    )
    return [encoder_grid_spec, language_grid_spec]


def resolve_hetero_gtp_degrees(args: argparse.Namespace) -> tuple[int, int]:
    """Return the language dense and expert GTP degrees."""
    _, gtp_weight_remat_size = resolve_tensor_parallel_weight_shards(
        args.mimo_llm_tp,
        getattr(args, "tensor_parallel_num_weight_shards", None),
        getattr(args, "gtp_weight_remat_size", 1),
    )
    _, expert_gtp_weight_remat_size = resolve_tensor_parallel_weight_shards(
        args.mimo_llm_expt_tp or 1,
        getattr(args, "expert_tensor_parallel_num_weight_shards", None),
        getattr(args, "expert_gtp_weight_remat_size", 1),
        shards_field="expert_tensor_parallel_num_weight_shards",
        tp_field="expert_tensor_parallel_size",
    )
    return gtp_weight_remat_size, expert_gtp_weight_remat_size


def _num_experts(args: argparse.Namespace) -> int:
    """Resolve MoE expert count from the stock --num-experts arg."""
    value = getattr(args, "num_experts", None)
    return int(value) if value else 0

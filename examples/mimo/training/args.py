# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hetero grid/topology CLI args + validation for the stock-args MIMO examples.

This is PR-E2 of the NMFW-516 hetero-MIMO upstreaming effort. It contributes the
heterogeneous parallelism/topology argument group and its validation, designed
to compose with Megatron's *stock* argument system as an
``extra_args_provider`` and a post-``validate_args`` validation hook.

Responsibilities (this PR only):
  * :func:`add_hetero_grid_args` registers the per-module parallelism knobs
    (``--encoder-*`` and ``--llm-*``) plus ``--llm-offset`` / ``--llm-only``.
    These are namespaced so they never collide with stock parallelism flags
    (``--tensor-model-parallel-size`` and friends are owned by the LLM grid but
    expressed via the namespaced ``--llm-tp`` so the encoder grid can differ).
  * :func:`validate_hetero_grid_args` ports the prototype's hetero invariants:
    disjoint + covering rank spans, fan-out divisibility, MoE EP divisibility,
    and the sample-vs-iter scheduler resolution. Call it AFTER stock
    ``validate_args`` so tokenizer/preset-derived sizes are populated.
  * :func:`build_module_grid_specs` maps parsed grid args onto the two
    :class:`~examples.mimo.training.topology.ModuleGridSpec` s that
    :func:`~examples.mimo.training.topology.create_topology` consumes.

Arg-ownership note: ``--llm-ep`` and ``--llm-expt-tp`` are *parallelism* knobs
and live here, not in the model provider's ``add_model_provider_args``. The
provider still reads them via ``getattr(args, "llm_ep", ...)`` /
``getattr(args, "llm_expt_tp", ...)`` fallbacks, so the two PRs compose cleanly
once stacked.
"""

from __future__ import annotations

import argparse
import math
from typing import List

from examples.mimo.training.topology import ModuleGridSpec
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

# Default name for the (single) vision encoder module grid. The E1 provider sets
# ``args.vision_encoder_key = "radio_encoder"``; we mirror that default here so
# the encoder ModuleGridSpec carries the same module name the provider/runtime
# look up. Resolved at spec-build time via ``getattr(args, "vision_encoder_key")``.
DEFAULT_ENCODER_MODULE_NAME = "radio_encoder"


def add_hetero_grid_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the hetero parallelism/topology arg group.

    Stock-hook compatible: returns the parser so it can be passed straight to
    ``pretrain(extra_args_provider=...)`` or composed with the model-provider
    args provider. Declares only namespaced ``--encoder-*`` / ``--llm-*`` knobs;
    never re-declares a stock flag (``--tensor-model-parallel-size``,
    ``--micro-batch-size``, ``--global-batch-size``, ``--num-experts``, ...).
    """
    grid = parser.add_argument_group("hetero module grids")

    # Encoder grid placement + factorization.
    grid.add_argument("--encoder-offset", type=int, default=0,
                      help="First global rank of the vision-encoder grid span.")
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
    # MoE expert parallelism for the language grid. Relocated here from the E1
    # model provider's add_model_provider_args (these are topology knobs); the
    # provider keeps a getattr(args, "llm_ep"/"llm_expt_tp", ...) fallback read.
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
    """Validate the disjoint-grid hetero layout. Returns ``(encoder_size, llm_size)``.

    Call AFTER stock ``validate_args`` (so ``micro_batch_size`` / ``num_experts``
    are populated) and after the model-provider preset (so ``num_experts`` reflects
    the active provider). Ports the prototype invariants:

      * CP must be 1 on both grids (Phase-2 limitation).
      * MoE: ``num_experts % llm_ep == 0`` (resolves PR-E1 open-Q3 -- the
        EP-divisibility check belongs here, not in the provider).
      * Global batch is keyed on ``llm_dp``: ``mbs * num_microbatches * llm_dp``.
      * Sample-vs-iter scheduler resolution when ``--train-samples`` is present.
      * ``--llm-only``: language grid must cover ``[0, world_size)`` exactly.
      * Fan-out divisibility: ``mbs * llm_dp % encoder_dp == 0``.
      * Encoder + LLM rank spans are DISJOINT and COVERING over ``world_size``.
    """
    if args.encoder_cp != 1 or args.llm_cp != 1:
        raise ValueError("hetero MIMO training currently supports CP=1 only")

    # MoE EP-divisibility (PR-E1 open-Q3 resolution). Stock arg is --num-experts,
    # surfaced on args as num_experts; the prototype also used num_moe_experts.
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
    encoder_ranks = set(range(args.encoder_offset, args.encoder_offset + encoder_size))
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

    Returns ``[encoder_spec, language_spec]`` (or just ``[language_spec]`` when
    ``--llm-only``). ``num_ranks`` is the ground truth ModuleGridSpec field;
    ``dp`` / ``expt_dp`` are derived in ``ModuleGridSpec.__post_init__`` from the
    tp/cp/pp/ep/expt_tp factorization, so we pass ``num_ranks = tp*cp*pp*dp`` and
    let the dataclass re-derive dp (matching the supplied value) and expt_dp.

    Must be called after :func:`validate_hetero_grid_args` so the spans are known
    to tile ``[0, world_size)``.
    """
    encoder_size, llm_size = validate_hetero_grid_args(args, world_size)

    language_spec = ModuleGridSpec(
        name=MIMO_LANGUAGE_MODULE_KEY,
        num_ranks=llm_size,
        tp=args.llm_tp,
        cp=args.llm_cp,
        pp=args.llm_pp,
        ep=args.llm_ep,
        rank_offset=args.llm_offset,
        expt_tp=_resolve_expt_tp(args.llm_expt_tp, args.llm_tp),
    )

    if args.llm_only:
        return [language_spec]

    encoder_name = getattr(args, "vision_encoder_key", None) or DEFAULT_ENCODER_MODULE_NAME
    encoder_spec = ModuleGridSpec(
        name=encoder_name,
        num_ranks=encoder_size,
        tp=args.encoder_tp,
        cp=args.encoder_cp,
        pp=args.encoder_pp,
        ep=1,
        rank_offset=args.encoder_offset,
        expt_tp=1,
    )
    return [encoder_spec, language_spec]


def _resolve_expt_tp(expt_tp, tp: int) -> int:
    """Expert TP defaults to 1 when unset.

    Matches ModuleGridSpec's convention (experts default to TP=1, set explicitly
    for MoE -- intentionally not Megatron's etp=tp). The 20L MoE layout (ep=4 over
    4 ranks) requires expt_tp=1: expt_tp*ep*pp must divide num_ranks.
    """
    return 1 if expt_tp is None else expt_tp


def _num_experts(args: argparse.Namespace) -> int:
    """Resolve MoE expert count from stock (--num-experts) or prototype args."""
    for attr in ("num_experts", "num_moe_experts"):
        value = getattr(args, attr, None)
        if value:
            return int(value)
    return 0


def _num_microbatches(args: argparse.Namespace) -> int:
    """Resolve the per-step microbatch count from whichever arg the loop exposes."""
    value = getattr(args, "num_microbatches", None)
    if value:
        return int(value)
    return 1

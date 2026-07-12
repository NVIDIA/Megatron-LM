# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Span group definitions for Megatron-LM telemetry.

Tries to import the real ``SpanGroup`` from ``nemo.lens``.  When nemo-lens
is not installed, a minimal stub is provided so that ``MegatronSpanGroup``
constants are always available.
"""

from typing import ClassVar, Final

try:
    from nemo.lens.groups import SpanGroup
except ImportError:

    class SpanGroup:
        """Minimal stub when nemo-lens is not installed."""

        JOB = "job"
        CHECKPOINT = "checkpoint"
        EVALUATE = "evaluate"
        MODEL_INIT = "model_init"
        LOAD_CHECKPOINT = "load_checkpoint"
        STEP = "step"
        FORWARD_BACKWARD = "forward_backward"
        OPTIMIZER = "optimizer"

        ALL_GROUPS: Final[frozenset] = frozenset(
            [
                JOB,
                CHECKPOINT,
                EVALUATE,
                MODEL_INIT,
                LOAD_CHECKPOINT,
                STEP,
                FORWARD_BACKWARD,
                OPTIMIZER,
            ]
        )

        GOODPUT_GROUPS: Final[frozenset] = frozenset(
            [JOB, CHECKPOINT, MODEL_INIT, LOAD_CHECKPOINT, STEP]
        )

        _PRESETS: ClassVar[dict] = {
            "default": frozenset([JOB, CHECKPOINT, EVALUATE]),
            "goodput": GOODPUT_GROUPS,
            "per_step": frozenset(
                [
                    JOB,
                    CHECKPOINT,
                    EVALUATE,
                    MODEL_INIT,
                    LOAD_CHECKPOINT,
                    STEP,
                    FORWARD_BACKWARD,
                    OPTIMIZER,
                ]
            ),
            "profiling": ALL_GROUPS,
            "all": ALL_GROUPS,
        }

        @classmethod
        def categories(cls) -> dict:
            return {
                g: ("goodput" if g in cls.GOODPUT_GROUPS else "profiling")
                for g in cls.ALL_GROUPS
            }

        @classmethod
        def resolve(cls, spec: str) -> frozenset:
            raise RuntimeError(
                "SpanGroup.resolve() requires nemo-lens to be installed. "
                "Install it with: pip install nemo-lens"
            )


class MegatronSpanGroup(SpanGroup):
    """Span groups for Megatron-LM instrumentation.

    Extends the shared groups with Megatron-specific fine-grained groups.
    """

    # ------------------------------------------------------------------ #
    # Fine-grained (included in "per_step" or "all")
    # ------------------------------------------------------------------ #

    MICROBATCH = "microbatch"
    """Per-microbatch forward/backward spans."""

    LAYER = "layer"
    """Per-transformer-layer forward (attention + MLP breakdown)."""

    COMMUNICATION = "communication"
    """P2P send/recv and gradient AllReduce/ReduceScatter."""

    ACTIVATION_OFFLOAD = "activation_offload"
    """GPU<->CPU activation offload/reload spans."""

    DATA_LOADING = "data_loading"
    """Data loading and batch preparation."""

    FIRST_ITERATION = "first_iteration"
    """The first training iteration actually executed in this process (post
    checkpoint-resume, post iteration-skip) — not necessarily iteration 1, and
    distinct from the per-step STEP span since it captures one-off warmup
    costs (compilation, CUDA graph capture, prefetch) absent from steady-state
    iterations."""

    TRACE_REGION = "trace_region"
    """Shadows every perfetto-native ``trace_region(...)`` marker with a lens
    span (see megatron.core.perfetto_trace) — ~85 checkpoint/dataset/load
    sub-phase markers, covered without per-site instrumentation. Verbose and
    fine-grained: deliberately NOT in the ``per_step`` preset (only ``all``);
    opt in explicitly, e.g. ``--otel-span-groups per_step,trace_region``."""

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    INFERENCE = "inference"
    """Inference server request spans."""

    # ------------------------------------------------------------------ #
    # All groups and presets
    # ------------------------------------------------------------------ #

    ALL_GROUPS: Final[frozenset] = SpanGroup.ALL_GROUPS | frozenset(
        [
            MICROBATCH,
            LAYER,
            COMMUNICATION,
            ACTIVATION_OFFLOAD,
            DATA_LOADING,
            FIRST_ITERATION,
            TRACE_REGION,
            INFERENCE,
        ]
    )

    # Semantic goodput/resiliency boundaries: the base goodput groups plus the
    # Megatron phases the goodput bucketer keys off (dataloader setup, the
    # one-off first iteration). Everything else (forward/backward, optimizer,
    # microbatch, layer, communication, activation offload, trace_region,
    # inference) is profiling detail. Drives both the "goodput" preset and the
    # per-span lens.span_category attribute.
    #
    # Listed self-contained (NOT `SpanGroup.GOODPUT_GROUPS | ...`) so an older
    # installed nemo-lens without GOODPUT_GROUPS can't AttributeError at class
    # definition and take the whole training run down with it -- telemetry must
    # never break training.
    # Goodput = resiliency overhead only: the cost of RESTART (job/container,
    # python+megatron init, model init, checkpoint load, dataloader, first-
    # iteration warmup) and the cost of DEFENSE (checkpoint save, sniff test, FT
    # heartbeat, weight-hash check), plus STEP for the productive-time baseline.
    # EVALUATE is deliberately NOT here -- evaluation is intended work, not a
    # resiliency cost, so it must not count against goodput. Likewise energy/
    # straggler monitors and steady-state logging are profiling, not goodput.
    GOODPUT_GROUPS: Final[frozenset] = frozenset(
        [
            SpanGroup.JOB,
            SpanGroup.CHECKPOINT,
            SpanGroup.MODEL_INIT,
            SpanGroup.LOAD_CHECKPOINT,
            SpanGroup.STEP,
            DATA_LOADING,
            FIRST_ITERATION,
        ]
    )

    @classmethod
    def categories(cls) -> dict:
        """{group: 'goodput'|'profiling'} for every group in ALL_GROUPS.

        Defined here (not only on the base) so it works even against an older
        nemo-lens whose SpanGroup predates categories().
        """
        return {
            g: ("goodput" if g in cls.GOODPUT_GROUPS else "profiling")
            for g in cls.ALL_GROUPS
        }

    _PRESETS: ClassVar[dict] = {
        "default": frozenset(
            [
                SpanGroup.JOB,
                SpanGroup.CHECKPOINT,
                SpanGroup.EVALUATE,
                FIRST_ITERATION,
                INFERENCE,
            ]
        ),
        "goodput": GOODPUT_GROUPS,
        "per_step": frozenset(
            [
                SpanGroup.JOB,
                SpanGroup.CHECKPOINT,
                SpanGroup.EVALUATE,
                SpanGroup.MODEL_INIT,
                SpanGroup.LOAD_CHECKPOINT,
                SpanGroup.STEP,
                SpanGroup.FORWARD_BACKWARD,
                SpanGroup.OPTIMIZER,
                COMMUNICATION,
                DATA_LOADING,
                FIRST_ITERATION,
                INFERENCE,
            ]
        ),
        "profiling": ALL_GROUPS,
        "all": ALL_GROUPS,
    }

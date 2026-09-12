# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

        _PRESETS: ClassVar[dict] = {
            "default": frozenset([JOB, CHECKPOINT, EVALUATE]),
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
        def resolve(cls, spec: str) -> frozenset:
            """Always raises; resolving a span-group spec needs nemo-lens."""
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

    _PRESETS: ClassVar[dict] = {
        "default": frozenset(
            [SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE, FIRST_ITERATION, INFERENCE]
        ),
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

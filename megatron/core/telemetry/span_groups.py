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
            "all": ALL_GROUPS,
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
            INFERENCE,
        ]
    )

    _PRESETS: ClassVar[dict] = {
        "default": frozenset(
            [
                SpanGroup.JOB,
                SpanGroup.CHECKPOINT,
                SpanGroup.EVALUATE,
                INFERENCE,
            ]
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
                INFERENCE,
            ]
        ),
        "all": ALL_GROUPS,
    }

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-LM telemetry helpers.

When ``nemo-lens`` is installed, the real implementations are used.
Otherwise, no-op fallbacks from ``_fallbacks`` keep everything working.

Submodules:
    _fallbacks        — No-op stubs matching the nemo.lens API.
    span_groups       — SpanGroup / MegatronSpanGroup constants and presets.
    training_metrics  — OTel metric recording for the training loop.
"""

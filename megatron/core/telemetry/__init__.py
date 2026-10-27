# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-LM telemetry helpers.

When ``nemo-lens`` is installed, the real implementations are used.
Otherwise, no-op fallbacks from ``fallbacks`` keep everything working.

Submodules:
    fallbacks         — No-op stubs matching the nemo.lens API.
    span_groups       — SpanGroup / MegatronSpanGroup constants and presets.
    training_metrics  — OTel metric recording for the training loop.
"""

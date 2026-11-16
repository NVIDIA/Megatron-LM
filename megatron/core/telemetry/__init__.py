# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-LM telemetry helpers.

Submodules:
    telemetry         - Megatron's Lens shim and span-group registration.
    training_metrics  - OTel metric recording for the training loop.
    fallbacks         - Legacy no-op stubs for MCore call sites not yet on ``telemetry``.
"""

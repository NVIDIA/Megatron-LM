# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Operation implementations used by Megatron Core's backend spec providers.

This package does not select backends or import optional kernel packages.
Model construction uses ``megatron.core.models.backends.BackendSpecProvider``;
the returned implementations own any shape- or phase-dependent dispatch.
"""

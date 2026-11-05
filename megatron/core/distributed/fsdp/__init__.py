# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron FSDP integration.

Import MCore wrappers explicitly from ``mcore_fsdp_adapter`` so low-level FSDP imports
do not initialize the adapters.
"""

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MoE operation implementations and backend adapters.

The fake DeepEP adapter models balanced assignment allocations on one device;
its handles are owned by the dispatch/combine autograd contexts.
"""

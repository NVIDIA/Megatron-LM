# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gated Delta Net (GDN) family of operations.

``common`` holds the shared base class, submodule spec, context-parallel all-to-alls
and ``GatedDeltaRuleInterface``, the keyword call surface both recurrences share.
``gdn`` (scalar gates) and ``gdn2`` (channel-wise gates) hold the two variants and
their Torch reference recurrences. Each module owns its parameters, context-parallel
execution and checkpoint mappings; global inference-cache allocation stays outside.

Inputs use BTHD layout and optional initial/final states are BHKV. FLA supplies the
packed training kernels and GDN's recurrent inference; the Torch references back
deterministic mode and reject packed ``cu_seqlens``.
"""

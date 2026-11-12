# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Loaders and ABI of the external indexer top-k plugins (internal).

The only code in Megatron Lite that imports the external LiteTopK and exact-tie top-k modules
(:mod:`.loader`), states the plugin ABI (:mod:`.abi`), manages the environment the plugins read
(:mod:`.env`) and owns the key caches they gather from (:mod:`.cache`).
"""

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared Qwen3MoE model helpers."""

from __future__ import annotations


def is_expert_param(name: str) -> bool:
    return "experts" in name and "router" not in name


__all__ = ["is_expert_param"]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Phase A tests for TE attention runtime-context parsing."""

from pathlib import Path
from types import SimpleNamespace

from megatron.training.theoretical_flops_usage import (
    build_runtime_context,
    parse_te_attention_runtime_context,
    set_te_attention_debug_env_if_needed,
)


def test_te_attention_runtime_context_parse():
    log_path = Path("tests/unit_tests/fixtures/theoretical_flops/te_dot_product_attention_debug.log")
    parsed = parse_te_attention_runtime_context(log_path.read_text(encoding="utf-8"))

    assert parsed["te_available_backends"] == (
        "{FlashAttention=True, FusedAttention=True (sub-backend 1), "
        "UnfusedDotProductAttention=True}"
    )
    assert parsed["te_selected_backend"] == "FusedAttention"
    assert parsed["te_fused_sub_backend"] == 1


def test_set_te_attention_debug_env_if_needed(monkeypatch):
    monkeypatch.delenv("NVTE_DEBUG", raising=False)
    monkeypatch.delenv("NVTE_DEBUG_LEVEL", raising=False)
    args = SimpleNamespace(
        report_theoretical_flops=True,
        capture_te_attention_backend=True,
        transformer_impl="transformer_engine",
        attention_backend="auto",
    )

    set_te_attention_debug_env_if_needed(args)
    context = build_runtime_context(args)

    assert context.nvte_debug == "1"
    assert context.nvte_debug_level == "2"

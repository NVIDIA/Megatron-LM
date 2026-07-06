# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Phase A tests for dense theoretical FLOPs reporting."""

import argparse
import json
from types import SimpleNamespace

from megatron.training.argument_utils import ArgumentGroupFactory
from megatron.training.config import TrainingConfig
from megatron.training.theoretical_flops_usage import (
    build_theoretical_flops_report,
    update_theoretical_flops_json_runtime_context,
    write_theoretical_flops_json,
)
from megatron.training.training import num_floating_point_operations


def _make_dense_gqa_args(**overrides):
    args = SimpleNamespace()
    args.num_layers = 4
    args.hidden_size = 2048
    args.num_attention_heads = 16
    args.seq_length = 4096
    args.padded_vocab_size = 32000
    args.swiglu = True
    args.ffn_hidden_size = 6144
    args.kv_channels = 128
    args.group_query_attention = True
    args.num_query_groups = 8
    args.attention_output_gate = False
    args.multi_latent_attention = False
    args.num_experts = None
    args.moe_layer_freq = 1
    args.moe_router_topk = 0
    args.moe_ffn_hidden_size = None
    args.moe_latent_size = None
    args.moe_shared_expert_intermediate_size = None
    args.mtp_num_layers = None
    args.experimental_attention_variant = None
    args.linear_attention_freq = None
    args.linear_key_head_dim = None
    args.linear_value_head_dim = None
    args.linear_num_key_heads = None
    args.linear_num_value_heads = None
    args.linear_conv_kernel_dim = None
    args.q_lora_rank = None
    args.qk_head_dim = None
    args.qk_pos_emb_head_dim = None
    args.kv_lora_rank = None
    args.v_head_dim = None
    args.hybrid_layer_pattern = None
    args.micro_batch_size = 2
    args.global_batch_size = 16
    args.data_parallel_size = 8
    args.world_size = 8
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.context_parallel_size = 1
    args.sequence_parallel = False
    args.bf16 = True
    args.fp16 = False
    args.fp8_format = None
    args.attention_backend = "auto"
    args.transformer_impl = "transformer_engine"
    args.theoretical_flops_output_dir = "./flops_analysis"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _entries_by_operator(report):
    return {entry.operator: entry for entry in report.entries}


def test_theoretical_flops_dense_gqa_total():
    args = _make_dense_gqa_args()

    report = build_theoretical_flops_report(args, num_microbatches=1)

    assert report.computed_total_flops == report.reference_total_flops
    assert report.computed_total_flops == num_floating_point_operations(args, args.global_batch_size)
    assert report.computed_total_flops == sum(entry.global_flops for entry in report.entries)


def test_theoretical_flops_shapes_tp2_sp():
    args = _make_dense_gqa_args(
        hidden_size=8,
        ffn_hidden_size=32,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        seq_length=16,
        padded_vocab_size=64,
        tensor_model_parallel_size=2,
        data_parallel_size=4,
        world_size=8,
        sequence_parallel=True,
    )

    entries = _entries_by_operator(build_theoretical_flops_report(args, num_microbatches=1))

    assert entries["qkv_projection"].shape == "(m,n,k)=(32, 8, 8)"
    assert entries["qkv_projection"].sequence_sharded is False
    assert entries["output_projection"].shape == "(m,n,k)=(16, 8, 4)"
    assert entries["output_projection"].sequence_sharded is True
    assert entries["fc1"].shape == "(m,n,k)=(32, 32, 8)"
    assert entries["fc1"].sequence_sharded is False
    assert entries["fc2"].shape == "(m,n,k)=(16, 8, 16)"
    assert entries["fc2"].sequence_sharded is True


def test_theoretical_flops_shapes_cp2():
    args = _make_dense_gqa_args(
        hidden_size=8,
        ffn_hidden_size=32,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        seq_length=16,
        padded_vocab_size=64,
        context_parallel_size=2,
        data_parallel_size=4,
        world_size=8,
    )

    entries = _entries_by_operator(build_theoretical_flops_report(args, num_microbatches=1))

    assert entries["qkv_projection"].shape == "(m,n,k)=(16, 16, 8)"
    assert entries["(gqa)core_attn(sbhd)"].shape == (
        "(mbs=2, seq=8, h=4, qk_d=2, v_d=2) (causal)"
    )


def test_theoretical_flops_json_roundtrip(tmp_path):
    args = _make_dense_gqa_args()
    report = build_theoretical_flops_report(args, num_microbatches=1)

    json_path = write_theoretical_flops_json(report, tmp_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert loaded["reference_total_flops"] == report.reference_total_flops
    assert isinstance(loaded["entries"][0]["global_flops"], int)
    assert loaded["runtime_context"]["te_selected_backend"] is None


def test_theoretical_flops_json_runtime_context_update(tmp_path):
    args = _make_dense_gqa_args(theoretical_flops_output_dir=str(tmp_path))
    report = build_theoretical_flops_report(args, num_microbatches=1)
    write_theoretical_flops_json(report, tmp_path)

    update_theoretical_flops_json_runtime_context(
        args,
        {
            "te_available_backends": "{FlashAttention=True}",
            "te_selected_backend": "FlashAttention",
            "capture_step": 1,
        },
    )

    loaded = json.loads((tmp_path / "theoretical_flops.json").read_text(encoding="utf-8"))
    assert loaded["runtime_context"]["te_selected_backend"] == "FlashAttention"
    assert loaded["runtime_context"]["capture_step"] == 1


def test_theoretical_flops_golden_dense_gqa_tp1():
    args = _make_dense_gqa_args()
    report = build_theoretical_flops_report(args, num_microbatches=1)
    fixture_path = (
        "tests/unit_tests/fixtures/theoretical_flops/dense_gqa_tp1_expected.json"
    )
    with open(fixture_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    actual_entries = [
        {
            "operator": entry.operator,
            "shape": entry.shape,
            "global_flops": entry.global_flops,
            "sequence_sharded": entry.sequence_sharded,
        }
        for entry in report.entries
    ]
    assert report.reference_total_flops == expected["reference_total_flops"]
    assert actual_entries == expected["entries"]
    assert report.runtime_context.te_selected_backend == expected["runtime_context"]["te_selected_backend"]


def test_theoretical_flops_regression_num_fp_ops():
    cases = [
        _make_dense_gqa_args(group_query_attention=False, num_query_groups=16),
        _make_dense_gqa_args(swiglu=False),
        _make_dense_gqa_args(attention_output_gate=True),
    ]

    for args in cases:
        report = build_theoretical_flops_report(args, num_microbatches=1)
        assert report.computed_total_flops == num_floating_point_operations(args, args.global_batch_size)


def test_theoretical_flops_argparse_wiring_offline_no_cuda():
    parser = argparse.ArgumentParser()
    ArgumentGroupFactory(TrainingConfig).build_group(parser, "training")

    args = parser.parse_args(
        [
            "--report-theoretical-flops",
            "--theoretical-flops-output-dir",
            "/tmp/flops",
            "--theoretical-flops-verbose",
            "--disable-capture-te-attention-backend",
            "--disable-reconcile-trace-after-profile",
        ]
    )

    assert args.report_theoretical_flops is True
    assert args.theoretical_flops_output_dir == "/tmp/flops"
    assert args.theoretical_flops_verbose is True
    assert args.capture_te_attention_backend is False
    assert args.reconcile_trace_after_profile is False

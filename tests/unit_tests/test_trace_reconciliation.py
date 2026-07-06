# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Phase A tests for PyTorch profiler trace reconciliation."""

import gzip
import json
from types import SimpleNamespace

from megatron.training.theoretical_flops_usage import (
    build_theoretical_flops_report,
    write_theoretical_flops_json,
)
from megatron.training.trace_reconciliation import (
    get_torch_profile_dir,
    parse_chrome_trace_gemm_events,
    reconcile_trace_vs_theory,
)


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
    args.profile_step_start = 2
    args.profile_step_end = 4
    args.report_theoretical_flops = True
    args.theoretical_flops_output_dir = "./flops_analysis"
    args.tensorboard_dir = "./tensorboard"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_trace_handler_export_path_selection():
    legacy_args = _make_dense_gqa_args(report_theoretical_flops=False)
    flops_args = _make_dense_gqa_args(
        report_theoretical_flops=True,
        theoretical_flops_output_dir="./flops_analysis",
    )

    assert str(get_torch_profile_dir(legacy_args)) == "tensorboard/../torch_profile"
    assert str(get_torch_profile_dir(flops_args)) == "flops_analysis/torch_profile"


def test_trace_reconciliation_mock_profile(tmp_path):
    args = _make_dense_gqa_args(theoretical_flops_output_dir=str(tmp_path))
    theory_path = write_theoretical_flops_json(build_theoretical_flops_report(args, num_microbatches=1), tmp_path)
    trace_path = tmp_path / "torch_profile" / "rank-0.json.gz"
    _write_mock_trace(
        trace_path,
        [
            ("aten::mm", (8192, 4096, 2048)),
            ("aten::mm", (8192, 2048, 2048)),
            ("aten::mm", (8192, 12288, 2048)),
            ("aten::addmm", (8192, 2048, 6144)),
            ("transformer_engine::gemm", (8192, 32000, 2048)),
        ],
    )

    events = parse_chrome_trace_gemm_events(trace_path)
    result = reconcile_trace_vs_theory(args, rank=0, trace_path=trace_path, theoretical_path=theory_path)

    assert len(events) == 5
    assert result.matched == 5
    assert result.unmatched_analytical == []
    assert result.unmatched_trace == []
    assert (tmp_path / "reconciliation_rank0.json").exists()


def test_trace_reconciliation_te_fusion_unmatched(tmp_path):
    args = _make_dense_gqa_args(theoretical_flops_output_dir=str(tmp_path))
    theory_path = write_theoretical_flops_json(build_theoretical_flops_report(args, num_microbatches=1), tmp_path)
    trace_path = tmp_path / "torch_profile" / "rank-0.json.gz"
    _write_mock_trace(
        trace_path,
        [
            ("aten::mm", (8192, 4096, 2048)),
            ("aten::mm", (8192, 2048, 2048)),
            ("transformer_engine::gemm_swiglu_fused", (8192, 6144, 2048)),
            ("aten::addmm", (8192, 2048, 6144)),
            ("transformer_engine::gemm", (8192, 32000, 2048)),
        ],
    )

    result = reconcile_trace_vs_theory(args, rank=0, trace_path=trace_path, theoretical_path=theory_path)

    fc1_unmatched = [
        entry for entry in result.unmatched_analytical if entry["submodule"] == "mlp" and entry["operator"] == "fc1"
    ]
    assert fc1_unmatched
    assert fc1_unmatched[0]["hint"] == "may be fused via TE gemm+activation kernel"
    assert result.warnings


def _write_mock_trace(path, gemm_shapes):
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_events = []
    for index, (name, shape) in enumerate(gemm_shapes):
        m, n, k = shape
        trace_events.append(
            {
                "name": name,
                "ph": "X",
                "ts": index,
                "dur": 1,
                "args": {
                    "Input Dims": [[m, k], [k, n]],
                },
            }
        )
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": trace_events}, f)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import json
from types import SimpleNamespace

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.transformer.experimental_attention_variant.dsa_diagnostics import (
    DSADiagnosticsCollector,
    compute_dsa_attention_diagnostics,
    expand_integer_ranges,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import add_megatron_arguments
from tests.unit_tests.test_utilities import Utils
from megatron.core.transformer.experimental_attention_variant.dsa_diagnostics import (
    _summarize_distribution_width,
)


def _diagnostic_config(tmp_path, **overrides):
    values = {
        "dsa_diagnostics": True,
        "dsa_diagnostics_layers": [7],
        "dsa_diagnostics_topk_values": [1, 2],
        "dsa_diagnostics_prefill_tail_offsets": [0, 2],
        "dsa_diagnostics_decode_offsets": [0, 2],
        "dsa_diagnostics_output_dir": str(tmp_path),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["0...4"], [0, 1, 2, 3, 4]),
        (["4...0"], [4, 3, 2, 1, 0]),
        (["0...8:2"], [0, 2, 4, 6, 8]),
        (["0...2", "2", "4,6...8"], [0, 1, 2, 4, 6, 7, 8]),
    ],
)
def test_expand_integer_ranges(tokens, expected):
    assert expand_integer_ranges(tokens) == expected


@pytest.mark.parametrize("token", ["bad", "0..2", "0...2:0"])
def test_expand_integer_ranges_rejects_invalid_values(token):
    with pytest.raises(ValueError):
        expand_integer_ranges([token])


def test_megatron_parser_expands_dsa_diagnostic_ranges():
    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    args, unknown = parser.parse_known_args(
        [
            "--dsa-diagnostics-layers",
            "1...7:3",
            "--dsa-diagnostics-topk-values",
            "512...2048:512",
            "--dsa-diagnostics-prefill-tail-offsets",
            "0...32",
            "--dsa-diagnostics-decode-offsets",
            "0...8",
        ]
    )
    assert not unknown
    assert args.dsa_diagnostics_layers == [1, 4, 7]
    assert args.dsa_diagnostics_topk_values == [512, 1024, 1536, 2048]
    assert args.dsa_diagnostics_prefill_tail_offsets == list(range(33))
    assert args.dsa_diagnostics_decode_offsets == list(range(9))


def test_dsa_diagnostics_selects_prefill_tail_and_decode_offsets_once(tmp_path):
    collector = DSADiagnosticsCollector(_diagnostic_config(tmp_path))
    collector.register_request(request_id=11, prompt_length=10)

    selected = collector.selected_queries(
        request_id=11, layer_number=7, query_start_position=6, query_length=7
    )
    assert [(item["position"], item["phase"], item["offset"]) for item in selected] == [
        (7, "prefill_tail", 2),
        (9, "prefill_tail", 0),
        (10, "decode", 0),
        (12, "decode", 2),
    ]
    assert not collector.selected_queries(11, 7, 6, 7)
    assert not collector.selected_queries(11, 8, 6, 7)


def test_dsa_diagnostics_preserves_prompt_origin_and_cleans_request_state(tmp_path):
    collector = DSADiagnosticsCollector(_diagnostic_config(tmp_path))
    collector.register_request(request_id=11, prompt_length=10)

    # Suspend/recompute can re-add a checkpointed request whose prompt now includes generated
    # tokens. Decode offsets must remain anchored to the original prompt.
    collector.register_request(request_id=11, prompt_length=12)
    selected = collector.selected_queries(
        request_id=11, layer_number=7, query_start_position=10, query_length=1
    )
    assert [(item["phase"], item["offset"]) for item in selected] == [("decode", 0)]

    # Once a request is gone, reusing its ID must not inherit deduplication state.
    collector.unregister_request(11)
    collector.register_request(request_id=11, prompt_length=10)
    selected = collector.selected_queries(
        request_id=11, layer_number=7, query_start_position=10, query_length=1
    )
    assert [(item["phase"], item["offset"]) for item in selected] == [("decode", 0)]


def test_dsa_diagnostics_writes_rank_local_jsonl(tmp_path):
    collector = DSADiagnosticsCollector(_diagnostic_config(tmp_path))
    collector.record({"request_id": 3, "layer": 7, "supports": {}})
    collector.flush()

    paths = list(tmp_path.glob("dsa_diag.*.jsonl"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["request_id"] == 3
    assert record["layer"] == 7
    assert record["dp_rank"] == 0
    assert record["pp_rank"] == 0


def test_compute_dsa_attention_diagnostics_matches_one_head_oracles():
    query = torch.tensor([[[[2.0, 0.0]]]])
    key = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
            [[[-1.0, 0.0]]],
        ]
    )
    value = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
            [[[2.0, 2.0]]],
        ]
    )
    metrics = compute_dsa_attention_diagnostics(
        query=query,
        key=key,
        value=value,
        indexer_support=torch.tensor([1, 0]),
        model_support=torch.tensor([1]),
        softmax_scale=1.0,
        topk_values=[1, 2],
        query_position=1,
    )

    assert metrics["valid_key_count"] == 2
    support_k1 = metrics["supports"]["1"]
    assert support_k1["sum_head_oracle"]["captured_mass_mean"] > support_k1["indexer"][
        "captured_mass_mean"
    ]
    assert support_k1["sum_head_oracle"]["captured_mass_mean"] == pytest.approx(
        support_k1["max_head_oracle"]["captured_mass_mean"]
    )
    assert support_k1["sum_head_oracle"]["captured_mass_mean"] == pytest.approx(
        support_k1["per_head_oracle"]["captured_mass_mean"]
    )
    assert metrics["supports"]["2"]["indexer"]["captured_mass_mean"] == pytest.approx(1.0)
    assert metrics["distribution_width"]["k50_per_head"] == [1]
    assert metrics["distribution_width"]["max_measured_topk"] == 2
    assert metrics["aggregate_distribution_width"]["sum_head_teacher"]["k50"] == 1
    assert metrics["aggregate_distribution_width"]["sum_head_teacher"][
        "max_measured_topk"
    ] == 2


def test_distribution_width_summary_reports_unresolved_measurements():
    rows = [
        {
            "layer": 7,
            "phase": "decode",
            "max_measured_topk": 8192,
            "k99": -1,
        },
        {
            "layer": 7,
            "phase": "decode",
            "max_measured_topk": 8192,
            "k99": 4096,
        },
    ]
    summary = _summarize_distribution_width(rows)
    assert summary == [
        {
            "layer": 7,
            "phase": "decode",
            "metric": "k99",
            "count": 2,
            "resolved_count": 1,
            "unresolved_count": 1,
            "unresolved_fraction": 0.5,
            "mean": 4096.0,
            "p50": 4096.0,
            "p90": 4096.0,
            "p99": 4096.0,
            "max": 4096.0,
            "unresolved_measurement_cap_min": 8192.0,
            "unresolved_measurement_cap_mean": 8192.0,
            "unresolved_measurement_cap_max": 8192.0,
        }
    ]


def test_dsa_diagnostics_rejects_cuda_graphs_in_transformer_config(tmp_path):
    with pytest.raises(AssertionError, match="dsa_diagnostics requires cuda_graph_impl='none'"):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            experimental_attention_variant="dsa",
            dsa_diagnostics=True,
            dsa_diagnostics_output_dir=str(tmp_path),
            cuda_graph_impl="local",
        )


def test_compute_dsa_attention_diagnostics_separates_shared_and_per_head_support():
    query = torch.tensor([[[[4.0, 0.0], [0.0, 4.0]]]])
    key = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
            [[[0.5, 0.5]]],
        ]
    )
    value = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
            [[[1.0, 1.0]]],
        ]
    )
    metrics = compute_dsa_attention_diagnostics(
        query=query,
        key=key,
        value=value,
        indexer_support=torch.tensor([0, 1, 2]),
        model_support=torch.tensor([0]),
        softmax_scale=1.0,
        topk_values=[1, 3],
        query_position=2,
        dump_support_indices=True,
    )

    support_k1 = metrics["supports"]["1"]
    assert support_k1["per_head_oracle"]["captured_mass_min"] > support_k1[
        "sum_head_oracle"
    ]["captured_mass_min"]
    assert support_k1["per_head_oracle"]["head_support_union_size"] == 2
    assert metrics["supports"]["3"]["indexer"]["captured_mass_mean"] == pytest.approx(1.0)
    assert support_k1["support_indices"]["per_head_oracle"] == [[0], [1]]


def test_compute_dsa_attention_diagnostics_tp_reconstructs_global_heads():
    if Utils.world_size < 2 or Utils.world_size % 2 != 0:
        pytest.skip("Launch with torchrun and a world size divisible by 2 for TP diagnostics.")

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    try:
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        device = torch.device("cuda", torch.cuda.current_device())
        query_vector = [4.0, 0.0] if tp_rank == 0 else [0.0, 4.0]
        query = torch.tensor([[[query_vector]]], device=device)
        key = torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[0.0, 1.0]]],
                [[[0.5, 0.5]]],
            ],
            device=device,
        )
        value = key.clone()
        metrics = compute_dsa_attention_diagnostics(
            query=query,
            key=key,
            value=value,
            indexer_support=torch.tensor([0, 1, 2], device=device),
            model_support=torch.tensor([0], device=device),
            softmax_scale=1.0,
            topk_values=[1],
            query_position=2,
            tp_group=parallel_state.get_tensor_model_parallel_group(),
        )

        per_head = metrics["supports"]["1"]["per_head_oracle"]
        assert len(per_head["captured_mass_per_head"]) == 2
        assert per_head["head_support_union_size"] == 2
        assert metrics["supports"]["1"]["sum_head_oracle"][
            "captured_mass_mean"
        ] == pytest.approx(metrics["supports"]["1"]["max_head_oracle"]["captured_mass_mean"])
    finally:
        Utils.destroy_model_parallel()

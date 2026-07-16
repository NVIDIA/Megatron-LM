# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from megatron.training.statistics_logging import save_raw_moments_by_name


def _read_records(filepath):
    return [json.loads(line) for line in filepath.read_text().strip().split("\n")]


def test_save_raw_moments_by_name_creates_jsonl(tmp_path):
    save_raw_moments_by_name(
        str(tmp_path),
        "custom_raw_moments",
        iteration=100,
        consumed_train_samples=8192,
        raw_moments_by_name=[
            (
                "decoder.layers.0.self_attention.linear_qkv.weight",
                {"count": 2, "sum_1": 1.5, "sum_2": 2.5, "sum_3": 3.5, "sum_4": 4.5},
            ),
            (
                "decoder.layers.0.mlp.linear_fc1.weight",
                {"count": 3, "sum_1": 2.25, "sum_2": 5.25, "sum_3": 8.25, "sum_4": 11.25},
            ),
        ],
        rank=7,
    )

    filepath = tmp_path / "training_stats" / "custom_raw_moments" / "rank7.jsonl"
    assert _read_records(filepath) == [
        {
            "iter": 100,
            "consumed_train_samples": 8192,
            "stat": "custom_raw_moments",
            "values": {
                "decoder.layers.0.self_attention.linear_qkv.weight": {
                    "count": 2.0,
                    "sum_1": 1.5,
                    "sum_2": 2.5,
                    "sum_3": 3.5,
                    "sum_4": 4.5,
                },
                "decoder.layers.0.mlp.linear_fc1.weight": {
                    "count": 3.0,
                    "sum_1": 2.25,
                    "sum_2": 5.25,
                    "sum_3": 8.25,
                    "sum_4": 11.25,
                },
            },
        }
    ]


def test_save_raw_moments_by_name_appends_across_calls(tmp_path):
    first_moments = {"count": 1, "sum_1": 1, "sum_2": 1, "sum_3": 1, "sum_4": 1}
    second_moments = {"count": 1, "sum_1": 2, "sum_2": 4, "sum_3": 8, "sum_4": 16}
    save_raw_moments_by_name(
        str(tmp_path),
        "param_raw_moments_by_param",
        iteration=100,
        consumed_train_samples=8192,
        raw_moments_by_name=[("layer.weight", first_moments)],
        rank=0,
    )
    save_raw_moments_by_name(
        str(tmp_path),
        "param_raw_moments_by_param",
        iteration=200,
        consumed_train_samples=16384,
        raw_moments_by_name=[("layer.weight", second_moments)],
        rank=0,
    )

    filepath = tmp_path / "training_stats" / "param_raw_moments_by_param" / "rank0.jsonl"
    records = _read_records(filepath)
    assert [record["iter"] for record in records] == [100, 200]
    assert records[0]["values"] == {
        "layer.weight": {"count": 1.0, "sum_1": 1.0, "sum_2": 1.0, "sum_3": 1.0, "sum_4": 1.0}
    }
    assert records[1]["values"] == {
        "layer.weight": {"count": 1.0, "sum_1": 2.0, "sum_2": 4.0, "sum_3": 8.0, "sum_4": 16.0}
    }


def test_save_raw_moments_by_name_skips_empty_values(tmp_path):
    save_raw_moments_by_name(
        str(tmp_path),
        "param_raw_moments_by_param",
        iteration=100,
        consumed_train_samples=8192,
        raw_moments_by_name=[],
        rank=0,
    )

    filepath = tmp_path / "training_stats" / "param_raw_moments_by_param" / "rank0.jsonl"
    assert not filepath.exists()

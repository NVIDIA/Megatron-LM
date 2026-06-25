# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from megatron.training.statistics_logging import (
    save_activation_raw_moments_by_layer,
    save_dgrad_raw_moments_by_layer,
    save_grad_raw_moments_by_param,
    save_param_raw_moments_by_param,
)


def _read_records(filepath):
    return [json.loads(line) for line in filepath.read_text().strip().split("\n")]


class TestSaveParamRawMomentsByParam:
    def test_creates_jsonl(self, tmp_path):
        save_param_raw_moments_by_param(
            str(tmp_path),
            iteration=100,
            consumed_train_samples=8192,
            param_raw_moments_by_param=[
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

        filepath = tmp_path / "training_stats" / "param_raw_moments_by_param" / "rank7.jsonl"
        assert filepath.exists()

        records = _read_records(filepath)
        assert records == [
            {
                "iter": 100,
                "consumed_train_samples": 8192,
                "stat": "param_raw_moments_by_param",
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

    def test_appends_across_calls(self, tmp_path):
        first_moments = {"count": 1, "sum_1": 1, "sum_2": 1, "sum_3": 1, "sum_4": 1}
        second_moments = {"count": 1, "sum_1": 2, "sum_2": 4, "sum_3": 8, "sum_4": 16}
        save_param_raw_moments_by_param(
            str(tmp_path),
            iteration=100,
            consumed_train_samples=8192,
            param_raw_moments_by_param=[("layer.weight", first_moments)],
            rank=0,
        )
        save_param_raw_moments_by_param(
            str(tmp_path),
            iteration=200,
            consumed_train_samples=16384,
            param_raw_moments_by_param=[("layer.weight", second_moments)],
            rank=0,
        )

        filepath = tmp_path / "training_stats" / "param_raw_moments_by_param" / "rank0.jsonl"
        records = _read_records(filepath)
        assert len(records) == 2
        assert records[0]["iter"] == 100
        assert records[0]["values"] == {
            "layer.weight": {"count": 1.0, "sum_1": 1.0, "sum_2": 1.0, "sum_3": 1.0, "sum_4": 1.0}
        }
        assert records[1]["iter"] == 200
        assert records[1]["values"] == {
            "layer.weight": {"count": 1.0, "sum_1": 2.0, "sum_2": 4.0, "sum_3": 8.0, "sum_4": 16.0}
        }

    def test_empty_values_do_not_create_file(self, tmp_path):
        save_param_raw_moments_by_param(
            str(tmp_path),
            iteration=100,
            consumed_train_samples=8192,
            param_raw_moments_by_param=[],
            rank=0,
        )

        filepath = tmp_path / "training_stats" / "param_raw_moments_by_param" / "rank0.jsonl"
        assert not filepath.exists()


class TestSaveGradRawMomentsByParam:
    def test_creates_jsonl(self, tmp_path):
        save_grad_raw_moments_by_param(
            str(tmp_path),
            iteration=100,
            consumed_train_samples=8192,
            grad_raw_moments_by_param=[
                (
                    "decoder.layers.0.self_attention.linear_qkv.weight",
                    {"count": 4, "sum_1": 3.5, "sum_2": 6.5, "sum_3": 9.5, "sum_4": 12.5},
                ),
                (
                    "decoder.layers.0.mlp.linear_fc1.weight",
                    {"count": 5, "sum_1": 4.25, "sum_2": 8.25, "sum_3": 12.25, "sum_4": 16.25},
                ),
            ],
            rank=3,
        )

        filepath = tmp_path / "training_stats" / "grad_raw_moments_by_param" / "rank3.jsonl"
        records = _read_records(filepath)
        assert records == [
            {
                "iter": 100,
                "consumed_train_samples": 8192,
                "stat": "grad_raw_moments_by_param",
                "gradient_stage": "pre_clip",
                "values": {
                    "decoder.layers.0.self_attention.linear_qkv.weight": {
                        "count": 4.0,
                        "sum_1": 3.5,
                        "sum_2": 6.5,
                        "sum_3": 9.5,
                        "sum_4": 12.5,
                    },
                    "decoder.layers.0.mlp.linear_fc1.weight": {
                        "count": 5.0,
                        "sum_1": 4.25,
                        "sum_2": 8.25,
                        "sum_3": 12.25,
                        "sum_4": 16.25,
                    },
                },
            }
        ]


class TestSaveActivationRawMomentsByLayer:
    def test_creates_jsonl(self, tmp_path):
        save_activation_raw_moments_by_layer(
            str(tmp_path),
            iteration=100,
            consumed_train_samples=8192,
            activation_raw_moments_by_layer=[
                (
                    "decoder.layers.0.self_attention.linear_qkv/output0",
                    {"count": 2, "sum_1": 1.5, "sum_2": 2.5, "sum_3": 3.5, "sum_4": 4.5},
                )
            ],
            rank=2,
        )

        filepath = tmp_path / "training_stats" / "activation_raw_moments_by_layer" / "rank2.jsonl"
        records = _read_records(filepath)
        assert records == [
            {
                "iter": 100,
                "consumed_train_samples": 8192,
                "stat": "activation_raw_moments_by_layer",
                "values": {
                    "decoder.layers.0.self_attention.linear_qkv/output0": {
                        "count": 2.0,
                        "sum_1": 1.5,
                        "sum_2": 2.5,
                        "sum_3": 3.5,
                        "sum_4": 4.5,
                    }
                },
            }
        ]


class TestSaveDgradRawMomentsByLayer:
    def test_creates_jsonl_with_loss_scale(self, tmp_path):
        save_dgrad_raw_moments_by_layer(
            str(tmp_path),
            iteration=100,
            consumed_train_samples=8192,
            dgrad_raw_moments_by_layer=[
                (
                    "decoder.layers.0.self_attention.linear_qkv/input0",
                    {"count": 3, "sum_1": 2.5, "sum_2": 4.5, "sum_3": 6.5, "sum_4": 8.5},
                )
            ],
            rank=5,
            loss_scale=128.0,
        )

        filepath = tmp_path / "training_stats" / "dgrad_raw_moments_by_layer" / "rank5.jsonl"
        records = _read_records(filepath)
        assert records == [
            {
                "iter": 100,
                "consumed_train_samples": 8192,
                "stat": "dgrad_raw_moments_by_layer",
                "gradient_stage": "backward_scaled",
                "loss_scale": 128.0,
                "values": {
                    "decoder.layers.0.self_attention.linear_qkv/input0": {
                        "count": 3.0,
                        "sum_1": 2.5,
                        "sum_2": 4.5,
                        "sum_3": 6.5,
                        "sum_4": 8.5,
                    }
                },
            }
        ]

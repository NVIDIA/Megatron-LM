# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging

import pytest
import torch
import torch.nn as nn

from megatron.training.activation_logging import ActivationLogger


@pytest.fixture()
def logger(tmp_path):
    return ActivationLogger(save_dir=str(tmp_path))


@pytest.fixture()
def simple_model():
    """A minimal model with two nn.Linear layers, wrapped as a single-element model chunk list."""
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 8))
    return [model]


class TestMakeTpeHook:
    """Tests for _make_tpe_hook regex layer extraction."""

    def test_extracts_decoder_layer_number(self, logger):
        hook = logger._make_tpe_hook("chunk0", "decoder.layers.3.mlp.experts.linear_fc1")
        assert hook is not None
        fake_tpe = [128, 64, 96, 80]
        hook(None, (torch.zeros(1), fake_tpe), {}, torch.zeros(1))
        assert logger._decoder_tpe_records[3] == [fake_tpe]

    def test_extracts_mtp_layer_number(self, logger):
        hook = logger._make_tpe_hook(
            "chunk0", "mtp.layers.0.mtp_model_layer.layers.1.mlp.experts.linear_fc1"
        )
        assert hook is not None
        fake_tpe = [50, 50]
        hook(None, (torch.zeros(1), fake_tpe), {}, torch.zeros(1))
        assert logger._mtp_tpe_records[(0, 1)] == [fake_tpe]

    def test_returns_none_for_non_matching_name(self, logger, caplog):
        with caplog.at_level(logging.WARNING):
            hook = logger._make_tpe_hook("chunk0", "some.module.without.layer.number")
        assert hook is None
        assert "Cannot extract layer number" in caplog.text


class TestSaveTpe:
    """Tests for save_tpe JSONL output."""

    def test_creates_jsonl(self, tmp_path, logger):
        logger._decoder_tpe_records[3].append([10, 20])
        logger._decoder_tpe_records[3].append([30, 40])
        logger._decoder_tpe_records[7].append([50, 60])
        logger._mtp_tpe_records[(0, 1)].append([70, 80])

        logger.save_tpe(iteration=100)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        filepath = tmp_path / "tokens_per_expert" / f"rank{rank}.jsonl"
        assert filepath.exists()

        records = [json.loads(line) for line in filepath.read_text().strip().split("\n")]
        assert records == [
            {"iter": 100, "block": "decoder", "layer": 3, "tpe": [[10, 20], [30, 40]]},
            {"iter": 100, "block": "decoder", "layer": 7, "tpe": [[50, 60]]},
            {"iter": 100, "block": "mtp", "mtp_idx": 0, "layer": 1, "tpe": [[70, 80]]},
        ]

    def test_appends_across_calls(self, tmp_path, logger):
        logger._decoder_tpe_records[0].append([10, 20])
        logger.save_tpe(iteration=100)

        logger._decoder_tpe_records[0].append([30, 40])
        logger.save_tpe(iteration=200)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        filepath = tmp_path / "tokens_per_expert" / f"rank{rank}.jsonl"
        records = [json.loads(line) for line in filepath.read_text().strip().split("\n")]
        assert len(records) == 2
        assert records[0]["iter"] == 100
        assert records[1]["iter"] == 200


class TestActivationHookLifecycle:
    """Tests for activation hook registration and removal."""

    def test_register_and_remove(self, logger, simple_model):
        logger.register_activation_hooks(simple_model)
        assert len(logger._activation_hooks) == 2

        logger.remove_activation_hooks()
        assert len(logger._activation_hooks) == 0

    def test_hooks_capture_activations(self, logger, simple_model):
        logger.register_activation_hooks(simple_model)

        simple_model[0](torch.randn(2, 16))

        assert len(logger._activations_state_dict) > 0
        logger.remove_activation_hooks()

    def test_removed_hooks_dont_capture(self, logger, simple_model):
        logger.register_activation_hooks(simple_model)
        logger.remove_activation_hooks()

        simple_model[0](torch.randn(2, 16))
        assert len(logger._activations_state_dict) == 0

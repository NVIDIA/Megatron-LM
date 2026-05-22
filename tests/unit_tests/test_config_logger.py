# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import json
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn

from megatron.core import config_logger


@dataclasses.dataclass
class ExampleConfig:
    hidden_size: int = 128
    dtype: torch.dtype = torch.float16


def test_config_logger_path_helpers():
    enabled = SimpleNamespace(config_logger_dir="/tmp/configs")
    disabled = SimpleNamespace(config_logger_dir="")
    missing = SimpleNamespace()

    assert config_logger.get_config_logger_path(enabled) == "/tmp/configs"
    assert config_logger.has_config_logger_enabled(enabled)
    assert not config_logger.has_config_logger_enabled(disabled)
    assert not config_logger.has_config_logger_enabled(missing)


def test_config_logger_path_counter_is_per_path():
    config_logger.__dict__["__config_logger_path_counts"].clear()

    assert config_logger.get_path_count("run") == 0
    assert config_logger.get_path_count("run") == 1
    assert config_logger.get_path_with_count("other") == "other.iter0"


def test_json_encoder_handles_mcore_types():
    payload = {
        "dtype": torch.float32,
        "module": nn.Sequential(nn.Linear(1, 1)),
        "leaf": nn.ReLU(),
        "dataclass": ExampleConfig(),
        "fn": test_json_encoder_handles_mcore_types,
    }

    encoded = json.loads(json.dumps(payload, cls=config_logger.JSONEncoderWithMcoreTypes))

    assert encoded["dtype"] == "torch.float32"
    assert "0" in encoded["module"]
    assert "ReLU" in encoded["leaf"]
    assert encoded["dataclass"]["hidden_size"] == 128
    assert "test_json_encoder_handles_mcore_types" in encoded["fn"]


def test_log_config_to_disk_writes_json_and_removes_self(tmp_path, monkeypatch):
    config_logger.__dict__["__config_logger_path_counts"].clear()
    monkeypatch.setattr(config_logger.parallel_state, "get_all_ranks", lambda: "0_0_0_0_0")

    class Owner:
        pass

    config = SimpleNamespace(config_logger_dir=str(tmp_path))
    payload = {"self": Owner(), "hidden": 1024, "dtype": torch.bfloat16}

    config_logger.log_config_to_disk(config, payload)

    output_path = tmp_path / "Owner.rank_0_0_0_0_0.iter0.json"
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data == {"hidden": 1024, "dtype": "torch.bfloat16"}
    assert "self" not in payload


def test_log_config_to_disk_writes_ordered_dict_as_torch_file(tmp_path):
    config_logger.__dict__["__config_logger_path_counts"].clear()

    config = SimpleNamespace(config_logger_dir=str(tmp_path))
    payload = OrderedDict([("a", torch.tensor([1, 2]))])

    config_logger.log_config_to_disk(config, payload, prefix="ordered", rank_str="rank0")

    output_path = tmp_path / "ordered.rank_rank0.iter0.pth"
    assert output_path.exists()
    loaded = torch.load(output_path)
    assert list(loaded) == ["a"]
    assert torch.equal(loaded["a"], torch.tensor([1, 2]))

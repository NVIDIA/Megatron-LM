# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for megatron.training.utils.checkpoint_utils."""

from dataclasses import dataclass, field
from unittest import mock

import pytest
import yaml

from megatron.training.utils.checkpoint_utils import (
    CONFIG_FILE,
    apply_run_config_backward_compat,
    get_checkpoint_run_config_filename,
    read_run_config,
)


@dataclass
class _LegacyConfig:
    """Config with a field that older checkpoints serialized but ``__init__`` no longer accepts."""

    active: int = 1
    removed: str = field(default="", init=False)


@pytest.fixture
def _allow_local_targets():
    """Disable the target allowlist so test-local dataclasses can be resolved."""
    from megatron.training.config.instantiate_utils import target_allowlist

    target_allowlist.disable()
    yield
    target_allowlist.enable()


class TestGetCheckpointRunConfigFilename:
    """``get_checkpoint_run_config_filename`` joins the checkpoint dir with CONFIG_FILE."""

    def test_joins_checkpoints_path_with_config_file(self):
        filename = get_checkpoint_run_config_filename("/some/checkpoint/dir")
        assert filename == f"/some/checkpoint/dir/{CONFIG_FILE}"

    def test_accepts_relative_path(self):
        filename = get_checkpoint_run_config_filename("relative/dir")
        assert filename == f"relative/dir/{CONFIG_FILE}"


class TestReadRunConfigNonDistributed:
    """``read_run_config`` without torch.distributed initialized reads the file directly."""

    @pytest.fixture(autouse=True)
    def _disable_distributed(self):
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            yield

    def test_reads_yaml_file(self, tmp_path):
        run_config_path = tmp_path / "run_config.yaml"
        data = {"_target_": "some.Config", "train": {"global_batch_size": 8}}
        with open(run_config_path, "w") as f:
            yaml.safe_dump(data, f)

        loaded = read_run_config(str(run_config_path))

        assert loaded == data

    def test_missing_file_raises_runtime_error(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.yaml"

        with pytest.raises(RuntimeError):
            read_run_config(str(missing_path))

    def test_strips_runtime_only_timers_target(self, tmp_path):
        run_config_path = tmp_path / "run_config.yaml"
        data = {"_target_": "some.Config", "timers": {"_target_": "megatron.core.timers.Timers"}}
        with open(run_config_path, "w") as f:
            yaml.safe_dump(data, f)

        loaded = read_run_config(str(run_config_path))

        assert loaded["timers"] is None

    def test_invalid_yaml_raises_runtime_error(self, tmp_path):
        run_config_path = tmp_path / "run_config.yaml"
        run_config_path.write_text("invalid: yaml: content: [")

        with pytest.raises(RuntimeError, match="Unable to load config file"):
            read_run_config(str(run_config_path))

    def test_strips_runtime_only_targets_in_nested_dicts_and_lists(self, tmp_path):
        run_config_path = tmp_path / "run_config.yaml"
        data = {
            "model": {
                "timers": {"_target_": "megatron.core.timers.Timers"},
                "keep": {"_target_": "some.other.Component", "value": 1},
                "nested": [
                    {"timers": {"_target_": "megatron.core.timers.Timers"}},
                    {"other": {"_target_": "another.Component", "value": 2}},
                ],
            },
            "tokenizer": {"type": "sentencepiece"},
        }
        with open(run_config_path, "w") as f:
            yaml.safe_dump(data, f)

        loaded = read_run_config(str(run_config_path))

        assert loaded["model"]["timers"] is None
        assert loaded["model"]["nested"][0]["timers"] is None
        assert loaded["model"]["keep"] == {"_target_": "some.other.Component", "value": 1}
        assert loaded["model"]["nested"][1]["other"] == {
            "_target_": "another.Component",
            "value": 2,
        }
        assert loaded["tokenizer"] == {"type": "sentencepiece"}

    def test_removes_init_false_fields_end_to_end(self, tmp_path, _allow_local_targets):
        """Exercises the real sanitizer (not a mock) through ``read_run_config``."""
        run_config_path = tmp_path / "run_config.yaml"
        data = {
            "model": {
                "_target_": f"{_LegacyConfig.__module__}.{_LegacyConfig.__qualname__}",
                "active": 2,
                "removed": "old",
            }
        }
        with open(run_config_path, "w") as f:
            yaml.safe_dump(data, f)

        loaded = read_run_config(str(run_config_path))

        assert "removed" not in loaded["model"]
        assert loaded["model"]["active"] == 2

    def test_runtime_only_sanitization_runs_before_backward_compat(self, tmp_path):
        run_config_path = tmp_path / "run_config.yaml"
        with open(run_config_path, "w") as f:
            yaml.safe_dump({"timers": {"_target_": "megatron.core.timers.Timers"}}, f)

        seen_by_compat = []

        def _record(config_dict):
            seen_by_compat.append(config_dict)
            return config_dict

        with mock.patch(
            "megatron.training.utils.checkpoint_utils.apply_run_config_backward_compat",
            side_effect=_record,
        ):
            read_run_config(str(run_config_path))

        # Timers must already be replaced by None when backward compat runs; otherwise the
        # compat step would try to resolve a target that cannot be rebuilt without runtime args.
        assert seen_by_compat == [{"timers": None}]

    def test_preserves_serialized_quantization_recipe(self, tmp_path):
        run_config_path = tmp_path / "run_config.yaml"
        data = {
            "_target_": "some.Config",
            "quant_recipe": {
                "_target_": (
                    "megatron.core.quantization.quant_config.RecipeConfig.from_config_dict"
                ),
                "config": {"matchers": None, "configs": {"default": {"format": "bf16"}}},
            },
        }
        with open(run_config_path, "w") as f:
            yaml.safe_dump(data, f)

        loaded = read_run_config(str(run_config_path))

        assert loaded == data


class TestReadRunConfigDistributed:
    """``read_run_config`` under torch.distributed reads on rank 0 and broadcasts."""

    def test_rank0_reads_and_broadcasts(self, tmp_path):
        run_config_path = tmp_path / "run_config.yaml"
        data = {"_target_": "some.Config", "train": {"global_batch_size": 8}}
        with open(run_config_path, "w") as f:
            yaml.safe_dump(data, f)

        broadcasted = []

        def _fake_broadcast(obj_list, src=0):
            broadcasted.append(obj_list[0])

        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("megatron.training.utils.checkpoint_utils.safe_get_rank", return_value=0),
            mock.patch(
                "megatron.training.utils.checkpoint_utils.safe_get_world_size", return_value=4
            ),
            mock.patch("torch.distributed.broadcast_object_list", side_effect=_fake_broadcast),
        ):
            loaded = read_run_config(str(run_config_path))

        assert loaded == data
        assert broadcasted == [data]

    def test_non_rank0_receives_broadcasted_value(self, tmp_path):
        run_config_path = tmp_path / "run_config.yaml"
        broadcast_result = {"_target_": "some.Config", "train": {"global_batch_size": 8}}

        def _fake_broadcast(obj_list, src=0):
            obj_list[0] = broadcast_result

        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("megatron.training.utils.checkpoint_utils.safe_get_rank", return_value=1),
            mock.patch(
                "megatron.training.utils.checkpoint_utils.safe_get_world_size", return_value=4
            ),
            mock.patch("torch.distributed.broadcast_object_list", side_effect=_fake_broadcast),
        ):
            loaded = read_run_config(str(run_config_path))

        assert loaded == broadcast_result

    def test_rank0_read_error_raises_on_all_ranks(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.yaml"

        def _fake_broadcast(obj_list, src=0):
            pass

        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("megatron.training.utils.checkpoint_utils.safe_get_rank", return_value=0),
            mock.patch(
                "megatron.training.utils.checkpoint_utils.safe_get_world_size", return_value=4
            ),
            mock.patch("torch.distributed.broadcast_object_list", side_effect=_fake_broadcast),
        ):
            with pytest.raises(RuntimeError, match="Unable to load config file"):
                read_run_config(str(missing_path))

    def test_non_rank0_raises_rank0_error_message(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.yaml"
        rank0_error = {
            "error": True,
            "msg": f"ERROR: Unable to load config file {missing_path}: boom",
        }

        def _fake_broadcast(obj_list, src=0):
            obj_list[0] = rank0_error

        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("megatron.training.utils.checkpoint_utils.safe_get_rank", return_value=1),
            mock.patch(
                "megatron.training.utils.checkpoint_utils.safe_get_world_size", return_value=4
            ),
            mock.patch("torch.distributed.broadcast_object_list", side_effect=_fake_broadcast),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                read_run_config(str(missing_path))


class TestApplyRunConfigBackwardCompat:
    """``apply_run_config_backward_compat`` delegates to sanitize_dataclass_config."""

    def test_delegates_to_sanitize_dataclass_config(self):
        config_dict = {"_target_": "some.Config", "a": 1}

        with mock.patch(
            "megatron.training.utils.checkpoint_utils.sanitize_dataclass_config",
            return_value={"sanitized": True},
        ) as mock_sanitize:
            result = apply_run_config_backward_compat(config_dict)

        mock_sanitize.assert_called_once_with(config_dict)
        assert result == {"sanitized": True}

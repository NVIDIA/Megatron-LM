# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for megatron.training.utils.checkpoint_utils."""

from unittest import mock

import pytest
import yaml

from megatron.training.utils.checkpoint_utils import (
    CONFIG_FILE,
    apply_run_config_backward_compat,
    get_checkpoint_run_config_filename,
    read_run_config,
)


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

    def test_strips_legacy_quantization_recipe(self, tmp_path):
        run_config_path = tmp_path / "run_config.yaml"
        data = {
            "_target_": "some.Config",
            "quant_recipe": {
                "_target_": "megatron.core.quantization.quant_config.RecipeConfig",
                "_call_": True,
            },
        }
        with open(run_config_path, "w") as f:
            yaml.safe_dump(data, f)

        loaded = read_run_config(str(run_config_path))

        assert loaded["quant_recipe"] is None


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
            mock.patch("megatron.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            mock.patch(
                "megatron.training.utils.checkpoint_utils.get_world_size_safe", return_value=4
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
            mock.patch("megatron.training.utils.checkpoint_utils.get_rank_safe", return_value=1),
            mock.patch(
                "megatron.training.utils.checkpoint_utils.get_world_size_safe", return_value=4
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
            mock.patch("megatron.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            mock.patch(
                "megatron.training.utils.checkpoint_utils.get_world_size_safe", return_value=4
            ),
            mock.patch("torch.distributed.broadcast_object_list", side_effect=_fake_broadcast),
        ):
            with pytest.raises(RuntimeError):
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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ModelOpt startup detection and checkpoint selection in megatron.post_training."""

from argparse import Namespace
from unittest import mock

import pytest
import torch

pytest.importorskip("modelopt", reason="modelopt is not installed")


class TestModelOptStartupDetection:
    """`should_use_modelopt_config` runs at startup, before any run config is registered, and decides
    which model config class the entry point builds; it inspects the load directory with the
    checkpoint policy built from the arguments. Run-time callers pass the registered section."""

    QUANTIZED = {"modelopt_state_dict": [("quantize", {})]}

    def _detect(self, monkeypatch, args, state, checkpoint=None):
        from megatron.post_training import checkpointing as modelopt_checkpointing
        from megatron.post_training.utils import should_use_modelopt_config
        from megatron.training import global_vars

        monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
        directory = mock.MagicMock()
        directory.__truediv__.return_value.is_dir.return_value = True
        monkeypatch.setattr(modelopt_checkpointing, "get_load_checkpoint_path", lambda *a: "/ckpt")
        monkeypatch.setattr(modelopt_checkpointing, "Path", lambda path: directory)
        read = mock.Mock(return_value=state)
        monkeypatch.setattr(
            modelopt_checkpointing.dist_checkpointing, "load_common_state_dict", read
        )
        before = vars(args).copy()
        selected = should_use_modelopt_config(args, checkpoint=checkpoint)
        assert vars(args) == before
        return read, selected

    def test_sharded_modelopt_checkpoint_is_detected_before_any_run_config(self, monkeypatch):
        args = Namespace(load="/synthetic/modelopt", ckpt_format="torch_dist")
        read, selected = self._detect(monkeypatch, args, self.QUANTIZED)
        assert read.called and selected is True

    def test_distillation_only_state_is_not_modelopt(self, monkeypatch):
        args = Namespace(load="/synthetic/modelopt", ckpt_format="torch_dist")
        _, selected = self._detect(monkeypatch, args, {"modelopt_state_dict": [("kd_loss", {})]})
        assert selected is False

    def test_legacy_format_is_inspected_with_the_startup_policy(self, monkeypatch):
        from megatron.post_training import checkpointing as modelopt_checkpointing
        from megatron.post_training.utils import should_use_modelopt_config
        from megatron.training import global_vars

        args = Namespace(load="/synthetic/modelopt", ckpt_format="torch")
        monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
        calls = []

        def base_load(path, run_args, rank0, *, checkpoint, **kwargs):
            calls.append((path, run_args, rank0, checkpoint))
            return {"modelopt_state": self.QUANTIZED}, path, False, None

        monkeypatch.setattr(modelopt_checkpointing, "_load_base_checkpoint", base_load)
        assert should_use_modelopt_config(args) is True
        assert not hasattr(args, "modelopt_enabled")
        ((path, run_args, rank0, checkpoint),) = calls
        assert path == "/synthetic/modelopt" and run_args is args and rank0 is False
        assert checkpoint.ckpt_format == "torch" and checkpoint.load == "/synthetic/modelopt"

    def test_run_time_callers_pass_the_registered_section(self, monkeypatch, run_config):
        run_config.checkpoint.load = "/section/modelopt"
        args = Namespace(load=None, ckpt_format="torch_dist")  # Deliberately stale.
        read, selected = self._detect(
            monkeypatch, args, self.QUANTIZED, checkpoint=run_config.checkpoint
        )
        assert read.called and selected is True


class TestModelOptCheckpointSelection:
    @pytest.mark.parametrize(
        "case, leaf",
        [
            ("text_tracker", "iter_0000010"),
            ("train_state", "iter_0000020"),
            ("train_state_only", "iter_0000020"),
            ("step", "iter_0000007"),
            ("step_without_tracker", "iter_0000007"),
            ("release", "release"),
            ("train_state_release", "release"),
            ("pretrained_fallback", "iter_0000020"),
        ],
    )
    def test_detection_restoration_and_weights_select_the_same_checkpoint(
        self, monkeypatch, tmp_path, case, leaf
    ):
        from megatron.post_training import checkpointing as modelopt_checkpointing
        from megatron.post_training.utils import should_use_modelopt_config
        from megatron.training import checkpointing
        from megatron.training.config import CheckpointConfig
        from megatron.training.state import TrainState
        from megatron.training.utils.checkpoint_utils import get_checkpoint_train_state_filename

        load_dir = tmp_path / "load"
        load_dir.mkdir()
        root = tmp_path / "pretrained" if case == "pretrained_fallback" else load_dir
        root.mkdir(exist_ok=True)
        if case not in ("train_state_only", "step_without_tracker"):
            (root / "latest_checkpointed_iteration.txt").write_text(
                "release" if case == "release" else "10"
            )
        if case in (
            "train_state",
            "train_state_only",
            "train_state_release",
            "pretrained_fallback",
        ):
            torch.save(
                TrainState(iteration=20, release=case == "train_state_release").state_dict(),
                get_checkpoint_train_state_filename(str(root), prefix="latest"),
            )
        selected = root / leaf
        (selected / "modelopt_state").mkdir(parents=True)
        pretrained = None
        if case == "pretrained_fallback":
            pretrained = str(root)
        elif case == "train_state_only":
            # The train-state file must count as a checkpoint even without a text tracker.
            pretrained = str(tmp_path / "unused-fallback")
        checkpoint = CheckpointConfig(
            load=str(load_dir),
            pretrained_checkpoint=pretrained,
            ckpt_step=7 if case.startswith("step") else None,
        )
        original = vars(checkpoint).copy()
        args = Namespace(load="/stale", ckpt_format="torch")
        read_state = mock.Mock(return_value={"modelopt_state_dict": [("quantize", {})]})
        restore = mock.Mock()
        monkeypatch.setattr(
            modelopt_checkpointing.dist_checkpointing, "load_common_state_dict", read_state
        )
        monkeypatch.setattr(modelopt_checkpointing, "restore_sharded_modelopt_state", restore)
        assert should_use_modelopt_config(args, checkpoint=checkpoint) is True
        model = mock.Mock()
        modelopt_checkpointing.load_modelopt_state(model, checkpoint=checkpoint)
        read_state.assert_called_once_with(str(selected / "modelopt_state"))
        restore.assert_called_once_with([model], selected)

        # The actual base loader selects the same directory; stop at the tensor reader.
        reads = []

        def read_weights(directory, run_args, rank0, state_dict, iteration, release, **kwargs):
            reads.append(
                checkpointing.get_checkpoint_name(
                    directory, iteration, release, return_base_dir=True
                )
            )
            return {}, reads[-1], release, checkpointing.CheckpointType.GLOBAL

        monkeypatch.setattr(
            checkpointing, "_get_checkpoint_format", mock.Mock(return_value="torch_dist")
        )
        monkeypatch.setattr(checkpointing, "_load_global_dist_base_checkpoint", read_weights)
        resolved, fallback = checkpointing.resolve_checkpoint_load_dir(checkpoint)
        assert fallback is (case == "pretrained_fallback")
        checkpointing._load_base_checkpoint(resolved, args, rank0=True, checkpoint=checkpoint)
        assert reads == [str(selected)]
        assert vars(checkpoint) == original and vars(args) == dict(
            load="/stale", ckpt_format="torch"
        )

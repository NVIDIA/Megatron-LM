# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ModelOpt startup detection in megatron.post_training."""

from argparse import Namespace
from unittest import mock

import pytest

pytest.importorskip("modelopt", reason="modelopt is not installed")


class TestModelOptStartupDetection:
    """`maybe_enable_modelopt` runs at startup, before any run config is registered, and decides
    which model config class the entry point builds; it inspects the load directory with the
    checkpoint policy built from the arguments. Run-time callers pass the registered section."""

    QUANTIZED = {"modelopt_state_dict": [("quantize", {})]}

    def _detect(self, monkeypatch, args, state, checkpoint=None):
        from megatron.post_training import checkpointing as modelopt_checkpointing
        from megatron.post_training.utils import maybe_enable_modelopt
        from megatron.training import global_vars

        monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
        directory = mock.MagicMock()
        directory.__truediv__.return_value.is_dir.return_value = True
        monkeypatch.setattr(modelopt_checkpointing, "get_sharded_load_dir", lambda path: directory)
        read = mock.Mock(return_value=state)
        monkeypatch.setattr(
            modelopt_checkpointing.dist_checkpointing, "load_common_state_dict", read
        )
        maybe_enable_modelopt(args, checkpoint=checkpoint)
        return read

    def test_sharded_modelopt_checkpoint_is_detected_before_any_run_config(self, monkeypatch):
        args = Namespace(load="/synthetic/modelopt", ckpt_format="torch_dist")
        read = self._detect(monkeypatch, args, self.QUANTIZED)
        assert read.called and args.modelopt_enabled is True

    def test_distillation_only_state_is_not_modelopt(self, monkeypatch):
        args = Namespace(load="/synthetic/modelopt", ckpt_format="torch_dist")
        self._detect(monkeypatch, args, {"modelopt_state_dict": [("kd_loss", {})]})
        assert not getattr(args, "modelopt_enabled", False)

    def test_legacy_format_is_inspected_with_the_startup_policy(self, monkeypatch):
        from megatron.post_training import checkpointing as modelopt_checkpointing
        from megatron.post_training.utils import maybe_enable_modelopt
        from megatron.training import global_vars

        args = Namespace(load="/synthetic/modelopt", ckpt_format="torch")
        monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
        calls = []

        def base_load(path, run_args, rank0, *, checkpoint, **kwargs):
            calls.append((path, run_args, rank0, checkpoint))
            return {"modelopt_state": self.QUANTIZED}, path, False, None

        monkeypatch.setattr(modelopt_checkpointing, "_load_base_checkpoint", base_load)
        maybe_enable_modelopt(args)
        assert args.modelopt_enabled is True
        ((path, run_args, rank0, checkpoint),) = calls
        assert path == "/synthetic/modelopt" and run_args is args and rank0 is False
        assert checkpoint.ckpt_format == "torch" and checkpoint.load == "/synthetic/modelopt"

    def test_run_time_callers_pass_the_registered_section(self, monkeypatch, run_config):
        run_config.checkpoint.load = "/section/modelopt"
        args = Namespace(load=None, ckpt_format="torch_dist")  # Deliberately stale.
        read = self._detect(monkeypatch, args, self.QUANTIZED, checkpoint=run_config.checkpoint)
        assert read.called and args.modelopt_enabled is True

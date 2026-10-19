# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock, patch

import pytest

from megatron.post_training import model_config as mc
from megatron.post_training import model_hooks
from megatron.training.argument_utils import (
    pretrain_cfg_container_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.config.container import InferenceConfigContainer, PretrainConfigContainer
from megatron.training.config.instantiate_utils import InstantiationMode
from megatron.training.models.gpt import GPTModelBuilder
from megatron.training.models.hybrid import HybridModelBuilder
from tests.unit_tests.post_training.test_modelopt_model_config import _make_args


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("kd", [False, True])
def test_whole_run_yaml_roundtrip(tmp_path, hybrid, training, kd):
    args = _make_args(
        export_model_type="HybridModel" if hybrid else "GPTModel",
        hybrid_layer_pattern="M*" if hybrid else None,
        export_kd_teacher_load=str(tmp_path) if kd else None,
    )
    args.load = "/student"
    # Build the real specs: a whole-container test must include their nested callables.
    model = mc.modelopt_model_config_from_args(args)
    if training:
        cfg = pretrain_cfg_container_from_args(args, model)
    else:
        cfg = mc.modelopt_inference_config_from_args(args, device="cpu")
    path = tmp_path / "run.yaml"
    cfg.to_yaml(str(path))
    restored = type(cfg).from_yaml(str(path), mode=InstantiationMode.STRICT)
    restored.validate()
    assert restored.model.get_builder_cls() is (HybridModelBuilder if hybrid else GPTModelBuilder)
    assert restored.checkpoint.load == "/student"
    assert restored.model.transformations == model.transformations
    assert restored.model_build == cfg.model_build
    assert not hasattr(restored.model.transformations, "load")
    if kd:
        assert (
            restored.model.distillation.teacher_model.get_builder_cls()
            is restored.model.get_builder_cls()
        )
    # Changing the only student checkpoint reference must affect transformation restoration.
    restored.checkpoint.load = "/replacement"
    with patch.object(
        model_hooks, "restore_modelopt_state_hook", return_value=lambda x: x
    ) as restore:
        restored.model.get_builder_cls()(restored.model).pre_wrap_hook(restored.checkpoint)
    restore.assert_called_once_with("/replacement", restored.checkpoint.ckpt_format)


def test_tokenizer_finalizes_same_model_used_for_build():
    cfg = mc.modelopt_inference_config_from_args(_make_args())
    model = cfg.model
    model.vocab_size = None
    resolve_tokenizer_vocab_size(cfg, 32768)
    with patch.object(GPTModelBuilder, "build_distributed_models") as build:
        cfg.build_models(Mock())
    assert cfg.model is model
    assert model.vocab_size == 32768
    assert model.should_pad_vocab is False
    build.assert_called_once()


@pytest.mark.parametrize("field", ["use_torch_fsdp2", "use_megatron_fsdp"])
def test_rejects_placement_conflict_before_build(field):
    cfg = mc.modelopt_inference_config_from_args(_make_args(), device="cpu")
    setattr(cfg.dist, field, True)
    with patch.object(GPTModelBuilder, "build_distributed_models") as build:
        with pytest.raises(ValueError, match="FSDP"):
            cfg.build_models(Mock())
    build.assert_not_called()


@pytest.mark.parametrize(
    "field,value", [("tp_comm_overlap", True), ("cross_entropy_fusion_impl", "te")]
)
def test_direct_config_validates_kd_incompatibilities(tmp_path, field, value):
    cfg = mc.modelopt_inference_config_from_args(_make_args(export_kd_teacher_load=str(tmp_path)))
    setattr(cfg.model.transformer, field, value)
    with pytest.raises(ValueError, match="incompatible"):
        cfg.validate()


def test_direct_training_config_rejects_manual_gc(tmp_path):
    args = _make_args(export_kd_teacher_load=str(tmp_path))
    cfg = pretrain_cfg_container_from_args(args, mc.modelopt_model_config_from_args(args))
    cfg.train.manual_gc = True
    with pytest.raises(ValueError, match="manual_gc"):
        cfg.validate()


@pytest.mark.parametrize("fail", [False, True])
def test_legacy_checkpoint_boundary_restores_args(fail):
    from argparse import Namespace

    from megatron.training.config.training_config import CheckpointConfig
    from megatron.training.config.utils import legacy_checkpoint_args

    args = Namespace(load="/old", ckpt_format="torch", unmigrated_setting=True)
    original = vars(args).copy()

    def load():
        with legacy_checkpoint_args(args, CheckpointConfig(load="/new", ckpt_format="torch_dist")):
            assert args.load == "/new"
            assert args.ckpt_format == "torch_dist"
            assert args.unmigrated_setting is True
            if fail:
                raise RuntimeError("load failed")

    if fail:
        with pytest.raises(RuntimeError, match="load failed"):
            load()
    else:
        load()
    assert vars(args) == original


def test_build_policy_conflict_is_rejected():
    cfg = mc.modelopt_inference_config_from_args(_make_args(), device="cpu")
    with pytest.raises(ValueError, match="conflicts"):
        cfg.build_models(Mock(), device="cuda")


def test_unknown_builder_is_not_imported():
    from megatron.training.config.instantiate_utils import InstantiationException

    cfg = mc.modelopt_inference_config_from_args(_make_args())
    data = cfg.to_dict()
    data["model"]["_builder_"] = "os.system"
    with pytest.raises(InstantiationException, match="allowlist"):
        InferenceConfigContainer.from_dict(data)


def test_teacher_inherits_configured_student_format():
    from argparse import Namespace

    from megatron.post_training import checkpointing
    from megatron.training.config.training_config import CheckpointConfig
    from megatron.training.models.distillation import DistillationConfig

    args = Namespace(ckpt_format="torch", finetune=False)
    observed = []
    with (
        patch.object(checkpointing, "get_args", return_value=args),
        patch.object(checkpointing, "unwrap_model"),
        patch.object(
            checkpointing,
            "load_checkpoint",
            side_effect=lambda *a, **kw: observed.append(args.ckpt_format),
        ),
    ):
        checkpointing.load_kd_teacher_checkpoint(
            [Mock()],
            DistillationConfig(teacher_load="/teacher"),
            student_checkpoint=CheckpointConfig(ckpt_format="torch_dist"),
        )
    assert observed == ["torch_dist"]
    assert vars(args) == {"ckpt_format": "torch", "finetune": False}


def test_legacy_inference_helper_reuses_declared_model():
    args = _make_args()
    cfg = mc.modelopt_inference_config_from_args(args)
    model = cfg.model
    with (
        patch("megatron.training.global_vars.get_run_config", return_value=cfg),
        patch("megatron.core.process_groups_config.ProcessGroupCollection.use_mpu_process_groups"),
        patch.object(cfg, "build_models") as build,
        patch.object(mc, "modelopt_inference_config_from_args") as translate,
    ):
        mc.build_modelopt_models(args)
        mc.build_modelopt_models(args)
    translate.assert_not_called()
    assert cfg.model is model
    assert build.call_count == 2


def test_inference_uses_config_without_cli_enable_flag():
    from argparse import Namespace

    from megatron.inference import utils

    args = Namespace(
        modelopt_enabled=False,
        load="/stale",
        ckpt_format="torch",
        inference_ckpt_non_strict=False,
        checkpoint_model_prefix="",
        transformer_impl="transformer_engine",
    )
    stage = Mock()
    cfg = Mock(
        model=Mock(transformations=object()),
        checkpoint=Namespace(load="/configured", ckpt_format="torch_dist"),
    )
    cfg.build_models.return_value = [stage]
    loaded = []
    with (
        patch.object(utils, "get_args", return_value=args),
        patch("megatron.training.global_vars.get_run_config", return_value=cfg),
        patch.object(utils.ProcessGroupCollection, "use_mpu_process_groups"),
        patch.object(utils, "get_model_builder") as legacy_builder,
        patch.object(
            utils,
            "load_checkpoint",
            side_effect=lambda **kw: loaded.append((args.load, args.ckpt_format)),
        ),
    ):
        assert utils.get_model_for_inference() is stage
    legacy_builder.assert_not_called()
    cfg.build_models.assert_called_once()
    assert loaded == [("/configured", "torch_dist")]
    assert (args.load, args.ckpt_format) == ("/stale", "torch")


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_adapter_uses_active_run(fail):
    from argparse import Namespace

    from megatron.training.config.utils import legacy_checkpoint_args

    args = Namespace(load="/old", ckpt_format="torch")
    cfg = Namespace(checkpoint=Namespace(load="/active", ckpt_format="torch_dist"))

    def load():
        with legacy_checkpoint_args():
            assert (args.load, args.ckpt_format) == ("/active", "torch_dist")
            if fail:
                raise RuntimeError("load failed")

    with (
        patch("megatron.training.global_vars.get_args", return_value=args),
        patch("megatron.training.global_vars.get_run_config", return_value=cfg),
    ):
        if fail:
            with pytest.raises(RuntimeError, match="load failed"):
                load()
        else:
            load()
    assert vars(args) == {"load": "/old", "ckpt_format": "torch"}

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock, patch

import pytest

from megatron.post_training import model_config as mc
from megatron.post_training import model_hooks
from megatron.post_training import utils as modelopt_utils
from megatron.training.argument_utils import (
    pretrain_cfg_container_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.config.container import InferenceConfigContainer, PretrainConfigContainer
from megatron.training.config.instantiate_utils import InstantiationMode
from megatron.training.models.gpt import GPTModelBuilder
from megatron.training.models.hybrid import HybridModelBuilder
from tests.unit_tests.post_training.test_modelopt_model_config import _make_args


@pytest.mark.parametrize(
    "option",
    [
        {"export_te_mcore_model": True},
        {"export_kd_teacher_load": "/teacher"},
        {"qad_train_target": "base"},
    ],
)
def test_modelopt_config_selection_from_explicit_options(option):
    args = _make_args(**option)
    args.load = None
    assert modelopt_utils.should_use_modelopt_config(args)


def test_modelopt_config_selection_from_checkpoint():
    args = _make_args()
    args.load = "/modelopt-checkpoint"
    with patch(
        "megatron.post_training.checkpointing.has_modelopt_state", return_value=True
    ) as has_state:
        assert modelopt_utils.should_use_modelopt_config(args)
    has_state.assert_called_once_with(args.load)


def test_plain_run_does_not_select_modelopt_config():
    args = _make_args()
    args.load = None
    assert not modelopt_utils.should_use_modelopt_config(args)


def test_legacy_modelopt_enable_flag_is_only_a_deprecated_input():
    args = _make_args(modelopt_enabled=True)
    args.load = None
    original_args = vars(args).copy()
    with pytest.warns(DeprecationWarning, match="modelopt-enabled is deprecated"):
        assert modelopt_utils.should_use_modelopt_config(args)
    assert vars(args) == original_args


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("kd", [False, True])
def test_whole_run_yaml_roundtrip(tmp_path, hybrid, training, kd):
    args = _make_args(
        hybrid_layer_pattern="M*" if hybrid else None,
        export_kd_teacher_load=str(tmp_path) if kd else None,
    )
    args.load = "/student"
    # Build the real specs: a whole-container test must include their nested callables.
    model = mc.modelopt_model_config_from_args(args)
    if training:
        cfg = pretrain_cfg_container_from_args(args, model)
        cfg.distillation = mc.distillation_config_from_args(args, model)
    else:
        cfg = mc.modelopt_inference_config_from_args(args, device="cpu")
    path = tmp_path / "run.yaml"
    cfg.to_yaml(str(path))
    restored = type(cfg).from_yaml(str(path), mode=InstantiationMode.STRICT)
    restored.validate()
    assert restored.model.get_builder_cls() is (HybridModelBuilder if hybrid else GPTModelBuilder)
    assert restored.checkpoint.load == "/student"
    assert restored.model.transformations == model.transformations
    if training:
        assert "model_build" not in cfg.to_dict()
        assert not hasattr(restored, "model_build")
    else:
        assert restored.model_build == cfg.model_build
    assert not hasattr(restored.model.transformations, "load")
    if training and kd:
        assert restored.distillation.teacher_load == str(tmp_path)
        assert (
            restored.distillation.teacher_model.get_builder_cls()
            is restored.model.get_builder_cls()
        )
        assert restored.distillation.loss_kwargs() == cfg.distillation.loss_kwargs()
    elif training:
        assert restored.distillation is None
    else:
        assert "distillation" not in cfg.to_dict()  # inference never distills
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
    with (
        patch.object(cfg, "validate") as validate,
        patch.object(GPTModelBuilder, "build_distributed_models") as build,
    ):
        cfg.build_models(Mock())
    validate.assert_not_called()
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
    args = _make_args(export_kd_teacher_load=str(tmp_path))
    model = mc.modelopt_model_config_from_args(args)
    cfg = pretrain_cfg_container_from_args(args, model)
    cfg.distillation = mc.distillation_config_from_args(args, model)
    setattr(cfg.model.transformer, field, value)
    with pytest.raises(ValueError, match="incompatible"):
        cfg.validate()


def test_direct_training_config_rejects_manual_gc(tmp_path):
    args = _make_args(export_kd_teacher_load=str(tmp_path))
    model = mc.modelopt_model_config_from_args(args)
    cfg = pretrain_cfg_container_from_args(args, model)
    cfg.distillation = mc.distillation_config_from_args(args, model)
    cfg.train.manual_gc = True
    with pytest.raises(ValueError, match="manual_gc"):
        cfg.validate()


def test_build_models_passes_run_distillation_to_builder(tmp_path):
    args = _make_args(export_kd_teacher_load=str(tmp_path))
    model = mc.modelopt_model_config_from_args(args)
    cfg = pretrain_cfg_container_from_args(args, model)
    cfg.distillation = mc.distillation_config_from_args(args, model)
    with patch.object(GPTModelBuilder, "build_distributed_models") as build:
        cfg.build_models(Mock())
    assert build.call_args.kwargs["distillation"] is cfg.distillation
    with pytest.raises(ValueError, match="container.distillation"):
        cfg.build_models(Mock(), distillation=None)


def test_build_policy_conflict_is_rejected():
    cfg = mc.modelopt_inference_config_from_args(_make_args(), device="cpu")
    with pytest.raises(ValueError, match="conflicts"):
        cfg.build_models(Mock(), device="cuda")


def test_serialized_builder_is_ignored():
    cfg = mc.modelopt_inference_config_from_args(_make_args())
    data = cfg.to_dict()
    data["model"]["_builder_"] = "os.system"
    restored = InferenceConfigContainer.from_dict(data)
    assert restored.model.get_builder_cls() is GPTModelBuilder


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
        load="/configured",
        ckpt_format="torch_dist",
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
        patch("megatron.training.global_vars.is_run_config_initialized", return_value=True),
        patch.object(utils.ProcessGroupCollection, "use_mpu_process_groups"),
        patch.object(utils, "get_model_builder") as legacy_builder,
        patch.object(
            utils,
            "load_checkpoint",
            side_effect=lambda **kw: loaded.append(kw["ddp_model"]),
        ),
    ):
        assert utils.get_model_for_inference() is stage
    legacy_builder.assert_not_called()
    cfg.build_models.assert_called_once()
    assert loaded == [[stage]]

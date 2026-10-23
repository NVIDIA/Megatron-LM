# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest

from megatron.post_training import model_config as mc
from megatron.post_training import model_hooks
from megatron.post_training import utils as modelopt_utils
from megatron.training.argument_utils import (
    inference_cfg_container_from_args,
    pretrain_cfg_container_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.config.container import InferenceConfigContainer
from megatron.training.config.instantiate_utils import InstantiationMode
from megatron.training.models.gpt import GPTModelBuilder
from megatron.training.models.hybrid import HybridModelBuilder
from tests.unit_tests.post_training.conftest import make_modelopt_test_args

_NON_MODELOPT_OPTION_OVERRIDES = (
    {"export_te_mcore_model": True},
    {"export_default_te_spec": True},
    {"export_force_local_attention": True},
    {"export_real_quant_cfg": "fp8_real_quant"},
    {"export_kd_teacher_model_config": "/teacher.yaml"},
    {"export_kd_teacher_ckpt_format": "torch_dist"},
    {"export_kd_cfg": "/kd.yaml"},
    {"qad_train_target": "base"},
    {"freeze_base_for_mtp": True},
    {"lsq_scale_lr": 1e-4},
    {"qk_l2_norm": True},
    {"enable_gpt_oss": True},
    {"export_offline_model": True},
)


@pytest.mark.parametrize(
    "overrides,load,checkpoint_has_state,expected,assert_args_unchanged",
    [
        ({"modelopt_enabled": True}, None, False, True, True),
        ({"export_kd_teacher_load": "/teacher"}, None, False, True, True),
        ({}, "/modelopt-checkpoint", True, True, False),
        ({}, None, False, False, False),
        *[(option, None, False, False, False) for option in _NON_MODELOPT_OPTION_OVERRIDES],
    ],
)
def test_should_use_modelopt_config(
    overrides, load, checkpoint_has_state, expected, assert_args_unchanged
):
    args = make_modelopt_test_args(**overrides)
    args.load = load
    original_args = vars(args).copy()
    patch_ctx = (
        patch("megatron.post_training.checkpointing.has_modelopt_state", return_value=True)
        if checkpoint_has_state
        else nullcontext()
    )
    with patch_ctx as has_state:
        result = modelopt_utils.should_use_modelopt_config(args)
    assert result is expected
    if checkpoint_has_state:
        has_state.assert_called_once_with(load, args.ckpt_format)
    if assert_args_unchanged:
        assert vars(args) == original_args


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("kd", [False, True])
def test_whole_run_yaml_roundtrip(tmp_path, hybrid, training, kd):
    args = make_modelopt_test_args(
        hybrid_layer_pattern="M*" if hybrid else None,
        export_kd_teacher_load=str(tmp_path) if kd else None,
    )
    args.load = "/student"
    # The transform config must preserve every setting needed to rebuild the specs.
    model = mc.modelopt_model_config_from_args(args)
    if training:
        cfg = pretrain_cfg_container_from_args(args, model)
        cfg.distillation = mc.distillation_config_from_args(args, model)
    else:
        cfg = mc.modelopt_inference_config_from_args(args)
    path = tmp_path / "run.yaml"
    cfg.to_yaml(str(path))
    assert "modelopt_gpt_layer_spec" not in path.read_text()
    assert "modelopt_mtp_layer_spec" not in path.read_text()
    restored = type(cfg).from_yaml(str(path), mode=InstantiationMode.STRICT)
    restored.validate()
    assert restored.model.get_builder_cls() is (HybridModelBuilder if hybrid else GPTModelBuilder)
    assert restored.checkpoint.load == "/student"
    assert restored.model.transformations == model.transformations
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
    cfg = mc.modelopt_inference_config_from_args(make_modelopt_test_args())
    model = cfg.model
    model.vocab_size = None
    resolve_tokenizer_vocab_size(cfg, 32768)
    with (
        patch("megatron.core.process_groups_config.ProcessGroupCollection.use_mpu_process_groups"),
        patch.object(GPTModelBuilder, "build_distributed_models") as build,
    ):
        mc.build_inference_models(cfg)
    assert cfg.model is model
    assert model.vocab_size == 32768
    assert model.should_pad_vocab is False
    build.assert_called_once()


@pytest.mark.parametrize(
    "field,value", [("tp_comm_overlap", True), ("cross_entropy_fusion_impl", "te")]
)
def test_direct_config_validates_kd_incompatibilities(tmp_path, field, value):
    args = make_modelopt_test_args(export_kd_teacher_load=str(tmp_path))
    model = mc.modelopt_model_config_from_args(args)
    cfg = pretrain_cfg_container_from_args(args, model)
    cfg.distillation = mc.distillation_config_from_args(args, model)
    setattr(cfg.model.transformer, field, value)
    with pytest.raises(ValueError, match="incompatible"):
        cfg.validate()


def test_direct_training_config_rejects_manual_gc(tmp_path):
    args = make_modelopt_test_args(export_kd_teacher_load=str(tmp_path))
    model = mc.modelopt_model_config_from_args(args)
    cfg = pretrain_cfg_container_from_args(args, model)
    cfg.distillation = mc.distillation_config_from_args(args, model)
    cfg.train.manual_gc = True
    with pytest.raises(ValueError, match="manual_gc"):
        cfg.validate()


def test_serialized_builder_is_ignored():
    cfg = mc.modelopt_inference_config_from_args(make_modelopt_test_args())
    data = cfg.to_dict()
    data["model"]["_builder_"] = "os.system"
    restored = InferenceConfigContainer.from_dict(data)
    assert restored.model.get_builder_cls() is GPTModelBuilder


def test_legacy_inference_helper_reuses_declared_model():
    args = make_modelopt_test_args()
    cfg = mc.modelopt_inference_config_from_args(args)
    model = cfg.model
    with (
        patch("megatron.training.global_vars.get_run_config", return_value=cfg),
        patch.object(mc, "build_inference_models") as build,
        patch.object(mc, "modelopt_inference_config_from_args") as translate,
    ):
        mc.build_modelopt_models(args)
    translate.assert_not_called()
    assert cfg.model is model
    build.assert_called_once()


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
    loaded = []
    with (
        patch.object(utils, "get_args", return_value=args),
        patch("megatron.training.global_vars.get_run_config", return_value=cfg),
        patch("megatron.training.global_vars.is_run_config_initialized", return_value=True),
        patch.object(utils, "build_inference_models", return_value=[stage]) as build,
        patch.object(utils, "get_model_builder") as legacy_builder,
        patch.object(
            utils, "load_checkpoint", side_effect=lambda **kw: loaded.append(kw["ddp_model"])
        ),
    ):
        assert utils.get_model_for_inference() is stage
    legacy_builder.assert_not_called()
    build.assert_called_once_with(cfg)
    assert loaded == [[stage]]


@pytest.mark.parametrize(
    "hybrid,builder_cls", [(False, GPTModelBuilder), (True, HybridModelBuilder)]
)
def test_checkpoint_detected_inference_defaults_unregistered_modelopt_options(hybrid, builder_cls):
    """Checkpoint detection works for callers that did not register ModelOpt CLI flags."""
    from megatron.inference import utils

    args = make_modelopt_test_args(hybrid_layer_pattern="M*" if hybrid else None)
    args.load = "/modelopt-checkpoint"
    args.inference_ckpt_non_strict = False
    modelopt_only_options = (
        "export_offline_model",
        "freeze_base_for_mtp",
        "export_default_te_spec",
        "export_force_local_attention",
        "export_real_quant_cfg",
    )
    for name in modelopt_only_options:
        delattr(args, name)
    cfg = inference_cfg_container_from_args(args, model_cfg=None, build_model_config=False)
    stage = Mock()

    with (
        patch.object(utils, "get_args", return_value=args),
        patch("megatron.training.global_vars.get_run_config", return_value=cfg),
        patch("megatron.training.global_vars.is_run_config_initialized", return_value=True),
        patch.object(utils, "should_use_modelopt_config", return_value=True),
        patch.object(mc, "build_inference_models", return_value=[stage]) as build,
        patch.object(utils, "load_checkpoint"),
    ):
        assert utils.get_model_for_inference() is stage

    assert cfg.model.get_builder_cls() is builder_cls
    assert cfg.model.transformations == mc.ModelTransformConfig()
    assert all(not hasattr(args, name) for name in modelopt_only_options)
    build.assert_called_once_with(cfg)


def test_inference_rejects_custom_pg_collection_for_checkpoint_detected_modelopt():
    from argparse import Namespace

    from megatron.inference import utils

    args = Namespace(load="/modelopt-ckpt", modelopt_enabled=False)
    with (
        patch.object(utils, "get_args", return_value=args),
        patch("megatron.training.global_vars.is_run_config_initialized", return_value=False),
        patch.object(utils, "should_use_modelopt_config", return_value=True),
        pytest.raises(ValueError, match="Custom inference process groups are not supported"),
    ):
        utils.get_model_for_inference(pg_collection=Mock())

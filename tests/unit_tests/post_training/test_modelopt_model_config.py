# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for building ModelOpt model configs/hooks that use the stock GPT/Hybrid builders."""

import functools
import sys
from argparse import Namespace
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.post_training import model_config as mc
from megatron.post_training import model_hooks
from megatron.post_training.arguments import add_modelopt_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.models.base import ModelConfig
from megatron.training.models.distillation import DistillationConfig
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig


def _make_args(extra_argv=(), **overrides):
    sys.argv = ['test_modelopt_model_config.py', *extra_argv]
    args = parse_args(extra_args_provider=add_modelopt_args)
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 8
    args.micro_batch_size = 1
    args.seq_length = 128
    args.max_position_embeddings = 128
    args.padded_vocab_size = 32000
    args.position_embedding_type = 'rope'
    args.apply_rope_fusion = False
    for name, value in overrides.items():
        setattr(args, name, value)
    validate_args(args)
    return args


class TestDistillationConfigFromArgs:
    def test_none_without_teacher(self):
        assert mc.distillation_config_from_args(_make_args()) is None

    def test_maps_kd_arguments(self):
        args = _make_args(
            export_kd_teacher_load="/ckpt/teacher",
            export_kd_teacher_model_config="/cfg/teacher.yaml",
            export_kd_teacher_ckpt_format="torch_dist",
            export_kd_cfg="/cfg/kd.yaml",
        )
        assert mc.distillation_config_from_args(args) == DistillationConfig(
            teacher_load="/ckpt/teacher",
            teacher_model_config="/cfg/teacher.yaml",
            teacher_ckpt_format="torch_dist",
            kd_cfg="/cfg/kd.yaml",
        )


class TestModelOptModelConfigFromArgs:
    def test_gpt_uses_stock_builder_and_modelopt_spec(self):
        args = _make_args(
            export_te_mcore_model=True,
            export_force_local_attention=True,
            export_real_quant_cfg="None",
        )

        config = mc.modelopt_model_config_from_args(args)

        assert isinstance(config, GPTModelConfig)
        assert config.get_builder_cls() is GPTModelBuilder
        spec = object()
        with patch.object(mc, "get_gpt_modelopt_spec", return_value=spec) as get_spec:
            assert config.transformer_layer_spec(config) is spec
        get_spec.assert_called_once_with(
            config=config.transformer,
            local_core_attention=True,
            remap_te_layernorm=True,
            real_quant_cfg="None",
            use_arbitrary_attention_mask=False,
        )

    def test_gpt_default_te_spec_honors_disable_moe_grouped_gemm(self):
        args = _make_args(export_default_te_spec=True)

        config = mc.modelopt_model_config_from_args(args, disable_moe_grouped_gemm=True)

        with patch.object(mc, "get_gpt_layer_with_transformer_engine_spec") as get_spec:
            config.transformer_layer_spec(config)
        assert get_spec.call_args.args[1] is False  # moe_grouped_gemm

    def test_hybrid_sets_stack_spec(self):
        args = _make_args(
            export_model_type="HybridModel", hybrid_layer_pattern="M*-", export_te_mcore_model=True
        )
        stack_spec = ModuleSpec(module=object)

        with patch.object(
            mc, "get_hybrid_stack_modelopt_spec", return_value=stack_spec
        ) as get_spec:
            config = mc.modelopt_model_config_from_args(args, disable_moe_grouped_gemm=True)

        assert isinstance(config, HybridModelConfig)
        assert config.get_builder_cls() is HybridModelBuilder
        assert config.hybrid_stack_spec is stack_spec
        get_spec.assert_called_once_with(
            remap_te_layernorm=True, use_default_te_spec=False, moe_grouped_gemm=False
        )

    def test_transformer_config_tweaks(self):
        args = _make_args(
            export_qk_l2_norm=True,
            export_moe_apply_probs_on_input=True,
            export_offline_model=True,
            enable_gpt_oss=True,
        )

        config = mc.modelopt_model_config_from_args(args)

        assert config.transformer.qk_l2_norm is True
        assert config.transformer.moe_apply_probs_on_input is True
        assert config.transformer.num_layers == 0
        assert config.transformer.original_num_layers == 2
        assert config.transformer.yarn_rotary_scaling_factor == 32.0

    def test_rejects_custom_spec(self):
        args = _make_args()
        args.spec = ["some.module", "some_spec"]
        with pytest.raises(ValueError, match="custom args.spec"):
            mc.modelopt_model_config_from_args(args)

    def test_rejects_freeze_alias_conflict(self):
        args = _make_args(freeze_base_for_mtp=True, qad_train_target="base")
        with pytest.raises(ValueError, match="conflicts with --qad-train-target"):
            mc.modelopt_model_config_from_args(args)

    @pytest.mark.parametrize("builder_cls", [GPTModelBuilder, HybridModelBuilder])
    def test_round_trip_reconstructs_setup_in_dependency_order(self, monkeypatch, builder_cls):
        events = []
        for name in (
            "restore_modelopt_state_hook",
            "qad_freeze_hook",
            "lsq_hook",
            "te_load_state_dict_hook",
            "kd_convert_hook",
        ):

            def factory(*args, _name=name, **kwargs):
                def hook(models):
                    events.append(_name)
                    return models

                return hook

            monkeypatch.setattr(model_hooks, name, factory)
        args = _make_args(
            freeze_base_for_mtp=True,
            lsq_scale_lr=1e-3,
            export_te_mcore_model=True,
            export_kd_teacher_load="/ckpt/teacher",
            export_model_type="HybridModel" if builder_cls is HybridModelBuilder else "GPTModel",
            hybrid_layer_pattern="M*" if builder_cls is HybridModelBuilder else None,
        )
        args.load = "/ckpt/student"
        # Use a serializable spec without requiring the optional hybrid spec backends.
        with patch.object(mc, "get_hybrid_stack_modelopt_spec", return_value=None):
            original = mc.modelopt_model_config_from_args(args)
        config = ModelConfig.from_dict(original.as_dict())
        assert config.pre_wrap_hooks == []
        assert config.transformations == original.transformations
        assert config.distillation.teacher_model.transformer.num_layers == 2
        assert config.distillation.teacher_model.transformations is None
        assert config.restore_modelopt_state is False

        def freeze(models):
            events.append("freeze")
            return models

        config.pre_wrap_hooks.append(freeze)
        builder = builder_cls(config)
        for _ in range(2):
            builder.pre_wrap_hook(Namespace(load="/ckpt/student", ckpt_format="torch_dist"))(
                [Mock()]
            )
        expected = [
            "restore_modelopt_state_hook",
            "qad_freeze_hook",
            "lsq_hook",
            "te_load_state_dict_hook",
            "freeze",
            "kd_convert_hook",
        ]
        assert events == expected * 2
        assert config.pre_wrap_hooks == [freeze]  # no duplicate registration across builds

    def test_no_runtime_hooks_stored_in_config(self):
        config = mc.modelopt_model_config_from_args(_make_args())
        assert config.pre_wrap_hooks == []
        assert config.transformations.qad_train_target is None
        assert config.distillation is None
        assert config.restore_modelopt_state is False

    def test_teacher_yaml_resolved_before_serialization(self, tmp_path):
        teacher_yaml = tmp_path / "teacher.yaml"
        teacher_yaml.write_text(yaml.safe_dump({"num_layers": 4}))
        args = _make_args(
            export_kd_teacher_load=str(tmp_path), export_kd_teacher_model_config=str(teacher_yaml)
        )
        config = mc.modelopt_model_config_from_args(args)
        teacher_yaml.unlink()
        restored = ModelConfig.from_dict(config.as_dict())
        assert restored.distillation.teacher_model.transformer.num_layers == 4
        with patch.object(GPTModelBuilder, "build_model", return_value=Mock()) as build:
            model_hooks._build_teacher(restored, Mock())
        build.assert_called_once()

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_post_training_execution_device_is_explicit(self, device):
        cfg = mc.modelopt_inference_config_from_args(
            _make_args(use_cpu_initialization=True), device=device
        )
        with patch.object(GPTModelBuilder, "build_distributed_models") as build:
            cfg.build_models(Mock())
        assert build.call_args.kwargs["device"] == device
        assert build.call_args.kwargs["checkpoint"] is cfg.checkpoint
        assert build.call_args.kwargs["wrap_with_ddp"] is False


class TestLoadTeacherModelConfig:
    def _args(self, **overrides):
        return Namespace(kv_channels=16, num_layers=4, hybrid_override_pattern=None, **overrides)

    def test_overlays_yaml_and_drops_kv_channels(self, tmp_path):
        teacher_yaml = tmp_path / "teacher.yaml"
        teacher_yaml.write_text(yaml.safe_dump({"num_layers": 2, "hybrid_override_pattern": "M*"}))
        distillation = DistillationConfig(
            teacher_load=str(tmp_path), teacher_model_config=str(teacher_yaml)
        )

        raw = mc._load_teacher_model_config(distillation, self._args(flextron=True))

        assert raw.num_layers == 2
        assert not hasattr(raw, "kv_channels")
        assert raw.flextron is False
        assert raw.hybrid_layer_pattern == "M*"

    def test_defaults_to_student_arch_without_yaml(self, tmp_path):
        distillation = DistillationConfig(teacher_load=str(tmp_path))

        raw = mc._load_teacher_model_config(distillation, self._args())

        assert raw.num_layers == 4
        assert raw.kv_channels == 16

    def test_missing_explicit_config_raises(self, tmp_path):
        distillation = DistillationConfig(
            teacher_load=str(tmp_path), teacher_model_config=str(tmp_path / "missing.yaml")
        )
        with pytest.raises(FileNotFoundError):
            mc._load_teacher_model_config(distillation, self._args())


class TestBuildTeacher:
    def test_uses_student_builder_with_teacher_architecture(self):
        student_spec = functools.partial(lambda model_config: model_config)
        student = GPTModelConfig(
            transformer=TransformerConfig(num_layers=4, hidden_size=128, num_attention_heads=1),
            vocab_size=32000,
            transformer_layer_spec=student_spec,
            distillation=DistillationConfig(teacher_load="/ckpt/teacher"),
            pre_wrap_hooks=[Mock()],
        )
        teacher_transformer = TransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=1
        )
        student.distillation.teacher_model = GPTModelConfig(
            transformer=teacher_transformer, vocab_size=None, transformer_layer_spec=student_spec
        )
        student_stage = Mock(pre_process=True, post_process=False, vp_stage=None)

        with patch.object(GPTModelBuilder, "build_model", autospec=True) as build_model:
            teacher = model_hooks._build_teacher(student, student_stage)

        builder = build_model.call_args.args[0]
        assert teacher is build_model.return_value
        teacher_config = builder._model_config
        assert teacher_config.transformer is teacher_transformer
        assert teacher_config.transformer_layer_spec is student_spec
        assert teacher_config.vocab_size == 32000
        assert teacher_config.distillation is None
        assert teacher_config.pre_wrap_hooks == []
        assert build_model.call_args.args[1] is student_stage.pg_collection
        assert build_model.call_args.kwargs == {
            "pre_process": True,
            "post_process": False,
            "vp_stage": None,
        }
        # The student config itself is untouched.
        assert student.transformer.num_layers == 4
        assert student.distillation is not None


class TestOnModelInitDevice:
    def test_passthrough_without_meta_device(self):
        hook = Mock()
        assert model_hooks.on_model_init_device(hook, False) is hook

    def test_hook_runs_under_meta_device(self):
        created = []

        def hook(models):
            created.append(torch.empty(1))
            return models

        wrapped = model_hooks.on_model_init_device(hook, True)
        assert wrapped(["model"]) == ["model"]

        assert created[0].device.type == "meta"
        assert torch.empty(1).device.type != "meta"


class TestKdConvertHook:
    def _student_config(self, **transformer_overrides):
        transformer = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_dtype=None,
        )
        for name, value in transformer_overrides.items():
            setattr(transformer, name, value)
        return GPTModelConfig(
            transformer=transformer,
            vocab_size=32000,
            distillation=DistillationConfig(teacher_load="/ckpt/teacher"),
        )

    def _run(self, student_config, teacher_transformer, models):
        student_config.distillation.teacher_model = GPTModelConfig(
            transformer=teacher_transformer, vocab_size=32000
        )
        hook = model_hooks.kd_convert_hook(student_config, te_mcore_model=False)
        return hook(models)

    @pytest.mark.parametrize(
        "attr", ["tensor_model_parallel_size", "pipeline_model_parallel_size", "pipeline_dtype"]
    )
    def test_rejects_mismatched_parallelism_or_dtype(self, attr):
        teacher = Mock(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_dtype=None,
        )
        setattr(teacher, attr, 2)
        student = self._student_config()
        student.distillation.teacher_model = GPTModelConfig(transformer=teacher, vocab_size=32000)
        with pytest.raises(ValueError, match=f"same {attr}"):
            student.distillation.validate(student)

    def test_rejects_multiple_model_chunks(self):
        with pytest.raises(ValueError, match="virtual pipeline"):
            self._run(self._student_config(), Mock(), [Mock(), Mock()])


class TestFreezeHookOrdering:
    def test_freeze_hook_appended_without_kd(self):
        from megatron.training.training import (
            _add_model_freeze_pre_wrap_hook,
            _freeze_all_model_chunks,
        )

        model_config = Mock(pre_wrap_hooks=[])

        _add_model_freeze_pre_wrap_hook(
            model_config, freeze_all_layers=True, freeze_base_model_for_mtp=False
        )

        assert model_config.pre_wrap_hooks == [_freeze_all_model_chunks]


class TestKdConvertHookDoesNotLoadTeacherWeights:
    def test_teacher_checkpoint_is_not_loaded_before_wrapping(self):
        """Teacher weights are loaded after mixed-precision wrapping (load_kd_teacher_checkpoint)."""
        student_config = GPTModelConfig(
            vocab_size=32000,
            transformer=TransformerConfig(
                num_layers=2,
                hidden_size=128,
                num_attention_heads=1,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                pipeline_dtype=None,
            ),
            distillation=DistillationConfig(teacher_load="/ckpt/teacher"),
        )
        teacher_transformer = student_config.transformer
        student_config.distillation.teacher_model = GPTModelConfig(
            transformer=teacher_transformer, vocab_size=32000
        )
        hook = model_hooks.kd_convert_hook(student_config, te_mcore_model=False)
        student_stage = Mock()

        with (
            patch.object(model_hooks, "_build_teacher") as build_teacher,
            patch.object(model_hooks, "mtd") as mtd,
            patch.object(model_hooks, "mtd_mcore"),
            patch("megatron.post_training.checkpointing.load_checkpoint") as load_checkpoint,
            patch(
                "megatron.post_training.checkpointing.load_kd_teacher_checkpoint"
            ) as load_teacher,
        ):
            converted = hook([student_stage])

        build_teacher.assert_called_once()
        assert converted == [mtd.convert.return_value]
        load_checkpoint.assert_not_called()
        load_teacher.assert_not_called()


class TestModelOptMtpSpec:
    @pytest.mark.parametrize("quant_cfg", ["None", "fp8_real_quant", "fp8_blockwise_real_quant"])
    def test_real_quant_decoder_keeps_standard_mtp_spec(self, quant_cfg):
        from megatron.core.transformer.transformer_layer import TransformerLayer
        from megatron.training.models.gpt import mtp_block_spec
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(1, 1)
        try:
            config = mc.modelopt_model_config_from_args(
                _make_args(mtp_num_layers=1, export_real_quant_cfg=quant_cfg)
            )
            decoder_spec = config.transformer_layer_spec(config)
            mtp_spec = mtp_block_spec(config, decoder_spec, pp_rank=0)
            layer = mtp_spec.layer_specs[0].submodules.mtp_model_layer
            expected = mc.modelopt_mtp_layer_spec(config)
            assert layer.module is TransformerLayer
            assert layer.submodules == expected.submodules
            if quant_cfg != "None":
                assert decoder_spec.layer_specs[0].module is not TransformerLayer
        finally:
            Utils.destroy_model_parallel()


class TestTeacherCheckpointConfig:
    @pytest.mark.parametrize("fail", [False, True])
    def test_uses_config_and_restores_runtime_args(self, fail):
        from megatron.post_training import checkpointing

        args = Namespace(finetune=False, ckpt_format="torch", export_kd_teacher_load="/stale")
        teacher = Mock()
        student = Mock(teacher_model=teacher)
        config = DistillationConfig(teacher_load="/configured", teacher_ckpt_format="torch_dist")

        def load(models, optimizer, scheduler, **kwargs):
            assert models == [teacher]
            assert args.export_kd_teacher_load == "/configured"
            assert args.ckpt_format == "torch_dist"
            assert args.finetune is True
            if fail:
                raise RuntimeError("load failed")

        with (
            patch.object(checkpointing, "get_args", return_value=args),
            patch.object(checkpointing, "unwrap_model", return_value=student),
            patch.object(checkpointing, "load_checkpoint", side_effect=load) as load_checkpoint,
        ):
            if fail:
                with pytest.raises(RuntimeError, match="load failed"):
                    checkpointing.load_kd_teacher_checkpoint([student], config)
            else:
                checkpointing.load_kd_teacher_checkpoint([student], config)
        load_checkpoint.assert_called_once()
        assert vars(args) == dict(
            finetune=False, ckpt_format="torch", export_kd_teacher_load="/stale"
        )

    def test_absent_legacy_teacher_path_is_not_left_on_args(self):
        from megatron.post_training import checkpointing

        args = Namespace(finetune=False, ckpt_format="torch")
        with (
            patch.object(checkpointing, "get_args", return_value=args),
            patch.object(checkpointing, "unwrap_model"),
            patch.object(checkpointing, "load_checkpoint"),
        ):
            checkpointing.load_kd_teacher_checkpoint(
                [Mock()], DistillationConfig(teacher_load="/teacher")
            )
        assert not hasattr(args, "export_kd_teacher_load")

    def test_modelopt_state_uses_explicit_path_and_format(self):
        from megatron.post_training import checkpointing

        args = Namespace(load="/student", ckpt_format="torch_dist")
        with (
            patch.object(checkpointing, "get_args", return_value=args),
            patch.object(
                checkpointing, "_load_base_checkpoint", return_value=({}, None, None)
            ) as load,
        ):
            checkpointing.load_modelopt_state(Mock(), "/explicit", ckpt_format="torch")
        load.assert_called_once_with("/explicit", rank0=False)


class TestConfigurationDrivenKDLoss:
    @pytest.mark.parametrize("training", [True, False])
    def test_kd_loss_without_legacy_cli_teacher_flag(self, training):
        from modelopt.torch.distill import DistillationModel

        from megatron.post_training.loss_func import loss_func

        model = Mock(spec=DistillationModel)
        model.training = training
        model.compute_kd_loss.return_value = {
            "kd_loss": torch.tensor(7.0),
            "logits_loss": torch.tensor(3.0),
            "intermediate_loss": torch.tensor(4.0),
        }
        with (
            patch("megatron.post_training.loss_func.unwrap_model", return_value=model),
            patch("megatron.post_training.loss_func.get_args", return_value=Namespace()),
        ):
            loss, tokens, report = loss_func(torch.ones(2), torch.tensor([1.0, 2.0]), model)
        assert loss.item() == (7.0 if training else 3.0)
        assert tokens.item() == 2
        assert report["total loss"][0].item() == 7.0
        model.compute_kd_loss.assert_called_once()

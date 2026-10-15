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

    @pytest.mark.parametrize("flag", ["manual_gc", "tp_comm_overlap"])
    def test_rejects_incompatible_flags(self, flag):
        args = _make_args(export_kd_teacher_load="/ckpt/teacher")
        setattr(args, flag, True)
        with pytest.raises(AssertionError, match="incompatible"):
            mc.distillation_config_from_args(args)


class TestModelOptModelConfigFromArgs:
    def test_gpt_uses_stock_builder_and_modelopt_spec(self):
        args = _make_args(
            export_te_mcore_model=True, export_force_local_attention=True, export_real_quant_cfg="None"
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

        with patch.object(mc, "get_hybrid_stack_modelopt_spec", return_value=stack_spec) as get_spec:
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

    def test_pre_wrap_hooks_registered_in_dependency_order(self, monkeypatch):
        for name in (
            "restore_modelopt_state_hook",
            "qad_freeze_hook",
            "lsq_hook",
            "te_load_state_dict_hook",
            "kd_convert_hook",
        ):
            monkeypatch.setattr(
                model_hooks, name, lambda *a, _name=name, **k: (_name, a), raising=True
            )
        args = _make_args(
            freeze_base_for_mtp=True,
            lsq_scale_lr=1e-3,
            export_te_mcore_model=True,
            export_kd_teacher_load="/ckpt/teacher",
        )
        args.load = "/ckpt/student"

        config = mc.modelopt_model_config_from_args(args)

        assert [hook[0] for hook in config.pre_wrap_hooks] == [
            "restore_modelopt_state_hook",
            "qad_freeze_hook",
            "lsq_hook",
            "te_load_state_dict_hook",
            "kd_convert_hook",
        ]
        assert config.pre_wrap_hooks[0][1] == ("/ckpt/student",)
        assert config.pre_wrap_hooks[1][1] == ("mtp",)
        # The restore hook owns restoration; the flag would make a runner restore again after KD.
        assert config.restore_modelopt_state is False
        assert config.distillation.teacher_load == "/ckpt/teacher"

    def test_no_optional_hooks_by_default(self):
        config = mc.modelopt_model_config_from_args(_make_args())
        # Only the (no-op without LSQ params) LSQ hook is always registered.
        assert len(config.pre_wrap_hooks) == 1
        assert config.distillation is None
        assert config.restore_modelopt_state is False


class TestLoadTeacherModelConfig:
    def _args(self, **overrides):
        return Namespace(kv_channels=16, num_layers=4, hybrid_override_pattern=None, **overrides)

    def test_overlays_yaml_and_drops_kv_channels(self, tmp_path):
        teacher_yaml = tmp_path / "teacher.yaml"
        teacher_yaml.write_text(yaml.safe_dump({"num_layers": 2, "hybrid_override_pattern": "M*"}))
        distillation = DistillationConfig(
            teacher_load=str(tmp_path), teacher_model_config=str(teacher_yaml)
        )

        raw = model_hooks._load_teacher_model_config(distillation, self._args(flextron=True))

        assert raw.num_layers == 2
        assert not hasattr(raw, "kv_channels")
        assert raw.flextron is False
        assert raw.hybrid_layer_pattern == "M*"

    def test_defaults_to_student_arch_without_yaml(self, tmp_path):
        distillation = DistillationConfig(teacher_load=str(tmp_path))

        raw = model_hooks._load_teacher_model_config(distillation, self._args())

        assert raw.num_layers == 4
        assert raw.kv_channels == 16

    def test_missing_explicit_config_raises(self, tmp_path):
        distillation = DistillationConfig(
            teacher_load=str(tmp_path), teacher_model_config=str(tmp_path / "missing.yaml")
        )
        with pytest.raises(FileNotFoundError):
            model_hooks._load_teacher_model_config(distillation, self._args())


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
        teacher_transformer = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=1)
        student_stage = Mock(pre_process=True, post_process=False, vp_stage=None)

        with patch.object(GPTModelBuilder, "build_model", autospec=True) as build_model:
            teacher = model_hooks._build_teacher(
                student, Namespace(), teacher_transformer, student_stage
            )

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

    def test_wrapper_keeps_kd_marker(self):
        def hook(models):
            return models

        hook.modelopt_kd_convert = True
        wrapped = model_hooks.on_model_init_device(hook, True)
        assert wrapped.modelopt_kd_convert is True


class TestKdConvertHook:
    def _student_config(self, **transformer_overrides):
        transformer = Mock(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_dtype=None,
        )
        for name, value in transformer_overrides.items():
            setattr(transformer, name, value)
        return Mock(
            transformer=transformer, distillation=DistillationConfig(teacher_load="/ckpt/teacher")
        )

    def _run(self, student_config, teacher_transformer, models):
        hook = model_hooks.kd_convert_hook(student_config, Namespace(), te_mcore_model=False)
        with patch.object(model_hooks, "_load_teacher_model_config"), patch.object(
            model_hooks, "core_transformer_config_from_args", return_value=teacher_transformer
        ):
            return hook(models)

    def test_marked_for_ordering(self):
        hook = model_hooks.kd_convert_hook(self._student_config(), Namespace(), False)
        assert hook.modelopt_kd_convert is True

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
        with pytest.raises(ValueError, match=f"same {attr}"):
            self._run(self._student_config(), teacher, [Mock()])

    def test_rejects_multiple_model_chunks(self):
        with pytest.raises(ValueError, match="virtual pipeline"):
            self._run(self._student_config(), Mock(), [Mock(), Mock()])


class TestFreezeHookOrdering:
    def test_freeze_hook_inserted_before_kd_conversion(self):
        from megatron.training.training import (
            _add_model_freeze_pre_wrap_hook,
            _freeze_base_model_for_mtp,
        )

        def restore(models):
            return models

        def kd(models):
            return models

        kd.modelopt_kd_convert = True
        model_config = Mock(pre_wrap_hooks=[restore, kd])

        _add_model_freeze_pre_wrap_hook(
            model_config, freeze_all_layers=False, freeze_base_model_for_mtp=True
        )

        assert model_config.pre_wrap_hooks == [restore, _freeze_base_model_for_mtp, kd]

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
        student_config = Mock(
            transformer=Mock(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                pipeline_dtype=None,
            ),
            distillation=DistillationConfig(teacher_load="/ckpt/teacher"),
        )
        teacher_transformer = student_config.transformer
        hook = model_hooks.kd_convert_hook(student_config, Namespace(), te_mcore_model=False)
        student_stage = Mock()

        with patch.object(model_hooks, "_load_teacher_model_config"), patch.object(
            model_hooks, "core_transformer_config_from_args", return_value=teacher_transformer
        ), patch.object(model_hooks, "_build_teacher") as build_teacher, patch.object(
            model_hooks, "mtd"
        ) as mtd, patch.object(model_hooks, "mtd_mcore"), patch(
            "megatron.post_training.checkpointing.load_checkpoint"
        ) as load_checkpoint, patch(
            "megatron.post_training.checkpointing.load_kd_teacher_checkpoint"
        ) as load_teacher:
            converted = hook([student_stage])

        build_teacher.assert_called_once()
        assert converted == [mtd.convert.return_value]
        load_checkpoint.assert_not_called()
        load_teacher.assert_not_called()

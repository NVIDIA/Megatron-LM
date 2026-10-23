# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for building ModelOpt model configs/hooks that use the stock GPT/Hybrid builders."""

import functools
import warnings
from argparse import Namespace
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.post_training import model_config as mc
from megatron.post_training import model_hooks
from megatron.training.argument_utils import gpt_config_from_args, hybrid_config_from_args
from megatron.training.models.base import ModelConfig
from megatron.training.models.distillation import DistillationConfig
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig, default_layer_spec
from megatron.training.models.hybrid import (
    HybridModelBuilder,
    HybridModelConfig,
    resolve_hybrid_stack_spec,
)
from megatron.training.models.transforms import ModelTransformConfig
from tests.unit_tests.post_training.conftest import make_modelopt_test_args


class TestDistillationConfigFromArgs:
    def test_none_without_teacher(self):
        args = make_modelopt_test_args()
        student = mc.modelopt_model_config_from_args(args)
        assert mc.distillation_config_from_args(args, student) is None

    def test_maps_kd_arguments_and_reads_loss_yaml(self, tmp_path):
        teacher_yaml = tmp_path / "teacher.yaml"
        teacher_yaml.write_text(
            yaml.safe_dump(
                {
                    "num_layers": 4,
                    "untie_embeddings_and_output_weights": True,
                    "rotary_base": 500000,
                }
            )
        )
        kd_yaml = tmp_path / "kd.yaml"
        kd_yaml.write_text(
            yaml.safe_dump(
                {"logit_kl_temperature": 2.0, "intermediate_layer_pairs": [["a", "b", "mse"]]}
            )
        )
        args = make_modelopt_test_args(
            export_kd_teacher_load=str(tmp_path),
            export_kd_teacher_model_config=str(teacher_yaml),
            export_kd_teacher_ckpt_format="torch_dist",
            export_kd_cfg=str(kd_yaml),
        )
        student = mc.modelopt_model_config_from_args(args)

        distillation = mc.distillation_config_from_args(args, student)

        assert distillation.teacher_load == str(tmp_path)
        assert distillation.teacher_model_config == str(teacher_yaml)
        assert distillation.teacher_ckpt_format == "torch_dist"
        assert distillation.kd_cfg is None
        assert distillation.logit_kl_temperature == 2.0
        assert distillation.intermediate_layer_pairs == [("a", "b", "mse")]
        assert distillation.teacher_model.transformer.num_layers == 4
        assert distillation.teacher_model.share_embeddings_and_output_weights is False
        assert distillation.teacher_model.rotary_base == 500000
        assert student.share_embeddings_and_output_weights is True
        assert distillation.teacher_model.transformations is not student.transformations
        assert distillation.teacher_model.transformations.qad_train_target is None
        assert distillation.teacher_model.transformations.lsq_scale_lr is None
        assert (
            distillation.teacher_model.transformations.real_quant_cfg
            == student.transformations.real_quant_cfg
        )
        assert student.transformer.num_layers == 2  # the student config is untouched

    def test_rejects_unknown_loss_yaml_keys(self, tmp_path):
        kd_yaml = tmp_path / "kd.yaml"
        kd_yaml.write_text(yaml.safe_dump({"criterion": "x"}))
        args = make_modelopt_test_args(
            export_kd_teacher_load=str(tmp_path), export_kd_cfg=str(kd_yaml)
        )
        with pytest.raises(ValueError, match="Unknown distillation settings"):
            mc.distillation_config_from_args(args, mc.modelopt_model_config_from_args(args))


class TestDistillationConfig:
    def test_deprecated_kd_cfg_is_read_into_loss_fields(self, tmp_path):
        kd_yaml = tmp_path / "kd.yaml"
        kd_yaml.write_text(yaml.safe_dump({"skip_lm_loss": False, "kd_loss_scale": 0.5}))
        with pytest.warns(DeprecationWarning, match="kd_cfg is deprecated"):
            config = DistillationConfig(teacher_load="/t", kd_cfg=str(kd_yaml))
        assert config.kd_cfg is None
        assert (config.skip_lm_loss, config.kd_loss_scale) == (False, 0.5)

    def test_deprecated_kd_cfg_conflicting_with_field_raises(self, tmp_path):
        kd_yaml = tmp_path / "kd.yaml"
        kd_yaml.write_text(yaml.safe_dump({"kd_loss_scale": 0.5}))
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="kd_loss_scale"):
            DistillationConfig(teacher_load="/t", kd_cfg=str(kd_yaml), kd_loss_scale=2.0)

    def test_loss_kwargs_only_contains_non_defaults(self):
        assert DistillationConfig(teacher_load="/t").loss_kwargs() == {}
        config = DistillationConfig(
            teacher_load="/t", logit_kl_topk=64, logit_layers=["out", "out"]
        )
        assert config.logit_layers == ("out", "out")
        assert config.loss_kwargs() == {"logit_kl_topk": 64, "logit_layers": ("out", "out")}

    def test_loss_kwargs_are_accepted_by_modelopt(self):
        import modelopt.torch.distill.plugins.megatron as mtd_mcore

        config = DistillationConfig(
            teacher_load="/t",
            intermediate_layer_pairs=[("decoder.layers.0", "decoder.layers.0", "cosine")],
            logit_kl_temperature=2.0,
            logit_kl_topk=8,
            skip_lm_loss=False,
            kd_loss_scale=0.5,
        )
        modelopt_config = mtd_mcore.DistillationConfig(**config.loss_kwargs())
        assert modelopt_config.logit_kl_topk == 8
        assert modelopt_config.kd_loss_scale == 0.5

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"logit_layers": ("a",)}, "logit_layers"),
            ({"intermediate_layer_pairs": [("a",)]}, "intermediate_layer_pairs"),
            ({"kd_loss_scale": 0.0}, "positive"),
        ],
    )
    def test_validate_rejects_invalid_loss(self, overrides, match):
        config = DistillationConfig(teacher_load="/t", teacher_model=Mock(), **overrides)
        with pytest.raises(ValueError, match=match):
            config.validate()

    def test_validate_requires_resolved_teacher(self):
        with pytest.raises(ValueError, match="resolved teacher_model"):
            DistillationConfig(teacher_load="/t").validate()

    @staticmethod
    def _student_config(**transformer_overrides):
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
        return GPTModelConfig(transformer=transformer, vocab_size=32000)

    @staticmethod
    def _distillation_with_teacher(**teacher_overrides):
        teacher = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_dtype=None,
        )
        for name, value in teacher_overrides.items():
            setattr(teacher, name, value)
        return DistillationConfig(
            teacher_load="/ckpt/teacher",
            teacher_model=GPTModelConfig(transformer=teacher, vocab_size=32000),
        )

    @pytest.mark.parametrize(
        "attr", ["tensor_model_parallel_size", "pipeline_model_parallel_size", "pipeline_dtype"]
    )
    def test_validate_student_rejects_mismatched_parallelism_or_dtype(self, attr):
        distillation = self._distillation_with_teacher(**{attr: 2})
        with pytest.raises(ValueError, match=f"same {attr}"):
            distillation.validate_student(self._student_config())


class TestModelOptModelConfigFromArgs:
    def test_gpt_uses_stock_builder_and_modelopt_spec(self):
        args = make_modelopt_test_args(
            export_force_local_attention=True, export_real_quant_cfg="None"
        )

        config = mc.modelopt_model_config_from_args(args)

        assert isinstance(config, GPTModelConfig)
        assert config.get_builder_cls() is GPTModelBuilder
        assert config.transformer_layer_spec is None
        assert config.mtp_layer_spec is None
        spec = object()
        with patch(
            "megatron.training.models.gpt.get_gpt_modelopt_spec", return_value=spec
        ) as get_spec:
            assert default_layer_spec(config, vp_stage=None) is spec
        get_spec.assert_called_once_with(
            config=config.transformer,
            local_core_attention=True,
            remap_te_layernorm=True,
            real_quant_cfg="None",
            use_arbitrary_attention_mask=False,
        )

    def test_gpt_default_te_spec_honors_disable_moe_grouped_gemm(self):
        args = make_modelopt_test_args(export_default_te_spec=True)

        config = mc.modelopt_model_config_from_args(args, disable_moe_grouped_gemm=True)

        with patch(
            "megatron.training.models.gpt.get_gpt_layer_with_transformer_engine_spec"
        ) as get_spec:
            default_layer_spec(config, vp_stage=None)
        assert get_spec.call_args.args[1] is False  # moe_grouped_gemm

    def test_hybrid_sets_stack_spec(self):
        args = make_modelopt_test_args(hybrid_layer_pattern="M*-")
        stack_spec = ModuleSpec(module=object)

        config = mc.modelopt_model_config_from_args(args, disable_moe_grouped_gemm=True)
        with patch(
            "megatron.training.models.hybrid.get_hybrid_stack_modelopt_spec",
            return_value=stack_spec,
        ) as get_spec:
            assert resolve_hybrid_stack_spec(config) is stack_spec

        assert isinstance(config, HybridModelConfig)
        assert config.get_builder_cls() is HybridModelBuilder
        assert config.hybrid_stack_spec is None
        get_spec.assert_called_once_with(
            remap_te_layernorm=True, use_default_te_spec=False, moe_grouped_gemm=False
        )

    def test_transformer_config_tweaks(self):
        args = make_modelopt_test_args(export_offline_model=True)

        config = mc.modelopt_model_config_from_args(args)

        assert config.transformer.num_layers == 0
        assert config.transformer.original_num_layers == 2

    def test_architecture_options_come_from_core_arguments(self):
        # The --export- spellings are deprecated aliases of the core flags.
        args = make_modelopt_test_args(["--export-qk-l2-norm", "--enable-gpt-oss"])

        config = mc.modelopt_model_config_from_args(args)

        assert config.transformer.qk_l2_norm is True
        assert config.position_embedding_type == "yarn"
        assert config.transformer.yarn_rotary_scaling_factor == 32.0
        assert config.transformer.yarn_correction_range_round_to_int is False

    def test_rejects_custom_spec(self):
        args = make_modelopt_test_args()
        args.spec = ["some.module", "some_spec"]
        with pytest.raises(ValueError, match="custom args.spec"):
            mc.modelopt_model_config_from_args(args)

    def test_rejects_freeze_alias_conflict(self):
        args = make_modelopt_test_args(freeze_base_for_mtp=True, qad_train_target="base")
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
        args = make_modelopt_test_args(
            freeze_base_for_mtp=True,
            lsq_scale_lr=1e-3,
            export_kd_teacher_load="/ckpt/teacher",
            hybrid_layer_pattern="M*" if builder_cls is HybridModelBuilder else None,
        )
        args.load = "/ckpt/student"
        original = mc.modelopt_model_config_from_args(args)
        distillation = mc.distillation_config_from_args(args, original)
        config = ModelConfig.from_dict(original.as_dict())
        assert config.pre_wrap_hooks == []
        assert config.transformations == original.transformations
        assert config.restore_modelopt_state is False

        def freeze(models):
            events.append("freeze")
            return models

        config.pre_wrap_hooks.append(freeze)
        builder = builder_cls(config)
        for _ in range(2):
            builder.pre_wrap_hook(
                Namespace(load="/ckpt/student", ckpt_format="torch_dist"), distillation
            )([Mock()])
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
        config = mc.modelopt_model_config_from_args(make_modelopt_test_args())
        assert config.pre_wrap_hooks == []
        assert config.transformations.qad_train_target is None
        assert config.restore_modelopt_state is False

    def test_teacher_yaml_resolved_before_serialization(self, tmp_path):
        teacher_yaml = tmp_path / "teacher.yaml"
        teacher_yaml.write_text(yaml.safe_dump({"num_layers": 4}))
        args = make_modelopt_test_args(
            export_kd_teacher_load=str(tmp_path), export_kd_teacher_model_config=str(teacher_yaml)
        )
        config = mc.modelopt_model_config_from_args(args)
        distillation = mc.distillation_config_from_args(args, config)
        teacher_yaml.unlink()
        assert distillation.teacher_model.transformer.num_layers == 4
        with patch.object(GPTModelBuilder, "build_model", return_value=Mock()) as build:
            model_hooks._build_teacher(config, distillation, Mock())
        build.assert_called_once()

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_post_training_execution_device_is_explicit(self, device):
        cfg = mc.modelopt_inference_config_from_args(
            make_modelopt_test_args(use_cpu_initialization=True)
        )
        with (
            patch(
                "megatron.core.process_groups_config.ProcessGroupCollection.use_mpu_process_groups"
            ),
            patch.object(GPTModelBuilder, "build_distributed_models") as build,
        ):
            mc.build_inference_models(cfg, device=device)
        assert build.call_args.kwargs["device"] == device
        assert build.call_args.kwargs["checkpoint"] is cfg.checkpoint
        assert build.call_args.kwargs["wrap_with_ddp"] is False


class TestModelType:
    @pytest.mark.parametrize(
        "pattern,expected", [(None, GPTModelConfig), ("M*-", HybridModelConfig)]
    )
    def test_layer_pattern_selects_config(self, pattern, expected):
        args = make_modelopt_test_args(hybrid_layer_pattern=pattern)
        assert isinstance(mc.modelopt_model_config_from_args(args), expected)

    @pytest.mark.parametrize(
        "flag,pattern,expected",
        [
            ("GPTModel", None, GPTModelConfig),
            ("HybridModel", "M*-", HybridModelConfig),
            ("MambaModel", None, HybridModelConfig),  # hybrid without a layer pattern
        ],
    )
    def test_deprecated_flag_is_honored(self, flag, pattern, expected):
        args = make_modelopt_test_args(export_model_type=flag, hybrid_layer_pattern=pattern)
        with pytest.warns(DeprecationWarning, match="--export-model-type is deprecated"):
            assert isinstance(mc.modelopt_model_config_from_args(args), expected)

    def test_deprecated_flag_conflicting_with_config_raises(self):
        args = make_modelopt_test_args(export_model_type="GPTModel")
        config = hybrid_config_from_args(args)
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="conflicts"):
            mc.apply_modelopt_model_config(config, args)

    @pytest.mark.parametrize("flag", [None, "GPTModel"])
    def test_gpt_config_with_layer_pattern_raises(self, flag):
        args = make_modelopt_test_args(hybrid_layer_pattern="M*-", export_model_type=flag)
        config = gpt_config_from_args(args)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="pretrain_hybrid.py"):
                mc.apply_modelopt_model_config(config, args)

    def test_entry_point_config_is_updated_in_place(self):
        args = make_modelopt_test_args()
        config = hybrid_config_from_args(args)
        assert mc.apply_modelopt_model_config(config, args) is config
        assert config.transformations is not None

    @pytest.mark.parametrize("default_te_spec", [False, True])
    def test_te_load_hook_registered_unless_default_te_spec(self, monkeypatch, default_te_spec):
        te_hook = Mock(return_value=lambda models: models)
        monkeypatch.setattr(model_hooks, "te_load_state_dict_hook", te_hook)
        monkeypatch.setattr(model_hooks, "lsq_hook", lambda *_: lambda models: models)
        config = mc.modelopt_model_config_from_args(
            make_modelopt_test_args(export_default_te_spec=default_te_spec)
        )
        config.get_builder_cls()(config).pre_wrap_hook()([Mock()])
        assert te_hook.called is not default_te_spec


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
            pre_wrap_hooks=[Mock()],
        )
        teacher_transformer = TransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=1
        )
        teacher_transformations = ModelTransformConfig(force_local_attention=True)
        distillation = DistillationConfig(
            teacher_load="/ckpt/teacher",
            teacher_model=GPTModelConfig(
                transformer=teacher_transformer,
                vocab_size=None,
                transformer_layer_spec=student_spec,
                transformations=teacher_transformations,
            ),
        )
        student_stage = Mock(pre_process=True, post_process=False, vp_stage=None)

        with patch.object(GPTModelBuilder, "build_model", autospec=True) as build_model:
            teacher = model_hooks._build_teacher(student, distillation, student_stage)

        builder = build_model.call_args.args[0]
        assert teacher is build_model.return_value
        teacher_config = builder._model_config
        assert teacher_config.transformer is teacher_transformer
        assert teacher_config.transformer_layer_spec is student_spec
        assert teacher_config.vocab_size == 32000
        assert teacher_config.transformations is teacher_transformations
        assert teacher_config.pre_wrap_hooks == []
        assert build_model.call_args.args[1] is student_stage.pg_collection
        assert build_model.call_args.kwargs == {
            "pre_process": True,
            "post_process": False,
            "vp_stage": None,
        }
        # The student config itself is untouched.
        assert student.transformer.num_layers == 4
        assert len(student.pre_wrap_hooks) == 1


class TestOnModelInitDevice:
    @pytest.mark.parametrize("init_meta", [False, True])
    def test_wraps_hook_for_meta_device(self, init_meta):
        if not init_meta:
            hook = Mock()
            assert model_hooks.on_model_init_device(hook, False) is hook
            return

        created = []

        def hook(models):
            created.append(torch.empty(1))
            return models

        wrapped = model_hooks.on_model_init_device(hook, True)
        assert wrapped(["model"]) == ["model"]
        assert created[0].device.type == "meta"


class TestKdConvertHook:
    def test_rejects_multiple_model_chunks(self):
        student_config = GPTModelConfig(
            transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=1),
            vocab_size=32000,
        )
        distillation = DistillationConfig(
            teacher_load="/ckpt/teacher",
            teacher_model=GPTModelConfig(transformer=student_config.transformer, vocab_size=32000),
        )
        hook = model_hooks.kd_convert_hook(student_config, distillation, remap_te_layernorm=False)
        with pytest.raises(ValueError, match="virtual pipeline"):
            hook([Mock(), Mock()])


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
        )
        distillation = DistillationConfig(
            teacher_load="/ckpt/teacher",
            teacher_model=GPTModelConfig(transformer=student_config.transformer, vocab_size=32000),
        )
        hook = model_hooks.kd_convert_hook(student_config, distillation, remap_te_layernorm=False)
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
                make_modelopt_test_args(mtp_num_layers=1, export_real_quant_cfg=quant_cfg)
            )
            decoder_spec = default_layer_spec(config, vp_stage=None)
            from megatron.training.models.gpt import _modelopt_mtp_layer_spec

            expected = _modelopt_mtp_layer_spec(config)
            with patch(
                "megatron.training.models.gpt._modelopt_mtp_layer_spec", return_value=expected
            ) as select_mtp_spec:
                mtp_spec = mtp_block_spec(config, decoder_spec, pp_rank=0)
            select_mtp_spec.assert_called_once_with(config)
            layer = mtp_spec.layer_specs[0].submodules.mtp_model_layer
            assert layer is expected
            assert layer.module is TransformerLayer
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

    def test_student_checkpoint_format_when_teacher_format_unset(self):
        from megatron.post_training import checkpointing
        from megatron.training.config.training_config import CheckpointConfig

        args = Namespace(ckpt_format="torch", finetune=False)
        observed = []

        def load(*_args, **_kwargs):
            observed.append(args.ckpt_format)

        with (
            patch.object(checkpointing, "get_args", return_value=args),
            patch.object(checkpointing, "unwrap_model"),
            patch.object(checkpointing, "load_checkpoint", side_effect=load),
        ):
            checkpointing.load_kd_teacher_checkpoint(
                [Mock()],
                DistillationConfig(teacher_load="/teacher"),
                student_checkpoint=CheckpointConfig(ckpt_format="torch_dist"),
            )
        assert observed == ["torch_dist"]
        assert vars(args) == {"ckpt_format": "torch", "finetune": False}

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

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PEFT pre-wrap registration and model lifecycle tests."""

from dataclasses import dataclass, fields
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.training import training as training_mod
from megatron.training.config.container import PretrainConfigContainer
from megatron.training.models.base import ModelConfig
from megatron.training.models.dist_utils import unimodal_build_distributed_models
from megatron.training.peft import register_peft_pre_wrap_hook


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)


class _TinyPEFT:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model_chunks, *, training):
        assert training is True
        self.calls += 1
        for chunk in model_chunks:
            chunk.base.requires_grad_(False)
            chunk.adapter = torch.nn.Linear(2, 2, bias=False)
        return model_chunks


@dataclass
class _SerializableTinyPEFT:
    rank: int

    def __call__(self, model_chunks, *, training):
        return model_chunks


def _model_config():
    return SimpleNamespace(pre_wrap_hooks=[])


def _empty_container(peft):
    container = PretrainConfigContainer.__new__(PretrainConfigContainer)
    for field in fields(PretrainConfigContainer):
        setattr(container, field.name, None)
    container.peft = peft
    return container


def test_pretrain_config_has_optional_peft_field():
    peft_field = next(field for field in fields(PretrainConfigContainer) if field.name == "peft")
    assert peft_field.default is None


def test_config_export_rejects_nonserializable_peft_instance():
    with pytest.raises(TypeError, match="serializable"):
        _empty_container(_TinyPEFT()).to_dict()


def test_config_export_preserves_dataclass_peft_state():
    result = _empty_container(_SerializableTinyPEFT(rank=8)).to_dict()
    assert result["peft"]["rank"] == 8


def test_peft_hook_freezes_base_and_keeps_adapter_trainable():
    config = _model_config()
    peft = _TinyPEFT()
    register_peft_pre_wrap_hook(config, peft)

    model = _TinyModel()
    result = config.pre_wrap_hooks[0]([model])

    assert result == [model]
    assert peft.calls == 1
    assert not model.base.weight.requires_grad
    assert model.adapter.weight.requires_grad


def test_re_registration_replaces_only_the_owned_hook():
    user_hook = lambda chunks: chunks
    config = SimpleNamespace(pre_wrap_hooks=[user_hook])
    first, second = _TinyPEFT(), _TinyPEFT()

    register_peft_pre_wrap_hook(config, first)
    register_peft_pre_wrap_hook(config, second)
    assert len(config.pre_wrap_hooks) == 2
    assert config.pre_wrap_hooks[0] is user_hook

    config.pre_wrap_hooks[1]([_TinyModel()])
    assert first.calls == 0
    assert second.calls == 1

    register_peft_pre_wrap_hook(config, None)
    assert config.pre_wrap_hooks == [user_hook]


def test_re_registration_keeps_later_freeze_hook_after_peft():
    config = _model_config()
    register_peft_pre_wrap_hook(config, _TinyPEFT())
    training_mod._add_model_freeze_pre_wrap_hook(
        config, freeze_all_layers=True, freeze_base_model_for_mtp=False
    )
    register_peft_pre_wrap_hook(config, _TinyPEFT())

    chunks = [_TinyModel()]
    for hook in config.pre_wrap_hooks:
        chunks = hook(chunks)

    assert not chunks[0].base.weight.requires_grad
    assert not chunks[0].adapter.weight.requires_grad


def test_peft_registration_precedes_existing_freeze_hook():
    config = _model_config()
    training_mod._add_model_freeze_pre_wrap_hook(
        config, freeze_all_layers=True, freeze_base_model_for_mtp=False
    )
    register_peft_pre_wrap_hook(config, _TinyPEFT())
    training_mod._add_model_freeze_pre_wrap_hook(
        config, freeze_all_layers=True, freeze_base_model_for_mtp=False
    )

    chunks = [_TinyModel()]
    for hook in config.pre_wrap_hooks:
        chunks = hook(chunks)

    assert not chunks[0].base.weight.requires_grad
    assert not chunks[0].adapter.weight.requires_grad


def test_non_callable_peft_is_rejected_without_mutating_hooks():
    config = _model_config()
    with pytest.raises(TypeError, match="callable"):
        register_peft_pre_wrap_hook(config, object())
    assert config.pre_wrap_hooks == []


@pytest.mark.parametrize("bad_result", [None, _TinyModel(), [object()], []])
def test_invalid_peft_result_fails_before_wrapping(bad_result):
    config = _model_config()
    register_peft_pre_wrap_hook(config, lambda chunks, *, training: bad_result)
    with pytest.raises(TypeError, match="list of modules"):
        config.pre_wrap_hooks[0]([_TinyModel()])


def test_peft_runs_before_distributed_wrap():
    from megatron.training.models import dist_utils

    config = _model_config()
    peft = _TinyPEFT()
    register_peft_pre_wrap_hook(config, peft)
    model = _TinyModel()
    events = []

    def pre_wrap(chunks):
        events.append("peft")
        return config.pre_wrap_hooks[0](chunks)

    def ddp_wrap(chunks, *args, **kwargs):
        events.append("ddp")
        assert not chunks[0].base.weight.requires_grad
        assert chunks[0].adapter.weight.requires_grad
        return chunks

    transformer_config = SimpleNamespace(
        virtual_pipeline_model_parallel_size=None,
        init_model_with_meta_device=False,
        use_cpu_initialization=True,
    )
    with (
        mock.patch.object(dist_utils, "build_virtual_pipeline_stages", return_value=[model]),
        mock.patch.object(dist_utils, "_print_num_params"),
        mock.patch.object(
            dist_utils, "_wrap_with_mp_wrapper", side_effect=lambda chunks, *_: chunks
        ),
        mock.patch.object(dist_utils, "_ddp_wrap", side_effect=ddp_wrap),
        mock.patch.object(dist_utils, "correct_amax_history_if_needed", None),
        mock.patch.object(
            dist_utils.tensor_parallel, "set_defaults_if_not_set_tensor_model_parallel_attributes"
        ),
    ):
        unimodal_build_distributed_models(
            lambda: model,
            transformer_config,
            SimpleNamespace(),
            ddp_config=SimpleNamespace(),
            pre_wrap_hook=pre_wrap,
        )

    assert events == ["peft", "ddp"]


def test_setup_registers_peft_before_model_builder():
    args = SimpleNamespace(
        skip_train=False,
        perform_rl_step=False,
        no_load_optim=True,
        logits_save_dir=None,
        logits_load_dir=None,
        moe_use_upcycling=False,
        load=None,
        pretrained_checkpoint=None,
        data_parallel_size=1,
        micro_batch_size=1,
        fp16=False,
        ckpt_convert_format=None,
        use_gloo_process_groups=False,
        dump_param_to_param_group_map=False,
        freeze_all_layers=False,
        freeze_base_model_for_mtp=False,
    )
    peft = _TinyPEFT()
    model = _TinyModel()
    model_config = mock.Mock(spec=ModelConfig)
    model_config.pre_wrap_hooks = []

    def build_model(**kwargs):
        assert len(model_config.pre_wrap_hooks) == 1
        chunks = model_config.pre_wrap_hooks[0]([model])
        assert not chunks[0].base.weight.requires_grad
        assert chunks[0].adapter.weight.requires_grad
        return chunks

    builder = mock.Mock()
    builder.build_distributed_models.side_effect = build_model
    model_config.get_builder_cls.return_value = mock.Mock(return_value=builder)
    cfg = SimpleNamespace(
        model=model_config,
        peft=peft,
        profiling=mock.Mock(),
        ddp=mock.Mock(),
        optimizer=SimpleNamespace(
            overlap_param_gather_with_optimizer_step=False,
            use_layer_wise_distributed_optimizer=False,
        ),
        dist=SimpleNamespace(use_megatron_fsdp=False, use_torch_fsdp2=False),
        rng=SimpleNamespace(data_parallel_random_init=False),
    )

    optimizer = object()
    scheduler = object()

    def build_optimizer(config, chunks, **kwargs):
        assert chunks == [model]
        assert not chunks[0].base.weight.requires_grad
        assert chunks[0].adapter.weight.requires_grad
        return optimizer

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.Mock()),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "unwrap_model", return_value=[model]),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "get_current_global_batch_size", return_value=1),
        mock.patch.object(
            training_mod,
            "get_megatron_optimizer_config",
            return_value=(SimpleNamespace(timers=None), None),
        ),
        mock.patch.object(training_mod, "get_megatron_optimizer", side_effect=build_optimizer),
        mock.patch.object(training_mod, "get_optimizer_param_scheduler", return_value=scheduler),
        mock.patch.object(training_mod.mpu, "model_parallel_is_initialized", return_value=False),
        mock.patch("megatron.training.utils.start_memory_history_recording"),
    ):
        built, actual_optimizer, actual_scheduler = training_mod.setup_model_and_optimizer(
            ModelType.encoder_or_decoder, cfg_container=cfg, pg_collection=mock.Mock()
        )

    assert built == [model]
    assert actual_optimizer is optimizer
    assert actual_scheduler is scheduler
    assert peft.calls == 1

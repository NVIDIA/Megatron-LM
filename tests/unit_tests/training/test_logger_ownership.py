# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Logging services and serialized metadata use their explicit configuration."""

import sys
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.training import arguments, global_vars
from megatron.training.argument_utils import logger_args_snapshot, logger_config_from_args
from megatron.training.async_utils import build_otel_worker_bootstrap
from megatron.training.config import LoggerConfig
from megatron.training.initialize import setup_logging, write_args_to_tensorboard
from megatron.training.training import _should_compute_params_norm


def test_cli_aliases_and_native_config_match():
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(
        [
            '--no-one-logger',
            '--one-logger-project',
            'project',
            '--one-logger-run-name',
            'run',
            '--one-logger-async',
            '--app-tag-run-name',
            'application',
            '--app-tag-run-version',
            '1.0',
            '--otel-enabled',
            '--otel-service-name',
            'service',
            '--otel-span-groups',
            'checkpoint',
        ]
    )
    config = logger_config_from_args(args)
    expected = LoggerConfig(
        enable_one_logger=False,
        one_logger_project='project',
        one_logger_run_name='run',
        one_logger_async=True,
        app_tag_run_name='application',
        app_tag_run_version='1.0',
        otel_enabled=True,
        otel_service_name='service',
        otel_span_groups='checkpoint',
    )
    assert asdict(config) == asdict(expected)


def test_normalization_and_snapshot_do_not_alias_or_mutate_args():
    args = Namespace(modules_to_filter=['module'], log_interval=7, iteration=12)
    config = logger_config_from_args(args)
    args.modules_to_filter.append('stale')
    assert config.modules_to_filter == ['module']
    config.log_interval = 11
    snapshot = logger_args_snapshot(args, config)
    assert snapshot.log_interval == 11 and args.log_interval == 7
    assert snapshot.iteration == args.iteration
    config.modules_to_filter.append('later')
    assert snapshot.modules_to_filter == ['module']
    for field in fields(config):
        if hasattr(args, field.name):
            delattr(args, field.name)
    assert logger_args_snapshot(args, config).log_interval == 11


@pytest.mark.parametrize('interval,valid', [(None, True), (20, True), (21, False)])
def test_native_memory_interval_validation(interval, valid):
    config = LoggerConfig(log_interval=10, log_memory_interval=interval)
    if valid:
        config.validate()
    else:
        with pytest.raises(AssertionError):
            config.validate()


@pytest.mark.parametrize('rank,enabled', [(0, False), (1, True)])
def test_tensorboard_uses_owned_directory_and_queue(monkeypatch, rank, enabled):
    factory = Mock()
    monkeypatch.setitem(
        sys.modules, 'torch.utils.tensorboard', SimpleNamespace(SummaryWriter=factory)
    )
    monkeypatch.setattr(global_vars, '_GLOBAL_TENSORBOARD_WRITER', None)
    args = Namespace(rank=rank, world_size=2, tensorboard_dir='stale', tensorboard_queue_size=99)
    config = LoggerConfig(tensorboard_dir='owned', tensorboard_queue_size=17)
    global_vars._set_tensorboard_writer(args, logger_config=config)
    if enabled:
        factory.assert_called_once_with(log_dir='owned', max_queue=17)
    else:
        factory.assert_not_called()


def test_wandb_uses_owned_settings_and_detached_metadata(monkeypatch, tmp_path):
    wandb = SimpleNamespace(init=Mock())
    monkeypatch.setitem(sys.modules, 'wandb', wandb)
    monkeypatch.setattr(global_vars, '_GLOBAL_WANDB_WRITER', None)
    config = LoggerConfig(
        wandb_project='project',
        wandb_exp_name='run',
        wandb_entity='team',
        wandb_save_dir=str(tmp_path),
        log_interval=17,
    )
    args = Namespace(rank=0, world_size=1, log_interval=99)
    before = vars(args).copy()
    global_vars._set_wandb_writer(args, logger_config=config)
    kwargs = wandb.init.call_args.kwargs
    assert kwargs['project'] == 'project' and kwargs['name'] == 'run'
    assert kwargs['entity'] == 'team' and kwargs['dir'] == str(tmp_path)
    assert kwargs['config']['log_interval'] == 17
    assert vars(args) == before


@pytest.mark.parametrize('wandb_project,asynchronous', [(None, False), ('project', True)])
def test_one_logger_async_policy(monkeypatch, wandb_project, asynchronous):
    factory = Mock()
    monkeypatch.setitem(sys.modules, 'one_logger', SimpleNamespace(OneLogger=factory))
    monkeypatch.setattr(global_vars, '_GLOBAL_ONE_LOGGER', None)
    config = LoggerConfig(
        one_logger_project='owned', one_logger_run_name='run', wandb_project=wandb_project
    )
    global_vars._set_one_logger(Namespace(rank=0, world_size=1), logger_config=config)
    factory.assert_called_once_with(
        config={'project': 'owned', 'name': 'run', 'async': asynchronous}
    )


def test_timers_use_owned_settings(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(global_vars, 'Timers', factory)
    monkeypatch.setattr(global_vars, '_GLOBAL_TIMERS', None)
    global_vars._set_timers(
        Namespace(), logger_config=LoggerConfig(timing_log_level=2, timing_log_option='all')
    )
    factory.assert_called_once_with(2, 'all')


def test_tensorboard_metadata_uses_snapshot(monkeypatch):
    from megatron.training import initialize

    args = Namespace(iteration=13, log_interval=99)
    writer = Mock()
    monkeypatch.setattr(initialize, 'get_args', lambda: args)
    monkeypatch.setattr(initialize, 'get_tensorboard_writer', lambda: writer)
    write_args_to_tensorboard(logger_config=LoggerConfig(log_interval=17))
    writer.add_text.assert_any_call('log_interval', '17', global_step=13)
    assert args.log_interval == 99


def test_async_worker_telemetry_uses_config(monkeypatch):
    monkeypatch.delenv('OTEL_SERVICE_NAME', raising=False)
    attrs = Mock(return_value={'test': 'resource'})
    monkeypatch.setattr(global_vars, 'build_telemetry_resource_attrs', attrs)
    args = Namespace(rank=0, world_size=2, otel_enabled=False, otel_service_name='stale')
    config = LoggerConfig(otel_enabled=True, otel_service_name='owned')
    result = build_otel_worker_bootstrap(args, logger_config=config)
    assert result['enabled'] is True and result['service_name'] == 'owned'
    assert result['resource_attrs'] == {'test': 'resource'}
    attrs.assert_called_once_with(args)


def test_disabled_async_telemetry_does_not_read_resource_metadata(monkeypatch):
    attrs = Mock(side_effect=AssertionError('disabled telemetry must not query NVML'))
    monkeypatch.setattr(global_vars, 'build_telemetry_resource_attrs', attrs)
    result = build_otel_worker_bootstrap(
        Namespace(rank=0, world_size=1), logger_config=LoggerConfig()
    )
    assert result['enabled'] is False and result['resource_attrs'] == {}
    attrs.assert_not_called()


def test_native_parameter_norm_cadence():
    config = LoggerConfig(
        log_params_norm=True, log_interval=20, tensorboard_dir='events', tensorboard_log_interval=5
    )
    assert _should_compute_params_norm(config, 1, True)
    assert _should_compute_params_norm(config, 5, False)
    assert not _should_compute_params_norm(config, 6, False)


def test_logging_level_config_overrides_environment(monkeypatch):
    from megatron.training import initialize

    set_level = Mock()
    monkeypatch.setenv('MEGATRON_LOGGING_LEVEL', '40')
    monkeypatch.setattr(initialize, 'is_rank0', lambda: True)
    monkeypatch.setattr(initialize.logging.getLogger(), 'setLevel', set_level)
    monkeypatch.setattr(initialize, 'get_args', lambda: Namespace(logging_level=50))
    setup_logging(logger_config=LoggerConfig(logging_level=20))
    set_level.assert_called_once_with(20)


def test_checkpoint_logging_does_not_override_current_run(monkeypatch):
    from megatron.training import checkpointing

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.load = 'checkpoint'
    args.log_interval = 17
    saved = Namespace(log_interval=99, otel_service_name='old-run')
    state = {'args': saved, 'iteration': 12, 'checkpoint_version': 3.0}
    monkeypatch.setattr(
        checkpointing,
        '_load_base_checkpoint',
        Mock(return_value=(state, 'checkpoint', False, None)),
    )
    checkpointing.load_args_from_checkpoint(args)
    config = logger_config_from_args(args)
    assert config.log_interval == 17 and config.otel_service_name is None


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('model_kind', ['gpt', 'hybrid'])
def test_native_builder_projects_logging_before_construction(monkeypatch, model_kind, enabled):
    from megatron.core.transformer import TransformerConfig
    from megatron.training.models import gpt, hybrid

    module = gpt if model_kind == 'gpt' else hybrid
    config_cls = gpt.GPTModelConfig if model_kind == 'gpt' else hybrid.HybridModelConfig
    builder_cls = gpt.GPTModelBuilder if model_kind == 'gpt' else hybrid.HybridModelBuilder
    transformer = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    transformer.log_max_attention_logit = not enabled
    config = config_cls(transformer=transformer, vocab_size=128, seq_length=16)
    models = [Mock()]

    def build(build_model, transformer_config, *args, **kwargs):
        assert transformer_config.log_max_attention_logit is enabled
        return models

    monkeypatch.setattr(module, 'unimodal_build_distributed_models', build)
    assert (
        builder_cls(config).build_distributed_models(
            Mock(), wrap_with_ddp=False, log_max_attention_logit=enabled
        )
        == models
    )


@pytest.mark.parametrize('enabled', [False, True])
def test_mimo_projects_logging_before_module_construction(monkeypatch, enabled):
    from examples.mimo.training import builder
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.spec_utils import ModuleSpec

    configs = [
        TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4) for _ in range(3)
    ]
    for config in configs:
        config.log_max_attention_logit = not enabled
    language = ModuleSpec(module=Mock, params={'config': configs[0]})
    encoder = ModuleSpec(
        module=Mock,
        submodules={
            'encoders': {
                'vision': ModuleSpec(module=Mock, params={'transformer_config': configs[1]})
            }
        },
    )
    projection = ModuleSpec(module=Mock, params={'config': configs[2]})
    topology = SimpleNamespace(grids={MIMO_LANGUAGE_MODULE_KEY: Mock(), 'vision': Mock()})
    provider = SimpleNamespace(
        special_token_ids=lambda args: {'vision': 1},
        encoder_specs={'vision': lambda *args: encoder},
        language_spec=lambda *args: language,
        language_input_projection_specs={'vision': lambda *args: projection},
    )
    monkeypatch.setattr(builder, 'get_args', lambda: Namespace())
    monkeypatch.setattr(builder, 'resolve_provider', lambda args: provider)
    monkeypatch.setattr(
        builder, '_resolve_role', lambda topology: (MIMO_LANGUAGE_MODULE_KEY, True, Mock())
    )
    model = Mock()

    def construct(*args, **kwargs):
        assert all(config.log_max_attention_logit is enabled for config in configs)
        return model

    monkeypatch.setattr(builder, 'MimoModel', construct)
    instance = builder.MimoModelBuilder(builder.MimoBuildConfig(_topology=topology))
    assert instance.build_model(Mock(), log_max_attention_logit=enabled) is model


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('source', ['args', 'yaml', 'provided'])
def test_legacy_gpt_logging_preserves_config_factory(monkeypatch, source, enabled):
    import gpt_builders

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    del args.log_max_attention_logit
    args.yaml_cfg = 'model.yaml' if source == 'yaml' else None
    config = Namespace(log_max_attention_logit=not enabled)
    from_args = Mock(return_value=config)
    from_yaml = Mock(return_value=config)
    monkeypatch.setattr(gpt_builders, 'core_transformer_config_from_args', from_args)
    monkeypatch.setattr(gpt_builders, 'core_transformer_config_from_yaml', from_yaml)
    monkeypatch.setattr(gpt_builders, '_get_transformer_layer_spec', Mock())
    model = Mock()

    def construct(**kwargs):
        assert kwargs['config'] is config
        assert config.log_max_attention_logit is enabled
        return model

    monkeypatch.setattr(gpt_builders, 'GPTModel', construct)
    assert (
        gpt_builders.gpt_builder(
            args,
            True,
            True,
            config=config if source == 'provided' else None,
            log_max_attention_logit=enabled,
        )
        is model
    )
    assert from_args.call_count == (source == 'args')
    assert from_yaml.call_count == (source == 'yaml')
    assert not hasattr(args, 'log_max_attention_logit')


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('legacy_field', ['stale', 'absent'])
@pytest.mark.parametrize('teacher_override', [None, False, True])
def test_teacher_logging_inherits_owned_policy_with_explicit_yaml_override(
    monkeypatch, tmp_path, enabled, legacy_field, teacher_override
):
    builder = pytest.importorskip('megatron.post_training.model_builder')
    args = Namespace(export_kd_teacher_model_config=None, kv_channels=8)
    if legacy_field == 'stale':
        args.log_max_attention_logit = not enabled
    before = vars(args).copy()
    if teacher_override is not None:
        (tmp_path / 'model_config.yaml').write_text(
            f'log_max_attention_logit: {str(teacher_override).lower()}\n'
        )
    monkeypatch.setattr(builder, 'get_args', lambda: args)
    teacher = builder._load_teacher_model_config(str(tmp_path), log_max_attention_logit=enabled)
    expected = enabled if teacher_override is None else teacher_override
    assert teacher.log_max_attention_logit is expected
    assert vars(args) == before

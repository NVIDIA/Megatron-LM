# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Logging services and serialized metadata use their run-owned configuration."""

import sys
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.training import arguments, global_vars
from megatron.training.argument_utils import _default_config_from_args
from megatron.training.async_utils import build_otel_worker_bootstrap
from megatron.training.config import LoggerConfig
from megatron.training.initialize import setup_logging, write_args_to_tensorboard
from megatron.training.training import _should_compute_params_norm


def test_cli_aliases_and_native_config_match(run_config):
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
            '--no-barrier-with-level-1-timing',
            '--run-workload-inspector-server',
        ]
    )
    config = _default_config_from_args(LoggerConfig, args)
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
        barrier_with_L1_time=False,
        run_workload_inspector_server=True,
    )
    assert asdict(config) == asdict(expected)


@pytest.mark.parametrize('inference', [False, True])
def test_normalization_does_not_alias_or_mutate_args(monkeypatch, inference):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig
    from megatron.training import argument_utils, training

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.modules_to_filter = ['module']
    args.log_interval = 7
    monkeypatch.setattr(
        training, 'get_megatron_optimizer_config', lambda args: (OptimizerConfig(), None)
    )
    monkeypatch.setattr(
        training, 'get_megatron_ddp_config', lambda args: DistributedDataParallelConfig()
    )
    if inference:
        cfg = argument_utils.inference_cfg_container_from_args(args, build_model_config=False)
    else:
        cfg = argument_utils.pretrain_cfg_container_from_args(args)
    config = cfg.logger
    args.modules_to_filter.append('stale')
    assert config.modules_to_filter == ['module']
    config.log_interval = 11
    assert args.log_interval == 7
    config.modules_to_filter.append('later')
    assert args.modules_to_filter == ['module', 'stale']


@pytest.mark.parametrize('interval,valid', [(None, True), (20, True), (21, False)])
def test_native_memory_interval_validation(interval, valid, run_config):
    config = LoggerConfig(log_interval=10, log_memory_interval=interval)
    run_config.logger = config
    if valid:
        run_config.validate()
    else:
        with pytest.raises(AssertionError):
            run_config.validate()


@pytest.mark.parametrize('rank,enabled', [(0, False), (1, True)])
def test_tensorboard_uses_owned_directory_and_queue(monkeypatch, rank, enabled, run_config):
    factory = Mock()
    monkeypatch.setitem(
        sys.modules, 'torch.utils.tensorboard', SimpleNamespace(SummaryWriter=factory)
    )
    monkeypatch.setattr(global_vars, '_GLOBAL_TENSORBOARD_WRITER', None)
    args = Namespace(rank=rank, world_size=2, tensorboard_dir='stale', tensorboard_queue_size=99)
    config = LoggerConfig(tensorboard_dir='owned', tensorboard_queue_size=17)
    run_config.logger = config
    global_vars._set_tensorboard_writer(args)
    if enabled:
        factory.assert_called_once_with(log_dir='owned', max_queue=17)
    else:
        factory.assert_not_called()


def test_wandb_uses_owned_settings_and_detached_metadata(monkeypatch, tmp_path, run_config):
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
    run_config.logger = config
    global_vars._set_wandb_writer(args)
    kwargs = wandb.init.call_args.kwargs
    assert kwargs['project'] == 'project' and kwargs['name'] == 'run'
    assert kwargs['entity'] == 'team' and kwargs['dir'] == str(tmp_path)
    assert kwargs['config']['log_interval'] == 17
    assert vars(args) == before


@pytest.mark.parametrize('wandb_project,asynchronous', [(None, False), ('project', True)])
def test_one_logger_async_policy(monkeypatch, wandb_project, asynchronous, run_config):
    factory = Mock()
    monkeypatch.setitem(sys.modules, 'one_logger', SimpleNamespace(OneLogger=factory))
    monkeypatch.setattr(global_vars, '_GLOBAL_ONE_LOGGER', None)
    config = LoggerConfig(
        one_logger_project='owned', one_logger_run_name='run', wandb_project=wandb_project
    )
    run_config.logger = config
    global_vars._set_one_logger(Namespace(rank=0, world_size=1))
    factory.assert_called_once_with(
        config={'project': 'owned', 'name': 'run', 'async': asynchronous}
    )


def test_timers_use_owned_settings(monkeypatch, run_config):
    factory = Mock()
    monkeypatch.setattr(global_vars, 'Timers', factory)
    monkeypatch.setattr(global_vars, '_GLOBAL_TIMERS', None)
    run_config.logger = LoggerConfig(timing_log_level=2, timing_log_option='all')
    global_vars._set_timers(Namespace())
    factory.assert_called_once_with(2, 'all')


def test_tensorboard_metadata_uses_config(monkeypatch, run_config):
    from megatron.training import initialize

    args = Namespace(iteration=13, log_interval=99)
    writer = Mock()
    monkeypatch.setattr(initialize, 'get_args', lambda: args)
    monkeypatch.setattr(initialize, 'get_tensorboard_writer', lambda: writer)
    run_config.logger = LoggerConfig(log_interval=17)
    write_args_to_tensorboard()
    writer.add_text.assert_any_call('log_interval', '17', global_step=13)
    assert all(call.args != ('log_interval', '99') for call in writer.add_text.call_args_list)
    assert args.log_interval == 99


def test_async_worker_telemetry_uses_config(monkeypatch, run_config):
    monkeypatch.delenv('OTEL_SERVICE_NAME', raising=False)
    attrs = Mock(return_value={'test': 'resource'})
    monkeypatch.setattr(global_vars, 'build_telemetry_resource_attrs', attrs)
    args = Namespace(rank=0, world_size=2, otel_enabled=False, otel_service_name='stale')
    config = LoggerConfig(otel_enabled=True, otel_service_name='owned')
    run_config.logger = config
    result = build_otel_worker_bootstrap(args)
    assert result['enabled'] is True and result['service_name'] == 'owned'
    assert result['resource_attrs'] == {'test': 'resource'}
    attrs.assert_called_once_with(args)


def test_disabled_async_telemetry_does_not_read_resource_metadata(monkeypatch, run_config):
    attrs = Mock(side_effect=AssertionError('disabled telemetry must not query NVML'))
    monkeypatch.setattr(global_vars, 'build_telemetry_resource_attrs', attrs)
    run_config.logger = LoggerConfig()
    result = build_otel_worker_bootstrap(Namespace(rank=0, world_size=1))
    assert result['enabled'] is False and result['resource_attrs'] == {}
    attrs.assert_not_called()


def test_native_parameter_norm_cadence(run_config):
    config = LoggerConfig(
        log_params_norm=True, log_interval=20, tensorboard_dir='events', tensorboard_log_interval=5
    )
    run_config.logger = config
    assert _should_compute_params_norm(Namespace(), 1, True)
    assert _should_compute_params_norm(Namespace(), 5, False)
    assert not _should_compute_params_norm(Namespace(), 6, False)


def test_logging_level_config_overrides_environment(monkeypatch, run_config):
    from megatron.training import initialize

    set_level = Mock()
    monkeypatch.setenv('MEGATRON_LOGGING_LEVEL', '40')
    monkeypatch.setattr(initialize, 'is_rank0', lambda: True)
    monkeypatch.setattr(initialize.logging.getLogger(), 'setLevel', set_level)
    monkeypatch.setattr(initialize, 'get_args', lambda: Namespace(logging_level=50))
    run_config.logger = LoggerConfig(logging_level=20)
    setup_logging()
    set_level.assert_called_once_with(20)


def test_checkpoint_logging_does_not_override_current_run(monkeypatch, run_config):
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
    config = _default_config_from_args(LoggerConfig, args)
    assert config.log_interval == 17 and config.otel_service_name is None


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('model_kind', ['gpt', 'hybrid'])
def test_native_builder_projects_logging_before_construction(
    monkeypatch, model_kind, enabled, run_config
):
    from megatron.core.transformer import TransformerConfig
    from megatron.training import training
    from megatron.training.models import gpt, hybrid

    config_cls = gpt.GPTModelConfig if model_kind == 'gpt' else hybrid.HybridModelConfig
    transformer = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    transformer.log_max_attention_logit = not enabled
    transformer.barrier_with_L1_time = not enabled
    run_config.model = config_cls(transformer=transformer, vocab_size=128, seq_length=16)
    run_config.logger.log_max_attention_logit = enabled
    run_config.logger.barrier_with_L1_time = enabled
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    monkeypatch.setattr(training, 'get_args', lambda: args)
    monkeypatch.setattr(training, 'get_timers', Mock())
    monkeypatch.setattr(training, 'get_one_logger', lambda: None)
    monkeypatch.setattr(training, 'has_nvidia_modelopt', False)
    monkeypatch.setattr(training, 'is_gtp_remat_active', lambda args: False)
    monkeypatch.setattr(training, '_add_model_freeze_pre_wrap_hook', Mock())
    monkeypatch.setattr('megatron.training.utils.start_memory_history_recording', Mock())

    class ObservedModel(Exception):
        pass

    def construct(model_config):
        assert model_config.transformer.log_max_attention_logit is enabled
        assert model_config.transformer.barrier_with_L1_time is enabled
        raise ObservedModel

    monkeypatch.setattr(config_cls, 'get_builder_cls', lambda self: construct)
    with pytest.raises(ObservedModel):
        training.setup_model_and_optimizer(Mock(), cfg_container=run_config)


@pytest.mark.parametrize('enabled', [False, True])
def test_mimo_projects_logging_before_module_construction(monkeypatch, enabled, run_config):
    run_config.logger.log_max_attention_logit = enabled
    run_config.logger.barrier_with_L1_time = enabled
    from examples.mimo.training import builder
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.spec_utils import ModuleSpec

    configs = [
        TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4) for _ in range(3)
    ]
    for config in configs:
        config.log_max_attention_logit = not enabled
        config.barrier_with_L1_time = not enabled
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
        assert all(config.barrier_with_L1_time is enabled for config in configs)
        return model

    monkeypatch.setattr(builder, 'MimoModel', construct)
    instance = builder.MimoModelBuilder(builder.MimoBuildConfig(_topology=topology))
    assert instance.build_model(Mock()) is model


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('source', ['args', 'yaml', 'provided', 'native'])
def test_legacy_gpt_logging_preserves_config_factory(monkeypatch, source, enabled, run_config):
    run_config.logger.log_max_attention_logit = enabled
    run_config.logger.barrier_with_L1_time = enabled
    import gpt_builders
    from megatron.core.transformer import TransformerConfig
    from megatron.training import argument_utils, yaml_arguments
    from megatron.training.models import GPTModelConfig

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    del args.log_max_attention_logit
    del args.barrier_with_L1_time
    args.yaml_cfg = 'model.yaml' if source == 'yaml' else None
    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    config.log_max_attention_logit = not enabled
    config.barrier_with_L1_time = not enabled
    if source == 'native':
        run_config.model = GPTModelConfig(transformer=config, vocab_size=128, seq_length=16)
    from_args = Mock(return_value=config)
    from_yaml = Mock(return_value=config)
    monkeypatch.setattr(argument_utils, 'core_transformer_config_from_args', from_args)
    monkeypatch.setattr(yaml_arguments, 'core_transformer_config_from_yaml', from_yaml)
    monkeypatch.setattr(gpt_builders, '_get_transformer_layer_spec', Mock())
    model = Mock()

    def construct(**kwargs):
        assert kwargs['config'] is config
        assert config.log_max_attention_logit is enabled
        assert config.barrier_with_L1_time is enabled
        return model

    monkeypatch.setattr(gpt_builders, 'GPTModel', construct)
    assert (
        gpt_builders.gpt_builder(args, True, True, config=config if source == 'provided' else None)
        is model
    )
    assert from_args.call_count == (source == 'args')
    assert from_yaml.call_count == (source == 'yaml')
    assert not hasattr(args, 'log_max_attention_logit')
    assert not hasattr(args, 'barrier_with_L1_time')


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('legacy_field', ['stale', 'absent'])
@pytest.mark.parametrize('teacher_override', [None, False, True])
def test_teacher_logging_inherits_owned_policy_with_explicit_yaml_override(
    monkeypatch, tmp_path, enabled, legacy_field, teacher_override, run_config
):
    run_config.logger.log_max_attention_logit = enabled
    run_config.logger.barrier_with_L1_time = enabled
    builder = pytest.importorskip('megatron.post_training.model_builder')
    args = Namespace(export_kd_teacher_model_config=None, kv_channels=8)
    if legacy_field == 'stale':
        args.log_max_attention_logit = not enabled
        args.barrier_with_L1_time = not enabled
    before = vars(args).copy()
    if teacher_override is not None:
        (tmp_path / 'model_config.yaml').write_text(
            f'log_max_attention_logit: {str(teacher_override).lower()}\n'
            f'barrier_with_L1_time: {str(teacher_override).lower()}\n'
        )
    monkeypatch.setattr(builder, 'get_args', lambda: args)
    teacher = builder._load_teacher_model_config(str(tmp_path))
    expected = enabled if teacher_override is None else teacher_override
    assert teacher.log_max_attention_logit is expected
    assert teacher.barrier_with_L1_time is expected
    assert vars(args) == before


@pytest.mark.parametrize('enabled', [False, True])
def test_finalization_is_idempotent_and_updates_derived_configs(run_config, enabled):
    from megatron.core.transformer import TransformerConfig
    from megatron.training.models import GPTModelConfig

    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
        vocab_size=128,
        seq_length=16,
    )
    run_config.logger.log_max_attention_logit = enabled
    run_config.logger.barrier_with_L1_time = enabled
    run_config.logger.log_num_zeros_in_grad = enabled
    run_config.finalize()
    run_config.finalize()
    assert run_config.model.transformer.log_max_attention_logit is enabled
    assert run_config.model.transformer.barrier_with_L1_time is enabled
    assert run_config.optimizer.log_num_zeros_in_grad is enabled
    assert run_config.optimizer.barrier_with_L1_time is enabled


def test_base_finalization_does_not_require_logger():
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.transformer import TransformerConfig
    from megatron.training.config.container import ConfigContainerBase

    @dataclass
    class CustomContainer(ConfigContainerBase):
        model: TransformerConfig
        optimizer: OptimizerConfig

    model = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    optimizer = OptimizerConfig()
    before = (model.log_max_attention_logit, optimizer.log_num_zeros_in_grad)
    ConfigContainerBase().finalize()
    CustomContainer(model=model, optimizer=optimizer).finalize()
    assert (model.log_max_attention_logit, optimizer.log_num_zeros_in_grad) == before


@pytest.mark.parametrize('model_type', ['gpt', 'hybrid', 'legacy'])
def test_inference_finalization_does_not_require_optimizer(model_type):
    from megatron.core.transformer import TransformerConfig
    from megatron.training.config import (
        CheckpointConfig,
        InferenceConfigContainer,
        InferenceSetupConfig,
    )
    from megatron.training.models import GPTModelConfig, HybridModelConfig

    model = None
    if model_type != 'legacy':
        config_cls = GPTModelConfig if model_type == 'gpt' else HybridModelConfig
        model = config_cls(
            transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
            vocab_size=128,
            seq_length=16,
        )
    cfg = InferenceConfigContainer(
        model=model,
        checkpoint=CheckpointConfig(),
        inference=InferenceSetupConfig(),
        logger=LoggerConfig(log_max_attention_logit=True, barrier_with_L1_time=False),
    )
    cfg.finalize()
    cfg.finalize()
    assert not hasattr(cfg, 'optimizer')
    if model is not None:
        assert model.transformer.log_max_attention_logit is True
        assert model.transformer.barrier_with_L1_time is False
    override = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    cfg.finalize_model_config(override)
    assert override.log_max_attention_logit is True
    assert override.barrier_with_L1_time is False


def test_mimo_finalization_ignores_non_model_state(run_config):
    from megatron.core.models.mimo.config.base_configs import MimoModelConfig
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.spec_utils import ModuleSpec

    @dataclass
    class Metadata:
        transformer: TransformerConfig

    active = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    ignored = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    spec = ModuleSpec(
        module=Mock,
        params={'config': active, 'metadata': Metadata(ignored)},
        metainfo={'config': ignored},
    )
    spec.submodules = {'shared': [spec, spec]}
    model = MimoModelConfig(
        language_model_spec=spec, module_to_grid_map={MIMO_LANGUAGE_MODULE_KEY: Metadata(ignored)}
    )
    run_config.logger.log_max_attention_logit = True
    run_config.finalize_model_config(model)
    assert active.log_max_attention_logit is True
    assert ignored.log_max_attention_logit is False
    run_config.finalize_model_config(Metadata(ignored))
    assert ignored.log_max_attention_logit is False


def test_explicit_transformer_override_does_not_replace_native_model(run_config):
    from megatron.core.transformer import TransformerConfig
    from megatron.training.argument_utils import get_transformer_config
    from megatron.training.models import GPTModelConfig

    native = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    override = TransformerConfig(num_layers=4, hidden_size=64, num_attention_heads=4)
    run_config.model = GPTModelConfig(transformer=native, vocab_size=128, seq_length=16)
    run_config.logger.log_max_attention_logit = True

    assert get_transformer_config(Namespace(), override) is override
    assert override.log_max_attention_logit is True
    assert run_config.model.transformer is native
    assert native.num_layers == 2


@pytest.mark.parametrize('provider', ['gpt', 'hybrid'])
@pytest.mark.parametrize('existing_model', ['matching', 'different', 'absent'])
def test_inference_reuses_matching_config_and_preserves_provider_override(
    monkeypatch, run_config, provider, existing_model
):
    from megatron.core.transformer import TransformerConfig
    from megatron.inference import utils
    from megatron.training.models import GPTModelConfig, HybridModelConfig

    config_cls = GPTModelConfig if provider == 'gpt' else HybridModelConfig
    other_cls = HybridModelConfig if provider == 'gpt' else GPTModelConfig

    def make_config(cls):
        return cls(
            transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
            vocab_size=128,
            seq_length=16,
        )

    existing = (
        None
        if existing_model == 'absent'
        else make_config(config_cls if existing_model == 'matching' else other_cls)
    )
    run_config.model = existing
    run_config.logger.log_max_attention_logit = True
    created = make_config(config_cls)
    factory = Mock(return_value=created)
    monkeypatch.setattr(utils, f'{provider}_config_from_args', factory)
    builder = Mock()
    monkeypatch.setattr(
        utils, 'GPTModelBuilder' if provider == 'gpt' else 'HybridModelBuilder', builder
    )

    utils.get_model_builder(Namespace(model_provider=provider))

    expected = existing if existing_model == 'matching' else created
    builder.assert_called_once_with(expected)
    assert factory.call_count == (existing_model != 'matching')
    assert expected.transformer.log_max_attention_logit is True
    assert run_config.model is existing


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('bootstrap', ['pretrain', 'args_only'])
def test_optimizer_receives_owned_logging_settings(monkeypatch, enabled, bootstrap, run_config):
    from megatron.training import training

    args = Namespace(
        skip_train=False,
        perform_rl_step=False,
        logits_save_dir=None,
        logits_load_dir=None,
        use_gloo_process_groups=False,
        dump_param_to_param_group_map=None,
    )
    if bootstrap == 'args_only':
        monkeypatch.setattr(global_vars, '_GLOBAL_ARGS', None)
        monkeypatch.setattr(global_vars, '_GLOBAL_RUN_CONFIG', None)
        monkeypatch.setattr(global_vars, 'initialize_runtime_services', Mock())
        global_vars.set_global_variables(args, build_tokenizer=False)
        run_config = global_vars.get_run_config()
        assert not hasattr(run_config, 'optimizer')
    config = Namespace(barrier_with_L1_time=not enabled, log_num_zeros_in_grad=not enabled)
    monkeypatch.setattr(training, 'get_args', lambda: args)
    monkeypatch.setattr(training, 'get_timers', Mock())
    monkeypatch.setattr(training, 'get_one_logger', lambda: None)
    monkeypatch.setattr(training, 'has_nvidia_modelopt', False)
    monkeypatch.setattr(training, 'is_gtp_remat_active', lambda args: False)
    monkeypatch.setattr(training, 'get_model', Mock(return_value=[Mock()]))
    monkeypatch.setattr(training, 'unwrap_model', lambda model: model)
    monkeypatch.setattr(training, 'get_megatron_optimizer_config', lambda args: (config, None))

    class ObservedOptimizer(Exception):
        pass

    def construct(config, model, **kwargs):
        assert config.barrier_with_L1_time is enabled
        assert config.log_num_zeros_in_grad is enabled
        raise ObservedOptimizer

    monkeypatch.setattr(training, 'get_megatron_optimizer', construct)
    with pytest.raises(ObservedOptimizer):
        run_config.logger = LoggerConfig(
            barrier_with_L1_time=enabled, log_num_zeros_in_grad=enabled
        )
        training.setup_model_and_optimizer(Mock(), model_provider_func=Mock())

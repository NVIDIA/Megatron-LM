# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Logging services use config; argument metadata retains its legacy format."""

import sys
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, fields
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
        training,
        'get_megatron_ddp_config',
        lambda args, *, use_torch_fsdp2: DistributedDataParallelConfig(),
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


def test_wandb_uses_owned_settings_and_legacy_metadata(monkeypatch, tmp_path, run_config):
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
    args = Namespace(rank=0, world_size=1, log_interval=99, run_workload_inspector_server=False)
    before = vars(args).copy()
    run_config.logger = config
    run_config.profiling.run_workload_inspector_server = True
    global_vars._set_wandb_writer(args)
    kwargs = wandb.init.call_args.kwargs
    assert kwargs['project'] == 'project' and kwargs['name'] == 'run'
    assert kwargs['entity'] == 'team' and kwargs['dir'] == str(tmp_path)
    assert kwargs['config'] is vars(args)
    assert kwargs['config'] == before
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


def test_finetune_reuses_registered_config(monkeypatch, run_config):
    import torch

    from tasks import finetune_utils

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.epochs, args.iteration = 0, 1
    args.main_grads_dtype = args.main_params_dtype = torch.float32
    args.exp_avg_dtype = args.exp_avg_sq_dtype = torch.float32
    timers = Mock()
    monkeypatch.setattr(global_vars, '_GLOBAL_ARGS', args)
    monkeypatch.setattr(global_vars, '_GLOBAL_TIMERS', None)
    monkeypatch.setattr(global_vars, 'Timers', Mock(return_value=timers))
    global_vars._set_timers(args)
    model, optimizer, scheduler = Mock(), Mock(), Mock()

    def setup(*unused):
        assert global_vars.get_run_config() is run_config
        return model, optimizer, scheduler

    monkeypatch.setattr(finetune_utils, 'setup_model_and_optimizer', setup)
    callback = Mock()
    finetune_utils.finetune(Mock(), Mock(), end_of_epoch_callback_provider=lambda: callback)
    callback.assert_called_once_with(model, epoch=-1, output_predictions=True)
    assert global_vars.get_run_config() is run_config
    assert global_vars.get_timers() is timers


def test_tensorboard_metadata_preserves_legacy_args(monkeypatch):
    from megatron.training import initialize

    args = Namespace(iteration=13, log_interval=99, run_workload_inspector_server=False)
    writer = Mock()
    monkeypatch.setattr(initialize, 'get_args', lambda: args)
    monkeypatch.setattr(initialize, 'get_tensorboard_writer', lambda: writer)
    monkeypatch.setattr(global_vars, '_GLOBAL_RUN_CONFIG', None)
    write_args_to_tensorboard()
    assert writer.add_text.call_count == len(vars(args))
    for name, value in vars(args).items():
        writer.add_text.assert_any_call(name, str(value), global_step=13)
    assert args.log_interval == 99
    assert args.run_workload_inspector_server is False


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
def test_core_metric_flags_keep_cli_defaults_and_owners(enabled):
    import torch

    from megatron.training.argument_utils import core_transformer_config_from_args
    from megatron.training.training import get_megatron_optimizer_config

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    flags = (
        ['--log-max-attention-logit', '--log-num-zeros-in-grad', '--no-barrier-with-level-1-timing']
        if enabled
        else []
    )
    args = parser.parse_args(flags)
    args.num_layers = 2
    args.hidden_size = 32
    args.num_attention_heads = 4
    args.params_dtype = torch.float32
    args.main_grads_dtype = args.main_params_dtype = torch.float32
    args.exp_avg_dtype = args.exp_avg_sq_dtype = torch.float32
    transformer = core_transformer_config_from_args(args)
    optimizer, _ = get_megatron_optimizer_config(args)
    assert transformer.log_max_attention_logit is enabled
    assert transformer.barrier_with_L1_time is (not enabled)
    assert optimizer.log_num_zeros_in_grad is enabled
    assert optimizer.barrier_with_L1_time is (not enabled)
    names = {item.name for item in fields(LoggerConfig)}
    assert not names & {'log_max_attention_logit', 'log_num_zeros_in_grad', 'barrier_with_L1_time'}


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('model_kind', ['gpt', 'hybrid'])
def test_native_builder_keeps_core_metric_settings(monkeypatch, model_kind, enabled, run_config):
    from megatron.core.transformer import TransformerConfig
    from megatron.training import training
    from megatron.training.models import gpt, hybrid

    config_cls = gpt.GPTModelConfig if model_kind == 'gpt' else hybrid.HybridModelConfig
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        log_max_attention_logit=enabled,
        barrier_with_L1_time=enabled,
    )
    run_config.model = config_cls(transformer=transformer, vocab_size=128, seq_length=16)
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.log_max_attention_logit = args.barrier_with_L1_time = not enabled
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
        assert model_config.transformer is transformer
        assert transformer.log_max_attention_logit is enabled
        assert transformer.barrier_with_L1_time is enabled
        raise ObservedModel

    monkeypatch.setattr(config_cls, 'get_builder_cls', lambda self: construct)
    with pytest.raises(ObservedModel):
        training.setup_model_and_optimizer(Mock(), cfg_container=run_config)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('supplied_model', [False, True])
def test_attention_metric_logging_reads_model_config(
    monkeypatch, enabled, supplied_model, run_config
):
    from megatron.core.transformer import TransformerConfig
    from megatron.training import training
    from megatron.training.models import GPTModelConfig

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(['--micro-batch-size', '1'])
    args.data_parallel_size = args.world_size = args.gtp_weight_remat_size = 1
    args.consumed_train_samples = args.skipped_train_samples = 0
    args.dsa_indexer_loss_coeff = None
    args.log_max_attention_logit = not enabled
    transformer = TransformerConfig(
        num_layers=2, hidden_size=32, num_attention_heads=4, log_max_attention_logit=enabled
    )
    run_config.model = GPTModelConfig(transformer=transformer, vocab_size=128, seq_length=16)
    run_config.logger = LoggerConfig(log_interval=100, tensorboard_log_interval=1)
    writer = Mock()
    monkeypatch.setattr(training, 'get_args', lambda: args)
    monkeypatch.setattr(training, 'get_timers', Mock())
    monkeypatch.setattr(training, 'get_tensorboard_writer', lambda: writer)
    monkeypatch.setattr(training, 'get_num_microbatches', lambda: 1)
    monkeypatch.setattr(training, 'one_logger_utils', Mock())
    monkeypatch.setattr(training, 'reduce_max_stat_across_model_parallel_group', lambda x, **kw: x)
    for getter in ('get_wandb_writer', 'get_one_logger', 'get_energy_monitor', 'get_telemetry'):
        monkeypatch.setattr(training, getter, lambda: None)
    model = [SimpleNamespace(config=transformer)] if supplied_model else None
    if supplied_model:
        run_config.model = None  # Legacy and MiMo use the built model's transformer config.
    training.training_log({}, {}, 0.01, 1, 1.0, False, 0, None, None, None, 7.0, model=model)
    logged = [
        call for call in writer.add_scalar.call_args_list if call.args[0] == 'max_attention_logit'
    ]
    assert len(logged) == int(enabled)
    if enabled:
        assert logged[0].args == ('max_attention_logit', 7.0, 1)

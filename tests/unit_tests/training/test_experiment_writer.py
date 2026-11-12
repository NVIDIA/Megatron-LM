# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional tracking SDKs are mocked; no account or network access is required."""

import sys
from argparse import ArgumentParser, Namespace
from itertools import product
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.timers import Timers
from megatron.training import arguments, global_vars, training
from megatron.training.argument_utils import _default_config_from_args
from megatron.training.config import LoggerConfig
from megatron.training.logging.experiment_writer import ExperimentWriter
from megatron.training.state import TrainState


@pytest.fixture
def tracking_sdks(monkeypatch):
    mlflow = Mock()
    mlflow.active_run.return_value = None
    client = mlflow.tracking.MlflowClient.return_value
    client.create_run.return_value.info.run_id = 'run-id'
    client.get_run.return_value.data.params = {}
    mlflow.entities.Param.side_effect = lambda key, value: SimpleNamespace(key=key, value=value)
    mlflow.entities.Metric.side_effect = lambda key, value, timestamp, step: SimpleNamespace(
        key=key, value=value, timestamp=timestamp, step=step
    )
    comet = Mock()
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    monkeypatch.setitem(sys.modules, 'comet_ml', comet)
    return mlflow, comet


@pytest.fixture
def experiment_writer(run_config, tracking_sdks):
    run_config.logger = LoggerConfig(
        mlflow_experiment='experiment',
        mlflow_run_name='run',
        mlflow_tracking_uri='https://tracking.example',
        mlflow_tags={'team': 'training'},
        comet_project='project',
        comet_experiment_name='run',
        comet_workspace='workspace',
        comet_tags=['training'],
    )
    writer = ExperimentWriter(run_config, wandb_writer=Mock())
    yield writer
    writer.finish()


def test_cli_config_initializes_backends(monkeypatch, run_config, tracking_sdks):
    parser = ArgumentParser()
    arguments._add_logging_args(parser)
    args = parser.parse_args(
        [
            '--mlflow-experiment',
            'experiment',
            '--mlflow-run-name',
            'run',
            '--mlflow-tags',
            '{"team": "training"}',
            '--no-mlflow-log-artifacts',
            '--comet-project',
            'project',
            '--comet-experiment-name',
            'run',
            '--comet-tags',
            'training',
            'sft',
        ]
    )
    run_config.logger = _default_config_from_args(LoggerConfig, args)
    run_config.logger.validate()
    mlflow, comet = tracking_sdks
    writer = ExperimentWriter(run_config)
    try:
        mlflow.set_experiment.assert_called_once_with('experiment')
        assert mlflow.tracking.MlflowClient.return_value.create_run.call_args.kwargs['tags'] == {
            'team': 'training'
        }
        comet.Experiment.return_value.add_tags.assert_called_once_with(['training', 'sft'])
        writer.checkpoint_saved('/checkpoint', 7)
        mlflow.tracking.MlflowClient.return_value.log_artifacts.assert_not_called()
    finally:
        writer.finish()


@pytest.mark.parametrize(
    'wandb_enabled,mlflow_enabled,comet_enabled', list(product([False, True], repeat=3))
)
@pytest.mark.parametrize('rank', [0, 1])
def test_runtime_service_is_optional_and_last_rank_only(
    monkeypatch,
    run_config,
    tracking_sdks,
    tmp_path,
    wandb_enabled,
    mlflow_enabled,
    comet_enabled,
    rank,
):
    run_config.logger = LoggerConfig(
        wandb_project='project' if wandb_enabled else None,
        wandb_exp_name='run',
        wandb_save_dir=str(tmp_path),
        mlflow_experiment='experiment' if mlflow_enabled else None,
        mlflow_run_name='run',
        comet_project='project' if comet_enabled else None,
        comet_experiment_name='run',
    )
    wandb = Mock()
    monkeypatch.setitem(sys.modules, 'wandb', wandb)
    monkeypatch.setattr(global_vars, '_GLOBAL_EXPERIMENT_WRITER', None)
    monkeypatch.setattr(global_vars, '_GLOBAL_WANDB_WRITER', None)
    monkeypatch.setattr(global_vars, 'safe_get_world_size', lambda: 2)
    try:
        global_vars._set_experiment_writer(Namespace(rank=rank))
        assert wandb.init.call_count == int(wandb_enabled and rank == 1)
        assert tracking_sdks[0].tracking.MlflowClient.call_count == int(
            mlflow_enabled and rank == 1
        )
        assert tracking_sdks[1].Experiment.call_count == int(comet_enabled and rank == 1)
        assert (global_vars.get_experiment_writer() is not None) == (
            (wandb_enabled or mlflow_enabled or comet_enabled) and rank == 1
        )
    finally:
        global_vars.finish_experiment_logging()


@pytest.mark.parametrize('backend', ['mlflow', 'comet_ml'])
def test_enabled_backend_requires_sdk(monkeypatch, run_config, backend):
    run_config.logger = LoggerConfig(
        mlflow_experiment='experiment' if backend == 'mlflow' else None,
        mlflow_run_name='run',
        comet_project='project' if backend == 'comet_ml' else None,
        comet_experiment_name='run',
    )
    monkeypatch.setitem(sys.modules, backend, None)
    with pytest.raises(ImportError, match='pip install'):
        ExperimentWriter(run_config)


def test_scalar_fanout_and_timer_interface(experiment_writer, tracking_sdks, monkeypatch):
    mlflow, comet = tracking_sdks
    metrics = {'loss[layer=0]': torch.tensor(1.5)}
    experiment_writer.log(metrics, step=7)
    experiment_writer.wandb.log.assert_called_once_with(metrics, 7)
    assert_mlflow_metric(mlflow, 'loss_layer_0_', 1.5, 7)
    comet.Experiment.return_value.log_metrics.assert_called_once_with(
        {'loss[layer=0]': 1.5}, step=7
    )
    timers = Timers(0, 'minmax')
    monkeypatch.setattr(timers, '_get_global_min_max_time', lambda *a: {'forward': (1.0, 2.0)})
    timers.write(['forward'], experiment_writer, iteration=8)
    assert_mlflow_metric(mlflow, 'forward-time', 2.0, 8)
    comet.Experiment.return_value.log_metrics.assert_called_with({'forward-time': 2.0}, step=8)


def test_checkpoint_reporting_and_idempotent_shutdown(experiment_writer, tracking_sdks, tmp_path):
    mlflow, comet = tracking_sdks
    experiment_writer.checkpoint_saved(str(tmp_path), 7)
    mlflow.tracking.MlflowClient.return_value.log_artifacts.assert_called_once_with(
        'run-id', str(tmp_path), artifact_path=f'checkpoints/{tmp_path.name}'
    )
    comet.Experiment.return_value.log_other.assert_any_call('last_saved_iteration', 7)
    experiment_writer.checkpoint_loaded('msc://store/checkpoint', 'msc://store')
    mlflow.tracking.MlflowClient.return_value.set_tag.assert_any_call(
        'run-id', 'last_loaded_checkpoint', 'msc://store/checkpoint'
    )
    mlflow.tracking.MlflowClient.return_value.log_artifacts.side_effect = RuntimeError(
        'tracking service unavailable'
    )
    # Artifact failures must not strand other ranks at a checkpoint barrier.
    experiment_writer.checkpoint_saved(str(tmp_path), 8)
    experiment_writer.finish('KILLED')
    experiment_writer.finish()
    mlflow.tracking.MlflowClient.return_value.set_terminated.assert_called_once_with(
        'run-id', status='KILLED'
    )
    experiment_writer.wandb.finish.assert_called_once_with(exit_code=1)
    comet.Experiment.return_value.end.assert_called_once()


@pytest.mark.parametrize('external_run', [False, True])
def test_mlflow_resume_keeps_original_parameters(run_config, tracking_sdks, external_run):
    mlflow, _ = tracking_sdks
    run_config.logger = LoggerConfig(
        mlflow_experiment='experiment', mlflow_run_name='run', mlflow_run_id='run-id'
    )
    mlflow.tracking.MlflowClient.return_value.get_run.return_value.data.params = {
        'logger.mlflow_run_name': 'original'
    }
    if external_run:
        mlflow.active_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id='run-id'))
    writer = ExperimentWriter(run_config)
    writer.finish()
    client = mlflow.tracking.MlflowClient.return_value
    params = {
        p.key: p.value
        for batch in client.log_batch.call_args_list
        for p in batch.kwargs.get('params', [])
    }
    assert 'logger.mlflow_run_name' not in params
    assert 'logger.mlflow_tracking_uri' not in params
    assert client.update_run.call_count == int(not external_run)
    assert client.set_terminated.call_count == int(not external_run)


def test_uncaught_exception_marks_runs_failed(monkeypatch, run_config, tracking_sdks):
    previous = Mock()
    monkeypatch.setattr(sys, 'excepthook', previous)
    run_config.logger = LoggerConfig(comet_project='project', comet_experiment_name='run')
    writer = ExperimentWriter(run_config)
    error = RuntimeError('training failed')
    sys.excepthook(type(error), error, None)
    assert sys.excepthook is previous
    previous.assert_called_once_with(type(error), error, None)
    tracking_sdks[1].Experiment.return_value.log_other.assert_called_with(
        'training_status', 'FAILED'
    )
    tracking_sdks[1].Experiment.return_value.end.assert_called_once()
    writer.finish()


def test_config_parameters_accept_mapping_keys(run_config, tracking_sdks, monkeypatch):
    run_config.logger = LoggerConfig(mlflow_experiment='experiment', mlflow_run_name='run')
    monkeypatch.setattr(
        run_config,
        'to_dict',
        lambda: {'layers': {0: {'pattern[*]': 'value'}}, 'api_key': 'not-a-real-secret'},
    )
    writer = ExperimentWriter(run_config)
    try:
        params = tracking_sdks[0].tracking.MlflowClient.return_value.log_batch.call_args.kwargs[
            'params'
        ]
        assert [(p.key, p.value) for p in params] == [('layers.0.pattern___', 'value')]
    finally:
        writer.finish()


@pytest.mark.parametrize('tensorboard_enabled', [False, True])
@pytest.mark.parametrize('write_to_tensorboard', [False, True])
def test_training_and_evaluation_log_without_tensorboard(
    monkeypatch,
    run_config,
    experiment_writer,
    tracking_sdks,
    tensorboard_enabled,
    write_to_tensorboard,
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(['--micro-batch-size', '1'])
    args.data_parallel_size = args.world_size = args.gtp_weight_remat_size = 1
    args.dsa_indexer_loss_coeff = None
    run_config.logger.log_interval = 100
    run_config.logger.tensorboard_log_interval = 1
    run_config.train.micro_batch_size = 1
    run_config.validation.eval_iters = 1
    monkeypatch.setattr(training, 'get_args', lambda: args)
    monkeypatch.setattr(training, 'get_train_state', lambda: TrainState())
    monkeypatch.setattr(training, 'get_timers', Mock())
    tensorboard_writer = Mock() if tensorboard_enabled else None
    monkeypatch.setattr(training, 'get_tensorboard_writer', lambda: tensorboard_writer)
    monkeypatch.setattr(training, 'get_experiment_writer', lambda: experiment_writer)
    monkeypatch.setattr(training, 'get_num_microbatches', lambda: 1)
    monkeypatch.setattr(training, 'one_logger_utils', Mock())
    monkeypatch.setattr(training, 'reduce_max_stat_across_model_parallel_group', lambda x, **kw: x)
    monkeypatch.setattr(training, 'is_last_rank', lambda: True)
    for getter in ('get_one_logger', 'get_energy_monitor', 'get_telemetry'):
        monkeypatch.setattr(training, getter, lambda: None)
    training.training_log(
        {'lm loss': torch.tensor(1.5)}, {}, 0.01, 1, 1.0, False, 0, None, None, None, None
    )
    mlflow, comet = tracking_sdks
    assert_mlflow_metric(mlflow, 'lm loss', 1.5, 1)
    comet.Experiment.return_value.log_metrics.assert_any_call({'learning-rate': 0.01}, step=1)
    monkeypatch.setattr(
        training, 'evaluate', lambda *a, **kw: ({'lm loss': torch.tensor(2.0)}, None, False)
    )
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar.assert_any_call('lm loss', torch.tensor(1.5), 1)
        tensorboard_writer.reset_mock()
    training.evaluate_and_print_results(
        'test', None, None, [], 2, None, None, write_to_tensorboard=write_to_tensorboard
    )
    experiment_writer.wandb.log.assert_any_call({'lm loss': torch.tensor(1.5)}, 1)
    experiment_writer.wandb.log.assert_any_call({'lm loss validation': 2.0}, 2)
    if tensorboard_writer is not None:
        assert bool(tensorboard_writer.add_scalar.call_count) == write_to_tensorboard
    assert_mlflow_metric(mlflow, 'lm loss validation', 2.0, 2)
    comet.Experiment.return_value.log_metrics.assert_any_call({'lm loss validation': 2.0}, step=2)


def assert_mlflow_metric(mlflow, key, value, step):
    metrics = [
        (m.key, m.value, m.step)
        for batch in mlflow.tracking.MlflowClient.return_value.log_batch.call_args_list
        for m in batch.kwargs.get('metrics', [])
    ]
    assert (key, value, step) in metrics


def test_finish_from_another_thread_uses_owned_run(experiment_writer, tracking_sdks):
    thread = Thread(target=experiment_writer.finish, args=('FAILED',))
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    tracking_sdks[0].tracking.MlflowClient.return_value.set_terminated.assert_called_once_with(
        'run-id', status='FAILED'
    )
    experiment_writer.wandb.finish.assert_called_once_with(exit_code=1)
    # No fluent active-run stack is created on the training thread.
    tracking_sdks[0].start_run.assert_not_called()


@pytest.mark.parametrize(
    'error,status',
    [
        (SystemExit(1), 'FAILED'),
        (SystemExit('missing checkpoint'), 'FAILED'),
        (SystemExit(0), 'KILLED'),
        (KeyboardInterrupt(), 'KILLED'),
        (RuntimeError('training failed'), 'FAILED'),
    ],
)
def test_training_boundary_records_failed_attempt(
    monkeypatch, experiment_writer, tracking_sdks, error, status
):
    monkeypatch.setattr(global_vars, '_GLOBAL_EXPERIMENT_WRITER', experiment_writer)

    @global_vars.with_experiment_logging
    def training_attempt():
        raise error

    with pytest.raises(type(error)):
        training_attempt()
    tracking_sdks[0].tracking.MlflowClient.return_value.set_terminated.assert_called_once_with(
        'run-id', status=status
    )
    tracking_sdks[1].Experiment.return_value.log_other.assert_called_with('training_status', status)
    experiment_writer.wandb.finish.assert_called_once_with(exit_code=1)


def test_mutable_metadata_reaches_all_backends(experiment_writer, tracking_sdks):
    experiment_writer.log_metadata({'slurm_job_name': 'resumed-job'})
    experiment_writer.wandb.config.update.assert_called_once_with(
        {'slurm_job_name': 'resumed-job'}, allow_val_change=True
    )
    tracking_sdks[0].tracking.MlflowClient.return_value.set_tag.assert_any_call(
        'run-id', 'slurm_job_name', 'resumed-job'
    )
    tracking_sdks[1].Experiment.return_value.log_other.assert_called_with(
        'slurm_job_name', 'resumed-job'
    )


@pytest.mark.parametrize('standalone', [False, True])
def test_evaluation_steps_reach_all_backends(experiment_writer, tracking_sdks, standalone):
    experiment_writer.log_evaluation({'reward': 1.5}, 7, standalone=standalone)
    key = 'eval_only/reward' if standalone else 'reward'
    assert_mlflow_metric(tracking_sdks[0], key, 1.5, 7)
    values = {key: 1.5}
    if standalone:
        values['eval_checkpoint_iter'] = 7
        experiment_writer.wandb.define_metric.assert_called_once_with(
            'eval_only/*', step_metric='eval_checkpoint_iter'
        )
        experiment_writer.wandb.log.assert_called_once_with(values)
    else:
        experiment_writer.wandb.log.assert_called_once_with(values, 7)
    tracking_sdks[1].Experiment.return_value.log_metrics.assert_called_once_with(values, step=7)


def test_inference_axis_and_resume_reach_all_backends(experiment_writer, tracking_sdks):
    experiment_writer.wandb.Api.return_value.run.return_value.scan_history.return_value = [
        {'inference/inference_step': 9},
        {'inference/inference_step': 3},
    ]
    tracking_sdks[0].tracking.MlflowClient.return_value.get_metric_history.return_value = [
        SimpleNamespace(value=12),
        SimpleNamespace(value=4),
    ]
    assert experiment_writer.start_inference_logging() == 12
    experiment_writer.wandb.define_metric.assert_called_once_with(
        'inference/*', step_metric='inference/inference_step'
    )
    metrics = {'inference/inference_step': 13, 'inference/active_request_count': 2}
    experiment_writer.log_inference(metrics)
    experiment_writer.wandb.log.assert_called_once_with(metrics, commit=True)
    assert_mlflow_metric(tracking_sdks[0], 'inference/active_request_count', 2, 13)
    tracking_sdks[1].Experiment.return_value.log_metrics.assert_called_once_with(metrics, step=13)

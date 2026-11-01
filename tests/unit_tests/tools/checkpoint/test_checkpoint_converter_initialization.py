# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The checkpoint loader and saver set up torch.distributed and their process groups."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core import parallel_state
from megatron.training.state import TrainState

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))

from loader_base import MegatronCheckpointLoaderBase
from saver_base import MegatronCheckpointSaverBase


def _describe_environment(converter):
    """torch.distributed and the global process groups that a converter has set up."""
    return dict(
        world_size=torch.distributed.get_world_size(),
        registered=parallel_state.get_registered_process_groups() is converter.pg_collection,
        tp_size=parallel_state.get_tensor_model_parallel_world_size(),
        pp_size=parallel_state.get_pipeline_model_parallel_world_size(),
    )


def _initialize_loader_environment():
    """Set up a loader for TP=2, PP=2, EP=1 as its process does and describe the result."""
    loader = MegatronCheckpointLoaderBase(args=None, queue=None)
    loader.margs = SimpleNamespace(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=2, expert_model_parallel_size=1
    )
    with (
        mock.patch('megatron.training.argument_utils.inference_cfg_container_from_args'),
        mock.patch('megatron.training.global_vars.set_global_variables'),
    ):
        loader.initialize_megatron_env()
    return _describe_environment(loader)


def _initialize_saver_environment():
    """Set up a saver for TP=2, PP=2, EP=1 as its process does and describe the result."""
    saver = MegatronCheckpointSaverBase(
        args=SimpleNamespace(
            target_tensor_parallel_size=2,
            target_pipeline_parallel_size=2,
            target_expert_parallel_size=1,
        ),
        queue=None,
    )
    saver.margs = SimpleNamespace()
    saver.md = SimpleNamespace(iteration=3)
    saver.import_model_provider = lambda: None
    with (
        mock.patch('megatron.training.argument_utils.inference_cfg_container_from_args'),
        mock.patch('megatron.training.global_vars.set_global_variables'),
        mock.patch('megatron.training.global_vars.get_args', return_value=saver.margs),
        mock.patch('megatron.training.global_vars.get_train_state', return_value=TrainState()),
    ):
        saver.initialize_megatron_env()
    return _describe_environment(saver)


def _run_in_new_process(function_name, env):
    """Run a function of this module in a process where torch.distributed is not initialized."""
    script = f"import json; import {__name__} as t; print(json.dumps(t.{function_name}()))"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_saver_starts_torch_distributed_without_a_port():
    """The saver starts torch.distributed even when MASTER_PORT is taken, for example by the
    saver of another conversion on the same host."""
    # The store server of another process holds the port.
    store = torch.distributed.TCPStore('localhost', 0, world_size=1, is_master=True)
    env = dict(os.environ, MASTER_ADDR='localhost', MASTER_PORT=str(store.port))

    description = _run_in_new_process('_initialize_saver_environment', env)

    assert description == dict(world_size=1, registered=True, tp_size=2, pp_size=2)


def test_loader_registers_its_collection():
    """The loader registers the collection it passes to its models with parallel_state."""
    description = _run_in_new_process('_initialize_loader_environment', dict(os.environ))

    assert description == dict(world_size=1, registered=True, tp_size=2, pp_size=2)

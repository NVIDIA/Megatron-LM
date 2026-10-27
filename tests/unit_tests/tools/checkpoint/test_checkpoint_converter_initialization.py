# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The checkpoint loader and saver set up torch.distributed and their process groups."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import torch

from megatron.training.state import TrainState

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))

from saver_base import MegatronCheckpointSaverBase


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
    return dict(
        world_size=torch.distributed.get_world_size(),
        tp_size=saver.pg_collection.tp.size(),
        pp_size=saver.pg_collection.pp.size(),
    )


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

    assert description == dict(world_size=1, tp_size=2, pp_size=2)

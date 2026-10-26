# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import inspect
from types import SimpleNamespace

import pytest

from megatron.training import training
from megatron.training.arguments import _add_distributed_args


def test_pipeline_prewarm_cli_flag():
    parser = _add_distributed_args(argparse.ArgumentParser())
    assert parser.parse_args([]).pipeline_model_parallel_prewarm is False
    assert parser.parse_args(['--pipeline-model-parallel-prewarm']).pipeline_model_parallel_prewarm


@pytest.mark.parametrize('prewarm', [False, True])
@pytest.mark.parametrize('fsdp', [None, 'use_megatron_fsdp', 'use_torch_fsdp2'])
def test_training_rejects_prewarm_with_fsdp_before_initialization(monkeypatch, prewarm, fsdp):
    args = SimpleNamespace(**({fsdp: True} if fsdp is not None else {}))
    monkeypatch.setattr(training, 'get_args', lambda: args)

    class InitializationReached(Exception):
        pass

    def get_timers():
        raise InitializationReached()

    monkeypatch.setattr(training, 'get_timers', get_timers)
    kwargs = {
        name: None
        for name, param in inspect.signature(training.train).parameters.items()
        if param.default is inspect.Parameter.empty
    }
    kwargs['config'] = SimpleNamespace(pipeline_model_parallel_prewarm=prewarm)
    if prewarm and fsdp is not None:
        with pytest.raises(ValueError, match='does not support Megatron-FSDP or Torch-FSDP2'):
            training.train(**kwargs)
    else:
        with pytest.raises(InitializationReached):
            training.train(**kwargs)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Missing state and corrupted wakeups must fail before rollout can proceed."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples/verl'))
from verl_mlite import compat

pytestmark = pytest.mark.optional


@pytest.fixture
def worker(monkeypatch):
    module = ModuleType('vllm.v1.worker.gpu_worker')

    class Worker:
        def __init__(self):
            layer = torch.nn.Module()
            layer.register_parameter('bias_vl', torch.nn.Parameter(torch.tensor([2.0, 3.0])))
            layer.register_buffer('state', torch.tensor([5.0, 7.0]))
            self.model_runner = SimpleNamespace(model=torch.nn.Sequential(layer))
            self.draft = torch.nn.Linear(2, 2)
            self.draft.register_buffer('state', torch.tensor([11.0, 13.0]))
            self.model_config = SimpleNamespace(
                sleep_preserve_parameter_names=['*.bias_vl', 'draft.*']
            )
            self._sleep_saved_parameters = {}
            self._sleep_saved_buffers = {}
            self._sleep_saved_draft_buffers = {}

        def get_draft_model(self):
            return self.draft

        def _get_sleep_parameters(self, model):
            out = dict(model.named_parameters())
            if self.draft is not None:
                out.update((f'draft.{n}', p) for n, p in self.draft.named_parameters())
            return out

        def sleep(self, level=1):
            if level != 2:
                return
            self._sleep_saved_buffers = {
                n: b.clone() for n, b in self.model_runner.model.named_buffers()
            }
            self._sleep_saved_draft_buffers = (
                {} if self.draft is None else {n: b.clone() for n, b in self.draft.named_buffers()}
            )
            # CPU parameters survive natively; buffers model the save/restore path.
            for b in list(self.model_runner.model.buffers()) + (
                [] if self.draft is None else list(self.draft.buffers())
            ):
                b.zero_()

        def wake_up(self, tags=None):
            if tags is not None and 'weights' not in tags:
                return
            for owner, saved in (
                (self.model_runner.model, self._sleep_saved_buffers),
                (self.draft, self._sleep_saved_draft_buffers),
            ):
                if owner is not None:
                    for n, b in owner.named_buffers():
                        if n in saved:
                            b.copy_(saved[n])
                saved.clear()

    module.Worker = Worker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv('VERL_MLITE_ROLLOUT_SLEEP_LEVEL', '2')
    monkeypatch.setattr(compat, '_vllm_importable', lambda: True)
    assert compat._patch_vllm_sleep_preserve_check()
    assert compat._patch_vllm_sleep_preserve_check()
    return Worker()


def test_split_wake_and_repeated_cycles(worker):
    worker.wake_up(['weights'])  # fresh worker has no outstanding sleep
    for _ in range(2):
        worker.sleep(2)
        worker.wake_up(['kv_cache'])
        assert worker._verl_mlite_l2_pending
        worker.wake_up(['weights'])
        assert not worker._verl_mlite_l2_pending
    worker.wake_up()


@pytest.mark.parametrize(
    'patterns', [[], ['draft.*'], ['*.bias_vl'], ['*.bias_vl', 'draft.*', 'missing.*']]
)
def test_missing_required_pattern_or_tensor_fails(worker, patterns):
    worker.model_config.sleep_preserve_parameter_names = patterns
    with pytest.raises(RuntimeError):
        worker.sleep(2)


def test_missing_draft_fails(worker):
    worker.draft = None
    with pytest.raises(RuntimeError, match='no tensor'):
        worker.sleep(2)


def test_spec_off_preserves_bias_without_draft(worker):
    worker.draft = None
    worker.model_config.sleep_preserve_parameter_names = ['*.bias_vl']
    worker.sleep(2)
    worker.wake_up()


@pytest.mark.parametrize('kind', ['buffer', 'parameter', 'draft', 'shape', 'dtype', 'removed'])
def test_corruption_or_missing_state_fails(worker, kind):
    worker.sleep(2)
    if kind == 'buffer':
        worker._sleep_saved_buffers['0.state'][0] += 1
    elif kind == 'parameter':
        with torch.no_grad():
            worker.model_runner.model[0].bias_vl[0] += 1
    elif kind == 'draft':
        with torch.no_grad():
            worker.draft.weight[0, 0] += 1
    elif kind == 'shape':
        worker.model_runner.model[0].state = torch.zeros(1)
    elif kind == 'dtype':
        worker.model_runner.model[0].state = torch.zeros(2, dtype=torch.float64)
    else:
        del worker.model_runner.model[0].state
    with pytest.raises((RuntimeError, KeyError)):
        worker.wake_up(['weights'])
    assert worker._verl_mlite_l2_pending


def test_missing_backup_fails_even_if_restored_value_would_be_zero(worker):
    worker.sleep(2)
    worker._sleep_saved_buffers.clear()
    with pytest.raises(RuntimeError, match='backups differ'):
        worker.wake_up()


def test_repeated_sleep_before_verification_fails(worker):
    worker.sleep(2)
    with pytest.raises(RuntimeError, match='repeated'):
        worker.sleep(2)


def test_level_one_debug_path_is_unchanged(worker):
    worker.sleep(1)
    worker.wake_up()
    assert not getattr(worker, '_verl_mlite_l2_pending', False)


def test_level_one_never_imports_cuda_worker(monkeypatch):
    monkeypatch.setenv('VERL_MLITE_ROLLOUT_SLEEP_LEVEL', '1')
    monkeypatch.setattr(
        compat, '_vllm_importable', lambda: pytest.fail('L1 must not import GPU worker')
    )
    assert not compat._patch_vllm_sleep_preserve_check()

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Algorithm selection, explicit callback ownership and shared training-loss checks."""

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.training import losses
from megatron.training.config import LogitsLoadConfig, LogitsSaveConfig, OfflineDistillationConfig


def _config(offline=None):
    return SimpleNamespace(
        offline_distillation=offline,
        distillation=None,
        rerun_state_machine=SimpleNamespace(
            check_for_nan_in_loss=False, check_for_spiky_loss=False, spiky_loss_factor=10
        ),
    )


@pytest.mark.parametrize('mask', [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
def test_lm_loss_preserves_sum_tokens_reporting_and_gradients(mask):
    output = torch.tensor([2.0, 5.0, 7.0], requires_grad=True)
    mask = torch.tensor(mask)
    loss, tokens, report = losses.language_model_loss(mask, output)
    assert loss.item() == (9 if mask.any() else 0)
    assert tokens.item() == mask.sum().item()
    torch.testing.assert_close(report['lm loss'], torch.stack([loss.detach(), tokens]))
    assert not report['lm loss'].requires_grad
    loss.backward()
    torch.testing.assert_close(output.grad, mask)


def test_plain_and_teacher_save_work_without_modelopt(monkeypatch):
    original_import = builtins.__import__

    def without_modelopt(name, *args, **kwargs):
        if name == 'megatron.post_training.utils':
            raise ImportError('ModelOpt is not installed')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_modelopt)
    for offline in (
        None,
        OfflineDistillationConfig(teacher_save=LogitsSaveConfig(directory='/cache')),
    ):
        callback = losses.build_loss_func(_config(offline))
        loss, tokens, _ = callback(torch.ones(2), torch.tensor([2.0, 3.0]))
        assert loss.item() == 5 and tokens.item() == 2


def test_caller_owns_and_reuses_offline_reader(monkeypatch):
    from megatron.training import distillation

    readers = []

    class Reader:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.calls = 0
            readers.append(self)

        def __call__(self, mask, output, model):
            self.calls += 1
            return losses.language_model_loss(mask, output * self.calls)

    monkeypatch.setattr(distillation, 'LossFuncCallable', Reader)
    load = LogitsLoadConfig(directory='/cache', loss_alpha=0.7, decode_threads=2)
    cfg = _config(OfflineDistillationConfig(student_load=load))
    model = torch.nn.Linear(1, 1)
    original_attributes = set(vars(model))
    original_config = set(vars(cfg))
    callback = losses.build_loss_func(cfg)
    for expected in (2, 4):
        loss, _, _ = callback(torch.ones(1), torch.tensor([2.0]), model=model)
        assert loss.item() == expected
    assert readers[0].options == dict(
        logprobs_dir='/cache',
        decode_threads=2,
        msc_prefetch_depth=2,
        kd_loss_alpha=0.7,
        ignore_errors=False,
        ignore_hash=False,
    )
    assert set(vars(model)) == original_attributes
    assert set(vars(cfg)) == original_config
    # Another explicit construction gets a fresh reader, with no hidden cache.
    another = losses.build_loss_func(cfg)
    assert another(torch.ones(1), torch.tensor([2.0]), model=model)[0].item() == 2
    assert len(readers) == 2


def test_online_kd_uses_actual_model_and_preserves_eval_behavior(monkeypatch):
    pytest.importorskip("modelopt.torch.distill")
    from megatron.post_training import loss_func as online

    class Model(torch.nn.Module):
        def compute_kd_loss(self, student_loss, loss_reduction_fn):
            intermediate = loss_reduction_fn(torch.tensor([3.0, 9.0]))
            return dict(
                kd_loss=student_loss + intermediate,
                logits_loss=student_loss,
                intermediate_loss=intermediate,
            )

    utils = ModuleType('megatron.post_training.utils')
    utils.is_distillation_model = lambda model: isinstance(model, Model)
    monkeypatch.setitem(sys.modules, 'megatron.post_training.utils', utils)
    monkeypatch.setattr(online, 'is_distillation_model', utils.is_distillation_model)
    model = Model()
    cfg = _config()
    cfg.distillation = object()
    callback = losses.build_loss_func(cfg)
    mask, output = torch.tensor([1.0, 0.0]), torch.tensor([2.0, 4.0])
    loss, tokens, report = callback(mask, output, model=model)
    assert loss.item() == 5 and tokens.item() == 1
    assert report['intermediate distillation loss'][0].item() == 3
    model.eval()
    assert callback(mask, output, model=model)[0].item() == 2


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 100.0])
@pytest.mark.parametrize('algorithm', ['lm', 'offline', 'online'])
def test_shared_checks_validate_the_selected_loss(monkeypatch, value, algorithm):
    # Use non-LM values for KD, ensuring checks see the selected objective, not the LM tensor.
    def kd(mask, output, model):
        return losses.language_model_loss(mask, output * value)

    utils = ModuleType('megatron.post_training.utils')
    utils.is_distillation_model = lambda model: algorithm == 'online'
    monkeypatch.setitem(sys.modules, 'megatron.post_training.utils', utils)
    if algorithm == 'online':
        online = ModuleType('megatron.post_training.loss_func')
        online.loss_func = kd
        monkeypatch.setitem(sys.modules, 'megatron.post_training.loss_func', online)
    offline = None
    if algorithm == 'offline':
        from megatron.training import distillation

        monkeypatch.setattr(distillation, 'LossFuncCallable', lambda **kwargs: kd)
        offline = OfflineDistillationConfig(student_load=LogitsLoadConfig(directory='/cache'))
    cfg = _config(offline)
    if algorithm == 'online':
        cfg.distillation = object()
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.rerun_state_machine.check_for_spiky_loss = True
    checks = []

    def validate_result(*, result, rejection_func, fatal, **kwargs):
        if rejection_func(result):
            checks.append(fatal)

    monkeypatch.setattr(
        losses,
        'get_rerun_state_machine',
        lambda: SimpleNamespace(
            validate_result=validate_result,
            is_unexpectedly_large=lambda result, threshold, context: result > threshold,
        ),
    )
    callback = losses.build_loss_func(cfg)
    callback(torch.ones(1), torch.tensor([value if algorithm == 'lm' else 1.0]))
    if value == 100:
        assert checks == [False]
    else:
        assert True in checks


def test_unconfigured_loss_does_not_inspect_model_or_import_modelopt(monkeypatch):
    original_import = builtins.__import__

    def no_modelopt(name, *args, **kwargs):
        if name.startswith('megatron.post_training'):
            pytest.fail('An unconfigured LM loss must not import ModelOpt helpers')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_modelopt)
    # A distillation-looking model still uses basic LM loss without a config request.
    model = SimpleNamespace(compute_kd_loss=lambda **kwargs: pytest.fail('unexpected KD'))
    callback = losses.build_loss_func(_config())
    assert callback(torch.ones(1), torch.tensor([3.0]), model=model)[0].item() == 3

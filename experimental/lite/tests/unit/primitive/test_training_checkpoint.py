from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from megatron.lite.primitive.ckpt import dcp
from megatron.lite.primitive.protocols import default_expert_classifier, default_placement_fn
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle


def _assert_state_equal(actual, expected) -> None:
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_state_equal(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_optimizer_checkpoint_roundtrips_rank_local_state(tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    loss = model(torch.ones(3, 4)).sum()
    loss.backward()
    optimizer.step()

    expected = copy.deepcopy(optimizer.state_dict())
    dcp._save_optimizer_checkpoint(optimizer, str(tmp_path))

    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()

    dcp._load_optimizer_checkpoint(optimizer, str(tmp_path))

    assert (tmp_path / "optimizer_rank_0.pt").exists()
    _assert_state_equal(optimizer.state_dict(), expected)


def test_runtime_checkpoint_api_passes_current_training_checkpoint_signature(monkeypatch, tmp_path) -> None:
    calls = {}

    def fake_save(model, optimizer, step, path, config, ps, **kwargs):
        calls["save"] = (model, optimizer, step, path, config, ps, kwargs)

    def fake_load(model, optimizer, path, config, ps, **kwargs):
        calls["load"] = (model, optimizer, path, config, ps, kwargs)
        return 7

    monkeypatch.setattr("megatron.lite.primitive.ckpt.save_training_checkpoint", fake_save)
    monkeypatch.setattr("megatron.lite.primitive.ckpt.load_training_checkpoint", fake_load)

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    model = torch.nn.Linear(1, 1)
    optimizer = object()
    parallel = SimpleNamespace(tp=1, etp=1, ep=1, pp=1, cp=1)
    ps = object()
    handle = ModelHandle(
        model=model,
        optimizer=optimizer,
        parallel_state=ps,
        config=SimpleNamespace(parallel=parallel),
    )

    runtime.save_checkpoint(
        handle,
        str(tmp_path),
        global_step=7,
        save_model=True,
        save_optimizer=False,
    )
    loaded_step = runtime.load_checkpoint(
        handle,
        str(tmp_path),
        load_model=False,
        load_optimizer=True,
    )

    assert calls["save"] == (
        model,
        optimizer,
        7,
        str(tmp_path),
        parallel,
        ps,
        {
            "get_placements": default_placement_fn,
            "is_expert": default_expert_classifier,
            "use_dcp": True,
            "save_rng": True,
            "save_model": True,
            "save_optimizer": False,
        },
    )
    assert calls["load"] == (
        model,
        optimizer,
        str(tmp_path),
        parallel,
        ps,
        {
            "get_placements": default_placement_fn,
            "is_expert": default_expert_classifier,
            "use_dcp": True,
            "load_rng": True,
            "load_parameter_state_update_legacy_format": False,
            "load_model": False,
            "load_optimizer": True,
        },
    )
    assert loaded_step == 7

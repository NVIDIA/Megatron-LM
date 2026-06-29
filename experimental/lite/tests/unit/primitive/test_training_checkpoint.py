# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

pytest.importorskip("megatron.core.dist_checkpointing")

from megatron.core.dist_checkpointing.strategies.torch import (
    _replace_state_dict_keys_with_sharded_keys,
)
from megatron.lite.primitive.ckpt import dcp
from megatron.lite.primitive.ckpt.distckpt import (
    _model_sharded_state_dict,
    _rank_offsets_and_replica_id,
    _single_or_all_model_state,
    _synchronize_native_optimizer_steps,
    attach_model_sharded_state_dict,
)
from megatron.lite.primitive.parallel import ParallelState
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


class FakeDistOpt:
    def __init__(self):
        self.save_model_sd = None
        self.load_model_sd = None
        self.loaded_state = None

    def sharded_state_dict(self, model_sd, is_loading: bool = False, metadata=None):
        assert metadata == DISTOPT_METADATA
        if is_loading:
            self.load_model_sd = model_sd
        else:
            self.save_model_sd = model_sd
        return {"is_loading": is_loading}

    def load_state_dict(self, state):
        self.loaded_state = state


class FakeWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.wrapper_load_called = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.wrapper_load_called = True
        return super().load_state_dict(*args, **kwargs)


DISTOPT_METADATA = {
    "distrib_optim_sharding_type": "fully_reshardable",
    "distrib_optim_fully_reshardable_mem_efficient": False,
    "chained_optim_avoid_prefix": True,
}


def test_dist_opt_checkpoint_dispatches_to_mcore_distckpt(monkeypatch, tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = FakeDistOpt()
    ps = ParallelState(pp_rank=1, tp_rank=2, dp_cp_rank=3)
    attach_model_sharded_state_dict([model], ps)
    saved = {}

    def fake_save(state_dict, checkpoint_dir, **kwargs):
        saved["state_dict"] = state_dict
        saved["checkpoint_dir"] = checkpoint_dir
        saved["kwargs"] = kwargs

    monkeypatch.setattr("megatron.lite.primitive.ckpt.distckpt.dist_checkpointing.save", fake_save)

    dcp.save_training_checkpoint(model, optimizer, 5, str(tmp_path), use_dcp=True)

    model_sd = saved["state_dict"]["model"]
    assert set(model_sd) == {"weight", "bias"}
    assert model_sd["weight"].replica_id == (0, 2, 3)
    assert optimizer.save_model_sd is model_sd
    assert saved["state_dict"]["optimizer"] == {"is_loading": False}
    assert saved["state_dict"]["step"] == 5
    assert saved["checkpoint_dir"] == str(tmp_path / "step_5")
    assert saved["kwargs"]["validate_access_integrity"] is False
    assert saved["kwargs"]["content_metadata"] == DISTOPT_METADATA
    assert not (tmp_path / "step_5" / "optimizer_rank_0.pt").exists()


def test_dist_opt_checkpoint_offsets_cover_tp_pp_ep_etp_topology() -> None:
    ps = ParallelState(
        pp_size=2,
        pp_rank=1,
        tp_size=2,
        tp_rank=1,
        ep_size=2,
        ep_rank=1,
        etp_size=2,
        etp_rank=1,
        dp_size=2,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        dp_cp_rank=0,
        expert_dp_size=1,
        expert_dp_rank=0,
    )

    dense_offsets, dense_replica = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Replicate(), Shard(0)], ps, expert=False
    )
    expert_offsets, expert_replica = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Shard(0), Shard(0)], ps, expert=True
    )

    assert dense_offsets == ((0, 1, 2),)
    assert dense_replica == (0, 0, 0)
    assert expert_offsets == ((0, 3, 4),)
    assert expert_replica == (0, 0, 0)


def test_dist_opt_replica_id_groups_sharded_axes_by_placement() -> None:
    placements = [Replicate(), Replicate(), Replicate(), Shard(0)]
    rank_offsets0, replica_id0 = _rank_offsets_and_replica_id(
        placements, ParallelState(tp_size=2, tp_rank=0), expert=False
    )
    rank_offsets1, replica_id1 = _rank_offsets_and_replica_id(
        placements, ParallelState(tp_size=2, tp_rank=1), expert=False
    )

    assert rank_offsets0 == ((0, 0, 2),)
    assert rank_offsets1 == ((0, 1, 2),)
    assert replica_id0 == replica_id1 == (0, 0, 0)

    expert_offsets, expert_replica_id = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Shard(0), Shard(1)],
        ParallelState(ep_size=2, ep_rank=1, etp_size=2, etp_rank=1),
        expert=True,
    )

    assert expert_offsets == ((0, 1, 2), (1, 1, 2))
    assert expert_replica_id == (0, 0, 0)


def test_dist_opt_replica_id_does_not_treat_pp_as_a_replica_axis() -> None:
    rank_offsets, replica_id = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Replicate(), Shard(0)],
        ParallelState(pp_size=2, pp_rank=1, tp_size=2, tp_rank=1),
        expert=False,
    )

    assert rank_offsets == ((0, 1, 2),)
    assert replica_id == (0, 0, 0)

    _rank_offsets, replica_id = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Replicate(), Replicate()],
        ParallelState(pp_size=2, pp_rank=1, tp_size=2, tp_rank=0),
        expert=False,
    )

    assert replica_id == (0, 0, 0)


def test_dist_opt_pp_rank_one_model_keys_survive_torch_dist_main_replica_filter() -> None:
    ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)
    model = torch.nn.Linear(4, 2)
    attach_model_sharded_state_dict([model], ps)

    model_sd = _model_sharded_state_dict(model)
    filtered_sd, _flat_mapping, _rename_mapping = _replace_state_dict_keys_with_sharded_keys(
        model_sd, keep_only_main_replica=True
    )

    assert set(filtered_sd) == {"model_pp1.weight", "model_pp1.bias"}


def test_dist_opt_model_state_keys_are_pp_and_vpp_aware() -> None:
    ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)
    single_chunk = torch.nn.Linear(4, 2)
    attach_model_sharded_state_dict([single_chunk], ps)

    single_sd = _model_sharded_state_dict(single_chunk)

    assert set(single_sd) == {"model_pp1"}
    assert set(single_sd["model_pp1"]) == {"weight", "bias"}
    assert single_sd["model_pp1"]["weight"].key == "model_pp1.weight"
    assert _single_or_all_model_state(single_sd) is single_sd

    chunks = [torch.nn.Linear(4, 2), torch.nn.Linear(4, 2)]
    attach_model_sharded_state_dict(chunks, ps)

    vpp_sd = _model_sharded_state_dict(chunks)

    assert set(vpp_sd) == {"model_pp1_vpp0", "model_pp1_vpp1"}
    assert set(vpp_sd["model_pp1_vpp0"]) == {"weight", "bias"}
    assert set(vpp_sd["model_pp1_vpp1"]) == {"weight", "bias"}
    assert vpp_sd["model_pp1_vpp0"]["weight"].key == "model_pp1_vpp0.weight"
    assert vpp_sd["model_pp1_vpp1"]["weight"].key == "model_pp1_vpp1.weight"
    assert _single_or_all_model_state(vpp_sd) is vpp_sd


def test_dist_opt_checkpoint_loads_from_mcore_distckpt(monkeypatch, tmp_path) -> None:
    wrapped_module = torch.nn.Linear(4, 2)
    model = FakeWrapper(wrapped_module)
    optimizer = FakeDistOpt()
    attach_model_sharded_state_dict([model], ParallelState())
    expected_weight = torch.full_like(wrapped_module.weight, 3.0)
    expected_bias = torch.full_like(wrapped_module.bias, -2.0)

    def fake_load(sharded_state_dict, checkpoint_dir, **kwargs):
        assert set(sharded_state_dict["model"]) == {"weight", "bias"}
        assert optimizer.load_model_sd is sharded_state_dict["model"]
        assert sharded_state_dict["optimizer"] == {"is_loading": True}
        assert checkpoint_dir == str(tmp_path / "step_5")
        assert kwargs["validate_access_integrity"] is False
        return {
            "step": 5,
            "model": {"weight": expected_weight, "bias": expected_bias},
            "optimizer": {"loaded": True},
        }

    monkeypatch.setattr("megatron.lite.primitive.ckpt.distckpt.dist_checkpointing.load", fake_load)

    step = dcp.load_training_checkpoint(model, optimizer, str(tmp_path / "step_5"), use_dcp=True)

    assert step == 5
    assert not model.wrapper_load_called
    torch.testing.assert_close(wrapped_module.weight, expected_weight)
    torch.testing.assert_close(wrapped_module.bias, expected_bias)
    assert optimizer.loaded_state == {"loaded": True}


def test_dist_opt_step_sync_traverses_multi_optimizer_chain_without_optimizer_property() -> None:
    class FakeTorchOptimizer:
        def __init__(self, steps):
            self.state = {
                object(): {"step": torch.tensor(step, dtype=torch.int64)} for step in steps
            }

    class FakeDistOpt:
        def __init__(self, steps):
            self.optimizer = FakeTorchOptimizer(steps)

    class FakeChainedOptimizer:
        def __init__(self):
            self.chained_optimizers = [FakeDistOpt([1, 3]), FakeDistOpt([2, 4])]

        @property
        def optimizer(self):
            raise AssertionError(
                "ChainedOptimizer has more than one optimizer when accessing self.optimizer"
            )

    chained = FakeChainedOptimizer()

    _synchronize_native_optimizer_steps(chained)

    for child in chained.chained_optimizers:
        steps = [int(state["step"].item()) for state in child.optimizer.state.values()]
        assert steps == [max(steps)] * len(steps)


def test_runtime_checkpoint_api_passes_current_training_checkpoint_signature(
    monkeypatch, tmp_path
) -> None:
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
        handle, str(tmp_path), global_step=7, save_model=True, save_optimizer=False
    )
    loaded_step = runtime.load_checkpoint(
        handle, str(tmp_path), load_model=False, load_optimizer=True
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

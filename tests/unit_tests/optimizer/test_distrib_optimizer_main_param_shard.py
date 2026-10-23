# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer


class _FakeInnerOptimizer:
    """Stands in for TE FusedAdam: param groups plus lazily created per-param state."""

    def __init__(self, params, state=None):
        self.param_groups = [{"params": params}]
        self.state = {} if state is None else state

    def get_unscaled_state(self, param, state_name):
        return self.state[param][state_name].float() * 2  # Visible marker of being unscaled.


def _distrib_optimizer(inner, model_params, use_precision_aware):
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.optimizer = inner
    optimizer.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=use_precision_aware
    )
    optimizer.model_param_group_index_map = {p: (0, i) for i, p in enumerate(model_params)}
    return optimizer


def test_main_param_shard_is_main_param_without_precision_aware_optimizer():
    """The default path must keep returning the very tensor that model_param.main_param holds."""
    model_param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    main_shard = torch.full((2,), 5.0)
    model_param.main_param = main_shard
    optimizer = _distrib_optimizer(_FakeInnerOptimizer([main_shard]), [model_param], False)

    assert optimizer.get_main_param_shard(model_param) is model_param.main_param


def test_main_param_shard_is_none_for_params_not_owned():
    owned = torch.nn.Parameter(torch.ones(4))
    other = torch.nn.Parameter(torch.ones(4))
    optimizer = _distrib_optimizer(_FakeInnerOptimizer([torch.ones(2)]), [owned], False)

    assert optimizer.get_main_param_shard(other) is None


def test_main_param_shard_reads_precision_aware_master_state():
    model_param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    shard = torch.ones(2, dtype=torch.bfloat16)
    master = torch.full((2,), 1.5)
    inner = _FakeInnerOptimizer([shard], {shard: {"master_param": master}})
    optimizer = _distrib_optimizer(inner, [model_param], True)

    result = optimizer.get_main_param_shard(model_param)
    assert result.dtype == torch.float32
    assert torch.equal(result, master * 2)


def test_main_param_shard_before_first_step_uses_model_param_values():
    """TE creates its state on the first step; before that the masters equal the model params."""
    model_param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    shard = torch.full((2,), 3.0, dtype=torch.bfloat16)
    optimizer = _distrib_optimizer(_FakeInnerOptimizer([shard]), [model_param], True)

    result = optimizer.get_main_param_shard(model_param)
    assert result.dtype == torch.float32
    assert torch.equal(result, shard.float())
    assert shard not in optimizer.optimizer.state  # Lookup must not create optimizer state.


def test_main_param_shard_reassembles_param_remainders():
    """With store_param_remainders the master is the BF16 param plus int16 low bits; the FP32
    value must be reconstructed exactly like the TE Adam kernel does."""
    generator = torch.Generator().manual_seed(0)
    fp32 = torch.randn(4096, generator=generator) * 10
    bits = fp32.view(torch.int32)
    high = (bits >> 16).to(torch.int16)
    remainder = (bits & 0xFFFF).to(torch.int16)
    high = high + (remainder < 0).to(torch.int16)  # BF16 param is rounded to nearest.
    shard = high.view(torch.bfloat16)
    assert (remainder < 0).any() and (remainder >= 0).any()

    model_param = torch.nn.Parameter(torch.ones(4096, dtype=torch.bfloat16))
    inner = _FakeInnerOptimizer([shard], {shard: {"master_param": remainder}})
    optimizer = _distrib_optimizer(inner, [model_param], True)

    result = optimizer.get_main_param_shard(model_param)
    assert result.dtype == torch.float32
    assert torch.equal(result, fp32)


class _FakeChainedMember:
    def __init__(self, shards):
        self._shards = shards

    def get_main_param_shard(self, model_param):
        return self._shards.get(model_param)


def test_chained_optimizer_returns_shard_from_owning_optimizer():
    dense, expert, unowned = (torch.nn.Parameter(torch.ones(1)) for _ in range(3))
    dense_shard, expert_shard = torch.ones(1), torch.zeros(1)
    chained = object.__new__(ChainedOptimizer)
    chained.chained_optimizers = [
        _FakeChainedMember({dense: dense_shard}),
        object(),  # Optimizers that do not track main params are skipped.
        _FakeChainedMember({expert: expert_shard}),
    ]

    assert chained.get_main_param_shard(dense) is dense_shard
    assert chained.get_main_param_shard(expert) is expert_shard
    assert chained.get_main_param_shard(unowned) is None


@pytest.mark.parametrize("stub, use_megatron_fsdp", [(True, False), (False, True)])
def test_main_param_shard_is_none_without_param_index(stub, use_megatron_fsdp):
    """Stub optimizers and Megatron-FSDP return from __init__ before building the param index
    map. A ChainedOptimizer asks them first when, e.g., the dense params are frozen and only the
    experts train, so they must answer None instead of raising."""
    expert = torch.nn.Parameter(torch.ones(1))
    expert_shard = torch.zeros(1)
    unindexed = object.__new__(DistributedOptimizer)
    unindexed.is_stub_optimizer = stub
    unindexed.ddp_config = SimpleNamespace(use_megatron_fsdp=use_megatron_fsdp)
    chained = object.__new__(ChainedOptimizer)
    chained.chained_optimizers = [unindexed, _FakeChainedMember({expert: expert_shard})]

    assert unindexed.get_main_param_shard(expert) is None
    assert chained.get_main_param_shard(expert) is expert_shard


def test_mimo_optimizer_returns_shard_from_owning_module_optimizer():
    from megatron.core.models.mimo.optimizer import MimoOptimizer, ModuleOptimizerInfo

    encoder, language, inactive, unowned = (torch.nn.Parameter(torch.ones(1)) for _ in range(4))
    encoder_shard, language_shard = torch.ones(1), torch.zeros(1)
    language_chain = object.__new__(ChainedOptimizer)
    language_chain.chained_optimizers = [_FakeChainedMember({language: language_shard})]

    def _info(optimizer, is_active=True):
        return ModuleOptimizerInfo(
            optimizer=optimizer, grid=None, pg_collection=None, is_active=is_active
        )

    mimo = MimoOptimizer(
        {
            "images": _info(_FakeChainedMember({encoder: encoder_shard})),
            "language": _info(language_chain),
            # Modules this rank does not run are not consulted.
            "audio": _info(_FakeChainedMember({inactive: torch.ones(1)}), is_active=False),
        },
        config=None,
    )

    assert mimo.get_main_param_shard(encoder) is encoder_shard
    assert mimo.get_main_param_shard(language) is language_shard
    assert mimo.get_main_param_shard(inactive) is None
    assert mimo.get_main_param_shard(unowned) is None

# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.models.test_moe_experts import (
    expert_type,
    initialize_expert_layer,
)
from tests.unit_tests.test_utilities import Utils


def _expert_params(model):
    if hasattr(model, "local_experts"):
        return [list(expert.parameters()) for expert in model.local_experts]
    if not hasattr(model.linear_fc1, "weight0"):
        pytest.skip("TE holds the grouped expert weights in one tensor")
    return [
        [getattr(model.linear_fc1, f"weight{i}"), getattr(model.linear_fc2, f"weight{i}")]
        for i in range(model.num_local_experts)
    ]


def _expert_values(gid, params):
    gen = torch.Generator().manual_seed(gid)
    return [torch.randn(p.shape, generator=gen) for p in params]


def _placed_layer(seed, expert_type, order, ep_rank):
    model = initialize_expert_layer(seed, True, expert_type)
    n = model.num_local_experts
    model.local_expert_indices = order[ep_rank * n : (ep_rank + 1) * n]
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestExpertPlacementCheckpoint:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.parametrize("expert_type", expert_type)
    @pytest.mark.parametrize("singleton_local_shards", [False, True])
    def test_experts_load_by_global_id(
        self, tmp_path_dist_ckpt, expert_type, singleton_local_shards
    ):
        """Experts saved under one placement load by global expert id under another."""
        if Utils.world_size % 2 != 0:
            pytest.skip("needs an even world size")
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        metadata = {"singleton_local_shards": singleton_local_shards}
        saved, *loaded = [
            torch.randperm(8, generator=torch.Generator().manual_seed(1)).tolist(),
            torch.randperm(8, generator=torch.Generator().manual_seed(2)).tolist(),
            list(range(8)),
        ]
        with TempNamedDir(tmp_path_dist_ckpt / "test_expert_placement_checkpoint") as ckpt_dir:
            model = _placed_layer(1, expert_type, saved, ep_rank)
            for gid, params in zip(model.local_expert_indices, _expert_params(model)):
                for p, value in zip(params, _expert_values(gid, params)):
                    p.data.copy_(value)
            save(model.sharded_state_dict(prefix="0.", metadata=metadata), ckpt_dir)
            for order in loaded:
                model = _placed_layer(2, expert_type, order, ep_rank)
                state_dict = load(
                    model.sharded_state_dict(prefix="0.", metadata=metadata), ckpt_dir
                )
                model.load_state_dict({k.removeprefix("0."): v for k, v in state_dict.items()})
                for gid, params in zip(model.local_expert_indices, _expert_params(model)):
                    for p, value in zip(params, _expert_values(gid, params)):
                        assert torch.equal(p.detach().cpu(), value), (order, gid)

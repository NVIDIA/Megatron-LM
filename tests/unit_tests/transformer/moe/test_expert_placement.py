# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
"""Tests for custom MoE expert placement (--moe-expert-placement).

CPU tests cover placement resolution and JSON parsing/validation. GPU tests verify that the
alltoall dispatcher routes tokens correctly under a non-contiguous placement (dispatch/combine
round-trip) and that a permuted placement produces the same result as the default placement.
"""

import dataclasses
import json

import pytest
import torch

from megatron.core.transformer.moe.moe_layer import get_expert_placement_order
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer


def _base_config(num_experts=8, ep_size=2, placement=None):
    return TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=8,
        num_moe_experts=num_experts,
        expert_model_parallel_size=ep_size,
        moe_token_dispatcher_type="alltoall",
        moe_expert_placement=placement,
    )


class TestExpertPlacementResolution:
    """CPU-only: get_expert_placement_order and the resulting per-rank partition."""

    def test_default_is_contiguous(self):
        cfg = _base_config(placement=None)
        assert get_expert_placement_order(cfg, layer_number=3, ep_size=2) == list(range(8))

    def test_default_key_applies_to_all_layers(self):
        perm = [3, 1, 0, 2, 7, 5, 4, 6]
        cfg = _base_config(placement={"default": perm, "layers": {}})
        assert get_expert_placement_order(cfg, layer_number=1, ep_size=2) == perm
        assert get_expert_placement_order(cfg, layer_number=9, ep_size=2) == perm

    def test_per_layer_overrides_default(self):
        perm = [3, 1, 0, 2, 7, 5, 4, 6]
        override = [0, 1, 2, 3, 4, 5, 6, 7]
        cfg = _base_config(placement={"default": perm, "layers": {5: override}})
        assert get_expert_placement_order(cfg, layer_number=5, ep_size=2) == override
        assert get_expert_placement_order(cfg, layer_number=6, ep_size=2) == perm

    def test_no_matching_entry_falls_back_to_contiguous(self):
        cfg = _base_config(placement={"default": None, "layers": {5: [7, 6, 5, 4, 3, 2, 1, 0]}})
        # layer 6 has no entry and no default -> contiguous
        assert get_expert_placement_order(cfg, layer_number=6, ep_size=2) == list(range(8))

    def test_rejects_non_permutation(self):
        cfg = _base_config(placement={"default": [0, 0, 1, 2, 3, 4, 5, 6], "layers": {}})
        with pytest.raises(AssertionError):
            get_expert_placement_order(cfg, layer_number=1, ep_size=2)

    @pytest.mark.parametrize(
        "perm,ep_size",
        [
            ([3, 1, 0, 2, 7, 5, 4, 6], 2),
            ([7, 6, 5, 4, 3, 2, 1, 0], 4),
            ([5, 2, 7, 0, 3, 6, 1, 4], 8),
        ],
    )
    def test_partition_covers_all_experts_once(self, perm, ep_size):
        cfg = _base_config(placement={"default": perm, "layers": {}})
        order = get_expert_placement_order(cfg, layer_number=1, ep_size=ep_size)
        num_local = 8 // ep_size
        owned = []
        for r in range(ep_size):
            owned.extend(order[r * num_local : (r + 1) * num_local])
        assert sorted(owned) == list(range(8))


class TestExpertPlacementParsing:
    """CPU-only: --moe-expert-placement JSON parsing / validation."""

    def _parse(self, tmp_path, data, num_experts=8, ep_size=4):
        from megatron.training.arguments import _parse_moe_expert_placement

        p = tmp_path / "placement.json"
        p.write_text(json.dumps(data))
        return _parse_moe_expert_placement(str(p), num_experts=num_experts, ep_size=ep_size)

    def test_valid(self, tmp_path):
        res = self._parse(
            tmp_path,
            {
                "num_experts": 8,
                "default": [3, 1, 0, 2, 7, 5, 4, 6],
                "layers": {"5": list(range(8))},
            },
        )
        assert res["default"] == [3, 1, 0, 2, 7, 5, 4, 6]
        assert res["layers"] == {5: list(range(8))}

    @pytest.mark.parametrize(
        "data",
        [
            {"default": [0, 0, 1, 2, 3, 4, 5, 6]},  # duplicate
            {"default": [0, 1, 2, 3, 4, 5, 6]},  # wrong length
            {},  # empty
            {"num_experts": 16, "default": list(range(8))},  # num_experts mismatch
        ],
    )
    def test_invalid_raises(self, tmp_path, data):
        with pytest.raises(ValueError):
            self._parse(tmp_path, data)


class _PlacementContainer(MoEModelTestContainer):
    """MoEModelTestContainer that injects a custom expert placement into the config."""

    def __init__(self, *args, moe_expert_placement=None, **kwargs):
        super().__init__(*args, **kwargs)
        if moe_expert_placement is not None:
            self.config = dataclasses.replace(
                self.config, moe_expert_placement=moe_expert_placement
            )
            self.moe_layer = self.new_moe_layer()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestExpertPlacementDispatch:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "ep_size,perm",
        [
            (2, [4, 5, 6, 7, 0, 1, 2, 3]),  # swap the two rank-halves
            (2, [3, 1, 0, 2, 7, 5, 4, 6]),  # arbitrary permutation
        ],
    )
    def test_dropless_roundtrip_with_placement(self, ep_size, perm):
        """A non-contiguous placement must still restore tokens through dispatch/combine."""
        container = _PlacementContainer(
            tp_size=1,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_token_dispatcher_type="alltoall",
            moe_expert_placement={"default": perm, "layers": {}},
        )
        # The dispatcher must have picked up the custom placement.
        assert container.moe_layer.token_dispatcher.expert_order_is_identity is False
        container.dispatcher_dropless_test()

    @pytest.mark.internal
    @pytest.mark.timeout(120)
    def test_identity_placement_matches_default(self):
        """An explicit identity placement must behave exactly like no placement."""
        container = _PlacementContainer(
            tp_size=1,
            ep_size=2,
            pp_size=1,
            num_moe_experts=8,
            moe_token_dispatcher_type="alltoall",
            moe_expert_placement={"default": list(range(8)), "layers": {}},
        )
        assert container.moe_layer.token_dispatcher.expert_order_is_identity is True
        container.dispatcher_dropless_test()

    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("ep_size", [2, 8])
    @pytest.mark.parametrize("dedup", [False, True])
    def test_placed_layer_matches_default_layer(self, ep_size, dedup):
        """Experts keyed by global id must compute the same layer under any placement."""
        if Utils.world_size % ep_size != 0:
            pytest.skip(f"needs a world size divisible by EP={ep_size}")
        container = MoEModelTestContainer(
            tp_size=1, ep_size=ep_size, pp_size=1, num_moe_experts=16, moe_router_topk=4
        )
        # The fp32 router gate runs through TE's general GEMM, whose rounding can move the
        # router weight gradient by ~1e-3 when a last-bit difference in the probability
        # gradient (e.g. the dedup combine's summation order) flips it. An fp64 router
        # takes the plain torch path and compares at fp32 rounding level.
        default = container.new_moe_layer(moe_router_dtype="fp64")
        perm = torch.randperm(16, generator=torch.Generator().manual_seed(0)).tolist()
        placed = container.new_moe_layer(
            moe_router_dtype="fp64",
            moe_expert_placement={"default": None, "layers": {0: perm}},
            moe_token_dedup=dedup,
        )
        layers = (default, placed)
        assert placed.token_dispatcher.expert_order == perm
        placed.router.load_state_dict(default.router.state_dict())
        for layer in layers:
            for expert, gid in zip(layer.experts.local_experts, layer.local_expert_indices):
                gen = torch.Generator().manual_seed(gid)
                for p in expert.parameters():
                    p.data.copy_(torch.randn(p.shape, generator=gen) * 0.1)
        gen = torch.Generator(device="cuda").manual_seed(torch.distributed.get_rank())
        x = torch.randn(32, 2, container.config.hidden_size, device="cuda", generator=gen)
        grad = torch.randn(x.shape, device="cuda", generator=gen)
        # Placement (and dedup) only change the order of expert rows and of a token's expert
        # outputs, so compare at fp32 rounding level with TF32 matmuls off: their
        # reduced-precision accumulation is order-sensitive at the 1e-3 level.
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_precision = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        try:
            results = []
            for layer in layers:
                inp = x.clone().requires_grad_()
                out, _ = layer(inp)
                out.backward(grad)
                results.append((out, inp.grad, layer.router.weight.grad))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
            torch.set_float32_matmul_precision(prev_precision)
        for ref, got in zip(*results):
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

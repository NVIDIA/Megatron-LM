# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the dtype of the expert-TP Reduce-Scatter in the alltoall token combine."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.typed_torch import apply_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import (
    MoEModelTestContainer,
    token_permutation,
    token_unpermutation,
)


def _make_dispatcher(tp_size, enabled, probs_dtype=torch.float32):
    """Builds a dispatcher with only the attributes ``combine_preprocess`` reads."""
    dispatcher = MoEAlltoAllTokenDispatcher.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = SimpleNamespace(
        moe_combine_reduce_in_activation_dtype=enabled, moe_permute_fusion=False
    )
    dispatcher.tp_size = tp_size
    dispatcher.tp_group = None
    dispatcher.num_local_experts = 1
    dispatcher.output_splits_tp = None
    dispatcher.probs = torch.zeros(1, dtype=probs_dtype)
    return dispatcher


class TestTpReduceDtype:
    """CPU tests for ``MoEAlltoAllTokenDispatcher._tp_reduce_dtype``."""

    @pytest.fixture(autouse=True)
    def _clear_nccl_algo(self, monkeypatch):
        monkeypatch.delenv("NCCL_ALGO", raising=False)

    @pytest.mark.parametrize("tp_size", [2, 4, 8])
    def test_disabled_keeps_probs_dtype(self, tp_size):
        dispatcher = _make_dispatcher(tp_size, enabled=False)
        hidden = torch.zeros(4, 8, dtype=torch.bfloat16)
        assert dispatcher._tp_reduce_dtype(hidden) == torch.float32

    @pytest.mark.parametrize("activation_dtype", [torch.bfloat16, torch.float16])
    def test_enabled_two_ranks_uses_activation_dtype(self, activation_dtype):
        dispatcher = _make_dispatcher(2, enabled=True)
        hidden = torch.zeros(4, 8, dtype=activation_dtype)
        assert dispatcher._tp_reduce_dtype(hidden) == activation_dtype

    @pytest.mark.parametrize("tp_size", [4, 8])
    def test_enabled_more_than_two_ranks_keeps_probs_dtype(self, tp_size):
        dispatcher = _make_dispatcher(tp_size, enabled=True)
        hidden = torch.zeros(4, 8, dtype=torch.bfloat16)
        assert dispatcher._tp_reduce_dtype(hidden) == torch.float32

    @pytest.mark.parametrize(
        "nccl_algo,expected",
        [
            ("NVLS", torch.float32),
            ("nvls", torch.float32),
            ("Ring,NVLS", torch.float32),
            ("reducescatter:nvls", torch.float32),
            ("^NVLS", torch.bfloat16),
            ("Ring", torch.bfloat16),
            ("PAT", torch.bfloat16),
        ],
    )
    def test_enabled_two_ranks_respects_nccl_algo(self, monkeypatch, nccl_algo, expected):
        monkeypatch.setenv("NCCL_ALGO", nccl_algo)
        dispatcher = _make_dispatcher(2, enabled=True)
        hidden = torch.zeros(4, 8, dtype=torch.bfloat16)
        assert dispatcher._tp_reduce_dtype(hidden) == expected

    @pytest.mark.parametrize(
        "tp_size,enabled,expected_reduce_dtype",
        [(2, False, torch.float32), (2, True, torch.bfloat16), (4, True, torch.float32)],
    )
    def test_combine_preprocess_reduce_scatter_dtype(
        self, monkeypatch, tp_size, enabled, expected_reduce_dtype
    ):
        seen = {}

        def fake_reduce_scatter(input_, group=None, input_split_sizes=None):
            seen["dtype"] = input_.dtype
            return input_.clone()

        monkeypatch.setattr(
            token_dispatcher, "reduce_scatter_to_sequence_parallel_region", fake_reduce_scatter
        )
        dispatcher = _make_dispatcher(tp_size, enabled=enabled)
        hidden = torch.randn(4, 8).to(torch.bfloat16)
        out = dispatcher.combine_preprocess(hidden)
        assert seen["dtype"] == expected_reduce_dtype
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, hidden)


class TestCombineReduceInActivationDtype:
    """Checks that the 2-rank bf16 reduce-scatter matches the FP32 path bit for bit."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(Utils.world_size % 8 != 0, reason="needs a multiple of 8 ranks")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    def test_two_rank_expert_tp_matches_fp32_reduce(self, monkeypatch):
        monkeypatch.setenv("NCCL_ALGO", "Ring")
        container = MoEModelTestContainer(
            tp_size=2,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
        )
        reference_layer = container.moe_layer
        bf16_layer = container.new_moe_layer(moe_combine_reduce_in_activation_dtype=True)
        tp_rank = reference_layer.token_dispatcher.tp_rank

        torch.manual_seed(1234)
        hidden_states = torch.randn((32, 8, reference_layer.config.hidden_size), device="cuda")
        probs, indices = apply_module(reference_layer.router)(hidden_states)

        def combine(layer):
            dispatcher = layer.token_dispatcher
            permuted, _, permuted_probs = token_permutation(
                dispatcher, hidden_states, probs, indices
            )
            # Give each expert-TP rank a different partial result so the reduction adds two
            # distinct bf16 values, as the second expert-TP shard of a real expert would.
            permuted = (
                permuted.float() * permuted_probs.unsqueeze(-1) * (1.0 + 0.37 * tp_rank)
            ).to(torch.bfloat16)
            restored, _ = token_unpermutation(dispatcher, permuted)
            return restored

        with torch.no_grad():
            expected = combine(reference_layer)
            actual = combine(bf16_layer)
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, expected)

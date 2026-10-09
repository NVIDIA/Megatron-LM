# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    deepseek_v4_hybrid_attention as dsv4_attention,
)
from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
    csa2_attention_spec,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    candidate_blocks_from_scores,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule, SinglePassMHCState
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.models.test_deepseek_v41 import groups, tiny_config


class _SinglePassBranch(torch.nn.Module):
    """One residual branch plus final contraction, including parameter gradients."""

    def __init__(self, config):
        super().__init__()
        self.mhc = HyperConnectionModule(config, 1)

    def forward(self, hidden, previous):
        """Recreate transient state on every replay."""
        state = SinglePassMHCState(previous)
        branch, residual_mix, output_mix, residual = self.mhc(
            hidden, mhc_state=state, return_residual=True
        )
        output = self.mhc.fused_h_res_h_post_bda(
            residual_mix, residual, output_mix, (branch, None), 0.0, True, False
        )
        return state.contract(output, self.mhc.n)


@pytest.mark.parametrize("fused", [False, True])
def test_single_pass_mhc_replay(groups, fused):
    seeded()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=fused,
    )
    module = _SinglePassBranch(config).cuda()
    inputs = (
        torch.randn(9, 2, 64, device="cuda", requires_grad=True),
        torch.randn(9, 2, 4, device="cuda", requires_grad=True),
    )
    with deterministic_algorithms(True):
        assert_module_replays_bit_exact(module, inputs)


def test_csa2_candidate_replay(groups):
    """Replay tied block selection and expansion, including duplicate invalid slots."""
    scores = torch.zeros(2, 9, 257, device="cuda")
    visible = torch.arange(9, device="cuda").unsqueeze(-1)
    scores.masked_fill_(torch.arange(257, device="cuda") >= visible, -torch.inf)
    with deterministic_algorithms(True):
        first = candidate_blocks_from_scores(scores, visible, 4, 2)
        second = candidate_blocks_from_scores(scores, visible, 4, 2)
        for a, b in zip((first.indices, first.to_mask(257)), (second.indices, second.to_mask(257))):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_csa2_module_replay(groups, monkeypatch):
    """Replay the shared DSv4 projection path with V4.1's weightless query norm disabled."""

    def unexpected_query_norm(*args, **kwargs):
        pytest.fail("V4.1 must not apply the V4 per-head query RMS normalization")

    monkeypatch.setattr(dsv4_attention, "_q_rms_norm", unexpected_query_norm)
    seeded()
    config = tiny_config(
        num_layers=1,
        csa_compress_ratios=[2],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0],
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        dsa_indexer_loss_coeff=0,
    )
    layer = build_module(
        csa2_attention_spec, config=config, layer_number=1, pg_collection=groups
    ).cuda()
    with deterministic_algorithms(True):
        assert_module_replays_bit_exact(
            layer, (torch.randn(9, 2, 32, device="cuda", requires_grad=True), None)
        )


def test_csa2_chunked_indexer_scores_replay():
    """The inference-only chunked CSA2 score path is bit-exact and matches dense scoring."""
    from megatron.core.transformer.experimental_attention_variant.csa2 import (
        _chunked_csa2_indexer_scores,
    )

    seeded()
    q = torch.randn(11, 2, 3, 8, device="cuda", dtype=torch.float32)
    indexer_k = torch.randn(7, 2, 8, device="cuda", dtype=torch.float32)
    weights = torch.randn(11, 2, 3, device="cuda", dtype=torch.float32)

    def chunked(query, keys, head_weights):
        with torch.no_grad():
            return _chunked_csa2_indexer_scores(query, keys, head_weights, key_chunk_size=3)

    with deterministic_algorithms(True):
        actual, _ = assert_replays_bit_exact(
            chunked, (q, indexer_k, weights), backward=False, what="CSA2 chunked indexer scores"
        )
    expected = torch.einsum("sbhd,tbd->bsht", q, indexer_k).relu()
    expected = (expected * weights.permute(1, 0, 2).unsqueeze(-1)).sum(dim=2)
    torch.testing.assert_close(actual["out"], expected, rtol=1e-5, atol=1e-6)

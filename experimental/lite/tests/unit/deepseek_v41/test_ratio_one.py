# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Official source stays external; execute its candidate/selection CPU semantics."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from megatron.lite.model.deepseek_v41.lite.ratio_one import (
    RatioOneAttention,
    candidate_pool,
    select_indices,
)
from megatron.lite.model.deepseek_v41.lite.state import KVSourceState


@pytest.fixture
def official_selection():
    """Execute exact official AST, without loading CUDA/quantization kernels.

    This is the source-pinned selection stage of Indexer.forward, from its
    first score einsum through its return. It is not a full official model.
    """
    if os.environ.get("DS41_RUN_OFFICIAL_REFERENCE") != "1":
        pytest.skip(
            "External official-source oracle is opt-in: set DS41_RUN_OFFICIAL_REFERENCE=1 "
            "and DS41_REFERENCE_DIR; see examples/deepseek_v41/README.md"
        )
    source = Path(os.environ.get("DS41_REFERENCE_DIR", "/tmp/ds41-fixture-reference")) / "model.py"
    raw = source.read_bytes()  # Missing external source is a failure, never a skip.
    assert (
        hashlib.sha256(raw).hexdigest()
        == "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    )
    tree = ast.parse(raw)
    pool = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "select_candidate_blocks"
    )
    indexer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    forward = next(
        n for n in indexer.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    begin = next(
        i
        for i, n in enumerate(forward.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "index_score" for t in n.targets)
    )
    namespace = {"torch": torch, "F": F, "world_size": 1}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[pool], type_ignores=[])), str(source), "exec"
        ),
        namespace,
    )
    # Only bind stage inputs. No original statement from the selected stage is
    # patched, and the original candidate function executes unchanged.
    stage = ast.parse(
        "def stage(self, q, index_k, weights, shared_attn, ratio, start_pos, seqlen, end_pos, offset): pass"
    ).body[0]
    stage.body = ast.parse("x = q").body + forward.body[begin:]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[stage], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace["select_candidate_blocks"], namespace["stage"]


@pytest.mark.parametrize("width", [0, 1, 7, 8, 9, 17])
def test_candidate_pool_matches_official_partial_newest_block(official_selection, width):
    reference, _ = official_selection
    torch.manual_seed(width + 11)
    scores = torch.randn(6, width)
    visible = torch.arange(6).remainder(width + 1).unsqueeze(-1)
    scores = scores.masked_fill(torch.arange(width) >= visible, -torch.inf)
    if width == 0:
        assert candidate_pool(scores, visible, blocks=2, block_size=8).shape == scores.shape
        return  # Official topk/reshape has no empty-K contract.
    actual = candidate_pool(scores, visible, blocks=2, block_size=8)
    assert torch.equal(actual, reference(scores, visible, 2, 8))


@pytest.mark.parametrize("ties", [False, True])
def test_candidate_publish_and_reindex_match_official_indices(official_selection, ties):
    _, reference = official_selection
    torch.manual_seed(331)
    q, k, weights = torch.randn(17, 2, 4), torch.randn(17, 4), torch.randn(17, 2)
    if ties:
        q.zero_()
    prefix = torch.tensor([0, 17], dtype=torch.int32)
    settings = dict(topk=3, candidate_blocks=2, block_size=4)
    owner = SimpleNamespace(
        is_candidate_source=True,
        uses_candidates=False,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        index_topk=3,
    )
    shared = SimpleNamespace(candidates=None)
    expected = reference(
        owner, q.unsqueeze(0), k.unsqueeze(0), weights.unsqueeze(0), shared, 1, 0, 17, 17, 0
    )
    actual, pool = select_indices(q, k, weights, prefix, 0, publish=True, **settings)
    assert torch.equal(actual, expected.squeeze(0))
    assert torch.equal(pool[0], shared.candidates.squeeze(0))
    reader = SimpleNamespace(is_candidate_source=False, uses_candidates=True, index_topk=3)
    q2 = q.flip(0)
    expected_reader = reference(
        reader, q2.unsqueeze(0), k.unsqueeze(0), weights.unsqueeze(0), shared, 1, 0, 17, 17, 0
    )
    actual_reader, _ = select_indices(q2, k, weights, prefix, 0, candidates=pool, **settings)
    assert torch.equal(actual_reader, expected_reader.squeeze(0))


def test_ratio_one_native_unfused_attention_backward_and_source_reuse():
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.manual_seed(15)
    config = SimpleNamespace(
        num_attention_heads=1,
        qk_pos_emb_head_dim=2,
        csa_window_size=2,
        v_head_dim=4,
        dsa_indexer_topk=3,
        ds41_candidate_blocks=2,
        ds41_candidate_block_size=2,
        apply_rope_fusion=False,
        rotary_interleaved=False,
        multi_latent_attention=True,
        mrope_section=None,
    )
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    state = KVSourceState()
    rope = lambda length, packed_seq: torch.zeros(length, 1, 1, 2)
    owner = RatioOneAttention(
        config,
        group=group,
        rotary_pos_emb=rope,
        layer=3,
        index_owner=3,
        candidate_owner=3,
        state=state,
    )
    reader = RatioOneAttention(
        config,
        group=group,
        rotary_pos_emb=rope,
        layer=4,
        index_owner=3,
        candidate_owner=3,
        state=state,
    )
    prefix = torch.tensor([0, 5, 6, 12], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cp_partition_mode="contiguous",
        cu_seqlens_q=prefix,
        cu_seqlens_kv=prefix,
        max_seqlen_q=6,
        max_seqlen_kv=6,
    )
    q = torch.randn(12, 1, 4, requires_grad=True)
    window, main = [torch.randn(12, 4, requires_grad=True) for _ in range(2)]
    iq, ik, weights = torch.randn(12, 1, 4), torch.randn(12, 4), torch.randn(12, 1)
    with state.microbatch():
        output = owner(q, window, main, iq, ik, weights, params)
        other = reader(q, window, main, -iq, -ik, -weights, params)
        assert torch.equal(output, other)
        (output.square().sum() + other.square().sum()).backward()
    assert all(torch.isfinite(t.grad).all() for t in (q, window, main))
    assert (
        torch.isfinite(owner.attn_sink.grad).all() and torch.isfinite(reader.attn_sink.grad).all()
    )


def test_ratio_one_only_composes_core_functions():
    import inspect

    from megatron.core.transformer.experimental_attention_variant import csa, dsa
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
    from megatron.lite.model.deepseek_v41.lite import ratio_one

    tree = ast.parse(inspect.getsource(ratio_one))
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "dist"
        for n in ast.walk(tree)
    )

    def bodies(module):
        return {
            ast.dump(ast.Module(body=n.body, type_ignores=[]), include_attributes=False)
            for n in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    assert not (bodies(ratio_one) & (bodies(csa) | bodies(dsa) | bodies(cp_utils)))


def test_candidate_block_ties_and_two_positive_infinities_match_official(official_selection):
    reference, _ = official_selection
    scores = torch.zeros(4, 17)
    scores[1, :8] = 2  # Within-block and cross-block ties.
    scores[2, 0] = torch.inf  # Newest block is also pinned to +inf.
    scores[3, 0] = scores[3, 8] = torch.inf
    visible = torch.tensor([[17], [17], [17], [17]])
    for blocks in (1, 2, 3):
        assert torch.equal(
            candidate_pool(scores, visible, blocks=blocks, block_size=8),
            reference(scores, visible, blocks, 8),
        )


def test_indexer_frozen_manifest_does_not_freeze_nonowning_main_kv():
    from megatron.lite.model.deepseek_v41.lite.native import freeze_indexer_sources

    main = torch.nn.Linear(4, 4, bias=False)
    indexer = torch.nn.Module()
    indexer.q = torch.nn.Linear(4, 4, bias=False)
    indexer.k = torch.nn.Linear(4, 4, bias=False)
    indexer.weights = torch.nn.Linear(4, 1, bias=False)
    object.__setattr__(indexer, "main_kv", main)
    names = freeze_indexer_sources(indexer, loss_coefficient=0)
    assert names == ("q.weight", "k.weight", "weights.weight")
    assert indexer.ds41_frozen_parameters == names
    assert all(not p.requires_grad for p in indexer.parameters())
    assert main.weight.requires_grad
    with pytest.raises(ValueError, match="auxiliary loss"):
        freeze_indexer_sources(indexer, loss_coefficient=0.1)


def test_candidate_pool_with_zero_query_rows_matches_official(official_selection):
    reference, _ = official_selection
    scores, visible = torch.empty(0, 6), torch.empty(0, 1, dtype=torch.int64)
    actual = candidate_pool(scores, visible, blocks=2, block_size=2)
    assert actual.shape == (0, 6)
    assert torch.equal(actual, reference(scores, visible, 2, 2))

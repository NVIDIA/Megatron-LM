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

from megatron.lite.primitive.modules.attention.ratio_one import candidate_pool, select_indices


@pytest.fixture
def official_selection():
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

    def execute(nodes):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, str(source), "exec"), namespace)

    execute([pool])
    stage = ast.parse(
        "def stage(self, q, index_k, weights, shared_attn, ratio, start_pos, seqlen, end_pos, offset): pass"
    ).body[0]
    stage.body = ast.parse("x = q").body + forward.body[begin:]
    execute([stage])
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
    from megatron.lite.model.deepseek_v41.lite.components.config import freeze_indexer_sources

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

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

from megatron.lite.primitive.modules.attention.ratio_one import select_indices


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

    def named(nodes, cls, name):
        return next(n for n in nodes if isinstance(n, cls) and n.name == name)

    pool = named(tree.body, ast.FunctionDef, "select_candidate_blocks")
    indexer = named(tree.body, ast.ClassDef, "Indexer")
    forward = named(indexer.body, ast.FunctionDef, "forward")
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

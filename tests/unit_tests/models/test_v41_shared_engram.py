# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 layer identity, packed gate masks and bounded contiguous CP history."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from megatron.core.models.deepseek_v41.engram import V41Engram, released_engram_config
from megatron.core.models.engram import Engram
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


def _config(tmp_path):
    options = dict(
        layer_ids=[0, 3],
        num_embeddings=[60, 120],
        max_ngram_size=3,
        vocab_size=11,
        n_heads=2,
        head_dim=4,
        pad_token_id=2,
        compressed_vocab_size=32,
    )
    p = tmp_path / "tokens.pt"
    torch.save(torch.arange(32), p)
    return options, released_engram_config(options, str(p))


def test_logical_seeds_and_gathered_legacy_table(tmp_path):
    """Map blocks to physical V layers and load their single gathered table by hash head."""
    options, shared = _config(tmp_path)
    assert shared.layer_ids == (1, 7)
    bound = ((1 << 63) - 1) // 32 // 2
    expected = np.random.default_rng(10007 * 3).integers(0, bound, size=3, dtype=np.int64) * 2 + 1
    assert tuple(expected) == shared.multipliers(7)
    cfg = SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        perform_initialization=False,
        init_method=torch.nn.init.normal_,
        hidden_size=8,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        layernorm_epsilon=1e-5,
        sequence_parallel=False,
        deterministic_mode=False,
    )
    groups = ProcessGroupCollection(ep=None, tp=None, cp=None, expt_dp=None)
    layer = V41Engram(cfg, shared, 7, groups)
    sizes = shared.table_sizes(7)
    old = torch.arange(sum(sizes) * shared.head_dim, dtype=torch.float32).view(-1, shared.head_dim)
    state = {k: v for k, v in layer.state_dict().items() if not k.startswith("embedding.tables.")}
    state["embed.tables.0.weight"] = old
    layer.load_state_dict(state)
    torch.testing.assert_close(torch.cat([t.weight for t in layer.embedding.tables]), old)
    with pytest.raises(ValueError, match="gathered"):
        layer.load_state_dict({**state, "embed.tables.0.weight": old[:3]})
    assert options["num_embeddings"] == [sum(shared.table_sizes(i)) for i in shared.layer_ids]


def test_packed_memory_matches_separate_documents_at_cp1_and_cp2(tmp_path):
    """A real document crosses the rank boundary and another starts after interior padding."""
    Utils.initialize_distributed()
    group = dist.group.WORLD
    size, rank = group.size(), group.rank()
    if size not in (1, 2):
        pytest.skip("run on one or two ranks")
    torch.cuda.set_device(rank)
    _, shared = _config(tmp_path)
    cfg = SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        perform_initialization=True,
        init_method=torch.nn.init.normal_,
        hidden_size=8,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        layernorm_epsilon=1e-5,
        sequence_parallel=False,
        deterministic_mode=False,
        linear_cp_layout="contiguous",
        attention_cp_layout="contiguous",
        cp_partition_mode="contiguous",
    )
    singleton = ProcessGroupCollection(ep=None, tp=None, cp=None, expt_dp=None)
    parallel = ProcessGroupCollection(ep=None, tp=None, cp=group, expt_dp=None)
    torch.manual_seed(77)
    reference = Engram(cfg, shared, 7, singleton).cuda()
    actual = V41Engram(cfg, shared, 7, parallel).cuda()
    actual.load_state_dict(reference.state_dict())
    for tokens, logical, physical in [
        ([7, 8, 9, 10, 11, 29, 13, 30], [0, 5, 6], [0, 6, 8]),
        ([7, 8], [0, 2], [0, 2]),  # local_length < ngram history at CP2
    ]:
        reference.zero_grad()
        actual.zero_grad()
        ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        cu = torch.tensor(logical, dtype=torch.int32, device="cuda")
        padded = torch.tensor(physical, dtype=torch.int32, device="cuda")
        p = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=padded,
            cu_seqlens_kv_padded=padded,
            max_seqlen_q=max(np.diff(logical)),
            max_seqlen_kv=max(np.diff(logical)),
        )
        width = ids.shape[1] // size
        lo, hi = rank * width, (rank + 1) * width
        h = (
            torch.arange(len(tokens) * 32, device="cuda", dtype=torch.float32).reshape(-1, 1, 32)
            / 256
        ).to(torch.bfloat16)
        hx = h[lo:hi].clone().requires_grad_()
        hr = h.clone().requires_grad_()
        lookup = actual.lookup(ids[:, lo:hi], p, local_sequence_length=width)
        got = actual(hx, lookup=lookup, add_to_hidden_states=True)
        parts = []
        for i, start in enumerate(physical[:-1]):
            end = start + logical[i + 1] - logical[i]
            parts.extend(
                [
                    reference(hr[start:end], ids[:, start:end], add_to_hidden_states=True),
                    hr[end : physical[i + 1]],
                ]
            )
        want = torch.cat(parts)[lo:hi]
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        want.float().sum().backward()
        got.float().sum().backward()
        torch.testing.assert_close(hx.grad, hr.grad[lo:hi], rtol=0, atol=0)
        for a, b in zip(actual.embedding.tables, reference.embedding.tables, strict=True):
            torch.testing.assert_close(a.weight.grad, b.weight.grad, rtol=0, atol=0)

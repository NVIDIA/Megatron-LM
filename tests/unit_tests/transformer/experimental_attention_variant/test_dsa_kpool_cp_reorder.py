# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KPool selection under all-gather CP sees the indexer K and the KPool gate in one row order.

DSAttention gathers the indexer K and, for KPool, the gate score over the CP group, then
reorders both into full-sequence order: the key through ``_permute_rows`` (inverse-permutation
backward) and the gate through ``index_select``. The KPool selector compresses keys with the
gate row by row, so both must arrive with the global length and the same order. CPU only: the
CP gather, the indexer projections and the KPool kernel are faked.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_kernels, dsa_layout
from megatron.core.transformer.forward_sharing import forward_sharing_lifetime
from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
    _make_config as make_small_dsa_config,
)

LOCAL_ROWS = 8
_FUSED_DSA_HOOKS = (
    "run_fused_dsa_attention",
    "run_fused_qk_topk",
    "run_fused_qk_topk_with_loss",
    "run_fused_qk_topk_with_scores",
    "run_fused_absorbed_sparse_attention",
)


class _FakeCPGroup:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _kpool_config(monkeypatch):
    from megatron.core import utils as core_utils

    # Config validation only; neither a CUDA device nor the kernel dependencies are needed.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core_utils, "_missing_cudnn_dsa_kernel_dependencies", lambda: [])
    return make_small_dsa_config(
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.auto,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=8,
        dsa_indexer_kpool=4,
        dsa_indexer_topk=4,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=True,
        use_cpu_initialization=True,
    )


@pytest.mark.parametrize(("cp_size", "cp_rank"), [(2, 1), (4, 2)])
@pytest.mark.parametrize("training", [True, False])
def test_kpool_gate_and_indexer_k_share_the_cp_reorder(monkeypatch, cp_size, cp_rank, training):
    config = _kpool_config(monkeypatch)
    real_indexer_cls = dsa_module.DSAIndexer

    def fake_build(spec, *args, **kwargs):
        if spec is real_indexer_cls:
            kwargs.setdefault("submodules", dsa_module.DSAIndexerSubmodules())
            return real_indexer_cls(*args, **kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(dsa_module, "build_module", fake_build)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    for name in _FUSED_DSA_HOOKS:
        if hasattr(dsa_kernels, name):
            monkeypatch.setattr(dsa_kernels, name, lambda *a, **k: None)
    monkeypatch.setattr(dsa_module, "_validate_nonpacked_cp_uniform_length", lambda **_k: None)
    group = _FakeCPGroup(cp_size, cp_rank)

    def pg_size(g=None):
        return g.size() if g is not None else 1

    monkeypatch.setattr(dsa_module, "get_pg_size", pg_size)
    monkeypatch.setattr(dsa_layout, "get_pg_size", pg_size)

    def fake_gather(tensor, group):
        # Rows of CP rank j carry the id j * 1000 + local_row in feature 0.
        pieces = []
        for j in range(group.size()):
            if j == group.rank():
                pieces.append(tensor)
            else:
                piece = torch.zeros_like(tensor)
                ids = j * 1000 + torch.arange(tensor.size(0), dtype=tensor.dtype)
                piece[..., 0] = ids.view(-1, *([1] * (tensor.dim() - 2))).expand_as(piece[..., 0])
                pieces.append(piece)
        return torch.cat(pieces, dim=0)

    monkeypatch.setattr(dsa_module, "gather_from_sequence_parallel_region", fake_gather)
    seen = {}

    def fake_kpool(q, k, weights, topk, kpool, gate, ape, **kwargs):
        seen.update(k=k.clone(), gate=gate.clone(), grad=torch.is_grad_enabled())
        batch, sq = q.size(1), q.size(0)
        return None, torch.arange(topk).expand(batch, sq, topk).contiguous()

    monkeypatch.setattr(dsa_module, "fused_qk_topk_kpool", fake_kpool)

    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=real_indexer_cls),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=group),
    )
    attention.cp_comm_type = "allgather"

    def fake_before_topk(x, _qr, _packed_seq_params):
        rows, batch = x.shape[:2]
        ids = (cp_rank * 1000 + torch.arange(rows, dtype=torch.float32)).view(rows, 1)
        q = torch.randn(rows, batch, 4, 8)
        k = torch.zeros(rows, batch, 8)
        k[..., 0] = ids
        gate = torch.zeros(rows, batch, 8)
        gate[..., 0] = ids
        attention.indexer._kpool_gate_score = gate
        return q, k, torch.ones(rows, batch, 4)

    attention.indexer.forward_before_topk = fake_before_topk
    attention.train(training)
    heads = config.num_attention_heads
    absorbed = config.kv_lora_rank + config.qk_pos_emb_head_dim
    with forward_sharing_lifetime(None, None, config):
        attention(
            query=torch.randn(LOCAL_ROWS, 1, heads, absorbed),
            key=torch.randn(LOCAL_ROWS, 1, 1, absorbed),
            value=None,
            attention_mask=None,
            x=torch.randn(LOCAL_ROWS, 1, config.hidden_size),
            qr=torch.randn(LOCAL_ROWS, 1, config.q_lora_rank),
            up_v_weight=torch.randn(heads, config.v_head_dim, config.kv_lora_rank),
            attn_mask_type=AttnMaskType.causal,
        )

    k_ids, gate_ids = seen["k"][:, 0, 0], seen["gate"][:, 0, 0]
    # Gathered to the global length and reordered: a real permutation of the gather order.
    assert k_ids.numel() == LOCAL_ROWS * cp_size
    assert not torch.equal(k_ids, torch.sort(k_ids).values)
    assert torch.equal(torch.sort(k_ids).values, torch.sort(gate_ids).values)
    # The KPool gate follows the indexer K row for row.
    assert torch.equal(k_ids, gate_ids)
    # KPool selection is discrete and runs without an autograd graph.
    assert seen["grad"] is False

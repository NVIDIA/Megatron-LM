# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DSA native context parallelism selecting through a reference binding (optional, Blackwell GPU).

The real ``_forward_dense_cp_native`` and ``_forward_packed_cp_native`` forwards of a DSA layer
run on one GPU as rank r of a contiguous 2-rank split, the collectives replaced by the whole
prompt's projections. The indexer operands are fixed: one nonzero query channel and one distinct
key magnitude per key, exact in BF16 and in FP8 rows, so the upstream masked selector (BF16
scores, cuDNN radix top-k) and the matched-precision reference selector (FP8, the binding of
``backend="reference"``) order every row's keys alike. For one sequence and two packed layouts
(sequence boundaries inside both ranks, and sequences of three tokens and of one token), with
2000 tokens (the gathered keys get the 512-row alignment padding), the bound forward must build
no dense causal mask, and its selection must equal the upstream selection and a float64 oracle on
every row. The cuDNN frontend 1.27 is needed (as for the upstream CSA THD tests), for example
through the runner's ``PYTHONPATH``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]

_TOKENS, _PARTS, _TOPK, _HEADS, _HEAD_DIM = 2000, 2, 64, 32, 128
_LAYER = dict(
    hidden_size=64,
    num_attention_heads=2,
    q_lora_rank=16,
    kv_lora_rank=16,
    qk_nope_head_dim=16,
    qk_rope_head_dim=16,
    v_head_dim=16,
    index_n_heads=_HEADS,
    index_head_dim=_HEAD_DIM,
    index_topk=_TOPK,
    rms_norm_eps=1e-5,
)
_LAYOUTS = {"dense": None, "packed": [0, 700, 1300, 2000], "short": [0, 3, 1000, 1001, 2000]}


def _key_values() -> torch.Tensor:
    """Distinct positive magnitudes in random order, exact in BF16 and in FP8 rows (128 values
    per binade: BF16 mantissas; an FP8 row with one nonzero value holds it exactly)."""
    generator = torch.Generator().manual_seed(7)
    index = torch.arange(_TOKENS, dtype=torch.float64)
    binade, step = torch.div(index, 128, rounding_mode="floor"), index % 128
    values = torch.exp2(binade) * (1 + step / 128)
    return values[torch.randperm(_TOKENS, generator=generator)]


def _oracle(key_values: torch.Tensor, rank: int, offsets: list[int]) -> torch.Tensor:
    """Per local row: the top-k keys of its sequence it sees causally, ascending, -1 padded."""
    local = _TOKENS // _PARTS
    out = torch.full((local, _TOPK), -1, dtype=torch.int32)
    for row in range(local):
        token = rank * local + row
        start = max(offsets[i] for i in range(len(offsets) - 1) if offsets[i] <= token)
        visible = key_values[start : token + 1]
        chosen = visible.topk(min(_TOPK, visible.numel())).indices + start
        out[row, : chosen.numel()] = chosen.sort().values.to(torch.int32)
    return out


def _run(monkeypatch, rank: int, packed: list[int] | None, *, bound: bool):
    """The rank's selection (ascending rows, on the host) and the number of masks built."""
    from megatron.lite.primitive.kernels.indexer_topk import sort_topk_rows_
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.modules.attention.indexer_topk import configure_indexer_topk

    device = torch.device("cuda", torch.cuda.current_device())
    key_values = _key_values()
    q = torch.zeros((_TOKENS, 1, _HEADS, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    q[..., 0] = 1
    k = torch.zeros((_TOKENS, 1, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    k[:, 0, 0] = key_values.to(device=device, dtype=torch.bfloat16)
    weights = torch.ones((_TOKENS, 1, _HEADS), dtype=torch.bfloat16, device=device)

    torch.manual_seed(0)
    layer = dsa.DynamicSparseAttention(**_LAYER, cp_size=_PARTS, cp_rank=rank)
    layer = layer.to(device, torch.bfloat16).eval()
    if bound:
        configure_indexer_topk(
            [layer], {"backend": "reference", "precision": "fast"}, native_format="fp8"
        )
    local = _TOKENS // _PARTS
    rows = slice(rank * local, (rank + 1) * local)
    monkeypatch.setattr(
        layer.indexer, "forward_before_topk", lambda *a, **kw: (q[rows], k[rows], weights[rows])
    )

    def gather(tensor, reorder, *, contiguous=False):
        # The indexer keys of the whole prompt; the attention KV of this rank, repeated.
        if tensor.dim() == 3 and tensor.shape[1:] == (1, _HEAD_DIM):
            return k.clone()
        return torch.cat([tensor] * _PARTS, dim=0).index_select(0, reorder)

    monkeypatch.setattr(layer, "_gather_projected_cp", gather)
    selected, masks = [], []
    kernels = dsa._dsa_kernels
    build_flat, build_mask = kernels.build_flat_topk_idxs, dsa._build_cp_causal_mask

    def flat(topk_indices, **kwargs):
        selected.append(topk_indices.clone())
        return build_flat(topk_indices, **kwargs)

    def sparse_attn(query, kv, sink, idxs, scale, topk_length=None, value_dim=None):
        return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

    def counted_mask(*args, **kwargs):
        masks.append(args)
        return build_mask(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(kernels, "build_flat_topk_idxs", flat)
        patch.setattr(kernels, "dsa_sparse_attn", sparse_attn)
        patch.setattr(dsa, "_build_cp_causal_mask", counted_mask)
        x = torch.randn(1, local, _LAYER["hidden_size"], device=device, dtype=torch.bfloat16)
        cos, sin = dsa.build_rope_cache(
            dim=_LAYER["qk_rope_head_dim"],
            max_position_embeddings=_TOKENS,
            rope_theta=1e4,
            device=device,
        )
        positions = torch.arange(_TOKENS, device=device)[rows].unsqueeze(0)
        with torch.no_grad():
            if packed is None:
                layer._forward_dense_cp_native(x, cos, sin, positions, index_share_state=None)
            else:
                params = SimpleNamespace(
                    cp_layout="contiguous",
                    cu_seqlens_q=torch.tensor(packed, dtype=torch.int32, device=device),
                    cu_seqlens_q_padded=None,
                )
                layer._forward_packed_cp_native(
                    x, cos, sin, positions, params, index_share_state=None
                )
    (topk,) = selected
    rows_selected = None if not bound else layer._indexer_topk.stats.rows
    return sort_topk_rows_(topk[0].to(torch.int32).clone()).cpu(), len(masks), rows_selected


@pytest.mark.parametrize("rank", range(_PARTS))
@pytest.mark.parametrize("name", list(_LAYOUTS))
def test_bound_native_cp_selects_like_upstream(monkeypatch, name, rank):
    packed = _LAYOUTS[name]
    upstream, upstream_masks, _ = _run(monkeypatch, rank, packed, bound=False)
    bound, bound_masks, bound_rows = _run(monkeypatch, rank, packed, bound=True)
    expected = _oracle(_key_values(), rank, [0, _TOKENS] if packed is None else packed)
    assert torch.equal(upstream, expected)
    assert torch.equal(bound, expected)
    assert (upstream_masks, bound_masks) == (1, 0)
    assert bound_rows == _TOKENS // _PARTS

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

@pytest.fixture(autouse=True)
def _te_import_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


class _FakeGroup:
    def __init__(self, size: int = 2, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _KeywordConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_api(group):
    from megatron.lite.primitive.modules.attention import magi

    def magi_attn_varlen_key(**kwargs):
        return SimpleNamespace(
            num_heads_q=kwargs["num_heads_q"],
            num_heads_kv=kwargs["num_heads_kv"],
            total_tokens=int(kwargs["cu_seqlens_q"][-1].item()),
            cp_group=group,
        )

    def dispatch(tensor, key, pad_value=0.0):
        del pad_value
        local_tokens = key.total_tokens // key.cp_group.size()
        start = key.cp_group.rank() * local_tokens
        return tensor.narrow(0, start, local_tokens)

    def undispatch(tensor, key):
        del key
        return tensor

    def calc_attn(query, key, value, runtime_key, softmax_scale=None):
        del key, value, runtime_key, softmax_scale
        return query, None

    return magi._MagiAttentionAPI(
        calc_attn=calc_attn,
        dispatch=dispatch,
        undispatch=undispatch,
        magi_attn_varlen_key=magi_attn_varlen_key,
        DispatchConfig=_KeywordConfig,
        DistAttnConfig=_KeywordConfig,
        OverlapConfig=_KeywordConfig,
        AttnOverlapMode=SimpleNamespace(STATIC="static", DYNAMIC="dynamic"),
    )


def test_pack_magi_forward_kwargs_dispatches_aligned_batch_before_qkv(monkeypatch):
    from megatron.lite.model.protocol_utils import pack_magi_forward_kwargs
    from megatron.lite.primitive.modules.attention import magi
    from megatron.lite.runtime.contracts.data import PackedBatch

    group = _FakeGroup()
    monkeypatch.setattr(magi, "_load_magi_attention_api", lambda: _fake_api(group))
    ps = SimpleNamespace(tp_size=1, cp_size=2, cp_rank=0, cp_group=group)
    model = SimpleNamespace(
        ps=ps,
        config=SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, head_dim=8),
    )
    batch = PackedBatch(
        input_ids=torch.arange(8),
        labels=torch.tensor([10, 11, 12, 20, 21, 22, 23, 24]),
        seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        loss_mask=torch.ones(8),
    )

    kwargs = pack_magi_forward_kwargs(model, batch)

    assert torch.equal(kwargs["input_ids"], torch.tensor([[0, 1, 2, 0, 3, 4]]))
    assert torch.equal(kwargs["labels"], torch.tensor([[11, 12, 0, 0, 21, 22]]))
    assert torch.equal(kwargs["position_ids"], torch.tensor([[0, 1, 2, 0, 0, 1]]))
    assert torch.equal(kwargs["loss_mask"], torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]]))
    packed_seq_params = kwargs["packed_seq_params"]
    assert packed_seq_params.qkv_format == "magi"
    assert packed_seq_params.magi_runtime_key.num_heads_q == 4
    assert packed_seq_params.magi_runtime_key.num_heads_kv == 2
    assert batch.extras["_mlite_magi_runtime_key"] is packed_seq_params.magi_runtime_key


def test_magi_dot_product_attention_uses_microbatch_runtime(monkeypatch):
    from megatron.lite.primitive.modules.attention import magi
    from megatron.lite.primitive.modules.attention.magi import MagiDotProductAttention
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    group = _FakeGroup()
    monkeypatch.setattr(magi, "_load_magi_attention_api", lambda: _fake_api(group))
    module = MagiDotProductAttention(head_dim=4)
    packed_seq_params = PackedSeqParams(
        qkv_format="magi", magi_runtime_key=SimpleNamespace(num_heads_q=4, num_heads_kv=2)
    )
    query = torch.arange(96, dtype=torch.bfloat16).view(6, 4, 4)
    key = torch.ones(6, 2, 4, dtype=torch.bfloat16)
    value = torch.ones_like(key)

    output = module(query, key, value, packed_seq_params=packed_seq_params)

    assert torch.equal(output, query)


def test_magi_runtime_lowers_chunk_size_to_avoid_tail_padding(monkeypatch):
    from megatron.lite.primitive.modules.attention import magi
    from megatron.lite.primitive.modules.attention.magi import (
        MagiAttentionConfig,
        build_magi_attention_runtime_key,
    )

    group = _FakeGroup()
    selected_chunks = []

    def magi_attn_varlen_key(**kwargs):
        chunk_size = kwargs["dist_attn_config"].dispatch_config.chunk_size
        selected_chunks.append(chunk_size)
        effective_chunk = 3 if chunk_size is None else chunk_size
        return SimpleNamespace(
            chunk_size=effective_chunk, pad_size=2 if effective_chunk == 3 else 0
        )

    api = magi._MagiAttentionAPI(
        calc_attn=lambda *args, **kwargs: None,
        dispatch=lambda *args, **kwargs: None,
        undispatch=lambda *args, **kwargs: None,
        magi_attn_varlen_key=magi_attn_varlen_key,
        DispatchConfig=_KeywordConfig,
        DistAttnConfig=_KeywordConfig,
        OverlapConfig=_KeywordConfig,
        AttnOverlapMode=SimpleNamespace(STATIC="static", DYNAMIC="dynamic"),
    )
    monkeypatch.setattr(magi, "_load_magi_attention_api", lambda: api)

    runtime_key = build_magi_attention_runtime_key(
        torch.tensor([0, 4, 8], dtype=torch.int32),
        num_heads_q=4,
        num_heads_kv=2,
        head_dim=8,
        cp_group=group,
        config=MagiAttentionConfig(),
    )

    assert selected_chunks == [None, 2]
    assert runtime_key.chunk_size == 2
    assert runtime_key.pad_size == 0


class _StaticQKV(torch.nn.Module):
    def __init__(self, qkv: torch.Tensor):
        super().__init__()
        self.qkv = qkv

    def forward(self, _x):
        return self.qkv


class _ZeroRotary(torch.nn.Module):
    def forward(self, seq_len, packed_seq=False):
        assert packed_seq is True
        return torch.zeros(seq_len, 1, 1, 2, dtype=torch.bfloat16)


def test_gqa_magi_branch_uses_dispatched_positions_and_runtime(monkeypatch):
    from megatron.lite.primitive.modules.attention import magi
    from megatron.lite.primitive.modules.attention.magi import MagiDotProductAttention
    from megatron.lite.primitive.modules.gqa import GQAttention
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    group = _FakeGroup()
    monkeypatch.setattr(magi, "_load_magi_attention_api", lambda: _fake_api(group))
    attention = GQAttention.__new__(GQAttention)
    torch.nn.Module.__init__(attention)
    attention.num_heads_local = 4
    attention.num_kv_heads_local = 2
    # lite replicates KV heads when num_key_value_heads < tp_size; the real
    # __init__ sets this, and the magi branch reads it like every other path.
    attention._replicate_kv = False
    attention.head_dim = 2
    attention.ps = SimpleNamespace(cp_size=2, cp_group=group)
    attention._output_gate = False
    attention._use_fp32_rope = False
    attention._qkv_layout = "flat"
    attention._mrope_section = None
    attention.attention_backend = "magi"
    qkv = torch.arange(96, dtype=torch.bfloat16).view(6, 1, 16)
    attention.qkv = _StaticQKV(qkv)
    attention.qkv_lora = None
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    attention.rotary = _ZeroRotary()
    attention.core_attn = MagiDotProductAttention(head_dim=2)
    attention.proj = torch.nn.Identity()
    attention.proj_lora = None
    packed_seq_params = PackedSeqParams(
        qkv_format="magi",
        max_seqlen_q=4,
        max_seqlen_kv=4,
        magi_runtime_key=SimpleNamespace(num_heads_q=4, num_heads_kv=2),
    )
    positions = torch.tensor([[0, 1, 2, 3, 0, 1]])

    output = attention(
        torch.zeros(6, 1, 8, dtype=torch.bfloat16),
        position_ids=positions,
        packed_seq_params=packed_seq_params,
    )

    assert output.shape == (6, 1, 8)
    assert torch.equal(output, qkv[..., :8])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"use_thd": False}, "use_thd"),
        ({"parallel": {"cp": 1}}, "CP>1"),
        ({"parallel": {"cp": 2, "pp": 2}}, "PP=1"),
        ({"mtp_enable": True}, "MTP"),
    ],
)
def test_qwen3_magi_config_rejects_unsupported_combinations(overrides, message):
    from megatron.lite.model.qwen3_moe.lite.protocol import (
        ImplConfig,
        _validate_magi_attention_config,
    )
    from megatron.lite.runtime.contracts import ParallelConfig

    overrides = dict(overrides)
    kwargs = {
        "attention_backend_override": "magi",
        "use_thd": True,
        "parallel": ParallelConfig(cp=2),
    }
    parallel_overrides = overrides.pop("parallel", None)
    kwargs.update(overrides)
    if parallel_overrides is not None:
        kwargs["parallel"] = ParallelConfig(**parallel_overrides)

    with pytest.raises(ValueError, match=message):
        _validate_magi_attention_config(ImplConfig(**kwargs))


def test_qwen3_magi_config_accepts_supported_static_cp():
    from megatron.lite.model.qwen3_moe.lite.protocol import (
        ImplConfig,
        _validate_magi_attention_config,
    )
    from megatron.lite.runtime.contracts import ParallelConfig

    _validate_magi_attention_config(
        ImplConfig(
            attention_backend_override="magi",
            use_thd=True,
            parallel=ParallelConfig(tp=2, cp=2),
        )
    )


def test_magi_attention_config_defaults_trust_auto():
    from megatron.lite.primitive.modules.attention.magi import (
        MagiAttentionConfig,
        resolve_magi_attention_config,
    )

    config = MagiAttentionConfig()
    # None/None = delegate chunk sizing and overlap staging to magi itself.
    assert config.chunk_size is None
    assert config.overlap_degree is None
    # The calibration seam is a pass-through until profiling says otherwise.
    resolved = resolve_magi_attention_config(config, total_tokens=1024, cp_size=4)
    assert resolved is config
    # Dynamic mode stays selectable alongside explicit expert overrides.
    assert MagiAttentionConfig(chunk_size=64, overlap_degree=2).overlap_degree == 2


def test_gqa_attention_backend_hot_swap_contract():
    from megatron.lite.primitive.modules.attention.magi import MagiDotProductAttention
    from megatron.lite.primitive.modules.gqa import GQAttention

    attention = object.__new__(GQAttention)
    torch.nn.Module.__init__(attention)
    attention._mrope_section = None
    attention._use_thd = True
    attention.head_dim = 8
    attention.num_heads_local = 4
    attention.num_kv_heads_local = 2
    attention.ps = SimpleNamespace(cp_size=1, cp_group=None, cp_global_ranks=None)
    attention.attention_backend = "te"
    attention.core_attn = torch.nn.Identity()  # stands in for the te module

    # te -> magi builds the parameter-free adapter and flips the attribute.
    attention.set_attention_backend("magi")
    assert isinstance(attention.core_attn, MagiDotProductAttention)
    assert attention.attention_backend == "magi"
    # Parameter-free contract: the swap can never touch checkpoints.
    assert dict(attention.core_attn.state_dict()) == {}

    # Same-backend call is a no-op preserving module identity.
    module = attention.core_attn
    attention.set_attention_backend("magi")
    assert attention.core_attn is module

    with pytest.raises(ValueError, match="Unsupported"):
        attention.set_attention_backend("flash")

    attention.attention_backend = "te"
    attention._mrope_section = [16, 24, 24]
    with pytest.raises(ValueError, match="MRoPE"):
        attention.set_attention_backend("magi")

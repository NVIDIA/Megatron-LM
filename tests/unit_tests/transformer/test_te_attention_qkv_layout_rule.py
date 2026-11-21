# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Which q, k and v TEDotProductAttention makes dense before Transformer Engine reads them.

TE's ``get_qkv_layout`` takes views sharing one storage at the offsets of a packed layout for a
dense packed buffer, and any other tensor for a dense tensor in its format. ``_dense_unpacked_qkv``
passes through only dense tensors and the packed views TE reads correctly, and makes every other
input contiguous. The GPU checks against TE are in ``test_te_attention_qkv_layout.py``.
"""

import pytest
import torch

from megatron.core.extensions.transformer_engine import _dense_unpacked_qkv

# Leading dimensions of q/k/v ([..., heads, head_dim]) in each format TEDotProductAttention takes.
FORMATS = {"sbhd": (5, 2), "bshd": (2, 5), "thd": (7,)}
HEADS, DIM = 4, 8


def _strided_view(*leading, heads=HEADS):
    """A [*leading, heads, DIM] view of a wider buffer: strided, at storage offset 0."""
    return torch.randn(*leading, heads, 3 * DIM)[..., :DIM]


def _fused_qkv(*leading, groups, heads_per_group, gate=False):
    """q, k and v as SelfAttention.get_query_key_value_tensors splits the fused QKV projection."""
    split = [heads_per_group * DIM] * (2 if gate else 1) + [DIM, DIM]
    mixed = torch.randn(*leading, groups, sum(split))
    query, *_, key, value = mixed.split(split, dim=-1)
    return query.reshape(*leading, groups * heads_per_group, DIM), key, value


def _assert_passed_through(out, inputs):
    assert all(a is b for a, b in zip(out, inputs))


def _assert_made_dense(out, inputs, which="qkv"):
    for name, made, original in zip("qkv", out, inputs):
        if name in which:
            assert made is not original and made.is_contiguous(), name
            assert torch.equal(made, original), name
        else:
            assert made is original, name


@pytest.mark.parametrize("fmt", FORMATS)
class TestDenseUnpackedQKV:

    def test_dense_inputs_are_passed_through(self, fmt):
        qkv = tuple(torch.randn(*FORMATS[fmt], HEADS, DIM) for _ in range(3))
        _assert_passed_through(_dense_unpacked_qkv(*qkv), qkv)

    def test_fused_qkv_views_are_passed_through(self, fmt):
        """MHA training: TE reads the views of the fused QKV projection as sbh3d (th3d, ...)."""
        qkv = _fused_qkv(*FORMATS[fmt], groups=HEADS, heads_per_group=1)
        assert not qkv[0].is_contiguous()
        _assert_passed_through(_dense_unpacked_qkv(*qkv), qkv)

    def test_fused_kv_views_are_passed_through(self, fmt):
        """Cross-attention: TE reads the views of the fused KV projection as sbhd_sbh2d (...)."""
        leading = FORMATS[fmt]
        key, value = torch.randn(*leading, HEADS, 2 * DIM).chunk(2, dim=-1)
        qkv = (torch.randn(*leading, HEADS, DIM), key, value)
        _assert_passed_through(_dense_unpacked_qkv(*qkv), qkv)

        qkv = (_strided_view(*leading), key, value)
        _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv, which="q")

    @pytest.mark.parametrize("groups", [1, 2], ids=["mqa", "gqa"])
    def test_fused_qkv_views_with_grouped_heads_are_made_dense(self, fmt, groups):
        """TE detects no packed layout in an MQA/GQA split; its own fallback copied them too."""
        qkv = _fused_qkv(*FORMATS[fmt], groups=groups, heads_per_group=HEADS // groups)
        strided = "".join(name for name, t in zip("qkv", qkv) if not t.is_contiguous())
        assert "k" in strided and "v" in strided
        _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv, which=strided)

    def test_strided_query_beside_separate_kv_is_made_dense(self, fmt):
        """Static inference without RoPE: the query is a view of the fused QKV projection, k and v
        come from the separate KV cache."""
        leading = FORMATS[fmt]
        qkv = (_strided_view(*leading), *(torch.randn(*leading, HEADS, DIM) for _ in range(2)))
        _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv, which="q")

    def test_strided_separate_kv_are_made_dense(self, fmt):
        leading = FORMATS[fmt]
        qkv = (torch.randn(*leading, HEADS, DIM), _strided_view(*leading), _strided_view(*leading))
        _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv, which="kv")

    def test_query_and_key_sharing_a_buffer_are_made_dense(self, fmt):
        """q and k in one buffer without v is not a packed layout TE knows."""
        leading = FORMATS[fmt]
        query, key = torch.randn(*leading, HEADS, 2 * DIM).chunk(2, dim=-1)
        qkv = (query, key, torch.randn(*leading, HEADS, DIM))
        _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv, which="qk")


def test_static_kv_cache_batch_slices_are_made_dense():
    """A batch smaller than the cache's max_batch_size reads strided slices of the cache."""
    key_cache, value_cache = (torch.randn(64, 4, 2, DIM) for _ in range(2))  # [seq, batch, g, hn]
    qkv = (torch.randn(16, 2, HEADS, DIM), key_cache[:16, 0:2], value_cache[:16, 0:2])
    _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv, which="kv")


def test_key_that_is_value_is_made_dense():
    """One strided cache slice passed as both key and value is not a packed KV buffer."""
    kv = torch.randn(64, 4, 2, DIM)[:16, 0:2]
    qkv = (torch.randn(16, 2, HEADS, DIM), kv, kv)
    _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv, which="kv")


@pytest.mark.parametrize(
    "leading, groups, heads_per_group",
    [((1, 2), 1, 4), ((2, 1), 1, 1), ((1, 1), 2, 1)],
    ids=["s1-b2-mqa", "s2-b1-g1", "s1-b1-g2"],
)
def test_gated_fused_qkv_at_two_tokens_per_group_is_made_dense(leading, groups, heads_per_group):
    """Gated attention without RoPE or qk-norm at s * b * groups == 2: k sits at offset numel(q),
    so TE takes q for a dense tensor and k/v for a packed KV buffer (sbhd_sb2hd/sbhd_sbh2d)."""
    qkv = _fused_qkv(*leading, groups=groups, heads_per_group=heads_per_group, gate=True)
    assert qkv[1].storage_offset() == qkv[0].numel()
    _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv)


def test_fused_qkv_views_of_a_batch_slice_are_made_dense():
    """TE detects sbh3d from offsets and equal strides, then reads a dense [s, b, h, 3, d]."""
    qkv = torch.randn(5, 4, HEADS, 3 * DIM)[:, :2].split(DIM, dim=-1)
    _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv)


def test_fused_qkv_views_of_a_transposed_buffer_are_made_dense():
    """bshd views of an sbhd fused QKV buffer: TE detects bsh3d and reads a dense [b, s, h, 3, d]."""
    qkv = torch.randn(5, 2, HEADS, 3 * DIM).transpose(0, 1).split(DIM, dim=-1)
    _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv)


def test_fused_kv_views_of_a_batch_slice_are_made_dense():
    key, value = torch.randn(5, 4, HEADS, 2 * DIM)[:, :2].chunk(2, dim=-1)
    qkv = (torch.randn(5, 2, HEADS, DIM), key, value)
    _assert_made_dense(_dense_unpacked_qkv(*qkv), qkv, which="kv")

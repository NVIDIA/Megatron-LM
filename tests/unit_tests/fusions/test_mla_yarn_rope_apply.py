# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings import rope_utils as rope_utils_module
from megatron.core.models.common.embeddings import should_use_fused_mla_rope
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.test_utilities import Utils

try:
    import triton

    from megatron.core.fusions import fused_mla_yarn_rope_apply as fused_mla_rope_module
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        HAVE_TRITON,
        _mla_rope_bwd_kernel,
        _mla_rope_fwd_inplace_kernel,
        fused_apply_mla_rope_for_q,
        fused_mla_rope_concat,
        fused_mla_rope_inplace,
        fused_mla_rope_kv_split,
        fused_mla_rope_out_of_place,
    )
except ImportError:
    HAVE_TRITON = False
    triton = None
    fused_mla_rope_module = None
    _mla_rope_bwd_kernel = None
    _mla_rope_fwd_inplace_kernel = None
    fused_apply_mla_rope_for_q = None
    fused_mla_rope_concat = None
    fused_mla_rope_inplace = None
    fused_mla_rope_kv_split = None
    fused_mla_rope_out_of_place = None


@pytest.mark.parametrize(
    ("apply_rope_fusion", "rope_type", "rotary_percent", "expected"),
    [
        (False, "rope", 1.0, False),
        (True, "rope", 1.0, True),
        (True, "rope", 0.5, False),
        (True, "yarn", 0.5, True),
    ],
)
def test_mla_rope_fusion_selection(apply_rope_fusion, rope_type, rotary_percent, expected):
    """Partial standard RoPE falls back while full RoPE and YaRN retain fusion."""
    config = SimpleNamespace(
        apply_rope_fusion=apply_rope_fusion, rope_type=rope_type, rotary_percent=rotary_percent
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert should_use_fused_mla_rope(config) is expected


def test_partial_standard_rope_fallback_warns_once(monkeypatch):
    """Partial standard RoPE reports its performance fallback only once."""
    config = SimpleNamespace(apply_rope_fusion=True, rope_type="rope", rotary_percent=0.5)
    monkeypatch.setattr(rope_utils_module, "_ROPE_FUSION_FALLBACK_WARNINGS", set())

    with pytest.warns(UserWarning, match="Falling back to the unfused path"):
        assert not should_use_fused_mla_rope(config)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert not should_use_fused_mla_rope(config)
    assert not recorded


def dtype_tols(dtype):
    if dtype == torch.float32:
        return dict(rtol=1.0e-6, atol=1.0e-6)
    elif dtype == torch.float16:
        return dict(rtol=3.0e-3, atol=1.0e-5)
    elif dtype == torch.bfloat16:
        return dict(rtol=2.0e-2, atol=5.0e-2)
    else:
        raise ValueError(f"Unsuppored dtype ({dtype})")


class FakeCPGroup:
    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class TestApplyRotaryPosEmbTHD:
    @pytest.mark.parametrize(
        ("unsupported_kwargs", "warning_text"),
        [
            ({"inverse": True}, "inverse RoPE is not supported"),
            (
                {"mla_rotary_interleaved": True, "mla_output_remove_interleaving": True},
                "MLA-style interleaving",
            ),
            ({"mscale": 2.0}, "mscale=2.0 is not supported"),
        ],
    )
    def test_unsupported_fusion_options_use_unfused(self, unsupported_kwargs, warning_text):
        cp_group = FakeCPGroup()
        t = torch.randn(4, 2, 8)
        freqs = torch.randn(2, 1, 1, 8)
        cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
        config = TransformerConfig(
            num_attention_heads=2, num_layers=1, apply_rope_fusion=True, rotary_interleaved=False
        )
        expected = rope_utils_module._apply_rotary_pos_emb_thd(
            t, cu_seqlens, freqs, cp_group=cp_group, max_seqlen=2, **unsupported_kwargs
        )

        fused_mock = MagicMock(return_value=t.clone())
        with (
            patch.object(rope_utils_module, "fused_apply_rotary_pos_emb_thd", fused_mock),
            pytest.warns(UserWarning, match=warning_text),
        ):
            output = apply_rotary_pos_emb(
                t,
                freqs,
                config,
                cu_seqlens=cu_seqlens,
                cp_group=cp_group,
                max_seqlen=2,
                **unsupported_kwargs,
            )

        fused_mock.assert_not_called()
        torch.testing.assert_close(output, expected)

    def test_packed_freqs_returns_offset_mapped_output_for_context_parallel(self):
        cp_group = FakeCPGroup(size=2, rank=0)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        t = torch.randn(4, 2, 8)
        freqs = torch.randn(8, 1, 1, 8)

        out = rope_utils_module._apply_rotary_pos_emb_thd(
            t, cu_seqlens, freqs, cp_group=cp_group, max_seqlen=4
        )

        expected_freqs = torch.cat([freqs[0:1], freqs[3:4], freqs[4:5], freqs[7:8]], dim=0)
        expected = rope_utils_module._apply_rotary_pos_emb_bshd(
            t.unsqueeze(1), expected_freqs
        ).squeeze(1)

        torch.testing.assert_close(out, expected)

    def test_max_seqlen_freqs_returns_sequence_mapped_output_for_context_parallel(self):
        cp_group = FakeCPGroup(size=2, rank=1)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        t = torch.randn(4, 2, 8)
        freqs = torch.randn(4, 1, 1, 8)

        out = rope_utils_module._apply_rotary_pos_emb_thd(
            t, cu_seqlens, freqs, cp_group=cp_group, max_seqlen=4
        )

        expected_freqs = torch.cat([freqs[1:2], freqs[2:3]], dim=0)
        expected_slices = []
        for x in torch.split(t, [2, 2]):
            expected_slices.append(
                rope_utils_module._apply_rotary_pos_emb_bshd(
                    x.unsqueeze(1), expected_freqs
                ).squeeze(1)
            )
        expected = torch.cat(expected_slices, dim=0)

        torch.testing.assert_close(out, expected)

    def test_missing_max_seqlen_preserves_legacy_packed_freq_mapping(self):
        cp_group = FakeCPGroup(size=2, rank=0)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        t = torch.randn(4, 2, 8)
        freqs = torch.randn(8, 1, 1, 8)

        legacy_out = rope_utils_module._apply_rotary_pos_emb_thd(
            t, cu_seqlens, freqs, cp_group=cp_group
        )
        explicit_out = rope_utils_module._apply_rotary_pos_emb_thd(
            t, cu_seqlens, freqs, cp_group=cp_group, max_seqlen=4
        )

        torch.testing.assert_close(legacy_out, explicit_out)

    def test_shared_max_seqlen_maps_asymmetric_query_sequences_from_zero(self):
        cp_group = FakeCPGroup(size=1, rank=0)
        cu_seqlens_q = torch.tensor([0, 3, 6], dtype=torch.int32)
        t = torch.randn(6, 2, 8)
        freqs = torch.randn(4, 1, 1, 8)

        max_seqlen_q = 3
        max_seqlen_kv = freqs.size(0)
        assert max_seqlen_q < max_seqlen_kv < t.size(0)
        combined_max_seqlen = max(max_seqlen_q, max_seqlen_kv)
        out = rope_utils_module._apply_rotary_pos_emb_thd(
            t, cu_seqlens_q, freqs, cp_group=cp_group, max_seqlen=combined_max_seqlen
        )
        compatibility_out = rope_utils_module._apply_rotary_pos_emb_thd(
            t, cu_seqlens_q, freqs, cp_group=cp_group
        )

        expected_freqs = freqs[torch.tensor([0, 1, 2, 0, 1, 2])]
        expected = rope_utils_module._apply_rotary_pos_emb_bshd(
            t.unsqueeze(1), expected_freqs
        ).squeeze(1)

        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(out, compatibility_out)


class _SaveOutputForBackward(torch.autograd.Function):
    """Minimal stand-in for a kernel whose backward consumes its output."""

    @staticmethod
    def forward(ctx, tensor):
        output = tensor.clone()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, _grad_output):
        (saved_output,) = ctx.saved_tensors
        return saved_output


def _test_fused_mla_rope_inplace(
    input_format, inverse=False, remove_interleaving=False, rope_first=False, num_heads=32
):
    assert fused_mla_rope_inplace is not None
    q_dim = 128
    emb_dim = 64
    dtype = torch.bfloat16
    transformer_config = TransformerConfig(
        num_attention_heads=num_heads,
        num_layers=1,
        rotary_interleaved=False,
        multi_latent_attention=True,
    )

    max_seqlen = None
    if input_format == "sbhd":
        cu_seqlens = None
        seqlen = 1024
        batch_size = 2
        yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=seqlen)
        freqs, mscale = yarn_rope(seqlen, 0)
        cos = (torch.cos(freqs) * mscale).to(dtype)
        sin = (torch.sin(freqs) * mscale).to(dtype)

        pytorch_fwd_input = torch.randn(
            (seqlen, batch_size, num_heads, q_dim + emb_dim), dtype=dtype, device='cuda'
        )
        pytorch_bwd_input = torch.randn(
            (seqlen, batch_size, num_heads, q_dim + emb_dim), dtype=dtype, device='cuda'
        )
    else:
        cu_seqlens = [0, 27, 54, 99, 128]
        total_seqlen = cu_seqlens[-1]
        max_seqlen = 0
        for i in range(len(cu_seqlens) - 1):
            max_seqlen = max(max_seqlen, cu_seqlens[i + 1] - cu_seqlens[i])
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device='cuda')
        yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=max_seqlen)
        freqs, mscale = yarn_rope(max_seqlen, 0)
        cos = (torch.cos(freqs) * mscale).to(dtype)
        sin = (torch.sin(freqs) * mscale).to(dtype)

        pytorch_fwd_input = torch.randn(
            (total_seqlen, num_heads, q_dim + emb_dim), dtype=dtype, device='cuda'
        )
        pytorch_bwd_input = torch.randn(
            (total_seqlen, num_heads, q_dim + emb_dim), dtype=dtype, device='cuda'
        )

    pytorch_fwd_input.requires_grad_(True)
    fused_fwd_input = pytorch_fwd_input.detach().clone()
    fused_fwd_input.requires_grad_(True)
    fused_bwd_input = pytorch_bwd_input.detach().clone()

    if rope_first:
        pe, no_pe = torch.split(pytorch_fwd_input, [emb_dim, q_dim], dim=-1)
    else:
        no_pe, pe = torch.split(pytorch_fwd_input, [q_dim, emb_dim], dim=-1)
    pe_output = apply_rotary_pos_emb(
        pe,
        freqs,
        transformer_config,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        mscale=mscale,
        cp_group=FakeCPGroup(),
        mla_rotary_interleaved=True,
        inverse=inverse,
        mla_output_remove_interleaving=remove_interleaving,
    )
    pytorch_output = (
        torch.concat([pe_output, no_pe], dim=-1)
        if rope_first
        else torch.concat([no_pe, pe_output], dim=-1)
    )
    pytorch_output.backward(pytorch_bwd_input, retain_graph=True)

    fused_output = fused_mla_rope_inplace(
        fused_fwd_input,
        cos,
        sin,
        q_dim,
        emb_dim,
        cu_seqlens_q=cu_seqlens,
        inverse=inverse,
        remove_interleaving=remove_interleaving,
        rope_first=rope_first,
    )
    fused_output.backward(fused_bwd_input, retain_graph=True)

    tols = dtype_tols(dtype)
    torch.testing.assert_close(
        pytorch_output.float(),
        fused_output.float(),
        msg=lambda msg: f"Mismatch in fwd: {msg}",
        **tols,
    )
    torch.testing.assert_close(
        pytorch_fwd_input.grad.float(),
        fused_fwd_input.grad.float(),
        msg=lambda msg: f"Mismatch in bwd: {msg}",
        **tols,
    )
    nope_slice = slice(emb_dim, None) if rope_first else slice(0, q_dim)
    torch.testing.assert_close(
        fused_output[..., nope_slice], pytorch_fwd_input.detach()[..., nope_slice], rtol=0, atol=0
    )
    torch.testing.assert_close(
        fused_fwd_input.grad[..., nope_slice], pytorch_bwd_input[..., nope_slice], rtol=0, atol=0
    )


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "sequence_lengths,global_start,local_rows",
    [
        pytest.param([8], -2, 12, id="single-sequence-boundary-and-tail-padding"),
        pytest.param([0, 1, 0, 3, 0, 5], -2, 13, id="zero-length-sequences"),
        pytest.param([1] * 257, -2, 261, id="many-short-sequences"),
    ],
)
def test_mla_rope_contiguous_thd_global_start_matches_position_ids(
    inverse, sequence_lengths, global_start, local_rows
):
    """Kernel-side THD row mapping must match explicit packed-sequence positions."""
    assert fused_mla_rope_inplace is not None
    torch.manual_seed(1234)

    sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device="cuda"), sequence_lengths.cumsum(dim=0))
    )
    global_rows = torch.arange(
        global_start, global_start + local_rows, dtype=torch.int32, device="cuda"
    )
    sequence_ids = torch.bucketize(
        global_rows, cu_seqlens[1:], out_int32=True, right=True
    ).clamp_max(cu_seqlens.numel() - 2)
    sequence_starts = cu_seqlens[sequence_ids]
    sequence_ends = cu_seqlens[sequence_ids + 1]
    valid_rows = (global_rows >= sequence_starts) & (global_rows < sequence_ends)
    position_ids = torch.where(valid_rows, global_rows - sequence_starts, 0)

    nope_dim = 4
    emb_dim = 8
    num_heads = 2
    max_sequence_length = max(sequence_lengths.max().item(), 1)
    freqs = torch.randn(max_sequence_length, emb_dim, dtype=torch.float32, device="cuda")
    cos = freqs.cos().contiguous()
    sin = freqs.sin().contiguous()
    values = torch.randn(
        local_rows, num_heads, nope_dim + emb_dim, dtype=torch.float32, device="cuda"
    )
    position_input = values.clone().requires_grad_(True)
    contiguous_input = values.clone().requires_grad_(True)

    position_output = fused_mla_rope_inplace(
        position_input,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q=cu_seqlens,
        inverse=inverse,
        remove_interleaving=True,
        position_ids=position_ids,
    )
    contiguous_output = fused_mla_rope_inplace(
        contiguous_input,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q=cu_seqlens,
        inverse=inverse,
        remove_interleaving=True,
        thd_global_start=global_start,
    )

    torch.testing.assert_close(contiguous_output, position_output)
    grad = torch.randn_like(position_output)
    position_output.backward(grad.clone())
    contiguous_output.backward(grad.clone())
    torch.testing.assert_close(contiguous_input.grad, position_input.grad)


def _test_fused_mla_rope_kv_split(input_format, remove_interleaving=False, num_heads=32):
    assert fused_mla_rope_kv_split is not None
    k_dim = 128
    v_dim = 128
    emb_dim = 64
    dtype = torch.bfloat16
    transformer_config = TransformerConfig(
        num_attention_heads=num_heads,
        num_layers=1,
        rotary_interleaved=False,
        multi_latent_attention=True,
    )

    max_seqlen = None
    if input_format == "sbhd":
        cu_seqlens = None
        seqlen = 1024
        batch_size = 2
        yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=seqlen)
        freqs, mscale = yarn_rope(seqlen, 0)
        cos = (torch.cos(freqs) * mscale).to(dtype)
        sin = (torch.sin(freqs) * mscale).to(dtype)

        pytorch_fwd_kv_input = torch.randn(
            (seqlen, batch_size, num_heads, k_dim + v_dim), dtype=dtype, device='cuda'
        )
        pytorch_fwd_emb_input = torch.randn(
            (seqlen, batch_size, 1, emb_dim), dtype=dtype, device='cuda'
        )
        pytorch_bwd_k_input = torch.randn(
            (seqlen, batch_size, num_heads, k_dim + emb_dim), dtype=dtype, device='cuda'
        )
        pytorch_bwd_v_input = torch.randn(
            (seqlen, batch_size, num_heads, v_dim), dtype=dtype, device='cuda'
        )
    else:
        cu_seqlens = [0, 27, 54, 99, 128]
        total_seqlen = cu_seqlens[-1]
        max_seqlen = 0
        for i in range(len(cu_seqlens) - 1):
            max_seqlen = max(max_seqlen, cu_seqlens[i + 1] - cu_seqlens[i])
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device='cuda')
        yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=max_seqlen)
        freqs, mscale = yarn_rope(max_seqlen, 0)
        cos = (torch.cos(freqs) * mscale).to(dtype)
        sin = (torch.sin(freqs) * mscale).to(dtype)

        pytorch_fwd_kv_input = torch.randn(
            (total_seqlen, num_heads, k_dim + v_dim), dtype=dtype, device='cuda'
        )
        pytorch_fwd_emb_input = torch.randn((total_seqlen, 1, emb_dim), dtype=dtype, device='cuda')
        pytorch_bwd_k_input = torch.randn(
            (total_seqlen, num_heads, k_dim + emb_dim), dtype=dtype, device='cuda'
        )
        pytorch_bwd_v_input = torch.randn(
            (total_seqlen, num_heads, v_dim), dtype=dtype, device='cuda'
        )

    pytorch_fwd_kv_input.requires_grad_(True)
    pytorch_fwd_emb_input.requires_grad_(True)
    fused_fwd_kv_input = pytorch_fwd_kv_input.detach()
    fused_fwd_kv_input.requires_grad_(True)
    fused_fwd_emb_input = pytorch_fwd_emb_input.detach()
    fused_fwd_emb_input.requires_grad_(True)
    fused_bwd_k_input = pytorch_bwd_k_input.detach()
    fused_bwd_v_input = pytorch_bwd_v_input.detach()

    pe_output = apply_rotary_pos_emb(
        pytorch_fwd_emb_input,
        freqs,
        transformer_config,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        mscale=mscale,
        cp_group=FakeCPGroup(),
        mla_rotary_interleaved=True,
        mla_output_remove_interleaving=remove_interleaving,
    )
    if input_format == "sbhd":
        pe_output = pe_output.expand(-1, -1, num_heads, -1)
    else:
        pe_output = pe_output.expand(-1, num_heads, -1)
    k, pytorch_v_output = torch.split(pytorch_fwd_kv_input, [k_dim, v_dim], dim=-1)
    pytorch_k_output = torch.concat([k, pe_output], dim=-1)
    torch.autograd.backward(
        (pytorch_k_output, pytorch_v_output), (pytorch_bwd_k_input, pytorch_bwd_v_input)
    )

    fused_k_output, fused_v_output = fused_mla_rope_kv_split(
        fused_fwd_kv_input,
        fused_fwd_emb_input,
        cos,
        sin,
        emb_dim,
        k_dim,
        v_dim,
        cu_seqlens_kv=cu_seqlens,
        remove_interleaving=remove_interleaving,
    )
    torch.autograd.backward(
        (fused_k_output, fused_v_output), (fused_bwd_k_input, fused_bwd_v_input)
    )

    tols = dtype_tols(dtype)
    torch.testing.assert_close(
        pytorch_k_output.float(),
        fused_k_output.float(),
        msg=lambda msg: f"Mismatch in k fwd: {msg}",
        **tols,
    )
    torch.testing.assert_close(
        pytorch_v_output.float(),
        fused_v_output.float(),
        msg=lambda msg: f"Mismatch in v fwd: {msg}",
        **tols,
    )
    torch.testing.assert_close(
        pytorch_fwd_kv_input.grad.float(),
        fused_fwd_kv_input.grad.float(),
        msg=lambda msg: f"Mismatch in kv bwd: {msg}",
        **tols,
    )
    torch.testing.assert_close(
        pytorch_fwd_emb_input.grad.float(),
        fused_fwd_emb_input.grad.float(),
        msg=lambda msg: f"Mismatch in emb bwd: {msg}",
        **tols,
    )


def _make_noncontiguous_leaf(values):
    storage = torch.empty(
        *values.shape[:-1], values.size(-1) * 2, dtype=values.dtype, device=values.device
    )
    result = storage[..., ::2]
    result.copy_(values)
    return result.detach().requires_grad_(True)


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_heads", [1, 64])
@pytest.mark.parametrize("layout", ["sbhd", "thd", "thd_cp_padding"])
def test_mla_rope_concat_matches_native(layout, num_heads, dtype):
    """Fused packing must match an independent PyTorch RoPE + concat reference."""
    assert fused_mla_rope_concat is not None
    nope_dim = 512
    emb_dim = 64
    config = TransformerConfig(
        num_attention_heads=num_heads,
        num_layers=1,
        rotary_interleaved=False,
        multi_latent_attention=True,
    )

    if layout == "sbhd":
        max_seqlen = 16
        input_shape = (max_seqlen, 2, num_heads)
        cu_seqlens = None
        cp_size, cp_rank = 1, 0
        valid_tokens = input_shape[0] * input_shape[1]
    elif layout == "thd":
        seqlens = [12, 20, 24]
        max_seqlen = max(seqlens)
        cu_seqlens = torch.tensor([0, 12, 32, 56], dtype=torch.int32, device="cuda")
        input_shape = (56, num_heads)
        cp_size, cp_rank = 1, 0
        valid_tokens = input_shape[0]
    else:
        global_seqlens = [16, 24, 32]
        max_seqlen = max(global_seqlens)
        cu_seqlens = torch.tensor([0, 16, 40, 72], dtype=torch.int32, device="cuda")
        cp_size, cp_rank = 2, 1
        valid_tokens = sum(global_seqlens) // cp_size
        input_shape = (valid_tokens + 4, num_heads)

    rotary = RotaryEmbedding(emb_dim, rotary_percent=1.0, rotary_base=10000)
    freqs = rotary(max_seqlen, packed_seq=cu_seqlens is not None)
    cos = freqs.cos().to(device="cuda", dtype=dtype).contiguous()
    sin = freqs.sin().to(device="cuda", dtype=dtype).contiguous()
    nope_values = torch.randn(*input_shape, nope_dim, dtype=dtype, device="cuda")
    rope_values = torch.randn(*input_shape, emb_dim, dtype=dtype, device="cuda")

    native_nope = _make_noncontiguous_leaf(nope_values)
    native_rope = _make_noncontiguous_leaf(rope_values)
    fused_nope = _make_noncontiguous_leaf(nope_values)
    fused_rope = _make_noncontiguous_leaf(rope_values)
    assert not fused_nope.is_contiguous() and not fused_rope.is_contiguous()

    rotated = apply_rotary_pos_emb(
        native_rope,
        freqs,
        config,
        cu_seqlens=cu_seqlens,
        cp_group=FakeCPGroup(cp_size, cp_rank),
        mla_rotary_interleaved=True,
        max_seqlen=max_seqlen if cu_seqlens is not None else None,
    )
    native_output = torch.cat((native_nope, rotated), dim=-1)
    fused_output = fused_mla_rope_concat(
        fused_nope, fused_rope, cos, sin, cu_seqlens=cu_seqlens, cp_rank=cp_rank, cp_size=cp_size
    )
    assert fused_output.is_contiguous()

    if layout == "sbhd":
        valid_output = (slice(None),)
    else:
        valid_output = (slice(0, valid_tokens),)
    tols = dtype_tols(dtype)
    torch.testing.assert_close(
        native_output[valid_output].float(), fused_output[valid_output].float(), **tols
    )

    grad = torch.randn_like(fused_output)
    if layout != "sbhd" and valid_tokens < grad.size(0):
        grad[valid_tokens:] = 0
    native_output.backward(grad)
    fused_output.backward(grad)
    torch.testing.assert_close(native_nope.grad.float(), fused_nope.grad.float(), **tols)
    torch.testing.assert_close(native_rope.grad.float(), fused_rope.grad.float(), **tols)


# -------------------------------------------------------------------------------------------------
# Partial head block coverage.
#
# Every kernel here is launched over cdiv(head_num, BLOCK_H) head programs, so a head count that
# BLOCK_H does not divide leaves the final program covering head rows that do not exist. BLOCK_H is
# autotuned over {1, 2, ..., 128} and the tuner may pick a divisor, so an awkward head count is not
# enough on its own -- a run can pass without the partial block ever being built. The tests below
# pin BLOCK_H instead, and each one spells out the head count and the head-block size it needs.
# -------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _pinned_block_h(block_h):
    """Leave every autotuned kernel one BLOCK_H (the head-block size) to choose from."""
    kernels = [
        fused_mla_rope_module._autotuned_mla_rope_fwd_inplace_kernel,
        fused_mla_rope_module._autotuned_mla_rope_bwd_inplace_kernel,
        fused_mla_rope_module._mla_rope_fwd_kv_split_kernel,
        fused_mla_rope_module._mla_rope_bwd_kv_split_kernel,
    ]
    saved = [(kernel, kernel.configs, kernel.cache) for kernel in kernels]
    try:
        for kernel in kernels:
            kernel.configs = [triton.Config({"BLOCK_H": block_h})]
            kernel.cache = {}
        yield
    finally:
        for kernel, configs, cache in saved:
            kernel.configs = configs
            kernel.cache = cache


def _thd_rope_tables(cu_seqlens_list, emb_dim, dtype):
    """cos/sin for a THD batch, plus the cu_seqlens tensor the kernels want."""
    max_seqlen = max(b - a for a, b in zip(cu_seqlens_list, cu_seqlens_list[1:]))
    yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=max_seqlen)
    freqs, mscale = yarn_rope(max_seqlen, 0)
    cos = (torch.cos(freqs) * mscale).to(dtype)
    sin = (torch.sin(freqs) * mscale).to(dtype)
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device="cuda")
    return cos, sin, cu_seqlens


def _run_kv_split_once(block_h, remove_interleaving, num_heads, seed=1234):
    """One seeded forward and backward of the KV-split path at a pinned BLOCK_H.

    Returns the four tensors whose values must not depend on the tiling: both forward outputs, the
    kv gradient, and the k_pos_emb gradient. The last one is the ``dEMB`` reduction, the one site in
    this file where an out-of-range lane survives into the result instead of being dropped by a
    masked store.
    """
    k_dim = v_dim = 128
    emb_dim = 64
    dtype = torch.bfloat16
    cu_seqlens_list = [0, 27, 54, 99, 128]
    total_seqlen = cu_seqlens_list[-1]
    cos, sin, cu_seqlens = _thd_rope_tables(cu_seqlens_list, emb_dim, dtype)

    torch.manual_seed(seed)
    kv = torch.randn((total_seqlen, num_heads, k_dim + v_dim), dtype=dtype, device="cuda")
    emb = torch.randn((total_seqlen, 1, emb_dim), dtype=dtype, device="cuda")
    dk = torch.randn((total_seqlen, num_heads, k_dim + emb_dim), dtype=dtype, device="cuda")
    dv = torch.randn((total_seqlen, num_heads, v_dim), dtype=dtype, device="cuda")
    kv.requires_grad_(True)
    emb.requires_grad_(True)

    with _pinned_block_h(block_h):
        k_out, v_out = fused_mla_rope_kv_split(
            kv,
            emb,
            cos,
            sin,
            emb_dim,
            k_dim,
            v_dim,
            cu_seqlens_kv=cu_seqlens,
            remove_interleaving=remove_interleaving,
        )
        torch.autograd.backward((k_out, v_out), (dk, dv))

    return k_out.detach(), v_out.detach(), kv.grad.clone(), emb.grad.clone()


def _q_inplace_test_data(input_format, emb_dim):
    num_heads = 32
    q_dim = 128
    total_seqlen = 128
    generator = torch.Generator(device="cuda").manual_seed(617)
    source = torch.randn(
        (total_seqlen, num_heads, q_dim + emb_dim),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    if input_format == "sbhd":
        batch_size = 2
        seq_num = None
        cu_seqlens = None
        max_seqlen = total_seqlen // batch_size
        token_idx = torch.arange(max_seqlen, device="cuda").repeat_interleave(batch_size)
    else:
        batch_size = None
        cu_seqlens = torch.tensor([0, 27, 54, 99, 128], dtype=torch.int32, device="cuda")
        seq_num = len(cu_seqlens) - 1
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        max_seqlen = max(lengths)
        token_idx = torch.cat([torch.arange(length, device="cuda") for length in lengths])

    angles = torch.randn(
        (max_seqlen, 1, 1, emb_dim // 2), dtype=torch.float32, device="cuda", generator=generator
    )
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat((cos_half, cos_half), dim=-1).to(torch.bfloat16)
    sin = torch.cat((sin_half, sin_half), dim=-1).to(torch.bfloat16)
    return source, cos, sin, token_idx, q_dim, batch_size, seq_num, cu_seqlens


def _q_inplace_torch_reference(source, cos, sin, token_idx, q_dim, direction):
    expected = source.clone()
    rotary = source[..., q_dim:].float()
    emb_dim = rotary.shape[-1]
    half = emb_dim // 2
    token_cos = cos.view(-1, emb_dim)[token_idx].unsqueeze(1).float()
    token_sin = sin.view(-1, emb_dim)[token_idx].unsqueeze(1).float()

    if direction == "forward":
        x_1, x_2 = rotary[..., 0::2], rotary[..., 1::2]
        left = x_1 * token_cos[..., :half] - x_2 * token_sin[..., :half]
        right = x_2 * token_cos[..., half:] + x_1 * token_sin[..., half:]
        converted = torch.cat((left, right), dim=-1)
    else:
        left, right = rotary[..., :half], rotary[..., half:]
        x_1 = left * token_cos[..., :half] + right * token_sin[..., half:]
        x_2 = -left * token_sin[..., :half] + right * token_cos[..., half:]
        converted = torch.empty_like(rotary)
        converted[..., 0::2] = x_1
        converted[..., 1::2] = x_2

    expected[..., q_dim:] = converted.to(source.dtype)
    return expected


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("remove_interleaving", [False, True])
class TestFusedMLARopePartialHeadBlock:
    @pytest.mark.parametrize("input_format", ["sbhd", "thd"])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_inplace_matches_unfused_reference(self, input_format, inverse, remove_interleaving):
        num_heads = 12  # Deliberately not a power of two.
        block_heads_partial = 8  # 12 % 8 == 4: the second of two programs is partial.
        with _pinned_block_h(block_heads_partial):
            _test_fused_mla_rope_inplace(
                input_format,
                inverse=inverse,
                remove_interleaving=remove_interleaving,
                num_heads=num_heads,
            )

    @pytest.mark.parametrize("input_format", ["sbhd", "thd"])
    def test_kv_split_matches_unfused_reference(self, input_format, remove_interleaving):
        """Covers the forward outputs, the kv gradient and the k_pos_emb (``dEMB``) gradient."""
        num_heads = 12  # Deliberately not a power of two.
        block_heads_partial = 8  # 12 % 8 == 4: the second of two programs is partial.
        with _pinned_block_h(block_heads_partial):
            _test_fused_mla_rope_kv_split(
                input_format, remove_interleaving=remove_interleaving, num_heads=num_heads
            )

    def test_kv_split_result_does_not_depend_on_block_size(self, remove_interleaving):
        """BLOCK_H is a tiling choice, so it must not change one bit of the result.

        This needs no reference implementation, which is what makes it a sharper check than the
        tolerance-based comparisons above: a lane that a mask should have excluded shows up as a
        difference between two block sizes whatever value the hardware happened to give it.
        """
        num_heads = 12  # Deliberately not a power of two.
        block_heads_partial = 8  # 12 % 8 == 4: the second of two programs is partial.
        block_heads_exact = 4  # 12 % 4 == 0: every program is full. Same arithmetic, no partial.
        partial = _run_kv_split_once(block_heads_partial, remove_interleaving, num_heads)
        exact = _run_kv_split_once(block_heads_exact, remove_interleaving, num_heads)
        for name, from_partial, from_exact in zip(("k", "v", "d_kv", "d_emb"), partial, exact):
            torch.testing.assert_close(
                from_partial,
                from_exact,
                rtol=0,
                atol=0,
                msg=lambda msg, name=name: f"{name} depends on BLOCK_H: {msg}",
            )

    def test_inplace_does_not_write_past_the_tensor(self, remove_interleaving):
        """The rotated tensor is a view of the head of a larger buffer whose tail is poisoned.

        ``stride_x_seq == head_num * stride_x_nheads``, so the head rows the final program covers
        but does not own are the next token's leading heads -- and, for the last token, memory past
        the end of the tensor. The poisoned tail is where an unmasked store lands.
        """
        num_heads = 12  # Deliberately not a power of two.
        block_heads_partial = 8  # 12 % 8 == 4: the second of two programs is partial.
        nope_dim = 128
        emb_dim = 64
        dtype = torch.bfloat16
        cu_seqlens_list = [0, 27, 54, 99, 128]
        total_seqlen = cu_seqlens_list[-1]
        cos, sin, cu_seqlens = _thd_rope_tables(cu_seqlens_list, emb_dim, dtype)

        # One guard token row already exceeds the four head rows the partial block over-covers.
        guard_rows = 2
        buffer = torch.randn(
            (total_seqlen + guard_rows, num_heads, nope_dim + emb_dim), dtype=dtype, device="cuda"
        )
        guard_before = buffer[total_seqlen:].clone()
        rotated = buffer[:total_seqlen]
        assert rotated.stride() == buffer.stride()

        with _pinned_block_h(block_heads_partial), torch.no_grad():
            fused_mla_rope_inplace(
                rotated,
                cos,
                sin,
                nope_dim,
                emb_dim,
                cu_seqlens_q=cu_seqlens,
                remove_interleaving=remove_interleaving,
            )

        overflow = int((buffer[total_seqlen:] != guard_before).sum())
        assert overflow == 0, (
            f"{overflow} element(s) written past the end of the tensor: the partial final head "
            f"block (head_num={num_heads}, BLOCK_H={block_heads_partial}) stored to head rows it "
            f"does not own"
        )


@pytest.mark.skipif(not HAVE_TRITON, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_format", ["sbhd", "thd"])
@pytest.mark.parametrize(
    "direction,emb_dim,block_h,num_warps,repeats",
    [("forward", 128, 1, 8, 200), ("backward", 64, 2, 4, 100)],
    ids=["forward-multi-warp", "backward-multi-warp"],
)
def test_rotary_q_inplace_conversion_loads_before_stores(
    input_format, direction, emb_dim, block_h, num_warps, repeats
) -> None:
    """Validate a single-warp control, then require exact multi-warp reproducibility."""
    source, cos, sin, token_idx, q_dim, batch_size, seq_num, cu_seqlens = _q_inplace_test_data(
        input_format, emb_dim
    )
    expected = _q_inplace_torch_reference(source, cos, sin, token_idx, q_dim, direction)
    kernel = _mla_rope_fwd_inplace_kernel if direction == "forward" else _mla_rope_bwd_kernel

    def launch(launch_block_h, launch_num_warps):
        actual = source.clone()
        grid = (source.shape[0], triton.cdiv(source.shape[1], launch_block_h))
        tensor_args = (actual,) if direction == "forward" else (actual, actual)
        backward_kwargs = {} if direction == "forward" else {"COPY_NOPE": False, "BLOCK_NOPE": 1}
        kernel[grid](
            *tensor_args,
            cos,
            sin,
            q_dim,
            emb_dim,
            source.shape[1],
            batch_size,
            seq_num,
            cu_seqlens,
            None,
            0,
            actual.stride(0),
            actual.stride(1),
            cos.stride(0),
            sin.stride(0),
            0,
            1,
            INVERSE=False,
            REMOVE_INTERLEAVING=False,
            ROPE_FIRST=False,
            HAS_THD_GLOBAL_START=False,
            BLOCK_H=launch_block_h,
            num_warps=launch_num_warps,
            num_stages=3,
            **backward_kwargs,
        )
        return actual

    control = launch(1, 1)
    torch.testing.assert_close(control.float(), expected.float(), **dtype_tols(source.dtype))
    for _ in range(repeats):
        actual = launch(block_h, num_warps)
        torch.testing.assert_close(actual, control, rtol=0, atol=0)


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_format", ["sbhd", "thd"])
class TestFusedMLARope:
    @pytest.mark.flaky_in_dev
    @pytest.mark.parametrize("rope_first", [False, True], ids=["nope-first", "rope-first"])
    def test_forward_backward_for_q(self, input_format, rope_first):
        _test_fused_mla_rope_inplace(input_format, rope_first=rope_first)

    @pytest.mark.parametrize("inverse", [False, True])
    @pytest.mark.parametrize("remove_interleaving", [False, True])
    def test_inplace_forward_backward(self, input_format, inverse, remove_interleaving):
        _test_fused_mla_rope_inplace(
            input_format, inverse=inverse, remove_interleaving=remove_interleaving
        )

    @pytest.mark.parametrize("remove_interleaving", [False, True])
    def test_kv_split_forward_backward(self, input_format, remove_interleaving):
        _test_fused_mla_rope_kv_split(input_format, remove_interleaving=remove_interleaving)


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_format", ["sbhd", "thd"])
def test_out_of_place_inverse_rope_preserves_upstream_saved_output(input_format):
    """Post-attention inverse RoPE must not overwrite an output saved for backward."""
    assert fused_mla_rope_out_of_place is not None
    seqlen = 32
    batch_size = 1
    num_heads = 2
    nope_dim = 16
    emb_dim = 64
    dtype = torch.bfloat16

    yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=seqlen)
    freqs, mscale = yarn_rope(seqlen, 0)
    cos = (torch.cos(freqs) * mscale).to(dtype)
    sin = (torch.sin(freqs) * mscale).to(dtype)

    if input_format == "sbhd":
        shape = (seqlen, batch_size, num_heads, nope_dim + emb_dim)
        cu_seqlens = None
    else:
        shape = (2 * seqlen, num_heads, nope_dim + emb_dim)
        cu_seqlens = torch.tensor([0, seqlen, 2 * seqlen], dtype=torch.int32, device="cuda")

    unsafe_source = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    unsafe_attention_output = _SaveOutputForBackward.apply(unsafe_source)
    unsafe_reference = unsafe_attention_output.detach().clone()
    unsafe_inverse_output = fused_mla_rope_inplace(
        unsafe_attention_output,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q=cu_seqlens,
        inverse=True,
        remove_interleaving=True,
    )

    assert unsafe_inverse_output.data_ptr() == unsafe_attention_output.data_ptr()
    assert not torch.equal(unsafe_attention_output, unsafe_reference)

    source = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    attention_output = _SaveOutputForBackward.apply(source)
    saved_reference = attention_output.detach().clone()

    inverse_output = fused_mla_rope_out_of_place(
        attention_output,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q=cu_seqlens,
        inverse=True,
        remove_interleaving=True,
    )
    expected_inverse_output = fused_mla_rope_inplace(
        saved_reference.clone(),
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q=cu_seqlens,
        inverse=True,
        remove_interleaving=True,
    )

    assert inverse_output.data_ptr() != attention_output.data_ptr()
    torch.testing.assert_close(attention_output, saved_reference, rtol=0, atol=0)
    torch.testing.assert_close(inverse_output, expected_inverse_output, rtol=0, atol=0)

    inverse_output.backward(torch.randn_like(inverse_output).contiguous())
    torch.testing.assert_close(source.grad, saved_reference, rtol=0, atol=0)


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_format", ["sbhd", "thd"])
def test_legacy_query_api_remains_in_place(input_format):
    """The legacy API keeps its original mutation behavior and allocation profile."""
    assert fused_apply_mla_rope_for_q is not None
    seqlen = 32
    batch_size = 1
    num_heads = 2
    nope_dim = 16
    emb_dim = 64
    dtype = torch.bfloat16

    yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=seqlen)
    freqs, mscale = yarn_rope(seqlen, 0)
    cos = (torch.cos(freqs) * mscale).to(dtype)
    sin = (torch.sin(freqs) * mscale).to(dtype)

    if input_format == "sbhd":
        shape = (seqlen, batch_size, num_heads, nope_dim + emb_dim)
        cu_seqlens = None
    else:
        shape = (2 * seqlen, num_heads, nope_dim + emb_dim)
        cu_seqlens = torch.tensor([0, seqlen, 2 * seqlen], dtype=torch.int32, device="cuda")

    query = torch.randn(shape, dtype=dtype, device="cuda")
    reference = query.clone()
    expected = fused_mla_rope_inplace(
        reference.clone(), cos, sin, nope_dim, emb_dim, cu_seqlens_q=cu_seqlens
    )
    output = fused_apply_mla_rope_for_q(
        query, cos, sin, qk_head_dim=nope_dim, emb_dim=emb_dim, cu_seqlens_q=cu_seqlens
    )

    assert output.data_ptr() == query.data_ptr()
    assert not torch.equal(query, reference)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


class TestApplyRotaryPosEmbMlaFusionConflict:
    """Test apply_rotary_pos_emb: mla_rotary_interleaved vs apply_rope_fusion conflict."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.seq_len = 16
        self.num_heads = 2
        self.kv_channels = 32
        self.rot_dim = self.kv_channels

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mla_rotary_interleaved_with_apply_rope_fusion_emits_warning_and_uses_unfused(self):
        """When apply_rope_fusion=True and mla_rotary_interleaved=True, expect warning and unfused path."""
        config = TransformerConfig(
            num_attention_heads=self.num_heads,
            num_layers=1,
            apply_rope_fusion=True,
            rotary_interleaved=False,
        )
        t = torch.randn(
            self.seq_len, 1, self.num_heads, self.kv_channels, device="cuda", dtype=torch.float32
        )
        freqs = torch.randn(self.seq_len, 1, 1, self.rot_dim, device="cuda", dtype=torch.float32)

        fused_mock = MagicMock(return_value=t.clone())
        with (
            patch.object(rope_utils_module, "fused_apply_rotary_pos_emb", fused_mock),
            patch.object(
                rope_utils_module,
                "_apply_rotary_pos_emb_bshd",
                wraps=rope_utils_module._apply_rotary_pos_emb_bshd,
            ) as unfused_spy,
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = apply_rotary_pos_emb(t, freqs, config, mla_rotary_interleaved=True)
            # Should have warned about MLA + fusion conflict
            mla_fusion_warnings = [
                x for x in w if "apply_rope_fusion does not support MLA-style" in str(x.message)
            ]
            assert (
                len(mla_fusion_warnings) >= 1
            ), "Expected warning when mla_rotary_interleaved and apply_rope_fusion both enabled"
            # Fused kernel must not be used
            fused_mock.assert_not_called()
            # Unfused path must have been used
            unfused_spy.assert_called_once()
            call_kw = unfused_spy.call_args[1]
            assert call_kw["mla_rotary_interleaved"] is True
        assert out.shape == t.shape


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_format", ["sbhd", "thd"])
class TestFusedApplyMLARope:
    @pytest.mark.flaky_in_dev
    @pytest.mark.parametrize("rope_first", [False, True], ids=["nope-first", "rope-first"])
    def test_forward_backward_for_q(self, input_format, rope_first):
        _test_fused_mla_rope_inplace(input_format, rope_first=rope_first)

    @pytest.mark.parametrize("remove_interleaving", [False, True])
    def test_kv_split_forward_backward(self, input_format, remove_interleaving):
        _test_fused_mla_rope_kv_split(input_format, remove_interleaving=remove_interleaving)

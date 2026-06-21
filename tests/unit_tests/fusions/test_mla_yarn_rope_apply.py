# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings import rope_utils as rope_utils_module
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )
except:
    fused_apply_mla_rope_for_kv = None
    fused_apply_mla_rope_for_q = None


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
    def test_packed_freqs_returns_offset_mapped_output_for_context_parallel(self):
        cp_group = FakeCPGroup(size=2, rank=0)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        t = torch.randn(4, 2, 8)
        freqs = torch.randn(8, 1, 1, 8)

        out = rope_utils_module._apply_rotary_pos_emb_thd(t, cu_seqlens, freqs, cp_group=cp_group)

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

        out = rope_utils_module._apply_rotary_pos_emb_thd(t, cu_seqlens, freqs, cp_group=cp_group)

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


def _test_fused_apply_mla_rope_for_q(input_format):
    assert fused_apply_mla_rope_for_q is not None
    num_heads = 32
    q_dim = 128
    emb_dim = 64
    dtype = torch.bfloat16
    transformer_config = TransformerConfig(
        num_attention_heads=num_heads,
        num_layers=1,
        rotary_interleaved=False,
        multi_latent_attention=True,
    )

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
    fused_fwd_input = pytorch_fwd_input.detach()
    fused_fwd_input.requires_grad_(True)
    fused_bwd_input = pytorch_bwd_input.detach()

    no_pe, pe = torch.split(pytorch_fwd_input, [q_dim, emb_dim], dim=-1)
    pe_output = apply_rotary_pos_emb(
        pe,
        freqs,
        transformer_config,
        cu_seqlens=cu_seqlens,
        mscale=mscale,
        cp_group=FakeCPGroup(),
        mla_rotary_interleaved=True,
    )
    pytorch_output = torch.concat([no_pe, pe_output], dim=-1)
    pytorch_output.backward(pytorch_bwd_input, retain_graph=True)

    fused_output = fused_apply_mla_rope_for_q(
        fused_fwd_input, cos, sin, q_dim, emb_dim, cu_seqlens_q=cu_seqlens
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


def _test_fused_apply_mla_rope_for_kv(input_format):
    assert fused_apply_mla_rope_for_kv is not None
    num_heads = 32
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
        mscale=mscale,
        cp_group=FakeCPGroup(),
        mla_rotary_interleaved=True,
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

    fused_k_output, fused_v_output = fused_apply_mla_rope_for_kv(
        fused_fwd_kv_input,
        fused_fwd_emb_input,
        cos,
        sin,
        emb_dim,
        k_dim,
        v_dim,
        cu_seqlens_kv=cu_seqlens,
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


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_format", ["sbhd", "thd"])
class TestFusedApplyMLARope:
    @pytest.mark.flaky_in_dev
    def test_forward_backward_for_q(self, input_format):
        _test_fused_apply_mla_rope_for_q(input_format)

    def test_forward_backward_for_kv(self, input_format):
        _test_fused_apply_mla_rope_for_kv(input_format)


def _test_fused_apply_mla_rope_for_q_does_not_modify_input(input_format):
    """Regression test for in-place aliasing bug.

    The Triton kernel inside fused_apply_mla_rope_for_q writes to Q in-place.
    If the caller passes a tensor that shares storage with a saved activation (e.g.
    the output of a linear layer kept by TransformerEngine for weight-grad computation),
    that activation gets silently overwritten, producing wrong dW -> corrupted weights
    -> NaN from iteration 2 onward.

    After the fix (q = q.clone() inside ApplyMLARotaryEmbQ.forward), the input tensor
    must remain byte-for-byte identical after the call.
    """
    assert fused_apply_mla_rope_for_q is not None
    num_heads = 16
    q_dim = 64
    emb_dim = 32
    dtype = torch.bfloat16

    if input_format == "sbhd":
        cu_seqlens = None
        seqlen, batch_size = 128, 2
        yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=seqlen)
        freqs, mscale = yarn_rope(seqlen, 0)
        shape = (seqlen, batch_size, num_heads, q_dim + emb_dim)
    else:
        raw_cu = [0, 27, 54, 99, 128]
        max_seqlen = max(raw_cu[i + 1] - raw_cu[i] for i in range(len(raw_cu) - 1))
        cu_seqlens = torch.tensor(raw_cu, dtype=torch.int32, device='cuda')
        yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=max_seqlen)
        freqs, mscale = yarn_rope(max_seqlen, 0)
        shape = (raw_cu[-1], num_heads, q_dim + emb_dim)

    cos = (torch.cos(freqs) * mscale).to(dtype)
    sin = (torch.sin(freqs) * mscale).to(dtype)

    q = torch.randn(shape, dtype=dtype, device='cuda')
    q_original = q.clone()

    # Simulate the aliasing that occurs in practice: a view of q that shares storage,
    # just as linear_q_up_proj's output does after .view() in MLASelfAttention.
    q_view = q.view(q.shape)

    _ = fused_apply_mla_rope_for_q(q_view, cos, sin, q_dim, emb_dim, cu_seqlens_q=cu_seqlens)

    # The original backing tensor must be untouched.
    assert torch.equal(q, q_original), (
        "fused_apply_mla_rope_for_q modified the input tensor in-place. "
        "This corrupts saved activations in the upstream linear layer and causes "
        "NaN from the second training iteration onward."
    )


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_format", ["sbhd", "thd"])
class TestFusedApplyMLARopeNoInputMutation:
    """Regression tests: fused_apply_mla_rope_for_q must not mutate its input."""

    def test_input_unchanged_after_rope_q(self, input_format):
        _test_fused_apply_mla_rope_for_q_does_not_modify_input(input_format)


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

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
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
        fused_apply_mla_rope_for_q,
        fused_mla_rope_inplace,
        fused_mla_rope_kv_split,
        fused_mla_rope_out_of_place,
    )
except Exception:
    fused_apply_mla_rope_for_q = None
    fused_mla_rope_inplace = None
    fused_mla_rope_kv_split = None
    fused_mla_rope_out_of_place = None


def dtype_tols(dtype):
    if dtype == torch.float32:
        return dict(rtol=1.0e-6, atol=1.0e-6)
    elif dtype == torch.float16:
        return dict(rtol=3.0e-3, atol=1.0e-5)
    elif dtype == torch.bfloat16:
        return dict(rtol=2.0e-2, atol=5.0e-2)
    else:
        raise ValueError(f"Unsuppored dtype ({dtype})")


def _benchmark_ms(function, warmup=20, iterations=100):
    """Measure average GPU time, including work joined from side streams."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _capture_cuda_graph(function):
    """Capture a callable whose side-stream work joins the capture stream."""
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        function()
    torch.cuda.synchronize()
    return graph


def _localization_available():
    try:
        from torch.cuda.green_contexts import is_localization_supported
        from torch.cuda.memory import get_num_locality_domains
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    try:
        supported = is_localization_supported(device)
    except TypeError:
        supported = is_localization_supported()
    return supported and get_num_locality_domains(device) == 2


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


def _test_fused_mla_rope_inplace(input_format, inverse=False, remove_interleaving=False):
    assert fused_mla_rope_inplace is not None
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
    fused_fwd_input = pytorch_fwd_input.detach()
    fused_fwd_input.requires_grad_(True)
    fused_bwd_input = pytorch_bwd_input.detach()

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
    pytorch_output = torch.concat([no_pe, pe_output], dim=-1)
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


def _test_fused_mla_rope_kv_split(
    input_format, remove_interleaving=False, use_vmm_outputs=False
):
    assert fused_mla_rope_kv_split is not None
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

    output_kwargs = {}
    vmm_allocators = []
    if use_vmm_outputs:
        try:
            from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator

            key_allocator = VMMRowSplitAllocator(fused_fwd_kv_input.device)
            value_allocator = VMMRowSplitAllocator(fused_fwd_kv_input.device)
            key_shape = (*fused_fwd_kv_input.shape[:-1], k_dim + emb_dim)
            value_shape = (*fused_fwd_kv_input.shape[:-1], v_dim)
            output_kwargs = {
                "out_key": key_allocator.allocate(key_shape, dtype),
                "out_value": value_allocator.allocate(value_shape, dtype),
            }
            vmm_allocators = [key_allocator, value_allocator]
        except (ImportError, RuntimeError, ValueError) as exc:
            pytest.skip(f"VMM-localized rotary outputs are unavailable: {exc}")

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
        **output_kwargs,
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
    for allocator in vmm_allocators:
        allocator.close()


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_format", ["sbhd", "thd"])
class TestFusedMLARope:
    @pytest.mark.flaky_in_dev
    @pytest.mark.parametrize("inverse", [False, True])
    @pytest.mark.parametrize("remove_interleaving", [False, True])
    def test_inplace_forward_backward(self, input_format, inverse, remove_interleaving):
        _test_fused_mla_rope_inplace(
            input_format, inverse=inverse, remove_interleaving=remove_interleaving
        )

    @pytest.mark.parametrize("remove_interleaving", [False, True])
    def test_kv_split_forward_backward(self, input_format, remove_interleaving):
        _test_fused_mla_rope_kv_split(input_format, remove_interleaving=remove_interleaving)

    def test_kv_split_forward_backward_with_vmm_outputs(self, input_format):
        if input_format != "sbhd":
            pytest.skip("The focused VMM prototype covers the profiled SBHD path")
        _test_fused_mla_rope_kv_split(input_format, use_vmm_outputs=True)


@pytest.mark.experimental
@pytest.mark.internal
@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark test - run with RUN_BENCHMARK_TESTS=1",
)
def test_mla_rope_qkv_three_mxfp8_quant_localization_performance():
    """Compare the profiled rotary Q/K/V plus three MXFP8 quant pattern."""
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator

    seqlen = 4096
    batch_size = 1
    num_heads = 128
    nope_dim = 128
    emb_dim = 64
    k_dim = 128
    v_dim = 128
    dtype = torch.bfloat16
    device = torch.device("cuda")

    yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=seqlen)
    freqs, mscale = yarn_rope(seqlen, 0)
    cos = (torch.cos(freqs) * mscale).to(dtype)
    sin = (torch.sin(freqs) * mscale).to(dtype)

    q_shape = (seqlen, batch_size, num_heads, nope_dim + emb_dim)
    kv_shape = (seqlen, batch_size, num_heads, k_dim + v_dim)
    key_shape = (seqlen, batch_size, num_heads, k_dim + emb_dim)
    value_shape = (seqlen, batch_size, num_heads, v_dim)
    q_seed = torch.randn(q_shape, dtype=dtype, device=device)
    q_ordinary = q_seed.clone()
    kv = torch.randn(kv_shape, dtype=dtype, device=device)
    k_pos_emb = torch.randn(
        (seqlen, batch_size, 1, emb_dim), dtype=dtype, device=device
    )
    key_ordinary = torch.empty(key_shape, dtype=dtype, device=device)
    value_ordinary = torch.empty(value_shape, dtype=dtype, device=device)

    allocators = [
        VMMRowSplitAllocator(device),
        VMMRowSplitAllocator(device),
        VMMRowSplitAllocator(device),
    ]
    try:
        q_localized = allocators[0].allocate(q_shape, dtype)
        key_localized = allocators[1].allocate(key_shape, dtype)
        value_localized = allocators[2].allocate(value_shape, dtype)
    except (RuntimeError, ValueError) as exc:
        for allocator in allocators:
            allocator.close()
        pytest.skip(f"VMM-localized rotary outputs are unavailable: {exc}")
    q_localized.copy_(q_seed)

    ordinary_inputs = [
        q_ordinary.view(seqlen, -1),
        key_ordinary.view(seqlen, -1),
        value_ordinary.view(seqlen, -1),
    ]
    localized_inputs = [
        q_localized.view(seqlen, -1),
        key_localized.view(seqlen, -1),
        value_localized.view(seqlen, -1),
    ]
    quantizers = []
    ordinary_quantized = []
    localized_workspaces = []
    try:
        for ordinary_input, localized_input in zip(ordinary_inputs, localized_inputs):
            quantizer = te.MXFP8Quantizer(
                fp8_dtype=te.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True,
            )
            quantizer.optimize_for_gemm = True
            quantizers.append(quantizer)
            ordinary_quantized.append(
                quantizer.make_empty(
                    tuple(ordinary_input.shape),
                    dtype=dtype,
                    device=device,
                )
            )
            localized_workspaces.append(
                te.MXFP8VMMWorkspace.from_vmm_input(localized_input, quantizer)
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        for workspace in localized_workspaces:
            workspace.close()
        for allocator in allocators:
            allocator.close()
        pytest.skip(f"VMM MXFP8 localization is unavailable: {exc}")

    @torch.no_grad()
    def ordinary_rotary():
        fused_mla_rope_inplace(q_ordinary, cos, sin, nope_dim, emb_dim)
        fused_mla_rope_kv_split(
            kv,
            k_pos_emb,
            cos,
            sin,
            emb_dim,
            k_dim,
            v_dim,
            out_key=key_ordinary,
            out_value=value_ordinary,
        )

    @torch.no_grad()
    def localized_rotary():
        # Q RoPE is in-place, so its upstream buffer must already be VMM-backed.
        fused_mla_rope_inplace(q_localized, cos, sin, nope_dim, emb_dim)
        fused_mla_rope_kv_split(
            kv,
            k_pos_emb,
            cos,
            sin,
            emb_dim,
            k_dim,
            v_dim,
            out_key=key_localized,
            out_value=value_localized,
        )

    @torch.no_grad()
    def ordinary_quant():
        for quantizer, input_tensor, output in zip(
            quantizers, ordinary_inputs, ordinary_quantized
        ):
            quantizer.update_quantized(input_tensor, output)

    @torch.no_grad()
    def localized_quant():
        for workspace in localized_workspaces:
            workspace.quantize()

    def ordinary_pipeline():
        ordinary_rotary()
        ordinary_quant()

    def localized_pipeline():
        localized_rotary()
        localized_quant()

    use_cuda_graph = os.getenv("MXFP8_LOCALIZATION_USE_CUDA_GRAPH") == "1"

    def maybe_capture(function):
        return _capture_cuda_graph(function).replay if use_cuda_graph else function

    ordinary_rotary_fn = maybe_capture(ordinary_rotary)
    localized_rotary_fn = maybe_capture(localized_rotary)
    ordinary_quant_fn = maybe_capture(ordinary_quant)
    localized_quant_fn = maybe_capture(localized_quant)
    ordinary_pipeline_fn = maybe_capture(ordinary_pipeline)
    localized_pipeline_fn = maybe_capture(localized_pipeline)

    ordinary_rotary_ms = _benchmark_ms(ordinary_rotary_fn)
    localized_rotary_ms = _benchmark_ms(localized_rotary_fn)
    ordinary_quant_ms = _benchmark_ms(ordinary_quant_fn)
    localized_quant_ms = _benchmark_ms(localized_quant_fn)
    ordinary_pipeline_ms = _benchmark_ms(ordinary_pipeline_fn)
    localized_pipeline_ms = _benchmark_ms(localized_pipeline_fn)

    q_ordinary.copy_(q_seed)
    q_localized.copy_(q_seed)
    ordinary_pipeline_fn()
    localized_pipeline_fn()
    torch.cuda.synchronize()
    torch.testing.assert_close(q_localized, q_ordinary, atol=0.0, rtol=0.0)
    torch.testing.assert_close(key_localized, key_ordinary, atol=0.0, rtol=0.0)
    torch.testing.assert_close(value_localized, value_ordinary, atol=0.0, rtol=0.0)
    for ordinary_output, workspace in zip(ordinary_quantized, localized_workspaces):
        for name in (
            "_rowwise_data",
            "_rowwise_scale_inv",
            "_columnwise_data",
            "_columnwise_scale_inv",
        ):
            torch.testing.assert_close(
                getattr(workspace.output, name),
                getattr(ordinary_output, name),
                atol=0.0,
                rtol=0.0,
            )

    execution = "CUDA Graph" if use_cuda_graph else "eager"
    print(
        f"\nMLA rotary Q/K/V + three MXFP8 quant ({execution}):"
        f"\n  ordinary-memory rotary:       {ordinary_rotary_ms:.3f} ms"
        f"\n  VMM-output rotary:            {localized_rotary_ms:.3f} ms"
        f"\n  full-chip three quant:        {ordinary_quant_ms:.3f} ms"
        f"\n  localized three quant:        {localized_quant_ms:.3f} ms"
        f"\n  ordinary full pipeline:       {ordinary_pipeline_ms:.3f} ms"
        f"\n  localized full pipeline:      {localized_pipeline_ms:.3f} ms"
        f"\n  end-to-end speedup:           "
        f"{ordinary_pipeline_ms / localized_pipeline_ms:.3f}x"
    )

    for workspace in localized_workspaces:
        workspace.close()
    for allocator in allocators:
        allocator.close()


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

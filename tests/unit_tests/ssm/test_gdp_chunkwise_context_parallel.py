# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.metadata
import itertools
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.context_parallel import gdp_common
from megatron.core.ssm.context_parallel.chunkwise import (
    CPBackwardPackedSummary,
    CPForwardPackedSummary,
    CPForwardResult,
    CPSavedContext,
    build_packed_sequence_cp_metadata,
)
from megatron.core.ssm.context_parallel.gdp_common import GDPInputs
from megatron.core.ssm.gated_delta_product import (
    HAVE_CUTEDSL_GDP_CP,
    HAVE_FLA_GDP_CP,
    GatedDeltaProductMixer,
)

pytestmark = pytest.mark.launch_on_gb200

if HAVE_FLA_GDP_CP:
    import megatron.core.ssm.context_parallel.gdp as gdp_cp_module
else:
    gdp_cp_module = None

if HAVE_CUTEDSL_GDP_CP:
    import cutlass.cute as cute

    import megatron.core.ssm.context_parallel.gdp_cutedsl as gdp_cutedsl_cp_module
else:
    cute = None
    gdp_cutedsl_cp_module = None


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def test_shared_autograd_adapter(monkeypatch):
    """Verify the common adapter's backend-independent autograd contract."""
    calls = {}
    group = _FakeGroup(rank=1, size=3)
    backend = SimpleNamespace(autograd_function=gdp_common.GDPChunkwiseContextParallel)

    def fake_forward(*, backend, inputs, cp_group, preceding_slice):
        calls["forward"] = (backend, inputs, cp_group, preceding_slice)
        return CPForwardResult(
            output=inputs.q + inputs.k + inputs.v + inputs.g + inputs.beta,
            saved_context=CPSavedContext(tensors=(inputs.q,), metadata="saved"),
        )

    def fake_backward(*, backend, output_grad, saved_context, cp_group, following_slice):
        calls["backward"] = (backend, output_grad, saved_context, cp_group, following_slice)
        return tuple(output_grad * multiplier for multiplier in range(1, 6))

    monkeypatch.setattr(gdp_common, "chunkwise_cp_forward", fake_forward)
    monkeypatch.setattr(gdp_common, "chunkwise_cp_backward", fake_backward)

    inputs = [torch.randn(2, 3, requires_grad=True) for _ in range(5)]
    output = gdp_common.gdp_chunkwise_context_parallel(
        *inputs,
        cu_seqlens=None,
        num_householder=1,
        scale=0.125,
        cp_group=group,
        backend=backend,
        preceding_rank_start=0,
        following_rank_stop=3,
    )
    output.sum().backward()

    assert calls["forward"][0] is backend
    assert calls["forward"][2] is group
    assert calls["forward"][3] == slice(0, 1)
    assert calls["backward"][0] is backend
    assert calls["backward"][2].metadata == "saved"
    assert calls["backward"][3] is group
    assert calls["backward"][4] == slice(2, 3)
    for multiplier, tensor in enumerate(inputs, start=1):
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, multiplier))


@pytest.mark.skipif(not HAVE_FLA_GDP_CP, reason="FLA GDP CP kernels are not installed")
def test_interleave_last_update():
    """Verify backward values align with each token's final Householder update."""
    assert gdp_cp_module is not None
    tensor = torch.arange(12).reshape(1, 3, 2, 2)

    interleaved = gdp_cp_module._interleave_last_update(tensor, num_householder=3)

    assert interleaved.shape == (1, 9, 2, 2)
    torch.testing.assert_close(interleaved[:, 2::3], tensor)
    torch.testing.assert_close(interleaved[:, 0::3], torch.zeros_like(tensor))
    torch.testing.assert_close(interleaved[:, 1::3], torch.zeros_like(tensor))


def test_reuse_rank_local_metadata():
    """Verify GDP reuses cached local sequence boundaries and CP summary bounds."""
    global_seq_idx = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.int32)
    metadata = build_packed_sequence_cp_metadata(global_seq_idx, cp_rank=1, cp_size=2)
    mixer = GatedDeltaProductMixer.__new__(GatedDeltaProductMixer)
    packed_seq_params = PackedSeqParams(qkv_format="thd", seq_idx=global_seq_idx)

    returned_metadata = mixer._chunkwise_packed_metadata(
        packed_seq_params, local_sequence_length=4, metadata=metadata
    )

    assert returned_metadata is metadata
    torch.testing.assert_close(
        returned_metadata.local_cu_seqlens, torch.tensor([0, 1, 4], dtype=torch.int32)
    )
    assert returned_metadata.preceding_rank_start == 0
    assert returned_metadata.following_rank_stop == 2


@pytest.mark.skipif(not HAVE_CUTEDSL_GDP_CP, reason="CuTeDSL GDP CP backend is unavailable")
def test_cutedsl_backend_adapts_four_function_protocol(monkeypatch):
    """Verify summaries and the recompute policy reach the CuTeDSL API."""
    assert gdp_cutedsl_cp_module is not None
    calls = {}
    tensor = torch.randn(2, 3)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    forward_summary = torch.randn(2, 4, 7)
    backward_summary = torch.randn(2, 4, 7)
    backend_local_context = object()
    backend_backward_context = object()

    def fake_forward_prepare(*args, **kwargs):
        calls["forward_prepare"] = (args, kwargs)
        return forward_summary, backend_local_context

    def fake_forward_apply(local_context, preceding_summaries, *, output_final_state):
        calls["forward_apply"] = (local_context, preceding_summaries, output_final_state)
        return tensor, SimpleNamespace(
            q=tensor,
            k=tensor,
            v=tensor,
            g=tensor,
            beta=tensor,
            cu_seqlens=cu_seqlens,
            state=tensor,
            initial_states=tensor,
            scale=0.125,
            num_householder=3,
            use_qk_l2norm_in_kernel=True,
            recompute_chunk_num=2,
        )

    def fake_backward_prepare(output_grad, saved_context, *, boundary_len=None):
        calls["backward_prepare"] = (output_grad, saved_context, boundary_len)
        return backward_summary, backend_backward_context

    gradients = tuple(torch.randn_like(tensor) for _ in range(5))

    def fake_backward_apply(backward_context, following_summaries, *, dht):
        calls["backward_apply"] = (backward_context, following_summaries, dht)
        return gradients

    monkeypatch.setattr(gdp_cutedsl_cp_module, "cutedsl_cp_forward_prepare", fake_forward_prepare)
    monkeypatch.setattr(gdp_cutedsl_cp_module, "cutedsl_cp_forward_apply", fake_forward_apply)
    monkeypatch.setattr(gdp_cutedsl_cp_module, "cutedsl_cp_backward_prepare", fake_backward_prepare)
    monkeypatch.setattr(gdp_cutedsl_cp_module, "cutedsl_cp_backward_apply", fake_backward_apply)

    backend = gdp_cutedsl_cp_module.CuTeDSLGatedDeltaProductCPBackend(recompute_chunk_num=2)
    inputs = GDPInputs(
        q=tensor,
        k=tensor,
        v=tensor,
        g=tensor,
        beta=tensor,
        cu_seqlens=cu_seqlens,
        num_householder=3,
        scale=0.125,
    )

    local_summary, local_context = backend.cp_forward_prepare(inputs)
    assert isinstance(local_summary, CPForwardPackedSummary)
    assert local_summary.packed is forward_summary
    assert calls["forward_prepare"][1]["recompute_chunk_num"] == 2

    prefix = CPForwardPackedSummary(packed=forward_summary.unsqueeze(0))
    output, saved_context = backend.cp_forward_apply(local_context, prefix)
    assert output is tensor
    assert calls["forward_apply"][0] is backend_local_context
    assert calls["forward_apply"][1] is prefix.packed
    assert calls["forward_apply"][2] is False

    local_backward_summary, backward_context = backend.cp_backward_prepare(tensor, saved_context)
    assert isinstance(local_backward_summary, CPBackwardPackedSummary)
    assert local_backward_summary.packed is backward_summary
    assert calls["backward_prepare"][1].initial_states is tensor
    assert calls["backward_prepare"][2] is None

    suffix = CPBackwardPackedSummary(packed=backward_summary.unsqueeze(0))
    assert backend.cp_backward_apply(backward_context, suffix) is gradients
    assert calls["backward_apply"][0] is backend_backward_context
    assert calls["backward_apply"][1] is suffix.packed
    assert calls["backward_apply"][2] is None


@pytest.mark.skipif(
    not HAVE_CUTEDSL_GDP_CP or not torch.cuda.is_available(),
    reason="CuTeDSL GDP CP backend or CUDA is unavailable",
)
def test_cutedsl_backend_compiles_boundary_variants_at_first_forward(monkeypatch):
    """Verify no CP boundary case compiles a kernel after the first forward."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("CuTeDSL GDP requires an SM100 GPU")
    if Version(importlib.metadata.version("gdp-attn-fused")) < Version("0.5.2"):
        pytest.skip("gdp-attn-fused < 0.5.2 keys the backward summary on token and sequence count")
    assert gdp_cutedsl_cp_module is not None
    num_tokens, num_heads, num_householder, key_dim, value_dim = 512, 4, 3, 128, 64
    widths = [
        num_householder * num_heads * value_dim,
        num_householder * num_heads * key_dim,
        num_heads * key_dim,
    ]
    torch.manual_seed(0)
    # Strided q/k views and a contiguous v, as GatedDeltaProductMixer passes them under CP.
    value, key, query = torch.randn(
        num_tokens, sum(widths), device="cuda", dtype=torch.bfloat16
    ).split(widths, dim=-1)
    inputs = GDPInputs(
        q=query,
        k=key,
        v=value.contiguous(),
        g=-0.1 * torch.rand(num_tokens, num_heads, device="cuda"),
        beta=torch.rand(
            num_tokens, num_householder * num_heads, device="cuda", dtype=torch.bfloat16
        ),
        cu_seqlens=torch.tensor([0, 100, 320, num_tokens], dtype=torch.int32, device="cuda"),
        num_householder=num_householder,
        scale=key_dim**-0.5,
    )
    backend = gdp_cutedsl_cp_module.CuTeDSLGatedDeltaProductCPBackend(recompute_chunk_num=1)
    compile_calls = []
    compile_fn = cute.compile

    def counting_compile(*args, **kwargs):
        compile_calls.append(args[0])
        return compile_fn(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", counting_compile)

    backend.cp_forward_prepare(inputs)
    num_compiles = len(compile_calls)
    if num_compiles == 0:
        pytest.skip("gdp_attn kernels were already compiled in this process")
    for num_preceding, num_following in itertools.product((0, 1), repeat=2):
        forward_summary, local_context = backend.cp_forward_prepare(inputs)
        preceding = forward_summary.packed.unsqueeze(0)[:num_preceding]
        output, saved_context = backend.cp_forward_apply(
            local_context, CPForwardPackedSummary(packed=preceding)
        )
        backward_summary, backward_context = backend.cp_backward_prepare(
            torch.ones_like(output), saved_context
        )
        following = backward_summary.packed.unsqueeze(0)[:num_following]
        backend.cp_backward_apply(backward_context, CPBackwardPackedSummary(packed=following))
    assert len(compile_calls) == num_compiles

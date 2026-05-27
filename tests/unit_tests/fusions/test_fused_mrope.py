# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings
from types import SimpleNamespace

import pytest
import torch

import megatron.core.models.common.embeddings.rope_utils as rope_utils
from megatron.core import parallel_state
from megatron.core.fusions import fused_mrope
from megatron.core.fusions.fused_mrope import (
    fused_apply_mrope,
    fused_apply_mrope_thd,
    get_fused_mrope_thd_unavailable_reason,
    get_fused_mrope_unavailable_reason,
    is_fused_mrope_available,
    mrope_freqs_to_rotary_emb,
)
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rope_utils import (
    _ROPE_FUSION_FALLBACK_WARNINGS,
    _apply_rotary_pos_emb_bshd,
    _apply_rotary_pos_emb_thd,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class FakeCPGroup:
    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class FakeDynamicInferenceContext:
    def is_dynamic_batching(self):
        return True

    def is_static_batching(self):
        return False


@pytest.fixture(autouse=True)
def clear_rope_fusion_fallback_warnings():
    _ROPE_FUSION_FALLBACK_WARNINGS.clear()
    yield
    _ROPE_FUSION_FALLBACK_WARNINGS.clear()


def _dtype_tols(dtype):
    if dtype == torch.bfloat16:
        return dict(rtol=2.0e-2, atol=5.0e-2)
    if dtype == torch.float16:
        return dict(rtol=3.0e-3, atol=1.0e-2)
    return dict(rtol=1.0e-6, atol=1.0e-6)


def _make_inputs(
    dtype=torch.bfloat16,
    requires_grad=False,
    head_dim=20,
    rotary_dim=16,
    mrope_section=None,
    interleaved_mrope=False,
    batch=2,
):
    seq = 32
    heads = 3
    if mrope_section is None:
        mrope_section = [3, 3, 2] if interleaved_mrope else [2, 3, 3]

    generator = torch.Generator(device="cuda").manual_seed(1234)
    t = torch.randn(
        seq,
        batch,
        heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
        requires_grad=requires_grad,
    )
    freqs = torch.randn(
        3, batch, seq, rotary_dim // 2, dtype=torch.float32, device="cuda", generator=generator
    )
    return t, freqs, mrope_section


def _make_position_ids(seq, batch):
    base = torch.arange(seq, device="cuda", dtype=torch.long)
    batch_offsets = torch.arange(batch, device="cuda", dtype=torch.long)
    return (
        torch.stack((base, base * 2 + 3, base * 3 + 5), dim=0)[:, None, :]
        + batch_offsets[None, :, None]
    ).contiguous()


def _make_thd_inputs(
    dtype=torch.bfloat16,
    requires_grad=False,
    interleaved_mrope=False,
    cp_size=1,
    padded_seq_lens=(12, 16),
):
    total_seq = sum(padded_seq_lens)
    local_seq = total_seq // cp_size
    heads = 3
    head_dim = 20
    rotary_dim = 16
    mrope_section = [3, 3, 2] if interleaved_mrope else [2, 3, 3]

    generator = torch.Generator(device="cuda").manual_seed(5678)
    t = torch.randn(
        local_seq,
        heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
        requires_grad=requires_grad,
    )
    freqs = torch.randn(
        3, 1, total_seq, rotary_dim // 2, dtype=torch.float32, device="cuda", generator=generator
    )
    cu_seqlens = torch.tensor(
        [0, padded_seq_lens[0], total_seq], dtype=torch.int32, device="cuda"
    )
    return t, freqs, cu_seqlens, mrope_section


def _make_mrope_config(
    num_attention_heads, mrope_section, interleaved_mrope=False, rotary_interleaved=False
):
    return TransformerConfig(
        num_attention_heads=num_attention_heads,
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
        mrope_interleaved=interleaved_mrope,
        rotary_interleaved=rotary_interleaved,
    )


def _fallback_warnings(recorded_warnings):
    return [
        warning
        for warning in recorded_warnings
        if issubclass(warning.category, UserWarning)
        and "Using unfused implementation" in str(warning.message)
    ]


@pytest.mark.parametrize("use_packed_seq", [False, True])
def test_gpt_mrope_eval_requests_raw_freqs_when_fusion_available(monkeypatch, use_packed_seq):
    monkeypatch.setattr(fused_mrope, "is_fused_mrope_available", lambda: True)

    captured_kwargs = {}

    def fake_rotary_pos_emb(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "raw-mrope-freqs"

    model = SimpleNamespace(
        training=False,
        pre_process=False,
        mtp_process=False,
        position_embedding_type="mrope",
        config=SimpleNamespace(
            multi_latent_attention=False,
            flash_decode=False,
            apply_rope_fusion=True,
            rotary_interleaved=False,
            cuda_graph_impl=None,
        ),
        rotary_pos_emb=fake_rotary_pos_emb,
        mrope_section=[2, 3, 3],
    )
    packed_seq_params = (
        SimpleNamespace(qkv_format="thd", cp_group=FakeCPGroup()) if use_packed_seq else None
    )

    output = GPTModel._preprocess(
        model,
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        position_ids=torch.zeros(3, 1, 4, dtype=torch.long),
        decoder_input=torch.zeros(4, 1, 12),
        packed_seq_params=packed_seq_params,
    )

    assert output[1] == "raw-mrope-freqs"
    assert captured_kwargs["return_raw_freqs"] is True
    assert captured_kwargs["packed_seq"] is use_packed_seq


def test_gpt_mrope_dynamic_inference_keeps_materialized_freqs(monkeypatch):
    monkeypatch.setattr(fused_mrope, "is_fused_mrope_available", lambda: True)

    captured_kwargs = {}

    def fake_rotary_pos_emb(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "materialized-mrope-freqs"

    model = SimpleNamespace(
        training=False,
        pre_process=False,
        mtp_process=False,
        position_embedding_type="mrope",
        config=SimpleNamespace(
            multi_latent_attention=False,
            flash_decode=False,
            apply_rope_fusion=True,
            rotary_interleaved=False,
            cuda_graph_impl=None,
        ),
        rotary_pos_emb=fake_rotary_pos_emb,
        mrope_section=[2, 3, 3],
    )

    output = GPTModel._preprocess(
        model,
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        position_ids=torch.zeros(3, 1, 4, dtype=torch.long),
        decoder_input=torch.zeros(4, 1, 12),
        inference_context=FakeDynamicInferenceContext(),
    )

    assert output[1] == "materialized-mrope-freqs"
    assert captured_kwargs["return_raw_freqs"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
@pytest.mark.parametrize("head_dim", [16, 20])
def test_fused_mrope_matches_unfused_forward_backward(interleaved_mrope, head_dim):
    t_ref, freqs, mrope_section = _make_inputs(
        requires_grad=True, head_dim=head_dim, interleaved_mrope=interleaved_mrope
    )
    t_fused = t_ref.detach().clone().requires_grad_(True)

    emb = mrope_freqs_to_rotary_emb(
        freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
    )
    ref = _apply_rotary_pos_emb_bshd(t_ref, emb, rotary_interleaved=False)
    out = fused_apply_mrope(
        t_fused, freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
    )

    tols = _dtype_tols(t_ref.dtype)
    torch.testing.assert_close(ref.float(), out.float(), **tols)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    out.backward(grad)
    torch.testing.assert_close(t_ref.grad.float(), t_fused.grad.float(), **tols)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_apply_rotary_pos_emb_bshd_eval_uses_triton_without_te(interleaved_mrope, monkeypatch):
    t, freqs, mrope_section = _make_inputs(interleaved_mrope=interleaved_mrope, batch=1)
    config = _make_mrope_config(t.shape[2], mrope_section, interleaved_mrope)

    emb = mrope_freqs_to_rotary_emb(
        freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
    )
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False)

    fused_calls = 0
    orig_fused_apply_mrope = rope_utils.fused_apply_mrope

    def wrapped_fused_apply_mrope(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return orig_fused_apply_mrope(*args, **kwargs)

    monkeypatch.setattr(rope_utils, "fused_apply_rotary_pos_emb", None)
    monkeypatch.setattr(rope_utils, "fused_apply_mrope", wrapped_fused_apply_mrope)
    with torch.no_grad(), warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup())

    assert fused_calls == 1
    assert not _fallback_warnings(recorded_warnings)
    assert not out.requires_grad
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize(
    "fallback_kwargs, warning_match",
    [
        ({"mscale": 1.25}, "mscale=1.25 is not supported by Triton fused mRoPE"),
        ({"inverse": True}, "inverse RoPE is not supported by Triton fused mRoPE"),
    ],
)
def test_apply_rotary_pos_emb_raw_mrope_fallbacks_match_unfused(
    fallback_kwargs, warning_match
):
    t, freqs, mrope_section = _make_inputs()
    config = TransformerConfig(
        num_attention_heads=t.shape[2],
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
    )

    with pytest.warns(UserWarning, match=warning_match):
        out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup(), **fallback_kwargs)

    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False, **fallback_kwargs)
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))

    with warnings.catch_warnings(record=True) as repeated_warnings:
        warnings.simplefilter("always")
        out_again = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup(), **fallback_kwargs)
    assert not repeated_warnings
    torch.testing.assert_close(ref.float(), out_again.float(), **_dtype_tols(t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize(
    "fallback_kwargs, config_kwargs, expected_warning_key, warning_text",
    [
        (
            {"mscale": 1.25},
            {},
            "triton-mrope-mscale",
            "mscale=1.25 is not supported by Triton fused mRoPE",
        ),
        (
            {"inverse": True},
            {},
            "triton-mrope-inverse",
            "inverse RoPE is not supported by Triton fused mRoPE",
        ),
        (
            {"mla_rotary_interleaved": True},
            {},
            "triton-mrope-mla-rotary-interleaved",
            "does not support MLA-style interleaving",
        ),
        (
            {},
            {"rotary_interleaved": True},
            "triton-mrope-unavailable-rotary-interleaved",
            "rotary_interleaved=True is not supported",
        ),
    ],
)
def test_apply_rotary_pos_emb_raw_mrope_fallback_emits_single_warning(
    fallback_kwargs, config_kwargs, expected_warning_key, warning_text
):
    t, freqs, mrope_section = _make_inputs()
    config = _make_mrope_config(t.shape[2], mrope_section, **config_kwargs)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup(), **fallback_kwargs)

    fallback_warnings = _fallback_warnings(recorded_warnings)
    assert len(fallback_warnings) == 1
    assert warning_text in str(fallback_warnings[0].message)
    assert _ROPE_FUSION_FALLBACK_WARNINGS == {expected_warning_key}

    emb = mrope_freqs_to_rotary_emb(
        freqs, mrope_section, rotary_interleaved=config.rotary_interleaved
    )
    ref = _apply_rotary_pos_emb_bshd(
        t, emb, rotary_interleaved=config.rotary_interleaved, **fallback_kwargs
    )
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))


def test_interleaved_mrope_rejects_inconsistent_sections():
    freqs = torch.randn(3, 2, 8, 8, dtype=torch.float32)

    with pytest.raises(AssertionError, match="interleaved mRoPE"):
        mrope_freqs_to_rotary_emb(freqs, [2, 3, 3], interleaved_mrope=True)


def test_raw_mrope_cpu_falls_back_to_unfused():
    t = torch.randn(8, 1, 3, 20, dtype=torch.float32)
    freqs = torch.randn(3, 1, 8, 8, dtype=torch.float32)
    mrope_section = [2, 3, 3]
    config = TransformerConfig(
        num_attention_heads=t.shape[2],
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
    )

    assert "CUDA tensors" in get_fused_mrope_unavailable_reason(t, freqs)
    with pytest.warns(UserWarning, match="CUDA tensors.*Using unfused implementation"):
        out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup())
    assert _ROPE_FUSION_FALLBACK_WARNINGS == {"triton-mrope-unavailable-device"}

    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False)
    torch.testing.assert_close(ref, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_raw_mrope_unsupported_dtype_falls_back_to_unfused():
    t, freqs, mrope_section = _make_inputs(dtype=torch.float64)
    config = TransformerConfig(
        num_attention_heads=t.shape[2],
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
    )

    assert "dtype" in get_fused_mrope_unavailable_reason(t, freqs)
    with pytest.warns(UserWarning, match="dtype.*Using unfused implementation"):
        out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup())
    assert _ROPE_FUSION_FALLBACK_WARNINGS == {"triton-mrope-unavailable-dtype"}

    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False)
    torch.testing.assert_close(ref, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_apply_rotary_pos_emb_dispatches_raw_mrope(interleaved_mrope):
    t, freqs, mrope_section = _make_inputs(interleaved_mrope=interleaved_mrope)
    config = TransformerConfig(
        num_attention_heads=t.shape[2],
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
        mrope_interleaved=interleaved_mrope,
    )

    out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup())

    emb = mrope_freqs_to_rotary_emb(
        freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
    )
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False)
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_raw_mrope_unsupported_freq_dtype_warning_key_is_dtype():
    t, freqs, mrope_section = _make_inputs()
    freqs = freqs.to(torch.float16)
    config = TransformerConfig(
        num_attention_heads=t.shape[2],
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
    )

    assert "float32" in get_fused_mrope_unavailable_reason(t, freqs)
    with pytest.warns(UserWarning, match="float32.*Using unfused implementation"):
        out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup())
    assert _ROPE_FUSION_FALLBACK_WARNINGS == {"triton-mrope-unavailable-dtype"}

    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False)
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("layout", ["bshd", "thd"])
def test_materialized_mrope_falls_back_without_te_fused_rope(monkeypatch, layout):
    if layout == "bshd":
        t, freqs, mrope_section = _make_inputs(batch=1)
        cu_seqlens = None
        monkeypatch.setattr(rope_utils, "fused_apply_rotary_pos_emb", None)
    else:
        t, freqs, cu_seqlens, mrope_section = _make_thd_inputs()
        monkeypatch.setattr(rope_utils, "fused_apply_rotary_pos_emb_thd", None)

    config = _make_mrope_config(t.shape[-2], mrope_section)
    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)

    with pytest.warns(UserWarning, match="Transformer Engine fused RoPE.*unavailable"):
        out = apply_rotary_pos_emb(t, emb, config, cu_seqlens, cp_group=FakeCPGroup())

    if layout == "bshd":
        ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False)
    else:
        ref = _apply_rotary_pos_emb_thd(t, cu_seqlens, emb, cp_group=FakeCPGroup())
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "fallback_kwargs, expected_warning_key, warning_text",
    [
        (
            {"mscale": 1.25},
            "te-rope-thd-mscale",
            "mscale=1.25 is not supported by TE's fused RoPE for THD layout",
        ),
        (
            {"inverse": True},
            "te-rope-thd-inverse",
            "inverse RoPE is not supported by TE's fused RoPE for THD layout",
        ),
        (
            {"mla_rotary_interleaved": True},
            "te-rope-thd-mla-rotary-interleaved",
            "does not support MLA-style interleaving",
        ),
    ],
)
def test_materialized_thd_mrope_option_fallbacks_do_not_call_te(
    monkeypatch, fallback_kwargs, expected_warning_key, warning_text
):
    t, freqs, cu_seqlens, mrope_section = _make_thd_inputs()
    config = _make_mrope_config(t.shape[1], mrope_section)
    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)

    def unexpected_te_thd_call(*args, **kwargs):
        raise AssertionError("TE THD fused RoPE should not be called")

    monkeypatch.setattr(rope_utils, "fused_apply_rotary_pos_emb_thd", unexpected_te_thd_call)
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        out = apply_rotary_pos_emb(
            t, emb, config, cu_seqlens, cp_group=FakeCPGroup(), **fallback_kwargs
        )

    fallback_warnings = _fallback_warnings(recorded_warnings)
    assert len(fallback_warnings) == 1
    assert warning_text in str(fallback_warnings[0].message)
    assert _ROPE_FUSION_FALLBACK_WARNINGS == {expected_warning_key}

    ref = _apply_rotary_pos_emb_thd(
        t, cu_seqlens, emb, cp_group=FakeCPGroup(), **fallback_kwargs
    )
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
@pytest.mark.parametrize("cp_size, cp_rank", [(1, 0), (2, 0), (2, 1)])
def test_fused_mrope_thd_matches_unfused_forward_backward(
    interleaved_mrope, cp_size, cp_rank, monkeypatch
):
    t_ref, freqs, cu_seqlens, mrope_section = _make_thd_inputs(
        requires_grad=True, interleaved_mrope=interleaved_mrope, cp_size=cp_size
    )
    t_fused = t_ref.detach().clone().requires_grad_(True)
    cp_group = FakeCPGroup(size=cp_size, rank=cp_rank)
    config = TransformerConfig(
        num_attention_heads=t_ref.shape[1],
        num_layers=1,
        context_parallel_size=cp_size,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
        mrope_interleaved=interleaved_mrope,
    )

    emb = mrope_freqs_to_rotary_emb(
        freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
    )
    ref = _apply_rotary_pos_emb_thd(t_ref, cu_seqlens, emb, cp_group=cp_group)

    fused_calls = 0
    orig_fused_apply_mrope_thd = rope_utils.fused_apply_mrope_thd

    def wrapped_fused_apply_mrope_thd(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return orig_fused_apply_mrope_thd(*args, **kwargs)

    def unexpected_pack(*args, **kwargs):
        raise AssertionError("raw THD mRoPE fusion should not materialize packed freqs")

    monkeypatch.setattr(rope_utils, "fused_apply_mrope_thd", wrapped_fused_apply_mrope_thd)
    monkeypatch.setattr(rope_utils, "_pack_thd_raw_mrope_freqs", unexpected_pack)
    out = apply_rotary_pos_emb(t_fused, freqs, config, cu_seqlens, cp_group=cp_group)
    assert fused_calls == 1

    tols = _dtype_tols(t_ref.dtype)
    torch.testing.assert_close(ref.float(), out.float(), **tols)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    out.backward(grad)
    torch.testing.assert_close(t_ref.grad.float(), t_fused.grad.float(), **tols)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_apply_rotary_pos_emb_thd_eval_uses_triton_without_te(interleaved_mrope, monkeypatch):
    t, freqs, cu_seqlens, mrope_section = _make_thd_inputs(
        interleaved_mrope=interleaved_mrope
    )
    config = _make_mrope_config(t.shape[1], mrope_section, interleaved_mrope)

    emb = mrope_freqs_to_rotary_emb(
        freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
    )
    ref = _apply_rotary_pos_emb_thd(t, cu_seqlens, emb, cp_group=FakeCPGroup())

    fused_calls = 0
    orig_fused_apply_mrope_thd = rope_utils.fused_apply_mrope_thd

    def wrapped_fused_apply_mrope_thd(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return orig_fused_apply_mrope_thd(*args, **kwargs)

    def unexpected_pack(*args, **kwargs):
        raise AssertionError("raw THD mRoPE fusion should not materialize packed freqs")

    monkeypatch.setattr(rope_utils, "fused_apply_rotary_pos_emb_thd", None)
    monkeypatch.setattr(rope_utils, "fused_apply_mrope_thd", wrapped_fused_apply_mrope_thd)
    monkeypatch.setattr(rope_utils, "_pack_thd_raw_mrope_freqs", unexpected_pack)
    with torch.no_grad(), warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        out = apply_rotary_pos_emb(t, freqs, config, cu_seqlens, cp_group=FakeCPGroup())

    assert fused_calls == 1
    assert not _fallback_warnings(recorded_warnings)
    assert not out.requires_grad
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_apply_rotary_pos_emb_thd_fused_dispatch_does_not_read_cuda_scalars(monkeypatch):
    t, freqs, cu_seqlens, mrope_section = _make_thd_inputs()
    config = _make_mrope_config(t.shape[1], mrope_section)

    def unexpected_item(_tensor):
        raise AssertionError("fused raw THD mRoPE dispatch should not call Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", unexpected_item)
    out = apply_rotary_pos_emb(t, freqs, config, cu_seqlens, cp_group=FakeCPGroup())

    assert out.shape == t.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize(
    "fallback_kwargs, config_kwargs, expected_warning_key, warning_text",
    [
        (
            {"mscale": 1.25},
            {},
            "triton-mrope-thd-mscale",
            "mscale=1.25 is not supported by Triton fused mRoPE for THD layout",
        ),
        (
            {"inverse": True},
            {},
            "triton-mrope-thd-inverse",
            "inverse RoPE is not supported by Triton fused mRoPE for THD layout",
        ),
        (
            {"mla_rotary_interleaved": True},
            {},
            "triton-mrope-thd-mla-rotary-interleaved",
            "does not support MLA-style interleaving",
        ),
        (
            {},
            {"rotary_interleaved": True},
            "triton-mrope-thd-rotary-interleaved",
            "currently supports rotary_interleaved=False",
        ),
    ],
)
def test_apply_rotary_pos_emb_thd_raw_mrope_fallback_emits_option_warning(
    fallback_kwargs, config_kwargs, expected_warning_key, warning_text
):
    t, freqs, cu_seqlens, mrope_section = _make_thd_inputs()
    config = _make_mrope_config(t.shape[1], mrope_section, **config_kwargs)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        out = apply_rotary_pos_emb(
            t, freqs, config, cu_seqlens, cp_group=FakeCPGroup(), **fallback_kwargs
        )

    fallback_warnings = _fallback_warnings(recorded_warnings)
    assert len(fallback_warnings) == 1
    assert warning_text in str(fallback_warnings[0].message)
    assert _ROPE_FUSION_FALLBACK_WARNINGS == {expected_warning_key}

    emb = mrope_freqs_to_rotary_emb(
        freqs, mrope_section, rotary_interleaved=config.rotary_interleaved
    )
    ref = _apply_rotary_pos_emb_thd(
        t,
        cu_seqlens,
        emb,
        rotary_interleaved=config.rotary_interleaved,
        cp_group=FakeCPGroup(),
        **fallback_kwargs,
    )
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_thd_raw_mrope_rejects_sequence_length_mismatch():
    t, freqs, cu_seqlens, mrope_section = _make_thd_inputs()
    config = _make_mrope_config(t.shape[1], mrope_section)
    bad_freqs = freqs[:, :, :-1, :].contiguous()

    with pytest.raises(ValueError, match="sequence length must match local tokens"):
        apply_rotary_pos_emb(t, bad_freqs, config, cu_seqlens, cp_group=FakeCPGroup())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_thd_raw_mrope_rejects_batch_dimension_greater_than_one():
    t, freqs, cu_seqlens, mrope_section = _make_thd_inputs()
    config = _make_mrope_config(t.shape[1], mrope_section)
    bad_freqs = freqs.expand(-1, 2, -1, -1).contiguous()

    with pytest.raises(ValueError, match="singleton batch dimension"):
        apply_rotary_pos_emb(t, bad_freqs, config, cu_seqlens, cp_group=FakeCPGroup())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_thd_raw_mrope_rejects_non_thd_tensor_shape():
    t, freqs, cu_seqlens, mrope_section = _make_thd_inputs()
    config = _make_mrope_config(t.shape[1], mrope_section)

    with pytest.raises(ValueError, match="raw mRoPE THD expects t"):
        apply_rotary_pos_emb(
            t[..., :8].unsqueeze(1), freqs, config, cu_seqlens, cp_group=FakeCPGroup()
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_fused_mrope_thd_public_api_matches_unfused():
    t, freqs, cu_seqlens, mrope_section = _make_thd_inputs()
    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)

    ref = _apply_rotary_pos_emb_thd(t, cu_seqlens, emb, cp_group=FakeCPGroup())
    assert (
        get_fused_mrope_thd_unavailable_reason(
            t, cu_seqlens, freqs, cp_size=1, cp_rank=0
        )
        is None
    )
    out = fused_apply_mrope_thd(t, cu_seqlens, freqs, mrope_section)

    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_fused_mrope_thd_fp32_compute_matches_explicit_cast_forward_backward():
    t_ref, freqs, cu_seqlens, mrope_section = _make_thd_inputs(requires_grad=True)
    t_fused = t_ref.detach().clone().requires_grad_(True)
    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)

    ref = _apply_rotary_pos_emb_thd(
        t_ref.float(), cu_seqlens, emb, cp_group=FakeCPGroup()
    ).to(t_ref.dtype)
    out = fused_apply_mrope_thd(
        t_fused, cu_seqlens, freqs, mrope_section, fp32_compute=True
    )

    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t_ref.dtype))

    grad = torch.randn_like(ref)
    ref.backward(grad)
    out.backward(grad)
    torch.testing.assert_close(t_ref.grad.float(), t_fused.grad.float(), **_dtype_tols(t_ref.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("return_raw_freqs", [False, True])
def test_mrope_packed_seq_keeps_global_freqs_with_context_parallel(return_raw_freqs):
    class FakeCPGroup2:
        def size(self):
            return 2

        def rank(self):
            return 0

    seq = 16
    batch = 1
    head_dim = 20
    rotary_dim = 16
    mrope_section = [2, 3, 3]
    cp_group = FakeCPGroup2()
    position_ids = _make_position_ids(seq, batch)
    rope = MultimodalRotaryEmbedding(
        head_dim,
        rotary_percent=rotary_dim / head_dim,
        cp_group=cp_group,
    )

    unpacked_freqs = rope(
        position_ids,
        mrope_section,
        cp_group=cp_group,
        return_raw_freqs=return_raw_freqs,
        packed_seq=False,
    )
    packed_freqs = rope(
        position_ids,
        mrope_section,
        cp_group=cp_group,
        return_raw_freqs=return_raw_freqs,
        packed_seq=True,
    )

    seq_dim = 2 if return_raw_freqs else 0
    assert unpacked_freqs.shape[seq_dim] == seq // cp_group.size()
    assert packed_freqs.shape[seq_dim] == seq


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.skipif(Utils.world_size < 2, reason="CP test requires at least 2 distributed ranks")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_raw_mrope_fusion_matches_unfused_with_context_parallel(interleaved_mrope):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        seq = 32
        batch = 2
        heads = 3
        head_dim = 20
        rotary_dim = 16
        mrope_section = [3, 3, 2] if interleaved_mrope else [2, 3, 3]
        position_ids = _make_position_ids(seq, batch)

        rope = MultimodalRotaryEmbedding(
            head_dim,
            rotary_percent=rotary_dim / head_dim,
            cp_group=cp_group,
            interleaved_mrope=interleaved_mrope,
        )
        raw_freqs = rope(
            position_ids,
            mrope_section,
            cp_group=cp_group,
            return_raw_freqs=True,
        )
        materialized_emb = rope(position_ids, mrope_section, cp_group=cp_group)
        raw_freqs_emb = mrope_freqs_to_rotary_emb(
            raw_freqs,
            mrope_section,
            interleaved_mrope=interleaved_mrope,
            rotary_interleaved=False,
        )
        torch.testing.assert_close(raw_freqs_emb, materialized_emb)

        local_seq = seq // cp_group.size()
        assert raw_freqs.shape == (3, batch, local_seq, rotary_dim // 2)
        assert materialized_emb.shape == (local_seq, batch, 1, rotary_dim)

        generator = torch.Generator(device="cuda").manual_seed(4321)
        t_ref = torch.randn(
            local_seq,
            batch,
            heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
            requires_grad=True,
        )
        t_fused = t_ref.detach().clone().requires_grad_(True)

        config = TransformerConfig(
            num_attention_heads=heads,
            num_layers=1,
            context_parallel_size=cp_group.size(),
            apply_rope_fusion=True,
            mrope_section=mrope_section,
            mrope_interleaved=interleaved_mrope,
        )

        ref = _apply_rotary_pos_emb_bshd(t_ref, materialized_emb, rotary_interleaved=False)
        out = apply_rotary_pos_emb(t_fused, raw_freqs, config, cp_group=cp_group)
        tols = _dtype_tols(t_ref.dtype)
        torch.testing.assert_close(ref.float(), out.float(), **tols)

        grad = torch.randn_like(ref)
        ref.backward(grad)
        out.backward(grad)
        torch.testing.assert_close(t_ref.grad.float(), t_fused.grad.float(), **tols)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.skipif(Utils.world_size < 2, reason="THD CP test requires at least 2 distributed ranks")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_raw_mrope_thd_fusion_matches_unfused_with_context_parallel(interleaved_mrope):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        t_ref, _, cu_seqlens, mrope_section = _make_thd_inputs(
            requires_grad=True, interleaved_mrope=interleaved_mrope, cp_size=cp_group.size()
        )
        t_fused = t_ref.detach().clone().requires_grad_(True)
        config = TransformerConfig(
            num_attention_heads=t_ref.shape[1],
            num_layers=1,
            context_parallel_size=cp_group.size(),
            apply_rope_fusion=True,
            mrope_section=mrope_section,
            mrope_interleaved=interleaved_mrope,
        )
        total_seq = int(cu_seqlens[-1].item())
        position_ids = _make_position_ids(total_seq, 1)
        rope = MultimodalRotaryEmbedding(
            t_ref.shape[-1],
            rotary_percent=16 / t_ref.shape[-1],
            cp_group=cp_group,
            interleaved_mrope=interleaved_mrope,
        )
        freqs = rope(
            position_ids,
            mrope_section,
            cp_group=cp_group,
            return_raw_freqs=True,
            packed_seq=True,
        )
        emb = rope(position_ids, mrope_section, cp_group=cp_group, packed_seq=True)

        raw_freqs_emb = mrope_freqs_to_rotary_emb(
            freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
        )
        assert freqs.shape == (3, 1, total_seq, 8)
        assert emb.shape == (total_seq, 1, 1, 16)
        torch.testing.assert_close(raw_freqs_emb, emb)

        ref = _apply_rotary_pos_emb_thd(t_ref, cu_seqlens, emb, cp_group=cp_group)
        out = apply_rotary_pos_emb(t_fused, freqs, config, cu_seqlens, cp_group=cp_group)
        tols = _dtype_tols(t_ref.dtype)
        torch.testing.assert_close(ref.float(), out.float(), **tols)

        grad = torch.randn_like(ref)
        ref.backward(grad)
        out.backward(grad)
        torch.testing.assert_close(t_ref.grad.float(), t_fused.grad.float(), **tols)
    finally:
        Utils.destroy_model_parallel()

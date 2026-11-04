# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static-batching TE attention with strided q, k or v matches an fp32 reference.

TE's ``get_qkv_layout`` takes a tensor that is not a view of a packed QKV or KV buffer to be dense
in its format (``sbhd_sbhd_sbhd``) without checking its strides, and the fused (cuDNN) backend
then reads it with dense strides. Static-batching attention without RoPE or q_layernorm passes
the query as a strided view of the fused QKV projection when a rank has one query head per KV
group (MHA) or one KV group (MQA), next to k and v from the separate static KV cache. A batch
smaller than the context's ``max_batch_size`` passes strided batch slices of that cache.

The cases compare against an fp32 attention over the very q, k and v Megatron hands its TE
wrapper. Their bound, rel L2 1e-2, is five times the bf16 error measured on H100 for these shapes
(1.7e-3 to 2.3e-3); a query read with dense strides misses by 0.1 to 0.25. They use only public
API, so they also run on releases without the fix. The rule itself is tested on CPU in
``test_te_attention_qkv_layout_rule.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

BATCH = 2
PROMPT = 16
HIDDEN = 256
HEADS = 8
HEAD_DIM = HIDDEN // HEADS
MAX_SEQ = 64
REL_TOL = 1e-2

BACKENDS = {
    "default": {},
    "cudnn": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "0"},
    "no_fused": {"NVTE_FUSED_ATTN": "0"},
}

# Query groups per rank whose query get_query_key_value_tensors leaves a strided view.
STRIDED_QUERY_GROUPS = {"mha": HEADS, "mqa": 1}


def _use_backend(monkeypatch, name):
    """Apply the backend environment and make TE select its backend again."""
    clear_nvte_env_vars()
    for key, value in BACKENDS[name].items():
        monkeypatch.setenv(key, value)
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as te_dpa,
    )

    monkeypatch.setattr(
        te_dpa,
        "_attention_backends",
        {
            "attention_params": None,
            "use_flash_attention": None,
            "flash_attention_backend": None,
            "use_fused_attention": None,
            "fused_attention_backend": None,
            "use_unfused_attention": None,
            "backend_selection_requires_update": False,
        },
    )


def _selected_backend():
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as te_dpa,
    )

    selected = te_dpa._attention_backends
    if selected["use_fused_attention"]:
        return f"FusedAttention (sub-backend {selected['fused_attention_backend']})"
    if selected["use_flash_attention"]:
        return f"FlashAttention ({selected['flash_attention_backend']})"
    if selected["use_unfused_attention"]:
        return "UnfusedDotProductAttention"
    return "no backend"


def _config(num_query_groups):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=num_query_groups,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def _self_attention(num_query_groups):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.attention import SelfAttention

    attention = SelfAttention(
        _config(num_query_groups),
        get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,  # as the GPT layer spec builds it
    )
    return attention.cuda().eval()


def _reference(query, key, value, causal):
    """fp32 attention. q [sq, b, np, hn], k/v [sk, b, ng, hn] -> [sq, b, np * hn]."""
    sq, b, heads, dim = query.shape
    sk = key.shape[0]
    key, value = (x.float().repeat_interleave(heads // x.shape[2], dim=2) for x in (key, value))
    scores = torch.einsum("qbhd,kbhd->bhqk", query.float(), key) * dim**-0.5
    if causal:
        mask = torch.ones(sq, sk, dtype=torch.bool, device=query.device).triu(sk - sq + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    out = torch.einsum("bhqk,kbhd->qbhd", torch.softmax(scores, dim=-1), value)
    return out.reshape(sq, b, heads * dim)


def _rel(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _static_prefill_and_decode(attention, max_batch_size=BATCH, contiguous_query=False):
    """Run a static prefill of PROMPT tokens and one decode step.

    Returns one record per step: the output, the fp32 reference over the q/k/v the step handed
    Megatron's TE wrapper, and their layouts.
    """
    context = StaticInferenceContext(max_batch_size=max_batch_size, max_sequence_length=MAX_SEQ)
    generator = torch.Generator(device="cuda").manual_seed(5)
    steps = []
    wrapper_forward = attention.core_attention.forward

    def spy(query, key, value, attention_mask, attn_mask_type, **kwargs):
        if contiguous_query:
            query = query.contiguous()
        out = wrapper_forward(query, key, value, attention_mask, attn_mask_type, **kwargs)
        causal = attn_mask_type == AttnMaskType.causal
        steps.append(
            SimpleNamespace(
                out=out,
                ref=_reference(query, key, value, causal),
                query_stride=tuple(query.stride()),
                query_dense=query.is_contiguous(),
                kv_dense=key.is_contiguous() and value.is_contiguous(),
            )
        )
        return out

    attention.core_attention.forward = spy
    with torch.no_grad(), InferenceMode.active():
        for tokens in (PROMPT, 1):
            hidden = torch.randn(tokens, BATCH, HIDDEN, generator=generator, device="cuda")
            attention.forward_pre_attn_and_core_attn(
                hidden.bfloat16(), None, inference_context=context
            )
            context.sequence_len_offset += tokens
            context.enable_decode_mode()
    assert len(steps) == 2, "harness: TE core attention not reached on every step"
    return dict(zip(("prefill", "decode"), steps))


def _te_wrapper_attention(strided_query):
    """Megatron's TEDotProductAttention on sbhd q/k/v; q optionally a strided view at offset 0."""
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    attention = TEDotProductAttention(
        _config(HEADS), layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
    )
    generator = torch.Generator(device="cuda").manual_seed(9)
    packed = torch.randn(PROMPT, BATCH, HEADS, 3 * HEAD_DIM, generator=generator, device="cuda")
    query = (packed * 0.5).bfloat16()[..., :HEAD_DIM]  # like the fused QKV split, offset 0
    if not strided_query:
        query = query.contiguous()
    key, value = (
        torch.randn(PROMPT, BATCH, HEADS, HEAD_DIM, generator=generator, device="cuda")
        for _ in range(2)
    )
    key, value = (key * 0.5).bfloat16(), value.bfloat16()
    with torch.no_grad():
        out = attention(query, key, value, None, AttnMaskType.causal)
    return out.reshape(PROMPT, BATCH, -1), _reference(query, key, value, causal=True), query


def _run(fn, backend):
    try:
        return fn()
    except (RuntimeError, ValueError) as error:
        # TE raises ValueError("No dot product attention backend is available ...").
        if backend == "cudnn" and "backend" in str(error).lower():
            pytest.skip(f"FusedAttention is unavailable for this shape: {error}")
        raise


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TE), reason="needs a GPU and Transformer Engine"
)
class TestStaticTEAttentionStridedInputs:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _check(self, steps, backend, what):
        for name, step in steps.items():
            error = _rel(step.out, step.ref)
            assert error <= REL_TOL, (
                f"{what}, static {name}: rel {error:.3g} from the fp32 reference (bound "
                f"{REL_TOL}) on {_selected_backend()} [{backend}]; query stride "
                f"{step.query_stride}, dense query {step.query_dense}, dense k/v {step.kv_dense}"
            )

    def test_control_contiguous_query_prefill_matches_reference(self, monkeypatch):
        _use_backend(monkeypatch, "default")
        steps = _static_prefill_and_decode(_self_attention(HEADS), contiguous_query=True)
        assert steps["prefill"].query_dense
        self._check(steps, "default", "MHA with the query made contiguous by the test")

    def test_control_strided_query_without_fused_attention(self, monkeypatch):
        """FlashAttention and unfused attention read the strided query by its own strides."""
        _use_backend(monkeypatch, "no_fused")
        steps = _static_prefill_and_decode(_self_attention(HEADS))
        self._check(steps, "no_fused", "MHA, strided query")

    @pytest.mark.parametrize("heads", STRIDED_QUERY_GROUPS)
    @pytest.mark.parametrize("backend", ["default", "cudnn"])
    def test_strided_query_matches_reference(self, monkeypatch, backend, heads):
        """No RoPE or q_layernorm, MHA or MQA per rank: the query is a strided view."""
        _use_backend(monkeypatch, backend)
        attention = _self_attention(STRIDED_QUERY_GROUPS[heads])
        steps = _run(lambda: _static_prefill_and_decode(attention), backend)
        for name, step in steps.items():
            assert not step.query_dense, f"harness: the {name} query reached TE dense"
            assert step.kv_dense, f"harness: the {name} k/v reached TE strided"
        self._check(steps, backend, f"{heads.upper()}, strided query")

    @pytest.mark.parametrize("backend", ["default", "cudnn"])
    def test_batch_below_max_batch_size_matches_reference(self, monkeypatch, backend):
        """k and v are batch slices of a cache built for twice the batch (GQA: dense query)."""
        _use_backend(monkeypatch, backend)
        attention = _self_attention(2)
        steps = _run(
            lambda: _static_prefill_and_decode(attention, max_batch_size=2 * BATCH), backend
        )
        for name, step in steps.items():
            assert step.query_dense, f"harness: the {name} query reached TE strided"
            assert not step.kv_dense, f"harness: the {name} k/v reached TE dense"
        self._check(steps, backend, "GQA, k/v batch slices of the static KV cache")

    def test_control_te_wrapper_contiguous_query_matches_reference(self, monkeypatch):
        _use_backend(monkeypatch, "default")
        out, ref, _ = _te_wrapper_attention(strided_query=False)
        error = _rel(out, ref)
        assert error <= REL_TOL, f"harness: rel {error:.3g} on {_selected_backend()}"

    @pytest.mark.parametrize("backend", ["default", "cudnn"])
    def test_te_wrapper_strided_query_matches_reference(self, monkeypatch, backend):
        """The strided query beside separate k/v, straight into Megatron's TE wrapper."""
        _use_backend(monkeypatch, backend)
        out, ref, query = _run(lambda: _te_wrapper_attention(strided_query=True), backend)
        assert not query.is_contiguous()
        error = _rel(out, ref)
        assert error <= REL_TOL, (
            f"TEDotProductAttention with a strided query (stride {tuple(query.stride())}) and "
            f"separate k/v: rel {error:.3g} on {_selected_backend()} [{backend}]"
        )

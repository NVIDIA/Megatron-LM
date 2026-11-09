# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the RoPE dispatch from the dynamic inference context.

Call sites: ``DynamicInferenceContext.apply_rotary_emb_query`` / ``apply_rotary_emb_key``
(megatron/core/inference/contexts/dynamic_context.py). They gather one frequency row per token
and call ``rope_utils.apply_rotary_pos_emb``, which runs Transformer Engine's
``fused_apply_rotary_pos_emb`` when ``config.apply_rope_fusion`` is set, ``mscale`` is 1 and the
layer is not MLA, and the unfused torch rotation otherwise. The attention layer rotates keys
before appending them to the KV cache (``Attention._adjust_key_value_for_inference``) and
queries before the attention kernel (``Attention.forward_pre_attn_and_core_attn``), with YaRN's
concentration factor as ``mscale``; the MLA layer rotates its RoPE parts the same way
(``multi_latent_attention.py``, packed ``[n, H, d]`` queries and ``[n, 1, d]`` keys). Also
``apply_rotary_pos_emb`` on packed (THD) sequences, whose fused path #8377 changed: packed
callers, not the dynamic context (which passes no ``cu_seqlens``).

Contract (from RoPE's definition, the table format and the call sites):

* Positions. Live token ``i`` of request ``r`` in a step sits at position ``c_r + i``, where
  ``c_r`` counts the request's tokens already in the KV cache: 0 for a prompt, the chunk offset
  for a later prefill chunk, the sequence length for a decode token; a speculative step runs the
  last sampled token and the ``k`` speculative ones at ``c_r .. c_r + k``. Its query row and its
  key row are both rotated at that position.
* Table. ``emb [L, 1, 1, d_rot]`` (``RotaryEmbedding``): row ``p`` holds pair ``i``'s angle in
  channels ``i`` and ``i + d_rot / 2`` (NeoX layout) or ``2i`` and ``2i + 1`` (interleaved).
  Packed sequences: a table of more rows than the longest sequence holds one row per packed
  token (token ``j`` takes row ``j``); otherwise token ``p`` of each sequence takes row ``p``.
* Rotation. Pairs ``(a, b) -> (a cos - b sin, a sin + b cos)`` with ``cos, sin`` of the angle,
  both times ``mscale``. NeoX pairs ``(i, i + d_rot / 2)``; interleaved pairs ``(2i, 2i + 1)``
  (``config.rotary_interleaved``); MLA (``config.multi_latent_attention``) pairs ``(2i, 2i + 1)``
  written back de-interleaved, ``[a'_0 .. a'_{d/2-1}, b'_0 .. b'_{d/2-1}]`` (the order MLA's q and
  k share; DeepSeek's definition up to that common permutation). Channels from ``d_rot`` on pass
  through unchanged. Shape and dtype kept.
* Rows past the live tokens (padding up to ``padded_active_token_count``) are unspecified.
  ``cu_seqlens_q`` given to the query method is ignored (frequencies are already per token).
* The fused and unfused paths meet the same contract.
* Layer: the key reaching the KV cache and the query reaching the attention kernel are the
  layer's own projections rotated as above, with ``mscale`` = YaRN's concentration factor
  ``0.1 ln s + 1`` (factor ``s``) and 1 without YaRN; the sink logits reaching the kernel are
  none (vanilla), zero per head (off-by-one) or the layer's learned ones (learnable).

Reference: fp64 rotation of the input rows by the table's fp32 angles (exact in fp64) at the
positions the test computes from its own request bookkeeping (not from the context's index
buffers, which it only cross-checks). Tolerance class: floating point, per token row, bounded by
the same rotation with torch's elementwise ops in the input dtype (cos/sin from the fp32 angles
times ``mscale``, rounded to the dtype). Pass-through channels exact; rows at angle 0 (position 0)
with ``mscale`` 1 exact (identity).

Variants (pairwise): TE fused and unfused; NeoX, interleaved and MLA pairing; ``mscale`` 1 and
1.369 (YaRN's factor 40, which must take the unfused path); full and half rotary dims; sbhd
``[n, 1, H, D]`` and packed ``[n, H, D]`` queries (#7211's packed branch); bf16 and fp16; a
prefill (prompts 1, 63, 64, 65, 200 and ``max_len - 8`` around the 64-token block, and a
chunk of 63 of a 129-token prompt), a mixed step (decodes next to the chunk's last 66 tokens at
63..128) and a decode-only step (keys rotated out of place), and a speculative verification
step (``k = 2``, three tokens per request). Requests are interleaved in the token rows, so the
gathered frequencies are not monotone. Production: Llama-3.1-8B's geometry (32/8 heads of 128,
base 5e5 with Llama 3 scaling) with prompts to 8000 tokens. Packed sequences: shared and
per-token tables, the per-token table longer than the tokens (fused path falls back), fused and
unfused. Inputs in NaN-guarded buffers; padding rows NaN.

Hardware: H100 (TE fused RoPE and torch).
"""

import itertools
import math
from typing import Dict, List, Tuple

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.attention import attention_reference as ar
from tests.unit_tests.test_utilities import Utils

HEADS, KV_HEADS, HEAD_DIM, MLA_ROPE = 8, 2, 128, 64
BLOCK = 64
MAX_LEN = 1024
BASE = 10000.0
YARN_FACTOR = 40.0
YARN_MSCALE = 0.1 * math.log(YARN_FACTOR) + 1.0  # YaRN's concentration factor, 1.3689
LONG = MAX_LEN - 8
PROMPTS = {0: 1, 1: 63, 2: 64, 3: 65, 4: 200, 5: LONG}
CHUNKED, CHUNKED_PROMPT, CHUNK = 6, 129, 63

H100 = ou.requires_sm(ou.HOPPER)


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------------------------
# Reference (from the contract)
# ---------------------------------------------------------------------------------------------


def pair_angles(table: torch.Tensor, positions: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """[L, d_rot / 2] angles of each row's pairs, read from the table rows at ``positions``."""
    rows = table.reshape(table.shape[0], -1)[positions]
    if interleaved:
        return rows[:, 0::2]
    return rows[:, : rows.shape[-1] // 2]


def rotation_reference(x, angles, mscale, *, pairing):
    """Rotate ``x [L, H, D]`` by ``angles [L, d_rot / 2]`` with amplitude ``mscale``."""
    cos = (torch.cos(angles) * mscale)[:, None, :]
    sin = (torch.sin(angles) * mscale)[:, None, :]
    return _rotate(x, cos, sin, pairing, ar.rope_reference)


def rotation_trusted(x, angles32, mscale, *, pairing):
    """The same rotation with torch ops in x's dtype; cos/sin from the fp32 angles."""
    cos = (torch.cos(angles32) * mscale)[:, None, :]
    sin = (torch.sin(angles32) * mscale)[:, None, :]
    return _rotate(x, cos, sin, pairing, ar.rope_lowp)


def _rotate(x, cos, sin, pairing, rope):
    rot = 2 * cos.shape[-1]
    y = rope(x, cos, sin, interleaved=pairing != "neox")
    if pairing == "mla":  # pairs (2i, 2i + 1), written back de-interleaved
        y = torch.cat([y[..., 0:rot:2], y[..., 1:rot:2], y[..., rot:]], dim=-1)
    return y


def _check_rotation(out, x, table, positions, mscale, pairing, what):
    """``out``/``x``: [L, H, D] live rows; ``positions``: [L]."""
    angles = pair_angles(table, positions, pairing == "interleaved")
    ref = ou.fp64_reference(rotation_reference, x, angles, mscale, pairing=pairing)
    trusted = rotation_trusted(x, angles, mscale, pairing=pairing)
    flat = lambda t: t.reshape(t.shape[0], -1)  # noqa: E731
    ou.assert_close_to_reference(flat(out), flat(ref), flat(trusted), dim=-1, what=what)
    rot = 2 * angles.shape[-1]
    ou.assert_exact(out[..., rot:], x[..., rot:], what=f"{what}: pass-through channels")
    identity = (angles == 0).all(dim=-1)  # rows at position 0 of a RoPE table
    if mscale == 1.0 and pairing != "mla" and bool(identity.any()):
        ou.assert_exact(out[identity], x[identity], what=f"{what}: rows at angle 0")


# ---------------------------------------------------------------------------------------------
# Context and steps
# ---------------------------------------------------------------------------------------------


def _model_config(heads=HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM, **extra):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=heads * head_dim,
        num_attention_heads=heads,
        num_query_groups=kv_heads,
        kv_channels=head_dim,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        **extra,
    )


def _rope_config(fusion: bool, pairing: str, dtype: torch.dtype):
    """The layer config the RoPE methods read (``apply_rope_fusion``, ``rotary_interleaved``,
    ``multi_latent_attention``)."""
    if pairing == "mla":
        from megatron.core.transformer.transformer_config import MLATransformerConfig

        return MLATransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=HEADS,
            q_lora_rank=None,
            kv_lora_rank=128,
            qk_head_dim=64,
            qk_pos_emb_head_dim=MLA_ROPE,
            v_head_dim=64,
            rope_type="yarn",  # MLA accepts apply_rope_fusion only with YaRN (training path)
            apply_rope_fusion=fusion,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            params_dtype=dtype,
            add_bias_linear=False,
        )
    return _model_config(apply_rope_fusion=fusion, rotary_interleaved=pairing == "interleaved")


def _context(config, *, speculative=0, max_len=MAX_LEN, max_tokens=2048, buffer_gb=0.05):
    """A dynamic context with 64-token blocks; chunked prefill on unless speculative."""
    from megatron.core.inference.config import InferenceConfig
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    inference = InferenceConfig(
        max_sequence_length=max_len,
        buffer_size_gb=buffer_gb,
        paused_buffer_size_gb=buffer_gb / 10,
        block_size_tokens=BLOCK,
        max_tokens=max_tokens,
        max_requests=32,
        num_cuda_graphs=None,
        use_flashinfer_fused_rope=False,
        unified_memory_level=0,
        num_speculative_tokens=speculative,
        enable_chunked_prefill=speculative == 0,
    )
    return DynamicInferenceContext(model_config=config, inference_config=inference)


def _request(request_id: int, prompt_len: int):
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

    # Distinct prompts, so that no request shares a prefix block with another.
    tokens = torch.arange(1, prompt_len + 1, device="cpu") + 1000 * request_id
    return DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=tokens,
        sampling_params=SamplingParams(num_tokens_to_generate=4, termination_id=-1),
    )


def _admit_chunk(context, request, length: int):
    """Admit the next ``length`` prompt tokens as the engine's chunked-prefill scheduler does."""
    if length < request.remaining_prompt_length:
        context.add_request(request, prefill_chunk_length=length)
        context.chunked_prefill_request_id = request.request_id
        request.remaining_prompt_tokens = request.remaining_prompt_tokens[length:]
        request.finished_chunk_token_count += length
    else:
        context.chunked_prefill_request_id = -1
        context.add_request(request)
        request.remaining_prompt_tokens = request.remaining_prompt_tokens.new_empty(0)


def _advance(context):
    active = context.total_request_count - context.paused_request_count
    k = context.num_speculative_tokens
    context.update_requests(
        active_requests_mask=torch.ones(active, dtype=torch.int32),
        new_tokens=torch.zeros(active, dtype=torch.int64),
        new_speculative_tokens=torch.zeros(k, active, dtype=torch.int64) if k else None,
    )


def _live_positions(context, expected: Dict[int, Tuple[int, int]], what: str) -> torch.Tensor:
    """Positions of the step's live token rows, from ``expected`` {request: (cached, new)} in the
    context's row order. Cross-checks the context's per-request lengths and offsets."""
    first, last = context.paused_request_count, context.total_request_count
    order = [int(r) for r in context.request_ids[first:last].tolist()]
    assert sorted(order) == sorted(expected), f"{what}: active requests {order}"
    positions: List[int] = []
    for i, rid in enumerate(order):
        cached, new = expected[rid]
        got = (
            int(context.request_kv_length_offsets[first + i]),
            int(context.request_query_lengths[first + i]),
        )
        assert got == (cached, new), f"{what}: request {rid} at {got}, expected {(cached, new)}"
        positions.extend(range(cached, cached + new))
    return torch.tensor(positions, device="cuda")


def _steps(context, scenario: str):
    """Drive ``context`` through a scenario; yield (step name, {request: (cached, new)})."""
    if scenario == "speculative":
        k = context.num_speculative_tokens
        for rid, prompt in PROMPTS.items():
            context.add_request(_request(rid, prompt))
        yield "prefill", {rid: (0, n) for rid, n in PROMPTS.items()}
        _advance(context)
        yield "verification", {rid: (n, k + 1) for rid, n in PROMPTS.items()}
        return
    for rid, prompt in PROMPTS.items():
        context.add_request(_request(rid, prompt))
    chunked = _request(CHUNKED, CHUNKED_PROMPT)
    _admit_chunk(context, chunked, CHUNK)
    prefill = {rid: (0, n) for rid, n in PROMPTS.items()}
    prefill[CHUNKED] = (0, CHUNK)
    yield "prefill", prefill
    _advance(context)
    _admit_chunk(context, chunked, chunked.remaining_prompt_length)
    mixed = {rid: (n, 1) for rid, n in PROMPTS.items()}
    mixed[CHUNKED] = (CHUNK, CHUNKED_PROMPT - CHUNK)
    yield "mixed", mixed
    _advance(context)
    decode = {rid: (n + 1, 1) for rid, n in PROMPTS.items()}
    decode[CHUNKED] = (CHUNKED_PROMPT, 1)
    yield "decode", decode


def _table(rot_dim: int, interleaved: bool, length: int, **rope):
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    rope.setdefault("rotary_base", BASE)
    module = RotaryEmbedding(rot_dim, 1.0, rotary_interleaved=interleaved, **rope)
    return module.get_emb(length)


# ---------------------------------------------------------------------------------------------
# The context's RoPE methods
# ---------------------------------------------------------------------------------------------


def _rotate_step(
    context,
    config,
    table,
    positions,
    *,
    mscale,
    pairing,
    layout,
    dtype,
    heads,
    kv_heads,
    head_dim,
    rot_dim,
    what,
):
    """Rotate random query and key rows of the current step; check every live row."""
    from megatron.core import parallel_state

    cp_group = parallel_state.get_context_parallel_group()
    n = context.padded_active_token_count
    live = positions.numel()
    assert n >= live, f"{what}: {n} padded rows for {live} live tokens"
    g = ou.make_generator(device="cuda")
    width = rot_dim if pairing == "mla" else head_dim
    q_shape = (n, heads, width) if layout == "packed" else (n, 1, heads, width)
    k_shape = (n, 1, width) if pairing == "mla" else (n, 1, kv_heads, width)
    q = ou.guarded_like(ou.poisoned_empty(q_shape, dtype))
    k = ou.guarded_like(ou.poisoned_empty(k_shape, dtype))
    q.tensor[:live] = torch.randn(live, *q_shape[1:], device="cuda", generator=g).to(dtype)
    k.tensor[:live] = torch.randn(live, *k_shape[1:], device="cuda", generator=g).to(dtype)
    q_in, k_in = q.tensor.clone(), k.tensor.clone()
    cu_seqlens_q = context.cu_query_lengths()[0]  # given by MLA; must not change the rows
    with torch.no_grad():
        q_out = context.apply_rotary_emb_query(
            q.tensor, table, config, cu_seqlens_q, cp_group, mscale=mscale
        )
        k_out = context.apply_rotary_emb_key(k.tensor, table, config, cp_group, mscale=mscale)
    torch.cuda.synchronize()
    ou.check_guards(query=q, key=k)
    assert q_out.shape == q_shape and q_out.dtype == dtype, f"{what}: query {q_out.shape}"
    assert k_out.shape[0] == n and k_out.dtype == dtype, f"{what}: key {k_out.shape}"
    rows = lambda t: t[:live].reshape(live, -1, width)  # noqa: E731
    _check_rotation(rows(q_out), rows(q_in), table, positions, mscale, pairing, f"{what}, query")
    _check_rotation(rows(k_out), rows(k_in), table, positions, mscale, pairing, f"{what}, key")


def _dispatch_cases():
    cases = ou.pairwise_cases(
        {
            "fusion": [False, True],
            "pairing": ["neox", "interleaved", "mla"],
            "mscale": [1.0, YARN_MSCALE],
            "rotary_percent": [1.0, 0.5],
            "layout": ["sbhd", "packed"],
            "dtype": [torch.bfloat16, torch.float16],
            "scenario": ["steps", "speculative"],
        }
    )
    # The TE fused kernel runs only with mscale 1 and no MLA: every such production layout.
    fused = dict(fusion=True, mscale=1.0, rotary_percent=1.0, layout="sbhd", dtype=torch.bfloat16)
    for pairing in ("neox", "interleaved"):
        for scenario in ("steps", "speculative"):
            cases.append(dict(fused, pairing=pairing, scenario=scenario))
    params, seen = [], set()
    for c in cases:
        if c["pairing"] == "mla":
            c["rotary_percent"] = 1.0  # MLA rotates its whole RoPE part
        name = "-".join(
            [
                "te_fused" if c["fusion"] else "unfused",
                c["pairing"],
                f"mscale={c['mscale']:.3g}",
                f"rot={c['rotary_percent']}",
                c["layout"],
                str(c["dtype"]).replace("torch.", ""),
                c["scenario"],
            ]
        )
        if name not in seen:
            seen.add(name)
            params.append(pytest.param(c, id=name, marks=H100))
    return params


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("case", _dispatch_cases())
def test_context_rotates_queries_and_keys_at_token_positions(case):
    """Every live query and key row of a prefill, mixed and decode-only step (or a prefill and a
    speculative verification step) is rotated at its token's position, on both dispatch paths."""
    pairing, dtype = case["pairing"], case["dtype"]
    speculative = 2 if case["scenario"] == "speculative" else 0
    context = _context(_model_config(), speculative=speculative)
    config = _rope_config(case["fusion"], pairing, dtype)
    if pairing == "mla":
        rot_dim, interleaved_table = MLA_ROPE, False
    else:
        rot_dim, interleaved_table = int(HEAD_DIM * case["rotary_percent"]), pairing != "neox"
    table = _table(rot_dim, interleaved_table, context.max_sequence_length_for_model)
    for name, expected in _steps(context, case["scenario"]):
        context.initialize_attention_state()
        what = f"{name} step {case}"
        positions = _live_positions(context, expected, what)
        _rotate_step(
            context,
            config,
            table,
            positions,
            mscale=case["mscale"],
            pairing=pairing,
            layout=case["layout"],
            dtype=dtype,
            heads=HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rot_dim=rot_dim,
            what=what,
        )


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "te_fused"])
@H100
def test_production_geometry(fusion):
    """Llama-3.1-8B's attention geometry (32 query and 8 KV heads of 128, base 5e5 with Llama 3
    frequency scaling) over prompts up to 8000 tokens: a prefill and a decode step."""
    heads, kv_heads, head_dim, max_len = 32, 8, 128, 8192
    prompts = {0: 1, 1: 255, 2: 256, 3: 257, 4: 2048, 5: 4095, 6: 4096, 7: 8000}
    context = _context(
        _model_config(heads, kv_heads, head_dim), max_len=max_len, max_tokens=20480, buffer_gb=0.3
    )
    config = _model_config(heads, kv_heads, head_dim, apply_rope_fusion=fusion)
    table = _table(
        head_dim,
        False,
        context.max_sequence_length_for_model,
        rotary_base=500000,
        rope_scaling=True,
    )
    for rid, prompt in prompts.items():
        context.add_request(_request(rid, prompt))
    for name, expected in (
        ("prefill", {rid: (0, n) for rid, n in prompts.items()}),
        ("decode", {rid: (n, 1) for rid, n in prompts.items()}),
    ):
        if name == "decode":
            _advance(context)
        context.initialize_attention_state()
        positions = _live_positions(context, expected, name)
        _rotate_step(
            context,
            config,
            table,
            positions,
            mscale=1.0,
            pairing="neox",
            layout="sbhd",
            dtype=torch.bfloat16,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rot_dim=head_dim,
            what=f"Llama-3.1-8B {name} step ({'TE fused' if fusion else 'unfused'})",
        )


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.sanitizer
@H100
def test_small_case_for_sanitizer():
    """A three-request prefill and decode through the TE fused and the unfused rotation (the
    compute-sanitizer memcheck case)."""
    prompts = {0: 3, 1: 65, 2: 1}
    for fusion in (True, False):
        context = _context(_model_config(), max_len=256, buffer_gb=0.01)
        config = _rope_config(fusion, "neox", torch.bfloat16)
        table = _table(HEAD_DIM // 2, False, context.max_sequence_length_for_model)
        for rid, prompt in prompts.items():
            context.add_request(_request(rid, prompt))
        for name, expected in (
            ("prefill", {rid: (0, n) for rid, n in prompts.items()}),
            ("decode", {rid: (n, 1) for rid, n in prompts.items()}),
        ):
            if name == "decode":
                _advance(context)
            context.initialize_attention_state()
            positions = _live_positions(context, expected, name)
            _rotate_step(
                context,
                config,
                table,
                positions,
                mscale=1.0,
                pairing="neox",
                layout="sbhd",
                dtype=torch.bfloat16,
                heads=HEADS,
                kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                rot_dim=HEAD_DIM // 2,
                what=f"sanitizer {name} (fusion={fusion})",
            )


# ---------------------------------------------------------------------------------------------
# The attention layer: RoPE before the cache append and the kernel, YaRN amplitude, sink logits
# ---------------------------------------------------------------------------------------------


def _layer(config):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType

    model_parallel_cuda_manual_seed(ou.DEFAULT_SEED)
    submodules = get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
    layer = SelfAttention(config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal)
    return layer.cuda().eval()


class _Probe:
    """Captures the layer's projections before RoPE, the keys it appends to the cache and the
    query and sink logits it gives the attention kernel (replaced by a stub returning zeros)."""

    def __init__(self, layer, context):
        self.pre, self.keys, self.kernel = [], [], []
        projections = layer.get_query_key_value_tensors

        def capture_projections(*args, **kwargs):
            out = projections(*args, **kwargs)
            self.pre.append([t.detach().clone() for t in out[:2]])
            return out

        append = context.append_key_value_cache

        def capture_append(layer_number, key, value):
            self.keys.append(key.detach().clone())
            return append(layer_number, key, value)

        def kernel(q, k, v, *args, softmax_offset=None, **kwargs):
            offset = None if softmax_offset is None else softmax_offset.detach().clone()
            self.kernel.append((q.detach().clone(), offset))
            return torch.zeros_like(q)

        layer.get_query_key_value_tensors = capture_projections
        context.append_key_value_cache = capture_append
        layer.flash_decode_and_prefill = kernel


LAYER_CASES = [
    # (id, apply_rope_fusion, YaRN scaling factor, softmax_type)
    ("unfused", False, None, "vanilla"),
    ("te_fused", True, None, "vanilla"),
    ("te_fused-yarn", True, YARN_FACTOR, "vanilla"),
    ("unfused-yarn-off_by_one", False, YARN_FACTOR, "off-by-one"),
    ("te_fused-learnable", True, None, "learnable"),
]


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "fusion, yarn, softmax_type", [pytest.param(*c[1:], id=c[0], marks=H100) for c in LAYER_CASES]
)
def test_attention_layer_rotates_at_token_positions(fusion, yarn, softmax_type):
    """A SelfAttention layer (TE spec) in prefill and decode steps: the keys it appends and the
    query it hands the attention kernel are its own projections rotated at each token's
    position, with YaRN's amplitude when the config scales RoPE; the kernel gets the layer's sink
    logits (none, zeros or the learned ones)."""
    config = _model_config(apply_rope_fusion=fusion, softmax_type=softmax_type)
    if yarn is not None:
        config.yarn_rotary_scaling_factor = yarn  # as megatron/training/argument_utils.py sets it
    layer = _layer(config)
    if softmax_type == "learnable":
        with torch.no_grad():
            g = ou.make_generator(seed=ou.DEFAULT_SEED + 3, device="cuda")
            offset = layer.core_attention.softmax_offset
            offset.copy_(torch.randn(offset.shape, device="cuda", generator=g).to(offset.dtype))
    mscale = 1.0 if yarn is None else 0.1 * math.log(yarn) + 1.0
    prompts = {0: 5, 1: 64, 2: 70}
    context = _context(config, max_len=256, buffer_gb=0.01)
    probe = _Probe(layer, context)
    table = _table(HEAD_DIM, False, context.max_sequence_length_for_model)
    for rid, prompt in prompts.items():
        context.add_request(_request(rid, prompt))
    from megatron.core.inference.utils import InferenceMode

    for name, expected in (
        ("prefill", {rid: (0, n) for rid, n in prompts.items()}),
        ("decode", {rid: (n, 1) for rid, n in prompts.items()}),
    ):
        if name == "decode":
            _advance(context)
        context.initialize_attention_state()
        positions = _live_positions(context, expected, name)
        live, n = positions.numel(), context.padded_active_token_count
        g = ou.make_generator(device="cuda")
        hidden = torch.zeros(n, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        hidden[:live] = torch.randn(live, 1, config.hidden_size, device="cuda", generator=g).to(
            hidden.dtype
        )
        probe.pre.clear()
        probe.keys.clear()
        probe.kernel.clear()
        with torch.no_grad(), InferenceMode.active():
            layer(hidden, attention_mask=None, inference_context=context, rotary_pos_emb=table)
        assert len(probe.pre) == len(probe.keys) == len(probe.kernel) == 1, f"{name}: one call"
        (q_pre, k_pre), key, (q, offset) = probe.pre[0], probe.keys[0], probe.kernel[0]
        what = f"{name} step (fusion={fusion}, yarn={yarn})"
        rows = lambda t: t[:live].reshape(live, -1, HEAD_DIM)  # noqa: E731
        _check_rotation(rows(q), rows(q_pre), table, positions, mscale, "neox", f"{what}, query")
        _check_rotation(rows(key), rows(k_pre), table, positions, mscale, "neox", f"{what}, key")
        heads = layer.num_attention_heads_per_partition
        if softmax_type == "vanilla":
            assert offset is None, f"{what}: vanilla softmax got sink logits"
        elif softmax_type == "off-by-one":
            assert offset is not None and offset.shape == (heads,), f"{what}: {offset}"
            assert bool((offset == 0).all()), f"{what}: off-by-one sink logits {offset}"
        else:
            ou.assert_exact(
                offset, layer.core_attention.softmax_offset.detach(), what=f"{what}: sink logits"
            )


# ---------------------------------------------------------------------------------------------
# Packed (THD) sequences: shared and per-token frequency tables (#8377)
# ---------------------------------------------------------------------------------------------

PACKED_LENGTHS = [5, 1, 9, 3]  # longest 9; 18 packed tokens


def _packed_table(rows: int, rot_dim: int, interleaved: bool, generator) -> torch.Tensor:
    """A frequency table of ``rows`` rows with arbitrary angles in RoPE's table layout."""
    angles = torch.rand(rows, rot_dim // 2, device="cuda", generator=generator) * 2 * math.pi
    angles = angles * torch.arange(1, rows + 1, device="cuda")[:, None]  # large angles too
    if interleaved:
        table = angles.repeat_interleave(2, dim=-1)
    else:
        table = torch.cat([angles, angles], dim=-1)
    return table[:, None, None, :].contiguous()


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("interleaved", [False, True], ids=["neox", "interleaved"])
@pytest.mark.parametrize("table_kind", ["shared", "per_token", "per_token_padded"])
@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "te_fused"])
@H100
def test_packed_sequences_take_their_table_rows(fusion, table_kind, interleaved):
    """Packed sequences of lengths 5, 1, 9, 3 (THD, no context parallelism): with a table of
    ``max_seqlen`` rows, token ``p`` of each sequence takes row ``p``; with a table of more rows
    than the longest sequence (one per packed token, or more), packed token ``j`` takes row ``j``,
    on both paths (#8377 runs the per-token case through TE's SBHD kernel when the table has
    exactly one row per token, and falls back to the unfused path otherwise)."""
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    g = ou.make_generator(device="cuda")
    total, longest = sum(PACKED_LENGTHS), max(PACKED_LENGTHS)
    cu = torch.tensor([0, *itertools.accumulate(PACKED_LENGTHS)], device="cuda").int()
    rot_dim = HEAD_DIM
    rows = {"shared": longest, "per_token": total, "per_token_padded": total + 4}[table_kind]
    table = _packed_table(rows, rot_dim, interleaved, g)
    if table_kind == "shared":
        positions = torch.cat([torch.arange(n) for n in PACKED_LENGTHS]).cuda()
    else:
        positions = torch.arange(total, device="cuda")
    config = _model_config(apply_rope_fusion=fusion, rotary_interleaved=interleaved)
    t = ou.guarded_like(torch.randn(total, HEADS, HEAD_DIM, device="cuda", generator=g).bfloat16())
    before = t.tensor.clone()
    with torch.no_grad():
        out = apply_rotary_pos_emb(
            t.tensor,
            table,
            config=config,
            cu_seqlens=cu,
            cp_group=parallel_state.get_context_parallel_group(),
            max_seqlen=longest,
        )
    torch.cuda.synchronize()
    ou.check_guards(t=t)
    ou.assert_exact(t.tensor, before, what="packed input (read-only)")
    pairing = "interleaved" if interleaved else "neox"
    _check_rotation(out, before, table, positions, 1.0, pairing, f"packed {table_kind}")


# ---------------------------------------------------------------------------------------------
# CPU checks of the reference itself (no kernel)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pairing", ["neox", "interleaved", "mla"])
def test_reference_rotates_complex_pairs(pairing):
    """The reference against complex multiplication by ``m exp(i theta)`` on each pair, in the
    output order the pairing defines."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 2, 8, generator=g, dtype=torch.float64)
    angles = torch.rand(3, 3, generator=g, dtype=torch.float64) * 7  # d_rot 6 of 8
    out = rotation_reference(x, angles, 1.5, pairing=pairing)
    if pairing == "neox":
        a, b = x[..., 0:3], x[..., 3:6]
    else:
        a, b = x[..., 0:6:2], x[..., 1:6:2]
    z = torch.complex(a, b) * 1.5 * torch.exp(1j * angles)[:, None, :]
    if pairing == "interleaved":
        expected = torch.stack([z.real, z.imag], dim=-1).flatten(-2)
    else:
        expected = torch.cat([z.real, z.imag], dim=-1)
    torch.testing.assert_close(out[..., :6], expected)
    torch.testing.assert_close(out[..., 6:], x[..., 6:])

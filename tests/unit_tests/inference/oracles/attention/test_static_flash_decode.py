# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for static ``flash_decode``: FA2 ``flash_attn_with_kvcache`` with in-kernel RoPE.

Call site: ``Attention.flash_decode`` (megatron/core/transformer/attention.py), the decode-phase
attention kernel of the static engine with ``flash_decode=True`` (required for static CUDA
graphs), and the static steps around it (``Attention.forward`` with a ``StaticInferenceContext``).

Contract (from the call site, the static engine's RoPE inputs and the flash-attn API):

* Inputs: ``query_layer [1, B, H, D]``, ``key_layer``/``value_layer [1, B, Hk, D]`` (this step's
  token per row), ``inference_key_memory``/``inference_value_memory [S_max, B_max, Hk, D]`` (the
  static cache, ``B <= B_max``), ``sequence_len_offset [B]`` int32 (row ``b``'s position
  ``pos_b``), ``rotary_cos``/``rotary_sin [S_max, d_rot / 2]`` = cos/sin of
  ``theta[p, i] = p * base ** (-2 i / d_rot)`` (or None: no RoPE), ``rotary_interleaved``.
* Effects: ``k' = rope(k, pos_b)`` and ``v`` are written to ``cache[pos_b, b]``; no other cache
  element changes (rows ``!= pos_b``, slots ``>= B``).
* Output ``[B, 1, H, D]``: ``attention(rope(q, pos_b), keys cache[0..pos_b, b])`` with the layer's
  scale, softcap, sink, and sliding window ``(W, 0)`` (keys ``j >= pos_b - W``).
* RoPE: pairs ``(i, i + d_rot/2)`` (or ``(2i, 2i+1)`` interleaved) rotated by ``theta``; channels
  past ``d_rot`` pass through. The call site rounds cos/sin to the query dtype before the call;
  the reference takes those rounded tables as the kernel's inputs.
* Static steps (layer level): the attention output that reaches ``linear_proj`` equals causal
  attention of each step's RoPE'd queries over every RoPE'd key so far (window applied), times
  ``sigmoid(gate)`` with ``attention_output_gate``, for ``flash_decode`` on and off, whether or not
  the batch fills the static cache (``B < B_max`` makes ``k``/``v`` strided batch slices).

Reference: fp64 RoPE from the definition (``attention_reference.rope_reference``) and fp64
attention over the reference keys. Tolerance class: floating point, bounded per row by the trusted
yardstick (RoPE and attention in bf16 with torch ops: ``rope_lowp``, ``attention_lowp``); the
value write is exact (a copy).

Variants covered: positions ``0, 1, 127, 128, 129, 255, 256, 257, S_max - 1`` in one batch (FA2's
128-key split-KV tiles), batch below the cache's ``B_max`` (poisoned unused slots); GQA 1, 4, 8;
head dim 64 and 128; NeoX and interleaved pairs; full and half rotary dim; no RoPE; softcap;
learnable sink; window ``127, 128, 129``. Layer level: ``flash_decode`` on/off x output gate x
window x full/partial batch, plus GPT and Hybrid static generation (control and #8863 case).

Open PRs (written against main):
* #8878 (t3-flashdecode): main passes no ``window_size`` to the decode kernel (R11-F3) and
  returns before the output gate (R11-F4): the ``window`` kernel cases and the layer cases with
  ``flash_decode`` and a window or the gate fail on main and pass with #8878.
* t3-p4n2 (branch fix/te-strided-query-layout): TE's FusedAttention reads strided ``k``/``v``
  batch slices of the static cache as dense, so the layer cases with a partial batch fail on main
  in every step that runs TE attention (prefill, and decode with ``flash_decode`` off) and pass
  with it; ``[batch=full-mha_without_rope]`` (a strided query view of the fused QKV output, P4-N2)
  likewise.
* Strided ``k``/``v`` (P4-N7, suspected): with a batch below ``B_max`` the static path hands TE
  ``k``/``v`` as batch slices of the cache. TE (2.20's ``get_qkv_layout``) takes them as
  ``sbhd_sbhd_sbhd`` without making them dense only when ``k`` and ``v`` have the same strides and
  zero offsets; a RoPE'd key is a new dense tensor, so the RoPE cases above never reach it.
  ``[batch=partial-gqa_without_rope]`` does (GQA keeps the query dense: its reshape copies), and
  asserts that it does before comparing; it fails on main only if P4-N7 is real. The
  flash-decode partial-batch cases reach it in their prefill (keys are RoPE'd before caching).
* #8863 (t3-r11f5): HybridModel never builds ``rotary_pos_cos/sin``, so its flash-decode static
  generation applies no RoPE (R11-F5): ``test_model_static_generation[hybrid]`` fails on main and
  passes with #8863 (``[gpt]`` is the control). #8864 (t3-p4n1) keeps the GPT case passing.
"""

from typing import List, Optional

import pytest
import torch

from megatron.core.transformer import attention as attention_module
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.attention import attention_reference as ar
from tests.unit_tests.test_utilities import Utils

HEADS = 8
ROPE_BASE = 10000.0
S_MAX = 512
POSITIONS = [0, 1, 127, 128, 129, 255, 256, 257, S_MAX - 1]
SOFTCAP = 8.0


def _require_fa2() -> None:
    if attention_module.flash_attn_with_kvcache is None:
        if ou.session_target_sm() is not None:
            pytest.fail("flash-attn 2 is not installed", pytrace=False)
        pytest.skip("flash-attn 2 is not installed")


@pytest.fixture
def model_parallel():
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)  # the model tests' weight init forks this RNG
    yield
    Utils.destroy_model_parallel()


def _config(
    *,
    kv_heads: int,
    head_dim: int,
    rotary_interleaved: bool = False,
    window: Optional[int] = None,
    softcap: Optional[float] = None,
    sink: bool = False,
    flash_decode: bool = True,
    gate: bool = False,
):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=HEADS * head_dim,
        num_attention_heads=HEADS,
        num_query_groups=kv_heads,
        kv_channels=head_dim,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        rotary_interleaved=rotary_interleaved,
        window_size=None if window is None else (window, 0),
        attn_logit_softcapping=softcap,
        softmax_type="learnable" if sink else "vanilla",
        flash_decode=flash_decode,
        attention_output_gate=gate,
        init_method_std=0.05,
    )


def _attention(config):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType

    submodules = get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
    module = SelfAttention(config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal)
    module = module.cuda().eval()
    if config.softmax_type == "learnable":
        g = ou.make_generator(seed=ou.DEFAULT_SEED + 1, device="cuda")
        with torch.no_grad():
            offset = module.core_attention.softmax_offset
            offset.copy_((torch.randn(offset.shape, device="cuda", generator=g) * 2 + 1))
    return module


def _tables(rot_dim: int, dtype: torch.dtype):
    """cos/sin [S_MAX, rot_dim / 2] of the exact fp64 angles, rounded to ``dtype`` (the call site
    rounds the model's tables to the query dtype before the kernel)."""
    theta = ar.rope_angles(torch.arange(S_MAX, device="cuda"), rot_dim, ROPE_BASE)
    return torch.cos(theta).to(dtype), torch.sin(theta).to(dtype)


# ---------------------------------------------------------------------------------------------
# Kernel level: Attention.flash_decode
# ---------------------------------------------------------------------------------------------


def _kernel_cases():
    # The window comes first so that its cases (R11-F3, #8878) can be selected by id prefix.
    factors = {
        "window": [None, 127, 128, 129],
        "kv_heads": [HEADS, HEADS // 4, HEADS // 8],
        "head_dim": [64, 128],
        "interleaved": [False, True],
        "rotary": ["full", "half", "none"],
        "softcap": [None, SOFTCAP],
        "sink": [False, True],
    }
    return [
        pytest.param(
            case,
            id="-".join(f"{k}={v}" for k, v in case.items() if v not in (None, False)),
            marks=ou.requires_sm(ou.HOPPER),
        )
        for case in ou.pairwise_cases(factors)
    ]


def _flash_decode_step(module, case, positions: List[int], g):
    """One ``flash_decode`` call on a poisoned static cache. Returns everything the checks need."""
    head_dim, kv_heads = case["head_dim"], case["kv_heads"]
    dtype = torch.bfloat16
    batch, batch_max = len(positions), len(positions) + 2
    pos = torch.tensor(positions, dtype=torch.int32, device="cuda")
    # The cache: earlier keys/values random (already RoPE'd), every other element NaN.
    k_mem = ou.poisoned_empty((S_MAX, batch_max, kv_heads, head_dim), dtype)
    v_mem = ou.poisoned_empty((S_MAX, batch_max, kv_heads, head_dim), dtype)
    for b, p in enumerate(positions):
        k_mem[:p, b] = torch.randn(p, kv_heads, head_dim, device="cuda", generator=g).to(dtype)
        v_mem[:p, b] = torch.randn(p, kv_heads, head_dim, device="cuda", generator=g).to(dtype)
    k_before, v_before = k_mem.clone(), v_mem.clone()
    q_scale = 4.0 if case["softcap"] is not None else 1.0
    q = (torch.randn(1, batch, HEADS, head_dim, device="cuda", generator=g) * q_scale).to(dtype)
    k = torch.randn(1, batch, kv_heads, head_dim, device="cuda", generator=g).to(dtype)
    v = torch.randn(1, batch, kv_heads, head_dim, device="cuda", generator=g).to(dtype)
    rot_dim = {"full": head_dim, "half": head_dim // 2, "none": 0}[case["rotary"]]
    cos, sin = _tables(rot_dim, dtype) if rot_dim else (None, None)
    with torch.no_grad():
        out = module.flash_decode(
            sequence_len_offset=pos,
            query_layer=q,
            key_layer=k,
            value_layer=v,
            inference_key_memory=k_mem,
            inference_value_memory=v_mem,
            rotary_cos=cos,
            rotary_sin=sin,
            rotary_interleaved=case["interleaved"],
        )
    torch.cuda.synchronize()
    return dict(
        out=out,
        q=q,
        k=k,
        v=v,
        k_mem=k_mem,
        v_mem=v_mem,
        k_before=k_before,
        v_before=v_before,
        cos=cos,
        sin=sin,
        positions=positions,
    )


def _rotate(x, cos, sin, positions, interleaved, lowp):
    """x [B, heads, D] rotated at each row's position (identity without tables)."""
    if cos is None:
        return x if lowp else x.double()
    idx = torch.tensor(positions, device=x.device)
    c, s = cos[idx][:, None, :], sin[idx][:, None, :]
    if lowp:
        return ar.rope_lowp(x, c, s, interleaved=interleaved)
    return ou.fp64_reference(ar.rope_reference, x, c, s, interleaved=interleaved)


def _check_flash_decode(module, step, case, window):
    out, q, k, v = step["out"], step["q"][0], step["k"][0], step["v"][0]
    positions, interleaved = step["positions"], case["interleaved"]
    cos, sin = step["cos"], step["sin"]
    k_mem, v_mem = step["k_mem"], step["v_mem"]
    batch = len(positions)
    idx = torch.arange(batch, device="cuda")
    pos = torch.tensor(positions, device="cuda")

    # Cache writes: k' at [pos_b, b] (bounded), v copied exactly, nothing else touched.
    k_ref = _rotate(k, cos, sin, positions, interleaved, lowp=False)
    k_trusted = _rotate(k, cos, sin, positions, interleaved, lowp=True)
    ou.assert_close_to_reference(
        k_mem[pos, idx], k_ref, k_trusted, what="RoPE'd key written to cache"
    )
    ou.assert_exact(v_mem[pos, idx], v, what="value written to cache")
    written = torch.zeros(k_mem.shape[:2], dtype=torch.bool, device="cuda")
    written[pos, idx] = True
    ou.assert_exact(k_mem[~written], step["k_before"][~written], what="K cache outside [pos_b, b]")
    ou.assert_exact(v_mem[~written], step["v_before"][~written], what="V cache outside [pos_b, b]")

    # Attention of the RoPE'd query over keys 0..pos_b, with k' (exact rotation) at pos_b.
    q_ref = _rotate(q, cos, sin, positions, interleaved, lowp=False)
    q_trusted = _rotate(q, cos, sin, positions, interleaved, lowp=True)
    scale = module.get_softmax_scale(q)
    sink = module._get_inference_softmax_offset()
    refs, trusted = [], []
    for b, p in enumerate(positions):
        keys_before = step["k_before"][:p, b]
        values = torch.cat([step["v_before"][:p, b], v[b : b + 1]])
        allowed = ar.allowed_keys(1, p + 1, window, "cuda")
        r_out, _ = ar.attention_fp64(
            q_ref[b : b + 1],
            torch.cat([keys_before.double(), k_ref[b : b + 1]]),
            values.double(),
            allowed,
            scale=scale,
            softcap=case["softcap"],
            sink=None if sink is None else sink.double(),
        )
        refs.append(r_out)
        trusted.append(
            ar.attention_lowp(
                q_trusted[b : b + 1],
                torch.cat([keys_before, k_trusted[b : b + 1]]),
                values,
                allowed,
                scale=scale,
                softcap=case["softcap"],
                sink=sink,
            )
        )
    assert out.shape == (batch, 1, HEADS, q.shape[-1]), f"output shape {tuple(out.shape)}"
    flat = lambda t: t.reshape(t.shape[0], -1)  # noqa: E731
    ou.assert_close_to_reference(
        flat(out[:, 0]),
        flat(torch.cat(refs)),
        flat(torch.cat(trusted)),
        dim=-1,
        what=f"flash_decode output {case}",
    )


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("case", _kernel_cases())
def test_flash_decode_matches_reference(case):
    """One decode call: RoPE at each row's position, the cache append, and attention over the
    row's prefix (positions around FA2's 128-key tiles) match the fp64 reference."""
    _require_fa2()
    config = _config(
        kv_heads=case["kv_heads"],
        head_dim=case["head_dim"],
        rotary_interleaved=case["interleaved"],
        window=case["window"],
        softcap=case["softcap"],
        sink=case["sink"],
    )
    module = _attention(config)
    step = _flash_decode_step(module, case, POSITIONS, ou.make_generator(device="cuda"))
    _check_flash_decode(module, step, case, case["window"])


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_flash_decode_small_case_for_sanitizer():
    """A small poisoned decode call (compute-sanitizer memcheck case)."""
    _require_fa2()
    case = dict(
        kv_heads=2,
        head_dim=64,
        interleaved=False,
        rotary="full",
        window=None,
        softcap=None,
        sink=False,
    )
    module = _attention(_config(kv_heads=2, head_dim=64))
    step = _flash_decode_step(module, case, [0, 3, 129], ou.make_generator(device="cuda"))
    _check_flash_decode(module, step, case, None)


# ---------------------------------------------------------------------------------------------
# Layer and model level: static steps
# ---------------------------------------------------------------------------------------------


class _Capture:
    """Records each step's attention input (``linear_qkv``'s input, before its fused norm) and the
    tensor that reaches ``linear_proj`` (core attention output, gated)."""

    def __init__(self, attention):
        self.inputs, self.outputs = [], []
        self._hooks = [
            attention.linear_qkv.register_forward_pre_hook(
                lambda mod, args: self.inputs.append(args[0].detach().clone())
            ),
            attention.linear_proj.register_forward_pre_hook(
                lambda mod, args: self.outputs.append(args[0].detach().clone())
            ),
        ]

    def remove(self):
        for hook in self._hooks:
            hook.remove()


def _static_oracle(attention, capture, window: Optional[int], rope: bool, what: str):
    """Check every captured step against fp64 attention over the RoPE'd history."""
    config = attention.config
    head_dim = config.kv_channels
    gate_on = config.attention_output_gate
    steps = []
    with torch.no_grad():
        for hidden in capture.inputs:
            qkv = attention.get_query_key_value_tensors(hidden, None, output_gate=gate_on)
            steps.append(qkv)
    keys, values = [], []
    start = 0
    for t, (qkv, got) in enumerate(zip(steps, capture.outputs)):
        q, k, v = qkv[:3]  # [s, B, H|Hk, D]
        s, batch = q.shape[:2]
        positions = torch.arange(start, start + s, device="cuda")
        if rope:
            theta = ar.rope_angles(positions, head_dim, ROPE_BASE)
            cos64 = torch.cos(theta)[:, None, None, :]
            sin64 = torch.sin(theta)[:, None, None, :]
            interleaved = config.rotary_interleaved
            q64 = ou.fp64_reference(ar.rope_reference, q, cos64, sin64, interleaved=interleaved)
            k64 = ou.fp64_reference(ar.rope_reference, k, cos64, sin64, interleaved=interleaved)
            q_lp = ar.rope_lowp(q, cos64, sin64, interleaved=interleaved)
            k_lp = ar.rope_lowp(k, cos64, sin64, interleaved=interleaved)
        else:
            q64, k64, q_lp, k_lp = q.double(), k.double(), q, k
        keys.append((k64, k_lp))
        values.append(v)
        k_all64 = torch.cat([x[0] for x in keys])
        k_all_lp = torch.cat([x[1] for x in keys])
        v_all = torch.cat(values)
        allowed = ar.allowed_keys(s, start + s, window, "cuda")
        scale = attention.get_softmax_scale(q)
        ref, trusted = [], []
        for b in range(batch):
            r_out, _ = ar.attention_fp64(
                q64[:, b], k_all64[:, b], v_all[:, b].double(), allowed, scale=scale
            )
            l_out = ar.attention_lowp(q_lp[:, b], k_all_lp[:, b], v_all[:, b], allowed, scale=scale)
            if gate_on:
                gate = qkv[3][:, b]
                r_out = r_out * torch.sigmoid(gate.double())
                l_out = (l_out * torch.sigmoid(gate.float())).to(l_out.dtype)
            ref.append(r_out.reshape(s, -1))
            trusted.append(l_out.reshape(s, -1))
        ref = torch.stack(ref, dim=1).reshape(s * batch, -1)
        trusted = torch.stack(trusted, dim=1).reshape(s * batch, -1)
        ou.assert_close_to_reference(
            got.reshape(s * batch, -1), ref, trusted, dim=-1, what=f"{what}, step {t}"
        )
        start += s


def _layer_cases():
    cases = []
    for batch in ("full", "partial"):
        for flash_decode in (True, False):
            for gate in (False, True):
                for window in (None, 8):
                    case_id = (
                        f"batch={batch}-flash_decode={flash_decode}-gate={gate}-window={window}"
                    )
                    cases.append(
                        pytest.param(
                            dict(flash_decode=flash_decode, gate=gate, window=window, batch=batch),
                            id=case_id,
                            marks=ou.requires_sm(ou.HOPPER),
                        )
                    )
    # MHA without RoPE: the query TE receives is a strided view of the fused QKV output.
    cases.append(
        pytest.param(
            dict(flash_decode=False, gate=False, window=None, batch="full", mha_without_rope=True),
            id="batch=full-mha_without_rope",
            marks=ou.requires_sm(ou.HOPPER),
        )
    )
    # GQA without RoPE, batch below B_max: dense query, k/v strided cache slices (P4-N7).
    cases.append(
        pytest.param(
            dict(
                flash_decode=False, gate=False, window=None, batch="partial", gqa_without_rope=True
            ),
            id="batch=partial-gqa_without_rope",
            marks=ou.requires_sm(ou.HOPPER),
        )
    )
    return cases


def _te_layout(q, k, v) -> str:
    """The layout Transformer Engine's detection assigns to (q, k, v), for messages only."""
    try:
        from transformer_engine.pytorch.attention.dot_product_attention.utils import get_qkv_layout

        return str(get_qkv_layout(q, k, v, qkv_format="sbhd")[0])
    except Exception as error:  # noqa: BLE001  (version-dependent internals; report only)
        return f"unknown ({type(error).__name__})"


class _StridedKVProbe:
    """Records, for every static TE core-attention call, whether it receives a dense query and
    ``k``/``v`` that are strided views with the same strides and zero offsets: the inputs TE takes
    as ``sbhd_sbhd_sbhd`` without making them dense."""

    def __init__(self, attention):
        self.calls = []
        self._hook = attention.core_attention.register_forward_pre_hook(self._record)

    def _record(self, module, args):
        q, k, v = args[:3]
        self.calls.append(
            dict(
                q_dense=q.is_contiguous(),
                kv_strided=not k.is_contiguous() and not v.is_contiguous(),
                kv_same_strides=k.stride() == v.stride(),
                zero_offsets=all(x.storage_offset() == 0 for x in (q, k, v)),
                te_layout=_te_layout(q, k, v),
                shapes=[tuple(x.shape) for x in (q, k, v)],
                strides=[x.stride() for x in (q, k, v)],
            )
        )

    def remove(self):
        self._hook.remove()

    def assert_reached(self):
        assert self.calls, "no static TE core-attention call was made"
        for step, call in enumerate(self.calls):
            reached = (
                call["q_dense"]
                and call["kv_strided"]
                and call["kv_same_strides"]
                and call["zero_offsets"]
            )
            assert reached, f"precondition not reached in step {step}: {call}"
        print(f"P4-N7 probe: TE layout {self.calls[0]['te_layout']}; {self.calls[0]}")


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("case", _layer_cases())
def test_static_steps_match_reference(case):
    """A static prefill and four decode steps of one attention layer: the tensor reaching
    ``linear_proj`` is the fp64 attention over the RoPE'd history (gated, windowed)."""
    _require_fa2()
    from megatron.core.inference.contexts.static_context import StaticInferenceContext
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    head_dim, kv_heads, prompt, decode_steps, batch_max, seq_max = 64, 2, 12, 4, 3, 64
    rope = not (case.get("mha_without_rope") or case.get("gqa_without_rope"))
    if case.get("mha_without_rope"):
        kv_heads = HEADS
    config = _config(
        kv_heads=kv_heads,
        head_dim=head_dim,
        window=case["window"],
        flash_decode=case["flash_decode"],
        gate=case["gate"],
    )
    attention = _attention(config)
    batch = batch_max if case["batch"] == "full" else batch_max - 1
    rotary = RotaryEmbedding(kv_channels=head_dim, rotary_percent=1.0, rotary_base=ROPE_BASE)
    context = StaticInferenceContext(batch_max, seq_max)
    # Poisoned cache, so reads outside [0, seq_end) x [0, B) show up as NaN.
    context.key_value_memory_dict[1] = (
        ou.poisoned_empty((seq_max, batch_max, kv_heads, head_dim), torch.bfloat16),
        ou.poisoned_empty((seq_max, batch_max, kv_heads, head_dim), torch.bfloat16),
    )
    g = ou.make_generator(device="cuda")
    capture = _Capture(attention)
    probe = _StridedKVProbe(attention) if case.get("gqa_without_rope") else None
    try:
        with torch.no_grad(), InferenceMode.active():
            for t, s in enumerate([prompt] + [1] * decode_steps):
                if t == 1:
                    context.enable_decode_mode()
                hidden = torch.randn(s, batch, config.hidden_size, device="cuda", generator=g)
                kwargs = dict(attention_mask=None, inference_context=context)
                if case["flash_decode"]:
                    cos, sin = rotary.get_cos_sin(seq_max)
                    kwargs.update(
                        rotary_pos_cos=cos,
                        rotary_pos_sin=sin,
                        sequence_len_offset=torch.full(
                            (batch,), context.sequence_len_offset, dtype=torch.int32, device="cuda"
                        ),
                    )
                elif rope:
                    kwargs.update(rotary_pos_emb=rotary(seq_max))
                attention(hidden.to(torch.bfloat16), **kwargs)
                context.increment_sequence_len_offset(s)
    finally:
        capture.remove()
        if probe is not None:
            probe.remove()
    if probe is not None:
        # The case must hand TE strided k/v slices it takes as dense, or it tests nothing.
        probe.assert_reached()
    _static_oracle(attention, capture, case["window"], rope=rope, what=f"static steps {case}")


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "model", [pytest.param(m, marks=ou.requires_sm(ou.HOPPER)) for m in ("gpt", "hybrid")]
)
def test_model_static_generation(model):
    """Static flash-decode generation of a RoPE model: in the prefill and in each decode step, the
    attention layer's output is the fp64 attention over keys RoPE'd at their positions.

    ``[hybrid]`` fails on main (HybridModel builds no cos/sin tables, so flash decoding applies no
    RoPE, R11-F5) and passes with #8863; ``[gpt]`` is the control."""
    _require_fa2()
    from megatron.core.inference.contexts.static_context import StaticInferenceContext
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.transformer.transformer_config import TransformerConfig

    vocab, seq_max, batch, prompt = 128, 64, 2, 12
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        init_method_std=0.1,
        bf16=True,
        params_dtype=torch.bfloat16,
        flash_decode=True,
    )
    if model == "gpt":
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel

        net = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=vocab,
            max_sequence_length=seq_max,
            position_embedding_type="rope",
        )
        attention = net.decoder.layers[0].self_attention
    else:
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
        from megatron.core.models.hybrid.hybrid_model import HybridModel

        net = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab,
            max_sequence_length=seq_max,
            hybrid_layer_pattern="*-",
            position_embedding_type="rope",
        )
        attention = net.decoder.layers[0].self_attention
    net = net.cuda().eval()
    context = StaticInferenceContext(batch, seq_max)
    g = ou.make_generator(device="cuda")
    tokens = torch.randint(0, vocab, (batch, prompt + 2), device="cuda", generator=g)
    capture = _Capture(attention)
    try:
        with torch.no_grad(), InferenceMode.active():
            start = 0
            for t, s in enumerate([prompt, 1, 1]):
                if t == 1:
                    context.enable_decode_mode()
                ids = tokens[:, start : start + s]
                positions = torch.arange(start, start + s, device="cuda").repeat(batch, 1)
                net(
                    input_ids=ids,
                    position_ids=positions,
                    attention_mask=None,
                    inference_context=context,
                    runtime_gather_output=True,
                )
                context.increment_sequence_len_offset(s)
                start += s
    finally:
        capture.remove()
    _static_oracle(attention, capture, None, rope=True, what=f"{model} static generation")

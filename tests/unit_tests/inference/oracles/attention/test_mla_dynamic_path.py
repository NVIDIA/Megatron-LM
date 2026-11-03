# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the MLA dynamic-inference path: latent cache, absorption, whole-pool up-projection.

Call site: ``MultiLatentAttention.forward`` with a ``DynamicInferenceContext``
(megatron/core/transformer/multi_latent_attention.py, ``cache_mla_latents=True``): the cached-latent
QKV path (latent ``[c, k_pe]`` appended to the 64-token page pool), prefill/mixed steps that
up-project the whole pool (``uncompress_kv_from_cache``) and run paged FlashAttention, decode-only
steps that absorb ``W_UK`` into the query, run FlashMLA over the latents and apply ``W_UV`` after.

Contract (from the MLA math; the training path is the semantic definition):

* For every live token of every step (prefill, decode-only, mixed), the tensor that reaches
  ``linear_proj`` equals the training forward of the same weights (``cache_mla_latents=False``)
  over that request's whole prefix: ``softmax(scale * (q_nope . k_nope + q_pe . k_pe)) v`` per
  head, bottom-right causal, with ``k_nope = W_UK c``, ``v = W_UV c``, ``c = RMSNorm(kv_latent)``
  (the layer's kv norm), ``k_pe`` RoPE'd at the key's position, the layer's ``softmax_scale``, and
  the layer's sink (``softmax_type`` learnable: ``exp(sink_h)`` in the denominator).
* Padding: rows past the live tokens are not part of the contract, but they must not move live
  rows: with an FP8 per-tensor recipe ``linear_proj`` scales its input by an amax over every row,
  so padding rows must reach it as zeros (whatever the attention kernel left there).
* Unused memory: a request's outputs do not depend on pool pages it does not own (free pages,
  other requests' pages, the dummy page), in bf16 and under FP8 per-tensor scaling.
* DSA cannot use the latent cache: its sparse attention needs an indexer cache the dynamic path
  does not have, so ``cache_mla_latents`` with ``experimental_attention_variant='dsa'`` must be
  rejected when the config is built.
* FA2 cannot read 64-token pages nor value heads narrower than query heads: a pinned FA2 must be
  refused with a ValueError, not fail inside flash-attn.

Reference: fp64 MLA attention built from the training path's own bf16 inputs (its RoPE'd ``q`` and
``k_pe`` as core attention receives them, and the latent before the kv norm), with the norm, the
up-projection (``W_UK``/``W_UV`` from the layer's ``linear_kv_up_proj``) and attention in fp64.
Tolerance class: floating point, bounded by the training forward in bf16 (an independent
implementation: unpaged, unabsorbed, TE attention), per token row. Padding and unused-memory
contracts: exact (metamorphic: poisoned vs clean runs are bitwise equal). DSA/FA2: the error.

Variants covered: prompts 63, 64, 65, 129 (around the 64-token page), a prefill step, a
decode-only step (absorbed, FlashMLA), a mixed step (decodes plus a 70-token prefill, whole-pool
up-projection), another decode-only step; vanilla and learnable sink; bf16 and FP8 tensorwise for
the padding and unused-memory contracts; poison values 1e4 and NaN.

Hardware: H100 with a FlashMLA build that has its SM90 kernels (#8701; the manifest runs the SM90
build) and FA3 or FA4 for the paged prefill (FA2 cannot read 64-token pages). The scenario also
runs on GB200, where it fails on main (FlashMLA has no SM100 dense decode, R11-F1; FA4's SM100
forward does not compile in the CI image, G1) and passes with #8873 and #8860.

Open PRs (written against main):
* #8571 (s8): main does not pass the sink to the dynamic MLA kernels, so ``[learnable]`` fails on
  main and passes with s8.
* #8874 (t3-r11f2): main does not zero the MLA attention output's padding rows under FP8 (R11-F2):
  ``test_padding_rows_do_not_reach_live_rows[fp8]`` fails on main and passes with #8874.
* #8702 (n6): main up-projects the whole latent pool through the FP8 linear, so one per-tensor
  scale covers unused pages (R11-F6): ``test_unused_pages_do_not_reach_live_rows[fp8-1e4]`` fails
  on main and passes with n6. ``[bf16-nan]`` also needs the latent tails to be finite or masked by
  the prefill kernel (#8872 zeroes new latent pages).
* #8570 (d5): ``test_dsa_with_latent_cache_is_rejected`` fails on main (accepted) and passes with
  d5. It runs on CPU.
* #8584 (n2): ``test_fa2_is_refused`` fails on main (RuntimeError inside flash-attn) and passes
  with n2.
* #8873 (t3-r11f1): GB200 decode-only steps; see above.
"""

import contextlib
import functools
import subprocess
import sys
from typing import Dict

import pytest
import torch

from megatron.core.transformer import attention as attention_module
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.attention import attention_reference as ar
from tests.unit_tests.test_utilities import Utils

HEADS, HIDDEN, LATENT, ROPE, QK, V = 16, 512, 512, 64, 64, 128
PROMPTS = {0: 63, 1: 64, 2: 65, 3: 129}
LATE_PROMPT = 70  # joins in the mixed step
DEVICES = [ou.requires_sm(ou.HOPPER, ou.BLACKWELL), pytest.mark.launch_on_gb200]

_FLASH_MLA_DECODE_PROBE = """
import sys

import torch
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

torch.cuda.set_device(int(sys.argv[1]))
q = torch.zeros(1, 1, 16, 576, device="cuda", dtype=torch.bfloat16)
kv_cache = torch.zeros(1, 64, 1, 576, device="cuda", dtype=torch.bfloat16)
cache_seqlens = torch.ones(1, device="cuda", dtype=torch.int32)
block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
metadata, num_splits = get_mla_metadata(cache_seqlens, 16, 1)
flash_mla_with_kvcache(q, kv_cache, block_table, cache_seqlens, 512, metadata, num_splits)
torch.cuda.synchronize()
"""


@functools.cache
def _flash_mla_decode_runs() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", _FLASH_MLA_DECODE_PROBE, str(torch.cuda.current_device())],
        capture_output=True,
        timeout=600,
    )
    return probe.returncode == 0


def _require_kernels() -> None:
    """Fail (not skip) where the scenario's kernels are missing on its hardware."""
    if not (attention_module.HAVE_FA4 or attention_module.HAVE_FA3):
        pytest.fail("the latent-cache prefill needs FlashAttention 3 or 4", pytrace=False)
    if ou.device_sm_major() == ou.HOPPER:
        if not attention_module.HAVE_FMLA or not _flash_mla_decode_runs():
            pytest.fail(
                "flash_mla with its SM90 decode kernels is required (the CI image builds it "
                "with FLASH_MLA_DISABLE_SM90=1; #8701)",
                pytrace=False,
            )


@pytest.fixture
def model_parallel():
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _config(*, cache: bool, sink: bool = False, fp8: bool = False, flash_version=None, **extra):
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        q_lora_rank=128,
        kv_lora_rank=LATENT,
        qk_head_dim=QK,
        qk_pos_emb_head_dim=ROPE,
        v_head_dim=V,
        qk_layernorm=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        rope_type="rope",
        apply_rope_fusion=False,
        cache_mla_latents=cache,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        softmax_type="learnable" if sink else "vanilla",
        flash_attention_version=flash_version,
        init_method_std=0.05,
    )
    if fp8:
        from megatron.core.enums import Fp8Recipe

        kwargs.update(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise)
    kwargs.update(extra)
    return MLATransformerConfig(**kwargs)


def _layer(config):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention

    submodules = get_gpt_layer_with_transformer_engine_submodules(
        multi_latent_attention=True, qk_layernorm=True
    ).self_attention.submodules
    module = MLASelfAttention(
        config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal
    )
    return module.cuda().eval()


def _weights(sink: bool = False, fp8: bool = False, flash_version=None):
    """A factory of latent-caching layers that all hold one set of random weights, and the training
    twin holding them too. The weights are copied before any inference forward, which deletes the
    fused KV up-projection."""
    first = _layer(_config(cache=True, sink=sink, fp8=fp8, flash_version=flash_version))
    if sink:
        g = ou.make_generator(seed=ou.DEFAULT_SEED + 1, device="cuda")
        with torch.no_grad():
            offset = first.core_attention.softmax_offset
            offset.copy_(torch.randn(offset.shape, device="cuda", generator=g) * 2 + 1)
    state = {k: v.clone() if torch.is_tensor(v) else v for k, v in first.state_dict().items()}
    del first

    def make_layer():
        layer = _layer(_config(cache=True, sink=sink, fp8=fp8, flash_version=flash_version))
        layer.load_state_dict(state)
        return layer

    twin = _layer(_config(cache=False, sink=sink, fp8=fp8))
    twin.load_state_dict(state)
    return make_layer, twin


def _context(config):
    from megatron.core.inference.config import InferenceConfig
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    return DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=256,
            buffer_size_gb=0.02,
            paused_buffer_size_gb=0.002,
            block_size_tokens=64,  # required with cache_mla_latents
            max_tokens=1024,
            max_requests=40,
            num_cuda_graphs=None,
            use_flashinfer_fused_rope=False,
            unified_memory_level=0,
        ),
    )


def _add_request(context, request_id: int, prompt_len: int):
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

    context.add_request(
        DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=torch.arange(1, prompt_len + 1, device="cpu"),
            sampling_params=SamplingParams(num_tokens_to_generate=8, termination_id=-1),
        )
    )


class _Scenario:
    """Drives a layer through dynamic steps with fixed per-request hidden states."""

    def __init__(self, layer, config, fp8: bool = False, kernel_wrapper=None):
        self.layer, self.config, self.fp8 = layer, config, fp8
        self.context = _context(config)
        self.context.memory_buffer.zero_()
        g = ou.make_generator(seed=ou.DEFAULT_SEED + 7, device="cuda")
        max_len = 256
        self.hidden: Dict[int, torch.Tensor] = {
            rid: torch.randn(max_len, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
            for rid in list(PROMPTS) + [len(PROMPTS)]
        }
        if kernel_wrapper is not None:
            layer.flash_decode_and_prefill = kernel_wrapper(layer.flash_decode_and_prefill)

    def step(self):
        """One step. Returns ({request: (offset, length, rows of linear_proj's input)},
        decode_only)."""
        from megatron.core.fp8_utils import get_fp8_context
        from megatron.core.inference.utils import InferenceMode

        context = self.context
        context.initialize_attention_state()
        hidden = torch.zeros(
            context.padded_active_token_count, 1, HIDDEN, device="cuda", dtype=torch.bfloat16
        )
        spans, start = {}, 0
        for index in range(context.paused_request_count, context.total_request_count):
            rid = int(context.request_ids[index])
            length = int(context.request_query_lengths[index])
            offset = int(context.request_kv_length_offsets[index])
            hidden[start : start + length, 0] = self.hidden[rid][offset : offset + length]
            spans[rid] = (offset, length, start)
            start += length
        captured = []
        hook = self.layer.linear_proj.register_forward_pre_hook(
            lambda mod, args: captured.append(args[0].detach().clone())
        )
        try:
            fp8_context = get_fp8_context(self.config) if self.fp8 else contextlib.nullcontext()
            with torch.no_grad(), InferenceMode.active(), fp8_context:
                self.layer(hidden, attention_mask=None, inference_context=context)
        finally:
            hook.remove()
        out = captured[0].reshape(captured[0].shape[0], -1)
        rows = {rid: (o, n, out[s : s + n]) for rid, (o, n, s) in spans.items()}
        decode_only = context.is_decode_only()
        return rows, decode_only

    def advance(self):
        active = self.context.total_request_count - self.context.paused_request_count
        self.context.update_requests(
            active_requests_mask=torch.ones(active, dtype=torch.int32),
            new_tokens=torch.zeros(active, dtype=torch.int64),
        )


def _training_reference(twin, hidden_prefix: torch.Tensor, rows: slice):
    """(fp64 reference, trusted bf16) for ``rows`` of the twin's forward over a request's prefix."""
    captured = {}
    hooks = [
        twin.core_attention.register_forward_pre_hook(
            lambda mod, args: captured.setdefault("qkv", [a.detach().clone() for a in args[:3]])
        ),
        twin.linear_kv_up_proj.register_forward_pre_hook(
            lambda mod, args: captured.setdefault("latent", args[0].detach().clone())
        ),
        twin.linear_proj.register_forward_pre_hook(
            lambda mod, args: captured.setdefault("out", args[0].detach().clone())
        ),
    ]
    try:
        with torch.no_grad():
            twin(hidden_prefix[:, None, :], attention_mask=None)
    finally:
        for hook in hooks:
            hook.remove()
    query, key, _ = (t[:, 0] for t in captured["qkv"])  # [L, H, QK + ROPE]
    length = query.shape[0]
    latent = captured["latent"][:, 0].double()  # [L, LATENT]
    up = twin.linear_kv_up_proj
    if hasattr(up, "layer_norm_weight"):
        eps = twin.config.layernorm_epsilon
        latent = latent * torch.rsqrt(latent.square().mean(-1, keepdim=True) + eps)
        latent = latent * up.layer_norm_weight.double()
    weight = up.weight.double().view(HEADS, QK + V, LATENT)
    k_nope = torch.einsum("lr,hdr->lhd", latent, weight[:, :QK])
    value = torch.einsum("lr,hdr->lhd", latent, weight[:, QK:])
    key64 = torch.cat([k_nope, key[..., QK:].double()], dim=-1)
    q_rows = query[rows].double()
    allowed = ar.allowed_keys(q_rows.shape[0], length, None, "cuda")
    sink = attention_module.Attention._get_inference_softmax_offset(twin)
    ref, _ = ar.attention_fp64(
        q_rows,
        key64,
        value,
        allowed,
        scale=twin.softmax_scale,
        sink=None if sink is None else sink.double(),
    )
    trusted = captured["out"][rows, 0]
    return ref.reshape(ref.shape[0], -1), trusted.reshape(trusted.shape[0], -1)


def _check_step(twin, scenario, rows, what):
    for rid, (offset, length, got) in rows.items():
        prefix = scenario.hidden[rid][: offset + length]
        ref, trusted = _training_reference(twin, prefix, slice(offset, offset + length))
        ou.assert_close_to_reference(got, ref, trusted, dim=-1, what=f"{what}, request {rid}")


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "sink",
    [
        pytest.param(False, id="vanilla", marks=DEVICES),
        pytest.param(True, id="learnable", marks=DEVICES),
    ],
)
def test_dynamic_steps_match_training(sink):
    """Prefill, decode-only (absorbed), mixed (whole-pool up-projection) and decode-only steps:
    every live token's attention output matches the fp64 MLA reference within the training
    path's own error."""
    if sink and not _te_supports_softmax_type():
        pytest.fail("softmax_type requires Transformer Engine >= 2.8", pytrace=False)
    _require_kernels()
    make_layer, twin = _weights(sink=sink)
    layer = make_layer()
    scenario = _Scenario(layer, layer.config)
    for rid, prompt in PROMPTS.items():
        _add_request(scenario.context, rid, prompt)
    for name, expect_decode_only in (
        ("prefill", False),
        ("decode", True),
        ("mixed", False),
        ("decode after mixed", True),
    ):
        if name == "mixed":
            _add_request(scenario.context, len(PROMPTS), LATE_PROMPT)
        rows, decode_only = scenario.step()
        assert decode_only is expect_decode_only, f"{name}: decode_only={decode_only}"
        _check_step(twin, scenario, rows, f"{name} step (sink={sink})")
        scenario.advance()


def _te_supports_softmax_type() -> bool:
    from megatron.core.utils import is_te_min_version

    return is_te_min_version("2.8.0")


def _poison_padding_rows(value: float):
    """Wrap the attention kernel so that its padding rows (past cu_seqlens_q[-1], which the
    kernel's contract leaves unspecified) hold ``value``."""

    def wrap(kernel):
        def wrapped(q, k, v, max_seqlen_q, max_seqlen_k, cu_seqlens_q, *args, **kwargs):
            out = kernel(q, k, v, max_seqlen_q, max_seqlen_k, cu_seqlens_q, *args, **kwargs)
            out[int(cu_seqlens_q[-1]) :] = value
            return out

        return wrapped

    return wrap


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "precision", [pytest.param(p, marks=ou.requires_sm(ou.HOPPER)) for p in ("bf16", "fp8")]
)
def test_padding_rows_do_not_reach_live_rows(precision):
    """A padded prefill step: live rows are finite and bitwise the same whether the attention
    kernel leaves NaN or zeros in the padding rows. ``[fp8]`` fails on main (R11-F2) and passes
    with #8874; ``[bf16]`` is the control."""
    _require_kernels()
    fp8 = precision == "fp8"
    make_layer, _ = _weights(fp8=fp8)
    results = []
    for value in (0.0, float("nan")):
        layer = make_layer()
        scenario = _Scenario(
            layer, layer.config, fp8=fp8, kernel_wrapper=_poison_padding_rows(value)
        )
        for rid, prompt in PROMPTS.items():
            _add_request(scenario.context, rid, prompt)
        assert scenario.context.padded_active_token_count > sum(PROMPTS.values()), "no padding"
        rows, _ = scenario.step()
        results.append({rid: got for rid, (_, _, got) in rows.items()})
    for rid in PROMPTS:
        assert torch.isfinite(results[1][rid]).all(), f"request {rid}: padding NaN reached it"
        ou.assert_exact(
            results[1][rid], results[0][rid], what=f"request {rid}, NaN vs zero padding"
        )


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "precision, poison",
    [
        pytest.param("bf16", 1e4, marks=ou.requires_sm(ou.HOPPER)),
        pytest.param("bf16", float("nan"), marks=ou.requires_sm(ou.HOPPER)),
        pytest.param("fp8", 1e4, marks=ou.requires_sm(ou.HOPPER)),
    ],
    ids=["bf16-1e4", "bf16-nan", "fp8-1e4"],
)
def test_unused_pages_do_not_reach_live_rows(precision, poison):
    """Prefill and mixed steps up-project the whole latent pool. Live rows are bitwise the same
    whether the pool's unused pages (and the unwritten tails of used ones) hold zeros or poison.
    ``[fp8-1e4]`` fails on main (one FP8 scale over the whole pool, R11-F6) and passes with n6."""
    _require_kernels()
    fp8 = precision == "fp8"
    make_layer, _ = _weights(fp8=fp8)
    results = []
    for fill in (0.0, poison):
        layer = make_layer()
        scenario = _Scenario(layer, layer.config, fp8=fp8)
        scenario.context.memory_buffer.fill_(fill)
        for rid, prompt in PROMPTS.items():
            _add_request(scenario.context, rid, prompt)
        outputs = {}
        rows, _ = scenario.step()  # prefill
        outputs.update({("prefill", rid): got for rid, (_, _, got) in rows.items()})
        scenario.advance()
        _add_request(scenario.context, len(PROMPTS), LATE_PROMPT)
        rows, decode_only = scenario.step()  # mixed
        assert not decode_only
        outputs.update({("mixed", rid): got for rid, (_, _, got) in rows.items()})
        results.append(outputs)
    for key, clean in results[0].items():
        assert torch.isfinite(results[1][key]).all(), f"{key}: poison reached a live row"
        ou.assert_exact(results[1][key], clean, what=f"{key}: poisoned vs clean pool")


@pytest.mark.usefixtures("model_parallel")
@ou.requires_sm(ou.HOPPER)
def test_fa2_is_refused():
    """FA2 cannot read the latent cache's 64-token pages nor value heads narrower than the query
    heads (192 vs 128): a pinned FA2 must be refused with a ValueError (#8584, n2). Fails on main,
    where flash-attn raises a RuntimeError at the first prefill."""
    if attention_module.flash_attn_varlen_func is None:
        pytest.fail("flash-attn 2 is not installed", pytrace=False)
    with pytest.raises(ValueError):
        make_layer, _ = _weights(flash_version=2)
        layer = make_layer()
        scenario = _Scenario(layer, layer.config)
        _add_request(scenario.context, 0, PROMPTS[0])
        scenario.step()


def test_dsa_with_latent_cache_is_rejected():
    """DSA's sparse attention cannot run on the dense latent-cache path, so the config must reject
    ``cache_mla_latents`` with a DSA core (#8570, d5). Fails on main (accepted). CPU only."""
    with pytest.raises(ValueError):
        _config(
            cache=True,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=64,
        )


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_small_scenario_for_sanitizer():
    """A prefill and one absorbed decode step of two short requests (compute-sanitizer case)."""
    _require_kernels()
    make_layer, twin = _weights()
    layer = make_layer()
    scenario = _Scenario(layer, layer.config)
    _add_request(scenario.context, 0, 3)
    _add_request(scenario.context, 1, 65)
    for name in ("prefill", "decode"):
        rows, _ = scenario.step()
        _check_step(twin, scenario, rows, f"sanitizer {name}")
        scenario.advance()

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the MLA dynamic-inference path: latent cache, absorption, whole-pool up-projection.

Call site: ``MultiLatentAttention.forward`` with a ``DynamicInferenceContext``
(megatron/core/transformer/multi_latent_attention.py, ``cache_mla_latents=True``): the cached-latent
QKV path (latent ``[c, k_pe]`` appended to the 64-token page pool), prefill/mixed steps that
up-project the whole pool (``uncompress_kv_from_cache``) and run paged FlashAttention, decode-only
steps that absorb ``W_UK`` into the query, run FlashMLA over the latents and apply ``W_UV`` after.
The same layer serves the MTP draft head: with a repeated MTP layer the context routes its latent
appends and reads to a reserved plane of the pool (``mtp_kv_layer_slot``).

Contract (from the MLA math; the training path is the semantic definition):

* For every live token of every step (prefill, decode-only, mixed), the tensor that reaches
  ``linear_proj`` equals the training forward of the same weights (``cache_mla_latents=False``)
  over that request's whole prefix: ``softmax(scale * (q_nope . k_nope + q_pe . k_pe)) v`` per
  head, bottom-right causal, with ``k_nope = W_UK c``, ``v = W_UV c``, ``c = RMSNorm(kv_latent)``
  (the layer's kv norm, applied once, with its gain), ``q = [q_nope, q_pe]`` split as
  ``[qk_head_dim, qk_pos_emb_head_dim]`` and the layer's sink (``softmax_type`` learnable:
  ``exp(sink_h)`` in the denominator).
* RoPE and scale (DeepSeek-V2/V3's definition). ``q_pe``/``k_pe`` rotate channel pairs
  ``(2i, 2i + 1)`` by ``p * f_i`` at the token's position ``p``. Plain RoPE: ``f_i = base **
  (-2i / d)``, unit amplitude, scale ``1 / sqrt(qk_head_dim + qk_pos_emb_head_dim)``. YaRN with
  factor ``s`` over ``P0`` original positions: ``f_i`` ramps from ``base ** (-2i / d)`` (pairs
  below ``floor(dim(beta_fast))``) to that over ``s`` (above ``ceil(dim(beta_slow))``), ``dim(r) = d
  ln(P0 / (2 pi r)) / (2 ln base)``; cos/sin carry the amplitude ``m(s, mscale) / m(s,
  mscale_all_dim)`` and the scale gains ``m(s, mscale_all_dim) ** 2``, where ``m(s, x) = 0.1 x ln s
  + 1``. No ``softmax_scale`` is configured.
* Tensor parallelism: each rank's ``linear_proj`` input holds its heads, as above. With sequence
  parallelism each rank feeds its sequence shard and the contract is unchanged (the latent cache
  holds every token on every rank).
* Speculative decoding: a decode request runs ``1 + num_speculative_tokens`` query tokens at
  consecutive positions, bottom-right causal over its prefix, as above. MTP draft head (repeated
  layer, draft-KV plane): every draft depth's output for request ``r`` at draft position ``p`` is
  the training forward of the draft layer's weights over that request's draft inputs at positions
  ``0..p`` (the commit pass writes the committed positions, ``base - 1 + depth`` is depth's write
  position, and nothing reads the main plane). The commit pass's own output is discarded by the
  controller and is not part of the contract.
* Chunked prefill: a prompt admitted in chunks gives the same rows as unchunked, whatever the
  chunk boundary is relative to the 64-token page.
* Padding: rows past the live tokens are not part of the contract, but they must not move live
  rows: with an FP8 per-tensor recipe ``linear_proj`` scales its input by an amax over every row,
  so padding rows must reach it as zeros (whatever the attention kernel left there). The check
  reads the layer's output, after ``linear_proj``, where the cross-row scale acts.
* Unused memory: a request's outputs do not depend on pool pages it does not own (free pages,
  other requests' pages, the dummy page), in bf16 and under FP8 per-tensor scaling.
* DSA cannot use the latent cache: its sparse attention needs an indexer cache the dynamic path
  does not have, so ``cache_mla_latents`` with ``experimental_attention_variant='dsa'`` must be
  rejected when the config is built.
* FA2 cannot read 64-token pages nor value heads narrower than query heads: a pinned FA2 must be
  refused with a ValueError, not fail inside flash-attn.

Reference: fp64 MLA attention built from the training path's own bf16 projections (its ``q`` before
RoPE, from ``linear_q_up_proj``, and the latent and ``k_pe`` before the kv norm and RoPE, from
``linear_kv_down_proj``), with the kv norm, the up-projection (``W_UK``/``W_UV`` from the layer's
``linear_kv_up_proj``), RoPE (cos/sin from the definition above, exact angles) and attention in
fp64, at the scale derived from the configuration as above (not read from a layer). Tolerance
class: floating point, bounded by the training forward in bf16 (an independent implementation:
unpaged, unabsorbed, TE attention), per token row. The trusted twin's attention module is built
with the reference scale, so it shares neither the layer's scale derivation (which multiplies
YaRN's factor into any configured scale) nor its kernels; its RoPE (the YaRN embedding module
both paths share) is itself checked against the fp64 rotation, bounded by the same rotation in
bf16 (``rope_lowp``). Padding and unused-memory contracts: exact (metamorphic: poisoned vs clean
runs are bitwise equal). DSA/FA2: the error. Comparisons of a scenario are collected and asserted
after its last step, so a failing rank cannot leave another waiting in a collective.

Variants covered: prompts 63, 64, 65, 129 (around the 64-token page), a prefill step, a
decode-only step (absorbed, FlashMLA), a mixed step (decodes plus a 70-token prefill, whole-pool
up-projection), another decode-only step; vanilla and learnable sink; plain RoPE, YaRN with
DeepSeek-V3's scaling (factor 40, ``mscale = mscale_all_dim = 1``: scale times 1.874, unit
amplitude) and with Megatron's MLA default (``mscale_all_dim = 0``: unit scale factor, amplitude
1.369); TP 2 without and with sequence parallelism (TE spec) and TP 2 with the inference-optimized
layers and sequence parallelism (the inference server's configuration), 16 heads per rank;
speculative decoding with 1 and 2 speculative tokens (FlashMLA with 2 and 3 query rows per
request, and the mixed step's varlen path), without the draft-KV plane and with it: the commit
pass and every draft depth after a prefill step (varlen path over the draft plane), after a
decode-only step (absorbed FlashMLA over the draft plane) and after a mixed step; chunked prefill
of a 129-token prompt split at 63, 64 and 65 tokens (page boundary - 1, at, + 1), the second chunk
in a mixed step and both reread by a decode-only step; bf16 and FP8 tensorwise for the padding and
unused-memory contracts; poison values 1e4 and NaN. DeepSeek-V3's head widths (``qk_head_dim``
128, ``qk_pos_emb_head_dim`` 64, ``v_head_dim`` 128), so the query's content and RoPE parts have
different widths; random RMSNorm gains (the q and kv norms), so a norm applied twice or skipped
changes the result (with unit gains RMSNorm is idempotent).

Hardware: H100 with a FlashMLA build that has its SM90 kernels (#8701; the manifest runs the SM90
build) and FA3 or FA4 for the paged prefill (FA2 cannot read 64-token pages); the TP cases on 2
H100s. The scenario also runs on GB200, where it fails on main (FlashMLA has no SM100 dense
decode, R11-F1; FA4's SM100 forward does not compile in the CI image, G1) and passes with #8873
and #8860.

Open PRs (written against main):
* #8571 (s8): main does not pass the sink to the dynamic MLA kernels, so ``[learnable]`` fails on
  main and passes with s8.
* #8874 (t3-r11f2): main does not zero the MLA attention output's padding rows under FP8 (R11-F2):
  ``test_padding_rows_do_not_reach_live_rows[fp8]`` fails on main and passes with #8874.
* #8702 (n6): main up-projects the whole latent pool through the FP8 linear, so one per-tensor
  scale covers unused pages (R11-F6): ``test_unused_pages_do_not_reach_live_rows[fp8-1e4]`` fails
  on main and passes with n6. ``[bf16-nan]`` also needs the latent tails to be finite or masked by
  the prefill kernel (#8872 zeroes new latent pages). Under sequence parallelism main caches a
  sequence shard of the latent next to the gathered ``k_pe`` and crashes at the first forward:
  ``test_tensor_parallel_steps_match_training[tp2-sp]`` and ``[tp2-sp-inference_optimized]`` fail
  on main and pass with n6.
* #8570 (d5): ``test_dsa_with_latent_cache_is_rejected`` fails on main (accepted) and passes with
  d5. It runs on CPU.
* #8584 (n2): ``test_fa2_is_refused`` fails on main (RuntimeError inside flash-attn) and passes
  with n2.
* #8873 (t3-r11f1): GB200 decode-only steps; see above.
"""

import contextlib
import functools
import math
import subprocess
import sys
from typing import Dict, List, Tuple

import pytest
import torch

from megatron.core.transformer import attention as attention_module
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.attention import attention_reference as ar
from tests.unit_tests.test_utilities import Utils

HEADS, HIDDEN, LATENT, ROPE, QK, V = 16, 512, 512, 64, 128, 128
TP_HEADS = 32  # 16 per rank at TP 2, as DeepSeek-V3's 128 heads at TP 8
ROPE_BASE = 10000.0
MAX_LEN = 256
PROMPTS = {0: 63, 1: 64, 2: 65, 3: 129}
LATE_PROMPT = 70  # joins in the mixed step
DEVICES = [ou.requires_sm(ou.HOPPER, ou.BLACKWELL), pytest.mark.launch_on_gb200]

# The rotary variants (MLATransformerConfig fields). YaRN: DeepSeek-V3's rope_scaling (factor 40
# over 4096 positions, beta 32/1, mscale = mscale_all_dim = 1) and Megatron's MLA default, which
# leaves mscale_all_dim at 0 (the factor moves from the softmax scale into the cos/sin amplitude).
_YARN = dict(
    rope_type="yarn",
    rotary_base=ROPE_BASE,
    rotary_scaling_factor=40.0,
    original_max_position_embeddings=4096,
    beta_fast=32.0,
    beta_slow=1.0,
    mscale=1.0,
)
ROPE_VARIANTS = {
    "rope": dict(rope_type="rope", rotary_base=ROPE_BASE),
    "yarn_deepseek_v3": dict(_YARN, mscale_all_dim=1.0),
    "yarn_mla_default": dict(_YARN, mscale_all_dim=0.0),
}


def _yarn_mscale(scale: float, mscale: float) -> float:
    """DeepSeek-V2/V3's ``yarn_get_mscale``: ``0.1 * mscale * ln(scale) + 1`` for scale > 1."""
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _softmax_scale(variant: str) -> float:
    """MLA's softmax scale from the definition: ``1 / sqrt(qk_head_dim + qk_pos_emb_head_dim)``,
    times ``yarn_get_mscale(factor, mscale_all_dim) ** 2`` under YaRN."""
    params = ROPE_VARIANTS[variant]
    scale = 1.0 / math.sqrt(QK + ROPE)
    if params["rope_type"] == "yarn":
        scale *= _yarn_mscale(params["rotary_scaling_factor"], params["mscale_all_dim"]) ** 2
    return scale


def _rope_tables(positions: torch.Tensor, variant: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp64 cos/sin [N, ROPE / 2] of the angles ``p * f_i`` at ``positions``, times the YaRN
    amplitude, from the definition in the module docstring."""
    params = ROPE_VARIANTS[variant]
    base = params["rotary_base"]
    i = torch.arange(ROPE // 2, dtype=torch.float64, device=positions.device)
    freq = base ** (-2.0 * i / ROPE)
    amplitude = 1.0
    if params["rope_type"] == "yarn":
        scale, original = (
            params["rotary_scaling_factor"],
            params["original_max_position_embeddings"],
        )

        def pair_index(rotations: float) -> float:
            # The pair whose wavelength fits ``rotations`` times into the original context.
            return ROPE * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(pair_index(params["beta_fast"])), 0)
        high = min(math.ceil(pair_index(params["beta_slow"])), ROPE - 1)
        if low == high:
            high += 0.001
        ramp = ((i - low) / (high - low)).clamp(0.0, 1.0)  # 0: extrapolate, 1: interpolate
        freq = freq / scale * ramp + freq * (1.0 - ramp)
        amplitude = _yarn_mscale(scale, params["mscale"]) / _yarn_mscale(
            scale, params["mscale_all_dim"]
        )
    theta = positions.to(torch.float64)[:, None] * freq[None, :]
    return torch.cos(theta) * amplitude, torch.sin(theta) * amplitude


def _mla_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, lowp: bool):
    """MLA's rotation of ``x [L, h, ROPE]``: pair ``(2i, 2i + 1)`` by angle ``i``, returned as
    ``[rotated even channels, rotated odd channels]``, the order in which the layer's RoPE leaves
    them (DeepSeek's de-interleaving; dot products do not depend on it). fp64 unless ``lowp``."""
    if lowp:
        y = ar.rope_lowp(x, cos, sin, interleaved=True)
    else:
        y = ou.fp64_reference(ar.rope_reference, x, cos, sin, interleaved=True)
    return torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)


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


@pytest.fixture
def tensor_parallel():
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if Utils.world_size < 2:
        pytest.fail(f"TP 2 needs 2 GPUs, the world size is {Utils.world_size}", pytrace=False)
    Utils.initialize_model_parallel(2, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    SymmetricMemoryManager.destroy()  # the inference-optimized layers register it on the TP group
    Utils.destroy_model_parallel()


def _config(
    *,
    cache: bool,
    sink: bool = False,
    fp8: bool = False,
    flash_version=None,
    variant: str = "rope",
    heads: int = HEADS,
    tp: int = 1,
    sp: bool = False,
    spec: str = "te",
    mtp: bool = False,
    **extra,
):
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=heads,
        q_lora_rank=128,
        kv_lora_rank=LATENT,
        qk_head_dim=QK,
        qk_pos_emb_head_dim=ROPE,
        v_head_dim=V,
        qk_layernorm=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        apply_rope_fusion=False,
        cache_mla_latents=cache,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        softmax_type="learnable" if sink else "vanilla",
        flash_attention_version=flash_version,
        init_method_std=0.05,
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
        **ROPE_VARIANTS[variant],
    )
    if spec == "inference_optimized":
        kwargs.update(transformer_impl="inference_optimized")
    if mtp:
        # A repeated one-layer MTP head: the context reserves a draft-KV plane for it.
        kwargs.update(mtp_num_layers=1, mtp_use_repeated_layer=True)
    if fp8:
        from megatron.core.enums import Fp8Recipe

        kwargs.update(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise)
    kwargs.update(extra)
    return MLATransformerConfig(**kwargs)


def _layer(config, spec: str = "te", core_scale=None):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_inference_submodules,
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention
    from megatron.core.transformer.spec_utils import build_module

    layer_submodules = (
        get_gpt_layer_with_inference_submodules
        if spec == "inference_optimized"
        else get_gpt_layer_with_transformer_engine_submodules
    )
    submodules = layer_submodules(
        multi_latent_attention=True, qk_layernorm=True
    ).self_attention.submodules
    module = MLASelfAttention(
        config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal
    )
    if core_scale is not None:
        # The trusted twin's attention takes the scale from the definition: the layer multiplies
        # YaRN's mscale**2 into any configured softmax_scale, so a configured value would still
        # move with the derivation under test.
        module.core_attention = build_module(
            submodules.core_attention,
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=core_scale,
            k_channels=QK + ROPE,
            v_channels=V,
            cp_comm_type=None,
            pg_collection=module.pg_collection,
        )
    return module.cuda().eval()


def _weights(
    sink: bool = False,
    fp8: bool = False,
    flash_version=None,
    variant: str = "rope",
    heads: int = HEADS,
    tp: int = 1,
    sp: bool = False,
    spec: str = "te",
    mtp: bool = False,
    seed: int = 0,
):
    """A factory of latent-caching layers that all hold one set of random weights, and the training
    twin holding them too. The weights are copied before any inference forward, which deletes the
    fused KV up-projection. ``seed`` gives another set (the MTP draft layer)."""

    def layer_config():
        return _config(
            cache=True,
            sink=sink,
            fp8=fp8,
            flash_version=flash_version,
            variant=variant,
            heads=heads,
            tp=tp,
            sp=sp,
            spec=spec,
            mtp=mtp,
        )

    first = _layer(layer_config(), spec)
    assert first.config.softmax_scale is None, "the scale must come from the MLA definition"
    # Random RMSNorm gains (the q and kv norms fused into the up-projections): with unit gains a
    # second normalization of already-normalized latents would change nothing.
    g = ou.make_generator(seed=ou.DEFAULT_SEED + 3 + 100 * seed, device="cuda")
    with torch.no_grad():
        gains = [p for name, p in first.named_parameters() if name.endswith("layer_norm_weight")]
        assert len(gains) == 2, "expected the q and kv up-projections' fused RMSNorm gains"
        for gain in gains:
            gain.copy_(torch.rand(gain.shape, device="cuda", generator=g) + 0.5)
        if tp > 1:
            # Replicated weights (the duplicated down-projections) must agree over the TP group:
            # under sequence parallelism a rank caches latents other ranks projected.
            group = first.pg_collection.tp
            for name, p in first.named_parameters():
                if "down_proj" in name:
                    src = torch.distributed.get_global_rank(group, 0)
                    torch.distributed.broadcast(p.data, src=src, group=group)
    if sink:
        g = ou.make_generator(seed=ou.DEFAULT_SEED + 1 + 100 * seed, device="cuda")
        with torch.no_grad():
            offset = first.core_attention.softmax_offset
            offset.copy_(torch.randn(offset.shape, device="cuda", generator=g) * 2 + 1)
    state = {k: v.clone() if torch.is_tensor(v) else v for k, v in first.state_dict().items()}
    del first

    def make_layer():
        layer = _layer(layer_config(), spec)
        layer.load_state_dict(state)
        return layer

    # The trusted training forward: TE spec, no sequence parallelism, scale from the definition.
    twin_config = _config(
        cache=False, sink=sink, fp8=fp8, variant=variant, heads=heads, tp=tp, mtp=mtp
    )
    twin = _layer(twin_config, "te", core_scale=_softmax_scale(variant))
    # The inference-optimized linears subclass TE's, so their state carries every twin key.
    missing, _ = twin.load_state_dict(state, strict=False)
    assert not missing, f"the twin's weights are not all in the layer's state: {missing}"
    return make_layer, twin


def _context(config, **overrides):
    from megatron.core.inference.config import InferenceConfig
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    kwargs = dict(
        max_sequence_length=MAX_LEN,
        buffer_size_gb=0.02,
        paused_buffer_size_gb=0.002,
        block_size_tokens=64,  # required with cache_mla_latents
        max_tokens=1024,
        max_requests=40,
        num_cuda_graphs=None,
        use_flashinfer_fused_rope=False,
        unified_memory_level=0,
    )
    kwargs.update(overrides)
    return DynamicInferenceContext(model_config=config, inference_config=InferenceConfig(**kwargs))


def _request(request_id: int, prompt_len: int):
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

    return DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=torch.arange(1, prompt_len + 1, device="cpu"),
        sampling_params=SamplingParams(num_tokens_to_generate=32, termination_id=-1),
    )


def _add_request(context, request_id: int, prompt_len: int):
    context.add_request(_request(request_id, prompt_len))


def _admit_chunk(context, request, length: int):
    """Admit the next ``length`` prompt tokens of ``request`` the way the engine's chunked-prefill
    scheduler does (``DynamicInferenceEngine.schedule_chunked_prefill``)."""
    if length < request.remaining_prompt_length:
        context.add_request(request, prefill_chunk_length=length)
        context.chunked_prefill_request_id = request.request_id
        request.remaining_prompt_tokens = request.remaining_prompt_tokens[length:]
        request.finished_chunk_token_count += length
    else:
        context.chunked_prefill_request_id = -1
        context.add_request(request)
        request.remaining_prompt_tokens = request.remaining_prompt_tokens.new_empty(0)


def _hidden_states(seed: int) -> Dict[int, torch.Tensor]:
    """Fixed per-request hidden states [MAX_LEN, HIDDEN], the same on every rank."""
    g = ou.make_generator(seed=seed, device="cuda")
    return {
        rid: torch.randn(MAX_LEN, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
        for rid in list(PROMPTS) + [len(PROMPTS)]
    }


def _linear_proj_input(layer, run) -> Tuple[torch.Tensor, Tuple]:
    """``run()``'s input to ``layer.linear_proj``, flattened to [tokens, heads * V], and what
    ``run()`` returned."""
    captured = []
    hook = layer.linear_proj.register_forward_pre_hook(
        lambda mod, args: captured.append(args[0].detach().clone())
    )
    try:
        out = run()
    finally:
        hook.remove()
    return captured[0].reshape(captured[0].shape[0], -1), out


class _Scenario:
    """Drives a layer through dynamic steps with fixed per-request hidden states."""

    def __init__(self, layer, config, fp8: bool = False, kernel_wrapper=None, **inference):
        self.layer, self.config, self.fp8 = layer, config, fp8
        self.context = _context(config, **inference)
        self.context.memory_buffer.zero_()
        self.hidden = _hidden_states(ou.DEFAULT_SEED + 7)
        group = layer.pg_collection.tp
        self.tp_size, self.tp_rank = group.size(), group.rank()
        self.sequence_parallel = config.sequence_parallel and self.tp_size > 1
        if kernel_wrapper is not None:
            layer.flash_decode_and_prefill = kernel_wrapper(layer.flash_decode_and_prefill)

    def _active(self):
        context = self.context
        index = range(context.paused_request_count, context.total_request_count)
        return [
            (
                int(context.request_ids[i]),
                int(context.request_kv_length_offsets[i]),
                int(context.request_query_lengths[i]),
            )
            for i in index
        ]

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
        for rid, offset, length in self._active():
            hidden[start : start + length, 0] = self.hidden[rid][offset : offset + length]
            spans[rid] = (offset, length, start)
            start += length
        if self.sequence_parallel:
            # Each rank feeds its sequence shard; linear_proj's input still holds every token.
            hidden = hidden.chunk(self.tp_size)[self.tp_rank].contiguous()

        def run():
            fp8_context = get_fp8_context(self.config) if self.fp8 else contextlib.nullcontext()
            with torch.no_grad(), InferenceMode.active(), fp8_context:
                return self.layer(hidden, attention_mask=None, inference_context=context)

        out, (layer_out, _) = _linear_proj_input(self.layer, run)
        # The token count this step ran with (set by initialize_attention_state above).
        self.padded_tokens = context.padded_active_token_count
        rows = {rid: (o, n, out[s : s + n]) for rid, (o, n, s) in spans.items()}
        # The layer's output rows (after linear_proj), where cross-row FP8 scaling shows.
        self.output_rows = None
        if not self.sequence_parallel:
            layer_out = layer_out.reshape(layer_out.shape[0], -1)
            self.output_rows = {
                rid: layer_out[s : s + n].clone() for rid, (_, n, s) in spans.items()
            }
        decode_only = context.is_decode_only()
        return rows, decode_only

    def draft_phase(
        self, draft, draft_hidden
    ) -> List[Tuple[str, Dict[int, Tuple[int, int, torch.Tensor]]]]:
        """The MTP draft loop after this step's main forward, driven as the MTP controller drives
        it with every draft accepted (``_compute_serial_mtp_and_sample``): a commit pass writing
        each request's committed positions ``[offset, offset + length - 1)`` from the draft inputs
        (``_mtp_setup_prefill_step``), then one forward per draft depth at ``offset + length - 1 +
        depth`` (``_mtp_begin_decode``/``_mtp_setup_decode_step``). Returns each draft depth's
        rows as {request: (position, 1, row)}; the commit pass's output is discarded, as the
        controller discards it."""
        from megatron.core.inference.utils import InferenceMode

        context = self.context
        active = self._active()
        rids = [rid for rid, _, _ in active]
        n = len(active)
        depths = []

        def forward(hidden):
            def run():
                with torch.no_grad(), InferenceMode.active():
                    return draft(hidden[:, None], attention_mask=None, inference_context=context)

            return _linear_proj_input(draft, run)[0]

        with context._mtp_forward_phase():
            counts = [length - 1 for _, _, length in active]
            total = sum(counts)
            if total:
                rows = context.request_to_kv_block_ids[
                    context.paused_request_count : context.total_request_count
                ]
                context._mtp_setup_prefill_step(
                    append_counts=torch.tensor(counts, dtype=torch.long, device="cuda"),
                    block_table_prefill=rows.to("cuda").to(context.gpu_view.mha_block_table.dtype),
                    padded_token_count=total,
                    padded_request_count=n,
                    request_start_positions=torch.tensor(
                        [offset for _, offset, _ in active], dtype=torch.long, device="cuda"
                    ),
                    total=total,
                )
                forward(
                    torch.cat(
                        [draft_hidden[rid][o : o + c] for (rid, o, _), c in zip(active, counts)]
                    )
                )
                context.mtp_metadata.end_forward()
            first = torch.tensor(
                [offset + length - 1 for _, offset, length in active],
                dtype=torch.int64,
                device="cuda",
            )
            context._mtp_begin_decode(n, n, first, graphed=False)
            for depth in range(context.num_speculative_tokens):
                context._mtp_setup_decode_step()
                positions = [int(p) + depth for p in first.tolist()]
                out = forward(
                    torch.stack([draft_hidden[rid][p] for rid, p in zip(rids, positions)])
                )
                context.mtp_metadata.advance_decode_step()
                depths.append(
                    (
                        f"draft depth {depth}",
                        {
                            rid: (p, 1, out[i : i + 1])
                            for i, (rid, p) in enumerate(zip(rids, positions))
                        },
                    )
                )
        return depths

    def advance(self):
        context = self.context
        active = context.total_request_count - context.paused_request_count
        k = context.num_speculative_tokens
        context.update_requests(
            active_requests_mask=torch.ones(active, dtype=torch.int32),
            new_tokens=torch.zeros(active, dtype=torch.int64),
            new_speculative_tokens=torch.zeros(k, active, dtype=torch.int64) if k else None,
        )


class _Checks:
    """Comparisons collected over a scenario and asserted after its last step, so that a failing
    rank cannot leave another waiting in a collective of a later forward."""

    def __init__(self):
        self.items = []

    def add(self, got, ref, trusted, what):
        self.items.append((got, ref, trusted, what))

    def run(self):
        failures = []
        for got, ref, trusted, what in self.items:
            try:
                ou.assert_close_to_reference(got, ref, trusted, dim=-1, what=what)
            except AssertionError as error:
                failures.append(str(error))
        assert not failures, f"{len(failures)} of {len(self.items)} checks failed:\n" + "\n".join(
            failures
        )


def _output(out):
    """A Megatron linear's output tensor (the wrappers return ``(output, bias)``)."""
    return out[0] if isinstance(out, tuple) else out


def _training_reference(twin, hidden_prefix: torch.Tensor, rows: slice, variant: str, checks, what):
    """(fp64 reference, trusted bf16) for ``rows`` of the twin's forward over a request's prefix.
    Also queues the check of the twin's own RoPE against the fp64 rotation."""
    captured = {}

    def record(name, value):
        # A hook that returns a value replaces the module's arguments or output: return None.
        captured.setdefault(name, value)

    hooks = [
        twin.linear_q_up_proj.register_forward_hook(
            lambda mod, args, out: record("q", _output(out).detach().clone())
        ),
        twin.linear_kv_down_proj.register_forward_hook(
            lambda mod, args, out: record("kv_down", _output(out).detach().clone())
        ),
        twin.core_attention.register_forward_pre_hook(
            lambda mod, args: record("qk", [a.detach().clone() for a in args[:2]])
        ),
        twin.linear_proj.register_forward_pre_hook(
            lambda mod, args: record("out", args[0].detach().clone())
        ),
    ]
    try:
        with torch.no_grad():
            twin(hidden_prefix[:, None, :], attention_mask=None)
    finally:
        for hook in hooks:
            hook.remove()
    heads = twin.num_attention_heads_per_partition
    length = hidden_prefix.shape[0]
    query = captured["q"][:, 0].reshape(length, heads, QK + ROPE)  # before RoPE
    # [L, LATENT + ROPE]: the latent before the kv norm, and k_pe before RoPE.
    kv_down = captured["kv_down"][:, 0]
    k_pe = kv_down[:, None, LATENT:]
    cos, sin = _rope_tables(torch.arange(length, device=query.device), variant)
    cos, sin = cos[:, None, :], sin[:, None, :]
    q_pe64 = _mla_rope(query[..., QK:], cos, sin, lowp=False)
    k_pe64 = _mla_rope(k_pe, cos, sin, lowp=False)

    # The twin's RoPE (YaRN embedding module and rotation, which the dynamic path shares) against
    # the definition, bounded by the same rotation with torch ops in bf16.
    q_rot, k_rot = (t[:, 0] for t in captured["qk"])  # [L, heads, QK + ROPE]
    checks.add(
        q_rot[..., QK:].reshape(length, -1),
        q_pe64.reshape(length, -1),
        _mla_rope(query[..., QK:], cos, sin, lowp=True).reshape(length, -1),
        f"{what}: the trusted twin's RoPE'd q_pe ({variant})",
    )
    checks.add(
        k_rot[:, :1, QK:].reshape(length, -1),
        k_pe64.reshape(length, -1),
        _mla_rope(k_pe, cos, sin, lowp=True).reshape(length, -1),
        f"{what}: the trusted twin's RoPE'd k_pe ({variant})",
    )

    latent = kv_down[:, :LATENT].double()
    up = twin.linear_kv_up_proj
    if hasattr(up, "layer_norm_weight"):
        eps = twin.config.layernorm_epsilon
        latent = latent * torch.rsqrt(latent.square().mean(-1, keepdim=True) + eps)
        latent = latent * up.layer_norm_weight.double()
    weight = up.weight.double().view(heads, QK + V, LATENT)
    k_nope = torch.einsum("lr,hdr->lhd", latent, weight[:, :QK])
    value = torch.einsum("lr,hdr->lhd", latent, weight[:, QK:])
    key64 = torch.cat([k_nope, k_pe64.expand(length, heads, ROPE)], dim=-1)
    q64 = torch.cat([query[..., :QK].double(), q_pe64], dim=-1)[rows]
    allowed = ar.allowed_keys(q64.shape[0], length, None, "cuda")
    sink = attention_module.Attention._get_inference_softmax_offset(twin)
    ref, _ = ar.attention_fp64(
        q64,
        key64,
        value,
        allowed,
        scale=_softmax_scale(variant),
        sink=None if sink is None else sink.double(),
    )
    trusted = captured["out"][rows, 0]
    return ref.reshape(ref.shape[0], -1), trusted.reshape(trusted.shape[0], -1)


def _check_rows(checks, twin, hidden, rows, what, variant="rope"):
    """Queue each request's rows against the twin over its prefix (rows ``[offset, offset + n)``)."""
    for rid, (offset, length, got) in rows.items():
        prefix = hidden[rid][: offset + length]
        label = f"{what}, request {rid}"
        ref, trusted = _training_reference(
            twin, prefix, slice(offset, offset + length), variant, checks, label
        )
        checks.add(got, ref, trusted, label)


def _run_scenario(layer, twin, variant: str, what: str, **inference):
    """Prefill, decode-only (absorbed), mixed (whole-pool up-projection) and decode-only steps,
    every live row queued against the training reference."""
    scenario = _Scenario(layer, layer.config, **inference)
    checks = _Checks()
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
        _check_rows(checks, twin, scenario.hidden, rows, f"{name} step ({what})", variant)
        scenario.advance()
    checks.run()


SCENARIO_CASES = [
    pytest.param(dict(sink=False, variant="rope"), id="vanilla", marks=DEVICES),
    pytest.param(dict(sink=True, variant="rope"), id="learnable", marks=DEVICES),
    pytest.param(
        dict(sink=False, variant="yarn_deepseek_v3"), id="yarn-deepseek_v3", marks=DEVICES
    ),
    pytest.param(
        dict(sink=False, variant="yarn_mla_default"), id="yarn-mla_default", marks=DEVICES
    ),
]


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("case", SCENARIO_CASES)
def test_dynamic_steps_match_training(case):
    """Prefill, decode-only (absorbed), mixed (whole-pool up-projection) and decode-only steps:
    every live token's attention output matches the fp64 MLA reference within the training
    path's own error. The YaRN cases take a softmax scale and a cos/sin amplitude other than 1
    (``mscale_all_dim`` 1 and 0), derived from the definition, not from the layer."""
    if case["sink"] and not _te_supports_softmax_type():
        pytest.fail("softmax_type requires Transformer Engine >= 2.8", pytrace=False)
    _require_kernels()
    make_layer, twin = _weights(sink=case["sink"], variant=case["variant"])
    _run_scenario(make_layer(), twin, case["variant"], f"sink={case['sink']}, {case['variant']}")


TP_CASES = [
    pytest.param(dict(sp=False, spec="te"), id="tp2", marks=DEVICES),
    pytest.param(dict(sp=True, spec="te"), id="tp2-sp", marks=DEVICES),
    pytest.param(
        dict(sp=True, spec="inference_optimized"), id="tp2-sp-inference_optimized", marks=DEVICES
    ),
]


@pytest.mark.usefixtures("tensor_parallel")
@pytest.mark.parametrize("case", TP_CASES)
def test_tensor_parallel_steps_match_training(case):
    """The dynamic steps at TP 2 (16 heads per rank), with and without sequence parallelism: each
    rank's attention output (its heads, every token) matches the fp64 reference built from the
    TP 2 training twin (TE spec, no sequence parallelism) within that twin's error.
    ``[tp2-sp]`` and ``[tp2-sp-inference_optimized]`` fail on main (n6: the cached latent is a
    sequence shard, the gathered ``k_pe`` is not; the concatenation raises) and pass with n6."""
    _require_kernels()
    make_layer, twin = _weights(tp=2, sp=case["sp"], heads=TP_HEADS, spec=case["spec"])
    _run_scenario(make_layer(), twin, "rope", f"TP 2, {case}")


@pytest.mark.usefixtures("tensor_parallel")
@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_tensor_parallel_small_scenario_for_sanitizer():
    """A TP 2 prefill and one absorbed decode step of two short requests (compute-sanitizer
    case for the 2-GPU entry)."""
    _require_kernels()
    make_layer, twin = _weights(tp=2, heads=TP_HEADS)
    layer = make_layer()
    scenario = _Scenario(layer, layer.config)
    _add_request(scenario.context, 0, 3)
    _add_request(scenario.context, 1, 65)
    checks = _Checks()
    for name in ("prefill", "decode"):
        rows, _ = scenario.step()
        _check_rows(checks, twin, scenario.hidden, rows, f"TP 2 sanitizer {name}")
        scenario.advance()
    checks.run()


MTP_CASES = [
    pytest.param(dict(k=1, draft_kv=False), id="spec1", marks=DEVICES),
    pytest.param(dict(k=1, draft_kv=True), id="spec1-draft_kv", marks=DEVICES),
    pytest.param(dict(k=2, draft_kv=True), id="spec2-draft_kv", marks=DEVICES),
]


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("case", MTP_CASES)
def test_speculative_steps_match_training(case):
    """Speculative decoding with ``k`` speculative tokens: prefill, verification (decode-only,
    ``k + 1`` query rows per request through FlashMLA), mixed (decodes of ``k + 1`` rows and a
    prefill, varlen path) and verification steps match the reference. With the draft-KV plane
    (a repeated MTP layer), the draft layer (its own weights) runs the commit pass and ``k``
    draft depths after every main step, and each depth's row at draft position ``p`` matches the
    draft layer's training forward over that request's draft inputs ``0..p``: after a prefill or
    mixed step the depths take the varlen path over the draft plane, after a verification step
    the absorbed FlashMLA path."""
    _require_kernels()
    k, draft_kv = case["k"], case["draft_kv"]
    make_layer, twin = _weights(mtp=draft_kv)
    layer = make_layer()
    scenario = _Scenario(layer, layer.config, num_speculative_tokens=k, buffer_size_gb=0.04)
    context = scenario.context
    assert context.enable_mtp_kv_cache is draft_kv, "the draft-KV plane was not set up as asked"
    if draft_kv:
        make_draft, draft_twin = _weights(mtp=True, seed=1)
        draft = make_draft()
        draft_hidden = _hidden_states(ou.DEFAULT_SEED + 11)
    checks = _Checks()
    for rid, prompt in PROMPTS.items():
        _add_request(context, rid, prompt)
    for name, expect_decode_only in (
        ("prefill", False),
        ("verification", True),
        ("mixed", False),
        ("verification after mixed", True),
    ):
        if name == "mixed":
            _add_request(context, len(PROMPTS), LATE_PROMPT)
        rows, decode_only = scenario.step()
        assert decode_only is expect_decode_only, f"{name}: decode_only={decode_only}"
        if decode_only:
            lengths = {rid: n for rid, (_, n, _) in rows.items()}
            assert set(lengths.values()) == {k + 1}, f"{name}: query rows per request {lengths}"
        what = f"{name} step (k={k}, draft_kv={draft_kv})"
        _check_rows(checks, twin, scenario.hidden, rows, what)
        if draft_kv:
            for label, draft_rows in scenario.draft_phase(draft, draft_hidden):
                _check_rows(checks, draft_twin, draft_hidden, draft_rows, f"{what}, {label}")
        scenario.advance()
    checks.run()


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "split", [pytest.param(s, id=f"split={s}", marks=DEVICES) for s in (63, 64, 65)]
)
def test_chunked_prefill_matches_training(split):
    """A 129-token prompt admitted in two chunks split one token before, at and after the
    64-token page boundary, next to a 40-token prompt: the first chunk (prefill step), the second
    chunk (mixed step: it reads the first chunk's latents through the whole-pool up-projection)
    and a decode-only step (absorbed, over both chunks' latents) match the unchunked reference."""
    _require_kernels()
    make_layer, twin = _weights()
    layer = make_layer()
    scenario = _Scenario(layer, layer.config, enable_chunked_prefill=True)
    context = scenario.context
    prompt = 129
    _add_request(context, 0, 40)
    chunked = _request(1, prompt)
    _admit_chunk(context, chunked, split)
    checks = _Checks()
    for name, expect_decode_only, span in (
        ("first chunk", False, (0, split)),
        ("last chunk", False, (split, prompt - split)),
        ("decode", True, (prompt, 1)),
    ):
        if name == "last chunk":
            _admit_chunk(context, chunked, chunked.remaining_prompt_length)
        rows, decode_only = scenario.step()
        assert decode_only is expect_decode_only, f"{name}: decode_only={decode_only}"
        assert rows[1][:2] == span, f"{name}: chunked request at {rows[1][:2]}, expected {span}"
        _check_rows(checks, twin, scenario.hidden, rows, f"{name} step (split {split})")
        scenario.advance()
    checks.run()


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
    """A padded prefill step: the layer's live output rows (after ``linear_proj``) are finite and
    bitwise the same whether the attention kernel leaves zeros or garbage in the padding rows.

    The garbage is what a per-tensor FP8 scale would see in stale memory: a large finite value
    and an infinity for ``[fp8]`` (an amax reduction built on fmax skips NaN, so NaN alone may not
    move it), NaN and a large value for the ``[bf16]`` control, whose linear_proj is row-local.
    ``[fp8]`` fails on main (R11-F2: ``linear_proj``'s input amax covers the padding rows) and
    passes with #8874, which zeroes them first."""
    _require_kernels()
    fp8 = precision == "fp8"
    poisons = (1e4, float("inf")) if fp8 else (float("nan"), 1e4)
    make_layer, _ = _weights(fp8=fp8)
    results = []
    for value in (0.0, *poisons):
        layer = make_layer()
        scenario = _Scenario(
            layer, layer.config, fp8=fp8, kernel_wrapper=_poison_padding_rows(value)
        )
        for rid, prompt in PROMPTS.items():
            _add_request(scenario.context, rid, prompt)
        scenario.step()
        assert scenario.padded_tokens > sum(
            PROMPTS.values()
        ), f"no padding rows: {scenario.padded_tokens} tokens for {sum(PROMPTS.values())} live"
        results.append(scenario.output_rows)
    clean = results[0]
    for value, poisoned in zip(poisons, results[1:]):
        for rid in PROMPTS:
            assert torch.isfinite(
                poisoned[rid]
            ).all(), f"request {rid}: padding rows holding {value} made live outputs non-finite"
            ou.assert_exact(
                poisoned[rid],
                clean[rid],
                what=f"request {rid}: layer output with padding rows holding {value} vs zero",
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
    checks = _Checks()
    for name in ("prefill", "decode"):
        rows, _ = scenario.step()
        _check_rows(checks, twin, scenario.hidden, rows, f"sanitizer {name}")
        scenario.advance()
    checks.run()

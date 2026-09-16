# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the single-GPU inference Triton kernels.

Inference kernels do not feed the training loss, but RL rollouts and batch-invariant
inference depend on them replaying bit-exactly. Covered here: KV-cache tensor ops and the
fused KV append, the inference MoE activations and MXFP8 quantisation, the inference MoE
permute/unpermute (batch-invariant path bit-exact; the atomic default path is the negative
control), the vLLM-derived fused MoE GEMM and top-k sum, the batch-invariant GEMM /
log-softmax / mean kernels, and the CUDA-graph routing-map padding mask. Multi-rank NVLS
symmetric-memory collectives are exempted in the manifest (they need NVLink peers).
"""

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    count_differing_replays,
    seeded,
)

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs a GPU and Triton"
)


def _dev_scalar(value, dtype=torch.int32):
    return torch.tensor([value], dtype=dtype, device="cuda")


# --- KV-cache tensor ops ---------------------------------------------------------------------


def test_tensor_ops_replay():
    from megatron.core.inference.contexts.attention_context.triton import tensor_ops

    seeded()
    src = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)
    pos = _dev_scalar(1533)

    def get_slice(src):
        out = torch.zeros_like(src)
        tensor_ops.tensor_get_slice_after(src, out, pos)
        return out

    assert_replays_bit_exact(get_slice, (src,), backward=False, what="tensor_get_slice_after")

    a = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)

    def merge(a, b):
        out = torch.zeros_like(a)
        tensor_ops.tensor_merge(a, b, pos, output_tensor=out)
        return out

    assert_replays_bit_exact(merge, (a, b), backward=False, what="tensor_merge")

    states = torch.randn(512, 8, 64, 128, device="cuda")
    idx = torch.randperm(512, device="cuda").to(torch.int64)
    idx[torch.rand(512, device="cuda") < 0.25] = -1
    new_states = torch.randn(512, 8, 64, 128, device="cuda")

    def masked_update(states, new_states):
        tensor_ops.tensor_masked_update(states, idx, new_states)
        return states

    assert_replays_bit_exact(
        masked_update, (states, new_states), backward=False, what="tensor_masked_update"
    )


def test_fused_kv_append_replays():
    from megatron.core.inference.contexts.fused_kv_append_kernel import (
        triton_append_key_value_cache,
    )

    seeded()
    n_tokens, heads, dim, block, blocks, layers = 8192, 8, 128, 64, 2048, 2
    key = torch.randn(n_tokens, 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(n_tokens, 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(blocks * block, device="cuda")[:n_tokens]
    token_to_block = (slots // block).to(torch.int32)
    token_to_pos = (slots % block).to(torch.int32)
    dummy_block = blocks - 1
    token_to_block[torch.rand(n_tokens, device="cuda") < 0.1] = dummy_block
    memory = torch.zeros(
        2, layers + 1, blocks, block, heads, dim, device="cuda", dtype=torch.bfloat16
    )

    def fn(key, value, memory):
        triton_append_key_value_cache(
            1, key, value, memory, n_tokens, token_to_block, token_to_pos, dummy_block
        )
        return memory

    assert_replays_bit_exact(
        fn, (key, value, memory), backward=False, what="triton_append_key_value_cache"
    )


# --- inference MoE activations / quantisation ------------------------------------------------


def _padded_layout(rows, ffn, n_used):
    perm = torch.arange(rows, device="cuda", dtype=torch.int32)
    perm[torch.rand(rows, device="cuda") < 0.1] = -1
    return perm, _dev_scalar(n_used)


def test_inference_moe_activations_replay():
    from megatron.core.inference.moe import activations

    seeded()
    rows, ffn, n_used = 8192, 2688, 6000
    perm, n_used_t = _padded_layout(rows, ffn, n_used)
    x2 = torch.randn(rows, 2 * ffn, device="cuda", dtype=torch.bfloat16)
    x1 = torch.randn(rows, ffn, device="cuda", dtype=torch.bfloat16) * 5

    def live(out):
        # Rows beyond n_used / padding rows are left uninitialised by design.
        return out[:n_used][perm[:n_used] >= 0]

    assert_replays_bit_exact(
        lambda x: live(activations.padded_squared_relu(x, perm, n_used_t, clamp_scale=10.0)),
        (x1,),
        backward=False,
        what="padded_squared_relu",
    )
    assert_replays_bit_exact(
        lambda x: live(activations.padded_swiglu(x, perm, n_used_t)),
        (x2,),
        backward=False,
        what="padded_swiglu",
    )
    assert_replays_bit_exact(
        lambda x: activations.bounded_silu_mul(x, n_used_t)[:n_used],
        (x2,),
        backward=False,
        what="bounded_silu_mul",
    )
    if hasattr(torch, "float8_e8m0fnu"):

        def quant(x):
            q = activations.squared_relu_and_quantize_mxfp8(x, perm, n_used_t)
            return live(q.data.view(torch.uint8)), q.scale.view(torch.uint8)

        assert_replays_bit_exact(
            quant, (x1,), backward=False, what="squared_relu_and_quantize_mxfp8"
        )


def test_batch_invariant_inference_activations_replay():
    from megatron.core.inference.moe import batch_invariant

    seeded()
    rows, ffn, n_used = 16384, 5120, 12000
    perm, n_used_t = _padded_layout(rows, ffn, n_used)
    probs = torch.rand(rows, device="cuda")
    x2 = torch.randn(rows, 2 * ffn, device="cuda", dtype=torch.bfloat16)
    x1 = torch.randn(rows, ffn, device="cuda", dtype=torch.bfloat16) * 5

    def live(out):
        return out[:n_used][perm[:n_used] >= 0]

    assert_replays_bit_exact(
        lambda x: live(batch_invariant.swiglu_with_probs(x, perm, n_used_t, probs)),
        (x2,),
        backward=False,
        what="swiglu_with_probs",
    )
    assert_replays_bit_exact(
        lambda x: live(
            batch_invariant.squared_relu_with_probs(x, perm, n_used_t, probs, clamp_scale=10.0)
        ),
        (x1,),
        backward=False,
        what="squared_relu_with_probs",
    )
    bound_elems = _dev_scalar(n_used * ffn)
    assert_replays_bit_exact(
        lambda x: batch_invariant.weighted_silu_mul_bounded(x, probs, bound_elems)[:n_used],
        (x2,),
        backward=False,
        what="weighted_silu_mul_bounded",
    )


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="needs torch float8 e8m0 dtype")
def test_mxfp8_quantize_replays():
    from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

    seeded()
    x = torch.randn(16384, 8192, device="cuda", dtype=torch.bfloat16)

    def fn(x):
        data, scale = mxfp8_quantize(x)
        return data.view(torch.uint8), scale.view(torch.uint8)

    assert_replays_bit_exact(fn, (x,), backward=False, what="mxfp8_quantize")


# --- inference MoE permute / unpermute --------------------------------------------------------


def _permute_case(num_tokens=8192, hidden=4096, topk=8, num_local=8):
    hidden_states = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    routing_map = torch.randint(0, 32, (num_tokens, topk), device="cuda")
    probs = torch.rand(num_tokens, topk, device="cuda")
    valid = _dev_scalar(num_tokens)
    return hidden_states, probs, routing_map, valid, num_local


_INT_VIEW = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def _bit_pattern(t):
    """Unsigned bit pattern of every element of a float tensor, as int64."""
    bits = t.contiguous().view(_INT_VIEW[t.dtype]).to(torch.int64)
    return bits & ((1 << (8 * t.element_size())) - 1)


def _row_keys(tokens, prob_bits):
    """Canonical sorted keys ``token << 32 | bits(prob)`` for a set of permuted rows."""
    return ((tokens.to(torch.int64) << 32) | prob_bits).sort().values


def _expected_rows_per_expert(probs, routing_map, num_local, prob_dtype):
    """For each local expert: the multiset of (token, probability) rows it must receive."""
    prob_bits = _bit_pattern(probs.to(prob_dtype))
    expected = []
    for expert in range(num_local):
        tokens, slots = (routing_map == expert).nonzero(as_tuple=True)
        expected.append(_row_keys(tokens, prob_bits[tokens, slots]))
    return expected


def test_inference_permute_row_order_is_a_permutation_per_expert():
    """The permute kernel claims rows with ``tl.atomic_add``: row order inside an expert block
    is scheduling-dependent, but the per-expert multiset of (token, probability) rows, the
    expert offsets and the gathered hidden states must not change -- and must match what
    ``routing_map`` / ``probs`` prescribe, so a corrupted probability is caught even though
    the rows may legitimately be reordered."""
    from megatron.core.inference.moe.permute import permute_tokens

    seeded()
    hidden_states, probs, routing_map, valid, num_local = _permute_case()
    runs = []
    for _ in range(3):
        perm_h, perm_p, perm_map, offs = permute_tokens(
            hidden_states, probs, routing_map, 0, num_local, valid
        )
        # Rows beyond the last expert offset are never written (the consumers skip them).
        n_used = int(offs[-1].item())
        runs.append(
            (
                perm_h[:n_used].clone(),
                perm_p[:n_used].clone(),
                perm_map[:n_used].clone(),
                offs.clone(),
            )
        )

    expected_offs = torch.stack([(routing_map == e).sum() for e in range(num_local)]).cumsum(0)
    expected_rows = _expected_rows_per_expert(probs, routing_map, num_local, runs[0][1].dtype)
    for perm_h, perm_p, perm_map, offs in runs:
        assert torch.equal(offs.to(torch.int64), expected_offs.to(offs.device))
        start = 0
        for expert, end in enumerate(offs.tolist()):
            live = perm_map[start:end] >= 0  # padding rows (alignment > 1) carry no token
            actual = _row_keys(perm_map[start:end][live], _bit_pattern(perm_p[start:end][live]))
            assert torch.equal(
                actual, expected_rows[expert]
            ), f"expert {expert}: (token, probability) rows differ from routing_map/probs"
            start = end
        live = perm_map >= 0
        assert torch.equal(perm_h[live], hidden_states[perm_map[live].long()])


def _unpermute_case():
    from megatron.core.inference.moe import permute

    seeded()
    num_tokens, hidden, topk, num_local = 8192, 4096, 8, 8
    hidden_states, probs, routing_map, valid, _ = _permute_case(num_tokens, hidden, topk, num_local)
    perm_h, perm_p, perm_map, offs, inverse_map = permute.permute_tokens(
        hidden_states,
        probs,
        routing_map,
        0,
        num_local,
        valid,
        return_batch_invariant_inverse_map=True,
    )
    n_used = offs[-1:].to(torch.int32)
    expert_output = torch.randn_like(perm_h)
    return expert_output, perm_p, perm_map, inverse_map, n_used, valid, num_tokens


def test_inference_unpermute_batch_invariant_path_replays():
    from megatron.core.inference.moe import batch_invariant

    expert_output, _, _, inverse_map, _, valid, _ = _unpermute_case()

    def bi(expert_output):
        return batch_invariant.unpermute_tokens_in_expert_order(
            expert_output, inverse_map, valid, None
        )

    assert_replays_bit_exact(
        bi, (expert_output,), replays=4, backward=False, what="unpermute_tokens_in_expert_order"
    )


@pytest.mark.xfail(
    strict=False,
    reason="Negative control: the default combine accumulates with fp32 tl.atomic_add, whose "
    "order is hardware dependent (it raced on GB300); recorded, not gated.",
)
def test_inference_default_unpermute_is_the_racy_path():
    from megatron.core.inference.moe import permute

    expert_output, perm_p, perm_map, _, n_used, valid, num_tokens = _unpermute_case()

    def default(expert_output):
        return permute.unpermute_tokens(expert_output, perm_p, perm_map, num_tokens, n_used, valid)

    differing = count_differing_replays(default, (expert_output,), replays=8, backward=False)
    assert differing > 0, (
        "the fp32 tl.atomic_add combine replayed bit-exactly 7 times; if upstream made it "
        "deterministic, drop this control and register the kernel as tested"
    )


# --- vLLM-derived fused MoE ---------------------------------------------------------------


def test_vllm_fused_moe_and_moe_sum_replay():
    from megatron.core.inference.moe.fused_moe import ActivationType
    from megatron.core.inference.moe.vllm_fused_moe import _moe_sum, vllm_fused_moe

    seeded()
    max_tokens, hidden, ffn, topk, experts = 4096, 4096, 2048, 8, 32
    hidden_states = torch.randn(max_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(max_tokens, topk, device="cuda")
    routing_map = torch.randint(0, experts, (max_tokens, topk), device="cuda")
    fc1 = torch.randn(experts, ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    fc2 = torch.randn(experts, hidden, ffn, device="cuda", dtype=torch.bfloat16) * 0.02
    valid = _dev_scalar(max_tokens - 100)

    def fn(h, p):
        out = vllm_fused_moe(
            h, p, fc1, fc2, ActivationType.SQUARED_RELU, experts, 0, valid, routing_map
        )
        return out[: max_tokens - 100]

    assert_replays_bit_exact(
        fn, (hidden_states, probs), replays=4, backward=False, what="vllm_fused_moe"
    )

    inp = torch.randn(max_tokens * topk, 2688, device="cuda", dtype=torch.bfloat16)

    def moe_sum(inp, p):
        out = _moe_sum(inp, p, max_tokens, topk, 2688, valid, routing_map, 0, experts)
        return out[: max_tokens - 100]

    assert_replays_bit_exact(moe_sum, (inp, probs), replays=4, backward=False, what="_moe_sum")


# --- batch-invariant kernels -----------------------------------------------------------------


def test_batch_invariant_kernels_replay():
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    seeded()
    a = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    assert_replays_bit_exact(
        lambda a, b: bik.matmul_persistent(a, b, bias),
        (a, b),
        replays=4,
        backward=False,
        what="matmul_persistent",
    )
    logits = torch.randn(8192, 32768, device="cuda")
    assert_replays_bit_exact(
        lambda x: bik.log_softmax(x, dim=-1),
        (logits,),
        replays=4,
        backward=False,
        what="bik log_softmax",
    )
    x = torch.randn(8192, 4096, device="cuda")
    assert_replays_bit_exact(
        lambda x: bik.mean_dim(x, 1), (x,), replays=4, backward=False, what="bik mean_dim"
    )


def test_mask_routing_padding_replays():
    from megatron.core.transformer.moe.inference_routing_mask_kernel import mask_routing_padding

    seeded()
    routing_map = torch.randint(0, 64, (8192, 8), device="cuda", dtype=torch.int64)
    real = _dev_scalar(37 + 8192)

    def fn(routing_map):
        mask_routing_padding(routing_map, real, tp_rank=1)
        return routing_map

    assert_replays_bit_exact(fn, (routing_map,), backward=False, what="mask_routing_padding")

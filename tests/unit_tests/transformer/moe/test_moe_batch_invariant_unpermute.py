# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise tests: fused batch-invariant MoE unpermute against the reference loop, and the
training combine against the inference combine it reproduces."""

import pytest
import torch

from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe import batch_invariant
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection, permute
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _adversarial(shape, dtype, seed):
    """Values over a wide exponent range plus signed zeros."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mantissa = torch.randn(shape, device="cuda", generator=gen)
    exponent = torch.randint(-40, 40, shape, device="cuda", generator=gen).float()
    values = (mantissa * torch.exp2(exponent)).to(dtype)
    values.view(-1)[::7] = -0.0
    values.view(-1)[3::11] = 0.0
    return values


def _routed(num_tokens, hidden, num_experts, topk, dtype, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=gen)
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, scores.topk(topk, dim=1).indices, True)
    tokens = torch.zeros(num_tokens, hidden, device="cuda", dtype=dtype)
    _, _, _, inverse_map, _ = permute(
        tokens,
        routing_map,
        num_out_tokens=num_tokens * topk,
        return_batch_invariant_inverse_map=True,
    )
    rows = _adversarial((num_tokens * topk, hidden), dtype, seed + 1)
    return rows, inverse_map


@pytest.mark.parametrize("collective", ["ordered", "multimem"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "num_tokens,hidden,num_experts,topk,ep_size",
    [
        (1, 8, 4, 1, 1),
        (37, 100, 8, 2, 2),
        (300, 520, 16, 4, 4),
        (1000, 2688, 128, 6, 8),
        (513, 1030, 64, 8, 8),
    ],
)
def test_fused_unpermute_matches_loop_bitwise(
    monkeypatch, collective, dtype, num_tokens, hidden, num_experts, topk, ep_size
):
    monkeypatch.setattr(batch_invariant, "get_batch_invariant_collective", lambda: collective)
    rows, inverse_map = _routed(num_tokens, hidden, num_experts, topk, dtype, seed=num_tokens)
    restore_shape = torch.Size((num_tokens, hidden))
    grad = _adversarial((num_tokens, hidden), dtype, seed=num_tokens + 2)

    fused_rows = rows.clone().requires_grad_()
    fused = batch_invariant.unpermute(
        fused_rows,
        restore_shape,
        probs=None,
        num_experts=num_experts,
        inverse_map=inverse_map,
        ep_size=ep_size,
    )
    fused.backward(grad)

    loop_rows = rows.clone().requires_grad_()
    loop = batch_invariant._unpermute_loop(
        loop_rows, restore_shape, None, num_experts, inverse_map, ep_size
    )
    loop.backward(grad)

    assert fused.dtype == loop.dtype and fused.shape == loop.shape
    assert torch.equal(_bits(fused), _bits(loop))
    assert torch.equal(_bits(fused_rows.grad), _bits(loop_rows.grad))


def test_fused_unpermute_rank_tree_rounding():
    """Rank partials round to fp32 before the cross-rank sum, so 1e20 + 1 - 1e20 + 1 is 0."""
    rows = torch.tensor([[1e20], [1.0], [-1e20], [1.0]], device="cuda").expand(4, 4).contiguous()
    inverse_map = torch.tensor([[[0, 1, 2, 3]], [[0, 1, 2, 3]]], device="cuda")
    out = batch_invariant.unpermute(
        rows, torch.Size((1, 4)), probs=None, num_experts=4, inverse_map=inverse_map, ep_size=2
    )
    assert torch.equal(out, torch.zeros(1, 4, device="cuda"))


def test_unpermute_with_probs_uses_loop():
    num_tokens, hidden, num_experts, topk = 64, 32, 8, 2
    rows, inverse_map = _routed(num_tokens, hidden, num_experts, topk, torch.bfloat16, seed=5)
    probs = torch.rand(num_tokens, num_experts, device="cuda")
    restore_shape = torch.Size((num_tokens, hidden))
    out = batch_invariant.unpermute(
        rows,
        restore_shape,
        probs=probs,
        num_experts=num_experts,
        inverse_map=inverse_map,
        ep_size=2,
    )
    ref = batch_invariant._unpermute_loop(rows, restore_shape, probs, num_experts, inverse_map, 2)
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_unpermute_backward_is_cuda_graph_capturable(dtype):
    """The fused forward and backward capture into one CUDA graph and replay bitwise.

    The engine's AllToAll combine takes this ``probs=None`` path inside captured graphs, so
    no step may depend on the inverse map's contents (no boolean indexing). The map holds
    an invalid slot and leaves rows 3 and 5 unused; their gradients must stay zero. The
    upstream gradient is non-contiguous and carries signed zeros.
    """
    num_experts, hidden = 4, 40
    # Tokens x top-2 slots; slot (1, 1) is invalid, rows 3 and 5 belong to no slot.
    inverse_rows = torch.tensor([[0, 4], [1, -1], [2, 6]], device="cuda")
    inverse_experts = torch.tensor([[0, 3], [1, -1], [2, 0]], device="cuda")
    inverse_map = torch.stack((inverse_rows, inverse_experts))
    restore_shape = torch.Size((3, hidden))
    rows = _adversarial((7, hidden), dtype, seed=11).requires_grad_()
    grad = _adversarial((hidden, 3), dtype, seed=12).t()  # non-contiguous
    assert not grad.is_contiguous()

    def step():
        rows.grad = None
        out = batch_invariant.unpermute(
            rows,
            restore_shape,
            probs=None,
            num_experts=num_experts,
            inverse_map=inverse_map,
            ep_size=2,
        )
        out.backward(grad)
        return out

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    rows.grad = None
    with torch.cuda.graph(graph):
        out = batch_invariant.unpermute(
            rows,
            restore_shape,
            probs=None,
            num_experts=num_experts,
            inverse_map=inverse_map,
            ep_size=2,
        )
        out.backward(grad)
    graph.replay()
    torch.cuda.synchronize()

    loop_rows = rows.detach().clone().requires_grad_()
    loop = batch_invariant._unpermute_loop(
        loop_rows, restore_shape, None, num_experts, inverse_map, 2
    )
    loop.backward(grad)
    assert torch.equal(_bits(out), _bits(loop))
    assert torch.equal(_bits(rows.grad), _bits(loop_rows.grad))
    assert torch.equal(_bits(rows.grad[[3, 5]]), _bits(torch.zeros_like(rows.grad[[3, 5]])))


# The training (AllToAll) combine must reproduce the combine of the inference-optimized model
# that produced the rollouts, which may run at another EP size: each rollout EP rank rounds
# the sum over its expert shard to one fp32 partial, so the shard boundaries set the bits.

NUM_EXPERTS, TOPK, HIDDEN = 8, 4, 256

# Token 0's combined value per (collective, EP size). "ordered" adds the fp32 shard partials
# in an fp32 chain, so once the two 2^-24 terms sit in different shards each one is a lost tie.
# "multimem" sums the partials exactly, so only a tie inside a shard partial (EP 2, where
# experts 0, 1, 3 share a shard) loses one.
_WITNESS = {
    "ordered": {1: 1.0078125, 2: 1.0, 4: 1.0, 8: 1.0},
    "multimem": {1: 1.0078125, 2: 1.0, 4: 1.0078125, 8: 1.0078125},
}


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _routed_outputs(num_tokens, seed):
    """Routing map and each routed expert's output row per token.

    Token 0 routes to experts 0, 1, 3, 4 with outputs 1, 2^-8, 2^-24, 2^-24 (see
    _WITNESS): one fp64 sum over all four gives 1 + 2^-8 + 2^-23, which rounds up to
    1.0078125 in bf16, but a 2^-24 that is a tie in an fp32 rounding is lost, giving 1.0.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(num_tokens, NUM_EXPERTS, device="cuda", generator=gen)
    scores[0] = torch.tensor([4.0, 3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    routing_map = torch.zeros_like(scores, dtype=torch.bool)
    routing_map.scatter_(1, scores.topk(TOPK, dim=1).indices, True)
    exponent = torch.randint(-12, 13, (num_tokens, NUM_EXPERTS, 1), device="cuda", generator=gen)
    outputs = torch.randn(num_tokens, NUM_EXPERTS, HIDDEN, device="cuda", generator=gen)
    outputs = (outputs * torch.exp2(exponent.float())).to(torch.bfloat16)
    outputs[0, [0, 1, 3, 4]] = torch.tensor([1.0, 2.0**-8, 2.0**-24, 2.0**-24]).to(outputs)[:, None]
    return routing_map, outputs


def _rollout_combine(routing_map, outputs, ep_size, collective):
    """The inference-optimized combine at ``ep_size``: each EP rank's batch-invariant kernel over
    its contiguous expert shard, then the cross-rank sum of ``collective`` in rank order."""
    from megatron.core.inference.moe.batch_invariant import unpermute_tokens_in_expert_order

    num_tokens = routing_map.shape[0]
    rows = outputs.view(-1, HIDDEN)
    row_ids = torch.arange(rows.shape[0], device="cuda", dtype=torch.int32)
    row_ids = torch.where(routing_map, row_ids.view(num_tokens, NUM_EXPERTS), -1)
    valid_tokens = torch.tensor(num_tokens, dtype=torch.int32, device="cuda")
    shard = NUM_EXPERTS // ep_size
    # "ordered" adds fp32 partials in rank order; "multimem" returns their correctly rounded
    # sum, which training mirrors with an fp64 accumulator.
    acc = torch.zeros(
        num_tokens,
        HIDDEN,
        dtype=torch.float64 if collective == "multimem" else torch.float32,
        device="cuda",
    )
    for rank in range(ep_size):
        local_rows = row_ids[:, rank * shard : (rank + 1) * shard].contiguous()
        acc += unpermute_tokens_in_expert_order(rows, local_rows, valid_tokens, None)
    return acc.to(torch.float32).to(torch.bfloat16)


def _training_combine(routing_map, outputs, combine_ep_size):
    """The real AllToAll dispatcher's batch-invariant combine on this rank's EP group of 1."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_token_dispatcher_type="alltoall",
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_backend=AttnBackend.flash,
        flash_attention_version=3,
        attention_dropout=0.0,
        batch_invariant_mode=True,
    )
    dispatcher = MoEAlltoAllTokenDispatcher(
        NUM_EXPERTS,
        list(range(NUM_EXPERTS)),
        config=config,
        pg_collection=get_default_pg_collection(),
    )
    if combine_ep_size is not None:
        # Set after the dispatcher is built, as RL does once it builds its rollout model.
        config.batch_invariant_moe_combine_ep_size = combine_ep_size
    num_tokens = routing_map.shape[0]
    hidden_states = torch.zeros(num_tokens, 1, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
        permuted, _ = dispatcher.dispatch_preprocess(
            hidden_states, routing_map, routing_map.float()
        )
        # Stand in for the experts: the row for slot (token, k) is that expert's output.
        rows, experts = dispatcher.batch_invariant_inverse_permutation_mapping
        tokens = torch.arange(num_tokens, device="cuda")[:, None].expand_as(rows)
        expert_output = torch.empty_like(permuted)
        expert_output[rows.reshape(-1)] = outputs[tokens.reshape(-1), experts.reshape(-1)]
        out = dispatcher.combine_postprocess(expert_output)
    return out.view(num_tokens, HIDDEN)


@pytest.mark.parametrize("collective", ["ordered", "multimem"])
@pytest.mark.parametrize("rollout_ep", [1, 2, 4, 8])
def test_alltoall_combine_reproduces_rollout_ep(
    model_parallel, monkeypatch, collective, rollout_ep
):
    """With batch_invariant_moe_combine_ep_size set to the rollout EP, training's combine on an
    EP group of 1 gives the rollout combine's bits."""
    monkeypatch.setattr(batch_invariant, "get_batch_invariant_collective", lambda: collective)
    routing_map, outputs = _routed_outputs(512, seed=rollout_ep)
    expected = _rollout_combine(routing_map, outputs, rollout_ep, collective)
    assert expected[0, 0].item() == _WITNESS[collective][rollout_ep]
    out = _training_combine(routing_map, outputs, combine_ep_size=rollout_ep)
    differing = (_bits(out) != _bits(expected)).any(dim=1)
    assert not differing.any(), (
        f"training combine differs from the EP={rollout_ep} rollout combine in tokens "
        f"{differing.nonzero().flatten().tolist()}"
    )


def test_alltoall_combine_defaults_to_own_ep(model_parallel):
    """Unset, the combine partitions by the layer's own EP group."""
    routing_map, outputs = _routed_outputs(512, seed=0)
    out = _training_combine(routing_map, outputs, combine_ep_size=None)
    expected = _rollout_combine(routing_map, outputs, 1, "ordered")
    assert torch.equal(_bits(out), _bits(expected))

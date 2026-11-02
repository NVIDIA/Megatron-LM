# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch-invariant tensor-parallel reduce-scatter of the inference-optimized layers.

Under batch-invariant mode a token's reduced row must not depend on which TP rank's chunk it
lands in, nor on the step's token count. NCCL's ring reduce-scatter adds a chunk's partials
starting at the chunk's owner and rounds after every hop, so with TP >= 3 the same partials get
different bits in different chunks. Every rank here gives the same partial to every row, so all
rows of every chunk and token count must equal the rank-order fp32 sum bit for bit.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.tensor_parallel import inference_layers
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not inference_layers.HAVE_TE, reason="Transformer Engine required"),
    pytest.mark.internal,
]

H = 4096
T_VALUES = (4, 64, 256, 4096)


def _tp_group(tp):
    if Utils.world_size < tp or Utils.world_size % tp:
        pytest.skip(f"needs a multiple of {tp} ranks, got {Utils.world_size}")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(123)
    return parallel_state.get_tensor_model_parallel_group()


@pytest.fixture(params=[2, 4], ids=["tp2", "tp4"])
def tp_group(request):
    yield _tp_group(request.param)
    SymmetricMemoryManager.destroy("tp")
    Utils.destroy_model_parallel()


def _config(tp):
    return TransformerConfig(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=32,
        tensor_model_parallel_size=tp,
        sequence_parallel=True,
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


def _reduce_scatter_fn(entry, tp_group):
    """[T, 1, H] per-rank partial -> [T / TP, 1, H] reduced chunk, through a production path."""
    tp = dist.get_world_size(tp_group)
    config = _config(tp)
    if entry == "reduce_scatter_to_sequence_parallel_region":
        return lambda x: inference_layers.inference_reduce_scatter_to_sequence_parallel_region(
            x, tp_group, config
        )
    # The row-parallel linear multiplies by an identity weight, which is exact, so its
    # partial is its input.
    layer = inference_layers.InferenceRowParallelLinear(
        H * tp,
        H,
        config=config,
        init_method=torch.nn.init.zeros_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
        tp_group=tp_group,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.eye(H))
    layer.eval()
    return lambda x: layer(x)[0]


def _partial(tp_group, integer=False):
    """This rank's partial of one token; the callers give it to every row."""
    gen = torch.Generator(device="cuda").manual_seed(1234 + dist.get_rank(tp_group))
    if integer:
        return torch.randint(-2, 3, (H,), device="cuda", generator=gen).to(torch.bfloat16)
    return torch.randn(H, device="cuda", generator=gen).to(torch.bfloat16)


def _rank_order_sum(partial, tp_group):
    partials = torch.empty(dist.get_world_size(tp_group), H, dtype=partial.dtype, device="cuda")
    dist.all_gather_into_tensor(partials, partial, group=tp_group)
    expected = partials[0].float()
    for p in partials[1:]:
        expected += p
    return expected.to(partial.dtype)


def _all_ranks_ok(ok):
    flag = torch.tensor([int(ok)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


@pytest.mark.parametrize(
    "entry", ["row_parallel_linear", "reduce_scatter_to_sequence_parallel_region"]
)
def test_reduce_scatter_is_position_invariant(tp_group, entry):
    tp = dist.get_world_size(tp_group)
    partial = _partial(tp_group)
    expected = _rank_order_sum(partial, tp_group).view(torch.int16)
    reduce_scatter = _reduce_scatter_fn(entry, tp_group)
    differing_rows = {}
    with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
        for t in T_VALUES:
            out = reduce_scatter(partial.expand(t, 1, H).contiguous())
            assert out.shape == (t // tp, 1, H), out.shape
            bad = int((out.view(-1, H).view(torch.int16) != expected).any(dim=-1).sum())
            if bad:
                differing_rows[t] = bad
    assert _all_ranks_ok(not differing_rows), (
        f"TP={tp}, rank {dist.get_rank(tp_group)}: rows differing from the rank-order sum "
        f"per token count: {differing_rows}"
    )


@pytest.mark.parametrize("tp,integer", [(2, False), (4, True)], ids=["tp2-random", "tp4-exact"])
def test_matches_nccl_where_its_order_cannot_matter(tp, integer):
    """Where NCCL's add order cannot change the bits (one add at TP=2, exact sums), the
    batch-invariant reduce-scatter must give NCCL's result, as training's TP=2 path does."""
    tp_group = _tp_group(tp)
    try:
        gen = torch.Generator(device="cuda").manual_seed(99 + dist.get_rank(tp_group))
        x = torch.randn(256, 1, H, device="cuda", generator=gen)
        x = (x.round().clamp(-2, 2) if integer else x).to(torch.bfloat16)
        nccl = torch.empty(256 // tp, 1, H, dtype=x.dtype, device="cuda")
        dist.reduce_scatter_tensor(nccl, x, group=tp_group)
        with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
            for entry in ("row_parallel_linear", "reduce_scatter_to_sequence_parallel_region"):
                out = _reduce_scatter_fn(entry, tp_group)(x)
                assert _all_ranks_ok(
                    torch.equal(out.view(torch.int16), nccl.view(torch.int16))
                ), entry
    finally:
        SymmetricMemoryManager.destroy("tp")
        Utils.destroy_model_parallel()


def test_reduce_scatter_replays_in_cuda_graph(tp_group):
    reduce_scatter = _reduce_scatter_fn("reduce_scatter_to_sequence_parallel_region", tp_group)
    x = torch.zeros(64, 1, H, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                reduce_scatter(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # thread_local, as Megatron's CUDA graphs capture NCCL collectives.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            out = reduce_scatter(x)
        for seed in (1, 2):
            gen = torch.Generator(device="cuda").manual_seed(seed + dist.get_rank(tp_group))
            x.copy_(torch.randn(x.shape, device="cuda", generator=gen))
            graph.replay()
            eager = reduce_scatter(x)
            assert _all_ranks_ok(torch.equal(out.view(torch.int16), eager.view(torch.int16)))

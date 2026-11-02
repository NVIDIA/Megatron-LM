# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch-invariant mode's tensor-parallel reduction (tensor_parallel.batch_invariant_reduction).

NCCL's ring reduce-scatter adds the partials of the chunk owned by ring position u as
``((x[u+1] + x[u+2]) + ...) + x[u]``, rounding after every add, and its all-reduce chunks the
tensor the same way. A token's chunk follows its row in the step, so with TP >= 3 the same
partials got different bits depending on the co-batched requests. In batch-invariant mode every
TP reduction must give each element the rank-order fp32 sum of its partials, rounded once,
whichever path reaches it: the inference-optimized layers, Transformer Engine's own reductions
(forward and backward), and Megatron's mappings, in training and in inference.
"""

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch.module.linear as te_linear

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.tensor_parallel import batch_invariant_reduction, inference_layers, mappings
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

H = 4096
T_VALUES = (4, 64, 256, 4096)


def _init_tp(tp):
    if Utils.world_size % tp:
        pytest.skip(f"needs a multiple of {tp} ranks, got {Utils.world_size}")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(123)
    return parallel_state.get_tensor_model_parallel_group()


@pytest.fixture
def tp_group(request):
    yield _init_tp(request.param)
    SymmetricMemoryManager.destroy("tp")
    Utils.destroy_model_parallel()


def _config(tp, sequence_parallel=True):
    return TransformerConfig(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=32,
        tensor_model_parallel_size=tp,
        sequence_parallel=sequence_parallel,
        normalization="RMSNorm",
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


def _identity_linear(cls, tp_group, sequence_parallel=True):
    """A TP linear whose per-rank weight is the identity, so its GEMM is exact: the partial each
    rank contributes to the reduction is its input (row parallel) or its output gradient
    (column parallel)."""
    tp = tp_group.size()
    if cls in (TEColumnParallelLinear, ColumnParallelLinear):
        sizes = dict(input_size=H, output_size=H * tp, gather_output=False)
    else:
        sizes = dict(input_size=H * tp, output_size=H, input_is_parallel=True)
    layer = cls(
        **sizes,
        config=_config(tp, sequence_parallel),
        init_method=torch.nn.init.zeros_,
        bias=False,
        skip_bias_add=True,
        is_expert=False,
        tp_group=tp_group,
    ).cuda()
    with torch.no_grad():
        layer.weight.copy_(torch.eye(H))
    return layer


def _symmetric_buffer(tp_group):
    buf = SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)
    assert buf.symm_mem_hdl is not None, f"no 'tp' symmetric memory: {buf.init_failure_reason}"
    return buf


def _column_parallel_dgrad(layer, tp_group):
    """The column-parallel input gradient, reduce-scattered from each rank's output gradient."""

    def run(grad_out):
        x = torch.zeros(
            grad_out.size(0) // tp_group.size(), 1, H, dtype=grad_out.dtype, device="cuda"
        )
        x.requires_grad_()
        with torch.enable_grad():
            layer(x)[0].backward(grad_out)
        return x.grad

    layer.train()
    return run


def _inference_reduce_scatter(tp_group):
    _symmetric_buffer(tp_group)
    config = _config(tp_group.size())
    return lambda x: inference_layers.inference_reduce_scatter_to_sequence_parallel_region(
        x, tp_group, config
    )


# name -> builder(tp_group) -> reduce(this rank's partial [T, 1, H]) -> this rank's output.
REDUCE_SCATTERS = {
    # Inference-optimized decode, from the "tp" symmetric buffer.
    "inference_row_parallel_linear": lambda g: _identity_linear(
        inference_layers.InferenceRowParallelLinear, g
    ).eval(),
    "inference_reduce_scatter_symmetric_memory": _inference_reduce_scatter,
    # Transformer Engine's reduce-scatter as its row-parallel Linear calls it, and its layers.
    "te_reduce_scatter_along_first_dim": lambda g: (
        lambda x: te_linear.reduce_scatter_along_first_dim(x, g)[0]
    ),
    "te_row_parallel_linear": lambda g: _identity_linear(TERowParallelLinear, g).train(),
    "te_column_parallel_dgrad": lambda g: _column_parallel_dgrad(
        _identity_linear(TEColumnParallelLinear, g), g
    ),
    # Megatron's mappings: local layers, embeddings, inference without symmetric memory, MoE.
    "local_row_parallel_linear": lambda g: _identity_linear(RowParallelLinear, g).train(),
    "local_column_parallel_dgrad": lambda g: _column_parallel_dgrad(
        _identity_linear(ColumnParallelLinear, g), g
    ),
    "reduce_scatter_to_sequence_parallel_region": lambda g: (
        lambda x: mappings.reduce_scatter_to_sequence_parallel_region(x, group=g)
    ),
    "async_reduce_scatter_along_first_dim": lambda g: (
        lambda x: mappings.async_reduce_scatter_along_first_dim(x, group=g).wait()
    ),
}
ALL_REDUCES = {
    "te_allreduce": lambda g: (lambda x: te_linear.allreduce(x, g)[0]),
    "te_row_parallel_linear_without_sp": lambda g: _identity_linear(
        TERowParallelLinear, g, sequence_parallel=False
    ).train(),
    "reduce_from_tensor_model_parallel_region": lambda g: (
        lambda x: mappings.reduce_from_tensor_model_parallel_region(x, group=g)
    ),
}


def _call(reduce, x):
    out = reduce(x)
    return out[0] if isinstance(out, tuple) else out


def _partial(tp_group, numel=H, seed=1234):
    """This rank's random bf16 partial."""
    gen = torch.Generator(device="cuda").manual_seed(seed + dist.get_rank(tp_group))
    return torch.randn(numel, device="cuda", generator=gen).to(torch.bfloat16)


def _rank_order_sum(partial, tp_group):
    """Every rank's partial summed in rank order in fp32, rounded once."""
    partials = torch.empty(
        dist.get_world_size(tp_group), *partial.shape, dtype=partial.dtype, device="cuda"
    )
    dist.all_gather_into_tensor(partials, partial.contiguous(), group=tp_group)
    expected = partials[0].float()
    for p in partials[1:]:
        expected += p
    return expected.to(partial.dtype)


def _all_ranks(ok):
    flag = torch.tensor([int(ok)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize("tp_group", [2, 4], indirect=True, ids=["tp2", "tp4"])
@pytest.mark.parametrize("entry", [*REDUCE_SCATTERS, *ALL_REDUCES])
def test_reduction_is_position_invariant(tp_group, entry):
    """Every rank gives every row the same partial, so every output row must equal the
    rank-order sum, in every chunk and at every token count."""
    tp = tp_group.size()
    all_reduce = entry in ALL_REDUCES
    partial = _partial(tp_group)
    expected = _bits(_rank_order_sum(partial, tp_group))
    reduce = (ALL_REDUCES if all_reduce else REDUCE_SCATTERS)[entry](tp_group)
    differing_rows = {}
    with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
        for t in T_VALUES:
            out = _call(reduce, partial.expand(t, 1, H).contiguous())
            assert out.shape == (t if all_reduce else t // tp, 1, H), out.shape
            bad = int((_bits(out).view(-1, H) != expected).any(dim=-1).sum())
            if bad:
                differing_rows[t] = bad
    assert _all_ranks(not differing_rows), (
        f"{entry}, TP={tp}, rank {dist.get_rank(tp_group)}: rows differing from the rank-order "
        f"sum per token count: {differing_rows}"
    )


@pytest.mark.parametrize("tp_group", [2, 4], indirect=True, ids=["tp2", "tp4"])
def test_rollout_reproduces_training(tp_group):
    """A token's row-parallel output has the same bits in a training forward (TE's reduction)
    and in a rollout decode step (inference-optimized, from symmetric memory), though the two
    steps put it in different rows and chunks."""
    tp = tp_group.size()
    layer = _identity_linear(inference_layers.InferenceRowParallelLinear, tp_group)
    gen = torch.Generator(device="cuda").manual_seed(7 + dist.get_rank(tp_group))
    tokens = torch.randn(8, 1, H, device="cuda", generator=gen).to(torch.bfloat16)
    # Training packs 256 rows (chunks of 256 / TP rows), the rollout step 64 (64 / TP rows).
    train_rows = torch.tensor([5, 37, 70, 101, 140, 166, 203, 250], device="cuda")
    rollout_rows = torch.tensor([60, 50, 3, 9, 20, 30, 40, 45], device="cuda")
    train_x = torch.randn(256, 1, H, device="cuda", generator=gen).to(torch.bfloat16)
    rollout_x = torch.randn(64, 1, H, device="cuda", generator=gen).to(torch.bfloat16)
    train_x[train_rows] = tokens
    rollout_x[rollout_rows] = tokens

    def gathered(out):
        full = torch.empty(out.size(0) * tp, *out.shape[1:], dtype=out.dtype, device="cuda")
        dist.all_gather_into_tensor(full, out.contiguous(), group=tp_group)
        return full

    with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
        train_out = gathered(layer.train()(train_x)[0])
        rollout_out = gathered(layer.eval()(rollout_x)[0])
    _symmetric_buffer(tp_group)  # the rollout step reduced from symmetric memory
    trained, rolled_out = _bits(train_out[train_rows]), _bits(rollout_out[rollout_rows])
    assert _all_ranks(torch.equal(trained, rolled_out)), (
        f"TP={tp}: hidden elements differing per token between training and rollout: "
        f"{(trained != rolled_out).sum(dim=(1, 2)).tolist()}"
    )
    assert _all_ranks(torch.equal(trained, _bits(_rank_order_sum(tokens, tp_group))))


def _wide_partials(numel, seed):
    """Random bf16 partials over a wide exponent range, with zeros of both signs and
    subnormals, so any rounding or sign difference between two adds shows up."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scale = torch.exp2(torch.randint(-40, 40, (numel,), device="cuda", generator=gen).float())
    x = (torch.randn(numel, device="cuda", generator=gen) * scale).to(torch.bfloat16)
    x[0::97] = -0.0
    x[1::89] = 0.0
    x[2::83] = -(2.0**-130)
    x[3::79] = 2.0**-133
    return x


@pytest.mark.internal
@pytest.mark.parametrize("numel", [8, 1 << 12, 1 << 20, 1 << 24])
def test_two_rank_nccl_add_equals_rank_order_sum(numel):
    """With two ranks and no symmetric buffer the reduction keeps NCCL's reduce-scatter and
    all-reduce, because their single add gives the bits of the rank-order fp32 sum. Checked
    against the kernel on both of its inputs (the all-to-all output and the symmetric buffer),
    at sizes across NCCL's algorithm and protocol choices."""
    tp_group = _init_tp(2)
    try:
        rank = dist.get_rank(tp_group)
        x = _wide_partials(2 * numel, seed=99 + dist.get_rank())
        nccl_rs = torch.empty(numel, dtype=x.dtype, device="cuda")
        dist.reduce_scatter_tensor(nccl_rs, x, group=tp_group)
        nccl_ar = x.clone()
        dist.all_reduce(nccl_ar, group=tp_group)

        partials = torch.empty_like(x)
        dist.all_to_all_single(partials, x, group=tp_group)
        ordered_rs = batch_invariant_reduction._rank_ordered_sum(
            torch.empty_like(nccl_rs), partials, 2, rank
        )
        symm = _symmetric_buffer(tp_group).maybe_get_tensor([2 * numel], dtype=x.dtype)
        symm["tensor"].copy_(x)
        symm_rs, _ = batch_invariant_reduction.batch_invariant_reduce_scatter(
            symm["tensor"], tp_group, symm_mem_hdl=symm["handle"]
        )
        gathered = torch.empty(2, 2 * numel, dtype=x.dtype, device="cuda")
        dist.all_gather_into_tensor(gathered, x, group=tp_group)
        ordered_ar = batch_invariant_reduction._rank_ordered_sum(
            torch.empty_like(x), gathered.view(-1), 2, rank
        )
        results = {
            "reduce_scatter_vs_all_to_all_sum": torch.equal(_bits(nccl_rs), _bits(ordered_rs)),
            "reduce_scatter_vs_symmetric_memory_sum": torch.equal(_bits(nccl_rs), _bits(symm_rs)),
            "all_reduce_vs_rank_order_sum": torch.equal(_bits(nccl_ar), _bits(ordered_ar)),
        }
        assert _all_ranks(all(results.values())), results
    finally:
        SymmetricMemoryManager.destroy("tp")
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("tp_group", [4], indirect=True, ids=["tp4"])
def test_layout_matches_nccl(tp_group):
    """With integer partials every sum is exact, so any add order gives the same bits: the
    batch-invariant reduce-scatter (all-to-all and symmetric memory) and all-reduce must then
    reproduce NCCL's, which pins which rows each rank gets."""
    gen = torch.Generator(device="cuda").manual_seed(5 + dist.get_rank(tp_group))
    x = torch.randint(-2, 3, (256, 1, H), device="cuda", generator=gen).to(torch.bfloat16)
    nccl_rs = torch.empty(64, 1, H, dtype=x.dtype, device="cuda")
    dist.reduce_scatter_tensor(nccl_rs, x, group=tp_group)
    nccl_ar = x.clone()
    dist.all_reduce(nccl_ar, group=tp_group)
    symm = _symmetric_buffer(tp_group).maybe_get_tensor(list(x.shape), dtype=x.dtype)
    symm["tensor"].copy_(x)
    reduce_scatter = batch_invariant_reduction.batch_invariant_reduce_scatter
    all_reduce = batch_invariant_reduction.batch_invariant_all_reduce
    symm_rs, _ = reduce_scatter(symm["tensor"], tp_group, symm_mem_hdl=symm["handle"])
    ok = {
        "all_to_all": torch.equal(_bits(reduce_scatter(x, tp_group)[0]), _bits(nccl_rs)),
        "symmetric_memory": torch.equal(_bits(symm_rs), _bits(nccl_rs)),
        "all_reduce": torch.equal(_bits(all_reduce(x.clone(), tp_group)[0]), _bits(nccl_ar)),
    }
    # A size that is not a multiple of TP takes the zero-padded all-reduce.
    odd = x.view(-1)[: 3 * H + 5]
    odd_nccl = odd.clone()
    dist.all_reduce(odd_nccl, group=tp_group)
    ok["all_reduce_padded"] = torch.equal(
        _bits(all_reduce(odd.clone(), tp_group)[0]), _bits(odd_nccl)
    )
    assert _all_ranks(all(ok.values())), ok


@pytest.mark.parametrize("tp_group", [4], indirect=True, ids=["tp4"])
def test_uneven_reduce_scatter_sums_in_rank_order(tp_group):
    """The variable-split reduce-scatter (the MoE combine across expert TP) sums in rank order."""
    rank = dist.get_rank(tp_group)
    splits = [3, 0, 5, 8]
    x = _partial(tp_group, numel=sum(splits) * H, seed=11).view(sum(splits), H)
    with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
        out = mappings.reduce_scatter_to_sequence_parallel_region(
            x, group=tp_group, input_split_sizes=splits
        )
    expected = _rank_order_sum(x, tp_group)[sum(splits[:rank]) : sum(splits[: rank + 1])]
    assert _all_ranks(torch.equal(_bits(out), _bits(expected)))


@pytest.mark.parametrize("tp_group", [4], indirect=True, ids=["tp4"])
@pytest.mark.parametrize("path", ["symmetric_memory", "all_to_all", "all_reduce"])
def test_reduction_replays_in_cuda_graph(tp_group, path):
    """The batch-invariant reduction captures in a CUDA graph and replays like eager."""
    if path == "symmetric_memory":
        _symmetric_buffer(tp_group)
        reduce = _identity_linear(inference_layers.InferenceRowParallelLinear, tp_group).eval()
    elif path == "all_to_all":
        reduce = lambda x: mappings.reduce_scatter_to_sequence_parallel_region(x, group=tp_group)
    else:
        reduce = lambda x: mappings.reduce_from_tensor_model_parallel_region(x, group=tp_group)
    x = torch.zeros(64, 1, H, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _call(reduce, x.clone())
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # thread_local, as Megatron's CUDA graphs capture NCCL collectives. The all-reduce
        # works in place, so each call reduces a copy of x.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            out = _call(reduce, x.clone())
        for seed in (1, 2):
            gen = torch.Generator(device="cuda").manual_seed(seed + dist.get_rank(tp_group))
            x.copy_(torch.randn(x.shape, device="cuda", generator=gen))
            graph.replay()
            eager = _call(reduce, x.clone())
            assert _all_ranks(torch.equal(_bits(out), _bits(eager)))


def test_te_reductions_are_restored_outside_batch_invariant_mode():
    names = ("reduce_scatter_along_first_dim", "allreduce", "symmetric_all_reduce")
    originals = {name: getattr(te_linear, name) for name in names}
    with set_batch_invariant_mode(True, backend="te_native"):
        assert all(getattr(te_linear, name) is not originals[name] for name in names)
    assert all(getattr(te_linear, name) is originals[name] for name in names)

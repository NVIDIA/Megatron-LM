# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native replay and independent numerical checks for shared-prefix attention.

Exact replay needs only ``torch.use_deterministic_algorithms(True)``, which
``--deterministic-mode`` enables. Outputs and gradients are compared with an fp64
reference; their error must stay within twice the error of ordinary FlashAttention over
the dense, unshared ``[prompt, completion]`` branches. CP1/2/4 tests use real NCCL
exchanges and real FlashAttention forward/backward. No CUDA compute or communication
implementation is substituted. The pass-plan, validation and cache tests run on CPU.
"""

import itertools
import random
from contextlib import contextmanager

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    deterministic_algorithms,
    run_once,
    seeded,
)
from tests.unit_tests.determinism.utils import RacingStreams

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native CUDA")


@pytest.fixture
def attention(monkeypatch):
    """Return the attention module with an empty plan cache."""
    from megatron.core.models.hybrid import shared_prefix_fused

    monkeypatch.setattr(shared_prefix_fused, "_PLAN_CACHE", {})
    return shared_prefix_fused


def _stars(*stars):
    """Pack ``(prompt_len, completion_lens)`` stars contiguously into a forest."""
    forest, offset = [], 0
    for prompt_len, completion_lens in stars:
        forest.append((offset, prompt_len, list(completion_lens)))
        offset += prompt_len + sum(completion_lens)
    return forest


def _random_stars(count, seed, *, prompt=300, completion=200):
    """Random stars with G in {1, 2..8, 16} and length-1 prompts and completions."""
    rng = random.Random(seed)
    stars = []
    for _ in range(count):
        prompt_len = rng.choice([1, 2, rng.randint(3, prompt)])
        group = rng.choice([1, 1, rng.randint(2, 8), 16])
        lengths = [rng.choice([1, rng.randint(2, completion)]) for _ in range(group)]
        stars.append((prompt_len, lengths))
    return _stars(*stars)


def _total(forest):
    offset, prompt_len, completion_lens = forest[-1]
    return offset + prompt_len + sum(completion_lens)


_FORESTS = {
    "g1": _stars((37, [53])),
    "length-one": _stars((1, [5, 1, 7]), (9, [1, 1])),
    "multi-star": _stars((8, [11, 7]), (10, [13, 9])),
    # Prompts and completions spanning several FlashAttention tiles.
    "multi-tile": _stars((300, [200, 129, 64]), (130, [257])),
    # Many stars: many zero-length-K gap sequences in the cross pass.
    "random-40": _random_stars(40, seed=1234),
}
_REPLAY_FORESTS = {
    "multi-star": _stars((512, [768, 768]), (256, [896, 896])),
    "g16": _stars((512, [224] * 16)),
}


def _segments(forest):
    """Yield ``(start, length, prefix_start, prefix_len)`` for every prompt and completion."""
    for offset, prompt_len, completion_lens in forest:
        yield offset, prompt_len, offset, 0
        start = offset + prompt_len
        for length in completion_lens:
            yield start, length, offset, prompt_len
            start += length


def _expected_visibility(forest):
    """Dense [total, total] mask: each row sees its prompt (if any) and its own segment causally."""
    total = _total(forest)
    visible = torch.zeros(total, total, dtype=torch.int32)
    for start, length, prefix_start, prefix_len in _segments(forest):
        visible[start : start + length, prefix_start : prefix_start + prefix_len] = 1
        visible[start : start + length, start : start + length] = torch.ones(
            length, length, dtype=torch.int32
        ).tril()
    return visible


def _plan_visibility(total, passes):
    """Count how often each (query, key) pair is attended across the plan's flash passes."""
    count = torch.zeros(total, total, dtype=torch.int32)
    for rows, k_idx, cu_q, cu_k, max_q, max_k, causal in passes:
        cu_q, cu_k = cu_q.tolist(), cu_k.tolist()
        assert cu_q[-1] == rows.stop - rows.start and len(cu_q) == len(cu_k)
        assert max_q == max(b - a for a, b in itertools.pairwise(cu_q))
        assert max_k == max(b - a for a, b in itertools.pairwise(cu_k))
        keys = torch.arange(total) if k_idx is None else k_idx.cpu()
        for q0, q1, k0, k1 in zip(cu_q, cu_q[1:], cu_k, cu_k[1:]):
            query_rows = torch.arange(rows.start + q0, rows.start + q1)
            seen = torch.ones(q1 - q0, k1 - k0, dtype=torch.int32)
            if causal:  # FlashAttention aligns the causal mask bottom-right
                seen = seen.tril(k1 - k0 - (q1 - q0))
            count[query_rows[:, None], keys[k0:k1][None, :]] += seen
    return count


@pytest.mark.parametrize(
    "forest",
    [_FORESTS[name] for name in ("g1", "length-one", "multi-star", "multi-tile")]
    + [_random_stars(6, seed, prompt=40, completion=20) for seed in range(20)],
)
def test_plan_attends_each_visible_pair_exactly_once(attention, forest):
    """The passes cover the shared-prefix mask exactly; zero-K gap rows attend nothing."""
    total, passes = attention._star_forest_plan_cached(forest, torch.device("cpu"))
    assert total == _total(forest)
    assert len(passes) == (2 if any(len(lens) > 1 for _, _, lens in forest) else 1)
    assert torch.equal(_plan_visibility(total, passes), _expected_visibility(forest))


@pytest.mark.parametrize(
    "forest,match",
    [
        ([], "at least one star"),
        ([(1, 4, [2])], "contiguously from token 0"),
        ([(0, 4, [2]), (7, 3, [1])], "contiguously from token 0"),
        ([(0, 0, [2])], "empty prompt or completion"),
        ([(0, 3, [2, 0])], "empty prompt or completion"),
    ],
)
def test_invalid_forest_is_rejected_before_attention(attention, forest, match):
    """Malformed forests raise before any attention kernel runs."""
    query = torch.zeros(100, 1, 2, 8)
    with pytest.raises(ValueError, match=match):
        attention.flash_composed_forest_attention(query, query, query, forest)


def _qkv(tokens, dtype, *, heads=8, kv_heads=2, dim=64, strided=True):
    """Use K/V views from an interleaved grouped QKV projection, as the backbone does."""
    width = heads // kv_heads
    storage = torch.randn(tokens, 1, kv_heads, (width + 2) * dim, device="cuda", dtype=dtype)
    q, k, v = storage.split([width * dim, dim, dim], dim=-1)
    q = q.reshape(tokens, 1, heads, dim)
    k, v = (tensor.reshape(tokens, 1, kv_heads, dim) for tensor in (k, v))
    if not strided:
        q, k, v = (tensor.contiguous() for tensor in (q, k, v))
    else:
        assert not k.is_contiguous() and not v.is_contiguous()
    return tuple(tensor.detach().requires_grad_(True) for tensor in (q, k, v))


def _reference(q, k, v, forest):
    """Independent prompt mask + softmax in the inputs' dtype, with no attention kernels."""
    heads, dim = q.shape[2:]
    expanded_k = k[:, 0].repeat_interleave(heads // k.shape[2], dim=1)
    expanded_v = v[:, 0].repeat_interleave(heads // v.shape[2], dim=1)
    blocks = []
    for start, length, prefix_start, prefix_len in _segments(forest):
        indices = torch.cat(
            [
                torch.arange(prefix_start, prefix_start + prefix_len, device=q.device),
                torch.arange(start, start + length, device=q.device),
            ]
        )
        scores = (
            torch.einsum("qhd,khd->hqk", q[start : start + length, 0], expanded_k[indices])
            * dim**-0.5
        )
        allowed = torch.arange(indices.numel(), device=q.device)[None, :] <= (
            prefix_len + torch.arange(length, device=q.device)[:, None]
        )
        probabilities = scores.masked_fill(~allowed[None], float("-inf")).softmax(-1)
        blocks.append(torch.einsum("hqk,khd->qhd", probabilities, expanded_v[indices]))
    total = _total(forest)
    if q.shape[0] > total:
        blocks.append(q.new_zeros(q.shape[0] - total, heads, dim))
    return torch.cat(blocks).reshape(q.shape[0], 1, heads * dim)


def _dense_branches(q, k, v, forest):
    """Ordinary causal FlashAttention over every unshared ``[prompt, completion]`` branch."""
    from flash_attn import flash_attn_varlen_func

    branches, picks, position = [], [], 0
    for offset, prompt_len, completion_lens in forest:
        prompt = torch.arange(offset, offset + prompt_len)
        start = offset + prompt_len
        for branch, length in enumerate(completion_lens):
            branches.append(torch.cat([prompt, torch.arange(start, start + length)]))
            if branch == 0:  # prompt rows come from the first branch
                picks.append(torch.arange(position, position + prompt_len))
            picks.append(torch.arange(position + prompt_len, position + prompt_len + length))
            position += prompt_len + length
            start += length
    index, pick = torch.cat(branches).cuda(), torch.cat(picks).cuda()
    lengths = [branch.numel() for branch in branches]
    cu = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_func(
        *(tensor[:, 0].index_select(0, index) for tensor in (q, k, v)),
        cu,
        cu,
        max(lengths),
        max(lengths),
        causal=True,
    ).index_select(0, pick)
    if q.shape[0] > out.shape[0]:
        out = torch.cat([out, out.new_zeros(q.shape[0] - out.shape[0], *out.shape[1:])])
    return out.reshape(q.shape[0], 1, -1)


def _relative_errors(result, truth):
    """Relative Frobenius error of the output and of each input gradient against fp64 truth."""
    outputs, grads = result
    observed = (outputs["out"], grads["in[0]"], grads["in[1]"], grads["in[2]"])
    return [
        ((actual.double() - expected).norm() / expected.norm()).item()
        for actual, expected in zip(observed, truth)
    ]


def _truth(inputs, forest, cotangent):
    reference_inputs = tuple(tensor.detach().double().requires_grad_() for tensor in inputs)
    output = _reference(*reference_inputs, forest)
    grads = torch.autograd.grad(output, reference_inputs, cotangent.double())
    return (output.detach(), *grads)


@_CUDA
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", sorted(_REPLAY_FORESTS))
@pytest.mark.parametrize("strided", [False, True])
def test_composed_attention_replays(attention, dtype, kind, strided):
    """Replay all Q/K/V gradients under contention with only Torch deterministic algorithms."""
    seeded()
    forest = _REPLAY_FORESTS[kind]
    inputs = _qkv(_total(forest) + 8, dtype, heads=32, strided=strided)
    cotangent = torch.randn(inputs[0].shape[0], 1, 32 * 64, device="cuda", dtype=dtype)

    def operation(q, k, v):
        return attention.flash_composed_forest_attention(q, k, v, forest)

    with deterministic_algorithms(True):
        outputs, grads = assert_replays_bit_exact(
            operation,
            inputs,
            grad_outputs={"out": cotangent},
            replays=3,
            contention=True,
            what=f"shared-prefix {kind}/{dtype}, deterministic algorithms",
        )
    assert set(grads) == {"in[0]", "in[1]", "in[2]"}
    assert all(
        torch.isfinite(tensor).all() for tensors in (outputs, grads) for tensor in tensors.values()
    )
    assert torch.count_nonzero(outputs["out"][-8:]) == 0
    assert all(torch.count_nonzero(gradient[-8:]) == 0 for gradient in grads.values())


@_CUDA
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", sorted(_FORESTS))
@pytest.mark.parametrize("deterministic", [False, True])
def test_composed_attention_matches_independent_reference(attention, dtype, kind, deterministic):
    """Fused error vs fp64 stays within 2x of dense unshared FlashAttention's error."""
    seeded()
    forest = _FORESTS[kind]
    inputs = _qkv(_total(forest) + 8, dtype)
    cotangent = torch.randn(inputs[0].shape[0], 1, 8 * 64, device="cuda", dtype=dtype)
    truth = _truth(inputs, forest, cotangent)

    def operation(q, k, v):
        return attention.flash_composed_forest_attention(q, k, v, forest)

    with deterministic_algorithms(deterministic):
        fused = run_once(operation, inputs, {"out": cotangent})
        baseline = run_once(
            lambda q, k, v: _dense_branches(q, k, v, forest), inputs, {"out": cotangent}
        )
    assert all(torch.isfinite(tensor).all() for tensors in fused for tensor in tensors.values())
    assert torch.count_nonzero(fused[0]["out"][-8:]) == 0
    assert all(torch.count_nonzero(gradient[-8:]) == 0 for gradient in fused[1].values())
    fused_errors = _relative_errors(fused, truth)
    baseline_errors = _relative_errors(baseline, truth)
    for name, fused_error, baseline_error in zip(
        ("out", "dq", "dk", "dv"), fused_errors, baseline_errors
    ):
        assert fused_error <= 2 * baseline_error, (
            f"{kind}/{dtype} {name}: fused relative error {fused_error:.3e} exceeds twice "
            f"the dense-branch FlashAttention error {baseline_error:.3e}"
        )


def _check_reference(result, reference_inputs, reference_output, cotangent, *, shard=None):
    reference_grads = torch.autograd.grad(reference_output, reference_inputs, cotangent.float())
    expected = reference_output.detach()
    if shard is not None:
        expected = shard(expected)
        reference_grads = tuple(shard(gradient) for gradient in reference_grads)
    outputs, grads = result
    assert set(outputs) == {"out"}
    assert set(grads) == {"in[0]", "in[1]", "in[2]"}
    # Same predeclared envelopes as the existing CUDA shared-prefix parity fixtures.
    # These are focused operator checks, not production full-model acceptance criteria.
    torch.testing.assert_close(outputs["out"].float(), expected, rtol=2e-2, atol=2e-2)
    for index, gradient in enumerate(reference_grads):
        torch.testing.assert_close(grads[f"in[{index}]"].float(), gradient, rtol=3e-2, atol=3e-2)
    assert all(torch.isfinite(tensor).all() for tensors in result for tensor in tensors.values())


def _zigzag_shard(tensor, lengths, size, rank):
    parts = []
    for sequence in tensor.split(lengths):
        chunks = sequence.chunk(2 * size)
        parts.extend((chunks[rank], chunks[2 * size - rank - 1]))
    return torch.cat(parts).contiguous()


def _collective_check(check):
    """Make every world rank observe a failed comparison before its next CP exchange."""
    failure = None
    try:
        check()
    except Exception as error:
        failure = str(error)
    success = torch.tensor(failure is None, dtype=torch.int32, device="cuda")
    # Synchronize over the world, including independent CP subgroups: a failing CP1
    # rank must not exit while another rank advances into a shared CP2/CP4 group.
    torch.distributed.all_reduce(success, op=torch.distributed.ReduceOp.MIN)
    if not success.item():
        raise AssertionError(failure or "native attention comparison failed on another rank")


def _collective_replays(operation, inputs, cotangent, what):
    """Use the byte-comparison harness with a collective verdict between native replays."""
    reference = run_once(operation, inputs, {"out": cotangent})
    for replay in range(2):
        with RacingStreams():
            actual = run_once(operation, inputs, {"out": cotangent})

        def compare():
            for expected, observed in zip(reference, actual):
                assert expected.keys() == observed.keys(), what
                for name in expected:
                    assert bytes_equal(
                        expected[name], observed[name]
                    ), f"{what}: replay {replay + 2} changed bit patterns in {name}"

        _collective_check(compare)
    return reference


@contextmanager
def _cp_groups():
    """Create CP1 plus every CP2/CP4 group size that divides the world."""
    from tests.unit_tests.test_utilities import Utils

    if Utils.world_size % 2:
        pytest.skip("native CP tests need a world size divisible by two")
    Utils.initialize_distributed()
    world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    groups = {}
    try:
        for size in (1, 2, 4):
            if world % size:
                continue
            for first in range(0, world, size):
                ranks = list(range(first, first + size))
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    groups[size] = group
        yield groups
    finally:
        torch.cuda.synchronize()
        for group in reversed(list(groups.values())):
            torch.distributed.destroy_process_group(group)


@_CUDA
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads,kv_heads", [(8, 2), (8, 8)])
def test_shared_prefix_cp_replays_and_matches_reference(attention, dtype, heads, kv_heads):
    """CP4 GQA replicates KV heads; compare its summed dK/dV against the global reference."""
    seeded()
    forest = _stars((64, [80, 80]), (32, [32, 32]))
    inputs = _qkv(320, dtype, heads=heads, kv_heads=kv_heads)
    reference_inputs = tuple(tensor.detach().float().requires_grad_() for tensor in inputs)
    reference_output = _reference(*reference_inputs, forest)
    cotangent = torch.randn_like(reference_output).to(dtype)
    with _cp_groups() as groups, deterministic_algorithms(True):
        for size, group in groups.items():

            def shard(tensor):
                return _zigzag_shard(tensor, [320], size, group.rank())

            local = tuple(shard(tensor).detach().requires_grad_() for tensor in inputs)

            def operation(q, k, v):
                return attention.flash_composed_forest_attention_cp(q, k, v, forest, cp_group=group)

            result = _collective_replays(
                operation, local, shard(cotangent), what=f"shared-prefix CP{size}/{dtype}"
            )
            # Each comparison needs a fresh reference autograd graph.
            reference_output = _reference(*reference_inputs, forest)
            _collective_check(
                lambda: _check_reference(
                    result, reference_inputs, reference_output, cotangent, shard=shard
                )
            )

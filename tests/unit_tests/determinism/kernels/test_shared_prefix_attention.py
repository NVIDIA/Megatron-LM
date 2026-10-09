# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native replay and independent numerical checks for shared-prefix attention.

The strict replay contract selects BOTH deterministic Torch accumulation and the
explicit FlashAttention deterministic-backward diagnostic. Production default
backward is checked against an independent reference, without claiming exact replay.
CP1/2/4 tests use real NCCL exchanges and real FlashAttention forward/backward.
No CUDA compute or communication implementation is substituted.
"""

import itertools
import runpy
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

_DEFERRED = ("NRL_SP_FUSED_KV_GATHER", "NRL_SP_FUSED_BACKWARD_GLUE", "NRL_SP_FUSED_DQ_ASSEMBLY")
_CONTROLS = (
    pytest.param(False, False, False, id="level-gather"),
    pytest.param(True, False, False, id="chain-gather"),
    pytest.param(True, True, False, id="chain-qslice"),
    pytest.param(True, True, True, id="full-context"),
)
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native CUDA")


@pytest.mark.parametrize("env_name", _DEFERRED)
@pytest.mark.parametrize("value", ["1", "TRUE", " true ", "yes"])
def test_deferred_variant_import_fails_closed(monkeypatch, env_name, value):
    """Execute the actual module's import guards; a resolver-only check is insufficient."""
    from megatron.core.models.hybrid import shared_prefix_fused as attention

    for name in (*_DEFERRED, "NRL_SP_FUSED_MERGE", "NRL_SP_MERGE_BT", "NRL_SP_MERGE_WARPS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, value)
    with pytest.raises(RuntimeError, match=env_name):
        runpy.run_path(attention.__file__, run_name="_shared_prefix_guard_probe")


@pytest.mark.parametrize("env_name", ["NRL_SP_MERGE_BT", "NRL_SP_MERGE_WARPS"])
def test_retired_tile_control_import_fails_closed(monkeypatch, env_name):
    from megatron.core.models.hybrid import shared_prefix_fused as attention

    for name in (*_DEFERRED, "NRL_SP_FUSED_MERGE", "NRL_SP_MERGE_BT", "NRL_SP_MERGE_WARPS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, "")
    with pytest.raises(RuntimeError, match=env_name):
        runpy.run_path(attention.__file__, run_name="_shared_prefix_guard_probe")


@pytest.fixture
def attention(monkeypatch):
    pytest.importorskip("flash_attn")
    from megatron.core.models.hybrid import shared_prefix_fused

    monkeypatch.setattr(shared_prefix_fused, "_PLAN_CACHE", {})
    monkeypatch.setattr(shared_prefix_fused, "_SP_STREAM_POOL", [])
    return shared_prefix_fused


def _select_controls(attention, monkeypatch, chain, qslice, streams, deterministic):
    monkeypatch.setattr(attention, "_SP_CHAINFIRST", chain)
    monkeypatch.setattr(attention, "_SP_QSLICE", qslice)
    monkeypatch.setattr(attention, "_SP_STREAMS", streams)
    monkeypatch.setenv("NRL_SP_DETERMINISTIC_BACKWARD", "1" if deterministic else "0")
    setting = attention._resolve_deterministic_backward_setting()
    assert setting is deterministic
    monkeypatch.setattr(attention, "_SP_DETERMINISTIC_BACKWARD", setting)


def _forest(kind, large=False):
    if kind == "multi-root":
        lengths = [512, 768, 768, 256, 896, 896] if large else [8, 11, 7, 10, 13, 9]
        parents = [-1, 0, 0, -1, 3, 3]
    else:
        lengths = [512, 640, 384, 512, 512, 768, 768] if large else [8, 9, 7, 11, 10, 13, 6]
        parents = [-1, 0, 1, 1, 0, 4, 0]
    starts = [0, *itertools.accumulate(lengths[:-1])]
    return starts, lengths, parents


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


def _reference(q, k, v, starts, lengths, parents):
    """Independent ancestor mask + FP32 softmax, with no attention implementation calls."""
    heads, dim = q.shape[2:]
    expanded_k = k[:, 0].repeat_interleave(heads // k.shape[2], dim=1)
    expanded_v = v[:, 0].repeat_interleave(heads // v.shape[2], dim=1)
    blocks = []
    for node, (start, length) in enumerate(zip(starts, lengths)):
        ancestors = []
        parent = parents[node]
        while parent != -1:
            ancestors.append(parent)
            parent = parents[parent]
        path = list(reversed(ancestors)) + [node]
        indices = torch.cat(
            [
                torch.arange(starts[index], starts[index] + lengths[index], device=q.device)
                for index in path
            ]
        )
        scores = (
            torch.einsum("qhd,khd->hqk", q[start : start + length, 0], expanded_k[indices])
            * dim**-0.5
        )
        ancestor_length = indices.numel() - length
        allowed = torch.arange(indices.numel(), device=q.device)[None, :] <= (
            ancestor_length + torch.arange(length, device=q.device)[:, None]
        )
        probabilities = scores.masked_fill(~allowed[None], float("-inf")).softmax(-1)
        blocks.append(torch.einsum("hqk,khd->qhd", probabilities, expanded_v[indices]))
    total = sum(lengths)
    if q.shape[0] > total:
        blocks.append(q.new_zeros(q.shape[0] - total, heads, dim))
    return torch.cat(blocks).reshape(q.shape[0], 1, heads * dim)


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


@_CUDA
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["multi-root", "deep"])
@pytest.mark.parametrize("chain,qslice,full_context", _CONTROLS)
@pytest.mark.parametrize("streams", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_composed_attention_replays(
    attention, monkeypatch, dtype, kind, chain, qslice, full_context, streams, strided
):
    """Replay all Q/K/V gradients under contention on 4096 active tokens plus trailing padding."""
    seeded()
    _select_controls(attention, monkeypatch, chain, qslice, streams, True)
    starts, lengths, parents = _forest(kind, large=True)
    inputs = _qkv(sum(lengths) + 8, dtype, heads=32, strided=strided)
    cotangent = torch.randn(inputs[0].shape[0], 1, 32 * 64, device="cuda", dtype=dtype)

    def operation(q, k, v):
        return attention.flash_composed_forest_attention_fused(
            q, k, v, starts, lengths, parents, full_context=full_context
        )

    with deterministic_algorithms(True):
        outputs, grads = assert_replays_bit_exact(
            operation,
            inputs,
            grad_outputs={"out": cotangent},
            replays=3,
            contention=True,
            what=f"shared-prefix {kind}/{dtype}, deterministic backward",
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
@pytest.mark.parametrize("kind", ["multi-root", "deep"])
@pytest.mark.parametrize("chain,qslice,full_context", _CONTROLS)
@pytest.mark.parametrize("deterministic", [False, True])
def test_composed_attention_matches_independent_reference(
    attention, monkeypatch, dtype, kind, chain, qslice, full_context, deterministic
):
    """Check default and diagnostic outputs/gradients; reference includes cross-pass coupling."""
    seeded()
    _select_controls(attention, monkeypatch, chain, qslice, True, deterministic)
    starts, lengths, parents = _forest(kind)
    inputs = _qkv(sum(lengths) + 8, dtype)
    reference_inputs = tuple(tensor.detach().float().requires_grad_() for tensor in inputs)
    reference_output = _reference(*reference_inputs, starts, lengths, parents)
    cotangent = torch.randn_like(reference_output).to(dtype)

    def operation(q, k, v):
        return attention.flash_composed_forest_attention_fused(
            q, k, v, starts, lengths, parents, full_context=full_context
        )

    with deterministic_algorithms(deterministic):
        result = run_once(operation, inputs, {"out": cotangent})
    _check_reference(result, reference_inputs, reference_output, cotangent)


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
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_distributed()
    world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    assert world >= 4 and world % 4 == 0, "native CP gates require world size divisible by four"
    groups = {}
    try:
        for size in (1, 2, 4):
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
@pytest.mark.parametrize("full_context", [False, True])
@pytest.mark.parametrize("heads,kv_heads", [(8, 2), (8, 8)])
def test_shared_prefix_cp_replays_and_matches_reference(
    attention, monkeypatch, dtype, full_context, heads, kv_heads
):
    """CP4 GQA replicates KV heads; compare its summed dK/dV against the global reference."""
    seeded()
    _select_controls(attention, monkeypatch, True, True, True, True)
    lengths, parents = [64, 80, 80, 32, 32, 32], [-1, 0, 0, -1, 3, 3]
    starts = [0, *itertools.accumulate(lengths[:-1])]
    forest = [(0, 64, [80, 80]), (224, 32, [32, 32])]
    inputs = _qkv(320, dtype, heads=heads, kv_heads=kv_heads)
    reference_inputs = tuple(tensor.detach().float().requires_grad_() for tensor in inputs)
    reference_output = _reference(*reference_inputs, starts, lengths, parents)
    cotangent = torch.randn_like(reference_output).to(dtype)
    with _cp_groups() as groups, deterministic_algorithms(True):
        for size in (1, 2, 4):
            group = groups[size]

            def shard(tensor):
                return _zigzag_shard(tensor, [320], size, group.rank())

            local = tuple(shard(tensor).detach().requires_grad_() for tensor in inputs)

            def operation(q, k, v):
                return attention.flash_composed_forest_attention_cp(
                    q, k, v, forest, cp_group=group, full_context=full_context
                )

            result = _collective_replays(
                operation, local, shard(cotangent), what=f"shared-prefix CP{size}/{dtype}"
            )
            # Each comparison needs a fresh reference autograd graph.
            reference_output = _reference(*reference_inputs, starts, lengths, parents)
            _collective_check(
                lambda: _check_reference(
                    result, reference_inputs, reference_output, cotangent, shard=shard
                )
            )

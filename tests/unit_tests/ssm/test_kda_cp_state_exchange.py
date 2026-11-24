# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The two-level CP state exchange composes the same boundary states as FLA's all-gather and merge.

- ``rank_chains`` reproduces FLA's per-rank ``pre_num_ranks`` / ``post_num_ranks``.
- On CPU, with simulated ranks (one thread each) and simulated all-gathers, the exchange's
  incoming states equal the sequential composition of the upstream summaries, for 4 to 64 ranks,
  several block sizes and random packed layouts, forwards and backwards, in FP64.
- On 4 GPUs (block size 2), the NCCL exchange equals the same composition, and FLA's
  ``chunk_kda`` with the exchange matches FLA's all-gather version and a CP1 reference in its
  output and input gradients.

Run the GPU tests with: torchrun --nproc-per-node 4 -m pytest -q <this file>
"""

import os
import threading

import pytest
import torch

from megatron.core.ssm.context_parallel import state_exchange
from megatron.core.ssm.context_parallel.state_exchange import exchange_boundary_state, rank_chains

_HEADS, _KEY, _VALUE = 2, 3, 4


def _random_layout(size, generator):
    part = int(torch.randint(1, 5, (1,), generator=generator))
    total = part * size
    count = int(torch.randint(0, 2 * size, (1,), generator=generator))
    cuts = torch.randint(1, total, (count,), generator=generator)
    return [0] + sorted(set(cuts.tolist())) + [total]


def _layouts(size, seed):
    generator = torch.Generator().manual_seed(seed)
    return [[0, 3 * size], [0, 1, 3 * size]] + [_random_layout(size, generator) for _ in range(10)]


def test_rank_chains_matches_fla():
    context = pytest.importorskip("fla.ops.cp.context")
    for size in (2, 5, 16, 64):
        for cu in _layouts(size, size):
            cu_t = torch.tensor(cu, dtype=torch.int64)
            chains = rank_chains(cu_t, size)
            for rank in range(size):
                ctx = context.get_cp_cu_seqlens(cu_t, cu_t, world_size=size, rank=rank)
                assert chains.pre[rank] == ctx.pre_num_ranks, (cu, rank)
                assert chains.post[rank] == ctx.post_num_ranks, (cu, rank)
                assert chains.num_seqs[rank] == len(ctx.cu_seqlens_cpu) - 1, (cu, rank)


def _summaries(size, dtype, device="cpu", key=_KEY, value=_VALUE):
    """Per-rank packed ``[delta_S | M]`` with contracting transitions, the same on every rank."""
    generator = torch.Generator().manual_seed(1234)
    out = []
    for _ in range(size):
        delta = torch.randn(_HEADS, key, value, generator=generator, dtype=torch.float64)
        trans = torch.randn(_HEADS, key, key, generator=generator, dtype=torch.float64)
        trans *= 0.4 / key**0.5
        out.append(torch.cat((delta, trans), dim=-1).to(dtype=dtype, device=device))
    return out


def _sequential_incoming(summaries, rank, pre, post, reverse):
    """FLA's merge loop: ``h = M h + delta`` over the upstream ranks, farthest first."""
    upstream = list(range(rank + post, rank, -1)) if reverse else list(range(rank - pre, rank))
    if not upstream:
        return None
    packed = [summaries[j].double().cpu() for j in upstream]
    key = packed[0].shape[-2]
    value = packed[0].shape[-1] - key
    h = torch.zeros(_HEADS, key, value, dtype=torch.float64)
    for p in packed:
        h = p[..., value:] @ h + p[..., :value]
    return h


class _Comm:
    """A simulated process group: all-gathers meet at a barrier of the member threads."""

    def __init__(self, members):
        self.members = members
        self.barrier = threading.Barrier(len(members))
        self.slots = [None] * len(members)


class _Handle:
    def __init__(self, comm, index):
        self.comm, self.index = comm, index

    def rank(self):
        return self.index

    def size(self):
        return len(self.comm.members)


def _fake_all_gather(output, local, group):
    comm = group.comm
    comm.slots[group.index] = local.clone()
    comm.barrier.wait()
    output.copy_(torch.stack(comm.slots))
    comm.barrier.wait()


def _simulate(summaries, chains, reverse, size):
    n = len(summaries)
    blocks = [_Comm(list(range(b * size, (b + 1) * size))) for b in range(n // size)]
    crosses = [_Comm(list(range(i, n, size))) for i in range(size)]
    world = _Comm(list(range(n)))
    results, errors = [None] * n, []

    def run(rank):
        def subgroups(group, block):
            assert block == size
            return (
                _Handle(blocks[rank // size], rank % size),
                _Handle(crosses[rank % size], rank // size),
            )

        try:
            results[rank] = exchange_boundary_state(
                summaries[rank].clone(),
                group=_Handle(world, rank),
                chains=chains,
                reverse=reverse,
                size=size,
                subgroups=subgroups,
                all_gather=_fake_all_gather,
            )
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
            for comm in blocks + crosses:
                comm.barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    return results


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("n", "size"), [(4, 2), (6, 3), (12, 4), (16, 4), (32, 8), (64, 8)])
def test_exchange_equals_sequential_composition_on_simulated_ranks(n, size, reverse):
    summaries = _summaries(n, torch.float64)
    for cu in _layouts(n, n + size):
        chains = rank_chains(torch.tensor(cu), n)
        results = _simulate(summaries, chains, reverse, size)
        for rank in range(n):
            expected = _sequential_incoming(
                summaries, rank, chains.pre[rank], chains.post[rank], reverse
            )
            if expected is None:
                assert results[rank] is None, (cu, rank)
            else:
                torch.testing.assert_close(results[rank], expected, rtol=1e-12, atol=1e-12)


def _init_cp(size):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available() or world < size:
        pytest.skip(f"needs {size} GPUs")
    from tests.unit_tests.test_utilities import Utils

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=size)
    from megatron.core import parallel_state

    return Utils, parallel_state.get_context_parallel_group()


_GPU_LAYOUTS = [
    [0, 1024],
    [0, 256, 512, 1024],
    [0, 300, 700, 1024],
    [0, 100, 600, 650, 1024],
    [0, 40, 1000, 1024],
]


@pytest.mark.parametrize(
    ("key", "value", "tol"),
    # 3 x 4 runs the PyTorch merge; 32 x 32 runs FLA's merge kernel, whose dots are TF32.
    [(3, 4, 1e-5), (32, 32, 5e-3)],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_nccl_exchange_equals_sequential_composition(reverse, key, value, tol):
    Utils, group = _init_cp(4)
    try:
        rank = group.rank()
        summaries = _summaries(4, torch.float32, device="cuda", key=key, value=value)
        for cu in _GPU_LAYOUTS:
            chains = rank_chains(torch.tensor(cu), 4)
            incoming = exchange_boundary_state(
                summaries[rank].clone(), group=group, chains=chains, reverse=reverse, size=2
            )
            expected = _sequential_incoming(
                summaries, rank, chains.pre[rank], chains.post[rank], reverse
            )
            if expected is None:
                assert incoming is None
            else:
                torch.testing.assert_close(incoming.double().cpu(), expected, rtol=tol, atol=tol)
    finally:
        Utils.destroy_model_parallel()


def _kda_inputs(total, heads, dim, generator):
    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)

    q = randn(1, total, heads, dim).bfloat16()
    k = randn(1, total, heads, dim).bfloat16()
    v = randn(1, total, heads, dim).bfloat16()
    g = torch.nn.functional.logsigmoid(randn(1, total, heads, dim) + 3.0)
    beta = torch.sigmoid(randn(1, total, heads)).bfloat16()
    grad = randn(1, total, heads, dim).bfloat16()
    return (q, k, v, g, beta), grad


def _run_chunk_kda(inputs, grad, cu_seqlens, cp_context):
    from fla.ops.kda import chunk_kda

    leaves = [x.detach().clone().requires_grad_(True) for x in inputs]
    out, _ = chunk_kda(
        *leaves, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens, cp_context=cp_context
    )
    out.backward(grad)
    return [out.detach().float()] + [x.grad.float() for x in leaves]


@pytest.mark.parametrize("cu_seqlens", _GPU_LAYOUTS)
def test_chunk_kda_exchange_matches_all_gather_and_cp1(cu_seqlens, monkeypatch):
    pytest.importorskip("fla.ops.kda")
    size = 4
    Utils, group = _init_cp(size)
    try:
        import fla.ops.kda.chunk_bwd as fla_kda_bwd
        import fla.ops.kda.chunk_fwd as fla_kda_fwd
        from fla.ops.cp import build_cp_context as fla_build_cp_context
        from fla.ops.cp import chunk_delta_h as fla_cp

        from megatron.core.ssm.gated_delta_net import kda_cp_state_exchange

        # Four GPUs: two blocks of two ranks (the default needs more than 8 ranks).
        def two(n):
            return 2 if n == 4 else None

        monkeypatch.setattr(state_exchange, "block_size", two)
        monkeypatch.setattr(kda_cp_state_exchange, "block_size", two)

        rank, total, heads, dim = group.rank(), cu_seqlens[-1], 4, 64
        generator = torch.Generator(device="cuda").manual_seed(77)
        inputs, grad = _kda_inputs(total, heads, dim, generator)
        cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
        reference = _run_chunk_kda(inputs, grad, cu, None)

        part = total // size
        local = slice(rank * part, (rank + 1) * part)
        local_inputs = [x[:, local] for x in inputs]
        context = kda_cp_state_exchange.build_cp_context(cu, group=group, cu_seqlens_cpu=cu.cpu())
        assert getattr(context, "megatron_rank_chains", None) is not None
        plain = fla_build_cp_context(cu.clone(), group=group, cu_seqlens_cpu=cu.cpu().clone())

        saved = (
            fla_kda_fwd.chunk_gated_delta_rule_fwd_h_pre_process,
            fla_kda_bwd.chunk_gated_delta_rule_bwd_dhu_pre_process,
        )
        try:
            fla_kda_fwd.chunk_gated_delta_rule_fwd_h_pre_process = (
                fla_cp.chunk_gated_delta_rule_fwd_h_pre_process
            )
            fla_kda_bwd.chunk_gated_delta_rule_bwd_dhu_pre_process = (
                fla_cp.chunk_gated_delta_rule_bwd_dhu_pre_process
            )
            all_gather = _run_chunk_kda(local_inputs, grad[:, local], None, plain)
            kda_cp_state_exchange.install()
            exchanged = _run_chunk_kda(local_inputs, grad[:, local], None, context)
        finally:
            (
                fla_kda_fwd.chunk_gated_delta_rule_fwd_h_pre_process,
                fla_kda_bwd.chunk_gated_delta_rule_bwd_dhu_pre_process,
            ) = saved

        names = ("output", "dq", "dk", "dv", "dg", "dbeta")
        for name, ref, ag, ex in zip(names, reference, all_gather, exchanged):
            ref = ref[:, local]
            scale = ref.abs().max().clamp_min(1e-6)
            for label, got in (("all-gather", ag), ("exchange", ex)):
                err = ((got - ref).abs().max() / scale).item()
                assert err < 2e-2, f"{label} {name} relative error {err:.3e} on {cu_seqlens}"
            err = ((ex - ag).abs().max() / scale).item()
            assert err < 1e-2, f"exchange vs all-gather {name} relative error {err:.3e}"
    finally:
        Utils.destroy_model_parallel()

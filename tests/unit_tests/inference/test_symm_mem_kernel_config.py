# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Grid sizes of the symmetric-memory barrier kernels.

A block that waits in a ``symm_mem_sync`` barrier keeps its SM until its peers' blocks arrive.
With shared-expert overlap two barrier kernels are in flight on each GPU, the shared expert's TP
collective on ``SharedExpertMLP.stream`` and the MoE dispatcher's on the current stream, so
neither grid may fill the GPU: two ranks could each fill their SMs with a different kernel's
blocks and wait forever on blocks the other cannot schedule. The test simulates an 8-SM GPU,
runs every NVLS collective that synchronizes through the barrier with enough work for more
blocks, and checks that each kernel launches at most half the SMs' worth of blocks and returns
the same bytes as at the default grid.
"""

import math

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton import (
    collectives,
    fused_collectives,
    is_device_nvls_capable,
    kernel_config,
    variable_collectives,
)
from megatron.core.inference.moe import batch_invariant
from megatron.core.inference.symmetric_memory import SymmetricMemoryBuffer
from tests.unit_tests.test_utilities import Utils

SIMULATED_SMS = 8
GRID_LIMIT = SIMULATED_SMS // 2
"""Two barrier kernels can be in flight on a GPU at once, so each may use half its SMs."""

TOKENS, HIDDEN, TOPK = 64, 1024, 8
"""Per rank: 8192 16-byte lanes (8 blocks of 1024 threads) for the uniform collectives, and up
to 64 one-token blocks for the variable-count ones."""

BARRIER_KERNELS = (
    (collectives, "_multimem_all_gather_kernel"),
    (collectives, "_multimem_all_gather_3_kernel"),
    (collectives, "_multimem_reduce_scatter_kernel"),
    (fused_collectives, "_multimem_reduce_scatter_residual_add_kernel"),
    (variable_collectives, "_multimem_all_gather_v_kernel"),
    (variable_collectives, "_multimem_all_gatherv_3tensor_kernel"),
    (variable_collectives, "_multimem_reduce_scatter_v_kernel"),
    (batch_invariant, "_ordered_reduce_scatter_v_kernel"),
)


class _GridRecorder:
    """Stands in for a Triton kernel and records the grid of each launch."""

    def __init__(self, kernel, grids):
        self._kernel, self._grids = kernel, grids

    def __getitem__(self, grid):
        self._grids.append(grid[0])
        return self._kernel[grid]


def _values(rank, shape, dtype, salt):
    """Deterministic data, different on every rank: multiples of 1/16 below 2 in magnitude, so
    every sum over the ranks is exact."""
    n = torch.arange(math.prod(shape), device="cuda").view(shape)
    v = (n * 7 + rank * 13 + salt) % 61 - 30
    return (v.to(torch.float32) / 16).to(dtype)


class TestBarrierKernelGrids:

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def _buffers(self, world, n):
        """``n`` symmetric buffers of 16 MiB; skips unless every rank has NVLink multicast."""
        if world < 2 or world & (world - 1):
            pytest.skip("the symm_mem_sync barrier needs a power-of-two world of at least 2")
        if not is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device())):
            pytest.skip("NVLS multicast needs a Hopper or newer GPU")
        buffers = [SymmetricMemoryBuffer(16, dist.group.WORLD) for _ in range(n)]
        ok = all(b.symm_mem_hdl is not None and b.symm_mem_hdl.multicast_ptr for b in buffers)
        missing = torch.tensor([0 if ok else 1], device="cuda")
        dist.all_reduce(missing)
        if int(missing):
            pytest.skip("symmetric memory with NVLink multicast is unavailable on some rank")
        return buffers

    def _run_collectives(self, buffers):
        """Run every barrier collective once on fresh inputs; return their outputs."""
        rank, world = dist.get_rank(), dist.get_world_size()
        ag, ag3, rs, fused, agv, agv_h, agv_r, agv_p, rsv, ordered = buffers
        out = {}

        def step(name, fn):
            # Every rank's inputs are in place before any peer reads or overwrites them.
            torch.cuda.synchronize()
            dist.barrier()
            out[name] = fn()
            torch.cuda.synchronize()
            dist.barrier()
            out[name] = [t.clone() for t in out[name]]

        x = _values(rank, (TOKENS, HIDDEN), torch.bfloat16, 0)

        def view(buf, rows, width, dtype, offset=0):
            nbytes = rows * width * dtype.itemsize
            return buf.symm_buffer[offset : offset + nbytes].view(dtype).view(rows, width)

        gathered = view(ag, world * TOKENS, HIDDEN, torch.bfloat16)
        step("all_gather", lambda: [collectives.multimem_all_gather(gathered, x, ag.symm_mem_hdl)])

        size = world * TOKENS * HIDDEN * 2
        outs3 = [view(ag3, world * TOKENS, HIDDEN, torch.bfloat16, i * size) for i in range(3)]
        ins3 = [_values(rank, (TOKENS, HIDDEN), torch.bfloat16, 1 + i) for i in range(3)]

        def all_gather_fused():
            collectives.multimem_all_gather_fused(
                outs3[0],
                ins3[0],
                0,
                outs3[1],
                ins3[1],
                size,
                outs3[2],
                ins3[2],
                2 * size,
                ag3.symm_mem_hdl,
            )
            return outs3

        step("all_gather_fused", all_gather_fused)

        rs_in = view(rs, world * TOKENS, HIDDEN, torch.bfloat16)
        rs_in.copy_(_values(rank, (world * TOKENS, HIDDEN), torch.bfloat16, 5))
        rs_out = torch.empty(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        step(
            "reduce_scatter",
            lambda: [collectives.multimem_reduce_scatter(rs_out, rs_in, rs.symm_mem_hdl)],
        )

        fused_in = view(fused, world * TOKENS, HIDDEN, torch.bfloat16)
        fused_in.copy_(_values(rank, (world * TOKENS, HIDDEN), torch.bfloat16, 6))
        residual_in = _values(rank, (TOKENS, HIDDEN), torch.bfloat16, 7)
        residual_out = torch.empty_like(residual_in)
        weights = _values(0, (HIDDEN,), torch.bfloat16, 8) + 1
        step(
            "fused_rs_norm_ag",
            lambda: [
                fused_collectives.fused_multimem_rs_add_norm_ag(
                    residual_out, fused_in, fused.symm_mem_hdl, residual_in, weights, 1e-5
                ),
                fused_in,
            ],
        )

        # Variable counts: rank r has 40 + r tokens; the grid is sized from TOKENS per rank.
        counts = [40 + q for q in range(world)]
        mine, first = counts[rank], sum(counts[:rank])
        offset = torch.tensor([first], dtype=torch.int32, device="cuda")
        ep_max = torch.tensor([max(counts)], dtype=torch.int32, device="cuda")
        var = dict(rank_token_offset=offset, ep_max_tokens=ep_max, per_rank_max_tokens=TOKENS)
        rows = world * TOKENS

        agv_out = view(agv, rows, HIDDEN, torch.bfloat16)
        agv_out.zero_()
        step(
            "all_gather_v",
            lambda: [
                variable_collectives.multimem_all_gather_v(
                    agv_out, x[:mine], agv.symm_mem_hdl, **var
                )
            ],
        )

        outs = [
            view(agv_h, rows, HIDDEN, torch.bfloat16),
            view(agv_r, rows, TOPK, torch.int64),
            view(agv_p, rows, TOPK, torch.float32),
        ]
        for t in outs:
            t.zero_()
        routing = torch.arange(first * TOPK, (first + mine) * TOPK, device="cuda").view(mine, TOPK)
        probs = _values(rank, (mine, TOPK), torch.float32, 9)
        handles = [b.symm_mem_hdl for b in (agv_h, agv_r, agv_p)]
        step(
            "all_gatherv_3tensor",
            lambda: list(
                variable_collectives.multimem_all_gatherv_3tensor(
                    *outs, x[:mine], routing, probs, *handles, **var
                )
            ),
        )

        for name, buf, combine in (
            ("reduce_scatter_v", rsv, variable_collectives.multimem_reduce_scatter_v),
            ("ordered_reduce_scatter_v", ordered, batch_invariant.ordered_reduce_scatter_v),
        ):
            partials = view(buf, rows, HIDDEN, torch.float32)
            partials.copy_(_values(rank, (rows, HIDDEN), torch.float32, 10))
            result = torch.empty(mine, HIDDEN, dtype=torch.float32, device="cuda")
            step(
                name,
                lambda combine=combine, partials=partials, result=result, buf=buf: [
                    combine(result, partials, buf.symm_mem_hdl, **var)
                ],
            )
        return out

    def test_barrier_kernels_leave_room_for_a_concurrent_one(self, monkeypatch):
        """On an 8-SM GPU every barrier kernel launches at most 4 blocks, with the same result
        as at the default grid; an explicit larger grid is refused."""
        world = dist.get_world_size() if dist.is_initialized() else 1
        buffers = self._buffers(world, 10)

        grids = {name: [] for _, name in BARRIER_KERNELS}
        for module, name in BARRIER_KERNELS:
            monkeypatch.setattr(module, name, _GridRecorder(getattr(module, name), grids[name]))

        expected = self._run_collectives(buffers)
        default_grids = {name: max(g) for name, g in grids.items()}
        # The default grids exceed the simulated limit, so the cap is exercised.
        assert all(g > GRID_LIMIT for g in default_grids.values()), default_grids

        for g in grids.values():
            g.clear()
        monkeypatch.setattr(
            kernel_config, "_multi_processor_count", lambda device: SIMULATED_SMS, raising=False
        )
        actual = self._run_collectives(buffers)

        too_large = {name: max(g) for name, g in grids.items() if not g or max(g) > GRID_LIMIT}
        assert not too_large, (
            f"on a {SIMULATED_SMS}-SM GPU these barrier kernels launched more than "
            f"{GRID_LIMIT} blocks (or none): {too_large}"
        )
        for name, tensors in expected.items():
            for want, got in zip(tensors, actual[name]):
                assert torch.equal(want, got), f"{name} changed with the capped grid"

        rank = dist.get_rank()
        gathered = buffers[0].symm_buffer[: world * TOKENS * HIDDEN * 2].view(torch.bfloat16)
        x = _values(rank, (TOKENS, HIDDEN), torch.bfloat16, 0)
        with pytest.raises(AssertionError, match="barrier kernels may use at most"):
            collectives.multimem_all_gather(
                gathered.view(world * TOKENS, HIDDEN),
                x,
                buffers[0].symm_mem_hdl,
                max_num_blocks=GRID_LIMIT + 1,
            )

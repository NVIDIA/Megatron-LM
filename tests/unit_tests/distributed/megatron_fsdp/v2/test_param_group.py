# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-process integration tests for ParameterGroup.

Builds a layer with mixed-dtype params (bf16 + uint8), splits them into
two ParameterGroups, and validates shard / unshard / reshard / reduce_grad
across all four sharding strategies.

Run with:
    torchrun --nproc_per_node=4 -m pytest megatron.core.distributed.fsdp.src.megatron_fsdp.tests.test_param_group -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import MixedPrecisionPolicy
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.param_group import ParameterGroup
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.utils import ParamGroupIdx

# ------------------------------------------------------------------ #
#  Process group — init once per pytest session, shared by all tests
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session", autouse=True)
def dist_env():
    """Initialize NCCL process group once and tear down at session end."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ------------------------------------------------------------------ #
#  Test model — contains bf16 and uint8 (simulated fp8) params
# ------------------------------------------------------------------ #


class MixedDtypeLayer(nn.Module):
    """A toy layer with large matrices, small biases, and quantized projections."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32, bias=False)  # bf16, shape [16, 32]
        self.linear2 = nn.Linear(32, 16, bias=True)  # bf16, shape [32, 16] + bias [16]
        self.norm = nn.LayerNorm(16)  # bf16, weight [16] + bias [16]
        self.quant_proj = nn.Linear(16, 8, bias=False)  # uint8, shape [16, 8]
        self.quant_gate = nn.Linear(16, 4, bias=False)  # uint8, shape [16, 4]


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _build_groups(strategy, mesh=None, mp_policy=None, outer_dp_sharding_strategy="no_shard"):
    """Create two ParameterGroups (bf16 + uint8) and call init_buffers.

    Returns (groups, originals, dp_group, rank, world_size, device) where
    `originals[i]` is a list of cloned param tensors before any sharding.
    """
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    dp_group = torch.distributed.group.WORLD if mesh is None else mesh.get_group(mesh_dim=1)

    # Fixed seed so all ranks start with identical weights
    torch.manual_seed(42)
    layer = MixedDtypeLayer()

    # Split params by dtype
    bf16_params, uint8_params = [], []
    for name, p in layer.named_parameters():
        if "quant" in name:
            uint8_params.append(
                nn.Parameter(p.data.to(torch.uint8).to(device), requires_grad=False)
            )
        else:
            bf16_params.append(nn.Parameter(p.data.to(torch.bfloat16).to(device)))

    # Build one ParameterGroup per dtype, each with its own param_group_id
    groups, originals = [], []
    for gid, params in enumerate([bf16_params, uint8_params]):
        if not params:
            continue
        originals.append([p.detach().clone() for p in params])
        pg = ParameterGroup(
            params=params,
            param_group_id=ParamGroupIdx(0, gid),
            mp_policy=mp_policy or MixedPrecisionPolicy(),
            mesh=mesh,
            sharding_strategy=strategy,
            outer_dp_sharding_strategy=outer_dp_sharding_strategy,
        )
        groups.append(pg)
    return groups, originals, dp_group, rank, torch.distributed.get_world_size(), device


def _build_hsdp_mesh(device):
    world_size = torch.distributed.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("HSDP mesh coverage requires an even world size >= 4")
    mesh = torch.arange(world_size, dtype=torch.int).reshape(2, world_size // 2)
    return DeviceMesh(device.type, mesh, mesh_dim_names=("dp_outer", "dp"))


def _flags(s):
    """Return (has_model_weight_buf, has_grad_buf, weight_distributed, grad_distributed).

    - has_model_weight_buf: model_weight_buffer is created for all supported strategies
    - has_grad_buf: always True when requires_grad, across all strategies
    - weight_distributed: only True for optim_grads_params (full FSDP)
    - grad_distributed: True for optim_grads and optim_grads_params
    """
    return {
        "no_shard": (True, True, False, False),
        "optim": (True, True, False, False),
        "optim_grads": (True, True, False, True),
        "optim_grads_params": (True, True, True, True),
    }[s]


# ------------------------------------------------------------------ #
#  PyTorch reference — thin wrappers around torch.distributed
# ------------------------------------------------------------------ #


class Ref:
    @staticmethod
    def all_gather(shard, group):
        ws = torch.distributed.get_world_size(group)
        out = torch.empty(shard.numel() * ws, dtype=shard.dtype, device=shard.device)
        torch.distributed.all_gather_into_tensor(out, shard, group=group)
        return out

    @staticmethod
    def reduce_scatter(full, group):
        ws = torch.distributed.get_world_size(group)
        ss = full.numel() // ws
        out = torch.empty(ss, dtype=full.dtype, device=full.device)
        torch.distributed.reduce_scatter_tensor(out, full, group=group)
        return out

    @staticmethod
    def all_reduce(t, group):
        torch.distributed.all_reduce(t, group=group)
        return t


# ------------------------------------------------------------------ #
#  Part 1: init_buffers — verify buffer creation and shard correctness
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
def test_init_buffers(strategy):
    groups, originals, dp_group, rank, ws, device = _build_groups(strategy)
    has_wbuf, _, w_dist, g_dist = _flags(strategy)

    for pg, orig in zip(groups, originals):
        # -- model_weight_buffer --
        if has_wbuf:
            assert pg.model_weight_buffer is not None
            wbuf = pg.model_weight_buffer
            assert wbuf.inner_sharded == w_dist

            # Per-param check: get_item should return this rank's portion of
            # the original param. A param may span shard boundaries, so the
            # returned slice can be shorter than the full param or even empty.
            for i, p in enumerate(orig):
                item = wbuf.get_item(i)
                if w_dist:
                    s, e = wbuf.buffer_index._get_item_self_range(i)
                    expected = p.flatten()[s:e]
                else:
                    expected = p.flatten()
                assert torch.equal(item, expected)
        # -- main_grad_buffer --
        if pg.requires_grad:
            assert pg.main_grad_buffer is not None
            assert pg.main_grad_buffer.inner_sharded == g_dist
            assert pg.main_grad_buffer.data is None  # lazy init

    torch.distributed.barrier()


# ------------------------------------------------------------------ #
#  Part 2: unshard + reshard — verify all-gather and cleanup
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
def test_unshard_reshard(strategy):
    if strategy not in ("no_shard", "optim_grads_params"):
        pytest.skip(
            "This test currently covers no_shard and optim_grads_params, "
            f"skipping {strategy}."
        )

    groups, originals, dp_group, rank, ws, device = _build_groups(strategy)
    _, _, w_dist, _ = _flags(strategy)

    for pg, orig in zip(groups, originals):
        wbuf = pg.model_weight_buffer
        assert wbuf is not None

        shard_before = wbuf.data.view(torch.uint8).clone()
        unsharded = wbuf.unshard()

        if not w_dist:
            # Non-distributed: unshard returns self.data directly, no comm
            assert unsharded is wbuf.data
        else:
            # Distributed: after all-gather, every param should be fully
            # recoverable from the unsharded buffer at its global offset
            for i, p in enumerate(orig):
                start, end = wbuf.buffer_index._get_item_global_range(i)
                recovered = unsharded[start:end]
                assert torch.equal(recovered, p.flatten())

        # Reshard: release temporary buffer, persistent shard must be intact
        wbuf.reshard()
        if w_dist:
            assert wbuf._unsharded_buffer is None
        # Compare the persistent storage bit-for-bit. The buffer can contain
        # uninitialized padding, including NaNs for which torch.equal is false
        # even when the before/after bit patterns are identical.
        assert torch.equal(wbuf.data.view(torch.uint8), shard_before)

    torch.distributed.barrier()


# ------------------------------------------------------------------ #
#  Part 3: reduce_grad — verify all-reduce / reduce-scatter
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
def test_reduce_grad(strategy):
    groups, _, dp_group, rank, ws, device = _build_groups(strategy)
    _, _, _, g_dist = _flags(strategy)

    for pg in groups:
        pg._init_dist_grads()  # lazily allocate grad buffer and dist_grads list
        gbuf = pg.main_grad_buffer
        if gbuf is None:
            # uint8 group has requires_grad=False, so no grad buffer
            continue

        if strategy == "no_shard":
            # No-shard: each rank fills with (rank+1), then all-reduce should
            # produce the sum across all ranks.
            gbuf.data.fill_(float(rank + 1))
            ref = torch.full_like(gbuf.data, float(rank + 1))
            Ref.all_reduce(ref, dp_group)
            gbuf.reduce_grad(reduce_scatter=False)
            assert torch.equal(gbuf.data, ref)
        else:
            # ZeRO-1/2/3: reduce-scatter a full gradient buffer and compare
            # this rank's optimizer-facing shard against the PyTorch reference.
            full_size = gbuf.buffer_index.bucket_meta.size
            full = torch.full((full_size,), float(rank + 1), dtype=gbuf.dtype, device=device)

            ref_shard = Ref.reduce_scatter(full.clone(), dp_group)

            if g_dist:
                # Pre-populate the allocator so reduce_grad sees the full temp buffer.
                bucket = gbuf.allocator.allocate(
                    key=gbuf.alloc_key, size=full_size, dtype=gbuf.dtype, device=device
                )
                bucket.data.copy_(full)
                gbuf.data.zero_()
            else:
                gbuf.data.copy_(full)
            gbuf.reduce_grad()

            # Only compare the shard region of self.data
            # shard_layout=(outer, inner): (0, 1) means inner sharded only.
            actual = gbuf.get_shard_view((0, 1))
            assert torch.equal(actual, ref_shard)

    torch.distributed.barrier()


@pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
@pytest.mark.parametrize("outer_strategy", ["no_shard", "optim"])
def test_hsdp_reduce_grad(strategy, outer_strategy):
    if outer_strategy == "optim" and strategy != "optim_grads_params":
        pytest.skip("Outer-DP optimizer sharding currently requires inner optim_grads_params.")

    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    mesh = _build_hsdp_mesh(device)
    groups, _, _, rank, _, device = _build_groups(
        strategy, mesh=mesh, outer_dp_sharding_strategy=outer_strategy
    )
    _, _, _, g_dist = _flags(strategy)

    for pg in groups:
        pg._init_dist_grads()  # lazily allocate grad buffer and dist_grads list
        gbuf = pg.main_grad_buffer
        if gbuf is None:
            continue
        assert not pg._full_grad_buffer_has_accumulated_grad
        assert not pg._reduced_grad_buffer_has_accumulated_grad

        if strategy == "no_shard":
            gbuf.data.fill_(float(rank + 1))
            ref = torch.full_like(gbuf.data, float(rank + 1))
            Ref.all_reduce(ref, pg.dp_group)
            Ref.all_reduce(ref, pg.outer_dp_group)
            pg.reduce_grad(is_last_microbatch=True)
            assert torch.equal(gbuf.data, ref)
        else:
            full_size = gbuf.buffer_index.bucket_meta.size
            full = torch.full((full_size,), float(rank + 1), dtype=gbuf.dtype, device=device)

            ref_shard = Ref.reduce_scatter(full.clone(), pg.dp_group)
            if outer_strategy == "optim":
                ref_shard = Ref.reduce_scatter(ref_shard, pg.outer_dp_group)
            else:
                Ref.all_reduce(ref_shard, pg.outer_dp_group)

            if g_dist:
                bucket = gbuf.allocator.allocate(
                    key=gbuf.alloc_key, size=full_size, dtype=gbuf.dtype, device=device
                )
                bucket.data.copy_(full)
                gbuf.data.zero_()
            else:
                gbuf.data.copy_(full)
            pg.reduce_grad(is_last_microbatch=True)

            if outer_strategy == "optim":
                # shard_layout=(outer, inner): (1, 1) means both dimensions are sharded.
                actual = gbuf.get_shard_view((1, 1))
            else:
                # shard_layout=(outer, inner): (0, 1) means inner sharded only.
                actual = gbuf.get_shard_view((0, 1))
            assert torch.equal(actual, ref_shard)

        assert pg._full_grad_buffer_has_accumulated_grad == (strategy == "no_shard")
        assert pg._reduced_grad_buffer_has_accumulated_grad

    torch.distributed.barrier()


@pytest.mark.parametrize("strategy", ["no_shard", "optim"])
def test_hsdp_reduce_grad_multi_microbatch(strategy):
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    mesh = _build_hsdp_mesh(device)
    groups, _, _, rank, _, _ = _build_groups(
        strategy, mesh=mesh, outer_dp_sharding_strategy="no_shard"
    )

    num_micro_batches = 3
    for pg in groups:
        pg._init_dist_grads()
        gbuf = pg.main_grad_buffer
        if gbuf is None:
            continue

        gbuf.data.zero_()
        full_batch_grad = torch.zeros_like(gbuf.data)
        for microbatch in range(num_micro_batches):
            micro_grad = torch.full_like(
                gbuf.data, float((microbatch + 1) * (rank + 1))
            )
            gbuf.data.add_(micro_grad)
            full_batch_grad.add_(micro_grad)
            is_last_microbatch = microbatch == num_micro_batches - 1
            pg.reduce_grad(is_last_microbatch=is_last_microbatch)
            if is_last_microbatch:
                assert pg._full_grad_buffer_has_accumulated_grad == (
                    strategy == "no_shard"
                )
                assert pg._reduced_grad_buffer_has_accumulated_grad
            else:
                assert pg._full_grad_buffer_has_accumulated_grad
                assert not pg._reduced_grad_buffer_has_accumulated_grad

        if strategy == "no_shard":
            ref = full_batch_grad
            Ref.all_reduce(ref, pg.dp_group)
            Ref.all_reduce(ref, pg.outer_dp_group)
            assert torch.equal(gbuf.data, ref)
        else:
            ref_shard = Ref.reduce_scatter(full_batch_grad, pg.dp_group)
            Ref.all_reduce(ref_shard, pg.outer_dp_group)
            # shard_layout=(outer, inner): (0, 1) means inner sharded only.
            actual = gbuf.get_shard_view((0, 1))
            assert torch.equal(actual, ref_shard)

        pg.zero_grad()
        assert not pg._full_grad_buffer_has_accumulated_grad
        assert not pg._reduced_grad_buffer_has_accumulated_grad

    torch.distributed.barrier()

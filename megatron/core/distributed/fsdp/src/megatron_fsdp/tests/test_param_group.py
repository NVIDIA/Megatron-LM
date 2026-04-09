"""Multi-process integration tests for ParameterGroup.

Builds a layer with mixed-dtype params (bf16 + uint8), splits them into
two ParameterGroups, and validates shard / unshard / reshard / reduce_grad
across all four sharding strategies.

Run with:
    torchrun --nproc_per_node=4 -m pytest megatron/core/distributed/fsdp_refactor/src/test_param_group.py -v
"""

import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "/home/tongliu/tongliu/megatron/fsdp/Megatron-LM")
from megatron.core.distributed.fsdp_refactor.src.param_group import ParameterGroup

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
    torch.distributed.destroy_process_group()


# ------------------------------------------------------------------ #
#  Test model — contains bf16 and uint8 (simulated fp8) params
# ------------------------------------------------------------------ #


class MixedDtypeLayer(nn.Module):
    """A toy layer with large matrices, small biases, and quantized projections."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 512, bias=False)  # bf16, shape [512, 256]
        self.linear2 = nn.Linear(512, 256, bias=True)  # bf16, shape [256, 512] + bias [256]
        self.norm = nn.LayerNorm(256)  # bf16, weight [256] + bias [256]
        self.quant_proj = nn.Linear(256, 128, bias=False)  # uint8, shape [128, 256]
        self.quant_gate = nn.Linear(256, 64, bias=False)  # uint8, shape [64, 256]


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _build_groups(strategy):
    """Create two ParameterGroups (bf16 + uint8) and call init_buffers.

    Returns (groups, originals, dp_group, rank, world_size, device) where
    `originals[i]` is a list of cloned param tensors before any sharding.
    """
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    dp_group = torch.distributed.group.WORLD

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
        pg = ParameterGroup(
            params=params,
            fsdp_unit_id=0,
            dp_group=dp_group,
            device=device,
            sharding_strategy=strategy,
            param_group_id=gid,
        )
        pg.init_buffers()
        groups.append(pg)
        # Snapshot original (pre-shard) values for later comparison
        originals.append([p.detach().clone() for p in params])
    return groups, originals, dp_group, rank, torch.distributed.get_world_size(), device


def _flags(s):
    """Return (has_model_weight_buf, has_grad_buf, weight_distributed, grad_distributed).

    - has_model_weight_buf: whether model_weight_buffer is created (False for no_shard)
    - has_grad_buf: always True when requires_grad, across all strategies
    - weight_distributed: only True for optim_grads_params (full FSDP)
    - grad_distributed: True for optim_grads and optim_grads_params
    """
    return {
        "no_shard": (False, True, False, False),
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
            assert wbuf.is_distributed == w_dist

            # Per-param check: get_item should return this rank's portion of
            # the original param. A param may span shard boundaries, so the
            # returned slice can be shorter than the full param or even empty.
            for i, p in enumerate(orig):
                item = wbuf.get_item(i)
                if w_dist:
                    s, e = wbuf.buffer_index._get_item_slice_in_shard(i)
                    expected = p.flatten()[s:e]
                else:
                    expected = p.flatten()
                assert torch.equal(item, expected)
        else:
            # no_shard: model_weight_buffer should not exist
            assert pg.model_weight_buffer is None

        # -- main_grad_buffer --
        if pg.requires_grad:
            assert pg.main_grad_buffer is not None
            assert pg.main_grad_buffer.is_distributed == g_dist
            # Grad buffer must be zero-initialized
            assert torch.all(pg.main_grad_buffer.data == 0)

    torch.distributed.barrier()


# ------------------------------------------------------------------ #
#  Part 2: unshard + reshard — verify all-gather and cleanup
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
def test_unshard_reshard(strategy):
    groups, originals, dp_group, rank, ws, device = _build_groups(strategy)
    _, _, w_dist, _ = _flags(strategy)

    for pg, orig in zip(groups, originals):
        wbuf = pg.model_weight_buffer
        if wbuf is None:
            # no_shard: nothing to unshard
            continue

        shard_before = wbuf.data.clone()
        unsharded, work = wbuf.unshard(async_op=False)

        if not w_dist:
            # Non-distributed: unshard returns self.data directly, no comm
            assert unsharded is wbuf.data
            assert work is None
        else:
            # Distributed: after all-gather, every param should be fully
            # recoverable from the unsharded buffer at its global offset
            for i, p in enumerate(orig):
                off, sz = wbuf.buffer_index._get_item_offset(i)
                recovered = unsharded[off : off + sz]
                assert torch.equal(recovered, p.flatten())

        # Reshard: release temporary buffer, persistent shard must be intact
        wbuf.reshard()
        if w_dist:
            assert wbuf._unsharded_buffer is None
        assert torch.equal(wbuf.data, shard_before)

    torch.distributed.barrier()


# ------------------------------------------------------------------ #
#  Part 3: reduce_grad — verify all-reduce / reduce-scatter
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
def test_reduce_grad(strategy):
    groups, _, dp_group, rank, ws, device = _build_groups(strategy)
    _, _, _, g_dist = _flags(strategy)

    for pg in groups:
        gbuf = pg.main_grad_buffer
        if gbuf is None:
            # uint8 group has requires_grad=False, so no grad buffer
            continue

        if not g_dist:
            # Non-distributed: each rank fills with (rank+1), then all-reduce
            # should produce sum across all ranks
            gbuf.data.fill_(float(rank + 1))
            ref = torch.full_like(gbuf.data, float(rank + 1))
            Ref.all_reduce(ref, dp_group)
            gbuf.reduce_grad(async_op=False)
            assert torch.equal(gbuf.data, ref)
        else:
            # Distributed: fill a full-size temp buffer with (rank+1),
            # reduce-scatter it, compare against PyTorch ref
            full_size = gbuf.buffer_index.bucket_meta.size
            full = torch.full((full_size,), float(rank + 1), dtype=gbuf.dtype, device=device)

            # Pre-populate the allocator so reduce_grad sees the data
            bucket = gbuf.allocator.allocate(
                param_group_id=gbuf.buffer_index.param_group_id,
                size=full_size,
                dtype=gbuf.dtype,
                device=device,
            )
            bucket.data.copy_(full)

            ref_shard = Ref.reduce_scatter(full.clone(), dp_group)

            gbuf.data.zero_()
            gbuf.reduce_grad(async_op=False)

            # Only compare the shard region of self.data
            sm = gbuf.buffer_index.shard_meta
            actual = gbuf.data[sm.local_data_index : sm.local_data_index + sm.size]
            assert torch.equal(actual, ref_shard)

    torch.distributed.barrier()

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

"""Multi-process correctness test for the single-root distributed Muon optimizer.

``test_dist_muon_matches_reference`` validates ``FullyShardV2Muon`` against a
full-tensor reference in five steps, per parameter and per step:

  1. construct a full gradient, identical on every rank;
  2. plant it into the sharded grad buffer (each rank keeps only its own slice);
  3. reference: run a full-tensor Muon update on every rank, mirroring the
     optimizer's Newton-Schulz path;
  4. distributed: ``opt.step()`` gathers each grad to its root rank, runs NS
     there, scatters the result back, and updates each rank's shard;
  5. compare the two results bit for bit (``torch.equal``).

Why exact equality: momentum and the weight update are elementwise, so they
commute with sharding; the gather/scatter just moves bytes; and both sides run
the SAME ``orthogonalize`` on the SAME assembled full gradient. The only
non-elementwise op is NS, and it sees identical input on same-arch GPUs. So any
divergence is a real bug in the gather / scatter / sharded-update logic.

The param set also carries a trailing 1D bias (shape ``(8,)``): Muon manages only
2D matrices, so its reference stays frozen at init and the exact-match check thus
also asserts Muon never touched it.

Run with:
    torchrun --nproc_per_node=4 -m pytest tests/unit_tests/distributed/megatron_fsdp/v2/test_dist_muon.py -v
"""

import shutil
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.tensor import DTensor, DeviceMesh

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fully_shard import fully_shard
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import (
    MixedPrecisionPolicy,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.param_group import ParameterGroup
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.utils import ParamGroupIdx
from megatron.core.optimizer.fully_shard_v2_muon import (
    HAVE_EMERGING_OPTIMIZERS,
    FullyShardV2Muon,
    FullyShardV2MuonOptimizer,
    _vanilla_muon_scale,
    _vanilla_newton_schulz,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig

if HAVE_EMERGING_OPTIMIZERS:
    from megatron.core.optimizer.fully_shard_v2_muon import (
        get_muon_scale_factor,
        newton_schulz_tp,
    )

STRATEGIES = ["optim", "optim_grads", "optim_grads_params"]
MESH_CASES = [
    pytest.param("world", id="world-mesh"),
    pytest.param("rank_pairs", id="rank-pair-mesh"),
]

_RANK_PAIR_MESH = None
SHARED_TMP_DIR = "/tmp/pytest-shared-tmp"

# Each set is a list of param shapes at a different scale: four 2D matrices (mixing
# wide rows<cols and tall rows>cols -> NS transposes) + a trailing 1D bias Muon must
# skip. Sizes are picked so each matrix straddles a shard boundary (ZeRO-2/3), so
# gather/scatter is real.
SHAPE_SETS = [
    pytest.param([(5, 8), (9, 8), (8, 6), (7, 8), (8,)], id="small"),  # dims < 10
    pytest.param([(200, 256), (400, 256), (256, 300), (256, 128), (256,)], id="medium"),
    pytest.param(
        [(1500, 2048), (3000, 2048), (2048, 1200), (2048, 2600), (2048,)], id="large"
    ),
]


class _MuonCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 32)
        self.mlp = nn.Linear(32, 24, bias=False)

    def forward(self, x):
        return self.mlp(self.proj(x))


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


def _read_full_weight(pg, idx, shape):
    """All-gather this param's full master weight and return it (reshaped, cloned)."""
    mbuf = pg.main_weight_buffer
    full = mbuf.unshard(bind_params=False)
    start, end = mbuf.buffer_index._get_item_global_range(idx)
    out = full[start:end].clone().view(shape)
    mbuf.reshard()
    return out


def _make_mesh(mesh_case, device):
    """Return either the default world mesh or a rank-pair subgroup mesh."""
    global _RANK_PAIR_MESH

    if mesh_case == "world":
        return None

    world_size = torch.distributed.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("rank-pair mesh coverage requires an even world size >= 4")

    if _RANK_PAIR_MESH is not None:
        return _RANK_PAIR_MESH

    rank = torch.distributed.get_rank()
    selected_group, selected_ranks = None, None
    for start in range(0, world_size, 2):
        ranks = list(range(start, start + 2))
        group = torch.distributed.new_group(ranks=ranks)
        if rank in ranks:
            selected_group = group
            selected_ranks = ranks

    assert selected_group is not None
    # Ranks 2/3 use group ranks 0/1 in a group whose global ranks are 2/3. This
    # catches P2P code that accidentally passes group ranks as global peers.
    _RANK_PAIR_MESH = DeviceMesh.from_group(
        [selected_group],
        device_type=device.type,
        mesh=selected_ranks,
        mesh_dim_names=("dp",),
    )
    return _RANK_PAIR_MESH


def _reference_orthogonalize(opt, grad):
    """Mirror FullyShardV2Muon's root-side Newton-Schulz path."""
    grad = grad.to(torch.float32)
    if HAVE_EMERGING_OPTIMIZERS:
        orth = newton_schulz_tp(
            grad,
            steps=opt._num_ns_steps,
            coefficient_type=opt._coefficient_type,
            tp_group=None,
            partition_dim=None,
            tp_mode="duplicated",
        )
        scale = get_muon_scale_factor(grad.size(-2), grad.size(-1), mode=opt._scale_mode)
    else:
        orth = _vanilla_newton_schulz(grad, steps=opt._num_ns_steps)
        scale = _vanilla_muon_scale(grad.size(-2), grad.size(-1))
    return orth * (scale * opt._extra_scale_factor)


def _build_muon_checkpoint_optimizer(device):
    torch.manual_seed(4321)
    model = _MuonCheckpointModel().to(device)
    fully_shard(
        model,
        mp_policy=MixedPrecisionPolicy(
            main_params_dtype=torch.float32, main_grads_dtype=torch.float32
        ),
        sharding_strategy="optim_grads_params",
        enable_async_reduce_grad=False,
    )
    opt = FullyShardV2MuonOptimizer(
        OptimizerConfig(lr=0.05, weight_decay=0.01),
        [model],
        lr=0.05,
        momentum=0.9,
        nesterov=False,
        weight_decay=0.01,
        num_ns_steps=5,
    )
    assert opt._momentum_buffers, "checkpoint test must manage at least one Muon param"
    return model, opt


def test_dist_muon_checkpoint_save_load_bitwise():
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    _, save_opt = _build_muon_checkpoint_optimizer(device)
    for param_idx, momentum_buffer in enumerate(save_opt._momentum_buffers):
        values = torch.arange(momentum_buffer.numel(), device=device, dtype=torch.float32)
        values = values.add_(rank * 1000 + param_idx * 17)
        momentum_buffer.copy_(values.to(momentum_buffer.dtype))
    expected = [momentum_buffer.clone() for momentum_buffer in save_opt._momentum_buffers]

    save_state = save_opt.sharded_state_dict(model_sharded_state_dict={})
    for param_name, param_state in save_state["state"].items():
        assert isinstance(param_state["momentum_buffer"], DTensor), param_name

    ckpt_dir = Path(SHARED_TMP_DIR) / "test_dist_muon_checkpoint_save_load_bitwise"
    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()

    dcp.save({"optimizer": save_state}, checkpoint_id=str(ckpt_dir))
    torch.distributed.barrier()

    _, load_opt = _build_muon_checkpoint_optimizer(device)
    for momentum_buffer in load_opt._momentum_buffers:
        momentum_buffer.fill_(-1)
    load_state = load_opt.sharded_state_dict(model_sharded_state_dict={}, is_loading=True)
    dcp.load({"optimizer": load_state}, checkpoint_id=str(ckpt_dir))
    load_opt.load_state_dict(load_state)

    for param_idx, (actual, ref) in enumerate(zip(load_opt._momentum_buffers, expected)):
        assert torch.equal(actual, ref), (
            f"Muon momentum checkpoint mismatch for param {param_idx}: "
            f"max abs diff {(actual - ref).abs().max().item() if actual.numel() else 0.0}"
        )

    torch.distributed.barrier()
    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    torch.distributed.barrier()


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("mesh_case", MESH_CASES)
@pytest.mark.parametrize("shapes", SHAPE_SETS)
def test_dist_muon_matches_reference(strategy, nesterov, mesh_case, shapes):
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    mesh = _make_mesh(mesh_case, device)

    # shapes (parametrized): 2D matrices Muon updates + a trailing 1D bias it skips
    # (``len(s) == 2`` tells them apart).
    lr, momentum, weight_decay, num_ns_steps = 0.1, 0.9, 0.01, 5

    # Identical init weights on every rank; fp32 master + fp32 grads -> fp32 update.
    torch.manual_seed(1234)
    params = [nn.Parameter(torch.randn(*s, dtype=torch.float32, device=device)) for s in shapes]
    pg = ParameterGroup(
        params=params,
        param_group_id=ParamGroupIdx(0, 0),
        mp_policy=MixedPrecisionPolicy(
            main_params_dtype=torch.float32, main_grads_dtype=torch.float32
        ),
        mesh=mesh,
        sharding_strategy=strategy,
    )
    # Pass only the 2D-matrix dist_params + their grad DTensors (in production the
    # FullyShardV2MuonOptimizer wrapper does this filtering). The 1D bias is excluded,
    # so the optimizer never touches it. grad DTensors may be None on ranks whose shard
    # of a param is empty.
    muon_params, muon_grads, muon_momentum_buffers = [], [], []
    for dp, dg in zip(pg.dist_params, pg.dist_grads):
        if dp.dim() == 2:
            muon_params.append(dp)
            muon_grads.append(dg)
            dtype = dg.dtype if dg is not None else dp.dtype
            muon_momentum_buffers.append(
                torch.zeros(dp.to_local().numel(), dtype=dtype, device=device)
            )
    opt = FullyShardV2Muon(
        muon_params,
        muon_grads,
        muon_momentum_buffers,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay,
        num_ns_steps=num_ns_steps,
    )

    # Reference, identical on every rank: a full master weight per param (the 1D
    # bias's stays frozen at init -> matching it asserts Muon skipped it) plus a
    # momentum buffer per param. Snapshot the inits to confirm matrices actually move.
    ref_w = [_read_full_weight(pg, i, s) for i, s in enumerate(shapes)]
    ref_buf = [torch.zeros(*s, dtype=torch.float32, device=device) for s in shapes]
    init_w = [w.clone() for w in ref_w]

    for step in range(3):
        # 1. Construct a full grad per param, identical across ranks (CPU RNG -> GPU).
        full_grads = []
        for i, s in enumerate(shapes):
            gen = torch.Generator(device="cpu").manual_seed(100 * step + i)
            full_grads.append(torch.randn(*s, generator=gen, dtype=torch.float32).to(device))

        # 2. Plant the full grads into the sharded grad buffer (each rank keeps its slice).
        for i, g in enumerate(full_grads):
            pg.main_grad_buffer.set_item(i, g)

        # 3. Reference: full-tensor Muon on every rank, 2D matrices only (the 1D bias
        #    is left frozen). pre_ns mirrors step()'s Phase 1: momentum + nesterov.
        for i, s in enumerate(shapes):
            if len(s) != 2:
                continue
            ref_buf[i].mul_(momentum).add_(full_grads[i])
            # Mirror step()'s exact ops so the comparison can be bit-exact: the
            # nesterov look-ahead is a single alpha-add (g + m*buf), not g + (m*buf).
            pre_ns = (
                full_grads[i].add(ref_buf[i], alpha=momentum)
                if nesterov
                else ref_buf[i].clone()
            )
            orth = _reference_orthogonalize(opt, pre_ns)
            if weight_decay != 0.0:
                ref_w[i].mul_(1.0 - lr * weight_decay)
            ref_w[i].add_(orth, alpha=-lr)

        # 4. Distributed Muon: gather -> NS on root -> scatter -> sharded update.
        opt.step()

        # 5. Compare every param exactly: 2D matrices match the updated reference,
        #    the 1D bias matches its untouched init.
        for i, s in enumerate(shapes):
            got = _read_full_weight(pg, i, s)
            assert torch.equal(got, ref_w[i]), (
                f"[{mesh_case}/{strategy}] param {i} shape {s} step {step}: max abs diff "
                f"{(got - ref_w[i]).abs().max().item():.3e}"
            )

    torch.distributed.barrier()

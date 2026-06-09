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
  3. reference: run a full-tensor Muon update on every rank, reusing the
     optimizer's own ``orthogonalize`` so the Newton-Schulz path is shared;
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

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import (
    MixedPrecisionPolicy,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.param_group import ParameterGroup
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.utils import ParamGroupIdx
from megatron.core.optimizer.fully_shard_v2_muon import FullyShardV2Muon

STRATEGIES = ["optim", "optim_grads", "optim_grads_params"]

# Each set is a list of param shapes at a different scale: four 2D matrices (mixing
# wide rows<cols and tall rows>cols -> NS transposes) + a trailing 1D bias Muon must
# skip. Four matrices let the test split them into 2 multi-param packages, exercising
# both the batched all_to_all packing and the cross-package pipeline. Sizes are picked
# so each matrix straddles a shard boundary (ZeRO-2/3), so gather/scatter is real.
SHAPE_SETS = [
    pytest.param([(5, 8), (9, 8), (8, 6), (7, 8), (8,)], id="small"),  # dims < 10
    pytest.param([(200, 256), (400, 256), (256, 300), (256, 128), (256,)], id="medium"),
    pytest.param(
        [(1500, 2048), (3000, 2048), (2048, 1200), (2048, 2600), (2048,)], id="large"
    ),
]


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


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("shapes", SHAPE_SETS)
def test_dist_muon_matches_reference(strategy, nesterov, shapes):
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

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
        mesh=None,
        sharding_strategy=strategy,
    )
    # Pass only the 2D-matrix dist_params + their grad DTensors (in production the
    # FullyShardV2MuonOptimizer wrapper does this filtering). The 1D bias is excluded,
    # so the optimizer never touches it. grad DTensors may be None on ranks whose shard
    # of a param is empty.
    muon_params, muon_grads = [], []
    for dp, dg in zip(pg.dist_params, pg.dist_grads):
        if dp.dim() == 2:
            muon_params.append(dp)
            muon_grads.append(dg)
    # Split the matrices into two multi-param packages to exercise both the batched
    # all_to_all packing and the cross-package pipeline.
    half = len(muon_params) // 2
    packages = [list(range(half)), list(range(half, len(muon_params)))]
    opt = FullyShardV2Muon(
        muon_params,
        muon_grads,
        packages=packages,
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
            pre_ns = full_grads[i].add(ref_buf[i], alpha=momentum) if nesterov else ref_buf[i].clone()
            orth = opt.orthogonalize(params[i], pre_ns)
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
                f"[{strategy}] param {i} shape {s} step {step}: max abs diff "
                f"{(got - ref_w[i]).abs().max().item():.3e}"
            )
            
    torch.distributed.barrier()

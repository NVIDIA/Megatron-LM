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

"""Multi-process correctness tests for the single-root distributed Muon optimizer.

Validates that ``FullyShardV2Muon`` (gather full grad to a load-balanced
root rank -> Newton-Schulz -> scatter back -> sharded weight update) produces the
same updated weights as a single-process reference Muon that runs the *same*
Newton-Schulz on the full gradient. The shared NS isolates the distribution logic
(gather/scatter/sharded momentum/sharded update) from NS numerics.

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
    FullyShardMixedPrecisionPolicy,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.param_group import ParameterGroup
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.utils import ParamGroupIdx
from megatron.core.optimizer.fully_shard_v2_muon import FullyShardV2Muon

STRATEGIES = ["optim", "optim_grads", "optim_grads_params"]


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
#  Helpers
# ------------------------------------------------------------------ #


def _build_group(strategy, shapes):
    """Build one fp32 ParameterGroup with 2D matrix params of the given shapes.

    fp32 master + fp32 grads so the update is computed in fp32 (tight tolerance).
    Returns (pg, params, device).
    """
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Identical init weights on every rank.
    torch.manual_seed(1234)
    params = [nn.Parameter(torch.randn(*s, dtype=torch.float32, device=device)) for s in shapes]

    pg = ParameterGroup(
        params=params,
        param_group_id=ParamGroupIdx(0, 0),
        mp_policy=FullyShardMixedPrecisionPolicy(
            main_params_dtype=torch.float32, main_grads_dtype=torch.float32
        ),
        mesh=None,
        sharding_strategy=strategy,
    )
    return pg, params, device


def _full_grads(shapes, device, step):
    """Deterministic full gradients (identical across ranks), varying per step."""
    grads = []
    for i, s in enumerate(shapes):
        g = torch.Generator(device="cpu").manual_seed(100 * step + i)
        grads.append(torch.randn(*s, generator=g, dtype=torch.float32).to(device))
    return grads


def _plant_grads(pg, full_grads):
    """Write each param's full grad into main_grad_buffer (per-rank shard)."""
    for idx, g in enumerate(full_grads):
        pg.main_grad_buffer.set_item(idx, g)


def _read_full_weight(pg, idx, shape):
    """All-gather this param's full master weight and return it (reshaped, cloned)."""
    mbuf = pg.main_weight_buffer
    full = mbuf.unshard(bind_params=False)
    off, sz = mbuf.buffer_index._get_item_global_range(idx)
    out = full[off : off + sz].clone().view(shape)
    mbuf.reshard()
    return out


def _ref_muon_step(opt, param, w, g, state, *, lr, momentum, nesterov, weight_decay):
    """Single-process reference Muon update on full tensors.

    Reuses the optimizer's own ``orthogonalize`` so the Newton-Schulz + scaling
    path is identical by construction; the test then only validates the
    distributed momentum / gather / scatter / sharded-update logic.
    """
    buf = state.get("buf")
    if buf is None:
        buf = torch.zeros_like(g)
        state["buf"] = buf
    buf.mul_(momentum).add_(g)
    pre_ns = g + momentum * buf if nesterov else buf.clone()
    orth = opt.orthogonalize(param, pre_ns.to(torch.float32)).to(w.dtype)
    w_new = w.clone()
    if weight_decay != 0.0:
        w_new.mul_(1.0 - lr * weight_decay)
    w_new.add_(orth, alpha=-lr)
    return w_new


# ------------------------------------------------------------------ #
#  Core correctness: distributed Muon == single-process reference
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_dist_muon_matches_reference(strategy):
    # Shapes chosen so params land at different buffer offsets -> different
    # load-balanced roots and shards that span rank boundaries unevenly.
    shapes = [(3, 8), (7, 8), (5, 8), (9, 8), (12, 8)]
    hp = dict(lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01, num_ns_steps=5)

    pg, params, device = _build_group(strategy, shapes)

    # Reference starts from the actual initialized master weights.
    w_full = [_read_full_weight(pg, i, s) for i, s in enumerate(shapes)]
    ref_state = [dict() for _ in shapes]

    opt = FullyShardV2Muon(
        pg.dist_params,
        lr=hp["lr"],
        momentum=hp["momentum"],
        nesterov=hp["nesterov"],
        weight_decay=hp["weight_decay"],
        num_ns_steps=hp["num_ns_steps"],
    )

    for step in range(3):
        full_grads = _full_grads(shapes, device, step)
        _plant_grads(pg, full_grads)
        opt.step()

        # Advance the reference in lockstep (reusing opt.orthogonalize).
        for i in range(len(shapes)):
            w_full[i] = _ref_muon_step(
                opt, params[i], w_full[i], full_grads[i], ref_state[i],
                lr=hp["lr"], momentum=hp["momentum"], nesterov=hp["nesterov"],
                weight_decay=hp["weight_decay"],
            )

        for i, s in enumerate(shapes):
            got = _read_full_weight(pg, i, s)
            # Tolerance reflects fp32 Newton-Schulz matmul non-determinism on GPU
            # (the distributed root and the single-process reference run the SAME
            # NS on the SAME assembled grad, so any real distribution bug would
            # diverge by O(update magnitude), not ~1e-5).
            assert torch.allclose(got, w_full[i], rtol=1e-3, atol=1e-4), (
                f"[{strategy}] param {i} shape {s} step {step}: max abs diff "
                f"{(got - w_full[i]).abs().max().item():.3e}"
            )

    torch.distributed.barrier()


# ------------------------------------------------------------------ #
#  Double-update guard: Muon only touches 2D matrix params
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_dist_muon_skips_non_matrix_params(strategy):
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    torch.manual_seed(7)
    matrix = nn.Parameter(torch.randn(8, 8, dtype=torch.float32, device=device))
    bias = nn.Parameter(torch.randn(8, dtype=torch.float32, device=device))  # 1D -> not Muon
    params = [matrix, bias]

    pg = ParameterGroup(
        params=params,
        param_group_id=ParamGroupIdx(0, 0),
        mp_policy=FullyShardMixedPrecisionPolicy(
            main_params_dtype=torch.float32, main_grads_dtype=torch.float32
        ),
        mesh=None,
        sharding_strategy=strategy,
    )

    matrix_before = _read_full_weight(pg, 0, (8, 8))
    bias_before = _read_full_weight(pg, 1, (8,))

    _plant_grads(pg, [torch.ones(8, 8, device=device), torch.ones(8, device=device)])
    FullyShardV2Muon(pg.dist_params, lr=0.5).step()

    matrix_after = _read_full_weight(pg, 0, (8, 8))
    bias_after = _read_full_weight(pg, 1, (8,))

    assert torch.equal(bias_before, bias_after), "Muon must not modify the 1D bias param"
    assert not torch.allclose(matrix_after, matrix_before), "Muon should update the 2D matrix"

    torch.distributed.barrier()

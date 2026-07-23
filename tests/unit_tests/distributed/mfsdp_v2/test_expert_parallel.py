# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP v2 composition with expert parallelism.

These exercise the ``(ep, fsdp)`` composition contract: ``fully_shard`` is handed
the full 2-D parallelism mesh, while ``Placements`` is 1-D and names only the
data-parallel (``fsdp``) axis. FSDP must shard and communicate over the ``fsdp``
axis alone, leaving the expert-parallel (``ep``) axis untouched.

The expert module is a real per-expert TransformerEngine ``GroupedLinear`` (one
weight tensor per local expert), sharded over the ``fsdp`` sub-mesh. Each fsdp rank
in a pair gets a distinct microbatch, so FSDP's expert-DP gradient averaging is
exercised for real; the sharded model is checked against an unsharded baseline that
trains on both microbatches with gradient averaging -- the true data-parallel
equivalent.
"""

import pytest
import torch
import transformer_engine.pytorch as te
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)

EP_SIZE = 4
FSDP_SIZE = 2
WORLD_SIZE = EP_SIZE * FSDP_SIZE

# Per-ep-rank grouped-expert GEMM shapes (dims are multiples of 16 for the grouped GEMM).
NUM_LOCAL_EXPERTS = 2
IN_FEATURES = 16
OUT_FEATURES = 32
TOKENS_PER_EXPERT = 4


def _dp_only_placements() -> Placements:
    """1-D placements that shard only over the ``fsdp`` axis of the 2-D mesh."""
    return Placements(
        dp_axes=["fsdp"],
        parameter=[Flat()],
        gradient=[Flat()],
        optimizer=[Flat()],
    )


def _build_expert(device: torch.device) -> "te.GroupedLinear":
    """Per-expert TE GroupedLinear: one plain ``weight{i}`` tensor per local expert."""
    return te.GroupedLinear(
        num_gemms=NUM_LOCAL_EXPERTS,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        params_dtype=torch.float32,
        device=device,
    )


def _expert_weights(layer: "te.GroupedLinear") -> list[torch.Tensor]:
    return [getattr(layer, f"weight{i}") for i in range(NUM_LOCAL_EXPERTS)]


def _grouped_forward(layer: "te.GroupedLinear", x: torch.Tensor) -> torch.Tensor:
    """Run the grouped GEMM with an even token split across local experts."""
    m_splits = [TOKENS_PER_EXPERT] * NUM_LOCAL_EXPERTS
    out = layer(x, m_splits)
    return out[0] if isinstance(out, tuple) else out


def test_expert_parallel_grouped_linear_matches_baseline(distributed_setup):
    """FSDP over the fsdp axis of an (ep, fsdp) mesh must match a data-parallel baseline."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size != WORLD_SIZE:
        pytest.skip(f"This test requires exactly {WORLD_SIZE} ranks (ep={EP_SIZE}, fsdp={FSDP_SIZE}).")
    if device.type != "cuda":
        pytest.skip("TransformerEngine GroupedLinear requires CUDA.")

    mesh = init_device_mesh(device.type, (EP_SIZE, FSDP_SIZE), mesh_dim_names=("ep", "fsdp"))
    # Experts are keyed by ep index: distinct across ep groups, identical within an
    # fsdp pair (both fsdp ranks host the same experts, which FSDP shards over fsdp).
    ep_index = mesh["ep"].get_local_rank()

    torch.manual_seed(1000 + ep_index)
    baseline = _build_expert(device)
    torch.manual_seed(1000 + ep_index)
    model = _build_expert(device)
    for baseline_weight, model_weight in zip(_expert_weights(baseline), _expert_weights(model)):
        baseline_weight.data.copy_(model_weight.data)

    fully_shard(model, mesh=mesh, placements=_dp_only_placements())

    # Structural: each expert weight shards over the size-2 fsdp sub-mesh, never the
    # 4-rank ep axis nor the full 8-rank mesh.
    for index, weight in enumerate(_expert_weights(model)):
        assert isinstance(weight, DTensor), f"weight{index} should be a DTensor after fully_shard."
        assert weight.device_mesh.size() == FSDP_SIZE, (
            f"weight{index} sharded over a mesh of size {weight.device_mesh.size()}, "
            f"expected the fsdp sub-mesh of size {FSDP_SIZE} (ep axis must be excluded)."
        )

    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    fsdp_index = mesh["fsdp"].get_local_rank()
    total_tokens = TOKENS_PER_EXPERT * NUM_LOCAL_EXPERTS
    # One DISTINCT microbatch per fsdp rank in the pair, so FSDP's expert-DP AVG
    # gradient reduction is exercised for real (not a no-op over identical data).
    # Both ranks generate the whole set deterministically from ep_index. The FSDP
    # model trains on this rank's microbatch; the unsharded baseline trains on the
    # whole set with gradient averaging -- the true data-parallel equivalent. Because
    # both apply the same averaged gradient every step, their weights stay in lockstep,
    # so the FSDP loss on this rank's microbatch must equal the baseline's loss on the
    # same microbatch. TE's grouped GEMM needs the input to require grad.
    microbatch_inputs, microbatch_targets = [], []
    for microbatch in range(FSDP_SIZE):
        torch.manual_seed(2000 + ep_index * FSDP_SIZE + microbatch)
        microbatch_inputs.append(
            torch.randn(total_tokens, IN_FEATURES, device=device, requires_grad=True)
        )
        microbatch_targets.append(torch.randn(total_tokens, OUT_FEATURES, device=device))

    def loss_on(module, x, target):
        return torch.nn.functional.mse_loss(_grouped_forward(module, x), target)

    def train_sharded() -> list[torch.Tensor]:
        x, target = microbatch_inputs[fsdp_index], microbatch_targets[fsdp_index]
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = loss_on(model, x, target)
            losses.append(loss.detach())
            loss.backward()  # FSDP AVG-reduces the gradient across the fsdp pair.
            optimizer.step()
        return losses

    def train_baseline() -> list[torch.Tensor]:
        losses = []
        for _ in range(5):
            baseline_optimizer.zero_grad()
            step_losses = []
            for x, target in zip(microbatch_inputs, microbatch_targets):
                loss = loss_on(baseline, x, target)
                step_losses.append(loss.detach())
                (loss / FSDP_SIZE).backward()  # accumulate, scaled to a mean.
            losses.append(step_losses[fsdp_index])
            baseline_optimizer.step()
        return losses

    sharded_losses = train_sharded()
    baseline_losses = train_baseline()

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        rtol=1e-4,
        atol=1e-5,
        msg="EP+FSDP sharded losses did not match the data-parallel unsharded baseline.",
    )

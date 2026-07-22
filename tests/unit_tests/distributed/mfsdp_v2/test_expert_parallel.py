# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP v2 composition with expert parallelism.

These exercise the ``(ep, fsdp)`` composition contract: ``fully_shard`` is handed
the full 2-D parallelism mesh, while ``Placements`` is 1-D and names only the
data-parallel (``fsdp``) axis. FSDP must shard and communicate over the ``fsdp``
axis alone, leaving the expert-parallel (``ep``) axis untouched.

The expert module is a real per-expert TransformerEngine ``GroupedLinear`` (one
weight tensor per local expert). Experts are distinct across ep groups and
identical within each fsdp pair, so single-rank SGD parity is the discriminating
check: if FSDP reduced gradients over the ``ep`` axis instead of ``fsdp``, the
distinct-per-ep data would make training diverge from the per-group baseline.
"""

import logging

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)

logger = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te

    HAVE_TE = True
except Exception:  # pragma: no cover - import guard
    te = None
    HAVE_TE = False

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


@pytest.mark.skipif(not HAVE_TE, reason="TransformerEngine is required.")
def test_expert_parallel_grouped_linear_matches_baseline(distributed_setup):
    """FSDP over the fsdp axis of an (ep, fsdp) mesh must match single-rank SGD."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size != WORLD_SIZE:
        pytest.skip(f"This test requires exactly {WORLD_SIZE} ranks (ep={EP_SIZE}, fsdp={FSDP_SIZE}).")
    if device.type != "cuda":
        pytest.skip("TransformerEngine GroupedLinear requires CUDA.")

    mesh = init_device_mesh(device.type, (EP_SIZE, FSDP_SIZE), mesh_dim_names=("ep", "fsdp"))
    # Experts (and data) are keyed by ep index: distinct across ep groups, identical
    # within an fsdp pair (both fsdp ranks share the same ep index).
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

    total_tokens = TOKENS_PER_EXPERT * NUM_LOCAL_EXPERTS
    torch.manual_seed(2000 + ep_index)
    # TE's grouped GEMM only builds the wgrad graph when the input requires grad
    # (as activations from prior layers do in a real model).
    x = torch.randn(total_tokens, IN_FEATURES, device=device, requires_grad=True)
    target = torch.randn(total_tokens, OUT_FEATURES, device=device)

    def train(module, module_optimizer, log_prefix) -> list[torch.Tensor]:
        losses = []
        for step in range(5):
            module_optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(_grouped_forward(module, x), target)
            losses.append(loss.detach())
            logger.debug(
                "%s parity: rank=%s, ep=%s, step=%s, loss=%s", log_prefix, rank, ep_index, step, loss
            )
            loss.backward()
            module_optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer, "Baseline")
    sharded_losses = train(model, optimizer, "EP-FSDP")

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        rtol=1e-4,
        atol=1e-5,
        msg="EP+FSDP per-expert sharded losses did not match single-rank baseline losses.",
    )

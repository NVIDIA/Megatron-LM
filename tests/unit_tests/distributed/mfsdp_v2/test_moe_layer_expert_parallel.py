# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP v2 composed with expert parallelism through a real MCore MoELayer.

Builds an ``EP=4`` ``MoELayer`` (router + all-to-all token dispatch + ``TEGroupedMLP``
experts) and shards its experts with mFSDP v2 over the ``fsdp`` (expert-data-parallel)
axis of a 2-D ``(ep, fsdp)`` mesh built to match ``parallel_state``'s expert groups.
Sharding the experts must be numerically transparent (matches an unsharded baseline)
and must place each expert weight on the size-2 expert-DP sub-mesh (ep excluded).
"""

import logging

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core import parallel_state
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)

try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except Exception:  # pragma: no cover - import guard
    HAVE_TE = False

try:
    from megatron.training.initialize import _set_random_seed
except Exception:  # pragma: no cover
    _set_random_seed = None

EP_SIZE = 4
FSDP_SIZE = 2
WORLD_SIZE = EP_SIZE * FSDP_SIZE
NUM_MOE_EXPERTS = 8  # 2 local experts per ep rank
HIDDEN = 16
FFN = 64
SEQ = 1
BATCH = 64


def _moe_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_MOE_EXPERTS,
        expert_model_parallel_size=EP_SIZE,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=FFN,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )


def _build_moe_layer(config: TransformerConfig) -> MoELayer:
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=NUM_MOE_EXPERTS, moe_grouped_gemm=True
        ).mlp
    )
    assert isinstance(submodules, MoESubmodules)
    return MoELayer(config, submodules)


def _build_expert_dp_mesh(device: torch.device):
    """2-D (fsdp, ep) mesh whose sub-meshes match parallel_state's expert groups."""
    mesh = init_device_mesh(device.type, (FSDP_SIZE, EP_SIZE), mesh_dim_names=("fsdp", "ep"))
    ep_ranks = sorted(dist.get_process_group_ranks(parallel_state.get_expert_model_parallel_group()))
    edp_ranks = sorted(dist.get_process_group_ranks(parallel_state.get_expert_data_parallel_group()))
    assert sorted(dist.get_process_group_ranks(mesh["ep"].get_group())) == ep_ranks, (
        "Mesh ep sub-group does not match parallel_state expert-model-parallel group; "
        "rank ordering assumption is wrong."
    )
    assert sorted(dist.get_process_group_ranks(mesh["fsdp"].get_group())) == edp_ranks, (
        "Mesh fsdp sub-group does not match parallel_state expert-data-parallel group."
    )
    return mesh


@pytest.mark.skipif(not HAVE_TE, reason="TransformerEngine is required.")
def test_moe_layer_experts_shard_over_expert_dp(distributed_setup):
    """Sharding a MoELayer's experts with mFSDP v2 is transparent and uses expert-DP."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size != WORLD_SIZE:
        pytest.skip(f"This test requires exactly {WORLD_SIZE} ranks (ep={EP_SIZE}, fsdp={FSDP_SIZE}).")
    if device.type != "cuda":
        pytest.skip("MoE GroupedMLP requires CUDA.")
    if _set_random_seed is None:
        pytest.skip("megatron.training is required for RNG setup.")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=EP_SIZE,
    )
    try:
        config = _moe_config()

        _set_random_seed(seed_=123, data_parallel_random_init=False)
        baseline = _build_moe_layer(config).cuda()
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        model = _build_moe_layer(config).cuda()
        # Identical starting point for the sharded and unsharded layers.
        for baseline_param, model_param in zip(baseline.parameters(), model.parameters()):
            baseline_param.data.copy_(model_param.data)

        mesh = _build_expert_dp_mesh(device)
        fully_shard(
            model.experts,
            mesh=mesh,
            placements=Placements(
                dp_axes=["fsdp"], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()]
            ),
        )

        # Structural: every expert parameter shards over the size-2 expert-DP sub-mesh.
        for name, param in model.experts.named_parameters():
            assert isinstance(param, DTensor), f"expert param {name!r} should be a DTensor."
            assert param.device_mesh.size() == FSDP_SIZE, (
                f"expert param {name!r} sharded over mesh of size {param.device_mesh.size()}, "
                f"expected the expert-DP sub-mesh of size {FSDP_SIZE}."
            )

        # foreach=False: after sharding only the experts, model.parameters() mixes plain
        # (router) tensors with sharded expert DTensors, which the fused foreach path
        # cannot batch together.
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.02, foreach=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=False)

        # Identical input on every rank so FSDP's expert-DP averaging is a no-op and
        # the sharded layer must match the unsharded baseline exactly.
        torch.manual_seed(4321)
        x = torch.randn(SEQ, BATCH, HIDDEN, device=device)
        target = torch.randn(SEQ, BATCH, HIDDEN, device=device)

        def train(layer, layer_optimizer, log_prefix) -> list[torch.Tensor]:
            losses = []
            for step in range(5):
                layer_optimizer.zero_grad()
                output = layer(x)
                output = output[0] if isinstance(output, tuple) else output
                loss = torch.nn.functional.mse_loss(output, target)
                losses.append(loss.detach())
                logger.debug("%s parity: step=%s, loss=%s", log_prefix, step, loss)
                loss.backward()
                layer_optimizer.step()
            return losses

        baseline_losses = train(baseline, baseline_optimizer, "Baseline")
        sharded_losses = train(model, optimizer, "EP-FSDP")

        torch.testing.assert_close(
            torch.stack(sharded_losses),
            torch.stack(baseline_losses),
            rtol=1e-4,
            atol=1e-5,
            msg="MoELayer expert-sharded losses did not match the unsharded baseline.",
        )
    finally:
        Utils.destroy_model_parallel()

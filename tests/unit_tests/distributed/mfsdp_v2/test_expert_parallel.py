# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP v2 composed with expert parallelism through a real MCore MoELayer.

Builds an ``EP=4`` ``MoELayer`` (router + all-to-all token dispatch + ``TEGroupedMLP``
experts) and shards its experts with mFSDP v2 over the ``dp`` (expert-data-parallel)
axis of a 2-D ``(dp, ep)`` mesh.
Sharding the experts must be numerically transparent (matches an unsharded baseline)
and must place each expert weight on the expert-DP sub-mesh (ep excluded).

Mesh topology and model shapes are test-local so different tests can pick different
``(ep, dp)`` splits.
"""


import dataclasses

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


@dataclasses.dataclass(frozen=True)
class ModelParallelSizes:
    """Parallelism sizes for the ``model_parallel`` fixture (each defaults to 1)."""

    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1


@pytest.fixture
def model_parallel(request, distributed_setup):
    """Set up and tear down model parallelism from a ``ModelParallelSizes`` request.param.

    The fixture skips when the world size is incompatible, yields the requested
    ``ModelParallelSizes`` and the resolved ``dp_size``, and tears down even if the test
    fails.
    """
    sizes: ModelParallelSizes = request.param
    non_dp = sizes.tp_size * sizes.pp_size * sizes.cp_size * sizes.ep_size
    if distributed_setup.world_size % non_dp != 0:
        pytest.skip(f"world_size {distributed_setup.world_size} is incompatible with {sizes}.")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=sizes.tp_size,
        pipeline_model_parallel_size=sizes.pp_size,
        context_parallel_size=sizes.cp_size,
        expert_model_parallel_size=sizes.ep_size,
    )
    dp_size = distributed_setup.world_size // non_dp
    yield sizes, dp_size
    Utils.destroy_model_parallel()


def _moe_config(
    num_routed_experts: int, expert_model_parallel_size: int, hidden_size: int, ffn_hidden_size: int
) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_moe_experts=num_routed_experts,
        expert_model_parallel_size=expert_model_parallel_size,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=ffn_hidden_size,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )


def _build_moe_layer(config: TransformerConfig) -> MoELayer:
    # Construct through HybridModel; the "E" layer in the pattern is a MoE layer whose
    # .mlp is a MoELayer (built internally via get_moe_module_spec). Return the extracted
    # MoELayer so the test shards just its experts.
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="E",
    )
    return model.decoder.layers[0].mlp


@pytest.mark.parametrize("model_parallel", [ModelParallelSizes(ep_size=2)], indirect=True)
def test_moe_layer_experts_shard_over_expert_dp(distributed_setup, model_parallel):
    """Sharding a MoELayer's experts with mFSDP v2 is transparent and uses expert-DP."""
    sizes, dp_size = model_parallel
    ep_size = sizes.ep_size
    num_routed_experts = 8  # num_routed_experts // ep local experts per ep rank
    hidden, ffn = 16, 64
    seq, batch = 1, 64
    device = distributed_setup.device

    config = _moe_config(num_routed_experts, ep_size, hidden, ffn)

    _set_random_seed(seed_=123, data_parallel_random_init=False)
    baseline = _build_moe_layer(config).cuda()
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    model = _build_moe_layer(config).cuda()
    # Identical starting point for the sharded and unsharded layers.
    for baseline_param, model_param in zip(baseline.parameters(), model.parameters()):
        baseline_param.data.copy_(model_param.data)

    # (dp, ep) with ep innermost matches MCore's default ep-fastest expert layout.
    mesh = init_device_mesh(device.type, (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    fully_shard(
        model.experts,
        mesh=mesh,
        placements=Placements(
            dp_axes=["dp"], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()]
        ),
    )

    # With ep innermost in the (dp, ep) mesh, this rank's expert-DP group is the set of
    # ranks sharing its ep index: {ep_idx, ep_idx + ep_size, ...}.
    ep_idx = distributed_setup.rank % ep_size
    expected_dp_ranks = list(range(ep_idx, distributed_setup.world_size, ep_size))

    # Structural: every expert parameter shards over exactly this expert-DP group -- the
    # ep axis is excluded and left to EP's alltoall dispatch.
    for name, param in model.experts.named_parameters():
        assert isinstance(param, DTensor), f"expert param {name!r} should be a DTensor."
        assert param.device_mesh.mesh.tolist() == expected_dp_ranks, (
            f"expert param {name!r} sharded over ranks {param.device_mesh.mesh.tolist()}, "
            f"expected the expert-DP group {expected_dp_ranks} (ep axis excluded)."
        )

    # foreach=False: after sharding only the experts, model.parameters() mixes plain
    # (router) tensors with sharded expert DTensors, which the fused foreach path cannot
    # batch together.
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.02, foreach=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=False)

    # Identical input on every rank. mFSDP shards only the experts, so with matching data
    # the expert-DP gradient reduction is a no-op and the sharded layer must reproduce the
    # unsharded one exactly. This checks that mFSDP's all-gather / reduce-scatter / DTensor
    # plumbing is numerically transparent inside a real MoE layer (router + all-to-all
    # dispatch + TEGroupedMLP).
    torch.manual_seed(4321)
    x = torch.randn(seq, batch, hidden, device=device)
    target = torch.randn(seq, batch, hidden, device=device)

    def train(layer, layer_optimizer) -> list[torch.Tensor]:
        losses = []
        for _ in range(5):
            layer_optimizer.zero_grad()
            # MoELayer.forward returns (output, mlp_bias); mlp_bias is None here.
            output, _ = layer(x)
            loss = torch.nn.functional.mse_loss(output, target)
            losses.append(loss.detach())
            loss.backward()
            layer_optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer)
    sharded_losses = train(model, optimizer)

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        rtol=1e-4,
        atol=1e-5,
        msg="MoELayer expert sharding was not numerically transparent vs the unsharded layer.",
    )

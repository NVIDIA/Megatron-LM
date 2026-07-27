# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP v2 tensor-parallel composability.

These tests establish the DP+TP mesh-ownership contract on a DP=2, TP=2 mesh:

- ``DBuffer`` owns only the data-parallel submesh.
- ``FsdpParameterGroup`` owns the full user-provided mesh and builds
  optimizer-facing DTensors that carry both DP and TP placements.
- The emitted DTensor placement tuple follows the user-provided mesh-axis order.

Run under torchrun with four ranks, for example::

    torchrun --nproc-per-node 4 -m pytest -q \
        tests/unit_tests/distributed/mfsdp_v2/test_tensor_parallel.py
"""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    set_tensor_model_parallel_attributes,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class TpModule(nn.Module):
    """Module holding column-, row-, and replicated-parallel parameters.

    Parameters are stored TP-local (as classic MCore / TE parameters are) and
    carry the MCore tensor-parallel attributes that ``fully_shard`` infers
    placement from. The module never runs forward; these tests only inspect the
    sharded-parameter and optimizer-state metadata that ``fully_shard`` installs.
    """

    tp_size = 2
    # Global (un-TP-sharded) parameter shapes.
    global_shapes = {
        "column_weight": torch.Size((16, 8)),  # column-parallel: shard out (dim 0)
        "row_weight": torch.Size((8, 8)),  # row-parallel: shard in (dim 1)
        "bias": torch.Size((8,)),  # replicated across TP
    }

    def __init__(self) -> None:
        super().__init__()
        column = self.global_shapes["column_weight"]
        self.column_weight = nn.Parameter(torch.randn(column[0] // self.tp_size, column[1]))
        set_tensor_model_parallel_attributes(self.column_weight, is_parallel=True, dim=0, stride=1)

        row = self.global_shapes["row_weight"]
        self.row_weight = nn.Parameter(torch.randn(row[0], row[1] // self.tp_size))
        set_tensor_model_parallel_attributes(self.row_weight, is_parallel=True, dim=1, stride=1)

        # bias carries no tensor-parallel attributes, so it is replicated over TP.
        self.bias = nn.Parameter(torch.randn(self.global_shapes["bias"]))


def _dp_placements() -> Placements:
    return Placements(dp_axes=["dp"], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _build_sharded_model(device, mesh_dim_names: tuple[str, str]) -> tuple[nn.Module, object]:
    """Shard a fresh TpModule on a DP=2, TP=2 mesh with the given axis order."""
    mesh = init_device_mesh(device.type, (2, 2), mesh_dim_names=mesh_dim_names)
    torch.manual_seed(2026)
    model = TpModule().to(device)
    fully_shard(model, mesh=mesh, placements=_dp_placements())
    return model, mesh


def _sharded_parameter(model: nn.Module, name: str) -> nn.Parameter:
    (group,) = model.parameter_groups
    index = group.parameter_names.index(name)
    return group.sharded_parameters[index]


def _skip_unless_four_ranks(distributed_setup) -> None:
    if distributed_setup.world_size != 4:
        pytest.skip("This test requires exactly 4 ranks for a DP=2, TP=2 mesh.")
    if distributed_setup.device.type != "cuda":
        pytest.skip("DTensor.from_local for these tests requires CUDA ranks.")


def test_dbuffer_mesh_is_dp_only(distributed_setup):
    """The group owns the full mesh (4 ranks); its DBuffers own the DP submesh (2 ranks)."""
    _skip_unless_four_ranks(distributed_setup)
    device = distributed_setup.device

    for mesh_dim_names in (("dp", "tp"), ("tp", "dp")):
        model, mesh = _build_sharded_model(device, mesh_dim_names)
        (group,) = model.parameter_groups

        assert mesh.size() == 4
        # The parameter group keeps the full user mesh.
        assert group.mesh is mesh
        # Every DBuffer owns only the DP submesh.
        assert group.main_weight.mesh.ndim == 1
        assert group.main_weight.mesh.size() == 2
        assert group.main_grad is not None
        assert group.main_grad.mesh.size() == 2


def test_full_mesh_parameter_placements_dp_tp(distributed_setup):
    """Sharded parameters carry both DP and TP placements in [dp, tp] mesh order."""
    _skip_unless_four_ranks(distributed_setup)
    device = distributed_setup.device
    model, mesh = _build_sharded_model(device, ("dp", "tp"))

    expected = {
        # DP is axis 0, TP is axis 1.
        "column_weight": (Shard(0), Shard(0)),  # DP shard(0), TP shard(0)
        "row_weight": (Shard(0), Shard(1)),  # DP shard(0), TP shard(1)
        "bias": (Shard(0), Replicate()),  # DP shard(0), TP replicate
    }
    for name, expected_placements in expected.items():
        parameter = _sharded_parameter(model, name)
        assert isinstance(parameter.data, DTensor)
        assert parameter.device_mesh == mesh
        assert parameter.placements == expected_placements, name


def test_full_mesh_parameter_placements_tp_dp(distributed_setup):
    """Reversing the mesh-axis order reverses the DTensor placement tuple."""
    _skip_unless_four_ranks(distributed_setup)
    device = distributed_setup.device
    model, mesh = _build_sharded_model(device, ("tp", "dp"))

    expected = {
        # TP is axis 0, DP is axis 1.
        "column_weight": (Shard(0), Shard(0)),
        "row_weight": (Shard(1), Shard(0)),
        "bias": (Replicate(), Shard(0)),
    }
    for name, expected_placements in expected.items():
        parameter = _sharded_parameter(model, name)
        assert parameter.device_mesh == mesh
        assert parameter.placements == expected_placements, name


def test_full_mesh_global_shape_and_stride(distributed_setup):
    """Sharded parameters reconstruct the global shape and contiguous stride."""
    _skip_unless_four_ranks(distributed_setup)
    device = distributed_setup.device
    model, _ = _build_sharded_model(device, ("dp", "tp"))

    for name, global_shape in TpModule.global_shapes.items():
        parameter = _sharded_parameter(model, name)
        assert parameter.shape == global_shape, name
        expected_stride = torch.empty(global_shape).stride()
        assert parameter.stride() == expected_stride, name


def test_optimizer_state_lives_on_full_mesh(distributed_setup):
    """Adam state should inherit the parameter's full mesh and placements."""
    _skip_unless_four_ranks(distributed_setup)
    device = distributed_setup.device
    model, mesh = _build_sharded_model(device, ("dp", "tp"))

    parameters = list(model.parameters())
    # No forward here; feed each sharded parameter a matching DTensor gradient so
    # a single Adam step exercises optimizer-state materialization.
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    before = [parameter.to_local().detach().clone() for parameter in parameters]
    reference = [tensor.clone() for tensor in before]

    # foreach=False forces the per-parameter path so Adam never buckets sharded
    # parameters with different TP placements into one _foreach call.
    optimizer = torch.optim.Adam(parameters, lr=0.1, foreach=False)
    optimizer.step()

    for parameter in parameters:
        state = optimizer.state[parameter]
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state[key]
            assert isinstance(moment, DTensor), key
            assert moment.device_mesh == mesh
            assert moment.placements == parameter.placements

    # One elementwise Adam step on the DTensor must match the same step applied to
    # a plain local tensor with an all-ones gradient (unsharded local reference).
    reference_optimizer = torch.optim.Adam(
        [torch.nn.Parameter(tensor) for tensor in reference], lr=0.1, foreach=False
    )
    for reference_parameter in reference_optimizer.param_groups[0]["params"]:
        reference_parameter.grad = torch.ones_like(reference_parameter)
    reference_optimizer.step()

    for parameter, before_tensor, reference_parameter in zip(
        parameters, before, reference_optimizer.param_groups[0]["params"]
    ):
        after = parameter.to_local()
        # Flat DP packing gives some ranks no shard of a given tensor (dp rank 0
        # here owns only column_weight), and an empty shard has nothing to update.
        if after.numel():
            assert not torch.equal(after, before_tensor)
        torch.testing.assert_close(after, reference_parameter.detach())


class TpMlp(nn.Module):
    """A column-parallel -> GELU -> row-parallel MLP of real MCore TP layers.

    The tensor-parallel layers carry out their own TP collectives (column has no
    forward all-reduce; row all-reduces its output), so this computes the same
    function as the unsharded ``RefMlp`` below, which is what makes a loss-curve
    comparison meaningful.
    """

    def __init__(self, config: TransformerConfig, tp_group, hidden: int, ffn: int) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=hidden,
            output_size=ffn,
            config=config,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            tp_group=tp_group,
        )
        self.fc2 = RowParallelLinear(
            input_size=ffn,
            output_size=hidden,
            config=config,
            init_method=config.init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=tp_group,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tensor-parallel MLP; the linears return ``(output, bias)``."""
        hidden, _ = self.fc1(x)
        hidden = torch.nn.functional.gelu(hidden)
        output, _ = self.fc2(hidden)
        return output


class RefMlp(nn.Module):
    """A plain, un-tensor-parallel MLP matching ``TpMlp``'s full weights."""

    def __init__(self, hidden: int, ffn: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn)
        self.fc2 = nn.Linear(ffn, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the reference MLP."""
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


def _all_gather_cat(local: torch.Tensor, group, dim: int) -> torch.Tensor:
    """All-gather a tensor-parallel shard across ``group`` and concatenate on ``dim``."""
    shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, local.contiguous(), group=group)
    return torch.cat(shards, dim=dim)


def test_fsdp_tp_loss_curve_matches_unsharded(distributed_setup):
    """FSDP+TP training should track an unsharded single-replica reference loss curve."""
    if distributed_setup.world_size != 4:
        pytest.skip("This test requires exactly 4 ranks for a DP=2, TP=2 mesh.")
    device = distributed_setup.device
    if device.type != "cuda":
        pytest.skip("MCore tensor-parallel layers require CUDA ranks.")

    tp_size, dp_size = 2, 2
    hidden, ffn, batch, steps = 8, 16, 4, 5

    # init_device_mesh lays ranks out row-major (tp innermost), so the tp axis is a
    # contiguous 2-rank group -- exactly the tensor-parallel group the TP layers use.
    # Pass those groups explicitly (via ProcessGroupCollection) instead of mutating
    # global parallel_state, so this test never depends on MPU init.
    mesh = init_device_mesh(device.type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = mesh.get_group("tp")
    pg_collection.dp = mesh.get_group("dp")

    # CPU initialization derives the shard rank/size from tp_group and needs no
    # model-parallel RNG tracker. With the same seed on every rank, each rank builds
    # the identical full master weight and keeps its own slice, so all-gathering the
    # slices reconstructs the full weight for the reference.
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=1,
        tensor_model_parallel_size=tp_size,
        # FSDP tracks gradients through .grad, not a fused weight.main_grad.
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
        use_cpu_initialization=True,
    )

    torch.manual_seed(1234)
    model = TpMlp(config, pg_collection.tp, hidden, ffn).to(device)

    # Rebuild the full (un-TP-sharded) weights before FSDP shards across DP, and
    # load them into the reference so both models start bit-identical.
    reference = RefMlp(hidden, ffn).to(device)
    with torch.no_grad():
        reference.fc1.weight.copy_(_all_gather_cat(model.fc1.weight, pg_collection.tp, dim=0))
        reference.fc1.bias.copy_(_all_gather_cat(model.fc1.bias, pg_collection.tp, dim=0))
        reference.fc2.weight.copy_(_all_gather_cat(model.fc2.weight, pg_collection.tp, dim=1))
        # Row-parallel bias is added after the output all-reduce, so it is replicated.
        reference.fc2.bias.copy_(model.fc2.bias)

    fully_shard(model.fc1, mesh=mesh, placements=_dp_placements())
    fully_shard(model.fc2, mesh=mesh, placements=_dp_placements())

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

    # Every rank sees the same input: TP ranks must (TP replicates the input), and
    # giving every DP rank the same data makes the DP-averaged gradient equal the
    # single-replica gradient, so each step's loss matches the reference.
    torch.manual_seed(2024)
    x = torch.randn(steps, batch, hidden, device=device)
    target = torch.randn(steps, batch, hidden, device=device)

    def train(module, module_optimizer):
        losses = []
        for step in range(steps):
            module_optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(module(x[step]), target[step])
            loss.backward()
            module_optimizer.step()
            losses.append(loss.detach())
        return losses

    reference_losses = train(reference, reference_optimizer)
    fsdp_losses = train(model, optimizer)

    torch.testing.assert_close(
        torch.stack(fsdp_losses),
        torch.stack(reference_losses),
        rtol=1e-3,
        atol=1e-4,
        msg="FSDP+TP loss curve did not match the unsharded reference.",
    )

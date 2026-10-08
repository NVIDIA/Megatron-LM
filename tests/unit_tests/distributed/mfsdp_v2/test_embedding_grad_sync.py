"""Embedding synchronization across stages with different MFSDP buffer layouts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard

from megatron.core.distributed.finalize_model_grads import _allreduce_fsdp_embedding_grad
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer


@pytest.mark.parametrize("replicated", [False, True])
def test_local_embedding_grad_uses_buffer_layout(distributed_setup, replicated):
    """Stage-local shard offsets do not change the logical rows being reduced."""
    setup = distributed_setup
    if setup.world_size != 4:
        pytest.skip("Requires four ranks for two pipeline stages and two DP ranks.")
    mesh = init_device_mesh(setup.device.type, (2, 2), mesh_dim_names=("pp", "dp"))
    stage = mesh.get_local_rank("pp")
    prefix = torch.full((1 + 6 * stage, 4), 17.0, device=setup.device)
    embedding = torch.arange(28, dtype=torch.float32, device=setup.device).reshape(7, 4)
    stage_grad = embedding + stage * 100
    placement = Replicate() if replicated else Shard(0)
    grad_buffer = DBuffer.distribute_tensors([prefix, stage_grad], mesh["dp"], [placement])
    weight = nn.Parameter(torch.zeros_like(grad_buffer.get_tensor_view(1)))
    weight.grad = grad_buffer.get_tensor_view(1)
    original_grad = weight.grad
    original_prefix = grad_buffer.get_tensor_view(0).clone()
    parameter_group = SimpleNamespace(
        pre_optimizer_main_grad=grad_buffer,
        fsdp_parameters=[SimpleNamespace(sharded=None), SimpleNamespace(sharded=weight)],
    )
    weight._mfsdp_parameter_group = lambda: parameter_group

    _allreduce_fsdp_embedding_grad(weight, weight.grad, mesh.get_group("pp"))

    assert weight.grad is original_grad
    full_buffer = grad_buffer.redistribute([Replicate()])
    torch.testing.assert_close(full_buffer.get_tensor_view(1), 2 * embedding + 100)
    torch.testing.assert_close(grad_buffer.get_tensor_view(0), original_prefix)

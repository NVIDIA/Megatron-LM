from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle
from tests.unit_tests.test_utilities import Utils


pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
    pytest.mark.xfail(
        reason="MLite distopt checkpoint continuity is covered by a follow-up bugfix PR.",
        strict=True,
    ),
]


class TinyDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_distopt():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for distopt smoke tests.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    Utils.set_world_size(
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _build_model_and_distopt():
    torch.manual_seed(2468)
    model = TinyDense().bfloat16().cuda()
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    wrapped = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        ddp_config,
        model,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(optimizer="adam", lr=1.0e-3, bf16=True, use_distributed_optimizer=True),
        [wrapped],
    )
    return wrapped, optimizer


def _train_step(model, optimizer, x: torch.Tensor):
    output = model(x)
    loss = output.float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if hasattr(model, "zero_grad_buffer"):
        model.zero_grad_buffer()
    return loss.detach()


def _local_named_params(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().float().clone() for name, param in model.named_parameters()}


def _assert_model_close(lhs, rhs):
    lhs_params = _local_named_params(lhs)
    rhs_params = _local_named_params(rhs)
    assert lhs_params.keys() == rhs_params.keys()
    for name in lhs_params:
        torch.testing.assert_close(lhs_params[name], rhs_params[name], atol=0.0, rtol=0.0)


def test_distopt_checkpoint_load_matches_uninterrupted_training_single_node(tmp_path):
    model_for_ckpt, optimizer_for_ckpt = _build_model_and_distopt()
    direct_model, direct_optimizer = _build_model_and_distopt()
    loaded_model, loaded_optimizer = _build_model_and_distopt()
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    torch.manual_seed(1357)
    x0 = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    x1 = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)

    _train_step(model_for_ckpt, optimizer_for_ckpt, x0)
    _train_step(direct_model, direct_optimizer, x0)

    runtime.save_checkpoint(
        ModelHandle(
            model=model_for_ckpt,
            optimizer=optimizer_for_ckpt,
            _extras={"model_chunks": [model_for_ckpt]},
        ),
        str(tmp_path),
        step=1,
    )
    assert runtime.load_checkpoint(
        ModelHandle(
            model=loaded_model,
            optimizer=loaded_optimizer,
            _extras={"model_chunks": [loaded_model]},
        ),
        str(tmp_path),
    ) == 1

    _train_step(direct_model, direct_optimizer, x1)
    _train_step(loaded_model, loaded_optimizer, x1)
    _assert_model_close(direct_model, loaded_model)

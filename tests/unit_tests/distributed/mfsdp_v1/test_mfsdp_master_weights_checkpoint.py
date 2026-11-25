# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.fp8_utils import HAVE_TE
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import is_te_min_version, is_torch_min_version
from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict
from tests.unit_tests.test_utilities import Utils


class TinySwiGLUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.linear_fc1 = torch.nn.Linear(16, 32, bias=False)

    def forward(self, inputs):
        gate, value = self.mlp.linear_fc1(inputs).chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * value


@pytest.mark.skipif(
    not is_torch_min_version("2.4.0"), reason="Megatron FSDP requires torch >= 2.4.0"
)
@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("2.1.0")), reason="Remainders require TE >= 2.1"
)
class TestMasterWeightsCheckpoint:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_model_and_optimizer(zero_init=False):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=1,
            bf16=True,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
        )
        module = TinySwiGLUModel().cuda()
        if zero_init:
            for parameter in module.parameters():
                torch.nn.init.zeros_(parameter)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=None,
                megatron_fsdp_use_decoupled_grad=True,
            ),
            module=Float16Module(config, module),
            fsdp_unit_modules=[TinySwiGLUModel],
            pg_collection=pg_collection,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                lr=1e-3,
                bf16=True,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=True,
            ),
            [model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )
        optimizer.zero_grad()
        model(torch.zeros(1, 16, device="cuda", dtype=torch.bfloat16)).sum().backward()
        model.finish_grad_sync()
        optimizer.step()
        return model, optimizer

    def test_save_load(self, tmp_path_dist_ckpt):
        torch.manual_seed(1234)
        model, optimizer = self._build_model_and_optimizer()
        restored_model = None
        try:
            assert optimizer.optimizer.master_weights
            assert optimizer.state
            for state in optimizer.state.values():
                remainder = state["master_param"].to_local()
                assert remainder.dtype == torch.int16
                remainder.random_(1, 32768)

            saved_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.sharded_param_state_fsdp_dtensor(),
            }
            checkpoint_state = preprocess_fsdp_dtensor_state_dict(
                SimpleNamespace(swiglu=True, num_experts=None), saved_state, model.module
            )
            for state in checkpoint_state["optimizer"]["state"].values():
                assert state["master_param"].shape == state["exp_avg"].shape
            dcp.save(checkpoint_state, checkpoint_id=tmp_path_dist_ckpt)

            restored_model, restored_optimizer = self._build_model_and_optimizer(zero_init=True)
            restored_state = {
                "model": restored_model.state_dict(),
                "optimizer": restored_optimizer.sharded_param_state_fsdp_dtensor(),
            }
            for parameter in restored_state["model"].values():
                assert torch.count_nonzero(parameter.to_local()) == 0
            for state in restored_optimizer.state.values():
                assert torch.count_nonzero(state["master_param"].to_local()) == 0

            dcp.load(
                preprocess_fsdp_dtensor_state_dict(
                    SimpleNamespace(swiglu=True, num_experts=None),
                    restored_state,
                    restored_model.module,
                ),
                checkpoint_id=tmp_path_dist_ckpt,
            )
            restored_model.load_state_dict(restored_state["model"])
            restored_optimizer.load_state_dict(restored_state["optimizer"])

            actual_weights = restored_model.state_dict()
            assert actual_weights.keys() == saved_state["model"].keys()
            for name, expected in saved_state["model"].items():
                actual = actual_weights[name]
                torch.testing.assert_close(actual.to_local(), expected.to_local(), rtol=0, atol=0)
            actual_states = restored_optimizer.sharded_param_state_fsdp_dtensor()["state"]
            assert actual_states.keys() == saved_state["optimizer"]["state"].keys()
            for name, expected_state in saved_state["optimizer"]["state"].items():
                torch.testing.assert_close(
                    actual_states[name]["master_param"].to_local(),
                    expected_state["master_param"].to_local(),
                    rtol=0,
                    atol=0,
                )
        finally:
            model.stop_communication()
            if restored_model is not None:
                restored_model.stop_communication()

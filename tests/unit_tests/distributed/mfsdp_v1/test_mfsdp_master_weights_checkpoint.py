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
from megatron.training.checkpointing import (
    _load_base_checkpoint,
    preprocess_fsdp_dtensor_state_dict,
)
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
    def _build_model_and_optimizer(
        main_params_dtype,
        store_param_remainders,
        zero_init=False,
        model_dtype=torch.bfloat16,
        master_weights=None,
    ):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=1,
            bf16=model_dtype == torch.bfloat16,
            fp16=model_dtype == torch.float16,
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
                megatron_fsdp_main_grads_dtype=torch.float32,
            ),
            module=Float16Module(config, module),
            fsdp_unit_modules=[TinySwiGLUModel],
            pg_collection=pg_collection,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                lr=1e-3,
                bf16=model_dtype == torch.bfloat16,
                fp16=model_dtype == torch.float16,
                loss_scale=1.0 if model_dtype == torch.float16 else None,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=True,
                main_params_dtype=main_params_dtype,
                store_param_remainders=store_param_remainders,
            ),
            [model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )
        if master_weights is not None:
            optimizer.optimizer.master_weights = master_weights
        optimizer.zero_grad()
        model(torch.zeros(1, 16, device="cuda", dtype=model_dtype)).sum().backward()
        model.finish_grad_sync()
        optimizer.step()
        return model, optimizer

    @staticmethod
    def _master_values(optimizer):
        # Materialize the distributed parameters used by checkpoint name mapping.
        for model in optimizer.model_chunks:
            model.state_dict()
        # Clone local values so subsequent updates cannot change the reference.
        values = {}
        for param in optimizer.state:
            value = optimizer.optimizer.get_unscaled_state(param, "master_param")
            if hasattr(value, "to_local"):
                value = value.to_local()
            values[optimizer._param_name(param)] = value.clone()
        return values

    @staticmethod
    def _step_with_fixed_gradient(model, optimizer):
        optimizer.zero_grad()
        # Reuse FSDP's gradient allocation and synchronization before installing the
        # same nonzero gradient in both the uninterrupted and resumed optimizers.
        model(
            torch.ones(1, 16, device="cuda", dtype=next(model.parameters()).dtype)
        ).sum().backward()
        model.finish_grad_sync()
        for group in optimizer.optimizer.param_groups:
            for param in group["params"]:
                param.decoupled_grad.fill_(0.125)
        optimizer.step()

    @staticmethod
    def _checkpoint_args():
        return SimpleNamespace(
            swiglu=True,
            num_experts=None,
            use_megatron_fsdp=True,
            non_persistent_global_ckpt_dir=None,
            non_persistent_ckpt_type=None,
            strict_fsdp_dtensor_load=True,
            dist_ckpt_strictness="raise_unexpected",
            finetune=False,
        )

    @staticmethod
    def _save_checkpoint(tmp_path_dist_ckpt, checkpoint_state):
        checkpoint_dir = tmp_path_dist_ckpt / "iter_0000001"
        dcp.save(checkpoint_state, checkpoint_id=checkpoint_dir)
        if Utils.rank == 0:
            (tmp_path_dist_ckpt / "latest_checkpointed_iteration.txt").write_text("1")
        torch.distributed.barrier()
        return dcp.FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata

    def test_scaled_fp16_masters_fall_back_to_model_weights(self):
        with pytest.warns(UserWarning, match="disables TE master weights"):
            model, optimizer = self._build_model_and_optimizer(torch.float16, False)
        try:
            assert not optimizer.optimizer.master_weights
            self._step_with_fixed_gradient(model, optimizer)
            assert optimizer.state
            assert all("master_param" not in state for state in optimizer.state.values())
        finally:
            model.stop_communication()

    @pytest.mark.parametrize(
        "model_dtype,store_param_remainders",
        [
            pytest.param(torch.float16, False, id="fp16_fp32_master"),
            pytest.param(torch.bfloat16, False, id="bf16_fp32_master"),
            pytest.param(torch.bfloat16, True, id="bf16_remainder"),
        ],
    )
    def test_save_load(self, tmp_path_dist_ckpt, model_dtype, store_param_remainders):
        torch.manual_seed(1234)
        model, optimizer = self._build_model_and_optimizer(
            torch.float32, store_param_remainders, model_dtype=model_dtype
        )
        restored_model = None
        try:
            assert optimizer.optimizer.master_weights
            assert optimizer.state
            for state in optimizer.state.values():
                master = state["master_param"].to_local()
                if store_param_remainders:
                    assert master.dtype == torch.int16
                    master.random_(1, 32768)
                else:
                    assert master.dtype == torch.float32
                    master.add_(0.00001)

            saved_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.sharded_param_state_fsdp_dtensor(),
            }
            args = self._checkpoint_args()
            checkpoint_state = preprocess_fsdp_dtensor_state_dict(args, saved_state, model.module)
            for state in checkpoint_state["optimizer"]["state"].values():
                assert state["master_param"].shape == state["exp_avg"].shape
            metadata = self._save_checkpoint(tmp_path_dist_ckpt, checkpoint_state)
            assert any(key.endswith(".master_param") for key in metadata)

            restored_model, restored_optimizer = self._build_model_and_optimizer(
                torch.float32, store_param_remainders, zero_init=True, model_dtype=model_dtype
            )
            restored_state = {
                "model": restored_model.state_dict(),
                "optimizer": restored_optimizer.sharded_param_state_fsdp_dtensor(is_loading=True),
                "_model": [restored_model.module],
            }
            for parameter in restored_state["model"].values():
                assert torch.count_nonzero(parameter.to_local()) == 0
            for state in restored_optimizer.state.values():
                assert torch.count_nonzero(state["master_param"].to_local()) == 0

            restored_state, _, _, _ = _load_base_checkpoint(
                str(tmp_path_dist_ckpt), args, sharded_state_dict=restored_state
            )
            assert "_fsdp_missing_master_params" not in restored_state["optimizer"]
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

    @pytest.mark.parametrize(
        "model_dtype,store_param_remainders",
        [
            pytest.param(torch.float16, False, id="fp16_fp32_master"),
            pytest.param(torch.bfloat16, False, id="bf16_fp32_master"),
            pytest.param(torch.bfloat16, True, id="bf16_remainder"),
        ],
    )
    def test_resume_without_master(self, tmp_path_dist_ckpt, model_dtype, store_param_remainders):
        models = []
        try:
            # Reproduce the old optimizer configuration, not just a new checkpoint
            # with one key removed. Populate moments with a real nonzero update.
            torch.manual_seed(1234)
            source, source_optimizer = self._build_model_and_optimizer(
                torch.float32, False, model_dtype=model_dtype, master_weights=False
            )
            models.append(source)
            self._step_with_fixed_gradient(source, source_optimizer)
            legacy_state = {
                "model": source.state_dict(),
                "optimizer": source_optimizer.sharded_param_state_fsdp_dtensor(),
            }
            assert all(
                "master_param" not in state for state in legacy_state["optimizer"]["state"].values()
            )
            checkpoint_dir = tmp_path_dist_ckpt / "iter_0000001"
            args = SimpleNamespace(
                swiglu=True,
                num_experts=None,
                use_megatron_fsdp=True,
                non_persistent_global_ckpt_dir=None,
                non_persistent_ckpt_type=None,
                strict_fsdp_dtensor_load=False,
                dist_ckpt_strictness="raise_unexpected",
                finetune=False,
            )
            dcp.save(
                preprocess_fsdp_dtensor_state_dict(args, legacy_state, source.module),
                checkpoint_id=checkpoint_dir,
            )
            metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata
            assert not any("master_param" in key for key in metadata)
            if Utils.rank == 0:
                (tmp_path_dist_ckpt / "latest_checkpointed_iteration.txt").write_text("1")
            torch.distributed.barrier()

            torch.manual_seed(5678)
            restored, restored_optimizer = self._build_model_and_optimizer(
                torch.float32, store_param_remainders, model_dtype=model_dtype
            )
            models.append(restored)
            initial_weights = restored.state_dict()
            assert any(
                not torch.equal(initial_weights[name].to_local(), value.to_local())
                for name, value in legacy_state["model"].items()
            )
            template = {
                "model": restored.state_dict(),
                "optimizer": restored_optimizer.sharded_param_state_fsdp_dtensor(is_loading=True),
                "_model": [restored.module],
            }
            loaded, _, _, _ = _load_base_checkpoint(
                str(tmp_path_dist_ckpt), args, sharded_state_dict=template
            )
            restored.load_state_dict(loaded["model"])
            restored_optimizer.load_state_dict(loaded["optimizer"])

            # Independent reference: masters start from checkpoint model values,
            # since old checkpoints did not retain extra master precision.
            reference, reference_optimizer = self._build_model_and_optimizer(
                torch.float32, store_param_remainders, model_dtype=model_dtype, zero_init=True
            )
            models.append(reference)
            reference.state_dict()
            reference.load_state_dict(legacy_state["model"])
            reference_state = {
                "state": {
                    name: dict(state) for name, state in legacy_state["optimizer"]["state"].items()
                },
                "param_to_group_meta": legacy_state["optimizer"]["param_to_group_meta"],
            }
            for param in reference_optimizer.state:
                name = reference_optimizer._param_name(param)
                reference_state["state"][name]["master_param"] = (
                    torch.zeros_like(param, dtype=torch.int16)
                    if store_param_remainders
                    else param.detach().float().clone()
                )
            reference_optimizer.load_state_dict(reference_state)

            expected_masters = self._master_values(reference_optimizer)
            actual_masters = self._master_values(restored_optimizer)
            for name, expected in expected_masters.items():
                torch.testing.assert_close(actual_masters[name], expected, rtol=0, atol=0)
            for name, expected in source.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[name].to_local(), expected.to_local(), rtol=0, atol=0
                )
            # Match moments and step before checking the first resumed update.
            actual_state = restored_optimizer.sharded_param_state_fsdp_dtensor()
            for name, expected in legacy_state["optimizer"]["state"].items():
                for key, value in expected.items():
                    torch.testing.assert_close(
                        actual_state["state"][name][key], value, rtol=0, atol=0
                    )
            assert (
                actual_state["param_to_group_meta"]
                == legacy_state["optimizer"]["param_to_group_meta"]
            )
            self._step_with_fixed_gradient(reference, reference_optimizer)
            self._step_with_fixed_gradient(restored, restored_optimizer)
            actual_masters = self._master_values(restored_optimizer)
            for name, expected in self._master_values(reference_optimizer).items():
                torch.testing.assert_close(actual_masters[name], expected, rtol=0, atol=0)
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[name].to_local(), expected.to_local(), rtol=0, atol=0
                )
        finally:
            for model in models:
                model.stop_communication()

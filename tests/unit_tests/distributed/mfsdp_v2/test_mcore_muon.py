# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MCore Muon/Adam parameter selection, joint clipping, and FSDP checkpoint resume."""

from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    preprocess_state_dict_for_uneven_dtensor,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.fsdp_dtensor_checkpoint import handle_swiglu_in_state_dict
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("emerging_optimizers")
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.muon import FsdpMuon


class TinyModel(MegatronModule):
    """Muon matrices mixed with embeddings, output weights, biases, and an explicit opt-out."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.embedding = nn.Embedding(8, 8)
        self.decoder = TransformerBlock(config, spec=get_gpt_layer_local_spec())
        self.fc2 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 4)
        self.embedding.weight.is_embedding_or_output_parameter = True
        self.output.weight.is_embedding_or_output_parameter = True
        self.fc2.weight.use_muon = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return output activations for a token batch."""
        hidden_states = self.decoder(self.embedding(tokens), attention_mask=None)
        return self.output(self.fc2(hidden_states).relu())


def _train(
    model: MegatronModule, optimizer: MegatronOptimizer, inputs: torch.Tensor
) -> torch.Tensor:
    """Return losses and the combined Muon/Adam gradient norms."""
    values = []
    for tokens in inputs:
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        losses = run_forward_backward_on_microbatches(
            model, lambda: model(tokens).float().square().mean()
        )
        success, norm, _ = optimizer.step()
        assert success
        values.append(
            torch.stack((losses[0], torch.as_tensor(norm, device=tokens.device).squeeze()))
        )
    return torch.stack(values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mcore_muon_adam_losses_and_resume(
    distributed_setup, tmp_path_dist_ckpt: Path, dtype: torch.dtype
) -> None:
    """Compare a real optimizer chain against unsharded training, then resume both states."""
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gradient_accumulation_fusion=False,
            attention_backend=AttnBackend.unfused,
        )
        optimizer_config = OptimizerConfig(
            optimizer="muon",
            lr=0.01,
            weight_decay=0.1,
            bf16=config.bf16,
            params_dtype=dtype,
            muon_split_qkv=False,
            clip_grad=0.01,
            log_num_zeros_in_grad=True,
        )
        torch.manual_seed(1234)
        reference = TinyModel(config).to(device=distributed_setup.device, dtype=dtype)
        module = TinyModel(config).to(device=distributed_setup.device, dtype=dtype)
        module.load_state_dict(reference.state_dict())
        reference.ddp_config = DistributedDataParallelConfig()
        reference_optimizer = get_megatron_optimizer(
            optimizer_config,
            [reference],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        def shard_and_optimize(
            module: MegatronModule,
        ) -> tuple[FullyShardedDataParallel, MegatronOptimizer]:
            model = FullyShardedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_megatron_fsdp=True,
                    megatron_fsdp_version=2,
                    data_parallel_sharding_strategy="optim_grads_params",
                ),
                module,
                pg_collection=pg_collection,
            )
            optimizer = get_megatron_optimizer(
                optimizer_config,
                [model],
                use_gloo_process_groups=False,
                pg_collection=pg_collection,
            )
            return model, optimizer

        model, optimizer = shard_and_optimize(module)
        assert isinstance(optimizer, ChainedOptimizer)
        assert len(optimizer.chained_optimizers) == 2
        muon_optimizer = next(
            opt for opt in optimizer.chained_optimizers if isinstance(opt.optimizer, FsdpMuon)
        )
        assert set(muon_optimizer.get_parameters()) == {
            param for param in model.module.decoder.parameters() if param.ndim == 2
        }
        parameters = optimizer.get_parameters()
        assert len(parameters) == len(set(parameters))

        inputs = torch.randint(0, 8, (5, 2, 4), device=distributed_setup.device)
        expected = _train(reference, reference_optimizer, inputs)
        actual = _train(model, optimizer, inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-6)

        with TempNamedDir(tmp_path_dist_ckpt / "muon_adam", sync=True) as ckpt_dir:
            state = {
                "model": model.state_dict_for_save_checkpoint(),
                "optimizer": optimizer.sharded_state_dict({}),
            }
            state["model"], state["optimizer"] = handle_swiglu_in_state_dict(
                model.module, state["model"], state["optimizer"]
            )
            preprocess_state_dict_for_uneven_dtensor(state)
            dcp.save(state, checkpoint_id=ckpt_dir)
            restored, restored_optimizer = shard_and_optimize(
                TinyModel(config).to(device=distributed_setup.device, dtype=dtype)
            )
            raw_state = {
                "model": restored.state_dict_for_save_checkpoint(),
                "optimizer": restored_optimizer.sharded_state_dict({}, is_loading=True),
            }
            state = raw_state.copy()
            state["model"], state["optimizer"] = handle_swiglu_in_state_dict(
                restored.module, raw_state["model"], raw_state["optimizer"]
            )
            preprocess_state_dict_for_uneven_dtensor(state)
            dcp.load(state, checkpoint_id=ckpt_dir)
            restored_optimizer.load_state_dict(raw_state["optimizer"])
            restored.load_state_dict(raw_state["model"])

        expected = _train(model, optimizer, inputs)
        actual = _train(restored, restored_optimizer, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_state = optimizer.sharded_state_dict({})["state"]
        actual_state = restored_optimizer.sharded_state_dict({})["state"]
        for name, param_state in expected_state.items():
            for key, value in param_state.items():
                torch.testing.assert_close(
                    actual_state[name][key].to_local(), value.to_local(), rtol=0, atol=0
                )
    finally:
        Utils.destroy_model_parallel()

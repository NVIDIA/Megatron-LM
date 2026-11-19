# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MCore Muon/Adam parameter selection, joint clipping, and loss parity, and AngularMuOwn."""

import pytest
import torch
from torch import nn

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.angular_muown import (
    FsdpAngularMuOwn,
    angular_multiplier,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.muon import FsdpMuon
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


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


@pytest.fixture
def pg_collection(distributed_setup):
    """Set up dense model parallelism and clean up its process groups afterward."""
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    destroy_model_parallel()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mcore_muon_adam_losses(distributed_setup, pg_collection, dtype: torch.dtype) -> None:
    """Compare parameter selection, joint clipping, and losses against unsharded training."""
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
    reference_model = TinyModel(config).to(device=distributed_setup.device, dtype=dtype)
    reference_model.ddp_config = DistributedDataParallelConfig()
    reference_optimizer = get_megatron_optimizer(
        optimizer_config,
        [reference_model],
        use_gloo_process_groups=False,
        pg_collection=pg_collection,
    )

    module = TinyModel(config).to(device=distributed_setup.device, dtype=dtype)
    module.load_state_dict(reference_model.state_dict())
    model = FullyShardedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            data_parallel_sharding_strategy="optim_grads_params",
            # FP32 gradient reduction keeps BF16 rounding from depending on the DP size.
            megatron_fsdp_grad_comm_dtype=torch.float32,
        ),
        module,
        pg_collection=pg_collection,
    )
    optimizer = get_megatron_optimizer(
        optimizer_config, [model], use_gloo_process_groups=False, pg_collection=pg_collection
    )
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
    expected = _train(reference_model, reference_optimizer, inputs)
    actual = _train(model, optimizer, inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-6)


def _tiny_transformer_config(dtype: torch.dtype) -> TransformerConfig:
    """Return the one-layer GPT block configuration used by these tests."""
    return TransformerConfig(
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


def test_angular_muown_config_validation() -> None:
    """AngularMuOwn is reachable only through Muon with shape scaling."""
    base = dict(optimizer="muon", lr=0.01, muon_split_qkv=False)
    OptimizerConfig(
        **base, muon_use_muown=True, muown_use_angular=True, muon_scale_mode="shape_scaling"
    ).validate()
    with pytest.raises(ValueError, match="shape_scaling"):
        OptimizerConfig(**base, muon_use_muown=True, muown_use_angular=True).validate()
    with pytest.raises(NotImplementedError, match="AngularMuOwn"):
        OptimizerConfig(**base, muon_use_muown=True, muon_scale_mode="shape_scaling").validate()
    with pytest.raises(ValueError, match="requires muon_use_muown"):
        OptimizerConfig(**base, muown_use_angular=True).validate()


def test_mcore_angular_muown(distributed_setup, pg_collection) -> None:
    """The MFSDP v2 factory builds FsdpAngularMuOwn for the Muon matrices and trains with it."""
    config = _tiny_transformer_config(torch.bfloat16)
    optimizer_config = OptimizerConfig(
        optimizer="muon",
        lr=0.01,
        weight_decay=0.1,
        bf16=True,
        params_dtype=torch.bfloat16,
        muon_split_qkv=False,
        muon_nesterov=True,
        muon_scale_mode="shape_scaling",
        muon_use_muown=True,
        muown_use_angular=True,
        muown_angular_warmup_iters=1,
        muown_angular_horizon_iters=4,
        muown_weight_decay_mult=0.5,
        clip_grad=1.0,
    )
    optimizer_config.validate()
    torch.manual_seed(1234)
    module = TinyModel(config).to(device=distributed_setup.device, dtype=torch.bfloat16)

    reference_model = TinyModel(config).to(device=distributed_setup.device, dtype=torch.bfloat16)
    reference_model.ddp_config = DistributedDataParallelConfig()
    with pytest.raises(NotImplementedError, match="Megatron-FSDP v2"):
        get_megatron_optimizer(
            optimizer_config,
            [reference_model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

    model = FullyShardedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            data_parallel_sharding_strategy="optim_grads_params",
            megatron_fsdp_grad_comm_dtype=torch.float32,
        ),
        module,
        pg_collection=pg_collection,
    )
    optimizer = get_megatron_optimizer(
        optimizer_config, [model], use_gloo_process_groups=False, pg_collection=pg_collection
    )
    muon_wrapper = next(
        opt for opt in optimizer.chained_optimizers if isinstance(opt.optimizer, FsdpAngularMuOwn)
    )
    assert set(muon_wrapper.get_parameters()) == {
        param for param in model.module.decoder.parameters() if param.ndim == 2
    }
    muon_optimizer = muon_wrapper.optimizer
    assert muon_optimizer.nesterov
    assert muon_optimizer.angular_warmup_steps == 1
    assert muon_optimizer.angular_horizon_steps == 4
    assert muon_optimizer.weight_decay_mult == 0.5
    assert muon_optimizer.row_scale_betas == (
        optimizer_config.adam_beta1,
        optimizer_config.adam_beta2,
    )

    inputs = torch.randint(0, 8, (5, 2, 4), device=distributed_setup.device)
    values = _train(model, optimizer, inputs)
    assert torch.isfinite(values).all()
    assert all(group["muown_step"] == len(inputs) for group in muon_optimizer.param_groups)
    assert muon_optimizer.last_angular_multiplier == pytest.approx(
        angular_multiplier(
            len(inputs),
            warmup_steps=1,
            horizon_steps=4,
            decay_scale=optimizer_config.muown_angular_decay_scale,
            decay_degree=optimizer_config.muown_angular_decay_degree,
            plateau_degree=optimizer_config.muown_angular_plateau_degree,
        )
    )
    for param, state in muon_optimizer.state.items():
        torch.testing.assert_close(
            state["row_scale"].to_local().abs(),
            param.to_local().norm(dim=1, keepdim=True),
            rtol=1e-5,
            atol=1e-7,
        )

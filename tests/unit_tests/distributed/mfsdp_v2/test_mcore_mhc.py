# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""mHC replay unit selection and training parity with MFSDP v2."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.fsdp_unit_collector import FsdpUnitCollector
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.extensions.transformer_engine import te
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("recompute_mhc", [False, True])
@pytest.mark.parametrize("custom_units", [False, True])
@pytest.mark.parametrize("norm_name", ["input_layernorm", "pre_mlp_layernorm"])
@pytest.mark.parametrize(
    "norm_type", [nn.LayerNorm, nn.RMSNorm, te.pytorch.LayerNorm, te.pytorch.RMSNorm]
)
def test_mhc_replay_unit_selection(recompute_mhc, custom_units, norm_name, norm_type):
    """Replay needs parameterized child units even with an explicit wrapping policy."""
    # Unit selection only needs the module tree, not initialized attention kernels.
    inner = TransformerLayer.__new__(TransformerLayer)
    nn.Module.__init__(inner)
    inner.input_layernorm = nn.LayerNorm(4, elementwise_affine=False)
    inner.pre_mlp_layernorm = nn.LayerNorm(4, elementwise_affine=False)
    norm = norm_type(4, device="cpu")
    setattr(inner, norm_name, norm)
    inner.mlp = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4))
    inner.is_moe_layer = False
    wrapper = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
    nn.Module.__init__(wrapper)
    wrapper.inner_layer = inner
    wrapper.hyper_connection = nn.Linear(4, 4)
    wrapper.config = SimpleNamespace(
        recompute_granularity="selective" if recompute_mhc else None,
        recompute_modules=["mhc"] if recompute_mhc else None,
    )
    root = nn.Sequential(wrapper, nn.LayerNorm(4))
    collector = FsdpUnitCollector(
        () if custom_units else (HyperConnectionHybridLayer, TransformerLayer)
    )
    expected = [norm] if recompute_mhc else []
    if not custom_units:
        expected.append(wrapper)
    expected.append(root)
    assert collector.collect(root) == expected

    # Each collection must reflect the current recompute configuration.
    wrapper.config.recompute_modules = []
    assert collector.collect(root) == ([root] if custom_units else [wrapper, root])


@pytest.fixture
def process_group_collection(distributed_setup):
    """Initialize TP=PP=1 so FSDP shards over the data-parallel ranks."""
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(1234)  # mHC also initializes parameters on the CPU.
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    destroy_model_parallel()


@pytest.mark.parametrize("recompute_mhc", [False, True], ids=["no-recompute", "recompute"])
@pytest.mark.parametrize("pattern", ["**", "E*"], ids=["attention", "moe"])
def test_mhc_training_matches_reference(process_group_collection, recompute_mhc, pattern):
    """Compare gradients and updates, including replay before an early MoE owner's hook."""
    if torch.distributed.get_world_size() < 2:
        pytest.skip("Parameter sharding requires at least two ranks.")
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        num_moe_experts=2 if "E" in pattern else None,
        moe_ffn_hidden_size=128 if "E" in pattern else None,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        add_bias_linear=False,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
        enable_mhc_connections=True,
        mhc_num_residual_streams=2,
        mhc_sinkhorn_iterations=3,
        use_fused_mhc=True,
        recompute_granularity="selective" if recompute_mhc else None,
        recompute_modules=["mhc"] if recompute_mhc else None,
        mhc_recompute_layer_num=2 if recompute_mhc else None,
    )
    reference, model = [
        HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_pattern=pattern,
            pg_collection=process_group_collection,
        ).cuda()
        for _ in range(2)
    ]
    model.load_state_dict(reference.state_dict())
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
            # Gather on demand so prefetch cannot hide missing replay hooks.
            suggested_communication_unit_size=0,
        ),
        module=model,
        pg_collection=process_group_collection,
    )

    input_ids = torch.arange(16, device="cuda").reshape(2, 8)
    position_ids = torch.arange(8, device="cuda").repeat(2, 1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="sgd", lr=0.01, sgd_momentum=0.0, weight_decay=0.0, clip_grad=0.0
        ),
        [model],
    )
    reference_parameters = dict(reference.named_parameters())
    parameters = dict(model.module.named_parameters())
    assert parameters.keys() == reference_parameters.keys()
    # Identical inputs across ranks make averaged FSDP gradients match the reference.
    for _ in range(3):
        losses = []
        for current_model, current_optimizer in (
            (reference, reference_optimizer),
            (model, optimizer),
        ):
            current_optimizer.zero_grad(set_to_none=True)

            def forward():
                output = current_model(
                    input_ids=input_ids, position_ids=position_ids, attention_mask=None
                )
                return output.square().mean()

            losses.extend(run_forward_backward_on_microbatches(current_model, forward))
        torch.testing.assert_close(losses[0], losses[1])
        for name, parameter in parameters.items():
            expected = reference_parameters[name].grad
            if expected is not None:
                assert parameter.grad is not None, name
                torch.testing.assert_close(
                    parameter.grad.full_tensor(), expected, rtol=1e-4, atol=1e-8, msg=name
                )

        reference_optimizer.step()
        success, _, _ = optimizer.step()
        assert success
        for name, parameter in parameters.items():
            torch.testing.assert_close(
                parameter.full_tensor(), reference_parameters[name], msg=name
            )

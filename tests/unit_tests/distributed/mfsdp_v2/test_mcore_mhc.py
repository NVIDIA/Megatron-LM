# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""mHC replay unit selection and training parity with MFSDP v2."""

from copy import deepcopy

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.fsdp_unit_collector import FsdpUnitCollector
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.uneven_dtensor import (
    chunk_metadata_by_fqn,
)
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("recompute_mhc", [False, True])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("fused_input_layernorm", [False, True])
def test_mhc_replay_unit_selection(
    process_group_collection, recompute_mhc, normalization, fused_input_layernorm
):
    """mHC replay gives standalone norms their own units alongside routed experts."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
        normalization=normalization,
        enable_mhc_connections=True,
        mhc_num_residual_streams=2,
        recompute_granularity="selective" if recompute_mhc else None,
        recompute_modules=["mhc"] if recompute_mhc else None,
        mhc_recompute_layer_num=2 if recompute_mhc else None,
    )
    stack_spec = deepcopy(hybrid_stack_spec)
    if not fused_input_layernorm:
        attention = stack_spec.submodules.attention_layer.submodules
        attention.input_layernorm = TENorm
        attention.self_attention.submodules.linear_qkv = TEColumnParallelLinear
    model = HybridModel(
        config=config,
        hybrid_stack_spec=stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E",
        pg_collection=process_group_collection,
    )
    attention, moe = model.decoder.layers
    assert isinstance(attention, HyperConnectionHybridLayer)
    assert isinstance(moe, HyperConnectionHybridLayer)
    collector = FsdpUnitCollector()
    expected = []
    if recompute_mhc and not fused_input_layernorm:
        expected.append(attention.inner_layer.input_layernorm)
    expected.append(attention)
    if recompute_mhc:
        expected.append(moe.inner_layer.pre_mlp_layernorm)
    expected.append(moe.inner_layer.mlp.experts)
    expected.extend([moe, model])
    assert collector.collect(model) == expected


def test_mhc_owns_mamba_norm_and_mixer(process_group_collection):
    """mHC owns a real Mamba layer, including its separately constructed input norm."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
        enable_mhc_connections=True,
        mhc_num_residual_streams=2,
        recompute_granularity="selective",
        recompute_modules=["mhc"],
        mhc_recompute_layer_num=2,
    )
    stack_spec = deepcopy(hybrid_stack_spec)
    stack_spec.submodules.mamba_layer.submodules.norm = TENorm
    stack_spec.submodules.mamba_layer.submodules.mixer.submodules.in_proj = TEColumnParallelLinear
    model = HybridModel(
        config=config,
        hybrid_stack_spec=stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="M*",
        pg_collection=process_group_collection,
    )
    mamba, attention = model.decoder.layers
    assert isinstance(mamba.inner_layer, MambaLayer)
    assert next(mamba.inner_layer.norm.parameters(), None) is not None
    collector = FsdpUnitCollector()

    assert collector.collect(model) == [mamba, attention, model]
    assert collector.collect(mamba) == [mamba]


@pytest.fixture
def process_group_collection(distributed_setup):
    """Initialize TP=PP=1 so FSDP shards over the data-parallel ranks."""
    if not torch.cuda.is_available():
        pytest.skip("MCore model construction requires CUDA.")
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
    # MFSDP shards packed buffers, not each parameter independently. DTensor's
    # full_tensor() assumes ordinary Shard(0) chunks, so compare the actual local
    # chunks with the corresponding reference slices, including empty shards.
    reference_slices = {
        name: tuple(
            slice(offset, offset + size) for offset, size in zip(chunk.offsets, chunk.sizes)
        )
        for name, chunk in chunk_metadata_by_fqn(model.module).items()
    }
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
                    parameter.grad.to_local(),
                    expected[reference_slices[name]],
                    msg=lambda detail: f"{name}: {detail}",
                )

        reference_optimizer.step()
        success, _, _ = optimizer.step()
        assert success
        for name, parameter in parameters.items():
            torch.testing.assert_close(
                parameter.to_local(), reference_parameters[name][reference_slices[name]], msg=name
            )

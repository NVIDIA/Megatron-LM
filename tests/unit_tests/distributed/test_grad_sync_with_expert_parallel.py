# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Optional

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from tests.unit_tests.test_utilities import Utils


class TestMoEModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_moe_experts: int,
        moe_grouped_gemm: bool,
        ep_size: int,
        etp_size: int,
    ):
        transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_token_dispatcher_type='alltoall',
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=moe_grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [MoELayer(transformer_config, submodules).cuda() for _ in range(num_layers)]
        )


def get_moe_model_and_buffers(
    num_layers: int,
    hidden_size: int,
    num_moe_experts: int,
    moe_grouped_gemm: bool,
    ep_size: int,
    bucket_size: Optional[int],
    etp_size: int,
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    num_distributed_optimizer_instances: int,
):
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
        bucket_size=bucket_size,
        average_in_collective=average_in_collective,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
    )
    model = TestMoEModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_moe_experts=num_moe_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        ep_size=ep_size,
        etp_size=etp_size,
    )
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config=ddp_config, module=model
    )
    assert len(model.buffers) == 1
    param_and_grad_buffer = model.buffers[0]
    ep_param_and_grad_buffer = (
        model.expert_parallel_buffers[0] if len(model.expert_parallel_buffers) else None
    )
    non_ep_bucket_groups = model.bucket_groups
    ep_bucket_groups = model.expert_parallel_bucket_groups

    return (
        model,
        param_and_grad_buffer,
        ep_param_and_grad_buffer,
        non_ep_bucket_groups,
        ep_bucket_groups,
    )


def _build_expert_linear(implementation: str, config: TransformerConfig) -> torch.nn.Module:
    common_kwargs = {
        "input_size": config.hidden_size,
        "output_size": config.ffn_hidden_size,
        "config": config,
        "init_method": config.init_method,
        "bias": False,
        "skip_bias_add": False,
        "is_expert": True,
    }
    if implementation == "native":
        return ColumnParallelLinear(
            **common_kwargs,
            gather_output=False,
            tp_group=parallel_state.get_expert_tensor_parallel_group(),
        )
    if implementation == "transformer_engine":
        return TEColumnParallelLinear(
            **common_kwargs,
            gather_output=False,
            tp_group=parallel_state.get_expert_tensor_parallel_group(),
        )
    if implementation == "transformer_engine_grouped":
        return TEColumnParallelGroupedLinear(
            num_gemms=config.num_moe_experts,
            **common_kwargs,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
    raise AssertionError(f"Unsupported implementation: {implementation}")


@pytest.mark.parametrize(
    ("tensor_model_parallel_size", "expert_tensor_parallel_size"), [(2, 1), (1, 2)]
)
@pytest.mark.parametrize(
    "implementation", ["native", "transformer_engine", "transformer_engine_grouped"]
)
def test_expert_grad_sync_uses_expert_data_parallel_group(
    implementation: str, tensor_model_parallel_size: int, expert_tensor_parallel_size: int
):
    """Expert gradients must not be reduced over ordinary DP when ETP differs from TP."""
    if Utils.world_size < 4 or Utils.world_size % 4 != 0:
        pytest.skip("Test requires a world size divisible by four")
    if Utils.world_size > 16:
        pytest.skip("Rank-encoded gradients are intended for small unit-test world sizes")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
    )
    try:
        # Per-token loss leaves DDP's pre-collective gradient scaling at one.
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            ffn_hidden_size=16,
            num_moe_experts=2,
            moe_ffn_hidden_size=16,
            moe_router_topk=2,
            tensor_model_parallel_size=tensor_model_parallel_size,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            calculate_per_token_loss=True,
            gradient_accumulation_fusion=False,
            perform_initialization=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        module = _build_expert_linear(implementation, config).cuda()
        model = DistributedDataParallel(
            config,
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                use_distributed_optimizer=False,
                average_in_collective=False,
            ),
            module=module,
        )

        expert_dp_group = parallel_state.get_expert_data_parallel_group(
            partial_expert_data_parallel=True
        )
        ordinary_dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=True
        )
        expert_dp_ranks = torch.distributed.get_process_group_ranks(expert_dp_group)
        ordinary_dp_ranks = torch.distributed.get_process_group_ranks(ordinary_dp_group)
        assert expert_dp_ranks != ordinary_dp_ranks

        # Powers of two give every rank set a distinct sum, exposing the wrong collective group.
        rank_value = float(2 ** torch.distributed.get_rank())
        expected_value = float(sum(2**rank for rank in expert_dp_ranks))
        ordinary_dp_value = float(sum(2**rank for rank in ordinary_dp_ranks))
        assert expected_value != ordinary_dp_value

        for param in model.parameters():
            param.main_grad.fill_(rank_value)
        model.finish_grad_sync()

        for param in model.parameters():
            torch.testing.assert_close(
                param.main_grad, torch.full_like(param.main_grad, expected_value), rtol=0, atol=0
            )

        assert not model.buffers
        assert len(model.expert_parallel_buffers) == 1
        assert all(param.allreduce is False for param in model.parameters())
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
@pytest.mark.parametrize("average_in_collective", [False, True])
@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize("etp_size", [1, 2])
@pytest.mark.parametrize("num_distributed_optimizer_instances", [1, 2])
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_grad_sync(
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    ep_size: int,
    etp_size: int,
    num_distributed_optimizer_instances: int,
):
    Utils.initialize_model_parallel(
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
    )

    if num_distributed_optimizer_instances > 1 and not use_distributed_optimizer:
        pytest.skip(
            "Multiple distributed optimizer instances requires distributed optimizer to be enabled"
        )

    (
        model,
        non_ep_param_and_grad_buffer,
        ep_param_and_grad_buffer,
        non_ep_bucket_groups,
        ep_bucket_groups,
    ) = get_moe_model_and_buffers(
        num_layers=2,
        hidden_size=512,
        num_moe_experts=4,
        moe_grouped_gemm=True,
        ep_size=ep_size,
        etp_size=etp_size,
        bucket_size=None,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
        average_in_collective=average_in_collective,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
    )

    param_to_bucket_group = {}
    for bucket_group in non_ep_bucket_groups:
        for param in bucket_group.params:
            assert param not in param_to_bucket_group
            param_to_bucket_group[param] = bucket_group
    for bucket_group in ep_bucket_groups:
        for param in bucket_group.params:
            assert param not in param_to_bucket_group
            param_to_bucket_group[param] = bucket_group

    non_ep_param_and_grad_buffer.grad_data.data.fill_(1.0)
    non_ep_expected_grad_data_value_after_collective = 1
    if (
        use_distributed_optimizer
        and (not average_in_collective)
        and parallel_state.get_data_parallel_rank(
            with_context_parallel=True, partial_data_parallel=True
        )
        != 0
    ):
        # With above conditions, the data in param_and_grad_buffer.grad_data[0] equals
        # 1/data_parallel_word_size.
        # When average_in_collective=False, the grad data is always first scaled by
        # 1/data_parallel_word_size and then summed by AR/RS.
        # When use_distributed_optimizer=True, only for rank=0,
        # param_and_grad_buffer.grad_data[0] is updated. For other ranks another shard of
        # grad_data is updated while param_and_grad_buffer.grad_data[0] is unchanged
        # (=1/data_parallel_word_size).
        non_ep_expected_grad_data_value_after_collective /= (
            parallel_state.get_data_parallel_world_size()
        )
    if ep_size > 1:
        # For MoE models with exper parallelism, each expert will receive tokens from EPxETP
        # times batches, such that the expert gradient will be EPxETP times after backward,
        # and the expected gradient after collective should be 1.0 as same as dense params.
        ep_param_and_grad_buffer.grad_data.data.fill_(float(ep_size * etp_size))
        ep_expected_grad_data_value_after_collective = 1
        if (
            use_distributed_optimizer
            and (not average_in_collective)
            and parallel_state.get_expert_data_parallel_rank(partial_expert_data_parallel=True) != 0
        ):
            # With above conditions, the data in param_and_grad_buffer.grad_data[0] equals 1/EDP.
            # When average_in_collective=False, the grad data is always first scaled by
            # expert_data_parallel_size and then summed by AR/RS.
            # After SUM collective in expert_data_group, the scale will be 1.0.
            ep_expected_grad_data_value_after_collective /= (
                parallel_state.get_expert_data_parallel_world_size()
            )

    register_grad_sync_context = (
        contextlib.nullcontext() if overlap_grad_reduce else pytest.raises(AssertionError)
    )

    # Call register_grad_ready for all params before starting test to seed tracking
    # data structures.
    params = list(model.parameters())
    for param in params:
        with register_grad_sync_context:
            bucket_group = param_to_bucket_group[param]
            bucket_group.register_grad_ready(param)
    # Call reset to set .is_first_batch to False.
    for param in params:
        bucket_group = param_to_bucket_group[param]
        bucket_group.reset()

    map_bucket_to_last_param_idx = {}
    for i, param in enumerate(params):
        if not (param in param_to_bucket_group):
            # it means this parameter is not on this device, skip
            continue
        bucket_group = param_to_bucket_group[param]
        if bucket_group in map_bucket_to_last_param_idx:
            param_idx = map_bucket_to_last_param_idx[bucket_group] + 1
        else:
            param_idx = 0
        map_bucket_to_last_param_idx[bucket_group] = param_idx

        finish_grad_sync_context = contextlib.nullcontext()
        if (
            param_idx < (len(bucket_group.params) - 1)
            and overlap_grad_reduce
            and num_distributed_optimizer_instances == 1
        ):
            # Can't finish grad sync until all params have been registered ready.
            finish_grad_sync_context = pytest.raises(AssertionError)

        with register_grad_sync_context:
            bucket_group.register_grad_ready(param)

        with finish_grad_sync_context:
            # When overlap_grad_reduce is True, this should throw an assertion error until all
            # params in the model have registered their grad above.
            # When overlap_grad_reduce is False, the collective is forced through.
            bucket_group.finish_grad_sync()

        if bucket_group in non_ep_bucket_groups:
            expected_grad_data_value = non_ep_expected_grad_data_value_after_collective
        else:
            expected_grad_data_value = ep_expected_grad_data_value_after_collective
        # Before gradient sync, the gradient value should keep original.
        if overlap_grad_reduce and param_idx < (len(bucket_group.params) - 1):
            if bucket_group in non_ep_bucket_groups:
                expected_grad_data_value = 1
            else:
                expected_grad_data_value = ep_size * etp_size

        if bucket_group in non_ep_bucket_groups:
            assert non_ep_param_and_grad_buffer.grad_data[0] == expected_grad_data_value
        else:
            assert ep_param_and_grad_buffer.grad_data[0] == expected_grad_data_value

        if not overlap_grad_reduce:
            # Reset grad_data for subsequent collectives.
            if bucket_group in non_ep_bucket_groups:
                non_ep_param_and_grad_buffer.grad_data.data.fill_(1.0)
            else:
                ep_param_and_grad_buffer.grad_data.data.fill_(float(ep_size * etp_size))

    Utils.destroy_model_parallel()

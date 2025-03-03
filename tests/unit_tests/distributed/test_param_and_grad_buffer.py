import contextlib
import math
from typing import Optional

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import partition_buckets
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import TestModel, Utils


def get_model_and_buffers(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    bias: bool,
    shared_embedding: bool,
    bucket_size: int,
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    num_distributed_optimizer_instances: int = 1,
):
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
        bucket_size=bucket_size,
        average_in_collective=average_in_collective,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
    )
    model = TestModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=bias,
        shared_embedding=shared_embedding,
    ).bfloat16()

    # Wrap with DistributedDataParallel, and get underlying buffer.
    # Use dummy TransformerConfig with mostly default values. Avoid divide-by-zero
    # errors for num_attention_heads and num_layers.
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config=ddp_config, module=model
    )
    assert len(model.buffers) == 1
    param_and_grad_buffer = model.buffers[0]
    bucket_groups = model.bucket_groups

    return model, param_and_grad_buffer, bucket_groups


@pytest.mark.parametrize("bucket_size", [None, 9000, 9025, 9050, 18000, 18050, 20000])
@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shared_embedding", [False, True])
def test_bucket_sizes(
    bucket_size: Optional[int], use_distributed_optimizer: bool, bias: bool, shared_embedding: bool
):
    Utils.initialize_model_parallel()

    if shared_embedding and bias:
        # Don't bother running shared_embedding + bias since gold values are trickier to compute.
        return

    input_dim = 95
    output_dim = 95
    num_layers = 10
    _, param_and_grad_buffer, _ = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=bias,
        shared_embedding=shared_embedding,
        bucket_size=bucket_size,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=True,
        average_in_collective=False,
    )

    actual_numel_in_each_bucket = [
        bucket.numel_unpadded for bucket in param_and_grad_buffer.buckets
    ]
    actual_numel_padded_in_each_bucket = [
        bucket.grad_data.numel() for bucket in param_and_grad_buffer.buckets
    ]

    def _pad_if_needed(numel_unpadded, divisor):
        if use_distributed_optimizer:
            return math.ceil(numel_unpadded / divisor) * divisor
        return numel_unpadded

    def _pad_bucket_if_needed(numel_unpadded):
        # Want 128-byte alignment for distributed optimizer.
        divisor = math.lcm(parallel_state.get_data_parallel_world_size(), 128)
        return _pad_if_needed(numel_unpadded, divisor)

    def _pad_param_if_needed(numel_unpadded):
        # Want 64-byte alignment for params.
        return _pad_if_needed(numel_unpadded, 64)

    if bucket_size is None:
        # If bucket_size is infinite (None), number of buckets should be 1.
        if shared_embedding and use_distributed_optimizer:
            assert len(param_and_grad_buffer.buckets) == 2
        else:
            assert len(param_and_grad_buffer.buckets) == 1
    else:
        # Else, compute number of buckets.
        numel_in_each_bucket = []
        numel_padded_in_each_bucket = []
        numel_in_last_bucket = 0
        param_sizes = []
        for _ in range(num_layers):
            param_sizes.append(input_dim * output_dim)
            if bias:  # Include bias term.
                param_sizes.append(output_dim)
        # Create separate bucket for first parameter from reverse direction.
        if shared_embedding and use_distributed_optimizer:
            numel_in_each_bucket.append(param_sizes[-1])
            numel_padded_in_each_bucket.append(_pad_bucket_if_needed(param_sizes[-1]))
            param_sizes = param_sizes[:-1]
        # Iterate through params in backward direction.
        for param_size in param_sizes[::-1]:
            numel_in_last_bucket = _pad_param_if_needed(numel_in_last_bucket)
            numel_in_last_bucket += param_size
            if numel_in_last_bucket >= bucket_size:
                numel_in_each_bucket.append(numel_in_last_bucket)
                numel_padded_in_each_bucket.append(_pad_bucket_if_needed(numel_in_last_bucket))
                numel_in_last_bucket = 0
        if numel_in_last_bucket > 0:
            numel_in_each_bucket.append(numel_in_last_bucket)
            numel_padded_in_each_bucket.append(_pad_bucket_if_needed(numel_in_last_bucket))

        assert len(param_and_grad_buffer.buckets) == len(
            numel_in_each_bucket
        ), f"Buckets don't match (got {actual_numel_in_each_bucket} but should be {numel_in_each_bucket})"
        assert actual_numel_in_each_bucket == numel_in_each_bucket, (
            f"Number of parameters in each bucket should be {numel_in_each_bucket}, "
            f"but is {actual_numel_in_each_bucket}"
        )
        if use_distributed_optimizer:
            assert all(
                [
                    x % parallel_state.get_data_parallel_world_size() == 0
                    for x in actual_numel_padded_in_each_bucket
                ]
            ), (
                f"Size of each padded bucket should be divisible by "
                f"{parallel_state.get_data_parallel_world_size()}"
            )
        assert actual_numel_padded_in_each_bucket == numel_padded_in_each_bucket, (
            f"Number of parameters in each padded bucket should be {numel_padded_in_each_bucket}, "
            f"but is {actual_numel_padded_in_each_bucket}"
        )

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
@pytest.mark.parametrize("average_in_collective", [False, True])
@pytest.mark.parametrize("num_distributed_optimizer_instances", [1, 2])
# @pytest.mark.flaky
def test_grad_sync(
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    num_distributed_optimizer_instances: int,
):
    Utils.initialize_model_parallel(
        num_distributed_optimizer_instances=num_distributed_optimizer_instances
    )
    # Skip test if num_distributed_optimizer_instances > 1 and not using distributed optimizer
    if num_distributed_optimizer_instances > 1 and not use_distributed_optimizer:
        pytest.skip("Multiple optimizer instances require distributed optimizer to be enabled")

    input_dim = 100
    output_dim = 100
    num_layers = 10
    model, param_and_grad_buffer, bucket_groups = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=True,
        shared_embedding=False,
        bucket_size=None,  # Group all params into single bucket.
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
        average_in_collective=average_in_collective,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
    )
    param_to_bucket_group = {}
    for bucket_group in bucket_groups:
        for param in bucket_group.params:
            assert param not in param_to_bucket_group
            param_to_bucket_group[param] = bucket_group

    param_and_grad_buffer.grad_data.data.fill_(1.0)
    expected_grad_data_value_after_collective = 1
    # under the following conditions, the data in param_and_grad_buffer.grad_data[0] equals to 1/DP
    # this is because when average_in_collective=False, the grad data is always first scaled by 1/DP and then summed by AR/RS
    # and when use_distributed_optimizer=True, only for rank=0 param_and_grad_buffer.grad_data[0] is updated, for other ranks
    # another shard of grad_data is updated while param_and_grad_buffer.grad_data[0] is unchanged (=1/DP)
    if (
        use_distributed_optimizer
        and (not average_in_collective)
        and parallel_state.get_data_parallel_rank(
            with_context_parallel=True, partial_data_parallel=True
        )
        != 0
    ):
        expected_grad_data_value_after_collective /= parallel_state.get_data_parallel_world_size()

    params = list(model.parameters())
    for i, param in enumerate(params):
        assert param in param_to_bucket_group
        bucket_group = param_to_bucket_group[param]
        register_grad_sync_context = (
            contextlib.nullcontext() if overlap_grad_reduce else pytest.raises(AssertionError)
        )
        finish_grad_sync_context = contextlib.nullcontext()
        if (
            i < (len(params) - 1)
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

        expected_grad_data_value = expected_grad_data_value_after_collective
        if overlap_grad_reduce and i < (len(params) - 1):
            expected_grad_data_value = 1
        assert param_and_grad_buffer.grad_data[0] == expected_grad_data_value

        if not overlap_grad_reduce:
            # Reset grad_data for subsequent collectives.
            param_and_grad_buffer.grad_data.data.fill_(1.0)

    Utils.destroy_model_parallel()

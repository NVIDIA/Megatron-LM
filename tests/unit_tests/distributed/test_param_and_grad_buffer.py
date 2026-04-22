# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib
import math
from typing import Optional
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import (
    _compute_tail_shrink_close_set,
    _ParamAndGradBuffer,
    partition_buckets,
)
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
    grad_reduce_in_fp32: bool = True,
    param_name_patterns_for_fp32_local_accumulation: tuple = (),
):
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
        bucket_size=bucket_size,
        average_in_collective=average_in_collective,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
        param_name_patterns_for_fp32_local_accumulation=param_name_patterns_for_fp32_local_accumulation,
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


def test_param_to_index_alignment_with_padding():
    """Ensure bucket-local param offsets honor padding when DistOpt pads params."""
    Utils.initialize_model_parallel()

    # With input_dim=4, output_dim=4:
    #   - weight: 4*4 = 16 elements
    #   - bias: 4 elements
    # Since 16 % 64 != 0, the bias must be padded away from the weight,
    # making padding observable.
    input_dim = 4
    output_dim = 4
    model, param_and_grad_buffer, _ = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=1,
        bias=True,
        shared_embedding=False,
        bucket_size=None,  # single bucket
        use_distributed_optimizer=True,  # enforces 64-element alignment
        overlap_grad_reduce=True,
        average_in_collective=False,
    )

    bucket = param_and_grad_buffer.buckets[0]
    naive_offset = 0
    padding_observed = False

    for param in bucket.params_list:
        global_start, global_end, _ = param_and_grad_buffer.param_index_map[param]
        expected_local_start = global_start - bucket.offset
        expected_local_end = global_end - bucket.offset
        local_start, local_end = bucket.param_to_index[param]

        # param_to_index should match the padded offsets used in the global buffer.
        assert (local_start, local_end) == (expected_local_start, expected_local_end)

        # At least one param should have been padded relative to naive packing.
        if local_start != naive_offset:
            padding_observed = True
        naive_offset = local_end

        # Verify the slice retrieved via param_to_index matches param.data view.
        param_slice = bucket.param_data.view(-1)[local_start:local_end]
        torch.testing.assert_close(param_slice, param.data.view(-1))

    assert padding_observed, (
        "Expected padding to be applied between params. "
        "Ensure model dimensions are chosen such that param sizes are not multiples of 64."
    )

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
@pytest.mark.parametrize("average_in_collective", [False, True])
@pytest.mark.parametrize("num_distributed_optimizer_instances", [1, 2])
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
    # Data in param_and_grad_buffer.grad_data[0] is 1/DP.
    # When average_in_collective=False, the grad data is always first scaled by 1/DP and then
    # summed by AR/RS.
    # When use_distributed_optimizer=True, only rank0's param_and_grad_buffer.grad_data[0] is
    # updated; other ranks update another shard of grad_data while keeping
    # param_and_grad_buffer.grad_data[0] unchanged (=1/DP).
    if (
        use_distributed_optimizer
        and (not average_in_collective)
        and parallel_state.get_data_parallel_rank(
            with_context_parallel=True, partial_data_parallel=True
        )
        != 0
    ):
        expected_grad_data_value_after_collective /= parallel_state.get_data_parallel_world_size()

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

    for i, param in enumerate(params):
        assert param in param_to_bucket_group
        bucket_group = param_to_bucket_group[param]
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


@pytest.mark.parametrize("force_all_reduce", [False, True])
def test_force_all_reduce_uses_correct_collective(force_all_reduce: bool):
    """Test that force_all_reduce=True causes all-reduce to be used instead of reduce-scatter."""
    Utils.initialize_model_parallel()

    input_dim = 100
    output_dim = 100
    num_layers = 2
    model, param_and_grad_buffer, _ = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=True,
        shared_embedding=False,
        bucket_size=None,
        use_distributed_optimizer=True,  # This normally uses reduce-scatter.
        overlap_grad_reduce=False,
        average_in_collective=False,
    )

    # Mock the collective operations to track which one is called.
    with (
        mock.patch('torch.distributed.all_reduce') as mock_all_reduce,
        mock.patch(
            'megatron.core.distributed.param_and_grad_buffer.dist_reduce_scatter_func'
        ) as mock_reduce_scatter,
    ):
        # Set up the mocks to be no-ops.
        mock_all_reduce.return_value = None
        mock_reduce_scatter.return_value = None

        # Trigger the grad sync via the DDP model's finish_grad_sync method.
        model.finish_grad_sync(force_all_reduce=force_all_reduce)

        if force_all_reduce:
            # When force_all_reduce=True, all_reduce should be called.
            assert (
                mock_all_reduce.called
            ), "Expected all_reduce to be called when force_all_reduce=True"
            assert (
                not mock_reduce_scatter.called
            ), "Expected reduce_scatter NOT to be called when force_all_reduce=True"
        else:
            # When force_all_reduce=False with distributed optimizer, reduce_scatter should be called.
            assert (
                mock_reduce_scatter.called
            ), "Expected reduce_scatter to be called when force_all_reduce=False"
            assert (
                not mock_all_reduce.called
            ), "Expected all_reduce NOT to be called when force_all_reduce=False"

    Utils.destroy_model_parallel()


def test_start_param_sync_dp_size_1():
    """When dp_size == 1 (e.g., expt_dp_size == 1), start_param_sync should set
    param_gather_dispatched=True and return immediately without launching any
    all-gather collective."""
    world_size = torch.distributed.get_world_size()
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)

    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=False,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        bucket_size=None,
    )
    module = TestModel(
        input_dim=32, output_dim=32, num_layers=2, bias=False, shared_embedding=False
    ).bfloat16()
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config=ddp_config, module=module
    )

    # Confirm dp_size == 1 in the test environment.
    for bg in model.bucket_groups:
        assert bg.intra_distributed_optimizer_instance_size == 1

    with mock.patch('torch.distributed.all_gather') as mock_all_gather:
        for bg in model.bucket_groups:
            assert not bg.param_gather_dispatched
            bg.start_param_sync()
            assert (
                bg.param_gather_dispatched
            ), "param_gather_dispatched should be True after start_param_sync with dp_size=1"
        # No all-gather should have been called.
        assert not mock_all_gather.called, "all_gather should not be called when dp_size == 1"

    Utils.destroy_model_parallel()


class TestFreeOverlapBuffers:
    """Tests for free_overlap_buffers() which releases GPU memory before async checkpoint saves."""

    @staticmethod
    def _make_model():
        """Create a DDP-wrapped model with overlap_param_gather enabled."""
        Utils.initialize_model_parallel()
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=None,
        )
        module = TestModel(
            input_dim=32, output_dim=32, num_layers=2, bias=False, shared_embedding=False
        ).bfloat16()
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            ddp_config=ddp_config,
            module=module,
        )
        return model

    def test_bucket_group_clears_buffers(self):
        """free_overlap_buffers on a bucket group should None-out per-bucket layerwise buffers."""
        model = self._make_model()

        for bg in model.bucket_groups:
            # Simulate buffers that would be allocated by start_param_sync.
            for bucket in bg.buckets:
                bucket.layerwise_gather_list = [torch.empty(8), torch.empty(8)]

            bg.free_overlap_buffers()

            for bucket in bg.buckets:
                assert (
                    bucket.layerwise_gather_list is None
                ), "layerwise_gather_list should be None after free_overlap_buffers"

        Utils.destroy_model_parallel()

    def test_bucket_group_waits_on_pending_handle(self):
        """free_overlap_buffers should wait() on any pending param_gather_handle."""
        model = self._make_model()

        for bg in model.bucket_groups:
            mock_handle = mock.MagicMock()
            bg.param_gather_handle = mock_handle

            bg.free_overlap_buffers()

            mock_handle.wait.assert_called_once()
            assert (
                bg.param_gather_handle is None
            ), "param_gather_handle should be None after free_overlap_buffers"

        Utils.destroy_model_parallel()

    def test_bucket_group_noop_when_no_buffers(self):
        """free_overlap_buffers should be safe to call when no buffers are allocated."""
        model = self._make_model()

        for bg in model.bucket_groups:
            assert bg.param_gather_handle is None
            for bucket in bg.buckets:
                assert bucket.layerwise_gather_list is None

            # Should not raise.
            bg.free_overlap_buffers()

        Utils.destroy_model_parallel()

    def test_ddp_free_overlap_buffers_delegates(self):
        """DDP.free_overlap_buffers should call free_overlap_buffers on all bucket groups."""
        model = self._make_model()

        with mock.patch.object(type(model.bucket_groups[0]), 'free_overlap_buffers') as mock_free:
            model.free_overlap_buffers()
            assert mock_free.call_count == len(
                model.bucket_groups + model.expert_parallel_bucket_groups
            ), "free_overlap_buffers should be called on every bucket group"

        Utils.destroy_model_parallel()


class TestFP32LocalGradAccumulation:
    """Tests for the FP32 local gradient accumulation feature
    (param_name_patterns_for_fp32_local_accumulation)."""

    @staticmethod
    def _make_model(patterns, bucket_size=None):
        """Create a DDP-wrapped model with FP32 local grad accumulation patterns."""
        return get_model_and_buffers(
            input_dim=100,
            output_dim=100,
            num_layers=3,
            bias=True,
            shared_embedding=False,
            bucket_size=bucket_size,
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            average_in_collective=False,
            grad_reduce_in_fp32=False,
            param_name_patterns_for_fp32_local_accumulation=patterns,
        )

    def test_config_validation_with_grad_reduce_in_fp32(self):
        """param_name_patterns_for_fp32_local_accumulation and grad_reduce_in_fp32 are
        mutually exclusive."""
        with pytest.raises(AssertionError):
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, param_name_patterns_for_fp32_local_accumulation=('all',)
            )

    def test_pattern_matching_creates_fp32_main_grad(self):
        """Params matching patterns should get a float32 main_grad and a
        main_grad_copy_in_grad_buffer; non-matching params should not."""
        Utils.initialize_model_parallel()
        # Match only weight params (not bias).
        model, buf, _ = self._make_model(patterns=('*.weight',))

        for name, param in model.module.named_parameters():
            if 'weight' in name:
                assert param.main_grad.dtype == torch.float32, f"{name} main_grad should be float32"
                assert hasattr(param, 'main_grad_copy_in_grad_buffer')
                assert param.main_grad_copy_in_grad_buffer is not None
                # The copy in grad buffer should be in the buffer's grad dtype (bf16).
                assert param.main_grad_copy_in_grad_buffer.dtype == buf.grad_dtype
            else:
                # Bias params should not be promoted.
                assert (
                    param.main_grad.dtype == buf.grad_dtype
                ), f"{name} main_grad should remain in grad_dtype"
                assert getattr(param, 'main_grad_copy_in_grad_buffer', None) is None

        Utils.destroy_model_parallel()

    def test_all_pattern_matches_every_param(self):
        """The 'all' pattern should match every parameter."""
        Utils.initialize_model_parallel()
        model, buf, _ = self._make_model(patterns=('all',))

        for name, param in model.module.named_parameters():
            assert (
                param.main_grad.dtype == torch.float32
            ), f"{name} main_grad should be float32 with 'all' pattern"
            assert getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None

        Utils.destroy_model_parallel()

    def test_bucket_tracks_params_with_extra_main_grads(self):
        """Each bucket's params_with_extra_main_grads should contain exactly
        the params that matched the patterns."""
        Utils.initialize_model_parallel()
        model, buf, _ = self._make_model(patterns=('*.weight',))

        promoted_params = set()
        for name, param in model.module.named_parameters():
            if 'weight' in name:
                promoted_params.add(param)

        bucket_promoted = set()
        for bucket in buf.buckets:
            for param in bucket.params_with_extra_main_grads:
                bucket_promoted.add(param)
            # Every param in params_with_extra_main_grads should also be in bucket.params.
            assert bucket.params_with_extra_main_grads == [] or set(
                bucket.params_with_extra_main_grads
            ).issubset(bucket.params)

        assert (
            bucket_promoted == promoted_params
        ), "Bucket-tracked promoted params should match the set of pattern-matched params"

        Utils.destroy_model_parallel()

    def test_no_patterns_means_no_extra_main_grads(self):
        """With no patterns, no params should have extra main_grads."""
        Utils.initialize_model_parallel()
        _, buf, _ = self._make_model(patterns=())

        assert len(buf.extra_main_grads) == 0
        for bucket in buf.buckets:
            assert len(bucket.params_with_extra_main_grads) == 0

        Utils.destroy_model_parallel()

    def test_reset_zeros_extra_main_grads(self):
        """reset() should zero out both grad_data and all extra main_grads."""
        Utils.initialize_model_parallel()
        _, buf, _ = self._make_model(patterns=('all',))

        # Fill extra main_grads and grad_data with non-zero values.
        buf.grad_data.fill_(1.0)
        for grad in buf.extra_main_grads:
            grad.fill_(42.0)

        buf.reset()

        assert torch.all(buf.grad_data == 0), "grad_data should be zeroed after reset"
        for grad in buf.extra_main_grads:
            assert torch.all(grad == 0), "extra main_grads should be zeroed after reset"

        Utils.destroy_model_parallel()

    def test_scale_gradients_scales_extra_main_grads(self):
        """scale_gradients() should scale both grad_data and extra main_grads."""
        Utils.initialize_model_parallel()
        _, buf, _ = self._make_model(patterns=('all',))

        buf.grad_data.fill_(2.0)
        for grad in buf.extra_main_grads:
            grad.fill_(4.0)

        buf.scale_gradients(0.5)

        assert torch.allclose(
            buf.grad_data, torch.tensor(1.0, dtype=buf.grad_data.dtype)
        ), "grad_data should be scaled"
        for grad in buf.extra_main_grads:
            assert torch.allclose(
                grad, torch.tensor(2.0, dtype=grad.dtype)
            ), "extra main_grads should be scaled"

        Utils.destroy_model_parallel()

    def test_grad_sync_copies_to_and_from_comm_buffer(self):
        """During grad sync, values in FP32 main_grad should be copied to the comm buffer
        before the collective, and the reduced result should be copied back afterward."""
        Utils.initialize_model_parallel()
        model, buf, bucket_groups = self._make_model(patterns=('all',))

        # Simulate accumulated gradients in FP32 main_grad.
        for param in model.parameters():
            param.main_grad.fill_(1.0)

        # Run grad sync (non-overlapped, so finish_grad_sync triggers start + wait).
        model.finish_grad_sync()

        # After sync, main_grad should contain the reduced result (not the original 1.0,
        # since the collective may have scaled / averaged). The key invariant is that
        # main_grad should equal main_grad_copy_in_grad_buffer (the comm buffer slice)
        # after the copy-back.
        for param in model.parameters():
            if getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None:
                torch.testing.assert_close(
                    param.main_grad,
                    param.main_grad_copy_in_grad_buffer.float(),
                    msg="main_grad should equal comm buffer after grad sync copy-back",
                )

        Utils.destroy_model_parallel()


class TestLastBucketScaleFactor:
    """Tests for the ``bucket_size_last_bucket_scale_factor`` option, which shrinks the
    exposed reduce-scatter at the end of backward by shifting params out of the final
    bucket into the preceding bucket."""

    @staticmethod
    def _fake_params(sizes, shared_embedding_last=False):
        out = []
        for i, sz in enumerate(sizes):
            p = torch.nn.Parameter(torch.zeros(sz), requires_grad=False)
            if shared_embedding_last and i == len(sizes) - 1:
                p.shared_embedding = True
            out.append((p, f"p{i}"))
        return out

    @staticmethod
    def _no_own_bucket(p):
        return False

    @staticmethod
    def _shared_embedding_own_bucket(p):
        return getattr(p, 'shared_embedding', False)

    def test_noop_when_scale_factor_is_one(self):
        params = self._fake_params([100] * 10)
        assert (
            _compute_tail_shrink_close_set(
                params_with_names=params,
                bucket_size=200,
                scale_factor=1.0,
                requires_own_bucket_fn=self._no_own_bucket,
            )
            is None
        )

    def test_noop_when_bucket_size_is_none(self):
        params = self._fake_params([100] * 5)
        assert (
            _compute_tail_shrink_close_set(
                params_with_names=params,
                bucket_size=None,
                scale_factor=0.5,
                requires_own_bucket_fn=self._no_own_bucket,
            )
            is None
        )

    def test_shared_embedding_disables_feature(self):
        # With shared-embedding + distopt, the last bucket is forced to be the shared
        # embedding param alone; shrinking the bucket before it would not reduce the
        # exposed tail. The helper must no-op in this case.
        params = self._fake_params([100] * 5, shared_embedding_last=True)
        assert (
            _compute_tail_shrink_close_set(
                params_with_names=params,
                bucket_size=200,
                scale_factor=0.5,
                requires_own_bucket_fn=self._shared_embedding_own_bucket,
            )
            is None
        )

    def test_split_case_carves_small_tail_from_last_full_bucket(self):
        # 10 params of 100, bucket_size=200 -> default layout [200]*5 (tail=0).
        # scale=0.5 -> target=100. Shifting the last close from iter 9 to iter 8 carves
        # a new 100-elem tail off the last full bucket, giving [200]*4 + [100, 100].
        params = self._fake_params([100] * 10)
        assert _compute_tail_shrink_close_set(
            params_with_names=params,
            bucket_size=200,
            scale_factor=0.5,
            requires_own_bucket_fn=self._no_own_bucket,
        ) == {1, 3, 5, 7, 8}

    def test_extend_case_absorbs_tail_into_penultimate(self):
        # 12 params of 100, bucket_size=500 -> default layout [500, 500, 200].
        # scale=0.2 -> target=100. Spread would overshoot (uniform threshold 550 closes
        # each bucket at 600, leaving no tail), so the feature falls back to moving params
        # only into the preceding bucket: shift last close from iter 9 to iter 10, giving
        # [500, 600, 100].
        params = self._fake_params([100] * 12)
        assert _compute_tail_shrink_close_set(
            params_with_names=params,
            bucket_size=500,
            scale_factor=0.2,
            requires_own_bucket_fn=self._no_own_bucket,
        ) == {4, 10}

    def test_spread_case_distributes_surplus_evenly(self):
        # 100 params of 10, bucket_size=100 -> default layout [100]*10 (tail=0, 10 buckets).
        # scale=0.2 -> target=20. Spread threshold = (1000-20)/9 ≈ 108.89, which closes
        # every bucket at 110 elements (11 params). Expected layout: [110]*9 + [10] — every
        # non-last bucket uniformly grows by 10, and a 10-elem tail is carved off. This is
        # the regime where spread dominates penultimate-only (which would produce
        # [100]*9 + [90, 10] — asymmetric).
        params = self._fake_params([10] * 100)
        expected = {10, 21, 32, 43, 54, 65, 76, 87, 98}
        assert (
            _compute_tail_shrink_close_set(
                params_with_names=params,
                bucket_size=100,
                scale_factor=0.2,
                requires_own_bucket_fn=self._no_own_bucket,
            )
            == expected
        )

    def test_noop_when_single_param_fills_last_bucket(self):
        # 3 params of 100, bucket_size=100 -> default [100]*3 (tail=0).
        # Splitting any param below target=50 is impossible (each param is 100). No-op.
        params = self._fake_params([100] * 3)
        assert (
            _compute_tail_shrink_close_set(
                params_with_names=params,
                bucket_size=100,
                scale_factor=0.5,
                requires_own_bucket_fn=self._no_own_bucket,
            )
            is None
        )

    def test_noop_when_moving_tail_would_inflate_penultimate(self):
        # 9 params of 100, bucket_size=200 -> default [200]*4 + [100], tail=100.
        # scale=0.4 -> target=80. The only way to shrink the 100-elem tail is to move
        # the sole tail param into the penultimate, eliminating the tail and pushing
        # the new last bucket (the extended penultimate) to 300 elements — strictly
        # worse. Helper must detect this and no-op.
        params = self._fake_params([100] * 9)
        assert (
            _compute_tail_shrink_close_set(
                params_with_names=params,
                bucket_size=200,
                scale_factor=0.4,
                requires_own_bucket_fn=self._no_own_bucket,
            )
            is None
        )

    def test_noop_when_default_tail_already_meets_target(self):
        # 11 params of 100, bucket_size=200 -> default [200]*5 + [100], tail=100.
        # scale=0.5 -> target=100. Default tail already meets target; no shift.
        params = self._fake_params([100] * 11)
        assert (
            _compute_tail_shrink_close_set(
                params_with_names=params,
                bucket_size=200,
                scale_factor=0.5,
                requires_own_bucket_fn=self._no_own_bucket,
            )
            is None
        )

    @pytest.mark.parametrize(
        "scale_factor,num_layers,bucket_size,expected_numel_per_bucket",
        [
            # scale=1.0 preserves default behavior — proves the feature is off-by-default
            # through the full DDP stack.
            (1.0, 10, 200, [200, 200, 200, 200, 200]),
            # scale<1.0 triggers the feature — proves the close set wires through to the
            # actual bucket layout. The no-op and EXTEND branches are covered by the unit
            # tests above; one passing integration case is enough to prove wiring.
            (0.5, 10, 200, [200, 200, 200, 200, 100, 100]),
        ],
    )
    def test_end_to_end_bucket_layout(
        self, scale_factor, num_layers, bucket_size, expected_numel_per_bucket
    ):
        """End-to-end check: configured DDP produces the expected bucket layout."""
        Utils.initialize_model_parallel()
        # use_distributed_optimizer=False skips per-param and per-bucket padding, so the
        # bucket numels match the raw param counts exactly. input_dim=output_dim=10 with
        # no bias gives a 100-elem weight per layer.
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=False,
            overlap_grad_reduce=True,
            bucket_size=bucket_size,
            bucket_size_last_bucket_scale_factor=scale_factor,
        )
        module = TestModel(
            input_dim=10, output_dim=10, num_layers=num_layers, bias=False, shared_embedding=False
        ).bfloat16()
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            ddp_config=ddp_config,
            module=module,
        )
        buf = model.buffers[0]
        actual = [b.numel_unpadded for b in buf.buckets]
        assert actual == expected_numel_per_bucket, (
            f"scale_factor={scale_factor}, bucket_size={bucket_size}, "
            f"num_layers={num_layers}: expected {expected_numel_per_bucket}, got {actual}"
        )
        Utils.destroy_model_parallel()

    # Lower boundary 0.0 is excluded; just-above upper boundary 1.01 is out of range.
    @pytest.mark.parametrize("invalid_value", [0.0, 1.01])
    def test_config_validation_rejects_out_of_range(self, invalid_value):
        with pytest.raises(AssertionError):
            DistributedDataParallelConfig(bucket_size_last_bucket_scale_factor=invalid_value)

    # Middle and upper-boundary (inclusive) values must be accepted.
    @pytest.mark.parametrize("valid_value", [0.5, 1.0])
    def test_config_validation_accepts_valid_range(self, valid_value):
        DistributedDataParallelConfig(bucket_size_last_bucket_scale_factor=valid_value)


class TestNVFP4IndexMaps:
    """Tests for NVFP4 dual index map (param_index_map and nvfp4_packed_param_index_map).

    These tests mock NVFP4 functions and CUDA so they run on CPU without GPUs.
    The mocking replaces is_nvfp4tensor (to treat regular bf16 params as NVFP4),
    get_nvfp4_rowwise_packed_shape (to halve last dim), modify_nvfp4_rowwise_storage
    (no-op), and torch.cuda.current_device (to allocate on CPU).
    """

    @staticmethod
    def _make_buffer(
        param_shapes,
        nvfp4_param_indices=None,
        use_distributed_optimizer=False,
        bucket_size=None,
        dp_world_size=1,
    ):
        """Create a _ParamAndGradBuffer with some params mocked as NVFP4.

        Args:
            param_shapes: List of (name, shape) tuples for each parameter.
            nvfp4_param_indices: Set of indices into param_shapes to treat as NVFP4.
            use_distributed_optimizer: Whether to use distributed optimizer.
            bucket_size: Bucket size for splitting.
            dp_world_size: Simulated data parallel world size.

        Returns:
            (buffer, params) where params is the ordered list of nn.Parameters.
        """
        params = []
        params_with_names = []
        param_to_name = {}
        for name, shape in param_shapes:
            param = torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16))
            params.append(param)
            params_with_names.append((param, name))
            param_to_name[param] = name

        if nvfp4_param_indices is None:
            nvfp4_param_indices = set()
        nvfp4_params = {params[i] for i in nvfp4_param_indices}
        has_nvfp4 = len(nvfp4_params) > 0

        def mock_is_nvfp4(t):
            return any(t is p for p in nvfp4_params)

        def mock_packed_shape(shape):
            packed = list(shape)
            packed[-1] = packed[-1] // 2
            return torch.Size(packed)

        mock_dp_group = mock.MagicMock()
        mock_dp_group.size.return_value = dp_world_size
        mock_pg = mock.MagicMock()

        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=use_distributed_optimizer,
            overlap_grad_reduce=False,
            bucket_size=bucket_size,
            average_in_collective=False,
        )

        with (
            mock.patch(
                'megatron.core.distributed.param_and_grad_buffer.is_nvfp4tensor',
                side_effect=mock_is_nvfp4,
            ),
            mock.patch(
                'megatron.core.distributed.param_and_grad_buffer.get_nvfp4_rowwise_packed_shape',
                side_effect=mock_packed_shape,
            ),
            mock.patch('megatron.core.fp4_utils.modify_nvfp4_rowwise_storage'),
            mock.patch('torch.cuda.current_device', return_value='cpu'),
            mock.patch(
                'megatron.core.distributed.param_and_grad_buffer.log_on_each_pipeline_stage'
            ),
        ):
            buffer = _ParamAndGradBuffer(
                ddp_config=ddp_config,
                param_dtype=torch.uint8 if has_nvfp4 else torch.bfloat16,
                grad_dtype=torch.bfloat16,
                params_with_names=params_with_names,
                data_parallel_group=mock_dp_group,
                bucket_size=bucket_size,
                param_to_name=param_to_name,
                gradient_scaling_factor=1.0,
                param_indices=list(range(len(params))),
                nccl_ub=False,
                pg_collection=mock_pg,
            )

        return buffer, params

    def test_exact_index_values_no_padding(self):
        """Verify exact index map values for a simple case without distributed optimizer."""
        param_shapes = [('layer0.weight', (100, 100)), ('layer1.weight', (100, 100))]
        buffer, params = self._make_buffer(param_shapes, nvfp4_param_indices={0, 1})

        # Buffer processes params in reverse order: params[1] first, params[0] second.
        assert buffer.param_index_map[params[1]] == (0, 10000, 0)
        assert buffer.param_index_map[params[0]] == (10000, 20000, 0)
        assert buffer.nvfp4_packed_param_index_map[params[1]] == (0, 5000, 0)
        assert buffer.nvfp4_packed_param_index_map[params[0]] == (5000, 10000, 0)

        assert buffer.numel == 20000
        assert buffer.nvfp4_packed_numel == 10000

    def test_non_nvfp4_exact_values_match_original(self):
        """Non-NVFP4 param_index_map values should be identical to original behavior."""
        param_shapes = [('layer0.weight', (100, 100)), ('layer1.weight', (100, 100))]
        buffer, params = self._make_buffer(param_shapes)

        # Without NVFP4 or distributed optimizer, no padding: offsets are contiguous.
        assert buffer.param_index_map[params[1]] == (0, 10000, 0)
        assert buffer.param_index_map[params[0]] == (10000, 20000, 0)
        assert buffer.numel == 20000

    def test_nvfp4_multi_bucket_param_to_index(self):
        """param_to_index in each bucket should be relative to that bucket's full-numel offset."""
        param_shapes = [
            ('layer0.weight', (100, 100)),
            ('layer1.weight', (100, 100)),
            ('layer2.weight', (100, 100)),
            ('layer3.weight', (100, 100)),
        ]
        # bucket_size=15000: each param is 10000 full numel, so 2 params per bucket.
        buffer, params = self._make_buffer(
            param_shapes, nvfp4_param_indices={0, 1, 2, 3}, bucket_size=15000
        )

        assert len(buffer.buckets) == 2
        for bucket in buffer.buckets:
            for param in bucket.params_list:
                global_start, global_end, _ = buffer.param_index_map[param]
                local_start, local_end = bucket.param_to_index[param]
                assert local_start == global_start - bucket.offset
                assert local_end == global_end - bucket.offset
                assert local_end - local_start == param.data.nelement()

    @pytest.mark.parametrize("dp_world_size", [1, 2, 4, 8])
    def test_nvfp4_with_distributed_optimizer(self, dp_world_size):
        """With distributed optimizer, both packed and unpacked indices should be padded."""
        param_shapes = [('layer0.weight', (98, 101)), ('layer1.weight', (98, 101))]
        buffer, params = self._make_buffer(
            param_shapes,
            nvfp4_param_indices={0, 1},
            use_distributed_optimizer=True,
            dp_world_size=dp_world_size,
        )

        # Param starts should be 64-aligned in both maps.
        for param in params:
            start, end, _ = buffer.param_index_map[param]
            assert start % 64 == 0, f"Unpacked start {start} should be 64-aligned"
            assert end - start == param.data.nelement()

        for param in params:
            start, end, _ = buffer.nvfp4_packed_param_index_map[param]
            assert start % 64 == 0, f"Packed start {start} should be 64-aligned"
            assert end - start == param.data.nelement() // 2

        # Buffer numel should be divisible by dp_world_size.
        assert buffer.numel % dp_world_size == 0
        assert buffer.nvfp4_packed_numel % dp_world_size == 0

    def test_nvfp4_mixed_params(self):
        """Test buffer with a mix of NVFP4 and non-NVFP4 params."""
        param_shapes = [
            ('linear.weight', (100, 100)),  # NVFP4.
            ('layernorm.weight', (100,)),  # Non-NVFP4 (bf16).
        ]
        buffer, params = self._make_buffer(param_shapes, nvfp4_param_indices={0})

        # Non-NVFP4 param should have same span in both maps.
        packed_start, packed_end, _ = buffer.nvfp4_packed_param_index_map[params[1]]
        unpacked_start, unpacked_end, _ = buffer.param_index_map[params[1]]
        assert packed_end - packed_start == unpacked_end - unpacked_start == 100

        # NVFP4 param should have half the span in packed map.
        packed_start, packed_end, _ = buffer.nvfp4_packed_param_index_map[params[0]]
        unpacked_start, unpacked_end, _ = buffer.param_index_map[params[0]]
        assert packed_end - packed_start == 5000  # numel // 2.
        assert unpacked_end - unpacked_start == 10000  # Full numel.

    def test_nvfp4_varied_param_sizes(self):
        """Test with different param sizes to verify offsets accumulate correctly."""
        param_shapes = [('small.weight', (10, 20)), ('large.weight', (100, 200))]
        buffer, params = self._make_buffer(param_shapes, nvfp4_param_indices={0, 1})

        # Reversed order: large (params[1]) processed first, then small (params[0]).
        large_packed_start = 0
        large_packed_end = 100 * 200 // 2  # 10000.
        small_packed_start = large_packed_end
        small_packed_end = small_packed_start + 10 * 20 // 2  # 10100.

        assert buffer.nvfp4_packed_param_index_map[params[1]] == (
            large_packed_start,
            large_packed_end,
            0,
        )
        assert buffer.nvfp4_packed_param_index_map[params[0]] == (
            small_packed_start,
            small_packed_end,
            0,
        )

        large_unpacked_start = 0
        large_unpacked_end = 100 * 200  # 20000.
        small_unpacked_start = large_unpacked_end
        small_unpacked_end = small_unpacked_start + 10 * 20  # 20200.

        assert buffer.param_index_map[params[1]] == (large_unpacked_start, large_unpacked_end, 0)
        assert buffer.param_index_map[params[0]] == (small_unpacked_start, small_unpacked_end, 0)

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings

import pytest
import torch

from megatron.core.extensions.transformer_engine import te_general_gemm
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import (
    VocabParallelEmbedding,
    copy_gtp_attributes,
    gtp_local_pad_zero_count,
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce,
)
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import init_method_normal
from tests.unit_tests.stream_contention import SideStreamContention, bit_equal, replay_count
from tests.unit_tests.test_utilities import Utils


class _FakeGroup:
    """Minimal mock for a dist process group — used in single-process unit tests."""

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class TestGtpLocalPadZeroCount:
    """gtp_local_pad_zero_count: how many elements of a [range_start, range_end) fragment of a
    GTP shard's flattened buffer are structural alignment padding (see
    generalized_tensor_parallelism._gtp_slice_one_param). Padding is a contiguous suffix of the
    *unsharded* padded buffer, sliced evenly across the GTP group -- usually it lands entirely on
    the last rank, but when pad_length exceeds one shard's own row count (small dim0 relative to
    pad_for_alignment * gtp_remat_size) it spills backward from the tail into lower-numbered
    ranks' shards too."""

    @staticmethod
    def _shard(dim0, dim1, pad_length, group):
        shard = torch.zeros(dim0, dim1)
        shard.pad_length = pad_length
        shard.group = group
        return shard

    @pytest.mark.parametrize(
        "pad_length,rank",
        [
            (0, 3),  # no padding at all (tail rank)
            # pad_length is stamped identically on every rank's shard object
            # (_gtp_slice_one_param), but when padding is smaller than one shard, only the
            # tail shard physically contains it -- rank 0 here has none.
            (3, 0),
        ],
        ids=["no_padding", "non_tail_rank_padding_fits_in_tail_shard"],
    )
    def test_returns_zero_when_this_rank_has_no_local_padding(self, pad_length, rank):
        shard = self._shard(8, 4, pad_length, group=_FakeGroup(size=4, rank=rank))
        assert gtp_local_pad_zero_count(shard, 0, shard.numel()) == 0

    def test_padding_spills_into_earlier_ranks_when_larger_than_one_shard(self):
        # dim0=1, pad_for_alignment=16, gtp_remat_size=4 -> alignment=64, pad_length=63,
        # shard_dim0=16: padding (63 rows) exceeds one shard's own row count (16), so every rank
        # except rank 0 is entirely padding, and rank 0 is 1 real row + 15 padding rows.
        dim0, pad_length, trailing = 16, 63, 8
        group_size = 4
        totals = []
        for rank in range(group_size):
            shard = self._shard(dim0, trailing, pad_length, group=_FakeGroup(group_size, rank))
            totals.append(gtp_local_pad_zero_count(shard, 0, shard.numel()))
        assert totals == [15 * trailing, 16 * trailing, 16 * trailing, 16 * trailing]
        assert sum(totals) == pad_length * trailing

    # dim0=8, dim1=4, pad_length=3 -> shard.numel()=32, pad_start=32-3*4=20. DP-optimizer bucket
    # slicing can hand count_zeros_fp32 any [range_start, range_end) fragment of this shard.
    @pytest.mark.parametrize(
        "range_start,range_end,expected",
        [
            (0, 32, 12),  # whole range: all 12 pad elements
            (0, 20, 0),  # fragment strictly before the pad region
            (18, 32, 12),  # fragment straddling the boundary: only the pad portion counts
            (21, 31, 10),  # fragment entirely inside padding: every element counts
        ],
        ids=["whole_range", "strictly_before_pad", "straddles_pad_boundary", "entirely_inside_pad"],
    )
    def test_tail_rank_fragment_ranges(self, range_start, range_end, expected):
        shard = self._shard(8, 4, pad_length=3, group=_FakeGroup(size=4, rank=3))
        assert gtp_local_pad_zero_count(shard, range_start, range_end) == expected

    def test_no_gtp_group_returns_zero(self):
        shard = torch.zeros(8, 4)
        shard.pad_length = 3
        assert gtp_local_pad_zero_count(shard, 0, shard.numel()) == 0


class TestCopyGtpAttributes:
    """copy_gtp_attributes(destination, source) must carry pad_length/group onto any param
    view/copy (e.g. an optimizer's master param) -- regression coverage for a real bug where
    a prior version dropped one of the two, which silently disabled GTP padding exclusion for
    optimizers relying on this fallback (e.g. LayerWiseDistributedOptimizer/Muon)."""

    def test_padding_exclusion_survives_the_copy(self):
        """End-to-end: gtp_local_pad_zero_count on a copy must match the original -- this is
        exactly the check that would have caught the pad_length/group-dropping bug directly."""
        source = TestGtpLocalPadZeroCount._shard(
            8, 4, pad_length=3, group=_FakeGroup(size=4, rank=3)
        )

        destination = torch.zeros(8, 4)
        copy_gtp_attributes(destination, source)

        expected = gtp_local_pad_zero_count(source, 0, source.numel())
        assert expected > 0, "test setup must exercise real padding"
        assert gtp_local_pad_zero_count(destination, 0, destination.numel()) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_linear_default_output_dtype_preserves_input_dtype(dtype):
    Utils.initialize_model_parallel(1, 1)

    try:
        input_data = torch.randn(4, 3, 16, device="cuda", dtype=dtype)
        weight = torch.randn(32, 16, device="cuda", dtype=dtype)
        output = linear_with_grad_accumulation_and_async_allreduce(
            input_data, weight, None, False, False, False, tp_group=None, output_dtype=None
        )
        reference = torch.nn.functional.linear(input_data, weight)

        assert output.dtype == input_data.dtype
        torch.testing.assert_close(output, reference)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("tensor_parallel,allreduce_dgrad", [(1, False), (8, True)])
def test_LinearWithFrozenWeight(tensor_parallel, allreduce_dgrad):
    Utils.initialize_model_parallel(tensor_parallel, 1)

    size_per_partition = int(8 / tensor_parallel)

    # Input is an 8x8 identity matrix.
    input_data = torch.eye(8).cuda()
    input_data.requires_grad = True

    # Weight is an 8x8 matrix of all ones. If tensor parallelism > 1, the weight is partitioned evenly across GPUs.
    weight = torch.ones((size_per_partition, 8)).cuda()

    # Bias is a vector of length 8 of all zeros. If tensor parallelism > 1, the bias is partitioned evenly across GPUs
    bias = torch.zeros((size_per_partition)).cuda()

    gradient_accumulation_fusion = False
    sequence_parallel = False
    grad_output_buffer = None
    wgrad_deferral_limit = None

    output_parallel = linear_with_frozen_weight(
        input_data,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    )
    output = gather_from_tensor_model_parallel_region(
        output_parallel
    )  # no-op if tensor_parallel == 1.
    output.sum().backward()

    expected_output = torch.ones(8).cuda()
    expected_grad = 8 * torch.ones(8).cuda()

    assert torch.allclose(output, expected_output)
    assert torch.allclose(input_data.grad, expected_grad)

    Utils.destroy_model_parallel()


def test_LinearWithFrozenWeight_3d_input_matches_torch_linear():
    Utils.initialize_model_parallel(1, 1)

    input_data = torch.randn(4, 3, 8, device="cuda", requires_grad=True)
    weight = torch.randn(6, 8, device="cuda")
    bias = torch.randn(6, device="cuda")

    expected_input = input_data.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.linear(expected_input, weight, bias)
    expected.sum().backward()

    actual = linear_with_frozen_weight(input_data, weight, bias, False, False, False, None, None)
    actual.sum().backward()

    assert torch.allclose(actual, expected)
    assert torch.allclose(input_data.grad, expected_input.grad)

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    te_general_gemm is None, reason="Transformer Engine general_gemm is not available"
)
def test_linear_with_grad_accumulation_supports_fp32_output_and_bf16_backward():
    Utils.initialize_model_parallel(1, 1)

    input_data = torch.randn(4, 3, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_input = input_data.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)

    output = linear_with_grad_accumulation_and_async_allreduce(
        input_data, weight, None, False, False, False, tp_group=None, output_dtype=torch.float32
    )
    output.sum().backward()

    reference_output = torch.nn.functional.linear(reference_input, reference_weight)
    reference_output.sum().backward()
    fp32_reference_output = torch.nn.functional.linear(
        input_data.detach().float(), weight.detach().float()
    )

    assert output.dtype == torch.float32
    assert input_data.grad.dtype == torch.bfloat16
    assert weight.grad.dtype == torch.bfloat16
    assert torch.allclose(output, fp32_reference_output, atol=1e-4, rtol=1e-4)
    assert torch.allclose(input_data.grad, reference_input.grad)
    assert torch.allclose(weight.grad, reference_weight.grad)

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    te_general_gemm is None, reason="Transformer Engine general_gemm is not available"
)
def test_linear_fp32_output_is_bitwise_exact_for_integer_bf16_operands():
    Utils.initialize_model_parallel(1, 1)

    generator = torch.Generator(device="cuda").manual_seed(1234)
    input_data = torch.randint(
        -8, 9, (4, 3, 512), device="cuda", dtype=torch.int32, generator=generator
    ).to(torch.bfloat16)
    weight = torch.randint(
        -8, 9, (128, 512), device="cuda", dtype=torch.int32, generator=generator
    ).to(torch.bfloat16)

    output = linear_with_grad_accumulation_and_async_allreduce(
        input_data, weight, None, False, False, False, tp_group=None, output_dtype=torch.float32
    )
    reference = torch.nn.functional.linear(input_data.float(), weight.float())

    # K * 8^2 = 32,768, so every possible integer product and partial sum is
    # exactly representable in FP32. Compare raw words instead of using a tolerance.
    assert output.dtype == torch.float32
    assert torch.equal(output.contiguous().view(torch.int32), reference.view(torch.int32))

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    te_general_gemm is None, reason="Transformer Engine general_gemm is not available"
)
def test_linear_fp32_output_matches_plain_te_general_gemm():
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    try:
        from transformer_engine.pytorch.module.base import get_workspace
    except ImportError:
        get_workspace = None

    Utils.initialize_model_parallel(1, 1)

    input_data = torch.randn(4, 3, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)
    wrapped_output = linear_with_grad_accumulation_and_async_allreduce(
        input_data, weight, None, False, False, False, tp_group=None, output_dtype=torch.float32
    )

    kwargs = {
        "out_dtype": torch.float32,
        "quantization_params": None,
        "gelu": None,
        "gelu_in": None,
        "accumulate": False,
        "layout": "TN",
        "out": None,
        "bias": None,
        "use_split_accumulator": False,
        "grad": False,
        "ub": None,
        "ub_type": None,
        "extra_output": None,
        "bulk_overlap": False,
    }
    if get_workspace is not None:
        kwargs["workspace"] = get_workspace()
    plain_te_output = general_gemm(weight, input_data.reshape(-1, 64), **kwargs)[0]
    plain_te_output = plain_te_output.reshape_as(wrapped_output)

    assert wrapped_output.dtype == torch.float32
    assert torch.equal(
        wrapped_output.contiguous().view(torch.int32),
        plain_te_output.contiguous().view(torch.int32),
    )

    Utils.destroy_model_parallel()


class TestVocabParallelEmbeddingDeterminism:
    """The embedding lookup is a single code path whose backward is bit-reproducible.

    ``F.embedding``'s CUDA backward sorts the ids and reduces each row's duplicates in a fixed
    order, so ``VocabParallelEmbedding`` needs no ``deterministic_mode`` special case. The ids
    are heavily duplicated (64 distinct ids over 32k positions plus one fully padded sequence),
    so thousands of rows accumulate into the same embedding row -- where an atomic reduction
    would show up as bit differences between replays.

    torch >= 2.11 adds a fused atomic path for tables with few segments
    (``min(num_ids, num_rows) * ceil(hidden / 1024) < 4 * num_SMs``); it is disabled under
    ``torch.use_deterministic_algorithms(True)`` and out of reach for these shapes (8192 rows,
    32k ids, hidden 1024), so both flag settings are expected to replay bit-exactly here.

    Replays run under side-stream contention when the process has more than one hardware queue;
    the unit-test bucket pins ``CUDA_DEVICE_MAX_CONNECTIONS=1``, where contention is a no-op and
    more replays are taken instead (see ``tests/unit_tests/stream_contention.py``).
    """

    @staticmethod
    def _module_and_inputs(deterministic_mode):
        model_parallel_cuda_manual_seed(123)
        config = ModelParallelConfig(
            params_dtype=torch.bfloat16, bf16=True, deterministic_mode=deterministic_mode
        )
        module = VocabParallelEmbedding(
            8192, 1024, init_method=init_method_normal(0.02), config=config
        ).cuda()
        torch.manual_seed(0)
        ids = torch.randint(0, 64, (8, 4096), device="cuda") * 100
        ids[0] = 3  # a fully padded sequence: 4096 copies of one id
        grad_output = torch.randn(8, 4096, 1024, device="cuda", dtype=torch.bfloat16)
        return module, ids, grad_output

    @staticmethod
    def _forward_backward(module, ids, grad_output):
        module.weight.grad = None
        out = module(ids)
        out.backward(grad_output)
        torch.cuda.synchronize()
        return out.detach().clone(), module.weight.grad.detach().clone()

    @pytest.mark.parametrize("deterministic_algorithms", [False, True])
    def test_backward_replays_bit_exact(self, deterministic_algorithms):
        Utils.initialize_model_parallel(1, 1)
        prev = torch.are_deterministic_algorithms_enabled()
        prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
        try:
            module, ids, grad_output = self._module_and_inputs(deterministic_algorithms)
            ref_out, ref_grad = self._forward_backward(module, ids, grad_output)
            for _ in range(replay_count() - 1):
                with SideStreamContention():
                    out, grad = self._forward_backward(module, ids, grad_output)
                assert bit_equal(out, ref_out)
                assert bit_equal(grad, ref_grad)
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)
            Utils.destroy_model_parallel()

    def test_deterministic_mode_does_not_change_the_lookup(self):
        """Both config settings run the same kernel: outputs and weight grads are bit-identical."""
        Utils.initialize_model_parallel(1, 1)
        try:
            results = []
            for deterministic_mode in (False, True):
                module, ids, grad_output = self._module_and_inputs(deterministic_mode)
                with warnings.catch_warnings():
                    # deterministic_mode without the torch flag warns once; not what is under test.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    results.append(self._forward_backward(module, ids, grad_output))
            (out_a, grad_a), (out_b, grad_b) = results
            assert bit_equal(out_a, out_b)
            assert bit_equal(grad_a, grad_b)
        finally:
            Utils.destroy_model_parallel()

    def test_warns_once_when_deterministic_mode_lacks_the_torch_flag(self):
        """config.deterministic_mode relies on torch.use_deterministic_algorithms(True); say so."""
        Utils.initialize_model_parallel(1, 1)
        prev = torch.are_deterministic_algorithms_enabled()
        prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            module, ids, _ = self._module_and_inputs(deterministic_mode=True)
            with pytest.warns(RuntimeWarning, match="use_deterministic_algorithms"):
                module(ids)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                module(ids)  # second forward: no repeat
            torch.use_deterministic_algorithms(True, warn_only=True)
            module, ids, _ = self._module_and_inputs(deterministic_mode=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                module(ids)  # flag set: nothing to warn about
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)
            Utils.destroy_model_parallel()

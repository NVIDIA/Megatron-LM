# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch

from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_get_slice_after,
    tensor_masked_update,
    tensor_merge,
)


def tensor_get_slice_after_pytorch(
    input_tensor: torch.Tensor, output_tensor: torch.Tensor, pos_on_device: torch.Tensor
) -> None:
    """Reference PyTorch implementation of tensor_get_slice_after."""

    assert input_tensor.ndim == output_tensor.ndim, "Rank mismatch"
    for i in range(1, input_tensor.ndim):
        assert input_tensor.shape[i] == output_tensor.shape[i], f"Dimension {i} must match"

    pos = pos_on_device[0].item()
    assert 0 <= pos <= input_tensor.shape[0]

    copy_size = min(input_tensor.shape[0] - pos, output_tensor.shape[0])
    if copy_size > 0:
        output_tensor[:copy_size].copy_(input_tensor[pos : pos + copy_size])


def tensor_merge_pytorch(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    output_tensor: torch.Tensor,
    pos_on_device: torch.Tensor,
) -> None:
    """Reference PyTorch implementation of tensor_merge."""

    assert tensor_a.ndim == tensor_b.ndim == output_tensor.ndim, "Rank mismatch across tensors"
    for i in range(1, tensor_a.ndim):
        assert (
            tensor_a.shape[i] == tensor_b.shape[i] == output_tensor.shape[i]
        ), f"Dimension {i} must match"

    pos = pos_on_device[0].item()
    assert 0 <= pos <= tensor_a.shape[0]
    assert output_tensor.shape[0] >= tensor_a.shape[0]

    if pos > 0:
        output_tensor[:pos].copy_(tensor_a[:pos])

    copy_size = min(tensor_b.shape[0], output_tensor.shape[0] - pos)
    if copy_size > 0:
        output_tensor[pos : pos + copy_size].copy_(tensor_b[:copy_size])


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def slice_params():
    return {"input_batch": 16, "output_batch": 20, "feature_dim": 256}


@pytest.mark.parametrize("pos_dtype", [torch.int64, torch.int32])
def test_get_slice_after_basic(device, slice_params, pos_dtype):
    params = slice_params
    input_tensor = torch.randn(params["input_batch"], params["feature_dim"], device=device)
    # The SSM layer passes an int32 position (MambaMetadata.device_decode_prefill).
    pos_on_device = torch.tensor([5], dtype=pos_dtype, device=device)

    output_ref = torch.zeros(params["output_batch"], params["feature_dim"], device=device)
    output_triton = torch.zeros_like(output_ref)
    output_ref[15:] = 123.0
    output_triton[15:] = 123.0

    tensor_get_slice_after_pytorch(input_tensor, output_ref, pos_on_device)
    tensor_get_slice_after(input_tensor, output_triton, pos_on_device, check_bounds=True)

    assert torch.equal(output_ref, output_triton)
    assert torch.equal(
        output_triton[: params["input_batch"] - pos_on_device[0].item()],
        input_tensor[pos_on_device[0].item() :],
    )


def test_get_slice_after_pos_zero(device, slice_params):
    params = slice_params
    input_tensor = torch.randn(params["input_batch"], params["feature_dim"], device=device)
    output_tensor = torch.zeros(params["output_batch"], params["feature_dim"], device=device)

    tensor_get_slice_after(
        input_tensor, output_tensor, torch.tensor([0], device=device), check_bounds=True
    )

    copy_size = min(params["input_batch"], params["output_batch"])
    assert torch.equal(output_tensor[:copy_size], input_tensor[:copy_size])


def test_get_slice_after_pos_full(device, slice_params):
    params = slice_params
    input_tensor = torch.randn(params["input_batch"], params["feature_dim"], device=device)
    output_tensor = torch.ones(params["output_batch"], params["feature_dim"], device=device)
    original = output_tensor.clone()

    tensor_get_slice_after(
        input_tensor,
        output_tensor,
        torch.tensor([params["input_batch"]], device=device),
        check_bounds=True,
    )

    assert torch.equal(output_tensor, original)


def test_get_slice_after_exact_fit(device):
    input_tensor = torch.randn(8, 256, device=device)
    output_tensor = torch.zeros(5, 256, device=device)

    tensor_get_slice_after(input_tensor, output_tensor, torch.tensor([3], device=device))

    assert torch.equal(output_tensor, input_tensor[3:8])


def test_get_slice_after_nd(device):
    input_tensor = torch.randn(6, 4, 8, device=device)
    output_tensor = torch.zeros(10, 4, 8, device=device)

    tensor_get_slice_after(
        input_tensor, output_tensor, torch.tensor([1], device=device), check_bounds=True
    )

    assert torch.equal(output_tensor[:5], input_tensor[1:6])


def test_get_slice_after_bounds(device, slice_params):
    params = slice_params
    input_tensor = torch.randn(params["input_batch"], params["feature_dim"], device=device)
    output_tensor = torch.zeros(params["output_batch"], params["feature_dim"], device=device)

    with pytest.raises(AssertionError):
        tensor_get_slice_after(
            input_tensor,
            output_tensor,
            torch.tensor([params["input_batch"] + 1], device=device),
            check_bounds=True,
        )


def test_get_slice_after_consistency(device):
    input_tensor = torch.randn(32, 128, device=device)
    output_ref = torch.zeros(16, 128, device=device)
    output_triton = torch.zeros_like(output_ref)
    pos_on_device = torch.tensor([8], device=device)

    tensor_get_slice_after_pytorch(input_tensor, output_ref, pos_on_device)
    tensor_get_slice_after(input_tensor, output_triton, pos_on_device)

    assert torch.equal(output_ref, output_triton)


@pytest.fixture
def merge_params():
    return {"tensor_a_batch": 8, "tensor_b_batch": 12, "output_batch": 32, "feature_dim": 256}


@pytest.mark.parametrize("pos_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("in_place", [False, True])
def test_tensor_merge_basic(device, merge_params, in_place, pos_dtype):
    params = merge_params
    pos_val = 5
    pos_on_device = torch.tensor([pos_val], dtype=pos_dtype, device=device)

    tensor_b = torch.randn(params["tensor_b_batch"], params["feature_dim"], device=device)

    if in_place:
        tensor_a = torch.randn(params["output_batch"], params["feature_dim"], device=device)
        output_triton = tensor_a.clone()

        output_ref = tensor_a.clone()
        tensor_merge_pytorch(tensor_a, tensor_b, output_ref, pos_on_device)
        tensor_merge(output_triton, tensor_b, pos_on_device, output_tensor=None, check_bounds=True)
    else:
        tensor_a = torch.randn(params["tensor_a_batch"], params["feature_dim"], device=device)
        output_ref = torch.zeros(params["output_batch"], params["feature_dim"], device=device)
        output_triton = torch.zeros_like(output_ref)

        tensor_merge_pytorch(tensor_a, tensor_b, output_ref, pos_on_device)
        tensor_merge(
            tensor_a, tensor_b, pos_on_device, output_tensor=output_triton, check_bounds=True
        )

    assert torch.equal(output_ref, output_triton)
    assert torch.equal(output_triton[:pos_val], tensor_a[:pos_val])
    assert torch.equal(output_triton[pos_val : pos_val + params["tensor_b_batch"]], tensor_b)


def test_tensor_merge_pos_zero(device, merge_params):
    params = merge_params
    tensor_a = torch.randn(params["tensor_a_batch"], params["feature_dim"], device=device)
    tensor_b = torch.randn(params["tensor_b_batch"], params["feature_dim"], device=device)
    output_tensor = torch.zeros(params["output_batch"], params["feature_dim"], device=device)

    tensor_merge(
        tensor_a,
        tensor_b,
        torch.tensor([0], device=device),
        output_tensor=output_tensor,
        check_bounds=True,
    )

    assert torch.equal(output_tensor[: params["tensor_b_batch"]], tensor_b)


def test_tensor_merge_pos_full(device, merge_params):
    params = merge_params
    tensor_a = torch.randn(params["tensor_a_batch"], params["feature_dim"], device=device)
    tensor_b = torch.randn(params["tensor_b_batch"], params["feature_dim"], device=device)
    output_tensor = torch.zeros(params["output_batch"], params["feature_dim"], device=device)

    tensor_merge(
        tensor_a,
        tensor_b,
        torch.tensor([params["tensor_a_batch"]], device=device),
        output_tensor=output_tensor,
        check_bounds=True,
    )

    assert torch.equal(output_tensor[: params["tensor_a_batch"]], tensor_a)
    assert torch.equal(
        output_tensor[
            params["tensor_a_batch"] : params["tensor_a_batch"] + params["tensor_b_batch"]
        ],
        tensor_b,
    )


def test_tensor_merge_small(device):
    tensor_a = torch.randn(3, 256, device=device)
    tensor_b = torch.randn(5, 256, device=device)
    output_tensor = torch.zeros(10, 256, device=device)

    tensor_merge(tensor_a, tensor_b, torch.tensor([2], device=device), output_tensor=output_tensor)

    assert torch.equal(output_tensor[:2], tensor_a[:2])
    assert torch.equal(output_tensor[2:7], tensor_b)


IN_PROJ_ROW = 2 * 8192 + 2 * 8 * 128 + 128
"""Elements per token of Nemotron-H 8B's Mamba in_proj output at TP 1 (z, x, B, C, dt): the rows
that the SSM layer's mixed-batch slice copies."""
D_INNER_ROW = 8192
"""Elements per token of Nemotron-H 8B's SSM output at TP 1: the rows that the merge copies."""
INT32_GUARD = 2**31
"""Elements in front of a large case's tensors. An element offset in [2**31, 2**32) computed in
int32 wraps to offset - 2**32, which lands here instead of faulting."""


def _behind_front_guard(shape, fill, device):
    """``(tensor, guard)``: a bf16 ``shape`` after ``INT32_GUARD`` elements, all set to ``fill``."""
    buffer = torch.full(
        (INT32_GUARD + math.prod(shape),), fill, dtype=torch.bfloat16, device=device
    )
    return buffer[INT32_GUARD:].view(shape), buffer[:INT32_GUARD]


def _require_free_memory(nbytes):
    """Skip, with the numbers, when the device has less than ``nbytes`` free."""
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < nbytes:
        pytest.skip(f"needs {nbytes / 2**30:.1f} GiB of device memory, {free / 2**30:.1f} free")


@pytest.mark.large
class TestInt32RowOffsets:
    """Rows that start at or past 2**31 elements, where int32 row offsets would wrap.

    The sources sit behind zeros and the output behind NaN, so a wrapped read leaves an output
    row unequal to its source and a wrapped write shows in the output's guard. Opt in with
    ``-m large`` (about 20 GiB of device memory each); run one case per process, since a wrapped
    offset can leave a sticky illegal-address error.
    """

    def test_get_slice_after(self, device):
        """The last of 115,706 in_proj rows starts past 2**31 elements. With position 0 it is read
        and written at that offset."""
        rows = INT32_GUARD // IN_PROJ_ROW + 2
        _require_free_memory(22 << 30)
        x, _ = _behind_front_guard((rows, 1, IN_PROJ_ROW), 0.0, device)
        x.normal_()
        out, out_guard = _behind_front_guard((rows, 1, IN_PROJ_ROW), float("nan"), device)
        pos = torch.tensor([0, rows], dtype=torch.int32, device=device)

        tensor_get_slice_after(x, out, pos)

        assert out_guard.isnan().all(), "wrote before the output"
        assert torch.equal(out, x)

    @pytest.mark.parametrize("source", ["tensor_a", "tensor_b"])
    def test_tensor_merge(self, device, source):
        """The last of 262,145 SSM output rows starts at 2**31 elements. It comes from tensor_a
        (position = rows) or tensor_b (position 0), so both of the kernel's source offsets reach
        it."""
        rows = INT32_GUARD // D_INNER_ROW + 1
        _require_free_memory(22 << 30)
        src, _ = _behind_front_guard((rows, 1, D_INNER_ROW), 0.0, device)
        src.normal_()
        other = torch.randn(3, 1, D_INNER_ROW, dtype=torch.bfloat16, device=device)
        out, out_guard = _behind_front_guard((rows, 1, D_INNER_ROW), float("nan"), device)
        tensor_a, tensor_b, pos_val = (
            (src, other, rows) if source == "tensor_a" else (other, src, 0)
        )
        pos = torch.tensor([pos_val, rows - pos_val], dtype=torch.int32, device=device)

        tensor_merge(tensor_a, tensor_b, pos, output_tensor=out)

        assert out_guard.isnan().all(), "wrote before the output"
        assert torch.equal(out, src)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_tensor_masked_update(device, ndim):
    """
    Tests tensor_masked_update for 2D, 3D, and 4D tensors.
    Covering 3 scenarios:
    1. idx has only valid values (arbitrary order).
    2. idx has mixed valid values and -1s (all -1s at the end).
    3. idx has all -1s.
    """

    num_states = 32
    batch_size = 8

    # Define shapes based on dimensionality
    if ndim == 2:
        shape_states = (num_states, 64)
        shape_new = (batch_size, 64)
    elif ndim == 3:
        shape_states = (num_states, 8, 8)
        shape_new = (batch_size, 8, 8)
    elif ndim == 4:
        shape_states = (num_states, 4, 4, 4)
        shape_new = (batch_size, 4, 4, 4)

    def allocate_tensors():
        states = torch.randn(shape_states, device=device)
        new_states = torch.randn(shape_new, device=device)
        return states, new_states

    # Scenario 1: no -1s
    states, new_states = allocate_tensors()
    idx = torch.randperm(num_states, device=device)[:batch_size]
    expected_states = states.clone()
    expected_states[idx] = new_states
    tensor_masked_update(states, idx, new_states)
    assert torch.equal(states, expected_states), f"Failed {ndim}D: all valid idx values"

    # Scenario 2: mix of regular values and -1s
    states, new_states = allocate_tensors()
    num_valid = batch_size // 2
    valid_indices = torch.randperm(num_states, device=device)[:num_valid]
    idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    idx[:num_valid] = valid_indices
    expected_states = states.clone()
    expected_states[valid_indices] = new_states[:num_valid]
    tensor_masked_update(states, idx, new_states)
    assert torch.equal(states, expected_states), f"Failed {ndim}D: mix of valid and mask values"

    # Scenario 3: all -1s
    states, new_states = allocate_tensors()
    idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    expected_states = states.clone()
    tensor_masked_update(states, idx, new_states)
    assert torch.equal(states, expected_states), f"Failed {ndim}D: all mask values"

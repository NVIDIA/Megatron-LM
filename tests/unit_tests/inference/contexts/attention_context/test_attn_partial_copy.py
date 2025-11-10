import torch
import pytest
import time
from megatron.core.inference.contexts.attention_context.triton.attn_partial_copy import (
    attn_partial_copy_triton,
)


def _assert_tensor_shapes_compatible(
    input_tensor: torch.Tensor, output_tensor: torch.Tensor, name: str
):
    """Assert that tensors have compatible shapes for partial copying."""
    assert (
        input_tensor.ndim == output_tensor.ndim
    ), f"{name}: Input and output tensors must have the same number of dimensions"

    # Check that all dimensions except the first (batch) are the same
    for i in range(1, input_tensor.ndim):
        assert (
            input_tensor.shape[i] == output_tensor.shape[i]
        ), f"{name}: Dimension {i} must match between input and output tensors"


def _assert_device_dc_valid(
    device_dc_tensor: torch.Tensor, input_tensor: torch.Tensor, output_tensor: torch.Tensor
):
    """Assert that device_dc is within valid bounds."""
    device_dc = device_dc_tensor[0].item()
    assert (
        0 <= device_dc <= input_tensor.shape[0]
    ), f"device_dc ({device_dc}) must be between 0 and input_tensor.shape[0] ({input_tensor.shape[0]})"

    # Check that we have enough space in output tensor
    copy_size = input_tensor.shape[0] - device_dc
    assert (
        copy_size <= output_tensor.shape[0]
    ), f"Copy size ({copy_size}) exceeds output_tensor.shape[0] ({output_tensor.shape[0]})"


def attn_partial_copy_pytorch(
    input_tensor: torch.Tensor, output_tensor: torch.Tensor, device_dc: torch.Tensor
) -> None:
    """
    PyTorch reference implementation for attn_partial_copy_triton.

    Args:
        input_tensor: Input tensor to copy from, shape [input_batch, ...]
        output_tensor: Output tensor to write to, shape [output_batch, ...]
        device_dc: Tensor with decode count in first element, shape [2] (e.g., [dc_count, pf_count])
    """

    _assert_tensor_shapes_compatible(input_tensor, output_tensor, "attn_partial_copy_pytorch")
    _assert_device_dc_valid(device_dc, input_tensor, output_tensor)

    # Extract device_dc value from tensor
    device_dc_val = device_dc[0].item()

    # Calculate copy size
    input_batch_size = input_tensor.shape[0]
    output_batch_size = output_tensor.shape[0]
    copy_size = min(input_batch_size - device_dc_val, output_batch_size)

    # Copy from input[device_dc:device_dc+copy_size] to output[0:copy_size]
    if copy_size > 0:
        output_tensor[:copy_size].copy_(input_tensor[device_dc_val : device_dc_val + copy_size])


@pytest.fixture
def device():
    """Get CUDA device if available, otherwise skip tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.fixture
def test_params():
    """Common test parameters."""
    return {'input_batch': 16, 'output_batch': 20, 'feature_dim': 256}


def test_basic_functionality(device, test_params):
    """Test basic partial copy functionality."""
    input_batch = test_params['input_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    # Create test tensors
    input_tensor = torch.randn(input_batch, feature_dim, dtype=torch.float32, device=device)
    device_dc = torch.tensor([5, 0], device=device)
    device_dc_val = device_dc[0].item()

    # Test with PyTorch implementation
    output_pt = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_pt[15:] = 999.0  # Fill some area that should remain unchanged

    attn_partial_copy_pytorch(input_tensor, output_pt, device_dc)

    # Test with Triton implementation
    output_tr = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_tr[15:] = 999.0  # Fill some area that should remain unchanged

    attn_partial_copy_triton(input_tensor, output_tr, device_dc, check_bounds=True)

    # Verify results
    copy_size = input_batch - device_dc_val

    # Check copied region
    assert torch.equal(
        output_pt[:copy_size], output_tr[:copy_size]
    ), "Copied region mismatch between PyTorch and Triton"
    assert torch.equal(
        output_pt[:copy_size], input_tensor[device_dc_val : device_dc_val + copy_size]
    ), "Copied region incorrect in output"

    # Check unchanged region
    assert torch.equal(output_pt[15:], output_tr[15:]), "Unchanged region mismatch"

    # Check full output
    assert torch.equal(output_pt, output_tr), "Full output mismatch between PyTorch and Triton"


def test_edge_case_device_dc_zero(device, test_params):
    """Test edge case: device_dc=0 (copy entire input)."""
    input_batch = test_params['input_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    input_tensor = torch.randn(input_batch, feature_dim, dtype=torch.float32, device=device)
    output_tensor = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)

    attn_partial_copy_triton(input_tensor, output_tensor, torch.tensor([0, 0], device=device))

    copy_size = min(input_batch, output_batch)
    assert torch.equal(
        output_tensor[:copy_size], input_tensor[:copy_size]
    ), "device_dc=0 test failed"


def test_edge_case_device_dc_full(device, test_params):
    """Test edge case: device_dc=input_batch (copy nothing)."""
    input_batch = test_params['input_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    input_tensor = torch.randn(input_batch, feature_dim, dtype=torch.float32, device=device)
    output_tensor = torch.ones(output_batch, feature_dim, dtype=torch.float32, device=device)
    original_output = output_tensor.clone()

    attn_partial_copy_triton(
        input_tensor, output_tensor, torch.tensor([input_batch, 0], device=device)
    )

    assert torch.equal(
        output_tensor, original_output
    ), "device_dc=input_batch test failed - output should remain unchanged"


def test_exact_size_match(device):
    """Test with exact size match between copy size and output size."""
    feature_dim = 256

    small_input = torch.randn(8, feature_dim, dtype=torch.float32, device=device)
    small_output = torch.zeros(5, feature_dim, dtype=torch.float32, device=device)

    attn_partial_copy_triton(small_input, small_output, torch.tensor([3, 0], device=device))

    # Copy size is 8 - 3 = 5, which exactly matches output size
    assert torch.equal(small_output, small_input[3:8]), "Exact size match test failed"


def test_1d_tensors(device):
    """Test with 1D tensors."""
    input_1d = torch.randn(10, dtype=torch.float32, device=device)
    output_1d = torch.zeros(8, dtype=torch.float32, device=device)

    attn_partial_copy_triton(input_1d, output_1d, torch.tensor([2, 0], device=device))

    assert torch.equal(output_1d, input_1d[2:10]), "1D tensor test failed"


def test_3d_tensors(device):
    """Test with 3D tensors."""
    input_3d = torch.randn(6, 4, 8, dtype=torch.float32, device=device)
    output_3d = torch.zeros(10, 4, 8, dtype=torch.float32, device=device)

    attn_partial_copy_triton(input_3d, output_3d, torch.tensor([1, 0], device=device))

    copy_size = min(6 - 1, 10)  # Should be 5
    assert torch.equal(output_3d[:copy_size], input_3d[1:6]), "3D tensor test failed"


def test_bounds_checking(device, test_params):
    """Test bounds checking functionality."""
    input_batch = test_params['input_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    input_tensor = torch.randn(input_batch, feature_dim, dtype=torch.float32, device=device)
    output_tensor = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)

    # This should fail - device_dc too large
    with pytest.raises(AssertionError):
        attn_partial_copy_triton(
            input_tensor,
            output_tensor,
            torch.tensor([input_batch + 1, 0], device=device),
            check_bounds=True,
        )


def test_pytorch_triton_consistency(device):
    """Test that PyTorch and Triton implementations produce identical results."""
    input_tensor = torch.randn(100, 128, dtype=torch.float32, device=device)
    device_dc = torch.tensor([25, 0], device=device)

    output_pt = torch.zeros(80, 128, dtype=torch.float32, device=device)
    output_tr = torch.zeros(80, 128, dtype=torch.float32, device=device)

    attn_partial_copy_pytorch(input_tensor, output_pt, device_dc)
    attn_partial_copy_triton(input_tensor, output_tr, device_dc)

    assert torch.equal(
        output_pt, output_tr
    ), "PyTorch and Triton implementations produce different results"


if __name__ == '__main__':
    # Run basic tests when executed directly
    print("Testing attn_partial_copy functionality...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available, skipping Triton tests")
        exit(0)

    # Test parameters
    input_batch = 16
    output_batch = 20
    feature_dim = 256

    print(f"\n=== Basic Functionality Test ===")

    # Create test tensors
    input_tensor = torch.randn(input_batch, feature_dim, dtype=torch.float32, device=device)

    # Test with device_dc = 5 (copy from index 5 to end)
    device_dc = torch.tensor([5, 0], device=device)
    device_dc_val = device_dc[0].item()

    # Test with PyTorch implementation
    output_pt = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_pt[15:] = 999.0  # Fill some area that should remain unchanged

    attn_partial_copy_pytorch(input_tensor, output_pt, device_dc)

    # Test with Triton implementation
    output_tr = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_tr[15:] = 999.0  # Fill some area that should remain unchanged

    attn_partial_copy_triton(input_tensor, output_tr, device_dc, check_bounds=True)

    # Verify results
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Output tensor shape: {output_pt.shape}")
    print(f"Device DC: {device_dc_val}")
    print(f"Copy size: {input_batch - device_dc_val}")

    # Check if PyTorch and Triton implementations match
    copy_size = input_batch - device_dc_val
    copied_region_match = torch.equal(output_pt[:copy_size], output_tr[:copy_size])
    unchanged_region_match = torch.equal(output_pt[15:], output_tr[15:])
    full_output_match = torch.equal(output_pt, output_tr)

    print(f"Copied region match: {copied_region_match}")
    print(f"Unchanged region match: {unchanged_region_match}")
    print(f"Full output match: {full_output_match}")

    # Verify against expected behavior
    expected_copied_match = torch.equal(
        output_pt[:copy_size], input_tensor[device_dc_val : device_dc_val + copy_size]
    )
    print(f"Copied region correct: {expected_copied_match}")

    # Test edge cases
    print(f"\n=== Edge Case Tests ===")

    # Test with device_dc = 0 (copy entire input)
    print(f"\n--- Edge Case: device_dc=0 ---")
    output_edge1 = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    attn_partial_copy_triton(input_tensor, output_edge1, torch.tensor([0, 0], device=device))
    copy_size_edge1 = min(input_batch, output_batch)
    expected_edge1 = torch.equal(output_edge1[:copy_size_edge1], input_tensor[:copy_size_edge1])
    print(f"device_dc=0 test: {expected_edge1}")

    # Test with device_dc = input_batch (copy nothing)
    print(f"\n--- Edge Case: device_dc=input_batch ---")
    output_edge2 = torch.ones(output_batch, feature_dim, dtype=torch.float32, device=device)
    original_output = output_edge2.clone()
    attn_partial_copy_triton(
        input_tensor, output_edge2, torch.tensor([input_batch, 0], device=device)
    )
    expected_edge2 = torch.equal(output_edge2, original_output)  # Should remain unchanged
    print(f"device_dc=input_batch test: {expected_edge2}")

    # Test with exact size match
    print(f"\n--- Edge Case: Exact size match ---")
    small_input = torch.randn(8, feature_dim, dtype=torch.float32, device=device)
    small_output = torch.zeros(5, feature_dim, dtype=torch.float32, device=device)
    attn_partial_copy_triton(small_input, small_output, torch.tensor([3, 0], device=device))
    copy_size_small = min(8 - 3, 5)  # Should be 5
    expected_small = torch.equal(small_output, small_input[3:8])
    print(f"Exact size match test: {expected_small}")

    # Test with 1D tensors
    print(f"\n--- Edge Case: 1D tensors ---")
    input_1d = torch.randn(10, dtype=torch.float32, device=device)
    output_1d = torch.zeros(8, dtype=torch.float32, device=device)
    attn_partial_copy_triton(input_1d, output_1d, torch.tensor([2, 0], device=device))
    expected_1d = torch.equal(output_1d, input_1d[2:10])
    print(f"1D tensor test: {expected_1d}")

    # Test with 3D tensors
    print(f"\n--- Edge Case: 3D tensors ---")
    input_3d = torch.randn(6, 4, 8, dtype=torch.float32, device=device)
    output_3d = torch.zeros(10, 4, 8, dtype=torch.float32, device=device)
    attn_partial_copy_triton(input_3d, output_3d, torch.tensor([1, 0], device=device))
    copy_size_3d = min(6 - 1, 10)  # Should be 5
    expected_3d = torch.equal(output_3d[:copy_size_3d], input_3d[1:6])
    print(f"3D tensor test: {expected_3d}")

    # Test bounds checking
    print(f"\n--- Edge Case: Bounds checking ---")
    try:
        # This should fail - device_dc too large
        attn_partial_copy_triton(
            input_tensor,
            output_pt,
            torch.tensor([input_batch + 1, 0], device=device),
            check_bounds=True,
        )
        print("Bounds check failed - should have raised assertion")
    except AssertionError as e:
        print(f"Bounds check passed - correctly caught: {str(e)[:50]}...")

    # Performance comparison test
    print(f"\n=== Performance Comparison ===")
    large_input = torch.randn(1000, 512, dtype=torch.float32, device=device)
    large_output_pt = torch.zeros(800, 512, dtype=torch.float32, device=device)
    large_output_tr = torch.zeros(800, 512, dtype=torch.float32, device=device)
    large_device_dc = torch.tensor([200, 0], device=device)

    # Warm up
    for _ in range(10):
        attn_partial_copy_pytorch(large_input, large_output_pt, large_device_dc)
        attn_partial_copy_triton(large_input, large_output_tr, large_device_dc)

    # Time PyTorch implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        attn_partial_copy_pytorch(large_input, large_output_pt, large_device_dc)
    torch.cuda.synchronize()
    pt_time = time.time() - start_time

    # Time Triton implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        attn_partial_copy_triton(large_input, large_output_tr, large_device_dc)
    torch.cuda.synchronize()
    tr_time = time.time() - start_time

    print(f"PyTorch time: {pt_time:.4f}s")
    print(f"Triton time: {tr_time:.4f}s")
    print(f"Speedup: {pt_time / tr_time:.2f}x")

    # Verify they still match
    final_match = torch.equal(large_output_pt, large_output_tr)
    print(f"Large tensor results match: {final_match}")

    print("\n✅ All tests completed!")

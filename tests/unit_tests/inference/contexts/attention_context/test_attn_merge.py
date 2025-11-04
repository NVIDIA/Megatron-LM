import torch
import pytest
from megatron.core.inference.contexts.attention_context.triton.attn_merge import attn_merge_triton


def attn_merge_pytorch(
    decode_tensor: torch.Tensor,
    prefill_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    device_dc: torch.Tensor,
    pf_useful_from_beginning: bool,
) -> None:
    """
    PyTorch reference implementation for attn_merge_triton.

    Args:
        decode_tensor: Tensor containing decode data, shape [decode_batch, ...]
        prefill_tensor: Tensor containing prefill data, shape [prefill_batch, ...]
        output_tensor: Output tensor to write to, shape [output_batch, ...]
        device_dc: Tensor with decode count in first element, shape [2] (e.g., [dc_count, pf_count])
        pf_useful_from_beginning: If True, copy prefill from beginning; else from device_dc
    """

    # Extract device_dc value from tensor
    device_dc_val = device_dc[0].item()

    # Copy decode_tensor[:device_dc] to output[:device_dc]
    if device_dc_val > 0:
        output_tensor[:device_dc_val].copy_(decode_tensor[:device_dc_val])

    # Copy prefill_tensor to output[device_dc:device_dc+prefill_batch_size]
    prefill_batch_size = prefill_tensor.shape[0]
    output_batch_size = output_tensor.shape[0]

    if prefill_batch_size > 0:
        copy_size = min(prefill_batch_size, output_batch_size - device_dc_val)
        if copy_size > 0:
            if pf_useful_from_beginning:
                # Mode 1: Copy prefill_tensor from beginning
                # output[device_dc:device_dc+copy_size] = prefill_tensor[0:copy_size]
                output_tensor[device_dc_val : device_dc_val + copy_size].copy_(
                    prefill_tensor[:copy_size]
                )
            else:
                # Mode 2: Copy prefill_tensor from device_dc to end
                # output[device_dc:device_dc+copy_size] = prefill_tensor[device_dc:device_dc+copy_size]
                if device_dc_val + copy_size <= prefill_batch_size:
                    output_tensor[device_dc_val : device_dc_val + copy_size].copy_(
                        prefill_tensor[device_dc_val : device_dc_val + copy_size]
                    )


@pytest.fixture
def device():
    """Get CUDA device if available, otherwise skip tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.fixture
def test_params():
    """Common test parameters."""
    return {'decode_batch': 8, 'prefill_batch': 12, 'output_batch': 32, 'feature_dim': 256}


def test_mode1_pf_useful_from_beginning(device, test_params):
    """Test Mode 1: pf_useful_from_beginning=True."""
    decode_batch = test_params['decode_batch']
    prefill_batch = test_params['prefill_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    device_dc = torch.tensor([5, 0], device=device)

    # Create test tensors
    decode_tensor = torch.randn(decode_batch, feature_dim, dtype=torch.float32, device=device)
    prefill_tensor = torch.randn(prefill_batch, feature_dim, dtype=torch.float32, device=device)

    # Test with PyTorch implementation
    output_pt = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_pt[15:25] = 999.0  # Fill some area that should remain unchanged

    attn_merge_pytorch(
        decode_tensor, prefill_tensor, output_pt, device_dc, pf_useful_from_beginning=True
    )

    # Test with Triton implementation
    output_tr = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_tr[15:25] = 999.0  # Fill some area that should remain unchanged

    attn_merge_triton(
        decode_tensor, prefill_tensor, output_tr, device_dc, pf_useful_from_beginning=True
    )

    # Verify results
    device_dc_val = device_dc[0].item()

    # Check decode region
    assert torch.equal(
        output_pt[:device_dc_val], output_tr[:device_dc_val]
    ), "Decode region mismatch between PyTorch and Triton"
    assert torch.equal(
        output_pt[:device_dc_val], decode_tensor[:device_dc_val]
    ), "Decode region incorrect in output"

    # Check prefill region
    assert torch.equal(
        output_pt[device_dc_val : device_dc_val + prefill_batch],
        output_tr[device_dc_val : device_dc_val + prefill_batch],
    ), "Prefill region mismatch between PyTorch and Triton"
    assert torch.equal(
        output_pt[device_dc_val : device_dc_val + prefill_batch], prefill_tensor
    ), "Prefill region incorrect in output"

    # Check unchanged region
    assert torch.equal(output_pt[15:25], output_tr[15:25]), "Unchanged region mismatch"

    # Check full output
    assert torch.equal(output_pt, output_tr), "Full output mismatch between PyTorch and Triton"


def test_mode2_pf_useful_from_device_dc(device, test_params):
    """Test Mode 2: pf_useful_from_beginning=False."""
    decode_batch = test_params['decode_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    device_dc = torch.tensor([5, 0], device=device)

    # Create test tensors
    decode_tensor = torch.randn(decode_batch, feature_dim, dtype=torch.float32, device=device)
    prefill_tensor_large = torch.randn(
        output_batch, feature_dim, dtype=torch.float32, device=device
    )

    # Test with PyTorch implementation
    output_pt = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_pt[20:30] = 888.0  # Fill some area that should remain unchanged

    attn_merge_pytorch(
        decode_tensor, prefill_tensor_large, output_pt, device_dc, pf_useful_from_beginning=False
    )

    # Test with Triton implementation
    output_tr = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_tr[20:30] = 888.0  # Fill some area that should remain unchanged

    attn_merge_triton(
        decode_tensor, prefill_tensor_large, output_tr, device_dc, pf_useful_from_beginning=False
    )

    # Verify results
    device_dc_val = device_dc[0].item()

    # Check decode region
    assert torch.equal(
        output_pt[:device_dc_val], output_tr[:device_dc_val]
    ), "Decode region mismatch between PyTorch and Triton"
    assert torch.equal(
        output_pt[:device_dc_val], decode_tensor[:device_dc_val]
    ), "Decode region incorrect in output"

    # Check prefill region
    copy_size = min(prefill_tensor_large.shape[0] - device_dc_val, output_batch - device_dc_val)
    if copy_size > 0:
        assert torch.equal(
            output_pt[device_dc_val : device_dc_val + copy_size],
            output_tr[device_dc_val : device_dc_val + copy_size],
        ), "Prefill region mismatch between PyTorch and Triton"
        assert torch.equal(
            output_pt[device_dc_val : device_dc_val + copy_size],
            prefill_tensor_large[device_dc_val : device_dc_val + copy_size],
        ), "Prefill region incorrect in output"

    # Check unchanged region
    assert torch.equal(output_pt[20:30], output_tr[20:30]), "Unchanged region mismatch"

    # Check full output
    assert torch.equal(output_pt, output_tr), "Full output mismatch between PyTorch and Triton"


def test_edge_case_device_dc_zero_mode1(device, test_params):
    """Test edge case: device_dc=0 with Mode 1."""
    decode_batch = test_params['decode_batch']
    prefill_batch = test_params['prefill_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    decode_tensor = torch.randn(decode_batch, feature_dim, dtype=torch.float32, device=device)
    prefill_tensor = torch.randn(prefill_batch, feature_dim, dtype=torch.float32, device=device)
    output_tensor = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)

    attn_merge_triton(
        decode_tensor,
        prefill_tensor,
        output_tensor,
        torch.tensor([0, 0], device=device),
        pf_useful_from_beginning=True,
    )

    assert torch.equal(
        output_tensor[:prefill_batch], prefill_tensor
    ), "device_dc=0 Mode 1 test failed"


def test_edge_case_device_dc_zero_mode2(device, test_params):
    """Test edge case: device_dc=0 with Mode 2."""
    decode_batch = test_params['decode_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    decode_tensor = torch.randn(decode_batch, feature_dim, dtype=torch.float32, device=device)
    prefill_tensor_large = torch.randn(
        output_batch, feature_dim, dtype=torch.float32, device=device
    )
    output_tensor = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)

    attn_merge_triton(
        decode_tensor,
        prefill_tensor_large,
        output_tensor,
        torch.tensor([0, 0], device=device),
        pf_useful_from_beginning=False,
    )

    copy_size = min(prefill_tensor_large.shape[0], output_batch)
    assert torch.equal(
        output_tensor[:copy_size], prefill_tensor_large[:copy_size]
    ), "device_dc=0 Mode 2 test failed"


def test_edge_case_device_dc_full_mode1(device, test_params):
    """Test edge case: device_dc=decode_batch with Mode 1."""
    decode_batch = test_params['decode_batch']
    prefill_batch = test_params['prefill_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    decode_tensor = torch.randn(decode_batch, feature_dim, dtype=torch.float32, device=device)
    prefill_tensor = torch.randn(prefill_batch, feature_dim, dtype=torch.float32, device=device)
    output_tensor = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)

    attn_merge_triton(
        decode_tensor,
        prefill_tensor,
        output_tensor,
        torch.tensor([decode_batch, 0], device=device),
        pf_useful_from_beginning=True,
    )

    assert torch.equal(output_tensor[:decode_batch], decode_tensor), "Decode region incorrect"
    assert torch.equal(
        output_tensor[decode_batch : decode_batch + prefill_batch], prefill_tensor
    ), "Prefill region incorrect"


def test_edge_case_device_dc_full_mode2(device, test_params):
    """Test edge case: device_dc=decode_batch with Mode 2."""
    decode_batch = test_params['decode_batch']
    output_batch = test_params['output_batch']
    feature_dim = test_params['feature_dim']

    decode_tensor = torch.randn(decode_batch, feature_dim, dtype=torch.float32, device=device)
    prefill_tensor_large = torch.randn(
        output_batch, feature_dim, dtype=torch.float32, device=device
    )
    output_tensor = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)

    attn_merge_triton(
        decode_tensor,
        prefill_tensor_large,
        output_tensor,
        torch.tensor([decode_batch, 0], device=device),
        pf_useful_from_beginning=False,
    )

    assert torch.equal(output_tensor[:decode_batch], decode_tensor), "Decode region incorrect"

    copy_size = min(prefill_tensor_large.shape[0] - decode_batch, output_batch - decode_batch)
    if copy_size > 0:
        assert torch.equal(
            output_tensor[decode_batch : decode_batch + copy_size],
            prefill_tensor_large[decode_batch : decode_batch + copy_size],
        ), "Prefill region incorrect"


def test_small_tensors_mode1(device):
    """Test with small tensors in Mode 1."""
    feature_dim = 256

    small_decode = torch.randn(3, feature_dim, dtype=torch.float32, device=device)
    small_prefill = torch.randn(5, feature_dim, dtype=torch.float32, device=device)
    small_output = torch.zeros(10, feature_dim, dtype=torch.float32, device=device)

    attn_merge_triton(
        small_decode,
        small_prefill,
        small_output,
        torch.tensor([2, 0], device=device),
        pf_useful_from_beginning=True,
    )

    assert torch.equal(
        small_output[:2], small_decode[:2]
    ), "Small tensor Mode 1 decode region incorrect"
    assert torch.equal(
        small_output[2:7], small_prefill
    ), "Small tensor Mode 1 prefill region incorrect"


def test_small_tensors_mode2(device):
    """Test with small tensors in Mode 2."""
    feature_dim = 256

    small_decode = torch.randn(3, feature_dim, dtype=torch.float32, device=device)
    small_prefill = torch.randn(5, feature_dim, dtype=torch.float32, device=device)
    small_output = torch.zeros(10, feature_dim, dtype=torch.float32, device=device)

    attn_merge_triton(
        small_decode,
        small_prefill,
        small_output,
        torch.tensor([2, 0], device=device),
        pf_useful_from_beginning=False,
    )

    assert torch.equal(
        small_output[:2], small_decode[:2]
    ), "Small tensor Mode 2 decode region incorrect"
    assert torch.equal(
        small_output[2:5], small_prefill[2:5]
    ), "Small tensor Mode 2 prefill region incorrect"


if __name__ == '__main__':
    # Run basic tests when executed directly
    print("Testing attn_merge functionality...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available, skipping Triton tests")
        exit(0)

    # Test parameters
    decode_batch = 8
    prefill_batch = 12
    output_batch = 32
    feature_dim = 256
    device_dc = torch.tensor([5, 0], device=device)

    # Create test tensors
    decode_tensor = torch.randn(decode_batch, feature_dim, dtype=torch.float32, device=device)
    prefill_tensor = torch.randn(prefill_batch, feature_dim, dtype=torch.float32, device=device)

    # Test Mode 1
    print(f"\n=== Testing Mode 1: pf_useful_from_beginning=True ===")

    output_pt1 = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_pt1[15:25] = 999.0

    attn_merge_pytorch(
        decode_tensor, prefill_tensor, output_pt1, device_dc, pf_useful_from_beginning=True
    )

    output_tr1 = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_tr1[15:25] = 999.0

    attn_merge_triton(
        decode_tensor, prefill_tensor, output_tr1, device_dc, pf_useful_from_beginning=True
    )

    # Test Mode 2
    print(f"\n=== Testing Mode 2: pf_useful_from_beginning=False ===")

    prefill_tensor_large = torch.randn(
        output_batch, feature_dim, dtype=torch.float32, device=device
    )

    output_pt2 = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_pt2[20:30] = 888.0

    attn_merge_pytorch(
        decode_tensor, prefill_tensor_large, output_pt2, device_dc, pf_useful_from_beginning=False
    )

    output_tr2 = torch.zeros(output_batch, feature_dim, dtype=torch.float32, device=device)
    output_tr2[20:30] = 888.0

    attn_merge_triton(
        decode_tensor, prefill_tensor_large, output_tr2, device_dc, pf_useful_from_beginning=False
    )

    # Verify results
    print(f"\n--- Mode 1 Results ---")
    device_dc_val = device_dc[0].item()
    print(f"Mode 1 - Full output match: {torch.equal(output_pt1, output_tr1)}")
    print(
        f"Mode 1 - Decode region correct: {torch.equal(output_pt1[:device_dc_val], decode_tensor[:device_dc_val])}"
    )
    print(
        f"Mode 1 - Prefill region correct: {torch.equal(output_pt1[device_dc_val:device_dc_val + prefill_batch], prefill_tensor)}"
    )

    print(f"\n--- Mode 2 Results ---")
    print(f"Mode 2 - Full output match: {torch.equal(output_pt2, output_tr2)}")
    print(
        f"Mode 2 - Decode region correct: {torch.equal(output_pt2[:device_dc_val], decode_tensor[:device_dc_val])}"
    )

    copy_size2 = min(prefill_tensor_large.shape[0] - device_dc_val, output_batch - device_dc_val)
    if copy_size2 > 0:
        print(
            f"Mode 2 - Prefill region correct: {torch.equal(output_pt2[device_dc_val:device_dc_val + copy_size2], prefill_tensor_large[device_dc_val:device_dc_val + copy_size2])}"
        )

    print("\n✅ All tests completed!")

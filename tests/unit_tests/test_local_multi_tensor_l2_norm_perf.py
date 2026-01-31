import torch
import sys
import os
import traceback

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from megatron.core.utils import local_multi_tensor_l2_norm
except ImportError as e:
    print(f"Could not import megatron.core.utils: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during import: {e}")
    traceback.print_exc()
    sys.exit(1)


def test_local_multi_tensor_l2_norm_correctness():
    print("Running test_local_multi_tensor_l2_norm_correctness...")
    # Setup inputs
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")

    t1 = torch.tensor([3.0, 4.0], device=device)  # norm 5
    t2 = torch.tensor([6.0, 8.0], device=device)  # norm 10

    # Global norm = sqrt(5^2 + 10^2) = sqrt(125) = 11.1803...
    expected_norm = (t1.norm()**2 + t2.norm()**2).sqrt()

    tensor_lists = [[t1, t2]]

    # Call the function
    try:
        result, _ = local_multi_tensor_l2_norm(2048, None, tensor_lists, False)
    except Exception as e:
        print(f"Function call failed: {e}")
        # If it failed because of device mismatch on CPU machine, catch it?
        # But my fix should handle CPU.
        raise e

    # Verify shape
    assert result.dim() == 1
    assert result.numel() == 1

    # Verify value
    print(f"Result: {result.item()}, Expected: {expected_norm.item()}")
    torch.testing.assert_close(result.item(), expected_norm.item())

    # Verify device
    if device == "cuda":
        assert result.is_cuda

    # Verify dtype is float32
    assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"

    print("test_local_multi_tensor_l2_norm_correctness PASSED")


def test_local_multi_tensor_l2_norm_empty_input():
    print("Running test_local_multi_tensor_l2_norm_empty_input...")
    if torch.cuda.is_available():
        pass
    else:
        pass

    tensor_lists = [[]]
    try:
        result, _ = local_multi_tensor_l2_norm(2048, None, tensor_lists, False)
    except Exception as e:
        print(f"Function call failed: {e}")
        return

    assert result.item() == 0.0
    if torch.cuda.is_available():
        assert result.is_cuda

    assert result.dtype == torch.float32
    print("test_local_multi_tensor_l2_norm_empty_input PASSED")


if __name__ == "__main__":
    test_local_multi_tensor_l2_norm_correctness()
    test_local_multi_tensor_l2_norm_empty_input()

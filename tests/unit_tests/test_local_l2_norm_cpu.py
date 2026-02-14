import pytest
import torch
import math
from megatron.core.utils import local_multi_tensor_l2_norm

def test_local_multi_tensor_l2_norm_cpu():
    # Test case 1: Single tensor
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor_lists = [[tensor1]]

    # Expected L2 norm: sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
    expected_norm = math.sqrt(14.0)

    computed_norm_tensor, _ = local_multi_tensor_l2_norm(None, None, tensor_lists, False)
    computed_norm = computed_norm_tensor.item()

    assert math.isclose(computed_norm, expected_norm, rel_tol=1e-5), \
        f"Expected {expected_norm}, got {computed_norm}"

    assert computed_norm_tensor.device.type == "cpu", \
        f"Expected device cpu, got {computed_norm_tensor.device}"

    # Test case 2: Multiple tensors
    tensor2 = torch.tensor([4.0, 5.0])
    tensor_lists = [[tensor1, tensor2]]

    # Expected: sqrt(14 + 16 + 25) = sqrt(55)
    expected_norm = math.sqrt(55.0)

    computed_norm_tensor, _ = local_multi_tensor_l2_norm(None, None, tensor_lists, False)
    computed_norm = computed_norm_tensor.item()

    assert math.isclose(computed_norm, expected_norm, rel_tol=1e-5), \
        f"Expected {expected_norm}, got {computed_norm}"

    # Test case 3: Empty lists
    tensor_lists = []
    # If list is empty, behavior should be robust (return 0.0)
    # The current implementation might crash or return something weird.
    # We expect 0.0
    computed_norm_tensor, _ = local_multi_tensor_l2_norm(None, None, tensor_lists, False)
    assert computed_norm_tensor.item() == 0.0

if __name__ == "__main__":
    test_local_multi_tensor_l2_norm_cpu()

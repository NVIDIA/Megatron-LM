import torch
import math
import pytest
from megatron.core.utils import local_multi_tensor_l2_norm

def test_l2_norm_simple():
    if not torch.cuda.is_available():
        # Fallback to CPU tensors if CUDA not available (utils usually handles CPU too?)
        # local_multi_tensor_l2_norm seems device agnostic in logic, but defaults to 'cuda' for zero tensor?
        # Let's check the code: "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
        device = 'cpu'
    else:
        device = 'cuda'

    t1 = torch.tensor([1.0, 2.0, 3.0], device=device)
    t2 = torch.tensor([4.0, 5.0, 6.0], device=device)
    # Expected: sqrt(14 + 77) = sqrt(91) ~= 9.539392

    res, _ = local_multi_tensor_l2_norm(None, None, [[t1, t2]], False)
    expected = math.sqrt(91)

    assert abs(res.item() - expected) < 1e-4, f"Expected {expected}, got {res.item()}"

def test_l2_norm_empty():
    res, _ = local_multi_tensor_l2_norm(None, None, [[]], False)
    assert res.item() == 0.0

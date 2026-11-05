# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Ported from https://github.com/NVIDIA-NeMo/nv-mistralai-megatron
# Revision: daa90bd2d26fb44f34b769961c1a1f94a53d6c35
# License: BSD-3-Clause (see the source repository LICENSE).

"""Gate explicit private-logits buffer reuse; default callers retain their inputs."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.ops.cross_entropy import triton_cross_entropy
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def tp1():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_private_buffer_gradients_match_out_of_place_bitwise(dtype):
    torch.manual_seed(2191)
    source = torch.randn(17, 3, 4097, device='cuda', dtype=dtype)
    ordinary = source.clone().requires_grad_()
    owned = source.clone().requires_grad_()
    labels = torch.randint(4097, (3, 17), device='cuda').transpose(0, 1)
    scale = torch.randn(17, 3, device='cuda')
    scale[::3] = 0
    group = parallel_state.get_tensor_model_parallel_group()
    reference = triton_cross_entropy(ordinary, labels, group)
    actual = triton_cross_entropy(owned, labels, group, inplace_backward=True)
    assert torch.equal(reference, actual)
    assert torch.equal(owned, source), 'Forward changed logits'
    expected_grad = torch.autograd.grad((reference * scale).sum(), ordinary)[0]
    actual_grad = torch.autograd.grad((actual * scale).sum(), owned)[0]
    assert actual_grad.data_ptr() == owned.data_ptr()
    assert torch.equal(actual_grad, expected_grad)
    assert torch.equal(ordinary, source), 'Default mode changed logits'


def test_consumed_buffer_cannot_silently_repeat_backward():
    logits = torch.randn(7, 127, device='cuda', requires_grad=True)
    labels = torch.randint(127, (7,), device='cuda')
    loss = triton_cross_entropy(
        logits, labels, parallel_state.get_tensor_model_parallel_group(), inplace_backward=True
    ).sum()
    torch.autograd.grad(loss, logits, retain_graph=True)
    with pytest.raises(RuntimeError, match='already consumed'):
        torch.autograd.grad(loss, logits)

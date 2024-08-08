import copy

import pytest
import torch

from megatron.core.utils import (
    local_multi_tensor_applier,
    local_multi_tensor_l2_norm,
    local_multi_tensor_scale,
)


def test_local_multi_tensor_l2_norm_and_scale():
    amp_C = pytest.importorskip("amp_C")
    multi_tensor_apply = pytest.importorskip("apex.multi_tensor_apply")

    torch.manual_seed(42)

    tensor_list = [torch.rand(5, 5).cuda() for _ in range(10)]
    tensor_list_copy = copy.deepcopy(tensor_list)

    norm_apex, _ = multi_tensor_apply.multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        torch.tensor([0], dtype=torch.int, device='cuda'),
        [tensor_list],
        False,
    )
    norm_local, _ = multi_tensor_apply.multi_tensor_applier(
        local_multi_tensor_l2_norm,
        torch.tensor([0], dtype=torch.int, device='cuda'),
        [tensor_list_copy],
        False,
    )
    torch.testing.assert_close(norm_apex, norm_local)

    clip_coeff = 0.05
    multi_tensor_apply.multi_tensor_applier(
        amp_C.multi_tensor_scale,
        torch.tensor([0], dtype=torch.int, device='cuda'),
        [tensor_list, tensor_list],
        clip_coeff,
    )
    multi_tensor_apply.multi_tensor_applier(
        local_multi_tensor_scale,
        torch.tensor([0], dtype=torch.int, device='cuda'),
        [tensor_list_copy, tensor_list_copy],
        clip_coeff,
    )
    torch.testing.assert_close(tensor_list, tensor_list_copy)


def test_local_multi_tensor_apply():
    amp_C = pytest.importorskip("amp_C")
    multi_tensor_apply = pytest.importorskip("apex.multi_tensor_apply")

    tensor_list = [torch.rand(5, 5).cuda() for _ in range(10)]

    norm_apex, _ = multi_tensor_apply.multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        torch.tensor([0], dtype=torch.int, device='cuda'),
        [tensor_list],
        False,
    )
    norm_local, _ = local_multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        torch.tensor([0], dtype=torch.int, device='cuda'),
        [tensor_list],
        False,
    )
    torch.testing.assert_close(norm_apex, norm_local)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.training.arguments import _validate_nccl_ub_rl_offload


@pytest.mark.parametrize(
    ("perform_rl_step", "nccl_ub", "offload", "cuda_graphs", "raises"),
    [
        (True, True, True, False, True),
        (False, True, True, False, False),
        (True, True, True, True, False),
        (True, False, True, False, False),
        (True, True, False, False, False),
    ],
)
def test_validate_nccl_ub_rl_offload(perform_rl_step, nccl_ub, offload, cuda_graphs, raises):
    args = SimpleNamespace(
        perform_rl_step=perform_rl_step,
        nccl_ub=nccl_ub,
        rl_offload_optimizer_during_inference=offload,
        rl_training_cuda_graphs=cuda_graphs,
    )

    if raises:
        with pytest.raises(ValueError, match="unsupported with --use-nccl-ub"):
            _validate_nccl_ub_rl_offload(args)
    else:
        _validate_nccl_ub_rl_offload(args)


@pytest.mark.parametrize("collection_name", ["buffers", "expert_parallel_buffers"])
def test_registered_grad_buffer_is_rejected_before_offload(collection_name):
    calls = []
    registered_buffer = SimpleNamespace(
        nccl_mem_pool=object(), offload_to_cpu=lambda **kwargs: calls.append(kwargs)
    )
    ddp = SimpleNamespace(buffers=[], expert_parallel_buffers=[])
    getattr(ddp, collection_name).append(registered_buffer)

    with pytest.raises(RuntimeError, match="owns an NCCL memory pool"):
        DistributedDataParallel.offload_grad_buffers(ddp, synchronize=False, empty_cache=False)

    assert calls == []


def test_unregistered_grad_buffers_are_offloaded():
    calls = []
    dense_buffer = SimpleNamespace(
        nccl_ub=True,
        nccl_mem_pool=None,
        offload_to_cpu=lambda **kwargs: calls.append(("dense", kwargs)),
    )
    expert_buffer = SimpleNamespace(
        offload_to_cpu=lambda **kwargs: calls.append(("expert", kwargs))
    )
    ddp = SimpleNamespace(buffers=[dense_buffer], expert_parallel_buffers=[expert_buffer])

    DistributedDataParallel.offload_grad_buffers(ddp, synchronize=False, empty_cache=False)

    expected_kwargs = {"move_params": False, "move_grads": True}
    assert calls == [("dense", expected_kwargs), ("expert", expected_kwargs)]

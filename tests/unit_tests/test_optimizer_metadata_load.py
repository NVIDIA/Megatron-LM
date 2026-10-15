# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from copy import deepcopy

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.test_utils import _init_distributed


def _build_optimizer(
    precision_aware: bool, moment_dtype: torch.dtype, remainders: bool
) -> tuple[DistributedDataParallel, ChainedOptimizer, DistributedOptimizer]:
    module = torch.nn.Sequential(
        *[
            torch.nn.Linear(512, 512, bias=False, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
    )
    for param in module.parameters():
        param.data.fill_(0.125)
    model = DistributedDataParallel(
        TransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            params_dtype=torch.bfloat16,
            bf16=True,
        ),
        DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=False,
        ),
        module,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.001,
            bf16=True,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=precision_aware,
            main_params_dtype=torch.float32,
            exp_avg_dtype=moment_dtype,
            exp_avg_sq_dtype=moment_dtype,
            store_param_remainders=remainders,
            clip_grad=0.0,
        ),
        [model],
    )
    return model, optimizer, optimizer.chained_optimizers[0]


@pytest.mark.parametrize(
    "precision_aware,moment_dtype,remainders",
    [
        (False, torch.float32, False),
        (True, torch.float32, False),
        (True, torch.bfloat16, False),
        (True, torch.float32, True),
        (True, torch.float16, False),
    ],
)
def test_te_metadata_load_preserves_buffers_and_bounds_initialization_memory(
    precision_aware: bool, moment_dtype: torch.dtype, remainders: bool
) -> None:
    if not is_te_min_version("2.18"):
        pytest.skip("Requires TE's state initializer with remainder storage")
    _init_distributed(int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0")))
    Utils.initialize_model_parallel()
    try:
        model, optimizer, distributed_optimizer = _build_optimizer(
            precision_aware, moment_dtype, remainders
        )
        inner = distributed_optimizer.optimizer
        metadata = distributed_optimizer.state_dict()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        distributed_optimizer.load_state_dict(metadata)
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated() - allocated
        peak = torch.cuda.max_memory_allocated() - allocated
        # Native initialization may clone one BF16 parameter to FP32. It must
        # not retain a second complete set of moments and main parameters.
        largest_param = max(p.numel() for group in inner.param_groups for p in group["params"])
        assert peak <= resident + largest_param * 6 + 32768
        for group in inner.param_groups:
            for param in group["params"]:
                state = inner.state[param]
                assert state["exp_avg"].dtype == moment_dtype
                assert state["exp_avg_sq"].dtype == moment_dtype
                if precision_aware:
                    assert state["master_param"].dtype == (
                        torch.int16 if remainders else torch.float32
                    )

        # Supply known restored values before taking a real optimizer step.
        # The cold metadata load above creates destinations; parameter loading
        # would normally fill them before training resumes.
        for group in inner.param_groups:
            for param in group["params"]:
                inner.state[param]["exp_avg"].zero_()
                inner.state[param]["exp_avg_sq"].zero_()
        optimizer.zero_grad()
        model.zero_grad_buffer()
        for param in model.parameters():
            param.main_grad.fill_(0.25)
        model.finish_grad_sync()
        assert optimizer.step()[0]

        metadata = distributed_optimizer.state_dict()
        expected_groups = deepcopy(metadata["optimizer"]["param_groups"])
        values = {
            param: {key: value.detach().cpu().clone() for key, value in state.items()}
            for param, state in inner.state.items()
        }
        pointers = {
            param: {key: value.data_ptr() for key, value in state.items()}
            for param, state in inner.state.items()
        }
        for group in inner.param_groups:
            group["lr"] = 0.9
            group["step"] = 99
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        distributed_optimizer.load_state_dict(metadata)
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() == allocated
        for param, state in inner.state.items():
            for key, value in state.items():
                assert value.data_ptr() == pointers[param][key]
                assert torch.equal(value.cpu(), values[param][key])
        for group, expected in zip(inner.param_groups, expected_groups, strict=True):
            assert group["lr"] == expected["lr"]
            assert group["step"] == expected["step"]
    finally:
        Utils.destroy_model_parallel()


def test_te_metadata_load_honors_registered_load_hooks() -> None:
    if not is_te_min_version("2.18"):
        pytest.skip("Requires TE 2.18+")
    _init_distributed(int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0")))
    Utils.initialize_model_parallel()
    try:
        _, _, distributed_optimizer = _build_optimizer(True, torch.float32, False)
        inner = distributed_optimizer.optimizer
        events = []
        pre_handle = inner.register_load_state_dict_pre_hook(
            lambda optimizer, state_dict: events.append("pre")
        )
        post_handle = inner.register_load_state_dict_post_hook(
            lambda optimizer: events.append("post")
        )
        try:
            distributed_optimizer.load_state_dict(distributed_optimizer.state_dict())
            assert events == ["pre", "post"]
        finally:
            pre_handle.remove()
            post_handle.remove()
    finally:
        Utils.destroy_model_parallel()

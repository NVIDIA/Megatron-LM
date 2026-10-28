# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from transformer_engine.pytorch.optimizers import FusedAdam

from megatron.core import dist_checkpointing
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.test_utils import _init_distributed


def _build(
    precision_aware: bool, fraction: float
) -> tuple[DistributedDataParallel, ChainedOptimizer, DistributedOptimizer]:
    module = torch.nn.Sequential(
        *[
            torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
            for _ in range(16)
        ]
    )
    for param in module.parameters():
        param.data.fill_(0.125)
    model = DistributedDataParallel(
        TransformerConfig(
            num_layers=1,
            hidden_size=256,
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
            clip_grad=0.0,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=precision_aware,
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=fraction,
            pin_cpu_params=False,
            pin_cpu_grads=False,
            overlap_cpu_optimizer_d2h_h2d=False,
        ),
        [model],
    )
    return model, optimizer, optimizer.chained_optimizers[0]


def _step(model: DistributedDataParallel, optimizer: ChainedOptimizer, step: int) -> None:
    optimizer.zero_grad()
    model.zero_grad_buffer()
    for param in model.parameters():
        param.main_grad.fill_((Utils.rank + 1) * 0.125 + step * 0.0625)
    model.finish_grad_sync()
    assert optimizer.step()[0]


def _snapshot(model: DistributedDataParallel, optimizer: DistributedOptimizer) -> dict[str, Any]:
    inner = optimizer.optimizer
    return {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        # FusedAdam uses a group step; DistOpt may add an unused per-tensor step on load.
        "gpu_steps": (
            [g.get("step", 0) for g in inner.gpu_optimizer.param_groups]
            if inner.gpu_optimizer is not None
            else []
        ),
        "optimizer": [
            {
                k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in inner.state[p].items()
                if not (
                    k == "step"
                    and inner.param_to_inner_param[p].is_cuda
                    and isinstance(inner.gpu_optimizer, FusedAdam)
                )
            }
            for g in inner.param_groups
            for p in g["params"]
        ],
    }


def _assert_equal(a: Any, b: Any, path: str = "state") -> None:
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_equal(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


@pytest.mark.parametrize("fraction", [0.0, 0.625, 1.0])
def test_hybrid_optimizer_cold_distributed_checkpoint_resume(
    tmp_path_dist_ckpt: Path, fraction: float
) -> None:
    """Cold resume must preserve subsequent parameters, Adam moments and group steps."""
    _init_distributed(int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0")))
    Utils.initialize_model_parallel()
    try:
        checkpoint = tmp_path_dist_ckpt / f"hybrid_resume_precision_aware_{fraction}"
        if Utils.rank == 0:
            checkpoint.mkdir(exist_ok=True)
        torch.distributed.barrier()
        model, optimizer, part = _build(True, fraction)
        _step(model, optimizer, 0)
        saved_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        dist_checkpointing.save(
            {"optimizer": part.sharded_state_dict(sharding_type="dp_reshardable")}, str(checkpoint)
        )
        _step(model, optimizer, 1)
        _step(model, optimizer, 2)
        expected = _snapshot(model, part)
        del part, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        model, optimizer, part = _build(True, fraction)
        # Match Bridge's sequence: create destination state, load DCP tensors,
        # restore model values, then pass loaded optimizer state to its loader.
        mapping = {
            "optimizer": part.sharded_state_dict(is_loading=True, sharding_type="dp_reshardable")
        }
        loaded = dist_checkpointing.load(mapping, str(checkpoint))
        model.load_state_dict(saved_model)
        part.load_state_dict(loaded["optimizer"])
        _step(model, optimizer, 1)
        _step(model, optimizer, 2)
        _assert_equal(_snapshot(model, part), expected)
    finally:
        Utils.destroy_model_parallel()

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native checkpoint continuation with partial CPU optimizer offload."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import (
    _build_sharded_state_dict_metadata,
    load_checkpoint,
    save_checkpoint,
)
from megatron.training.training import preprocess_common_state_dict
from tests.unit_tests.determinism.kernels.harness import bytes_equal
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils


class HybridOptimizerModel(torch.nn.Module):
    """Small replicated parameters with a native FP32 subset when requested."""

    def __init__(self, mixed: bool):
        super().__init__()
        self.config = TransformerConfig(
            num_layers=1, hidden_size=16, num_attention_heads=1, bf16=True
        )
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.linspace(-0.5, 0.5, 256, device='cuda')
                    .reshape(16, 16)
                    .to(torch.float32 if mixed and index >= 4 else torch.bfloat16)
                )
                for index in range(8)
            ]
        )

    def forward(self, step: int) -> torch.Tensor:
        """Produce known nonzero gradients through autograd and DDP."""
        gradient = (
            torch.arange(256, device='cuda', dtype=torch.float32).remainder(13) - 6
        ).reshape(16, 16) * (0.0078125 * step)
        return sum((parameter.float() * gradient).sum() for parameter in self.weights)

    def state_dict_for_save_checkpoint(self, **kwargs) -> dict:
        """Expose the native checkpoint interface used by production model modules."""
        return self.state_dict(**kwargs)

    def sharded_state_dict(self, prefix: str = '', **kwargs) -> dict:
        """Replicate model tensors across the synthetic TP and DP ranks."""
        return {
            prefix
            + name: ShardedTensor.from_rank_offsets(
                prefix + name,
                value,
                replica_id=(
                    0,
                    parallel_state.get_tensor_model_parallel_rank(),
                    parallel_state.get_data_parallel_rank(),
                ),
            )
            for name, value in self.state_dict(keep_vars=True).items()
        }


@pytest.mark.parametrize('precision_aware,mixed', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(
    'checkpoint_format', ['dp_reshardable', 'fully_reshardable', 'dp_zero_gather_scatter', 'torch']
)
def test_hybrid_checkpoint_continuation(
    tmp_path_dist_ckpt, monkeypatch, precision_aware: bool, mixed: bool, checkpoint_format: str
) -> None:
    """Restore step two and compare all owner states after steps three through five."""
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip('Requires an even world size for DP=2')
    tp = Utils.world_size // 2
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(123)

    def construct():
        model = HybridOptimizerModel(mixed)
        model = DistributedDataParallel(
            model.config,
            DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=True),
            model,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer='adam',
                lr=0.01,
                weight_decay=0.01,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=precision_aware,
                optimizer_cpu_offload=True,
                optimizer_offload_fraction=0.5,
                overlap_cpu_optimizer_d2h_h2d=True,
                clip_grad=0.0,
            ),
            [model],
        )
        inner = optimizer.chained_optimizers[0].optimizer
        assert inner.cpu_optimizers and inner.gpu_optimizer is not None
        return model, optimizer

    def train_step(model, optimizer, step):
        model.zero_grad_buffer()
        optimizer.zero_grad()
        model(step).backward()
        model.finish_grad_sync()
        success, _, _ = optimizer.step()
        assert success
        torch.cuda.synchronize()
        inner = optimizer.chained_optimizers[0].optimizer
        assert all(group['step'] == step for group in inner.gpu_optimizer.param_groups)
        for child in inner.cpu_optimizers:
            assert all(state['step'].item() == step for state in child.state.values())

    def snapshot(model, optimizer):
        inner = optimizer.chained_optimizers[0].optimizer
        return deepcopy({'model': model.module.state_dict(), 'optimizer': inner.state_dict()})

    def assert_equal(left, right):
        if isinstance(left, torch.Tensor):
            assert bytes_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert_equal(a, b)
        else:
            assert left == right

    try:
        with TempNamedDir(tmp_path_dist_ckpt / 'hybrid_optimizer', sync=True) as directory:
            args = parse_args(ignore_unknown_args=True)
            init_basic_mock_args(args, tp, 1, bf16=True)
            init_checkpointing_mock_args(args, directory)
            args.save_tokenizer_assets = False
            args.dist_ckpt_optim_fully_reshardable = checkpoint_format == 'fully_reshardable'
            if checkpoint_format == 'torch':
                args.ckpt_format = 'torch'
                args.use_dist_ckpt = False
                # Only unpickle the checkpoint created by this test. Legacy Torch
                # checkpoints include Namespace/RNG objects, not just tensor weights.
                monkeypatch.delenv('TORCH_FORCE_WEIGHTS_ONLY_LOAD', raising=False)
                monkeypatch.setenv('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
            args.optimizer_cpu_offload = True
            args.optimizer_offload_fraction = 0.5
            args.overlap_cpu_optimizer_d2h_h2d = True
            args.use_precision_aware_optimizer = precision_aware

            def metadata(*metadata_args, **kwargs):
                result = _build_sharded_state_dict_metadata(*metadata_args, **kwargs)
                if checkpoint_format == 'dp_zero_gather_scatter':
                    result['distrib_optim_sharding_type'] = checkpoint_format
                return result

            with (
                patch('megatron.training.checkpointing.get_args', return_value=args),
                patch(
                    'megatron.training.checkpointing._build_sharded_state_dict_metadata', metadata
                ),
            ):
                model, optimizer = construct()
                for step in (1, 2):
                    train_step(model, optimizer, step)
                save_checkpoint(
                    2,
                    [model],
                    optimizer,
                    None,
                    0,
                    preprocess_common_state_dict_fn=preprocess_common_state_dict,
                )
                expected = {}
                for step in (3, 4, 5):
                    train_step(model, optimizer, step)
                    expected[step] = snapshot(model, optimizer)

                restored_model, restored_optimizer = construct()
                with (
                    patch('megatron.training.checkpointing.check_checkpoint_args'),
                    patch('megatron.training.checkpointing.update_num_microbatches'),
                ):
                    iteration, _ = load_checkpoint([restored_model], restored_optimizer, None)
                assert iteration == 2
                for step in (3, 4, 5):
                    train_step(restored_model, restored_optimizer, step)
                    assert_equal(expected[step], snapshot(restored_model, restored_optimizer))
    finally:
        Utils.destroy_model_parallel()

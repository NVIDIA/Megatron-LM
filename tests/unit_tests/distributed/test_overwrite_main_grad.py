# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""overwrite_main_grad: the gradients of buffers that are not zero-filled equal the zero-filled
ones, and the preconditions are enforced."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64


def _config():
    return TransformerConfig(
        num_layers=1,
        num_attention_heads=1,
        gradient_accumulation_fusion=True,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


class _Model(torch.nn.Module):
    """Linear layers whose weight gradient a fused GEMM writes into main_grad and whose bias
    gradient autograd produces (the backward hook adds it into main_grad)."""

    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                ColumnParallelLinear(
                    HIDDEN,
                    HIDDEN,
                    config=config,
                    init_method=torch.nn.init.xavier_normal_,
                    bias=True,
                    gather_output=False,
                )
                for _ in range(6)
            ]
        )

    def forward(self, inputs):
        for layer in self.layers:
            inputs, _ = layer(inputs)
        return inputs


def _build(overwrite_main_grad, overlap_grad_reduce=True, use_distributed_optimizer=True):
    torch.manual_seed(1234)
    config = _config()
    ddp_config = DistributedDataParallelConfig(
        overwrite_main_grad=overwrite_main_grad,
        overlap_grad_reduce=overlap_grad_reduce,
        use_distributed_optimizer=use_distributed_optimizer,
        bucket_size=8192,  # Several bucket groups.
    )
    return DistributedDataParallel(config, ddp_config, _Model(config))


def _bucket_groups(model):
    return model.bucket_groups + model.expert_parallel_bucket_groups


def _train_batch(model, batch):
    model.zero_grad_buffer()
    # Rank-dependent inputs so that the reduction changes the gradients.
    torch.manual_seed(100 * batch + torch.distributed.get_rank())
    inputs = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device='cuda')
    model(inputs).float().square().mean().backward()
    model.finish_grad_sync()
    torch.cuda.synchronize()


def _grads(model):
    return {name: param.main_grad.clone() for name, param in model.named_parameters()}


class TestOverwriteMainGrad:

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("overlap_grad_reduce", [False, True])
    @pytest.mark.parametrize("use_distributed_optimizer", [False, True])
    def test_gradients_match_zero_filled_buffers(
        self, overlap_grad_reduce, use_distributed_optimizer
    ):
        ref_model = _build(False, overlap_grad_reduce, use_distributed_optimizer)
        model = _build(True, overlap_grad_reduce, use_distributed_optimizer)
        # The second and third iterations run without the zero fill.
        for batch in range(3):
            _train_batch(ref_model, batch)
            _train_batch(model, batch)
            ref_grads, grads = _grads(ref_model), _grads(model)
            for name, ref_grad in ref_grads.items():
                if name.endswith('bias'):
                    # Same autograd gradient added into zeros on both sides.
                    torch.testing.assert_close(grads[name], ref_grad, rtol=0, atol=0)
                else:
                    # A different GEMM writes the weight gradient (no accumulation).
                    torch.testing.assert_close(grads[name], ref_grad)

    def test_only_the_autograd_main_grads_are_zeroed(self):
        model = _build(True)
        weights = [layer.weight for layer in model.module.layers]
        biases = [layer.bias for layer in model.module.layers]
        assert all(param.overwrite_main_grad for param in weights + biases)

        _train_batch(model, 0)
        # The first iteration zero-filled whole buffers; nothing recorded before its backward.
        assert all(
            bucket.main_grads_to_zero is None
            for bg in _bucket_groups(model)
            for bucket in bg.buckets
        )

        _train_batch(model, 1)
        recorded = {
            grad.data_ptr()
            for bucket_group in _bucket_groups(model)
            for bucket in bucket_group.buckets
            for grad in bucket.main_grads_to_zero
        }
        assert recorded == {bias.main_grad.data_ptr() for bias in biases}
        for buffer in model.buffers + model.expert_parallel_buffers:
            assert len(buffer.main_grads_to_zero) == sum(
                len(bucket.main_grads_to_zero) for bucket in buffer.buckets
            )

        model.zero_grad_buffer()
        torch.cuda.synchronize()
        assert all(torch.count_nonzero(bias.main_grad) == 0 for bias in biases)
        assert all(torch.count_nonzero(weight.main_grad) > 0 for weight in weights)

        for bucket_group in _bucket_groups(model):
            for param in bucket_group.params:
                param.main_grad.fill_(1.0)
            bucket_group.zero_grads()
        torch.cuda.synchronize()
        assert all(torch.count_nonzero(bias.main_grad) == 0 for bias in biases)
        assert all(torch.all(weight.main_grad == 1.0) for weight in weights)

    def test_backward_inside_no_sync_raises(self):
        model = _build(True)
        model.zero_grad_buffer()
        inputs = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device='cuda')
        with model.no_sync():
            with pytest.raises(RuntimeError, match="no_sync"):
                model(inputs).float().square().mean().backward()

    def test_several_micro_batches_are_refused(self):
        dp_size = torch.distributed.get_world_size()
        init_num_microbatches_calculator(
            rank=0, global_batch_size=2 * dp_size, micro_batch_size=1, data_parallel_size=dp_size
        )
        try:
            with pytest.raises(ValueError, match="2 micro-batches per step"):
                _build(True)
        finally:
            destroy_num_microbatches_calculator()

    def test_shared_parameters_are_refused(self):
        config = _config()
        module = _Model(config)
        module.layers[1].weight = module.layers[0].weight
        with pytest.raises(ValueError, match="shared parameters"):
            DistributedDataParallel(
                config, DistributedDataParallelConfig(overwrite_main_grad=True), module
            )

    def test_per_layer_cuda_graphs_are_refused(self):
        config = _config()
        config.cuda_graph_impl = 'local'
        with pytest.raises(ValueError, match="cuda_graph_impl"):
            DistributedDataParallel(
                config, DistributedDataParallelConfig(overwrite_main_grad=True), _Model(config)
            )

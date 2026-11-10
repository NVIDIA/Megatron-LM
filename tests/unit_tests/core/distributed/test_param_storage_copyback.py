# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import TestModel, Utils


@pytest.mark.parametrize('sync_mode', ['synchronous', 'overlap', 'force', 'force_pending'])
def test_param_sync_refreshes_relocated_storage(sync_mode):
    """All gathered shards reach relocated parameters and existing CUDA graphs."""
    Utils.initialize_model_parallel()
    try:
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            pytest.skip('Requires at least two optimizer shards')
        rank = torch.distributed.get_rank()
        overlap = sync_mode != 'synchronous'
        config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_param_gather=overlap,
            overlap_grad_reduce=False,
            bucket_size=None,
        )
        module = TestModel(129, 128, num_layers=2, bias=False).bfloat16().cuda()
        params = list(module.parameters())
        layout = DistributedOptimizer.compute_full_param_layout(params, None, world_size, config)
        model = DistributedDataParallel(
            TransformerConfig(num_layers=1, num_attention_heads=1),
            config,
            module,
            full_param_layout=layout,
        )
        assert len(model.buffers) == 1
        buffer = model.buffers[0]
        assert len(buffer.buckets) == 1

        # Reproduce inference packing after DDP and optimizer views are established.
        serving = torch.stack([p.detach() for p in params])
        grads = [p.main_grad for p in params]
        for index, param in enumerate(params):
            param.data = serving[index]
        serving_ptr = serving.data_ptr()
        pointers = [p.data_ptr() for p in params]
        shard_size = buffer.param_data.numel() // world_size
        optimizer_shard = buffer.param_data[rank * shard_size : (rank + 1) * shard_size]

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.stack([p.float().sum() for p in params])
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            graph_output = torch.stack([p.float().sum() for p in params])

        for step in range(1, 4):
            # Each rank changes only its own shard; a shard boundary cuts a parameter.
            optimizer_shard.fill_(step * 10 + rank)
            model.reset_param_sync_dispatch_state()
            if sync_mode == 'force':
                model.start_param_sync(force_sync=True)
            else:
                model.start_param_sync()
                if sync_mode == 'force_pending':
                    model.start_param_sync(force_sync=True)
                elif sync_mode == 'overlap':
                    for group in model.bucket_groups:
                        group.finish_param_sync(skip_next_bucket_dispatch=True)

            expected_buffer = torch.arange(world_size, device='cuda', dtype=torch.bfloat16)
            expected_buffer = (expected_buffer + step * 10).repeat_interleave(shard_size)
            expected_params = []
            for index, param in enumerate(params):
                start, end, _ = buffer.param_index_map[param]
                expected = expected_buffer[start:end].view_as(param)
                expected_params.append(expected)
                torch.testing.assert_close(param, expected, rtol=0, atol=0)
                torch.testing.assert_close(serving[index], expected, rtol=0, atol=0)
                assert param.main_grad is grads[index]
                assert param.data_ptr() == pointers[index]
            assert serving.data_ptr() == serving_ptr
            graph.replay()
            torch.testing.assert_close(
                graph_output,
                torch.stack([p.float().sum() for p in expected_params]),
                rtol=0,
                atol=0,
            )
    finally:
        Utils.destroy_model_parallel()

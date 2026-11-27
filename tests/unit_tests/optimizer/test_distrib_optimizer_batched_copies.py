# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from tests.unit_tests.optimizer.test_num_zeros_in_grads import _model_and_optimizer
from tests.unit_tests.test_utilities import Utils


def _shards(dist_opt, model_groups, main_groups):
    for model_group, main_group in zip(model_groups, main_groups):
        for model_param, main_param in zip(model_group, main_group):
            param_range = dist_opt._get_model_param_range_map(model_param)["param"]
            yield model_param, main_param, param_range


class TestBatchedCopies:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("fp32_param", [False, True])
    def test_main_grads_equal_per_param_casts(self, fp32_param):
        torch.manual_seed(0)
        model, optimizer = _model_and_optimizer(fp32_param, bucket_size=2000)
        model(torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")).float().sum().backward()
        model.finish_grad_sync()
        optimizer.prepare_grads()
        dist_opt = getattr(optimizer, "chained_optimizers", [optimizer])[0]
        for model_groups, main_groups in (
            (dist_opt.model_float16_groups, dist_opt.shard_fp32_from_float16_groups),
            (dist_opt.model_fp32_groups, dist_opt.shard_fp32_groups),
        ):
            for model_param, main_param, rng in _shards(dist_opt, model_groups, main_groups):
                shard = model_param.main_grad.view(-1)[rng.start : rng.end]
                assert main_param.grad.dtype == torch.float32
                assert torch.equal(main_param.grad, shard.float())
                if shard.dtype == torch.float32:
                    # As with ".float()", an fp32 grad shard is used in place.
                    assert main_param.grad.data_ptr() == shard.data_ptr()

    def test_copy_main_params_to_param_buffer(self):
        torch.manual_seed(0)
        _, optimizer = _model_and_optimizer(fp32_param=False, bucket_size=2000)
        dist_opt = getattr(optimizer, "chained_optimizers", [optimizer])[0]
        for main_group in dist_opt.shard_fp32_from_float16_groups:
            for main_param in main_group:
                main_param.data.normal_()
        dist_opt._copy_main_params_to_param_buffer()
        for model_param, main_param, _ in _shards(
            dist_opt, dist_opt.model_float16_groups, dist_opt.shard_fp32_from_float16_groups
        ):
            world_range = dist_opt._get_model_param_range_map(model_param)["gbuf_world_in_bucket"]
            gbuf_index, _, bucket_id = dist_opt.model_param_gbuf_map[model_param]
            param_data = dist_opt.buffers[gbuf_index].buckets[bucket_id].param_data
            staged = param_data.view(-1)[world_range.start : world_range.end]
            assert torch.equal(staged, main_param.to(staged.dtype))

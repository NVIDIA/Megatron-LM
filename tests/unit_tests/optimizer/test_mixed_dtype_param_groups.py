# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
import torch.nn as nn

from megatron.core.dist_checkpointing import load, save
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import (
    convert_module_to_dtype_except_fp32_marked,
    mark_keep_in_fp32,
)
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.test_utils import _init_distributed


class MixedDtypeNet(nn.Module):
    """Module whose single optimizer param group mixes bf16 and fp32 params.

    Mirrors hyper-connections under bf16 training
    (megatron/core/transformer/hyper_connection.py): most weights are
    converted to bf16 while parameters marked with ``mark_keep_in_fp32`` stay
    fp32, so the params of one optimizer group live in different-dtype grad
    buffers.

    The fp32 param sits BETWEEN the two bf16 params in ``named_parameters()``
    order, so no matter which direction DDP walks the params when grouping
    them into per-dtype buffers (it walks in reverse), the gbuf-iteration
    group order interleaves the dtypes — [bf16, fp32, bf16]-ish — while
    ``_build_model_and_main_param_groups()`` installs the group as
    [fp32 shard, bf16 main params...]. A trailing (or leading) fp32 param
    can coincidentally match the installed order and mask the bug; the
    middle position cannot.

    Every parameter has a distinct numel, and each parameter's numel is a
    multiple of lcm(dp, 128) for dp up to 8, so on every DP rank up to world
    size 8 each rank owns a piece of every param and a lookup at the wrong
    group order can never return a tensor of matching size.
    """

    def __init__(self):
        super().__init__()
        # 8192 elements; bf16 after conversion.
        self.linear1 = nn.Linear(128, 64, bias=False)
        # 512 elements; stays fp32. Kept 2-D so the standard weight-decay
        # overrides leave it in the same param group as the linear weights.
        # Deliberately declared BETWEEN the bf16 linears — see class docstring.
        self.alpha = mark_keep_in_fp32(nn.Parameter(torch.ones(16, 32)))
        # 2048 elements; bf16 after conversion.
        self.linear2 = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        y = self.linear2(self.linear1(x))
        return y.float() * self.alpha.mean()


@pytest.mark.skipif(
    not is_torch_min_version("2.6a0"), reason="dp_reshardable requires PyTorch 2.6a0 or later"
)
def test_distrib_optimizer_mixed_dtype_param_group_dp_reshardable(tmp_path_dist_ckpt):
    """Mixed fp32/bf16 param group: dp_reshardable save/load round-trip.

    Regression test for two bugs surfaced by param groups that mix fp32 and
    float16 model params (e.g. hyper-connections params marked keep_in_fp32
    under bf16 training):

    1. ``_build_model_and_main_param_groups()`` installs each optimizer
       group's params as [*fp32 shards, *fp32-from-float16 shards], which
       reorders the group relative to the gbuf-iteration order recorded in
       ``model_param_group_index_map``. Unless the map is rebuilt to match
       the installed order, ``_get_main_param_and_optimizer_states()``
       fetches the wrong param's shard and checkpoint save with
       ``metadata={'distrib_optim_sharding_type': 'dp_reshardable'}`` crashes
       on a shape assertion in ``sharded_param_state_dp_reshardable()``.

    2. For fp32 model params the optimizer group entry is an
       autograd-tracked view of the model param (built without detach()), so
       the load path's in-place copy in
       ``_set_main_param_and_optimizer_states()`` must run under no_grad or
       ``load_parameter_state_from_dp_reshardable()`` raises "a view of a
       leaf Variable that requires grad is being used in an in-place
       operation".
    """
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()

    try:
        model = MixedDtypeNet().cuda()
        # Same conversion path as Float16Module: everything becomes bf16
        # except params marked with mark_keep_in_fp32.
        convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
        assert model.linear1.weight.dtype == torch.bfloat16
        assert model.linear2.weight.dtype == torch.bfloat16
        assert model.alpha.dtype == torch.float32

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )
        optimizer_config = OptimizerConfig(
            optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=True
        )
        optim = get_megatron_optimizer(optimizer_config, [model])
        distrib_optim = optim.chained_optimizers[0]

        # Sanity check of the repro precondition: at least one param group on
        # this rank must mix fp32 and bf16 model params.
        assert any(
            fp32_params and float16_params
            for fp32_params, float16_params in zip(
                distrib_optim.model_fp32_groups, distrib_optim.model_float16_groups
            )
        ), "setup error: expected a param group mixing fp32 and bf16 model params"

        # The map must point at the very shard that was installed for each
        # model param, not merely at a shard of the right size.
        for model_param, (
            group_index,
            group_order,
        ) in distrib_optim.model_param_group_index_map.items():
            shard = distrib_optim.optimizer.param_groups[group_index]["params"][group_order]
            local_range = distrib_optim._get_model_param_range_map(model_param)["gbuf_local"]
            assert shard.numel() == local_range.size, (
                f"optimizer group ({group_index}, {group_order}) holds a shard of numel "
                f"{shard.numel()}, but the {tuple(model_param.shape)} {model_param.dtype} "
                f"model param owns a local gbuf range of numel {local_range.size}"
            )
            if model_param.dtype == torch.float32:
                # fp32 shards are views into the model param itself.
                assert (
                    shard.untyped_storage().data_ptr() == model_param.untyped_storage().data_ptr()
                ), "fp32 group entry is not a view of the mapped model param"
            else:
                # float16/bf16 shards are the allocated fp32 main-param copies.
                assert (
                    shard is model_param.main_param
                ), "float16 group entry is not the mapped model param's main_param"

        # One step so the Adam state (exp_avg, exp_avg_sq) exists in the saved
        # parameter state.
        inputs = torch.randn(8, 128, dtype=torch.bfloat16, device='cuda')
        loss = model(inputs).sum()
        loss.backward()
        optim.step()

        metadata = {'distrib_optim_sharding_type': 'dp_reshardable'}

        # The customer-visible save crash path: dp_reshardable checkpoint save
        # asserts every saved tensor matches its local gbuf range.
        sharded_state_dict = optim.sharded_state_dict(model.sharded_state_dict(), metadata=metadata)
        assert 'param_state' in sharded_state_dict

        # Snapshot the state the checkpoint should preserve. Skip 'step': it
        # round-trips as a LocalNonpersistentObject pointing at the live
        # tensor, so it cannot be perturbed-and-restored.
        saved_states = {}
        for model_param in distrib_optim.model_param_group_index_map:
            tensors = distrib_optim._get_main_param_and_optimizer_states(model_param)
            saved_states[model_param] = {
                key: tensor.detach().clone() for key, tensor in tensors.items() if key != 'step'
            }

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_mixed_dtype_param_group_dp_reshardable', sync=True
        ) as ckpt_dir:
            save(sharded_state_dict, ckpt_dir)

            # Perturb the live main params and optimizer state. The fp32 model
            # param's main param is an autograd-tracked view of the model
            # param itself, so these writes must run under no_grad — the same
            # property the optimizer load path has to respect.
            with torch.no_grad():
                for model_param, saved in saved_states.items():
                    live = distrib_optim._get_main_param_and_optimizer_states(model_param)
                    for key in saved:
                        live[key].fill_(123.0)
                        assert not torch.equal(live[key], saved[key])

            # The customer-visible load crash path: resume from the
            # dp_reshardable checkpoint. Without the no_grad guard in
            # _set_main_param_and_optimizer_states(), load_state_dict() raises
            # a leaf-view in-place RuntimeError on the fp32 main-param view.
            load_sharded_state_dict = optim.sharded_state_dict(
                model.sharded_state_dict(), is_loading=True, metadata=metadata
            )
            loaded_state_dict = load(load_sharded_state_dict, ckpt_dir)
            optim.load_state_dict(loaded_state_dict)

            for model_param, saved in saved_states.items():
                restored = distrib_optim._get_main_param_and_optimizer_states(model_param)
                for key, saved_tensor in saved.items():
                    assert torch.equal(restored[key], saved_tensor), (
                        f"optimizer state '{key}' of the {tuple(model_param.shape)} "
                        f"{model_param.dtype} model param was not restored from the checkpoint"
                    )
    finally:
        Utils.destroy_model_parallel()

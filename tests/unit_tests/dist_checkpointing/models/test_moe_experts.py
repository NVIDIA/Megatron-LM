# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support, fp8_autocast

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

fp8_available, reason_for_no_fp8 = check_fp8_support()


def initialize_expert_layer(seed, glu=True, expert_type='sequential', fp8=False, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    num_moe_experts = 8
    num_local_experts = num_moe_experts // parallel_state.get_expert_model_parallel_world_size()
    default_config_kwargs = dict(
        num_layers=pp_size,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=num_moe_experts,
        use_cpu_initialization=True,
        gated_linear_unit=glu,
        fp8="hybrid" if fp8 else None,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    if expert_type == 'grouped':
        model = GroupedMLP(num_local_experts, transformer_config)
    elif expert_type == 'te_grouped':
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=True
        )
        model = TEGroupedMLP(
            num_local_experts,
            transformer_config,
            transformer_layer_spec.submodules.mlp.submodules.experts.submodules,
        )
    elif expert_type == 'sequential':
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        model = SequentialMLP(
            num_local_experts,
            transformer_config,
            transformer_layer_spec.submodules.mlp.submodules.experts.submodules,
        )
    elif expert_type == 'te_sequential':
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        model = SequentialMLP(
            num_local_experts,
            transformer_config,
            transformer_layer_spec.submodules.mlp.submodules.experts.submodules,
        )
    else:
        raise ValueError(
            'expert_type can only be one of ["sequential", "te_sequential", "grouped",'
            ' "te_grouped"]'
        )
    return model


def get_pp_offsets():
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    return ((0, pp_rank, pp_size),)


expert_type = ['sequential', 'grouped']
src_dest_expert_type = [('sequential', 'grouped'), ('grouped', 'sequential')]
if is_te_min_version("1.7.0.dev0"):
    expert_type.append('te_sequential')
    src_dest_expert_type.append(('sequential', 'te_sequential'))
    src_dest_expert_type.append(('te_sequential', 'sequential'))
if is_te_min_version("1.9.0.dev0"):
    expert_type.append('te_grouped')
    src_dest_expert_type.append(('te_sequential', 'te_grouped'))
    src_dest_expert_type.append(('te_grouped', 'te_sequential'))


class TestExpertLayerReconfiguration:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "use_fpsl,src_tp_pp_ep_etp,dest_tp_pp_ep_etp,use_glu",
        [
            # changing PP is impossible because the number of layers must be the same
            (False, (2, 4, 1, 2), (2, 4, 1, 2), False),
            (True, (2, 4, 1, 2), (2, 4, 1, 2), False),
            (False, (2, 4, 1, 2), (1, 4, 1, 2), False),
            (True, (2, 1, 1, 2), (1, 1, 1, 2), False),
            (False, (1, 1, 1, 1), (1, 1, 1, 1), False),
            (True, (1, 1, 1, 1), (1, 1, 4, 1), False),
            (False, (1, 1, 8, 1), (1, 1, 2, 1), False),
            (False, (2, 2, 2, 2), (4, 2, 1, 4), False),
            (True, (1, 1, 4, 1), (8, 1, 1, 1), False),
            (False, (1, 8, 1, 1), (1, 8, 1, 1), False),
            (False, (1, 1, 4, 1), (2, 1, 1, 2), False),
            (False, (2, 1, 4, 1), (2, 1, 1, 4), False),
            (False, (1, 1, 1, 1), (1, 1, 1, 1), True),
            (False, (1, 1, 1, 1), (1, 1, 4, 1), True),
            (True, (1, 1, 1, 1), (2, 1, 1, 1), True),
            (False, (1, 1, 4, 1), (8, 1, 1, 8), True),
        ],
    )
    @pytest.mark.parametrize("expert_type", expert_type)
    @pytest.mark.parametrize(
        "load_order,store_order",
        [
            ("tp-ep-dp-pp", "tp-ep-dp-pp"),
            # ("tp-ep-dp-pp", "ep-tp-dp-pp"),
            # ("ep-tp-dp-pp", "ep-tp-dp-pp"),
            # ("ep-tp-dp-pp", "tp-ep-dp-pp"),
        ],
    )
    def test_parallel_reconfiguration_e2e(
        self,
        tmp_path_dist_ckpt,
        src_tp_pp_ep_etp,
        dest_tp_pp_ep_etp,
        use_glu,
        use_fpsl,
        expert_type,
        load_order,
        store_order,
    ):
        """Test model saving and loading with different TP/PP/EP/ETP(expert-tensor-parallel)"""
        src_tp, src_pp, src_ep, src_etp = src_tp_pp_ep_etp
        dest_tp, dest_pp, dest_ep, dest_etp = dest_tp_pp_ep_etp
        if expert_type == 'grouped':
            add_bias_linear = False
        else:
            add_bias_linear = True
        # Save checkpoint A
        Utils.initialize_model_parallel(
            src_tp,
            src_pp,
            expert_model_parallel_size=src_ep,
            expert_tensor_parallel_size=src_etp,
            order=store_order,
        )
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_expert_layer_reconfiguration_model_A'
        ) as ckpt_dir_A, TempNamedDir(
            tmp_path_dist_ckpt / 'test_expert_layer_reconfiguration_model_B'
        ) as ckpt_dir_B:
            model_A = initialize_expert_layer(
                1, use_glu, expert_type, add_bias_linear=add_bias_linear
            )
            sharded_state_dict = model_A.sharded_state_dict(sharded_offsets=get_pp_offsets())

            save_strategy = get_default_save_sharded_strategy()
            if use_fpsl:
                save_strategy = FullyParallelSaveStrategyWrapper(
                    save_strategy,
                    parallel_state.get_data_parallel_group(with_context_parallel=True),
                    True,
                )
            save(sharded_state_dict, ckpt_dir_A, save_strategy)
            Utils.destroy_model_parallel()

            # Load checkpoint A with different TP/PP/EP and save as checkpoint B
            # No FPS this time, only FPL
            Utils.initialize_model_parallel(
                dest_tp,
                dest_pp,
                expert_model_parallel_size=dest_ep,
                expert_tensor_parallel_size=dest_etp,
                order=load_order,
            )
            model_B = initialize_expert_layer(
                1, use_glu, expert_type, add_bias_linear=add_bias_linear
            )
            if use_fpsl:
                load_strategy = get_default_load_sharded_strategy(ckpt_dir_A)
                load_strategy = FullyParallelLoadStrategyWrapper(
                    load_strategy,
                    parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
            else:
                load_strategy = None
            state_dict = load(
                model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()),
                ckpt_dir_A,
                load_strategy,
            )
            model_B.load_state_dict(state_dict)
            save(model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "src_tp_pp_exp,dest_tp_pp_exp,use_glu",
        [
            # changing PP is impossible because the number of layers must be the same
            ((2, 4, 1), (2, 4, 1), False),
            ((1, 1, 1), (1, 1, 4), False),
            ((2, 2, 2), (4, 2, 1), False),
            ((1, 1, 4), (8, 1, 1), False),
            ((2, 1, 4), (1, 1, 8), False),
            ((2, 4, 1), (2, 4, 1), True),
            ((1, 1, 1), (1, 1, 4), True),
            ((2, 2, 2), (4, 2, 1), True),
            ((1, 1, 4), (8, 1, 1), True),
            ((2, 1, 4), (1, 1, 8), True),
        ],
    )
    @pytest.mark.parametrize("src_module,dest_module", src_dest_expert_type)
    def test_sequential_grouped_mlp_interchangeable(
        self, tmp_path_dist_ckpt, src_tp_pp_exp, dest_tp_pp_exp, use_glu, src_module, dest_module
    ):
        """Test model saving and loading with different TP/PP/expert parallelism"""
        src_tp, src_pp, src_exp = src_tp_pp_exp
        dest_tp, dest_pp, dest_exp = dest_tp_pp_exp
        if src_module == 'grouped' or dest_module == 'grouped':
            add_bias_linear = False
        else:
            add_bias_linear = True
        # Save checkpoint A
        Utils.initialize_model_parallel(src_tp, src_pp, expert_model_parallel_size=src_exp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_sequential_grouped_mlp_interchangeable_model_A'
        ) as ckpt_dir_A, TempNamedDir(
            tmp_path_dist_ckpt / 'test_sequential_grouped_mlp_interchangeable_model_B'
        ) as ckpt_dir_B:

            model_A = initialize_expert_layer(
                1, use_glu, expert_type=src_module, add_bias_linear=add_bias_linear
            )
            sharded_state_dict = model_A.sharded_state_dict(sharded_offsets=get_pp_offsets())

            save_strategy = get_default_save_sharded_strategy()
            save(sharded_state_dict, ckpt_dir_A, save_strategy)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(dest_tp, dest_pp, expert_model_parallel_size=dest_exp)
            model_B = initialize_expert_layer(
                1, use_glu, expert_type=dest_module, add_bias_linear=add_bias_linear
            )
            load_strategy = None
            state_dict = load(
                model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()),
                ckpt_dir_A,
                load_strategy,
            )
            model_B.load_state_dict(state_dict)
            save(model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_te_min_version("1.11.0"),
        reason="FP8 support of TEGroupedMLP is only available in TE 1.11.0 and later.",
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize(
        "src_module,dst_module,src_tp_pp_exp,dest_tp_pp_exp",
        [
            # Changing tp/pp/dp doesn't affect _extra_state
            ('te_sequential', 'te_grouped', (1, 1, 1), (1, 1, 4)),
            ('te_sequential', 'te_grouped', (1, 1, 4), (1, 1, 1)),
            ('te_grouped', 'te_sequential', (1, 1, 1), (1, 1, 4)),
            ('te_grouped', 'te_sequential', (1, 1, 4), (1, 1, 1)),
        ],
    )
    def test_sequential_grouped_mlp_extra_state(
        self, tmp_path_dist_ckpt, src_tp_pp_exp, dest_tp_pp_exp, src_module, dst_module
    ):
        """Test saving and loading _extra_state"""
        src_tp, src_pp, src_exp = src_tp_pp_exp
        dest_tp, dest_pp, dest_exp = dest_tp_pp_exp
        use_glu = True
        Utils.initialize_model_parallel(src_tp, src_pp, expert_model_parallel_size=src_exp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_grouped_mlp_extra_state_model_A'
        ) as ckpt_dir_A, TempNamedDir(
            tmp_path_dist_ckpt / 'test_grouped_mlp_extra_state_model_B'
        ) as ckpt_dir_B, fp8_autocast():
            tokens_per_expert = torch.tensor([16] * (8 // src_exp))
            input_tensor = torch.randn(tokens_per_expert.sum(), 16, device="cuda")

            # Save checkpoint A
            model_A = initialize_expert_layer(1, use_glu, expert_type=src_module, fp8=True)
            model_A = model_A.cuda()
            # fp8 meta is initialized at the first step
            model_A(input_tensor, tokens_per_expert)
            sharded_state_dict = model_A.sharded_state_dict(sharded_offsets=get_pp_offsets())

            save_strategy = get_default_save_sharded_strategy()
            save(sharded_state_dict, ckpt_dir_A, save_strategy)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(dest_tp, dest_pp, expert_model_parallel_size=dest_exp)
            load_strategy = None

            # model_A load checkpoint A
            model_A = initialize_expert_layer(1, use_glu, expert_type=src_module, fp8=True)
            model_A = model_A.cuda()
            state_dict = load(
                model_A.sharded_state_dict(sharded_offsets=get_pp_offsets()),
                ckpt_dir_A,
                load_strategy,
            )
            model_A.load_state_dict(state_dict)

            # model_B load checkpoint A
            model_B = initialize_expert_layer(1, use_glu, expert_type=dst_module, fp8=True)
            model_B = model_B.cuda()
            state_dict = load(
                model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()),
                ckpt_dir_A,
                load_strategy,
            )
            model_B.load_state_dict(state_dict)

            # Should be bitwise equal
            if src_module == "te_grouped":
                model_A, model_B = model_B, model_A
            torch.testing.assert_close(
                torch.cat(
                    [
                        model_A.local_experts[i]
                        .linear_fc1.fp8_meta["scaling_fwd"]
                        .amax_history.view(-1, 1)
                        for i in range(8 // dest_exp)
                    ],
                    dim=1,
                ).view(1024, -1),
                model_B.linear_fc1.fp8_meta["scaling_fwd"].amax_history,
                rtol=0,
                atol=0,
            )

            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_te_min_version("1.9.0"),
        reason="TEGroupedMLP is only supported in TE 1.9.0 and later.",
    )
    @pytest.mark.parametrize("ep_size", [1, 2])
    def test_te_grouped_linear_torch_native(self, tmp_path_dist_ckpt, ep_size):
        """Test saving and loading torch native checkpoints"""
        use_glu = True
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep_size)
        with TempNamedDir(tmp_path_dist_ckpt / 'test_te_grouped_linear_torch_native') as ckpt_dir:
            tokens_per_expert = torch.tensor([16] * (8 // ep_size))
            input_tensor = torch.randn(tokens_per_expert.sum(), 16, device="cuda")

            # Save checkpoint
            model = initialize_expert_layer(1, use_glu, expert_type="te_grouped")
            model = model.cuda()
            model(input_tensor, tokens_per_expert)
            torch.save(model.state_dict(), ckpt_dir / f"model_ep{torch.distributed.get_rank()}.pt")

            # Load checkpoint
            state_dict = torch.load(ckpt_dir / f"model_ep{torch.distributed.get_rank()}.pt")
            model.load_state_dict(state_dict)

            Utils.destroy_model_parallel()

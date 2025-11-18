# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.sparse_attention import (
    Indexer,
    IndexerSubmodules,
    SparseAttention,
    SparseAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class MockState:
    """Mock optimizer/scheduler state for checkpointing."""

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict


def create_checkpoint_args(save_dir, load_dir=None):
    """Create args for Megatron checkpointing."""
    args = SimpleNamespace()
    args.save = save_dir
    args.load = load_dir if load_dir is not None else save_dir
    args.ckpt_format = 'torch'
    args.use_distributed_optimizer = True
    args.use_dist_ckpt = False
    args.finetune = False
    args.no_load_optim = False
    args.no_load_rng = False
    args.perform_initialization = True
    args.bf16 = True
    args.pipeline_model_parallel_size = 1
    args.tensor_model_parallel_size = 1
    args.num_layers_per_virtual_pipeline_stage = None
    return args


class TestIndexerCheckpointing:
    """Test checkpoint save and load for Indexer."""

    def _create_config(self):
        """Helper to create MLA config."""
        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            # Sparse attention specific configs
            index_n_heads=8,
            index_head_dim=64,
            index_topk=32,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
        )

    def _create_indexer(self, config, pg_collection):
        """Helper to create indexer."""
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = IndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )

        return Indexer(config, indexer_submodules, pg_collection)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_indexer_save_load_checkpoint(self, tmp_path_dist_ckpt):
        """Test that indexer can be saved and loaded using Megatron checkpointing."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        # Create indexer directly (it's already a MegatronModule)
        indexer = self._create_indexer(config, pg_collection).cuda()

        # Save original state
        original_state_dict = {k: v.clone().cpu() for k, v in indexer.state_dict().items()}

        with TempNamedDir(tmp_path_dist_ckpt / 'test_indexer_mcore_checkpoint') as ckpt_dir:
            # Setup args for checkpointing
            args = create_checkpoint_args(ckpt_dir)
            set_args(args)

            # Save checkpoint using Megatron API
            iteration = 100
            optimizer = MockState({"optimizer": "state"})
            opt_param_scheduler = MockState({"scheduler": "state"})

            save_checkpoint(iteration, [indexer], optimizer, opt_param_scheduler, 0)

            # Verify checkpoint file exists
            ckpt_path = ckpt_dir / "iter_0000100" / "mp_rank_00" / "model_optim_rng.pt"
            assert os.path.exists(ckpt_path), f"Checkpoint file should exist at {ckpt_path}"

            # Create new indexer with different initialization
            new_indexer = self._create_indexer(config, pg_collection).cuda()
            new_optimizer = MockState({"optimizer": "dummy"})
            new_opt_param_scheduler = MockState({"scheduler": "dummy"})

            # Load checkpoint using Megatron API
            loaded_iter, _ = load_checkpoint(
                [new_indexer], new_optimizer, new_opt_param_scheduler, strict=True
            )

            assert loaded_iter == iteration, f"Loaded iteration should be {iteration}"

            # Verify weights match after loading
            for key in original_state_dict:
                assert torch.allclose(
                    original_state_dict[key],
                    new_indexer.state_dict()[key].cpu(),
                    rtol=1e-5,
                    atol=1e-5,
                ), f"Loaded weights should match original for {key}"

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tp_size", [1, 2, 4])
    def test_indexer_checkpoint_with_tp(self, tmp_path_dist_ckpt, tp_size):
        """Test indexer checkpoint save/load with Megatron API and tensor parallelism."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        # Create indexer directly (it's already a MegatronModule)
        indexer = self._create_indexer(config, pg_collection).cuda()

        # Get original state on all ranks
        original_state_dict = {k: v.clone().cpu() for k, v in indexer.state_dict().items()}

        with TempNamedDir(tmp_path_dist_ckpt / f'test_indexer_mcore_tp{tp_size}') as ckpt_dir:
            # Setup args for checkpointing
            args = create_checkpoint_args(ckpt_dir)
            args.tensor_model_parallel_size = tp_size
            set_args(args)

            # Save checkpoint using Megatron API
            iteration = 200
            optimizer = MockState({"optimizer": "state"})
            opt_param_scheduler = MockState({"scheduler": "state"})

            save_checkpoint(iteration, [indexer], optimizer, opt_param_scheduler, 0)

            # Create new indexer with different initialization
            new_indexer = self._create_indexer(config, pg_collection).cuda()
            new_optimizer = MockState({"optimizer": "dummy"})
            new_opt_param_scheduler = MockState({"scheduler": "dummy"})

            # Load checkpoint using Megatron API
            loaded_iter, _ = load_checkpoint(
                [new_indexer], new_optimizer, new_opt_param_scheduler, strict=True
            )

            assert loaded_iter == iteration

            # Verify weights match on all ranks
            for key in original_state_dict:
                assert torch.allclose(
                    original_state_dict[key],
                    new_indexer.state_dict()[key].cpu(),
                    rtol=1e-5,
                    atol=1e-5,
                ), f"Loaded weights should match original for {key} on TP rank {parallel_state.get_tensor_model_parallel_rank()}"

            # Verify weights are identical across all TP ranks (duplicated)
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                for key, param in new_indexer.state_dict().items():
                    param_list = [torch.zeros_like(param) for _ in range(world_size)]
                    torch.distributed.all_gather(param_list, param)
                    for i in range(1, world_size):
                        assert torch.equal(
                            param_list[0], param_list[i]
                        ), f"Parameter {key} should be identical across all ranks after loading"

        Utils.destroy_model_parallel()


class TestSparseAttentionCheckpointing:
    """Test checkpoint save and load for SparseAttention."""

    def _create_config(self):
        """Helper to create MLA config."""
        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            # Sparse attention specific configs
            index_n_heads=8,
            index_head_dim=64,
            index_topk=32,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            indexer_loss_coeff=0.1,
            use_sparse_indexer_loss=False,
        )

    def _create_sparse_attention(self, config, pg_collection):
        """Helper to create sparse attention."""
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = IndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )

        indexer_spec = ModuleSpec(
            module=Indexer, submodules=indexer_submodules, params={'config': config}
        )

        sparse_attention_submodules = SparseAttentionSubmodules(indexer=indexer_spec)

        return SparseAttention(
            config=config,
            submodules=sparse_attention_submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=pg_collection,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tp_size", [1, 2, 4])
    def test_sparse_attention_checkpoint_with_tp(self, tmp_path_dist_ckpt, tp_size):
        """Test sparse attention checkpoint save/load with Megatron API and tensor parallelism."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        # Create sparse attention directly (it's already a MegatronModule)
        sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()

        # Get original indexer state on all ranks
        original_indexer_state = {
            k: v.clone().cpu() for k, v in sparse_attention.indexer.state_dict().items()
        }

        with TempNamedDir(tmp_path_dist_ckpt / f'test_sparse_attn_mcore_tp{tp_size}') as ckpt_dir:
            # Setup args for checkpointing
            args = create_checkpoint_args(ckpt_dir)
            args.tensor_model_parallel_size = tp_size
            set_args(args)

            # Save checkpoint using Megatron API
            iteration = 300
            optimizer = MockState({"optimizer": "state"})
            opt_param_scheduler = MockState({"scheduler": "state"})

            save_checkpoint(iteration, [sparse_attention], optimizer, opt_param_scheduler, 0)

            # Create new sparse attention with different initialization
            new_sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()
            new_optimizer = MockState({"optimizer": "dummy"})
            new_opt_param_scheduler = MockState({"scheduler": "dummy"})

            # Load checkpoint using Megatron API
            loaded_iter, _ = load_checkpoint(
                [new_sparse_attention], new_optimizer, new_opt_param_scheduler, strict=True
            )

            assert loaded_iter == iteration

            # Verify indexer weights match on all ranks
            for key in original_indexer_state:
                assert torch.allclose(
                    original_indexer_state[key],
                    new_sparse_attention.indexer.state_dict()[key].cpu(),
                    rtol=1e-5,
                    atol=1e-5,
                ), f"Loaded indexer weights should match original for {key} on TP rank {parallel_state.get_tensor_model_parallel_rank()}"

            # Verify indexer weights are identical across all TP ranks (duplicated)
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                for key, param in new_sparse_attention.indexer.state_dict().items():
                    param_list = [torch.zeros_like(param) for _ in range(world_size)]
                    torch.distributed.all_gather(param_list, param)
                    for i in range(1, world_size):
                        assert torch.equal(
                            param_list[0], param_list[i]
                        ), f"Indexer parameter {key} should be identical across all ranks after loading"

        Utils.destroy_model_parallel()

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for MimoOptimizer.

Unit tests (no distributed):
    pytest tests/unit_tests/models/test_mimo_optimizer.py -v -k "not distributed"

Integration tests (requires torchrun):
    torchrun --nproc_per_node=2 tests/unit_tests/models/test_mimo_optimizer.py
"""

import pytest
import torch

from megatron.core.models.mimo.optimizer import ModuleOptimizerInfo, MimoOptimizer


class TestModuleOptimizerInfo:
    """Tests for ModuleOptimizerInfo dataclass."""

    def test_create_active(self):
        info = ModuleOptimizerInfo(
            optimizer=None,
            grid=None,
            pg_collection=None,
            is_active=True,
        )
        assert info.is_active is True

    def test_create_inactive(self):
        info = ModuleOptimizerInfo(
            optimizer=None,
            grid=None,
            pg_collection=None,
            is_active=False,
        )
        assert info.is_active is False


class TestMimoOptimizerUnit:
    """Unit tests for MimoOptimizer (no distributed required)."""

    def test_init_empty(self):
        """Test initialization with no active optimizers."""
        from megatron.core.optimizer.optimizer_config import OptimizerConfig

        config = OptimizerConfig(optimizer='adam', lr=1e-4)
        module_infos = {
            "encoder": ModuleOptimizerInfo(None, None, None, is_active=False),
            "language": ModuleOptimizerInfo(None, None, None, is_active=False),
        }
        opt = MimoOptimizer(module_infos, config)

        assert opt.is_stub_optimizer is True
        assert len(opt._active_optimizers) == 0

    def test_param_groups_empty(self):
        """Test param_groups property with no active optimizers."""
        from megatron.core.optimizer.optimizer_config import OptimizerConfig

        config = OptimizerConfig(optimizer='adam', lr=1e-4)
        module_infos = {}
        opt = MimoOptimizer(module_infos, config)

        assert opt.param_groups == []

    def test_state_dict_empty(self):
        """Test state_dict with no active optimizers."""
        from megatron.core.optimizer.optimizer_config import OptimizerConfig

        config = OptimizerConfig(optimizer='adam', lr=1e-4)
        module_infos = {
            "encoder": ModuleOptimizerInfo(None, None, None, is_active=False),
        }
        opt = MimoOptimizer(module_infos, config)

        state = opt.state_dict()
        assert "encoder" in state
        assert state["encoder"] is None


# ============================================================================
# Integration tests (require torchrun)
# ============================================================================

def run_distributed_test():
    """Run distributed integration test."""
    import torch.distributed as dist

    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.models.mimo import MimoModel, MimoModelConfig, get_mimo_optimizer
    from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

    def create_grid(offset=0, tp=1, pp=1, dp=1):
        grid = HyperCommGrid(
            shape=[tp, 1, pp, dp, 1],
            dim_names=["tp", "cp", "pp", "dp", "ep"],
            rank_offset=offset,
            backend="nccl",
        )
        grid.create_pg(["tp"])
        grid.create_pg(["cp"])
        grid.create_pg(["pp"])
        grid.create_pg(["dp"])
        grid.create_pg(["dp", "cp"])
        grid.create_pg(["ep"])
        # Required by _get_pg_collection_for_optimizer
        grid.create_pg(["tp", "pp"])
        grid.create_pg(["tp", "ep", "pp"])
        grid.create_pg(["dp", "ep"])
        return grid

    def get_pg_collection(grid):
        pg = ProcessGroupCollection()
        pg.tp = grid.get_pg("tp")
        pg.cp = grid.get_pg("cp")
        pg.pp = grid.get_pg("pp")
        pg.ep = grid.get_pg("ep")
        pg.dp = grid.get_pg("dp")
        pg.dp_cp = grid.get_pg(["dp", "cp"])

        if pg.pp:
            pp_ranks = sorted(dist.get_process_group_ranks(pg.pp))
            pos_embd_pg = dist.new_group(ranks=[pp_ranks[0]])
            embd_ranks = [pp_ranks[0]]
            if pp_ranks[-1] != pp_ranks[0]:
                embd_ranks.append(pp_ranks[-1])
            embd_pg = dist.new_group(ranks=embd_ranks)
            pg.pos_embd = pos_embd_pg if is_pp_first_stage(pg.pp) else None
            pg.embd = embd_pg if (is_pp_last_stage(pg.pp) or is_pp_first_stage(pg.pp)) else None

        return pg

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}/{world_size}] Starting MimoOptimizer test")

    # Create grids: encoder on rank 0, LLM on rank 1
    encoder_grid = create_grid(offset=0, tp=1, pp=1, dp=1)
    llm_grid = create_grid(offset=1, tp=1, pp=1, dp=1)

    hidden_size = 64
    num_layers = 2
    vocab_size = 1000
    seq_len = 64

    # Create model specs
    encoder_pg = get_pg_collection(encoder_grid)
    llm_pg = get_pg_collection(llm_grid)

    lm_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )

    encoder_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": True,
            "post_process": True,
            "pg_collection": llm_pg,
        },
    )

    encoder_spec = ModuleSpec(
        module=TransformerBlock,
        params={
            "config": encoder_config,
            "spec": get_gpt_layer_with_transformer_engine_spec(),
            "pg_collection": encoder_pg,
            "pre_process": True,
            "post_process": True,
        },
    )

    vision_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={"encoders": {"clip": encoder_spec}, "input_projections": []},
    )

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_spec},
        special_token_ids={"images": 50257},
        module_to_grid_map={"images": encoder_grid, "language": llm_grid},
        language_module_key="language",
    )

    mimo_model = MimoModel(mimo_config)
    mimo_model.to(torch.device("cuda")).to(torch.bfloat16)

    # Wrap with DDP
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        bucket_size=10000,
        use_distributed_optimizer=True,
    )

    if mimo_model.language_model is not None:
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=ddp_config,
            module=mimo_model.language_model,
            pg_collection=llm_pg,
        )

    if "images" in mimo_model.modality_submodules and mimo_model.modality_submodules["images"] is not None:
        submodule = mimo_model.modality_submodules["images"]
        mimo_model.modality_submodules["images"] = DistributedDataParallel(
            config=submodule.encoders['clip'].config,
            ddp_config=ddp_config,
            module=submodule,
            pg_collection=encoder_pg,
        )

    # Create optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=True,
    )

    optimizer = get_mimo_optimizer(mimo_model, opt_config)

    print(f"[Rank {rank}] Created optimizer with {len(optimizer._active_optimizers)} active optimizers")

    # Verify structure
    assert "images" in optimizer.module_infos
    assert "language" in optimizer.module_infos

    if rank == 0:
        assert optimizer.module_infos["images"].is_active is True
        assert optimizer.module_infos["language"].is_active is False
    else:
        assert optimizer.module_infos["images"].is_active is False
        assert optimizer.module_infos["language"].is_active is True

    # Test zero_grad and basic operations
    optimizer.zero_grad()

    # Test state dict
    state = optimizer.state_dict()
    assert "images" in state
    assert "language" in state

    print(f"[Rank {rank}] MimoOptimizer test PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_distributed_test()

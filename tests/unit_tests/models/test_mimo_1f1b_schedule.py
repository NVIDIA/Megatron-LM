# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Integration tests for MIMO model with 1F1B pipeline schedule.

Run with:
    torchrun --nproc_per_node=2 tests/unit_tests/models/test_mimo_1f1b_schedule.py
"""

import logging
from typing import Dict

import torch
import torch.distributed as dist

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
)
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import MultiModuleProcessGroupCollection, ProcessGroupCollection
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TERowParallelLinear = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    """Create a HyperCommGrid with specified parallelism."""
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1],
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
    return grid


def get_pg_collection(grid):
    """Get ProcessGroupCollection from grid."""
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.get_pg("tp")
    pg_collection.cp = grid.get_pg("cp")
    pg_collection.pp = grid.get_pg("pp")
    pg_collection.ep = grid.get_pg("ep")
    pg_collection.dp = grid.get_pg("dp")
    pg_collection.dp_cp = grid.get_pg(["dp", "cp"])
    return pg_collection


def add_embedding_groups(pg_collection):
    """Add embedding groups to process group collection."""
    if not pg_collection.pp:
        return pg_collection

    pp_ranks = sorted(dist.get_process_group_ranks(pg_collection.pp))
    pos_embd_ranks = [pp_ranks[0]]
    embd_ranks = [pp_ranks[0]]
    if pp_ranks[-1] != pp_ranks[0]:
        embd_ranks.append(pp_ranks[-1])

    pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
    embd_pg = dist.new_group(ranks=embd_ranks)

    pg_collection.pos_embd = pos_embd_pg if is_pp_first_stage(pg_collection.pp) else None
    pg_collection.embd = (
        embd_pg
        if (is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp))
        else None
    )

    return pg_collection


def get_pg_collection_with_embedding_groups(grid):
    """Get ProcessGroupCollection with embedding groups."""
    return add_embedding_groups(get_pg_collection(grid))


def is_rank_in_grid(grid):
    """Check if current rank is in grid."""
    rank = dist.get_rank()
    return grid.rank_offset <= rank < grid.rank_offset + grid.size


# ============================================================================
# Model Spec Helpers
# ============================================================================


def get_language_model_spec(
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    vocab_size: int,
    seq_len: int,
    pg_collection: ProcessGroupCollection,
):
    """Get the language model spec."""
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    pre_process = (pp_rank == 0)
    post_process = (pp_rank == pp_size - 1)

    logger.info(
        f"[get_language_model_spec] Rank {dist.get_rank()}: PP rank={pp_rank}/{pp_size}, "
        f"pre_process={pre_process}, post_process={post_process}"
    )

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1

    lm_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='te',
    )
    language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": language_layer_spec,
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": pre_process,
            "post_process": post_process,
            "pg_collection": pg_collection,
        },
    )
    return language_model_spec


def get_projection_config(hidden_size: int) -> TransformerConfig:
    """Return a TransformerConfig for the vision projection MLP."""
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=1,
    )
    cfg.ffn_hidden_size = hidden_size
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu
    return cfg


def get_projection_layer_spec() -> ModuleSpec:
    """Layer spec for the vision-projection MLP."""
    if TEColumnParallelLinear is None or TERowParallelLinear is None:
        raise RuntimeError("TEColumnParallelLinear and TERowParallelLinear are required")
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )


def get_vision_submodules_spec(
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    language_hidden_size: int,
    pg_collection: ProcessGroupCollection,
):
    """Get the submodule spec for the vision modality."""
    vision_layer_spec = get_gpt_layer_with_transformer_engine_spec()

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1

    # Calculate pre/post process based on PP rank (same as language model spec)
    pp_rank = dist.get_rank(pg_collection.pp)
    pre_process = (pp_rank == 0)
    post_process = (pp_rank == pp_size - 1)

    logger.info(
        f"[get_vision_submodules_spec] Rank {dist.get_rank()}: PP rank={pp_rank}/{pp_size}, "
        f"pre_process={pre_process}, post_process={post_process}"
    )

    vision_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )
    vision_encoder_spec = ModuleSpec(
        module=TransformerBlock,
        params={
            "config": vision_config,
            "spec": vision_layer_spec,
            "pg_collection": pg_collection,
            "pre_process": pre_process,
            "post_process": post_process,
        },
    )

    vision_projection_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": get_projection_config(hidden_size=language_hidden_size),
            "submodules": get_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": vision_config.hidden_size,
            "tp_group": pg_collection.tp,
        },
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {"clip_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )

    return vision_submodule_spec


def get_mimo_model(
    encoder_name: str,
    language_module_name: str,
    encoder_grid: HyperCommGrid,
    llm_grid: HyperCommGrid,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    seq_len: int,
):
    """Create MIMO model with TransformerBlock encoder and GPTModel LLM."""
    language_pg_collection = get_pg_collection_with_embedding_groups(llm_grid)
    vision_pg_collection = get_pg_collection_with_embedding_groups(encoder_grid)

    # Always create full specs on all ranks (POC pattern)
    language_model_spec = get_language_model_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        vocab_size=vocab_size,
        seq_len=seq_len,
        pg_collection=language_pg_collection,
    )

    vision_submodule_spec = get_vision_submodules_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        language_hidden_size=hidden_size,
        pg_collection=vision_pg_collection,
    )

    module_to_grid_map = {
        encoder_name: encoder_grid,
        language_module_name: llm_grid,
    }
    topology = {
        encoder_name: [language_module_name],
        language_module_name: [],
    }

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={encoder_name: vision_submodule_spec},
        special_token_ids={encoder_name: 50257},
        module_to_grid_map=module_to_grid_map,
        language_module_key=language_module_name,
    )

    logger.info(f"[Rank {dist.get_rank()}] Creating MimoModel...")
    mimo_model = MimoModel(mimo_config)
    logger.info(f"[Rank {dist.get_rank()}] MimoModel created successfully")

    mimo_model.to(torch.device("cuda")).to(torch.bfloat16)

    # Wrap with DDP
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        bucket_size=10000,
        use_distributed_optimizer=True,
    )

    if mimo_model.language_model is not None:
        logger.info(f"[Rank {dist.get_rank()}] Wrapping language_model with DDP")
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=ddp_config,
            module=mimo_model.language_model,
            pg_collection=language_pg_collection,
        )

    if encoder_name in mimo_model.modality_submodules:
        submodule = mimo_model.modality_submodules[encoder_name]
        if submodule is not None:
            logger.info(f"[Rank {dist.get_rank()}] Wrapping {encoder_name} submodule with DDP")
            submodule = DistributedDataParallel(
                config=submodule.encoders['clip_encoder'].config,
                ddp_config=ddp_config,
                module=submodule,
                pg_collection=vision_pg_collection,
            )
            mimo_model.modality_submodules[encoder_name] = submodule

    return mimo_model, module_to_grid_map, topology


# ============================================================================
# Data Iterator
# ============================================================================


class DataIterator:
    """Simple data iterator for testing.

    Returns batches matching the POC's MockVLMDataset structure:
    - input_ids: [batch_size, seq_length] with image_seq_length image tokens at start
    - labels: [batch_size, seq_length]
    - loss_mask: [batch_size, seq_length]
    - position_ids: [batch_size, seq_length]
    - modality_inputs: {modality_name: {encoder_name: {'hidden_states': tensor, 'attention_mask': None}}}
    """

    def __init__(self, hidden_size, seq_length, micro_batch_size, vocab_size, encoder_name,
                 image_token_id=50257, image_seq_length=None):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.vocab_size = vocab_size
        self.encoder_name = encoder_name
        self.image_token_id = image_token_id
        # Use half the sequence for image tokens by default
        self.image_seq_length = image_seq_length or (seq_length // 2)

    def __iter__(self):
        return self

    def __next__(self):
        # Create encoder input: [image_seq_length, batch_size, hidden_size]
        # This matches the number of image tokens in input_ids
        encoder_hidden_states = torch.randn(
            self.image_seq_length, self.micro_batch_size, self.hidden_size,
            device='cuda', dtype=torch.bfloat16
        )

        # Create input_ids with image tokens at the beginning (like MockVLMDataset)
        # Shape: [batch_size, seq_length]
        image_tokens = torch.full(
            (self.micro_batch_size, self.image_seq_length),
            self.image_token_id,
            dtype=torch.long, device='cuda'
        )
        text_tokens = torch.randint(
            1, self.vocab_size,  # Avoid 0 (pad token)
            (self.micro_batch_size, self.seq_length - self.image_seq_length),
            device='cuda'
        )
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)

        # Create labels (copy of input_ids, with image tokens set to -100)
        labels = input_ids.clone()
        labels[input_ids == self.image_token_id] = -100

        # Create loss_mask (0 for image tokens, 1 for text tokens)
        loss_mask = torch.ones(
            self.micro_batch_size, self.seq_length,
            device='cuda', dtype=torch.float32
        )
        loss_mask[input_ids == self.image_token_id] = 0.0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(
                self.seq_length, device='cuda'
            ).unsqueeze(0).expand(self.micro_batch_size, -1).clone(),
            # modality_inputs structure from POC
            "modality_inputs": {
                self.encoder_name: {
                    "clip_encoder": {
                        'hidden_states': encoder_hidden_states,
                        'attention_mask': None,
                    }
                }
            },
        }


# ============================================================================
# Test Runner
# ============================================================================


def run_mimo_1f1b_test(
    encoder_tp: int,
    encoder_pp: int,
    encoder_dp: int,
    encoder_offset: int,
    llm_tp: int,
    llm_pp: int,
    llm_dp: int,
    llm_offset: int,
    hidden_size: int = 256,
    num_layers: int = 2,
    vocab_size: int = 1000,
    seq_length: int = 64,
    micro_batch_size: int = 2,
    num_microbatches: int = 4,
):
    """Run MIMO model through 1F1B schedule and verify."""
    encoder_name = "images"
    language_module_name = "language_module"

    logger.info(f"[Rank {dist.get_rank()}] Creating grids...")
    encoder_grid = create_hypercomm_grid(
        offset=encoder_offset, tp=encoder_tp, cp=1, pp=encoder_pp, dp=encoder_dp
    )
    llm_grid = create_hypercomm_grid(
        offset=llm_offset, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp
    )

    torch.manual_seed(12345)

    logger.info(f"[Rank {dist.get_rank()}] Creating MIMO model...")
    mimo_model, module_to_grid_map, topology = get_mimo_model(
        encoder_name=encoder_name,
        language_module_name=language_module_name,
        encoder_grid=encoder_grid,
        llm_grid=llm_grid,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_len=seq_length,
    )

    # Add schedule-related functions to the model's existing config (TransformerConfig)
    # Don't replace it with ModelParallelConfig - schedule expects TransformerConfig attributes
    def no_sync_func():
        from contextlib import contextmanager, ExitStack

        @contextmanager
        def combined_no_sync():
            with ExitStack() as stack:
                if mimo_model.language_model is not None:
                    stack.enter_context(mimo_model.language_model.no_sync())
                for submodule in mimo_model.modality_submodules.values():
                    if submodule is not None:
                        stack.enter_context(submodule.no_sync())
                yield

        return combined_no_sync()

    def finalize_grads_func(*args, **kwargs):
        if mimo_model.language_model is not None:
            llm_pg = get_pg_collection_with_embedding_groups(llm_grid)
            finalize_model_grads([mimo_model.language_model], num_tokens=None, pg_collection=llm_pg)
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                encoder_pg = get_pg_collection_with_embedding_groups(encoder_grid)
                finalize_model_grads([submodule], num_tokens=None, pg_collection=encoder_pg)

    # Add schedule functions to existing model config
    mimo_model.config.no_sync_func = no_sync_func
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float)) else loss
    )

    # Create MimoOptimizer
    logger.info(f"[Rank {dist.get_rank()}] Creating MimoOptimizer...")
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=True,
    )
    optimizer = get_mimo_optimizer(mimo_model, opt_config)
    logger.info(f"[Rank {dist.get_rank()}] MimoOptimizer created with {len(optimizer._active_optimizers)} active optimizers")

    logger.info(f"[Rank {dist.get_rank()}] Creating communicator...")
    communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, mimo_model.config, dim_mapping={'s': 0, 'h': 2, 'b': 1}
    )

    # Create data iterator on:
    # - Encoder's first PP stage (needs modality_inputs)
    # - LLM's first PP stage (needs input_ids for embeddings)
    # - LLM's last PP stage (needs labels for loss)
    data_iterator = None

    encoder_needs_data = (
        is_rank_in_grid(encoder_grid) and
        is_pp_first_stage(encoder_grid.get_pg("pp"))
    )
    llm_needs_data = (
        is_rank_in_grid(llm_grid) and
        (is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp")))
    )

    if encoder_needs_data or llm_needs_data:
        logger.info(f"[Rank {dist.get_rank()}] Creating data iterator (encoder={encoder_needs_data}, llm={llm_needs_data})")
        data_iterator = DataIterator(hidden_size, seq_length, micro_batch_size, vocab_size, encoder_name)

    # Build MultiModuleProcessGroupCollection
    # Only include pg_collections for modules this rank participates in
    module_pgs = {}
    if is_rank_in_grid(encoder_grid):
        module_pgs[encoder_name] = get_pg_collection_with_embedding_groups(encoder_grid)
    if is_rank_in_grid(llm_grid):
        module_pgs[language_module_name] = get_pg_collection_with_embedding_groups(llm_grid)

    # Set language_model_module_name only if this rank participates in LLM
    lang_module_name = language_module_name if is_rank_in_grid(llm_grid) else None

    pg_collection = MultiModuleProcessGroupCollection(
        module_pgs=module_pgs,
        language_model_module_name=lang_module_name,
    )

    def step_func(data_iterator, model):
        from functools import partial

        def loss_func(loss_mask, output_tensor):
            """Loss function matching POC pattern."""
            if output_tensor is None:
                return torch.tensor(0.0, device='cuda', requires_grad=True), {'loss_reduced': 0.0}

            # Handle dict output (from encoder or intermediate LLM stages)
            if isinstance(output_tensor, dict):
                if language_module_name in output_tensor:
                    output = output_tensor[language_module_name]
                else:
                    output = list(output_tensor.values())[0] if output_tensor else None
            else:
                output = output_tensor

            if output is None:
                return torch.tensor(0.0, device='cuda', requires_grad=True), {'loss_reduced': 0.0}

            loss = output.float().sum()
            return loss, {'loss_reduced': loss}

        batch = next(data_iterator) if data_iterator is not None else {'input_ids': None}
        # MimoModel.forward() returns (output_tensor, loss_mask) tuple
        output_tensor, loss_mask = model(**batch)
        # Return only output_tensor, bind loss_mask to loss_func via partial
        return output_tensor, partial(loss_func, loss_mask)

    logger.info(f"[Rank {dist.get_rank()}] Running 1F1B schedule with {num_microbatches} microbatches...")

    # Zero gradients before forward/backward
    optimizer.zero_grad()

    losses = schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=step_func,
        data_iterator=data_iterator,
        model=[mimo_model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
        p2p_communicator=communicator,
        pg_collection=pg_collection,
    )

    # Optimizer step with global gradient clipping
    logger.info(f"[Rank {dist.get_rank()}] Running optimizer step...")
    success, grad_norm, num_zeros = optimizer.step()
    logger.info(f"[Rank {dist.get_rank()}] Optimizer step: success={success}, grad_norm={grad_norm}")

    # Verify results on last LLM stage
    if is_rank_in_grid(llm_grid):
        if is_pp_last_stage(llm_grid.get_pg("pp")):
            logger.info(f"[Rank {dist.get_rank()}] Last LLM stage - got {len(losses)} losses")
            assert len(losses) > 0, "Expected losses on last LLM stage"
            for loss_dict in losses:
                assert 'loss_reduced' in loss_dict, "Expected 'loss_reduced' in loss dict"

    logger.info(f"[Rank {dist.get_rank()}] Test completed successfully!")
    return losses


def get_test_configs():
    """Get predefined test configurations for different GPU counts.

    Returns:
        Dict mapping world_size to list of test configurations.
    """
    return {
        # 2 GPUs: Encoder PP=1, LLM PP=1 (baseline)
        2: [
            {
                "name": "baseline_2gpu",
                "encoder_tp": 1, "encoder_pp": 1, "encoder_dp": 1, "encoder_offset": 0,
                "llm_tp": 1, "llm_pp": 1, "llm_dp": 1, "llm_offset": 1,
                "hidden_size": 256, "num_layers": 2, "vocab_size": 1000,
                "seq_length": 64, "micro_batch_size": 2, "num_microbatches": 4,
            },
        ],
        # 4 GPUs: Encoder PP=1, LLM PP=3 (tests keyed output fix)
        4: [
            {
                "name": "lm_pp3_4gpu",
                "encoder_tp": 1, "encoder_pp": 1, "encoder_dp": 1, "encoder_offset": 0,
                "llm_tp": 1, "llm_pp": 3, "llm_dp": 1, "llm_offset": 1,
                "hidden_size": 256, "num_layers": 2, "vocab_size": 1000,
                "seq_length": 64, "micro_batch_size": 2, "num_microbatches": 4,
            },
        ],
        # 8 GPUs: Multiple configurations
        8: [
            # Config 1: Encoder TP=2 PP=1, LLM TP=2 PP=3 (heterogeneous)
            # Encoder: 2 ranks (0-1), LLM: 6 ranks (2-7)
            # num_layers must be divisible by pp, so use 3
            {
                "name": "encoder_tp2_llm_tp2_pp3_8gpu",
                "encoder_tp": 2, "encoder_pp": 1, "encoder_dp": 1, "encoder_offset": 0,
                "llm_tp": 2, "llm_pp": 3, "llm_dp": 1, "llm_offset": 2,
                "hidden_size": 256, "num_layers": 3, "vocab_size": 1000,
                "seq_length": 64, "micro_batch_size": 2, "num_microbatches": 4,
            },
            # Config 2: Encoder PP=2, LLM PP=2 with TP=2 each
            # Encoder: 4 ranks (0-3), LLM: 4 ranks (4-7)
            {
                "name": "full_pp_8gpu",
                "encoder_tp": 2, "encoder_pp": 2, "encoder_dp": 1, "encoder_offset": 0,
                "llm_tp": 2, "llm_pp": 2, "llm_dp": 1, "llm_offset": 4,
                "hidden_size": 256, "num_layers": 2, "vocab_size": 1000,
                "seq_length": 64, "micro_batch_size": 2, "num_microbatches": 4,
            },
        ],
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MIMO 1F1B Schedule Test")
    parser.add_argument("--config", type=str, default=None,
                        help="Specific config name to run (e.g., 'baseline_2gpu')")
    parser.add_argument("--list-configs", action="store_true",
                        help="List available configurations and exit")
    args = parser.parse_args()

    # List configs if requested
    if args.list_configs:
        configs = get_test_configs()
        print("Available configurations:")
        for world_size, config_list in configs.items():
            print(f"\n  {world_size} GPUs:")
            for cfg in config_list:
                print(f"    - {cfg['name']}")
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    logger.info(f"Rank {rank}/{world_size} initialized")

    configs = get_test_configs()

    if world_size not in configs:
        logger.error(f"No configurations for world_size={world_size}. Available: {list(configs.keys())}")
        dist.destroy_process_group()
        return

    # Filter configs if specific one requested
    test_configs = configs[world_size]
    if args.config:
        test_configs = [c for c in test_configs if c["name"] == args.config]
        if not test_configs:
            logger.error(f"Config '{args.config}' not found for {world_size} GPUs")
            dist.destroy_process_group()
            return

    # Run all matching configs
    for config in test_configs:
        name = config.pop("name")
        logger.info(f"Running test: {name}")
        try:
            run_mimo_1f1b_test(**config)
            logger.info(f"Test {name} PASSED")
        except Exception as e:
            logger.error(f"Test {name} FAILED: {e}")
            raise
        finally:
            config["name"] = name  # Restore for potential reuse

    dist.destroy_process_group()
    logger.info("All tests completed successfully!")


if __name__ == "__main__":
    main()

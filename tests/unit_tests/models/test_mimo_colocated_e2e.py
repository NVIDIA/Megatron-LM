# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""End-to-end integration test for MIMO model with colocated modules (no pipeline parallelism).

Both encoder and LLM share the same ranks (offset=0) but use different TP/DP
configurations. Communication between heterogeneous TP/DP layouts is handled by
ColocatedBridgeCommunicator.

Run with:
    uv run python -m torch.distributed.run --nproc_per_node=8 -m pytest tests/unit_tests/models/test_mimo_colocated_e2e.py -v
"""

import logging
from contextlib import ExitStack, contextmanager
from functools import partial

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TERowParallelLinear = None

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions (copied from test_mimo_1f1b_schedule.py to avoid
# cross-test process group conflicts)
# ============================================================================

_active_grids: list = []
_embedding_pg_cache: dict = {}


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    """Create a HyperCommGrid with specified parallelism."""
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1, 1],  # [tp, cp, pp, dp, ep, expt_dp]
        dim_names=["tp", "cp", "pp", "dp", "ep", "expt_dp"],
        rank_offset=offset,
        backend="nccl",
    )
    grid.create_pg(["tp"])
    grid.create_pg(["cp"])
    grid.create_pg(["pp"])
    grid.create_pg(["dp"])
    grid.create_pg(["dp", "cp"])
    grid.create_pg(["ep"])
    grid.create_pg(["expt_dp"])
    # Required by _get_pg_collection_for_optimizer
    grid.create_pg(["tp", "pp"])
    grid.create_pg(["tp", "ep", "pp"])
    grid.create_pg(["dp", "ep"])
    grid.create_pg(["tp", "cp", "ep", "pp", "dp"])
    _active_grids.append(grid)
    return grid


def destroy_all_grids():
    """Destroy all tracked grids and bridge communicator PGs."""
    for grid in _active_grids:
        grid.destroy()
    _active_grids.clear()
    _embedding_pg_cache.clear()
    BridgeCommunicator.destroy_broadcast_pgs()


def get_pg_collection(grid):
    """Get ProcessGroupCollection from grid."""
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.get_pg("tp")
    pg_collection.cp = grid.get_pg("cp")
    pg_collection.pp = grid.get_pg("pp")
    pg_collection.ep = grid.get_pg("ep")
    pg_collection.dp = grid.get_pg("dp")
    pg_collection.dp_cp = grid.get_pg(["dp", "cp"])
    pg_collection.expt_dp = grid.get_pg("expt_dp")
    return pg_collection


def create_all_embedding_groups(grids):
    """Create embedding PGs for all grids upfront.

    dist.new_group is a collective -- ALL ranks must call it, even non-members.
    We create all embedding groups in a consistent order across all ranks to
    avoid hangs from asymmetric new_group calls.
    """
    for grid in grids:
        pp_group = grid.get_pg("pp")
        if not pp_group:
            continue

        pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
        cache_key = tuple(pp_ranks)

        if cache_key not in _embedding_pg_cache:
            pos_embd_ranks = [pp_ranks[0]]
            embd_ranks = [pp_ranks[0]]
            if pp_ranks[-1] != pp_ranks[0]:
                embd_ranks.append(pp_ranks[-1])
            _embedding_pg_cache[cache_key] = (
                dist.new_group(ranks=pos_embd_ranks),
                dist.new_group(ranks=embd_ranks),
            )


def add_embedding_groups(pg_collection, is_language_model=False):
    """Add cached embedding groups to a process group collection."""
    if not pg_collection.pp:
        return pg_collection

    pp_ranks = sorted(dist.get_process_group_ranks(pg_collection.pp))
    cache_key = tuple(pp_ranks)
    pos_embd_pg, embd_pg = _embedding_pg_cache[cache_key]

    pg_collection.pos_embd = pos_embd_pg if is_pp_first_stage(pg_collection.pp) else None

    if is_language_model:
        pg_collection.embd = (
            embd_pg
            if (is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp))
            else None
        )
    else:
        pg_collection.embd = None

    return pg_collection


def get_pg_collection_with_embedding_groups(grid, is_language_model=False):
    """Get ProcessGroupCollection with embedding groups (PGs must be pre-created)."""
    return add_embedding_groups(get_pg_collection(grid), is_language_model=is_language_model)


# ============================================================================
# Model Spec Helpers
# ============================================================================


def get_language_model_spec(
    num_layers, hidden_size, num_attention_heads, vocab_size, seq_len, pg_collection
):
    """Get the language model spec."""
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1

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
    return ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": (pp_rank == 0),
            "post_process": (pp_rank == pp_size - 1),
            "pg_collection": pg_collection,
        },
    )


def get_projection_config(hidden_size):
    """Return a TransformerConfig for the vision projection MLP."""
    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    cfg.ffn_hidden_size = hidden_size
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu
    return cfg


def get_projection_layer_spec():
    """Layer spec for the vision-projection MLP."""
    if TEColumnParallelLinear is None or TERowParallelLinear is None:
        raise RuntimeError("TEColumnParallelLinear and TERowParallelLinear are required")
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )


def get_vision_submodules_spec(
    num_layers, hidden_size, num_attention_heads, language_hidden_size, pg_collection
):
    """Get the submodule spec for the vision modality."""
    from megatron.core.transformer.transformer_block import TransformerBlock

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1
    pp_rank = dist.get_rank(pg_collection.pp)

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
            "spec": get_gpt_layer_with_transformer_engine_spec(),
            "pg_collection": pg_collection,
            "pre_process": (pp_rank == 0),
            "post_process": (pp_rank == pp_size - 1),
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

    return ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {"clip_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )


# ============================================================================
# Data Iterator
# ============================================================================


class DataIterator:
    """Simple data iterator returning VLM-like batches."""

    def __init__(
        self,
        hidden_size,
        seq_length,
        micro_batch_size,
        vocab_size,
        encoder_name,
        image_token_id=50257,
        image_seq_length=None,
    ):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.vocab_size = vocab_size
        self.encoder_name = encoder_name
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length or (seq_length // 2)

    def __iter__(self):
        return self

    def __next__(self):
        encoder_hidden_states = torch.randn(
            self.image_seq_length,
            self.micro_batch_size,
            self.hidden_size,
            device='cuda',
            dtype=torch.bfloat16,
        )

        image_tokens = torch.full(
            (self.micro_batch_size, self.image_seq_length),
            self.image_token_id,
            dtype=torch.long,
            device='cuda',
        )
        text_tokens = torch.randint(
            1,
            self.vocab_size,
            (self.micro_batch_size, self.seq_length - self.image_seq_length),
            device='cuda',
        )
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)

        labels = input_ids.clone()
        labels[input_ids == self.image_token_id] = -100

        loss_mask = torch.ones(
            self.micro_batch_size, self.seq_length, device='cuda', dtype=torch.float32
        )
        loss_mask[input_ids == self.image_token_id] = 0.0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(self.seq_length, device='cuda')
            .unsqueeze(0)
            .expand(self.micro_batch_size, -1)
            .clone(),
            "modality_inputs": {
                self.encoder_name: {
                    "clip_encoder": {'hidden_states': encoder_hidden_states, 'attention_mask': None}
                }
            },
        }


# ============================================================================
# Model Creation for Colocated Config
# ============================================================================


def get_mimo_model_colocated(
    encoder_name, encoder_grid, llm_grid, hidden_size, num_layers, vocab_size, seq_len
):
    """Create MIMO model with colocated grids for same-rank heterogeneous TP/DP."""
    language_pg = get_pg_collection_with_embedding_groups(llm_grid, is_language_model=True)
    vision_pg = get_pg_collection_with_embedding_groups(encoder_grid, is_language_model=False)

    language_model_spec = get_language_model_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        vocab_size=vocab_size,
        seq_len=seq_len,
        pg_collection=language_pg,
    )
    vision_submodule_spec = get_vision_submodules_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        language_hidden_size=hidden_size,
        pg_collection=vision_pg,
    )

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={encoder_name: vision_submodule_spec},
        special_token_ids={encoder_name: 50257},
        module_to_grid_map={encoder_name: encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid},
    )

    mimo_model = MimoModel(mimo_config)
    mimo_model.to(torch.device("cuda")).to(torch.bfloat16)

    # Set model_type so forward_backward_no_pipelining's get_model_type() works
    mimo_model.model_type = ModelType.encoder_or_decoder

    # Wrap with DDP
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True, bucket_size=10000, use_distributed_optimizer=True
    )

    if mimo_model.language_model is not None:
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=ddp_config,
            module=mimo_model.language_model,
            pg_collection=language_pg,
        )

    if encoder_name in mimo_model.modality_submodules:
        submodule = mimo_model.modality_submodules[encoder_name]
        if submodule is not None:
            submodule = DistributedDataParallel(
                config=submodule.encoders['clip_encoder'].config,
                ddp_config=ddp_config,
                module=submodule,
                pg_collection=vision_pg,
            )
            mimo_model.modality_submodules[encoder_name] = submodule

    return mimo_model, language_pg, vision_pg


# ============================================================================
# Test Runner
# ============================================================================


def loss_func(loss_mask, output_tensor):
    """Compute loss from model output."""
    if output_tensor is None:
        return torch.tensor(0.0, device='cuda', requires_grad=True), {'loss_reduced': 0.0}

    loss = output_tensor.float().sum()
    return loss, {'loss_reduced': loss.detach().item()}


def forward_step(data_iterator, model, encoder_grid, llm_grid, encoder_name):
    """Forward step with data slicing for heterogeneous DP."""
    batch = next(data_iterator) if data_iterator is not None else {'input_ids': None}

    if batch.get('input_ids') is None:
        output_tensor, loss_mask = model(**batch)
        return output_tensor, partial(loss_func, loss_mask)

    encoder_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()

    if encoder_dp > llm_dp:
        # Fan-in: data loaded with LLM DP (larger batch per rank)
        # Slice modality_inputs for encoder's smaller batch
        scale = encoder_dp // llm_dp
        encoder_dp_idx = encoder_grid.get_pg("dp").rank()
        slot = encoder_dp_idx % scale

        if 'modality_inputs' in batch and batch['modality_inputs'] is not None:
            for mod_name, mod_data in batch['modality_inputs'].items():
                for enc_name, enc_data in mod_data.items():
                    for key, tensor in enc_data.items():
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            # Encoder inputs are [seq, batch, hidden] -- slice batch dim
                            batch_size = tensor.shape[1]  # batch is dim 1
                            slice_size = batch_size // scale
                            start = slot * slice_size
                            enc_data[key] = tensor[:, start : start + slice_size, :].contiguous()

    elif llm_dp > encoder_dp:
        # Fan-out: slice LLM inputs for LLM's smaller batch
        scale = llm_dp // encoder_dp
        llm_dp_idx = llm_grid.get_pg("dp").rank()
        slot = llm_dp_idx % scale

        batch_size = batch['input_ids'].shape[0]
        slice_size = batch_size // scale
        start = slot * slice_size

        for key in ['input_ids', 'labels', 'loss_mask', 'position_ids']:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key][start : start + slice_size].contiguous()

    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(loss_func, loss_mask)


def run_colocated_test(
    encoder_tp,
    encoder_dp,
    llm_tp,
    llm_dp,
    hidden_size=256,
    num_layers=2,
    vocab_size=1000,
    seq_length=64,
    micro_batch_size=2,
    num_microbatches=2,
):
    """Run MIMO model through forward_backward_no_pipelining with colocated modules."""
    # Clear NVTE env vars that the conftest set_env fixture sets to '0'.
    # GPTModel (LanguageModule) asserts these are unset or match the attention backend.
    import os

    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    encoder_name = "images"

    # Both grids at offset=0 (colocated on same ranks)
    encoder_grid = create_hypercomm_grid(offset=0, tp=encoder_tp, cp=1, pp=1, dp=encoder_dp)
    llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=1, dp=llm_dp)

    # Create all embedding PGs upfront -- dist.new_group is a collective
    create_all_embedding_groups([encoder_grid, llm_grid])

    torch.manual_seed(12345)

    mimo_model, language_pg, vision_pg = get_mimo_model_colocated(
        encoder_name=encoder_name,
        encoder_grid=encoder_grid,
        llm_grid=llm_grid,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_len=seq_length,
    )

    # Create MIMO optimizer (handles per-module DP groups, global grad norm)
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=True,
    )
    optimizer = get_mimo_optimizer(mimo_model, opt_config)

    # Build schedule functions
    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    stack.enter_context(submodule.no_sync())
            yield

    def finalize_grads_func(*args, **kwargs):
        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model], num_tokens=None, pg_collection=language_pg
            )
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                finalize_model_grads([submodule], num_tokens=None, pg_collection=vision_pg)

    mimo_model.config.no_sync_func = no_sync_func
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )

    # Create data iterator -- all ranks need data since PP=1 and all are colocated
    data_iterator = DataIterator(
        hidden_size, seq_length, micro_batch_size, vocab_size, encoder_name
    )

    # Run multiple iterations of forward_backward + optimizer step
    all_losses = []
    num_iterations = 3
    rank = dist.get_rank()
    optimizer.zero_grad()

    for iteration in range(num_iterations):
        losses = schedule.forward_backward_no_pipelining(
            forward_step_func=partial(
                forward_step,
                encoder_grid=encoder_grid,
                llm_grid=llm_grid,
                encoder_name=encoder_name,
            ),
            data_iterator=data_iterator,
            model=[mimo_model],
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
            pg_collection=language_pg,
        )

        # MIMO optimizer step (handles per-module DP groups + global grad norm)
        success, grad_norm, num_zeros = optimizer.step()
        assert success, f"Rank {rank}: Optimizer step failed at iteration {iteration}"
        optimizer.zero_grad()

        all_losses.extend(losses)
        logger.info(f"Rank {rank}: iteration {iteration} completed with {len(losses)} microbatches")

    # Verify losses from all iterations
    assert len(all_losses) > 0, f"Rank {rank}: Expected non-empty losses list"

    loss_values = []
    for i, loss_dict in enumerate(all_losses):
        assert 'loss_reduced' in loss_dict, f"Rank {rank}: Missing 'loss_reduced' in microbatch {i}"
        loss_val = loss_dict['loss_reduced']
        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()
        assert loss_val == loss_val, f"Rank {rank}: Loss is NaN at microbatch {i}"  # NaN check
        assert abs(loss_val) != float('inf'), f"Rank {rank}: Loss is inf at microbatch {i}"
        logger.info(f"Rank {rank}: microbatch {i} loss = {loss_val}")
        loss_values.append(loss_val)

    # At least one microbatch should have non-zero loss
    assert any(
        v != 0.0 for v in loss_values
    ), f"Rank {rank}: All losses are zero -- model did not compute anything"

    # Verify we got losses from all iterations (num_iterations * num_microbatches)
    expected_total = num_iterations * num_microbatches
    assert len(all_losses) == expected_total, (
        f"Rank {rank}: Expected {expected_total} loss entries "
        f"({num_iterations} iterations x {num_microbatches} microbatches), "
        f"got {len(all_losses)}"
    )

    # Oracle check 1: cross-rank loss consistency within the LLM DP group.
    # All TP ranks in the same DP replica should see identical losses (same
    # batch, same weights after param-sync). A silently wrong bridge would
    # route the wrong batch slice and break this invariance.
    llm_dp_pg = llm_grid.get_pg('dp')
    llm_tp_pg = llm_grid.get_pg('tp')
    per_rank = torch.tensor(loss_values, device='cuda', dtype=torch.float64)
    gathered_tp = [torch.empty_like(per_rank) for _ in range(dist.get_world_size(llm_tp_pg))]
    dist.all_gather(gathered_tp, per_rank, group=llm_tp_pg)
    for other in gathered_tp:
        torch.testing.assert_close(per_rank, other, rtol=1e-5, atol=1e-5)

    # Oracle check 2: training signal — sum of losses in iteration 0 should
    # differ from iteration (num_iterations-1). If the optimizer step or the
    # backward path is silently a no-op (e.g. gradients landing on the wrong
    # params), the loss trajectory stays flat and this fires.
    first_iter_sum = sum(loss_values[:num_microbatches])
    last_iter_sum = sum(loss_values[-num_microbatches:])
    assert first_iter_sum != last_iter_sum, (
        f"Rank {rank}: loss unchanged across {num_iterations} iterations "
        f"(first={first_iter_sum}, last={last_iter_sum}) — optimizer step may "
        f"not be updating parameters"
    )
    # And across the full DP group the iteration-0 mean should also change —
    # this rules out the case where per-rank drift happens to cancel.
    first_iter_mean = torch.tensor([first_iter_sum], device='cuda', dtype=torch.float64)
    last_iter_mean = torch.tensor([last_iter_sum], device='cuda', dtype=torch.float64)
    dist.all_reduce(first_iter_mean, group=llm_dp_pg)
    dist.all_reduce(last_iter_mean, group=llm_dp_pg)
    assert not torch.equal(
        first_iter_mean, last_iter_mean
    ), f"Rank {rank}: DP-reduced loss unchanged across iterations"

    return all_losses


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh requires PyTorch 2.3+",
)
class TestMimoColocatedE2E:
    """Test MIMO model with colocated modules and forward_backward_no_pipelining."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def test_colocated_fan_in_8gpu(self):
        """Encoder TP2/DP4, LLM TP4/DP2 -- fan-in case."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_colocated_test(
            encoder_tp=2,
            encoder_dp=4,
            llm_tp=4,
            llm_dp=2,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=2,
        )

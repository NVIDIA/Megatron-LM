# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Integration tests for MIMO model with 1F1B pipeline schedule.

Run with:
    uv run python -m torch.distributed.run --nproc-per-node=2 -m pytest tests/unit_tests/models/test_mimo_1f1b_schedule.py -v
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
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
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
# Helper Functions (with grid tracking and PG caching from edc8159)
# ============================================================================

_active_grids: list = []
_embedding_pg_cache: dict = {}


def build_no_sync_func(mimo_model):
    """Build a no_sync_func that stacks DDP no_sync over each sub-module.

    Shared by 1F1B pipeline tests and colocated-correctness tests — both need
    DDP's gradient sync disabled during microbatches and resumed via the
    schedule's finalize_grads_func.
    """

    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    stack.enter_context(submodule.no_sync())
            yield

    return no_sync_func


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

    dist.new_group is a collective — ALL ranks must call it, even non-members.
    We create all embedding groups in a consistent order across all ranks to
    avoid hangs from asymmetric new_group calls.

    Args:
        grids: List of all HyperCommGrids that need embedding groups.
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
    """Add cached embedding groups to a process group collection.

    Must call create_all_embedding_groups() first to ensure PGs exist.

    Args:
        pg_collection: ProcessGroupCollection to add embedding groups to.
        is_language_model: If True, set embd group for word embedding sync.
    """
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
        # Encoder submodules have no shared word embeddings to sync
        pg_collection.embd = None

    return pg_collection


def get_pg_collection_with_embedding_groups(grid, is_language_model=False):
    """Get ProcessGroupCollection with embedding groups (PGs must be pre-created)."""
    return add_embedding_groups(get_pg_collection(grid), is_language_model=is_language_model)


def is_rank_in_grid(grid):
    """Check if current rank is in grid."""
    rank = dist.get_rank()
    return grid.rank_offset <= rank < grid.rank_offset + grid.size


# ============================================================================
# Model Spec Helpers
# ============================================================================


def get_language_model_spec(
    num_layers,
    hidden_size,
    num_attention_heads,
    vocab_size,
    seq_len,
    pg_collection,
    bf16=True,
    bias=True,
    dropout=True,
    per_token_loss=False,
):
    """Get the language model spec.

    ``bf16=False`` switches pipeline dtype and autocast to fp32. Correctness
    tests also pass ``bias=False, dropout=False`` to remove bias-update and
    stochastic noise from the cross-config diff signal.

    ``per_token_loss=True`` sets ``calculate_per_token_loss=True`` on the
    TransformerConfig, which pins DDP's gradient_scaling_factor to 1.0
    (pure SUM reduction). Callers that flip this must supply a 3-tuple
    loss_func and drive the external divide in their finalize hook.
    """
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1

    pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
    extra_kwargs = {}
    if not bias:
        extra_kwargs['add_bias_linear'] = False
    if not dropout:
        extra_kwargs['attention_dropout'] = 0.0
        extra_kwargs['hidden_dropout'] = 0.0

    lm_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=pipeline_dtype,
        bf16=bf16,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='te',
        calculate_per_token_loss=per_token_loss,
        **extra_kwargs,
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


def get_projection_config(hidden_size, bias=True):
    """Return a TransformerConfig for the vision projection MLP."""
    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    cfg.ffn_hidden_size = hidden_size
    cfg.bias_activation_fusion = bool(bias)
    cfg.add_bias_linear = bool(bias)
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
    num_layers,
    hidden_size,
    num_attention_heads,
    language_hidden_size,
    pg_collection,
    bf16=True,
    bias=True,
    dropout=True,
    per_token_loss=False,
):
    """Get the submodule spec for the vision modality.

    ``bias=False`` / ``dropout=False`` mirror the LM-spec kwargs for
    correctness tests. ``per_token_loss=True`` sets
    ``calculate_per_token_loss=True`` on the encoder's TransformerConfig so
    the encoder DDP also pure-SUMs across DP (needed for the heterogeneous-DP
    colocated path).
    """
    from megatron.core.transformer.transformer_block import TransformerBlock

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1
    pp_rank = dist.get_rank(pg_collection.pp)

    pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
    extra_kwargs = {}
    if not bias:
        extra_kwargs['add_bias_linear'] = False
    if not dropout:
        extra_kwargs['attention_dropout'] = 0.0
        extra_kwargs['hidden_dropout'] = 0.0

    vision_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=pipeline_dtype,
        bf16=bf16,
        calculate_per_token_loss=per_token_loss,
        **extra_kwargs,
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
            "config": get_projection_config(hidden_size=language_hidden_size, bias=bias),
            "submodules": get_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": vision_config.hidden_size,
            "tp_group": pg_collection.tp,
        },
    )

    return ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": pg_collection},
        submodules={
            "encoders": {"clip_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )


def get_mimo_model(
    encoder_name,
    encoder_grid,
    llm_grid,
    hidden_size,
    num_layers,
    vocab_size,
    seq_len,
    ddp_config=None,
    bf16=True,
    bias=True,
    dropout=True,
    per_token_loss=False,
):
    """Create MIMO model with TransformerBlock encoder and GPTModel LLM.

    Args:
        ddp_config: Optional override for the Megatron DDP config. Default
            matches the 1F1B schedule tests' config.
        bf16: If True (default) build the model in bf16; if False build in
            fp32 end-to-end for deterministic numerics in correctness tests.
        bias: If False, disable ``add_bias_linear`` in LM/vision configs and
            the projection MLP — removes bias-update noise from diffs.
        dropout: If False, force attention/hidden dropout to 0.0.
        per_token_loss: If True, set ``calculate_per_token_loss=True`` on
            both sub-model configs. This pins the encoder and LLM DDP
            gradient_scaling_factor to 1.0 (pure SUM across DP). The caller
            MUST supply a 3-tuple loss_func ``(sum_loss, num_tokens,
            log_dict)`` and a custom ``finalize_model_grads_func`` that
            divides grads by the correct global divisor on both sides;
            hetero-DP callers use this to land ``1/B_full`` on both encoder
            and LLM without relying on the per-DDP built-in scaling.
    """
    language_pg = get_pg_collection_with_embedding_groups(llm_grid, is_language_model=True)
    vision_pg = get_pg_collection_with_embedding_groups(encoder_grid, is_language_model=False)

    language_model_spec = get_language_model_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        vocab_size=vocab_size,
        seq_len=seq_len,
        pg_collection=language_pg,
        bf16=bf16,
        bias=bias,
        dropout=dropout,
        per_token_loss=per_token_loss,
    )
    vision_submodule_spec = get_vision_submodules_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        language_hidden_size=hidden_size,
        pg_collection=vision_pg,
        bf16=bf16,
        bias=bias,
        dropout=dropout,
        per_token_loss=per_token_loss,
    )

    module_to_grid_map = {encoder_name: encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid}
    topology = {encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []}

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={encoder_name: vision_submodule_spec},
        special_token_ids={encoder_name: 50257},
        module_to_grid_map=module_to_grid_map,
    )

    mimo_model = MimoModel(mimo_config)
    mimo_model.to(torch.device("cuda"))
    if bf16:
        mimo_model.to(torch.bfloat16)

    # Wrap with DDP (caller may override e.g. for heterogeneous-DP scaling).
    if ddp_config is None:
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

    return mimo_model, module_to_grid_map, topology, language_pg, vision_pg


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
# Test Runner
# ============================================================================


def run_mimo_1f1b_test(
    encoder_tp,
    encoder_pp,
    encoder_dp,
    encoder_offset,
    llm_tp,
    llm_pp,
    llm_dp,
    llm_offset,
    hidden_size=256,
    num_layers=2,
    vocab_size=1000,
    seq_length=64,
    micro_batch_size=2,
    num_microbatches=4,
):
    """Run MIMO model through 1F1B schedule and verify."""
    # Clear NVTE env vars that the conftest set_env fixture sets to '0'.
    # GPTModel (LanguageModule) asserts these are unset or match the attention backend.
    import os

    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    encoder_name = "images"

    encoder_grid = create_hypercomm_grid(
        offset=encoder_offset, tp=encoder_tp, cp=1, pp=encoder_pp, dp=encoder_dp
    )
    llm_grid = create_hypercomm_grid(offset=llm_offset, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)

    # Create all embedding PGs upfront — dist.new_group is a collective that
    # requires ALL ranks to participate, so we must create them before any
    # rank-specific pg_collection calls.
    create_all_embedding_groups([encoder_grid, llm_grid])

    torch.manual_seed(12345)

    mimo_model, module_to_grid_map, topology, language_pg, vision_pg = get_mimo_model(
        encoder_name=encoder_name,
        encoder_grid=encoder_grid,
        llm_grid=llm_grid,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_len=seq_length,
    )

    no_sync_func = build_no_sync_func(mimo_model)

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

    communicator = MultiModulePipelineCommunicator(
        module_to_grid_map,
        topology,
        mimo_model.config,
        dim_mapping={'s': 0, 'h': 2, 'b': 1},
        module_output_ndim={encoder_name: 2},
    )

    # Compute per-rank micro-batch size for asymmetric DP.
    # The LLM's MBS is the schedule-level MBS. The encoder's MBS is adjusted
    # by the DP ratio so that total work is conserved across the bridge.
    llm_mbs = micro_batch_size
    encoder_mbs = micro_batch_size * llm_dp // encoder_dp

    # Create data iterator on ranks that need it, with per-role micro-batch size.
    # Encoder ranks use encoder_mbs, LLM ranks use llm_mbs.
    data_iterator = None
    encoder_needs_data = is_rank_in_grid(encoder_grid) and is_pp_first_stage(
        encoder_grid.get_pg("pp")
    )
    llm_needs_data = is_rank_in_grid(llm_grid) and (
        is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp"))
    )
    if encoder_needs_data and not llm_needs_data:
        data_iterator = DataIterator(hidden_size, seq_length, encoder_mbs, vocab_size, encoder_name)
    elif llm_needs_data and not encoder_needs_data:
        data_iterator = DataIterator(hidden_size, seq_length, llm_mbs, vocab_size, encoder_name)
    elif encoder_needs_data and llm_needs_data:
        # Colocated: both encoder and LLM on same rank. Use LLM's MBS since
        # the LLM drives the schedule. (encoder_dp == llm_dp when colocated)
        data_iterator = DataIterator(
            hidden_size, seq_length, micro_batch_size, vocab_size, encoder_name
        )

    # Build MultiModuleProcessGroupCollection (reuse pre-created pg_collections)
    module_pgs = {}
    language_model_module_name = None
    if is_rank_in_grid(encoder_grid):
        module_pgs[encoder_name] = vision_pg
    if is_rank_in_grid(llm_grid):
        module_pgs[MIMO_LANGUAGE_MODULE_KEY] = language_pg
        language_model_module_name = MIMO_LANGUAGE_MODULE_KEY

    pg_collection = MultiModuleProcessGroupCollection(
        module_pgs=module_pgs, language_model_module_name=language_model_module_name
    )

    def step_func(data_iterator, model):
        def loss_func(loss_mask, output_tensor):
            if output_tensor is None:
                return torch.tensor(0.0, device='cuda', requires_grad=True), {'loss_reduced': 0.0}

            if isinstance(output_tensor, dict):
                output = output_tensor.get(
                    MIMO_LANGUAGE_MODULE_KEY, next(iter(output_tensor.values()), None)
                )
            else:
                output = output_tensor

            if output is None:
                return torch.tensor(0.0, device='cuda', requires_grad=True), {'loss_reduced': 0.0}

            loss = output.float().sum()
            return loss, {'loss_reduced': loss}

        batch = next(data_iterator) if data_iterator is not None else {'input_ids': None}
        output_tensor, loss_mask = model(**batch)
        return output_tensor, partial(loss_func, loss_mask)

    optimizer.zero_grad()

    try:
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
        success, grad_norm, num_zeros = optimizer.step()
        assert success, "Optimizer step failed"
        assert grad_norm is not None and grad_norm > 0, (
            f"Expected positive grad norm, got {grad_norm}"
        )

        # Verify results on last LLM stage
        if is_rank_in_grid(llm_grid) and is_pp_last_stage(llm_grid.get_pg("pp")):
            assert len(losses) > 0, "Expected losses on last LLM stage"
            for loss_dict in losses:
                assert 'loss_reduced' in loss_dict

        return losses
    finally:
        mimo_model.destroy()


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh requires PyTorch 2.3+",
)
class TestMimo1F1BSchedule:
    """Test MIMO model with 1F1B pipeline schedule."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def test_baseline_2gpu(self):
        """Encoder PP=1, LLM PP=1 on 2 GPUs."""
        if self.world_size != 2:
            pytest.skip(f"Requires 2 GPUs, got {self.world_size}")

        run_mimo_1f1b_test(
            encoder_tp=1,
            encoder_pp=1,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=1,
            llm_pp=1,
            llm_dp=1,
            llm_offset=1,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=4,
        )

    def test_lm_pp3_4gpu(self):
        """Encoder PP=1, LLM PP=3 on 4 GPUs."""
        if self.world_size != 4:
            pytest.skip(f"Requires 4 GPUs, got {self.world_size}")

        run_mimo_1f1b_test(
            encoder_tp=1,
            encoder_pp=1,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=1,
            llm_pp=3,
            llm_dp=1,
            llm_offset=1,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=4,
        )

    def test_encoder_tp2_llm_tp2_pp3_8gpu(self):
        """Encoder TP=2 PP=1, LLM TP=2 PP=3 on 8 GPUs."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        run_mimo_1f1b_test(
            encoder_tp=2,
            encoder_pp=1,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=2,
            llm_pp=3,
            llm_dp=1,
            llm_offset=2,
            hidden_size=256,
            num_layers=3,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=4,
        )

    def test_full_pp_8gpu(self):
        """Encoder PP=2, LLM PP=2 with TP=2 each on 8 GPUs."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        run_mimo_1f1b_test(
            encoder_tp=2,
            encoder_pp=2,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=2,
            llm_pp=2,
            llm_dp=1,
            llm_offset=4,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=4,
        )

    def test_fan_in_dp4_to_dp1_llm_tp2_pp2_8gpu(self):
        """Fan-in 4→1: Encoder DP=4 → LLM TP=2 PP=2 DP=1, on 8 GPUs.

        High fan-in ratio. Each encoder rank processes MBS=1, bridge concatenates
        4 × [img_seq, H] → [4*img_seq, H]. LLM has both TP and PP.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        run_mimo_1f1b_test(
            encoder_tp=1,
            encoder_pp=1,
            encoder_dp=4,
            encoder_offset=0,
            llm_tp=2,
            llm_pp=2,
            llm_dp=1,
            llm_offset=4,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=4,
            num_microbatches=4,
        )

    def test_fan_out_dp1_to_dp4_enc_tp2_pp2_8gpu(self):
        """Fan-out 1→4: Encoder TP=2 PP=2 DP=1 → LLM DP=4, on 8 GPUs.

        Encoder has PP and TP. Bridge fan-out splits encoder output into
        4 parts for 4 LLM DP ranks each with MBS=1.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        run_mimo_1f1b_test(
            encoder_tp=2,
            encoder_pp=2,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=1,
            llm_pp=1,
            llm_dp=4,
            llm_offset=4,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=1,
            num_microbatches=4,
        )

    def test_fan_in_dp2_to_dp1_llm_pp3_8gpu(self):
        """Fan-in 2→1: Encoder DP=2 → LLM TP=2 PP=3, on 8 GPUs.

        Tests fan-in with deep LLM pipeline (PP=3). The 2D tensor goes through
        bridge fan-in then P2P across 3 LLM PP stages.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        run_mimo_1f1b_test(
            encoder_tp=1,
            encoder_pp=1,
            encoder_dp=2,
            encoder_offset=0,
            llm_tp=2,
            llm_pp=3,
            llm_dp=1,
            llm_offset=2,
            hidden_size=256,
            num_layers=3,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=4,
        )

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DDP and the distributed optimizer with the Engram tables sharded over DP x CP (4 GPUs).

Every case trains the same tiny model on EP2 x expert-DP2 in several table layouts -- rows
sharded over EP (the EP layout), over all of dp_cp (``parallel_size=4``, one owner per row), or
over the EP blocks of dp_cp (``parallel_size=2``) -- and compares them::

    torchrun --nproc-per-node 4 -m pytest \\
        tests/unit_tests/models/engram/test_table_sharding_optimizer.py

The loss is linear in the table rows and in the dense output, with integer coefficients, so
every gradient is a small dyadic constant: the layouts must then agree bit for bit on loss,
table gradients and parameters. Clipping is disabled in that comparison because the global
gradient norm is a sum of per-rank partial norms whose rounding depends on how rows are
partitioned; the norm itself is compared to fp32 round-off, and a clipped run to a tolerance.

The cases look rows up with NCCL all-to-alls (``communication_backend='nccl'``). The bitwise
training comparison also runs the sharded layouts with the symmetric transport, whose backward
adds the table gradients row-sparse into DDP's ``main_grad``; that parametrization needs the
transport's libraries (CuTe DSL >= 4.8.1, nccl4py >= 0.6.0 with a matching NCCL runtime and
device bitcode, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) and skips without them.
"""

import os
import shutil
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.models.engram.symmetric_lookup import (
    initialize_engram_transports,
    select_engram_lookup,
)
from megatron.core.models.engram.table_sharding import (
    ENGRAM_PARAMETER_DOMAIN,
    initialize_engram_parallel_state,
    resolve_engram_groups,
    tag_engram_tables,
)
from megatron.core.optimizer import (
    OptimizerConfig,
    get_engram_config_overrides,
    get_megatron_optimizer,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    get_parameter_parallel_domain,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_rank, get_pg_size
from tests.unit_tests.test_utilities import Utils

from ._test_utils import closing_symmetric_transports, symmetric_transport_or_skip

WORLD_SIZE = 4
EP_SIZE = 2

# CI runs the cases on its 4-GPU nodes, which select them by the launch_on_gb200 marker.
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE,
        reason="requires torchrun with exactly four GPUs",
    ),
    pytest.mark.launch_on_gb200,
]

TABLE_SIZES = (11, 13, 5)
EMBEDDING_DIM = 4
DENSE_IN = 8
DENSE_OUT = 6
NUM_MICROBATCHES = 2
LR = 1.0e-2
NO_CLIP = 1.0e9
# parallel_size of the sharded layouts: one owner per row, and the EP layout's groups.
SHARDED_SIZES = (WORLD_SIZE, EP_SIZE)


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(expert_model_parallel_size=EP_SIZE)
    yield
    Utils.destroy_model_parallel()


def _engram_config(parallel_size=None, communication_backend="nccl"):
    """A small configuration; the qwen variant needs no tokenizer artifact.

    The NCCL lookup unless a case names the symmetric backend, the sharded layout's default.
    """
    return EngramConfig(
        global_vocab_sizes=(11, 11),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=1,
        memory_dim=4,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        variant="qwen",
        unigram_vocab_size=32,
        parallel_size=parallel_size,
        communication_backend=communication_backend,
    )


def _pg_collection(parallel_size, backend="nccl"):
    """The MPU groups, with the Engram groups set up on every rank as the model builder does."""
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    initialize_engram_parallel_state(pg_collection, _engram_config(parallel_size, backend))
    return pg_collection


def _dyadic(shape, seed, scale=8.0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randint(-8, 9, shape, generator=generator) / scale).cuda()


class _TableModel(torch.nn.Module):
    """Engram tables plus a replicated dense projection; the loss is linear in both."""

    def __init__(self, parallel_size, pg_collection, backend="nccl"):
        super().__init__()
        config = SimpleNamespace(
            use_cpu_initialization=False,
            params_dtype=torch.bfloat16,
            perform_initialization=False,
            sequence_parallel=False,
            deterministic_mode=False,
        )
        engram_config = _engram_config(parallel_size, backend)
        row_group, replica_group, domain = resolve_engram_groups(pg_collection, engram_config)
        make_lookup = select_engram_lookup(engram_config, row_group)
        self.embedding = EPShardedMultiTableEmbedding(
            config=config,
            table_sizes=TABLE_SIZES,
            embedding_dim=EMBEDDING_DIM,
            init_method=lambda _: None,
            ep_group=row_group,
            tp_group=pg_collection.tp,
            expt_dp_group=replica_group,
        )
        if domain == ENGRAM_PARAMETER_DOMAIN:
            tag_engram_tables(self.embedding)
        # What the Engram layer calls: the embedding itself, or the symmetric lookup bound to it.
        self.lookup = make_lookup(self.embedding)
        self.proj = torch.nn.Linear(
            DENSE_IN, DENSE_OUT, bias=False, dtype=torch.bfloat16, device="cuda"
        )
        self.bias = torch.nn.Parameter(torch.zeros(DENSE_OUT, dtype=torch.bfloat16, device="cuda"))
        with torch.no_grad():
            for table_id, table in enumerate(self.embedding.tables):
                full = _dyadic((TABLE_SIZES[table_id], EMBEDDING_DIM), seed=table_id)
                table.weight.copy_(full[table.row_start : table.row_end])
            self.proj.weight.copy_(_dyadic((DENSE_OUT, DENSE_IN), seed=100))
            self.bias.copy_(_dyadic((DENSE_OUT,), seed=101))

    def forward(self, hash_ids, dense_input, table_coefficients, dense_coefficients):
        rows = self.lookup(hash_ids)
        dense = self.proj(dense_input) + self.bias
        return (rows.float() * table_coefficients).sum() + (
            dense.float() * dense_coefficients
        ).sum()

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state = self.embedding.sharded_state_dict(prefix=f"{prefix}embedding.", metadata=metadata)
        dp_rank = torch.distributed.get_rank()
        for name, parameter in (("proj.weight", self.proj.weight), ("bias", self.bias)):
            state[f"{prefix}{name}"] = ShardedTensor.from_rank_offsets(
                f"{prefix}{name}", parameter, replica_id=(0, 0, dp_rank)
            )
        return state


def _build(
    parallel_size,
    recipe="adam",
    clip_grad=NO_CLIP,
    overlap=False,
    backend="nccl",
    **optimizer_kwargs,
):
    """The tiny model of one table layout (None is the EP layout) with DDP and its optimizer."""
    from megatron.training.training import wrap_model_chunks_with_ddp

    pg_collection = _pg_collection(parallel_size, backend)
    model = _TableModel(parallel_size, pg_collection, backend)
    muon = recipe == "muon"
    model = wrap_model_chunks_with_ddp(
        [model],
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(
            use_distributed_optimizer=True, grad_reduce_in_fp32=True, overlap_grad_reduce=overlap
        ),
        use_layer_wise_distributed_optimizer=muon,
        pg_collection=pg_collection,
    )[0]
    # Once DDP holds the tables in its buffers, as the model builder does; a no-op for NCCL.
    initialize_engram_transports([model], pg_collection, _engram_config(parallel_size, backend))
    config = OptimizerConfig(
        optimizer=recipe,
        lr=LR,
        min_lr=LR,
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=not muon,
        clip_grad=clip_grad,
        **(
            dict(muon_tp_mode="duplicated", use_layer_wise_distributed_optimizer=True)
            if muon
            else {}
        ),
        **optimizer_kwargs,
    )
    optimizer = get_megatron_optimizer(
        config,
        [model],
        config_overrides=get_engram_config_overrides(config, lr_multiplier=5.0, weight_decay=0.0),
        pg_collection=pg_collection,
        use_gloo_process_groups=False,
    )
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=LR,
        max_lr=LR,
        min_lr=LR,
        lr_warmup_steps=0,
        lr_decay_steps=100,
        lr_decay_style="constant",
        start_wd=0.01,
        end_wd=0.01,
        wd_incr_steps=100,
        wd_incr_style="constant",
    )
    return model, optimizer, scheduler


def _inputs(rank, step, microbatch, exact):
    seed = 10000 * step + 100 * rank + microbatch
    generator = torch.Generator().manual_seed(seed)
    num_tokens = 0 if (rank == 3 and microbatch == 0) else 5 + 2 * rank
    hash_ids = torch.stack(
        [torch.randint(0, rows, (num_tokens,), generator=generator) for rows in TABLE_SIZES], dim=-1
    )
    hash_ids[:2] = 0  # hot rows shared by both expert-DP replicas
    shape = (1, num_tokens, len(TABLE_SIZES), EMBEDDING_DIM)
    if exact:
        table_coefficients = torch.randint(-3, 4, shape, generator=generator).float()
    else:
        table_coefficients = torch.randn(shape, generator=generator)
    dense_input = (torch.randint(-4, 5, (3, DENSE_IN), generator=generator) / 4.0).bfloat16()
    dense_coefficients = torch.randint(-2, 3, (3, DENSE_OUT), generator=generator).float()
    return (
        hash_ids.view(1, num_tokens, len(TABLE_SIZES)).cuda(),
        dense_input.cuda(),
        table_coefficients.cuda(),
        dense_coefficients.cuda(),
    )


def _tables(model):
    return model.module.embedding.tables


def _iteration(model, optimizer, scheduler, step, exact=True, overlap=False):
    rank = torch.distributed.get_rank()
    model.zero_grad_buffer()
    optimizer.zero_grad()
    losses = []
    for microbatch in range(NUM_MICROBATCHES):
        last = microbatch == NUM_MICROBATCHES - 1
        with nullcontext() if last else model.no_sync():
            loss = model(*_inputs(rank, step, microbatch, exact))
            loss.backward()
        losses.append(loss.detach())
    # Before the DP reduction: the owner-local sums the backward all-to-all accumulated.
    unsynced = None if overlap else [table.weight.main_grad.clone() for table in _tables(model)]
    model.finish_grad_sync()
    scheduler.step(1)
    update_successful, grad_norm, _ = optimizer.step()
    assert update_successful
    return torch.stack(losses), grad_norm, unsynced


def _assemble(tables, shards):
    """Full tables from row shards, taking every row once (replica 0 of its shard)."""
    full_tables = []
    for table, shard in zip(tables, shards):
        full = torch.zeros(
            (table.global_num_embeddings, EMBEDDING_DIM), dtype=torch.float32, device="cuda"
        )
        if get_pg_rank(table.expt_dp_group) == 0:
            full[table.row_start : table.row_end] = shard.float()
        torch.distributed.all_reduce(full)
        full_tables.append(full)
    return full_tables


def _global_tables(model):
    return _assemble(_tables(model), [table.weight.detach() for table in _tables(model)])


def _dense_parameters(model):
    return [model.module.proj.weight.detach().clone(), model.module.bias.detach().clone()]


def _table_optimizer(optimizer, model):
    """The chained sub-optimizer that owns the table shards."""
    table_params = {table.weight for table in _tables(model)}
    for sub_optimizer in optimizer.chained_optimizers:
        if isinstance(sub_optimizer, DistributedOptimizer) and any(
            param in table_params for param in sub_optimizer.model_param_group_index_map
        ):
            return sub_optimizer
    raise AssertionError("no DistributedOptimizer owns the Engram tables")


@pytest.mark.parametrize("recipe", ["adam", "muon"])
def test_optimizer_routing_and_engram_overrides(recipe):
    for parallel_size, expected_dp_size in ((None, 2), (WORLD_SIZE, 1), (EP_SIZE, 2)):
        model, optimizer, scheduler = _build(parallel_size, recipe)
        domain = "expert" if parallel_size is None else ENGRAM_PARAMETER_DOMAIN
        for table in _tables(model):
            assert get_parameter_parallel_domain(table.weight) == domain
        table_optimizer = _table_optimizer(optimizer, model)
        # The table optimizer comes last in every layout, so its chain position, and with it
        # its fully reshardable checkpoint, carries over between layouts.
        assert optimizer.chained_optimizers[-1] is table_optimizer
        assert table_optimizer.data_parallel_group.size() == expected_dp_size
        # Gradient statistics reduce over the whole world in every layout.
        assert table_optimizer.get_grad_stats_parallel_group().size() == WORLD_SIZE
        table_params = {table.weight for table in _tables(model)}
        owned = set(table_optimizer.model_param_group_index_map)
        if parallel_size == WORLD_SIZE:
            # The owner keeps every table shard whole, and nothing else.
            assert owned == table_params
        else:
            # Byte-level ZeRO over the replica pair: this rank holds part of the tables.
            assert owned and owned <= table_params
        scheduler.step(1)
        for param in owned:
            group_index, _ = table_optimizer.model_param_group_index_map[param]
            group = table_optimizer.optimizer.param_groups[group_index]
            assert get_parameter_parallel_domain(group) == domain
            # The checkpoint identifier calls the tables expert-parallel in every layout.
            assert group["is_expert_parallel"]
            assert group["lr"] == pytest.approx(5.0 * LR)
            assert group["weight_decay"] == 0.0


@pytest.mark.parametrize("backend", ["nccl", pytest.param("symmetric", marks=pytest.mark.internal)])
@pytest.mark.parametrize("parallel_size", SHARDED_SIZES)
@pytest.mark.parametrize("recipe", ["adam", "muon"])
@pytest.mark.parametrize("overlap", [False, True], ids=["sync", "overlap"])
def test_training_matches_ep_layout_bitwise(backend, parallel_size, recipe, overlap):
    """The sharded layout, with either backend, trains bit for bit like the EP layout's NCCL
    lookup: with dyadic inputs, every sum is exact in any order."""
    if backend == "symmetric":
        symmetric_transport_or_skip()
    with closing_symmetric_transports():
        _check_training_matches_ep_layout(backend, parallel_size, recipe, overlap)


def _check_training_matches_ep_layout(backend, parallel_size, recipe, overlap):
    runs = {}
    for layout, layout_backend in ((None, "nccl"), (parallel_size, backend)):
        torch.manual_seed(0)
        model, optimizer, scheduler = _build(
            layout, recipe, overlap=overlap, backend=layout_backend
        )
        if layout_backend == "symmetric":
            assert all(hasattr(table.weight, "main_grad") for table in _tables(model))
        losses, grad_norms, unsynced = [], [], []
        for step in range(3):
            loss, grad_norm, step_unsynced = _iteration(
                model, optimizer, scheduler, step, overlap=overlap
            )
            losses.append(loss)
            grad_norms.append(grad_norm)
            if step_unsynced is not None:
                replica_group = _tables(model)[0].expt_dp_group
                if get_pg_size(replica_group) > 1:
                    # DDP sums the replicas of each row shard; reproduce that sum here.
                    for gradient in step_unsynced:
                        torch.distributed.all_reduce(gradient, group=replica_group)
                unsynced.append(_assemble(_tables(model), step_unsynced))
        runs[layout] = (
            torch.stack(losses),
            grad_norms,
            unsynced,
            _global_tables(model),
            _dense_parameters(model),
        )

    loss_ep, norm_ep, grad_ep, tables_ep, dense_ep = runs[None]
    loss_sh, norm_sh, grad_sh, tables_sh, dense_sh = runs[parallel_size]
    assert torch.equal(loss_ep, loss_sh)
    for step_ep, step_sh in zip(grad_ep, grad_sh):
        for table_ep, table_sh in zip(step_ep, step_sh):
            assert torch.equal(table_ep, table_sh)
    for value_ep, value_sh in zip(norm_ep, norm_sh):
        assert float(value_sh) == pytest.approx(float(value_ep), rel=1e-6, abs=0.0)
    for table_ep, table_sh in zip(tables_ep, tables_sh):
        assert torch.equal(table_ep, table_sh)
    for value_ep, value_sh in zip(dense_ep, dense_sh):
        assert torch.equal(value_ep, value_sh)


def test_grad_norm_counts_each_table_row_once():
    """The table's share of the global norm equals the norm of the full-table gradient."""
    model, optimizer, scheduler = _build(WORLD_SIZE, "adam")
    model.zero_grad_buffer()
    optimizer.zero_grad()
    rank = torch.distributed.get_rank()
    for microbatch in range(NUM_MICROBATCHES):
        with nullcontext() if microbatch == NUM_MICROBATCHES - 1 else model.no_sync():
            model(*_inputs(rank, 0, microbatch, exact=True)).backward()
    model.finish_grad_sync()
    # With one owner per row the synced main_grad is the complete, scaled table gradient.
    synced = _assemble(_tables(model), [table.weight.main_grad for table in _tables(model)])
    table_norm_squared = sum(float(gradient.square().sum()) for gradient in synced)
    # The dense main_grad is reduce-scattered, so rebuild its exact value from the inputs.
    weight_grad = torch.zeros((DENSE_OUT, DENSE_IN), device="cuda")
    bias_grad = torch.zeros((DENSE_OUT,), device="cuda")
    for source_rank in range(WORLD_SIZE):
        for microbatch in range(NUM_MICROBATCHES):
            _, dense_input, _, dense_coefficients = _inputs(source_rank, 0, microbatch, True)
            weight_grad += dense_coefficients.t() @ dense_input.float()
            bias_grad += dense_coefficients.sum(0)
    dense_norm_squared = float(
        (weight_grad / WORLD_SIZE).square().sum() + (bias_grad / WORLD_SIZE).square().sum()
    )
    optimizer.prepare_grads()
    grad_norm = float(optimizer.get_grad_norm())
    assert grad_norm**2 == pytest.approx(table_norm_squared + dense_norm_squared, rel=1e-6)


@pytest.mark.parametrize("recipe", ["adam", "muon"])
def test_clipped_random_training_matches_ep_layout(recipe):
    results = {}
    for layout in (None, WORLD_SIZE):
        model, optimizer, scheduler = _build(layout, recipe, clip_grad=1.0)
        norms = [
            float(_iteration(model, optimizer, scheduler, step, exact=False)[1])
            for step in range(3)
        ]
        results[layout] = (norms, _global_tables(model))
    for norm_ep, norm_sh in zip(results[None][0], results[WORLD_SIZE][0]):
        assert norm_sh == pytest.approx(norm_ep, rel=1e-3)
    for table_ep, table_sh in zip(results[None][1], results[WORLD_SIZE][1]):
        torch.testing.assert_close(table_sh, table_ep, rtol=2e-2, atol=2e-3)


def _shared_checkpoint_dir():
    path = [
        (
            tempfile.mkdtemp(prefix="engram-table-sharding-opt-")
            if torch.distributed.get_rank() == 0
            else None
        )
    ]
    torch.distributed.broadcast_object_list(path, src=0)
    return path[0]


def _remove_checkpoint_dir(checkpoint_dir):
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


# The metadata training writes with --dist-ckpt-optim-fully-reshardable (see
# megatron/training/checkpointing.py): optimizer state in the model's global row space, so it
# loads into either table layout.
_CHECKPOINT_METADATA = {
    "distrib_optim_sharding_type": "fully_reshardable",
    "chained_optim_avoid_prefix": True,
}


@pytest.mark.parametrize("recipe", ["adam", "muon"])
@pytest.mark.parametrize("source, target", [(None, WORLD_SIZE), (WORLD_SIZE, None)])
def test_checkpoint_with_optimizer_state_loads_across_layouts(recipe, source, target):
    """Save after two steps in one layout, resume in the other, and match the uninterrupted run.

    The tiny model has no other expert-parallel Adam parameters, so the table optimizer sits at
    the same chain position in both layouts, as in the Muon recipe of the real model.
    """
    reference, reference_optimizer, reference_scheduler = _build(target, recipe)
    for step in range(3):
        _iteration(reference, reference_optimizer, reference_scheduler, step)

    saved, saved_optimizer, saved_scheduler = _build(source, recipe)
    for step in range(2):
        _iteration(saved, saved_optimizer, saved_scheduler, step)
    checkpoint_dir = _shared_checkpoint_dir()
    model_state = saved.module.sharded_state_dict()
    save(
        {
            "model": model_state,
            "optimizer": saved_optimizer.sharded_state_dict(
                model_state, metadata=_CHECKPOINT_METADATA
            ),
        },
        checkpoint_dir,
    )
    torch.distributed.barrier()

    resumed, resumed_optimizer, resumed_scheduler = _build(target, recipe)
    model_state = resumed.module.sharded_state_dict()
    loaded = load(
        {
            "model": model_state,
            "optimizer": resumed_optimizer.sharded_state_dict(
                model_state, is_loading=True, metadata=_CHECKPOINT_METADATA
            ),
        },
        checkpoint_dir,
    )
    resumed.module.load_state_dict(loaded["model"])
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    for _ in range(2):
        resumed_scheduler.step(1)
    _iteration(resumed, resumed_optimizer, resumed_scheduler, step=2)

    for table_reference, table_resumed in zip(_global_tables(reference), _global_tables(resumed)):
        assert torch.equal(table_reference, table_resumed)
    for value_reference, value_resumed in zip(
        _dense_parameters(reference), _dense_parameters(resumed)
    ):
        assert torch.equal(value_reference, value_resumed)
    _remove_checkpoint_dir(checkpoint_dir)


def _communicator_state(group):
    """Whether ``group`` created its NCCL communicator, and how many collectives it ran.

    Both probes are private torch APIs. A torch without them fails the test instead of
    skipping it: this is the only check that a single-rank replica group never gets a
    communicator, so a renamed API must be noticed and the probe updated.
    """
    backend = group._get_backend(torch.device("cuda"))
    for owner, probe in ((backend, "_is_initialized"), (group, "_get_sequence_number_for_group")):
        if not hasattr(owner, probe):
            pytest.fail(f"this torch has no {type(owner).__name__}.{probe}; update the probe")
    return backend._is_initialized(), group._get_sequence_number_for_group()


@pytest.mark.parametrize("recipe", ["adam", "muon"])
@pytest.mark.parametrize("overlap", [False, True], ids=["sync", "overlap"])
def test_single_rank_replica_group_never_creates_a_communicator(recipe, overlap):
    """With one owner per row, nothing runs on the single-rank replica group.

    Two steps, a parameter broadcast, and checkpoint saves and loads in both optimizer formats
    leave its NCCL communicator uncreated, which saves the memory of a single-rank
    communicator.
    """
    model, optimizer, scheduler = _build(WORLD_SIZE, recipe, overlap=overlap)
    row_group = model.module.embedding.ep_group
    replica_group = _tables(model)[0].expt_dp_group
    assert replica_group.size() == 1
    assert model.parameter_replica_groups[ENGRAM_PARAMETER_DOMAIN] is replica_group
    for step in range(2):
        _iteration(model, optimizer, scheduler, step, overlap=overlap)
    model.broadcast_params()
    for metadata in (_CHECKPOINT_METADATA, {"distrib_optim_sharding_type": "dp_reshardable"}):
        checkpoint_dir = _shared_checkpoint_dir()
        model_state = model.module.sharded_state_dict()
        save(
            {
                "model": model_state,
                "optimizer": optimizer.sharded_state_dict(model_state, metadata=metadata),
            },
            checkpoint_dir,
        )
        torch.distributed.barrier()
        # The same layout reuses the same groups, so the load runs on this replica group too.
        resumed, resumed_optimizer, _ = _build(WORLD_SIZE, recipe, overlap=overlap)
        assert _tables(resumed)[0].expt_dp_group is replica_group
        model_state = resumed.module.sharded_state_dict()
        loaded = load(
            {
                "model": model_state,
                "optimizer": resumed_optimizer.sharded_state_dict(
                    model_state, is_loading=True, metadata=metadata
                ),
            },
            checkpoint_dir,
        )
        resumed.module.load_state_dict(loaded["model"])
        resumed_optimizer.load_state_dict(loaded["optimizer"])
        for table_saved, table_loaded in zip(_global_tables(model), _global_tables(resumed)):
            assert torch.equal(table_saved, table_loaded)
        _remove_checkpoint_dir(checkpoint_dir)
    torch.cuda.synchronize()

    initialized, num_collectives = _communicator_state(replica_group)
    assert not initialized, "a communicator was created for the single-rank replica group"
    assert num_collectives == 0, "a collective ran on the single-rank replica group"
    # The probe does see collectives: the lookups ran on the row group, dp_cp itself.
    initialized, num_collectives = _communicator_state(row_group)
    assert initialized and num_collectives > 0

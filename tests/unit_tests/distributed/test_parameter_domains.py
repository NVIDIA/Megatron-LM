# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parameter replica domains in DDP and the distributed optimizer.

A table sharded by rows over part of dp_cp (its shard group) and replicated over the rest (its
replica group) is an extra parameter domain: its parameters get DDP buffers of their own, reduced
over the replica group, and a DistributedOptimizer of their own, last in the chain. Here the
lookup is emulated by gathering the rows over the shard group; its backward sums every rank's
gradient into each row's owner, as a sharded lookup's exchange does.

The loss is linear in the table and in a dense projection, with small integer coefficients, so
every gradient is a dyadic constant and reductions are exact in any order: the domain must then
train bit for bit like the same table held as a dense parameter.

The params norm and the training wrappers that lay out and build DDP are checked for domains
too.

The collective tests need a world size that is a multiple of four (one GPU per rank)::

    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/distributed/test_parameter_domains.py
"""

import inspect
import math
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed import distributed_data_parallel as ddp_module
from megatron.core.distributed.param_and_grad_buffer import (
    _CompletedCollectiveHandle,
    _ParamAndGradBuffer,
    get_all_bucket_groups,
    get_all_buffers,
    group_params_for_buffers,
)
from megatron.core.optimizer import (
    HAVE_EMERGING_OPTIMIZERS,
    ChainedOptimizer,
    OptimizerConfig,
    get_megatron_optimizer,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import get_param_group_identifier_tuple
from megatron.core.optimizer.param_layout import BufferKey
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    create_row_parallel_groups,
    replica_group_shard,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import check_param_hashes_across_dp_replicas, get_pg_rank
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

_DOMAIN = 'table'
# The rows split evenly over 1, 2, 4 or 8 shards. With four or eight ranks, every rank holds
# part of each buffer whose optimizer state the tests save, also with replica groups of two.
_ROWS, _DIM = 24, 11
_IN, _OUT = 16, 8
_MICROBATCHES = 2
_LR = 1.0e-2
# Large enough never to clip: clipping would scale by the gradient norm, whose rounding depends
# on how the rows are partitioned.
_NO_CLIP = 1.0e9
_DP_RESHARDABLE = {'distrib_optim_sharding_type': 'dp_reshardable'}
# Replica groups spanning dp_cp (rows not sharded), of two ranks, or of one rank.
_REPLICA_IDS = ['replicas_dp_cp', 'replicas_2', 'replicas_1']


def _require_ranks() -> None:
    if not torch.cuda.is_available() or Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip('needs a world size that is a multiple of four, one GPU per rank')


def _ranks(group):
    return torch.distributed.get_process_group_ranks(group)


def _dyadic(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-8, 9, shape, generator=generator) / 8.0


def _inputs(rank, step, microbatch):
    """Inputs of one microbatch on one rank: (dense input, dense and row coefficients)."""
    generator = torch.Generator().manual_seed(10000 * step + 100 * rank + microbatch)
    dense_input = torch.randint(-4, 5, (3, _IN), generator=generator) / 4.0
    dense_coefficients = torch.randint(-2, 3, (3, _OUT), generator=generator).float()
    row_coefficients = torch.randint(-3, 4, (_ROWS, _DIM), generator=generator).float()
    # Like a lookup, a rank reads only some of the rows; the others get no gradient from it.
    row_coefficients[torch.rand(_ROWS, generator=generator) < 0.5] = 0.0
    return dense_input, dense_coefficients, row_coefficients


def _cuda_inputs(rank, step, microbatch):
    dense_input, dense_coefficients, row_coefficients = _inputs(rank, step, microbatch)
    return dense_input.bfloat16().cuda(), dense_coefficients.cuda(), row_coefficients.cuda()


class _GatherRows(torch.autograd.Function):
    """Gather the row shards of a table over its shard group.

    The backward sums every rank's gradient of the full table into the owner of each row, as the
    exchange of a lookup sharded by rows does.
    """

    @staticmethod
    def forward(ctx, shard, group):
        ctx.group = group
        rows = shard.new_empty((shard.shape[0] * group.size(), *shard.shape[1:]))
        torch.distributed.all_gather_into_tensor(rows, shard.contiguous(), group=group)
        return rows

    @staticmethod
    def backward(ctx, grad_rows):
        grad_shard = grad_rows.new_empty(
            (grad_rows.shape[0] // ctx.group.size(), *grad_rows.shape[1:])
        )
        torch.distributed.reduce_scatter_tensor(grad_shard, grad_rows.contiguous(), group=ctx.group)
        return grad_shard, None


class _TableModel(torch.nn.Module):
    """A dense projection and a table, plus an optional expert-parallel weight.

    With a shard group the table holds this rank's row shard and belongs to the extra domain;
    without one it is a dense parameter holding every row: the reference.
    """

    def __init__(self, shard_group=None, with_table=True, with_expert=False):
        super().__init__()
        self.shard_group = shard_group
        self.proj = torch.nn.Linear(_IN, _OUT, bias=False, dtype=torch.bfloat16, device='cuda')
        with torch.no_grad():
            self.proj.weight.copy_(_dyadic((_OUT, _IN), seed=0))
        self.row_start, self.row_end = 0, _ROWS
        self.table = None
        if with_table:
            if shard_group is not None:
                rows_per_shard = _ROWS // shard_group.size()
                self.row_start = shard_group.rank() * rows_per_shard
                self.row_end = self.row_start + rows_per_shard
            rows = _dyadic((_ROWS, _DIM), seed=1)[self.row_start : self.row_end]
            self.table = torch.nn.Parameter(rows.to(device='cuda', dtype=torch.bfloat16))
            if shard_group is not None:
                # The tags of a table sharded by rows: not replicated over the dense DP group,
                # and replicated over the replica group of its domain.
                self.table.allreduce = False
                self.table.parallel_domain = _DOMAIN
        self.expert = None
        if with_expert:
            weight = _dyadic((_OUT, _IN), seed=2).to(device='cuda', dtype=torch.bfloat16)
            self.expert = torch.nn.Parameter(weight)
            self.expert.allreduce = False

    def forward(self, dense_input, dense_coefficients, row_coefficients):
        loss = (self.proj(dense_input).float() * dense_coefficients).sum()
        if self.table is not None:
            rows = self.table
            if self.shard_group is not None and self.shard_group.size() > 1:
                rows = _GatherRows.apply(rows, self.shard_group)
            loss = loss + (rows.float() * row_coefficients).sum()
        if self.expert is not None:
            expert = torch.nn.functional.linear(dense_input, self.expert)
            loss = loss + (expert.float() * dense_coefficients).sum()
        return loss


def _process_groups(replica_size=None):
    """MPU groups, plus the table domain's groups when ``replica_size`` is given.

    ``replica_size`` is the size of the replica groups, or ``'dp_cp'`` for a table whose rows are
    not sharded at all. Creating the groups is a collective over the world.
    """
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    if replica_size is None:
        return pg_collection, None, None
    dp_cp_size = pg_collection.dp_cp.size()
    num_replicas = dp_cp_size if replica_size == 'dp_cp' else replica_size
    shard_group, replica_group, _ = create_row_parallel_groups(
        pg_collection.dp_cp,
        dp_cp_size // num_replicas,
        world_group=torch.distributed.group.WORLD,
        name='TEST_TABLE',
    )
    pg_collection.parameter_replica_groups = {_DOMAIN: replica_group}
    return pg_collection, shard_group, replica_group


def test_replica_group_shard_does_not_depend_on_the_data_parallel_size():
    """An extra domain's DistributedOptimizer is named by its replica group's row shard, which
    stays when the data-parallel size changes at a fixed number of shards: stage 1 of PP2 with
    two row shards holds ranks [2, 3] at DP2 and [4, 5, 6, 7] at DP4."""
    assert replica_group_shard([2, 3], [2]) == (0, 2)
    assert replica_group_shard([2, 3], [3]) == (1, 2)
    assert replica_group_shard([4, 5, 6, 7], [4, 6]) == (0, 2)
    assert replica_group_shard([4, 5, 6, 7], [5, 7]) == (1, 2)
    # One row shard: the replica group is the whole dp_cp group.
    assert replica_group_shard([4, 5, 6, 7], [4, 5, 6, 7]) == (0, 1)
    # Contiguous blocks, a size that does not divide the parent, and ranks outside it.
    for replica_ranks in ([4, 5], [4, 5, 6], [0, 2]):
        with pytest.raises(ValueError, match='strided slice'):
            replica_group_shard([4, 5, 6, 7], replica_ranks)


def _optimizer_index(pg_collection, replica_group):
    """What names a domain's DistributedOptimizer: model-parallel rank and row shard."""
    num_shards = pg_collection.dp_cp.size() // replica_group.size()
    return (
        get_pg_rank(pg_collection.mp) * num_shards + get_pg_rank(pg_collection.dp_cp) % num_shards
    )


def _ddp_config(**kwargs):
    kwargs.setdefault('use_distributed_optimizer', True)
    return DistributedDataParallelConfig(grad_reduce_in_fp32=True, **kwargs)


def _wrap(module, pg_collection, ddp_config, config=None):
    """Wrap ``module`` in DDP, laying out each domain's buffers for its replica group."""
    full_param_layout = None
    if ddp_config.use_distributed_optimizer:
        full_param_layout = DistributedOptimizer.compute_full_param_layout(
            [param for param in module.parameters() if param.requires_grad],
            ddp_config.bucket_size,
            pg_collection.dp_cp.size(),
            ddp_config,
            expert_data_parallel_world_size=pg_collection.expt_dp.size(),
            parallel_group_sizes={
                domain: group.size()
                for domain, group in pg_collection.parameter_replica_groups.items()
            },
        )
    model = DistributedDataParallel(
        config or TransformerConfig(num_attention_heads=1, num_layers=1),
        ddp_config,
        module,
        pg_collection=pg_collection,
        full_param_layout=full_param_layout,
    )
    model.broadcast_params()
    return model


def _build(pg_collection, shard_group, ddp_config, with_table=True):
    """A DDP-wrapped model and its Adam distributed optimizer."""
    model = _wrap(_TableModel(shard_group, with_table=with_table), pg_collection, ddp_config)
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer='adam',
            lr=_LR,
            bf16=True,
            use_distributed_optimizer=True,
            clip_grad=_NO_CLIP,
            overlap_param_gather=ddp_config.overlap_param_gather,
        ),
        [model],
        pg_collection=pg_collection,
        use_gloo_process_groups=False,
    )
    return model, optimizer


def _backward(model, step):
    """Forward and backward of every microbatch of ``step``; returns the losses."""
    rank = torch.distributed.get_rank()
    losses = []
    for microbatch in range(_MICROBATCHES):
        last = microbatch == _MICROBATCHES - 1
        with nullcontext() if last else model.no_sync():
            loss = model(*_cuda_inputs(rank, step, microbatch))
            loss.backward()
        losses.append(loss.detach())
    return losses


def _train(model, optimizer, steps, first_step=0):
    """Train for ``steps`` steps; returns the losses and gradient norms."""
    losses, grad_norms = [], []
    for step in range(first_step, first_step + steps):
        model.zero_grad_buffer()
        optimizer.zero_grad()
        losses += _backward(model, step)
        model.finish_grad_sync()
        update_successful, grad_norm, _ = optimizer.step()
        assert update_successful
        # Depending on the optimizer path, the norm comes back as a float or a 1-element tensor.
        grad_norms.append(float(grad_norm))
    if model.ddp_config.overlap_param_gather:
        # The gathers of the last update would otherwise wait for the next forward.
        model.start_param_sync(force_sync=True)
    return torch.stack(losses), grad_norms


def _expected_grads(ranks, step, scale):
    """The reduced gradients of the full table and of the projection over ``ranks``."""
    table = torch.zeros(_ROWS, _DIM)
    proj = torch.zeros(_OUT, _IN)
    for rank in ranks:
        for microbatch in range(_MICROBATCHES):
            dense_input, dense_coefficients, row_coefficients = _inputs(rank, step, microbatch)
            table += row_coefficients
            proj += dense_coefficients.t() @ dense_input
    return table * scale, proj * scale


def _reduced_range(buffer, param, use_distributed_optimizer):
    """The range of ``buffer`` within ``param`` that this rank holds reduced.

    The distributed optimizer reduce-scatters, so a rank holds only its shard of each bucket.
    Returns (start of the param, start of the range, end of the range).
    """
    start, end, bucket_id = buffer.param_index_map[param]
    low, high = start, end
    if use_distributed_optimizer:
        bucket = buffer.buckets[bucket_id]
        group = buffer.data_parallel_group
        shard_numel = bucket.grad_data.numel() // group.size()
        low = max(start, bucket.offset + group.rank() * shard_numel)
        high = min(end, bucket.offset + (group.rank() + 1) * shard_numel)
    return start, low, max(low, high)


def _assert_reduced(grad_data, reduced_range, expected):
    start, low, high = reduced_range
    expected = expected.reshape(-1)[low - start : high - start]
    torch.testing.assert_close(grad_data[low:high], expected.to(grad_data), rtol=0, atol=0)


def _owned_numel(distopt):
    """Number of elements of the main params this rank's shard of ``distopt`` holds."""
    return sum(
        param.numel() for group in distopt.optimizer.param_groups for param in group['params']
    )


# The tests run every collective before asserting on values that may differ between ranks: a
# rank that failed early would leave the others waiting in the next collective.


def _world_sum(value):
    total = torch.tensor([value], dtype=torch.int64, device='cuda')
    torch.distributed.all_reduce(total)
    return int(total.item())


def _from_every_rank(value):
    values = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(values, value)
    return values


def _communicator_state(group):
    """Whether ``group`` created its NCCL communicator, and how many collectives it ran.

    These are private torch APIs: without them the test cannot tell, so it skips. Callers probe
    after their last collective, and every rank runs the same torch, so all ranks skip together.
    """
    try:
        backend = group._get_backend(torch.device('cuda'))
        return backend._is_initialized(), group._get_sequence_number_for_group()
    except (AttributeError, RuntimeError) as error:
        pytest.skip(f'this torch cannot probe NCCL communicators: {error!r}')


def _assert_never_communicated(group):
    """``group`` never ran a collective, so it never created an NCCL communicator."""
    initialized, num_collectives = _communicator_state(group)
    assert not initialized, 'a communicator was created for the group'
    assert num_collectives == 0, 'a collective ran on the group'


def _spy_on_param_sync(bucket_groups):
    """Record each param sync of ``bucket_groups``, with the handle it left, and each
    post-all-gather processing."""
    events = []
    for bucket_group in bucket_groups:

        def start_param_sync(
            *args, bucket_group=bucket_group, start=bucket_group.start_param_sync, **kwargs
        ):
            start(*args, **kwargs)
            events.append(('start', type(bucket_group.param_gather_handle)))

        def post_param_sync(post=bucket_group._post_param_sync):
            events.append(('post',))
            post()

        bucket_group.start_param_sync = start_param_sync
        bucket_group._post_param_sync = post_param_sync
    return events


def _checkpoint_keys(optimizer):
    """Keys of the optimizer's dp_reshardable checkpoint."""
    state = optimizer.sharded_state_dict({}, metadata=_DP_RESHARDABLE)
    return sorted(
        {
            value.key
            for value in nested_values(state)
            if isinstance(value, (ShardedObject, ShardedTensor))
        }
    )


def _named_param(*shape, dtype=torch.bfloat16, domain=None, expert=False, layer_wise=False):
    param = torch.nn.Parameter(torch.zeros(*shape, dtype=dtype))
    if expert or domain is not None:
        param.allreduce = False
    if domain is not None:
        param.parallel_domain = domain
    if layer_wise:
        param.is_managed_by_layer_wise_optimizer = True
    return param


class TestBufferKeys:
    """Buffer grouping and layouts of extra domains (no torch.distributed needed)."""

    def test_params_are_grouped_by_domain(self):
        """Each domain gets buffers of its own, split by dtype as usual.

        A table keeps ``allreduce=False`` beside its domain; the domain decides, so its key is not
        expert-parallel. Positions are counted within each domain.
        """
        params = {
            'dense': _named_param(4, 4),
            'expert': _named_param(4, 4, expert=True),
            'table': _named_param(6, 4, domain='table'),
            'table_fp32': _named_param(6, 4, dtype=torch.float32, domain='table'),
            'table_2': _named_param(6, 4, domain='table'),
            'other': _named_param(6, 4, domain='other'),
        }
        names = {param: name for name, param in params.items()}

        groups = group_params_for_buffers(list(params.values()), grad_reduce_in_fp32=True)

        bf16, fp32 = torch.bfloat16, torch.float32
        assert {
            key: ([names[param] for param in group_params], indices)
            for key, (group_params, indices) in groups.items()
        } == {
            BufferKey(bf16, fp32, False): (['dense'], [0]),
            BufferKey(bf16, fp32, True): (['expert'], [0]),
            BufferKey(bf16, fp32, False, False, 'table'): (['table', 'table_2'], [0, 1]),
            BufferKey(fp32, fp32, False, False, 'table'): (['table_fp32'], [0]),
            BufferKey(bf16, fp32, False, False, 'other'): (['other'], [0]),
        }
        assert {key.domain for key in groups} == {'dense', 'expert', 'table', 'other'}

    def test_dense_and_expert_keys_are_unchanged(self):
        """Without a domain a key equals, and hashes like, the key built before domains."""
        for is_expert_parallel, domain in ((False, 'dense'), (True, 'expert')):
            key = BufferKey(torch.bfloat16, torch.float32, is_expert_parallel)
            explicit = BufferKey(torch.bfloat16, torch.float32, is_expert_parallel, False, None)
            assert key == explicit and hash(key) == hash(explicit)
            assert key.parallel_domain is None and key.domain == domain

    def test_layout_shards_each_domain_over_its_replica_group(self):
        """A domain's buckets are padded for, and sharded over, its own replica group."""
        dense, expert = _named_param(5, 7), _named_param(5, 7, expert=True)
        table = _named_param(5, 7, domain='table')
        ddp_config = _ddp_config()

        layout = DistributedOptimizer.compute_full_param_layout(
            [dense, expert, table],
            None,
            4,
            ddp_config,
            expert_data_parallel_world_size=2,
            parallel_group_sizes={'table': 3},
        )

        layouts = {key.domain: per_buffer for key, per_buffer in layout.layouts.items()}
        assert {
            domain: per_buffer.num_optimizer_shards for domain, per_buffer in layouts.items()
        } == {'dense': 4, 'expert': 2, 'table': 3}
        for domain, group_size in (('dense', 4), ('expert', 2), ('table', 3)):
            for _, bucket_end in layouts[domain].bucket_indices:
                assert bucket_end % math.lcm(group_size, 128) == 0

        with pytest.raises(ValueError, match='parallel_group_sizes'):
            DistributedOptimizer.compute_full_param_layout(
                [dense, table], None, 4, ddp_config, parallel_group_sizes={'other': 3}
            )

    def test_layer_wise_layout_leaves_domains_to_the_distributed_optimizer(self):
        """LayerWise never owns a domain: its params get DistributedOptimizer's layout."""
        dense = _named_param(8, 8, layer_wise=True)
        table = _named_param(5, 7, domain='table')
        ddp_config = _ddp_config()

        layout = LayerWiseDistributedOptimizer.compute_full_param_layout(
            [dense, table], None, 4, ddp_config, parallel_group_sizes={'table': 2}
        )
        [table_layout] = [
            per_buffer for key, per_buffer in layout.layouts.items() if key.domain == 'table'
        ]
        assert table_layout.num_optimizer_shards == 2

        managed_table = _named_param(5, 7, domain='table', layer_wise=True)
        with pytest.raises(ValueError, match='cannot be managed'):
            LayerWiseDistributedOptimizer.compute_full_param_layout(
                [dense, managed_table], None, 4, ddp_config, parallel_group_sizes={'table': 2}
            )


class TestBufferLists:
    """The buffers and bucket groups of every domain (no torch.distributed needed)."""

    def test_chunks_without_extra_domains(self):
        """Chunks other than DistributedDataParallel, mocks included, have no extra domain."""
        chunk = mock.MagicMock()
        chunk.buffers, chunk.expert_parallel_buffers = ['dense'], ['expert']
        chunk.bucket_groups, chunk.expert_parallel_bucket_groups = ['dense'], ['expert']
        assert get_all_buffers(chunk) == ['dense', 'expert']
        assert get_all_bucket_groups(chunk) == ['dense', 'expert']
        assert get_all_bucket_groups(SimpleNamespace()) == []

    def test_ddp_built_without_init(self):
        """A DistributedDataParallel built without __init__, as test doubles are, has no
        extra domain."""
        ddp = object.__new__(DistributedDataParallel)
        ddp.buffers, ddp.expert_parallel_buffers = ['dense'], ['expert']
        ddp.bucket_groups, ddp.expert_parallel_bucket_groups = ['dense'], ['expert']
        assert ddp.buffers_by_domain == {'dense': ['dense'], 'expert': ['expert']}
        assert ddp.bucket_groups_by_domain == {'dense': ['dense'], 'expert': ['expert']}
        assert ddp.all_buffers == get_all_buffers(ddp) == ['dense', 'expert']
        assert ddp.all_bucket_groups == get_all_bucket_groups(ddp) == ['dense', 'expert']


class TestGradSync:
    """DDP reduces a domain over its replica group and scales it to the usual dp_cp mean."""

    @pytest.fixture(autouse=True)
    def teardown(self):
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('replica_size', ['dp_cp', 2, 1], ids=_REPLICA_IDS)
    @pytest.mark.parametrize('reduction', ['sum', 'average_in_collective', 'per_token_loss'])
    @pytest.mark.parametrize(
        'use_distributed_optimizer', [False, True], ids=['all_reduce', 'reduce_scatter']
    )
    @pytest.mark.parametrize('overlap_grad_reduce', [False, True], ids=['sync', 'overlap'])
    def test_grad_sync(
        self, replica_size, reduction, use_distributed_optimizer, overlap_grad_reduce
    ):
        _require_ranks()
        Utils.initialize_model_parallel()
        pg_collection, shard_group, replica_group = _process_groups(replica_size)
        dp_cp = pg_collection.dp_cp
        per_token_loss = reduction == 'per_token_loss'
        ddp_config = _ddp_config(
            use_distributed_optimizer=use_distributed_optimizer,
            overlap_grad_reduce=overlap_grad_reduce,
            average_in_collective=reduction == 'average_in_collective',
        )
        config = TransformerConfig(
            num_attention_heads=1, num_layers=1, calculate_per_token_loss=per_token_loss
        )
        model = _wrap(_TableModel(shard_group), pg_collection, ddp_config, config)
        module = model.module

        # The domain has a buffer and bucket groups of its own, over its replica group.
        assert model.parameter_replica_groups == {
            'dense': model.dp_cp_group,
            'expert': model.expt_dp_group,
            _DOMAIN: replica_group,
        }
        assert list(model.buffers_by_domain) == ['dense', 'expert', _DOMAIN]
        assert model.buffers_by_domain['dense'] is model.buffers
        assert model.buffers_by_domain['expert'] is model.expert_parallel_buffers
        [table_buffer] = model.buffers_by_domain[_DOMAIN]
        assert list(table_buffer.param_index_map) == [module.table]
        assert table_buffer.data_parallel_group is replica_group
        assert model.all_buffers == model.buffers + model.expert_parallel_buffers + [table_buffer]
        table_bucket_groups = model.bucket_groups_by_domain[_DOMAIN]
        assert model.param_to_bucket_group[module.table] in table_bucket_groups
        assert model.all_bucket_groups == (
            model.bucket_groups + model.expert_parallel_bucket_groups + table_bucket_groups
        )
        # Only a single-rank replica group makes the collectives identities.
        identity = replica_group.size() == 1
        assert all(group.collectives_are_identity == identity for group in table_bucket_groups)
        assert not any(group.collectives_are_identity for group in model.bucket_groups)

        # Scaling: the reduction must give the dp_cp mean (or sum, under per-token loss).
        if per_token_loss:
            scale, table_scaling = 1.0, 1.0
        elif ddp_config.average_in_collective:
            scale, table_scaling = 1.0 / dp_cp.size(), replica_group.size() / dp_cp.size()
        else:
            scale = table_scaling = 1.0 / dp_cp.size()
        assert table_buffer.gradient_scaling_factor == table_scaling
        [dense_buffer] = model.buffers

        reduced, handles = [], []
        for step in range(2):
            model.zero_grad_buffer()
            _backward(model, step)
            # With overlap, the last backward of the second step dispatched every reduction.
            handles.append([group.grad_reduce_handle for group in table_bucket_groups])
            model.finish_grad_sync()
            reduced.append((table_buffer.grad_data.clone(), dense_buffer.grad_data.clone()))
        table_range = _reduced_range(table_buffer, module.table, use_distributed_optimizer)
        proj_range = _reduced_range(dense_buffer, module.proj.weight, use_distributed_optimizer)
        num_reduced = _world_sum(table_range[2] - table_range[1])

        for step, (table_grads, dense_grads) in enumerate(reduced):
            expected_table, expected_proj = _expected_grads(_ranks(dp_cp), step, scale)
            rows = expected_table[module.row_start : module.row_end]
            _assert_reduced(table_grads, table_range, rows)
            _assert_reduced(dense_grads, proj_range, expected_proj)
        # Each table element is reduced once with the distributed optimizer, whose replicas each
        # hold part of the shard, else on every replica.
        assert num_reduced == _ROWS * _DIM * (
            1 if use_distributed_optimizer else replica_group.size()
        )
        if overlap_grad_reduce:
            # An identity reduction leaves a completed handle to wait on.
            assert all(handle is not None for handle in handles[1])
            assert all(
                isinstance(handle, _CompletedCollectiveHandle) == identity for handle in handles[1]
            )

    def test_invalid_domains_are_rejected(self):
        _require_ranks()
        Utils.initialize_model_parallel()
        pg_collection, shard_group, replica_group = _process_groups(2)
        config = TransformerConfig(num_attention_heads=1, num_layers=1)

        def wrap(module, pg_collection):
            DistributedDataParallel(
                config, DistributedDataParallelConfig(), module, pg_collection=pg_collection
            )

        # A tagged parameter whose domain has no registered replica group.
        unregistered = ProcessGroupCollection.use_mpu_process_groups()
        with pytest.raises(ValueError, match='no replica group'):
            wrap(_TableModel(shard_group), unregistered)

        # 'dense' and 'expert' are implicit.
        reserved = ProcessGroupCollection.use_mpu_process_groups()
        reserved.parameter_replica_groups = {'expert': replica_group}
        with pytest.raises(ValueError, match='implicit'):
            wrap(_TableModel(with_table=False), reserved)

        # The backward hook would add a placeholder gradient into main_grad.
        module = _TableModel(shard_group)
        module.table.zero_out_wgrad = True
        with pytest.raises(ValueError, match='zero_out_wgrad'):
            wrap(module, pg_collection)

        # A domain's gradients get no GTP-remat correction.
        gtp_remat = ProcessGroupCollection.use_mpu_process_groups()
        gtp_remat.parameter_replica_groups = {_DOMAIN: replica_group}
        gtp_remat.gtp_remat = replica_group
        with pytest.raises(ValueError, match='GTP'):
            wrap(_TableModel(shard_group), gtp_remat)

    def test_replica_group_outside_dp_cp_is_rejected(self):
        """The scaling assumes that a domain replicates over part of this stage's dp_cp."""
        _require_ranks()
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        # The world spans both pipeline stages, so it is part of neither stage's dp_cp.
        pg_collection.parameter_replica_groups = {_DOMAIN: torch.distributed.group.WORLD}
        with pytest.raises(ValueError, match='must be part of dp_cp'):
            DistributedDataParallel(
                TransformerConfig(num_attention_heads=1, num_layers=1),
                DistributedDataParallelConfig(),
                _TableModel(with_table=False),
                pg_collection=pg_collection,
            )

    def test_replica_group_that_is_not_a_strided_slice_is_rejected(self):
        """Replica groups are the strided slices of dp_cp that create_row_parallel_groups builds:
        contiguous blocks would reduce correctly but get no stable optimizer name."""
        _require_ranks()
        Utils.initialize_model_parallel()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        dp_cp_ranks = torch.distributed.get_process_group_ranks(pg_collection.dp_cp)
        half = len(dp_cp_ranks) // 2
        # Every rank creates both groups, in the same order.
        blocks = [
            torch.distributed.new_group(ranks=ranks)
            for ranks in (dp_cp_ranks[:half], dp_cp_ranks[half:])
        ]
        block = blocks[get_pg_rank(pg_collection.dp_cp) // half]
        pg_collection.parameter_replica_groups = {_DOMAIN: block}
        with pytest.raises(ValueError, match='strided slice'):
            DistributedDataParallel(
                TransformerConfig(num_attention_heads=1, num_layers=1),
                DistributedDataParallelConfig(),
                _TableModel(with_table=False),
                pg_collection=pg_collection,
            )

    def test_domain_buffers_skip_nccl_user_buffers(self, monkeypatch):
        """Only dense and expert buffers register NCCL user buffers.

        Registration warms the buffer's group up between barriers over the whole world, so a
        domain held by some pipeline stages only would leave the stages with different barrier
        counts. A recorder takes the setting DDP requests and builds the buffer without it.
        """
        _require_ranks()
        Utils.initialize_model_parallel()
        pg_collection, shard_group, _ = _process_groups(2)
        signature = inspect.signature(_ParamAndGradBuffer)
        requested = {}
        # Before torch 2.11, nccl_ub rejects expandable segments; the allocator is set up by now.
        monkeypatch.delenv('PYTORCH_CUDA_ALLOC_CONF', raising=False)

        class RecordingBuffer(_ParamAndGradBuffer):
            def __init__(self, *args, **kwargs):
                arguments = signature.bind(*args, **kwargs).arguments
                nccl_ub = arguments['nccl_ub']
                arguments['nccl_ub'] = False
                super().__init__(**arguments)
                requested[id(self)] = nccl_ub

        monkeypatch.setattr(ddp_module, '_ParamAndGradBuffer', RecordingBuffer)
        model = _wrap(_TableModel(shard_group), pg_collection, _ddp_config(nccl_ub=True))

        assert {
            domain: [requested[id(buffer)] for buffer in buffers]
            for domain, buffers in model.buffers_by_domain.items()
        } == {'dense': [True], 'expert': [], _DOMAIN: [False]}


class TestDomainOptimizer:
    """A DistributedOptimizer per domain, last in the chain, matches the dense reference."""

    @pytest.fixture(autouse=True)
    def teardown(self):
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('replica_size', ['dp_cp', 2, 1], ids=_REPLICA_IDS)
    @pytest.mark.parametrize(
        'overlap_grad_reduce, overlap_param_gather',
        [(False, False), (True, False), (True, True)],
        ids=['sync', 'overlap_grad_reduce', 'overlap_param_gather'],
    )
    def test_training_matches_dense_reference(
        self, replica_size, overlap_grad_reduce, overlap_param_gather
    ):
        _require_ranks()
        Utils.initialize_model_parallel()
        overlap = dict(
            overlap_grad_reduce=overlap_grad_reduce, overlap_param_gather=overlap_param_gather
        )
        reference_pg_collection, _, _ = _process_groups()
        pg_collection, shard_group, replica_group = _process_groups(replica_size)
        reference, reference_optimizer = _build(
            reference_pg_collection, None, _ddp_config(**overlap)
        )
        model, optimizer = _build(pg_collection, shard_group, _ddp_config(**overlap))
        table = model.module.table

        assert len(reference_optimizer.chained_optimizers) == 1
        dense, table_optimizer = optimizer.chained_optimizers
        assert isinstance(table_optimizer, DistributedOptimizer)
        assert dense.data_parallel_group is pg_collection.dp_cp
        assert table_optimizer.data_parallel_group is replica_group
        assert table_optimizer.data_parallel_group_gloo is None
        assert table_optimizer.data_parallel_group_idx == _optimizer_index(
            pg_collection, replica_group
        )
        # Gradient statistics reduce over one group, so the chain computes one global norm.
        assert table_optimizer.get_grad_stats_parallel_group() is (
            dense.get_grad_stats_parallel_group()
        )
        # The table's groups report is_expert_parallel ("not replicated over the dense DP
        # group"), so they keep their checkpoint identifier in every layout.
        for group in table_optimizer.optimizer.param_groups:
            assert group['parallel_domain'] == _DOMAIN and group['is_expert_parallel']
        assert all('parallel_domain' not in group for group in dense.optimizer.param_groups)
        assert table not in dense.model_param_group_index_map

        num_owned = _world_sum(_owned_numel(table_optimizer))
        reference_losses, reference_norms = _train(reference, reference_optimizer, steps=3)
        losses, grad_norms = _train(model, optimizer, steps=3)
        replicas_match = check_param_hashes_across_dp_replicas([model], cross_check=True)

        # Each element of the table is owned once over its replicas.
        assert set(table_optimizer.model_param_group_index_map) <= {table}
        assert num_owned == _ROWS * _DIM
        assert torch.equal(losses, reference_losses)
        assert grad_norms == pytest.approx(reference_norms, rel=1e-6)
        reference_table = reference.module.table[model.module.row_start : model.module.row_end]
        assert torch.equal(table, reference_table)
        assert torch.equal(model.module.proj.weight, reference.module.proj.weight)
        # The replicas of each row shard stay identical.
        assert replicas_match

    @pytest.mark.parametrize('replica_size', ['dp_cp', 1], ids=_REPLICA_IDS[::2])
    def test_stage_without_the_domain_builds_an_empty_optimizer(self, replica_size):
        """Pipeline stages agree on the chain although only the first holds the table.

        Param groups are aligned over the world, so the second stage sees the domain's group,
        empty, and builds an empty optimizer at the same position; the global gradient norm and
        the step then line up across stages.
        """
        _require_ranks()
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        has_table = parallel_state.is_pipeline_first_stage()
        reference_pg_collection, _, _ = _process_groups()
        pg_collection, shard_group, replica_group = _process_groups(replica_size)
        reference, reference_optimizer = _build(
            reference_pg_collection, None, _ddp_config(), with_table=has_table
        )
        model, optimizer = _build(pg_collection, shard_group, _ddp_config(), with_table=has_table)

        chain = optimizer.chained_optimizers
        assert _from_every_rank(len(chain)) == [2] * torch.distributed.get_world_size()
        dense, table_optimizer = chain
        assert dense.data_parallel_group is pg_collection.dp_cp
        assert table_optimizer.data_parallel_group is replica_group
        # The second stage's index is not the global rank of its replica group's first member,
        # which would change with the data-parallel size.
        index = _optimizer_index(pg_collection, replica_group)
        assert table_optimizer.data_parallel_group_idx == index
        if get_pg_rank(pg_collection.mp) > 0 and replica_size == 'dp_cp':
            assert index != torch.distributed.get_global_rank(replica_group, 0)
        empty_without_table = has_table or (
            model.buffers_by_domain[_DOMAIN] == []
            and model.bucket_groups_by_domain[_DOMAIN] == []
            and table_optimizer.buffers == []
            and len(table_optimizer.optimizer.param_groups) > 0
            and _owned_numel(table_optimizer) == 0
        )
        assert all(_from_every_rank(empty_without_table))

        reference_losses, reference_norms = _train(reference, reference_optimizer, steps=2)
        losses, grad_norms = _train(model, optimizer, steps=2)

        assert torch.equal(losses, reference_losses)
        assert grad_norms == pytest.approx(reference_norms, rel=1e-6)
        assert torch.equal(model.module.proj.weight, reference.module.proj.weight)
        if has_table:
            module = model.module
            reference_table = reference.module.table[module.row_start : module.row_end]
            assert torch.equal(module.table, reference_table)

    @pytest.mark.parametrize('replica_size', [2, 1], ids=_REPLICA_IDS[1:])
    def test_dp_reshardable_checkpoint_round_trip(self, replica_size, tmp_path_dist_ckpt):
        """Resuming from a dp_reshardable checkpoint matches the uninterrupted run."""
        _require_ranks()
        Utils.initialize_model_parallel()
        pg_collection, shard_group, replica_group = _process_groups(replica_size)
        uninterrupted, uninterrupted_optimizer = _build(pg_collection, shard_group, _ddp_config())
        saved, saved_optimizer = _build(pg_collection, shard_group, _ddp_config())
        # The domain's optimizer is last in the chain and names its objects after the
        # model-parallel rank and the row shard of its replica group.
        for optimizer in (uninterrupted_optimizer, saved_optimizer):
            _, table_optimizer = optimizer.chained_optimizers
            assert table_optimizer.data_parallel_group is replica_group

        _train(uninterrupted, uninterrupted_optimizer, steps=3)
        _train(saved, saved_optimizer, steps=2)
        state = saved_optimizer.sharded_state_dict({}, metadata=_DP_RESHARDABLE)
        index = _optimizer_index(pg_collection, replica_group)
        prefix = f'chained_1.optimizer.distributed.dp_group_idx_{index}.'
        keys = [value.key for value in nested_values(state) if isinstance(value, ShardedTensor)]

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_parameter_domains_dp_reshardable', sync=True
        ) as checkpoint_dir:
            save(state, checkpoint_dir)
            resumed, resumed_optimizer = _build(pg_collection, shard_group, _ddp_config())
            resumed.module.load_state_dict(saved.module.state_dict())
            loaded = load(
                resumed_optimizer.sharded_state_dict({}, is_loading=True, metadata=_DP_RESHARDABLE),
                checkpoint_dir,
            )
            resumed_optimizer.load_state_dict(loaded)
        _train(resumed, resumed_optimizer, steps=1, first_step=2)

        assert any(key.startswith(prefix) for key in keys)
        assert torch.equal(resumed.module.table, uninterrupted.module.table)
        assert torch.equal(resumed.module.proj.weight, uninterrupted.module.proj.weight)

    def test_unsupported_setups_are_rejected(self):
        _require_ranks()
        Utils.initialize_model_parallel()
        pg_collection, shard_group, _ = _process_groups(2)

        # The domain's replica group is used only through a DistributedOptimizer.
        model = _wrap(
            _TableModel(shard_group), pg_collection, _ddp_config(use_distributed_optimizer=False)
        )
        config = OptimizerConfig(optimizer='adam', lr=_LR, bf16=True)
        with pytest.raises(ValueError, match='distributed optimizer'):
            get_megatron_optimizer(
                config, [model], pg_collection=pg_collection, use_gloo_process_groups=False
            )

        model = _wrap(
            _TableModel(shard_group),
            pg_collection,
            _ddp_config(overlap_grad_reduce=True, overlap_param_gather=True),
        )
        config = OptimizerConfig(
            optimizer='adam',
            lr=_LR,
            bf16=True,
            use_distributed_optimizer=True,
            overlap_param_gather=True,
            overlap_param_gather_with_optimizer_step=True,
        )
        with pytest.raises(ValueError, match='overlap_param_gather_with_optimizer_step'):
            get_megatron_optimizer(
                config, [model], pg_collection=pg_collection, use_gloo_process_groups=False
            )


class TestSingleRankReplicas:
    """When the rows are sharded over all of dp_cp, each replica group has a single rank."""

    @pytest.fixture(autouse=True)
    def teardown(self):
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('overlap', [False, True], ids=['sync', 'overlap'])
    def test_replica_group_never_communicates(self, overlap, tmp_path_dist_ckpt):
        """Training, broadcasts, hash checks and checkpoints never touch the replica group.

        So its NCCL communicator is never created, which saves the memory of a single-rank
        communicator on every rank.
        """
        _require_ranks()
        Utils.initialize_model_parallel()
        pg_collection, shard_group, replica_group = _process_groups(1)
        assert replica_group.size() == 1
        # The only row shard spans dp_cp, whose group is reused.
        assert shard_group is pg_collection.dp_cp
        model, optimizer = _build(
            pg_collection,
            shard_group,
            _ddp_config(overlap_grad_reduce=overlap, overlap_param_gather=overlap),
        )
        table = model.module.table
        _, table_optimizer = optimizer.chained_optimizers
        assert table_optimizer.data_parallel_group is replica_group

        _train(model, optimizer, steps=2)
        model.broadcast_params()
        replicas_match = check_param_hashes_across_dp_replicas([model], cross_check=True)
        # The legacy gather of the optimizer state is a local copy, without a Gloo group.
        state = table_optimizer.get_parameter_state_dp_zero()
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_parameter_domains_single_rank_replicas', sync=True
        ) as checkpoint_dir:
            save(optimizer.sharded_state_dict({}, metadata=_DP_RESHARDABLE), checkpoint_dir)
        torch.cuda.synchronize()

        # The single replica owns the whole shard.
        assert _owned_numel(table_optimizer) == table.numel()
        assert replicas_match
        [table_buffer] = model.buffers_by_domain[_DOMAIN]
        start, end, _ = table_buffer.param_index_map[table]
        [dtype_state] = state[0].values()
        assert torch.equal(dtype_state['param'][start:end], table.main_param.cpu())
        _assert_never_communicated(replica_group)
        # The probe does see collectives: dp_cp, which carries the dense gradients, ran some.
        initialized, num_collectives = _communicator_state(pg_collection.dp_cp)
        assert initialized and num_collectives > 0

    @pytest.mark.parametrize(
        'overlap_param_gather', [False, True], ids=['sync_param_gather', 'overlap_param_gather']
    )
    def test_identity_param_sync_still_post_processes(self, overlap_param_gather):
        """The skipped all-gather still runs the post-all-gather processing.

        That processing re-quantizes MXFP8 params and copies params that inference packing
        serves from storage of their own. With an overlapped gather it runs in the forward
        pre-hook, only if the dispatch left a handle to wait on: the identity leaves a completed
        one.
        """
        _require_ranks()
        Utils.initialize_model_parallel()
        pg_collection, shard_group, _ = _process_groups(1)
        model, optimizer = _build(
            pg_collection,
            shard_group,
            _ddp_config(
                overlap_grad_reduce=overlap_param_gather, overlap_param_gather=overlap_param_gather
            ),
        )
        table = model.module.table
        [table_buffer] = model.buffers_by_domain[_DOMAIN]
        start, end, _ = table_buffer.param_index_map[table]
        gathered = table_buffer.param_data[start:end].view_as(table)
        bucket_groups = model.bucket_groups_by_domain[_DOMAIN]
        assert bucket_groups and all(group.collectives_are_identity for group in bucket_groups)
        events = _spy_on_param_sync(bucket_groups)
        # Serve the table from storage of its own, as inference packing does: only the
        # post-all-gather processing copies the gathered values into it.
        table.data = table.data.clone()

        model.zero_grad_buffer()
        optimizer.zero_grad()
        _backward(model, step=0)
        model.finish_grad_sync()
        events.clear()
        update_successful, _, _ = optimizer.step()
        events_in_step = list(events)
        stale_after_step = not torch.equal(table, gathered)
        events.clear()
        if overlap_param_gather:
            # The next forward's pre-hook publishes the update.
            model(*_cuda_inputs(torch.distributed.get_rank(), step=1, microbatch=0))

        assert update_successful
        if overlap_param_gather:
            # The step leaves the update in the buffer; the pre-hook waits on the completed
            # handle, then post-processes.
            assert events_in_step == [] and stale_after_step
            expected_events = [('start', _CompletedCollectiveHandle), ('post',)]
            assert events == expected_events * len(bucket_groups)
        else:
            # The optimizer step syncs the params itself, through DDP's post-processing.
            assert set(events_in_step) == {('start', type(None)), ('post',)}
            assert events_in_step[0][0] == 'start' and events_in_step[-1] == ('post',)
        assert torch.equal(table, gathered)


class TestDefaultPath:
    """Models without an extra domain build the buffers and chains they built before."""

    @pytest.fixture(autouse=True)
    def teardown(self):
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('expert_parallel', ['two', 'world'])
    def test_buffers_and_chain_are_unchanged(self, expert_parallel):
        """Dense and expert buffers and optimizers are the same with and without a domain.

        The expert-parallel size covers a single-rank expert DP group: unlike a domain's, its
        collectives are kept.
        """
        _require_ranks()
        expert_model_parallel_size = 2 if expert_parallel == 'two' else Utils.world_size
        Utils.initialize_model_parallel(expert_model_parallel_size=expert_model_parallel_size)

        def build(pg_collection, shard_group, with_table):
            ddp_config = _ddp_config(overlap_grad_reduce=True, overlap_param_gather=True)
            module = _TableModel(shard_group, with_table=with_table, with_expert=True)
            # The layout is computed inside DDP, as for any caller that passes none.
            model = DistributedDataParallel(
                TransformerConfig(num_attention_heads=1, num_layers=1),
                ddp_config,
                module,
                pg_collection=pg_collection,
            )
            model.broadcast_params()
            # The parallel_state groups, with their Gloo groups: extra domains come from DDP.
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    optimizer='adam',
                    lr=_LR,
                    bf16=True,
                    use_distributed_optimizer=True,
                    overlap_param_gather=True,
                ),
                [model],
            )
            _train(model, optimizer, steps=1)
            return model, optimizer

        # Without a domain: the parallel_state groups, with their Gloo groups, as before.
        before, before_optimizer = build(None, None, with_table=False)
        pg_collection, shard_group, replica_group = _process_groups(2)
        after, after_optimizer = build(pg_collection, shard_group, with_table=True)

        assert before.parameter_replica_groups == {
            'dense': before.dp_cp_group,
            'expert': before.expt_dp_group,
        }
        assert list(before.buffers_by_domain) == ['dense', 'expert']
        assert before.all_buffers == before.buffers + before.expert_parallel_buffers
        assert before.all_bucket_groups == (
            before.bucket_groups + before.expert_parallel_bucket_groups
        )
        assert not any(group.collectives_are_identity for group in before.all_bucket_groups)
        bf16, fp32 = torch.bfloat16, torch.float32
        assert set(before.full_param_layout.layouts) == {
            BufferKey(bf16, fp32, False),
            BufferKey(bf16, fp32, True),
        }

        # The dense and expert buffers are the same with the domain.
        def describe(model, buffers):
            names = {param: name for name, param in model.module.named_parameters()}
            return [
                (
                    {names[param]: index for param, index in buffer.param_index_map.items()},
                    buffer.numel,
                    buffer.gradient_scaling_factor,
                    _ranks(buffer.data_parallel_group),
                )
                for buffer in buffers
            ]

        for domain in ('dense', 'expert'):
            assert describe(after, after.buffers_by_domain[domain]) == describe(
                before, before.buffers_by_domain[domain]
            )
            assert not any(
                group.collectives_are_identity for group in after.bucket_groups_by_domain[domain]
            )

        # So are the dense and MoE optimizers; the domain's comes last.
        def describe_optimizer(distopt):
            return (
                _ranks(distopt.data_parallel_group),
                distopt.data_parallel_group_gloo,
                distopt.data_parallel_group_idx,
                [
                    (get_param_group_identifier_tuple(group), len(group['params']))
                    for group in distopt.optimizer.param_groups
                ],
            )

        assert isinstance(before_optimizer, ChainedOptimizer)
        assert [type(child) for child in before_optimizer.chained_optimizers] == [
            DistributedOptimizer,
            DistributedOptimizer,
        ]
        dense, expert = before_optimizer.chained_optimizers
        assert dense.data_parallel_group_idx == get_pg_rank(pg_collection.mp)
        assert expert.data_parallel_group_idx == get_pg_rank(pg_collection.tp_ep_pp_with_egtp_remat)
        assert dense.data_parallel_group_gloo is not None
        assert not any(
            'parallel_domain' in group
            for child in before_optimizer.chained_optimizers
            for group in child.optimizer.param_groups
        )
        assert [describe_optimizer(child) for child in after_optimizer.chained_optimizers[:2]] == [
            describe_optimizer(child) for child in before_optimizer.chained_optimizers
        ]
        table_optimizer = after_optimizer.chained_optimizers[2]
        assert table_optimizer.data_parallel_group is replica_group

        # And so are the checkpoint keys; the domain's optimizer adds keys of its own.
        before_keys = _checkpoint_keys(before_optimizer)
        after_keys = _checkpoint_keys(after_optimizer)
        assert [key for key in after_keys if not key.startswith('chained_2.')] == before_keys
        assert any(key.startswith('chained_2.') for key in after_keys)


class TestParamsNorm:
    """The params norm counts each element of a domain once, as for a dense parameter."""

    @pytest.fixture(autouse=True)
    def teardown(self):
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('replica_size', ['dp_cp', 2, 1], ids=_REPLICA_IDS)
    def test_params_norm_matches_dense_reference(self, replica_size, monkeypatch, run_config):
        """A row shard counts on replica rank 0, or as the distributed optimizer's fragments,
        and the counts add up over dp_cp."""
        from megatron.training.utils import common_utils

        _require_ranks()
        Utils.initialize_model_parallel()
        # bf16 params: the norm reads the distributed optimizer's fp32 fragments, or with
        # force_create_fp32_copy fp32 copies of the whole params.
        monkeypatch.setattr(
            common_utils, 'get_args', lambda: SimpleNamespace(use_megatron_fsdp=False, bf16=True)
        )
        # The norm reads the FSDP switch and the precision from the registered run config.
        run_config.model.transformer.bf16 = True
        reference_pg_collection, _, _ = _process_groups()
        pg_collection, shard_group, _ = _process_groups(replica_size)
        reference, _ = _build(reference_pg_collection, None, _ddp_config())
        model, _ = _build(pg_collection, shard_group, _ddp_config())

        norms = []
        for force_create_fp32_copy in (False, True):
            expected = common_utils.calc_params_l2_norm(
                [reference], force_create_fp32_copy, pg_collection=reference_pg_collection
            )
            # The domains come from pg_collection, or else from the DDP-wrapped chunks.
            for groups in (pg_collection, None):
                actual = common_utils.calc_params_l2_norm(
                    [model], force_create_fp32_copy, pg_collection=groups
                )
                norms.append((actual, expected))

        for actual, expected in norms:
            assert actual == pytest.approx(expected, rel=1e-6)


class TestTrainingWrappers:
    """The training wrappers lay each domain out for its replica group and reject FSDP.

    These run without torch.distributed: DDP, the layout and CUDA streams are mocked.
    """

    def test_ddp_wrap(self):
        from megatron.training.models import dist_utils

        replica_group = mock.Mock()
        replica_group.size.return_value = 2
        pg_collection = mock.Mock()
        pg_collection.pp.rank.return_value = 0
        pg_collection.intra_dp_cp.size.return_value = 4
        pg_collection.intra_expt_dp.size.return_value = 2
        pg_collection.parameter_replica_groups = {_DOMAIN: replica_group}
        param = SimpleNamespace(requires_grad=True, nelement=lambda: 1)
        chunk = mock.Mock()
        chunk.parameters.return_value = [param]
        ddp_config = mock.Mock(
            num_buckets=None,
            bucket_size=8,
            overlap_grad_reduce=True,
            use_distributed_optimizer=True,
        )
        unsupported = [{'use_megatron_fsdp': True}]
        if dist_utils.HAVE_FSDP2:
            unsupported.append({'use_torch_fsdp2': True})

        with (
            mock.patch.object(dist_utils, 'DistributedDataParallel') as ddp,
            mock.patch.object(dist_utils, 'DistributedOptimizer') as distopt,
            mock.patch.object(dist_utils, 'get_model_config') as get_model_config,
            mock.patch('torch.cuda.current_stream'),
            mock.patch('torch.cuda.stream', new_callable=mock.MagicMock),
        ):
            get_model_config.return_value.cuda_graph_impl = 'none'
            dist_utils._ddp_wrap([chunk], False, ddp_config, False, pg_collection=pg_collection)
            for kwargs in unsupported:
                with pytest.raises(ValueError, match='standard DistributedDataParallel'):
                    dist_utils._ddp_wrap(
                        [chunk], False, ddp_config, False, pg_collection=pg_collection, **kwargs
                    )

        distopt.compute_full_param_layout.assert_called_once_with(
            [param],
            8,
            4,
            ddp_config,
            expert_data_parallel_world_size=2,
            parallel_group_sizes={_DOMAIN: 2},
        )
        ddp.assert_called_once()
        assert ddp.call_args.kwargs['pg_collection'] is pg_collection

    def test_wrap_model_chunks_with_ddp(self):
        from megatron.training import training

        class FakeDDP:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeFSDP(FakeDDP):
            pass

        replica_group = object()
        param = SimpleNamespace(requires_grad=True, nelement=lambda: 1)
        chunk = SimpleNamespace(parameters=lambda: [param])
        pg_collection = SimpleNamespace(
            dp_cp=object(), expt_dp=object(), parameter_replica_groups={_DOMAIN: replica_group}
        )
        ddp_config = SimpleNamespace(bucket_size=8, use_distributed_optimizer=True)
        layout = object()

        with (
            mock.patch.object(training, 'DDP', FakeDDP),
            mock.patch.object(
                training,
                'get_pg_size',
                side_effect=lambda group: 2 if group is replica_group else 4,
            ),
            mock.patch.object(
                training.DistributedOptimizer, 'compute_full_param_layout', return_value=layout
            ) as compute_layout,
        ):
            [wrapped] = training.wrap_model_chunks_with_ddp(
                [chunk], object(), ddp_config, DP=FakeDDP, pg_collection=pg_collection
            )
            with pytest.raises(ValueError, match='standard DistributedDataParallel'):
                training.wrap_model_chunks_with_ddp(
                    [chunk], object(), ddp_config, DP=FakeFSDP, pg_collection=pg_collection
                )

        compute_layout.assert_called_once_with(
            [param],
            8,
            4,
            ddp_config,
            expert_data_parallel_world_size=4,
            parallel_group_sizes={_DOMAIN: 2},
        )
        assert wrapped.kwargs['pg_collection'] is pg_collection
        assert wrapped.kwargs['full_param_layout'] is layout


# The layer-wise optimizer tests do not run on the LTS image (as in test_layer_wise_optimizer.py).
_skip_on_lts = pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', '24.01')) <= Version('25.05'),
    reason='Skip layer-wise optimizer for LTS test',
)


class _LayerWiseModel(torch.nn.Module):
    """Muon matrices with Adam biases, an expert-parallel table and a domain table."""

    def __init__(self, shard_group):
        super().__init__()
        self.shard_group = shard_group
        self.fc1 = torch.nn.Linear(_IN, 16, dtype=torch.bfloat16, device='cuda')
        self.fc2 = torch.nn.Linear(16, _OUT, dtype=torch.bfloat16, device='cuda')
        self.expert_table = torch.nn.Parameter(
            _dyadic((12, _DIM), seed=3).to(device='cuda', dtype=torch.bfloat16)
        )
        rows_per_shard = _ROWS // shard_group.size()
        row_start = shard_group.rank() * rows_per_shard
        rows = _dyadic((_ROWS, _DIM), seed=1)[row_start : row_start + rows_per_shard]
        self.table = torch.nn.Parameter(rows.to(device='cuda', dtype=torch.bfloat16))
        # Tables are embeddings: Adam updates them, not Muon.
        for table in (self.expert_table, self.table):
            table.allreduce = False
            table.is_embedding_or_output_parameter = True
        self.table.parallel_domain = _DOMAIN

    def forward(self, dense_input, dense_coefficients, row_coefficients):
        dense = self.fc2(torch.relu(self.fc1(dense_input))).float()
        rows = _GatherRows.apply(self.table, self.shard_group)
        return (
            (dense * dense_coefficients).sum()
            + (rows.float() * row_coefficients).sum()
            + self.expert_table.float().square().sum()
        )


@_skip_on_lts
@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason='needs emerging_optimizers')
class TestLayerWiseChain:
    """On the layer-wise path the domain's optimizer follows LayerWise and the fallbacks."""

    @pytest.fixture(autouse=True)
    def teardown(self):
        yield
        Utils.destroy_model_parallel()

    def test_domain_optimizer_comes_last(self):
        from megatron.training.training import wrap_model_chunks_with_ddp

        _require_ranks()
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        pg_collection, shard_group, replica_group = _process_groups(2)
        # Forces use_distributed_optimizer in DDP and lays the domain out for its replica group.
        model = wrap_model_chunks_with_ddp(
            [_LayerWiseModel(shard_group)],
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(),
            use_layer_wise_distributed_optimizer=True,
            pg_collection=pg_collection,
        )[0]
        model.broadcast_params()
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer='muon',
                lr=_LR,
                bf16=True,
                use_distributed_optimizer=False,
                clip_grad=1.0,
                muon_tp_mode='duplicated',
                use_layer_wise_distributed_optimizer=True,
            ),
            [model],
            pg_collection=pg_collection,
            use_gloo_process_groups=False,
        )

        children = optimizer.chained_optimizers
        assert [type(child) for child in children] == [
            LayerWiseDistributedOptimizer,
            DistributedOptimizer,
            DistributedOptimizer,
            DistributedOptimizer,
        ]
        _, dense, expert, table_optimizer = children
        assert _ranks(dense.data_parallel_group) == _ranks(pg_collection.dp_cp)
        assert _ranks(expert.data_parallel_group) == _ranks(pg_collection.expt_dp)
        assert table_optimizer.data_parallel_group is replica_group
        assert table_optimizer.data_parallel_group_idx == _optimizer_index(
            pg_collection, replica_group
        )

        table = model.module.table
        table_before = table.detach().clone()
        num_owned = _world_sum(_owned_numel(table_optimizer))
        _train(model, optimizer, steps=1)
        replicas_match = check_param_hashes_across_dp_replicas([model], cross_check=True)

        # Each element of the table is owned once over its replicas, which stay identical.
        assert set(table_optimizer.model_param_group_index_map) <= {table}
        assert num_owned == _ROWS * _DIM
        assert not torch.equal(table, table_before)
        assert replicas_match

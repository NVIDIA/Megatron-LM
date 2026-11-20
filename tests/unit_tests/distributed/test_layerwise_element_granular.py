# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Element-granular dense LayerWise buffer (``layerwise_element_granular_dense``).

The layout keeps every param's LayerWise owner, so training is bitwise identical to the
shard-aligned layout with the fp32-accumulating reduce-scatter at any data-parallel size, and
with the plain reduce-scatter at data-parallel size 2 (one addition per element).
"""

import math
import os
import random

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

import megatron.core.distributed.param_and_grad_buffer as param_and_grad_buffer
import megatron.core.optimizer.optimizer as megatron_optimizer_module
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.param_layout import (
    compute_owner_exchange_plan,
    compute_owner_exchange_splits,
    owner_exchange_runs,
    owner_exchange_segments,
    pad_param_start,
    pad_to_divisor,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip layer-wise optimizer for LTS test",
)

WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))
TRANSPORTS = ["a2a", "p2p"]


# ---------------------------------------------------------------------------------------------
# Layout (CPU)
# ---------------------------------------------------------------------------------------------


class _ShapeOnlyParam:
    """What the layout code reads of a param, without its storage."""

    def __init__(self, rows, cols):
        self.shape = (rows, cols)

    @property
    def data(self):
        return self

    def nelement(self):
        return self.shape[0] * self.shape[1]

    def dim(self):
        return 2


def _params(shapes):
    return [torch.nn.Parameter(torch.empty(*shape)) for shape in shapes]


def _ddp_config(pad_for_busbw=False):
    return DistributedDataParallelConfig(pad_buckets_for_high_nccl_busbw=pad_for_busbw)


def _owner_layout(params, bucket_size, dp_size, ddp_config):
    layout = LayerWiseDistributedOptimizer._compute_per_buffer_param_layout(
        params, bucket_size, dp_size, ddp_config, log_layout=False
    )
    owners = {}
    for param, (start, _, bucket_id) in layout.param_index_map.items():
        bucket_start, bucket_end = layout.bucket_indices[bucket_id]
        owners[param] = (start - bucket_start) // ((bucket_end - bucket_start) // dp_size)
    return owners, layout


def _bucket_ranges(layout):
    ranges = {}
    for param, (start, end, bucket_id) in layout.param_index_map.items():
        ranges.setdefault(bucket_id, []).append((start, end, layout.param_to_owner_shard[param]))
    return ranges


@pytest.mark.parametrize(
    "dp_size, bucket_size, pad_for_busbw",
    [
        (1, None, False),
        (2, 10**7, True),
        (4, 3000, False),
        (64, None, True),  # more ranks than params
        (96, 3000, True),  # not a power of two
        (256, 3000, False),
    ],
)
def test_element_granular_layout_invariants(dp_size, bucket_size, pad_for_busbw):
    """Owners are the shard-aligned layout's shards; buckets are contiguous backprop ranges
    of 64-aligned, non-overlapping, owner-major params with slice-aligned ends."""
    rng = random.Random(dp_size * 7 + (bucket_size or 0))
    params = _params([(rng.randint(1, 96), rng.randint(1, 96)) for _ in range(40)])
    ddp_config = _ddp_config(pad_for_busbw)
    owners, _ = _owner_layout(params, bucket_size, dp_size, ddp_config)
    total = sum(p.numel() for p in params)
    eg_bucket_size = None if bucket_size is None else max(1, math.ceil(total / 5))
    layout = LayerWiseDistributedOptimizer._compute_element_granular_param_layout(
        params, bucket_size, eg_bucket_size, dp_size, ddp_config
    )

    assert layout.param_to_owner_shard == owners
    assert layout.num_optimizer_shards == dp_size
    assert set(layout.param_index_map) == set(params)
    divisor = dp_size * LayerWiseDistributedOptimizer._shard_divisor(dp_size, ddp_config)

    # Buckets tile the buffer and end on a slice boundary.
    previous_end = 0
    for bucket_start, bucket_end in layout.bucket_indices:
        assert bucket_start == previous_end
        assert (bucket_end - bucket_start) % divisor == 0
        assert ((bucket_end - bucket_start) // dp_size) % 64 == 0
        previous_end = bucket_end
    if bucket_size is None:
        assert len(layout.bucket_indices) == 1

    # Bucket ids follow the backprop order, as _ParamAndGradBuffer requires.
    previous_bucket = 0
    for param in reversed(params):
        start, end, bucket_id = layout.param_index_map[param]
        assert bucket_id in (previous_bucket, previous_bucket + 1)
        previous_bucket = bucket_id
        bucket_start, bucket_end = layout.bucket_indices[bucket_id]
        assert bucket_start <= start and end <= bucket_end
        assert start % 64 == 0 and end - start == param.numel()

    # No overlap, and owners do not decrease along a bucket.
    for bucket_id, ranges in _bucket_ranges(layout).items():
        ranges.sort()
        for (_, end, owner), (next_start, _, next_owner) in zip(ranges, ranges[1:]):
            assert end <= next_start
            assert owner <= next_owner
        # Padding beyond alignment: less than one divisor per bucket, 63 elements per param.
        bucket_start, bucket_end = layout.bucket_indices[bucket_id]
        used = sum(end - start for start, end, _ in ranges)
        assert (bucket_end - bucket_start) - used < divisor + 64 * len(ranges)


def _random_owner_major_bucket(rng, dp_size):
    owners = sorted(rng.randrange(dp_size) for _ in range(rng.randint(1, 3 * dp_size)))
    bucket_start = rng.choice([0, 64 * dp_size * 7])
    cursor, ranges = bucket_start, []
    for owner in owners:
        cursor = pad_param_start(cursor)
        numel = rng.randint(1, 5000)
        ranges.append((cursor, cursor + numel, owner))
        cursor += numel
    bucket_end = bucket_start + pad_to_divisor(cursor - bucket_start, dp_size * 64)
    return ranges, bucket_start, bucket_end


@pytest.mark.parametrize("dp_size", [2, 3, 96])
def test_owner_exchange_plan_is_symmetric_and_covers_owned_params(dp_size):
    """p2p: every send has its receive in the same order, and each owner's params are covered
    exactly once by its own slice plus what it receives."""
    ranges, bucket_start, bucket_end = _random_owner_major_bucket(
        random.Random(100 * dp_size), dp_size
    )
    shard = (bucket_end - bucket_start) // dp_size
    plans = [
        compute_owner_exchange_plan(ranges, bucket_start, bucket_end, dp_size, rank)
        for rank in range(dp_size)
    ]
    sends, receives = {}, {}
    for rank, plan in enumerate(plans):
        for lo, hi, peer in plan.to_owners:
            assert rank * shard <= lo < hi <= (rank + 1) * shard and peer != rank
            sends.setdefault((rank, peer), []).append((lo, hi))
        for lo, hi, peer in plan.from_slices:
            assert peer * shard <= lo < hi <= (peer + 1) * shard and peer != rank
            receives.setdefault((peer, rank), []).append((lo, hi))
    assert sends == receives
    for start, end, owner in ranges:
        lo, hi = start - bucket_start, end - bucket_start
        covered = max(0, min(hi, (owner + 1) * shard) - max(lo, owner * shard))
        for piece_lo, piece_hi, _ in plans[owner].from_slices:
            covered += max(0, min(hi, piece_hi) - max(lo, piece_lo))
        assert covered == hi - lo
    # Runs merge consecutive same-owner pieces: never two adjacent runs of one owner.
    for slice_runs in owner_exchange_runs(ranges, bucket_start, bucket_end, dp_size):
        assert all(a[2] != b[2] for a, b in zip(slice_runs, slice_runs[1:]))


@pytest.mark.parametrize("dp_size", [1, 3, 16, 96])
def test_stage_a_split_sizes_are_contiguous(dp_size):
    """a2a: for every (source, owner) pair the piece is one contiguous range of the source's
    slice and of the owner's segment, both ends agree on its size, and every element of every
    segment and of every slice is covered exactly once."""
    ranges, bucket_start, bucket_end = _random_owner_major_bucket(
        random.Random(100 * dp_size), dp_size
    )
    numel = bucket_end - bucket_start
    shard = numel // dp_size
    segments = owner_exchange_segments(ranges, bucket_start, bucket_end, dp_size)
    assert segments[0][0] == 0 and segments[-1][1] == numel
    assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))
    for start, end, owner in ranges:
        assert segments[owner][0] <= start - bucket_start
        assert end - bucket_start <= segments[owner][1]

    splits = [
        compute_owner_exchange_splits(ranges, bucket_start, bucket_end, dp_size, rank)
        for rank in range(dp_size)
    ]
    assert len({split.has_remote_pieces for split in splits}) == 1
    for rank, split in enumerate(splits):
        assert split.segment == segments[rank]
        assert sum(split.slice_to_owner_sizes) == shard
        assert sum(split.owner_from_slice_sizes) == segments[rank][1] - segments[rank][0]
        for peer in range(dp_size):
            assert split.slice_to_owner_sizes[peer] == splits[peer].owner_from_slice_sizes[rank]

    # Stage A as the collective delivers it: owner o concatenates, in source order, the piece
    # each source cut from its slice behind the pieces of the owners before o.
    for owner in range(dp_size):
        delivered = []
        for source in range(dp_size):
            offset = source * shard + sum(splits[source].slice_to_owner_sizes[:owner])
            delivered += range(offset, offset + splits[source].slice_to_owner_sizes[owner])
        assert delivered == list(range(*segments[owner]))
    # Stage A^-1: slice r concatenates, in owner order, the piece each owner cut from its
    # segment behind the pieces of the slices before r.
    for rank in range(dp_size):
        delivered = []
        for owner in range(dp_size):
            offset = segments[owner][0] + sum(splits[owner].owner_from_slice_sizes[:rank])
            delivered += range(offset, offset + splits[owner].owner_from_slice_sizes[rank])
        assert delivered == list(range(rank * shard, (rank + 1) * shard))


@pytest.mark.parametrize("pad_for_busbw", [False, True])
def test_projector_buffer_padding(pad_for_busbw):
    """A buffer with fewer params than data-parallel ranks (the 2-param vision projector):
    the shard-aligned layout pads it to dp x the largest param, this layout by alignment."""
    dp_size = 64
    params = [_ShapeOnlyParam(10240, 2048), _ShapeOnlyParam(10240, 10240)]
    total = sum(p.nelement() for p in params)
    ddp_config = _ddp_config(pad_for_busbw)
    layout = LayerWiseDistributedOptimizer._compute_element_granular_param_layout(
        params, total // 10, math.ceil(total / 10), dp_size, ddp_config
    )
    owners, owner_layout = _owner_layout(params, total // 10, dp_size, ddp_config)
    assert layout.param_to_owner_shard == owners
    assert layout.bucket_indices[-1][1] / total - 1 < 1e-3
    assert owner_layout.bucket_indices[-1][1] == dp_size * 104_857_600


def test_shared_embedding_is_rejected():
    params = _params([(8, 8), (8, 8)])
    params[0].shared_embedding = True
    with pytest.raises(ValueError, match="shared embedding"):
        LayerWiseDistributedOptimizer._compute_element_granular_param_layout(
            params, None, None, 2, _ddp_config()
        )


def test_config_validation():
    # The config is checked when a wrapper finalizes it, not at construction.
    with pytest.raises(ValueError, match="layerwise_owner_exchange"):
        DistributedDataParallelConfig(layerwise_owner_exchange="ring").finalize()
    with pytest.raises(AssertionError, match="layerwise_dense_num_buckets"):
        DistributedDataParallelConfig(layerwise_dense_num_buckets=0).finalize()
    flag_on = DistributedDataParallelConfig(layerwise_element_granular_dense=True)
    with pytest.raises(ValueError, match="layerwise_element_granular_dense"):
        OptimizerConfig(optimizer='adam', use_distributed_optimizer=True).validate_run(
            ddp_config=flag_on, use_torch_fsdp2=False, ckpt_format="torch_dist"
        )
    OptimizerConfig(optimizer='muon', use_distributed_optimizer=True).validate_run(
        ddp_config=flag_on, use_torch_fsdp2=False, ckpt_format="torch_dist"
    )


# ---------------------------------------------------------------------------------------------
# Training (GPU)
# ---------------------------------------------------------------------------------------------


class _MixedShapes(nn.Module):
    """Linear stack whose weights straddle the data-parallel slices of the layout."""

    def __init__(self, dims=(96, 320, 192, 448, 160, 256, 64)):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.gelu(x)
        return x


def _translate(params_per_rank, from_chunks, to_chunks):
    """Map per-rank param lists of one model onto the same-named params of another."""
    names, by_name = {}, {}
    for index, (from_chunk, to_chunk) in enumerate(zip(from_chunks, to_chunks)):
        names.update({id(p): (index, n) for n, p in from_chunk.module.named_parameters()})
        by_name.update({(index, n): p for n, p in to_chunk.module.named_parameters()})
    return [[by_name[names[id(p)]] for p in owned] for owned in params_per_rank]


def _layerwise_buffers(chunks):
    return [
        buffer
        for chunk in chunks
        for buffer in chunk.buffers
        if getattr(buffer.params[0], 'is_managed_by_layer_wise_optimizer', False)
    ]


def _build(
    element_granular,
    *,
    transport="a2a",
    overlap=True,
    fp32_accum=True,
    bucket_size=10**6,
    copy_from=None,
    clip_grad=1.0,
    num_chunks=1,
    num_buckets=3,
    ddp_kwargs=None,
    transformer_kwargs=None,
):
    """Model chunks wrapped in DDP with a LayerWise layout, and their Muon optimizer."""
    from megatron.training.training import wrap_model_chunks_with_ddp

    torch.manual_seed(1234)
    chunks = [_MixedShapes().bfloat16().cuda() for _ in range(num_chunks)]
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=overlap,
        overlap_param_gather=overlap,
        bucket_size=bucket_size,
        reduce_scatter_with_fp32_accumulation=fp32_accum,
        layerwise_element_granular_dense=element_granular,
        layerwise_dense_num_buckets=num_buckets,
        layerwise_owner_exchange=transport,
        **(ddp_kwargs or {}),
    )
    # As get_model does: only the first chunk of the first pipeline stage is bucketed.
    disable_bucketing = [index > 0 for index in range(num_chunks)]
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    chunks = wrap_model_chunks_with_ddp(
        chunks,
        TransformerConfig(num_attention_heads=1, num_layers=1, **(transformer_kwargs or {})),
        ddp_config,
        use_layer_wise_distributed_optimizer=True,
        bucket_sizes=[
            None if (disable or pp_rank > 0) else bucket_size for disable in disable_bucketing
        ],
        disable_bucketing_per_chunk=disable_bucketing,
    )
    for index, chunk in enumerate(chunks):
        if copy_from is not None:
            chunk.module.load_state_dict(copy_from[index].module.state_dict())
        else:
            chunk.broadcast_params()
    optimizer_config = OptimizerConfig(
        optimizer='muon',
        lr=0.02,
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=False,
        clip_grad=clip_grad,
        muon_tp_mode="duplicated",
        use_layer_wise_distributed_optimizer=True,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()
    optimizer = get_megatron_optimizer(
        optimizer_config, chunks, pg_collection=pg_collection, use_gloo_process_groups=False
    )
    return chunks, optimizer


def _train(chunks, optimizer, steps, on_step=None):
    """``steps`` Muon steps on per-rank data; returns the grad norm of every step."""
    data_rank = torch.distributed.get_rank()
    grad_norms = []
    for step in range(steps):
        for chunk in chunks:
            chunk.zero_grad_buffer()
        optimizer.zero_grad()
        generator = torch.Generator(device='cuda').manual_seed(100 * step + data_rank)
        x = torch.randn(32, 96, device='cuda', generator=generator).bfloat16()
        y = torch.randn(32, 64, device='cuda', generator=generator)
        loss = sum(F.mse_loss(chunk(x).float(), y) for chunk in chunks)
        loss.backward()
        for chunk in chunks:
            chunk.finish_grad_sync()
        success, grad_norm, _ = optimizer.step()
        assert success
        grad_norms.append(grad_norm)
        if on_step is not None:
            on_step(step)
    if chunks[0].ddp_config.overlap_param_gather:
        for chunk in chunks:
            chunk.start_param_sync(force_sync=True)
    return grad_norms


def _assert_same_training_state(chunks, optimizer, ref_chunks, ref_optimizer):
    """Model params and FP32 masters equal the reference run's; DP replicas agree."""
    for chunk, ref_chunk in zip(chunks, ref_chunks):
        for (name, p), (_, ref_p) in zip(
            chunk.module.named_parameters(), ref_chunk.module.named_parameters()
        ):
            assert torch.equal(p.data, ref_p.data), f"{name} differs from the shard-aligned run"
    masters, ref_masters = optimizer.get_parameters(), ref_optimizer.get_parameters()
    assert len(masters) == len(ref_masters)
    for p, ref_p in zip(masters, ref_masters):
        assert torch.equal(p.data, ref_p.data), "FP32 main params differ"
    dp_group = parallel_state.get_data_parallel_group()
    for chunk in chunks:
        for name, p in chunk.module.named_parameters():
            gathered = [torch.empty_like(p.data) for _ in range(dp_group.size())]
            torch.distributed.all_gather(gathered, p.data.contiguous(), group=dp_group)
            assert all(torch.equal(g, gathered[0]) for g in gathered), name


def _assert_exchange_is_exercised(chunks):
    """Some rank of the job holds a piece owned by another rank, in both descriptions."""
    moved = 0
    for buffer in _layerwise_buffers(chunks):
        assert buffer.param_to_owner_shard is not None
        for bucket in buffer.buckets:
            moved += sum(end - start for start, end, _ in bucket.owner_exchange_plan.to_owners)
            assert bucket.owner_exchange_splits is not None
    moved = torch.tensor([moved], device='cuda')
    torch.distributed.all_reduce(moved)
    assert moved.item() > 0, "the test model must exercise the slice <-> owner exchange"


@pytest.mark.skipif(WORLD_SIZE < 4, reason="needs at least 4 ranks")
class TestElementGranularBitwise:
    """Train the same model with the shard-aligned and with the element-granular dense
    LayerWise layout and compare parameters, FP32 masters and grad norms bitwise."""

    def teardown_method(self, method):
        # A bucket group built with reduce_scatter_with_fp32_accumulation rebinds this
        # module-level function for every later group; do not leak it into other tests.
        param_and_grad_buffer.dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
        Utils.destroy_model_parallel()

    def _compare(self, steps=4, after_step=None, **kwargs):
        """Train both layouts and compare; ``after_step(chunks)`` runs after every step."""
        ref_chunks, ref_optimizer = _build(False, **kwargs)
        chunks, optimizer = _build(True, copy_from=ref_chunks, **kwargs)
        _assert_exchange_is_exercised(chunks)
        for buffer in _layerwise_buffers(ref_chunks):
            assert buffer.param_to_owner_shard is None
        lw, ref_lw = optimizer.chained_optimizers[0], ref_optimizer.chained_optimizers[0]
        assert isinstance(lw, LayerWiseDistributedOptimizer)
        ref_owned = _translate(ref_lw.dp_cp_params_list, ref_chunks, chunks)
        assert [[id(p) for p in owned] for owned in lw.dp_cp_params_list] == [
            [id(q) for q in owned] for owned in ref_owned
        ]
        before = [
            {n: p.data.clone() for n, p in chunk.module.named_parameters()} for chunk in chunks
        ]

        def hook(model_chunks):
            return None if after_step is None else (lambda step: after_step(model_chunks))

        ref_norms = _train(ref_chunks, ref_optimizer, steps, on_step=hook(ref_chunks))
        norms = _train(chunks, optimizer, steps, on_step=hook(chunks))
        assert norms == ref_norms
        _assert_same_training_state(chunks, optimizer, ref_chunks, ref_optimizer)
        # A layout that leaves a gradient slot zero still trains: every matrix must have moved.
        for chunk, chunk_before in zip(chunks, before):
            for name, p in chunk.module.named_parameters():
                if p.dim() == 2:
                    assert not torch.equal(p.data, chunk_before[name]), f"{name} did not move"
        return chunks, optimizer

    @pytest.mark.parametrize("transport", TRANSPORTS)
    @pytest.mark.parametrize(
        "dp_size, overlap, fp32_accum, bucket_size",
        [
            ("world", True, True, 10**6),  # production-style RS, owners from one bucket
            ("world", True, True, 120_000),  # owners from several shard-aligned buckets
            ("world", False, True, None),  # synchronous grad reduce and param gather
            (2, True, False, 10**6),  # plain NCCL reduce-scatter: one addition per element
        ],
    )
    def test_bitwise_identical_to_shard_aligned(
        self, transport, dp_size, overlap, fp32_accum, bucket_size
    ):
        tp_size = 1 if dp_size == "world" else WORLD_SIZE // dp_size
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
        chunks, _ = self._compare(
            transport=transport, overlap=overlap, fp32_accum=fp32_accum, bucket_size=bucket_size
        )
        if overlap:
            assert len(_layerwise_buffers(chunks)[0].buckets) == 3

    def test_bitwise_with_pipeline_stages(self):
        """Two pipeline stages: each stage has its own buffers and data-parallel group."""
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        self._compare(transport="p2p")

    def test_bitwise_with_unbucketed_second_chunk(self):
        """Two model chunks as under interleaved pipeline schedules: the second chunk's
        buffer has bucketing disabled and is one bucket."""
        Utils.initialize_model_parallel()
        chunks, _ = self._compare(num_chunks=2)
        first, second = _layerwise_buffers(chunks)
        assert len(first.buckets) == 3 and len(second.buckets) == 1

    def test_bitwise_with_fp32_local_accumulation_outside_the_buffer(self):
        """param_name_patterns_for_fp32_local_accumulation that matches no param of the
        element-granular buffer (here: the biases, which Adam owns) is accepted."""
        Utils.initialize_model_parallel()
        self._compare(ddp_kwargs=dict(param_name_patterns_for_fp32_local_accumulation=("*bias",)))

    @pytest.mark.parametrize(
        "transport, ddp_kwargs",
        [
            pytest.param("a2a", dict(grad_buffer_offload=True), id="offload-a2a"),
            pytest.param(
                "p2p", dict(nccl_ub=True, grad_buffer_offload=True), id="nccl_ub_offload-p2p"
            ),
        ],
    )
    def test_bitwise_with_nccl_ub_and_grad_buffer_offload(self, transport, ddp_kwargs):
        """User-buffer registration of the DDP buffers ("p2p" exchanges views of them) and
        gradient-buffer offload between steps: the gradient storage is released and
        re-created, with a separate NCCL pool under nccl_ub, while the exchange keeps views
        of it."""
        Utils.initialize_model_parallel()

        def offload_and_restore(model_chunks):
            for chunk in model_chunks:
                chunk.offload_grad_buffers()
            for chunk in model_chunks:
                chunk.restore_grad_buffers()

        self._compare(
            transport=transport,
            ddp_kwargs=ddp_kwargs,
            after_step=offload_and_restore if ddp_kwargs.get("grad_buffer_offload") else None,
        )

    def test_bitwise_with_full_iteration_grad_copy(self):
        """cuda_graph_impl='full_iteration' makes LayerWise convert gradients through
        GraphedGradCopyFloat16Optimizer, which reads the owners' main_grad views: stage A must
        have filled them before the conversion."""
        from megatron.core.optimizer.layer_wise_optimizer import GraphedGradCopyFloat16Optimizer

        Utils.initialize_model_parallel()
        _, optimizer = self._compare(
            transport="p2p", transformer_kwargs=dict(cuda_graph_impl="full_iteration")
        )
        layerwise = optimizer.chained_optimizers[0]
        assert any(
            isinstance(child, GraphedGradCopyFloat16Optimizer)
            for child in layerwise.chained_optimizers
        )

    def test_a2a_with_nccl_ub_is_rejected(self):
        """The "a2a" exchange receives into scratch tensors that are not registered with
        NCCL, and such an all_to_all_single does not complete: the combination raises."""
        Utils.initialize_model_parallel()
        with pytest.raises(ValueError, match="not supported with nccl_ub"):
            _build(True, transport="a2a", ddp_kwargs=dict(nccl_ub=True))

    def test_fp32_local_accumulation_inside_the_buffer_is_rejected(self):
        Utils.initialize_model_parallel()
        with pytest.raises(ValueError, match="param_name_patterns_for_fp32_local_accumulation"):
            _build(
                True,
                ddp_kwargs=dict(
                    param_name_patterns_for_fp32_local_accumulation=("*layers.0.weight",)
                ),
            )

    def test_without_layerwise_layout_is_rejected(self):
        from megatron.training.training import wrap_model_chunks_with_ddp

        Utils.initialize_model_parallel()
        ddp_config = DistributedDataParallelConfig(layerwise_element_granular_dense=True)
        transformer_config = TransformerConfig(num_attention_heads=1, num_layers=1)
        for kwargs in (
            dict(use_layer_wise_distributed_optimizer=False),
            dict(use_layer_wise_distributed_optimizer=True, use_layer_wise_param_layout=False),
        ):
            with pytest.raises(ValueError, match="layerwise_element_granular_dense"):
                wrap_model_chunks_with_ddp(
                    [_MixedShapes().bfloat16().cuda()], transformer_config, ddp_config, **kwargs
                )

    @pytest.mark.parametrize(
        "transport, clip_grad",
        [
            pytest.param("a2a", 1e-3, id="binding-a2a"),
            pytest.param("p2p", 1e6, id="not_binding-p2p"),
        ],
    )
    def test_grad_norm_and_clip_coefficient(self, transport, clip_grad, monkeypatch):
        """The global grad norm, the clip coefficient and the clipped gradients equal the
        shard-aligned layout's bitwise, with the clip binding and with it not binding."""
        Utils.initialize_model_parallel()
        records = []
        original_clip = megatron_optimizer_module.clip_grad_by_total_norm_fp32

        def recording_clip(parameters, max_norm, total_norm, *args, **kwargs):
            coefficient = max_norm / (total_norm + 1.0e-6)
            original_clip(parameters, max_norm, total_norm, *args, **kwargs)
            grads = [p.grad.clone() for p in parameters if p.grad is not None]
            records[-1].append((max_norm, total_norm, coefficient, grads))

        monkeypatch.setattr(
            megatron_optimizer_module, "clip_grad_by_total_norm_fp32", recording_clip
        )
        kwargs = dict(transport=transport, clip_grad=clip_grad)
        ref_chunks, ref_optimizer = _build(False, **kwargs)
        chunks, optimizer = _build(True, copy_from=ref_chunks, **kwargs)
        _assert_exchange_is_exercised(chunks)

        records.append([])
        ref_norms = _train(ref_chunks, ref_optimizer, steps=2)
        ref_records = records.pop()
        records.append([])
        norms = _train(chunks, optimizer, steps=2)
        eg_records = records.pop()

        assert norms == ref_norms
        assert len(eg_records) == len(ref_records) > 0
        for (max_norm, total_norm, coefficient, grads), ref_record in zip(eg_records, ref_records):
            assert (max_norm, total_norm, coefficient) == ref_record[:3]
            assert (coefficient < 1.0) == (clip_grad < 1.0), "the clip is not in the tested regime"
            assert len(grads) == len(ref_record[3])
            for grad, ref_grad in zip(grads, ref_record[3]):
                assert torch.equal(grad, ref_grad)
        _assert_same_training_state(chunks, optimizer, ref_chunks, ref_optimizer)


@pytest.mark.skipif(WORLD_SIZE < 4, reason="needs data-parallel size 4")
class TestElementGranularPlainReduceScatter:
    """Plain NCCL reduce-scatter at data-parallel size 4: the ring adds the ranks' values of
    an element in an order that depends on the element's slice, so the two layouts may round
    differently. With exactly summable gradients each owner must still see the exact sum."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _reduced_owned_grads(element_granular, transport, grad_fn):
        """Fill every rank's gradients with ``grad_fn(param_index, shape, rank)`` and return
        what this rank's owned params hold after the reduce-scatter (and stage A)."""
        chunks, optimizer = _build(
            element_granular, transport=transport, overlap=False, fp32_accum=False, bucket_size=None
        )
        dp_rank = parallel_state.get_data_parallel_rank()
        params = list(chunks[0].module.parameters())
        chunks[0].zero_grad_buffer()
        for index, param in enumerate(params):
            param.main_grad.copy_(grad_fn(index, param.shape, dp_rank).to(param.main_grad))
        chunks[0].finish_grad_sync()
        owned = {id(p) for p in optimizer.chained_optimizers[0].dp_cp_params_list[dp_rank]}
        reduced = {
            index: param.main_grad.clone()
            for index, param in enumerate(params)
            if id(param) in owned
        }
        assert all(params[index].dim() == 2 for index in reduced)
        if element_granular:
            _assert_exchange_is_exercised(chunks)
        return reduced, params

    @pytest.mark.parametrize("transport", TRANSPORTS)
    def test_plain_rs_dp4_exact_grads_bitwise(self, transport):
        """Gradients that are small integers: every summation order is exact in bf16, so the
        reduced gradient of each owner equals the shard-aligned layout's and the exact sum."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=WORLD_SIZE // 4)
        dp_size = 4

        def integers(index, shape, rank):
            generator = torch.Generator().manual_seed(1000 * index + rank)
            # Multiples of dp_size: the 1/dp gradient scaling before the collective is exact.
            return dp_size * torch.randint(-8, 9, tuple(shape), generator=generator).float()

        ref, _ = self._reduced_owned_grads(False, transport, integers)
        reduced, params = self._reduced_owned_grads(True, transport, integers)
        assert reduced.keys() == ref.keys() and len(reduced) > 0
        for index, grad in reduced.items():
            exact = sum(integers(index, params[index].shape, rank) for rank in range(dp_size))
            exact = (exact / dp_size).to(grad)
            assert torch.equal(grad, exact), f"param {index}: reduced gradient is not the sum"
            assert torch.equal(grad, ref[index]), f"param {index}: layouts differ"


@pytest.mark.skipif(WORLD_SIZE < 4, reason="needs data-parallel size 4")
class TestElementGranularRejections:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_multiple_distributed_optimizer_instances_are_rejected(self):
        Utils.initialize_model_parallel(num_distributed_optimizer_instances=2)
        with pytest.raises(ValueError, match="num_distributed_optimizer_instances"):
            _build(True, fp32_accum=False, ddp_kwargs=dict(num_distributed_optimizer_instances=2))

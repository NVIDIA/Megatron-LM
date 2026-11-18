# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engram communication_backend="static": bit parity with NCCL, CUDA-graph replay, guards."""

import copy
import tempfile
from dataclasses import asdict, fields, replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
from megatron.core.models.deepseek_v41.config import DeepSeekV41Config, EngramConfig
from megatron.core.models.deepseek_v41.engram import Engram, EngramHasher
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
from megatron.core.models.engram import distributed_embedding as embedding_impl
from megatron.core.models.engram import static_exchange
from megatron.core.models.engram.distributed_embedding import (
    EngramRuntime,
    EPShardedMultiTableEmbedding,
    get_contiguous_row_range,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
_GB200 = (
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(not _IS_BLACKWELL, reason='TE MXFP8 needs Blackwell'),
)


def _gb200(function):
    for mark in reversed(_GB200):
        function = mark(function)
    return function


def _layout(vocab_size=211, max_ngram_size=4, n_heads=4, head_dim=256, layers=(0,)):
    """A small layout with the released distinct-prime bucket allocation."""
    args = SimpleNamespace(
        engram_layer_ids=list(layers),
        engram_num_embeddings=[0] * len(layers),
        engram_max_ngram_size=max_ngram_size,
        engram_vocab_size=vocab_size,
        engram_n_heads=n_heads,
        engram_head_dim=head_dim,
    )
    layout = EngramLayout.from_args(args)
    rows = tuple(sum(sum(order) for order in primes) for primes in layout.primes)
    return replace(layout, num_embeddings=rows)


def _column_ranges(layout, index=0):
    primes = [prime for order in layout.primes[index] for prime in order]
    return tuple((sum(primes[:column]), prime) for column, prime in enumerate(primes))


def _hashes(ranges, tokens, pattern, seed, device='cuda'):
    """[1, tokens, columns, 1] IDs inside each column's range; rank-dependent draws.

    ``duplicates`` sends most tokens of every rank to one shared hot row per column (like
    masked/padding tokens) and the rest to a few shared warm rows, giving long segments.
    """
    shared = torch.Generator().manual_seed(seed)
    local = torch.Generator().manual_seed(
        seed * 1009 + (dist.get_rank() if dist.is_initialized() else 0)
    )
    columns = []
    for offset, size in ranges:
        ids = torch.randint(0, size, (tokens,), generator=local)
        if pattern == 'duplicates':
            hot = torch.randint(0, size, (1,), generator=shared)
            warm = torch.randint(0, size, (3,), generator=shared)
            choice = torch.rand(tokens, generator=local)
            ids = torch.where(choice < 0.6, hot, torch.where(choice < 0.9, warm[ids % 3], ids))
        columns.append(ids + offset)
    return torch.stack(columns, -1).view(1, tokens, len(ranges), 1).to(device)


# Hash columns whose ranges share rows: unsorted offsets, columns 0 and 4
# identical, column 2 spanning the table and column 3 a small range at its end.
_SHARED_ROWS = 4001
_SHARED_RANGES = ((1000, 3000), (0, 3000), (0, _SHARED_ROWS), (3990, 11), (1000, 3000))
# (first row, end row, pool size, columns, share of tokens) of the rows sent through
# several columns of the same token.
_SHARED_POOLS = (
    (0, 1000, 8, (1, 2), 0.25),
    (1000, 3000, 16, (0, 1, 2, 4), 0.4),
    (3990, 4000, 2, (0, 2, 3, 4), 0.15),
)


def _shared_row_hashes(tokens, seed, rank=None, device='cuda'):
    """[1, tokens, 5, 1] IDs inside ``_SHARED_RANGES``; most tokens repeat one row in columns.

    Each token picks one pool (or, for 20% of the tokens, none) and sends one of the pool's
    rows, shared by all ranks, through every column of that pool, so owners receive rows
    requested through different columns of different tokens.
    """
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    shared = torch.Generator().manual_seed(seed)
    local = torch.Generator().manual_seed(seed * 1009 + rank)
    columns = [
        torch.randint(offset, offset + size, (tokens,), generator=local)
        for offset, size in _SHARED_RANGES
    ]
    choice = torch.rand(tokens, generator=local)
    low = 0.0
    for first, end, count, members, share in _SHARED_POOLS:
        pool = torch.randint(first, end, (count,), generator=shared)
        rows = pool[torch.randint(0, count, (tokens,), generator=local)]
        picked = (choice >= low) & (choice < low + share)
        low += share
        for column in members:
            columns[column] = torch.where(picked, rows, columns[column])
    return torch.stack(columns, -1).view(1, tokens, len(_SHARED_RANGES), 1).to(device)


def _order_sensitive_gradient(tokens, width, generator):
    """BF16 upstream gradient whose per-row FP32 sums depend on the summation order.

    A quarter of the elements are +-2**20 (exact in BF16). Adding an O(1) value to them in
    FP32 rounds it to a multiple of 1/8; where a row's large values cancel, those rounding
    errors survive the final rounding to BF16, so another order of a row's contributions
    changes the result bits. N(0, 1) BF16 values mostly sum exactly in FP32, which would
    hide the order.
    """
    gradient = torch.randn(tokens, width, device='cuda', dtype=torch.bfloat16, generator=generator)
    large = torch.rand(tokens, width, device='cuda', generator=generator) < 0.25
    signs = torch.randint(0, 2, (tokens, width), device='cuda', generator=generator) * 2 - 1
    return torch.where(large, (signs * 2.0**20).to(torch.bfloat16), gradient)


def _bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _assert_bitwise(actual, expected, what):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, what
    differing = int((_bytes(actual) != _bytes(expected)).sum())
    assert differing == 0, f'{what}: {differing} bytes differ'


class _BitwiseChecks:
    """Collect bitwise comparisons and fail on every rank together, after all collectives.

    A rank that raised at its first mismatch would leave its peers waiting in the next
    all-to-all.
    """

    _MISMATCH = 1 << 62

    def __init__(self):
        self.results = []

    def add(self, actual, expected, what):
        """Record the number of differing bytes (shape or dtype mismatches count as all)."""
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            differing = self._MISMATCH
        else:
            differing = int((_bytes(actual) != _bytes(expected)).sum())
        self.results.append((what, differing))

    def add_all(self, actual, expected, what):
        """Compare two lists of output tensors pairwise; their lengths must match too."""
        count = 0 if len(actual) == len(expected) else self._MISMATCH
        self.results.append((f'{what} tensor count', count))
        for index, (a, e) in enumerate(zip(actual, expected)):
            self.add(a, e, f'{what}[{index}]')

    def assert_equal(self):
        """Fail if any rank recorded a difference."""
        counts = torch.tensor([count for _, count in self.results], dtype=torch.int64)
        counts = counts.cuda()
        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.MAX)
        failures = [
            f'{what}: '
            + ('shape, dtype or count differ' if count >= self._MISMATCH else f'{count} bytes')
            for (what, _), count in zip(self.results, counts.tolist())
            if count
        ]
        assert not failures, '; '.join(failures)


def _embedding_config(params_dtype=torch.bfloat16, **overrides):
    config = SimpleNamespace(
        use_cpu_initialization=False,
        params_dtype=params_dtype,
        perform_initialization=True,
        sequence_parallel=False,
        deterministic_mode=True,
        init_method=lambda t: torch.nn.init.normal_(t, std=0.1),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _copy_table(source, target):
    with torch.no_grad():
        if is_mxfp8tensor(source):
            target._rowwise_data.copy_(source._rowwise_data)
            target._rowwise_scale_inv.copy_(source._rowwise_scale_inv)
        else:
            target.copy_(source)


def _attach_main_grad(parameter):
    parameter.main_grad = torch.zeros(parameter.shape, dtype=torch.float32, device='cuda')
    parameter.grad_added_to_main_grad = False


def _detach_main_grad(parameter):
    for name in ('main_grad', 'grad_added_to_main_grad'):
        if hasattr(parameter, name):
            delattr(parameter, name)
    parameter.grad = None


def _ddp_post_hook(parameter):
    """Megatron DDP's backward post-hook: add param.grad unless already in main_grad."""
    if parameter.grad is not None and not parameter.grad_added_to_main_grad:
        parameter.main_grad.add_(parameter.grad)
    parameter.grad = None
    parameter.grad_added_to_main_grad = False


def _lookup_outputs(model, hashes, quantized):
    """Run one lookup of ``[..., C, 1]`` IDs; return its stored tensors and decoded rows.

    MXFP8 returns the rowwise data and scales, plus the columnwise copy when it exists (it
    does for wgrad); BF16 returns the rows. Decoded rows are ``[tokens, C * D]``.
    """
    columns, width = hashes.shape[-2], model.embedding_dim
    tokens = hashes.numel() // columns
    if quantized:
        activation = model.lookup_quantized(hashes, columns * width)
        assert is_mxfp8tensor(activation)
        outputs = [getattr(activation, name) for name in _MXFP8_STORAGE]
        decoded = activation.dequantize()[:tokens]
        return [output for output in outputs if output is not None], decoded
    rows = model(hashes)
    assert rows.shape == (*hashes.shape, width)
    return [rows], rows.reshape(tokens, columns * width)


def _lookup_and_backward(model, hashes, gradient, quantized):
    """Run one lookup and its backward; return the forward bytes to compare."""
    outputs, decoded = _lookup_outputs(model, hashes, quantized)
    (decoded * gradient).sum().backward()
    return [output.detach().clone() for output in outputs]


# ---------------------------------------------------------------------------- (1) parity


@_gb200
@pytest.mark.parametrize('pattern', ['uniform', 'duplicates'])
@pytest.mark.parametrize('storage', ['mxfp8', 'bf16'])
@pytest.mark.parametrize('ep_size', [1, 2, 4])
def test_static_lookup_matches_nccl_bitwise(ep_size, storage, pattern):
    """Forward bytes and table gradients equal the NCCL backend, with and without ranges.

    Gradients are compared as a dense gradient (no main_grad), through main_grad with fused
    accumulation, and through main_grad when gradient_accumulation_fusion is off (DDP adds).
    The upstream gradient makes every row's sum order-sensitive, and the second microbatch
    passes a strided B=2 view of the IDs.
    """
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        assert pg.ep.size() == ep_size
        layout = _layout()
        ranges = _column_ranges(layout)
        rows, width, quantized = layout.num_embeddings[0], 256, storage == 'mxfp8'
        torch.manual_seed(5)
        config = _embedding_config()

        def build(backend, column_ranges=None, model_config=config):
            return EPShardedMultiTableEmbedding(
                model_config,
                (rows,),
                width,
                config.init_method,
                pg.ep,
                fp8_storage=quantized,
                communication_backend=backend,
                column_ranges=column_ranges,
            )

        reference = build('nccl')
        no_fusion = copy.copy(config)
        no_fusion.gradient_accumulation_fusion = False
        models = {
            'ranges': build('static', ranges),
            'fallback': build('static'),
            'ranges, fusion off': build('static', ranges, no_fusion),
        }
        for model in models.values():
            _copy_table(reference.tables[0].weight, model.tables[0].weight)
            assert model.static_layout.shards == ep_size
            assert set(model.state_dict()) == {'tables.0.weight'}
        layout_widths = models['ranges'].static_layout.widths
        if ep_size > 1:
            # The layout must actually save slots compared with sending every column.
            assert sum(layout_widths) < ep_size * len(ranges)

        tokens = 512 if pattern == 'duplicates' else 192
        generator = torch.Generator(device='cuda').manual_seed(17 + dist.get_rank())
        checks = _BitwiseChecks()
        for mode in ('dense', 'main_grad'):
            for model in (reference, *models.values()):
                weight = model.tables[0].weight
                _detach_main_grad(weight)
                if mode == 'main_grad':
                    _attach_main_grad(weight)
            # Two microbatches accumulate into the same main_grad.
            expected_outputs, actual_outputs = [], {name: [] for name in models}
            for microbatch in range(2):
                hashes = _hashes(ranges, tokens, pattern, seed=31 + microbatch)
                if microbatch:
                    # A strided [B=2, S, C, 1] view of [B, S, layers, C] hashes, as
                    # Engram.lookup passes one layer's columns; same requests in order.
                    batched = hashes.view(2, tokens // 2, len(ranges), 1)
                    hashes = torch.stack((batched.roll(1, 1), batched), 2)[:, :, 1]
                    assert not hashes.is_contiguous()
                gradient = _order_sensitive_gradient(tokens, len(ranges) * width, generator)
                # Signed zeros: rows reached only by these tokens must keep +0.0 sums.
                gradient[:8] = -0.0
                expected_outputs.append(
                    _lookup_and_backward(reference, hashes, gradient, quantized)
                )
                reference_weight = reference.tables[0].weight
                if mode == 'main_grad':
                    _ddp_post_hook(reference_weight)
                for name, model in models.items():
                    actual_outputs[name].append(
                        _lookup_and_backward(model, hashes, gradient, quantized)
                    )
                    weight = model.tables[0].weight
                    if mode == 'main_grad':
                        fused = name != 'ranges, fusion off'
                        assert torch._is_zerotensor(weight.grad) == fused, name
                        assert weight.grad_added_to_main_grad == fused, name
                        _ddp_post_hook(weight)
            for name, model in models.items():
                for microbatch, (actual, expected) in enumerate(
                    zip(actual_outputs[name], expected_outputs)
                ):
                    checks.add_all(actual, expected, f'{name} {mode} mb{microbatch} output')
                weight = model.tables[0].weight
                if mode == 'main_grad':
                    checks.add(
                        weight.main_grad, reference.tables[0].weight.main_grad, f'{name} main_grad'
                    )
                else:
                    checks.add(weight.grad, reference.tables[0].weight.grad, f'{name} grad')
                    weight.grad = None
            reference.tables[0].weight.grad = None
        checks.assert_equal()
        torch.cuda.synchronize()
    finally:
        Utils.destroy_model_parallel()


@_gb200
@pytest.mark.parametrize('storage', ['mxfp8', 'bf16'])
@pytest.mark.parametrize('ep_size', [1, 2, 4])
def test_static_lookup_shared_rows_match_nccl(ep_size, storage):
    """Hash columns whose ranges share rows: forward bytes and gradients equal NCCL.

    The ranges are unsorted, overlap, repeat and span the table, and most tokens request one
    row through several columns. With an order-sensitive upstream gradient, any slot layout
    that changes a row's (source rank, t * C + c) summation order, such as column-major
    owner blocks, changes main_grad and the dense gradient.
    """
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        quantized, width, columns = storage == 'mxfp8', 256, len(_SHARED_RANGES)
        config = _embedding_config()
        torch.manual_seed(7)
        reference, static = (
            EPShardedMultiTableEmbedding(
                config,
                (_SHARED_ROWS,),
                width,
                config.init_method,
                pg.ep,
                fp8_storage=quantized,
                communication_backend=backend,
                column_ranges=ranges,
            )
            for backend, ranges in (('nccl', None), ('static', _SHARED_RANGES))
        )
        _copy_table(reference.tables[0].weight, static.tables[0].weight)
        # Every owner serves several of the columns, so its rows are shared between columns.
        assert static.static_layout.has_column_ranges and min(static.static_layout.widths) >= 2

        tokens = 384
        generator = torch.Generator(device='cuda').manual_seed(11 + dist.get_rank())
        checks = _BitwiseChecks()
        for mode in ('main_grad', 'dense'):
            for model in (reference, static):
                _detach_main_grad(model.tables[0].weight)
                if mode == 'main_grad':
                    _attach_main_grad(model.tables[0].weight)
            # Two microbatches accumulate into the same main_grad / dense gradient.
            for microbatch in range(2):
                hashes = _shared_row_hashes(tokens, seed=60 + microbatch)
                gradient = _order_sensitive_gradient(tokens, columns * width, generator)
                outputs = []
                for model in (static, reference):
                    outputs.append(_lookup_and_backward(model, hashes, gradient, quantized))
                    if mode == 'main_grad':
                        _ddp_post_hook(model.tables[0].weight)
                checks.add_all(*outputs, f'{mode} mb{microbatch} output')
            actual, expected = static.tables[0].weight, reference.tables[0].weight
            if mode == 'main_grad':
                checks.add(actual.main_grad, expected.main_grad, 'main_grad')
            else:
                checks.add(actual.grad, expected.grad, 'dense grad')
        checks.assert_equal()
    finally:
        Utils.destroy_model_parallel()


@_gb200
def test_static_lookup_large_shard_addresses():
    """Owner shards above 8.4M rows: row * 256 offsets exceed int32 in gather and backward."""
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        rows, width = 4 * 8_400_000 + 3, 256
        config = _embedding_config()
        torch.manual_seed(9)
        # No grad: skip the CPU copy of the high-precision initializer of each 2 GB shard.
        with torch.no_grad():
            models = [
                EPShardedMultiTableEmbedding(
                    config,
                    (rows,),
                    width,
                    config.init_method,
                    pg.ep,
                    fp8_storage=True,
                    communication_backend=backend,
                )
                for backend in ('nccl', 'static')
            ]
        _copy_table(models[0].tables[0].weight, models[1].tables[0].weight)
        assert models[1].tables[0].local_num_embeddings > 2**31 // width
        tokens, columns = 256, 4
        generator = torch.Generator(device='cuda').manual_seed(3 + dist.get_rank())
        # Mostly the top rows of each owner's shard, plus a few anywhere in the table.
        owner = torch.randint(0, 4, (tokens, columns), device='cuda', generator=generator)
        starts = torch.tensor(
            [get_contiguous_row_range(rows, rank, 4)[0] for rank in range(4)], device='cuda'
        )
        # Every owner holds at least 8_400_000 local rows.
        local = 8_399_999 - torch.randint(
            0, 20_000, (tokens, columns), device='cuda', generator=generator
        )
        hashes = (starts[owner] + local).view(1, tokens, columns, 1)
        hashes[0, :16] = torch.randint(
            0, rows, (16, columns, 1), device='cuda', generator=generator
        )
        gradient = torch.randn(
            tokens, columns * width, device='cuda', dtype=torch.bfloat16, generator=generator
        )
        results = []
        for model in models:
            weight = model.tables[0].weight
            _attach_main_grad(weight)
            results.append(_lookup_and_backward(model, hashes, gradient, True))
            _ddp_post_hook(weight)
        for index, (actual, expected) in enumerate(zip(results[1], results[0])):
            _assert_bitwise(actual, expected, f'output[{index}]')
        _assert_bitwise(
            models[1].tables[0].weight.main_grad, models[0].tables[0].weight.main_grad, 'main_grad'
        )
        assert torch.count_nonzero(models[1].tables[0].weight.main_grad[8_380_000:8_400_000])
    finally:
        Utils.destroy_model_parallel()


@_gb200
@pytest.mark.parametrize('use_layout', [False, True])
@pytest.mark.parametrize('storage', ['mxfp8', 'bf16'])
def test_static_lookup_with_ddp_overlapped_param_gather(storage, use_layout):
    """EP2 x replica-DP2 training with Megatron DDP, overlapped param gather and LayerWise
    Sinkhorn: the static backend publishes params before reading rows (ensure_params_ready),
    its ZeroTensor passes DDP's hooks, and three steps match the NCCL backend bit for bit."""
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import (
        OptimizerConfig,
        _get_param_groups,
        get_standard_config_overrides,
    )
    from megatron.core.optimizer.emerging_optimizers import Sinkhorn, _muon_param_overrides_factory
    from megatron.core.optimizer.layer_wise_optimizer import (
        LayerWiseDistributedOptimizer,
        tag_params_for_buffer_routing,
    )
    from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

    Utils.initialize_distributed()
    if dist.get_world_size() != 4:
        pytest.skip('requires EP2 and two expert-DP replicas')
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        quantized = storage == 'mxfp8'
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=False,
            gradient_accumulation_fusion=True,
            deterministic_mode=True,
            **(dict(fp8='e4m3', fp8_recipe=Fp8Recipe.mxfp8, fp8_param=True) if quantized else {}),
        )
        ranges = ((0, 97), (97, 101))
        tokens = 160
        generator = torch.Generator(device='cuda').manual_seed(3 + dist.get_rank())
        # The last step repeats the first IDs, so its rows must show the two updates.
        steps = [
            (
                _hashes(ranges, tokens, pattern, seed=seed),
                torch.randn(
                    tokens, 2 * 32, device='cuda', dtype=torch.bfloat16, generator=generator
                ),
            )
            for seed, pattern in ((50, 'uniform'), (51, 'duplicates'), (50, 'uniform'))
        ]
        runs = {}
        initial = None
        for backend in ('nccl', 'static'):
            torch.manual_seed(123)
            model = EPShardedMultiTableEmbedding(
                config,
                (198,),
                32,
                config.init_method,
                pg.ep,
                pg.tp,
                pg.expt_dp,
                fp8_storage=quantized,
                communication_backend=backend,
                column_ranges=ranges if backend == 'static' else None,
            ).cuda()
            convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
            weight = model.tables[0].weight
            if initial is None:
                initial = [b.clone() for b in _parameter_bytes(weight)]
            else:
                with torch.no_grad():
                    for target, source in zip(_parameter_bytes(weight), initial):
                        target.copy_(source)
            weight.lr_mult = 5.0
            tag_params_for_buffer_routing([model], use_sinkhorn_for_embeddings=True)
            optimizer_config = OptimizerConfig(
                optimizer='muon',
                bf16=True,
                lr=0.02,
                clip_grad=0.0,
                muon_use_sinkhorn=True,
                use_distributed_optimizer=True,
                use_layer_wise_param_layout=use_layout,
            )
            overrides = get_standard_config_overrides(optimizer_config)
            overrides.update(_muon_param_overrides_factory())
            groups = _get_param_groups([model], optimizer_config, overrides)
            assert all(group['optimizer'] == 'sinkhorn' for group in groups if group['params'])
            ddp = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    use_distributed_optimizer=True,
                    use_layer_wise_param_layout=use_layout,
                    fp8_param_gather=quantized,
                    reuse_grad_buf_for_mxfp8_param_ag=quantized,
                ),
                model,
                pg_collection=pg,
            )
            optimizer = LayerWiseDistributedOptimizer(
                [Sinkhorn([{'params': [weight], 'is_expert_parallel': True}], lr=0.02)],
                optimizer_config,
                pg_collection=pg,
                model_chunks=[ddp],
            )
            records = []
            for hashes, gradient in steps:
                ddp.zero_grad_buffer()
                optimizer.zero_grad()
                optimizer.prepare_model_params_for_param_sync()
                # Neither lookup calls table.forward(), where DDP installs its hooks.
                outputs = _lookup_and_backward(model, hashes, gradient, quantized)
                if backend == 'static':
                    assert weight.grad is None and weight.grad_added_to_main_grad
                ddp.finish_grad_sync()
                records.append((outputs, weight.main_grad.clone()))
                assert not optimizer.prepare_grads()
                assert optimizer.step_with_ready_grads()
            runs[backend] = records
            del optimizer, ddp
        for step, ((actual, actual_grad), (expected, expected_grad)) in enumerate(
            zip(runs['static'], runs['nccl'])
        ):
            for index, (a, e) in enumerate(zip(actual, expected)):
                _assert_bitwise(a, e, f'step {step} output[{index}]')
            _assert_bitwise(actual_grad, expected_grad, f'step {step} reduced main_grad')
        # The steps changed the table: later lookups read the published update.
        assert not torch.equal(runs['static'][2][0][0], runs['static'][0][0][0])
    finally:
        Utils.destroy_model_parallel()


# ------------------------------------------------------------------------- (2) CUDA graphs


@_gb200
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('storage', ['mxfp8', 'bf16'])
@pytest.mark.parametrize('ep_size', [1, 2, 4])
def test_static_lookup_cuda_graph_replay(ep_size, storage, training):
    """Capture once; replays on new IDs equal eager runs bit for bit.

    Training captures forward + backward into main_grad; evaluation captures the no-grad
    forward that a full-iteration validation graph runs.
    """
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        layout = _layout()
        ranges = _column_ranges(layout)
        quantized = storage == 'mxfp8'
        config = _embedding_config(gradient_accumulation_fusion=True)
        torch.manual_seed(11)
        model = EPShardedMultiTableEmbedding(
            config,
            (layout.num_embeddings[0],),
            256,
            config.init_method,
            pg.ep,
            fp8_storage=quantized,
            communication_backend='static',
            column_ranges=ranges,
        )
        weight = model.tables[0].weight
        _attach_main_grad(weight)
        tokens = 256
        id_sets = [
            _hashes(ranges, tokens, pattern, seed=100 + index)
            for index, pattern in enumerate(('uniform', 'duplicates', 'uniform', 'duplicates'))
        ]
        generator = torch.Generator(device='cuda').manual_seed(29 + dist.get_rank())
        static_ids = id_sets[0].clone()
        static_grad = torch.randn(
            tokens, len(ranges) * 256, device='cuda', dtype=torch.bfloat16, generator=generator
        )

        def step():
            if training:
                return _lookup_and_backward(model, static_ids, static_grad, quantized)
            with torch.no_grad():
                outputs, _ = _lookup_outputs(model, static_ids, quantized)
            return outputs

        def run_eager(ids):
            static_ids.copy_(ids)
            weight.main_grad.zero_()
            outputs = [output.clone() for output in step()]
            if training:
                assert torch._is_zerotensor(weight.grad) and weight.grad_added_to_main_grad
            else:
                # Evaluation needs no wgrad copy and leaves the gradients alone.
                assert len(outputs) == (2 if quantized else 1)
                assert weight.grad is None and not weight.grad_added_to_main_grad
            _ddp_post_hook(weight)
            return outputs, weight.main_grad.clone()

        expected = [run_eager(ids) for ids in id_sets]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                step()
                _ddp_post_hook(weight)
        torch.cuda.current_stream().wait_stream(stream)
        static_ids.copy_(id_sets[0])
        weight.main_grad.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode='thread_local'):
            captured = step()
        _ddp_post_hook(weight)
        # Replays run all-to-alls: compare on every rank only after the last one.
        checks = _BitwiseChecks()
        for index in (1, 2, 3):
            static_ids.copy_(id_sets[index])
            weight.main_grad.zero_()
            graph.replay()
            torch.cuda.synchronize()
            outputs, main_grad = expected[index]
            checks.add_all(captured, outputs, f'replay {index} output')
            checks.add(weight.main_grad, main_grad, f'replay {index} main_grad')
        del graph
        torch.cuda.synchronize()
        checks.assert_equal()
    finally:
        Utils.destroy_model_parallel()


# ----------------------------------------------------------------- (3) Engram module level


def _engram_options(backend, vocab_size=61):
    layout = _layout(vocab_size, max_ngram_size=3, n_heads=4, head_dim=32, layers=(0, 3))
    return layout, EngramConfig(
        layer_ids=[0, 3],
        num_embeddings=list(layout.num_embeddings),
        max_ngram_size=3,
        vocab_size=vocab_size,
        n_heads=4,
        head_dim=32,
        pad_token_id=0,
        compressed_vocab_size=50,
        communication_backend=backend,
    )


def _engram_config(fp8_param, fusion, options):
    config = TransformerConfig(
        num_layers=8,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        mhc_num_residual_streams=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=fusion,
        fp8='e4m3',
        fp8_recipe=Fp8Recipe.mxfp8,
        fp8_param=fp8_param,
        deterministic_mode=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    config.engram_config = options
    return config


_MXFP8_STORAGE = (
    '_rowwise_data',
    '_rowwise_scale_inv',
    '_columnwise_data',
    '_columnwise_scale_inv',
)


def _parameter_bytes(parameter):
    """Every stored byte, including MXFP8 scale padding that quantization leaves unset."""
    if is_mxfp8tensor(parameter):
        return [
            getattr(parameter, name)
            for name in _MXFP8_STORAGE
            if getattr(parameter, name, None) is not None
        ]
    return [parameter.detach()]


def _gradient(parameter):
    main_grad = getattr(parameter, 'main_grad', None)
    return main_grad if main_grad is not None else parameter.grad


@_gb200
@pytest.mark.parametrize('fusion', [False, True])
@pytest.mark.parametrize('fp8_param', [True, False])
@pytest.mark.parametrize('ep_size', [1, 2, 4])
def test_engram_static_matches_nccl(ep_size, fp8_param, fusion):
    """Engram.lookup + forward + backward through the real hasher layout, bit for bit."""
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        layout, options = _engram_options('nccl')
        hasher = EngramHasher(options, torch.arange(50)).cuda()
        modules = {}
        for backend in ('nccl', 'static'):
            config = _engram_config(
                fp8_param, fusion, replace(options, communication_backend=backend)
            )
            torch.manual_seed(123)
            model_parallel_cuda_manual_seed(123)
            # Logical block 3 is the layout's second Engram layer (hash index 1).
            modules[backend] = Engram(config, hasher.layout, 3, pg).cuda()
        reference, static = modules['nccl'], modules['static']
        with torch.no_grad():
            # Same values from the same seeds; copy the whole storage so that unset MXFP8
            # scale padding cannot differ between the two modules either.
            for expected, actual in zip(reference.parameters(), static.parameters()):
                for e, a in zip(_parameter_bytes(expected), _parameter_bytes(actual)):
                    a.copy_(e)
        assert static.hash_index == 1 and static.embed.communication.backend == 'static'
        assert static.embed.column_ranges == _column_ranges(layout, 1)
        assert static.embed.static_layout.has_column_ranges
        for (name, expected), actual in zip(reference.named_parameters(), static.parameters()):
            for e, a in zip(_parameter_bytes(expected), _parameter_bytes(actual)):
                _assert_bitwise(a, e, f'initial {name}')
        assert is_mxfp8tensor(static.embed.tables[0].weight) == fp8_param
        if fusion:
            for module in modules.values():
                for parameter in (module.embed.tables[0].weight, module.wkv.weight):
                    _attach_main_grad(parameter)

        tokens = 96
        generator = torch.Generator(device='cuda').manual_seed(7 + dist.get_rank())
        for pattern in ('uniform', 'duplicates'):
            if pattern == 'uniform':
                ids = torch.randint(0, 50, (1, tokens), device='cuda', generator=generator)
            else:
                # Repeated n-grams hash to the same rows in every column.
                ids = torch.randint(0, 3, (1, tokens), device='cuda', generator=generator)
            hashes = hasher(ids)
            hidden = torch.randn(
                tokens, 1, 128, device='cuda', dtype=torch.bfloat16, generator=generator
            )
            upstream = torch.randn(
                tokens, 1, 128, device='cuda', dtype=torch.bfloat16, generator=generator
            )
            results = {}
            for backend, module in modules.items():
                for parameter in module.parameters():
                    if not hasattr(parameter, 'main_grad'):
                        parameter.grad = None
                    else:
                        parameter.main_grad.zero_()
                inputs = hidden.clone().requires_grad_()
                with get_fp8_context(module.config):
                    embeddings = module.lookup(hashes)
                    output = module(inputs, embeddings=embeddings)
                output.backward(upstream)
                if fusion:
                    for parameter in (module.embed.tables[0].weight, module.wkv.weight):
                        _ddp_post_hook(parameter)
                results[backend] = (
                    output.detach().clone(),
                    inputs.grad.clone(),
                    {name: _gradient(p).clone() for name, p in module.named_parameters()},
                )
            (out_s, dx_s, grads_s), (out_r, dx_r, grads_r) = results['static'], results['nccl']
            _assert_bitwise(out_s, out_r, f'{pattern} output')
            _assert_bitwise(dx_s, dx_r, f'{pattern} hidden grad')
            assert grads_s.keys() == grads_r.keys()
            for name in grads_r:
                _assert_bitwise(grads_s[name], grads_r[name], f'{pattern} grad {name}')
            table = 'embed.tables.0.weight'
            assert torch.count_nonzero(grads_r[table]) > 0
            for name in ('wkv.weight', 'q_weight', 'k_weight'):
                assert torch.count_nonzero(grads_r[name]) > 0, name
    finally:
        Utils.destroy_model_parallel()


# ------------------------------------------------------------------- kernels vs reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('weight_dtype', [torch.bfloat16, torch.float32])
def test_static_accumulate_matches_embedding_row_gradients(weight_dtype):
    """Sequential per-row FP32 sums with hot rows, signed zeros and empty owners."""
    pytest.importorskip('triton')
    Utils.initialize_distributed()
    generator = torch.Generator(device='cuda').manual_seed(41)
    rows, width = 3001, 256
    # Segment lengths up to 65536: one hot row, a Zipf-like tail, unowned (-1) slots.
    ids = torch.cat(
        (
            torch.full((65536,), 17, device='cuda'),
            (torch.rand(40000, device='cuda', generator=generator) ** 4 * 2900).long(),
            torch.full((5000,), -1, device='cuda'),
            torch.tensor([2999, 3000], device='cuda'),
        )
    )
    ids = ids[torch.randperm(ids.numel(), device='cuda', generator=generator)]
    grads = torch.randn(ids.numel(), width, device='cuda', generator=generator)
    grads *= torch.logspace(-6, 6, ids.numel(), device='cuda')[:, None]
    grads = grads.to(torch.bfloat16 if weight_dtype == torch.bfloat16 else torch.float32)
    grads[ids == 2999] = -0.0
    garbage = torch.full_like(grads[0], float('nan'))
    grads[ids < 0] = garbage
    keys = torch.where(ids >= 0, ids, static_exchange.SENTINEL)
    sorted_keys, order = torch.sort(keys, stable=True)
    owned = ids >= 0
    expected = embedding_impl._embedding_row_gradients(
        ids[owned], grads[owned], (rows, width), weight_dtype
    )
    for use_fused in (True, False):
        dense = torch.zeros(rows, width, device='cuda', dtype=weight_dtype)
        static_exchange.static_accumulate(
            dense, sorted_keys, order, grads, weight_dtype, accumulate=False, use_fused=use_fused
        )
        _assert_bitwise(dense, expected, f'dense fused={use_fused}')
        main_grad = torch.randn(rows, width, device='cuda', generator=generator)
        reference = main_grad.clone().add_(expected)
        static_exchange.static_accumulate(
            main_grad, sorted_keys, order, grads, weight_dtype, accumulate=True, use_fused=use_fused
        )
        _assert_bitwise(main_grad, reference, f'main_grad fused={use_fused}')
    assert torch.equal(expected[2999], torch.zeros_like(expected[2999]))
    assert not torch.signbit(expected[2999]).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('ranges', [True, False])
@pytest.mark.parametrize('shards', [1, 3, 8])
def test_static_pack_gather_scatter_match_reference(shards, ranges):
    """Fused pack/gather/scatter equal the torch reference; every slot is accounted for."""
    pytest.importorskip('triton')
    Utils.initialize_distributed()
    layout_ranges = _column_ranges(_layout()) if ranges else None
    rows = _layout().num_embeddings[0]
    generator = torch.Generator(device='cuda').manual_seed(shards)
    tokens, width = 77, 64
    for rank in range(shards):
        layout = static_exchange.StaticLookupLayout(
            rows,
            [get_contiguous_row_range(rows, k, shards) for k in range(shards)],
            rank,
            layout_ranges,
        )
        metadata = layout.metadata().cuda()
        if ranges:
            addresses = _hashes(layout_ranges, tokens, 'duplicates', seed=rank).view(tokens, -1)
        else:
            addresses = torch.randint(0, rows, (tokens * 3, 1), device='cuda', generator=generator)
        results = [
            static_exchange.static_pack(
                addresses, layout, metadata, write_inverse=True, use_fused=use_fused
            )
            for use_fused in (True, False)
        ]
        for actual, expected in zip(results[0], results[1]):
            assert torch.equal(actual, expected)
        send_ids, inverse, violations = results[0]
        assert int(violations) == 0
        # Each request owns exactly the slot its inverse names, holding its local row.
        count = addresses.shape[0]
        owners = [
            next(k for k in range(shards) if get_contiguous_row_range(rows, k, shards)[1] > a)
            for a in addresses.flatten().tolist()
        ]
        local = addresses.flatten() - torch.tensor(
            [get_contiguous_row_range(rows, k, shards)[0] for k in owners], device='cuda'
        )
        assert torch.equal(send_ids[inverse], local)
        assert int((send_ids >= 0).sum()) == addresses.numel()
        assert send_ids.numel() == count * layout.slots
        # Out-of-range IDs are counted, never written outside the buffers.
        if ranges:
            bad = addresses.clone()
            bad[3, 1] = layout_ranges[0][0]
            bad[5, 0] = rows + 5
            for use_fused in (True, False):
                _, _, violations = static_exchange.static_pack(
                    bad, layout, metadata, write_inverse=True, use_fused=use_fused
                )
                assert int(violations) == 2

        table = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16, generator=generator)
        quantized = torch.randint(
            0, 256, (rows, width), device='cuda', dtype=torch.uint8, generator=generator
        )
        scales = torch.randint(
            0, 256, (rows + 31, 4), device='cuda', dtype=torch.uint8, generator=generator
        )
        received = torch.cat((send_ids, torch.full((5,), -1, device='cuda')))
        owned = received >= 0
        for data, scale in ((table, None), (quantized, scales)):
            gathered = [
                static_exchange.static_gather(
                    received, data, scale, width, write_keys=True, use_fused=use_fused
                )
                for use_fused in (True, False)
            ]
            assert torch.equal(gathered[0][0][owned], gathered[1][0][owned])
            assert torch.equal(gathered[0][1], gathered[1][1])
            assert torch.equal(
                gathered[0][1][~owned], torch.full_like(received[~owned], static_exchange.SENTINEL)
            )
        grads = torch.randn(
            addresses.numel(), width, device='cuda', dtype=torch.bfloat16, generator=generator
        )
        scattered = [
            static_exchange.static_scatter(grads, inverse, send_ids.numel(), use_fused=use_fused)
            for use_fused in (True, False)
        ]
        assert torch.equal(scattered[0][inverse], grads)
        assert torch.equal(scattered[0][inverse], scattered[1][inverse])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('shards', [2, 3, 4, 8])
def test_static_owner_order_with_shared_rows(shards):
    """Each owner visits a row's requests in NCCL's (source rank, t * C + c) order.

    NCCL delivers requests by source rank, then original index ``t * C + c``, and
    ``_embedding_row_gradients`` sums every row in that order. The static owner sorts its
    received slots stably by row, so the slot layout must keep that order for any data.
    Token-major blocks ``[r][t][j]`` do; with columns that share rows,
    column-major blocks ``[r][j][t]`` do not, which the negative control confirms for this
    data. Checked for the fused and the reference pack.
    """
    pytest.importorskip('triton')
    Utils.initialize_distributed()
    rows, columns, tokens = _SHARED_ROWS, len(_SHARED_RANGES), 64
    partition = [get_contiguous_row_range(rows, k, shards) for k in range(shards)]
    layout = static_exchange.StaticLookupLayout(rows, partition, 0, _SHARED_RANGES)
    metadata = layout.metadata().cuda()
    # Requests [source rank, t * C + c] of every requester.
    requests = torch.cat(
        [_shared_row_hashes(tokens, seed=5, rank=rank) for rank in range(shards)]
    ).view(shards, tokens * columns)
    sentinel = static_exchange.SENTINEL

    def visits(received, width, owner_columns, token_major):
        """(row, source rank, t * C + c) of every owned slot, in the owner's sorted order."""
        keys = torch.where(received >= 0, received, sentinel)
        sorted_keys, order = torch.sort(keys, stable=True)
        owned = sorted_keys != sentinel
        source, slot = order[owned] // (tokens * width), order[owned] % (tokens * width)
        if token_major:
            token, index = slot // width, slot % width
        else:
            index, token = slot // tokens, slot % tokens
        return torch.stack((sorted_keys[owned], source, token * columns + owner_columns[index]))

    column_major_differs = False
    for use_fused in (True, False):
        sent = [
            static_exchange.static_pack(
                requests[rank].view(tokens, columns),
                layout,
                metadata,
                write_inverse=True,
                use_fused=use_fused,
            )[0]
            for rank in range(shards)
        ]
        for owner, (start, end) in enumerate(partition):
            width, offset = layout.widths[owner], layout.offsets[owner]
            owner_columns = torch.tensor(layout.owner_columns[owner], device='cuda')
            # NCCL's arrival order (source rank, then t * C + c), stably sorted by row.
            source, index = ((requests >= start) & (requests < end)).nonzero(as_tuple=True)
            local = requests[source, index] - start
            ordered = torch.sort(local, stable=True).indices
            expected = torch.stack((local[ordered], source[ordered], index[ordered]))
            # The owner's receive buffer: every requester's static block for this owner.
            blocks = torch.stack([ids[tokens * offset : tokens * (offset + width)] for ids in sent])
            actual = visits(blocks.reshape(-1), width, owner_columns, token_major=True)
            assert torch.equal(actual, expected), f'owner {owner} of {shards}, fused={use_fused}'
            column_major = blocks.view(shards, tokens, width).transpose(1, 2).reshape(-1)
            column_major_differs |= not torch.equal(
                visits(column_major, width, owner_columns, token_major=False), expected
            )
    assert column_major_differs, 'the data does not distinguish the two block layouts'


# --------------------------------------------------------------------------- (4) config


def _engram_options_dict(**overrides):
    values = dict(
        layer_ids=[0],
        num_embeddings=[96],
        max_ngram_size=2,
        vocab_size=17,
        n_heads=2,
        head_dim=4,
        pad_token_id=2,
        compressed_vocab_size=128,
    )
    values.update(overrides)
    return values


def test_engram_config_static_backend():
    assert (
        EngramConfig(**_engram_options_dict(communication_backend='static')).communication_backend
        == 'static'
    )
    for field in ('prefetch', 'defer_grad_transfer'):
        with pytest.raises(ValueError, match='static'):
            EngramConfig(**_engram_options_dict(communication_backend='static', **{field: True}))
        # The runtime options stay valid with the host-planned backends.
        assert getattr(EngramConfig(**_engram_options_dict(**{field: True})), field)
    with pytest.raises(ValueError, match='static'):
        EngramConfig(**_engram_options_dict(communication_backend='device'))


@pytest.mark.parametrize('backend', ['nccl', 'hybridep', 'symmetric', 'static'])
def test_full_iteration_graphs_need_static_engram(backend):
    from tests.unit_tests.models.test_v41_state_integration import _config

    cfg = _config()
    values = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}
    values.update(
        cuda_graph_impl='full_iteration',
        cuda_graph_modules=[],
        engram_config=EngramConfig(**_engram_options_dict(communication_backend=backend)),
    )
    if backend != 'static':
        with pytest.raises(NotImplementedError, match="cuda_graph_impl='full_iteration'"):
            DeepSeekV41Config(**values)
        return
    assert values['cuda_graph_warmup_steps'] >= 1
    config = DeepSeekV41Config(**values)
    assert config.cuda_graph_impl == 'full_iteration'
    assert config.engram_config.communication_backend == 'static'
    values['cuda_graph_warmup_steps'] = 0
    with pytest.raises(ValueError, match='cuda_graph_warmup_steps'):
        DeepSeekV41Config(**values)


def _cpu_config():
    config = _embedding_config(params_dtype=torch.float32, use_cpu_initialization=True)
    config.init_method = lambda t: torch.nn.init.normal_(t, std=0.1)
    return config


def test_static_backend_guards_and_cpu_reference():
    """Plumbing guards, state-dict hygiene and the reference path on CPU (one shard)."""
    config = _cpu_config()
    init = config.init_method
    with pytest.raises(ValueError, match='exactly one table'):
        EPShardedMultiTableEmbedding(config, (7, 5), 4, init, communication_backend='static')
    with pytest.raises(ValueError, match='inside'):
        EPShardedMultiTableEmbedding(
            config, (12,), 4, init, communication_backend='static', column_ranges=((0, 5), (5, 8))
        )
    with pytest.raises(ValueError, match='single table'):
        EPShardedMultiTableEmbedding(config, (7, 5), 4, init, column_ranges=((0, 7),))
    model = EPShardedMultiTableEmbedding(
        config, (12,), 4, init, communication_backend='static', column_ranges=((0, 5), (5, 7))
    )
    assert set(model.state_dict()) == {'tables.0.weight'}
    assert model.static_layout.widths == (2,) and model.static_layout.slots == 2
    with pytest.raises(RuntimeError, match='static'):
        model.communication.exchange_counts(torch.zeros(1, dtype=torch.long))
    with pytest.raises(RuntimeError, match='static'):
        model.communication.exchange(torch.zeros(1), [1], [1])
    with pytest.raises(ValueError, match='static'):
        EngramRuntime(1).attach(model)
    with pytest.raises(ValueError, match='columns'):
        model(torch.zeros(4, 3, 1, dtype=torch.long))

    weight = model.tables[0].weight
    ids = torch.tensor([[[0], [5]], [[4], [11]], [[0], [5]], [[2], [5]]])
    output = model(ids)
    torch.testing.assert_close(output, weight.detach()[ids], rtol=0, atol=0)
    gradient = torch.randn(output.shape)
    gradient[0] = -0.0
    output.backward(gradient)
    expected = embedding_impl._embedding_row_gradients(
        ids.flatten(), gradient.reshape(-1, 4), tuple(weight.shape), weight.dtype
    )
    _assert_bitwise(weight.grad, expected, 'CPU dense gradient')
    with pytest.raises(RuntimeError, match='bucket range'):
        model(torch.tensor([[[5], [5]]]))  # column 0 covers rows [0, 5)


def _gloo_static_worker(rank, path):
    """EP2 on CPU: reference path parity with NCCL-style Gloo routing, and the token check."""
    dist.init_process_group(
        'gloo', init_method='file://' + path, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        config = _cpu_config()
        ranges = ((0, 5), (5, 7), (12, 3))
        models = [
            EPShardedMultiTableEmbedding(
                config,
                (15,),
                4,
                config.init_method,
                dist.group.WORLD,
                communication_backend=backend,
                column_ranges=column_ranges,
            )
            for backend, column_ranges in (('nccl', None), ('static', ranges), ('static', None))
        ]
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict())
        # Rows [0, 8) and [8, 15): column 1 spans both owners.
        assert models[1].static_layout.owner_columns == ((0, 1), (1, 2))
        generator = torch.Generator().manual_seed(rank)
        ids = torch.stack(
            [
                torch.randint(offset, offset + size, (6,), generator=generator)
                for offset, size in ranges
            ],
            -1,
        ).unsqueeze(-1)
        gradient = torch.randn(6, 3, 1, 4, generator=generator)
        outputs = []
        for model in models:
            output = model(ids)
            output.backward(gradient)
            outputs.append(output.detach())
        for model, output in zip(models[1:], outputs[1:]):
            _assert_bitwise(output, outputs[0], 'gloo static output')
            _assert_bitwise(
                model.tables[0].weight.grad, models[0].tables[0].weight.grad, 'gloo grad'
            )
        # Different token counts on the two ranks must fail before any exchange.
        with pytest.raises(RuntimeError, match='same token count'):
            models[1](ids[: 4 + rank])
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_static_lookup_and_token_check():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_gloo_static_worker, args=(directory + '/rendezvous',), nprocs=2, join=True)

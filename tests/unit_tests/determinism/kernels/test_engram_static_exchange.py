# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the static Engram exchange kernels (``static_exchange.py``).

Pack, owner gather and gradient scatter write unique destinations. The owner accumulation
gives each sorted row segment exactly one program, which adds its rows sequentially; hot rows
with tens of thousands of contributions make that loop long. Replays run under side-stream
contention, standalone and through the complete EP lookup with its all-to-alls.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
from megatron.core.models.engram.distributed_embedding import (
    EPShardedMultiTableEmbedding,
    get_contiguous_row_range,
)
from megatron.core.models.engram.static_exchange import (
    HAVE_TRITON,
    SENTINEL,
    StaticLookupLayout,
    static_accumulate,
    static_gather,
    static_pack,
    static_scatter,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason='needs a GPU and Triton'
)

_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _proxy_like_ranges():
    """24 hash columns (3 orders x 8 heads) of distinct primes, as in V4.1."""
    layout = EngramLayout.from_args(
        SimpleNamespace(
            engram_layer_ids=[0],
            engram_num_embeddings=[0],
            engram_max_ngram_size=4,
            engram_vocab_size=997,
            engram_n_heads=8,
            engram_head_dim=256,
        )
    )
    primes = [prime for order in layout.primes[0] for prime in order]
    return tuple((sum(primes[:column]), prime) for column, prime in enumerate(primes))


def _addresses(ranges, tokens, generator):
    """In-range IDs; most tokens share one hot row per column (like padding tokens)."""
    columns = []
    for offset, size in ranges:
        ids = torch.randint(0, size, (tokens,), device='cuda', generator=generator)
        hot = torch.rand(tokens, device='cuda', generator=generator) < 0.5
        columns.append(torch.where(hot, size // 2, ids) + offset)
    return torch.stack(columns, -1)


def _layout(ranges, shards, rank):
    rows = sum(size for _, size in ranges)
    return StaticLookupLayout(
        rows, [get_contiguous_row_range(rows, k, shards) for k in range(shards)], rank, ranges
    )


@pytest.mark.parametrize('shards', [1, 8])
def test_engram_static_pack_replays(shards):
    seeded(7)
    ranges = _proxy_like_ranges()
    layout = _layout(ranges, shards, rank=shards // 2)
    metadata = layout.metadata().cuda()
    generator = torch.Generator(device='cuda').manual_seed(3)
    # A strided [T, C] view, as Engram passes one layer of [B, S, layers, C] hashes.
    hashes = torch.stack((_addresses(ranges, 4096, generator),) * 2, 1)[:, 1]

    def pack(addresses):
        return static_pack(addresses, layout, metadata, write_inverse=shards > 1)

    reference, _ = assert_replays_bit_exact(
        pack, (hashes,), backward=False, contention=True, what='Engram static pack'
    )
    assert int(reference['out[2]']) == 0


@pytest.mark.parametrize('storage', ['mxfp8', 'bf16'])
def test_engram_static_gather_replays(storage):
    seeded(11)
    rows, width = 50_000, 256
    generator = torch.Generator(device='cuda').manual_seed(5)
    ids = torch.randint(0, rows, (131_072,), device='cuda', generator=generator)
    ids[torch.rand(ids.shape, device='cuda', generator=generator) < 0.25] = -1
    ids[:4096] = rows - 1
    owned = ids >= 0
    if storage == 'mxfp8':
        data = torch.randint(0, 256, (rows, width), dtype=torch.uint8, device='cuda')
        scales = torch.randint(0, 256, (rows + 48, width // 32), dtype=torch.uint8, device='cuda')
    else:
        data = torch.randn(rows, width, dtype=torch.bfloat16, device='cuda')
        scales = None

    def gather(received):
        payload, keys = static_gather(received, data, scales, width, write_keys=True)
        # Unowned slots are never written; only owned rows are defined.
        return payload[owned], keys

    reference, _ = assert_replays_bit_exact(
        gather, (ids,), backward=False, contention=True, what=f'Engram static {storage} gather'
    )
    assert torch.equal(reference['out[1]'][~owned], torch.full_like(ids[~owned], SENTINEL))


def test_engram_static_scatter_replays():
    seeded(13)
    tokens, columns, width = 4096, 24, 256
    generator = torch.Generator(device='cuda').manual_seed(9)
    slots = tokens * 31
    inverse = torch.randperm(slots, device='cuda', generator=generator)[: tokens * columns]
    grads = torch.randn(tokens * columns, width, dtype=torch.bfloat16, device='cuda')

    def scatter(values):
        return static_scatter(values, inverse, slots)[inverse]

    assert_replays_bit_exact(
        scatter, (grads,), backward=False, contention=True, what='Engram static gradient scatter'
    )


@pytest.mark.parametrize('accumulate', [True, False])
@pytest.mark.parametrize('weight_dtype', [torch.bfloat16, torch.float32])
def test_engram_static_accumulate_replays(weight_dtype, accumulate):
    seeded(17)
    rows, width = 20_000, 256
    generator = torch.Generator(device='cuda').manual_seed(21)
    ids = torch.cat(
        (
            torch.full((40_000,), 7, device='cuda'),  # one hot row
            (torch.rand(90_000, device='cuda', generator=generator) ** 3 * rows).long(),
            torch.full((1_000,), -1, device='cuda'),  # unowned slots
        )
    )
    ids = ids[torch.randperm(ids.numel(), device='cuda', generator=generator)]
    grads = torch.randn(ids.numel(), width, device='cuda', generator=generator)
    grads = grads.to(weight_dtype)
    grads[ids < 0] = float('nan')
    sorted_keys, order = torch.sort(torch.where(ids >= 0, ids, SENTINEL), stable=True)
    initial = torch.randn(rows, width, device='cuda') if accumulate else None

    def reduce(values):
        if accumulate:
            target = initial.clone()
        else:
            target = torch.zeros(rows, width, dtype=weight_dtype, device='cuda')
        static_accumulate(target, sorted_keys, order, values, weight_dtype, accumulate=accumulate)
        return target

    reference, _ = assert_replays_bit_exact(
        reduce, (grads,), backward=False, contention=True, what='Engram static row accumulation'
    )
    assert torch.isfinite(reference['out']).all()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize('storage', ['mxfp8', 'bf16'])
@pytest.mark.parametrize('ep_size', [1, 4])
def test_engram_static_lookup_replays(ep_size, storage):
    """Complete static lookup: pack, ID/payload/gradient all-to-alls, repack and main_grad."""
    if storage == 'mxfp8' and not _IS_BLACKWELL:
        pytest.skip('TE MXFP8 needs Blackwell')
    Utils.initialize_distributed()
    if torch.distributed.get_world_size() % ep_size:
        pytest.skip('world size must be divisible by EP size')
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    try:
        seeded(23)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        ranges = _proxy_like_ranges()
        rows = sum(size for _, size in ranges)
        config = SimpleNamespace(
            use_cpu_initialization=False,
            params_dtype=torch.bfloat16,
            perform_initialization=True,
            sequence_parallel=False,
            deterministic_mode=True,
            gradient_accumulation_fusion=True,
            init_method=lambda t: torch.nn.init.normal_(t, std=0.1),
        )
        model = EPShardedMultiTableEmbedding(
            config,
            (rows,),
            256,
            config.init_method,
            pg.ep,
            fp8_storage=storage == 'mxfp8',
            communication_backend='static',
            column_ranges=ranges,
        )
        weight = model.tables[0].weight
        weight.main_grad = torch.zeros(weight.shape, dtype=torch.float32, device='cuda')
        weight.grad_added_to_main_grad = False
        tokens = 1024
        generator = torch.Generator(device='cuda').manual_seed(31 + pg.ep.rank())
        hashes = _addresses(ranges, tokens, generator).view(1, tokens, len(ranges), 1)
        upstream = torch.randn(
            tokens, len(ranges) * 256, dtype=torch.bfloat16, device='cuda', generator=generator
        )

        def lookup(ids):
            weight.main_grad.zero_()
            weight.grad = None
            if storage == 'mxfp8':
                activation = model.lookup_quantized(ids, len(ranges) * 256)
                outputs = [
                    activation._rowwise_data,
                    activation._rowwise_scale_inv,
                    activation._columnwise_data,
                    activation._columnwise_scale_inv,
                ]
                values = activation.dequantize()[:tokens]
            else:
                values = model(ids).reshape(tokens, -1)
                outputs = [values]
            (values * upstream).sum().backward()
            assert weight.grad_added_to_main_grad
            return [output.detach() for output in outputs] + [weight.main_grad]

        assert_replays_bit_exact(
            lookup,
            (hashes,),
            backward=False,
            contention=True,
            what=f'Engram static {storage} EP{ep_size} lookup forward/backward',
        )
    finally:
        Utils.destroy_model_parallel()

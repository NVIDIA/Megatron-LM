# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from itertools import groupby, product
from math import ceil, lcm

import pytest

from megatron.core.datasets.data_schedule import (
    DefaultDynamicCPScheduler,
    _get_dynamic_cp_nvlink_domains,
)
from megatron.core.datasets.data_schedule_utils import (
    _dcp_communication_overhead,
    _dcp_communication_tokens,
    align_sample_id_groups,
    next_hdp_group_packing_aware,
    reorder_dcp_groups,
)

_COST = (1024.0, 2304.0)


def _groups(layout):
    return [(list(samples), len(list(ranks))) for samples, ranks in groupby(layout)]


def _local_length(length, cp_size, sp_size=1):
    alignment = cp_size * lcm(2 if cp_size > 1 else 1, sp_size)
    return ceil(length / alignment) * alignment // cp_size


def _layout_cost(layout, lengths, domains, sp_size=1, padding_alignment=None, capacity=None):
    costs = []
    start = 0
    for samples, cp_size in _groups(layout):
        local_lengths = [_local_length(lengths[sid], cp_size, sp_size) for sid in samples]
        compute = sum(length**2 * cp_size for length in local_lengths)
        crosses = len(set(domains[start : start + cp_size])) > 1
        tokens = _dcp_communication_tokens(sum(local_lengths), capacity, padding_alignment)
        extra = _dcp_communication_overhead(compute, tokens, cp_size, crosses, _COST)
        costs.extend([compute + extra] * cp_size)
        start += cp_size
    return costs


@pytest.mark.parametrize('arbitrary', [False, True])
@pytest.mark.parametrize('sp_size', [1, 4])
def test_disabled_topology_preserves_every_scheduler_output(arbitrary, sp_size):
    rng = random.Random(1729)
    for _ in range(20):
        parent = rng.choice([2, 4, 8, 16])
        samples = [(sid, rng.randrange(1, parent * 256)) for sid in range(rng.randrange(1, 12))]
        kwargs = dict(
            total_gpus=parent,
            max_seq_len_per_rank=256,
            allow_arbitrary_group_starts=arbitrary,
            sequence_parallel_size=sp_size,
        )
        expected = next_hdp_group_packing_aware(samples, **kwargs)
        for domains, cost in [
            (None, _COST),
            ([0] * parent, _COST),
            (list(range(parent)), (0.0, 0.0)),
            (list(range(parent)), (1024.0, 1024.0)),
        ]:
            actual = next_hdp_group_packing_aware(
                samples, **kwargs, nvlink_domains=domains, communication_cost=cost
            )
            assert actual == expected


@pytest.mark.parametrize(
    'domains,cost',
    [([0] * 8, _COST), ([0] * 4 + [1] * 4, (1024.0, 1024.0)), ([0] * 4 + [1] * 4, (0.0, 0.0))],
)
@pytest.mark.parametrize('padding', [None, 'max'])
def test_single_domain_keeps_three_sequences_in_one_cp8_pack(domains, cost, padding):
    # Rebalancing before the legacy full-parent packing fallback would split
    # this into two rounds, despite all three sequences fitting one CP8 pack.
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=4096,
        cp_size=8,
        dp_size=1,
        microbatch_group_size_per_vp_stage=None,
        allow_arbitrary_group_starts=True,
        nvlink_domains=domains,
        communication_cost=cost,
        padding_alignment=padding,
    )
    assert scheduler.get_groups_and_subsamples([(sid, 8193) for sid in range(3)]) == [
        [[0, 1, 2]] * 8
    ]


@pytest.mark.parametrize('cp_size', range(1, 17))
def test_topology_does_not_restrict_arbitrary_cp_size(cp_size):
    length = cp_size * 128
    microbatches, leftovers, times, samples = next_hdp_group_packing_aware(
        [(0, length)],
        total_gpus=cp_size,
        max_seq_len_per_rank=128,
        allow_arbitrary_group_starts=True,
        nvlink_domains=[rank // 8 for rank in range(cp_size)],
    )
    assert not leftovers
    assert samples == [[0]] * cp_size
    assert microbatches == [[length]] * cp_size
    assert times == _layout_cost(samples, {0: length}, [rank // 8 for rank in range(cp_size)])


def test_reorder_fits_cp5_plus_cp3_inside_each_nvl8():
    layout = [[0]] * 5 + [[1]] * 5 + [[2]] * 3 + [[3]] * 3
    lengths = {0: 2304, 1: 2304, 2: 1408, 3: 1408}
    domains = [0] * 8 + [1] * 8
    result = reorder_dcp_groups(layout, lengths, domains, _COST)

    assert sorted(_groups(result)) == sorted(_groups(layout))
    assert [size for _, size in _groups(result)] == [5, 3, 5, 3]
    assert max(_layout_cost(result, lengths, domains)) < max(_layout_cost(layout, lengths, domains))
    start = 0
    for _, size in _groups(result):
        assert len(set(domains[start : start + size])) == 1
        start += size
    assert layout == [[0]] * 5 + [[1]] * 5 + [[2]] * 3 + [[3]] * 3


def test_reorder_compacts_holes_without_merging_every_sample():
    layout = [[0]] * 5 + [[]] * 3 + [[1]] * 3 + [[]] * 5
    result = reorder_dcp_groups(layout, {0: 2304, 1: 1408}, [0] * 8 + [1] * 8, _COST)

    assert len(result) == 16
    assert all(result)
    assert sorted(samples for samples, _ in _groups(result)) == [[0], [1]]
    assert all(size >= minimum for (_, size), minimum in zip(sorted(_groups(result)), [5, 3]))


@pytest.mark.parametrize('cp_size', [2, 3, 5])
@pytest.mark.parametrize('capacity', [None, 2, 3])
def test_reorder_rejects_cp1_expansion_that_exceeds_padded_capacity(cp_size, capacity):
    # Each one-token sequence grows from one local row on CP1 to two zigzag
    # rows on CP>1. A two-sequence pack therefore needs four, not two, rows.
    layout = [[0, 1]] + [[] for _ in range(cp_size - 1)]
    with pytest.raises(ValueError, match='per-rank token capacity'):
        reorder_dcp_groups(
            layout, {0: 1, 1: 1}, list(range(cp_size)), max_seq_len_per_rank=capacity
        )
    assert layout == [[0, 1]] + [[] for _ in range(cp_size - 1)]


@pytest.mark.parametrize('cp_size', [2, 3, 5])
def test_reorder_allows_cp1_expansion_when_explicit_capacity_fits_padding(cp_size):
    result = reorder_dcp_groups(
        [[0, 1]] + [[] for _ in range(cp_size - 1)],
        {0: 1, 1: 1},
        list(range(cp_size)),
        max_seq_len_per_rank=4,
    )
    assert result == [[0, 1]] * cp_size
    assert sum(_local_length(1, cp_size) for _ in result[0]) == 4


@pytest.mark.parametrize('capacity', [None, 2])
def test_reorder_skips_padding_unsafe_expansion_and_uses_legal_group(capacity):
    lengths = {0: 1, 1: 1, 2: 4}
    result = reorder_dcp_groups(
        [[0, 1], [2], [2], []], lengths, [0, 0, 1, 1], max_seq_len_per_rank=capacity
    )
    assert sorted(_groups(result)) == [([0, 1], 1), ([2], 3)]
    for samples, cp_size in _groups(result):
        assert sum(_local_length(lengths[sid], cp_size) for sid in samples) <= 2


def test_reorder_validates_explicit_capacity_even_without_empty_ranks():
    layout = [[0]] * 3
    with pytest.raises(ValueError, match='per-rank token capacity'):
        reorder_dcp_groups(layout, {0: 13}, [0, 0, 1], max_seq_len_per_rank=5)
    assert reorder_dcp_groups(layout, {0: 13}, [0, 0, 1], max_seq_len_per_rank=6) == layout


@pytest.mark.parametrize('sp_size', [1, 4])
@pytest.mark.parametrize('padding', [None, 'max'])
def test_spare_ranks_are_distributed_across_all_packs(sp_size, padding):
    # Giving all eight spare ranks to one CP3 pack creates CP11 and leaves
    # seven slow CP3 packs. CP4 for each pack fills all four NVL8 domains.
    layout = [[sid] for sid in range(8) for _ in range(3)] + [[] for _ in range(8)]
    result = reorder_dcp_groups(
        layout,
        {sid: 4096 for sid in range(8)},
        [rank // 8 for rank in range(32)],
        sequence_parallel_size=sp_size,
        max_seq_len_per_rank=2048,
        padding_alignment=padding,
    )
    assert result == [[sid] for sid in range(8) for _ in range(4)]


def test_rank_allocation_optimizes_total_work_under_the_final_bottleneck():
    # A later pack sets max=147456. Keeping only one lexicographic prefix state
    # incorrectly chooses CP2+2+4 (sum=688192); the exact tie is CP1+3+4.
    lengths, domains = {0: 243, 1: 188, 2: 758}, [0] * 4 + [1] * 4
    result = reorder_dcp_groups(
        [[0], [1], [2], [2]] + [[] for _ in range(4)],
        lengths,
        domains,
        sequence_parallel_size=4,
        max_seq_len_per_rank=512,
    )
    assert sorted(_groups(result)) == [([0], 1), ([1], 3), ([2], 4)]
    costs = _layout_cost(result, lengths, domains, 4)
    assert (max(costs), sum(costs)) == (147456, 686224)


def test_rank_allocation_considers_group_order_before_filling_spare_ranks():
    # NVL8 plus a four-rank tail: keeping pack order forces CP7+5 across the
    # boundary. Ordering the larger group first permits domain-local CP8+4.
    lengths = {0: 336, 1: 282, 2: 345, 3: 390, 4: 390}
    domains = [0] * 8 + [1] * 4
    result = reorder_dcp_groups(
        [[0, 1]] * 3 + [[2, 3, 4]] * 5 + [[] for _ in range(4)],
        lengths,
        domains,
        sequence_parallel_size=4,
        max_seq_len_per_rank=256,
        padding_alignment=256,
    )
    assert result == [[2, 3, 4]] * 8 + [[0, 1]] * 4
    costs = _layout_cost(result, lengths, domains, 4, 256, 256)
    assert (max(costs), sum(costs)) == (58752, 665856)


@pytest.mark.parametrize('sp_size', [1, 4])
@pytest.mark.parametrize('padding', [None, 'max'])
def test_rank_allocation_matches_bounded_fixed_order_oracle(sp_size, padding):
    # Exhaustion is intentionally confined to this tiny CPU test, never used
    # by the training scheduler. Whole-pack reordering may improve further.
    rng, domains = random.Random(58085), [0] * 4 + [1] * 4
    for _ in range(10):
        lengths = {0: rng.randint(1, 512), 1: rng.randint(1, 512), 2: rng.randint(512, 1024)}
        candidates = []
        for extras in product(range(5), repeat=3):
            if sum(extras) != 4:
                continue
            sizes = [minimum + extra for minimum, extra in zip([1, 1, 2], extras)]
            if any(_local_length(lengths[sid], cp, sp_size) > 512 for sid, cp in enumerate(sizes)):
                continue
            layout = [[sid] for sid, cp in enumerate(sizes) for _ in range(cp)]
            costs = _layout_cost(layout, lengths, domains, sp_size, padding, 512)
            candidates.append((max(costs), sum(costs)))
        result = reorder_dcp_groups(
            [[0], [1], [2], [2]] + [[] for _ in range(4)],
            lengths,
            domains,
            sequence_parallel_size=sp_size,
            max_seq_len_per_rank=512,
            padding_alignment=padding,
        )
        costs = _layout_cost(result, lengths, domains, sp_size, padding, 512)
        assert (max(costs), sum(costs)) <= min(candidates)
        assert sorted(samples for samples, _ in _groups(result)) == [[0], [1], [2]]


@pytest.mark.parametrize('sp_size', [1, 4])
def test_whole_group_reordering_never_increases_complete_layout_proxy(sp_size):
    rng = random.Random(491)
    domains = [rank // 8 for rank in range(32)]
    for _ in range(30):
        layout, lengths = [], {}
        while len(layout) < len(domains):
            cp_size = rng.randint(1, len(domains) - len(layout))
            samples = list(range(len(lengths), len(lengths) + rng.randint(1, 3)))
            lengths.update({sid: rng.randint(1, 8192) for sid in samples})
            layout.extend([list(samples) for _ in range(cp_size)])
        result = reorder_dcp_groups(layout, lengths, domains, _COST, sp_size)
        before = _layout_cost(layout, lengths, domains, sp_size)
        after = _layout_cost(result, lengths, domains, sp_size)
        assert (max(after), sum(after)) <= (max(before), sum(before))
        assert sorted(_groups(result)) == sorted(_groups(layout))


@pytest.mark.parametrize('sp_size', [1, 2, 4, 8])
def test_cross_domain_scheduler_respects_padded_token_capacity(sp_size):
    lengths = {0: 2040, 1: 128}
    _, leftovers, times, layout = next_hdp_group_packing_aware(
        list(lengths.items()),
        total_gpus=6,
        max_seq_len_per_rank=510,
        allow_arbitrary_group_starts=True,
        sequence_parallel_size=sp_size,
        nvlink_domains=[0] * 3 + [1] * 3,
    )

    assert not leftovers
    assert sorted(samples for samples, _ in _groups(layout)) == [[0], [1]]
    for samples, cp_size in _groups(layout):
        local_lengths = [_local_length(lengths[sid], cp_size, sp_size) for sid in samples]
        assert sum(local_lengths) <= 510
        assert all(length % sp_size == 0 for length in local_lengths)
    assert times == _layout_cost(layout, lengths, [0] * 3 + [1] * 3, sp_size)


def test_packed_sequences_recompute_exposed_communication_once():
    # The long sequence's computation hides the short one's transfer in the same
    # packed attention call. Summing independent per-sequence penalties is wrong.
    long_compute = 4096**2 * 4
    short_compute = 128**2 * 4
    assert _dcp_communication_overhead(short_compute, 128, 4, True, _COST) > 0
    assert _dcp_communication_overhead(long_compute + short_compute, 4224, 4, True, _COST) == 0
    _, leftovers, times, layout = next_hdp_group_packing_aware(
        [(0, 16384), (1, 512)],
        total_gpus=4,
        max_seq_len_per_rank=8192,
        min_cp_size=4,
        allow_arbitrary_group_starts=True,
        nvlink_domains=[0, 0, 1, 1],
    )
    assert not leftovers
    assert layout == [[0, 1]] * 4
    assert times == [long_compute + short_compute] * 4


@pytest.mark.parametrize(
    'local_tokens,alignment,expected',
    [
        (1, None, 1),
        (511, None, 511),
        (4096, None, 4096),
        (1, 256, 256),
        (255, 256, 256),
        (256, 256, 256),
        (257, 256, 512),
        (4096, 256, 4096),
        (1, 'max', 4096),
        (511, 'max', 4096),
        (4096, 'max', 4096),
    ],
)
def test_communication_payload_counts_padding(local_tokens, alignment, expected):
    assert _dcp_communication_tokens(local_tokens, 4096, alignment) == expected


@pytest.mark.parametrize('alignment,payload', [(None, 500), (512, 512), ('max', 4096)])
def test_cp5_communication_uses_actual_padded_payload_not_valid_tokens(alignment, payload):
    # Both sequences share one CP5 pack: 400 + 100 valid local rows, but max
    # padding sends the full 4096-row communication buffer on every ring hop.
    compute = (400**2 + 100**2) * 5
    _, leftovers, times, layout = next_hdp_group_packing_aware(
        [(0, 2000), (1, 500)],
        total_gpus=5,
        max_seq_len_per_rank=4096,
        min_cp_size=5,
        allow_arbitrary_group_starts=True,
        nvlink_domains=[0, 0, 0, 0, 1],
        padding_alignment=alignment,
    )
    expected_extra = _dcp_communication_overhead(compute, payload, 5, True, _COST)
    assert not leftovers
    assert layout == [[0, 1]] * 5
    assert times == [compute + expected_extra] * 5
    if alignment == 'max':
        unpadded_extra = _dcp_communication_overhead(compute, 500, 5, True, _COST)
        assert expected_extra > 8 * unpadded_extra


@pytest.mark.parametrize('alignment,payload', [(None, 3100), (1024, 4096), ('max', 4096)])
def test_padded_payload_overhead_uses_whole_pack_compute_overlap(alignment, payload):
    # A long sequence hides most of the fixed-buffer transfer; packing a short
    # sequence must recompute that overlap, not charge a second buffer transfer.
    per_sequence_compute = [3000**2 * 5, 100**2 * 5]
    compute = sum(per_sequence_compute)
    _, leftovers, times, layout = next_hdp_group_packing_aware(
        [(0, 15000), (1, 500)],
        total_gpus=5,
        max_seq_len_per_rank=4096,
        min_cp_size=5,
        allow_arbitrary_group_starts=True,
        nvlink_domains=[0, 0, 0, 0, 1],
        padding_alignment=alignment,
    )
    expected_extra = _dcp_communication_overhead(compute, payload, 5, True, _COST)
    incorrectly_summed = sum(
        _dcp_communication_overhead(
            work, _dcp_communication_tokens(tokens, 4096, alignment), 5, True, _COST
        )
        for work, tokens in zip(per_sequence_compute, [3000, 100])
    )
    assert not leftovers
    assert layout == [[0, 1]] * 5
    assert times == [compute + expected_extra] * 5
    assert expected_extra < incorrectly_summed
    assert (expected_extra > 0) == (alignment is not None)


def test_integer_payload_padding_rounds_the_pack_once():
    compute = 2 * 100**2 * 5
    _, leftovers, times, layout = next_hdp_group_packing_aware(
        [(0, 500), (1, 500)],
        total_gpus=5,
        max_seq_len_per_rank=4096,
        min_cp_size=5,
        allow_arbitrary_group_starts=True,
        nvlink_domains=[0, 0, 0, 0, 1],
        padding_alignment=256,
    )
    # ceil((100 + 100) / 256) * 256, not two separately rounded 256-row payloads.
    assert not leftovers
    assert layout == [[0, 1]] * 5
    assert times == [compute + _dcp_communication_overhead(compute, 256, 5, True, _COST)] * 5
    assert times != [compute + _dcp_communication_overhead(compute, 512, 5, True, _COST)] * 5


@pytest.mark.parametrize('crosses', [False, True])
@pytest.mark.parametrize('cp_size', [1, 3, 8, 16])
def test_communication_overhead_matches_exposed_ring_steps(crosses, cp_size):
    compute, tokens = 16384.0, 128
    expected = (cp_size - 1) * (
        max(compute / cp_size, _COST[1] * tokens) - max(compute / cp_size, _COST[0] * tokens)
    )
    assert _dcp_communication_overhead(compute, tokens, cp_size, crosses, _COST) == (
        expected if crosses else 0
    )
    assert _dcp_communication_overhead(1e12, tokens, cp_size, crosses, _COST) == 0
    assert _dcp_communication_overhead(compute, tokens, cp_size, crosses, (0.0, 0.0)) == 0


@pytest.mark.parametrize(
    'cost', [(-1.0, 2.0), (2.0, 1.0), (float('nan'), 2.0), (1.0, float('inf')), (1.0,)]
)
def test_scheduler_rejects_invalid_communication_cost(cost):
    with pytest.raises(ValueError):
        next_hdp_group_packing_aware(
            [(0, 128)],
            2,
            128,
            allow_arbitrary_group_starts=True,
            nvlink_domains=[0, 1],
            communication_cost=cost,
        )


@pytest.mark.parametrize('domains', [[], [0], [0, 1, 2]])
def test_scheduler_rejects_wrong_domain_vector_size(domains):
    with pytest.raises(ValueError):
        next_hdp_group_packing_aware(
            [(0, 128)], 2, 128, allow_arbitrary_group_starts=True, nvlink_domains=domains
        )


def test_topology_aware_scheduler_rejects_legacy_real_group_placement():
    with pytest.raises(ValueError, match='arbitrary logical CP groups'):
        next_hdp_group_packing_aware(
            [(0, 128)], 2, 128, allow_arbitrary_group_starts=False, nvlink_domains=[0, 1]
        )


@pytest.mark.parametrize('tp_size,pp_size,parent', [(1, 1, 16), (2, 2, 8), (4, 3, 8), (2, 2, 3)])
def test_nvlink_domains_are_consistent_across_tp_and_pp_planes(tp_size, pp_size, parent):
    # Include a non-node-aligned PP stride: TP2 x parent3 x PP2 needs a
    # conservative boundary from the second PP plane even on the first plane.
    results = []
    stride = tp_size * parent
    for pp in range(pp_size):
        for tp in range(tp_size):
            offset = pp * stride + tp
            for dp in range(parent):
                results.append(
                    _get_dynamic_cp_nvlink_domains(
                        list(range(offset, offset + stride, tp_size)),
                        list(range(pp * stride + dp * tp_size, pp * stride + (dp + 1) * tp_size)),
                        [stage * stride + dp * tp_size + tp for stage in range(pp_size)],
                        offset + dp * tp_size,
                        stride * pp_size,
                        8,
                    )
                )
    boundaries = [
        any(
            (pp * stride + tp + (dp - 1) * tp_size) // 8 != (pp * stride + tp + dp * tp_size) // 8
            for pp in range(pp_size)
            for tp in range(tp_size)
        )
        for dp in range(1, parent)
    ]
    expected = [sum(boundaries[:dp]) for dp in range(parent)]
    assert results == [expected] * (tp_size * pp_size * parent)


def test_vpp_alignment_keeps_all_samples_after_topology_reordering():
    lengths = {0: 2304, 1: 2304, 2: 1408, 3: 1408}
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=512,
        cp_size=16,
        dp_size=1,
        microbatch_group_size_per_vp_stage=2,
        allow_arbitrary_group_starts=True,
        nvlink_domains=[0] * 8 + [1] * 8,
    )
    layouts = scheduler.get_groups_and_subsamples(list(lengths.items()))
    assert len(layouts) % 2 == 0
    seen = []
    for layout in layouts:
        assert len(layout) == 16 and all(layout)
        for samples, cp_size in _groups(layout):
            seen.extend(samples)
            assert sum(_local_length(lengths[sid], cp_size) for sid in samples) <= 512
    assert sorted(seen) == sorted(lengths)


@pytest.mark.parametrize('sp_size', [1, 4])
@pytest.mark.parametrize('padding', [None, 'max'])
def test_vpp_split_allocates_spare_ranks_before_expansion_is_locked_in(sp_size, padding):
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1024,
        cp_size=16,
        dp_size=1,
        microbatch_group_size_per_vp_stage=2,
        allow_arbitrary_group_starts=True,
        sequence_parallel_size=sp_size,
        nvlink_domains=[0] * 8 + [1] * 8,
        padding_alignment=padding,
    )
    result = scheduler.get_groups_and_subsamples([(sid, 4096) for sid in range(4)])
    assert result == [[[2]] * 8 + [[3]] * 8, [[0]] * 8 + [[1]] * 8]
    assert sorted(
        sid for layout in result for samples, _ in _groups(layout) for sid in samples
    ) == list(range(4))


def test_missing_topology_vpp_keeps_legacy_fill_instead_of_noop_callback():
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1024,
        cp_size=16,
        dp_size=1,
        microbatch_group_size_per_vp_stage=2,
        allow_arbitrary_group_starts=True,
        nvlink_domains=None,
    )
    result = scheduler.get_groups_and_subsamples([(sid, 4096) for sid in range(4)])
    # Exact legacy a9b7cb530 output. A disabled reorder callback would return
    # holes here; the callback must be omitted so the legacy CP4+12 fill runs.
    assert result == [[[2]] * 4 + [[3]] * 12, [[0]] * 4 + [[1]] * 12]
    assert all(all(layout) for layout in result)


@pytest.mark.parametrize(
    'domains,cost',
    [([0] * 16, _COST), ([0] * 8 + [1] * 8, (1024.0, 1024.0)), ([0] * 8 + [1] * 8, (0.0, 0.0))],
)
def test_zero_network_penalty_vpp_preserves_legacy_fill(domains, cost):
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1024,
        cp_size=16,
        dp_size=1,
        microbatch_group_size_per_vp_stage=2,
        allow_arbitrary_group_starts=True,
        nvlink_domains=domains,
        communication_cost=cost,
    )
    result = scheduler.get_groups_and_subsamples([(sid, 4096) for sid in range(4)])
    assert result == [[[2]] * 4 + [[3]] * 12, [[0]] * 4 + [[1]] * 12]
    assert all(all(layout) for layout in result)
    assert sorted(
        sid for layout in result for samples, _ in _groups(layout) for sid in samples
    ) == list(range(4))


def test_vpp_fill_callback_observes_unexpanded_split_groups():
    seen = []

    def fill(layout):
        seen.append([list(samples) for samples in layout])
        return reorder_dcp_groups(
            layout, {sid: 4096 for sid in range(4)}, [0] * 8 + [1] * 8, max_seq_len_per_rank=1024
        )

    result = align_sample_id_groups(
        [[[sid] for sid in range(4) for _ in range(4)]],
        2,
        allow_arbitrary_group_starts=True,
        fill_empty_ranks=fill,
    )
    assert len(seen) == 2
    assert all(sum(not samples for samples in layout) == 8 for layout in seen)
    assert all([cp for samples, cp in _groups(layout) if samples] == [4, 4] for layout in seen)
    assert all([cp for _, cp in _groups(layout)] == [8, 8] for layout in result)

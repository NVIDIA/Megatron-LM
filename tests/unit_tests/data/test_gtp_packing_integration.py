# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-group transport tests for full-domain GTP packing."""

import os
import sys
import traceback
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import wrap_data_iterator
from megatron.core.process_groups_config import ProcessGroupCollection


def _initialize(tp, pp, cp, dynamic):
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(
            'nccl',
            timeout=timedelta(seconds=120),
            device_id=torch.device('cuda', torch.cuda.current_device()),
        )
    if dist.get_world_size() % (tp * pp * cp * 2):
        pytest.skip("Topology does not fit this torchrun world")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        gtp_remat_size=2,
        dynamic_context_parallel=dynamic,
        create_gloo_process_groups=False,
        distributed_timeout_minutes=2,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=64,
        microbatch_group_size_per_vp_stage=pp,
        virtual_pipeline_model_parallel_size=None,
        sequence_packing_scheduler='default_dynamic_cp' if dynamic else 'dp_balanced',
        min_dynamic_context_parallel_size=1,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        gtp_weight_remat_size=2,
        gtp_remat_num_sequence_shards=1,
        sequence_parallel=False,
    )
    return groups, config


def _sample(sample_id, length):
    dev = torch.cuda.current_device()
    return {
        'tokens': torch.full((length,), sample_id, dtype=torch.int64, device=dev),
        'labels': torch.full((length,), sample_id + 1, dtype=torch.int64, device=dev),
        'loss_mask': torch.ones(length, dtype=torch.float32, device=dev),
        'position_ids': torch.arange(length, dtype=torch.int64, device=dev),
        'original_seq_len': torch.tensor([length], dtype=torch.int64, device=dev),
        'padded_seq_len': torch.tensor([length], dtype=torch.int64, device=dev),
    }


def _finish():
    if sys.exc_info()[0] is not None:
        # Expose the original failed assertion immediately. A finally barrier
        # would hide it while peers wait in a later test collective.
        traceback.print_exc()
        return
    torch.cuda.synchronize()
    dist.barrier()
    parallel_state.destroy_model_parallel()


@pytest.mark.parametrize('tp,pp,cp', [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)])
@pytest.mark.parametrize('dynamic', [False, True])
def test_global_gtp_transport(tp, pp, cp, dynamic):
    groups, config = _initialize(tp, pp, cp, dynamic)
    try:
        owner = groups.dp_gtp_remat.rank()
        # Opposite GTP coordinates have strongly asymmetric workloads.
        long_length = 64 * groups.dp_cp_gtp_remat.size() if dynamic else 64
        length = 16 if groups.gtp_remat.rank() == 0 else long_length
        samples = [_sample(1000 + owner * 8 + i, length) for i in range(8)]
        iterator = iter(samples) if groups.tp.rank() == 0 else None
        new_iterator, rounds, total, total_squared = wrap_data_iterator(
            iterator, config, 8, pg_collection=groups
        )
        records = []
        if groups.tp.rank() == 0:
            for _ in range(rounds):
                microbatch = next(new_iterator)
                values = microbatch.get('tokens')
                if values is None:
                    values = microbatch['labels'] - 1
                bounds = microbatch['cu_seqlens_padded'].tolist()
                ids = []
                for start, stop in zip(bounds, bounds[1:]):
                    sample_id = int(values[start].item())
                    assert torch.all(values[start:stop] == sample_id)
                    ids.append(sample_id)
                runtime_cp = int(microbatch['local_cp_size']) if dynamic else cp
                cp_group = (
                    parallel_state.get_dynamic_data_context_parallel_groups(group_size=runtime_cp)
                    if dynamic
                    else groups.cp
                )
                records.append((ids, dist.get_process_group_ranks(cp_group)))
            assert len(records) == rounds
            all_records = [None] * groups.dp_cp_gtp_remat.size()
            dist.all_gather_object(all_records, records, group=groups.dp_cp_gtp_remat)
            full_ranks = dist.get_process_group_ranks(groups.dp_cp_gtp_remat)
            occurrences = {}
            for slot, rank_records in enumerate(all_records):
                assert len(rank_records) == rounds
                for step, (ids, members) in enumerate(rank_records):
                    for sample_id in ids:
                        occurrences.setdefault(sample_id, []).append(
                            (step, full_ranks[slot], members)
                        )
            assert set(occurrences) == set(range(1000, 1000 + groups.dp_gtp_remat.size() * 8))
            if dynamic:
                assert any(
                    len(members) == len(full_ranks)
                    for rank_records in all_records
                    for _, members in rank_records
                )
            for placements in occurrences.values():
                assert len({step for step, _, _ in placements}) == 1
                actual_ranks = sorted(rank for _, rank, _ in placements)
                assert all(sorted(members) == actual_ranks for _, _, members in placements)
        expected = groups.dp_gtp_remat.size() // 2 * 8
        assert total == expected * (16 + long_length)
        assert total_squared == expected * (16**2 + long_length**2)
    finally:
        _finish()

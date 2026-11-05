# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate full-domain GTP packing's VPP alignment and independent rewind state."""

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
from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode, get_rerun_state_machine


def _snapshot(sample):
    return {key: tensor.detach().cpu().clone() for key, tensor in sample.items()}


@pytest.mark.parametrize("mtp_num_layers", [None, 1])
def test_gtp_vpp_packing_replay(mtp_num_layers):
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(
            'nccl',
            timeout=timedelta(seconds=120),
            device_id=torch.device('cuda', torch.cuda.current_device()),
        )
    if dist.get_world_size() % 4:
        pytest.skip('Requires a multiple of four ranks for PP2 x GTP2')
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        context_parallel_size=1,
        gtp_remat_size=2,
        dynamic_context_parallel=True,
        create_gloo_process_groups=False,
        distributed_timeout_minutes=2,
    )
    pg = ProcessGroupCollection.use_mpu_process_groups()
    machine = get_rerun_state_machine()
    old_mode = machine.get_mode()
    machine.set_mode(RerunMode.VALIDATE_RESULTS)
    try:
        config = SimpleNamespace(
            max_seqlen_per_dp_cp_rank=64,
            microbatch_group_size_per_vp_stage=2,
            virtual_pipeline_model_parallel_size=2,
            sequence_packing_scheduler='default_dynamic_cp',
            min_dynamic_context_parallel_size=1,
            pipeline_model_parallel_layout=None,
            mtp_num_layers=mtp_num_layers,
            gtp_weight_remat_size=2,
            gtp_remat_num_sequence_shards=1,
            sequence_parallel=False,
        )
        length = 16 if pg.gtp_remat.rank() == 0 else 64
        owner = pg.dp_gtp_remat.rank()
        device = torch.cuda.current_device()
        samples = []
        for index in range(8):
            token_id = 1000 + owner * 8 + index
            samples.append(
                {
                    'tokens': torch.full((length,), token_id, device=device, dtype=torch.int64),
                    'labels': torch.full((length,), token_id + 1, device=device, dtype=torch.int64),
                    'loss_mask': torch.ones(length, device=device, dtype=torch.float32),
                    'position_ids': torch.arange(length, device=device, dtype=torch.int64),
                    'original_seq_len': torch.tensor([length], device=device),
                    'padded_seq_len': torch.tensor([length], device=device),
                }
            )
        source = RerunDataIterator(iter(samples))
        data_vp = 0 if pg.pp.rank() == 0 else 1
        sources = [None, None]
        sources[data_vp] = source
        packed, rounds, total, squared = wrap_data_iterator(sources, config, 8, pg_collection=pg)
        assert rounds > 0 and rounds % 2 == 0
        assert total == pg.dp_gtp_remat.size() // 2 * 8 * (16 + 64)
        assert squared == pg.dp_gtp_remat.size() // 2 * 8 * (16**2 + 64**2)
        assert len(packed) == 2 and packed[0] is not packed[1]
        originals = [[next(iterator) for _ in range(rounds)] for iterator in packed]
        expected = [[_snapshot(sample) for sample in chunk] for chunk in originals]
        metadata_keys = {'cu_seqlens', 'cu_seqlens_padded', 'max_seqlen', 'local_cp_size'}
        for vp in range(2):
            for sample in originals[vp]:
                if vp == data_vp:
                    assert ('tokens' if pg.pp.rank() == 0 else 'labels') in sample
                else:
                    assert set(sample) == metadata_keys
        assert originals[0][0] is not originals[1][0]
        # Batch preparation replaces dictionary entries; sibling VPP chunks
        # and the saved replay snapshot must retain their own entries.
        originals[0][0].pop('cu_seqlens')
        originals[0][0]['local_cp_size'] = torch.tensor(999, device=device)
        assert 'cu_seqlens' in originals[1][0]
        assert int(originals[1][0]['local_cp_size']) != 999
        for vp, iterator in enumerate(packed):
            iterator.rewind()
            for step in range(rounds):
                replay = _snapshot(next(iterator))
                assert set(replay) == set(expected[vp][step])
                for key in replay:
                    assert torch.equal(replay[key], expected[vp][step][key])
            iterator.advance()
            assert not iterator.saved_microbatches
        assert len(source.saved_microbatches) == 8
        dist.barrier()
    finally:
        machine.set_mode(old_mode)
        if sys.exc_info()[0] is not None:
            traceback.print_exc()
        else:
            torch.cuda.synchronize()
            dist.barrier()
            parallel_state.destroy_model_parallel()

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 Engram, checkpoint and pipeline state transport contracts."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.lite.protocol import forward_step
from megatron.lite.runtime.contracts import PackedBatch


@pytest.mark.parametrize("rank", [0, 1])
def test_forward_step_rows_are_original_unpadded_rows(rank):
    class Rows(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ps = SimpleNamespace(cp_size=2, cp_rank=rank, cp_group=object())

        def forward(self, input_ids, **kwargs):
            local = input_ids.numel()
            mask = kwargs["loss_mask"].bool()
            return {
                "log_probs": torch.zeros(local),
                "rows": torch.arange(local) + rank * local,  # physical, as the model reports
                "valid_mask": mask,
            }

    ids = torch.arange(12)
    output = forward_step(Rows(), PackedBatch(ids, ids, torch.tensor([5, 1, 6])))
    # Documents padded to 8 + 4 + 8 physical rows, contiguous CP halves of 10.
    expected = (
        [0, 1, 2, 3, 4, -1, -1, -1, 5, -1] if rank == 0 else [-1, -1, 6, 7, 8, 9, 10, 11, -1, -1]
    )
    assert output["rows"].tolist() == expected
    # Padding rows are never valid; document ends stay masked by the label roll.
    assert not output["valid_mask"][output["rows"] < 0].any()
    assert output["valid_mask"].sum() == (4 if rank == 0 else 5)


def test_checkpoint_metadata_keeps_engram_rows_at_global_offsets(tmp_path):
    import torch.distributed as dist

    from megatron.lite.model.deepseek_v41.lite.checkpoint import attach_checkpoint_metadata

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1)
    try:
        group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)  # rank 1 of a 2-way row shard
        table = torch.nn.Module()
        table.register_buffer("weight", torch.ones(3, 4, dtype=torch.bfloat16))
        table.lookup = SimpleNamespace(group=group, boundaries=(0, 3, 6))
        engram = torch.nn.Module()
        engram.embed, engram.wkv = table, torch.nn.Linear(4, 4, bias=False)
        engram.register_buffer("cache", torch.zeros(2), persistent=False)
        model = torch.nn.Module()
        model.engrams = torch.nn.ModuleDict({"1": engram})
        ps = SimpleNamespace(dp_cp_rank=1, dp_cp_group=dist.group.WORLD)
        attach_checkpoint_metadata([model], ps)
        state = model.sharded_state_dict()
        assert model._mlite_dist_opt_sharded_state_dict
        assert set(state) == {"engrams.1.embed.weight", "engrams.1.wkv.weight"}
        rows = state["engrams.1.embed.weight"]
        assert rows.global_shape == (6, 4) and rows.global_offset == (3, 0)
        assert rows.replica_id == (0, 0, 0)  # the row group covers dp x cp: no replicas
        assert state["engrams.1.wkv.weight"].global_shape == (4, 4)
        table.lookup.boundaries = (0, 2, 6)
        with pytest.raises(ValueError, match="equal Core row shards"):
            model.sharded_state_dict()
    finally:
        dist.destroy_process_group()

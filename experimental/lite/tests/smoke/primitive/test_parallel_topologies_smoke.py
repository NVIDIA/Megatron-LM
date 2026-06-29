# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.parallel import (
    PackedSeqParams,
    contiguous_to_zigzag_chunks,
    init_parallel,
    zigzag_split_for_cp,
    zigzag_to_contiguous_chunks,
)

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu, pytest.mark.distributed]


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for distributed primitive smoke tests.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _assert_group_size(group, expected: int):
    if expected == 1:
        return
    assert group is not None
    assert dist.get_world_size(group) == expected


def _topologies(world_size: int):
    yield "dp_only", SimpleNamespace(tp=1, ep=1, etp=1, cp=1, pp=1)
    if world_size >= 2:
        yield "tp2", SimpleNamespace(tp=2, ep=1, etp=1, cp=1, pp=1)
        yield "cp2", SimpleNamespace(tp=1, ep=1, etp=1, cp=2, pp=1)
        yield "pp2", SimpleNamespace(tp=1, ep=1, etp=1, cp=1, pp=2)
        yield "ep2", SimpleNamespace(tp=1, ep=2, etp=1, cp=1, pp=1)
        yield "etp2", SimpleNamespace(tp=1, ep=1, etp=2, cp=1, pp=1)
    if world_size >= 4:
        yield "tp2_ep2_pp2", SimpleNamespace(tp=2, ep=2, etp=1, cp=1, pp=2)
    if world_size >= 8:
        yield "tp2_ep2_cp2_pp2", SimpleNamespace(tp=2, ep=2, etp=1, cp=2, pp=2)
        yield "tp2_ep2_etp2_pp2", SimpleNamespace(tp=2, ep=2, etp=2, cp=1, pp=2)


def test_parallel_state_builds_expected_primitive_groups():
    world_size = dist.get_world_size()
    for name, cfg in _topologies(world_size):
        if world_size % (cfg.tp * cfg.cp * cfg.pp) != 0:
            continue
        if world_size % (cfg.etp * cfg.ep * cfg.pp) != 0:
            continue

        ps = init_parallel(cfg)
        dense_dp = world_size // (cfg.tp * cfg.cp * cfg.pp)
        expert_dp = world_size // (cfg.etp * cfg.ep * cfg.pp)

        assert ps.tp_size == cfg.tp
        assert ps.cp_size == cfg.cp
        assert ps.pp_size == cfg.pp
        assert ps.ep_size == cfg.ep
        assert ps.dp_size == dense_dp
        assert ps.expert_dp_size == expert_dp
        assert 0 <= ps.tp_rank < cfg.tp
        assert 0 <= ps.cp_rank < cfg.cp
        assert 0 <= ps.pp_rank < cfg.pp
        assert 0 <= ps.ep_rank < cfg.ep
        _assert_group_size(ps.tp_group, cfg.tp)
        _assert_group_size(ps.cp_group, cfg.cp)
        _assert_group_size(ps.pp_group, cfg.pp)
        _assert_group_size(ps.ep_group, cfg.ep)
        _assert_group_size(ps.dp_group, dense_dp)
        _assert_group_size(ps.dp_cp_group, dense_dp * cfg.cp)
        _assert_group_size(ps.ep_dp_group, expert_dp)
        # tp_ep follows Megatron Core's expert tensor + expert model group,
        # not the dense-layer TP group.
        _assert_group_size(ps.tp_ep_group, cfg.etp * cfg.ep)
        _assert_group_size(ps.etp_group, cfg.etp)
        if cfg.pp > 1:
            assert ps.pp_next_rank in ps.pp_global_ranks
            assert ps.pp_prev_rank in ps.pp_global_ranks


def test_cp_zigzag_contiguous_chunk_swap_roundtrip():
    if dist.get_world_size() < 2:
        pytest.skip("CP chunk swap smoke requires at least 2 ranks.")

    ps = init_parallel(SimpleNamespace(tp=1, ep=1, etp=1, cp=2, pp=1))
    full = torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 8, 1) + 100 * ps.dp_rank
    local_zigzag = zigzag_split_for_cp(full, ps.cp_rank, cp_size=2, seq_dim=1)

    local_contiguous = zigzag_to_contiguous_chunks(local_zigzag, ps.cp_group, seq_dim=1)
    expected = full[:, ps.cp_rank * 4 : (ps.cp_rank + 1) * 4, :]
    assert torch.equal(local_contiguous, expected)

    restored = contiguous_to_zigzag_chunks(local_contiguous, ps.cp_group, seq_dim=1)
    assert torch.equal(restored, local_zigzag)


def test_gdn_rejects_thd_context_parallel_until_validated():
    if dist.get_world_size() < 2:
        pytest.skip("GDN THD+CP guard smoke requires at least 2 ranks.")

    pytest.importorskip("transformer_engine.pytorch")
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet

    ps = init_parallel(SimpleNamespace(tp=1, ep=1, etp=1, cp=2, pp=1))
    gdn = (
        GatedDeltaNet(
            hidden_size=16,
            linear_num_key_heads=2,
            linear_key_head_dim=4,
            linear_num_value_heads=2,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=2,
            rms_norm_eps=1e-6,
            ps=ps,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    packed_seq_params = PackedSeqParams.from_cu_seqlens(cu_seqlens, max_seqlen=8)
    local_hidden = torch.randn(4, 1, 16, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="all-gather CP"):
        gdn(local_hidden, packed_seq_params=packed_seq_params)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay native logical CP reductions and MTP halos without subgroup communicators."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.context_parallel.sequence_roll import roll_contiguous
from megatron.core.dynamic_cp_group import LogicalCPGroup
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_token_prediction import bind_native_mtp_cp_group, roll_tensor
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, bytes_equal
from tests.unit_tests.test_utilities import Utils, _restore_torch_settings, _snapshot_torch_settings

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("changed", [False, True])
def test_cudnn_fixture_restoration_respects_frozen_settings(monkeypatch, changed):
    """Restoring an unchanged snapshot must not write frozen flags or skip real changes."""
    snapshot = _snapshot_torch_settings()
    initial = tuple(not value if changed else value for value in snapshot["cudnn"])
    writes = []

    class CudnnState:
        deterministic, benchmark, allow_tf32 = initial

        def __setattr__(self, name, value):
            assert changed, "Unchanged frozen flags must not be assigned"
            writes.append(name)
            object.__setattr__(self, name, value)

    state = CudnnState()
    monkeypatch.setattr(torch.backends, "cudnn", state)
    _restore_torch_settings(snapshot)
    names = ("deterministic", "benchmark", "allow_tf32")
    assert tuple(getattr(state, name) for name in names) == snapshot["cudnn"]
    assert writes == (list(names) if changed else [])


@pytest.fixture
def native_parent():
    """Use one real WORLD parent, retaining no CP subgroup ProcessGroups."""
    Utils.initialize_distributed()
    if dist.get_world_size() != 8:
        pytest.skip("mixed CP3/CP5 layouts require eight ranks")
    transport = pytest.importorskip("transformer_engine.pytorch.attention.native_cp_transport")
    parent = dist.group.WORLD
    dist.barrier(device_ids=[torch.cuda.current_device()])
    reason = transport.get_native_cp_transport_unavailable_reason(parent)
    if reason:
        pytest.skip(reason)
    transport.initialize_native_cp_transport(parent, 4 * 1024 * 1024)
    try:
        yield parent
    finally:
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])
        transport.destroy_native_cp_transport(parent)


def _group(parent, sizes):
    ranks = dist.get_process_group_ranks(parent)
    start = 0
    for size in sizes:
        if start <= parent.rank() < start + size:
            group = LogicalCPGroup.from_parent_interval(ranks, start, size, dist.get_rank())
            bind_native_mtp_cp_group(group, parent)
            return group
        start += size
    raise AssertionError("layout must cover the parent")


@pytest.mark.parametrize("sizes", [(3, 5), (5, 3), (1,) * 8])
def test_native_reduction_and_contiguous_roll_replays(native_parent, sizes):
    group = _group(native_parent, sizes)
    value = torch.full((127,), dist.get_rank() + 1.0, device="cuda", requires_grad=True)
    outputs, gradients = assert_replays_bit_exact(
        lambda x: reduce_from_tensor_model_parallel_region(x, group),
        (value,),
        contention=True,
        what=f"native reduction {sizes}",
    )
    assert bytes_equal(outputs["out"], torch.full_like(value, sum(r + 1 for r in group.ranks)))
    assert bytes_equal(gradients["in[0]"], torch.ones_like(value))

    length = 16
    first = group.rank() * length
    value = torch.arange(first, first + length, device="cuda", dtype=torch.float32)
    value.requires_grad_()
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, length * group.size()], device="cuda", dtype=torch.int32),
        local_cp_size=group.size(),
        cp_group=group if group.size() > 1 else None,
        cp_singleton_group=group if group.size() == 1 else None,
        cp_partition_mode="contiguous",
    )
    outputs, gradients = assert_replays_bit_exact(
        lambda x: roll_contiguous(x, -1, group, params),
        (value,),
        contention=True,
        what=f"native differentiable MTP boundary {sizes}",
    )
    expected = value.detach() + 1
    if group.rank() == group.size() - 1:
        expected[-1] = 0
    expected_grad = torch.ones_like(value)
    if group.rank() == 0:
        expected_grad[0] = 0
    assert bytes_equal(outputs["out"], expected)
    assert bytes_equal(gradients["in[0]"], expected_grad)


@pytest.mark.parametrize("sizes", [(3, 5), (5, 3)])
@pytest.mark.parametrize("transport_kind", ["native", "legacy"])
def test_native_zigzag_token_halos_replay(native_parent, sizes, transport_kind):
    """Legacy control must use its subgroup communicator, not implicitly WORLD."""
    group = _group(native_parent, sizes)
    if transport_kind == "legacy":
        first = 0
        for size in sizes:
            ranks = list(range(first, first + size))
            candidate = dist.new_group(ranks)
            if dist.get_rank() in ranks:
                group = candidate
            first += size
    length = 32 * group.size()
    tokens = torch.arange(length, device="cuda").view(1, -1)

    def shard(tensor):
        chunks = tensor.chunk(2 * group.size(), -1)
        return torch.cat((chunks[group.rank()], chunks[-group.rank() - 1]), -1).contiguous()

    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, length], device="cuda", dtype=torch.int32),
        local_cp_size=group.size(),
        cp_group=group,
    )
    outputs, _ = assert_replays_bit_exact(
        lambda x: roll_tensor(x, cp_group=group, packed_seq_params=params, return_sum=False)[0],
        (shard(tokens),),
        backward=False,
        contention=True,
        what=f"native zigzag token halos {sizes}",
    )
    expected = torch.roll(tokens, -1, -1)
    expected[..., -1] = 0
    assert bytes_equal(outputs["out"], shard(expected))
    if transport_kind == "legacy":
        dist.destroy_process_group(group)


def test_native_attention_runtime_cp_replays(native_parent, monkeypatch):
    """Real TE fused attention and Q/K/V backward survive CP1/3/5 topology switches."""
    Utils.initialize_model_parallel()
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        pg.dp_cp = native_parent
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=64,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            deterministic_mode=True,
            use_native_cp_transport=True,
            dynamic_context_parallel=True,
            max_seqlen_per_dp_cp_rank=1024,
        )
        core = TEDotProductAttention(
            config, 1, AttnMaskType.causal, attention_type="self", pg_collection=pg
        ).cuda()
        original_group, original_ranks = core.cp_group, core.cp_global_ranks
        lengths = (960, 1920)  # Both divisible by 2*CP3 and 2*CP5.
        cu = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
        references = {}
        for sizes in ((3, 5), (1,) * 8, (5, 3), (3, 5)):
            group = _group(native_parent, sizes)
            local_tokens = sum(lengths) // group.size()
            # Recreate identical operands for repeated layouts, independent of RNG state.
            generator = torch.Generator(device="cuda").manual_seed(1234 + group.rank())
            inputs = {
                name: torch.randn(
                    local_tokens,
                    heads,
                    64,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                ).requires_grad_()
                for name, heads in (("query", 4), ("key", 2), ("value", 2))
            }
            packed = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu,
                cu_seqlens_kv_padded=cu,
                max_seqlen_q=max(lengths),
                max_seqlen_kv=max(lengths),
                local_cp_size=group.size(),
                cp_group=group if group.size() > 1 else None,
                cp_singleton_group=group if group.size() == 1 else None,
            )
            result = assert_replays_bit_exact(
                lambda **operands: core(
                    **operands,
                    attention_mask=None,
                    attn_mask_type=AttnMaskType.causal,
                    packed_seq_params=packed,
                ),
                inputs,
                contention=True,
                what=f"native TE attention {sizes}",
            )
            assert set(result[1]) == {"in.query", "in.key", "in.value"}
            assert core.cp_group is original_group and core.cp_global_ranks == original_ranks
            if sizes in references:
                for expected, actual in zip(references[sizes], result):
                    assert expected.keys() == actual.keys()
                    assert all(bytes_equal(expected[name], actual[name]) for name in expected)
            references[sizes] = result
    finally:
        Utils.destroy_model_parallel()

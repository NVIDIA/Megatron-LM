# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay native logical CP reductions and MTP halos without subgroup communicators."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.context_parallel.sequence_roll import roll_contiguous
from megatron.core.dynamic_cp_group import LogicalCPGroup
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
    exchange_cp_boundary_hidden,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    bind_native_mtp_cp_group,
    roll_tensor,
)
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


@pytest.mark.parametrize("replicated", [False, True])
def test_mtp_precomputed_sp_mask_replays(replicated):
    """Precomputed SP masks are not sharded twice; full masks are sharded exactly once."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        tp = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"]).tp
        full_mask = (torch.arange(16, device="cuda") % 3 == 0).repeat(2, 1)
        local_mask = full_mask.chunk(tp.size(), -1)[tp.rank()].contiguous()

        def transformer(hidden_states, decoder_input, padding_mask, **kwargs):
            assert bytes_equal(padding_mask, local_mask)
            return (hidden_states + decoder_input).masked_fill(padding_mask.T.unsqueeze(-1), 0)

        layer = SimpleNamespace(
            config=SimpleNamespace(sequence_parallel=True, recompute_granularity=None),
            tp_group=tp,
            training=True,
            mtp_layer_pattern=None,
            _get_precomputed_embeddings=lambda **kwargs: (
                kwargs["decoder_input"],
                kwargs["hidden_states"],
            ),
            _proj_and_transformer_layer=transformer,
        )
        inputs = {
            "hidden_states": torch.randn(8, 2, 16, device="cuda", requires_grad=True),
            "decoder_input": torch.randn(8, 2, 16, device="cuda", requires_grad=True),
            "padding_mask": full_mask if replicated else local_mask,
        }
        outputs, gradients = assert_replays_bit_exact(
            lambda **kwargs: MultiTokenPredictionLayer.forward(
                layer, None, None, attention_mask=None, **kwargs
            )[0],
            inputs,
            contention=True,
            what="precomputed MTP SP mask dispatch",
        )
        expected = (inputs["hidden_states"] + inputs["decoder_input"]).masked_fill(
            local_mask.T.unsqueeze(-1), 0
        )
        assert bytes_equal(outputs["out"], expected)
        expected_grad = torch.ones_like(expected).masked_fill(local_mask.T.unsqueeze(-1), 0)
        assert set(gradients) == {"in.hidden_states", "in.decoder_input"}
        assert all(bytes_equal(gradient, expected_grad) for gradient in gradients.values())
    finally:
        Utils.destroy_model_parallel()


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


@pytest.mark.parametrize("sizes", [(3, 5), (5, 3), (1,) * 8])
def test_native_all_gather_reduce_scatter_replays(native_parent, sizes):
    """Logical AG/RS and both autograd directions match integer-exact references."""
    from transformer_engine.pytorch.attention.native_cp_transport import get_native_cp_transport

    group = _group(native_parent, sizes)
    transport = get_native_cp_transport(native_parent)
    # Each FP32 shard exceeds the entire arena, forcing multiple bounded
    # chunks even if staging later stops splitting the arena into two halves.
    rows = transport.payload_bytes // (3 * 4) + 17

    def pattern(count):
        return torch.arange(count * 6, device="cuda", dtype=torch.float32).view(count, 6) % 17

    base = pattern(rows)
    value = (base + 32 * dist.get_rank())[:, ::2].requires_grad_()
    assert not value.is_contiguous() and value.nbytes > transport.payload_bytes

    def gather_then_scatter(tensor):
        gathered = gather_from_sequence_parallel_region(tensor, group=group)
        reduced = reduce_scatter_to_sequence_parallel_region(gathered, group=group)
        return gathered.detach(), reduced

    outputs, gradients = assert_replays_bit_exact(
        gather_then_scatter,
        (value,),
        contention=True,
        what=f"chunked logical CP AG then RS {sizes}",
    )
    expected_gather = torch.cat([(base + 32 * rank)[:, ::2] for rank in group.ranks])
    assert bytes_equal(outputs["out[0]"], expected_gather)
    assert bytes_equal(outputs["out[1]"], group.size() * value.detach())
    assert bytes_equal(gradients["in[0]"], torch.full_like(value, group.size()))

    # Different replicas contribute different values: a local multiply cannot
    # accidentally satisfy this reduce-scatter oracle. Distinct upstream
    # gradients also verify the logical ordering of its all-gather backward.
    full_pattern = pattern(group.size() * rows)
    full = (full_pattern + 32 * dist.get_rank())[:, ::2].requires_grad_()
    upstream = torch.full_like(value, dist.get_rank() + 1)
    outputs, gradients = assert_replays_bit_exact(
        lambda tensor: reduce_scatter_to_sequence_parallel_region(tensor, group=group),
        (full,),
        grad_outputs={"out": upstream},
        contention=True,
        what=f"chunked logical CP RS and AG backward {sizes}",
    )
    selected = full_pattern[group.rank() * rows : (group.rank() + 1) * rows, ::2]
    expected = group.size() * selected + 32 * sum(group.ranks)
    expected_grad = torch.cat([torch.full_like(value, rank + 1) for rank in group.ranks])
    assert bytes_equal(outputs["out"], expected)
    assert bytes_equal(gradients["in[0]"], expected_grad)


@pytest.mark.parametrize("sizes", [(3, 5), (5, 3), (1,) * 8])
def test_native_dsv4_boundary_exchange_replays(native_parent, sizes):
    """Native left halos and reverse gradient halos retain physical boundary zeros."""
    from transformer_engine.pytorch.attention.native_cp_transport import set_native_cp_parent_group

    group = _group(native_parent, sizes)
    # Bind CP1 too: its no-peer exchange is an explicit native no-op.
    set_native_cp_parent_group(group, native_parent)
    base = torch.arange(19 * 10, device="cuda", dtype=torch.float32).view(19, 1, 10) % 17
    value = (base + 32 * dist.get_rank())[..., ::2].requires_grad_()
    outputs, gradients = assert_replays_bit_exact(
        lambda tensor: exchange_cp_boundary_hidden(tensor, 4, 8, group),
        (value,),
        contention=True,
        what=f"native DSv4 boundary exchange {sizes}",
    )
    expected = torch.zeros_like(value[-8:])
    if group.rank() > 0:
        expected = (base[-8:] + 32 * group.ranks[group.rank() - 1])[..., ::2]
    expected_grad = torch.zeros_like(value)
    if group.rank() + 1 < group.size():
        expected_grad[-8:] = 1
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

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.swa_context_parallel import (
    _build_swa_p2p_halo_plan,
    build_swa_p2p_nonpacked_segments,
    build_swa_p2p_packed_segments,
    get_swa_p2p_halos,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


def _chunk_for_rank_segment(rank: int, segment: int, cp_size: int) -> int:
    if segment == 0:
        return rank
    return 2 * cp_size - rank - 1


def _positions_for_rank(rank: int, cp_size: int, chunk_len: int, device=None) -> torch.Tensor:
    positions = []
    for segment in (0, 1):
        chunk = _chunk_for_rank_segment(rank, segment, cp_size)
        start = chunk * chunk_len
        positions.extend(range(start, start + chunk_len))
    return torch.tensor(positions, dtype=torch.float32, device=device)


def _part_positions(part, cp_size: int, chunk_len: int) -> list[int]:
    chunk = _chunk_for_rank_segment(part.src_rank, part.src_segment, cp_size)
    start = chunk * chunk_len + part.src_offset
    return list(range(start, start + part.length))


def test_swa_p2p_halo_plan_matches_zigzag_predecessors():
    cp_size = 4
    chunk_len = 3
    window_size = 5

    segments_by_rank = build_swa_p2p_nonpacked_segments(2 * chunk_len, cp_size)
    _, parts_by_dst = _build_swa_p2p_halo_plan(segments_by_rank, window_size)

    for rank in range(cp_size):
        for segment in (0, 1):
            dst_chunk = _chunk_for_rank_segment(rank, segment, cp_size)
            segment_start = dst_chunk * chunk_len
            expected = list(range(max(0, segment_start - window_size), segment_start))

            actual = []
            for part in parts_by_dst[(rank, segment)]:
                actual.extend(_part_positions(part, cp_size, chunk_len))

            assert actual == expected


def test_swa_p2p_packed_segments_keep_sequence_boundaries():
    cp_size = 2
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

    segments_by_rank = build_swa_p2p_packed_segments(cu_seqlens, cp_size)
    _, parts_by_dst = _build_swa_p2p_halo_plan(segments_by_rank, window_size=4)

    seq1_first_segment_rank0 = 2
    assert segments_by_rank[0][seq1_first_segment_rank0].global_start == 8
    assert parts_by_dst[(0, seq1_first_segment_rank0)] == []


def test_swa_p2p_packed_segments_follow_local_thd_order():
    cp_size = 4
    cu_seqlens = torch.tensor([0, 16, 40], dtype=torch.int32)

    segments_by_rank = build_swa_p2p_packed_segments(cu_seqlens, cp_size)

    rank0_segments = segments_by_rank[0]
    assert [segment.global_start for segment in rank0_segments] == [0, 14, 16, 37]
    assert [segment.length for segment in rank0_segments] == [2, 2, 3, 3]
    assert [segment.local_offset for segment in rank0_segments] == [0, 2, 4, 7]


@pytest.mark.parametrize(
    ("cu_seqlens", "cp_size", "match"),
    [
        (torch.tensor([0, 10], dtype=torch.int32), 4, "divisible"),
        (torch.tensor([0, 8, 7], dtype=torch.int32), 2, "monotonic"),
        (torch.tensor([0, 8], dtype=torch.int32), 0, "positive cp_size"),
    ],
)
def test_swa_p2p_packed_segments_reject_invalid_metadata(cu_seqlens, cp_size, match):
    with pytest.raises(ValueError, match=match):
        build_swa_p2p_packed_segments(cu_seqlens, cp_size)


@pytest.mark.skipif(Utils.world_size < 2, reason="swa_p2p halo exchange needs >=2 ranks")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="unit distributed tests use NCCL/CUDA")
def test_swa_p2p_halo_exchange_forward_backward():
    cp_size = 2
    chunk_len = 2
    window_size = 3

    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = cp_group.rank()
        device = torch.device("cuda", torch.cuda.current_device())

        local_positions = _positions_for_rank(cp_rank, cp_size, chunk_len, device=device)
        input_ = local_positions.view(-1, 1, 1, 1).clone().detach().requires_grad_(True)

        front_halo, back_halo = get_swa_p2p_halos(input_, window_size, cp_group)

        for segment, halo in enumerate((front_halo, back_halo)):
            dst_chunk = _chunk_for_rank_segment(cp_rank, segment, cp_size)
            segment_start = dst_chunk * chunk_len
            expected_positions = torch.arange(
                max(0, segment_start - window_size),
                segment_start,
                dtype=input_.dtype,
                device=device,
            )
            torch.testing.assert_close(halo.flatten(), expected_positions)

        (front_halo.sum() + back_halo.sum()).backward()

        global_seq_len = 2 * cp_size * chunk_len
        expected_global_grad = torch.zeros(global_seq_len, dtype=input_.dtype, device=device)
        for rank in range(cp_size):
            for segment in (0, 1):
                dst_chunk = _chunk_for_rank_segment(rank, segment, cp_size)
                segment_start = dst_chunk * chunk_len
                expected_global_grad[max(0, segment_start - window_size) : segment_start] += 1

        expected_local_grad = expected_global_grad.index_select(0, local_positions.long()).view_as(
            input_
        )
        torch.testing.assert_close(input_.grad, expected_local_grad)
    finally:
        Utils.destroy_model_parallel()


def _zigzag_split(tensor: torch.Tensor, cp_rank: int, cp_size: int, dim: int = 0):
    if cp_size == 1:
        return tensor
    chunk_len = tensor.size(dim) // (2 * cp_size)
    front = tensor.narrow(dim, cp_rank * chunk_len, chunk_len)
    back = tensor.narrow(dim, (2 * cp_size - cp_rank - 1) * chunk_len, chunk_len)
    return torch.cat((front, back), dim=dim).contiguous()


def _zigzag_merge(chunks: list[torch.Tensor], cp_size: int) -> torch.Tensor:
    chunk_len = chunks[0].size(0) // 2
    parts = [None] * (2 * cp_size)
    for rank, chunk in enumerate(chunks):
        parts[rank] = chunk[:chunk_len]
        parts[2 * cp_size - rank - 1] = chunk[chunk_len:]
    return torch.cat(parts, dim=0)


class _GatherZigZag(torch.autograd.Function):
    """Gather non-packed zigzag CP outputs with backward sharding."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, cp_size: int):
        ctx.cp_size = cp_size
        ctx.cp_rank = parallel_state.get_context_parallel_rank()
        gathered = [torch.empty_like(local) for _ in range(cp_size)]
        dist.all_gather(
            gathered, local.contiguous(), group=parallel_state.get_context_parallel_group()
        )
        return _zigzag_merge(gathered, cp_size)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _zigzag_split(grad, ctx.cp_rank, ctx.cp_size).contiguous(), None


class _GatherPackedZigZag(torch.autograd.Function):
    """Gather packed THD zigzag CP outputs with backward sharding."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, seqlens: tuple[int, ...], cp_size: int):
        ctx.seqlens = seqlens
        ctx.cp_size = cp_size
        ctx.cp_rank = parallel_state.get_context_parallel_rank()
        gathered = [torch.empty_like(local) for _ in range(cp_size)]
        dist.all_gather(
            gathered, local.contiguous(), group=parallel_state.get_context_parallel_group()
        )

        outputs = []
        offsets = [0 for _ in range(cp_size)]
        for seq_len in seqlens:
            local_len = seq_len // cp_size
            seq_chunks = []
            for rank, rank_tensor in enumerate(gathered):
                seq_chunks.append(rank_tensor[offsets[rank] : offsets[rank] + local_len])
                offsets[rank] += local_len
            outputs.append(_zigzag_merge(seq_chunks, cp_size))
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        offset = 0
        chunks = []
        for seq_len in ctx.seqlens:
            seq_grad = grad[offset : offset + seq_len, 0, :]
            chunks.append(_zigzag_split(seq_grad, ctx.cp_rank, ctx.cp_size))
            offset += seq_len
        return torch.cat(chunks, dim=0).unsqueeze(1).contiguous(), None, None


def _make_config(
    *,
    num_layers: int,
    cp_size: int,
    cp_comm_type,
    hidden_size=64,
    num_attention_heads=4,
    num_query_groups=4,
    window_size=(6, 0),
    window_attn_skip_freq=None,
) -> TransformerConfig:
    hierarchical_context_parallel_sizes = None
    cp_comm_types = cp_comm_type if isinstance(cp_comm_type, list) else [cp_comm_type]
    if "a2a+p2p" in cp_comm_types:
        hierarchical_context_parallel_sizes = [2, 2]

    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=4 * hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        context_parallel_size=cp_size,
        cp_comm_type=cp_comm_type,
        no_rope_freq=[1] * num_layers,
        window_size=window_size,
        window_attn_skip_freq=window_attn_skip_freq,
        hierarchical_context_parallel_sizes=hierarchical_context_parallel_sizes,
    )


def _build_attention(
    config: TransformerConfig,
    cp_comm_type,
    layer_number: int = 1,
    attn_mask_type: AttnMaskType = AttnMaskType.causal,
):
    attention = SelfAttention(
        config,
        get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
        layer_number=layer_number,
        attn_mask_type=attn_mask_type,
        cp_comm_type=cp_comm_type,
    )
    return attention.cuda()


def _broadcast_module_from_cp_rank0(module: torch.nn.Module):
    cp_group = parallel_state.get_context_parallel_group()
    src_rank = dist.get_global_rank(cp_group, 0)
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor, src=src_rank, group=cp_group)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol=2e-2, rtol=2e-2):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def _assert_attention_parity(
    attention_mask: torch.Tensor | None = None,
    *,
    hidden_size=64,
    num_attention_heads=8,
    num_query_groups=2,
    window_size=(6, 0),
):
    cp_size = 4
    seq_len = 64
    batch_size = 1
    device = torch.device("cuda", torch.cuda.current_device())
    cp_rank = parallel_state.get_context_parallel_rank()
    attn_mask_type = AttnMaskType.arbitrary if attention_mask is not None else AttnMaskType.causal

    model_parallel_cuda_manual_seed(1234)
    cp_config = _make_config(
        num_layers=1,
        cp_size=cp_size,
        cp_comm_type="swa_p2p",
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        window_size=window_size,
    )
    ref_config = _make_config(
        num_layers=1,
        cp_size=1,
        cp_comm_type=None,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        window_size=window_size,
    )
    cp_attention = _build_attention(cp_config, "swa_p2p", attn_mask_type=attn_mask_type)
    _broadcast_module_from_cp_rank0(cp_attention)
    ref_attention = _build_attention(ref_config, None, attn_mask_type=attn_mask_type)
    ref_attention.load_state_dict(cp_attention.state_dict())

    torch.manual_seed(123)
    full_hidden = torch.randn(seq_len, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    torch.manual_seed(456)
    grad_output = torch.randn_like(full_hidden)

    local_hidden = _zigzag_split(full_hidden, cp_rank, cp_size).detach().requires_grad_(True)
    ref_hidden = full_hidden.detach().clone().requires_grad_(True)

    local_mask = None
    if attention_mask is not None:
        local_mask = _zigzag_split(attention_mask, cp_rank, cp_size, dim=2)

    cp_output, _ = cp_attention(local_hidden, local_mask)
    gathered_output = _GatherZigZag.apply(cp_output, cp_size)
    ref_output, _ = ref_attention(ref_hidden, attention_mask)

    _assert_close(gathered_output.detach(), ref_output.detach())

    gathered_output.backward(grad_output)
    ref_output.backward(grad_output)

    _assert_close(local_hidden.grad, _zigzag_split(ref_hidden.grad, cp_rank, cp_size))
    cp_group = parallel_state.get_context_parallel_group()
    for (_name, cp_param), ref_param in zip(
        cp_attention.named_parameters(), ref_attention.parameters()
    ):
        if cp_param.grad is None or ref_param.grad is None:
            assert cp_param.grad is None and ref_param.grad is None
            continue
        cp_grad = cp_param.grad.detach().clone()
        dist.all_reduce(cp_grad, group=cp_group)
        _assert_close(cp_grad, ref_param.grad, atol=3e-2, rtol=3e-2)


def _make_reset_attention_mask(
    seq_len: int, document_ends: tuple[int, ...], device, left_window: int
) -> torch.Tensor:
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    start = 0
    for end in document_ends:
        mask[start:end, start:end] = torch.triu(
            torch.ones((end - start, end - start), dtype=torch.bool, device=device), diagonal=1
        )
        start = end
    query_positions = torch.arange(seq_len, device=device).view(-1, 1)
    key_positions = torch.arange(seq_len, device=device).view(1, -1)
    mask |= key_positions < query_positions - left_window
    return mask.view(1, 1, seq_len, seq_len)


def _make_packed_seq_params(seqlens: tuple[int, ...], device) -> PackedSeqParams:
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    for idx, seq_len in enumerate(seqlens):
        cu[idx + 1] = cu[idx] + seq_len
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(seqlens),
        max_seqlen_kv=max(seqlens),
    )


@pytest.mark.skipif(Utils.world_size < 4, reason="swa_p2p integration tests need CP=4")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="unit distributed tests use NCCL/CUDA")
@pytest.mark.skipif(not is_te_min_version("1.2.0"), reason="SWA requires Transformer Engine >= 1.2")
def test_swa_p2p_attention_matches_cp1_swa_forward_backward():
    Utils.initialize_model_parallel(context_parallel_size=4)
    try:
        _assert_attention_parity()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(Utils.world_size < 4, reason="swa_p2p integration tests need CP=4")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="unit distributed tests use NCCL/CUDA")
@pytest.mark.skipif(not is_te_min_version("1.2.0"), reason="SWA requires Transformer Engine >= 1.2")
def test_swa_p2p_attention_large_window_matches_cp1_swa_forward_backward():
    Utils.initialize_model_parallel(context_parallel_size=4)
    try:
        _assert_attention_parity(window_size=(10, 0))
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(Utils.world_size < 4, reason="swa_p2p integration tests need CP=4")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="unit distributed tests use NCCL/CUDA")
@pytest.mark.skipif(not is_te_min_version("1.2.0"), reason="SWA requires Transformer Engine >= 1.2")
def test_swa_p2p_attention_with_reset_mask_matches_cp1_swa():
    Utils.initialize_model_parallel(context_parallel_size=4)
    try:
        mask = _make_reset_attention_mask(
            seq_len=64,
            document_ends=(19, 43, 64),
            device=torch.device("cuda", torch.cuda.current_device()),
            left_window=6,
        )
        _assert_attention_parity(mask)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(Utils.world_size < 4, reason="swa_p2p integration tests need CP=4")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="unit distributed tests use NCCL/CUDA")
@pytest.mark.skipif(not is_te_min_version("1.2.0"), reason="SWA requires Transformer Engine >= 1.2")
def test_swa_p2p_packed_attention_matches_cp1_swa_forward_backward():
    cp_size = 4
    seqlens = (32, 24)
    device = torch.device("cuda", torch.cuda.current_device())

    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    try:
        cp_rank = parallel_state.get_context_parallel_rank()
        model_parallel_cuda_manual_seed(1234)
        hidden_size = 64
        cp_attention = _build_attention(
            _make_config(
                num_layers=1,
                cp_size=cp_size,
                cp_comm_type="swa_p2p",
                hidden_size=hidden_size,
                num_attention_heads=8,
                num_query_groups=2,
            ),
            "swa_p2p",
        )
        _broadcast_module_from_cp_rank0(cp_attention)
        ref_attention = _build_attention(
            _make_config(
                num_layers=1,
                cp_size=1,
                cp_comm_type=None,
                hidden_size=hidden_size,
                num_attention_heads=8,
                num_query_groups=2,
            ),
            None,
        )
        ref_attention.load_state_dict(cp_attention.state_dict())

        torch.manual_seed(123)
        seq_tensors = [
            torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device)
            for seq_len in seqlens
        ]
        full_hidden = torch.cat(seq_tensors, dim=0).unsqueeze(1)
        local_hidden = torch.cat(
            [_zigzag_split(seq_tensor, cp_rank, cp_size) for seq_tensor in seq_tensors], dim=0
        ).unsqueeze(1)
        local_hidden = local_hidden.detach().requires_grad_(True)
        ref_hidden = full_hidden.detach().clone().requires_grad_(True)

        packed_seq_params = _make_packed_seq_params(seqlens, device)
        cp_output, _ = cp_attention(
            local_hidden, attention_mask=None, packed_seq_params=packed_seq_params
        )
        gathered_output = _GatherPackedZigZag.apply(cp_output, seqlens, cp_size)
        ref_output, _ = ref_attention(
            ref_hidden, attention_mask=None, packed_seq_params=packed_seq_params
        )

        _assert_close(gathered_output.detach(), ref_output.detach())

        torch.manual_seed(456)
        grad_output = torch.randn_like(ref_output)
        gathered_output.backward(grad_output)
        ref_output.backward(grad_output)

        grad_chunks = []
        offset = 0
        for seq_len in seqlens:
            grad_chunks.append(
                _zigzag_split(ref_hidden.grad[offset : offset + seq_len, 0], cp_rank, cp_size)
            )
            offset += seq_len
        expected_local_grad = torch.cat(grad_chunks, dim=0).unsqueeze(1)
        _assert_close(local_hidden.grad, expected_local_grad)

        cp_group = parallel_state.get_context_parallel_group()
        for cp_param, ref_param in zip(cp_attention.parameters(), ref_attention.parameters()):
            if cp_param.grad is None or ref_param.grad is None:
                assert cp_param.grad is None and ref_param.grad is None
                continue
            cp_grad = cp_param.grad.detach().clone()
            dist.all_reduce(cp_grad, group=cp_group)
            _assert_close(cp_grad, ref_param.grad, atol=3e-2, rtol=3e-2)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(Utils.world_size < 4, reason="swa_p2p integration tests need CP=4")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="unit distributed tests use NCCL/CUDA")
@pytest.mark.skipif(not is_te_min_version("1.2.0"), reason="SWA requires Transformer Engine >= 1.2")
def test_swa_p2p_packed_attention_rejects_explicit_attention_mask():
    cp_size = 4
    seq_len = 32
    device = torch.device("cuda", torch.cuda.current_device())

    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    try:
        cp_rank = parallel_state.get_context_parallel_rank()
        hidden_size = 64
        cp_attention = _build_attention(
            _make_config(
                num_layers=1,
                cp_size=cp_size,
                cp_comm_type="swa_p2p",
                hidden_size=hidden_size,
                num_attention_heads=8,
                num_query_groups=2,
            ),
            "swa_p2p",
            attn_mask_type=AttnMaskType.arbitrary,
        )

        torch.manual_seed(123)
        full_hidden = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device)
        local_hidden = _zigzag_split(full_hidden, cp_rank, cp_size).unsqueeze(1)
        packed_seq_params = _make_packed_seq_params((seq_len,), device)
        attention_mask = torch.zeros((1, 1, local_hidden.size(0), seq_len), dtype=torch.bool)
        attention_mask = attention_mask.to(device)

        with pytest.raises(NotImplementedError, match="explicit attention_mask"):
            cp_attention(
                local_hidden, attention_mask=attention_mask, packed_seq_params=packed_seq_params
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(Utils.world_size < 4, reason="mixed CP communication test needs CP=4")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="unit distributed tests use NCCL/CUDA")
@pytest.mark.skipif(not is_te_min_version("1.12.0"), reason="a2a+p2p requires TE >= 1.12")
def test_swa_p2p_can_mix_with_a2a_p2p_full_attention_layer():
    Utils.initialize_model_parallel(
        context_parallel_size=4, hierarchical_context_parallel_sizes=[2, 2]
    )
    try:
        config = _make_config(
            num_layers=2,
            cp_size=4,
            cp_comm_type=["swa_p2p", "a2a+p2p"],
            window_attn_skip_freq=[1, 0],
        )
        spec = get_gpt_layer_with_transformer_engine_spec()
        layer_1 = TransformerLayer(config, spec.submodules, layer_number=1).cuda()
        layer_2 = TransformerLayer(config, spec.submodules, layer_number=2).cuda()

        torch.manual_seed(123)
        hidden_states = torch.randn(16, 1, 64, dtype=torch.bfloat16, device="cuda")
        hidden_states.requires_grad_(True)
        output, _ = layer_1(hidden_states=hidden_states, attention_mask=None)
        output, _ = layer_2(hidden_states=output, attention_mask=None)
        output.float().sum().backward()
        assert hidden_states.grad is not None
    finally:
        Utils.destroy_model_parallel()

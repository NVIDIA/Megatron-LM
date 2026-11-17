# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed document, zigzag CP, and both training-phase correctness contracts."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import (
    simplified_sparse_attention_cudnn as adapter,
)


def _positions(lengths, size, rank, device="cpu"):
    positions, offset = [], 0
    for length in lengths:
        if size == 1:
            positions.extend(range(offset, offset + length))
        else:
            chunk = length // (2 * size)
            positions.extend(range(offset + rank * chunk, offset + (rank + 1) * chunk))
            positions.extend(
                range(offset + length - (rank + 1) * chunk, offset + length - rank * chunk)
            )
        offset += length
    return torch.tensor(positions, dtype=torch.int64, device=device)


def _packed(lengths, real_lengths, size, rank, device="cpu", group=None):
    physical = torch.tensor([0, *lengths], dtype=torch.int32, device=device).cumsum(0).int()
    logical = torch.tensor([0, *real_lengths], dtype=torch.int32, device=device).cumsum(0).int()
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical.clone(),
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical.clone(),
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        cp_group=group,
        # Packed metadata names a CP size only together with its group.
        local_cp_size=size if group is not None else None,
    )


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_row_bounds_and_flash_teacher_match_each_rank_document_positions(monkeypatch, size, packed):
    lengths = [16, 24] if packed else [40]
    real_lengths = [13, 21] if packed else lengths
    key = torch.arange(sum(lengths)).reshape(-1, 1, 1)
    starts = torch.cat(
        [torch.full((length,), sum(lengths[:i])) for i, length in enumerate(lengths)]
    )
    for rank in range(size):
        group = SimpleNamespace(size=lambda: size, rank=lambda: rank)
        monkeypatch.setattr(
            torch.distributed, "all_reduce", lambda count, **kwargs: count.fill_(sum(real_lengths))
        )
        positions = _positions(lengths, size, rank)
        params = _packed(lengths, real_lengths, size, rank, group=group) if packed else None
        query = torch.empty(positions.numel(), 1, 16, 256)
        layout = adapter.build_attention_layout(query, params, group)
        torch.testing.assert_close(layout.key_starts, starts[positions].to(torch.int32))
        torch.testing.assert_close(layout.key_ends, (positions + 1).to(torch.int32))
        assert layout.num_real_tokens.item() == sum(real_lengths)
        if size > 1:
            gathered = torch.cat([key[_positions(lengths, size, other)] for other in range(size)])
            torch.testing.assert_close(gathered[layout.key_reorder], key)
        teacher_key, _, cu_q, cu_k, max_q, max_k = adapter._phase1_flash_inputs(key, key, layout)
        assert cu_q.dtype == cu_k.dtype == torch.int32
        assert max_q >= int((cu_q[1:] - cu_q[:-1]).max())
        assert max_k >= int((cu_k[1:] - cu_k[:-1]).max())
        for segment in range(cu_q.numel() - 1):
            q_start, q_end = int(cu_q[segment]), int(cu_q[segment + 1])
            k_start, k_end = int(cu_k[segment]), int(cu_k[segment + 1])
            for local_row in range(q_start, q_end):
                # FlashAttention aligns the two segments at their right ends.
                visible_end = k_end - (q_end - local_row - 1)
                expected = torch.arange(
                    int(starts[positions[local_row]]), int(positions[local_row]) + 1
                )
                torch.testing.assert_close(teacher_key[k_start:visible_end, 0, 0], expected)


def test_packed_validity_uses_logical_lengths_without_an_explicit_mask():
    params = _packed([16], [13], 1, 0)
    layout = adapter.build_attention_layout(torch.empty(16, 16, 256), params, None)
    torch.testing.assert_close(layout.query_valid, torch.arange(16) < 13)
    assert layout.num_real_tokens.item() == 13


@pytest.mark.parametrize("real_lengths", [[0, 1], [0, 0]])
def test_packed_validity_preserves_empty_documents_and_all_padding_ranks(monkeypatch, real_lengths):
    lengths = [8, 8]
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    params = _packed(lengths, real_lengths, 2, 1, group=group)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda count, **kwargs: count.fill_(sum(real_lengths))
    )
    layout = adapter.build_attention_layout(torch.empty(8, 16, 256), params, group)
    assert not layout.query_valid.any()
    assert layout.num_real_tokens.item() == max(1, sum(real_lengths))


def test_packed_rejects_different_cp_group():
    other_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    params = _packed([16], [13], 1, 0, group=other_group)
    with pytest.raises(ValueError, match="CP group"):
        adapter.build_attention_layout(torch.empty(16, 16, 256), params, None)


def _oracle(inputs, lengths, real_lengths, coefficient, per_token, phase1):
    q, k, v, qi, ki = (tensor[:, 0].float() for tensor in inputs)
    rows = q.shape[0]
    valid = torch.cat(
        [
            torch.arange(length, device=q.device) < count
            for length, count in zip(lengths, real_lengths)
        ]
    )
    allowed = torch.zeros(rows, rows, dtype=torch.bool, device=q.device)
    offset = 0
    for length, count in zip(lengths, real_lengths):
        allowed[offset : offset + count, offset : offset + count] = torch.ones(
            count, count, device=q.device, dtype=torch.bool
        ).tril()
        offset += length
    logits = qi[:, 0] @ ki[:, 0].T * 128**-0.5
    if not phase1:
        indices = logits.detach().masked_fill(~allowed, -torch.inf).topk(512, dim=-1).indices
        allowed &= torch.zeros_like(allowed).scatter_(1, indices, True)
    scores = torch.einsum("qhd,kd->hqk", q, k[:, 0]) * 256**-0.5
    probability = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(-1)
    probability = probability * valid.reshape(1, -1, 1)
    output = torch.einsum("hqk,kd->qhd", probability, v[:, 0]).reshape(rows, 1, -1)
    target = probability.detach().mean(0)
    log_prediction = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min).log_softmax(-1)
    terms = target * (target.clamp_min(1e-10).log() - log_prediction)
    denominator = 1 if per_token else sum(real_lengths)
    loss = terms.masked_fill(~allowed, 0).sum() * coefficient / denominator
    return output, loss


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("phase1", [False, True])
def test_real_cp2_matches_global_outputs_loss_and_all_input_gradients(
    monkeypatch, packed, per_token, phase1
):
    """Compare actual BF16 collectives and kernels to an independent global oracle.

    The longer document exceeds top-k, so phase 2 exercises sparse selection.
    Phase 1 must instead retain full support and leave all backbone inputs frozen.
    """
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel import mappings
    from tests.unit_tests.test_utilities import Utils

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("The cuDNN kernels require SM100 or SM103")
    try:
        kernels = adapter.bind_cudnn_sparse_attention(dense_teacher=phase1)
    except ImportError as error:
        pytest.skip(str(error))
    if Utils.world_size < 2:
        pytest.skip("CP2 requires at least two processes")
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        group = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["cp"]).cp
        rank = group.rank()
        lengths = [1024, 512] if packed else [1536]
        real_lengths = [1019, 509] if packed else lengths
        rows, coefficient = sum(lengths), 0.03
        positions = _positions(lengths, 2, rank, "cuda")
        torch.manual_seed(655)
        global_inputs = tuple(
            torch.randn(rows, 1, heads, width, dtype=torch.bfloat16, device="cuda")
            for heads, width in ((16, 256), (1, 256), (1, 256), (1, 128), (1, 128))
        )
        # Strictly ordered representable scores avoid ambiguous top-k ties.
        global_inputs[3].zero_()
        global_inputs[3][..., :2] = 1
        row = torch.arange(rows, device="cuda")
        global_inputs[4][:, 0, 0, 0] = (row // 32) * 0.125
        global_inputs[4][:, 0, 0, 1] = (row % 32) / 256
        inputs = tuple(tensor[positions].detach().requires_grad_() for tensor in global_inputs)
        refs = tuple(tensor.float().requires_grad_() for tensor in global_inputs)
        params = _packed(lengths, real_lengths, 2, rank, "cuda", group) if packed else None
        layout = adapter.build_attention_layout(inputs[0], params, group)
        communicated = []
        reduce_scatter = mappings.dist_reduce_scatter_func

        def record_reduce_scatter(output, tensor, **kwargs):
            communicated.append(tensor.dtype)
            return reduce_scatter(output, tensor, **kwargs)

        monkeypatch.setattr(mappings, "dist_reduce_scatter_func", record_reduce_scatter)
        run = adapter.run_dense_teacher if phase1 else adapter.run_sparse_attention
        output, loss = run(
            kernels,
            *(tensor[:, 0] if packed else tensor for tensor in inputs),
            **({} if phase1 else {"topk": 512}),
            loss_coeff=coefficient,
            calculate_per_token_loss=per_token,
            layout=layout,
            cp_group=group,
        )
        output = output.reshape(positions.numel(), 1, -1)
        expected_output, expected_loss = _oracle(
            refs, lengths, real_lengths, coefficient, per_token, phase1
        )
        upstream = torch.randn_like(expected_output).to(torch.bfloat16)
        if phase1:
            assert not output.requires_grad
            loss.backward()
            expected_loss.backward()
        else:
            ((output.float() * upstream[positions]).sum() + loss).backward()
            ((expected_output * upstream).sum() + expected_loss).backward()
        torch.testing.assert_close(output.float(), expected_output[positions], rtol=2e-2, atol=2e-2)
        total_loss = loss.detach().clone()
        torch.distributed.all_reduce(total_loss, group=group)
        torch.testing.assert_close(total_loss, expected_loss, rtol=2e-2, atol=2e-4)
        assert communicated == [torch.bfloat16] * (1 if phase1 else 3)
        for index, (actual, expected) in enumerate(zip(inputs, refs)):
            if phase1 and index < 3:
                assert actual.grad is None
                continue
            expected_grad = expected.grad[positions]
            torch.testing.assert_close(actual.grad.float(), expected_grad, rtol=4e-2, atol=4e-2)
            relative_error = torch.linalg.vector_norm(
                actual.grad.float() - expected_grad
            ) / torch.linalg.vector_norm(expected_grad)
            assert relative_error.item() < 0.08
            if packed:
                assert torch.count_nonzero(actual.grad[~layout.query_valid]) == 0
    finally:
        Utils.destroy_model_parallel()

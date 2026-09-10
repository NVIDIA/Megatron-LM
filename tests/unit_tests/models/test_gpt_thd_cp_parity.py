# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical parity coverage for packed inputs through ``pretrain_gpt.py``."""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

import pretrain_gpt
from megatron.core import parallel_state
from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_EQUAL_SEQUENCE_LENGTHS = (16, 16)
_UNEVEN_SEQUENCE_LENGTHS = (3, 13)
_VOCAB_SIZE = 128
_RTOL = 1.0e-2
_ATOL = 2.0e-2


class _NullStragglerTimer:
    """No-op replacement for pretrain_gpt's global straggler timer."""

    def __call__(self, **_kwargs):
        return nullcontext()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _build_model(cp_size: int, mtp: bool = False) -> GPTModel:
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        context_parallel_size=cp_size,
        cp_comm_type="p2p" if cp_size > 1 else None,
        attention_cp_layout="zigzag",
        cross_entropy_loss_fusion=False,
        mtp_num_layers=1 if mtp else None,
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    mtp_block_spec = (
        get_gpt_mtp_block_spec(config=config, spec=layer_spec, use_transformer_engine=True)
        if mtp
        else None
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=max(_EQUAL_SEQUENCE_LENGTHS),
    )
    return model.cuda().eval()


def _make_data_iterator(thd: bool, sequence_lengths: tuple[int, ...]):
    offset = 0
    sequences = []
    for sequence_length in sequence_lengths:
        sequences.append(torch.arange(offset, offset + sequence_length, dtype=torch.int64))
        offset += sequence_length

    if thd:
        tokens = torch.cat(sequences).unsqueeze(0).remainder(_VOCAB_SIZE)
        position_ids = torch.cat(
            [
                torch.arange(sequence_length, dtype=torch.int64)
                for sequence_length in sequence_lengths
            ]
        ).unsqueeze(0)
    else:
        assert len(set(sequence_lengths)) == 1, "BSHD reference requires equal sequence lengths"
        tokens = torch.stack(sequences).remainder(_VOCAB_SIZE)
        position_ids = (
            torch.arange(sequence_lengths[0], dtype=torch.int64)
            .unsqueeze(0)
            .expand_as(tokens)
            .contiguous()
        )

    batch = {
        "tokens": tokens,
        "labels": (tokens + 1).remainder(_VOCAB_SIZE),
        "loss_mask": torch.ones_like(tokens, dtype=torch.float32),
        "position_ids": position_ids,
    }
    if thd:
        batch["cu_seqlens"] = torch.tensor(
            [[0, *torch.tensor(sequence_lengths).cumsum(0).tolist()]], dtype=torch.int32
        )
        batch["max_seqlen"] = torch.tensor([max(sequence_lengths)], dtype=torch.int32)
    return iter([batch])


def _merge_zigzag_rank_chunks(rank_chunks: list[torch.Tensor]) -> torch.Tensor:
    """Undo the two-chunk zigzag partition used by context parallelism."""
    half = rank_chunks[0].numel() // 2
    pieces = [None] * (2 * len(rank_chunks))
    for cp_rank, chunk in enumerate(rank_chunks):
        pieces[cp_rank] = chunk[:half]
        pieces[2 * len(rank_chunks) - cp_rank - 1] = chunk[half:]
    return torch.cat(pieces)


def _gather_global_losses(
    local_losses: torch.Tensor, cp_size: int, thd: bool, sequence_lengths: tuple[int, ...]
) -> torch.Tensor:
    if cp_size == 1:
        return local_losses.flatten()

    gathered = [torch.empty_like(local_losses) for _ in range(cp_size)]
    dist.all_gather(
        gathered, local_losses.contiguous(), group=parallel_state.get_context_parallel_group()
    )

    if thd:
        padded_lengths = [
            ((sequence_length + 2 * cp_size - 1) // (2 * cp_size)) * (2 * cp_size)
            for sequence_length in sequence_lengths
        ]
        local_offsets = [0]
        for padded_length in padded_lengths:
            local_offsets.append(local_offsets[-1] + padded_length // cp_size)

        sequences = []
        for sequence_length, start, end in zip(
            sequence_lengths, local_offsets[:-1], local_offsets[1:]
        ):
            padded_sequence = _merge_zigzag_rank_chunks(
                [rank_losses.flatten()[start:end] for rank_losses in gathered]
            )
            sequences.append(padded_sequence[:sequence_length])
        return torch.cat(sequences)

    sequences = []
    for sequence_idx in range(len(sequence_lengths)):
        sequences.append(
            _merge_zigzag_rank_chunks([rank_losses[sequence_idx] for rank_losses in gathered])
        )
    return torch.cat(sequences)


def _run_pretrain_gpt_forward(
    cp_size: int,
    thd: bool,
    sequence_lengths: tuple[int, ...] = _EQUAL_SEQUENCE_LENGTHS,
    mtp: bool = False,
) -> torch.Tensor:
    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    try:
        model = _build_model(cp_size, mtp)
        micro_batch_size = 1 if thd else len(sequence_lengths)
        args = SimpleNamespace(
            context_parallel_size=cp_size,
            create_attention_mask_in_dataloader=False,
            dataloader_inter_document_masking=False,
            hybrid_context_parallel=False,
            micro_batch_size=micro_batch_size,
            overlap_moe_expert_parallel_comm=False,
            pipeline_model_parallel_size=1,
            seq_length=sum(sequence_lengths) if thd else sequence_lengths[0],
            sequence_packing_scheduler=None,
            sft=thd,
        )
        config = SimpleNamespace(
            attention_cp_layout="zigzag",
            mtp_num_layers=1 if mtp else None,
            pipeline_model_parallel_layout=None,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
        )
        timers = MagicMock()
        original_get_batch = pretrain_gpt.get_batch

        def get_batch_with_layout_assertions(data_iterator, vp_stage=None):
            cp_batch = original_get_batch(data_iterator, vp_stage)
            assert isinstance(cp_batch, ContextParallelBatch)
            if cp_size > 1 and thd:
                assert cp_batch.boundary_layout == "zigzag"
                assert set(cp_batch.batches_by_layout) == {"zigzag"}
                assert set(cp_batch.packed_seq_params_by_layout) == {"zigzag"}
                assert cp_batch.thd_plan is None
                packed_seq_params = cp_batch.get_packed_seq_params("zigzag")
                assert packed_seq_params is not None
                expected_cu_seqlens = torch.tensor(
                    [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
                    dtype=torch.int32,
                    device="cuda",
                )
                torch.testing.assert_close(packed_seq_params.cu_seqlens_q, expected_cu_seqlens)
                torch.testing.assert_close(packed_seq_params.cu_seqlens_kv, expected_cu_seqlens)
                if sequence_lengths == _UNEVEN_SEQUENCE_LENGTHS:
                    assert packed_seq_params.pad_between_seqs
                    assert packed_seq_params.cu_seqlens_q_padded is not None
                    assert packed_seq_params.cu_seqlens_kv_padded is not None
                    cp_alignment = 2 * cp_size
                    padded_sequence_lengths = [
                        ((sequence_length + cp_alignment - 1) // cp_alignment) * cp_alignment
                        for sequence_length in sequence_lengths
                    ]
                    expected_cu_seqlens_padded = torch.tensor(
                        [0, *torch.tensor(padded_sequence_lengths).cumsum(0).tolist()],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    torch.testing.assert_close(
                        packed_seq_params.cu_seqlens_q_padded, expected_cu_seqlens_padded
                    )
                    torch.testing.assert_close(
                        packed_seq_params.cu_seqlens_kv_padded, expected_cu_seqlens_padded
                    )
            return cp_batch

        with (
            torch.no_grad(),
            patch.object(pretrain_gpt, "get_args", return_value=args),
            patch.object(pretrain_gpt, "core_transformer_config_from_args", return_value=config),
            patch.object(pretrain_gpt, "get_batch", side_effect=get_batch_with_layout_assertions),
            patch.object(pretrain_gpt, "get_timers", return_value=timers),
            patch.object(pretrain_gpt, "stimer", _NullStragglerTimer()),
            patch.object(pretrain_gpt, "update_seqlen_stats_from_cu_seqlens"),
        ):
            losses, _ = pretrain_gpt.forward_step(_make_data_iterator(thd, sequence_lengths), model)

        return _gather_global_losses(losses, cp_size, thd, sequence_lengths).float().cpu()
    finally:
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)


def test_pretrain_gpt_thd_cp1_smoke():
    """Exercise the synthetic THD forward path when only one GPU is available."""
    losses = _run_pretrain_gpt_forward(cp_size=1, thd=True)

    assert losses.numel() == sum(_EQUAL_SEQUENCE_LENGTHS)
    assert torch.isfinite(losses).all()


def test_pretrain_gpt_thd_cp2_matches_bshd_cp2_and_thd_cp1():
    """THD+CP2 must match both BSHD+CP2 and the unsharded THD reference."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("THD context-parallel parity requires at least two ranks")

    thd_cp1 = _run_pretrain_gpt_forward(cp_size=1, thd=True)
    bshd_cp2 = _run_pretrain_gpt_forward(cp_size=2, thd=False)
    thd_cp2 = _run_pretrain_gpt_forward(cp_size=2, thd=True)

    assert thd_cp1.shape == bshd_cp2.shape == thd_cp2.shape
    _assert_close(thd_cp2, bshd_cp2)
    _assert_close(thd_cp2, thd_cp1)


def test_pretrain_gpt_uneven_thd_cp2_matches_thd_cp1():
    """THD+CP2 must preserve unequal documents that require CP padding."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("THD context-parallel parity requires at least two ranks")

    thd_cp1 = _run_pretrain_gpt_forward(
        cp_size=1, thd=True, sequence_lengths=_UNEVEN_SEQUENCE_LENGTHS
    )
    thd_cp2 = _run_pretrain_gpt_forward(
        cp_size=2, thd=True, sequence_lengths=_UNEVEN_SEQUENCE_LENGTHS
    )

    assert thd_cp1.shape == thd_cp2.shape
    _assert_close(thd_cp2, thd_cp1)


def test_pretrain_gpt_thd_cp2_with_mtp_matches_thd_cp1():
    """GPT MTP must consume the same zigzag THD view as the main attention block."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("THD context-parallel parity requires at least two ranks")

    thd_cp1 = _run_pretrain_gpt_forward(cp_size=1, thd=True, mtp=True)
    thd_cp2 = _run_pretrain_gpt_forward(cp_size=2, thd=True, mtp=True)

    assert thd_cp1.shape == thd_cp2.shape
    _assert_close(thd_cp2, thd_cp1)

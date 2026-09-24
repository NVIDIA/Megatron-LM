# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed-layout and context-parallel contracts for CuTe sparse attention."""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import dsa_cute_kernels, dsa_gqa
from tests.unit_tests.transformer.experimental_attention_variant.test_dsa_cute_kernels import (
    _config,
    _core_with_indexer,
    _inputs,
)


def _metadata_core():
    core = dsa_gqa.DSGQACoreAttention.__new__(dsa_gqa.DSGQACoreAttention)
    torch.nn.Module.__init__(core)
    core.config = SimpleNamespace(attention_cp_layout="zigzag")
    return core


def _packed(boundaries, valid):
    # Q and KV describe the same physical, padded self-attention sequence.
    return PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries, real_token_mask_q=valid
    )


@pytest.mark.parametrize("comm_type", ["allgather", ["allgather"]])
def test_cute_cp_configuration_accepts_allgather(comm_type):
    assert _config(context_parallel_size=2, cp_comm_type=comm_type).context_parallel_size == 2


@pytest.mark.parametrize("comm_type", ["p2p", "a2a", None])
def test_cute_cp_configuration_rejects_other_collectives(comm_type):
    with pytest.raises((ValueError, AssertionError)):
        _config(context_parallel_size=2, cp_comm_type=comm_type)


@pytest.mark.parametrize("backend", ["reference", "torch-min-memory", "triton-min-memory"])
def test_cute_cp_does_not_enable_unsupported_backends(backend):
    with pytest.raises((ValueError, AssertionError)):
        _config(dsa_gqa_backend=backend, context_parallel_size=2, cp_comm_type="allgather")


def test_cute_cp_configuration_rejects_contiguous_layout():
    with pytest.raises((ValueError, AssertionError)):
        _config(context_parallel_size=2, cp_comm_type="allgather", attention_cp_layout="contiguous")


def test_unpadded_metadata_restores_global_order_and_is_reused_without_host_checks(monkeypatch):
    core = _metadata_core()
    arguments = dict(sq=8, cp_size=4, cp_rank=1, device=torch.device("cpu"), stream_id=7)

    def unexpected_host_read(*args, **kwargs):
        pytest.fail("Fixed-shape CP metadata must not read tensor values back to the host")

    with monkeypatch.context() as no_host_reads:
        no_host_reads.setattr(torch.Tensor, "item", unexpected_host_read)
        positions, reorder, starts, ends = core._get_unpadded_cp_metadata(**arguments)
        cached = core._get_unpadded_cp_metadata(**arguments)
    assert all(first is second for first, second in zip((positions, reorder, starts, ends), cached))
    # Independent rank-major layout: each rank owns one front and one mirrored back chunk.
    gathered_positions = torch.tensor(
        [
            0,
            1,
            2,
            3,
            28,
            29,
            30,
            31,
            4,
            5,
            6,
            7,
            24,
            25,
            26,
            27,
            8,
            9,
            10,
            11,
            20,
            21,
            22,
            23,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ]
    )
    torch.testing.assert_close(positions, torch.tensor([4, 5, 6, 7, 24, 25, 26, 27]))
    torch.testing.assert_close(gathered_positions[reorder], torch.arange(32))
    torch.testing.assert_close(starts, torch.zeros(8, dtype=torch.int64))
    torch.testing.assert_close(ends, positions + 1)


def test_unpadded_metadata_cache_is_stream_scoped_bounded_and_inference_safe():
    core = _metadata_core()
    arguments = dict(sq=8, cp_size=4, cp_rank=1, device=torch.device("cpu"), stream_id=7)
    with torch.inference_mode():
        inference_metadata = core._get_unpadded_cp_metadata(**arguments)
    assert all(not tensor.is_inference() for tensor in inference_metadata)
    training_metadata = core._get_unpadded_cp_metadata(**arguments)
    assert all(first is second for first, second in zip(inference_metadata, training_metadata))
    data = torch.randn(32, 2, requires_grad=True)
    data.index_select(0, training_metadata[1]).sum().backward()
    torch.testing.assert_close(data.grad, torch.ones_like(data))

    different_stream = core._get_unpadded_cp_metadata(**{**arguments, "stream_id": 8})
    assert different_stream[0] is not training_metadata[0]
    replacement = core._get_unpadded_cp_metadata(**arguments)
    # A single entry is replaced, rather than retaining tensors for every historical stream.
    assert replacement[0] is not training_metadata[0]
    different_shape = core._get_unpadded_cp_metadata(**{**arguments, "sq": 4})
    assert different_shape[0].numel() == 4


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1)])
def test_packed_layout_respects_document_boundaries_padding_and_local_order(cp_size, cp_rank):
    core = _metadata_core()
    boundaries = torch.tensor([0, 8, 16], dtype=torch.int32)
    global_valid = torch.tensor([True] * 7 + [False] + [True] * 6 + [False] * 2)
    positions = (
        torch.arange(16)
        if cp_size == 1
        else (
            torch.tensor([0, 1, 6, 7, 8, 9, 14, 15])
            if cp_rank == 0
            else torch.tensor([2, 3, 4, 5, 10, 11, 12, 13])
        )
    )
    local_valid = global_valid[positions]
    query = torch.empty(positions.numel(), 1, 16, 256)
    bounds, valid, reorder = core._get_cute_layout(
        query, _packed(boundaries, local_valid), cp_size, cp_rank
    )
    starts, ends = bounds
    torch.testing.assert_close(valid.flatten(), local_valid)
    torch.testing.assert_close(starts, torch.where(positions < 8, 0, 8))
    torch.testing.assert_close(ends, positions + 1)
    if cp_size > 1:
        gathered_positions = torch.tensor([0, 1, 6, 7, 8, 9, 14, 15, 2, 3, 4, 5, 10, 11, 12, 13])
        torch.testing.assert_close(gathered_positions[reorder], torch.arange(16))


@pytest.mark.parametrize("error", ["missing_valid_rows", "different_kv_boundaries"])
def test_packed_layout_requires_explicit_valid_rows_and_shared_qkv_boundaries(error):
    core = _metadata_core()
    boundaries = torch.tensor([0, 8, 16], dtype=torch.int32)
    packed = _packed(boundaries, torch.ones(8, dtype=torch.bool))
    if error == "missing_valid_rows":
        packed.real_token_mask_q = None
    else:
        packed.cu_seqlens_kv = boundaries.clone()
    with pytest.raises((ValueError, NotImplementedError)):
        core._get_cute_layout(torch.empty(8, 1, 16, 256), packed, 2, 0)


@pytest.mark.parametrize("boundaries", [[0, 8, 20], [4, 8, 16]])
def test_cpu_packed_boundaries_must_cover_exact_physical_extent(boundaries):
    core = _metadata_core()
    packed = _packed(torch.tensor(boundaries, dtype=torch.int32), torch.ones(8, dtype=torch.bool))
    with pytest.raises(ValueError):
        core._get_cute_layout(torch.empty(8, 1, 16, 256), packed, 2, 0)


def test_core_rejects_different_runtime_cp_group_even_when_its_size_matches(monkeypatch):
    core = _core_with_indexer(monkeypatch, 0.1)
    core.config.context_parallel_size = 2
    core.cp_comm_type = "allgather"
    configured_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    core.indexer.pg_collection.cp = configured_group
    packed = _packed(torch.tensor([0, 8, 16], dtype=torch.int32), torch.ones(8, dtype=torch.bool))
    packed.local_cp_size = 2
    packed.cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

    def unexpected_collective(*args, **kwargs):
        pytest.fail("A different runtime CP group must be rejected before any collective")

    monkeypatch.setattr(dsa_gqa, "gather_from_sequence_parallel_region", unexpected_collective)
    query, key, value, _, _ = _inputs(sequence_length=8, device="cpu")
    hidden = torch.zeros(8, 1, 16, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="configured|fixed|group|domain"):
        core._forward_cute(query, key, value, hidden, None, packed_seq_params=packed)


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("disabled_by", ["eval", "no_grad", "zero_coefficient"])
def test_disabled_indexer_loss_skips_valid_count_reduction(monkeypatch, cp_size, disabled_by):
    core = _core_with_indexer(monkeypatch, 0.0 if disabled_by == "zero_coefficient" else 0.1)
    core.train(disabled_by != "eval")
    core.config.context_parallel_size = cp_size
    core.config.calculate_per_token_loss = False
    core.cp_comm_type = "allgather"
    core.indexer.pg_collection.cp = SimpleNamespace(size=lambda: cp_size, rank=lambda: 0)
    packed = _packed(
        torch.tensor([0, 4 * cp_size, 8 * cp_size], dtype=torch.int32),
        torch.tensor([True] * 6 + [False] * 2),
    )
    monkeypatch.setattr(
        dsa_gqa,
        "gather_from_sequence_parallel_region",
        lambda value, *, group: torch.cat((value, value), dim=0),
    )

    def unexpected_count_collective(*args, **kwargs):
        pytest.fail("Disabled auxiliary loss must not reduce valid-row counts")

    monkeypatch.setattr(torch.distributed, "all_reduce", unexpected_count_collective)
    original_sum = torch.Tensor.sum

    def reject_valid_count_sum(tensor, *args, **kwargs):
        if tensor.dtype == torch.bool:
            pytest.fail("Disabled auxiliary loss must not compute valid-row counts")
        return original_sum(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", reject_valid_count_sum)
    seen = {}

    def attention(q, k, v, qi, ki, **kwargs):
        seen.update(kwargs)
        return q.flatten(2), q.new_zeros((), dtype=torch.float32)

    monkeypatch.setattr(dsa_cute_kernels, "run_cute_sparse_attention", attention)
    query, key, value, _, _ = _inputs(sequence_length=8, device="cpu")
    hidden = torch.zeros(8, 1, 16, dtype=torch.bfloat16)
    with torch.set_grad_enabled(disabled_by != "no_grad"):
        output = core._forward_cute(query, key, value, hidden, None, packed_seq_params=packed)
    assert output.shape == (8, 1, 16 * 256)
    assert seen["loss_coeff"] == 0.0
    assert seen["loss_denominator"] == 1
    assert seen["query_valid_rows"].shape == (8,)


@pytest.mark.parametrize("packed_input", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_core_gathers_only_k_v_indexer_k_and_keeps_local_query_order(
    monkeypatch, packed_input, per_token
):
    core = _core_with_indexer(monkeypatch, 0.1)
    core.config.context_parallel_size = 2
    core.config.cp_comm_type = "allgather"
    core.config.calculate_per_token_loss = per_token
    core.cp_comm_type = "allgather"
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    core.pg_collection = core.indexer.pg_collection = SimpleNamespace(cp=group, dp_cp=None)
    monkeypatch.setattr(
        dsa_gqa.DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kw: None
    )
    monkeypatch.setattr(dsa_gqa.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
    gathered = []

    def gather(value, *, group):
        gathered.append((value, group))
        return torch.cat((value, value + 1), dim=0)

    monkeypatch.setattr(dsa_gqa, "gather_from_sequence_parallel_region", gather)
    all_reduces = []

    def all_reduce(value, group=None, **kwargs):
        all_reduces.append(value.clone())
        value.add_(8)  # Peer contributes eight real rows.

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    query, key, value, _, _ = _inputs(sequence_length=8, device="cpu")
    hidden = torch.randn(8, 1, 16, dtype=torch.bfloat16, requires_grad=True)
    seen = {}

    def attention(q, k, v, qi, ki, **kwargs):
        seen.update(query=q, key=k, value=v, qi=qi, ki=ki, **kwargs)
        # Keep the indexer and backbone branches differentiable without duplicating kernel math.
        output = q.flatten(2) + k.sum() * 0 + v.sum() * 0
        loss = (qi.float().square().sum() + ki.float().square().sum()) * 0.001
        return output, loss

    monkeypatch.setattr(dsa_cute_kernels, "run_cute_sparse_attention", attention)
    packed = None
    if packed_input:
        boundaries = torch.tensor([0, 8, 16], dtype=torch.int32)
        packed = _packed(boundaries, torch.tensor([True] * 6 + [False] * 2))
        packed.local_cp_size = 2
        packed.cp_group = group
        query, key, value = (tensor.squeeze(1) for tensor in (query, key, value))
    output = core._forward_cute(query, key, value, hidden, None, packed_seq_params=packed)
    assert output.shape == (8, 1, 16 * 256)
    assert len(gathered) == 3
    assert [item[0].shape[-1] for item in gathered] == [256, 256, 128]
    assert all(item[1] is group and item[0].dtype == torch.bfloat16 for item in gathered)
    assert seen["query"].shape == (8, 1, 16, 256)
    assert seen["qi"].shape[0] == 8
    assert seen["key"].shape == seen["value"].shape == (16, 1, 1, 256)
    assert seen["ki"].shape == (16, 1, 1, 128)
    if per_token:
        assert seen["loss_denominator"] == 1
    elif not packed_input:
        assert seen["loss_denominator"] == 16
    else:
        # The loss is a contribution to the globally normalized objective. The training
        # schedule already compensates for CP gradient averaging in its backward scale.
        torch.testing.assert_close(torch.as_tensor(seen["loss_denominator"]), torch.tensor(14.0))
        assert len(all_reduces) == 1
    output.float().sum().backward()
    assert hidden.grad is None
    assert all(parameter.grad is not None for parameter in core.indexer.parameters())


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("checkpoint_core", [False, True])
def test_transformer_layer_accepts_thd_and_preserves_indexer_backpropagation(
    monkeypatch, cp_size, checkpoint_core
):
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.experimental_attention_variant.dsa_layer_specs import (
        dsa_stack_spec,
    )
    from megatron.core.transformer.spec_utils import build_module
    from tests.unit_tests.test_utilities import Utils

    if not torch.cuda.is_available():
        pytest.skip("Transformer Engine requires CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) < cp_size:
        pytest.skip("Not enough distributed processes for the requested CP group")
    Utils.initialize_model_parallel(1, 1, context_parallel_size=cp_size)
    try:
        model_parallel_cuda_manual_seed(4711)
        config = _config(
            hidden_size=256,
            ffn_hidden_size=512,
            hidden_dropout=0.0,
            context_parallel_size=cp_size,
            cp_comm_type="allgather",
            recompute_granularity="selective" if checkpoint_core else None,
            recompute_modules=["core_attn"] if checkpoint_core else [],
        )
        layer = build_module(dsa_stack_spec.submodules.attention_layer, config=config).cuda()
        monkeypatch.setattr(
            dsa_gqa.DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kw: None
        )
        monkeypatch.setattr(dsa_gqa.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
        calls = []

        def attention(q, k, v, qi, ki, **kwargs):
            calls.append((q.shape, k.shape, kwargs["row_bounds"], kwargs["query_valid_rows"]))
            output = q + k.mean(0, keepdim=True) + v.mean(0, keepdim=True)
            output = output * kwargs["query_valid_rows"][:, None, None, None]
            loss = qi.float().square().sum() + ki.float().square().sum()
            return output.flatten(2), loss * kwargs["loss_coeff"] / kwargs["loss_denominator"]

        monkeypatch.setattr(dsa_cute_kernels, "run_cute_sparse_attention", attention)
        boundaries = torch.tensor([0, 16 * cp_size, 32 * cp_size], device="cuda", dtype=torch.int32)
        packed = _packed(boundaries, torch.ones(32, device="cuda", dtype=torch.bool))
        hidden = torch.randn(32, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        output, _ = layer(hidden, attention_mask=None, packed_seq_params=packed)
        assert output.shape == hidden.shape
        output.float().square().mean().backward()
        assert calls and all(call[0][0] == 32 and call[1][0] == 32 * cp_size for call in calls)
        assert all(call[2][0].shape == call[3].shape == (32,) for call in calls)
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        for name, parameter in layer.named_parameters():
            if "linear_qkv.weight" in name or ".indexer." in name:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                assert torch.count_nonzero(parameter.grad) > 0, name
    finally:
        Utils.destroy_model_parallel()


def _document_oracle(inputs, doc_starts, valid, coefficient, denominator):
    query, key, value, qi, ki = (tensor.float() for tensor in inputs)
    query, key, value = query[:, 0], key[:, 0, 0], value[:, 0, 0]
    positions = torch.arange(query.size(0), device=query.device)
    allowed = (positions[None, :] >= doc_starts[:, None]) & (
        positions[None, :] <= positions[:, None]
    )
    scores = torch.einsum("qhd,kd->hqk", query, key) * 256**-0.5
    probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    output = torch.einsum("hqk,kd->qhd", probabilities, value) * valid[:, None, None]
    target = probabilities.detach().mean(0)
    index_scores = qi[:, 0, 0] @ ki[:, 0, 0].T * 128**-0.5
    log_prediction = index_scores.masked_fill(~allowed, -torch.inf).log_softmax(-1)
    terms = target * (target.clamp_min(1e-10).log() - log_prediction)
    loss = terms.masked_fill(~allowed | ~valid[:, None], 0).sum() * coefficient / denominator
    return output.reshape(query.size(0), 1, -1), loss


@pytest.mark.parametrize("packed_input", [False, True])
def test_cp2_real_collectives_and_cute_gradients_match_global_reference(monkeypatch, packed_input):
    """Full remote-K gradient paths must reduce back to the owning CP shard."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("CuTe simplified sparse attention requires SM103")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("CP2 parity requires at least two distributed processes")
    pytest.importorskip("simplified_sparse_attention")
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import mappings
    from tests.unit_tests.test_utilities import Utils

    # Observe real collective payloads while retaining NCCL execution and autograd.
    gather_dtypes, reduction_dtypes = [], []
    original_gather = mappings.dist_all_gather_func
    original_reduce_scatter = mappings.dist_reduce_scatter_func

    def record_gather(output, value, **kwargs):
        gather_dtypes.append((output.dtype, value.dtype))
        return original_gather(output, value, **kwargs)

    def record_reduce_scatter(output, value, **kwargs):
        reduction_dtypes.append((output.dtype, value.dtype))
        return original_reduce_scatter(output, value, **kwargs)

    monkeypatch.setattr(mappings, "dist_all_gather_func", record_gather)
    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", record_reduce_scatter)

    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        group = parallel_state.get_context_parallel_group()
        rank = torch.distributed.get_rank(group)
        torch.manual_seed(1729)
        full_inputs = _inputs(sequence_length=128)
        reference_inputs = tuple(t.detach().float().requires_grad_() for t in full_inputs)
        positions = torch.arange(128, device="cuda")
        if packed_input:
            # Two physical 64-row documents, with 58 and 61 real tokens.
            local_positions = torch.cat(
                [
                    torch.arange(start + chunk * 16, start + (chunk + 1) * 16, device="cuda")
                    for start in (0, 64)
                    for chunk in (rank, 3 - rank)
                ]
            )
            doc_starts = torch.where(positions < 64, 0, 64)
            valid = (positions < 58) | ((positions >= 64) & (positions < 125))
            boundaries = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
            packed = _packed(boundaries, valid[local_positions])
            denominator = 119
        else:
            local_positions = torch.cat(
                (
                    torch.arange(rank * 32, (rank + 1) * 32, device="cuda"),
                    torch.arange((3 - rank) * 32, (4 - rank) * 32, device="cuda"),
                )
            )
            doc_starts = torch.zeros(128, device="cuda", dtype=torch.int64)
            valid = torch.ones(128, device="cuda", dtype=torch.bool)
            packed = None
            denominator = 128
        inputs = tuple(t.detach()[local_positions].requires_grad_() for t in full_inputs)
        bounds, local_valid, reorder = _metadata_core()._get_cute_layout(inputs[0], packed, 2, rank)
        gathered = {
            index: dsa_gqa.gather_from_sequence_parallel_region(inputs[index], group=group)[reorder]
            for index in (1, 2, 4)
        }
        output, loss = dsa_cute_kernels.run_cute_sparse_attention(
            inputs[0],
            gathered[1],
            gathered[2],
            inputs[3],
            gathered[4],
            topk=512,
            softmax_scale=256**-0.5,
            loss_coeff=0.1,
            loss_denominator=denominator,
            row_bounds=bounds,
            query_valid_rows=local_valid,
        )
        reference_output, reference_loss = _document_oracle(
            reference_inputs, doc_starts, valid, 0.1, denominator
        )
        upstream = torch.randn_like(reference_output).to(torch.bfloat16)
        (output.float() * upstream[local_positions]).sum().add(loss).backward()
        assert gather_dtypes == [(torch.bfloat16, torch.bfloat16)] * 3
        assert reduction_dtypes == [(torch.bfloat16, torch.bfloat16)] * 3
        (reference_output * upstream).sum().add(reference_loss).backward()
        torch.testing.assert_close(
            output.float(), reference_output[local_positions], rtol=2e-2, atol=2e-2
        )
        total_loss = loss.detach().clone()
        torch.distributed.all_reduce(total_loss, group=group)
        torch.testing.assert_close(total_loss, reference_loss, rtol=2e-2, atol=2e-4)
        for name, actual, reference in zip(
            ("Q", "K", "V", "indexer Q", "indexer K"), inputs, reference_inputs
        ):
            expected = reference.grad[local_positions]
            actual_grad = actual.grad.float()
            torch.testing.assert_close(actual_grad, expected, rtol=4e-2, atol=4e-2)
            relative_error = torch.linalg.vector_norm(
                actual_grad - expected
            ) / torch.linalg.vector_norm(expected).clamp_min(1e-12)
            assert (
                relative_error.item() < 0.04
            ), f"{name}: gradient RMS error {relative_error.item():.4f}"
    finally:
        Utils.destroy_model_parallel()

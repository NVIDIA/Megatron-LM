# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact BI DSA forward/replay contracts and independent fixed-support gradient checks."""

import pytest
import torch

from megatron.core.inference.contexts.attention_context.dsa_metadata import DSARowPlan
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant.dsa_gqa_inference import (
    _materialized_topk,
    _sparse_attention_rows,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    _triton_dispatch_enabled,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200

pytest.importorskip("deep_select")

from megatron.core.transformer.experimental_attention_variant.dsa_batch_invariant import (  # noqa: E402
    attention_contiguous,
    contiguous_plan,
    route_contiguous,
    score_width,
    select,
)


@pytest.fixture(scope="module", autouse=True)
def initialize_distributed():
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    with set_batch_invariant_mode(True, backend="te_native"):
        yield
    Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def require_blackwell_and_deepselect():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Batch-invariant DSA currently targets GB200")
    pytest.importorskip("deep_select")


@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_deepselect_with_same_scores(topk):
    """Make sure the results is invariant if there are multiple identical scores."""
    width = score_width(8192, topk)
    length = topk + 73
    for ties in (True, False):
        scores = torch.zeros(1, width, device="cuda", dtype=torch.float32)
        if not ties:
            scores[:, : topk - 11] = 1
        lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
        expected = select(scores, lengths, topk)
        assert torch.all(expected[:, 1:] > expected[:, :-1])
        for batch in (1, 8, 128, 256):
            pitched = torch.empty(batch, width + 256, device="cuda")[:, :width]
            pitched[:] = scores
            torch.testing.assert_close(
                select(pitched, lengths.expand(batch).contiguous(), topk),
                expected.expand(batch, -1),
                atol=0,
                rtol=0,
            )
        wider = torch.nn.functional.pad(scores, (0, width), value=-float("inf"))
        assert torch.equal(select(wider, lengths, topk), expected)
    lengths = torch.tensor([0, 19, topk - 1], device="cuda", dtype=torch.int32)
    selected = select(scores.expand(3, -1).contiguous(), lengths, topk)
    for row, length in enumerate((0, 19, topk - 1)):
        assert torch.equal(selected[row, :length], torch.arange(length, device="cuda"))
        assert torch.all(selected[row, length:] == -1)


PAGE_SIZE = 128


def _reversed_page_table(batch, length):
    """Non-contiguous physical pages in reverse request order, including a partial final page."""
    pages = -(-length // PAGE_SIZE)
    return torch.arange(batch * pages, device="cuda", dtype=torch.int32).flip(0).view(batch, pages)


def _paged(tensor, table):
    """Scatter SBHD sequences into a ``[pages, PAGE_SIZE, ...]`` cache through ``table``."""
    length, batch = tensor.shape[:2]
    padded = torch.nn.functional.pad(
        tensor.permute(1, 0, 2, 3), (0, 0, 0, 0, 0, table.size(1) * PAGE_SIZE - length)
    )
    cache = tensor.new_empty(table.numel(), PAGE_SIZE, *tensor.shape[2:])
    cache[table.long().flatten()] = padded.reshape_as(cache)
    return cache


def test_indexer_scores_match_across_layouts_schedules_and_decode_rows():
    """Every score row is independent of KV layout, capacity, worker grid and row dispatch."""
    from megatron.core.transformer.experimental_attention_variant.dsa_gqa_inference_kernels import (
        dsa_gqa_indexer_scores,
    )

    torch.manual_seed(43)
    length, batch, index_dim = 2305, 2, 128
    width = score_width(4096, 512)
    iq = torch.randn(length, batch, 1, index_dim, device="cuda", dtype=torch.bfloat16)
    ik = torch.randn_like(iq)
    table = _reversed_page_table(batch, length)
    paged_keys = _paged(ik, table)[:, :, 0]
    rows = iq.permute(1, 0, 2, 3).reshape(-1, index_dim).contiguous()
    plan = contiguous_plan(batch, length, 0, length, iq.device)

    def scores(cache, block_table, capacity, workers=None, decode_rows=False):
        out = torch.empty(batch * length, capacity, device="cuda", dtype=torch.float32)
        dsa_gqa_indexer_scores(
            rows,
            cache,
            block_table,
            plan.row_lengths,
            plan.block_starts,
            plan.block_rows,
            plan.block_requests,
            out=out,
            fill_invalid=True,
            batch_invariant=True,
            row_requests=plan.row_requests if decode_rows else None,
            workers=workers,
        )
        return out[:, :width]

    # Training scores row blocks of one contiguous sequence; decode-only steps score each
    # row by its request. Both must reproduce the same bits for every causal prefix.
    reference = scores(ik[:, :, 0].permute(1, 0, 2), plan.block_table, width)
    for options in (
        dict(capacity=width),
        dict(capacity=16384),
        dict(capacity=16384, workers=16),
        dict(capacity=16384, workers=128),
        dict(capacity=width, decode_rows=True),
        dict(capacity=16384, workers=64, decode_rows=True),
    ):
        assert torch.equal(scores(paged_keys, table, **options), reference), options


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_contiguous_paged_chunked_decode_and_graph_are_exact(dim, topk, record_property):
    torch.manual_seed(41)
    length, batch, heads, index_dim = 2305, 2, 16, 128
    width = score_width(4096, topk)
    query = torch.randn(length, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(length, batch, 1, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    iq = torch.randn(length, batch, 1, index_dim, device="cuda", dtype=torch.bfloat16)
    ik = torch.randn_like(iq)
    scale = dim**-0.5
    with _triton_dispatch_enabled(True, width):
        _, indices = route_contiguous(iq, ik, 0, length, topk, width, index_dim**-0.5)
    expected = attention_contiguous(query, key, value, indices, scale, 0)

    table = _reversed_page_table(batch, length)
    kc, vc = _paged(key, table), _paged(value, table)
    ikc = _paged(ik, table)[:, :, 0]
    readout = torch.randn(heads * dim, 31, device="cuda", dtype=torch.float32)
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        _log_softmax_batch_invariant,
        mm_batch_invariant,
    )

    def logprobs(output):
        # Each token uses the same fixed readout; avoid making cuBLAS part of the DSA contract.
        logits = mm_batch_invariant(output.reshape(-1, heads * dim).float(), readout)
        return _log_softmax_batch_invariant(logits, -1, False)

    expected_logprobs = logprobs(expected.permute(1, 0, 2, 3).contiguous()).reshape(
        batch, length, -1
    )

    for start, stop in ((0, length), (0, 19), (19, 511), (511, length), (length - 1, length)):
        plan = contiguous_plan(batch, length, start, stop, query.device)._replace(block_table=table)
        rows = query[start:stop].permute(1, 0, 2, 3).reshape(-1, heads, dim).contiguous()
        irows = iq[start:stop].permute(1, 0, 2, 3).reshape(-1, index_dim).contiguous()

        def run():
            return _sparse_attention_rows(
                query=rows,
                key_cache=kc,
                value_cache=vc,
                index_query=irows,
                index_key_cache=ikc,
                plan=plan,
                max_context_length=width,
                index_topk=topk,
                softmax_scale=scale,
                batch_invariant=True,
            )

        selected = _materialized_topk(
            index_query=irows,
            index_key_cache=ikc,
            plan=plan,
            max_context_length=width,
            index_topk=topk,
            batch_invariant=True,
        )
        assert torch.equal(selected.reshape(batch, stop - start, topk), indices[:, start:stop])
        reference = expected[start:stop].permute(1, 0, 2, 3).reshape_as(rows)
        output = run()
        assert torch.equal(output, reference)
        assert torch.equal(logprobs(output), expected_logprobs[:, start:stop].reshape(-1, 31))
        graph = torch.cuda.CUDAGraph()
        run()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graphed = run()
        for _ in range(2):
            graph.replay()
            assert torch.equal(graphed, reference)
    record_property("attention_max_abs_diff", 0.0)
    record_property("fp32_logprob_max_abs_diff", 0.0)
    record_property("indexer_score_max_abs_diff", 0.0)

    # Reorder and interleave independent requests with a graph-padding row.
    positions = torch.tensor([2, length - 1, 511, length - 1, 0], device="cuda")
    requests = torch.tensor([1, 0, 1, 1, 0], device="cuda", dtype=torch.int32)
    valid = torch.tensor([True, True, True, True, False], device="cuda")
    mixed_query = query[positions, requests.long()].contiguous()
    mixed_index = iq[positions, requests.long(), 0].contiguous()
    mixed_plan = contiguous_plan(1, 5, 0, 5, query.device)._replace(
        row_requests=requests.masked_fill(~valid, -1),
        row_lengths=(positions.to(torch.int32) + 1).masked_fill(~valid, 0),
        block_starts=torch.arange(5, device="cuda", dtype=torch.int32),
        block_rows=valid.to(torch.int32),
        block_requests=requests,
        block_table=table,
    )
    mixed_reference = expected[positions, requests.long()].masked_fill(~valid[:, None, None], 0)
    # Decode-only steps score through the per-row dispatch instead of row blocks.
    for decode_only in (False, True):
        mixed = _sparse_attention_rows(
            query=mixed_query,
            key_cache=kc,
            value_cache=vc,
            index_query=mixed_index,
            index_key_cache=ikc,
            plan=mixed_plan._replace(decode_only=decode_only),
            max_context_length=width,
            index_topk=topk,
            softmax_scale=scale,
            batch_invariant=True,
        )
        assert torch.equal(mixed, mixed_reference)
    swapped = _sparse_attention_rows(
        query=mixed_query,
        key_cache=kc,
        value_cache=vc,
        index_query=mixed_index,
        index_key_cache=ikc,
        plan=mixed_plan._replace(
            row_requests=(1 - requests).masked_fill(~valid, -1),
            block_requests=1 - requests,
            block_table=table.flip(0).contiguous(),
        ),
        max_context_length=width,
        index_topk=topk,
        softmax_scale=scale,
        batch_invariant=True,
    )
    assert torch.equal(swapped, mixed_reference)


@pytest.mark.parametrize(
    "length,cache_routing,dim,topk",
    [(73, True, 128, 512), (769, False, 128, 512), (2113, False, 256, 2048)],
)
def test_training_backward_and_no_grad_recompute(
    length, cache_routing, dim, topk, record_property, monkeypatch
):
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
        SimplifiedDSGQAIndexer,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa_layer_specs import (
        dsa_stack_spec,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
        _linear,
        dsa_min_memory_gqa,
        dsa_min_memory_gqa_forward_only,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(72)
    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    batch, heads, hidden = 1, 16, 256
    max_seq_length = 4096 if length > 1024 else 1024
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_query_groups=1,
        kv_channels=dim,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0,
        batch_invariant_mode=True,
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_mode="simplified",
        add_bias_linear=False,
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=topk,
        dsa_kernel_backend="min-memory-triton",
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=dsa_stack_spec,
        vocab_size=128,
        max_sequence_length=max_seq_length,
        hybrid_layer_pattern="*",
    ).cuda()
    indexer = next(
        module for module in model.modules() if isinstance(module, SimplifiedDSGQAIndexer)
    )
    query = torch.randn(
        length, batch, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    key = torch.randn(
        length, batch, 1, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    value = torch.randn_like(key, requires_grad=True)
    states = torch.randn(length, batch, hidden, device="cuda", dtype=torch.bfloat16)
    args = (query, key, value, states, indexer, dim**-0.5)
    cache_kwargs = dict(
        cache_routing=cache_routing,
        cache_indexer_k=cache_routing,
        cache_selected_scores=cache_routing,
    )
    output, loss = dsa_min_memory_gqa(*args, loss_coeff=0.1, use_indexer_rope=False, **cache_kwargs)
    recomputed = dsa_min_memory_gqa_forward_only(*args, use_indexer_rope=False, query_chunk_size=19)
    assert torch.equal(output, recomputed)
    grad = torch.randn_like(output)
    (output.float().mul(grad.float()).sum() + loss).backward()
    output, loss = output.detach(), loss.detach()
    for linear in (indexer.linear_q, indexer.linear_k):
        assert linear.weight.grad is not None
        assert torch.isfinite(linear.weight.grad).all()
        assert torch.count_nonzero(linear.weight.grad) > 0
    with _triton_dispatch_enabled(True, score_width(max_seq_length, topk)), torch.no_grad():
        iq = _linear(states, indexer.linear_q.weight).unsqueeze(2)
        ik = _linear(states, indexer.linear_k.weight).unsqueeze(2)
        _, indices = route_contiguous(
            iq, ik, 0, length, topk, score_width(max_seq_length, topk), 128**-0.5
        )
    support = torch.zeros(length, length + 1, device="cuda", dtype=torch.bool)
    support.scatter_(1, indices[0].long().masked_fill(indices[0] < 0, length), True)
    support = support[:, :length]
    qref, kref, vref = [tensor.detach().double().requires_grad_() for tensor in (query, key, value)]
    logits = torch.einsum("sbhd,tbhd->bhst", qref, kref.expand(-1, -1, heads, -1)) * dim**-0.5
    probs = logits.masked_fill(~support, -float("inf")).softmax(-1)
    ref = torch.einsum("bhst,tbhd->sbhd", probs, vref.expand(-1, -1, heads, -1))
    record_property(
        "attention_vs_fp64_max_abs_diff",
        (output.reshape_as(ref).float() - ref.float()).abs().max().item(),
    )
    torch.testing.assert_close(output.reshape_as(ref).float(), ref.float(), rtol=2e-2, atol=2e-2)
    (ref * grad.reshape_as(ref).double()).sum().backward()
    for name, actual, reference in zip(("dQ", "dK", "dV"), (query, key, value), (qref, kref, vref)):
        record_property(
            f"{name}_max_abs_diff",
            (actual.grad.float() - reference.grad.float()).abs().max().item(),
        )
        torch.testing.assert_close(
            actual.grad.float(), reference.grad.float(), rtol=2e-2, atol=2e-2
        )

    weights = [
        linear.weight.detach().double().requires_grad_()
        for linear in (indexer.linear_q, indexer.linear_k)
    ]
    projected = [torch.nn.functional.linear(states.double(), weight) for weight in weights]
    qi, ki = [
        precise + (rounded[:, :, 0].double() - precise).detach()
        for precise, rounded in zip(projected, (iq, ik))
    ]
    student = (
        (torch.einsum("sbd,tbd->bst", qi, ki) * 128**-0.5)
        .masked_fill(~support, -float("inf"))
        .softmax(-1)
    )
    teacher = probs.detach().sum(1) / heads
    refloss = (teacher * ((teacher + 1e-10).log() - (student + 1e-10).log())).sum() * (0.1 / length)
    refloss.backward()
    for name, actual, reference in zip(
        ("indexer_dWq", "indexer_dWk"), (indexer.linear_q.weight, indexer.linear_k.weight), weights
    ):
        record_property(
            f"{name}_max_abs_diff",
            (actual.grad.float() - reference.grad.float()).abs().max().item(),
        )
        torch.testing.assert_close(
            actual.grad.float(), reference.grad.float(), rtol=2e-2, atol=2e-5
        )

    parameters = (query, key, value, indexer.linear_q.weight, indexer.linear_k.weight)
    for parameter in parameters:
        parameter.grad = None
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_loss = dsa_min_memory_gqa(
            *args, loss_coeff=0.1, use_indexer_rope=False, **cache_kwargs
        )
        (graph_output.float().mul(grad.float()).sum() + graph_loss).backward()
    graph.replay()
    assert torch.equal(graph_output, output)
    record_property("training_graph_forward_max_abs_diff", 0.0)
    for name, actual, reference in zip(
        ("dQ", "dK", "dV", "indexer_dWq", "indexer_dWk"), parameters, (qref, kref, vref, *weights)
    ):
        record_property(
            f"graph_{name}_max_abs_diff",
            (actual.grad.float() - reference.grad.float()).abs().max().item(),
        )
        torch.testing.assert_close(
            actual.grad.float(),
            reference.grad.float(),
            rtol=2e-2,
            atol=2e-5 if name.startswith("indexer") else 2e-2,
        )

    with torch.no_grad():
        for linear in (indexer.linear_q, indexer.linear_k):
            previous = linear.weight.clone()
            linear.weight.add_(linear.weight.grad, alpha=-0.1)
            assert not torch.equal(linear.weight, previous)
    updated, _ = dsa_min_memory_gqa(*args, loss_coeff=0.1, use_indexer_rope=False)
    assert torch.equal(
        updated, dsa_min_memory_gqa_forward_only(*args, use_indexer_rope=False, query_chunk_size=31)
    )
    with _triton_dispatch_enabled(True, score_width(max_seq_length, topk)), torch.no_grad():
        iq = _linear(states, indexer.linear_q.weight).reshape(length, 128)
        ik = _linear(states, indexer.linear_k.weight).reshape(1, length, 128)
        for start in (0, length - 1):
            plan = contiguous_plan(batch, length, start, length, query.device)
            inferred = _sparse_attention_rows(
                query=query[start:, 0],
                key_cache=key.permute(1, 0, 2, 3),
                value_cache=value.permute(1, 0, 2, 3),
                index_query=iq[start:],
                index_key_cache=ik,
                plan=plan,
                max_context_length=score_width(max_seq_length, topk),
                index_topk=topk,
                softmax_scale=dim**-0.5,
                batch_invariant=True,
            )
            assert torch.equal(inferred.reshape(length - start, 1, -1), updated[start:])
    record_property("training_recompute_and_updated_inference_max_abs_diff", 0.0)


@pytest.mark.parametrize("interleaved", [False, True])
@torch.no_grad()
def test_indexer_projection_norm_and_rope_match_training(interleaved, record_property):
    from types import SimpleNamespace

    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
        SimplifiedDSGQAIndexer,
        SimplifiedDSGQAIndexerSubmodules,
        _DSAIndexerInputNormSpec,
        _normalized_indexer_input,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
        _project_simplified_k_index_block,
        _project_simplified_q_index_tile,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(37)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=1,
        kv_channels=128,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0,
        add_bias_linear=False,
        batch_invariant_mode=True,
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        dsa_indexer_loss_coeff=0.1,
        rotary_interleaved=interleaved,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
    )
    indexer = SimplifiedDSGQAIndexer(
        config, SimplifiedDSGQAIndexerSubmodules(linear_q=TELinear, linear_k=TELinear)
    ).eval()
    states = torch.randn(1025, 2, 1024, device="cuda", dtype=torch.bfloat16)
    norm = _DSAIndexerInputNormSpec(
        "RMSNorm", torch.randn(1024, device="cuda", dtype=torch.bfloat16), None, 1e-6, False
    )
    normalized = _normalized_indexer_input(states, norm)
    with _triton_dispatch_enabled(True, score_width(4096, 512)):
        args = (states, 0, 1025)
        rest = (128, 128, indexer.rotary_pos_emb, interleaved, True, norm)
        train_q = _project_simplified_q_index_tile(*args, indexer.linear_q.weight, *rest)
        train_k = _project_simplified_k_index_block(*args, indexer.linear_k.weight, *rest)
    positions = torch.tensor([7, 511, 1024], device="cuda")
    # The real indexer consumes these dynamic GPU metadata fields, including padding position.
    context = SimpleNamespace(
        gpu_view=SimpleNamespace(
            token_to_block_idx=torch.zeros(3, device="cuda", dtype=torch.long),
            token_to_pos_ids=positions,
            token_to_position_in_request=positions,
        ),
        kv_block_allocator=SimpleNamespace(dummy_block_idx=-1),
    )
    dynamic_q, dynamic_k = indexer.forward_qk_dynamic(normalized[positions, :1], True, context)
    assert torch.equal(dynamic_q, train_q[positions, :1])
    assert torch.equal(dynamic_k, train_k[positions, :1])
    record_property("indexer_qk_max_abs_diff", 0.0)


def test_tied_routing_forward_replays(monkeypatch, record_property):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Batch-invariant DSA currently targets SM100")
    pytest.importorskip("deep_select")
    torch.manual_seed(39)
    query = torch.randn(3, 16, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(8, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    index_query = torch.zeros(3, 128, device="cuda", dtype=torch.bfloat16)
    index_key = torch.ones(8, 128, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.tensor([[7, 2, 4, 0, 5, 1, 3, 6]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1024, 769, 0], device="cuda", dtype=torch.int32)
    requests = torch.tensor([0, 0, -1], device="cuda", dtype=torch.int32)
    direct_rows = False

    def forward(query, key, value, index_query, index_key, table, lengths, requests):
        plan = DSARowPlan(
            requests,
            lengths,
            torch.arange(3, device="cuda", dtype=torch.int32),
            (lengths > 0).to(torch.int32),
            torch.zeros_like(requests),
            lengths[:0],
            table,
            direct_rows,
        )
        return _sparse_attention_rows(
            query=query,
            key_cache=key,
            value_cache=value,
            index_query=index_query,
            index_key_cache=index_key,
            plan=plan,
            max_context_length=1024,
            index_topk=512,
            softmax_scale=128**-0.5,
            batch_invariant=True,
        )

    inputs = (query, key, value, index_query, index_key, table, lengths, requests)
    expected = forward(*inputs)
    direct_rows = True
    assert torch.equal(forward(*inputs), expected)
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa_inference

    monkeypatch.setattr(dsa_gqa_inference, "SCORE_BUFFER_BYTES", 1024 * torch.float32.itemsize)
    outputs, _ = assert_replays_bit_exact(
        forward,
        inputs,
        backward=False,
        replays=3,
        contention=True,
        what="batch-invariant simplified DSA",
    )
    assert torch.isfinite(outputs["out"]).all()
    assert torch.count_nonzero(outputs["out"][2]) == 0
    assert torch.equal(outputs["out"], expected)
    record_property("score_window_max_abs_diff", 0.0)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The embedding weight gradient accumulated into main_grad gives main_grad the same bits as
PyTorch's embedding backward followed by ``main_grad.add_`` (what Megatron's DDP hook does)."""

import dataclasses

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.fusions.fused_embedding_wgrad import (
    HAVE_TRITON,
    MAX_TOKENS,
    MIN_TOKENS,
    accumulate_embedding_wgrad,
    can_accumulate_embedding_wgrad,
)
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs a GPU and Triton"
)

VOCAB, HIDDEN = 32768, 1024


def _token_ids(kind, num_tokens, vocab=VOCAB):
    if kind == "uniform":
        return torch.randint(0, vocab, (num_tokens,), device="cuda")
    if kind == "zipf":
        ranks = torch.rand(num_tokens, device="cuda").clamp_min(1e-6).pow(-1.1).floor() - 1
        return ranks.clamp(0, vocab - 1).long()
    if kind == "few":
        return torch.randint(0, 37, (num_tokens,), device="cuda") * (vocab // 37)
    if kind == "runs":
        # Token counts around the 10-row partial sums, the rest unique.
        counts = [1, 9, 10, 11, 19, 20, 21, 30, 31]
        ids = [1000 + t for t, c in enumerate(counts) for _ in range(c)]
        ids += range(5000, 5000 + num_tokens - len(ids))
        ids = torch.tensor(ids, device="cuda")
        return ids[torch.randperm(num_tokens, device="cuda")]
    if kind == "same":
        return torch.full((num_tokens,), 5, device="cuda")
    if kind == "edges":
        ids = torch.randint(0, vocab, (num_tokens,), device="cuda")
        ids[:100] = 0
        ids[100:200] = vocab - 1
        return ids
    raise ValueError(kind)


def _assert_same_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    mismatch = actual.view(torch.uint8) != expected.view(torch.uint8)
    assert not bool(mismatch.any()), f"{int(mismatch.sum())} bytes differ"


@pytest.mark.parametrize("kind", ["uniform", "zipf", "few", "runs", "same", "edges"])
@pytest.mark.parametrize("num_tokens", [MIN_TOKENS, 4096, MAX_TOKENS])
@pytest.mark.parametrize("main_grad_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float32])
def test_matches_dense_backward_and_add(kind, num_tokens, main_grad_dtype, grad_dtype):
    torch.manual_seed(0)
    ids = _token_ids(kind, num_tokens).view(1, num_tokens)
    main_grad = (torch.randn(VOCAB, HIDDEN, device="cuda") * 0.01).to(main_grad_dtype)
    expected = main_grad.clone()
    for _ in range(2):  # two microbatches accumulate
        grad = torch.randn(1, num_tokens, HIDDEN, device="cuda", dtype=grad_dtype)
        assert can_accumulate_embedding_wgrad(ids, grad, main_grad)
        expected.add_(torch.ops.aten.embedding_dense_backward(grad, ids, VOCAB, -1, False))
        accumulate_embedding_wgrad(ids, grad, main_grad)
        _assert_same_bits(main_grad, expected)


def test_transposed_gradient():
    """VocabParallelEmbedding with reduce_scatter_embeddings receives a [b, s, h] view of an
    [s, b, h] gradient."""
    torch.manual_seed(0)
    ids = _token_ids("zipf", 4096).view(2, 2048)
    grad = torch.randn(2048, 2, HIDDEN, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    main_grad = torch.zeros(VOCAB, HIDDEN, device="cuda")
    expected = torch.ops.aten.embedding_dense_backward(grad, ids, VOCAB, -1, False).float()
    accumulate_embedding_wgrad(ids, grad, main_grad)
    _assert_same_bits(main_grad, expected)


@pytest.mark.parametrize("layout", ["sliced", "column", "expanded", "broadcast"])
def test_strided_token_ids(layout):
    """Token id views whose flattened form is not contiguous, or not a view at all."""
    torch.manual_seed(0)
    if layout == "sliced":
        ids = torch.arange(8192, device="cuda").view(1, 8192)[:, ::2]
    elif layout == "column":
        ids = _token_ids("zipf", 8192).view(4096, 2)[:, 1].view(1, 4096)
    elif layout == "expanded":
        ids = _token_ids("zipf", 2048).view(1, 2048).expand(2, 2048)
    else:
        ids = torch.full((1, 1), 5, device="cuda").expand(1, 4096)
    grad = torch.randn(*ids.shape, HIDDEN, device="cuda", dtype=torch.bfloat16)
    main_grad = torch.zeros(VOCAB, HIDDEN, device="cuda")
    assert can_accumulate_embedding_wgrad(ids, grad, main_grad)
    expected = torch.ops.aten.embedding_dense_backward(grad, ids, VOCAB, -1, False).float()
    accumulate_embedding_wgrad(ids, grad, main_grad)
    _assert_same_bits(main_grad, expected)


def test_token_count_window():
    main_grad = torch.zeros(VOCAB, HIDDEN, device="cuda")
    for num_tokens, supported in [
        (MIN_TOKENS - 1, False),
        (MIN_TOKENS, True),
        (MAX_TOKENS, True),
        (MAX_TOKENS + 1, False),
    ]:
        ids = torch.zeros(num_tokens, dtype=torch.long, device="cuda")
        grad = torch.zeros(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        assert can_accumulate_embedding_wgrad(ids, grad, main_grad) == supported


def _with_ddp_hook(param):
    """Give ``param`` a main_grad and the post-accumulation hook of Megatron's DDP."""
    param.main_grad = torch.zeros(param.shape, dtype=torch.float32, device=param.device)
    param.grad_added_to_main_grad = False

    def hook(*unused):
        if param.grad is not None and not param.grad_added_to_main_grad:
            param.main_grad.add_(param.grad.data)
        param.grad = None

    grad_acc = param.expand_as(param).grad_fn.next_functions[0][0]
    grad_acc.register_hook(hook)
    return grad_acc


class TestVocabParallelEmbeddingWgradFusion:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "tp,reduce_scatter_embeddings,deterministic_mode",
        [(1, False, False), (2, False, False), (2, True, False), (1, False, True)],
    )
    def test_main_grad_matches_unfused(self, tp, reduce_scatter_embeddings, deterministic_mode):
        """Three microbatches: inside the fused token window, below it and above it."""
        if Utils.world_size % tp:
            pytest.skip(f"world size {Utils.world_size} is not a multiple of TP {tp}")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=8,
            params_dtype=torch.bfloat16,
            bf16=True,
            tensor_model_parallel_size=tp,
            sequence_parallel=reduce_scatter_embeddings,
            deterministic_mode=deterministic_mode,
        )
        modules = {}
        for fused in (False, True):
            modules[fused] = VocabParallelEmbedding(
                VOCAB,
                HIDDEN,
                init_method=init_method_normal(0.02),
                reduce_scatter_embeddings=reduce_scatter_embeddings,
                config=dataclasses.replace(config, gradient_accumulation_fusion=fused),
            )
        with torch.no_grad():
            modules[True].weight.copy_(modules[False].weight)
        hooks = [_with_ddp_hook(m.weight) for m in modules.values()]

        torch.manual_seed(1)
        for num_tokens in (4096, 2048, 10240):
            ids = _token_ids("zipf", num_tokens).view(2, num_tokens // 2)
            outputs = {fused: m(ids) for fused, m in modules.items()}
            grad = torch.randn_like(outputs[False])
            for out in outputs.values():
                out.backward(grad)
            _assert_same_bits(modules[True].weight.main_grad, modules[False].weight.main_grad)

        assert modules[False].weight.grad_added_to_main_grad is False
        assert modules[True].weight.grad_added_to_main_grad is not deterministic_mode
        assert modules[True].weight.grad is None
        del hooks

    @pytest.mark.parametrize("overlap_grad_reduce", [False, True])
    @pytest.mark.parametrize("grad_reduce_in_fp32", [False, True])
    def test_main_grad_matches_unfused_under_ddp(self, overlap_grad_reduce, grad_reduce_in_fp32):
        """Megatron DDP's own hooks and gradient buffers, data-parallel over all ranks."""
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=8,
            params_dtype=torch.bfloat16,
            bf16=True,
        )
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            overlap_grad_reduce=overlap_grad_reduce,
            bucket_size=None,
        )
        modules = {
            fused: VocabParallelEmbedding(
                VOCAB,
                HIDDEN,
                init_method=init_method_normal(0.02),
                config=dataclasses.replace(config, gradient_accumulation_fusion=fused),
            )
            for fused in (False, True)
        }
        with torch.no_grad():
            modules[True].weight.copy_(modules[False].weight)
        models = {
            fused: DistributedDataParallel(config, ddp_config=ddp_config, module=m)
            for fused, m in modules.items()
        }

        torch.manual_seed(1 + Utils.rank)
        batches = []
        for num_tokens in (4096, 2048, 4096):
            ids = _token_ids("zipf", num_tokens).view(2, num_tokens // 2)
            batches.append((ids, torch.randn(2, num_tokens // 2, HIDDEN, device="cuda")))
        for model in models.values():
            model.zero_grad_buffer()
            for i, (ids, grad) in enumerate(batches):
                if i < len(batches) - 1:
                    with model.no_sync():
                        model(ids).backward(grad.to(torch.bfloat16))
                else:
                    model(ids).backward(grad.to(torch.bfloat16))
            model.finish_grad_sync()
        _assert_same_bits(modules[True].weight.main_grad, modules[False].weight.main_grad)
        assert modules[True].weight.grad is None

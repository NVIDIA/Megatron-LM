# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The embedding weight gradient accumulated into main_grad gives main_grad the same bits as
PyTorch's embedding backward followed by ``main_grad.add_`` (what Megatron's DDP hook does)."""

import contextlib
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

    def test_shared_weight_two_uses_return_zero_tensor(self):
        """A weight used twice on one stage, as by the main and MTP embeddings, is marked
        ``zero_out_wgrad`` but with an fp32 main_grad still accumulates into it; autograd sums
        two ZeroTensors instead of two dense [vocab, hidden] gradients."""
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=8,
            params_dtype=torch.bfloat16,
            bf16=True,
            gradient_accumulation_fusion=True,
        )
        module = VocabParallelEmbedding(
            VOCAB, HIDDEN, init_method=init_method_normal(0.02), config=config
        )
        weight = module.weight
        weight.zero_out_wgrad = True
        weight.main_grad = torch.zeros(weight.shape, dtype=torch.float32, device="cuda")
        weight.grad_added_to_main_grad = False

        torch.manual_seed(1)
        ids = [_token_ids("zipf", 4096).view(2, 2048) for _ in range(2)]
        grads = [torch.randn(2, 2048, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in ids]
        torch.autograd.backward([module(i) for i in ids], grads)

        assert torch._is_zerotensor(weight.grad)
        assert weight.grad_added_to_main_grad
        # Each use adds what PyTorch's embedding backward followed by main_grad.add_ would.
        expected = torch.zeros_like(weight.main_grad)
        for i, g in zip(ids, grads):
            expected.add_(torch.ops.aten.embedding_dense_backward(g, i, VOCAB, -1, False))
        torch.testing.assert_close(weight.main_grad, expected)

    @pytest.mark.parametrize(
        "dtype, start, use", [(torch.float16, 65024.0, 2048.0), (torch.bfloat16, 256.0, 1.0)]
    )
    def test_shared_weight_low_precision_main_grad_sums_uses_first(self, dtype, start, use):
        """With an fp16/bf16 main_grad, the +use and -use of a shared weight cancel before DDP's
        hook adds them. Adding them one at a time would round ``start + use`` first: fp16
        overflows to inf, and bf16 rounds 257 to 256 and ends at 255."""
        Utils.initialize_model_parallel()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=8,
            params_dtype=dtype,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            gradient_accumulation_fusion=True,
        )
        module = VocabParallelEmbedding(
            VOCAB, HIDDEN, init_method=init_method_normal(0.02), config=config
        )
        module.weight.zero_out_wgrad = True
        model = DistributedDataParallel(
            config,
            ddp_config=DistributedDataParallelConfig(grad_reduce_in_fp32=False, bucket_size=None),
            module=module,
        )
        main_grad = module.weight.main_grad
        assert main_grad.dtype == dtype

        model.zero_grad_buffer()
        token = 7
        main_grad[token].fill_(start)
        ids = torch.full((1, 1), token, device="cuda")
        grads = [
            torch.full((1, 1, HIDDEN), sign * use, device="cuda", dtype=dtype) for sign in (1, -1)
        ]
        with model.no_sync():
            torch.autograd.backward([model(ids), model(ids)], grads)

        assert torch.equal(main_grad[token], torch.full_like(main_grad[token], start))

    @pytest.mark.parametrize("overlap_grad_reduce", [False, True])
    @pytest.mark.parametrize("unfused_use", [False, True])
    def test_shared_weight_under_ddp(self, overlap_grad_reduce, unfused_use, monkeypatch):
        """Megatron DDP's hook on a ``zero_out_wgrad`` weight used by two fused embeddings and,
        optionally, an unfused consumer (a tied output layer computed outside the fusion): the
        hook skips the ZeroTensors, the unfused gradient lands in main_grad exactly once, and
        the last microbatch still marks the bucket ready for the data-parallel reduction.
        Measured against an fp64 oracle, with the dense (unfused) path as the error baseline."""
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
            grad_reduce_in_fp32=True, overlap_grad_reduce=overlap_grad_reduce, bucket_size=None
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
        for m in modules.values():
            m.weight.zero_out_wgrad = True
        models = {
            fused: DistributedDataParallel(config, ddp_config=ddp_config, module=m)
            for fused, m in modules.items()
        }

        torch.manual_seed(1)
        batches = []
        for num_tokens in (4096, 2048):  # inside and below the fused token window
            uses = [_token_ids("zipf", num_tokens).view(2, num_tokens // 2) for _ in range(2)]
            grads = [
                torch.randn(2, num_tokens // 2, HIDDEN, device="cuda", dtype=torch.bfloat16)
                for _ in uses
            ]
            hidden = torch.randn(128, HIDDEN, device="cuda", dtype=torch.bfloat16)
            logits_grad = torch.randn(128, VOCAB, device="cuda", dtype=torch.bfloat16)
            batches.append((uses, grads, hidden, logits_grad))

        expected = torch.zeros(VOCAB, HIDDEN, dtype=torch.float64, device="cuda")
        for uses, grads, hidden, logits_grad in batches:
            for ids, grad in zip(uses, grads):
                expected += torch.ops.aten.embedding_dense_backward(
                    grad.double(), ids, VOCAB, -1, False
                )
            if unfused_use:
                expected += logits_grad.double().t() @ hidden.double()

        errors = {}
        for fused, model in models.items():
            weight = modules[fused].weight
            zero_tensor_adds = []
            add_ = weight.main_grad.add_

            def recording_add_(other, *args, **kwargs):
                if torch._is_zerotensor(other):
                    zero_tensor_adds.append(other.shape)
                return add_(other, *args, **kwargs)

            monkeypatch.setattr(weight.main_grad, "add_", recording_add_)
            # Two iterations: with overlap_grad_reduce, DDP launches the reduction from the hook
            # only once the first iteration has recorded how often each param becomes ready, and
            # finish_grad_sync() then fails unless the last microbatch marked the bucket ready.
            for _ in range(2):
                model.zero_grad_buffer()
                for i, (uses, grads, hidden, logits_grad) in enumerate(batches):
                    last = i == len(batches) - 1
                    with contextlib.nullcontext() if last else model.no_sync():
                        outputs = [model(ids) for ids in uses]
                        if unfused_use:
                            outputs.append(torch.nn.functional.linear(hidden, weight))
                            grads = grads + [logits_grad]
                        torch.autograd.backward(outputs, grads)
                model.finish_grad_sync()
                assert weight.grad is None
            # Every rank saw the same batches, so the data-parallel mean is this rank's sum.
            errors[fused] = (weight.main_grad.double() - expected).abs().max().item()
            # With only fused uses the hook's param.grad is a ZeroTensor: nothing to add.
            assert not zero_tensor_adds, zero_tensor_adds

        # Dropping or double-counting any use would leave an error the size of that use.
        assert errors[True] <= 1.5 * errors[False] + 1e-6 * expected.abs().max().item(), errors

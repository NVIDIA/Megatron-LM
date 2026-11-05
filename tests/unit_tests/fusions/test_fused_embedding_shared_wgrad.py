# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared embeddings accumulate each consumer directly into DDP's main_grad.

PP1 MTP/tied weights keep zero_out_wgrad=True so other fused consumers cannot leak
uninitialized placeholders into autograd. This flag must not force embedding
consumers through a shared BF16 leaf sum before their FP32 main_grad accumulation.
"""

import os
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.fusions.fused_embedding_wgrad import HAVE_TRITON, can_accumulate_embedding_wgrad
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, VocabParallelEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

VOCAB, HIDDEN = 512, 64


@pytest.fixture(scope="module")
def shared_embedding_ddp():
    """Work under standalone torchrun and the regular distributed unit-test runner."""
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    owns_world = not dist.is_initialized()
    if owns_world:
        if "RANK" not in os.environ:
            pytest.skip("Requires torchrun with at least two CUDA ranks")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    if dist.get_world_size() < 2:
        if owns_world:
            dist.destroy_process_group()
        pytest.skip("Requires at least two data-parallel ranks")
    owns_parallel = not ps.model_parallel_is_initialized()
    if owns_parallel:
        ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    assert ps.get_tensor_model_parallel_world_size() == 1
    assert ps.get_pipeline_model_parallel_world_size() == 1
    assert ps.get_context_parallel_world_size() == 1
    assert ps.get_gtp_weight_remat_world_size() == 1
    deterministic = torch.are_deterministic_algorithms_enabled()
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_parallel_cuda_manual_seed(123)
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if owns_parallel:
        ps.destroy_model_parallel()
    if owns_world:
        dist.destroy_process_group()


class _SharedEmbeddingConsumers(torch.nn.Module):
    def __init__(self, config, sharing):
        super().__init__()
        self.sharing = sharing
        self.embedding = VocabParallelEmbedding(
            VOCAB, HIDDEN, config=config, init_method=lambda value: value
        )
        self.embedding.weight.zero_out_wgrad = True
        with torch.no_grad():
            generator = torch.Generator().manual_seed(921)
            self.embedding.weight.copy_(
                torch.randn(VOCAB, HIDDEN, generator=generator).bfloat16().cuda()
            )
        if sharing == "native_fused":
            self.linear = ColumnParallelLinear(
                HIDDEN,
                VOCAB,
                config=config,
                init_method=lambda value: value,
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                skip_weight_param_allocation=True,
            )

    def forward(self, ids, linear_input):
        outputs = [self.embedding(value) for value in ids]
        if self.sharing == "unfused":
            outputs.append(F.linear(linear_input, self.embedding.weight))
        elif self.sharing == "native_fused":
            outputs.append(self.linear(linear_input, weight=self.embedding.weight)[0])
        return outputs


def _inputs(rank, step, microbatch, uses, num_tokens, main_grad, sharing):
    ids, gradients, dense = [], [], []
    generator = torch.Generator().manual_seed(1300 + rank * 83 + step * 19 + microbatch)
    token_ids = torch.randint(0, VOCAB, (1, num_tokens), generator=generator).cuda()
    for use in range(uses):
        # Shared IDs and a half-BF16-ULP contribution distinguish per-use FP32
        # accumulation from rounding their sum in the BF16 shared leaf.
        scale = (rank + 1) * (1.0 if use % 2 == 0 else 1.0 / 256)
        grad = torch.full((1, num_tokens, HIDDEN), scale, dtype=torch.bfloat16, device="cuda")
        assert can_accumulate_embedding_wgrad(token_ids, grad, main_grad) == (num_tokens == 4096)
        ids.append(token_ids)
        gradients.append(grad)
        dense.append(torch.ops.aten.embedding_dense_backward(grad, token_ids, VOCAB, -1, False))
    linear_input = torch.full(
        (16, 1, HIDDEN), 1.0 / 32, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    if sharing != "none":
        linear_grad = torch.full(
            (16, 1, VOCAB), (rank + 1) / 32, dtype=torch.bfloat16, device="cuda"
        )
        gradients.append(linear_grad)
        # Small powers of two make this independent FP32 matmul oracle exact.
        linear_dense = (
            linear_grad.reshape(-1, VOCAB).float().T
            @ linear_input.detach().reshape(-1, HIDDEN).float()
        )
        dense.append(linear_dense.bfloat16() if sharing == "unfused" else linear_dense)
    return ids, linear_input, gradients, dense


@pytest.mark.parametrize("num_tokens", [32, 4096], ids=["dense_fallback", "triton_fast_path"])
@pytest.mark.parametrize("sharing", ["none", "unfused", "native_fused"])
@pytest.mark.parametrize("grad_reduce_in_fp32", [True, False])
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
def test_shared_embedding_main_grad_under_real_ddp(
    shared_embedding_ddp, num_tokens, sharing, grad_reduce_in_fp32, overlap_grad_reduce
):
    if num_tokens == 4096 and not HAVE_TRITON:
        pytest.skip("Fast embedding wgrad path requires Triton")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        perform_initialization=False,
        gradient_accumulation_fusion=True,
        calculate_per_token_loss=True,
        deterministic_mode=False,
    )
    module = _SharedEmbeddingConsumers(config, sharing).cuda()
    model = DistributedDataParallel(
        config,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            overlap_grad_reduce=overlap_grad_reduce,
            bucket_size=None,
            use_distributed_optimizer=False,
        ),
        module=module,
    )
    weight = module.embedding.weight
    assert weight.zero_out_wgrad
    assert len(list(module.parameters())) == 1
    assert weight.main_grad.dtype == (torch.float32 if grad_reduce_in_fp32 else torch.bfloat16)
    for step, (rounds, uses) in enumerate(((3, 2), (1, 6), (4, 2)), 1):
        model.zero_grad_buffer()
        expected = torch.zeros_like(weight.main_grad)
        for microbatch in range(rounds):
            ids, linear_input, gradients, dense = _inputs(
                dist.get_rank(), step, microbatch, uses, num_tokens, expected, sharing
            )
            order = []
            context = model.no_sync() if microbatch < rounds - 1 else nullcontext()
            with context:
                outputs = model(ids, linear_input)
                for value, indices in zip(outputs[:uses], ids):
                    torch.testing.assert_close(value, weight.detach()[indices], rtol=0, atol=0)
                for index, out in enumerate(outputs):
                    out.register_hook(lambda grad, index=index: (order.append(index), grad)[1])
                if sharing == "native_fused":
                    try:
                        from transformer_engine.pytorch.module.base import get_dummy_wgrad
                    except ImportError:
                        # Native linear uses freshly allocated zeros without TE.
                        pass
                    else:
                        get_dummy_wgrad(list(weight.shape), weight.dtype).fill_(12345.0)
                loss = sum(
                    (out.float() * grad.float()).sum() for out, grad in zip(outputs, gradients)
                )
                loss.backward()
            assert sorted(order) == list(range(len(outputs)))
            # The ordinary linear contribution reaches DDP only after all
            # fused consumers have already added their wgrad to main_grad.
            if sharing == "unfused":
                order = [index for index in order if index != uses] + [uses]
            for index in order:
                expected.add_(dense[index])
            assert weight.grad is None
            assert weight.zero_out_wgrad
            assert weight.grad_added_to_main_grad
        model.finish_grad_sync()
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(weight.main_grad, expected, rtol=0, atol=0)
        assert bool(torch.isfinite(weight.main_grad).all())
        if overlap_grad_reduce:
            assert model.bucket_groups
            for group in model.bucket_groups:
                assert group.per_param_grad_ready_counts[weight] == 1
                # The first batch's golden count is saved by the next reset.
                assert group.golden_per_param_grad_ready_counts.get(weight, 0) == (
                    0 if step == 1 else 1
                )
                assert group.is_first_batch == (step == 1)

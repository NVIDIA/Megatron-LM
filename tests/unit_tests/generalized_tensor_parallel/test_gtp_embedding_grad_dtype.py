# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Embedding GTP must reduce local wgrad in its main-grad accumulation dtype."""

import dataclasses
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding


@pytest.fixture(scope="module")
def embedding_groups():
    """Standalone torchrun support, also usable with the regular unit-test runner."""
    owns_world = not dist.is_initialized()
    if owns_world:
        if "RANK" not in os.environ or not torch.cuda.is_available():
            pytest.skip("Requires torchrun with at least two CUDA ranks")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    world, rank = dist.get_world_size(), dist.get_rank()
    if world < 2 or world % 2:
        pytest.skip("Requires an even CUDA world size >= 2")
    singletons = [dist.new_group([r]) for r in range(world)]
    pairs = [dist.new_group([r, r + 1]) for r in range(0, world, 2)]
    yield singletons[rank], pairs[rank // 2]
    if owns_world:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def preserve_gtp_config():
    previous = dataclasses.asdict(gtp.GTP_CONFIG)
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    gtp.reset_gtp_state()
    yield
    gtp.wait_for_gtp_grad_reduction_on_current_stream()
    gtp.reset_gtp_state()
    gtp._GTP_PARAMS.clear()
    gtp.update_gtp_config(**previous)
    torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("main_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_embedding_reduction_uses_main_grad_dtype(
    embedding_groups, main_dtype, asynchronous, per_token
):
    singleton, pair = embedding_groups
    pair_rank = dist.get_rank(pair)
    gtp.update_gtp_config(
        calculate_per_token_loss=per_token,
        weight_prefetch=False,
        async_reduction=asynchronous,
        reduce_scatter_with_fp32_accumulation=False,
    )
    config = ModelParallelConfig(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        gradient_accumulation_fusion=True,
        deterministic_mode=False,
    )
    pgs = ProcessGroupCollection(tp=singleton, gtp_remat=pair, dp_cp=singleton)
    model = VocabParallelEmbedding(
        32, 16, config=config, init_method=lambda x: x, tp_group=singleton, pg_collection=pgs
    )
    canonical = (torch.arange(512, device="cuda").reshape(32, 16).float() / 64).bfloat16()
    with torch.no_grad():
        model.weight.copy_(canonical.chunk(2, dim=0)[pair_rank])
    model.weight.main_grad = torch.zeros_like(model.weight, dtype=main_dtype)
    model.weight.grad_added_to_main_grad = False
    model.weight.zero_out_wgrad = True
    model.weight._debug_name = "embedding.word_embeddings.weight"

    original_rs = model.weight.wgrad_reduce_scatter
    observed_dtypes = []

    def observed_rs(wgrad, *args, **kwargs):
        observed_dtypes.append(wgrad.dtype)
        # The one-weight chain head naturally chooses sync; exercise the real
        # async implementation as well, including repeated-use draining.
        kwargs["async_op"] = asynchronous
        return original_rs(wgrad, *args, **kwargs)

    model.weight.wgrad_reduce_scatter = observed_rs
    ids = torch.tensor([[0, 1, 0, 2] if pair_rank == 0 else [0, 1, 3]], device="cuda")
    # 1 + 2^-8 is exactly between adjacent BF16 values. A BF16 collective
    # rounds away the small peer's contribution before an FP32 main-grad add.
    scale = 1.0 if pair_rank == 0 else 2**-8
    expected = torch.zeros_like(model.weight.main_grad)
    low_precision_expected = torch.zeros_like(expected)
    for _ in range(3):
        # Main and MTP embedding lookups share a parameter before either backward.
        outputs = [model(ids), model(ids)]
        for output in reversed(outputs):
            torch.testing.assert_close(output, canonical[ids], rtol=0, atol=0)
            upstream = torch.full_like(output, scale)
            dense = torch.ops.aten.embedding_dense_backward(upstream, ids, 32, -1, False)
            assert dense.dtype == torch.bfloat16
            send = dense.to(main_dtype)
            low_send = dense.clone()
            if not per_token:
                send = send * 0.5
                low_send.mul_(0.5)
            reduced = torch.empty_like(expected)
            low_reduced = torch.empty_like(expected, dtype=torch.bfloat16)
            dist.reduce_scatter_tensor(reduced, send.contiguous(), group=pair)
            dist.reduce_scatter_tensor(low_reduced, low_send, group=pair)
            expected.add_(reduced)
            low_precision_expected.add_(low_reduced)
            output.backward(upstream)
            model.weight.grad = None
    if asynchronous:
        model.weight._wait_reduce_scatter(finalize_grad=True)
    gtp.wait_for_gtp_grad_reduction_on_current_stream()
    assert observed_dtypes == [main_dtype] * 6
    torch.testing.assert_close(model.weight.main_grad, expected, rtol=0, atol=0)
    if main_dtype == torch.float32:
        # Only the shard owning the deliberately repeated tokens sees the signal.
        signal = torch.tensor(int(not torch.equal(expected, low_precision_expected)), device="cuda")
        dist.all_reduce(signal, group=pair)
        assert signal.item() > 0, "The regression must expose BF16 collective rounding"
    else:
        torch.testing.assert_close(expected, low_precision_expected, rtol=0, atol=0)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 model, vision, Engram and DSpark integration on a reduced-width backbone."""

from copy import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.deepseek_v41.config import (
    DeepSeekV41Config,
    DSparkConfig,
    EngramConfig,
    VisionConfig,
)
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
from megatron.core.models.deepseek_v41.image_processing import ImageInput
from megatron.core.models.deepseek_v41.model import DeepSeekV41Model
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config


def tiny_config(dtype=torch.float32, *, all_components=False, **overrides):
    """Keep every attention role while reducing width, experts and memory table rows."""
    values = dict(
        num_layers=12,
        mhc_single_pass=True,
        csa_compress_ratios=[r for ratio in (0, 2, 2, 1, 1, 1) for r in (ratio, 0)],
        csa2_kv_source_layers=[2, 6],
        csa2_index_source_layers=[2, 6, 8],
        csa2_candidate_source_layer=6,
        dsa_indexer_loss_coeff=0.01,
        moe_token_dispatcher_type="alltoall",
    )
    values.update(overrides)
    config = DeepSeekV41Config.from_config(_make_config(params_dtype=dtype, **values))
    if all_components:
        config.engram_config = EngramConfig(
            layer_ids=[1, 4],
            num_embeddings=[0, 0],
            max_ngram_size=3,
            vocab_size=101,
            n_heads=2,
            head_dim=8,
            pad_token_id=2,
            compressed_vocab_size=128,
        )
        args = SimpleNamespace(
            **{"engram_" + k: v for k, v in asdict(config.engram_config).items()}
        )
        layout = EngramLayout.from_args(args)
        config.engram_config = replace(
            config.engram_config,
            num_embeddings=[sum((sum(row) for row in layer)) for layer in layout.primes],
        )
        config.vision_config = VisionConfig(
            num_hidden_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=48,
            patch_size=2,
            rope_theta=10000,
            downsample_ratio=3,
            max_image_tokens=32,
            min_pixels=36,
        )
    return config


@pytest.fixture
def groups():
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("all_components", [False, True])
def test_full_recompute_matches_eager_for_deepseek_model(groups, all_components):
    """Replayed decoder layers preserve Engram, vision, and shared CSA2 gradients."""
    eager = tiny_config(all_components=all_components, dsa_indexer_loss_coeff=0)
    recompute = tiny_config(
        all_components=all_components,
        dsa_indexer_loss_coeff=0,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    models = [
        DeepSeekV41Model(config, 128, 64, pg_collection=groups, token_map=torch.arange(128)).cuda()
        for config in (eager, recompute)
    ]
    models[1].load_state_dict(models[0].state_dict())
    ids = torch.randint(0, 126, (1, 17), device="cuda")
    positions = torch.arange(17, device="cuda").expand_as(ids)
    images = None
    if all_components:
        image = ImageInput(
            1,
            torch.randn(9, 3, 2, 2, device="cuda"),
            3,
            3,
            torch.tensor([0, 1, 2, 3], device="cuda"),
        )
        images = [[image]]
    losses = [
        model(ids, positions, labels=ids.roll(-1, 1), images=images).mean() for model in models
    ]
    torch.testing.assert_close(losses[1], losses[0], atol=0, rtol=0)
    for loss in losses:
        loss.backward()
    for (name, expected), (_, actual) in zip(
        models[0].named_parameters(), models[1].named_parameters()
    ):
        assert (actual.grad is None) == (expected.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-4, rtol=2e-2, msg=name)


def test_packed_thd_text_matches_independent_sequences(groups):
    """The model passes logical and physical THD prefixes through every CSA2 layer."""
    config = tiny_config(dsa_indexer_loss_coeff=0)
    model = DeepSeekV41Model(config, 128, 64, pg_collection=groups).cuda().eval()
    ids = torch.randint(0, 126, (1, 12), device="cuda")
    positions = torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 3, 0, 0, 0]], device="cuda")
    logical = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    physical = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=7,
        max_seqlen_kv=7,
    )
    actual = model(ids, positions, packed_seq_params=packed)
    expected = torch.cat(
        [
            model(ids[:, start : start + length], positions[:, start : start + length])
            for start, length in ((0, 3), (5, 4))
        ],
        dim=1,
    )
    torch.testing.assert_close(actual[:, [0, 1, 2, 5, 6, 7, 8]], expected, atol=2e-5, rtol=2e-4)


def test_packed_thd_full_recompute_matches_eager(groups):
    """Replaying shared packed CSA2 state must retain source gradients."""
    eager = tiny_config(dsa_indexer_loss_coeff=0)
    recompute = tiny_config(
        dsa_indexer_loss_coeff=0,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    models = [
        DeepSeekV41Model(config, 128, 64, pg_collection=groups).cuda()
        for config in (eager, recompute)
    ]
    models[1].load_state_dict(models[0].state_dict())
    ids = torch.randint(0, 126, (1, 7), device="cuda")
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], device="cuda")
    prefixes = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=prefixes,
        cu_seqlens_kv=prefixes,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    losses = [
        model(ids, positions, labels=ids.roll(-1, 1), packed_seq_params=packed).mean()
        for model in models
    ]
    torch.testing.assert_close(losses[1], losses[0], atol=0, rtol=0)
    for loss in losses:
        loss.backward()
    for (name, expected), (_, actual) in zip(
        models[0].named_parameters(), models[1].named_parameters()
    ):
        assert (actual.grad is None) == (expected.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-4, rtol=2e-2, msg=name)


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("all_components", [False, True])
def test_packed_thd_cp_full_model_matches_unsplit(recompute, all_components):
    """Cross-layer Full/Reindex/Reuse and MoE agree across one CP cut."""
    if Utils.world_size not in (2, 4):
        pytest.skip("run with torchrun --nproc-per-node=2 or 4")
    world = Utils.world_size
    Utils.initialize_model_parallel(context_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    singleton_groups = [torch.distributed.new_group([rank]) for rank in range(world)]
    reference_groups = copy(groups)
    reference_groups.cp = singleton_groups[torch.distributed.get_rank()]
    try:
        overrides = (
            dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
            if recompute
            else {}
        )
        cp_config = tiny_config(
            all_components=all_components,
            dsa_indexer_loss_coeff=0,
            context_parallel_size=world,
            **overrides,
        )
        ref_config = tiny_config(
            all_components=all_components,
            dsa_indexer_loss_coeff=0,
            context_parallel_size=1,
            **overrides,
        )
        distributed_model = (
            DeepSeekV41Model(
                cp_config,
                128,
                64,
                pg_collection=groups,
                token_map=torch.arange(128) if all_components else None,
            )
            .cuda()
            .train()
        )
        for tensor in distributed_model.state_dict().values():
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                torch.distributed.broadcast(tensor, src=0, group=groups.cp)
        reference_model = (
            DeepSeekV41Model(
                ref_config,
                128,
                64,
                pg_collection=reference_groups,
                token_map=torch.arange(128) if all_components else None,
            )
            .cuda()
            .train()
        )
        reference_model.load_state_dict(distributed_model.state_dict())
        total_rows = 4 * world
        ids = torch.randint(0, 126, (1, total_rows), device="cuda")
        torch.distributed.broadcast(ids, src=0, group=groups.cp)
        positions = torch.tensor([[0, 1, 2] + list(range(total_rows - 3))], device="cuda")
        logical = torch.tensor([0, 3, total_rows], device="cuda", dtype=torch.int32)
        rank = torch.distributed.get_rank()
        packed_cp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=logical,
            cu_seqlens_kv=logical,
            max_seqlen_q=total_rows - 3,
            max_seqlen_kv=total_rows - 3,
            cp_partition_mode="contiguous",
        )
        packed_ref = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=logical,
            cu_seqlens_kv=logical,
            max_seqlen_q=total_rows - 3,
            max_seqlen_kv=total_rows - 3,
        )
        images = None
        if all_components:
            patches = torch.randn(9, 3, 2, 2, device="cuda")
            torch.distributed.broadcast(patches, src=0, group=groups.cp)
            images = [[], [ImageInput(0, patches, 3, 3, torch.tensor([0, 1, 2, 3], device="cuda"))]]
        actual = distributed_model(
            ids[:, rank * 4 : (rank + 1) * 4],
            positions[:, rank * 4 : (rank + 1) * 4],
            packed_seq_params=packed_cp,
            images=images,
        )
        expected = reference_model(ids, positions, packed_seq_params=packed_ref, images=images)
        error = (actual - expected[:, rank * 4 : (rank + 1) * 4]).abs().amax()
        torch.distributed.all_reduce(error, op=torch.distributed.ReduceOp.MAX, group=groups.cp)
        if error > 1e-4:
            pytest.fail(f"CP full-model maximum logit error {error.item():.6g}")
        actual.float().square().sum().backward()
        expected.float().square().sum().backward()
        source = distributed_model.decoder.layers[2].inner_layer.self_attention
        reference_source = reference_model.decoder.layers[2].inner_layer.self_attention
        grad = source.core_attention.compressor.linear_wkv.weight.grad
        reference_grad = reference_source.core_attention.compressor.linear_wkv.weight.grad
        torch.distributed.all_reduce(grad, group=groups.cp)
        grad_error = (grad - reference_grad).abs().amax()
        torch.distributed.all_reduce(grad_error, op=torch.distributed.ReduceOp.MAX, group=groups.cp)
        if grad_error > 1e-3:
            pytest.fail(f"CP full-model compressor gradient error {grad_error.item():.6g}")
        if all_components:
            cp_parameters = dict(distributed_model.named_parameters())
            ref_parameters = dict(reference_model.named_parameters())
            for name in (
                "vision.patch_embed.proj.weight",
                "aligner.w1.weight",
                "decoder.layers.2.engram.wkv.weight",
            ):
                cp_grad = cp_parameters[name].grad
                if cp_grad is None:
                    cp_grad = torch.zeros_like(cp_parameters[name])
                torch.distributed.all_reduce(cp_grad, group=groups.cp)
                reference_grad = ref_parameters[name].grad
                error = (cp_grad - reference_grad).abs().amax()
                torch.distributed.all_reduce(
                    error, op=torch.distributed.ReduceOp.MAX, group=groups.cp
                )
                if error > 1e-3:
                    pytest.fail(f"CP {name} gradient error {error.item():.6g}")
    finally:
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        Utils.inited = False


def test_two_pending_microbatches_and_causality(groups):
    config = tiny_config(dsa_indexer_loss_coeff=0)
    model = DeepSeekV41Model(config, 128, 64, pg_collection=groups).cuda().eval()
    ids = torch.randint(0, 126, (1, 17), device="cuda")
    positions = torch.arange(17, device="cuda").expand_as(ids)
    first = model(ids, positions)
    changed = ids.clone()
    changed[:, 10:] = (changed[:, 10:] + 1) % 128
    second = model(changed, positions)
    torch.testing.assert_close(first[:, :10], second[:, :10], atol=2e-06, rtol=2e-05)
    (first.square().mean() + second.square().mean()).backward()
    assert (
        model.decoder.layers[
            2
        ].inner_layer.self_attention.core_attention.compressor.linear_wkv.weight.grad
        is not None
    )


@pytest.mark.parametrize("all_components", [False, True])
def test_forward_backward_and_optimizer_step(groups, all_components):
    """Train the supported components with an actual optimizer update."""
    config = tiny_config(all_components=all_components)
    model = DeepSeekV41Model(
        config, 128, 64, pg_collection=groups, token_map=torch.arange(128)
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ids = torch.randint(0, 126, (1, 17), device="cuda")
    positions = torch.arange(17, device="cuda").expand_as(ids)
    kwargs = {}
    if all_components:
        image = ImageInput(
            1,
            torch.randn(9, 3, 2, 2, device="cuda"),
            3,
            3,
            torch.tensor([0, 1, 2, 3], device="cuda"),
        )
        kwargs["images"] = [[image]]
    loss = model(ids, positions, labels=ids.roll(-1, 1), **kwargs).mean()
    loss.backward()
    assert torch.isfinite(loss)
    if all_components:
        for component in ("engram", "vision", "aligner"):
            gradients = [
                p.grad
                for name, p in model.named_parameters()
                if component in name and p.grad is not None
            ]
            assert gradients and any((g.abs().sum() > 0 for g in gradients))
            assert all((torch.isfinite(g).all() for g in gradients))
    optimizer.step()

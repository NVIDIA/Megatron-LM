# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 model, vision, Engram and DSpark integration on a reduced-width backbone."""

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
            num_embeddings=[sum(sum(row) for row in layer) for layer in layout.primes],
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
        config.dspark_config = DSparkConfig(
            num_layers=2,
            block_size=3,
            noise_token_id=127,
            target_layer_ids=[4, 5],
            markov_rank=8,
            n_routed_experts=2,
            num_experts_per_tok=1,
        )
    return config


@pytest.fixture
def groups():
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("all_components", [False, True])
def test_forward_backward_and_optimizer_step(groups, all_components):
    torch.manual_seed(101)
    config = tiny_config(all_components=all_components)
    model = DeepSeekV41Model(
        config, 128, 64, pg_collection=groups, token_map=torch.arange(128)
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
        kwargs.update(images=[[image]], draft_anchor_positions=torch.tensor([[8]], device="cuda"))
    output = model(ids, positions, labels=ids.roll(-1, 1), **kwargs)
    loss = output.backbone.mean() + output.draft_loss if all_components else output.mean()
    loss.backward()
    assert torch.isfinite(loss)
    if all_components:
        for name in ("engram", "vision", "aligner", "dspark"):
            gradients = [
                p.grad for n, p in model.named_parameters() if name in n and p.grad is not None
            ]
            assert gradients and any(g.abs().sum() > 0 for g in gradients), name
            assert all(torch.isfinite(g).all() for g in gradients), name
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = model(ids, positions, labels=ids.roll(-1, 1), **kwargs)
    assert torch.isfinite(second.backbone if all_components else second).all()


def test_draft_loss_does_not_update_backbone(groups):
    config = tiny_config(all_components=True)
    model = DeepSeekV41Model(
        config, 128, 64, pg_collection=groups, token_map=torch.arange(128)
    ).cuda()
    ids = torch.randint(0, 126, (1, 17), device="cuda")
    positions = torch.arange(17, device="cuda").expand_as(ids)
    result = model(ids, positions, draft_anchor_positions=torch.tensor([[8]], device="cuda"))
    result.draft_loss.backward()
    for name, parameter in model.named_parameters():
        if not name.startswith("dspark."):
            assert parameter.grad is None, name
    assert model.dspark.main_proj.weight.grad.abs().sum() > 0


def test_two_pending_microbatches_and_causality(groups):
    config = tiny_config(dsa_indexer_loss_coeff=0)
    model = DeepSeekV41Model(config, 128, 64, pg_collection=groups).cuda().eval()
    ids = torch.randint(0, 126, (1, 17), device="cuda")
    positions = torch.arange(17, device="cuda").expand_as(ids)
    first = model(ids, positions)
    changed = ids.clone()
    changed[:, 10:] = (changed[:, 10:] + 1) % 128
    second = model(changed, positions)
    torch.testing.assert_close(first[:, :10], second[:, :10], atol=2e-6, rtol=2e-5)
    (first.square().mean() + second.square().mean()).backward()
    assert (
        model.decoder.layers[
            2
        ].inner_layer.self_attention.core_attention.compressor.linear_wkv.weight.grad
        is not None
    )


def test_draft_stage_freezes_backbone_weights_and_routing_biases(groups):
    """AdamW decay and expert-bias updates must not change the frozen teacher."""
    from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias

    config = tiny_config(all_components=True)
    model = DeepSeekV41Model(
        config, 128, 64, pg_collection=groups, token_map=torch.arange(128)
    ).cuda()
    model.freeze_backbone_for_draft_training()
    model.train()
    assert not model.decoder.training and model.dspark.training
    before = {
        name: tensor.clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("dspark.") and isinstance(tensor, torch.Tensor)
    }
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    ids = torch.randint(0, 126, (1, 17), device="cuda")
    output = model(
        ids,
        torch.arange(17, device="cuda").expand_as(ids),
        draft_anchor_positions=torch.tensor([[8]], device="cuda"),
    )
    output.draft_loss.backward()
    optimizer.step()
    _update_router_expert_bias([model], config, groups.tp_dp_cp)
    after = model.state_dict()
    for name, tensor in before.items():
        torch.testing.assert_close(tensor, after[name], atol=0, rtol=0, msg=name)

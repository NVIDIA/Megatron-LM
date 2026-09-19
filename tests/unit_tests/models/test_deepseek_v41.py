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
            num_embeddings=[sum((sum(row) for row in layer)) for layer in layout.primes],
        )
    return config


@pytest.fixture
def groups():
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


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
    loss = model(ids, positions, labels=ids.roll(-1, 1), **kwargs).mean()
    loss.backward()
    assert torch.isfinite(loss)
    if all_components:
        for component in ("engram",):
            gradients = [
                p.grad
                for name, p in model.named_parameters()
                if component in name and p.grad is not None
            ]
            assert gradients and any((g.abs().sum() > 0 for g in gradients))
            assert all((torch.isfinite(g).all() for g in gradients))
    optimizer.step()

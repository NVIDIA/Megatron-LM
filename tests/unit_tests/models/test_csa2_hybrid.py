# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 and single-pass mHC work independently of the DeepSeek model recipe."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_csa2_stack_spec, hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config


@pytest.fixture
def groups():
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("single_pass", [False, True])
def test_csa2_with_dense_layers_and_two_pending_microbatches(groups, single_pass):
    """Shared KV/index gradients survive mixed layer patterns and overlapping forwards."""
    config = _make_config(
        num_layers=6,
        csa_compress_ratios=[2, 0, 2, 0, 2, 0],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0, 4],
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        dsa_indexer_loss_coeff=0.01,
        enable_mhc_connections=single_pass,
        mhc_single_pass=single_pass,
        num_moe_experts=None,
        moe_ffn_hidden_size=None,
        activation_func_clamp_value=None,
        moe_shared_expert_intermediate_size=None,
        moe_router_enable_expert_bias=False,
        ffn_hidden_size=64,
    )
    model = HybridModel(
        config,
        hybrid_csa2_stack_spec,
        vocab_size=128,
        max_sequence_length=64,
        hybrid_layer_pattern="V-V-V-",
        position_embedding_type="none",
        pg_collection=groups,
    ).cuda()
    assert type(model.decoder) is HybridStack
    assert len(model.decoder.layers) == 6
    assert not hasattr(model.decoder, "hc_head_fn")
    ids = torch.randint(0, 128, (2, 17), device="cuda")
    positions = torch.arange(17, device="cuda").expand_as(ids)
    first = model(ids, positions, None, labels=ids.roll(-1, 1)).mean()
    second = model(ids.roll(1, 1), positions, None, labels=ids).mean()
    (first + second).backward()
    full_layer = model.decoder.layers[0]
    if single_pass:
        full_layer = full_layer.inner_layer
    owner = full_layer.self_attention.core_attention
    for parameter in (owner.compressor.linear_wkv.weight, owner.indexer.linear_wk.weight):
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("single_trainable_weight", [False, True])
def test_csa2_full_recompute_matches_eager_with_pending_microbatches(
    groups, single_trainable_weight
):
    """Layer replay must use the original shared KV, indices, and mHC mix."""
    common = dict(
        num_layers=6,
        csa_compress_ratios=[2, 0, 2, 0, 2, 0],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0, 4],
        csa2_candidate_source_layer=0,
        csa2_candidate_topk_blocks=2,
        csa2_candidate_block_size=2,
        dsa_indexer_loss_coeff=0,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        num_moe_experts=None,
        moe_ffn_hidden_size=None,
        activation_func_clamp_value=None,
        moe_shared_expert_intermediate_size=None,
        moe_router_enable_expert_bias=False,
        ffn_hidden_size=64,
    )
    configs = [
        _make_config(**common),
        _make_config(
            **common,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        ),
    ]
    models = [
        HybridModel(
            config,
            hybrid_csa2_stack_spec,
            vocab_size=128,
            max_sequence_length=64,
            hybrid_layer_pattern="V-V-V-",
            position_embedding_type="none",
            pg_collection=groups,
        ).cuda()
        for config in configs
    ]
    models[1].load_state_dict(models[0].state_dict())
    if single_trainable_weight:
        for model in models:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            owner = model.decoder.layers[0].inner_layer.self_attention.core_attention
            owner.compressor.linear_wkv.weight.requires_grad_(True)
    ids = torch.randint(0, 128, (2, 17), device="cuda")
    positions = torch.arange(17, device="cuda").expand_as(ids)
    losses = [
        [model(tokens, positions, None, labels=ids).mean() for tokens in (ids, ids.roll(1, 1))]
        for model in models
    ]
    for actual, expected in zip(losses[1], losses[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for model_losses in losses:
        sum(model_losses).backward()
    for (name, expected), (_, actual) in zip(
        models[0].named_parameters(), models[1].named_parameters()
    ):
        assert (actual.grad is None) == (expected.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-4, rtol=2e-2, msg=name)


def _disable_gradient_boundary(block):
    """Run every layer under autograd, as without frozen-prefix inference."""
    block._gradient_boundary_layer_cache = {False: None, True: None}


def _assert_trainable_grads_match(models, reference, atol, rtol):
    """Each parameter gets a gradient exactly when the reference does, and the same one."""
    reference_params = dict(reference.named_parameters())
    assert any(parameter.grad is not None for parameter in reference.parameters())
    for model in models:
        for name, parameter in model.named_parameters():
            expected = reference_params[name].grad
            if expected is None:
                assert parameter.grad is None, name
                continue
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected, atol=atol, rtol=rtol, msg=name)


@pytest.mark.parametrize(
    "kv_source_layers,index_source_layers,first_trainable",
    [([0], [0, 4], 3), ([0, 2], [0, 2, 4], 2)],
    ids=["boundary-after-kv-source", "boundary-on-kv-source"],
)
def test_automatic_frozen_prefix_boundary_matches_eager_and_is_cached(
    groups, kv_source_layers, index_source_layers, first_trainable
):
    """Frozen-prefix inference keeps the trainable layers' gradients exact.

    The reference runs every layer under autograd. The boundary falls either after a
    kv_source layer whose shared KV the trainable layers read, or on a kv_source layer
    whose compressor is trainable and which owns the candidates (block 20 in
    DeepSeek-V4.1-Flash).
    """
    common = dict(
        num_layers=6,
        csa_compress_ratios=[2, 0, 2, 0, 2, 0],
        csa2_kv_source_layers=kv_source_layers,
        csa2_index_source_layers=index_source_layers,
        csa2_candidate_source_layer=kv_source_layers[-1],
        csa2_candidate_topk_blocks=2,
        csa2_candidate_block_size=2,
        dsa_indexer_loss_coeff=0,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        num_moe_experts=None,
        moe_ffn_hidden_size=None,
        activation_func_clamp_value=None,
        moe_shared_expert_intermediate_size=None,
        moe_router_enable_expert_bias=False,
        ffn_hidden_size=64,
    )
    recompute = dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    configs = [_make_config(**common), _make_config(**common), _make_config(**common, **recompute)]
    models = [
        HybridModel(
            config,
            hybrid_csa2_stack_spec,
            vocab_size=128,
            max_sequence_length=64,
            hybrid_layer_pattern="V-V-V-",
            position_embedding_type="none",
            pg_collection=groups,
        ).cuda()
        for config in configs
    ]
    for model in models[1:]:
        model.load_state_dict(models[0].state_dict())
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for layer in model.decoder.layers[first_trainable:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    reference, eager, recomputed = models
    _disable_gradient_boundary(reference.decoder)

    ids = torch.randint(0, 128, (2, 17), device="cuda")
    positions = torch.arange(17, device="cuda").expand_as(ids)
    losses = [model(ids, positions, None, labels=ids).mean() for model in models]
    torch.testing.assert_close(losses[1], losses[0], atol=0, rtol=0)
    torch.testing.assert_close(losses[2], losses[0], atol=0, rtol=0)
    for loss in losses:
        loss.backward()
    for model in (eager, recomputed):
        assert model.decoder._gradient_boundary_layer_cache[False] == first_trainable
    _assert_trainable_grads_match((eager, recomputed), reference, atol=2e-4, rtol=2e-2)


@pytest.mark.parametrize("fused", [False, True])
def test_single_pass_mhc_on_standard_dense_hybrid(groups, fused):
    """A standard attention/MLP model can opt in without any CSA2 or DeepSeek configuration."""
    config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=fused,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        hidden_dropout=0,
        attention_dropout=0,
        gradient_accumulation_fusion=False,
    )
    model = HybridModel(
        config,
        hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=64,
        hybrid_layer_pattern="*-*-",
        position_embedding_type="none",
        pg_collection=groups,
    ).cuda()
    convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
    assert model.decoder.forward_adapter is None
    assert not hasattr(model.decoder, "hc_head_fn")
    ids = torch.randint(0, 128, (2, 16), device="cuda")
    positions = torch.arange(16, device="cuda").expand_as(ids)
    losses = [model(ids, positions, None, labels=ids.roll(-1, 1)).mean() for _ in range(2)]
    sum(losses).backward()
    for layer in model.decoder.layers:
        grad = layer.hyper_connection.mapping_proj.weight.grad
        assert grad is not None and grad.isfinite().all() and grad.abs().sum() > 0


@pytest.mark.parametrize("csa2", [False, True])
def test_standard_transformer_block_state(groups, csa2):
    """The shared state interfaces also work with full attention/MLP transformer layers."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.transformer.transformer_block import TransformerBlock

    if csa2:
        config = _make_config(
            mhc_single_pass=True,
            num_moe_experts=None,
            moe_ffn_hidden_size=None,
            activation_func_clamp_value=None,
            moe_shared_expert_intermediate_size=None,
            moe_router_enable_expert_bias=False,
            ffn_hidden_size=64,
            dsa_indexer_loss_coeff=0.01,
        )
    else:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            enable_mhc_connections=True,
            mhc_single_pass=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            use_cpu_initialization=True,
            hidden_dropout=0,
            attention_dropout=0,
            gradient_accumulation_fusion=False,
        )
    spec = get_transformer_block_with_experimental_attention_variant_spec(config, pp_rank=0)
    block = TransformerBlock(config, spec=spec, pg_collection=groups).cuda()
    convert_module_to_dtype_except_fp32_marked(block, config.params_dtype)
    inputs = [
        torch.randn(
            16, 2, config.hidden_size, device="cuda", dtype=config.params_dtype, requires_grad=True
        )
        for _ in range(2)
    ]
    outputs = [block(x, attention_mask=None) for x in inputs]
    sum(output.float().square().mean() for output in outputs).backward()
    for x in inputs:
        assert x.grad is not None and x.grad.isfinite().all()
    for layer in block.layers:
        assert layer.mlp_hyper_connection.mapping_proj.weight.grad is not None


@pytest.mark.parametrize(
    "recompute_method,expected_checkpoints",
    [("uniform", 2), ("block", 1)],
)
def test_automatic_frozen_prefix_boundary_matches_eager_for_standard_transformer_block(
    groups, monkeypatch, recompute_method, expected_checkpoints
):
    """The reentrant checkpoint path trains every layer after a frozen prefix.

    Reentrant checkpoints only propagate gradients when an input requires grad, so the
    boundary activation must. With the block method, ``recompute_num_layers`` counts
    trainable layers from the boundary: one layer here, not none.
    """
    from megatron.core import tensor_parallel
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.transformer.transformer_block import TransformerBlock

    common = dict(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        enable_mhc_connections=False,
        mhc_single_pass=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        hidden_dropout=0,
        attention_dropout=0,
        gradient_accumulation_fusion=False,
    )
    recompute = dict(
        recompute_granularity="full", recompute_method=recompute_method, recompute_num_layers=1
    )
    configs = [
        TransformerConfig(**common),
        TransformerConfig(**common),
        TransformerConfig(**common, **recompute),
    ]
    blocks = [
        TransformerBlock(
            config,
            spec=get_transformer_block_with_experimental_attention_variant_spec(config, pp_rank=0),
            pg_collection=groups,
        ).cuda()
        for config in configs
    ]
    for block in blocks[1:]:
        block.load_state_dict(blocks[0].state_dict())
    for block in blocks:
        for layer in block.layers[:2]:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
    reference, eager, recomputed = blocks
    _disable_gradient_boundary(reference)

    checkpoint_calls = []
    checkpoint = tensor_parallel.checkpoint

    def counting_checkpoint(*args, **kwargs):
        checkpoint_calls.append(1)
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(tensor_parallel, "checkpoint", counting_checkpoint)
    input_ref = torch.randn(16, 2, common["hidden_size"], device="cuda", dtype=torch.bfloat16)
    outputs = [block(input_ref.clone(), attention_mask=None) for block in blocks]
    torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
    torch.testing.assert_close(outputs[2], outputs[0], atol=0, rtol=0)
    for output in outputs:
        output.float().square().mean().backward()
    assert len(checkpoint_calls) == expected_checkpoints
    for block in (eager, recomputed):
        assert block._gradient_boundary_layer_cache[False] == 2
    _assert_trainable_grads_match((eager, recomputed), reference, atol=2e-4, rtol=2e-2)


def test_single_pass_selective_recompute_matches_eager(groups):
    """Checkpoint replay must use each microbatch's incoming mix, not a later layer's state."""
    from dataclasses import replace

    config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        hidden_dropout=0,
        attention_dropout=0,
        gradient_accumulation_fusion=False,
    )
    recompute_config = replace(
        config,
        recompute_granularity="selective",
        recompute_modules=["mhc"],
        mhc_recompute_layer_num=2,
    )
    models = [
        HybridModel(
            cfg,
            hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=64,
            hybrid_layer_pattern="*-*-",
            position_embedding_type="none",
            pg_collection=groups,
        ).cuda()
        for cfg in (config, recompute_config)
    ]
    for model in models:
        convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
    models[1].load_state_dict(models[0].state_dict())
    ids = torch.randint(0, 128, (2, 16), device="cuda")
    positions = torch.arange(16, device="cuda").expand_as(ids)
    outputs = [
        [model(tokens, positions, None, labels=ids).mean() for tokens in (ids, ids.roll(1, 1))]
        for model in models
    ]
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for losses in outputs:
        sum(losses).backward()
    for (name, expected), (_, actual) in zip(
        models[0].named_parameters(), models[1].named_parameters()
    ):
        assert (actual.grad is None) == (expected.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-4, rtol=2e-2, msg=name)

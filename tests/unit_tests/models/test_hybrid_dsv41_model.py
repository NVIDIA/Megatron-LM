# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 text-backbone integration through the existing Hybrid model builder.

These tests exercise real attention, mHC, routed/shared experts, and language-model
endpoints. They leave module-level reference math to the CSA2 and mHC test suites.
"""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.csa2 import CompressedSparseAttention2
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Production Hybrid model requires CUDA"
    ),
    pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is not installed"),
]


@pytest.fixture
def pg_collection():
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        Utils.destroy_model_parallel()


def _build_model(pg_collection, dtype):
    # Hybrid numbers attention and MoE sublayers independently. Zero entries on
    # the E sublayers are placeholders, not additional attention modules.
    config = _make_config(
        params_dtype=dtype,
        num_layers=12,
        csa_compress_ratios=[0, 0, 2, 0, 2, 0, 1, 0, 1, 0, 1, 0],
        csa2_kv_source_layers=[2, 6],
        csa2_index_source_layers=[2, 6, 8],
        csa2_candidate_source_layer=6,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0,
        moe_shared_expert_gate=False,
        moe_shared_expert_overlap=False,
        moe_token_dispatcher_type="allgather",
        moe_grouped_gemm=False,
        cross_entropy_loss_fusion=False,
    )
    model_config = HybridModelConfig(
        transformer=config,
        hybrid_stack_spec=hybrid_dsv4_stack_spec(config),
        hybrid_layer_pattern="DE" * 6,
        vocab_size=64,
        seq_length=16,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
        parallel_output=False,
    )
    bare_model = HybridModelBuilder(model_config).build_model(pg_collection).cuda()
    # Use the training wrapper, which preserves the FP32-marked compressor,
    # attention sink, and mHC tensors. A blanket .bfloat16() would change them.
    model = Float16Module(config, bare_model) if dtype == torch.bfloat16 else bare_model
    model.train()
    return model, bare_model


def _batch(length):
    tokens = torch.arange(2 * (length + 1), device="cuda").view(2, length + 1) % 64
    input_ids = tokens[:, :-1].contiguous()
    labels = tokens[:, 1:].contiguous()
    position_ids = torch.arange(length, device="cuda").expand_as(input_ids)
    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    loss_mask[:, 1::3] = 0
    return dict(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        loss_mask=loss_mask,
    )


def _masked_loss(token_losses, loss_mask):
    # HybridModel returns per-token CE. The training loss function owns masking
    # and reduction; the model's loss_mask argument also serves the MTP path.
    return (token_losses.float() * loss_mask).sum() / loss_mask.sum()


def _assert_live_gradient(parameter, name):
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name
    assert parameter.grad.float().abs().sum() > 0, name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hybrid_dsv41_logits_masked_ce_and_backbone_gradients(pg_collection, dtype):
    """The configured DE stack reaches both LM endpoints and every trained branch."""
    model, bare_model = _build_model(pg_collection, dtype)
    assert bare_model.pg_collection is pg_collection
    assert bare_model.embedding.word_embeddings.weight is not bare_model.output_layer.weight
    assert bare_model.embedding.word_embeddings.weight.dtype == dtype
    assert bare_model.output_layer.weight.dtype == dtype
    assert len(bare_model.decoder.layers) == 12
    assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in bare_model.decoder.layers)
    assert not any("hc_head_" in name for name, _ in bare_model.named_parameters())

    marked_parameters = [
        parameter
        for parameter in bare_model.parameters()
        if getattr(parameter, "keep_in_fp32", False)
    ]
    assert marked_parameters
    assert all(parameter.dtype == torch.float32 for parameter in marked_parameters)

    attention_layers = [
        layer.inner_layer.self_attention for layer in bare_model.decoder.layers[::2]
    ]
    moe_layers = [layer.inner_layer.mlp for layer in bare_model.decoder.layers[1::2]]
    assert all(
        isinstance(layer.core_attention, CompressedSparseAttention2) for layer in attention_layers
    )
    assert all(isinstance(layer, MoELayer) for layer in moe_layers)
    assert all(isinstance(layer.experts, SequentialMLP) for layer in moe_layers)
    assert all(isinstance(layer.shared_experts, SharedExpertMLP) for layer in moe_layers)

    batch = _batch(7)
    with torch.no_grad():
        logits = model(**{key: value for key, value in batch.items() if key != "labels"})
    assert logits.shape == (2, 7, 64)
    assert logits.dtype == torch.float32
    expected_losses = F.cross_entropy(
        logits.flatten(0, 1), batch["labels"].flatten(), reduction="none"
    ).view_as(batch["labels"])

    selected_experts = {}

    def record_routing(router, inputs, output):
        # Router flattens [sequence, batch]. Experts used only by masked final
        # tokens can correctly receive zero gradient, so require supervised use.
        supervised_tokens = batch["loss_mask"].T.reshape(-1).bool()
        selected_experts[router] = output[1].detach()[supervised_tokens].any(dim=0)

    handles = [layer.router.register_forward_hook(record_routing) for layer in moe_layers]
    try:
        token_losses = model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    assert token_losses.shape == batch["labels"].shape
    tolerance = 2e-4 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(token_losses, expected_losses, atol=tolerance, rtol=tolerance)
    token_losses.retain_grad()
    loss = _masked_loss(token_losses, batch["loss_mask"])
    torch.testing.assert_close(
        loss, expected_losses[batch["loss_mask"].bool()].mean(), atol=tolerance, rtol=tolerance
    )
    loss.backward()
    torch.testing.assert_close(
        token_losses.grad, batch["loss_mask"] / batch["loss_mask"].sum(), atol=0, rtol=0
    )

    _assert_live_gradient(bare_model.embedding.word_embeddings.weight, "embedding")
    _assert_live_gradient(bare_model.output_layer.weight, "LM head")
    _assert_live_gradient(bare_model.decoder.final_norm.weight, "final norm")
    for index, layer in enumerate(bare_model.decoder.layers):
        _assert_live_gradient(layer.hyper_connection.mapping_proj.weight, f"mHC {index}")
    for index, attention in enumerate(attention_layers):
        # Checking the main query projection reaches all attention modes without
        # requiring an auxiliary indexer loss in this backbone integration step.
        _assert_live_gradient(attention.linear_q_up_proj.weight, f"attention Q {index}")
        indexer = attention.core_attention.indexer
        if indexer is not None:
            assert all(parameter.grad is None for parameter in indexer.parameters())
        compressor = attention.core_attention.compressor
        if compressor is not None:
            _assert_live_gradient(compressor.linear_wkv.weight, f"global KV owner {index}")
            if compressor.linear_wgate is not None:
                _assert_live_gradient(compressor.linear_wgate.weight, f"KV compressor gate {index}")
    for index, moe in enumerate(moe_layers):
        _assert_live_gradient(moe.router.weight, f"router {index}")
        _assert_live_gradient(moe.shared_experts.linear_fc1.weight, f"shared FC1 {index}")
        _assert_live_gradient(moe.shared_experts.linear_fc2.weight, f"shared FC2 {index}")
        selected = selected_experts[moe.router].nonzero().flatten().tolist()
        assert selected
        for expert_index in selected:
            expert = moe.experts.local_experts[expert_index]
            _assert_live_gradient(expert.linear_fc1.weight, f"expert FC1 {index}/{expert_index}")
            _assert_live_gradient(expert.linear_fc2.weight, f"expert FC2 {index}/{expert_index}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hybrid_dsv41_live_forwards_and_state_dict_roundtrip(pg_collection, dtype):
    """Restored serial execution matches two live graphs backpropagated in reverse."""
    model, bare_model = _build_model(pg_collection, dtype)
    restored, restored_bare = _build_model(pg_collection, dtype)
    incompatible = restored_bare.load_state_dict(deepcopy(bare_model.state_dict()), strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    batches = [_batch(5), _batch(7)]

    # Both CSA2 sharing and Single-Pass mHC must keep the earlier graph intact.
    outputs = [model(**batch) for batch in batches]
    forward_tolerance = 2e-4 if dtype == torch.bfloat16 else 2e-6
    for index in (1, 0):
        expected = restored(**batches[index])
        torch.testing.assert_close(
            outputs[index], expected, atol=forward_tolerance, rtol=forward_tolerance
        )
        _masked_loss(expected, batches[index]["loss_mask"]).backward()
        _masked_loss(outputs[index], batches[index]["loss_mask"]).backward()

    restored_parameters = dict(restored_bare.named_parameters())
    tolerance = dict(atol=2e-6, rtol=2e-5)
    if dtype == torch.bfloat16:
        tolerance = dict(atol=2e-5, rtol=2e-2)
    for name, parameter in bare_model.named_parameters():
        expected_gradient = restored_parameters[name].grad
        if expected_gradient is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected_gradient, **tolerance, msg=name)

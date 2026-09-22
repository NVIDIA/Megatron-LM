# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint regressions for repeated MTP sharing."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
    get_transformer_layer_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.forward_sharing import (
    forward_sharing_lifetime,
    get_forward_sharing_state,
    mtp_repeated_sharing_lifetime,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
    BOTH_SHARED_COMPONENTS,
    LATENT_KV,
    SPARSE_ATTENTION_INDEX,
    _causal_segment_mask,
    _make_config,
)


@pytest.fixture
def cpu_checkpoint_rng(monkeypatch):
    """Keep real checkpoint/autograd, replacing only CUDA RNG access for CPU tensors."""
    monkeypatch.setattr(
        tensor_parallel.random, "_get_all_rng_states", lambda: (torch.get_rng_state(),)
    )
    monkeypatch.setattr(tensor_parallel.random, "_set_all_rng_states", torch.set_rng_state)


@pytest.mark.parametrize(
    "components", [[LATENT_KV], [SPARSE_ATTENTION_INDEX], BOTH_SHARED_COMPONENTS]
)
@pytest.mark.parametrize("source_has_lengths", [False, True])
@pytest.mark.parametrize("carrier", ["config", "mask", "packed"])
@pytest.mark.parametrize("consumer_only", [False, True])
@pytest.mark.parametrize("recompute_method", ["uniform", "block"])
def test_full_checkpoint_preserves_sharing_gradients_and_per_depth_snapshots(
    monkeypatch,
    cpu_checkpoint_rng,
    components,
    source_has_lengths,
    carrier,
    consumer_only,
    recompute_method,
):
    """Consumer KV gradients traverse source checkpoint outputs, not captured Python tensors."""
    monkeypatch.setattr(dsa_module, "build_module", lambda *_a, **_k: nn.Identity())
    config = _make_config(
        num_layers=3,
        mtp_num_layers=2,
        mtp_repeated_layer_shared_components=components,
        recompute_granularity="full",
        recompute_method=recompute_method,
        recompute_num_layers=1,
    )
    core = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(),
        is_mtp_layer=True,
    )
    mask = torch.zeros(1) if carrier == "mask" else None
    packed = SimpleNamespace() if carrier == "packed" else None
    weight = nn.Parameter(torch.tensor(1.25))
    inputs = [torch.randn(4, 1, 3, requires_grad=True) for _ in range(2)]
    decoder_inputs = [torch.randn_like(value) for value in inputs]
    seen = {True: [], False: []}

    def projected_forward(hidden_states, decoder_input, attention_mask=None, **_kwargs):
        state = get_forward_sharing_state(packed, attention_mask, config)
        is_source, payload = core._prepare_mtp_sharing_payload(packed, attention_mask)
        if is_source or SPARSE_ATTENTION_INDEX not in components:
            topk = state.get(dsa_module._DSAIndexSharingPayload).topk_by_layer[core.source_layer]
        else:
            topk = payload.topk_by_layer[core.layer_number]
        seen[is_source].append(topk.clone())
        combined = hidden_states + decoder_input
        if is_source:
            if LATENT_KV in components:
                payload.latent_kv_by_layer[core.layer_number] = combined * weight
            if SPARSE_ATTENTION_INDEX in components:
                payload.topk_by_layer[core.layer_number] = topk
                if source_has_lengths:
                    payload.topk_length_by_layer[core.layer_number] = torch.ones(
                        4, dtype=torch.int32
                    )
            result = combined.square()
        else:
            result = combined
            if LATENT_KV in components:
                result = result + 3 * payload.latent_kv_by_layer[core.layer_number]
        return result + topk.sum().to(result.dtype)

    layer = SimpleNamespace(
        config=config,
        mtp_layer_pattern=None,
        mtp_model_layer=SimpleNamespace(self_attention=SimpleNamespace(core_attention=core)),
        _proj_and_transformer_layer=projected_forward,
    )
    outputs = []
    with forward_sharing_lifetime(packed, mask, config) as state:
        with mtp_repeated_sharing_lifetime(state, core.layer_number) as flags:
            ordinary = state.get_or_create(dsa_module._DSAIndexSharingPayload)
            for depth in range(2):
                flags[core.layer_number] = depth == 0
                ordinary.topk_by_layer[core.source_layer] = torch.full(
                    (4,), depth, dtype=torch.int64
                )
                outputs.append(
                    MultiTokenPredictionLayer._checkpointed_forward(
                        layer,
                        inputs[depth],
                        decoder_inputs[depth],
                        attention_mask=mask,
                        packed_seq_params=packed,
                    )
                )
            exported = core.get_mtp_checkpoint_tensors(packed, mask)
            if SPARSE_ATTENTION_INDEX in components:
                assert (exported[-1].numel() > 0) is source_has_lengths
            if LATENT_KV in components:
                assert exported[0].grad_fn is not None
    assert state._entries == {}
    # Another forward on the same carrier cannot overwrite the saved flags or indices.
    with forward_sharing_lifetime(packed, mask, config):
        state.get_or_create(dsa_module._DSAIndexSharingPayload).topk_by_layer[core.source_layer] = (
            torch.full((4,), 99)
        )
    loss = outputs[1].sum() if consumer_only else sum(value.sum() for value in outputs)
    loss.backward()

    reference_weight = nn.Parameter(weight.detach().clone())
    reference_inputs = [value.detach().clone().requires_grad_(True) for value in inputs]
    combined = reference_inputs[0] + decoder_inputs[0]
    consumer_topk_sum = 0 if SPARSE_ATTENTION_INDEX in components else 4
    expected_outputs = [
        combined.square(),
        reference_inputs[1] + decoder_inputs[1] + consumer_topk_sum,
    ]
    if LATENT_KV in components:
        expected_outputs[1] = expected_outputs[1] + 3 * combined * reference_weight
    expected_loss = (
        expected_outputs[1].sum()
        if consumer_only
        else sum(value.sum() for value in expected_outputs)
    )
    expected_loss.backward()
    for actual, expected in zip(outputs, expected_outputs):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(inputs, reference_inputs):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad)
    if LATENT_KV in components:
        torch.testing.assert_close(weight.grad, reference_weight.grad)
    else:
        assert weight.grad is None
    assert all(torch.equal(indices, torch.zeros(4, dtype=torch.int64)) for indices in seen[True])
    expected_index = 0 if SPARSE_ATTENTION_INDEX in components else 1
    assert len(seen[False]) == (2 if recompute_method == "uniform" else 1)
    assert all(
        torch.equal(indices, torch.full((4,), expected_index, dtype=torch.int64))
        for indices in seen[False]
    )
    assert state._entries == {}


def test_selective_core_checkpoint_uses_private_source_and_consumer_snapshots(
    monkeypatch, cpu_checkpoint_rng
):
    """The existing snapshot wrapper restores flags and ordinary indices after forward cleanup."""
    monkeypatch.setattr(dsa_module, "build_module", lambda *_a, **_k: nn.Identity())
    config = _make_config(
        num_layers=3, recompute_granularity="selective", recompute_modules=["core_attn"]
    )
    core = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(),
        is_mtp_layer=True,
    )
    seen = []
    mask = torch.zeros(1)

    def attention_call(q, k, attention_mask=None, **_kwargs):
        state = get_forward_sharing_state(None, attention_mask, config)
        is_source, payload = core._prepare_mtp_sharing_payload(None, attention_mask)
        if is_source:
            payload.topk_by_layer[core.layer_number] = state.get(
                dsa_module._DSAIndexSharingPayload
            ).topk_by_layer[core.source_layer]
        topk = payload.topk_by_layer[core.layer_number]
        seen.append(topk.clone())
        return q * k + topk.sum()

    attention = SimpleNamespace(
        config=config, core_attention=attention_call, attn_mask_type=AttnMaskType.causal
    )
    inputs = [torch.randn(2, requires_grad=True) for _ in range(2)]
    outputs = []
    with forward_sharing_lifetime(None, mask, config) as state:
        with mtp_repeated_sharing_lifetime(state, core.layer_number) as flags:
            ordinary = state.get_or_create(dsa_module._DSAIndexSharingPayload)
            ordinary.topk_by_layer[core.source_layer] = torch.zeros(2, dtype=torch.int64)
            for depth in range(2):
                flags[core.layer_number] = depth == 0
                outputs.append(
                    AbsorbedMLASelfAttention._checkpointed_attention_forward(
                        attention,
                        inputs[depth],
                        inputs[depth],
                        inputs[depth],
                        inputs[depth],
                        mask,
                        torch.ones(1),
                    )
                )
                ordinary.topk_by_layer[core.source_layer] = torch.ones(2, dtype=torch.int64)
    sum(value.sum() for value in outputs).backward()
    assert len(seen) == 4 and all(torch.count_nonzero(value) == 0 for value in seen)
    for value in inputs:
        torch.testing.assert_close(value.grad, 2 * value.detach())
    assert state._entries == {}


@pytest.mark.parametrize("shares_latent_kv", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("q_lora_rank", [None, 16])
def test_mla_projection_checkpoint_keeps_shared_kv_storage(
    monkeypatch, cpu_checkpoint_rng, shares_latent_kv, packed, q_lora_rank
):
    """Exercise real projection/checkpoint storage handling with CPU linears and identity RoPE."""
    from megatron.core.transformer.experimental_attention_variant import absorbed_mla as mla_module

    config = _make_config(q_lora_rank=q_lora_rank, apply_rope_fusion=False)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    monkeypatch.setattr(mla_module, "apply_rotary_pos_emb", lambda tensor, *_a, **_k: tensor)
    monkeypatch.setattr(mla_module, "should_use_fused_mla_rope", lambda _config: False)

    class Linear(nn.Linear):
        def forward(self, value):
            return super().forward(value), None

    class Rotary:
        def get_rotary_seq_len(self, *_args):
            return 4

        def __call__(self, *_args, **_kwargs):
            return torch.zeros(4, 1, 1, 4)

    config.rope_type = "rope"
    attention = SimpleNamespace(
        config=config,
        rotary_pos_emb=Rotary(),
        tp_group=group,
        pg_collection=SimpleNamespace(cp=group),
        num_attention_heads_per_partition=4,
        q_head_dim=12,
        recompute_up_proj=True,
        cache_mla_latents=False,
        core_attention=SimpleNamespace(mtp_latent_kv_share=shares_latent_kv),
        linear_q_down_proj=Linear(64, 16, bias=False),
        linear_q_up_proj=Linear(16, 48, bias=False),
        linear_q_proj=Linear(64, 48, bias=False),
        linear_kv_down_proj=Linear(64, 20, bias=False),
        q_layernorm=nn.Identity(),
        kv_layernorm=nn.Identity(),
        _get_kv_up_weights=lambda: (torch.ones(4, 8, 16), None),
    )
    params = (
        SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            cu_seqlens_q=torch.tensor([0, 4]),
            cu_seqlens_kv=torch.tensor([0, 4]),
            max_seqlen_q=4,
            max_seqlen_kv=4,
        )
        if packed
        else None
    )
    hidden = torch.randn(4, 1, 64, requires_grad=True)
    query, key, _ = AbsorbedMLASelfAttention.get_query_key_value_tensors(
        attention, hidden, packed_seq_params=params
    )
    assert len(attention.qkv_up_checkpoint.outputs) == (1 if shares_latent_kv else 2)
    checkpoint = attention.qkv_up_checkpoint
    source_key = key
    output = query.square().sum() + key.square().sum()
    checkpoint.discard_output_and_register_recompute(output)
    assert query.untyped_storage().nbytes() == 0
    if not shares_latent_kv:
        assert key.untyped_storage().nbytes() == 0
        return

    assert source_key.untyped_storage().nbytes() > 0
    assert source_key.grad_fn is not None
    consumer_hidden = torch.randn_like(hidden, requires_grad=True)
    _, consumer_key, _ = AbsorbedMLASelfAttention.get_query_key_value_tensors(
        attention, consumer_hidden, packed_seq_params=params, reuse_mtp_latent_kv=True
    )
    assert consumer_key is None
    # A loss depending only on consumer KV still trains the source projection.
    source_key.square().sum().backward()
    assert hidden.grad is not None and hidden.grad.norm() > 0
    assert attention.linear_kv_down_proj.weight.grad.norm() > 0


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    ("shared_components", "shares_latent_kv"),
    [
        (None, False),
        ([LATENT_KV], True),
        ([SPARSE_ATTENTION_INDEX], False),
        (BOTH_SHARED_COMPONENTS, True),
    ],
)
def test_selective_mla_up_projection_preserves_shared_latent_kv_gradient(
    monkeypatch, shared_components, shares_latent_kv
):
    """Latent-KV sharing checkpoints Q only and keeps producer K graph-connected."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1122)
        config = _make_config(
            mtp_repeated_layer_shared_components=shared_components,
            recompute_granularity="selective",
            recompute_modules=["mla_up_proj"],
        )
        attention_spec = get_dsa_module_spec_for_backend(config, backend=TESpecProvider())
        attention = (
            build_module(
                attention_spec, config=config, layer_number=1, is_mtp_layer=bool(shared_components)
            )
            .bfloat16()
            .cuda()
        )

        checkpoint_output_arities = []
        original_checkpoint = CheckpointWithoutOutput.checkpoint

        def observed_checkpoint(checkpoint, run_function, *args):
            outputs = original_checkpoint(checkpoint, run_function, *args)
            checkpoint_output_arities.append(
                1 if isinstance(outputs, torch.Tensor) else len(outputs)
            )
            return outputs

        monkeypatch.setattr(CheckpointWithoutOutput, "checkpoint", observed_checkpoint)

        total_tokens = 9
        valid = _causal_segment_mask(total_tokens, None, device="cuda")
        attention_mask = torch.zeros(
            (1, 1, total_tokens, total_tokens), dtype=torch.float32, device="cuda"
        ).masked_fill(~valid.view(1, 1, total_tokens, total_tokens), float("-inf"))
        source_hidden = torch.randn(
            total_tokens,
            1,
            config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )

        if shared_components is None:
            output, _ = attention(source_hidden, attention_mask=attention_mask)
            output.float().square().mean().backward()
            expected_checkpoint_count = 1
        else:
            with forward_sharing_lifetime(attention_mask=attention_mask, config=config):
                state = get_forward_sharing_state(attention_mask=attention_mask, config=config)
                layer_number = attention.core_attention.layer_number
                with mtp_repeated_sharing_lifetime(state, layer_number) as flags:
                    flags[layer_number] = True
                    attention(source_hidden, attention_mask=attention_mask)
                    payload = state.get(dsa_module._DSAMTPRepeatedSharingPayload)
                    shared_key = payload.latent_kv_by_layer.get(layer_number)
                    consumer_hidden = torch.randn_like(source_hidden, requires_grad=True)
                    flags[layer_number] = False
                    consumer_output, _ = attention(consumer_hidden, attention_mask=attention_mask)
            consumer_output.float().square().mean().backward()
            expected_checkpoint_count = 2

            assert consumer_hidden.grad is not None
            assert torch.isfinite(consumer_hidden.grad).all()
            if shares_latent_kv:
                assert shared_key.untyped_storage().nbytes() > 0
                assert shared_key.grad_fn is not None
                assert source_hidden.grad is not None
                assert source_hidden.grad.float().norm() > 0

        expected_arity = 1 if shares_latent_kv else 2
        assert checkpoint_output_arities == [expected_arity] * expected_checkpoint_count
        if shared_components is None:
            assert source_hidden.grad is not None
        if source_hidden.grad is not None:
            assert torch.isfinite(source_hidden.grad).all()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.launch_on_gb200
def test_real_mtp_block_full_recompute_replays_computing_indexer_loss(monkeypatch):
    """Full MTP recompute reruns the source DSA indexer with gradients enabled."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(5678)
        torch.manual_seed(5678)
        torch.cuda.manual_seed(5678)

        config = _make_config(
            mtp_num_layers=2,
            dsa_indexer_loss_coeff=0.1,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        decoder_layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(
            config=config, backend=TESpecProvider()
        )
        mtp_spec = get_gpt_mtp_block_spec(
            config=config, spec=decoder_layer_specs[-1], use_transformer_engine=True
        )
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_spec).bfloat16().cuda()
        attention = mtp.layers[0].mtp_model_layer.self_attention
        assert not attention.core_attention.skip_topk

        indexer_grad_modes = []
        indexer_hook = attention.core_attention.indexer.linear_wq_b.register_forward_pre_hook(
            lambda *_args: indexer_grad_modes.append(torch.is_grad_enabled())
        )
        tracked_losses = []
        monkeypatch.setattr(
            dsa_module.DSAIndexerLossLoggingHelper,
            "save_loss_to_tracker",
            staticmethod(lambda **kwargs: tracked_losses.append(kwargs["loss"].detach().clone())),
        )

        total_tokens = 9
        embedding = nn.Embedding(32, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        def embed(input_ids, position_ids):
            del position_ids
            return embedding(input_ids).transpose(0, 1).contiguous()

        input_ids = torch.arange(total_tokens, device="cuda").view(1, total_tokens)
        position_ids = input_ids.clone()
        hidden_states = torch.randn(
            total_tokens,
            1,
            config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        valid = _causal_segment_mask(total_tokens, None, device="cuda")
        attention_mask = torch.zeros(
            (1, 1, total_tokens, total_tokens), dtype=torch.float32, device="cuda"
        ).masked_fill(~valid.view(1, 1, total_tokens, total_tokens), float("-inf"))

        output = mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            embedding=embed,
        )
        assert indexer_grad_modes == [False]
        assert tracked_losses == []

        output[total_tokens:].float().square().mean().backward()

        assert indexer_grad_modes == [False, True]
        assert len(tracked_losses) == 1
        assert torch.isfinite(tracked_losses[0])
        assert tracked_losses[0] > 0
        assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
        assert embedding.weight.grad is not None and torch.isfinite(embedding.weight.grad).all()
        for name in ("linear_wq_b.weight", "linear_wk.weight", "linear_weights_proj.weight"):
            parameter = dict(attention.core_attention.indexer.named_parameters())[name]
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.float().norm() > 0
        indexer_hook.remove()
    finally:
        Utils.destroy_model_parallel()

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Exact canonical GDP mixer parity through real Megatron execution paths."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import (
    GatedDeltaProductMixer,
    GatedDeltaProductMixerSubmodules,
)
from megatron.core.tensor_parallel.inference_layers import (
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), (
        (a.float() - b.float()).abs().max().item()
    )


@pytest.fixture
def canonical_mixer():
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    enable_batch_invariant_mode(backend='triton')
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        normalization='RMSNorm',
        bf16=True,
        params_dtype=torch.bfloat16,
        mamba_num_heads=4,
        mamba_num_groups=4,
        mamba_head_dim=16,
        mamba_state_dim=32,
        gdp_num_householder=3,
        is_hybrid_model=True,
        batch_invariant_mode=True,
        batch_invariant_backend='triton',
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    pg = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    mixer = (
        GatedDeltaProductMixer(
            cfg,
            GatedDeltaProductMixerSubmodules(
                in_proj=InferenceLayerNormColumnParallelLinear, out_proj=InferenceRowParallelLinear
            ),
            64,
            layer_number=1,
            pg_collection=pg,
        )
        .cuda()
        .bfloat16()
    )
    yield mixer
    disable_batch_invariant_mode()
    Utils.destroy_model_parallel()


def test_training_static_decode(canonical_mixer):
    mixer = canonical_mixer
    x = torch.randn(23, 2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    mixer.train()
    expected = mixer(x)[0]
    expected.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mixer.parameters())
    mixer.eval()
    context = StaticInferenceContext(max_batch_size=2, max_sequence_length=128)
    with torch.no_grad():
        parts = [mixer(x[:5].detach(), inference_context=context)[0]]
        context.sequence_len_offset = 5
        for token in range(5, 23):
            parts.append(mixer(x[token : token + 1].detach(), inference_context=context)[0])
            context.sequence_len_offset += 1
        exact(expected, torch.cat(parts))
        isolated = StaticInferenceContext(max_batch_size=1, max_sequence_length=128)
        alone = mixer(x[:, 1:2].detach(), inference_context=isolated)[0]
        exact(expected[:, 1:2], alone)


def test_packed_training_and_serving(canonical_mixer):
    from types import SimpleNamespace

    from megatron.core.packed_seq_params import PackedSeqParams

    mixer = canonical_mixer
    x = torch.randn(23, 1, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, 5, 23], device='cuda', dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=18, max_seqlen_kv=18
    )
    mixer.train()
    expected = mixer(x, packed_seq_params=params)[0]
    expected.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()
    with torch.no_grad():
        separate = torch.cat([mixer(x[:5].detach())[0], mixer(x[5:].detach())[0]])
        exact(expected, separate)
        mixer.eval()
        conv, state = mixer.allocate_inference_cache(3, 128)
        slots = torch.tensor([2, 0], device='cuda', dtype=torch.int32)
        context = SimpleNamespace(
            mamba_slot_allocator=None,
            mamba_metadata=SimpleNamespace(cu_seqlens=cu, batch_indices_prefill=slots),
        )
        projected = mixer.in_proj(x.detach())[0]
        actual = mixer.out_proj(mixer.ssm_prefill(projected, conv, state, context))[0]
        exact(expected, actual)
        next_x = torch.randn(1, 2, 64, device='cuda', dtype=torch.bfloat16)
        next_projected = mixer.in_proj(next_x)[0].transpose(0, 1)
        decoded = mixer.out_proj(
            mixer.ssm_decode(next_projected, conv, state, batch_indices=slots).transpose(0, 1)
        )[0]
        separate_next = torch.cat(
            [
                mixer(torch.cat([x[:5].detach(), next_x[:, :1]]))[0][-1:],
                mixer(torch.cat([x[5:].detach(), next_x[:, 1:]]))[0][-1:],
            ],
            1,
        )
        exact(decoded, separate_next)


def build_hybrid(canonical_mixer, pattern):
    from dataclasses import replace

    from megatron.core.inference.utils import InferenceMode
    from megatron.core.models.hybrid.hybrid_layer_specs import (
        gated_delta_product_inference_stack_spec,
    )
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    cfg = replace(
        canonical_mixer.config,
        num_layers=3,
        hidden_size=128,
        num_attention_heads=2,
        kv_channels=64,
        num_query_groups=2,
        ffn_hidden_size=256,
    )
    return HybridModel(
        config=cfg,
        hybrid_stack_spec=gated_delta_product_inference_stack_spec,
        vocab_size=128,
        max_sequence_length=128,
        hybrid_layer_pattern=pattern,
        logit_dtype=torch.float32,
    ).cuda()


@pytest.mark.parametrize("pattern", ["M-M", "M*-"])
def test_hybrid_logits_and_raw_logprobs(canonical_mixer, pattern):
    from megatron.core.inference.utils import InferenceMode

    model = build_hybrid(canonical_mixer, pattern)
    tokens = torch.randint(0, 128, (2, 23), device='cuda')
    positions = torch.arange(23, device='cuda').expand(2, -1)
    model.train()
    expected = model(tokens, positions, None, runtime_gather_output=True)
    expected.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    context = StaticInferenceContext(max_batch_size=2, max_sequence_length=128)
    with torch.inference_mode(), InferenceMode.active():
        parts = [
            model(
                tokens[:, :5],
                positions[:, :5],
                None,
                inference_context=context,
                runtime_gather_output=True,
            )
        ]
        context.sequence_len_offset = 5
        for token in range(5, 23):
            parts.append(
                model(
                    tokens[:, token : token + 1],
                    positions[:, token : token + 1],
                    None,
                    inference_context=context,
                    runtime_gather_output=True,
                )
            )
            context.sequence_len_offset += 1
        actual = torch.cat(parts, 1)
        # Static prefill exposes its final token, followed by every decoded token.
        exact(actual, expected[:, 4:])
        lp = torch.log_softmax(actual.float(), -1).gather(-1, tokens[:, 4:, None]).squeeze(-1)
        reference = (
            torch.log_softmax(expected[:, 4:].float(), -1)
            .gather(-1, tokens[:, 4:, None])
            .squeeze(-1)
        )
        assert torch.isfinite(lp).all()
        exact(lp, reference)


def test_hybrid_dynamic_packed_prefill_and_graph(canonical_mixer):
    # The engine requires an attention layer to allocate its paged KV cache.
    pattern = "M*-"
    from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
    from megatron.core.inference.contexts import DynamicInferenceContext
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams
    from megatron.core.inference.utils import InferenceMode

    model = build_hybrid(canonical_mixer, pattern)
    prompts = [torch.randint(0, 128, (1, t), device="cuda") for t in (5, 19)]
    continuations = torch.randint(0, 128, (2, 18), device="cuda")
    model.train()
    with torch.no_grad():
        complete = [torch.cat([p, continuations[i : i + 1]], 1) for i, p in enumerate(prompts)]
        complete_logits = [
            model(
                p, torch.arange(p.shape[1], device="cuda")[None], None, runtime_gather_output=True
            )
            for p in complete
        ]
        expected = torch.cat(
            [
                model(
                    p,
                    torch.arange(p.shape[1], device="cuda")[None],
                    None,
                    runtime_gather_output=True,
                )
                for p in prompts
            ],
            1,
        )
    model.eval()
    ctx = DynamicInferenceContext(
        model_config=model.config,
        inference_config=InferenceConfig(
            max_sequence_length=128,
            buffer_size_gb=0.1,
            block_size_tokens=256,
            materialize_only_last_token_logits=False,
            mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
            num_cuda_graphs=0,
            use_cuda_graphs_for_non_decode_steps=False,
            max_requests=4,
            max_tokens=128,
            enable_prefix_caching=False,
        ),
    )
    for i, prompt in enumerate(prompts):
        ctx.add_request(
            DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=prompt[0].cpu(),
                sampling_params=SamplingParams(num_tokens_to_generate=32, termination_id=-1),
            )
        )
    ctx.initialize_attention_state()
    ids, positions = ctx.current_input_and_position_ids()
    initial = (ctx.mamba_conv_states.clone(), ctx.mamba_ssm_states.clone())

    def restore():
        ctx.mamba_conv_states.copy_(initial[0])
        ctx.mamba_ssm_states.copy_(initial[1])

    def forward():
        return model(ids, positions, None, inference_context=ctx, runtime_gather_output=True)

    with torch.inference_mode(), InferenceMode.active():
        actual = forward()
        exact(actual[:, :24], expected)
        restore()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = forward()
        restore()
        graph.replay()
        torch.cuda.synchronize()
        exact(replayed[:, :24], expected)
        exact(
            torch.log_softmax(replayed[:, :24].float(), -1), torch.log_softmax(expected.float(), -1)
        )
        # Exercise the real scheduler transition, paged KV cache, and GDP slots
        # across unequal sequence positions and several internal block boundaries.
        for step in range(continuations.shape[1]):
            ctx.update_requests(
                active_requests_mask=torch.ones(2, dtype=torch.int32),
                new_tokens=continuations[:, step],
            )
            ctx.initialize_attention_state()
            ids, positions = ctx.current_input_and_position_ids()
            decoded = forward()[:, :2]
            reference = torch.cat(
                [
                    logits[:, p.shape[1] + step : p.shape[1] + step + 1]
                    for p, logits in zip(prompts, complete_logits)
                ],
                1,
            )
            exact(decoded, reference)
            exact(torch.log_softmax(decoded.float(), -1), torch.log_softmax(reference.float(), -1))

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Paged CuTe decode kernels and mixed-batch DSA-GQA inference parity."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import DSGroupedSelfAttention
from megatron.core.transformer.experimental_attention_variant.dsa_gqa_decode_kernel import (
    HAVE_CUTEDSL,
    dsa_gqa_decode_attention,
    dsa_gqa_decode_indexer_score,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not HAVE_CUTEDSL, reason="CuTeDSL is required")


@pytest.fixture(scope="module", autouse=True)
def initialize_decode_test_distributed():
    Utils.initialize_distributed()


@pytest.mark.parametrize("head_dim", [128, 192, 256])
def test_paged_decode_kernels_match_reference(head_dim):
    torch.manual_seed(17)
    device = torch.device("cuda", torch.cuda.current_device())
    batch, pages_per_request, page_size, heads = 4, 3, 64, 4
    lengths = torch.tensor([3, 63, 65, 137], dtype=torch.int32, device=device)
    block_table = (
        torch.randperm(batch * pages_per_request, device=device)
        .reshape(batch, pages_per_request)
        .to(torch.int32)
    )
    keys = torch.randn(
        batch * pages_per_request, page_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    values = torch.randn_like(keys)
    index_q = torch.randn(batch, head_dim, device=device, dtype=torch.bfloat16)
    scores = torch.empty(batch, 151, device=device, dtype=torch.float32)
    dsa_gqa_decode_indexer_score(index_q, keys[:, :, 0], block_table, lengths, out=scores)
    expected_scores = torch.full_like(scores, -torch.inf)
    for request, length in enumerate(lengths.tolist()):
        positions = torch.arange(length, device=device)
        k = keys[block_table[request, positions // page_size].long(), positions % page_size, 0]
        expected_scores[request, :length] = k.float() @ index_q[request].float()
    torch.testing.assert_close(scores, expected_scores, atol=2e-4, rtol=2e-4)

    indices = scores.topk(16, dim=-1).indices.to(torch.int32)
    # Attention must accept a query view with the pitched token stride produced by QKV.
    q = torch.randn(batch, heads + 2, head_dim, device=device, dtype=torch.bfloat16)[:, :heads]
    output = torch.empty(batch, heads, head_dim, device=device, dtype=torch.bfloat16)
    scale = head_dim**-0.5
    dsa_gqa_decode_attention(q, keys, values, block_table, indices, lengths, scale, out=output)
    expected = torch.empty_like(output)
    for request, length in enumerate(lengths.tolist()):
        selected = indices[request, : min(length, indices.size(1))].long()
        pages = block_table[request, selected // page_size].long()
        k = keys[pages, selected % page_size, 0]
        v = values[pages, selected % page_size, 0]
        probs = (q[request].float() @ k.float().T * scale).softmax(-1).to(torch.bfloat16)
        expected[request] = (probs.float() @ v.float()).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


def _dynamic_packed_seq_params(inference_context: DynamicInferenceContext) -> PackedSeqParams:
    request_count = inference_context.get_active_request_count()
    cu_seqlens_q, max_seqlen_q = inference_context.cu_query_lengths()
    cu_seqlens_kv, _, max_seqlen_kv = inference_context.cu_kv_lengths()
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q[: request_count + 1],
        cu_seqlens_kv=cu_seqlens_kv[: request_count + 1],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        total_tokens=inference_context.active_token_count,
    )


class TestDSGQADynamicInference:
    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("fused", [False, True])
    def test_dynamic_query_rope_uses_token_positions_with_padding(self, packed, fused):
        config = TransformerConfig(
            num_layers=1, hidden_size=128, num_attention_heads=2, apply_rope_fusion=fused
        )
        device = torch.device("cuda", torch.cuda.current_device())
        query = torch.randn(8, 1, 2, 64, device=device, dtype=torch.bfloat16)
        # Two decodes followed by a three-token prefill and three padding tokens.
        positions = torch.tensor([13, 9, 0, 1, 2, 0, 0, 0], device=device)
        cu_lengths = torch.tensor([0, 1, 2, 5], device=device, dtype=torch.int32)
        cp_group = parallel_state.get_context_parallel_group()
        rope = RotaryEmbedding(64, rotary_percent=1.0, cp_group=cp_group)
        frequencies = rope(16)
        selected = frequencies[positions]
        first, second = query.float().chunk(2, dim=-1)
        expected = (
            query.float() * selected.cos() + torch.cat((-second, first), dim=-1) * selected.sin()
        ).to(torch.bfloat16)
        context = SimpleNamespace(
            padded_active_token_count=8, gpu_view=SimpleNamespace(token_to_pos_ids=positions)
        )
        actual = DynamicInferenceContext.apply_rotary_emb_query(
            context,
            query.squeeze(1) if packed else query,
            frequencies,
            config,
            cu_lengths if packed else None,
            cp_group,
        )
        torch.testing.assert_close(
            actual, expected.squeeze(1) if packed else expected, atol=2e-2, rtol=2e-2
        )

    @pytest.mark.parametrize("learned_k", [False, True])
    @pytest.mark.parametrize("use_rope", [False, True])
    def test_mixed_prefill_reuses_training_and_decode_uses_cute(
        self, monkeypatch, learned_k, use_rope
    ):
        dtype = torch.bfloat16
        hidden_size = 4096
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            ffn_hidden_size=16384,
            num_attention_heads=16,
            num_query_groups=1,
            kv_channels=256,
            params_dtype=dtype,
            bf16=True,
            normalization="RMSNorm",
            add_bias_linear=False,
            attention_dropout=0.0,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_simplified_use_learned_k=learned_k,
            dsa_indexer_n_heads=1,
            dsa_indexer_head_dim=128 if learned_k else 256,
            dsa_indexer_topk=8,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
        )
        inference_context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=128,
                max_requests=3,
                max_tokens=64,
                block_size_tokens=64,
                buffer_size_gb=0.05,
                unified_memory_level=0,
                num_cuda_graphs=None,
                use_flashinfer_fused_rope=False,
            ),
        )
        token_offset = 0
        for request_id, prompt_length in enumerate((13, 9)):
            inference_context.add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=torch.arange(
                        token_offset, token_offset + prompt_length, dtype=torch.long, device="cpu"
                    ),
                    sampling_params=SamplingParams(num_tokens_to_generate=2),
                )
            )
            token_offset += prompt_length
        inference_context.initialize_attention_state()

        attention = DSGroupedSelfAttention(
            config=config,
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        attention.eval()
        rope = RotaryEmbedding(
            config.kv_channels,
            rotary_percent=config.rotary_percent,
            cp_group=attention.pg_collection.cp,
        )

        def rotary(seq_len):
            return rope(seq_len) if use_rope else None

        def run_step() -> tuple[torch.Tensor, torch.Tensor]:
            hidden_states = torch.randn(
                inference_context.padded_active_token_count,
                1,
                hidden_size,
                dtype=dtype,
                device=torch.cuda.current_device(),
            )
            with torch.inference_mode(), InferenceMode.active():
                output, bias = attention(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    inference_context=inference_context,
                    packed_seq_params=_dynamic_packed_seq_params(inference_context),
                    rotary_pos_emb=rotary(128),
                )
            assert bias is None
            return output, hidden_states

        prefill_output, prompt_hidden_states = run_step()
        assert prefill_output.shape == (inference_context.padded_active_token_count, 1, hidden_size)

        inference_context.update_requests(
            active_requests_mask=torch.ones(2, dtype=torch.int32),
            new_tokens=torch.tensor([100, 101], dtype=torch.long),
        )
        inference_context.add_request(
            DynamicInferenceRequest(
                request_id=2,
                prompt_tokens=torch.arange(
                    token_offset, token_offset + 5, dtype=torch.long, device="cpu"
                ),
                sampling_params=SamplingParams(num_tokens_to_generate=2),
            )
        )
        inference_context.initialize_attention_state()
        assert inference_context.num_decode_requests == 2
        assert inference_context.num_prefill_requests == 1

        from megatron.core.transformer.experimental_attention_variant import dsa_gqa_decode_kernel

        indexer_calls = []
        attention_calls = []
        original_indexer = dsa_gqa_decode_kernel.dsa_gqa_decode_indexer_score
        original_attention = dsa_gqa_decode_kernel.dsa_gqa_decode_attention

        def capture_indexer(*args, **kwargs):
            indexer_calls.append(tuple(args[0].shape))
            return original_indexer(*args, **kwargs)

        def capture_attention(*args, **kwargs):
            attention_calls.append((tuple(args[0].shape), tuple(args[4].shape)))
            return original_attention(*args, **kwargs)

        monkeypatch.setattr(dsa_gqa_decode_kernel, "dsa_gqa_decode_indexer_score", capture_indexer)
        monkeypatch.setattr(dsa_gqa_decode_kernel, "dsa_gqa_decode_attention", capture_attention)
        core_attention = attention.core_attention
        training_forward = core_attention.forward
        prefill_calls = []

        def capture_prefill(*args, **kwargs):
            prefill_calls.append(kwargs["query"].size(0))
            return training_forward(*args, **kwargs)

        monkeypatch.setattr(core_attention, "forward", capture_prefill)
        mixed_output, mixed_hidden_states = run_step()

        assert indexer_calls == [(2, 128 if learned_k else 256)]
        assert attention_calls == [((2, 16, 256), (2, 8))]
        assert prefill_calls == [5]
        assert mixed_output.shape == (inference_context.padded_active_token_count, 1, hidden_size)
        assert mixed_output.dtype == dtype

        monkeypatch.setattr(core_attention, "forward", training_forward)
        prompt_start = 0
        for request_idx, prompt_length in enumerate((13, 9)):
            request_hidden_states = torch.cat(
                (
                    prompt_hidden_states[prompt_start : prompt_start + prompt_length],
                    mixed_hidden_states[request_idx : request_idx + 1],
                )
            )
            with torch.inference_mode():
                training_output, bias = attention(
                    hidden_states=request_hidden_states,
                    attention_mask=None,
                    rotary_pos_emb=rotary(request_hidden_states.size(0)),
                )
            assert bias is None
            torch.testing.assert_close(
                mixed_output[request_idx], training_output[-1], rtol=2e-2, atol=2e-2
            )
            prompt_start += prompt_length

        # All three requests now decode, exercising cache reuse across a second step.
        inference_context.update_requests(
            active_requests_mask=torch.ones(3, dtype=torch.int32),
            new_tokens=torch.tensor([102, 103, 104], dtype=torch.long),
        )
        inference_context.initialize_attention_state()
        assert inference_context.num_decode_requests == 3
        assert inference_context.num_prefill_requests == 0
        monkeypatch.setattr(core_attention, "forward", capture_prefill)
        prefill_calls.clear()
        decode_output, decode_hidden_states = run_step()
        assert prefill_calls == []
        assert indexer_calls[-1] == (3, 128 if learned_k else 256)
        assert attention_calls[-1] == ((3, 16, 256), (3, 8))
        monkeypatch.setattr(core_attention, "forward", training_forward)
        histories = [
            torch.cat((prompt_hidden_states[:13], mixed_hidden_states[:1])),
            torch.cat((prompt_hidden_states[13:22], mixed_hidden_states[1:2])),
            mixed_hidden_states[2:7],
        ]
        for request_idx, history in enumerate(histories):
            full_prefix = torch.cat((history, decode_hidden_states[request_idx : request_idx + 1]))
            with torch.inference_mode():
                expected, _ = attention(
                    hidden_states=full_prefix,
                    attention_mask=None,
                    rotary_pos_emb=rotary(full_prefix.size(0)),
                )
            torch.testing.assert_close(
                decode_output[request_idx], expected[-1], rtol=2e-2, atol=2e-2
            )

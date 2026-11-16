# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import megatron.core.transformer.attention as attention_module
from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb_with_cos_sin
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestRotaryEmbeddingWithPrecomputedCosSin:

    def setup_method(self):
        self.batch_size = 3
        self.seq_len = 4
        self.d_rot = 6
        self.rotary_embedding = RotaryEmbedding(kv_channels=4, rotary_percent=1.0)

    def test_output_shapes_match(self):

        # Create input tensors
        t = torch.randn(self.seq_len, self.batch_size, 2, self.d_rot * 2, device="cuda")
        rotary_pos_cos, rotary_pos_sin = self.rotary_embedding.get_cos_sin(self.seq_len)

        # Test using Flash Decoding optimized kernel which requires precomputed cos & sin tensors
        expected_shape = torch.Size(
            [self.seq_len, self.batch_size, self.seq_len // 2, self.seq_len * self.batch_size]
        )
        output_flash_rotary = apply_rotary_pos_emb_with_cos_sin(
            t, rotary_pos_cos, rotary_pos_sin, rotary_interleaved=True
        )

        assert (
            output_flash_rotary.shape == expected_shape
        ), f"Outputs do not match: {output_flash_rotary.shape} != {expected_shape}"


@pytest.mark.parametrize(
    ("batch_invariant_mode", "num_requests", "tokens_per_request", "padded_token_count"),
    [
        (False, 728, 1, 728),
        (True, 728, 1, 728),
        (True, 728, 1, 768),
        (False, 20, 3, 60),
        (True, 20, 3, 64),
    ],
)
def test_decode_attention_preserves_batch_invariant_token_padding(
    monkeypatch, batch_invariant_mode, num_requests, tokens_per_request, padded_token_count
):
    """Only batch-invariant token-only rows bypass the attention kernel."""
    attention = object.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        window_size=None, window_attn_skip_freq=None, attn_logit_softcapping=None
    )
    attention.layer_number = 1
    attention.batch_invariant_mode = batch_invariant_mode
    attention.flash_attention_version = 4
    attention.train(False)

    kernel_queries = []

    def fake_fa4_varlen(q, _k, _v, **_kwargs):
        kernel_queries.append(q.clone())
        return q.clone(), None

    monkeypatch.setattr(attention_module, "HAVE_FA4", True)
    monkeypatch.setattr(attention_module, "flash_attn4_varlen_func", fake_fa4_varlen, raising=False)

    metadata_token_count = num_requests * tokens_per_request
    num_heads = 2
    head_dim = 4
    q = torch.arange(padded_token_count * num_heads * head_dim, dtype=torch.float32).reshape(
        padded_token_count, 1, num_heads, head_dim
    )
    cu_seqlens = torch.arange(0, metadata_token_count + 1, tokens_per_request, dtype=torch.int32)
    seqlens_k = torch.ones(num_requests, dtype=torch.int32)

    output = attention.flash_decode_and_prefill(
        q=q,
        k=torch.empty(0),
        v=torch.empty(0),
        max_seqlen_q=tokens_per_request,
        max_seqlen_k=1,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        seqlens_k=seqlens_k,
        block_table=torch.zeros((num_requests, 1), dtype=torch.int32),
        is_decode_only=True,
    )

    assert kernel_queries[0].shape == (metadata_token_count, num_heads, head_dim)
    assert torch.equal(kernel_queries[0], q[:metadata_token_count, 0])
    assert output.shape == q.shape
    assert torch.equal(output[:metadata_token_count], q[:metadata_token_count])
    assert torch.count_nonzero(output[metadata_token_count:]) == 0


@pytest.mark.skipif(
    attention_module.flash_attn_with_kvcache is None, reason="flash decode needs flash-attn 2"
)
class TestStaticFlashDecodeStep:
    """A static flash_decode decode step must compute what every other path computes.

    flash_decode's kernel stands in for the KV-cache update and core attention only: the layer's
    sliding window must reach it, and the steps after core attention, such as the output gate,
    must still run, also with the RoPE the kernel applies and inside a CUDA graph, which static
    CUDA graphs require flash_decode for.
    """

    batch, heads, groups, hidden = 2, 4, 2, 128  # head dim 32
    cached = 12  # tokens already in the KV cache; this step's token is at position 12
    max_sequence_length = 32
    window = (4, 0)

    # A bf16 kernel against an fp32 reference on the same bf16 q/k/v/gate. Flash-attention rounds
    # the probabilities and the output to bf16, and the gated product is rounded once more: each
    # a relative error of at most 2**-9 per element. Over 13 keys that bounds the relative L2
    # error at about 1e-2 (2e-3 measured on H100). A missing window or gate costs 0.7 to 1.
    rel_tol = 2e-2

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _attention(self, window_size, attention_output_gate):
        pytest.importorskip("transformer_engine")
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_submodules,
        )

        config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden,
            num_attention_heads=self.heads,
            num_query_groups=self.groups,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            flash_decode=True,
            window_size=window_size,
            attention_output_gate=attention_output_gate,
        )
        attention = SelfAttention(
            config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        return attention.cuda().eval()

    def _decode_step(self, attention, rope=False, cuda_graph=False):
        """Run one decode step over a seeded KV cache.

        With ``rope``, the kernel rotates the step's query and key (the cached keys stand for
        already rotated ones). With ``cuda_graph``, the step is captured in a CUDA graph after an
        eager run and the replayed output is returned.

        Returns the core-attention output, the step's q/k/v (and gate), the cached K/V and the
        RoPE cos/sin tables (or None).
        """
        dim = attention.hidden_size_per_attention_head
        generator = torch.Generator(device="cuda").manual_seed(0)
        cache_shape = (self.max_sequence_length, self.batch, self.groups, dim)
        key_memory = torch.zeros(cache_shape, device="cuda", dtype=torch.bfloat16)
        value_memory = torch.zeros_like(key_memory)
        for memory in (key_memory, value_memory):
            memory[: self.cached] = torch.randn(
                self.cached, *cache_shape[1:], generator=generator, device="cuda"
            )
        cached_keys = key_memory[: self.cached].clone()
        cached_values = value_memory[: self.cached].clone()

        context = StaticInferenceContext(self.batch, self.max_sequence_length)
        context.sequence_len_offset = self.cached
        context.enable_decode_mode()
        context.key_value_memory_dict[attention.layer_number] = (key_memory, value_memory)

        hidden_states = torch.randn(
            1, self.batch, self.hidden, generator=generator, device="cuda"
        ).bfloat16()
        sequence_len_offset = torch.full(
            (self.batch,), self.cached, dtype=torch.int32, device="cuda"
        )
        rotary = None
        if rope:
            rotary = RotaryEmbedding(kv_channels=dim, rotary_percent=1.0).get_cos_sin(
                self.max_sequence_length
            )

        def step():
            return attention.forward_pre_attn_and_core_attn(
                hidden_states,
                None,
                inference_context=context,
                rotary_pos_cos=None if rotary is None else rotary[0],
                rotary_pos_sin=None if rotary is None else rotary[1],
                sequence_len_offset=sequence_len_offset,
            )

        projected = []
        get_query_key_value_tensors = attention.get_query_key_value_tensors

        def capture(*args, **kwargs):
            outputs = get_query_key_value_tensors(*args, **kwargs)
            projected.extend(t.detach().clone() for t in outputs)
            return outputs

        with torch.no_grad(), InferenceMode.active():
            with mock.patch.object(attention, "get_query_key_value_tensors", side_effect=capture):
                output = step()
            if cuda_graph:
                # The eager step warmed up the kernels. Replaying rewrites this step's key and
                # value at the same cache position, so the cache stays as the reference sees it.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = step()
                graph.replay()
                torch.cuda.synchronize()
        return output, projected, cached_keys, cached_values, rotary

    def _reference(self, projected, cached_keys, cached_values, window_size, rotary=None):
        """fp32 attention of the step's query over the keys its window allows, then the gate."""
        query, key, value = projected[:3]
        heads, dim = query.shape[2], query.shape[3]
        repeats = heads // key.shape[2]
        if rotary is not None:
            # The kernel rotates the step's query and key at its position with the bf16 tables
            # flash_decode passes, rotating the two halves of each head (rotary_interleaved=False).
            cos, sin = (t[self.cached].bfloat16().float() for t in rotary)

            def rotate(x):
                x1, x2 = x.float().chunk(2, dim=-1)
                return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

            query, key = rotate(query), rotate(key)
        keys = torch.cat([cached_keys.float(), key.float()]).repeat_interleave(repeats, dim=2)
        values = torch.cat([cached_values, value]).float().repeat_interleave(repeats, dim=2)
        scores = torch.einsum("bhd,kbhd->bhk", query[0].float(), keys) * dim**-0.5
        if window_size is not None:
            positions = torch.arange(self.cached + 1, device=scores.device)
            scores = scores.masked_fill(positions < self.cached - window_size[0], float("-inf"))
        output = torch.einsum("bhk,kbhd->bhd", torch.softmax(scores, dim=-1), values)
        output = output.reshape(1, self.batch, heads * dim)
        if len(projected) == 4:
            output = output * torch.sigmoid(projected[3].float().reshape(output.shape))
        return output

    def _check(self, window_size, attention_output_gate, rope=False, cuda_graph=False):
        attention = self._attention(window_size, attention_output_gate)
        output, projected, cached_keys, cached_values, rotary = self._decode_step(
            attention, rope=rope, cuda_graph=cuda_graph
        )
        assert len(projected) == (4 if attention_output_gate else 3)
        reference = self._reference(projected, cached_keys, cached_values, window_size, rotary)
        error = ((output.float() - reference).norm() / reference.norm()).item()
        assert error <= self.rel_tol, (
            f"static flash_decode step (window_size={window_size}, "
            f"attention_output_gate={attention_output_gate}, rope={rope}, "
            f"cuda_graph={cuda_graph}) differs from the reference by rel {error:.3g}"
        )

    @pytest.mark.parametrize("window_size", [None, window], ids=["full", "window"])
    @pytest.mark.parametrize("attention_output_gate", [False, True], ids=["ungated", "gated"])
    def test_decode_step_matches_reference(self, window_size, attention_output_gate):
        self._check(window_size, attention_output_gate)

    @pytest.mark.parametrize("cuda_graph", [False, True], ids=["eager", "cuda_graph"])
    def test_windowed_gated_decode_step_with_rope_matches_reference(self, cuda_graph):
        """The window and gate hold together with the RoPE the kernel applies, eagerly and
        replayed from a CUDA graph."""
        self._check(self.window, True, rope=True, cuda_graph=cuda_graph)

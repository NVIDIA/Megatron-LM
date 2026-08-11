# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native parity coverage for repeated-MTP DSA IndexShare and KVShare."""

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
    get_transformer_layer_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, CudaGraphModule
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.multi_token_prediction import (
    MTPDSAIterationContext,
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils


def _make_config(**overrides) -> MLATransformerConfig:
    kwargs = dict(
        num_layers=2,
        mtp_num_layers=7,
        mtp_use_repeated_layer=True,
        dsa_mtp_index_kv_share=True,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=8,
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_head_dim=8,
        qk_pos_emb_head_dim=4,
        v_head_dim=8,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=3,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_scoring_relu=False,
        dsa_kernel_backend="none",
        attention_backend=AttnBackend.unfused,
        add_bias_linear=False,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_base=10000,
        gradient_accumulation_fusion=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 2, multiplier=2.0),
        use_cpu_initialization=False,
        perform_initialization=True,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"experimental_attention_variant": None}, "requires experimental_attention_variant"),
        ({"mtp_use_repeated_layer": False}, "requires mtp_use_repeated_layer"),
        ({"mtp_num_layers": 1}, "requires mtp_num_layers > 1"),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["mla_up_proj"]},
            "not compatible with selective mla_up_proj recompute",
        ),
        (
            {"cuda_graph_impl": "transformer_engine", "cuda_graph_modules": ["attn"]},
            "does not yet support CUDA graph scopes that capture attention",
        ),
    ],
)
def test_iteration_sharing_rejects_incompatible_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_config(**overrides)


def test_iteration_sharing_accepts_moe_only_cuda_graph_scope():
    config = _make_config(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=["moe_router", "moe_preprocess"],
        num_moe_experts=4,
    )

    assert config.cuda_graph_modules == [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess]


def _rope_positions(total_tokens: int, segment_lengths: list[int] | None, device) -> torch.Tensor:
    if segment_lengths is None:
        return torch.arange(total_tokens, device=device)
    return torch.cat([torch.arange(length, device=device) for length in segment_lengths])


def _apply_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float, interleaved: bool
) -> torch.Tensor:
    dtype = x.dtype
    dim = x.size(-1)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))
    freqs = torch.outer(positions.float(), freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs).view(x.size(0), 1, 1, -1)
    if interleaved:
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    else:
        x_pairs = x.float().reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x_complex = torch.view_as_complex(x_pairs)
    output = torch.view_as_real(x_complex * freqs).flatten(-2)
    if not interleaved:
        output = torch.cat([output[..., 0::2], output[..., 1::2]], dim=-1)
    return output.to(dtype=dtype)


def _causal_segment_mask(
    total_tokens: int, segment_lengths: list[int] | None, device
) -> torch.Tensor:
    valid = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)
    start = 0
    for length in segment_lengths or [total_tokens]:
        valid[start : start + length, start : start + length] = torch.tril(
            torch.ones((length, length), dtype=torch.bool, device=device)
        )
        start += length
    return valid


class _NativeSharedAbsorbedDSA(nn.Module):
    """Pure-PyTorch repeated-iteration DSA oracle with explicit source tensor reuse."""

    _PARAMETER_MAP = {
        "q_down_weight": "linear_q_down_proj.weight",
        "q_norm_weight": "q_layernorm.weight",
        "q_up_weight": "linear_q_up_proj.weight",
        "kv_down_weight": "linear_kv_down_proj.weight",
        "kv_norm_weight": "kv_layernorm.weight",
        "kv_up_weight": "linear_kv_up_proj.weight",
        "output_weight": "linear_proj.weight",
        "index_q_weight": "core_attention.indexer.linear_wq_b.weight",
        "index_k_weight": "core_attention.indexer.linear_wk.weight",
        "index_k_norm_weight": "core_attention.indexer.k_norm.weight",
        "index_k_norm_bias": "core_attention.indexer.k_norm.bias",
        "index_weight_proj": "core_attention.indexer.linear_weights_proj.weight",
    }

    def __init__(self, real_attention, config: MLATransformerConfig):
        super().__init__()
        self.config = config
        real_parameters = dict(real_attention.named_parameters())
        for native_name, real_name in self._PARAMETER_MAP.items():
            setattr(self, native_name, nn.Parameter(real_parameters[real_name].detach().clone()))

    def _project_query(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        qr = F.linear(hidden_states, self.q_down_weight)
        qr = F.rms_norm(
            qr, (self.config.q_lora_rank,), self.q_norm_weight, self.config.layernorm_epsilon
        )
        query = F.linear(qr, self.q_up_weight).view(
            hidden_states.size(0),
            hidden_states.size(1),
            self.config.num_attention_heads,
            self.config.qk_head_dim + self.config.qk_pos_emb_head_dim,
        )
        query_nope, query_rope = torch.split(
            query, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )
        query_rope = _apply_rope(query_rope, positions, self.config.rotary_base, interleaved=True)

        kv_up = self.kv_up_weight.view(
            self.config.num_attention_heads,
            self.config.qk_head_dim + self.config.v_head_dim,
            self.config.kv_lora_rank,
        )
        k_up = kv_up[:, : self.config.qk_head_dim]
        query_absorbed = torch.einsum("sbhd,hdc->sbhc", query_nope, k_up)
        return torch.cat([query_absorbed, query_rope], dim=-1), qr, kv_up

    def _project_key(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        kv_combined = F.linear(hidden_states, self.kv_down_weight)
        kv_latent, key_rope = torch.split(
            kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        kv_latent = F.rms_norm(
            kv_latent,
            (self.config.kv_lora_rank,),
            self.kv_norm_weight,
            self.config.layernorm_epsilon,
        )
        key_rope = _apply_rope(
            key_rope.unsqueeze(2), positions, self.config.rotary_base, interleaved=True
        )
        return torch.cat([kv_latent.unsqueeze(2), key_rope], dim=-1)

    def _compute_topk(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        source = hidden_states.detach()
        qr = qr.detach()
        query = F.linear(qr, self.index_q_weight).view(
            hidden_states.size(0),
            hidden_states.size(1),
            self.config.dsa_indexer_n_heads,
            self.config.dsa_indexer_head_dim,
        )
        query_rope, query_nope = torch.split(
            query,
            [
                self.config.qk_pos_emb_head_dim,
                self.config.dsa_indexer_head_dim - self.config.qk_pos_emb_head_dim,
            ],
            dim=-1,
        )
        query_rope = _apply_rope(query_rope, positions, self.config.rotary_base, interleaved=False)
        query = torch.cat([query_rope, query_nope], dim=-1)

        key = F.linear(source, self.index_k_weight)
        key = F.layer_norm(
            key,
            (self.config.dsa_indexer_head_dim,),
            self.index_k_norm_weight,
            self.index_k_norm_bias,
            self.config.dsa_indexer_k_norm_epsilon or self.config.layernorm_epsilon,
        )
        key = key.unsqueeze(2)
        key_rope, key_nope = torch.split(
            key,
            [
                self.config.qk_pos_emb_head_dim,
                self.config.dsa_indexer_head_dim - self.config.qk_pos_emb_head_dim,
            ],
            dim=-1,
        )
        key_rope = _apply_rope(key_rope, positions, self.config.rotary_base, interleaved=False)
        key = torch.cat([key_rope, key_nope], dim=-1).squeeze(2)

        weights = F.linear(source, self.index_weight_proj)
        weights = weights * (self.config.dsa_indexer_n_heads**-0.5)
        weights = weights * (self.config.dsa_indexer_head_dim**-0.5)
        scores = torch.einsum("sbhd,tbd->bsht", query.float(), key.float())
        scores = (scores * weights.transpose(0, 1).unsqueeze(-1)).sum(dim=2)
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        topk_scores, topk = scores.topk(min(self.config.dsa_indexer_topk, key.size(0)), dim=-1)
        return topk.masked_fill(topk_scores == float("-inf"), -1)

    def forward_iteration(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        shared_key: torch.Tensor | None,
        shared_topk: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, qr, kv_up = self._project_query(hidden_states, positions)
        key = self._project_key(hidden_states, positions) if shared_key is None else shared_key
        topk = (
            self._compute_topk(hidden_states, qr, positions, valid)
            if shared_topk is None
            else shared_topk
        )

        scores = torch.einsum("sbhd,tbnd->bhst", query.float(), key.float())
        scores = scores * (self.config.qk_head_dim + self.config.qk_pos_emb_head_dim) ** -0.5
        selected = torch.zeros_like(valid, dtype=torch.int32).unsqueeze(0)
        selected.scatter_add_(-1, topk.clamp_min(0), (topk >= 0).to(dtype=selected.dtype))
        sparse_valid = valid.unsqueeze(0) & (selected > 0)
        scores = scores.masked_fill(~sparse_valid.unsqueeze(1), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(dtype=key.dtype)
        latent_value = key[..., : self.config.kv_lora_rank]
        latent_output = torch.einsum("bhst,tbnd->sbhd", probabilities, latent_value)
        v_up = kv_up[:, self.config.qk_head_dim :]
        output = torch.einsum("sbhc,hdc->sbhd", latent_output, v_up)
        output = F.linear(output.reshape(*hidden_states.shape[:-1], -1), self.output_weight)
        return output, key, topk


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return F.cosine_similarity(
        left.flatten().double().unsqueeze(0), right.flatten().double().unsqueeze(0)
    ).item()


def _tensor_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double()
    right = right.double()
    denominator = (left.square() + right.square()).sum()
    if denominator == 0:
        return 1.0
    return (2.0 * (left * right).sum() / denominator).item()


def _assert_similarity(left: torch.Tensor, right: torch.Tensor, tolerance: float = 2e-2):
    assert torch.isfinite(left).all()
    assert torch.isfinite(right).all()
    assert _cosine_similarity(left, right) > 1 - tolerance
    assert _tensor_similarity(left, right) > 1 - tolerance


@pytest.mark.parametrize("segment_lengths", [None, [5, 4]])
def test_repeated_mtp_dsa_index_and_kv_share_native_parity(monkeypatch, segment_lengths):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)

        config = _make_config()
        attention_spec = get_dsa_module_spec_for_backend(config, backend=TESpecProvider())
        real_attention = (
            build_module(attention_spec, config=config, layer_number=1, is_mtp_layer=True)
            .bfloat16()
            .cuda()
        )
        native_attention = _NativeSharedAbsorbedDSA(real_attention, config).bfloat16().cuda()

        total_tokens = sum(segment_lengths) if segment_lengths is not None else 9
        positions = _rope_positions(total_tokens, segment_lengths, device="cuda")
        valid = _causal_segment_mask(total_tokens, segment_lengths, device="cuda")
        if segment_lengths is None:
            attention_mask = torch.zeros(
                (1, 1, total_tokens, total_tokens), dtype=torch.float32, device="cuda"
            ).masked_fill(~valid.view(1, 1, total_tokens, total_tokens), float("-inf"))
            packed_seq_params = None
        else:
            cu_seqlens = torch.tensor(
                [0, segment_lengths[0], total_tokens], dtype=torch.int32, device="cuda"
            )
            attention_mask = None
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max(segment_lengths),
                max_seqlen_kv=max(segment_lengths),
            )

        call_counts = {"q": 0, "kv": 0, "indexer": 0, "sparse": 0}
        hooks = [
            real_attention.linear_q_down_proj.register_forward_hook(
                lambda *_args: call_counts.__setitem__("q", call_counts["q"] + 1)
            ),
            real_attention.linear_kv_down_proj.register_forward_hook(
                lambda *_args: call_counts.__setitem__("kv", call_counts["kv"] + 1)
            ),
            real_attention.core_attention.indexer.linear_wq_b.register_forward_hook(
                lambda *_args: call_counts.__setitem__("indexer", call_counts["indexer"] + 1)
            ),
        ]
        original_sparse_attention = dsa_module._run_sparse_attention

        def counted_sparse_attention(*args, **kwargs):
            call_counts["sparse"] += 1
            return original_sparse_attention(*args, **kwargs)

        monkeypatch.setattr(dsa_module, "_run_sparse_attention", counted_sparse_attention)

        real_inputs = [
            torch.randn(
                total_tokens,
                1,
                config.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
                requires_grad=True,
            )
            for _ in range(config.mtp_num_layers)
        ]
        native_inputs = [value.detach().clone().requires_grad_(True) for value in real_inputs]
        output_grads = [torch.randn_like(value) for value in real_inputs]

        real_outputs = []
        native_outputs = []
        real_shared = None
        native_key = native_topk = None
        for iteration in range(config.mtp_num_layers):
            context = MTPDSAIterationContext(iteration, shared_tensors=real_shared)
            real_output, _ = real_attention(
                real_inputs[iteration],
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
                mtp_dsa_context=context,
            )
            if context.is_source:
                real_shared = context.require_source_tensors()
            real_outputs.append(real_output)

            native_output, current_key, current_topk = native_attention.forward_iteration(
                native_inputs[iteration], positions, valid, native_key, native_topk
            )
            if iteration == 0:
                native_key, native_topk = current_key, current_topk
            native_outputs.append(native_output)

        assert real_shared is not None
        assert torch.equal(real_shared.topk_indices, native_topk)
        for row, indices in enumerate(real_shared.topk_indices[0]):
            indices = indices[indices >= 0]
            assert valid[row, indices].all()
        for output, reference in zip(real_outputs, native_outputs):
            _assert_similarity(output, reference)

        torch.autograd.backward(real_outputs, output_grads)
        torch.autograd.backward(native_outputs, output_grads)
        for real_input, native_input in zip(real_inputs, native_inputs):
            _assert_similarity(real_input.grad, native_input.grad)

        real_parameters = dict(real_attention.named_parameters())
        for native_name, real_name in native_attention._PARAMETER_MAP.items():
            native_parameter = getattr(native_attention, native_name)
            real_parameter = real_parameters[real_name]
            if native_parameter.grad is None or real_parameter.grad is None:
                assert native_parameter.grad is None and real_parameter.grad is None
                continue
            _assert_similarity(real_parameter.grad, native_parameter.grad)

        assert call_counts == {"q": 7, "kv": 1, "indexer": 1, "sparse": 7}
        for hook in hooks:
            hook.remove()
    finally:
        Utils.destroy_model_parallel()


def test_indexer_loss_is_computed_only_by_the_source_iteration(monkeypatch):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(2345)
        torch.manual_seed(2345)
        torch.cuda.manual_seed(2345)

        config = _make_config(dsa_indexer_loss_coeff=0.1)
        attention_spec = get_dsa_module_spec_for_backend(config, backend=TESpecProvider())
        attention = (
            build_module(attention_spec, config=config, layer_number=1, is_mtp_layer=True)
            .bfloat16()
            .cuda()
        )
        tracked_losses = []
        monkeypatch.setattr(
            dsa_module.DSAIndexerLossLoggingHelper,
            "save_loss_to_tracker",
            staticmethod(lambda **kwargs: tracked_losses.append(kwargs["loss"])),
        )

        total_tokens = 9
        valid = _causal_segment_mask(total_tokens, None, device="cuda")
        attention_mask = torch.zeros(
            (1, 1, total_tokens, total_tokens), dtype=torch.float32, device="cuda"
        ).masked_fill(~valid.view(1, 1, total_tokens, total_tokens), float("-inf"))
        shared = None
        outputs = []
        for iteration in range(config.mtp_num_layers):
            context = MTPDSAIterationContext(iteration, shared_tensors=shared)
            hidden_states = torch.randn(
                total_tokens,
                1,
                config.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
                requires_grad=True,
            )
            output, _ = attention(
                hidden_states, attention_mask=attention_mask, mtp_dsa_context=context
            )
            if context.is_source:
                shared = context.require_source_tensors()
            outputs.append(output)

        sum(output.float().sum() for output in outputs).backward()

        assert len(tracked_losses) == 1
        assert tracked_losses[0].requires_grad
        for name in ("linear_wq_b.weight", "linear_wk.weight", "linear_weights_proj.weight"):
            parameter = dict(attention.core_attention.indexer.named_parameters())[name]
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    finally:
        Utils.destroy_model_parallel()


def test_source_indexer_loss_and_gradients_match_unshared_oracle(monkeypatch):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(3456)
        torch.manual_seed(3456)
        torch.cuda.manual_seed(3456)

        shared_config = _make_config(dsa_indexer_loss_coeff=0.1)
        oracle_config = _make_config(dsa_indexer_loss_coeff=0.1, dsa_mtp_index_kv_share=False)
        shared_spec = get_dsa_module_spec_for_backend(shared_config, backend=TESpecProvider())
        oracle_spec = get_dsa_module_spec_for_backend(oracle_config, backend=TESpecProvider())
        shared_attention = (
            build_module(shared_spec, config=shared_config, layer_number=1, is_mtp_layer=True)
            .bfloat16()
            .cuda()
        )
        oracle_attention = (
            build_module(oracle_spec, config=oracle_config, layer_number=1, is_mtp_layer=True)
            .bfloat16()
            .cuda()
        )
        oracle_attention.load_state_dict(shared_attention.state_dict())

        tracked_losses = []
        monkeypatch.setattr(
            dsa_module.DSAIndexerLossLoggingHelper,
            "save_loss_to_tracker",
            staticmethod(lambda **kwargs: tracked_losses.append(kwargs["loss"])),
        )

        total_tokens = 9
        valid = _causal_segment_mask(total_tokens, None, device="cuda")
        attention_mask = torch.zeros(
            (1, 1, total_tokens, total_tokens), dtype=torch.float32, device="cuda"
        ).masked_fill(~valid.view(1, 1, total_tokens, total_tokens), float("-inf"))
        source_input = torch.randn(
            total_tokens, 1, shared_config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )

        shared_tensors = None
        shared_outputs = []
        for iteration in range(shared_config.mtp_num_layers):
            context = MTPDSAIterationContext(iteration, shared_tensors=shared_tensors)
            hidden_states = (
                source_input.detach().clone() if iteration == 0 else torch.randn_like(source_input)
            ).requires_grad_(True)
            output, _ = shared_attention(
                hidden_states, attention_mask=attention_mask, mtp_dsa_context=context
            )
            if context.is_source:
                shared_tensors = context.require_source_tensors()
            shared_outputs.append(output)

        oracle_input = source_input.detach().clone().requires_grad_(True)
        oracle_output, _ = oracle_attention(oracle_input, attention_mask=attention_mask)

        assert len(tracked_losses) == 2
        torch.testing.assert_close(tracked_losses[0], tracked_losses[1], rtol=0, atol=0)

        sum(output.float().sum() for output in shared_outputs).backward()
        oracle_output.float().sum().backward()

        shared_indexer_parameters = dict(shared_attention.core_attention.indexer.named_parameters())
        oracle_indexer_parameters = dict(oracle_attention.core_attention.indexer.named_parameters())
        for name in ("linear_wq_b.weight", "linear_wk.weight", "linear_weights_proj.weight"):
            shared_grad = shared_indexer_parameters[name].grad
            oracle_grad = oracle_indexer_parameters[name].grad
            assert shared_grad is not None and oracle_grad is not None
            torch.testing.assert_close(shared_grad, oracle_grad, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("dynamic_cp", [False, True])
def test_cp2_reuses_global_source_kv_and_topk_with_gradient_flow(monkeypatch, dynamic_cp):
    if Utils.world_size < 2:
        pytest.skip("CP2 MTP DSA sharing requires at least two distributed ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(3456)
        torch.manual_seed(3456 + parallel_state.get_context_parallel_rank())
        torch.cuda.manual_seed(3456 + parallel_state.get_context_parallel_rank())

        config = _make_config(context_parallel_size=2, cp_comm_type="all_gather")
        attention_spec = get_dsa_module_spec_for_backend(config, backend=TESpecProvider())
        attention = (
            build_module(
                attention_spec,
                config=config,
                layer_number=1,
                cp_comm_type=config.cp_comm_type,
                is_mtp_layer=True,
            )
            .bfloat16()
            .cuda()
        )
        call_counts = {"kv": 0, "indexer": 0, "sparse": 0}
        hooks = [
            attention.linear_kv_down_proj.register_forward_hook(
                lambda *_args: call_counts.__setitem__("kv", call_counts["kv"] + 1)
            ),
            attention.core_attention.indexer.linear_wq_b.register_forward_hook(
                lambda *_args: call_counts.__setitem__("indexer", call_counts["indexer"] + 1)
            ),
        ]
        original_sparse_attention = dsa_module._run_sparse_attention

        def counted_sparse_attention(*args, **kwargs):
            call_counts["sparse"] += 1
            return original_sparse_attention(*args, **kwargs)

        monkeypatch.setattr(dsa_module, "_run_sparse_attention", counted_sparse_attention)

        local_tokens = 8
        global_tokens = local_tokens * 2
        cu_seqlens = torch.tensor([0, global_tokens], dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens.clone(),
            cu_seqlens_kv_padded=cu_seqlens.clone(),
            max_seqlen_q=global_tokens,
            max_seqlen_kv=global_tokens,
            local_cp_size=2 if dynamic_cp else None,
            cp_group=(parallel_state.get_context_parallel_group() if dynamic_cp else None),
        )
        source_hidden = torch.randn(
            local_tokens,
            1,
            config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        consumer_hidden = torch.randn_like(source_hidden, requires_grad=True)

        source_context = MTPDSAIterationContext(iteration=0)
        attention(
            source_hidden,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
            mtp_dsa_context=source_context,
        )
        shared = source_context.require_source_tensors()
        consumer_context = MTPDSAIterationContext(iteration=1, shared_tensors=shared)
        consumer_output, _ = attention(
            consumer_hidden,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
            mtp_dsa_context=consumer_context,
        )
        consumer_output.float().square().mean().backward()

        assert shared.key.size(0) == global_tokens
        assert torch.all(shared.topk_indices >= -1)
        assert torch.all(shared.topk_indices < global_tokens)
        assert source_hidden.grad is not None and source_hidden.grad.float().norm() > 0
        assert consumer_hidden.grad is not None and consumer_hidden.grad.float().norm() > 0
        assert torch.isfinite(source_hidden.grad).all()
        assert torch.isfinite(consumer_hidden.grad).all()
        assert call_counts == {"kv": 1, "indexer": 1, "sparse": 2}
        for hook in hooks:
            hook.remove()
    finally:
        Utils.destroy_model_parallel()


def test_tp2_sequence_parallel_reuses_global_source_kv_and_topk(monkeypatch):
    if Utils.world_size < 2:
        pytest.skip("TP2 MTP DSA sharing requires at least two distributed ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(3789)
        torch.manual_seed(3789 + parallel_state.get_tensor_model_parallel_rank())
        torch.cuda.manual_seed(3789 + parallel_state.get_tensor_model_parallel_rank())

        config = _make_config(tensor_model_parallel_size=2, sequence_parallel=True)
        attention_spec = get_dsa_module_spec_for_backend(config, backend=TESpecProvider())
        attention = (
            build_module(attention_spec, config=config, layer_number=1, is_mtp_layer=True)
            .bfloat16()
            .cuda()
        )
        call_counts = {"kv": 0, "indexer": 0, "sparse": 0}
        hooks = [
            attention.linear_kv_down_proj.register_forward_hook(
                lambda *_args: call_counts.__setitem__("kv", call_counts["kv"] + 1)
            ),
            attention.core_attention.indexer.linear_wq_b.register_forward_hook(
                lambda *_args: call_counts.__setitem__("indexer", call_counts["indexer"] + 1)
            ),
        ]
        original_sparse_attention = dsa_module._run_sparse_attention

        def counted_sparse_attention(*args, **kwargs):
            call_counts["sparse"] += 1
            return original_sparse_attention(*args, **kwargs)

        monkeypatch.setattr(dsa_module, "_run_sparse_attention", counted_sparse_attention)

        local_tokens = 8
        global_tokens = local_tokens * 2
        source_hidden = torch.randn(
            local_tokens,
            1,
            config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        consumer_hidden = torch.randn_like(source_hidden, requires_grad=True)
        valid = _causal_segment_mask(global_tokens, None, device="cuda")
        attention_mask = torch.zeros(
            (1, 1, global_tokens, global_tokens), dtype=torch.float32, device="cuda"
        ).masked_fill(~valid.view(1, 1, global_tokens, global_tokens), float("-inf"))
        position_ids = torch.arange(global_tokens, device="cuda").view(1, global_tokens)

        source_context = MTPDSAIterationContext(iteration=0)
        attention(
            source_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mtp_dsa_context=source_context,
        )
        shared = source_context.require_source_tensors()
        consumer_context = MTPDSAIterationContext(iteration=1, shared_tensors=shared)
        consumer_output, _ = attention(
            consumer_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mtp_dsa_context=consumer_context,
        )
        consumer_output.float().square().mean().backward()

        assert shared.key.size(0) == global_tokens
        # Absorbed MLA gathers the sequence before its Q up-projection, so DSA sees
        # global query rows even though the TransformerLayer input/output remain SP-local.
        assert shared.topk_indices.size(1) == global_tokens
        assert torch.all(shared.topk_indices >= -1)
        assert torch.all(shared.topk_indices < global_tokens)
        assert source_hidden.grad is not None and source_hidden.grad.float().norm() > 0
        assert consumer_hidden.grad is not None and consumer_hidden.grad.float().norm() > 0
        assert torch.isfinite(source_hidden.grad).all()
        assert torch.isfinite(consumer_hidden.grad).all()
        assert call_counts == {"kv": 1, "indexer": 1, "sparse": 2}
        for hook in hooks:
            hook.remove()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_mxfp8", [False, True])
def test_full_recompute_threads_shared_kv_as_checkpoint_tensor_inputs(use_mxfp8):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(4321)
        scale = nn.Parameter(torch.tensor(1.25, device="cuda"))
        dummy_layer = SimpleNamespace(
            config=SimpleNamespace(
                fp8="e4m3" if use_mxfp8 else None,
                fp8_recipe=Fp8Recipe.mxfp8 if use_mxfp8 else None,
                fp4=False,
                distribute_saved_activations=False,
                recompute_method="uniform",
                recompute_num_layers=1,
            ),
            scale=scale,
        )

        def projected_forward(self, hidden_states, decoder_input, mtp_dsa_context=None, **_kwargs):
            combined = hidden_states + decoder_input
            if mtp_dsa_context.is_source:
                key = combined * self.scale
                topk = torch.zeros(
                    (1, combined.size(0), 1), dtype=torch.int64, device=combined.device
                )
                mtp_dsa_context.capture(key, topk, None)
                return combined.square()
            return combined + 3 * mtp_dsa_context.shared_tensors.key

        dummy_layer._proj_and_transformer_layer = types.MethodType(projected_forward, dummy_layer)

        hidden = [torch.randn(4, 1, 3, device="cuda", requires_grad=True) for _ in range(3)]
        decoder = [torch.randn_like(value) for value in hidden]
        shared = None
        outputs = []
        for iteration in range(3):
            context = MTPDSAIterationContext(iteration, shared_tensors=shared)
            output, shared = MultiTokenPredictionLayer._checkpointed_forward(
                dummy_layer,
                hidden_states=hidden[iteration],
                decoder_input=decoder[iteration],
                mtp_dsa_context=context,
            )
            outputs.append(output)

        sum(output.sum() for output in outputs).backward()

        reference_scale = nn.Parameter(scale.detach().clone())
        reference_hidden = [value.detach().clone().requires_grad_(True) for value in hidden]
        source = (reference_hidden[0] + decoder[0]) * reference_scale
        reference_outputs = [
            (reference_hidden[0] + decoder[0]).square(),
            reference_hidden[1] + decoder[1] + 3 * source,
            reference_hidden[2] + decoder[2] + 3 * source,
        ]
        sum(output.sum() for output in reference_outputs).backward()

        assert shared is not None and shared.key.grad_fn is not None
        torch.testing.assert_close(scale.grad, reference_scale.grad)
        for actual, expected in zip(hidden, reference_hidden):
            torch.testing.assert_close(actual.grad, expected.grad)
    finally:
        Utils.destroy_model_parallel()


def test_mtp_block_passes_one_source_state_through_all_repeated_iterations():
    observed = []
    sequence_roll_context = object()

    class FakeRepeatedLayer:
        def __call__(
            self,
            input_ids,
            position_ids,
            hidden_states,
            padding_mask=None,
            sequence_roll_context=None,
            roll_depth=0,
            mtp_dsa_context=None,
            **_kwargs,
        ):
            assert mtp_dsa_context is not None
            observed.append(
                (
                    mtp_dsa_context.iteration,
                    mtp_dsa_context.shared_tensors,
                    sequence_roll_context,
                    roll_depth,
                )
            )
            if mtp_dsa_context.is_source:
                topk = torch.zeros((1, hidden_states.size(0), 1), dtype=torch.int64)
                mtp_dsa_context.capture(hidden_states * 2, topk, None)
                shared = mtp_dsa_context.require_source_tensors()
            else:
                shared = mtp_dsa_context.shared_tensors
            return hidden_states + 1, input_ids, position_ids, padding_mask, shared

    block = SimpleNamespace(
        config=SimpleNamespace(
            pipeline_model_parallel_size=1,
            mtp_num_layers=7,
            mtp_detach_heads=False,
            dsa_mtp_index_kv_share=True,
        ),
        vp_stage=None,
        mtp_use_repeated_layer=True,
        layers=[FakeRepeatedLayer()],
    )
    hidden_states = torch.randn(4, 1, 3)
    output = MultiTokenPredictionBlock.forward(
        block,
        input_ids=torch.arange(4).view(1, 4),
        position_ids=torch.arange(4).view(1, 4),
        hidden_states=hidden_states,
        attention_mask=None,
        sequence_roll_context=sequence_roll_context,
    )

    assert output.shape == (32, 1, 3)
    assert [iteration for iteration, *_ in observed] == list(range(7))
    source_state = observed[1][1]
    assert source_state is not None
    assert observed[0][1] is None
    assert all(shared is source_state for _, shared, *_ in observed[1:])
    assert all(context is sequence_roll_context for *_, context, _ in observed)
    assert [roll_depth for *_, roll_depth in observed] == list(range(7))


@pytest.mark.parametrize("full_recompute", [False, True])
def test_real_mtp_block_threads_sharing_through_transformer_and_absorbed_dsa(
    monkeypatch, full_recompute
):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(5678)
        torch.manual_seed(5678)
        torch.cuda.manual_seed(5678)

        recompute_overrides = {}
        if full_recompute:
            recompute_overrides = {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            }
        config = _make_config(**recompute_overrides)
        decoder_layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(
            config=config, backend=TESpecProvider()
        )
        mtp_spec = get_gpt_mtp_block_spec(
            config=config, spec=decoder_layer_specs[-1], use_transformer_engine=True
        )
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_spec).bfloat16().cuda()
        attention = mtp.layers[0].mtp_model_layer.self_attention

        call_counts = {"q": 0, "kv": 0, "indexer": 0, "sparse": 0}
        hooks = [
            attention.linear_q_down_proj.register_forward_hook(
                lambda *_args: call_counts.__setitem__("q", call_counts["q"] + 1)
            ),
            attention.linear_kv_down_proj.register_forward_hook(
                lambda *_args: call_counts.__setitem__("kv", call_counts["kv"] + 1)
            ),
            attention.core_attention.indexer.linear_wq_b.register_forward_hook(
                lambda *_args: call_counts.__setitem__("indexer", call_counts["indexer"] + 1)
            ),
        ]
        original_sparse_attention = dsa_module._run_sparse_attention

        def counted_sparse_attention(*args, **kwargs):
            call_counts["sparse"] += 1
            return original_sparse_attention(*args, **kwargs)

        monkeypatch.setattr(dsa_module, "_run_sparse_attention", counted_sparse_attention)

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
        output[total_tokens:].float().square().mean().backward()

        assert output.shape == (total_tokens * (config.mtp_num_layers + 1), 1, config.hidden_size)
        assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
        assert embedding.weight.grad is not None and torch.isfinite(embedding.weight.grad).all()
        recompute_multiplier = 2 if full_recompute else 1
        assert call_counts == {
            "q": 7 * recompute_multiplier,
            "kv": 1 * recompute_multiplier,
            "indexer": 1 * recompute_multiplier,
            "sparse": 7 * recompute_multiplier,
        }
        for hook in hooks:
            hook.remove()
    finally:
        Utils.destroy_model_parallel()

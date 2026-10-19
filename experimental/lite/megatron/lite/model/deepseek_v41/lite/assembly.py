# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 source/module composition; all kernels and collectives remain in Core."""
from types import SimpleNamespace

import torch
from torch import nn

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
    rotate_activation,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    use_fused_dsa_kernels,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.lite.model.deepseek_v41.lite.compressor import DS41Compressor
from megatron.lite.model.deepseek_v41.lite.index import IndexProjection
from megatron.lite.model.deepseek_v41.lite.native import (
    attach_ratio_two_sources,
    freeze_indexer_sources,
)
from megatron.lite.model.deepseek_v41.lite.ratio_one import RatioOneAttention
from megatron.lite.model.deepseek_v41.lite.sources import DS41IndexKey, publish_latent


class FrozenIndexerCSA(CompressedSparseAttention):
    """Core 7b150acf: zero auxiliary loss uses one forward layout, with autograd."""

    def train(self, mode=True):
        super().train(mode)
        self.training = bool(mode and self.config.dsa_indexer_loss_coeff)
        return self


class ModelNorm(WrappedTorchNorm):
    def __new__(cls, config, hidden_size, eps, **kwargs):
        return (
            super().__new__(cls, config, hidden_size, eps, **kwargs).to(dtype=config.params_dtype)
        )


class FP32Projection(nn.Module):
    """Release main-compressor projections accumulate from FP32 input/weights."""

    def __init__(self, input_size, output_size, *, config, init_method, **unused):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                output_size, input_size, dtype=torch.float32, device=torch.cuda.current_device()
            )
        )
        if config.perform_initialization:
            init_method(self.weight)

    def forward(self, x):
        return torch.nn.functional.linear(x.float(), self.weight), None

    def backward_dw(self):
        pass  # Normal torch autograd; delayed weight gradients are rejected.


def remote_compressor(config, group, rotary):
    """Non-owning Core metadata for a KV source on an earlier PP stage.

    Cached forward outputs arrive as explicit pipeline tensors. No projection
    is run here and no duplicate owner parameter is registered.
    """
    result = DS41Compressor.__new__(DS41Compressor)
    nn.Module.__init__(result)
    result.config, result.compress_ratio, result.head_dim = config, 2, config.v_head_dim
    result.pg_collection = SimpleNamespace(cp=group)
    result.rotary_pos_emb = rotary
    for name in ("linear_wkv", "linear_wgate"):
        object.__setattr__(
            result, name, SimpleNamespace(weight=torch.empty(0, dtype=torch.float32))
        )
    return result


def compressor_reader(owner, state, layer, source):
    """Parameter-free view of a native owner's source/layout metadata.

    Only DS41Compressor's source overrides execute on a reader. Core's
    _cp_requires_single_projection still sees the actual owner's weights.
    """
    view = DS41Compressor.__new__(DS41Compressor)
    nn.Module.__init__(view)
    for name, value in vars(owner).items():
        if name not in {"_parameters", "_buffers", "_modules"}:
            object.__setattr__(view, name, value)
    for members in (owner._parameters, owner._buffers, owner._modules):
        for name, value in members.items():
            object.__setattr__(view, name, value)
    view.bind_source(state, layer, source)
    return view


class SourceContext:
    def __init__(self, release, config, groups, state, policies):
        self.release, self.config, self.groups, self.state = release, config, groups, state
        self.policies, self.main_owners = policies, {}


class DS41AttentionSources(nn.Module):
    """Core Hybrid's core_attention ModuleSpec, with DS4.1 state sources.

    Ratio0/2 call an unmodified CompressedSparseAttention.forward. Ratio1
    calls the explicitly approved DS4.1 orchestration over Core operators.
    """

    def __init__(
        self,
        *,
        config,
        context,
        layer_number,
        compress_ratio,
        rotary_pos_emb,
        pg_collection,
        **kwargs,
    ):
        super().__init__()
        self.context, self.config, self.layer = context, config, layer_number - 1
        self.ratio, self.rotary = compress_ratio, rotary_pos_emb
        policy = context.policies[self.layer]
        self.policy = policy
        self.group = pg_collection.cp
        state = context.state
        if self.ratio in (0, 2):
            self.native = FrozenIndexerCSA(
                config=config,
                submodules=CompressedSparseAttentionSubmodules(),
                layer_number=layer_number,
                compress_ratio=self.ratio,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
                **kwargs,
            ).train()
        if self.ratio == 0:
            return
        if self.ratio == 2:
            if self.layer == policy.kv:
                main = DS41Compressor(
                    config,
                    CompressorSubmodules(FP32Projection, FP32Projection, ModelNorm),
                    2,
                    config.v_head_dim,
                    rotary_pos_emb=rotary_pos_emb,
                    pg_collection=pg_collection,
                )
                main.ape.data.zero_()
                main.ape.requires_grad_(False)  # DS4.1 has no compressor APE.
                main.bind_source(state, self.layer, policy.kv)
                publish_latent(main, state, policy.kv)
                context.main_owners[policy.kv] = main
            else:
                main = compressor_reader(
                    context.main_owners.get(policy.kv)
                    or remote_compressor(config, pg_collection.cp, rotary_pos_emb),
                    state,
                    self.layer,
                    policy.kv,
                )
            indexer = CSAIndexer(
                config,
                CSAIndexerSubmodules(
                    TELinear,
                    TELinear,
                    ModuleSpec(
                        DS41IndexKey,
                        params=dict(
                            main_compressor=main, state=state, layer=self.layer, owner=policy.kv
                        ),
                    ),
                ),
                2,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
            )
            for attribute, channel in [
                ("linear_wq_b", "index_q"),
                ("linear_weights_proj", "index_weights"),
            ]:
                original = getattr(indexer, attribute) if self.layer == policy.index else None
                setattr(
                    indexer,
                    attribute,
                    IndexProjection(original, state, self.layer, policy.index, channel),
                )
            attach_ratio_two_sources(self.native, compressor=main, indexer=indexer)
        elif self.ratio == 1:

            def projection(inputs, outputs):
                return TELinear(
                    inputs,
                    outputs,
                    parallel_mode="duplicated",
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                )

            if self.layer == policy.kv:
                # Release ratio-one wkv is a BF16 linear; only ratio > 1 is FP32.
                self.main_projection = projection(config.hidden_size, config.v_head_dim)
                self.main_norm = nn.RMSNorm(
                    config.v_head_dim, eps=config.layernorm_epsilon, dtype=config.params_dtype
                )
                self.index_key = nn.Sequential(
                    nn.Linear(
                        config.v_head_dim,
                        config.dsa_indexer_head_dim,
                        bias=False,
                        dtype=config.params_dtype,
                    ),
                    nn.RMSNorm(
                        config.dsa_indexer_head_dim,
                        eps=config.layernorm_epsilon,
                        dtype=config.params_dtype,
                    ),
                )
                freeze_indexer_sources(self.index_key, loss_coefficient=0)

            self.index_q = IndexProjection(
                (
                    projection(
                        config.q_lora_rank, config.dsa_indexer_n_heads * config.dsa_indexer_head_dim
                    )
                    if self.layer == policy.index
                    else None
                ),
                state,
                self.layer,
                policy.index,
                "index_q",
            )
            self.index_weights = IndexProjection(
                (
                    projection(config.hidden_size, config.dsa_indexer_n_heads)
                    if self.layer == policy.index
                    else None
                ),
                state,
                self.layer,
                policy.index,
                "index_weights",
            )
            freeze_indexer_sources(self.index_q, loss_coefficient=0)
            freeze_indexer_sources(self.index_weights, loss_coefficient=0)
            self.native = RatioOneAttention(
                config,
                group=self.group,
                rotary_pos_emb=rotary_pos_emb,
                layer=self.layer,
                index_owner=policy.index,
                candidate_owner=context.release.candidate_source_layer_id,
                state=state,
                use_flash=use_fused_dsa_kernels(config),
            )
        else:
            raise ValueError("DS4.1 supports literal ratios zero, one and two")

    def _rope(self, value, params):
        prefix = params.cu_seqlens_q_padded
        if prefix is None:
            prefix = params.cu_seqlens_q
        freqs = self.rotary(int(params.max_seqlen_q), packed_seq=True)
        if isinstance(freqs, tuple):
            freqs = freqs[0]
        return cp_utils.apply_thd_cp_local_rope_unfused(
            value,
            freqs,
            value.shape[-1] - self.config.qk_pos_emb_head_dim,
            self.config.qk_pos_emb_head_dim,
            prefix,
            self.group.rank() * value.shape[0],
            self.config,
        )

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        *,
        x,
        qr,
        packed_seq_params,
        boundary_hidden=None,
        boundary_kv=None,
        **kwargs,
    ):
        params, state, policy = packed_seq_params, self.context.state, self.policy
        if self.ratio == 0:
            return self.native(
                query,
                key,
                value,
                attention_mask,
                x=x,
                qr=qr,
                packed_seq_params=params,
                boundary_hidden=boundary_hidden,
                boundary_kv=boundary_kv,
                **kwargs,
            )
        if self.layer == policy.kv:
            state.values[("attention_inputs", policy.kv)] = (x, boundary_hidden, params)
        if self.ratio == 2:
            return self.native(
                query,
                key,
                value,
                attention_mask,
                x=x,
                qr=qr,
                packed_seq_params=params,
                boundary_hidden=boundary_hidden,
                boundary_kv=boundary_kv,
                **kwargs,
            )
        prefix = params.cu_seqlens_q_padded
        if prefix is None:
            prefix = params.cu_seqlens_q
        geometry = ("ratio1_kv", tuple(x.shape), prefix.data_ptr(), state.grad_mode)

        def main_source():
            latent = self.main_norm(self.main_projection(x)[0].to(x.dtype))
            return latent, self._rope(latent, params)

        latent, main = state.resolve(self.layer, policy.kv, geometry, main_source)
        index_k = state.resolve(
            self.layer,
            policy.kv,
            ("ratio1_index_k", geometry),
            lambda: rotate_activation(self._rope(self.index_key(latent.detach()), params)),
        )
        q = self.index_q(qr.detach())[0].reshape(
            x.shape[0], self.config.dsa_indexer_n_heads, self.config.dsa_indexer_head_dim
        )
        q = rotate_activation(self._rope(q, params))
        weights = self.index_weights(x.detach())[0].squeeze(1) * (
            self.config.dsa_indexer_head_dim**-0.5 * self.config.dsa_indexer_n_heads**-0.5
        )
        return self.native(
            query,
            key.reshape(key.shape[0], -1),
            main.squeeze(1),
            q,
            index_k.squeeze(1),
            weights,
            params,
        )

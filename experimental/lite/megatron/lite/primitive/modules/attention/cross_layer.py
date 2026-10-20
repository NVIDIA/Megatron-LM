# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cross-layer Core compressor and index-projection source delegation."""
from typing import Any

import torch
from torch import nn

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    Compressor,
    _apply_rope,
    _apply_unfused_rope,
    rotate_activation,
)
from megatron.lite.primitive.parallel.source_state import KVSourceState


class FrozenIndexerCSA(CompressedSparseAttention):
    """Core 7b150acf: zero auxiliary loss uses one forward layout, with autograd."""

    def train(self, mode=True):
        super().train(mode)
        self.training = bool(mode and self.config.dsa_indexer_loss_coeff)
        return self


class SharedCompressor(Compressor):
    """Select owner outputs without changing Core's layout or CP collectives.

    Core contract: dev 7b150acfacee9d1126025f6cbe0f49c588975c51,
    Compressor.forward and Compressor._forward_thd_cp. Signature tests protect
    both overrides. The parent remains responsible for every compression op.
    """

    def bind_source(self, state: KVSourceState, layer: int, owner: int) -> None:
        if not 0 <= owner <= layer:
            raise ValueError("KV source must be this layer or an earlier owner")
        self.source_state, self.source_layer, self.source_owner = state, layer, owner

    def _resolve_source(self, x, prefix, mode, max_length, global_start, produce):
        if not hasattr(self, "source_state"):
            raise RuntimeError("shared-attention compressor has no bound KV source")
        geometry = (
            mode,
            tuple(x.shape),
            x.dtype,
            x.device,
            self.compress_ratio,
            self.head_dim,
            prefix.data_ptr() if prefix is not None else None,
            max_length,
            global_start,
            self.source_state.grad_mode,
            id(self.pg_collection.cp),
        )
        return self.source_state.resolve(self.source_layer, self.source_owner, geometry, produce)

    def forward(self, x: torch.Tensor, packed_seq_params: PackedSeqParams | None = None) -> Any:
        prefix = (
            None
            if packed_seq_params is None
            else (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
        )
        parent = super().forward
        return self._resolve_source(
            x,
            prefix,
            "forward",
            None if packed_seq_params is None else packed_seq_params.max_seqlen_q,
            None,
            lambda: parent(x, packed_seq_params),
        )

    def _forward_thd_cp(
        self, x, boundary_hidden, cu_seqlens, layout, max_seqlen_q, pre_compacted_hidden=None
    ):
        parent = super()._forward_thd_cp
        return self._resolve_source(
            x,
            cu_seqlens,
            "thd_cp",
            max_seqlen_q,
            layout.global_start,
            lambda: parent(
                x,
                boundary_hidden,
                cu_seqlens,
                layout,
                max_seqlen_q,
                pre_compacted_hidden=pre_compacted_hidden,
            ),
        )


class IndexProjection(nn.Module):
    """Preserve owner q/weights without registering reader projection parameters.

    Core dev 7b150acf CSAIndexer uses linear_wq_b.forward and
    linear_weights_proj.forward before calling cp_utils.compute_cp_indexer_topk.
    Their tuple-return contract is tested with the native scorer. Balanced
    indexer projection and quantized projection modes need separate validation.
    """

    def __init__(
        self, linear: nn.Module | None, state: KVSourceState, layer: int, owner: int, channel: str
    ) -> None:
        super().__init__()
        if not 0 <= owner <= layer or channel not in {"index_q", "index_weights"}:
            raise ValueError("Invalid shared-attention index source or projection channel")
        if (linear is not None) != (layer == owner):
            raise ValueError("Only the index owner may register a projection")
        self.linear = linear
        self.state, self.layer, self.owner, self.channel = state, layer, owner, channel

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        geometry = (self.channel, tuple(x.shape), x.dtype, x.device, self.state.grad_mode)
        return self.state.resolve(
            self.layer, self.owner, geometry, lambda: self.linear(x, *args, **kwargs)
        )

    def backward_dw(self) -> None:
        """Leave deferred owner weight gradients to the registered owner only."""
        if self.linear is not None and hasattr(self.linear, "backward_dw"):
            self.linear.backward_dw()


def publish_latent(compressor, state, owner):
    """Capture the native norm result, before Core applies main-KV RoPE.

    Core dev 7b150acf Compressor._forward_thd calls norm once before RoPE.
    No compressed row mapping, pooling or CP communication is implemented here.
    """

    def capture(module, inputs, output):
        if not state.active:
            raise RuntimeError("shared-attention compressor latent requires an active microbatch")
        state.values[("main_latent", owner, state.grad_mode)] = output

    compressor.norm.register_forward_hook(capture)


class LatentIndexKey(nn.Module):
    """Replace CSAIndexer.compressor's state source, not its native CP consumer.

    Core dev 7b150acf forward/_forward_thd_cp input/output contracts are
    preserved. Native CSA still owns index-K gathering, row layout, top-k and
    attention. Each KV owner owns one index-key projection; readers have none.
    """

    def __init__(self, main_compressor, state, layer, owner, *, config, rotary_pos_emb, **unused):
        super().__init__()
        # A reference to an already registered owner is not another parameter
        # owner. In particular freezing the indexer must not freeze main KV.
        object.__setattr__(self, "main", main_compressor)
        self.state, self.layer, self.owner = state, layer, owner
        self.config, self.rotary_pos_emb = config, rotary_pos_emb
        self.compress_ratio = main_compressor.compress_ratio
        self.head_dim = config.dsa_indexer_head_dim
        if layer == owner:
            device = main_compressor.linear_wkv.weight.device
            self.wk = nn.Linear(
                config.v_head_dim,
                self.head_dim,
                bias=False,
                dtype=config.params_dtype,
                device=device,
            )
            self.norm = nn.RMSNorm(
                self.head_dim,
                eps=config.layernorm_epsilon,
                dtype=config.params_dtype,
                device=device,
            )

    def _cp_requires_single_projection(self):
        return self.main._cp_requires_single_projection()

    def backward_dw(self):
        # This projection uses normal torch autograd. Delayed wgrad is not a
        pass

    def _resolve(self, x, prefix, maximum, mode, produce):
        geometry = (
            "index_k",
            mode,
            tuple(x.shape),
            x.dtype,
            x.device,
            prefix.data_ptr(),
            maximum,
            self.state.grad_mode,
        )
        return self.state.resolve(self.layer, self.owner, geometry, produce)

    def _project(self):
        latent = self.state.values[("main_latent", self.owner, self.state.grad_mode)]
        return self.norm(self.wk(latent))

    def forward(self, x, packed_seq_params=None):
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("shared-attention index source requires packed THD")
        prefix = packed_seq_params.cu_seqlens_q_padded
        if prefix is None:
            prefix = packed_seq_params.cu_seqlens_q
        maximum = packed_seq_params.max_seqlen_q

        def produce():
            original, boundary, params = self.state.values[("attention_inputs", self.owner)]
            _, compressed_prefix = self.main(original, params)
            key = self._project()
            key = _apply_rope(
                key,
                self.head_dim - self.config.qk_pos_emb_head_dim,
                self.config.qk_pos_emb_head_dim,
                self.rotary_pos_emb,
                self.config,
                rotary_seq_len=0,
                ratio=self.compress_ratio,
                cp_group=self.main.pg_collection.cp,
                cu_seqlens=compressed_prefix,
                max_seqlen_rope=(maximum // self.compress_ratio) * self.compress_ratio,
            )
            return rotate_activation(key), compressed_prefix

        return self._resolve(x, prefix, maximum, "thd", produce)

    def _forward_thd_cp(
        self, x, boundary_hidden, cu_seqlens, layout, max_seqlen_q, pre_compacted_hidden=None
    ):
        def produce():
            original, boundary, params = self.state.values[("attention_inputs", self.owner)]
            _, compressed_prefix = self.main._forward_thd_cp(
                original, boundary, cu_seqlens, layout, max_seqlen_q, pre_compacted_hidden=None
            )
            key = self._project()
            frequencies = self.rotary_pos_emb(max_seqlen_q, packed_seq=True)
            if isinstance(frequencies, tuple):
                frequencies = frequencies[0]
            frequencies = frequencies.index_select(0, layout.position_ids.long())
            key = _apply_unfused_rope(
                key,
                frequencies,
                self.head_dim - self.config.qk_pos_emb_head_dim,
                self.config.qk_pos_emb_head_dim,
                self.config,
                None,
                self.main.pg_collection.cp,
            )
            return rotate_activation(key), compressed_prefix

        return self._resolve(x, cu_seqlens, max_seqlen_q, "thd_cp", produce)


def compressor_reader(owner, state, layer, source):
    """Parameter-free view of a native owner's source/layout metadata."""
    view = SharedCompressor.__new__(SharedCompressor)
    nn.Module.__init__(view)
    for name, value in vars(owner).items():
        if name not in {"_parameters", "_buffers", "_modules"}:
            object.__setattr__(view, name, value)
    for members in (owner._parameters, owner._buffers, owner._modules):
        for name, value in members.items():
            object.__setattr__(view, name, value)
    view.bind_source(state, layer, source)
    return view

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 index keys come from the main compressor's pre-RoPE latent."""
import torch
from torch import nn

from megatron.core.transformer.experimental_attention_variant.csa import (
    _apply_rope,
    _apply_unfused_rope,
    rotate_activation,
)


def publish_latent(compressor, state, owner):
    """Capture the native norm result, before Core applies main-KV RoPE.

    Core dev 7b150acf Compressor._forward_thd calls norm once before RoPE.
    No compressed row mapping, pooling or CP communication is implemented here.
    """

    def capture(module, inputs, output):
        if not state.active:
            raise RuntimeError("DS4.1 compressor latent requires an active microbatch")
        state.values[("main_latent", owner, state.grad_mode)] = output

    compressor.norm.register_forward_hook(capture)


class DS41IndexKey(nn.Module):
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
        # supported DS4.1 build configuration.
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
            raise ValueError("DS4.1 index source requires packed THD")
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

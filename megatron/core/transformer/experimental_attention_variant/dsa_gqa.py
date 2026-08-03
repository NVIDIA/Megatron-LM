# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GQA-flavored DeepSeek Sparse Attention (DSA) building blocks for `main`.

Main's shipped DSA path (`dsa.py` + `dsa_cudnn_kernels.py`) is written for the
absorbed-MLA layout: the indexer's query projection consumes the MLA low-rank
query `qr`, and the sparse-attention *output* runs through FlashMLA (single latent
KV head). This module provides the two pieces needed to run the same DSA machinery
over plain **GQA** Q/K/V instead:

  * :class:`DSGQAIndexer` — a drop-in indexer whose query projection reads the
    transformer hidden states (``x``) directly instead of the MLA ``qr``. K and
    weights already come from ``x`` in main's :class:`DSAIndexer`, so this only
    swaps the Q projection (``linear_wq_b(qr)`` -> ``linear_q(x)``) and applies
    RoPE over the full indexer head dim (no MLA ``qk_pos_emb_head_dim`` split).

  * :class:`DSGQASelfAttention` — a :class:`SelfAttention` subclass that feeds the
    DSA core its required ``x``/``qr`` inputs. A plain ``SelfAttention`` never
    supplies them (only MLA's overridden forward does), so this mirrors the
    sanctioned ``requires_dsa_inputs`` channel and passes ``x = qr = hidden_states``
    (the GQA indexer reads ``x`` and ignores ``qr``).

The DSA *core* is unchanged: wire a plain :class:`DSAttention` with a
:class:`DSGQAIndexer` submodule. Because ``value`` is a real GQA value tensor
(not ``None``), ``DSAttention.forward`` takes its non-absorbed path: cuDNN indexer
top-k (``run_fused_qk_topk`` — GQA-safe) with a PyTorch fallback, the PyTorch
reference indexer loss, and the PyTorch ``unfused_dsa_fn`` sparse-attention output.
A Triton attention output can be swapped in later; this is the option-1 first cut.
"""
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAttention,
    rotate_activation,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class DSGQAIndexerSubmodules:
    """Submodules for :class:`DSGQAIndexer`.

    Mirrors :class:`~megatron.core.transformer.experimental_attention_variant.dsa.DSAIndexerSubmodules`
    but replaces the MLA ``linear_wq_b`` (query-from-``qr``) with ``linear_q``
    (query-from-hidden).

    Args:
        linear_q: Linear projection for the indexer query, from hidden states.
        linear_wk: Linear projection for the indexer key, from hidden states.
        k_norm: Normalization applied to the indexer key.
        linear_weights_proj: Linear projection for the per-head index weights.
    """

    linear_q: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


class DSGQAIndexer(DSAIndexer):
    """DSA lightning indexer for GQA: indexer query is projected from hidden states.

    Reuses :class:`DSAIndexer`'s ``_apply_rope`` / ``forward`` / ``forward_with_scores``.
    Overrides ``__init__`` (to avoid the MLA-only ``qk_pos_emb_head_dim`` /
    ``q_lora_rank`` config reads) and ``forward_before_topk`` (Q from ``x``).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSGQAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        # NB: intentionally skip DSAIndexer.__init__ — it reads MLA-only config
        # fields (qk_pos_emb_head_dim, q_lora_rank) that are absent on a plain GQA
        # TransformerConfig. Initialize MegatronModule directly and build the
        # GQA-appropriate submodules here.
        MegatronModule.__init__(self, config=config)
        self.hidden_size = self.config.hidden_size
        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk
        # No MLA positional/no-positional split: the indexer applies RoPE over its
        # full head dim. Setting this makes the inherited _apply_rope a no-op split
        # (x_nope has width 0).
        self.qk_pos_emb_head_dim = self.index_head_dim
        self.softmax_scale: float = self.index_head_dim**-0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        # rope_type / rotary_* are MLATransformerConfig fields; a plain GQA
        # TransformerConfig lacks them (the GQA model gets its rope params from
        # args, not config). Resolve with getattr defaults matching the standard
        # RoPE setup.
        self.rope_type = getattr(self.config, "rope_type", None) or "rope"
        rotary_percent = getattr(self.config, "rotary_percent", 1.0)
        rotary_base = getattr(self.config, "rotary_base", 10000)

        # Position embedding (over the full indexer head dim).
        if self.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=rotary_percent,
                rotary_base=rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        # Indexer query from hidden states (the GQA change vs. DSAIndexer.linear_wq_b).
        self.linear_q = build_module(
            submodules.linear_q,
            self.hidden_size,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        k_norm_eps = (
            self.config.dsa_indexer_k_norm_epsilon
            if self.config.dsa_indexer_k_norm_epsilon is not None
            else self.config.layernorm_epsilon
        )
        self.k_norm = build_module(
            submodules.k_norm, config=k_norm_config, hidden_size=self.index_head_dim, eps=k_norm_eps
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        # Indexer projections are duplicated across tensor-parallel ranks, so their
        # gradients should be averaged during final gradient synchronization.
        for param in self.parameters():
            setattr(param, "average_gradients_across_tp_domain", True)

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All indexer computations before top-k. GQA: query is projected from ``x``.

        ``qr`` is accepted for interface parity with ``DSAIndexer`` (DSAttention
        passes it) but ignored — the GQA indexer has no MLA low-rank query.
        """
        del qr  # GQA indexer projects the query from hidden states, not from qr.
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        # RoPE params.
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        if packed_seq:
            cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # Gather sequence-parallel shards (only x is needed for GQA).
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)

        seqlen, bsz, _ = x.size()

        # Query linear (from hidden) + RoPE.
        # [seqlen, batch, hidden] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.linear_q(x)
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_q)

        # Key linear (from hidden) + norm + RoPE.
        # [seqlen, batch, hidden] -> [seqlen, batch, index_head_dim]
        k, _ = self.linear_wk(x)
        if self.config.dsa_indexer_k_norm_fp32:
            k_dtype = k.dtype
            k = self.k_norm(k.float()).to(dtype=k_dtype)
        else:
            k = self.k_norm(k)
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_kv)
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        if self.config.dsa_indexer_rotate_activation:
            q = rotate_activation(q)
            k = rotate_activation(k)

        # Per-head index weights (from hidden).
        # [seqlen, batch, hidden] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        return q, k, weights


class DSGQAttention(DSAttention):
    """DSA core that supplies its own ``x``/``qr`` from stashed hidden states.

    Main's base ``Attention.forward`` calls the core attention inline as
    ``core_attention(query, key, value, attention_mask, ...)`` — it never passes
    the DSA ``x``/``qr`` (only MLA's overridden forward does). Rather than
    reimplement that forward, this subclass makes ``x``/``qr`` optional and fills
    them from ``_dsa_hidden_states`` (stashed by :class:`DSGQASelfAttention` during
    QKV projection) when the caller omits them. Works for both the static and
    activation-checkpointed core-call paths.
    """

    def forward(self, query, key, value, attention_mask, x=None, qr=None, **kwargs):
        """Fill x/qr from stashed hidden states and expand GQA K/V to full heads."""
        if x is None:
            x = getattr(self, "_dsa_hidden_states", None)
        if qr is None:
            qr = x  # GQA indexer reads x and ignores qr; a real tensor is enough.
        if x is None:
            raise RuntimeError(
                "DSGQAttention has no hidden states to use as DSA x/qr. Expected "
                "DSGQASelfAttention to stash them via get_query_key_value_tensors."
            )
        # Main's reference DSA path (compute_dsa_indexer_loss + unfused_dsa_fn)
        # requires one KV head per query head (MQA nk==1 or MHA nk==np); it does
        # not do genuine GQA head-grouping. Expand GQA K/V [s, b, nk, h] ->
        # [s, b, np, h] by repeating each group's head (contiguous grouping: query
        # head i attends KV group i // (np // nk)). NB: this materializes MHA-sized
        # K/V — the memory-efficient GQA path is the (later) Triton kernel.
        if value is not None:
            nq = query.size(2)
            nk = key.size(2)
            if nk != nq and nk > 0 and nq % nk == 0:
                rep = nq // nk
                key = key.repeat_interleave(rep, dim=2)
                value = value.repeat_interleave(rep, dim=2)
        return super().forward(query, key, value, attention_mask, x, qr, **kwargs)


class DSGQASelfAttention(SelfAttention):
    """GQA self-attention that feeds the DSA core its required ``x``/``qr`` inputs.

    A plain ``SelfAttention`` never supplies the DSA core with hidden states; only
    MLA's overridden forward does. This subclass stashes ``hidden_states`` on the
    DSA core during QKV projection so :class:`DSGQAttention` can supply them as
    ``x``/``qr``. Pair it with a :class:`DSGQAttention` core.
    """

    def get_query_key_value_tensors(self, hidden_states, *args, **kwargs):
        """Stash hidden states on the DSA core for use as its x/qr inputs."""
        if getattr(self.core_attention, "requires_dsa_inputs", False):
            self.core_attention._dsa_hidden_states = hidden_states
        return super().get_query_key_value_tensors(hidden_states, *args, **kwargs)

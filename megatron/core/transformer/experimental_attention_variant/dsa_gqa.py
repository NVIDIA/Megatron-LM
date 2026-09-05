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
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAttention,
    rotate_activation,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


@dataclass
class DSGQAIndexerSubmodules:
    """Submodules for :class:`DSGQAIndexer`.

    Mirrors ``DSAIndexerSubmodules`` from
    :mod:`megatron.core.transformer.experimental_attention_variant.dsa`
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
        # RoPE dim the min-memory kernels apply over the indexer head (branch uses
        # int(index_head_dim * rotary_percent); == index_head_dim for percent=1.0).
        self.index_rotary_dim = self.index_head_dim

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
        # The min-memory kernels read `indexer.linear_k` (they pull raw .weight).
        self.linear_k = self.linear_wk

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
        """Dispatch to the min-memory kernels (default) or main's reference DSA path.

        Backend selected by ``config.dsa_gqa_kernel``.
        - min_memory: the streamed GQA kernels (genuine GQA, no K/V expansion, O(tile)
          memory); ``dsa_min_memory_use_triton`` / ``dsa_min_memory_use_cudnn`` pick the
          fast backends (default off -> PyTorch reference implementation).
        - reference: main's DSAttention path (dense teacher + unfused_dsa_fn) with GQA
          K/V expanded to full heads. Kept for numerical A/B; OOMs at long seq.
        """
        if x is None:
            x = getattr(self, "_dsa_hidden_states", None)
        if qr is None:
            qr = x  # reference path: GQA indexer reads x and ignores qr.
        if x is None:
            raise RuntimeError(
                "DSGQAttention has no hidden states to use as DSA x/qr. Expected "
                "DSGQASelfAttention to stash them via get_query_key_value_tensors."
            )

        if getattr(self.config, "dsa_gqa_kernel", "min_memory") == "min_memory":
            return self._forward_min_memory(
                query,
                key,
                value,
                x,
                attn_mask_type=kwargs.get("attn_mask_type"),
                attention_bias=kwargs.get("attention_bias"),
                packed_seq_params=kwargs.get("packed_seq_params"),
            )

        # --- reference path: expand GQA K/V to full heads, use main's DSAttention ---
        # compute_dsa_indexer_loss + unfused_dsa_fn require one KV head per query head
        # (MQA nk==1 or MHA nk==np). Expand [s, b, nk, h] -> [s, b, np, h] (contiguous
        # grouping). Materializes MHA-sized K/V.
        if value is not None:
            nq = query.size(2)
            nk = key.size(2)
            if nk != nq and nk > 0 and nq % nk == 0:
                rep = nq // nk
                key = key.repeat_interleave(rep, dim=2)
                value = value.repeat_interleave(rep, dim=2)
        return super().forward(query, key, value, attention_mask, x, qr, **kwargs)

    def _forward_min_memory(
        self, query, key, value, x, attn_mask_type=None, attention_bias=None, packed_seq_params=None
    ):
        """Run the branch's streamed min-memory GQA-DSA kernels.

        Genuine GQA (no K/V head expansion). Returns a single tensor with the indexer
        KL loss attached via DSAIndexerLossAutoScaler (identity forward, gradient inject).
        """

        from megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexerLossAutoScaler,
            DSAIndexerLossLoggingHelper,
        )
        from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
            _cudnn_available_for_indexer,
            dsa_dense_indexer_loss,
            dsa_min_memory_gqa,
            dsa_min_memory_gqa_forward_only,
        )

        if attention_bias is not None:
            raise NotImplementedError(
                "DSGQAttention min-memory path does not support attention_bias."
            )
        if packed_seq_params is not None:
            raise NotImplementedError(
                "DSGQAttention min-memory path does not support packed sequences."
            )
        if attn_mask_type is not None and attn_mask_type != AttnMaskType.causal:
            raise NotImplementedError("DSGQAttention min-memory path supports causal masking only.")

        cfg = self.config
        use_triton = bool(getattr(cfg, "dsa_min_memory_use_triton", False))
        # dsa_use_cudnn is the pre-existing field on this path; keep honouring it.
        use_cudnn = bool(getattr(cfg, "dsa_min_memory_use_cudnn", False)) or bool(
            getattr(cfg, "dsa_use_cudnn", False)
        )
        # Indexer applies its own RoPE (rotary_pos_emb always built); enable it.
        use_indexer_rope = True
        qcs = getattr(cfg, "dsa_kernel_query_block_size", None)
        kcs = getattr(cfg, "dsa_kernel_key_block_size", None)
        sparse_loss = getattr(cfg, "dsa_indexer_use_sparse_loss", False)
        coeff = getattr(cfg, "dsa_indexer_loss_coeff", 0.0) or 0.0

        # One-time backend report (per layer/process) so the effective backend is
        # unambiguous — cudnn_indexer_active tells you whether cuDNN actually engaged
        # (it silently falls back to the PyTorch oracle if `from cudnn import DSA`
        # fails or index_n_heads is not in {32,64}). NB: config.dsa_kernel_backend is
        # NOT read by this path; only these env-derived flags matter.
        if not getattr(self, "_dsa_backend_logged", False):
            self._dsa_backend_logged = True
            from megatron.core.transformer.experimental_attention_variant import (
                dsa_min_memory_triton,
            )

            cudnn_active = _cudnn_available_for_indexer(use_cudnn, self.indexer.index_n_heads)
            # triton_active = the request AND Triton actually importable; if False while
            # use_triton=True, the Triton kernels silently fell back to the PyTorch oracle.
            triton_active = bool(use_triton and dsa_min_memory_triton.HAVE_TRITON)
            log_single_rank(
                logger,
                logging.INFO,
                f"[DSGQAttention layer{self.layer_number}] min-memory backend: "
                f"use_triton={use_triton} triton_active={triton_active} "
                f"use_cudnn={use_cudnn} cudnn_indexer_active={cudnn_active}",
            )

        if not torch.is_grad_enabled():
            return dsa_min_memory_gqa_forward_only(
                query=query,
                key=key,
                value=value,
                hidden_states=x.detach(),
                indexer=self.indexer,
                softmax_scale=self.softmax_scale,
                use_indexer_rope=use_indexer_rope,
                query_chunk_size=qcs,
                key_chunk_size=kcs,
                cache_indexer_k=getattr(cfg, "dsa_kernel_cache_indexer_k", False),
                use_triton=use_triton,
                use_cudnn=use_cudnn,
            )

        sparse_loss_coeff = coeff if sparse_loss else 0.0
        output, indexer_loss = dsa_min_memory_gqa(
            query=query,
            key=key,
            value=value,
            hidden_states=x.detach(),
            indexer=self.indexer,
            softmax_scale=self.softmax_scale,
            loss_coeff=sparse_loss_coeff,
            use_indexer_rope=use_indexer_rope,
            query_chunk_size=qcs,
            key_chunk_size=kcs,
            cache_routing=getattr(cfg, "dsa_kernel_cache_routing", False),
            cache_indexer_k=getattr(cfg, "dsa_kernel_cache_indexer_k", False),
            cache_selected_scores=getattr(cfg, "dsa_kernel_cache_selected_scores", False),
            use_triton=use_triton,
            use_cudnn=use_cudnn,
        )

        # Default to the dense (full-KV) teacher unless sparse (top-k) loss is requested.
        if coeff > 0 and not sparse_loss:
            indexer_loss = dsa_dense_indexer_loss(
                query=query.detach(),
                key=key.detach(),
                hidden_states=x.detach(),
                indexer=self.indexer,
                softmax_scale=self.softmax_scale,
                loss_coeff=coeff,
                use_indexer_rope=use_indexer_rope,
                query_chunk_size=qcs,
                key_chunk_size=kcs,
                use_triton=use_triton,
            )

        if coeff > 0:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss, layer_number=self.layer_number, num_layers=cfg.num_layers
            )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        return output


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

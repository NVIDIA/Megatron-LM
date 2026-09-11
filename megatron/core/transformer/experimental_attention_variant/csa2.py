# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native V4.1 CSA2 training math with explicit, forward-local cross-layer sharing.

This uses the existing DSv4 attention wrapper and sparse attention reference. Shared graph
tensors belong to one stack forward and are passed explicitly; modules retain no activation
state or inference caches.
"""

from copy import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
    _apply_rope,
    get_window_topk_idxs,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import get_pg_size


@dataclass
class CSA2State:
    """Shared attention tensors for one full-sequence stack forward.

    Source layer IDs are zero-based, matching the configuration. ``global_indices`` uses
    logical compressed positions (and -1 for unavailable positions), never offsets into a
    consumer's concatenated local/global KV tensor. Float tensors retain their autograd graph,
    so every consumer contributes to its Full owner's gradient. Create a fresh state for each
    forward, including when another microbatch's backward is still outstanding.
    """

    global_kv: torch.Tensor | None = None
    indexer_k: torch.Tensor | None = None
    global_indices: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
    kv_source_layer: int | None = None
    index_source_layer: int | None = None
    candidate_source_layer: int | None = None
    sequence_length: int | None = None
    batch_size: int | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    last_layer: int | None = None

    def validate_forward(self, layer_idx: int, query: torch.Tensor) -> None:
        """Reject reuse across forwards or incompatible sequence/batch layouts."""
        if self.last_layer is not None and layer_idx <= self.last_layer:
            raise ValueError(
                "CSA2State layers must run in increasing order; create a fresh CSA2State "
                "for each stack forward."
            )
        sequence_length, batch_size = query.shape[:2]
        if self.sequence_length is None:
            self.sequence_length, self.batch_size = sequence_length, batch_size
            self.device, self.dtype = query.device, query.dtype
        elif (sequence_length, batch_size, query.device, query.dtype) != (
            self.sequence_length,
            self.batch_size,
            self.device,
            self.dtype,
        ):
            raise ValueError(
                "CSA2State sequence length, batch size, device and dtype must match this "
                "forward; create a fresh CSA2State for each stack forward."
            )


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor | int, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Select block-max candidates, always including the newest visible position's block.

    ``logits`` has shape ``[..., global positions]`` and is already causally masked.
    ``compress_lens`` broadcasts against its leading dimensions with a trailing singleton
    dimension. The result is a position mask; the caller still applies the causal score mask
    because a selected block can include positions after the query.
    """
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("CSA2 candidate block count and size must be positive.")
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    newest_block = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == newest_block, torch.inf
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class CSA2Compressor(MegatronModule):
    """Produce pre-RoPE global latents from complete, non-overlapping token groups."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: CompressorSubmodules,
        compress_ratio: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        if compress_ratio not in (1, 2):
            raise ValueError("CSA2 compression requires ratio 1 or 2.")
        self.compress_ratio = compress_ratio
        self.tp_group = pg_collection.tp
        projection_config = copy(config)
        if compress_ratio == 2:
            projection_config.params_dtype = torch.float32
            projection_config.bf16 = False
        linear_kwargs = dict(
            config=projection_config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_wkv = build_module(
            submodules.linear_wkv, config.hidden_size, config.v_head_dim, **linear_kwargs
        )
        self.linear_wgate = None
        if compress_ratio == 2:
            self.linear_wgate = build_module(
                submodules.linear_wgate, config.hidden_size, config.v_head_dim, **linear_kwargs
            )
            mark_keep_in_fp32(self.linear_wkv.weight)
            mark_keep_in_fp32(self.linear_wgate.weight)
        self.norm = build_module(
            submodules.norm,
            config=config,
            hidden_size=config.v_head_dim,
            eps=config.attention_latent_norm_epsilon,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[sequence, batch, hidden]`` to ``[sequence // ratio, batch, head_dim]``."""
        if self.compress_ratio == 1:
            latent, _ = self.linear_wkv(x)
        else:
            # An incomplete final group remains available through the sliding window only.
            complete_length = x.shape[0] // self.compress_ratio * self.compress_ratio
            xf = x[:complete_length].float()
            latent, _ = self.linear_wkv(xf)
            gate, _ = self.linear_wgate(xf)
            shape = (complete_length // self.compress_ratio, self.compress_ratio, *latent.shape[1:])
            latent = (latent.reshape(shape) * gate.reshape(shape).softmax(dim=1)).sum(dim=1)
        latent = latent.to(x.dtype)
        if latent.shape[0] == 0:
            # TE RMSNorm cannot normalize zero rows. Keep its zero weight gradient in
            # the graph when the sequence has no complete compression group.
            return latent * self.norm.weight
        return self.norm(latent)

    def backward_dw(self) -> None:
        """Follow the existing DSv4 linear gradient interface."""
        self.linear_wkv.backward_dw()
        if self.linear_wgate is not None:
            self.linear_wgate.backward_dw()


@dataclass
class CSA2IndexerSubmodules:
    """Projection and normalization specs for Full and Reindex CSA2 indexers."""

    linear_wq_b: ModuleSpec | type
    linear_wk: ModuleSpec | type
    k_norm: ModuleSpec | type
    linear_weights_proj: ModuleSpec | type


class CSA2Indexer(MegatronModule):
    """Select global positions using keys derived from the pre-RoPE compressor latent."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: CSA2IndexerSubmodules,
        compress_ratio: int,
        pg_collection: ProcessGroupCollection,
        owns_k: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.compress_ratio = compress_ratio
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.n_heads = config.dsa_indexer_n_heads
        self.head_dim = config.dsa_indexer_head_dim
        self.topk = config.dsa_indexer_topk
        self.owns_k = owns_k
        linear_kwargs = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            **linear_kwargs,
        )
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj, config.hidden_size, self.n_heads, **linear_kwargs
        )
        self.linear_wk = self.k_norm = None
        if owns_k:
            self.linear_wk = build_module(
                submodules.linear_wk, config.v_head_dim, self.head_dim, **linear_kwargs
            )
            self.k_norm = build_module(
                submodules.k_norm,
                config=config,
                hidden_size=self.head_dim,
                eps=config.attention_latent_norm_epsilon,
            )

    def project_keys(self, latent: torch.Tensor, rotary_pos_emb: nn.Module) -> torch.Tensor:
        """Produce graph-connected rotated index keys at a Full owner."""
        if not self.owns_k:
            raise ValueError("A CSA2 Reindex layer must consume its Full owner's indexer_k.")
        k, _ = self.linear_wk(latent)
        # An incomplete first compression group produces no key rows.
        k = self.k_norm(k) if k.shape[0] else k * self.k_norm.weight
        pos_dim = self.config.qk_pos_emb_head_dim
        return _apply_rope(
            k,
            nope_dim=self.head_dim - pos_dim,
            pos_dim=pos_dim,
            rotary_pos_emb_module=rotary_pos_emb,
            config=self.config,
            rotary_seq_len=k.shape[0],
            ratio=self.compress_ratio,
            cp_group=self.cp_group,
        )

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        rotary_pos_emb: nn.Module,
        *,
        indexer_k: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return differentiable causal scores ``[batch, sequence, global positions]``.

        The discrete top-k has no LM-loss gradient. A future indexer auxiliary loss can
        consume these scores explicitly without retaining a graph in module state. Full layers
        may pass a pre-RoPE latent as before; Reindex layers pass ``latent=None`` and the Full
        owner's ``indexer_k``. Supplying shared keys preserves their graph for auxiliary losses.
        """
        if indexer_k is None:
            if latent is None:
                raise ValueError("CSA2 indexer scores require latent or shared indexer_k.")
            indexer_k = self.project_keys(latent, rotary_pos_emb)
        elif latent is not None:
            raise ValueError("Provide either latent or indexer_k, not both.")
        q, _ = self.linear_wq_b(qr)
        q = q.reshape(*q.shape[:-1], self.n_heads, self.head_dim)
        pos_dim = self.config.qk_pos_emb_head_dim
        rope_kwargs = dict(
            nope_dim=self.head_dim - pos_dim,
            pos_dim=pos_dim,
            rotary_pos_emb_module=rotary_pos_emb,
            config=self.config,
            cp_group=self.cp_group,
        )
        q = _apply_rope(q, rotary_seq_len=q.shape[0], **rope_kwargs)
        weights, _ = self.linear_weights_proj(x)
        weights = weights.float() * (self.head_dim**-0.5 * self.n_heads**-0.5)
        scores = torch.einsum("sbhd,tbd->bsht", q.float(), indexer_k.float()).relu()
        scores = (scores * weights.permute(1, 0, 2).unsqueeze(-1)).sum(dim=2)
        visible = torch.arange(1, x.shape[0] + 1, device=x.device) // self.compress_ratio
        causal = torch.arange(indexer_k.shape[0], device=x.device)[None, :] < visible[:, None]
        scores = scores.masked_fill(~causal, -torch.inf)
        if candidates is not None:
            scores = scores.masked_fill(~candidates, -torch.inf)
        return scores

    def select_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """Return sorted logical positions, with -1 for masked or unavailable entries."""
        indices = scores.topk(min(self.topk, scores.shape[-1]), dim=-1, sorted=False).indices
        indices = indices.sort(dim=-1).values
        return indices.masked_fill(~scores.gather(-1, indices).isfinite(), -1).int()

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        rotary_pos_emb: nn.Module,
        *,
        indexer_k: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return sorted global-position indices, with ``-1`` for unavailable positions."""
        scores = self.forward_before_topk(
            x, qr, latent, rotary_pos_emb, indexer_k=indexer_k, candidates=candidates
        )
        return self.select_indices(scores)

    def backward_dw(self) -> None:
        """Follow the existing DSv4 linear gradient interface."""
        for linear in (self.linear_wq_b, self.linear_wk, self.linear_weights_proj):
            if linear is not None:
                linear.backward_dw()


class CompressedSparseAttention2(MegatronModule):
    """Native SWA, Full, Reindex and Reuse behind the existing DSv4 attention interface."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: CompressedSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        pg_collection: ProcessGroupCollection,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str | None = None,
        rotary_pos_emb: nn.Module | None = None,
        compress_ratio: int | None = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        if config.dsv4_version != "v4.1" or attn_mask_type != AttnMaskType.causal:
            raise ValueError("CSA2 requires V4.1 causal attention.")
        if attention_type != "self" or is_mtp_layer:
            raise NotImplementedError("CSA2 currently supports backbone self-attention only.")
        if get_pg_size(pg_collection.tp) != 1 or get_pg_size(pg_collection.cp) != 1:
            raise NotImplementedError("Native CSA2 currently requires TP=CP=1.")
        if (attention_dropout or config.attention_dropout) != 0:
            raise NotImplementedError("Native CSA2 does not implement attention dropout.")
        if config.dsa_indexer_loss_coeff:
            raise NotImplementedError("CSA2 indexer auxiliary loss is not integrated yet.")
        layer_idx = layer_number - 1
        ratio = config.csa_compress_ratios[layer_idx]
        if compress_ratio is not None and compress_ratio != ratio:
            raise ValueError("CSA2 compress_ratio must match the configured layer ratio.")
        self.layer_idx = layer_idx
        self.is_kv_source = layer_idx in config.csa2_kv_source_layers
        self.is_index_source = layer_idx in config.csa2_index_source_layers
        self.kv_source_layer = (
            max(source for source in config.csa2_kv_source_layers if source <= layer_idx)
            if ratio
            else None
        )
        self.index_source_layer = (
            max(source for source in config.csa2_index_source_layers if source <= layer_idx)
            if ratio
            else None
        )
        self.is_candidate_source = layer_idx == config.csa2_candidate_source_layer
        self.uses_candidates = (
            config.csa2_candidate_source_layer is not None
            and layer_idx > config.csa2_candidate_source_layer
        )
        self.compress_ratio = ratio
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.rotary_pos_emb = rotary_pos_emb
        self.softmax_scale = config.v_head_dim**-0.5 if softmax_scale is None else softmax_scale
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.attn_sink = mark_keep_in_fp32(
            nn.Parameter(
                torch.zeros(config.num_attention_heads, dtype=torch.float32, device=device)
            )
        )
        self.compressor = self.indexer = None
        if self.is_kv_source:
            kwargs = dict(config=config, compress_ratio=ratio, pg_collection=pg_collection)
            self.compressor = build_module(submodules.compressor, **kwargs)
        if self.is_index_source:
            self.indexer = build_module(
                submodules.indexer,
                config=config,
                compress_ratio=ratio,
                pg_collection=pg_collection,
                owns_k=self.is_kv_source,
            )

    def _shared_global_attention(
        self, x: torch.Tensor, qr: torch.Tensor, state: CSA2State
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Publish or consume shared graph tensors and logical position indices."""
        if self.is_kv_source:
            latent = self.compressor(x)
            # Derive index K before main KV RoPE. Both tensors keep the Full owner's graph;
            # only the subsequent discrete selection runs without gradient recording.
            state.indexer_k = self.indexer.project_keys(latent, self.rotary_pos_emb)
            pos_dim = self.config.qk_pos_emb_head_dim
            state.global_kv = _apply_rope(
                latent,
                self.config.v_head_dim - pos_dim,
                pos_dim,
                self.rotary_pos_emb,
                self.config,
                latent.shape[0],
                ratio=self.compress_ratio,
                cp_group=self.cp_group,
            )
            state.kv_source_layer = self.layer_idx
            state.global_indices = state.candidates = None
            state.index_source_layer = state.candidate_source_layer = None
        elif (
            state.kv_source_layer != self.kv_source_layer
            or state.global_kv is None
            or state.indexer_k is None
        ):
            raise ValueError(
                f"CSA2 layer {self.layer_idx} requires KV and indexer K from Full layer "
                f"{self.kv_source_layer} in the same forward's CSA2State."
            )

        if self.is_index_source:
            candidates = None
            if self.uses_candidates:
                if (
                    state.candidate_source_layer != self.config.csa2_candidate_source_layer
                    or state.candidates is None
                ):
                    raise ValueError(
                        f"CSA2 layer {self.layer_idx} requires candidates from layer "
                        f"{self.config.csa2_candidate_source_layer} in the same CSA2State."
                    )
                candidates = state.candidates
            with torch.no_grad():
                scores = self.indexer.forward_before_topk(
                    x,
                    qr,
                    None,
                    self.rotary_pos_emb,
                    indexer_k=state.indexer_k,
                    candidates=candidates,
                )
                if self.is_candidate_source:
                    visible = (
                        torch.arange(1, x.shape[0] + 1, device=x.device) // self.compress_ratio
                    ).unsqueeze(-1)
                    state.candidates = select_candidate_blocks(
                        scores,
                        visible,
                        self.config.csa2_candidate_topk_blocks,
                        self.config.csa2_candidate_block_size,
                    )
                    state.candidate_source_layer = self.layer_idx
                # The candidate source itself selects over all causal positions. Only later
                # Reindex layers apply its candidate mask, matching the reference ordering.
                state.global_indices = self.indexer.select_indices(scores)
            state.index_source_layer = self.layer_idx
        elif state.index_source_layer != self.index_source_layer or state.global_indices is None:
            raise ValueError(
                f"CSA2 Reuse layer {self.layer_idx} requires indices from layer "
                f"{self.index_source_layer} in the same forward's CSA2State."
            )
        return state.global_kv, state.global_indices

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
        x: torch.Tensor,
        qr: torch.Tensor,
        boundary_hidden: torch.Tensor | None = None,
        boundary_kv: torch.Tensor | None = None,
        csa2_state: CSA2State | None = None,
    ) -> torch.Tensor:
        """Attend jointly to causal window and selected global latents using one softmax."""
        if packed_seq_params is not None or boundary_hidden is not None or boundary_kv is not None:
            raise NotImplementedError(
                "Native CSA2 currently supports unpacked sequences with CP=1."
            )
        if query.ndim != 4 or query.shape[0] == 0:
            raise ValueError("CSA2 expects a nonempty [sequence, batch, heads, head_dim] query.")
        if csa2_state is None:
            if self.compress_ratio and not self.is_kv_source:
                raise ValueError(
                    "CSA2 Reindex/Reuse requires an explicit csa2_state shared with its Full "
                    "owner for this forward."
                )
            csa2_state = CSA2State()
        csa2_state.validate_forward(self.layer_idx, query)
        seq_len, batch = query.shape[:2]
        if attention_mask is not None:
            causal_mask = torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool).triu(
                1
            )
            if (
                attention_mask.dtype != torch.bool
                or attention_mask.shape[-2:] != (seq_len, seq_len)
                or not torch.equal(attention_mask, causal_mask.expand_as(attention_mask))
            ):
                raise NotImplementedError(
                    "Native CSA2 only supports an ordinary causal attention mask."
                )
        indices = get_window_topk_idxs(self.config.csa_window_size, batch, seq_len, query.device)
        kv = key.squeeze(-2)
        if self.compress_ratio:
            global_kv, global_indices = self._shared_global_attention(x, qr, csa2_state)
            # Keep the shared indices logical: every consumer applies its own concatenation
            # offset out of place, without corrupting the indices seen by the next layer.
            global_indices = torch.where(global_indices >= 0, global_indices + seq_len, -1)
            kv = torch.cat((kv, global_kv), dim=0)
            indices = torch.cat((indices, global_indices), dim=-1)
        # Gather in FP32 so repeated KV positions also accumulate their backward contributions
        # in FP32. Casting after the gather would scatter-add those gradients in BF16.
        output = unfused_compressed_sparse_attn(
            query, kv.float(), self.attn_sink, indices, self.softmax_scale
        )
        csa2_state.last_layer = self.layer_idx
        return output

    def backward_dw(self) -> None:
        """Flush the compressor and indexer linears using the existing DSv4 interface."""
        if self.compressor is not None:
            self.compressor.backward_dw()
        if self.indexer is not None:
            self.indexer.backward_dw()

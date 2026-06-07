# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# FlexAttention module for BlockMask-based attention

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

# Import flex_attention for BlockMask support
try:
    from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    compiled_flex_attention = torch.compile(torch_flex_attention)
    HAVE_FLEX_ATTENTION = True
except ImportError:
    HAVE_FLEX_ATTENTION = False
    compiled_flex_attention = None


from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide


# =============================================================================
# Ulysses Context Parallel A2A helpers
# =============================================================================

class _UlyssesA2AFunction(torch.autograd.Function):
    """
    Forward Ulysses A2A + restore scatter.

    Forward:  [Lund+Lgen, 1, nh/tp, dim] -> all_to_all (if cp>1) -> restore -> [U+G, 1, H_cp, dim]
    Backward: un-restore -> all_to_all (if cp>1) -> [Lund+Lgen, 1, nh/tp, dim]

    When cp_size == 1, all_to_all is skipped but the restore scatter still runs because
    FlexAttention's BlockMask uses original global token order, not the compact [und|gen] layout.
    """

    @staticmethod
    def forward(ctx, q, k, v, und_idx, gen_idx, Lund, Lgen, cp_group):
        ctx.save_for_backward(und_idx, gen_idx)
        ctx.Lund, ctx.Lgen = Lund, Lgen
        ctx.cp_group = cp_group
        cp_size = cp_group.size() if cp_group is not None else 1
        ctx.cp_size = cp_size

        local_seq = q.shape[0]   # Lund + Lgen
        nh_tp    = q.shape[2]
        head_dim = q.shape[3]
        H_cp = nh_tp // cp_size
        ctx.nh_tp, ctx.H_cp, ctx.head_dim = nh_tp, H_cp, head_dim

        U = und_idx.shape[0]
        G = gen_idx.shape[0]
        ctx.U, ctx.G = U, G

        def _a2a_and_restore(x):
            # x: [local_seq, 1, nh_x, head_dim]  (nh_x may differ for k/v in GQA)
            x = x.squeeze(1)  # [local_seq, nh_x, head_dim]
            nh_x = x.shape[1]
            H_cp_x = nh_x // cp_size  # per-tensor head count after CP split

            if cp_size > 1:
                # Reshape -> [local_seq, cp, H_cp_x, dim] -> permute -> [cp, local_seq, H_cp_x, dim]
                # -> view [cp*local_seq, H_cp_x, dim]  (one contiguous chunk per destination rank)
                x_send = (
                    x.reshape(local_seq, cp_size, H_cp_x, head_dim)
                     .permute(1, 0, 2, 3).contiguous()
                     .view(cp_size * local_seq, H_cp_x, head_dim)
                )
                x_recv = torch.empty_like(x_send)
                dist.all_to_all_single(x_recv, x_send, group=cp_group)
                # x_recv: [cp*local_seq, H_cp_x, head_dim] in rank-interleaved layout
            else:
                x_recv = x.contiguous()  # [local_seq, nh_x, head_dim]

            # Restore scatter: rank-interleaved layout -> original global token order
            # Each rank r contributes: und tokens at positions und_idx[r*Lund:(r+1)*Lund]
            #                          gen tokens at positions gen_idx[r*Lgen:(r+1)*Lgen]
            restored = x_recv.new_zeros(U + G, H_cp_x, head_dim)
            for r in range(cp_size):
                idx_und_r = und_idx[r * Lund : min((r + 1) * Lund, U)]
                idx_gen_r = gen_idx[r * Lgen : min((r + 1) * Lgen, G)]
                src_und = r * local_seq
                src_gen = r * local_seq + Lund
                if idx_und_r.shape[0] > 0:
                    restored[idx_und_r] = x_recv[src_und : src_und + idx_und_r.shape[0]]
                if idx_gen_r.shape[0] > 0:
                    restored[idx_gen_r] = x_recv[src_gen : src_gen + idx_gen_r.shape[0]]

            return restored.unsqueeze(1)  # [U+G, 1, H_cp_x, head_dim]

        return _a2a_and_restore(q), _a2a_and_restore(k), _a2a_and_restore(v)

    @staticmethod
    def backward(ctx, dq_full, dk_full, dv_full):
        und_idx, gen_idx = ctx.saved_tensors
        Lund, Lgen   = ctx.Lund, ctx.Lgen
        cp_group     = ctx.cp_group
        cp_size      = ctx.cp_size
        nh_tp, H_cp, head_dim = ctx.nh_tp, ctx.H_cp, ctx.head_dim
        U, G         = ctx.U, ctx.G
        local_seq    = Lund + Lgen

        def _unrestore_and_a2a(dx):
            # dx: [U+G, 1, H_cp_x, head_dim]  (H_cp_x may differ for k/v in GQA)
            dx = dx.squeeze(1)  # [U+G, H_cp_x, head_dim]
            H_cp_x = dx.shape[1]
            nh_x = H_cp_x * cp_size

            # Un-restore: original global order -> rank-interleaved
            dx_ri = dx.new_zeros(cp_size * local_seq, H_cp_x, head_dim)
            for r in range(cp_size):
                idx_und_r = und_idx[r * Lund : min((r + 1) * Lund, U)]
                idx_gen_r = gen_idx[r * Lgen : min((r + 1) * Lgen, G)]
                dst_und = r * local_seq
                dst_gen = r * local_seq + Lund
                if idx_und_r.shape[0] > 0:
                    dx_ri[dst_und : dst_und + idx_und_r.shape[0]] = dx[idx_und_r]
                if idx_gen_r.shape[0] > 0:
                    dx_ri[dst_gen : dst_gen + idx_gen_r.shape[0]] = dx[idx_gen_r]

            if cp_size > 1:
                dx_out = torch.empty_like(dx_ri)
                dist.all_to_all_single(dx_out, dx_ri.contiguous(), group=cp_group)
                dx_local = (
                    dx_out.view(cp_size, local_seq, H_cp_x, head_dim)
                          .permute(1, 0, 2, 3).contiguous()
                          .reshape(local_seq, nh_x, head_dim)
                )
            else:
                dx_local = dx_ri.view(local_seq, nh_x, head_dim)

            return dx_local.unsqueeze(1)  # [local_seq, 1, nh_x, head_dim]

        return (
            _unrestore_and_a2a(dq_full),
            _unrestore_and_a2a(dk_full),
            _unrestore_and_a2a(dv_full),
            None, None, None, None, None,
        )


class _UlyssesA2AInvFunction(torch.autograd.Function):
    """
    Un-restore scatter + inverse Ulysses A2A + split + re-pad.

    Forward:  [U+G, 1, H_cp, dim] -> un-restore -> all_to_all_inv (if cp>1) -> split+repad
              -> (attn_und [Lund, 1, nh/tp, dim], attn_gen [Lgen, 1, nh/tp, dim])
    Backward: un-repad+cat -> forward all_to_all (if cp>1) -> restore -> [U+G, 1, H_cp, dim]
    """

    @staticmethod
    def forward(ctx, ctx_attn, und_idx, gen_idx, Lund, Lgen,
                actual_lund, actual_lgen, cp_group):
        ctx.save_for_backward(und_idx, gen_idx)
        ctx.Lund, ctx.Lgen         = Lund, Lgen
        ctx.actual_lund            = actual_lund
        ctx.actual_lgen            = actual_lgen
        ctx.cp_group               = cp_group
        cp_size = cp_group.size() if cp_group is not None else 1
        ctx.cp_size = cp_size

        U = und_idx.shape[0]
        G = gen_idx.shape[0]
        ctx.U, ctx.G               = U, G
        H_cp     = ctx_attn.shape[2]
        head_dim = ctx_attn.shape[3]
        nh_tp    = H_cp * cp_size
        ctx.H_cp, ctx.head_dim, ctx.nh_tp = H_cp, head_dim, nh_tp

        local_seq = Lund + Lgen
        x = ctx_attn.squeeze(1)  # [U+G, H_cp, head_dim]

        # Un-restore: original global order -> rank-interleaved
        x_ri = x.new_zeros(cp_size * local_seq, H_cp, head_dim)
        for r in range(cp_size):
            idx_und_r = und_idx[r * Lund : min((r + 1) * Lund, U)]
            idx_gen_r = gen_idx[r * Lgen : min((r + 1) * Lgen, G)]
            dst_und = r * local_seq
            dst_gen = r * local_seq + Lund
            if idx_und_r.shape[0] > 0:
                x_ri[dst_und : dst_und + idx_und_r.shape[0]] = x[idx_und_r]
            if idx_gen_r.shape[0] > 0:
                x_ri[dst_gen : dst_gen + idx_gen_r.shape[0]] = x[idx_gen_r]

        if cp_size > 1:
            x_out = torch.empty_like(x_ri)
            dist.all_to_all_single(x_out, x_ri.contiguous(), group=cp_group)
            x_local = (
                x_out.view(cp_size, local_seq, H_cp, head_dim)
                     .permute(1, 0, 2, 3).contiguous()
                     .reshape(local_seq, nh_tp, head_dim)
            )
        else:
            x_local = x_ri.view(local_seq, nh_tp, head_dim)

        # x_local layout after A2A_inv: [und_real | und_pad | gen_real | gen_pad]
        # gen tokens start at Lund (not actual_lund)
        attn_und = x_local[:actual_lund]                 # [actual_lund, nh_tp, head_dim]
        attn_gen = x_local[Lund : Lund + actual_lgen]    # [actual_lgen, nh_tp, head_dim]

        if actual_lund < Lund:
            attn_und = torch.cat(
                [attn_und, x_local.new_zeros(Lund - actual_lund, nh_tp, head_dim)], dim=0
            )
        if actual_lgen < Lgen:
            attn_gen = torch.cat(
                [attn_gen, x_local.new_zeros(Lgen - actual_lgen, nh_tp, head_dim)], dim=0
            )

        # [Lund, 1, nh_tp, head_dim], [Lgen, 1, nh_tp, head_dim]
        return attn_und.unsqueeze(1), attn_gen.unsqueeze(1)

    @staticmethod
    def backward(ctx, d_attn_und, d_attn_gen):
        und_idx, gen_idx = ctx.saved_tensors
        Lund, Lgen         = ctx.Lund, ctx.Lgen
        actual_lund        = ctx.actual_lund
        actual_lgen        = ctx.actual_lgen
        cp_group           = ctx.cp_group
        cp_size            = ctx.cp_size
        H_cp, head_dim, nh_tp = ctx.H_cp, ctx.head_dim, ctx.nh_tp
        U, G               = ctx.U, ctx.G
        local_seq          = Lund + Lgen

        d_und = d_attn_und.squeeze(1)   # [Lund, nh_tp, head_dim]
        d_gen = d_attn_gen.squeeze(1)   # [Lgen, nh_tp, head_dim]

        # Gradient through re-pad: only real-token slots carry gradient.
        # x_local layout: [und_real | und_pad | gen_real | gen_pad]; gen starts at Lund
        d_x_local = d_und.new_zeros(local_seq, nh_tp, head_dim)
        d_x_local[:actual_lund]              = d_und[:actual_lund]
        d_x_local[Lund : Lund + actual_lgen] = d_gen[:actual_lgen]

        if cp_size > 1:
            # Forward A2A is the backward of A2A_inv
            d_send = (
                d_x_local.reshape(local_seq, cp_size, H_cp, head_dim)
                          .permute(1, 0, 2, 3).contiguous()
                          .view(cp_size * local_seq, H_cp, head_dim)
            )
            d_recv = torch.empty_like(d_send)
            dist.all_to_all_single(d_recv, d_send, group=cp_group)
            d_ri = d_recv
        else:
            d_ri = d_x_local.view(cp_size * local_seq, H_cp, head_dim)

        # Backward through un-restore: rank-interleaved -> original global order
        d_ctx = d_ri.new_zeros(U + G, H_cp, head_dim)
        for r in range(cp_size):
            idx_und_r = und_idx[r * Lund : min((r + 1) * Lund, U)]
            idx_gen_r = gen_idx[r * Lgen : min((r + 1) * Lgen, G)]
            src_und = r * local_seq
            src_gen = r * local_seq + Lund
            if idx_und_r.shape[0] > 0:
                d_ctx[idx_und_r] = d_ri[src_und : src_und + idx_und_r.shape[0]]
            if idx_gen_r.shape[0] > 0:
                d_ctx[idx_gen_r] = d_ri[src_gen : src_gen + idx_gen_r.shape[0]]

        return d_ctx.unsqueeze(1), None, None, None, None, None, None, None


# =============================================================================
# FlexAttention module
# =============================================================================

class FlexAttention(MegatronModule):
    """
    FlexAttention module for BlockMask-based attention.

    This module uses PyTorch's flex_attention for efficient sparse attention
    with BlockMask. It handles the padding/unpadding logic required when
    the actual sequence length differs from the BlockMask size.

    When packed_seq_params carries MoT fields (padded_und_seqlen is not None),
    the forward dispatches to _forward_mot which performs:
      Ulysses A2A + restore -> attention kernel -> un-restore + A2A_inv
    This path works for both cp_size == 1 (restore/un-restore only, no A2A)
    and cp_size > 1 (full Ulysses communication).

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        """
        Initialize FlexAttention.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
            layer_number (int): Layer number in the transformer.
            attn_mask_type (AttnMaskType): Type of attention mask.
            attention_type (str): Type of attention (e.g., "self", "cross").
            attention_dropout (float, optional): Dropout rate for attention.
            softmax_scale (float, optional): Scale factor for softmax.
            cp_comm_type (str, optional): Context parallel communication type.
            pg_collection (ProcessGroupCollection, optional): Process group collection.
        """
        super().__init__(config=config)

        if not HAVE_FLEX_ATTENTION:
            raise ImportError(
                "FlexAttention requires PyTorch with flex_attention support. "
                "Please upgrade to PyTorch 2.5+ or use DotProductAttention instead."
            )

        self.config: TransformerConfig = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(
                pg_collection, 'tp'
            ), "FlexAttention pg_collection must have tp process group"
        self.pg_collection = pg_collection
        self.tp_group = self.pg_collection.tp
        self.cp_group = getattr(pg_collection, 'cp', None)
        self.cp_size  = self.cp_group.size() if self.cp_group is not None else 1

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        # Softmax scale
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        # Dropout (note: flex_attention doesn't directly support dropout,
        # so we apply it after attention if needed)
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """
        Forward pass for FlexAttention.

        Args:
            query (Tensor): Query tensor of shape [seq_len, batch_size, num_heads, head_dim].
            key (Tensor): Key tensor of shape [seq_len, batch_size, num_kv_heads, head_dim].
            value (Tensor): Value tensor of shape [seq_len, batch_size, num_kv_heads, head_dim].
            attention_mask (Tensor): BlockMask for attention.
            attn_mask_type (AttnMaskType, optional): Attention mask type (unused).
            attention_bias (Tensor, optional): Attention bias (not supported).
            packed_seq_params (PackedSeqParams, optional): When this carries MoT fields
                (padded_und_seqlen is not None), the MoT CP path is used.

        Returns:
            Tensor: Output tensor of shape [seq_len, batch_size, hidden_size_per_partition].
        """
        assert attention_bias is None, "Attention bias is not supported for FlexAttention."
        assert (
            packed_seq_params is not None
            and getattr(packed_seq_params, 'padded_und_seqlen', None) is not None
        ), "FlexAttention requires MoTPackedSeqParams with padded_und_seqlen"

        return self._forward_mot(query, key, value, packed_seq_params, attention_mask)

    def _forward_mot(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_seq_params,   # MoTPackedSeqParams
        block_mask,
    ) -> Tensor:
        """
        MoT-aware attention forward: Ulysses A2A + restore -> kernel -> un-restore + A2A_inv.

        Works for both cp_size == 1 (restore/un-restore only, no A2A communication) and
        cp_size > 1 (full Ulysses all-to-all).

        Args:
            query, key, value: [Lund+Lgen, 1, nh/tp, head_dim]
            packed_seq_params: MoTPackedSeqParams with und/gen index arrays and seqlens.
            block_mask: BlockMask built over the full [U+G] global token sequence.

        Returns:
            Tensor: [Lund+Lgen, 1, hidden_size_per_partition]
        """
        und_idx    = packed_seq_params.packed_und_token_indexes   # [U]
        gen_idx    = packed_seq_params.packed_gen_token_indexes   # [G]
        Lund       = packed_seq_params.padded_und_seqlen
        Lgen       = packed_seq_params.padded_gen_seqlen
        actual_lund = len(packed_seq_params.local_und_token_indexes)
        actual_lgen = len(packed_seq_params.local_gen_token_indexes)

        # ── A2A + restore: [Lund+Lgen, 1, nh/tp, dim] -> [U+G, 1, H_cp, dim] ──
        q_full, k_full, v_full = _UlyssesA2AFunction.apply(
            query, key, value, und_idx, gen_idx, Lund, Lgen, self.cp_group
        )
        # q/k/v_full: [U+G, 1, H_cp, head_dim] in original global token order

        # ── Attention kernel ──────────────────────────────────────────────────
        seq_len_full  = q_full.shape[0]      # U + G
        H_cp          = q_full.shape[2]
        head_dim      = q_full.shape[3]
        num_kv_H_cp   = k_full.shape[2]

        padded_seq_len = block_mask.shape[2]
        pad_size = padded_seq_len - seq_len_full

        q_f = q_full.squeeze(1).permute(1, 0, 2)  # [H_cp, U+G, head_dim]
        k_f = k_full.squeeze(1).permute(1, 0, 2)
        v_f = v_full.squeeze(1).permute(1, 0, 2)

        if pad_size > 0:
            q_f = torch.cat([q_f, q_full.new_zeros(H_cp,        pad_size, head_dim)], dim=1)
            k_f = torch.cat([k_f, k_full.new_zeros(num_kv_H_cp, pad_size, head_dim)], dim=1)
            v_f = torch.cat([v_f, v_full.new_zeros(num_kv_H_cp, pad_size, head_dim)], dim=1)

        ctx_raw = compiled_flex_attention(
            q_f.unsqueeze(0),   # [1, H_cp, padded_seq_len, head_dim]
            k_f.unsqueeze(0),
            v_f.unsqueeze(0),
            enable_gqa=True,
            block_mask=block_mask,
            scale=self.softmax_scale,
        )
        # ctx_raw: [1, H_cp, padded_seq_len, head_dim]
        ctx_attn = ctx_raw[0, :, :seq_len_full, :]          # [H_cp, U+G, head_dim]
        ctx_attn = ctx_attn.permute(1, 0, 2).unsqueeze(1)   # [U+G, 1, H_cp, head_dim]

        if self.training and self.config.attention_dropout > 0:
            ctx_attn = self.attention_dropout(ctx_attn)

        # ── Un-restore + A2A_inv + split + re-pad ────────────────────────────
        attn_und, attn_gen = _UlyssesA2AInvFunction.apply(
            ctx_attn, und_idx, gen_idx, Lund, Lgen, actual_lund, actual_lgen, self.cp_group
        )
        # attn_und: [Lund, 1, nh/tp, head_dim]   attn_gen: [Lgen, 1, nh/tp, head_dim]

        # Reshape and cat -> [Lund+Lgen, 1, hidden_size_per_partition]
        return torch.cat([
            attn_und.reshape(Lund, 1, self.hidden_size_per_partition),
            attn_gen.reshape(Lgen, 1, self.hidden_size_per_partition),
        ], dim=0)

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict (empty for FlexAttention as it has no learnable parameters)."""
        return {}

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


class MSAIndexerLossLoggingHelper:
    """Helper for logging MSA indexer (KL) losses across layers."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        if layer_number is None:
            return
        tracker = MSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(
                num_layers, device=torch.cuda.current_device()
            )
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        tracker = MSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        tracker = MSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]
        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        if tracker.get("reduce_group") is not None:
            torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
        if tracker.get("avg_group") is not None:
            torch.distributed.all_reduce(
                values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
    ):
        MSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        indexer_loss_values = tracker["values"] * loss_scale
        num_layers = indexer_loss_values.shape[0]
        avg_indexer_loss = indexer_loss_values.sum() / num_layers
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss
        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)
        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)
        MSAIndexerLossLoggingHelper.clean_loss_in_tracker()


class MSAIndexerLossAutoscaler(torch.autograd.Function):
    """Attaches the indexer KL loss to the attention output for gradient flow."""

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indexer_loss,) = ctx.saved_tensors
        if MSAIndexerLossAutoscaler.main_loss_backward_scale is None:
            MSAIndexerLossAutoscaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        scale = MSAIndexerLossAutoscaler.main_loss_backward_scale
        scaled_grad = torch.ones_like(indexer_loss) * scale
        return grad_output, scaled_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        if MSAIndexerLossAutoscaler.main_loss_backward_scale is None:
            MSAIndexerLossAutoscaler.main_loss_backward_scale = scale
        else:
            MSAIndexerLossAutoscaler.main_loss_backward_scale.copy_(scale)


@dataclass
class MSASelfAttentionSubmodules:
    """Submodule specs for MSASelfAttention."""

    linear_qkv: Union[ModuleSpec, type] = None
    linear_idx_q: Union[ModuleSpec, type] = None
    linear_idx_k: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


def _build_block_mask(
    block_indices: torch.Tensor,
    num_blocks: int,
    block_size: int,
    seqlen: int,
    num_kv_groups: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a per-group token-level mask from selected block indices.

    Args:
        block_indices: [batch, seqlen, num_kv_groups, k] selected block ids per query.
        num_blocks: total number of KV blocks.
        block_size: tokens per block.
        seqlen: sequence length.
        num_kv_groups: H_kv.
        batch: batch size.
        device: target device.
        dtype: target dtype.

    Returns:
        mask: [batch, num_kv_groups, seqlen, seqlen] with 0 for selected tokens, -inf elsewhere.
    """
    mask = torch.full(
        (batch, num_kv_groups, seqlen, seqlen),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    for b in range(batch):
        for r in range(num_kv_groups):
            indices = block_indices[b, :, r, :]
            for qi in range(seqlen):
                blocks = indices[qi].tolist()
                for blk in blocks:
                    if blk < 0 or blk >= num_blocks:
                        continue
                    start = blk * block_size
                    end = min(start + block_size, seqlen)
                    if start >= seqlen:
                        break
                    if start >= qi:
                        break
                    mask[b, r, qi, start:end] = 0.0
    return mask


def _compute_msa_kl_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    idx_scores: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    num_blocks: int,
    softmax_scale: float,
    loss_coeff: float,
    num_kv_groups: int,
    num_query_heads: int,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """Compute MSA KL loss: teacher = group-averaged Main Branch, student = index distribution.

    Args:
        query: [sq, b, np, hn] Q after RoPE (main branch).
        key: [sk, b, num_kv_groups, hn] K after RoPE.
        idx_scores: [sq, b, num_kv_groups, sk] index scores (before softmax).
        block_indices: [b, sq, num_kv_groups, k] selected block ids.
        block_size: B_k.
        num_blocks: total blocks.
        softmax_scale: 1/sqrt(d_h).
        loss_coeff: lambda coefficient.
        num_kv_groups: H_kv.
        num_query_heads: H_q.
        pg_collection: process groups.

    Returns:
        Scalar KL loss.
    """
    sq, b_size, np, hn = query.shape
    sk = key.shape[0]
    device = query.device

    q_per_group = np // num_kv_groups

    causal_mask = torch.triu(
        torch.full((sq, sk), float("-inf"), dtype=torch.float32, device=device),
        diagonal=1,
    )

    all_kl = []
    for r in range(num_kv_groups):
        start_h = r * q_per_group
        end_h = start_h + q_per_group
        q_group = query[:, :, start_h:end_h, :]
        k_group = key[:, :, r : r + 1, :]

        q_group_2d = q_group.reshape(sq, -1, hn)
        k_group_2d = k_group.reshape(sk, -1, hn)
        main_scores = torch.bmm(q_group_2d.float(), k_group_2d.float().transpose(1, 2))
        main_scores = main_scores.reshape(sq, b_size, q_per_group, sk) * softmax_scale

        idx_scores_group = idx_scores[:, :, r : r + 1, :].float()
        idx_scores_group = idx_scores_group.expand(-1, -1, q_per_group, -1)

        for b_idx in range(b_size):
            for qi in range(sq):
                blocks = block_indices[b_idx, qi, r]
                block_set = set()
                for blk_idx in range(blocks.shape[0]):
                    blk = int(blocks[blk_idx])
                    if blk < 0 or blk >= num_blocks:
                        continue
                    start = blk * block_size
                    end = min(start + block_size, sk)
                    for kj in range(start, end):
                        if kj <= qi:
                            block_set.add(kj)
                if not block_set:
                    continue
                tok_list = sorted(block_set)
                tok_tensor = torch.tensor(tok_list, device=device, dtype=torch.long)

                main_scores_slice = main_scores[qi, b_idx, :, :]
                main_on_tok = main_scores_slice[:, tok_tensor]
                main_on_tok = main_on_tok + causal_mask[qi, tok_tensor].unsqueeze(0)
                main_probs = F.softmax(main_on_tok, dim=-1, dtype=torch.float32)
                teacher = main_probs.mean(dim=0, keepdim=True)

                idx_on_tok = idx_scores_group[qi, b_idx, 0:1, tok_tensor]
                idx_on_tok = idx_on_tok + causal_mask[qi, tok_tensor].unsqueeze(0)
                idx_probs = F.softmax(idx_on_tok, dim=-1, dtype=torch.float32)

                kl = teacher * (
                    torch.log(teacher + 1e-10) - torch.log(idx_probs + 1e-10)
                )
                all_kl.append(kl.sum())

    if not all_kl:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    kl_tensor = torch.stack(all_kl)
    return kl_tensor.mean() * loss_coeff


def _block_max_pool(
    scores: torch.Tensor,
    block_size: int,
    num_blocks: int,
    seqlen: int,
) -> torch.Tensor:
    """Apply causal max-pooling over KV blocks.

    Args:
        scores: [sq, b, H_kv, sk] token-level index scores.
        block_size: B_k.
        num_blocks: total blocks.
        seqlen: sequence length.

    Returns:
        block_scores: [sq, b, H_kv, num_blocks] block-level max scores (causal).
    """
    sq, b_size, n_kv, sk = scores.shape
    device = scores.device
    padded_sk = num_blocks * block_size
    pad = padded_sk - sk
    if pad > 0:
        scores = F.pad(scores, (0, pad))

    scores = scores.reshape(sq, b_size, n_kv, num_blocks, block_size)

    causal_per_block = torch.full(
        (sq, num_blocks, block_size), float("-inf"), device=device, dtype=scores.dtype
    )
    for qi in range(sq):
        cur_block = qi // block_size
        for blk in range(num_blocks):
            if blk > cur_block:
                continue
            elif blk == cur_block:
                limit = qi - blk * block_size + 1
                if limit <= 0:
                    continue
                causal_per_block[qi, blk, :limit] = 0.0
            else:
                causal_per_block[qi, blk, :] = 0.0

    causal_per_block = causal_per_block.reshape(1, 1, 1, num_blocks, block_size)
    scores = scores + causal_per_block
    block_scores = scores.amax(dim=-1)
    return block_scores


def _select_topk_blocks(
    block_scores: torch.Tensor,
    k: int,
    seqlen: int,
    block_size: int,
    num_blocks: int,
) -> torch.Tensor:
    """Select top-k block indices for each query and GQA group.

    Always includes the local block that contains the current query position.

    Args:
        block_scores: [sq, b, H_kv, num_blocks] block scores.
        k: number of blocks to select.
        seqlen: sequence length.
        block_size: B_k.
        num_blocks: total blocks.

    Returns:
        block_indices: [b, sq, H_kv, k] selected block ids (sorted).
    """
    sq, b_size, n_kv, _ = block_scores.shape
    device = block_scores.device

    local_blocks = torch.arange(sq, device=device) // block_size
    local_blocks = local_blocks.clamp(max=num_blocks - 1)

    block_indices_list = []
    for b_idx in range(b_size):
        per_sample = []
        for r in range(n_kv):
            per_group = []
            for qi in range(sq):
                scores_1d = block_scores[qi, b_idx, r, :]
                local_blk = int(local_blocks[qi])

                valid = scores_1d != float("-inf")
                valid_indices = torch.where(valid)[0]
                valid_scores = scores_1d[valid_indices]

                effective_k = min(k, valid_indices.shape[0])
                if effective_k == 0:
                    topk = torch.full((k,), -1, device=device, dtype=torch.long)
                else:
                    topk_vals, topk_idxs = valid_scores.topk(effective_k)
                    topk = valid_indices[topk_idxs]

                    if local_blk not in topk and local_blk < num_blocks:
                        if topk.shape[0] < k:
                            topk = torch.cat(
                                [topk, torch.tensor([local_blk], device=device)]
                            )
                        else:
                            topk = torch.cat(
                                [topk[:-1], torch.tensor([local_blk], device=device)]
                            )

                    if topk.shape[0] < k:
                        pad = torch.full(
                            (k - topk.shape[0],), -1, device=device, dtype=torch.long
                        )
                        topk = torch.cat([topk, pad])
                    else:
                        topk = topk[:k]

                per_group.append(topk)
            per_sample.append(torch.stack(per_group, dim=0))
        block_indices_list.append(torch.stack(per_sample, dim=0))

    return torch.stack(block_indices_list, dim=0)


class MSASelfAttention(MegatronModule):
    """MiniMax Sparse Attention (MSA) built on top of GQA.

    Implements the two-branch design from the MSA paper:
    - Index Branch: lightweight Q/K projections per GQA group that select top-k KV blocks.
    - Main Branch: standard GQA attention restricted to selected blocks.

    Reference: https://github.com/MiniMax-AI/MSA
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MSASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = "self",
        softmax_scale: Optional[float] = None,
        cp_comm_type: str = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: Optional[str] = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.cp_comm_type = cp_comm_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "cp"]
            )
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = config.num_query_groups
        self.kv_channels = config.kv_channels
        self.head_dim = config.kv_channels

        self.num_query_groups_per_partition = (
            self.num_query_groups // self.pg_collection.tp.size()
        )
        self.num_attention_heads_per_partition = (
            self.num_attention_heads // self.pg_collection.tp.size()
        )
        self.q_per_group = self.num_attention_heads // self.num_query_groups

        self.query_projection_size = self.kv_channels * self.num_attention_heads
        self.kv_projection_size = self.kv_channels * self.num_query_groups

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.kv_channels)
        self.softmax_scale = softmax_scale

        self.msa_block_size = config.msa_block_size
        self.msa_k = config.msa_num_selected_blocks
        self.msa_idx_dim = config.msa_indexer_head_dim
        self.msa_loss_coeff = config.msa_loss_coeff
        self.msa_warmup = config.msa_warmup

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
            tp_group=self.pg_collection.tp,
            name=(name + ".linear_qkv") if name is not None else None,
        )

        idx_q_out = self.num_query_groups * self.msa_idx_dim
        self.linear_idx_q = build_module(
            submodules.linear_idx_q,
            self.hidden_size,
            idx_q_out,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
            name=(name + ".linear_idx_q") if name is not None else None,
        )
        self.linear_idx_k = build_module(
            submodules.linear_idx_k,
            self.hidden_size,
            self.msa_idx_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
            name=(name + ".linear_idx_k") if name is not None else None,
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
            name=(name + ".linear_proj") if name is not None else None,
        )

        self.rotary_pos_emb = RotaryEmbedding(
            self.kv_channels,
            rotary_percent=self.config.rotary_percent,
            rotary_base=self.config.rotary_base,
            cp_group=self.pg_collection.cp,
        )

    def _get_qkv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GQA QKV projection from hidden states.

        Returns: (query, key, value) with shapes
            query: [sq, b, np_per_partition, hn]
            key:   [sq, b, nk_per_partition, hn]
            value: [sq, b, nk_per_partition, hn]
        """
        sq, b_size = hidden_states.shape[:2]
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        tp = self.pg_collection.tp.size()
        if self.num_query_groups < tp:
            raise NotImplementedError(
                "MSA does not yet support num_query_groups < tp_size"
            )
        nk = self.num_query_groups_per_partition
        np = self.num_attention_heads_per_partition
        hn = self.head_dim

        qkv = mixed_qkv.view(sq, b_size, nk, (np // nk + 2) * hn)
        q_part, k_part, v_part = qkv.split([(np // nk) * hn, hn, hn], dim=-1)
        query = q_part.reshape(sq, b_size, np, hn)
        key = k_part.reshape(sq, b_size, nk, hn)
        value = v_part.reshape(sq, b_size, nk, hn)
        return query, key, value

    def _apply_rope(
        self, query: torch.Tensor, key: torch.Tensor, rotary_pos_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings."""
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, query, self.config, None
        )
        if rotary_pos_emb is None:
            pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
        else:
            pos_emb = rotary_pos_emb

        if isinstance(pos_emb, tuple):
            q_pos_emb, k_pos_emb = pos_emb
        else:
            q_pos_emb = k_pos_emb = pos_emb

        query = apply_rotary_pos_emb(
            query,
            q_pos_emb,
            config=self.config,
            mscale=1.0,
            cp_group=self.pg_collection.cp,
        )
        key = apply_rotary_pos_emb(
            key,
            k_pos_emb,
            config=self.config,
            mscale=1.0,
            cp_group=self.pg_collection.cp,
        )
        return query, key

    def _index_forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Index Branch: compute Q_idx, K_idx, and token-level scores.

        Returns:
            q_idx: [sq, b, H_kv, d_idx]
            k_idx: [sq, b, 1, d_idx]
            scores: [sq, b, H_kv, sk] per-group token-level index scores
        """
        sq, b_size, _ = hidden_states.shape
        nk = self.num_query_groups

        q_idx, _ = self.linear_idx_q(hidden_states)
        k_idx, _ = self.linear_idx_k(hidden_states)

        q_idx = q_idx.reshape(sq, b_size, nk, self.msa_idx_dim)
        k_idx = k_idx.reshape(sq, b_size, 1, self.msa_idx_dim)

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, q_idx, self.config, None
        )
        pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
        if isinstance(pos_emb, tuple):
            q_pos_emb, _ = pos_emb
        else:
            q_pos_emb = pos_emb

        q_idx = apply_rotary_pos_emb(
            q_idx,
            q_pos_emb,
            config=self.config,
            mscale=1.0,
            cp_group=self.pg_collection.cp,
        )
        k_idx = apply_rotary_pos_emb(
            k_idx,
            q_pos_emb,
            config=self.config,
            mscale=1.0,
            cp_group=self.pg_collection.cp,
        )

        k_idx_t = k_idx.reshape(sq, b_size, self.msa_idx_dim).transpose(1, 2)
        q_idx_2d = q_idx.reshape(sq, b_size * nk, self.msa_idx_dim)
        k_idx_2d = k_idx_t.unsqueeze(0).expand(nk, -1, -1, -1)
        k_idx_2d = k_idx_2d.reshape(b_size * nk, self.msa_idx_dim, sq)
        idx_scores = torch.bmm(q_idx_2d.float(), k_idx_2d.float())
        idx_scores = idx_scores.reshape(sq, b_size, nk, sq) * (self.msa_idx_dim**-0.5)
        return q_idx, k_idx, idx_scores

    def _sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Main Branch block-sparse attention.

        Args:
            query: [sq, b, np, hn]
            key: [sq, b, nk, hn]
            value: [sq, b, nk, hn]
            block_indices: [b, sq, nk, k]

        Returns:
            context: [sq, b, np, hn]
        """
        sq, b_size, np, hn = query.shape
        nk = key.shape[2]
        device = query.device

        q_per_group = np // nk
        num_blocks = (sq + self.msa_block_size - 1) // self.msa_block_size

        causal_mask = torch.triu(
            torch.full((sq, sq), float("-inf"), dtype=torch.float32, device=device),
            diagonal=1,
        )

        context = torch.zeros_like(query, dtype=query.dtype)
        for r in range(nk):
            start_h = r * q_per_group
            end_h = start_h + q_per_group
            q_group = query[:, :, start_h:end_h, :]
            k_group = key[:, :, r : r + 1, :]
            v_group = value[:, :, r : r + 1, :]

            q_2d = q_group.reshape(sq, b_size * q_per_group, hn)
            k_2d = k_group.reshape(sq, b_size, hn).transpose(0, 1)
            k_2d = k_2d.unsqueeze(1).expand(-1, q_per_group, -1, -1)
            k_2d = k_2d.reshape(b_size * q_per_group, sq, hn)
            scores = torch.bmm(q_2d.float(), k_2d.float().transpose(1, 2))
            scores = scores.reshape(sq, b_size, q_per_group, sq) * self.softmax_scale

            for b_idx in range(b_size):
                for qi in range(sq):
                    blocks = block_indices[b_idx, qi, r]
                    tok_mask = torch.full(
                        (sq,), float("-inf"), device=device, dtype=torch.float32
                    )
                    for blk_idx in range(blocks.shape[0]):
                        blk = int(blocks[blk_idx])
                        if blk < 0 or blk >= num_blocks:
                            continue
                        start = blk * self.msa_block_size
                        end = min(start + self.msa_block_size, sq)
                        if start < sq:
                            tok_mask[start:end] = 0.0

                    scores_h = scores[:, b_idx, :, :]
                    masked = (
                        scores_h[qi, :, :]
                        + causal_mask[qi : qi + 1, :]
                        + tok_mask.unsqueeze(0)
                    )
                    probs = F.softmax(masked, dim=-1, dtype=torch.float32)
                    v_2d = v_group[:, b_idx, 0:1, :].reshape(sq, hn)
                    out_h = probs.to(v_2d.dtype) @ v_2d
                    context[qi, b_idx, start_h:end_h] = out_h

        return context

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_context: Optional = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        rotary_pos_cos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for MSA."""
        del rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin, attention_bias
        del packed_seq_params, sequence_len_offset
        inference_context = inference_context or inference_params

        sq, b_size, _ = hidden_states.shape
        nk = self.num_query_groups
        np = self.num_attention_heads
        device = hidden_states.device

        query, key, value = self._get_qkv(hidden_states)

        if rotary_pos_emb is not None:
            query, key = self._apply_rope(query, key, rotary_pos_emb)

        if self.training and torch.is_grad_enabled() and self.msa_loss_coeff > 0:
            hidden_states_detached = hidden_states.detach()
            q_idx, k_idx, idx_scores = self._index_forward(hidden_states_detached)
            num_blocks = (sq + self.msa_block_size - 1) // self.msa_block_size
            block_scores = _block_max_pool(
                idx_scores, self.msa_block_size, num_blocks, sq
            )
            block_indices = _select_topk_blocks(
                block_scores, self.msa_k, sq, self.msa_block_size, num_blocks
            )

            if not self.msa_warmup:
                context = self._sparse_attention(query, key, value, block_indices)
            else:
                nk_actual = key.shape[2]
                q_per_group = np // nk_actual
                causal_mask = torch.triu(
                    torch.full(
                        (sq, sq), float("-inf"), dtype=torch.float32, device=device
                    ),
                    diagonal=1,
                )
                for r in range(nk_actual):
                    start_h = r * q_per_group
                    end_h = start_h + q_per_group
                    q_g = query[:, :, start_h:end_h, :]
                    k_g = key[:, :, r : r + 1, :]
                    q2d = q_g.reshape(sq, b_size * q_per_group, -1)
                    k2d = k_g.reshape(sq, b_size, -1).transpose(0, 1)
                    k2d = k2d.unsqueeze(1).expand(-1, q_per_group, -1, -1)
                    k2d = k2d.reshape(b_size * q_per_group, sq, -1)
                    scores = torch.bmm(q2d.float(), k2d.float().transpose(1, 2))
                    scores = (
                        scores.reshape(sq, b_size, q_per_group, sq) * self.softmax_scale
                    )
                    scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
                    probs = F.softmax(scores, dim=-1, dtype=torch.float32)
                    v_g = value[:, :, r : r + 1, :]
                    v2d = v_g.reshape(sq, b_size, -1).transpose(0, 1)
                    out_g = probs.float() @ v2d.unsqueeze(1).float()
                    context = out_g.to(query.dtype).reshape(sq, b_size, q_per_group, -1)
                    if r == 0:
                        context_full = torch.zeros(
                            sq, b_size, np, -1, device=device, dtype=query.dtype
                        )
                    context_full[:, :, start_h:end_h, :] = context

            kl_loss = _compute_msa_kl_loss(
                query,
                key,
                idx_scores,
                block_indices,
                self.msa_block_size,
                num_blocks,
                self.softmax_scale,
                self.msa_loss_coeff,
                nk,
                np,
                self.pg_collection,
            )
            MSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=kl_loss,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers,
            )
            context = context_full if self.msa_warmup else context
            context = MSAIndexerLossAutoscaler.apply(context, kl_loss)
        else:
            hidden_states_detached = hidden_states.detach()
            q_idx, k_idx, idx_scores = self._index_forward(hidden_states_detached)
            num_blocks = (sq + self.msa_block_size - 1) // self.msa_block_size
            block_scores = _block_max_pool(
                idx_scores, self.msa_block_size, num_blocks, sq
            )
            block_indices = _select_topk_blocks(
                block_scores, self.msa_k, sq, self.msa_block_size, num_blocks
            )
            context = self._sparse_attention(query, key, value, block_indices)

        context_out = context.reshape(sq, b_size, -1)
        output, bias = self.linear_proj(context_out)
        return output, bias

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Faithful standard-residual and native-mHC Engram module."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, get_pg_size, nvtx_range_pop, nvtx_range_push

from .config import EngramConfig
from .distributed_embedding import EPShardedMultiTableEmbedding
from .hashing import build_ngram_hashes, slice_hashes_for_sequence_parallel

logger = logging.getLogger(__name__)


class _SequenceParallelConvHalo(torch.autograd.Function):
    """Differentiable fetch of the previous TP rank's trailing convolution context.

    Under sequence parallelism each TP rank owns one contiguous sequence slice, so the causal
    short convolution needs the last ``history_length`` positions of the previous rank's slice.
    The first rank keeps zeros, matching the true sequence start. Backward returns the received
    context's gradient to the sending rank's trailing positions.
    """

    @staticmethod
    def _exchange(send_tensor, recv_tensor, send_peer, recv_peer, group):
        operations = []
        if send_tensor is not None:
            operations.append(
                torch.distributed.P2POp(
                    torch.distributed.isend,
                    send_tensor.contiguous(),
                    torch.distributed.get_global_rank(group, send_peer),
                    group,
                )
            )
        if recv_tensor is not None:
            operations.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv,
                    recv_tensor,
                    torch.distributed.get_global_rank(group, recv_peer),
                    group,
                )
            )
        for request in torch.distributed.batch_isend_irecv(operations):
            request.wait()

    @staticmethod
    def forward(ctx, local_slice: Tensor, history_length: int, group) -> Tensor:
        ctx.group = group
        ctx.history_length = history_length
        ctx.local_shape = local_slice.shape
        rank = get_pg_rank(group)
        size = get_pg_size(group)
        ctx.rank, ctx.size = rank, size
        send_tensor = local_slice[-history_length:] if rank < size - 1 else None
        recv_tensor = local_slice.new_zeros((history_length, *local_slice.shape[1:]))
        _SequenceParallelConvHalo._exchange(
            send_tensor, recv_tensor if rank > 0 else None, rank + 1, rank - 1, group
        )
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_history: Tensor):
        rank, size = ctx.rank, ctx.size
        send_tensor = grad_history if rank > 0 else None
        recv_tensor = (
            grad_history.new_empty((ctx.history_length, *ctx.local_shape[1:]))
            if rank < size - 1
            else None
        )
        _SequenceParallelConvHalo._exchange(send_tensor, recv_tensor, rank - 1, rank + 1, ctx.group)
        grad_local = grad_history.new_zeros(ctx.local_shape)
        if recv_tensor is not None:
            grad_local[-ctx.history_length :] = recv_tensor
        return grad_local, None, None


class EngramGroupRMSNorm(torch.nn.Module):
    """RMSNorm over groups of ``group_size`` channels with optional zero-centered gamma.

    One weight of width ``num_groups * group_size`` normalized per group is mathematically
    identical to independent per-group RMSNorms, and matches the official Qwen PLE layout
    (one norm per role across all residual streams). Normalization and scaling run in fp32
    for half-precision inputs, mirroring the reference implementation; wider dtypes (fp64
    parity tests) keep their own precision.
    """

    def __init__(self, width: int, group_size: int, eps: float, zero_centered: bool, **factory):
        super().__init__()
        if width % group_size != 0:
            raise ValueError(f"width ({width}) must be divisible by group_size ({group_size}).")
        self.group_size = group_size
        self.eps = eps
        self.zero_centered = zero_centered
        initial = torch.zeros(width, **factory) if zero_centered else torch.ones(width, **factory)
        self.weight = nn.Parameter(initial)

    def forward(self, hidden: Tensor) -> Tensor:
        compute = hidden.float() if hidden.dtype in (torch.float16, torch.bfloat16) else hidden
        grouped = compute.reshape(*compute.shape[:-1], -1, self.group_size)
        normed = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = normed.flatten(-2)
        gamma = self.weight.to(normed.dtype)
        if self.zero_centered:
            gamma = 1.0 + gamma
        return (normed * gamma).type_as(hidden)


class Engram(MegatronModule):
    """DeepSeek Engram injection for one selected global transformer layer."""

    def __init__(
        self,
        config,
        engram_config: EngramConfig,
        layer_number: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        self.engram_config = engram_config
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.num_streams = config.num_residual_streams if config.enable_hyper_connections else 1
        self.hidden_size = config.hidden_size
        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
        )

        if engram_config.tokenizer_remap is not None:
            self.register_buffer(
                "tokenizer_remap", engram_config.tokenizer_remap.to(device=device), persistent=False
            )
        else:
            # The qwen variant hashes raw token IDs directly.
            self.tokenizer_remap = None
        self.register_buffer(
            "hash_multipliers",
            torch.tensor(engram_config.multipliers(layer_number), dtype=torch.int64, device=device),
            persistent=False,
        )
        table_sizes = engram_config.table_sizes(layer_number)
        self.register_buffer(
            "table_sizes",
            torch.tensor(table_sizes, dtype=torch.int64, device=device),
            persistent=False,
        )

        self.embedding = EPShardedMultiTableEmbedding(
            config=config,
            table_sizes=table_sizes,
            embedding_dim=engram_config.head_dim,
            init_method=config.init_method,
            ep_group=pg_collection.ep,
            tp_group=pg_collection.tp,
            expt_dp_group=pg_collection.expt_dp,
        )
        factory_kwargs = {"device": device, "dtype": config.params_dtype}
        # Official Qwen PLE layout: one fused key projection over all residual streams, one
        # shared value projection, and one group RMSNorm per role. This is mathematically
        # identical to per-stream projections/norms and maps 1:1 onto the HF weights. The
        # qwen variant uses bias-free projections and zero-centered gamma.
        use_bias = engram_config.variant_spec.projection_bias
        zero_centered = engram_config.variant_spec.zero_centered_gamma
        stream_width = self.num_streams * config.hidden_size
        self.value_projection = nn.Linear(
            engram_config.total_memory_dim, config.hidden_size, bias=use_bias, **factory_kwargs
        )
        self.key_projection = nn.Linear(
            engram_config.total_memory_dim, stream_width, bias=use_bias, **factory_kwargs
        )
        norm_kwargs = dict(
            group_size=config.hidden_size,
            eps=config.layernorm_epsilon,
            zero_centered=zero_centered,
            **factory_kwargs,
        )
        self.key_norm = EngramGroupRMSNorm(stream_width, **norm_kwargs)
        self.query_norm = EngramGroupRMSNorm(stream_width, **norm_kwargs)
        self.conv_norm = EngramGroupRMSNorm(stream_width, **norm_kwargs)
        channels = self.num_streams * config.hidden_size
        dilation = engram_config.max_ngram_order
        # The causal left context is provided explicitly: zeros at the true sequence start and,
        # under sequence parallelism, the previous TP rank's trailing positions (halo exchange).
        self.conv_history_length = (engram_config.kernel_size - 1) * dilation
        self.short_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=engram_config.kernel_size,
            groups=channels,
            bias=False,
            padding=0,
            dilation=dilation,
            **factory_kwargs,
        )
        self._initialize_dense_parameters()
        self._log_allocation()

    @property
    def local_sparse_parameter_count(self) -> int:
        """Rank-local sparse-table parameter count."""
        return self.embedding.local_parameter_count

    @property
    def global_sparse_parameter_count(self) -> int:
        """Global logical sparse-table parameter count."""
        return self.embedding.global_parameter_count

    def _initialize_dense_parameters(self) -> None:
        if self.config.perform_initialization:
            self.config.init_method(self.value_projection.weight)
            self.config.init_method(self.key_projection.weight)
        with torch.no_grad():
            if self.value_projection.bias is not None:
                self.value_projection.bias.zero_()
            if self.key_projection.bias is not None:
                self.key_projection.bias.zero_()
            self.short_conv.weight.zero_()

        if self.config.sequence_parallel:
            # Under SP every TP rank sees a distinct sequence slice, so all Engram gradients
            # need a TP sum. Sparse tables are already marked in EPShardedEmbeddingTable and are
            # reduced on a dedicated unflattened path; mark only the dense parameters here.
            for parameter in self.parameters():
                if not getattr(parameter, "is_engram_embedding", False):
                    parameter.sequence_parallel = True

    def _log_allocation(self) -> None:
        """Emit one machine-readable ownership record per instantiated module and rank."""
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        ep_rank = (
            torch.distributed.get_rank(self.pg_collection.ep)
            if torch.distributed.is_initialized() and self.pg_collection.ep is not None
            else 0
        )
        local_rows = [table.local_num_embeddings for table in self.embedding.tables]
        row_ranges = [(table.row_start, table.row_end) for table in self.embedding.tables]
        # DEBUG: one record per rank and layer is far too chatty for large jobs at INFO.
        logger.debug(
            "[Engram allocation] rank=%s layer=%s ep_rank=%s local_rows=%s "
            "row_ranges=%s global_rows=%s local_sparse_parameters=%s "
            "global_sparse_parameters=%s",
            global_rank,
            self.layer_number,
            ep_rank,
            local_rows,
            row_ranges,
            list(self.embedding.table_sizes),
            self.local_sparse_parameter_count,
            self.global_sparse_parameter_count,
        )

    def _convolution_history(self, normed: Tensor) -> Tensor:
        """Return the causal left context: zeros at the sequence start, SP halo otherwise."""
        history_shape = (self.conv_history_length, *normed.shape[1:])
        if not self.config.sequence_parallel or get_pg_size(self.tp_group) <= 1:
            return normed.new_zeros(history_shape)
        if normed.shape[0] < self.conv_history_length:
            raise RuntimeError(
                "Engram sequence-parallel convolution requires each TP rank to own at least "
                f"{self.conv_history_length} positions "
                f"(kernel_size - 1 = {self.engram_config.kernel_size - 1} times dilation "
                f"{self.engram_config.max_ngram_order}); got {normed.shape[0]}."
            )
        return _SequenceParallelConvHalo.apply(normed, self.conv_history_length, self.tp_group)

    def _short_convolution(self, value: Tensor) -> Tensor:
        sequence_length, batch_size, num_streams, hidden_size = value.shape
        normed = self.conv_norm(value.flatten(start_dim=-2)).view_as(value)
        with_history = torch.cat((self._convolution_history(normed), normed), dim=0)
        channels_first = with_history.permute(1, 2, 3, 0).reshape(
            batch_size, num_streams * hidden_size, sequence_length + self.conv_history_length
        )
        # Valid causal convolution over [history + local]: output length equals sequence_length.
        convolved = self.short_conv(channels_first)
        return (
            F.silu(convolved)
            .view(batch_size, num_streams, hidden_size, sequence_length)
            .permute(3, 0, 1, 2)
            .contiguous()
        )

    def forward(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        """Return an Engram residual with the same standard or mHC layout as input."""
        if hidden_states.ndim != 3:
            raise ValueError(f"Engram hidden_states must be [S,B,H], got {hidden_states.shape}.")
        expected_hidden = self.num_streams * self.hidden_size
        if hidden_states.shape[-1] != expected_hidden:
            raise ValueError(
                f"Engram expected hidden width {expected_hidden}, got {hidden_states.shape[-1]}."
            )

        message = "engram.hash"
        nvtx_range_push(message)
        try:
            hash_ids = build_ngram_hashes(
                input_ids=input_ids,
                tokenizer_remap=self.tokenizer_remap,
                multipliers=self.hash_multipliers,
                table_sizes=self.table_sizes,
                max_ngram_order=self.engram_config.max_ngram_order,
                num_hash_heads=self.engram_config.num_hash_heads,
                boundary_token_id=self.engram_config.hash_boundary_token_id,
                reset_at_boundary=self.engram_config.variant_spec.resets_windows_at_boundary_token,
            )
            hash_ids = slice_hashes_for_sequence_parallel(
                hash_ids, hidden_states.shape[0], self.tp_group
            )
        finally:
            nvtx_range_pop(message)

        message = "engram.lookup"
        nvtx_range_push(message)
        try:
            memory = self.embedding(hash_ids).flatten(start_dim=-2).transpose(0, 1).contiguous()
        finally:
            nvtx_range_pop(message)
        streams = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_streams, self.hidden_size
        )

        message = "engram.gate-projection"
        nvtx_range_push(message)
        try:
            shared_value = self.value_projection(memory)
            key = self.key_norm(self.key_projection(memory)).view_as(streams)
            query = self.query_norm(hidden_states).view_as(streams)
            score = (key * query).sum(dim=-1) / math.sqrt(self.hidden_size)
            score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
            gate = score.sigmoid().unsqueeze(-1)
            value = gate * shared_value.unsqueeze(2)
        finally:
            nvtx_range_pop(message)

        message = "engram.short-conv"
        nvtx_range_push(message)
        try:
            output = value + self._short_convolution(value)
        finally:
            nvtx_range_pop(message)
        return output.reshape(hidden_states.shape)

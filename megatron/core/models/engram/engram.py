# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native Engram residual injection, including the DeepSeek-V4.1 variant.

The row-sharded lookup and original variants come from Megatron-LM PR #7231
(75eed192). V4.1 follows the published inference/model.py algebra with ordinary
trainable PyTorch projections and embeddings in place of inference quantization.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size, nvtx_range_pop, nvtx_range_push

from .config import EngramConfig
from .distributed_embedding import EPShardedMultiTableEmbedding
from .hashing import build_ngram_hashes

logger = logging.getLogger(__name__)


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


def _row_cu_seqlens(packed_seq_params: PackedSeqParams | None) -> Tensor | None:
    """Return the packed document boundaries for this microbatch, or None when unpacked.

    ``get_batch`` keeps ``cu_seqlens`` as ``(1, N)`` for consistency and squeezes it to 1-D
    before building PackedSeqParams, so accept either and hand back a 1-D tensor.
    """
    if packed_seq_params is None:
        return None
    cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
    if cu_seqlens is None:
        return None
    return cu_seqlens.reshape(-1)


class Engram(MegatronModule):
    """Token-conditioned memory for one selected global transformer layer."""

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
        if config.sequence_parallel or get_pg_size(pg_collection.tp) != 1:
            raise ValueError("Engram model integration currently requires TP=1 without SP.")
        if get_pg_size(getattr(pg_collection, "cp", None)) != 1:
            raise ValueError("Engram model integration currently requires CP=1.")
        if get_pg_size(pg_collection.ep) != getattr(config, "expert_model_parallel_size", 1):
            raise ValueError("Engram requires an explicit EP group matching the model config.")
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
        if engram_config.variant_spec.joint_key_value:
            self.wkv = nn.Linear(
                engram_config.total_memory_dim,
                stream_width + config.hidden_size,
                bias=use_bias,
                **factory_kwargs,
            )
            self.value_projection = self.key_projection = None
        else:
            self.wkv = None
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
        self.conv_norm = self.short_conv = None
        self.conv_history_length = engram_config.conv_history_length
        if engram_config.variant_spec.short_convolution:
            self.conv_norm = EngramGroupRMSNorm(stream_width, **norm_kwargs)
            channels = self.num_streams * config.hidden_size
            self.short_conv = nn.Conv1d(
                channels,
                channels,
                kernel_size=engram_config.kernel_size,
                groups=channels,
                bias=False,
                padding=0,
                dilation=engram_config.max_ngram_order,
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
        projections = (self.wkv, self.value_projection, self.key_projection)
        with torch.no_grad():
            for projection in projections:
                if projection is not None:
                    if self.config.perform_initialization:
                        self.config.init_method(projection.weight)
                    if projection.bias is not None:
                        projection.bias.zero_()
            if self.short_conv is not None:
                self.short_conv.weight.zero_()

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

    def _convolve_with_history(self, normed: Tensor, history: Tensor) -> Tensor:
        """Valid causal convolution of ``normed`` given its explicit left context."""
        sequence_length, batch_size, num_streams, hidden_size = normed.shape
        with_history = torch.cat((history, normed), dim=0)
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

    def _short_convolution(self, value: Tensor) -> Tensor:
        """Native causal convolution for the original Engram/PLE variants."""
        normed = self.conv_norm(value.flatten(start_dim=-2)).view_as(value)
        history = normed.new_zeros((self.conv_history_length, *normed.shape[1:]))
        return self._convolve_with_history(normed, history)

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        add_residual: bool = False,
    ) -> Tensor:
        """Return an Engram increment with the input's standard or mHC layout.

        V4.1 keeps the increment in FP32. With ``add_residual=True`` the same
        FP32 input supplies both the gate and residual branches; their gradients
        accumulate before casting back to the input dtype. Other variants retain
        their activation dtype.

        ``packed_seq_params`` carries the packed (THD) document boundaries. It is only used to
        stop an n-gram window from reaching back into the previous document of the same row;
        the causal convolution deliberately still mixes across the boundary, matching both
        published reference implementations.
        """
        if input_ids is None:
            raise ValueError("Engram requires input_ids for the current microbatch.")
        if hidden_states.ndim != 3:
            raise ValueError(f"Engram hidden_states must be [S,B,H], got {hidden_states.shape}.")
        if input_ids.shape != (hidden_states.shape[1], hidden_states.shape[0]):
            raise ValueError("Engram input_ids must have shape [B,S] matching hidden_states.")
        expected_hidden = self.num_streams * self.hidden_size
        if hidden_states.shape[-1] != expected_hidden:
            raise ValueError(
                f"Engram expected hidden width {expected_hidden}, got {hidden_states.shape[-1]}."
            )

        row_cu_seqlens = _row_cu_seqlens(packed_seq_params)
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
                cu_seqlens=row_cu_seqlens,
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
            if self.engram_config.variant_spec.fp32_gate:
                # Keep normalization, the signed-square-root gate and the residual
                # increment in FP32. Flattened norm parameters remain vectors for
                # optimizer classification even though each stream has its own gamma.
                key, shared_value = self.wkv(memory).split(
                    (self.num_streams * self.hidden_size, self.hidden_size), dim=-1
                )
                key = key.float().view_as(streams)
                query = streams.float()
                gamma = (self.query_norm.weight.float() * self.key_norm.weight.float()).view(
                    self.num_streams, self.hidden_size
                )
                rstd = torch.rsqrt(query.square().mean(-1) + self.query_norm.eps)
                rstd = rstd * torch.rsqrt(key.square().mean(-1) + self.key_norm.eps)
                score = (query * gamma * key).sum(-1) * rstd * self.hidden_size**-0.5
                gate = torch.sigmoid(torch.copysign(score.abs().clamp_min(1e-6).sqrt(), score))
                increment = gate.unsqueeze(-1) * shared_value.float().unsqueeze(2)
                if add_residual:
                    return (query + increment).reshape(hidden_states.shape).to(hidden_states.dtype)
                return increment.reshape(hidden_states.shape)
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
        output = output.reshape(hidden_states.shape)
        return hidden_states + output if add_residual else output

    def add_to_residual(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> Tensor:
        """Inject memory before mHC's read/mapping operations, casting exactly once."""
        return self(hidden_states, input_ids, packed_seq_params, add_residual=True)

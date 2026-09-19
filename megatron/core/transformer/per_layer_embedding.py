# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-Layer (hashed n-gram) Embedding of Qwen4-Exp / Qwen3.8-Flash-Next.

A PLE module sits at the start of a decoder layer and adds a token-local lexical signal to every
hyper-connection residual stream:

    e      = concat_h  E_h[ hash_h(n-gram ending at t) ]          # [.., ple_embed_dim]
    key    = GroupRMSNorm(W_k e)        (one key per residual stream)
    value  = W_v e                       (shared across streams)
    gate_j = signed_sqrt( <key_j, GroupRMSNorm(x)_j> / sqrt(C) )
    g      = sigmoid(gate_j) * value                                # [.., n, C]
    out    = g + silu( DilatedDepthwiseCausalConv1d( GroupRMSNorm(g) ) )

The hashed n-gram table is the bulk of the parameters (51B for Qwen3.8-Flash-Next) and is stored
vocab-parallel across the tensor-parallel group with :class:`VocabParallelEmbedding`.

The n-gram ids depend on the raw token ids, which decoder layers do not receive. ``GPTModel``
therefore calls :meth:`PerLayerEmbedding.prepare` with the token ids (and packed-sequence
boundaries) before running the decoder; the module keeps them until the next ``prepare``.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import torch.distributed.nn.functional as dist_nn_functional
except ImportError:  # pragma: no cover - torch built without distributed support
    dist_nn_functional = None

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.transformer.hyper_connection import gated_residual_group_rmsnorm
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_ngram_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> List[int]:
    """Deterministic odd hash multipliers for one PLE module (matches the HF reference)."""
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PLE_LAYER_PRIME * ple_layer_index
    return [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64) % half_bound) + 1
        for index in range(ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    # Deterministic Miller-Rabin for 64-bit integers.
    exponent, shifts = value - 1, 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    """Return the ``count``-th prime strictly greater than ``start``."""
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def build_ngram_vocab_layout(
    ngram_vocab_size_base: int, ngram_heads: int, ple_layer_index: int
) -> Tuple[List[int], List[int], int]:
    """Per-head prime vocabulary sizes, row offsets and the total row count of one PLE table."""
    sizes, offsets, total = [], [], 0
    for head_idx in range(ngram_heads):
        global_head_idx = ple_layer_index * ngram_heads + head_idx
        size = find_nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return sizes, offsets, total


def ngram_segment_positions(
    token_ids: Tensor, eos_token_id: int, cu_seqlens: Optional[Tensor] = None
) -> Tensor:
    """Position of every token inside its n-gram segment.

    A segment starts after an ``eos_token_id`` token and at every packed-sequence boundary
    (``cu_seqlens``, THD layout with ``token_ids`` of shape ``[1, T]``).

    Args:
        token_ids: [b, s] token ids.
        eos_token_id: Reset token id.
        cu_seqlens: Optional cumulative sequence lengths of packed sequences.

    Returns:
        [b, s] int64 positions within the segment (0 for the first token of a segment).
    """
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    segment_start = previous_eos + 1
    if cu_seqlens is not None:
        assert batch_size == 1, "packed sequences (THD) require a single flattened batch row"
        starts = cu_seqlens.to(device=token_ids.device, dtype=torch.long)[:-1]
        doc_index = torch.searchsorted(starts, positions, right=True) - 1
        doc_start = starts[doc_index.clamp_min(0)].unsqueeze(0)
        segment_start = torch.maximum(segment_start, doc_start)
    return positions.unsqueeze(0) - segment_start


def shift_tokens_in_segment(
    token_ids: Tensor, position_in_segment: Tensor, shift: int, eos_token_id: int
) -> Tensor:
    """Token ``shift`` steps back, or ``eos_token_id`` when that crosses a segment start."""
    if shift == 0:
        return token_ids
    seq_len = token_ids.shape[1]
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    source = (positions - shift).clamp_min(0)
    shifted = token_ids.gather(1, source.unsqueeze(0).expand_as(token_ids))
    valid = position_in_segment >= shift
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


class NGramEmbedding(MegatronModule):
    """Hashed n-gram embedding table of one PLE module (``ple.ple_embedding`` in HF).

    Every n-gram order (bigrams, trigrams, ...) owns ``heads_per_ngram`` independently hashed
    heads; each head indexes its own prime-sized slice of a single vocab-parallel table whose rows
    are ``embedding_dim // ngram_heads`` wide. The concatenation of all heads is the embedding.
    """

    def __init__(self, config: TransformerConfig, embedding_dim: int, ple_layer_index: int):
        super().__init__(config)
        self.config = config
        self.ngram_size = config.ple_ngram_size
        self.heads_per_ngram = config.ple_heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.eos_token_id = config.ple_eos_token_id
        assert embedding_dim % self.ngram_heads == 0
        self.head_dim = embedding_dim // self.ngram_heads

        sizes, offsets, total = build_ngram_vocab_layout(
            config.ple_ngram_vocab_size_base, self.ngram_heads, ple_layer_index
        )
        self.head_vocab_sizes, self.head_offsets, self.total_vocab_size = sizes, offsets, total
        divisor = config.ple_ngram_vocab_divisible_by
        self.padded_vocab_size = math.ceil(total / divisor) * divisor

        assert config.ple_unigram_vocab_size is not None, (
            "ple_unigram_vocab_size (the token vocabulary size the hash multipliers were derived "
            "from) must be set when PLE layers are enabled"
        )
        multipliers = build_ngram_layer_multipliers(
            config.ple_unigram_vocab_size, self.ngram_size, ple_layer_index, config.ple_seed
        )
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        # Derived deterministically from the config; kept as persistent buffers so checkpoints
        # (Megatron and HF, which stores them too) carry them and can be cross-checked on load.
        self.register_buffer(
            "layer_multipliers", torch.tensor(multipliers, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes", torch.tensor(sizes, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "ngram_heads_offsets", torch.tensor(offsets, dtype=torch.long, device=device)
        )

        tp_group = get_tensor_model_parallel_group_if_none(None)
        self.ngram_embedding = VocabParallelEmbedding(
            self.padded_vocab_size,
            self.head_dim,
            init_method=config.init_method,
            reduce_scatter_embeddings=config.sequence_parallel,
            config=config,
            tp_group=tp_group,
        )

    def compute_ngram_ids(self, token_ids: Tensor, cu_seqlens: Optional[Tensor] = None) -> Tensor:
        """Hashed table rows for every token and head.

        Args:
            token_ids: [b, s] token ids.
            cu_seqlens: Optional packed-sequence boundaries (THD).

        Returns:
            [b, s, ngram_heads] int64 row ids into the (global) n-gram table.
        """
        token_ids = token_ids.long()
        position_in_segment = ngram_segment_positions(token_ids, self.eos_token_id, cu_seqlens)
        shifted = [
            shift_tokens_in_segment(token_ids, position_in_segment, shift, self.eos_token_id)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids, shifted[position] * self.layer_multipliers[position]
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))
        return torch.cat(blocks, dim=-1)

    def forward(self, ngram_ids: Tensor) -> Tensor:
        """Look up ``[b, s, heads]`` ids -> ``[s_local, b, embedding_dim]`` (sequence-first).

        With sequence parallelism the lookup is reduce-scattered along the sequence, so the output
        covers this rank's ``s // tp`` tokens only.
        """
        emb = self.ngram_embedding(ngram_ids)  # [s(/tp), b, heads, d] if SP else [b, s, heads, d]
        if not self.ngram_embedding.reduce_scatter_embeddings:
            emb = emb.transpose(0, 1)
        return emb.flatten(-2).contiguous()


class PerLayerEmbedding(MegatronModule):
    """Qwen4-Exp PLE layer: gate hashed n-gram values into every residual stream.

    Built from a ``per_layer_embedding`` layer-spec slot; ``layer_number`` (one-indexed) selects the
    PLE index through ``config.ple_layer_ids``.
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        assert (
            layer_number in config.ple_layer_ids
        ), f"layer {layer_number} is not a PLE layer ({config.ple_layer_ids})"
        self.ple_layer_index = list(config.ple_layer_ids).index(layer_number)
        self.n = config.mhc_num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.embed_dim = config.ple_embed_dim or config.hidden_size
        hc_hidden_size = self.n * self.hidden_size
        dtype = config.params_dtype

        self.ple_embedding = NGramEmbedding(config, self.embed_dim, self.ple_layer_index)
        self.key_proj = nn.Linear(self.embed_dim, hc_hidden_size, bias=False, dtype=dtype)
        self.value_proj = nn.Linear(self.embed_dim, self.hidden_size, bias=False, dtype=dtype)
        for name in ("norm_key", "norm_query", "norm_conv"):
            norm = nn.Module()
            norm.weight = nn.Parameter(torch.zeros(hc_hidden_size, dtype=dtype))
            setattr(self, name, norm)
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.conv_dilation = config.ple_ngram_size
        self.halo = (self.conv_kernel_size - 1) * self.conv_dilation
        # Depthwise conv weight in the nn.Conv1d layout [channels, 1, kernel].
        self.conv1d = nn.Module()
        self.conv1d.weight = nn.Parameter(
            torch.zeros(hc_hidden_size, 1, self.conv_kernel_size, dtype=dtype)
        )

        if config.perform_initialization:
            nn.init.normal_(self.key_proj.weight, mean=0.0, std=config.init_method_std)
            nn.init.normal_(self.value_proj.weight, mean=0.0, std=config.init_method_std)

        # Replicated parameters need their gradients all-reduced across TP under SP.
        if config.sequence_parallel:
            for name, param in self.named_parameters():
                if not name.startswith("ple_embedding.ngram_embedding."):
                    setattr(param, 'sequence_parallel', True)

        self.tp_group = get_tensor_model_parallel_group_if_none(None)
        self._ngram_ids: Optional[Tensor] = None
        self._position_in_segment: Optional[Tensor] = None
        self._position_in_sequence: Optional[Tensor] = None

    # ------------------------------------------------------------------ inputs
    def prepare(self, token_ids: Tensor, cu_seqlens: Optional[Tensor] = None) -> None:
        """Stash the n-gram ids and positions for the next forward pass.

        Args:
            token_ids: [b, s] raw token ids of the full (not sequence-parallel-sharded) batch.
            cu_seqlens: Packed-sequence boundaries for THD inputs (``[1, T]`` token ids).
        """
        assert (
            parallel_state.get_context_parallel_world_size() == 1
        ), "PerLayerEmbedding does not support context parallelism."
        with torch.no_grad():
            self._ngram_ids = self.ple_embedding.compute_ngram_ids(token_ids, cu_seqlens)
            # Conv taps must not read across packed-sequence boundaries.
            batch_size, seq_len = token_ids.shape
            positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
            if cu_seqlens is not None:
                starts = cu_seqlens.to(device=token_ids.device, dtype=torch.long)[:-1]
                doc_index = torch.searchsorted(starts, positions, right=True) - 1
                positions = positions - starts[doc_index.clamp_min(0)]
            self._position_in_sequence = positions.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _local_positions(self, seq_len_local: int) -> Tensor:
        """[s_local, b] positions (within their document) of this rank's tokens."""
        positions = self._position_in_sequence.transpose(0, 1)  # [s, b]
        if self.config.sequence_parallel and self.tp_group.size() > 1:
            rank = self.tp_group.rank()
            positions = positions[rank * seq_len_local : (rank + 1) * seq_len_local]
        assert positions.shape[0] == seq_len_local, (
            f"PLE positions cover {positions.shape[0]} tokens but the layer input has "
            f"{seq_len_local}; was `prepare` called for this batch?"
        )
        return positions

    # ------------------------------------------------------------------ conv
    def _causal_dilated_conv(self, x: Tensor, positions: Tensor) -> Tensor:
        """Depthwise causal conv with dilation over the sequence dim of ``x`` [s_local, b, D].

        Taps that reach before the token's document start read zeros. Under sequence
        parallelism the ``halo`` rows preceding this rank's chunk are fetched from the previous
        tensor-parallel rank.
        """
        s_local = x.shape[0]
        if self.config.sequence_parallel and self.tp_group.size() > 1:
            assert s_local >= self.halo, (
                f"sequence-parallel chunk ({s_local} tokens) shorter than the PLE conv receptive "
                f"field ({self.halo})"
            )
            # Differentiable all-gather of every rank's trailing `halo` rows; rank r consumes
            # rank r-1's. Rank 0 has no predecessor but must keep the gathered tensors in its
            # autograd graph (multiplied by zero): the all-gather backward is a collective, so
            # every rank has to reach it or the tensor-parallel group deadlocks in backward.
            tails = dist_nn_functional.all_gather(x[-self.halo :].contiguous(), group=self.tp_group)
            rank = self.tp_group.rank()
            prev = tails[rank - 1] if rank > 0 else tails[-1] * 0.0
            x_ext = torch.cat([prev, x], dim=0)
        else:
            x_ext = F.pad(x, (0, 0, 0, 0, self.halo, 0))
        weight = self.conv1d.weight.squeeze(1).to(x.dtype)  # [D, K]
        kernel_size, dilation = self.conv_kernel_size, self.conv_dilation
        out = torch.zeros_like(x, dtype=torch.float32)
        for tap in range(kernel_size):
            offset = (kernel_size - 1 - tap) * dilation  # tap k reads x[t - offset]
            src = x_ext[self.halo - offset : self.halo - offset + s_local]
            valid = (positions >= offset).unsqueeze(-1)
            out = out + torch.where(valid, src.float(), 0.0) * weight[:, tap].float()
        return out.to(x.dtype)

    # ------------------------------------------------------------------ forward
    def forward(self, hidden_states: Tensor) -> Tensor:
        """[s_local, b, n*C] streams -> [s_local, b, n*C] PLE update (to be added by the layer)."""
        if self._ngram_ids is None:
            raise RuntimeError("PerLayerEmbedding.prepare() must be called before forward().")
        s_local, batch_size, hc_hidden_size = hidden_states.shape
        n, C = self.n, self.hidden_size
        dtype = hidden_states.dtype

        embeddings = self.ple_embedding(self._ngram_ids).to(dtype)  # [s_local, b, embed_dim]
        assert (
            embeddings.shape[0] == s_local
        ), f"PLE embeddings cover {embeddings.shape[0]} tokens but the layer input has {s_local}"
        key = gated_residual_group_rmsnorm(
            self.key_proj(embeddings), self.norm_key.weight, n, self.norm_eps
        )
        value = self.value_proj(embeddings)  # [s, b, C]
        query = gated_residual_group_rmsnorm(
            hidden_states, self.norm_query.weight, n, self.norm_eps
        )

        gate = (
            key.float().view(s_local, batch_size, n, C)
            * query.float().view(s_local, batch_size, n, C)
        ).sum(dim=-1, keepdim=True) / math.sqrt(C)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = (torch.sigmoid(gate) * value.float().unsqueeze(-2)).to(dtype)  # [s, b, n, C]
        gated_value = gated_value.view(s_local, batch_size, hc_hidden_size)
        gated_value_normed = gated_residual_group_rmsnorm(
            gated_value, self.norm_conv.weight, n, self.norm_eps
        )
        positions = self._local_positions(s_local)
        conv_out = self._causal_dilated_conv(gated_value_normed, positions)
        return gated_value + F.silu(conv_out.float()).to(dtype)

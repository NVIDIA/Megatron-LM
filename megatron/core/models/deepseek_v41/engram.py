# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# The logical-block hash seed convention follows DeepSeek-V4.1-Flash (see LICENSE.deepseek).

"""V4.1 layout and configuration at the shared Engram boundary."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from megatron.core.models.engram import Engram, EngramConfig, EngramLookupResult
from megatron.core.models.engram.hashing import build_ngram_hashes
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
    exchange_cp_boundary_hidden,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    get_thd_token_metadata,
)
from megatron.core.utils import get_pg_rank, get_pg_size


def released_multipliers(block: int, order: int, compressed_vocab_size: int) -> tuple[int, ...]:
    """Derive the released constants from the logical block, independent of its physical V layer."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rng = np.random.default_rng(10007 * block)
    return tuple(int(x) for x in 2 * rng.integers(0, bound, size=order, dtype=np.int64) + 1)


class _TensorMapConfig(EngramConfig):
    """Read a released 1-D torch token map into the common Engram configuration."""

    def _load_tokenizer_map(self) -> None:
        remap = torch.load(self.tokenizer_map_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(remap, torch.Tensor)
            or remap.ndim != 1
            or not remap.numel()
            or remap.dtype not in (torch.int32, torch.int64)
            or remap.min().item() < 0
            or not 0 <= self.boundary_token_id < remap.numel()
        ):
            raise ValueError("Expected a 1-D integer V4.1 tokenizer map with a valid pad ID")
        self.tokenizer_remap = remap.long()
        self.hash_boundary_token_id = int(remap[self.boundary_token_id])
        vocab = int(remap.max()) + 1
        self.layer_multipliers = {
            layer: released_multipliers((layer - 1) // 2, self.max_ngram_order, vocab)
            for layer in self.layer_ids
        }

    def validate_vocabulary(self, expected_tokenizer_vocab_size: int) -> None:
        """Map additional padded vocabulary IDs to the released pad-token identity."""
        if self.tokenizer_remap.numel() < expected_tokenizer_vocab_size:
            self.tokenizer_remap = torch.nn.functional.pad(
                self.tokenizer_remap,
                (0, expected_tokenizer_vocab_size - self.tokenizer_remap.numel()),
                value=self.hash_boundary_token_id,
            )
        super().validate_vocabulary(expected_tokenizer_vocab_size)


def validate_v41_layers(shared: EngramConfig, num_layers: int) -> None:
    """Check V-layer placement and the release hash seeds for directly supplied shared configs."""
    if shared.variant != "deepseek_v41" or any(
        layer % 2 == 0 or not 1 <= layer <= num_layers for layer in shared.layer_ids
    ):
        raise ValueError("V4.1 Engram must select 1-based V layers of the backbone")
    vocab = int(shared.tokenizer_remap.max()) + 1
    if any(
        shared.multipliers(layer)
        != released_multipliers((layer - 1) // 2, shared.max_ngram_order, vocab)
        for layer in shared.layer_ids
    ):
        raise ValueError("V4.1 tokenizer-map multipliers must use the original logical block IDs")


def released_engram_config(options: Mapping, tokenizer_map_path: str | None) -> EngramConfig:
    """Map the released config onto the common per-head tables and validate row identity."""
    if any(options.get(k) for k in ("prefetch", "defer_grad_transfer")):
        raise ValueError(
            "The shared V4.1 Engram path does not implement asynchronous lookup tickets"
        )
    if not tokenizer_map_path:
        raise ValueError("V4.1 Engram requires its tokenizer-map artifact")
    blocks = tuple(options["layer_ids"])
    order, heads = options["max_ngram_size"], options["n_heads"]
    config_cls = (
        _TensorMapConfig if Path(tokenizer_map_path).suffix in (".pt", ".pth") else EngramConfig
    )
    shared = config_cls(
        variant="deepseek_v41",
        layer_ids=tuple(2 * block + 1 for block in blocks),
        global_vocab_sizes=(options["vocab_size"],) * (order - 1),
        max_ngram_order=order,
        num_hash_heads=heads,
        memory_dim=heads * options["head_dim"],
        kernel_size=0,
        hash_seed=0,
        boundary_token_id=options["pad_token_id"],
        tokenizer_map_path=str(tokenizer_map_path),
        parallel_size=options.get("parallel_size"),
        communication_backend=options.get("communication_backend"),
        **{
            key: options[key]
            for key in (
                "symmetric_workspace_mb",
                "embedding_lr_multiplier",
                "embedding_weight_decay",
                "freeze_embedding",
            )
            if key in options
        },
    )
    if tuple(sum(shared.table_sizes(layer)) for layer in shared.layer_ids) != tuple(
        options["num_embeddings"]
    ):
        raise ValueError("V4.1 token-table rows do not match the released per-block layout")
    if int(shared.tokenizer_remap.max()) + 1 != options["compressed_vocab_size"]:
        raise ValueError("V4.1 token map disagrees with the released compressed vocabulary size")
    validate_v41_layers(shared, max(shared.layer_ids))
    return shared


def consume_legacy_hash_buffers(shared: EngramConfig | None, state_dict, prefix: str) -> None:
    """Validate and discard the old model's persistent copies of the artifact constants."""
    prefix += "engram_hash."
    if not any(key.startswith(prefix) for key in state_dict):
        return
    if shared is None:
        raise ValueError("The checkpoint contains Engram hashes but the model has no memory")
    sizes = torch.tensor([shared.table_sizes(layer) for layer in shared.layer_ids])
    if any(
        prefix + key not in state_dict for key in ("token_map", "primes", "offsets", "multipliers")
    ):
        raise ValueError("Incomplete legacy V4.1 Engram hash buffers")
    token_map = state_dict[prefix + "token_map"]
    remap = shared.tokenizer_remap
    if (
        token_map.numel() > remap.numel()
        or (remap[token_map.numel() :] != shared.hash_boundary_token_id).any()
    ):
        raise ValueError("Legacy Engram vocabulary disagrees with the tokenizer artifact")
    constants = {
        "token_map": remap[: token_map.numel()],
        "multipliers": torch.tensor([shared.multipliers(layer) for layer in shared.layer_ids]),
        "primes": sizes.view(
            len(shared.layer_ids), shared.max_ngram_order - 1, shared.num_hash_heads
        ),
        "offsets": sizes.cumsum(-1) - sizes,
    }
    if any(
        not torch.equal(state_dict[prefix + key].cpu(), value.cpu())
        for key, value in constants.items()
    ):
        raise ValueError(
            "Legacy Engram hash constants disagree with the selected tokenizer artifact"
        )
    for key in constants:
        del state_dict[prefix + key]


class V41Engram(Engram):
    """Use the shared tables, transport and gated fusion with V4.1's contiguous token history."""

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # A gathered legacy table lays out complete heads consecutively. Sharded legacy
        # optimizer checkpoints must first be reconstructed with their original layout.
        old = prefix + "embed.tables.0.weight"
        if old in state_dict:
            weight = state_dict[old]
            sizes = self.engram_config.table_sizes(self.layer_number)
            keys = [f"{prefix}embedding.tables.{i}.weight" for i in range(len(sizes))]
            if tuple(weight.shape) != (sum(sizes), self.engram_config.head_dim) or any(
                k in state_dict for k in keys
            ):
                raise ValueError("Expected one gathered V4.1 Engram table in the legacy layout")
            for key, chunk, table in zip(
                keys, weight.split(sizes), self.embedding.tables, strict=True
            ):
                state_dict[key] = chunk[table.row_start : table.row_end]
            del state_dict[old]
        super()._load_from_state_dict(state_dict, prefix, *args)

    def lookup(
        self,
        input_ids: torch.Tensor,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        local_sequence_length: int | None = None,
    ) -> EngramLookupResult:
        """Hash this rank's tokens with bounded left context and the packed physical boundaries."""
        length = input_ids.shape[1]
        if local_sequence_length is not None and local_sequence_length != length:
            raise ValueError("V4.1 Engram requires one token ID per local hidden position")
        tokens = self.tokens_with_history(input_ids)
        return self.lookup_with_history(input_ids, tokens, packed_seq_params)

    def tokens_with_history(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Exchange the bounded token history once per microbatch on this physical stage."""
        size, rank = get_pg_size(self.cp_group), get_pg_rank(self.cp_group)
        if size > 1 and self.cp_layout != "contiguous":
            raise ValueError("V4.1 Engram requires contiguous context parallelism")
        length = input_ids.shape[1]
        halo = self.engram_config.max_ngram_order - 1
        tokens = input_ids
        if size > 1 and length:
            rows = input_ids.T.contiguous()
            if length >= halo:
                left = exchange_cp_boundary_hidden(rows, 1, halo, self.cp_group)
                if rank:
                    tokens = torch.cat((left.T, input_ids), dim=1)
            else:
                # Tiny shards may need history from several ranks. Each shard is shorter than halo.
                peers = [torch.empty_like(rows) for _ in range(size)]
                torch.distributed.all_gather(peers, rows, group=self.cp_group)
                if rank:
                    tokens = torch.cat((torch.cat(peers[:rank])[-halo:].T, input_ids), dim=1)
        return tokens

    def lookup_with_history(self, input_ids, tokens, packed_seq_params):
        """Use the shared token row for one layer's hash and differentiable table transport."""
        padded = None if packed_seq_params is None else packed_seq_params.cu_seqlens_q_padded
        if get_pg_size(self.cp_group) == 1 and padded is None:
            return super().lookup(
                input_ids, packed_seq_params, local_sequence_length=input_ids.shape[1]
            )
        rank = get_pg_rank(self.cp_group)
        length = input_ids.shape[1]
        prefix = tokens.shape[1] - length
        bounds = valid = physical = None
        if packed_seq_params is not None:
            logical = packed_seq_params.cu_seqlens_q.reshape(-1)
            physical = logical if padded is None else padded.reshape(-1)
            first = rank * length - prefix
            positions = torch.arange(tokens.shape[1], device=tokens.device) + first
            _, _, keep = get_thd_token_metadata(logical, physical, positions)
            tokens = tokens.masked_fill(~keep.unsqueeze(0), self.engram_config.boundary_token_id)
            # Reset history at both ends of each real document, including interior padding at CP1.
            edges = torch.stack((physical[:-1], physical[:-1] + logical.diff()), dim=1).flatten()
            bounds = (torch.cat((edges, physical[-1:])) - first).clamp(0, tokens.shape[1])
            valid = keep[prefix:].view(1, length, 1)
        ids = build_ngram_hashes(
            tokens,
            self.tokenizer_remap,
            self.hash_multipliers,
            self.table_sizes,
            self.engram_config.max_ngram_order,
            self.engram_config.num_hash_heads,
            self.engram_config.hash_boundary_token_id,
            cu_seqlens=bounds,
        )
        # The parent binds the selected NCCL or symmetric table lookup once during construction.
        memory = self._lookup(ids[:, prefix:].contiguous()).flatten(-2).transpose(0, 1).contiguous()
        return EngramLookupResult(memory, valid, physical)

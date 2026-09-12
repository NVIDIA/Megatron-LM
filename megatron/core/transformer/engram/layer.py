# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer-owned Engram memory and sequence-layout adaptation."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributed import ProcessGroup
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    make_sharded_object_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import make_sharded_tensor_for_checkpoint

from .addressing import NgramHashMapping
from .config import (
    _ENGRAM_CHECKPOINT_SCHEMA_VERSION,
    ENGRAM_TABLE_LR_MULTIPLIER,
    ENGRAM_TABLE_WEIGHT_DECAY,
    EngramConfig,
    _parallel_world_size,
)
from .fusion import EngramLayer
from .tokenizer import CompressedTokenizer


class ParallelSequenceLayout:
    """Restore and repartition SP and the standard attention zigzag CP layout."""

    @staticmethod
    def _restore_cp(tensors: Sequence[Tensor], sequence_dim: int) -> Tensor:
        chunks = [None] * (2 * len(tensors))
        for rank, tensor in enumerate(tensors):
            first, second = tensor.chunk(2, dim=sequence_dim)
            chunks[rank] = first
            chunks[2 * len(tensors) - rank - 1] = second
        return torch.cat(chunks, dim=sequence_dim)

    @staticmethod
    def _select_cp(
        tensor: Tensor, sequence_dim: int, cp_group: ProcessGroup | None = None
    ) -> Tensor:
        cp_size = (
            torch.distributed.get_world_size(group=cp_group)
            if cp_group is not None
            else _parallel_world_size(parallel_state.get_context_parallel_world_size)
        )
        if cp_size == 1:
            return tensor
        cp_rank = (
            torch.distributed.get_rank(group=cp_group)
            if cp_group is not None
            else parallel_state.get_context_parallel_rank()
        )
        chunks = tensor.chunk(2 * cp_size, dim=sequence_dim)
        return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]), dim=sequence_dim)

    @classmethod
    def gather(
        cls,
        hidden_states: Tensor,
        compressed_input_ids: Tensor,
        sequence_parallel: bool,
        tp_group: ProcessGroup | None = None,
        cp_group: ProcessGroup | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Gather sequence-parallel hidden states and compressed IDs onto the full sequence."""
        if sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=tp_group
            )

        cp_size = (
            torch.distributed.get_world_size(group=cp_group)
            if cp_group is not None
            else _parallel_world_size(parallel_state.get_context_parallel_world_size)
        )
        if cp_size == 1:
            return hidden_states, compressed_input_ids

        # Compatibility for standalone layout helpers; Hybrid passes an explicit CP group.
        cp_group = parallel_state.get_context_parallel_group() if cp_group is None else cp_group
        hidden_states = cls._restore_cp(differentiable_all_gather(hidden_states, group=cp_group), 0)

        def gather_ids(value: Tensor) -> Tensor:
            gathered = [torch.empty_like(value) for _ in range(cp_size)]
            torch.distributed.all_gather(gathered, value, group=cp_group)
            return cls._restore_cp(gathered, 1)

        return hidden_states, gather_ids(compressed_input_ids)

    @classmethod
    def scatter(
        cls,
        hidden_states: Tensor,
        sequence_parallel: bool,
        tp_group: ProcessGroup | None = None,
        cp_group: ProcessGroup | None = None,
    ) -> Tensor:
        """Reshard fused hidden states back to the caller's sequence-parallel layout."""
        hidden_states = cls._select_cp(hidden_states, 0, cp_group=cp_group)
        if sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(
                hidden_states, group=tp_group
            )
        return hidden_states


class Engram(nn.Module):
    """Engram memory with a global addressing layout and explicitly owned local layers."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
        *,
        tokenizer_lookup: Tensor | None = None,
        pad_id: int | None = None,
        local_layer_ids: Sequence[int] | None = None,
        engram_config: EngramConfig | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__()
        if engram_config is None:
            if config is None:
                raise ValueError("Engram requires config or engram_config")
            engram_config = EngramConfig.from_transformer_config(config, pad_id=pad_id)
        if tokenizer_lookup is None:
            raise ValueError('Engram requires a precomputed tokenizer lookup table')
        if pg_collection is not None:
            required_groups = {'tp', 'pp', 'cp', 'dp_cp'}
            if engram_config.table_backend == 'row_a2a':
                required_groups.update({'dp', 'tp_dp_cp', 'mp'})
            missing_groups = sorted(
                name for name in required_groups if getattr(pg_collection, name, None) is None
            )
            if missing_groups:
                raise ValueError(
                    "Engram ProcessGroupCollection is missing required groups: "
                    + ", ".join(missing_groups)
                )
        self.compressed_tokenizer = CompressedTokenizer(tokenizer_lookup)
        self.pg_collection = pg_collection
        cfg = engram_config
        self.engram_config = cfg
        if cfg.pad_id is None:
            raise ValueError(
                "Engram requires the raw pad token ID from the main Megatron tokenizer"
            )
        self.hash_mapping = NgramHashMapping(
            hash_table_min_sizes=cfg.hash_table_min_sizes,
            max_ngram_size=cfg.max_ngram_size,
            num_hash_heads_per_ngram=cfg.num_hash_heads_per_ngram,
            layer_ids=cfg.layer_ids,
            pad_id=cfg.pad_id,
            seed=cfg.seed,
            compressed_tokenizer=self.compressed_tokenizer,
        )
        self.layers = nn.ModuleDict()
        owned_layer_ids = set(cfg.layer_ids if local_layer_ids is None else local_layer_ids)
        if not owned_layer_ids or not owned_layer_ids.issubset(cfg.layer_ids):
            raise ValueError('Local Engram layer IDs must be a non-empty subset of configured IDs')
        for layer_id in cfg.layer_ids:
            if layer_id not in owned_layer_ids:
                continue
            table_sizes = [
                size
                for sizes_for_ngram in self.hash_mapping.hash_moduli_by_layer[layer_id]
                for size in sizes_for_ngram
            ]
            self.layers[str(layer_id)] = EngramLayer(
                cfg, layer_id, table_sizes, pg_collection=pg_collection
            )

    def get_extra_state(self) -> dict[str, Any]:
        """Return schema metadata that must match when loading an Engram checkpoint."""
        cfg = self.engram_config
        state = {
            "schema_version": _ENGRAM_CHECKPOINT_SCHEMA_VERSION,
            "raw_vocab_size": int(self.compressed_tokenizer.lookup_table.numel()),
            "hash_table_min_sizes": cfg.hash_table_min_sizes,
            "hash_moduli_by_layer": self.hash_mapping.hash_moduli_by_layer,
            "max_ngram_size": cfg.max_ngram_size,
            "embedding_dim_per_ngram": cfg.embedding_dim_per_ngram,
            "num_hash_heads_per_ngram": cfg.num_hash_heads_per_ngram,
            "layer_ids": cfg.layer_ids,
            "pad_id": cfg.pad_id,
            "seed": cfg.seed,
            "kernel_size": cfg.kernel_size,
            "optimizer_policy": {
                "table_lr_multiplier": ENGRAM_TABLE_LR_MULTIPLIER,
                "table_weight_decay": ENGRAM_TABLE_WEIGHT_DECAY,
            },
            "implementation": "deepseek-engram-demo-torch-port",
        }
        if cfg.table_backend == "row_a2a":
            state["table_backend"] = "row_a2a"
        return state

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Validate loaded Engram metadata against the current module configuration."""
        expected = self.get_extra_state()
        if not isinstance(state, dict):
            raise ValueError(f"Invalid Engram checkpoint metadata: {state!r}")
        schema_version = state.get("schema_version")
        if schema_version != _ENGRAM_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Engram checkpoint schema version: {schema_version}")
        if state != expected:
            raise ValueError(f"Engram checkpoint architecture mismatch: {state} != {expected}")

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Replicate layer-owned addressing state across its TP/DP copies."""
        if sharded_offsets:
            raise ValueError("Engram does not accept parent sharded offsets")

        def group_rank(name, fallback):
            group = getattr(self.pg_collection, name, None)
            return torch.distributed.get_rank(group=group) if group is not None else fallback()

        tp_rank = group_rank(
            'tp',
            lambda: (
                parallel_state.get_tensor_model_parallel_rank()
                if parallel_state.model_parallel_is_initialized()
                else 0
            ),
        )
        dp_rank = group_rank(
            'dp_cp',
            lambda: (
                parallel_state.get_data_parallel_rank(with_context_parallel=True)
                if parallel_state.model_parallel_is_initialized()
                else 0
            ),
        )
        # The parent layer supplies its global position in the checkpoint prefix.
        # PP rank is ownership, not replication: every key needs a replica zero
        # even when its sole owner is on a nonzero pipeline rank.
        shared_replica_id = (0, tp_rank, dp_rank)
        tp_group = getattr(self.pg_collection, 'tp', None)
        dp_cp_group = getattr(self.pg_collection, 'dp_cp', None)
        result = {
            f'{prefix}_extra_state': make_sharded_object_for_checkpoint(
                self.get_extra_state(), f'{prefix}_extra_state', replica_id=shared_replica_id
            ),
            f'{prefix}compressed_tokenizer.lookup_table': make_sharded_tensor_for_checkpoint(
                self.compressed_tokenizer.lookup_table,
                f'{prefix}compressed_tokenizer.lookup_table',
                replica_id=shared_replica_id,
                tp_group=tp_group,
                dp_cp_group=dp_cp_group,
            ),
        }
        for name, buffer in self.hash_mapping.named_buffers(recurse=False):
            if name in self.hash_mapping._non_persistent_buffers_set:
                continue
            key = f'{prefix}hash_mapping.{name}'
            result[key] = make_sharded_tensor_for_checkpoint(
                buffer,
                key,
                replica_id=shared_replica_id,
                tp_group=tp_group,
                dp_cp_group=dp_cp_group,
            )
        layer_metadata = dict(metadata or {})
        if dp_cp_group is not None:
            layer_metadata['dp_cp_group'] = dp_cp_group
        for layer_id, layer in self.layers.items():
            result.update(
                sharded_state_dict_default(
                    layer, f'{prefix}layers.{layer_id}.', metadata=layer_metadata, tp_group=tp_group
                )
            )
        return result

    def compress_input_ids(self, input_ids: Tensor) -> Tensor:
        """Project raw token IDs into this memory module's compressed vocabulary."""
        return self.compressed_tokenizer(input_ids)

    def forward(
        self, hidden_states: Tensor, layer_id: int, compressed_input_ids: Tensor | None = None
    ) -> Tensor:
        """Look up this layer's table and fuse retrieved memory into hidden states."""
        layer_key = str(layer_id)
        if layer_key not in self.layers:
            return hidden_states
        if compressed_input_ids is None:
            raise ValueError(f"Engram layer ID {layer_id} requires compressed input IDs")
        tp_group = getattr(self.pg_collection, 'tp', None)
        cp_group = getattr(self.pg_collection, 'cp', None)
        hidden_states, compressed_input_ids = ParallelSequenceLayout.gather(
            hidden_states,
            compressed_input_ids,
            sequence_parallel=self.engram_config.sequence_parallel,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        hash_input_ids = self.hash_mapping.forward_compressed(compressed_input_ids, layer_id)
        embeddings = (
            self.layers[layer_key].multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        )
        hidden_states = self.layers[layer_key](hidden_states, embeddings)
        return ParallelSequenceLayout.scatter(
            hidden_states,
            sequence_parallel=self.engram_config.sequence_parallel,
            tp_group=tp_group,
            cp_group=cp_group,
        )

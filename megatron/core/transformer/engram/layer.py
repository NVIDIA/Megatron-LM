# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer-owned Engram memory and sequence-layout adaptation."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

from .addressing import NgramHashMapping
from .config import (
    _ENGRAM_CHECKPOINT_SCHEMA_VERSION,
    ENGRAM_TABLE_LR_MULTIPLIER,
    ENGRAM_TABLE_WEIGHT_DECAY,
    EngramConfig,
)
from .fusion import EngramLayer
from .memory import build_multi_head_embedding
from .parallel import EngramParallelGroups, ParallelSequenceLayout
from .tokenizer import CompressedTokenizer


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
        parallel_groups: EngramParallelGroups | None = None,
    ) -> None:
        super().__init__()
        if engram_config is None:
            if config is None:
                raise ValueError("Engram requires config or engram_config")
            engram_config = EngramConfig.from_transformer_config(config, pad_id=pad_id)
        if tokenizer_lookup is None:
            raise ValueError('Engram requires a precomputed tokenizer lookup table')
        # Standalone callers may bootstrap from initialized Megatron groups.
        if pg_collection is None and parallel_state.model_parallel_is_initialized():
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
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
        self.parallel_groups = parallel_groups
        if cfg.table_backend == 'row_a2a' and self.parallel_groups is None:
            if pg_collection is None:
                raise ValueError('Engram row tables require explicit model process groups')
            self.parallel_groups = EngramParallelGroups.create(
                pg_collection.tp_dp_cp,
                cfg.row_parallel_size,
                stats_group=torch.distributed.group.WORLD,
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
            memory = build_multi_head_embedding(
                cfg, layer_id, table_sizes, pg_collection, self.parallel_groups
            )
            self.layers[str(layer_id)] = EngramLayer(cfg, memory, pg_collection=pg_collection)

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
        """Use native replica metadata and each lookup backend's checkpoint factory."""
        if sharded_offsets:
            raise ValueError('Engram does not accept parent sharded offsets')
        tp_group = getattr(self.pg_collection, 'tp', None)
        metadata = dict(metadata or {})
        dp_cp_group = getattr(self.pg_collection, 'dp_cp', None)
        if dp_cp_group is not None:
            metadata['dp_cp_group'] = dp_cp_group
        result = make_sharded_tensors_for_checkpoint(
            {'_extra_state': self.get_extra_state()},
            prefix,
            tp_group=tp_group,
            dp_cp_group=metadata.get('dp_cp_group'),
        )
        modules = {
            'compressed_tokenizer': self.compressed_tokenizer,
            'hash_mapping': self.hash_mapping,
        }
        modules.update({f'layers.{key}': value for key, value in self.layers.items()})
        for name, module in modules.items():
            result.update(
                sharded_state_dict_default(
                    module, f'{prefix}{name}.', metadata=metadata, tp_group=tp_group
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

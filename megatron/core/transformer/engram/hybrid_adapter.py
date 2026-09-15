# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Engram ownership and pre-attention composition for HybridModel."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from torch import Tensor

from megatron.core.context_parallel import ContextParallelBatch, convert_cp_layout
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, get_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

from .layer import Engram


def resolve_engram_targets(
    memory_ids: Sequence[int],
    hybrid_layer_pattern: str,
    target_layer_indices: Sequence[int] | None = None,
) -> dict[int, int]:
    """Map main-stack attention positions to stable Engram memory IDs.

    Without explicit positions, a memory ID selects that occurrence of ``*``.
    Explicit positions keep memory identity independent of the surrounding stack.
    Pipeline separators and MTP suffixes never contribute to main-stack positions.
    Other supported Hybrid layer families may occur between selected attentions.
    """
    parsed = parse_hybrid_pattern(hybrid_layer_pattern)
    main_pattern = (parsed.main_pattern or '').replace('|', '')
    memory_ids = tuple(memory_ids)
    if not memory_ids or len(set(memory_ids)) != len(memory_ids) or min(memory_ids) < 0:
        raise ValueError('Engram memory IDs must be non-empty, unique, and non-negative')
    attention_positions = [i for i, symbol in enumerate(main_pattern) if symbol == '*']
    if target_layer_indices is None:
        if max(memory_ids) >= len(attention_positions):
            raise ValueError('Engram memory ID is outside the available attention occurrences')
        positions = [attention_positions[memory_id] for memory_id in memory_ids]
    else:
        positions = list(target_layer_indices)
    if len(positions) != len(memory_ids) or len(set(positions)) != len(positions):
        raise ValueError(
            'Engram target positions must be unique and match the number of memory IDs'
        )
    if any(position not in attention_positions for position in positions):
        raise ValueError(
            "Every Engram target position must select a standard attention ('*') layer"
        )
    return dict(zip(positions, memory_ids, strict=True))


class EngramAttentionLayer(TransformerLayer):
    """Own one Engram memory and apply its residual before attention normalization."""

    accepts_token_context = True

    def __init__(
        self,
        *args: Any,
        memory_id: int,
        engram_model_config: TransformerConfig,
        tokenizer_lookup: Tensor,
        pad_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engram_memory_id = memory_id
        # Resolve hashes against the complete memory-ID list, then allocate only
        # this consumer's parameters. Moving a layer between PP/VPP segments does
        # not change its table capacities, addressing seeds, or checkpoint keys.
        self.engram = Engram(
            config=engram_model_config,
            local_layer_ids=[memory_id],
            tokenizer_lookup=tokenizer_lookup,
            pad_id=pad_id,
            pg_collection=self.pg_collection,
        )

    def forward(
        self, *args: Any, token_context: Tensor | None = None, **kwargs: Any
    ) -> tuple[Tensor, Tensor | None]:
        """Fuse explicit raw tokens before the parent captures its attention residual."""
        if token_context is None:
            raise ValueError('Engram attention requires explicit token context')
        compressed_tokens = self.engram.compress_input_ids(token_context)
        if args:
            args = (self.engram(args[0], self.engram_memory_id, compressed_tokens), *args[1:])
        else:
            kwargs['hidden_states'] = self.engram(
                kwargs['hidden_states'], self.engram_memory_id, compressed_tokens
            )
        return super().forward(*args, **kwargs)


class EngramHybridProvider:
    """Prepare layer specifications and raw tokens without owning model parameters.

    Every consumer registers its own memory, fusion, and addressing state. This
    provider retains only construction inputs; no microbatch state survives a call.
    """

    def __init__(
        self,
        config: TransformerConfig,
        tokenizer_lookup: Tensor,
        pad_id: int,
        hybrid_layer_pattern: str,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        self.targets = resolve_engram_targets(
            config.engram_layer_ids or (), hybrid_layer_pattern, config.engram_target_layer_indices
        )
        main_pattern = (parse_hybrid_pattern(hybrid_layer_pattern).main_pattern or '').replace(
            '|', ''
        )
        if config.num_layers != len(main_pattern):
            raise ValueError('Engram main hybrid pattern length must match config.num_layers')
        unsupported = (
            'enable_mhc_connections',
            'freeze_base_model_for_mtp',
            'fine_grained_activation_offloading',
            'cpu_offloading',
            'moe_paged_stash',
            'overlap_moe_expert_parallel_comm',
            'use_megatron_fsdp',
            'use_torch_fsdp2',
            'init_model_with_meta_device',
            'multi_latent_attention',
            'experimental_attention_variant',
            'fp8',
            'fp4',
        )
        enabled = [name for name in unsupported if getattr(config, name, False)]
        if enabled:
            raise ValueError('Engram does not support: ' + ', '.join(enabled))
        if config.cuda_graph_impl != 'none':
            raise ValueError('Engram does not support CUDA graphs')
        if pg_collection is None:
            raise ValueError('Engram requires explicit model process groups')
        if tokenizer_lookup is None or pad_id is None:
            raise ValueError('Engram requires an explicit tokenizer lookup and raw pad token ID')
        self.config = config
        self.tokenizer_lookup = tokenizer_lookup
        self.pad_id = pad_id
        self.pg_collection = pg_collection

    def layer_spec_overrides(
        self, submodules: object, layer_config_list: Sequence[TransformerConfig], layer_offset: int
    ) -> dict[int, ModuleSpec]:
        """Replace selected standard attention implementations in this PP/VPP segment."""
        local_targets = {
            position: memory_id
            for position, memory_id in self.targets.items()
            if layer_offset <= position < layer_offset + len(layer_config_list)
        }
        if not local_targets:
            return {}
        attention_spec = getattr(submodules, 'attention_layer', None)
        if (
            not isinstance(attention_spec, ModuleSpec)
            or get_module(attention_spec) is not TransformerLayer
        ):
            raise ValueError('Engram requires a standard TransformerLayer attention ModuleSpec')
        if any(
            type(layer_config_list[position - layer_offset]) is not AttentionLayerConfig
            for position in local_targets
        ):
            raise ValueError('Engram targets require standard attention layer configurations')
        return {
            position: replace(
                attention_spec,
                module=EngramAttentionLayer,
                params={
                    **attention_spec.params,
                    'memory_id': memory_id,
                    'engram_model_config': self.config,
                    'tokenizer_lookup': self.tokenizer_lookup,
                    'pad_id': self.pad_id,
                },
            )
            for position, memory_id in local_targets.items()
        }

    def prepare(
        self,
        input_ids: Tensor | None,
        *,
        inference_context: BaseInferenceContext | None,
        packed_seq_params: PackedSeqParams | None,
        cp_batch: ContextParallelBatch | None,
    ) -> Tensor:
        """Return this microbatch's raw tokens in the consumer attention CP layout.

        Tokens are replicated across TP ranks, including when hidden states use
        sequence parallelism. Layer-local Engram gathers hidden states and tokens
        into the same complete sequence before hashing and convolution.
        """
        if inference_context is not None or InferenceMode.is_active():
            raise NotImplementedError('Engram does not support inference or decoding history')
        packed_cp = cp_batch is not None and (
            cp_batch.thd_plan is not None
            or any(value is not None for value in cp_batch.packed_seq_params_by_layout.values())
        )
        if packed_seq_params is not None or packed_cp:
            raise ValueError('Engram does not support packed sequences')
        if input_ids is None:
            raise ValueError('Engram requires input_ids to build the microbatch token context')
        if input_ids.ndim != 2:
            raise ValueError('Engram input_ids must have shape [batch, sequence]')
        source_layout = (
            cp_batch.boundary_layout if cp_batch is not None else self.config.linear_cp_layout
        )
        target_layout = self.config.attention_cp_layout
        if cp_batch is not None and target_layout in cp_batch.batches_by_layout:
            tokens = cp_batch.get_batch(target_layout).get('tokens')
            if tokens is not None:
                return tokens
        if self.pg_collection.cp.size() == 1 or source_layout == target_layout:
            return input_ids
        return (
            convert_cp_layout(
                input_ids.transpose(0, 1).contiguous(),
                source_layout,
                target_layout,
                cp_group=self.pg_collection.cp,
            )
            .transpose(0, 1)
            .contiguous()
        )

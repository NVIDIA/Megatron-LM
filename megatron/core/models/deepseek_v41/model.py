# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 text training composed on main's HybridModel and typed pipeline schedules."""

from dataclasses import replace

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.spec_utils import ModuleSpec

from .engram import V41Engram, consume_legacy_hash_buffers, validate_v41_layers
from .hybrid_adapter import V41ForwardContext, deepseek_v41_stack_spec
from .state import BatchShape, V41PipelineCodec, V41StateCodec


class DeepSeekV41Model(HybridModel):
    """Keep embeddings, layer execution, output head and checkpointing in HybridModel."""

    def __init__(self, config, vocab_size, max_sequence_length, *, pg_collection, **kwargs):
        pattern = kwargs.pop('hybrid_layer_pattern', config.hybrid_pattern)
        stack = deepseek_v41_stack_spec
        options = config.engram_config
        if options is not None:
            validate_v41_layers(options, config.num_layers)
            options.validate_startup(
                config, expected_tokenizer_vocab_size=vocab_size, hybrid_layer_pattern=pattern
            )
            memory = ModuleSpec(module=V41Engram, params={"engram_config": options})
            layer = stack.submodules.csa2_layer
            layer = replace(layer, submodules=replace(layer.submodules, engram=memory))
            stack = replace(stack, submodules=replace(stack.submodules, csa2_layer=layer))
        super().__init__(
            config=config,
            hybrid_stack_spec=stack,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_layer_pattern=pattern,
            position_embedding_type='none',
            pg_collection=pg_collection,
            pipeline_state_codec=V41PipelineCodec(config, pg_collection.cp),
            **kwargs,
        )
        if options is not None:
            # This adapter owns lookup before graph or checkpoint regions. The generic
            # inner-layer injection would otherwise add the same memory a second time.
            for layer in self.decoder.layers:
                inner = getattr(layer, "inner_layer", None)
                if getattr(inner, "engram", None) is not None:
                    layer.engram, inner.engram = inner.engram, None

    def _load_from_state_dict(self, state_dict, prefix, *args):
        consume_legacy_hash_buffers(self.config.engram_config, state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask=None,
        *,
        packed_seq_params=None,
        labels=None,
        padding_mask=None,
        capture_layers=None,
        graph_slots=1,
        graph_backend='torch',
        **kwargs,
    ):
        """Run text training with packed CP data and explicit cross-stage state.

        ``capture_layers`` is an explicit warmup/capture request for zero-based physical
        layers. Normal training reuses those profiles. Engram runs eagerly before its
        captured consumer. Reserve one slot per outstanding microbatch; full recomputation
        uses a separate execution profile.
        """
        adapter = self.decoder.forward_adapter
        stage = self.pipeline_stage.enter(input_ids, packed_seq_params)
        # The typed stage owns token delivery, including on data-less downstream ranks.
        input_ids = stage.values.get("tokens", input_ids)
        codec = V41StateCodec(self.config, BatchShape(*stage.metadata), self.pg_collection.cp)
        # CSA2 derives padding and causal visibility from packed prefixes. An SBHD
        # padding mask would silently disagree with the sparse attention support.
        if padding_mask is not None:
            raise ValueError('V4.1 padding is represented by packed logical/physical prefixes')
        if attention_mask is not None:
            raise ValueError('V4.1 attention support is defined by its causal packed layout')
        csa, mhc, _ = codec.restore(stage.values, adapter.start)
        context = V41ForwardContext(
            cross_layer_state=csa,
            mhc_state=mhc,
            codec=codec,
            values=stage.values,
            token_ids=stage.values.get("tokens"),
            metadata=stage.metadata,
            packed_seq_params=stage.packed_seq_params,
            capture=(
                dict(layers=set(capture_layers), slots=graph_slots, backend=graph_backend)
                if capture_layers is not None
                else None
            ),
        )
        return super().forward(
            input_ids,
            position_ids,
            None,
            labels=labels,
            packed_seq_params=stage.packed_seq_params,
            forward_context=context,
            pipeline_state=context,
            **kwargs,
        )

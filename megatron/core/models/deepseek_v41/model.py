# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 text training composed on main's HybridModel and typed pipeline schedules."""

import torch

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.pipeline_payload import TensorStatePayload
from megatron.core.utils import get_pg_size

from .engram import Engram, EngramHasher
from .engram_hash import build_compressed_token_map
from .hybrid_adapter import V41ForwardContext, deepseek_v41_stack_spec
from .state import BatchShape, V41StateCodec


class DeepSeekV41Model(HybridModel):
    """Keep embeddings, layer execution, output head and checkpointing in HybridModel."""

    pipeline_payload_factory = TensorStatePayload

    def __init__(
        self,
        config,
        vocab_size,
        max_sequence_length,
        *,
        pg_collection,
        token_map=None,
        tokenizer=None,
        **kwargs,
    ):
        pattern = kwargs.pop('hybrid_layer_pattern', config.hybrid_pattern)
        super().__init__(
            config=config,
            hybrid_stack_spec=deepseek_v41_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_layer_pattern=pattern,
            position_embedding_type='none',
            pg_collection=pg_collection,
            **kwargs,
        )
        self.engram_hash = None
        self._engram_hash_replica_index = 0
        options = config.engram_config
        local_memory = (
            []
            if options is None
            else [
                layer
                for layer in self.decoder.layers
                if (layer.layer_number - 1) // 2 in options.layer_ids
                and layer.layer_number % 2 == 1
            ]
        )
        if local_memory:
            # Hasher buffers are identical on every Engram-bearing PP/VPP stage.
            # Select one main replica by the first configured Engram owner, even
            # when it is not on the embedding stage.
            self._engram_hash_replica_index = min(
                options.layer_ids.index((layer.layer_number - 1) // 2) for layer in local_memory
            )
            if token_map is None:
                if tokenizer is None:
                    raise ValueError(
                        'Engram requires the released tokenizer or compressed token map'
                    )
                token_map, _ = build_compressed_token_map(tokenizer)
                token_map = torch.tensor(token_map, dtype=torch.long)
            self.engram_hash = EngramHasher(options, token_map)
            for layer in local_memory:
                layer.engram = Engram(
                    config, self.engram_hash.layout, (layer.layer_number - 1) // 2, pg_collection
                )
        self._incoming_payload = None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Save shared hash buffers once across PP while retaining TP/DP replica IDs."""
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.engram_hash is not None:
            for name in self.engram_hash.state_dict():
                shard = state[f'{prefix}engram_hash.{name}']
                shard.replica_id = (self._engram_hash_replica_index, *shard.replica_id[1:])
        return state

    def set_input_tensor(self, input_tensor):
        """Retain the received payload only until this microbatch's forward starts."""
        value = input_tensor[0] if isinstance(input_tensor, list) else input_tensor
        self._incoming_payload = value if isinstance(value, TensorStatePayload) else None
        if self._incoming_payload is not None:
            hidden, _ = value.restore()
            super().set_input_tensor(hidden)
        else:
            super().set_input_tensor(value)

    def pipeline_payload_spec(self, shape: BatchShape):
        """Return matching host descriptors for this physical/virtual stage."""
        adapter = self.decoder.forward_adapter
        codec = V41StateCodec(self.config, shape, self.pg_collection.cp)
        return (
            None if self.pre_process else codec.descriptor(adapter.start),
            None if self.post_process else codec.descriptor(adapter.end),
        )

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
        layers. Normal training reuses those profiles. Variable-split Engram lookups and
        full recomputation stay outside capture. Reserve one slot per outstanding microbatch.
        """
        adapter = self.decoder.forward_adapter
        incoming, self._incoming_payload = self._incoming_payload, None
        if incoming is None:
            if not self.pre_process:
                raise ValueError('Non-embedding V4.1 stages require a typed tensor-state payload')
            shape = BatchShape.from_batch(
                input_ids, packed_seq_params, get_pg_size(self.pg_collection.cp)
            )
            values = {}
            codec = V41StateCodec(self.config, shape, self.pg_collection.cp)
            if codec.last_token_layer >= 0:
                values['tokens'] = input_ids
            if packed_seq_params is not None:
                cu = packed_seq_params.cu_seqlens_q
                padded = packed_seq_params.cu_seqlens_q_padded
                values.update(cu=cu.int(), cu_padded=(cu if padded is None else padded).int())
        else:
            if incoming.spec.boundary_id != f'v41:{adapter.start}':
                raise ValueError('Received V4.1 payload belongs to another pipeline boundary')
            _, values = incoming.restore()
            codec = V41StateCodec(
                self.config, BatchShape(*incoming.spec.metadata), self.pg_collection.cp
            )
            packed_seq_params = codec.packed(values)
        # CSA2 derives padding and causal visibility from packed prefixes. An SBHD
        # padding mask would silently disagree with the sparse attention support.
        if padding_mask is not None:
            raise ValueError('V4.1 padding is represented by packed logical/physical prefixes')
        if attention_mask is not None:
            raise ValueError('V4.1 attention support is defined by its causal packed layout')
        csa, mhc, _ = codec.restore(values, adapter.start)
        context = V41ForwardContext(
            cross_layer_state=csa,
            mhc_state=mhc,
            codec=codec,
            values=values,
            hasher=self.engram_hash,
            capture=(
                dict(layers=set(capture_layers), slots=graph_slots, backend=graph_backend)
                if capture_layers is not None
                else None
            ),
        )
        output = super().forward(
            input_ids,
            position_ids,
            None,
            labels=labels,
            packed_seq_params=packed_seq_params,
            forward_context=context,
            **kwargs,
        )
        if self.post_process:
            return output
        return codec.payload(output, context.values, adapter.end)

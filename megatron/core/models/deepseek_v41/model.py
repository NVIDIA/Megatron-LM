# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 text training composed on main's HybridModel and typed pipeline schedules."""

import torch

from megatron.core.models.engram.distributed_embedding import EngramRuntime
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import TensorStatePayload
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    get_thd_token_metadata,
)
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
        pipeline_control_group=None,
        engram_runtime=None,
        engram_input_cache=None,
        **kwargs,
    ):
        if get_pg_size(pg_collection.pp) > 1 and pipeline_control_group is None:
            raise ValueError('V4.1 PP requires an explicitly created pipeline control group')
        self.pipeline_control_group = pipeline_control_group
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
        self.engram_runtime = engram_runtime
        self.engram_input_cache = engram_input_cache if engram_input_cache is not None else {}
        if options is not None and (
            options.prefetch
            or options.defer_grad_transfer
            or options.communication_backend == 'symmetric'
        ):
            if self.engram_runtime is None:
                self.engram_runtime = EngramRuntime(
                    options.buffer_budget_mb, defer_grad_transfer=options.defer_grad_transfer
                )
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
                if self.engram_runtime is not None:
                    self.engram_runtime.attach(layer.engram.embed)
        self._incoming_payload = None

    def stage_engram_inputs(self, batches):
        """Make token/THD metadata available before PP activations; share it across VPP."""
        if self.vp_stage not in (None, 0):
            if len(self.engram_input_cache) != len(batches):
                raise RuntimeError('Engram VPP lookahead must prepare chunk zero first')
            return
        self.engram_input_cache.clear()
        if self.engram_runtime is not None:
            self.engram_runtime.reset()
        pp = self.pg_collection.pp
        source = pp is None or pp.rank() == 0
        descriptors = None
        if source:
            descriptors = []
            for batch in batches:
                tokens, packed = batch.get_batch()['tokens'], batch.get_packed_seq_params()
                descriptors.append(
                    (
                        tuple(tokens.shape),
                        (
                            None
                            if packed is None
                            else (packed.cu_seqlens_q.numel(), packed.max_seqlen_q)
                        ),
                    )
                )
        if get_pg_size(pp) > 1:
            metadata = [descriptors]
            torch.distributed.broadcast_object_list(
                metadata,
                src=torch.distributed.get_global_rank(pp, 0),
                group=self.pipeline_control_group,
            )
            descriptors = metadata[0]
        device = next(self.parameters()).device
        for index, (batch, (shape, lengths)) in enumerate(zip(batches, descriptors)):
            packed = batch.get_packed_seq_params() if source else None
            tokens = (
                batch.get_batch()['tokens'].clone()
                if source
                else torch.empty(shape, dtype=torch.long, device=device)
            )
            tensors = [tokens]
            if lengths is not None:
                count, maximum = lengths
                cu = (
                    packed.cu_seqlens_q.int().clone()
                    if source
                    else torch.empty(count, dtype=torch.int32, device=device)
                )
                padded = (
                    (
                        packed.cu_seqlens_q
                        if packed.cu_seqlens_q_padded is None
                        else packed.cu_seqlens_q_padded
                    )
                    .int()
                    .clone()
                    if source
                    else torch.empty_like(cu)
                )
                tensors.extend((cu, padded))
            if get_pg_size(pp) > 1:
                for tensor in tensors:
                    torch.distributed.broadcast(
                        tensor, src=torch.distributed.get_global_rank(pp, 0), group=pp
                    )
            if lengths is not None:
                packed = PackedSeqParams(
                    qkv_format='thd',
                    cu_seqlens_q=cu,
                    cu_seqlens_kv=cu,
                    cu_seqlens_q_padded=padded,
                    cu_seqlens_kv_padded=padded,
                    max_seqlen_q=maximum,
                    max_seqlen_kv=maximum,
                    cp_group=self.pg_collection.cp,
                )
            self.engram_input_cache[index] = (tokens, packed)

    def prefetch_engram_batch(self, index):
        """Return model-local tickets; only tensors/metadata travel between PP stages."""
        if self.engram_hash is None:
            return {}
        tokens, packed = self.engram_input_cache[index]
        valid = sequence = None
        if packed is not None:
            group = self.pg_collection.cp
            rank = group.rank() if group is not None else 0
            rows = torch.arange(tokens.shape[1], device=tokens.device) + rank * tokens.shape[1]
            sequence, _, valid = get_thd_token_metadata(
                packed.cu_seqlens_q, packed.cu_seqlens_q_padded, rows
            )
            sequence, valid = sequence.unsqueeze(0), valid.unsqueeze(0)
        hashes = self.engram_hash(tokens, valid, sequence, cp_group=self.pg_collection.cp)
        tickets = {}
        for layer in self.decoder.layers:
            engram = getattr(layer, 'engram', None)
            if engram is not None:
                tickets[layer.layer_number - 1] = self.engram_runtime.prefetch(
                    engram.embed, hashes[:, :, engram.hash_index].unsqueeze(-1)
                )
        return {'engram_tickets': tickets}

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
        engram_tickets=None,
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
                # Loader buffers may be reused while a pipeline backward is outstanding.
                values['tokens'] = input_ids.clone()
            if packed_seq_params is not None:
                cu = packed_seq_params.cu_seqlens_q
                padded = packed_seq_params.cu_seqlens_q_padded
                values.update(
                    cu=cu.int().clone(), cu_padded=(cu if padded is None else padded).int().clone()
                )
        else:
            if incoming.boundary_id != f'v41:{adapter.start}':
                raise ValueError('Received V4.1 payload belongs to another pipeline boundary')
            _, values = incoming.restore()
            codec = V41StateCodec(
                self.config, BatchShape(*incoming.metadata), self.pg_collection.cp
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
            engram_tickets=engram_tickets or {},
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

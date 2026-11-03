# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1's live fields on the common execution and pipeline tensor contract."""

from dataclasses import astuple, dataclass, replace

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelinePayloadSpec,
    PipelineTensorSpec,
    TensorStatePayload,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CSA2State,
    csa2_source_layers,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    build_csa2_thd_layout,
    get_thd_compressed_capacity,
)
from megatron.core.transformer.hyper_connection import SinglePassMHCState
from megatron.core.transformer.state_boundary import TensorField, TensorSchema


@dataclass(frozen=True)
class BatchShape:
    """Host geometry shared by PP plans and execution; no device length reads."""

    sequence: int
    batch: int
    max_seqlen: int = 0
    num_sequences: int = 0
    cp_size: int = 1

    @property
    def packed(self):
        """Whether packed prefixes are part of this profile."""
        return self.num_sequences > 0

    @classmethod
    def from_batch(cls, tokens, packed, cp_size=1):
        """Use batch capacity and explicit max_seqlen, never a CUDA scalar."""
        if packed is None:
            if cp_size != 1:
                raise ValueError("V4.1 CP batches require packed contiguous metadata")
            return cls(tokens.shape[1], tokens.shape[0])
        if tokens.shape[0] != 1 or packed.qkv_format != 'thd':
            raise ValueError("Packed V4.1 batches use one physical token row")
        return cls(
            tokens.shape[1], 1, packed.max_seqlen_q, packed.cu_seqlens_q.numel() - 1, cp_size
        )


def live_fields(config, boundary):
    """Find only producer tensors read after this boundary and before replacement."""
    live = set()
    for layer in range(boundary, config.num_layers):
        if not config.csa_compress_ratios[layer]:
            continue
        owner, index = csa2_source_layers(config, layer)
        if owner < boundary:
            live.add('global_kv')
            if layer in config.csa2_index_source_layers:
                live.add('indexer_k')
        if index < boundary:
            live.add('global_indices')
        source = config.csa2_candidate_source_layer
        if (
            source is not None
            and source < boundary <= layer
            and layer in config.csa2_index_source_layers
        ):
            live.update(('candidate_ids', 'candidate_counts'))
    return live


class V41StateCodec:
    """One V4.1-specific schema shared by checkpoint, graph and PP boundaries.

    Derived packed layouts and flattened KV are rebuilt at boundaries. Eager layers
    keep their cached views. Integer selection fields never receive gradients.
    """

    def __init__(self, config, shape: BatchShape, cp_group=None):
        self.config, self.shape, self.cp_group = config, shape, cp_group
        options = getattr(config, 'engram_config', None)
        self.last_token_layer = (
            max((2 * i for i in options.layer_ids), default=-1) if options else -1
        )

    def schema(self, boundary, *, loss=False):
        """Describe a boundary using source schedules and host-known capacities."""
        c, p = self.config, self.shape
        fields = []

        def add(key, shape, dtype, grad=False):
            fields.append(TensorField(key, tuple(shape), dtype, 'contiguous', grad))

        if boundary > 0 and c.mhc_single_pass:
            add('mhc', (p.sequence, p.batch, c.mhc_num_residual_streams), torch.float32, True)
        if p.packed:
            for key in ('cu', 'cu_padded'):
                add(key, (p.num_sequences + 1,), torch.int32)
        if boundary <= self.last_token_layer:
            add('tokens', (p.batch, p.sequence), torch.int64)
        live = live_fields(c, boundary)
        if live:
            owner = max(i for i in c.csa2_kv_source_layers if i < boundary)
            ratio = c.csa_compress_ratios[owner]
            capacity = (
                get_thd_compressed_capacity(
                    p.sequence * p.cp_size, p.max_seqlen, p.num_sequences, ratio
                )
                if p.packed
                else p.sequence // ratio
            )
            if 'global_kv' in live:
                add('global_kv', (capacity, p.batch, c.v_head_dim), c.params_dtype, True)
            if 'indexer_k' in live:
                add(
                    'indexer_k',
                    (capacity, p.batch, c.dsa_indexer_head_dim),
                    c.params_dtype,
                    (c.dsa_indexer_loss_coeff or 0) > 0,
                )
            query_shape = (p.sequence,) if p.packed else (p.batch, p.sequence)
            if 'global_indices' in live:
                add(
                    'global_indices', (*query_shape, min(c.dsa_indexer_topk, capacity)), torch.int32
                )
            if 'candidate_ids' in live:
                max_keys = p.max_seqlen // ratio if p.packed else capacity
                blocks = (max_keys + c.csa2_candidate_block_size - 1) // c.csa2_candidate_block_size
                add(
                    'candidate_ids',
                    (*query_shape, min(c.csa2_candidate_topk_blocks, blocks)),
                    torch.int32,
                )
                add('candidate_counts', query_shape, torch.int32)
        if loss:
            add('indexer_loss', (), torch.float32, True)
        return TensorSchema(tuple(fields))

    def packed(self, state):
        """Restore packed metadata from explicitly transported prefix tensors."""
        if not self.shape.packed:
            return None
        return PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=state['cu'],
            cu_seqlens_kv=state['cu'],
            cu_seqlens_q_padded=state['cu_padded'],
            cu_seqlens_kv_padded=state['cu_padded'],
            max_seqlen_q=self.shape.max_seqlen,
            max_seqlen_kv=self.shape.max_seqlen,
            cp_group=self.cp_group,
        )

    def restore(self, state, boundary):
        """Create fresh Python state; original float tensors keep their gradient edges."""
        c = self.config
        result = CSA2State(attention_layer_ids=frozenset(range(0, c.num_layers, 2)))
        result.defer_indexer_loss = True
        result.global_kv = state.get('global_kv')
        result.indexer_k = state.get('indexer_k')
        result.global_indices = state.get('global_indices')
        prior = [i for i in c.csa2_kv_source_layers if i < boundary]
        if prior:
            result.kv_source_layer = prior[-1]
            result.index_source_layer = max(i for i in c.csa2_index_source_layers if i < boundary)
        if 'candidate_ids' in state:
            result.candidates = CSA2CandidateBlocks(
                state['candidate_ids'], state['candidate_counts'], c.csa2_candidate_block_size
            )
            result.candidate_source_layer = c.csa2_candidate_source_layer
        packed = self.packed(state)
        if packed is not None:
            result.thd_layout = build_csa2_thd_layout(
                packed, self.shape.sequence, cp_group=self.cp_group
            )
            if prior:
                result.compressed_layout = result.thd_layout.for_compression(
                    c.csa_compress_ratios[prior[-1]]
                )
        if c.dsa_kernel_backend == 'cudnn':
            result.prepare_fused_kv()
        return result, SinglePassMHCState(state.get('mhc')), packed

    def snapshot(self, csa, mhc, state, boundary, *, loss=False, schema=None):
        """Publish live fields and discard consumed owners without detaching their uses."""
        values = dict(state)
        values.update(
            global_kv=csa.global_kv,
            indexer_k=csa.indexer_k,
            global_indices=csa.global_indices,
            mhc=mhc.pre_mix,
            indexer_loss=csa.indexer_loss,
        )
        if csa.candidates is not None:
            values.update(
                candidate_ids=csa.candidates.indices, candidate_counts=csa.candidates.lengths
            )
        schema = self.schema(boundary, loss=loss) if schema is None else schema
        values = {f.key: values[f.key] for f in schema.fields}
        for field in schema.fields:
            if not field.differentiable:
                values[field.key] = values[field.key].detach()
        schema.pack(values)
        return values

    def descriptor(self, boundary):
        """Prepare a wire schema before any activation has been computed."""
        c, p = self.config, self.shape
        width = c.hidden_size * c.mhc_num_residual_streams
        hidden = TensorField(
            '__hidden__', (p.sequence, p.batch, width), c.params_dtype, 'strided', True
        )
        specs = tuple(
            PipelineTensorSpec.from_field(f) for f in (hidden, *self.schema(boundary).fields)
        )
        return PipelinePayloadSpec(specs, astuple(p), f'v41:{boundary}')

    def payload(self, hidden, state, boundary):
        """Create an owned hidden root while sharing side-field storage with autograd."""
        payload = TensorStatePayload.from_state(
            hidden, state, self.schema(boundary), boundary_id=f'v41:{boundary}'
        )
        return replace(payload, spec=replace(payload.spec, metadata=astuple(self.shape)))

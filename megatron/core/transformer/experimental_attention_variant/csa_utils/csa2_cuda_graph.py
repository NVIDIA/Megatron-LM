# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit tensor boundaries for CSA2 Hybrid Transformer Engine CUDA Graphs.

Only tensors consumed or produced by the current layer cross its graph boundary.
Each capture invocation reconstructs fresh Python state. Replay publishes graph
outputs to the caller's forward-local state before the ordinary layer continuation.
"""

from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CSA2State,
    CSA2StateCodec,
    csa2_field_factory,
    csa2_source_layers,
    prepare_csa2_boundary,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    CSA2THDCompressionLayout,
    CSA2THDLayout,
    build_csa2_thd_layout,
    get_thd_compressed_capacity,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    use_fused_dsa_kernels,
)
from megatron.core.transformer.state_boundary import (
    BoundarySchema,
    StateBoundary,
    StatePlacement,
    StateRegion,
    TensorField,
    TensorSchema,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig

_PREFIX = "csa2_graph_"
_LAYOUT_FIELDS = ("cu_seqlens", "cu_seqlens_padded", "sequence_ids", "position_ids", "valid_tokens")
_COMPRESSED_FIELDS = (
    "cu_seqlens",
    "cu_seqlens_padded",
    "sequence_ids",
    "position_ids",
    "source_indices",
    "valid_groups",
)


class CSA2GraphState:
    """CSA2 graph fields and native cache/aux-loss semantics for the common boundary."""

    def __init__(
        self,
        config: MLATransformerConfig,
        *,
        layer_number: int,
        is_attention: bool,
        cp_group: torch.distributed.ProcessGroup | None = None,
        placement: tuple[StatePlacement, ...] | None = None,
    ) -> None:
        self.config = config
        # Standalone callers describe one local graph. Hybrid always supplies
        # its complete bound physical/virtual placement to every backend.
        self.placement = (
            (StatePlacement(0, 2 * config.num_layers, 0),) if placement is None else placement
        )
        if config.context_parallel_size > 1 and cp_group is None:
            raise ValueError("CSA2 CP CUDA Graphs require an explicit CP group")
        self.cp_group = cp_group
        self.cp_size, self.cp_rank = (
            (cp_group.size(), cp_group.rank()) if cp_group is not None else (1, 0)
        )
        if self.cp_size != config.context_parallel_size:
            raise ValueError("CSA2 CUDA Graph CP group size must match the static configuration")
        self.layer_idx = layer_number - 1
        self.is_attention = is_attention
        self.indexer_loss_enabled = (config.dsa_indexer_loss_coeff or 0.0) > 0
        self.ratio = config.csa_compress_ratios[self.layer_idx] if is_attention else 0
        self.full = bool(self.ratio and self.layer_idx in config.csa2_kv_source_layers)
        self.reindex = bool(self.ratio and self.layer_idx in config.csa2_index_source_layers)
        self.candidate = config.csa2_candidate_source_layer
        self.kv_source, self.index_source = csa2_source_layers(config, self.layer_idx)
        self.codec = CSA2StateCodec(indexer_k_differentiable=self.indexer_loss_enabled)
        self.input_fields = tuple(
            self._field_name(field) for field in self._state_schema(0, 1, None).inputs
        )
        # Cache tensor-free declarations only. Every capture/replay still restores
        # or binds that invocation's native state and packed batch tensors.
        self._graph_regions: dict[tuple, StateRegion] = {}

    @staticmethod
    def _field_name(field: TensorField) -> str:
        return field.key.split("/", 1)[1].split(":", 1)[0]

    def _state_schema(self, seq, batch, params) -> BoundarySchema:
        factory = csa2_field_factory(
            self.config,
            seq,
            batch,
            qkv_format="sbhd" if params is None else "thd",
            max_seqlen=None if params is None else params.max_seqlen_q,
            num_sequences=None if params is None else params.cu_seqlens_q.numel() - 1,
            cp_size=self.cp_size,
        )
        boundary = StateBoundary(
            f"csa2.decoder/capture:L{self.layer_idx}",
            "capture",
            2 * self.layer_idx,
            2 * self.layer_idx + 2,
        )
        schema, _ = prepare_csa2_boundary(self.config, factory, boundary, self.placement)
        return schema

    def _packed_params(self, kwargs: Mapping[str, Any]) -> PackedSeqParams | None:
        if not self.is_attention:
            return None
        if kwargs.get("packed_seq_params") is not None:
            params = kwargs["packed_seq_params"]
        elif self.cp_size > 1:
            raise ValueError("CSA2 CP CUDA Graphs require contiguous THD metadata")
        else:
            return None
        if self.cp_size > 1 and (
            params.qkv_format != "thd" or params.cp_partition_mode != "contiguous"
        ):
            raise ValueError("CSA2 CP CUDA Graphs require contiguous THD metadata")
        if params.local_cp_size not in (None, self.cp_size):
            raise ValueError("CSA2 CUDA Graph local_cp_size must match the captured CP group")
        if params.cp_group is not None:
            group = params.cp_group
            if (group.size(), group.rank()) != (self.cp_size, self.cp_rank) or (
                self.cp_size > 1 and group is not self.cp_group
            ):
                raise ValueError("CSA2 CUDA Graph replay must use the captured CP group")
        return params

    def _graph_region(
        self, hidden: Tensor, params: PackedSeqParams | None, *, prepare: bool = False
    ) -> StateRegion:
        packed_profile = None
        if params is not None:
            if (
                params.qkv_format != "thd"
                or type(params.max_seqlen_q) is not int
                or type(params.max_seqlen_kv) is not int
                or type(params.cp_partition_mode) not in (str, type(None))
            ):
                raise ValueError("CSA2 graph profiles require THD host metadata")
            logical = params.cu_seqlens_q
            physical = params.cu_seqlens_q_padded
            physical = logical if physical is None else physical
            packed_profile = (
                params.qkv_format,
                params.max_seqlen_q,
                params.max_seqlen_kv,
                params.cp_partition_mode,
                tuple(logical.shape),
                logical.dtype,
                tuple(physical.shape),
                physical.dtype,
            )
        profile = (*hidden.shape[:2], packed_profile)
        region = self._graph_regions.get(profile)
        if region is None:
            if not prepare:
                raise ValueError(
                    "CSA2 graph replay requires a prepared static shape/layout profile"
                )
            region = self._build_graph_region(hidden, params)
            self._graph_regions[profile] = region
        return region

    def _build_graph_region(self, hidden: Tensor, params: PackedSeqParams | None) -> StateRegion:
        """Plan dependencies only while preparing samples or capturing a new profile."""
        schema = self._state_schema(*hidden.shape[:2], params)
        outputs = list(schema.outputs)
        if self.reindex and self.indexer_loss_enabled:
            outputs.append(
                TensorField(
                    f"csa2.decoder/indexer_loss:L{self.layer_idx}",
                    (),
                    torch.float32,
                    "thd" if params is not None else "sbhd",
                    True,
                )
            )
        if params is not None:
            prefix = params.cu_seqlens_q
            padded = params.cu_seqlens_q_padded
            padded = prefix if padded is None else padded
            capacity = (
                get_thd_compressed_capacity(
                    hidden.shape[0] * self.cp_size,
                    params.max_seqlen_q,
                    prefix.numel() - 1,
                    self.ratio,
                )
                if self.ratio
                else 0
            )
            for group, names, length in (
                ("layout", _LAYOUT_FIELDS, hidden.shape[0]),
                ("compressed", _COMPRESSED_FIELDS if self.ratio else (), capacity),
            ):
                for name in names:
                    if name in ("cu_seqlens", "cu_seqlens_padded"):
                        value = prefix if name == "cu_seqlens" else padded
                        shape, dtype = tuple(value.shape), value.dtype
                    else:
                        shape = (length, self.ratio) if name == "source_indices" else (length,)
                        dtype = torch.bool if name.startswith("valid_") else torch.int64
                    outputs.append(
                        TensorField(f"csa2.graph/{group}.{name}:batch", shape, dtype, "thd", False)
                    )
        return StateRegion(BoundarySchema(schema.boundary_id, schema.inputs, tuple(outputs)), self)

    def get_static_inputs(self, static_inputs: Mapping[str, Any]) -> dict[str, Tensor]:
        """Extend DSv4 sample inputs with this layer's actual shared dependencies."""
        owned = {}
        hidden = static_inputs["hidden_states"]
        params = self._packed_params(static_inputs)
        schema = self._graph_region(hidden, params, prepare=True).schema
        for field in schema.inputs:
            name = self._field_name(field)
            fill = -1 if name in ("global_indices", "candidate_indices") else 0
            owned[_PREFIX + name] = torch.full(
                field.shape,
                fill,
                dtype=field.dtype,
                device=hidden.device,
                requires_grad=field.differentiable,
            )
        return owned

    @staticmethod
    def _value(name: str, state: CSA2State) -> Tensor | None:
        if name.startswith("candidate_"):
            return None if state.candidates is None else getattr(state.candidates, name[10:])
        return getattr(state, name)

    def restore_inputs(
        self, hidden: Tensor, kwargs: Mapping[str, Any]
    ) -> tuple[StateRegion, CSA2State]:
        """Restore CSA2 input fields and bind only its native attention argument."""
        values = {name: kwargs[_PREFIX + name] for name in self.input_fields}
        params = self._packed_params(kwargs)
        region = self._graph_region(hidden, params, prepare=True)
        layout = (
            build_csa2_thd_layout(params, hidden.shape[0], cp_group=self.cp_group)
            if params is not None
            else None
        )
        compressed = (
            layout.for_compression(self.ratio) if layout is not None and self.ratio else None
        )
        template = CSA2State(
            kv_source_layer=self.kv_source,
            index_source_layer=self.index_source,
            candidate_source_layer=self.candidate if "candidate_indices" in values else None,
            sequence_length=hidden.shape[0],
            batch_size=hidden.shape[1],
            device=hidden.device,
            dtype=self.config.params_dtype,
            thd_layout=layout,
            compressed_layout=compressed,
            defer_indexer_loss=True,
        )
        metadata = dict(self.codec.metadata(template))
        metadata.update(
            candidate_block_size=self.config.csa2_candidate_block_size,
            packed_kv=use_fused_dsa_kernels(self.config),
        )
        schema = region.schema
        prefixes = self.codec.fields(template)
        prefix_tensors = self.codec.export(template, prefixes)
        state = self.codec.restore(
            (*schema.inputs, *prefixes),
            (*(values[self._field_name(field)] for field in schema.inputs), *prefix_tensors),
            tuple(metadata.items()),
        )
        return region, state

    def export(self, state: CSA2State, fields: tuple[TensorField, ...]) -> tuple[Tensor, ...]:
        """Use the declared field order for canonical values and graph-produced caches."""
        tensors = []
        for field in TensorSchema(fields).packed_fields:
            name = self._field_name(field)
            if name.startswith("layout."):
                tensor = getattr(state.thd_layout, name.split(".", 1)[1])
            elif name.startswith("compressed."):
                tensor = getattr(state.compressed_layout, name.split(".", 1)[1])
            else:
                tensor = self.codec.export(state, (field,))[0]
            tensors.append(tensor)
        TensorSchema(fields).validate(tensors)
        return tuple(tensors)

    def restore(
        self, fields: tuple[TensorField, ...], tensors: tuple[Tensor, ...], metadata: tuple
    ) -> dict[str, Tensor]:
        """Decode named publications; the owning component applies native cache semantics."""
        schema = TensorSchema(fields)
        schema.validate(tensors)
        return {
            self._field_name(field): tensor for field, tensor in zip(schema.packed_fields, tensors)
        }

    def prepare_replay(
        self, hidden: Tensor, kwargs: Mapping[str, Any]
    ) -> tuple[StateRegion, dict[str, Tensor], Callable]:
        """Declare inputs and the native state/cache publication for one replay."""
        bindings = kwargs.get("cross_layer_state")
        state = None if bindings is None else bindings.attention_kwargs().get("csa2_state")
        owned = {}
        params = self._packed_params(kwargs)
        if self.is_attention:
            if state is None or (
                state.last_layer is not None and state.last_layer >= self.layer_idx
            ):
                raise ValueError("CSA2 CUDA Graph replay requires fresh, ordered forward state")
            if state.thd_layout is not None and (
                state.thd_layout.cp_size,
                state.thd_layout.cp_rank,
            ) != (self.cp_size, self.cp_rank):
                raise ValueError("CSA2 CUDA Graph state belongs to a different CP shard")
            if self.ratio and not self.full and state.kv_source_layer != self.kv_source:
                raise ValueError("CSA2 CUDA Graph replay received the wrong shared KV owner")
            if self.ratio and not self.reindex and state.index_source_layer != self.index_source:
                raise ValueError("CSA2 CUDA Graph replay received the wrong index owner")
        for name in self.input_fields:
            value = self._value(name, state)
            if value is None:
                raise ValueError(f"CSA2 CUDA Graph replay requires {name}")
            if name == "indexer_k" and not self.indexer_loss_enabled:
                value = value.detach()
            owned[_PREFIX + name] = value
        if params is not None:
            if (
                params.qkv_format != "thd"
                or params.max_seqlen_q != self.config.max_seqlen_per_dp_cp_rank * self.cp_size
                or params.max_seqlen_kv != params.max_seqlen_q
            ):
                raise ValueError("CSA2 CUDA Graph THD requires the configured static max_seqlen")

        def restore(result, values):
            if "indexer_loss" in values:
                # TE jointly backpropagates every graph output, including zeros
                # for unused ones. Keep the unconditional auxiliary gradient
                # injector on the runtime hidden branch, as in eager training.
                result = (
                    DSAIndexerLossAutoScaler.apply(result[0], values["indexer_loss"]),
                    *result[1:],
                )
                logged = values["indexer_loss"].detach()
                if self.config.calculate_per_token_loss:
                    tokens = (
                        params.cu_seqlens_q[-1].clamp_min(1)
                        if params is not None
                        else hidden.shape[0] * hidden.shape[1]
                    )
                    logged = logged / tokens
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=logged,
                    layer_number=self.layer_idx + 1,
                    num_layers=self.config.num_layers,
                    reduce_group=self.cp_group if self.cp_size > 1 else None,
                )
            if self.is_attention:
                if self.full:
                    state.global_kv, state.indexer_k = values.get("global_kv"), values.get(
                        "indexer_k"
                    )
                    if state.indexer_k is not None and not self.indexer_loss_enabled:
                        state.indexer_k = state.indexer_k.detach()
                    state.global_kv_flat = state.indexer_k_flat = None
                    state.kv_source_layer = self.layer_idx
                    state.candidates = None
                    state.candidate_source_layer = None
                if self.reindex:
                    state.global_indices = values.get("global_indices")
                    state.index_source_layer = self.layer_idx
                    state.fused_indices = state.fused_topk_length = None
                    state.fused_q_padding_mask = None
                    state.fused_window_size = None
                if "candidate_indices" in values:
                    state.candidates = CSA2CandidateBlocks(
                        values["candidate_indices"],
                        values["candidate_lengths"],
                        self.config.csa2_candidate_block_size,
                    )
                    state.candidate_source_layer = self.layer_idx
                state.last_layer = self.layer_idx
                state.sequence_length, state.batch_size = hidden.shape[:2]
                state.device, state.dtype = hidden.device, self.config.params_dtype
                if params is not None:
                    layout_values = {name: values["layout." + name] for name in _LAYOUT_FIELDS}
                    state.thd_layout = CSA2THDLayout(
                        total_tokens=hidden.shape[0],
                        max_seqlen=params.max_seqlen_q,
                        cp_size=self.cp_size,
                        cp_rank=self.cp_rank,
                        **layout_values,
                    )
                    if self.ratio:
                        compressed_values = {
                            name: values["compressed." + name] for name in _COMPRESSED_FIELDS
                        }
                        state.compressed_layout = CSA2THDCompressionLayout(
                            ratio=self.ratio,
                            total_tokens=hidden.shape[0] * self.cp_size,
                            capacity=get_thd_compressed_capacity(
                                hidden.shape[0] * self.cp_size,
                                params.max_seqlen_q,
                                params.cu_seqlens_q.numel() - 1,
                                self.ratio,
                            ),
                            max_seqlen=params.max_seqlen_q // self.ratio,
                            **compressed_values,
                        )
                    state.fused_window_indices = None
            return result

        return self._graph_region(hidden, params), owned, restore

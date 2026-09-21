# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 state declarations for the common Hybrid execution boundaries."""

from typing import TYPE_CHECKING, Any

from torch import Tensor
from torch.nn import Module

from megatron.core.models.hybrid.hybrid_state import HybridStateDeclaration
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CSA2State,
    CSA2StateCodec,
    csa2_field_factory,
    csa2_state_dependencies,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_cuda_graph import (
    CSA2GraphState,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    CSA2PipelineBoundary,
    csa2_pipeline_boundary,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    build_csa2_thd_layout,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    use_fused_dsa_kernels,
)
from megatron.core.transformer.state_boundary import (
    BoundarySchema,
    StateBoundary,
    StatePlacement,
    StateRegion,
    prepare_state_region,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

if TYPE_CHECKING:
    from megatron.core.models.hybrid.hybrid_state import HybridRuntimeContext
    from megatron.core.transformer.module import CudaGraphCaptureRegion


class CSA2HybridAdapter(HybridStateDeclaration):
    """Declare native fields, sources and cache semantics without owning a backend."""

    context_attribute = "cross_layer_state"

    def __init__(
        self,
        config: MLATransformerConfig,
        *,
        layer_type_list: list[str],
        pp_layer_offset: int,
        is_mtp_layer: bool,
        pg_collection: ProcessGroupCollection | None = None,
        **kwargs,
    ) -> None:
        if (
            config.experimental_attention_variant != "dsv4_hybrid"
            or config.dsv4_version != "v4.1"
            or is_mtp_layer
        ):
            raise ValueError("CSA2 Hybrid adapters require a V4.1 backbone stack")
        self.config = config
        self.layer_pattern, self.layer_offset = "".join(layer_type_list), pp_layer_offset
        self.cp_group = None if pg_collection is None else pg_collection.cp
        self.codec = CSA2StateCodec(
            indexer_k_differentiable=(config.dsa_indexer_loss_coeff or 0) > 0
        )
        self.pattern, self.placement = None, ()

    def bind_placement(self, pattern: str, placement: tuple[StatePlacement, ...]) -> None:
        """Use the host's existing order to identify the last attention at a cut."""
        self.pattern, self.placement = pattern, placement

    def validate_runtime(self, runtime: "HybridRuntimeContext") -> None:
        """Keep native CSA2 restrictions separate from Hybrid/backend capabilities."""
        if runtime.tensor_parallel_size != 1 or runtime.sequence_parallel:
            raise ValueError("CSA2 state requires TP=1 without sequence parallelism")
        if runtime.context_parallel_size > 1 and self.cp_group is None:
            raise ValueError("CSA2 state with context parallelism requires an explicit CP group")
        if self.cp_group is not None and self.cp_group.size() != runtime.context_parallel_size:
            raise ValueError("CSA2 state CP group size must match the runtime configuration")

    def layer_kwargs(self, layer: Module, state: CSA2State) -> dict[str, dict[str, Any]]:
        """Bind only branches that implement the cross-layer attention contract."""
        if isinstance(layer, TransformerLayer):
            return {"attention": state.attention_kwargs()}
        if getattr(layer, "supports_cross_layer_state", False):
            return {"layer": {"cross_layer_state": state}}
        return {}

    def recompute_boundary_tensors(self, state: CSA2State) -> tuple[Tensor, ...]:
        """Protect native shared activations during selective recompute."""
        return state.recompute_boundary_tensors()

    def graph_state(
        self, layer: Module, symbol: str, region: "CudaGraphCaptureRegion"
    ) -> CSA2GraphState | None:
        """Supply native graph fields and auxiliary-loss/cache publication."""
        if "attention" not in region.branches or symbol not in ("D", "W"):
            return None
        return CSA2GraphState(
            self.config,
            layer_number=layer.layer_number,
            is_attention=symbol in ("D", "W"),
            cp_group=self.cp_group,
            placement=self.placement,
        )

    def initial_state(self, hidden: Tensor, packed_seq_params: PackedSeqParams | None) -> CSA2State:
        """Build forward-local native state and, when packed, its layout snapshot."""
        state = CSA2State()
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            state.thd_layout = build_csa2_thd_layout(
                packed_seq_params, hidden.shape[0], cp_group=self.cp_group
            )
        return state

    def pipeline_region(
        self,
        boundary: Any,
        hidden: Tensor,
        packed_seq_params: PackedSeqParams | None,
        *,
        requires_grad: bool = True,
    ) -> StateRegion:
        """Describe canonical native tensors and immutable metadata at a host cut."""
        if not isinstance(boundary, CSA2PipelineBoundary):
            boundary = csa2_pipeline_boundary(
                self.config,
                self.pattern,
                boundary.layer_offset,
                boundary.qkv_format,
                self.placement,
            )
        fields = boundary.tensor_fields(
            self.config,
            *hidden.shape[:2],
            packed_seq_params,
            requires_grad=requires_grad,
            cp_group=self.cp_group,
        )
        cp_size, cp_rank = (
            (1, 0) if self.cp_group is None else (self.cp_group.size(), self.cp_group.rank())
        )
        template = CSA2State(
            kv_source_layer=boundary.kv_source_layer,
            index_source_layer=boundary.index_source_layer,
            candidate_source_layer=boundary.candidate_source_layer,
            sequence_length=hidden.shape[0],
            batch_size=hidden.shape[1],
            device=hidden.device,
            dtype=self.config.params_dtype,
            last_layer=boundary.last_attention_layer,
        )
        metadata = dict(self.codec.metadata(template))
        metadata.update(
            candidate_block_size=boundary.candidate_block_size,
            layout=(
                None
                if boundary.qkv_format == "sbhd"
                else (hidden.shape[0], packed_seq_params.max_seqlen_q, cp_rank, cp_size)
            ),
            compress_ratio=boundary.compress_ratio if boundary.qkv_format == "thd" else None,
            packed_kv=use_fused_dsa_kernels(self.config),
        )
        metadata = tuple(metadata.items())
        return StateRegion(
            BoundarySchema(f"csa2.decoder/pp:{boundary.layer_offset}", fields, fields),
            self.codec,
            metadata,
            metadata,
        )

    def checkpoint_region(
        self, start: int, end: int, hidden: Tensor, state: CSA2State
    ) -> StateRegion:
        """Describe a host-selected full-recompute group without changing its layer order."""
        start, end = start + self.layer_offset, end + self.layer_offset
        config, layout = self.config, state.thd_layout
        if not self.placement:
            raise ValueError("Bind CSA2 placement before querying checkpoint regions")
        placement = self.placement
        factory = csa2_field_factory(
            config,
            hidden.shape[0],
            hidden.shape[1],
            qkv_format="thd" if layout is not None else "sbhd",
            max_seqlen=None if layout is None else layout.max_seqlen,
            num_sequences=None if layout is None else layout.cu_seqlens.numel() - 1,
            cp_size=1 if layout is None else layout.cp_size,
        )
        boundary = StateBoundary(
            f"csa2.decoder/checkpoint:{start}:{end}", "checkpoint", 2 * start, 2 * end
        )
        region = prepare_state_region(
            csa2_state_dependencies(config, factory, placement), boundary, placement, self.codec
        )
        schema = region.schema
        codec = self.codec
        metadata = dict(codec.metadata(state))
        metadata.update(
            sequence_length=hidden.shape[0],
            batch_size=hidden.shape[1],
            device=str(hidden.device),
            dtype=str(config.params_dtype).removeprefix("torch."),
            packed_kv=use_fused_dsa_kernels(config),
        )
        # Prefix tensors are ordinary explicit layout inputs. The codec rebuilds
        # physical indices instead of closing over cached tensors from another call.
        prefixes = tuple(f for f in codec.fields(state) if "/cu_seqlens" in f.key)
        inputs = (*schema.inputs, *prefixes)
        retained = (*region.retained_fields, *prefixes)
        input_metadata = tuple(metadata.items())
        for layer in range(start, end):
            symbol = self.layer_pattern[layer - self.layer_offset]
            if symbol in ("D", "W"):
                metadata["last_layer"] = layer
            if layer in config.csa2_kv_source_layers:
                metadata["kv_source_layer"] = layer
                metadata["candidate_source_layer"] = None
                metadata["candidate_block_size"] = None
                metadata["compress_ratio"] = (
                    config.csa_compress_ratios[layer] if layout is not None else None
                )
            if layer in config.csa2_index_source_layers:
                metadata["index_source_layer"] = layer
            if layer == config.csa2_candidate_source_layer:
                metadata["candidate_source_layer"] = layer
                metadata["candidate_block_size"] = config.csa2_candidate_block_size
        return StateRegion(
            BoundarySchema(schema.boundary_id, inputs, schema.outputs),
            codec,
            input_metadata,
            tuple(metadata.items()),
            retained,
        )

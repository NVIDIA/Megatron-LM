# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSv4.1 state lifecycle and logical pipeline boundaries for HybridStack."""

from typing import Any

from torch import Tensor
from torch.distributed import ProcessGroup

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelinePayloadFactory,
    PipelinePayloadSpec,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2State
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    CSA2PipelineChunk,
    CSA2PipelinePayload,
    build_csa2_pipeline_plan,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    use_fused_dsa_kernels,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig


class CSA2HybridAdapter:
    """Own CSA2 semantics while leaving Hybrid's layer loop and mHC execution generic.

    The adapter stores only construction metadata and an optional static PP-1
    plan. Each forward creates or restores its own CSA2/mHC state; received THD
    metadata and derived attention caches belong to that forward's context.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        *,
        layer_type_list: list[str],
        pp_layer_offset: int,
        pre_process: bool,
        post_process: bool,
        is_mtp_layer: bool,
    ) -> None:
        if (
            config.experimental_attention_variant != "dsv4_hybrid"
            or config.dsv4_version != "v4.1"
            or is_mtp_layer
        ):
            raise ValueError("CSA2 Hybrid adapters require a V4.1 backbone stack")
        self.config = config
        self.layer_pattern = "".join(layer_type_list)
        self.layer_offset = pp_layer_offset
        self.pre_process = pre_process
        self.post_process = post_process
        self._pipeline_chunk: CSA2PipelineChunk | None = None
        self._pipeline_chunks: dict[str, CSA2PipelineChunk] = {}

    def configure_pipeline(self, chunk: CSA2PipelineChunk) -> None:
        """Bind a static logical chunk for local split execution."""
        if (
            chunk.layer_offset != self.layer_offset
            or chunk.layer_pattern != self.layer_pattern
            or self.pre_process != (chunk.incoming is None)
            or self.post_process != (chunk.outgoing is None)
        ):
            raise ValueError("CSA2 pipeline chunk does not match this HybridStack segment")
        if self._pipeline_chunk is not None and self._pipeline_chunk != chunk:
            raise ValueError("CSA2 pipeline chunk is already configured with a different plan")
        self._pipeline_chunk = chunk
        boundary = chunk.outgoing if chunk.outgoing is not None else chunk.incoming
        if boundary is not None:
            self._pipeline_chunks[boundary.qkv_format] = chunk

    def configure_distributed_pipeline(
        self, pattern: str, pp_group: ProcessGroup, vp_stage: int | None = None
    ) -> PipelinePayloadFactory | None:
        """Bind this physical/virtual chunk's packed and unpacked boundaries."""
        if pp_group.size() == 1:
            return
        vp_size = self.config.virtual_pipeline_model_parallel_size or 1
        if (vp_size > 1 and vp_stage is None) or not 0 <= (vp_stage or 0) < vp_size:
            raise ValueError("CSA2 pipeline requires a valid vp_stage for each virtual chunk")
        if len(pattern.split("|")) != pp_group.size() * vp_size:
            raise ValueError("CSA2 pipeline segment count must equal PP size times VPP size")
        chunk_index = (vp_stage or 0) * pp_group.size() + pp_group.rank()
        chunks = tuple(
            build_csa2_pipeline_plan(
                self.config, pattern, pp_size=pp_group.size(), qkv_format=layout
            )[chunk_index]
            for layout in ("sbhd", "thd")
        )
        self.configure_pipeline(chunks[0])
        self._pipeline_chunks = dict(zip(("sbhd", "thd"), chunks))
        return self.make_pipeline_payload

    def make_pipeline_payload(
        self, tensors: tuple[Tensor, ...], metadata: tuple[int, int]
    ) -> CSA2PipelinePayload:
        """Reconstruct a received input using this adapter's existing boundary plan."""
        offset, max_seqlen = metadata
        chunk = self._pipeline_chunks.get("sbhd" if max_seqlen == -1 else "thd")
        if chunk is None or chunk.incoming is None or chunk.incoming.layer_offset != offset:
            raise ValueError("CSA2 message does not match the incoming pipeline boundary")
        payload = CSA2PipelinePayload(
            chunk.incoming, tensors, None if max_seqlen == -1 else max_seqlen
        )
        payload.validate()
        return payload

    def pipeline_payload_spec(
        self,
        seq_length: int,
        micro_batch_size: int,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        requires_grad: bool = True,
    ) -> tuple[PipelinePayloadSpec | None, PipelinePayloadSpec | None]:
        """Describe both boundaries for one prepared batch without reading device values."""
        layout = (
            "thd"
            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            else "sbhd"
        )
        chunk = self._pipeline_chunks[layout]
        return tuple(
            (
                boundary.payload_spec(
                    self.config,
                    seq_length,
                    micro_batch_size,
                    packed_seq_params,
                    requires_grad=requires_grad,
                )
                if boundary is not None
                else None
            )
            for boundary in (chunk.incoming, chunk.outgoing)
        )

    @property
    def consumes_input_tensor(self) -> bool:
        """A receiving chunk consumes exactly one submitted payload per forward."""
        return self._pipeline_chunk is not None and self._pipeline_chunk.incoming is not None

    def validate_input(self, input_tensor: Tensor | CSA2PipelinePayload | None) -> None:
        """Check a payload against this chunk without retaining any activation state."""
        chunk = self._pipeline_chunk
        if isinstance(input_tensor, CSA2PipelinePayload):
            if chunk is None:
                raise ValueError(
                    "CSA2 pipeline payload requires adapter.configure_pipeline() first"
                )
            if chunk.incoming is None:
                raise ValueError("The first CSA2 pipeline chunk cannot receive a payload")
            chunk = self._pipeline_chunks.get(input_tensor.boundary.qkv_format)
            if chunk is None or input_tensor.boundary != chunk.incoming:
                raise ValueError("CSA2 pipeline payload does not match the incoming boundary")
        elif chunk is not None and input_tensor is not None:
            raise ValueError("Configured CSA2 chunks receive a payload through set_input_tensor()")

    def prepare_forward(
        self,
        hidden_states: Any,
        packed_seq_params: PackedSeqParams | None,
        inference_context: BaseInferenceContext | None,
    ) -> tuple[Any, PackedSeqParams | None, HybridStackForwardContext]:
        """Create fresh state or restore the incoming snapshot, including THD caches."""
        chunk = self._pipeline_chunk
        if chunk is not None and inference_context is not None:
            raise ValueError("CSA2 pipeline chunks require a backbone training forward")
        mhc_state = None
        if self.consumes_input_tensor:
            if hidden_states is None:
                raise ValueError("CSA2 receiving chunk requires a payload from set_input_tensor()")
            self.validate_input(hidden_states)
            hidden_states, csa2_state, mhc_state, restored_params = hidden_states.restore(
                use_fused_kernels=use_fused_dsa_kernels(self.config)
            )
            if packed_seq_params is not None:
                if csa2_state.thd_layout is not None:
                    csa2_state.thd_layout.validate_compatible(
                        packed_seq_params, hidden_states.shape[0]
                    )
                elif packed_seq_params.qkv_format == "thd":
                    raise ValueError("CSA2 pipeline payload expects SBHD, not THD")
            packed_seq_params = restored_params
        else:
            csa2_state = CSA2State()
        return (
            hidden_states,
            packed_seq_params,
            HybridStackForwardContext(layer_kwargs={"csa2_state": csa2_state}, mhc_state=mhc_state),
        )

    def finalize_forward(
        self,
        output: Tensor | tuple[Tensor, Tensor],
        packed_seq_params: PackedSeqParams | None,
        context: HybridStackForwardContext,
    ) -> Tensor | tuple[Tensor, Tensor] | CSA2PipelinePayload:
        """Snapshot only live outgoing dependencies while preserving their autograd edges."""
        layout = (
            "thd"
            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            else "sbhd"
        )
        chunk = self._pipeline_chunks.get(layout, self._pipeline_chunk)
        if chunk is not None and chunk.outgoing is not None:
            if not isinstance(output, Tensor):
                raise TypeError("CSA2 pipeline chunks require a tensor before payload export")
            return chunk.outgoing.export_payload(
                output,
                context.layer_kwargs["csa2_state"],
                context.mhc_state,
                packed_seq_params=packed_seq_params,
            )
        return output

    def recompute_boundary_tensors(self, context: HybridStackForwardContext) -> tuple[Tensor, ...]:
        """Guard shared KV backward even when it arrives before the hidden-state gradient."""
        state = context.layer_kwargs["csa2_state"]
        return tuple(tensor for tensor in (state.global_kv, state.indexer_k) if tensor is not None)

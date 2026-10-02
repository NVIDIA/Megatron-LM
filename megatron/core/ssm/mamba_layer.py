# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
except ImportError:
    from megatron.core.telemetry.fallbacks import managed_span as _otel_managed_span

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.context_parallel.chunkwise import PackedSequenceCPMetadata
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule, TwoStageAttentionLayer
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params

_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_Q = "_mamba_packed_seq_cg_cu_seqlens_q"
_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_KV = "_mamba_packed_seq_cg_cu_seqlens_kv"
_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_Q_PADDED = "_mamba_packed_seq_cg_cu_seqlens_q_padded"
_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_KV_PADDED = "_mamba_packed_seq_cg_cu_seqlens_kv_padded"
_MAMBA_PACKED_SEQ_CG_SEQ_IDX = "_mamba_packed_seq_cg_seq_idx"
_MAMBA_PACKED_SEQ_CG_INPUT_NAMES = (
    _MAMBA_PACKED_SEQ_CG_CU_SEQLENS_Q,
    _MAMBA_PACKED_SEQ_CG_CU_SEQLENS_KV,
    _MAMBA_PACKED_SEQ_CG_CU_SEQLENS_Q_PADDED,
    _MAMBA_PACKED_SEQ_CG_CU_SEQLENS_KV_PADDED,
    _MAMBA_PACKED_SEQ_CG_SEQ_IDX,
)


@dataclass
class MambaLayerSubmodules:
    """
    Configuration class for specifying the submodules of a Mamba layer.

    This class defines the structure and default implementations for various
    components of a Mamba layer, allowing for flexible customization of the
    layer's architecture.

    Args:
        norm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        mixer (Union[ModuleSpec, type]): Specification for the along-sequence mixing mechanism.
        mamba_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the mixer.
    """

    norm: LayerNormBuilder = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MambaLayer(GraphableMegatronModule, TwoStageAttentionLayer):
    """
    A single Mamba layer.

    Mamba layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    #: Whether this layer class owns a wide-residual connection around its mixer.
    supports_wide_residual_connections: bool = False

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaLayerSubmodules,
        layer_number: int = 1,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
        name: str | None = None,
        is_mtp_layer: bool = False,
    ):
        """Initialize Mamba Layer.

        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config)
        if (
            config.wide_residual is not None
            and not self.supports_wide_residual_connections
            and not is_mtp_layer
        ):
            raise ValueError(
                f"{type(self).__name__} does not implement wide-residual streams. Build the "
                "hybrid stack with WideResidualMambaLayer when wide_residual is configured."
            )
        assert pg_collection is not None, "pg_collection must be provided for MambaLayer"
        self.tp_group = pg_collection.tp

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        self.hidden_dropout = config.hidden_dropout
        self.mixer = build_module(
            submodules.mixer,
            self.config,
            d_model=self.config.hidden_size,
            layer_number=layer_number,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=(name + f".mixer") if name is not None else None,
        )
        self.norm = submodules.norm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.mamba_bda = build_module(submodules.mamba_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def create_mcore_cudagraph_manager(self, config):
        """Register the mamba layer for cudagraphs."""
        assert self.config.cuda_graph_impl == "local"

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        if (
            not self.config.cuda_graph_modules
            and self.config.inference_cuda_graph_scope != InferenceCudaGraphScope.block
        ) or CudaGraphModule.mamba in self.config.cuda_graph_modules:
            self.cudagraph_manager = CudaGraphManager(config)

    def mamba_state_shapes_per_request(self) -> Tuple[Tuple[int], Tuple[int]]:
        """Returns the Mamba conv and ssm states shapes per request."""
        return self.mixer.mamba_state_shapes_per_request()

    def supports_two_stage_attention(self) -> bool:
        """Return whether the configured sequence mixer supports two-stage execution."""
        return (
            isinstance(self.mixer, TwoStageAttentionLayer)
            and self.mixer.supports_two_stage_attention()
        )

    def _prepare_mixer_input(self, hidden_states: Tensor) -> Tensor:
        """Convert a branch input to parameter precision and normalize it."""
        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        return apply_module(self.norm)(hidden_states)

    def _prepare_residual(self, hidden_states: Tensor) -> Tensor:
        """Preserve an ordinary residual stream in its configured dtype."""

        return hidden_states.float() if self.config.fp32_residual_connection else hidden_states

    def _apply_mixer_bda(self, mixer_out_with_bias, residual: Tensor) -> Tensor:
        """Apply the layer's bias-dropout-add tail to a projected mixer output."""

        with self.bias_dropout_add_exec_handler():
            return self.mamba_bda(training=self.training, fused=self.config.bias_dropout_fusion)(
                mixer_out_with_bias, residual, self.hidden_dropout
            )

    def _prepare_mixer_state(
        self,
        hidden_states: Tensor,
        residual_stream_recompute_context: ResidualStreamRecomputeContext | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, ...] | None, ResidualStreamRecomputeContext | None]:
        """Prepare branch input and residual state for the shared mixer schedule."""

        residual = self._prepare_residual(hidden_states)
        hidden_states = self._prepare_mixer_input(hidden_states)
        return hidden_states, residual, None, None

    def _apply_mixer_update(
        self,
        mixer_out_with_bias,
        residual: Tensor,
        connection_state: tuple[Tensor, ...] | None = None,
        recompute_context: ResidualStreamRecomputeContext | None = None,
    ) -> Tensor:
        """Complete the ordinary mixer update using its saved residual."""

        return self._apply_mixer_bda(mixer_out_with_bias, residual)

    def forward_pre_attn_and_core_attn(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,  # Not used in MambaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
        sequence_len_offset: Optional[int] = None,  # Not used in MambaLayer
        padding_mask: Optional[Tensor] = None,  # Not used in MambaLayer
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        packed_sequence_cp_metadata: PackedSequenceCPMetadata | None = None,
        residual_stream_recompute_context: ResidualStreamRecomputeContext | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        """Run normalization, input projection, and the selective SSM/SSD.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention. Not used by this layer.
            inference_context (BaseInferenceContext, optional): Must be ``None`` because two-stage
                mixer execution is training-only.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            Core SSM result and residual, followed by connection state for a connected layer.
            Pass the entire tuple to the layer's ``forward_post_core_attn`` method.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)
        assert inference_context is None, "Two-stage mixer execution does not support inference."

        hidden_states, residual, connection_state, _ = self._prepare_mixer_state(
            hidden_states, residual_stream_recompute_context=residual_stream_recompute_context
        )

        ssm_output = self.mixer.forward_pre_attn_and_core_attn(
            hidden_states,
            packed_seq_params=packed_seq_params,
            packed_sequence_cp_metadata=packed_sequence_cp_metadata,
        )
        if connection_state is None:
            return ssm_output, residual
        return ssm_output, residual, connection_state

    def forward_post_core_attn(
        self,
        ssm_output: Tensor,
        residual: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply Mamba's output projection and ordinary bias-dropout-add update."""

        del inference_context, padding_mask
        mixer_out_with_bias = self.mixer.forward_post_core_attn(ssm_output)
        return self._apply_mixer_bda(mixer_out_with_bias, residual)

    def _get_residual_connection(self):
        """Return an optional architecture-owned connection around the Mamba mixer."""

        return None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,  # Not used in MambaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
        residual_stream_recompute_context: ResidualStreamRecomputeContext | None = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        packed_sequence_cp_metadata: PackedSequenceCPMetadata | None = None,
    ):
        """
        Perform a forward pass through the Mamba layer.

        This method implements the core computation of a Mamba layer, including
        the convolution and the selective SSM/SSD.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention. Not used by this layer.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            packed_sequence_cp_metadata (PackedSequenceCPMetadata, optional): Rank-local
                packed-sequence metadata for chunkwise CP.
            residual_stream_recompute_context (ResidualStreamRecomputeContext, optional):
                Call-local ordered replay state for a configured residual connection.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Whole-layer + mixer lens spans, mirroring transformer_layer.py so the hybrid
        # model's Mamba layers aren't a blind spot in the per-layer breakdown (they were
        # ~34s of uninstrumented first-iteration warmup). No-op unless the 'layer' span
        # group is enabled, so zero cost on normal runs.
        with _otel_managed_span(
            'layer', 'megatron.layer.forward', **{'megatron.layer_number': self.layer_number}
        ):
            hidden_states, residual, connection_state, recompute_context = (
                self._prepare_mixer_state(
                    hidden_states,
                    residual_stream_recompute_context=residual_stream_recompute_context,
                )
            )

            # Mamba mixer: conv + selective SSM/SSD -- the compute block, analog of the
            # transformer layer's self_attention/mlp (this is where the SSD kernel autotune
            # lands on the first pass).
            with _otel_managed_span('layer', 'megatron.layer.mamba'):
                if packed_sequence_cp_metadata is None:
                    mixer_out_with_bias = self.mixer(
                        hidden_states,
                        inference_context=inference_context,
                        packed_seq_params=packed_seq_params,
                    )
                else:
                    mixer_out_with_bias = self.mixer(
                        hidden_states,
                        inference_context=inference_context,
                        packed_seq_params=packed_seq_params,
                        packed_sequence_cp_metadata=packed_sequence_cp_metadata,
                    )

            hidden_states = self._apply_mixer_update(
                mixer_out_with_bias, residual, connection_state, recompute_context=recompute_context
            )

            return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the mamba layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the mamba layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Prepare static inputs for CUDA graph capture.

        When packed sequences are in use (SFT), also sets up shared CUDA graph
        buffer tensors and a dummy PackedSeqParams so the graph captures the
        packed-sequence code path (Mamba with seq_idx).
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)

        if getattr(self.config, 'cuda_graph_max_packed_seqs', None) is not None:
            self._cuda_graph_seq_length = seq_length
            max_seqs = self.config.cuda_graph_max_packed_seqs
            # Compute total_tokens as seen by Mamba SSM after CP all_to_all.
            mamba_cp_size = self.mixer.cp.cp_size
            total_tokens = (seq_length // self.config.context_parallel_size) * mamba_cp_size
            device = static_inputs["hidden_states"].device
            self._cuda_graph_mamba_total_tokens = total_tokens
            self._use_pp_packed_mamba_cg_inputs = (
                self.config.cuda_graph_impl == "transformer_engine"
                and self.config.pipeline_model_parallel_size > 1
            )

            if self._use_pp_packed_mamba_cg_inputs:
                # TE owns these ordinary Tensor kwargs per graph slot. This is
                # required for 1F1B schedules where a later forward must not
                # overwrite sequence boundaries used by an earlier backward.
                _, buffers = PackedSeqParams.create_dummy_for_cuda_graph(
                    seq_length, max_seqs=max_seqs, device=device
                )
                seq_idx_buf = torch.zeros(1, total_tokens, dtype=torch.int32, device=device)
                self._cuda_graph_packed_seq_target_len = buffers['cu_seqlens_q'].shape[0]
            else:
                # With PP=1, forward/backward microbatch lifetimes do not
                # overlap, so layers can share one set of staging tensors.
                buffers = PackedSeqParams.get_or_create_shared_cg_buffers(
                    seq_length, max_seqs, device, tag='mamba'
                )
                seq_idx_buf = PackedSeqParams.get_or_create_shared_seq_idx_buffer(
                    total_tokens, device
                )
                buffers['seq_idx'] = seq_idx_buf
                self._cuda_graph_psp_buffers = buffers

            self._cuda_graph_mamba_seq_idx_spec = (
                seq_idx_buf.shape,
                seq_idx_buf.dtype,
                seq_idx_buf.device,
            )

            # Correct cu_seqlens for Mamba's CP all_to_all sequence gathering.
            # pre_conv_ssm gathers: [seq_length/cp, b, d] -> [seq_length, b, d/cp]
            if mamba_cp_size > 1:
                for k in (
                    'cu_seqlens_q',
                    'cu_seqlens_kv',
                    'cu_seqlens_q_padded',
                    'cu_seqlens_kv_padded',
                ):
                    buffers[k][1:] = total_tokens

            if self._use_pp_packed_mamba_cg_inputs:
                static_inputs[_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_Q] = buffers['cu_seqlens_q']
                static_inputs[_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_KV] = buffers['cu_seqlens_kv']
                static_inputs[_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_Q_PADDED] = buffers[
                    'cu_seqlens_q_padded'
                ]
                static_inputs[_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_KV_PADDED] = buffers[
                    'cu_seqlens_kv_padded'
                ]
                static_inputs[_MAMBA_PACKED_SEQ_CG_SEQ_IDX] = seq_idx_buf
            else:
                self._cuda_graph_psp = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=buffers['cu_seqlens_q'],
                    cu_seqlens_kv=buffers['cu_seqlens_kv'],
                    cu_seqlens_q_padded=buffers['cu_seqlens_q_padded'],
                    cu_seqlens_kv_padded=buffers['cu_seqlens_kv_padded'],
                    max_seqlen_q=total_tokens,
                    max_seqlen_kv=total_tokens,
                    total_tokens=total_tokens,
                    seq_idx=seq_idx_buf,
                )

        return static_inputs

    @staticmethod
    def _decompose_packed_seq_params_to_cg_kwargs(kwargs, target_len):
        """Replace Mamba's PackedSeqParams with graph-slot-owned Tensor inputs."""
        packed_seq_params = kwargs.pop('packed_seq_params', None)
        if packed_seq_params is None:
            return

        packed_seq_params.ensure_cg_padded(target_len)
        if packed_seq_params.seq_idx is None:
            raise ValueError("Packed Mamba CUDA graph replay requires a precomputed seq_idx.")
        kwargs[_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_Q] = packed_seq_params._cg_padded_q
        kwargs[_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_KV] = packed_seq_params._cg_padded_kv
        kwargs[_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_Q_PADDED] = packed_seq_params._cg_padded_qp
        kwargs[_MAMBA_PACKED_SEQ_CG_CU_SEQLENS_KV_PADDED] = packed_seq_params._cg_padded_kvp
        kwargs[_MAMBA_PACKED_SEQ_CG_SEQ_IDX] = packed_seq_params.seq_idx

    def _reconstruct_packed_seq_params_from_cg_kwargs(self, kwargs):
        """Rebuild Mamba's PackedSeqParams from graph-slot-owned Tensor inputs."""
        graph_inputs = [kwargs.pop(name, None) for name in _MAMBA_PACKED_SEQ_CG_INPUT_NAMES]
        if all(value is None for value in graph_inputs):
            return
        if not all(value is not None for value in graph_inputs):
            raise ValueError("Packed Mamba CUDA graphs require all sequence metadata inputs.")

        total_tokens = self._cuda_graph_mamba_total_tokens
        kwargs['packed_seq_params'] = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=graph_inputs[0],
            cu_seqlens_kv=graph_inputs[1],
            cu_seqlens_q_padded=graph_inputs[2],
            cu_seqlens_kv_padded=graph_inputs[3],
            max_seqlen_q=total_tokens,
            max_seqlen_kv=total_tokens,
            total_tokens=total_tokens,
            seq_idx=graph_inputs[4],
        )

    def _seq_idx_matches_cuda_graph(self, seq_idx):
        """Return whether ``seq_idx`` matches the tensor signature captured by the graph."""
        if seq_idx is None:
            return False
        shape, dtype, device = self._cuda_graph_mamba_seq_idx_spec
        return seq_idx.shape == shape and seq_idx.dtype == dtype and seq_idx.device == device

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Inject dummy PSP for CUDA graph capture so Mamba captures the packed-seq code path."""
        if any(name in kwargs for name in _MAMBA_PACKED_SEQ_CG_INPUT_NAMES):
            kwargs = dict(kwargs)
            self._reconstruct_packed_seq_params_from_cg_kwargs(kwargs)
        elif hasattr(self, '_cuda_graph_psp') and kwargs.get('packed_seq_params') is None:
            kwargs = dict(kwargs)
            kwargs['packed_seq_params'] = self._cuda_graph_psp
        return self.forward(*args, **kwargs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for Mamba layer using TE interface.

        Uses graph-slot-owned metadata for PP schedules, or copies metadata into
        shared staging buffers for PP=1. Falls back to eager execution when the
        runtime metadata does not fit the captured graph signature.
        """
        assert kwargs.get('inference_context') is None, (
            "CUDA graph accepts only Tensor inputs. inference_context is excluded from input list. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )
        psp = kwargs.get('packed_seq_params')
        if psp is not None and hasattr(self, '_cuda_graph_mamba_seq_idx_spec'):
            if psp.seq_idx is None:
                raise ValueError("Packed Mamba CUDA graph replay requires a precomputed seq_idx.")
            if not self._seq_idx_matches_cuda_graph(psp.seq_idx):
                # Graph inputs have a fixed shape, dtype, and device. This also
                # prevents PP=1 copy_ from broadcasting a one-token seq_idx
                # across the entire captured buffer.
                return self.forward(*args, **kwargs)
        if psp is not None and getattr(self, '_use_pp_packed_mamba_cg_inputs', False):
            if psp.cu_seqlens_q.shape[0] > self._cuda_graph_packed_seq_target_len:
                return self.forward(*args, **kwargs)
            kwargs = dict(kwargs)
            self._decompose_packed_seq_params_to_cg_kwargs(
                kwargs, self._cuda_graph_packed_seq_target_len
            )
        elif psp is not None and hasattr(self, '_cuda_graph_psp_buffers'):
            bucket_max = self._cuda_graph_psp_buffers['cu_seqlens_q'].shape[0]  # max_seqs + 1
            if psp.cu_seqlens_q.shape[0] > bucket_max:
                # Actual N_docs exceeds bucket -> fall back to non-CG forward.
                return self.forward(*args, **kwargs)

            bufs = self._cuda_graph_psp_buffers
            target_len = bufs['cu_seqlens_q'].shape[0]

            # PSP-identity gate: shared buffers need only be updated ONCE per
            # micro-batch. Use 'is' to avoid false-positive cache hits from
            # CPython id() recycling.
            if bufs.get('_last_updated_psp') is not psp:
                psp.ensure_cg_padded(target_len)
                bufs['cu_seqlens_q'].copy_(psp._cg_padded_q)
                bufs['cu_seqlens_kv'].copy_(psp._cg_padded_kv)
                bufs['cu_seqlens_q_padded'].copy_(
                    psp._cg_padded_qp if psp._cg_padded_qp is not None else psp._cg_padded_q
                )
                bufs['cu_seqlens_kv_padded'].copy_(
                    psp._cg_padded_kvp if psp._cg_padded_kvp is not None else psp._cg_padded_kv
                )
                # Copy seq_idx into shared buffer (computed by __post_init__).
                if 'seq_idx' in bufs and psp.seq_idx is not None:
                    bufs['seq_idx'].copy_(psp.seq_idx)
                bufs['_last_updated_psp'] = psp

            # The dummy PSP was injected inside capture and is not part of TE's callable
            # signature. Its tensor fields already alias the staging buffers updated above.
            kwargs = dict(kwargs)
            kwargs.pop('packed_seq_params')

        return super()._te_cuda_graph_replay(*args, **kwargs)

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        # Training and validation mode CUDA graphs.
        if (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('inference_context') is None
            and not torch.is_inference_mode_enabled()  # for inference eager dummy_forward
        ):
            return True
        elif InferenceMode.is_active() and (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('attention_mask') is None
            and kwargs.get('inference_context') is not None
            and not self.config.cuda_graph_modules  # empty-list = per-layer CUDA graphs
        ):
            context = kwargs['inference_context']
            using_cuda_graph = (context.is_static_batching() and context.is_decode_only()) or (
                not context.is_static_batching() and context.using_cuda_graph_this_step()
            )
            return using_cuda_graph
        return False

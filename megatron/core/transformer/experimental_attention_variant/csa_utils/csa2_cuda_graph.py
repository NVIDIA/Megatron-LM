# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit tensor boundaries for CSA2 Hybrid Transformer Engine CUDA Graphs.

Only tensors consumed or produced by the current layer cross its graph boundary.
Each capture invocation reconstructs fresh Python state. Replay publishes graph
outputs to the caller's forward-local state before the ordinary layer continuation.
"""

from typing import Callable

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2State
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    CSA2THDCompressionLayout,
    CSA2THDLayout,
    build_csa2_thd_layout,
    get_thd_compressed_capacity,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    use_fused_dsa_kernels,
)
from megatron.core.transformer.hyper_connection import SinglePassMHCState
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


class CSA2CudaGraphAdapter:
    """Static per-layer graph schema; never owns a microbatch's activation state."""

    def __init__(
        self,
        config: MLATransformerConfig,
        *,
        layer_number: int,
        is_attention: bool,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        self.config = config
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
        self.single_pass = config.enable_hyper_connections and config.mhc_single_pass
        self.graph_mhc = self.single_pass and not (
            config.recompute_granularity == "selective"
            and "mhc" in (config.recompute_modules or [])
        )
        self.indexer_loss_enabled = (config.dsa_indexer_loss_coeff or 0.0) > 0
        self.ratio = config.csa_compress_ratios[self.layer_idx] if is_attention else 0
        self.full = bool(self.ratio and self.layer_idx in config.csa2_kv_source_layers)
        self.reindex = bool(self.ratio and self.layer_idx in config.csa2_index_source_layers)
        self.candidate = config.csa2_candidate_source_layer
        self.kv_source = next(
            (n for n in reversed(config.csa2_kv_source_layers) if n <= self.layer_idx), None
        )
        self.index_source = next(
            (n for n in reversed(config.csa2_index_source_layers) if n <= self.layer_idx), None
        )
        inputs, outputs = [], []
        if self.graph_mhc:
            if self.layer_idx:
                inputs.append("pre_mix")
            outputs.append("pre_mix")
        if self.ratio:
            if self.full:
                outputs.extend(("global_kv", "indexer_k"))
            else:
                inputs.extend(("global_kv", "indexer_k" if self.reindex else "global_indices"))
            if self.reindex:
                outputs.append("global_indices")
                if self.candidate == self.layer_idx:
                    outputs.extend(("candidate_indices", "candidate_lengths"))
                elif self.candidate is not None and self.layer_idx > self.candidate:
                    inputs.extend(("candidate_indices", "candidate_lengths"))
                if self.indexer_loss_enabled:
                    outputs.append("indexer_loss")
        self.input_fields, self.output_fields = tuple(inputs), tuple(outputs)
        self._static_shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] | None = None

    def _packed_params(self, kwargs: dict) -> PackedSeqParams | None:
        if not self.is_attention:
            return None
        if kwargs.get("packed_seq_params") is not None:
            params = kwargs["packed_seq_params"]
        elif "cu_seqlens_q" in kwargs:
            max_seqlen = self.config.max_seqlen_per_dp_cp_rank * self.cp_size
            params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=kwargs["cu_seqlens_q"],
                cu_seqlens_kv=kwargs["cu_seqlens_kv"],
                cu_seqlens_q_padded=kwargs["cu_seqlens_q_padded"],
                cu_seqlens_kv_padded=kwargs["cu_seqlens_kv_padded"],
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
                cp_partition_mode=self.config.cp_partition_mode,
                local_cp_size=self.cp_size if self.cp_size > 1 else None,
                cp_group=self.cp_group if self.cp_size > 1 else None,
                pad_between_seqs=True,
            )
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

    def get_static_inputs(self, static_inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Extend DSv4 sample inputs with this layer's actual shared dependencies."""
        # CSA2 uses sparse indices and packed prefixes for visibility, never this mask.
        static_inputs.pop("attention_mask", None)
        hidden = static_inputs["hidden_states"]
        seq, batch = hidden.shape[:2]
        params = self._packed_params(static_inputs)
        capacity = 0
        if self.ratio:
            capacity = (
                get_thd_compressed_capacity(
                    seq * self.cp_size,
                    params.max_seqlen_q,
                    params.cu_seqlens_q.numel() - 1,
                    self.ratio,
                )
                if params is not None
                else seq // self.ratio
            )
        leading = (seq,) if params is not None else (batch, seq)
        max_keys = (
            params.max_seqlen_q // self.ratio if params is not None and self.ratio else capacity
        )
        candidate_width = min(
            self.config.csa2_candidate_topk_blocks,
            (max_keys + max(1, self.config.csa2_candidate_block_size) - 1)
            // max(1, self.config.csa2_candidate_block_size),
        )
        shapes = {
            "pre_mix": (seq, batch, self.config.num_residual_streams),
            "global_kv": (capacity, batch, self.config.v_head_dim),
            "indexer_k": (capacity, batch, self.config.dsa_indexer_head_dim),
            "global_indices": (*leading, min(self.config.dsa_indexer_topk, capacity)),
            "candidate_indices": (*leading, candidate_width),
            "candidate_lengths": leading,
        }
        for name in self.input_fields:
            dtype, requires_grad, fill = self.config.params_dtype, True, 0
            if name == "pre_mix":
                dtype = hidden.dtype if self.config.use_fused_mhc else torch.float32
                fill = 1 / self.config.num_residual_streams
            elif name == "indexer_k":
                requires_grad = self.indexer_loss_enabled
            elif name in ("global_indices", "candidate_indices", "candidate_lengths"):
                dtype, requires_grad = torch.int32, False
                fill = 0 if name == "candidate_lengths" else -1
            static_inputs[_PREFIX + name] = torch.full(
                shapes[name], fill, dtype=dtype, device=hidden.device, requires_grad=requires_grad
            )
        self._static_shapes = {
            name: (tuple(tensor.shape), tensor.dtype) for name, tensor in static_inputs.items()
        }
        return static_inputs

    @staticmethod
    def _value(name: str, state: CSA2State, mhc: SinglePassMHCState | None) -> Tensor | None:
        if name == "pre_mix":
            return None if mhc is None else mhc.pre_mix
        if name.startswith("candidate_"):
            return None if state.candidates is None else getattr(state.candidates, name[10:])
        return getattr(state, name)

    def capture(self, function: Callable, *args, **kwargs) -> tuple[Tensor, ...]:
        """Run capture/warmup with fresh state and export all changing tensor values."""
        values = {name: kwargs.pop(_PREFIX + name) for name in self.input_fields}
        hidden = args[0] if args else kwargs["hidden_states"]
        params = self._packed_params(kwargs)
        layout = (
            build_csa2_thd_layout(params, hidden.shape[0], cp_group=self.cp_group)
            if params is not None
            else None
        )
        compressed = (
            layout.for_compression(self.ratio) if layout is not None and self.ratio else None
        )
        state = CSA2State(
            global_kv=values.get("global_kv"),
            indexer_k=values.get("indexer_k"),
            global_indices=values.get("global_indices"),
            kv_source_layer=self.kv_source,
            index_source_layer=self.index_source,
            thd_layout=layout,
            compressed_layout=compressed,
            defer_indexer_loss=True,
        )
        if "candidate_indices" in values:
            state.candidates = CSA2CandidateBlocks(
                values["candidate_indices"],
                values["candidate_lengths"],
                self.config.csa2_candidate_block_size,
            )
            state.candidate_source_layer = self.candidate
        if use_fused_dsa_kernels(self.config):
            state.prepare_fused_kv()
        mhc = SinglePassMHCState(values.get("pre_mix")) if self.graph_mhc else None
        if self.is_attention:
            kwargs["cross_layer_state"] = state
            if params is not None:
                # The inner layer may reconstruct raw prefixes without a CP group.
                # Bind this capture's static communicator before its attention runs.
                for suffix in ("q", "kv", "q_padded", "kv_padded"):
                    kwargs.pop("cu_seqlens_" + suffix, None)
                kwargs["packed_seq_params"] = params
        if mhc is not None:
            kwargs["mhc_state"] = mhc
        outputs = tuple(function(*args, **kwargs))
        shared = tuple(self._value(name, state, mhc) for name in self.output_fields)
        if any(value is None for value in shared):
            raise RuntimeError("CSA2 CUDA Graph capture did not produce its declared state")
        # Preserve the selection-only contract even if TE wraps floating outputs
        # in an autograd Function because its other outputs require gradients.
        if "indexer_k" in self.output_fields and not self.indexer_loss_enabled:
            shared = tuple(
                value.detach() if name == "indexer_k" else value
                for name, value in zip(self.output_fields, shared)
            )
        if layout is not None:
            shared += tuple(getattr(state.thd_layout, name) for name in _LAYOUT_FIELDS)
            if self.ratio:
                shared += tuple(
                    getattr(state.compressed_layout, name) for name in _COMPRESSED_FIELDS
                )
        return outputs + shared

    def replay(self, function: Callable, *args, **kwargs):
        """Publish graph tensors to this forward's state before running any eager tail."""
        # Validation/forward-only uses the normal layer contract and owns no graph slot.
        if not torch.is_grad_enabled():
            return function.__self__.forward(*args, **kwargs)
        state = kwargs.pop("cross_layer_state", None)
        mhc = kwargs.pop("mhc_state", None)
        kwargs.pop("attention_mask", None)
        hidden = args[0] if args else kwargs["hidden_states"]
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
            value = self._value(name, state, mhc)
            if value is None:
                raise ValueError(f"CSA2 CUDA Graph replay requires {name}")
            if name == "indexer_k" and not self.indexer_loss_enabled:
                value = value.detach()
            kwargs[_PREFIX + name] = value
        if self.graph_mhc and mhc is None:
            raise ValueError("CSA2 CUDA Graph replay requires single-pass mHC state")
        if params is not None:
            if (
                params.qkv_format != "thd"
                or params.max_seqlen_q != self.config.max_seqlen_per_dp_cp_rank * self.cp_size
                or params.max_seqlen_kv != params.max_seqlen_q
            ):
                raise ValueError("CSA2 CUDA Graph THD requires the configured static max_seqlen")
            for suffix in ("q", "kv"):
                logical = getattr(params, "cu_seqlens_" + suffix)
                physical = getattr(params, "cu_seqlens_" + suffix + "_padded")
                kwargs["cu_seqlens_" + suffix] = logical
                kwargs["cu_seqlens_" + suffix + "_padded"] = (
                    logical if physical is None else physical
                )
        kwargs.pop("packed_seq_params", None)
        if self._static_shapes is not None:
            values = {**kwargs, "hidden_states": hidden}
            for name, (shape, dtype) in self._static_shapes.items():
                value = values.get(name)
                if name == "padding_mask" and value is None:
                    # The ordinary MoE path interprets None as no masked tokens.
                    value = kwargs[name] = torch.zeros(shape, dtype=dtype, device=hidden.device)
                if value is None or tuple(value.shape) != shape or value.dtype != dtype:
                    raise ValueError(
                        f"CSA2 CUDA Graph input {name} does not match its captured shape/dtype"
                    )

        def restore(outputs):
            count = len(self.output_fields)
            if params is not None:
                count += len(_LAYOUT_FIELDS) + (len(_COMPRESSED_FIELDS) if self.ratio else 0)
            if count == 0:
                result, shared = outputs, ()
            else:
                result, shared = outputs[:-count], outputs[-count:]
            values = dict(zip(self.output_fields, shared))
            if "indexer_loss" in values:
                # TE jointly backpropagates every graph output, including zeros
                # for unused ones. Keep the unconditional auxiliary gradient
                # injector on the runtime hidden branch, as in eager training.
                result = (
                    DSAIndexerLossAutoScaler.apply(result[0], values["indexer_loss"]),
                    *result[1:],
                )
            if self.graph_mhc:
                mhc.pre_mix = values["pre_mix"]
            if self.is_attention:
                if self.full:
                    state.global_kv, state.indexer_k = values["global_kv"], values["indexer_k"]
                    if not self.indexer_loss_enabled:
                        state.indexer_k = state.indexer_k.detach()
                    state.global_kv_flat = state.indexer_k_flat = None
                    state.kv_source_layer = self.layer_idx
                    state.candidates = None
                    state.candidate_source_layer = None
                if self.reindex:
                    state.global_indices = values["global_indices"]
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
                    start = len(self.output_fields)
                    layout_values = dict(zip(_LAYOUT_FIELDS, shared[start:]))
                    state.thd_layout = CSA2THDLayout(
                        total_tokens=hidden.shape[0],
                        max_seqlen=params.max_seqlen_q,
                        cp_size=self.cp_size,
                        cp_rank=self.cp_rank,
                        **layout_values,
                    )
                    if self.ratio:
                        compressed_values = dict(
                            zip(_COMPRESSED_FIELDS, shared[start + len(_LAYOUT_FIELDS) :])
                        )
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

        if not self.is_attention and not self.graph_mhc:
            # A plain FFN has no model state to publish. Its existing partial replay
            # may rebuild kwargs when attention is outside the requested graph scope.
            return function(*args, **kwargs)
        return function(*args, **kwargs, _te_graph_output_handler=restore)

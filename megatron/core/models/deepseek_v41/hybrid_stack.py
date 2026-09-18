# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HybridStack specialisation for DeepSeek-V4.1.

Adds to the generic stack:

* one :class:`DSv41SharedState` per forward, handed to every layer (CSA2 sharing and the
  single-pass ``H_pre`` handoff);
* Engram hashing of the input ids and Engram modules attached to the configured layers;
* the V4.1 output contraction (aggregate with the last layer's ``H_pre``; no separate head
  mixing parameters);
* the pipeline handoff of ``H_pre`` across stage boundaries, packed behind the residual
  streams of the transferred tensor.
"""

from contextlib import nullcontext
from typing import List, Optional, Union

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.deepseek_v41.engram import (
    EngramMemory,
    EngramTableLayout,
    NgramHasher,
    load_token_map,
)
from megatron.core.models.deepseek_v41.hyper_connection import SinglePassHyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.csa2.roles import (
    SYMBOLS_PER_MODEL_LAYER,
    is_attention_position,
    model_layer_id_from_layer_number,
)
from megatron.core.transformer.experimental_attention_variant.csa2.state import (
    DSv41SharedState,
    StateKey,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import WrappedTensor

_ATTENTION_SYMBOLS = {Symbols.WINDOW, Symbols.DS_ATTENTION}


def validate_dsv41_layer_type_list(layer_type_list, pp_layer_offset: int) -> None:
    """Every model layer must be one attention symbol followed by one MoE symbol."""
    if pp_layer_offset % SYMBOLS_PER_MODEL_LAYER != 0:
        raise ValueError(
            f"DeepSeek-V4.1 pipeline stages must start at a model-layer boundary; "
            f"pp_layer_offset={pp_layer_offset} is odd"
        )
    if len(layer_type_list) % SYMBOLS_PER_MODEL_LAYER != 0:
        raise ValueError(
            f"DeepSeek-V4.1 pipeline stages must hold whole model layers; got "
            f"{len(layer_type_list)} pattern symbols"
        )
    for i, symbol in enumerate(layer_type_list):
        expected_attention = i % SYMBOLS_PER_MODEL_LAYER == 0
        if expected_attention and symbol not in _ATTENTION_SYMBOLS:
            raise ValueError(
                f"DeepSeek-V4.1 pattern position {pp_layer_offset + i} must be an attention "
                f"symbol ({sorted(_ATTENTION_SYMBOLS)}), got '{symbol}'"
            )
        if not expected_attention and symbol != Symbols.MOE:
            raise ValueError(
                f"DeepSeek-V4.1 pattern position {pp_layer_offset + i} must be the MoE symbol "
                f"'{Symbols.MOE}', got '{symbol}'"
            )


def validate_dsv41_pipeline_segment(
    config: TransformerConfig, layer_type_list, pp_layer_offset: int
) -> None:
    """Every CSA2 consumer on this stage must find its KV / index / candidate source here."""
    from megatron.core.transformer.experimental_attention_variant.csa2.attention import (
        get_csa2_plan,
    )

    plan = get_csa2_plan(config)
    first = model_layer_id_from_layer_number(pp_layer_offset + 1)
    last = model_layer_id_from_layer_number(pp_layer_offset + len(layer_type_list))
    for layer_id in range(first, last + 1):
        layer_plan = plan[layer_id]
        needed = {
            "csa2_kv_source_layers": layer_plan.kv_source,
            "csa2_index_source_layers": layer_plan.index_source,
            "csa2_candidate_source_layer": (
                plan.candidate_source_layer if layer_plan.uses_candidates else None
            ),
        }
        for field, source in needed.items():
            if source is not None and not first <= source <= last:
                raise ValueError(
                    f"model layer {layer_id} reads from source layer {source} ({field}), which "
                    f"is not on this pipeline stage (model layers {first}-{last}); move the "
                    "pipeline boundary so every CSA2 source stays with its consumers"
                )


def build_engram_layout(config: TransformerConfig) -> Optional[EngramTableLayout]:
    """Bucket layout from the config, or None when Engram is disabled."""
    if not config.engram_layer_ids:
        return None
    return EngramTableLayout.build(
        layer_ids=config.engram_layer_ids,
        num_embeddings=config.engram_num_embeddings,
        max_ngram_size=config.engram_max_ngram_size,
        n_heads=config.engram_n_heads,
        head_dim=config.engram_head_dim,
        bucket_size=config.engram_bucket_size,
    )


class DSv41HybridStack(HybridStack):
    """HybridStack with DeepSeek-V4.1 cross-layer state, Engram and single-pass mHC head."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: HybridStackSubmodules,
        pre_process: bool = True,
        layer_type_list: Optional[list] = None,
        pp_layer_offset: int = 0,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
        mtp_layer_number: Optional[int] = None,
        hash_moe_layer_threshold: Optional[int] = None,
        name: str | None = None,
    ) -> None:
        if getattr(config, "dsv4_version", "v4") != "v4.1":
            raise ValueError("DSv41HybridStack requires dsv4_version='v4.1'")
        if not config.enable_hyper_connections:
            raise ValueError("DSv41HybridStack requires enable_hyper_connections=True")
        if is_mtp_layer:
            raise NotImplementedError("DeepSeek-V4.1 MTP layers are out of scope")
        assert layer_type_list is not None, "layer_type_list must be provided"
        validate_dsv41_layer_type_list(layer_type_list, pp_layer_offset)
        validate_dsv41_pipeline_segment(config, layer_type_list, pp_layer_offset)

        self._engram_layout = build_engram_layout(config)
        self._engram_pg_collection = pg_collection
        self._current_state: Optional[DSv41SharedState] = None
        self._first_layer_number = pp_layer_offset + 1

        super().__init__(
            config=config,
            submodules=submodules,
            pre_process=pre_process,
            layer_type_list=layer_type_list,
            pp_layer_offset=pp_layer_offset,
            post_layer_norm=post_layer_norm,
            post_process=post_process,
            device=device,
            dtype=dtype,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            mtp_layer_number=mtp_layer_number,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
            name=name,
        )

        self.engram_hasher: Optional[NgramHasher] = None
        has_local_engram = any(getattr(layer, "engram", None) is not None for layer in self.layers)
        if has_local_engram and not pre_process:
            local_engram_layers = [
                model_layer_id_from_layer_number(layer.layer_number)
                for layer in self.layers
                if getattr(layer, "engram", None) is not None
            ]
            raise ValueError(
                f"Engram layers {local_engram_layers} fall on a pipeline stage without the "
                "embedding; only the first stage receives input_ids. Move the pipeline "
                "boundary after the last Engram layer."
            )
        if has_local_engram:
            token_map = load_token_map(
                config.engram_token_map_path, config.engram_compressed_vocab_size
            )
            self.engram_hasher = NgramHasher(
                self._engram_layout,
                token_map,
                config.engram_compressed_vocab_size,
                config.engram_pad_token_id,
            )

    # ---- construction hooks -------------------------------------------------------------------

    def _engram_shard_group(self):
        if self.config.engram_shard_group == "none" or self._engram_pg_collection is None:
            return None
        group = getattr(self._engram_pg_collection, "ep", None)
        if group is None or torch.distributed.get_world_size(group) == 1:
            return None
        return group

    def _wrap_hyper_connection_layer(self, layer: MegatronModule, layer_number: int):
        engram = None
        engram_index = None
        if self._engram_layout is not None and is_attention_position(layer_number):
            model_layer_id = model_layer_id_from_layer_number(layer_number)
            if model_layer_id in self._engram_layout.layer_ids:
                engram_index = self._engram_layout.layer_ids.index(model_layer_id)
                engram = EngramMemory(
                    self.config,
                    self._engram_layout,
                    engram_index,
                    shard_group=self._engram_shard_group(),
                    name=f"engram.{model_layer_id}",
                )
        return SinglePassHyperConnectionHybridLayer(
            self.config, layer, engram=engram, engram_index=engram_index
        )

    def _build_output_contract_params(self) -> None:
        # V4.1 aggregates the head input with the last layer's H_pre: no extra parameters.
        return None

    def _extra_layer_kwargs(self) -> dict:
        return {"dsv41_state": self._current_state}

    def _contract_output_streams(self, hidden_states: Tensor) -> Tensor:
        # Same fp32 weighted sum as the per-layer aggregation: rounding H_pre (sigmoid weights in
        # (0, 1)) to bf16 before the head contraction would cost ~3 significant digits on every
        # logit while every other mHC mix in the stack is computed in fp32.
        last_layer = self.layers[-1]
        h_pre = self._current_state.final_h_pre(last_layer.layer_number)
        return last_layer._aggregate_op(hidden_states, h_pre, self.config.num_residual_streams)

    # ---- full activation recomputation ----------------------------------------------------------

    def _live_state_keys(self, first_pending_index: int) -> List[StateKey]:
        """Shared-state entries that layers ``self.layers[first_pending_index:]`` (or the head /
        the outgoing pipeline handoff) still read, given every earlier layer has run."""
        from megatron.core.transformer.experimental_attention_variant.csa2.attention import (
            get_csa2_plan,
        )

        plan = get_csa2_plan(self.config)
        first_pending_number = self._first_layer_number + first_pending_index
        done_layer_numbers = range(self._first_layer_number, first_pending_number)
        done_models = {model_layer_id_from_layer_number(n) for n in done_layer_numbers}
        last_local_model = model_layer_id_from_layer_number(
            self._first_layer_number + len(self.layers) - 1
        )
        pending_models = set(
            range(model_layer_id_from_layer_number(first_pending_number), last_local_model + 1)
        )
        keys: List[StateKey] = []
        for layer_id in sorted(done_models):
            lp = plan[layer_id]
            if lp.runs_compressor and any(plan[c].kv_source == layer_id for c in pending_models):
                keys += [("kv", layer_id), ("ik", layer_id)]
            if lp.runs_indexer and any(plan[c].index_source == layer_id for c in pending_models):
                keys.append(("topk", layer_id))
            if lp.is_candidate_source and any(plan[c].uses_candidates for c in pending_models):
                keys.append(("cand", layer_id))
        if first_pending_index > 0:
            keys.append(("hpre", first_pending_number - 1))
        return keys

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_pos_emb,
        packed_seq_params: Optional[PackedSeqParams],
        padding_mask,
        input_ids: Optional[Tensor],
        use_inner_quantization_context: bool,
    ) -> Tensor:
        """Full recompute with the shared state threaded explicitly through every checkpoint.

        Each chunk receives the live shared tensors as checkpoint inputs, rebuilds a fresh
        state (so replay publishes without conflicts), runs its layers and returns the tensors
        later chunks still need as checkpoint outputs; gradients therefore cross chunk
        boundaries like any other checkpointed activation.
        """
        from megatron.core import tensor_parallel
        from megatron.core.transformer.experimental_attention_variant.csa2.attention import (
            get_csa2_plan,
        )

        if self.config.fp8 or self.config.fp4:
            raise NotImplementedError(
                "DeepSeek-V4.1 full recompute with FP8/FP4 (TE checkpoint) is not wired yet"
            )
        plan = get_csa2_plan(self.config)
        ratios = {lp.layer_id: lp.compress_ratio for lp in plan.layers if lp.runs_compressor}
        outer_state = self._current_state
        engram_hash_ids = outer_state.engram_hash_ids

        def make_chunk(start: int, end: int, keys_in: List[StateKey], keys_out: List[StateKey]):
            def chunk(hidden, *tensors_in):
                state = DSv41SharedState()
                state.engram_hash_ids = engram_hash_ids
                state.load(keys_in, tensors_in, ratios)
                for index in range(start, end):
                    layer = self.layers[index]
                    # FP8/FP4 per-layer contexts are rejected above; bf16 needs none.
                    with nullcontext():
                        hidden, _ = layer(
                            hidden_states=hidden,
                            attention_mask=attention_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                            input_ids=input_ids,
                            dsv41_state=state,
                        )
                return (hidden, *state.export(keys_out))

            return chunk

        def run_chunk(start: int, end: int, use_checkpoint: bool, hidden, keys_in, tensors_in):
            keys_out = self._live_state_keys(end)
            fn = make_chunk(start, end, keys_in, keys_out)
            if use_checkpoint:
                outputs = tensor_parallel.checkpoint(
                    fn, self.config.distribute_saved_activations, hidden, *tensors_in
                )
            else:
                outputs = fn(hidden, *tensors_in)
            return outputs[0], keys_out, list(outputs[1:])

        keys: List[StateKey] = []
        tensors: List[Tensor] = []
        if not self.pre_process:
            # The predecessor stage's H_pre (published by _split_incoming_handoff) feeds the
            # first local layer; it enters the first chunk as a checkpoint input so the replay
            # sees it and its gradient reaches the pipeline input tensor.
            keys = [("hpre", self._first_layer_number - 1)]
            tensors = outer_state.export(keys)
        n_layers = len(self.layers)
        if self.config.recompute_method == 'uniform':
            start = 0
            while start < n_layers:
                end = min(start + self.config.recompute_num_layers, n_layers)
                hidden_states, keys, tensors = run_chunk(
                    start, end, True, hidden_states, keys, tensors
                )
                start = end
        elif self.config.recompute_method == 'block':
            for index in range(n_layers):
                use_checkpoint = index < self.config.recompute_num_layers
                hidden_states, keys, tensors = run_chunk(
                    index, index + 1, use_checkpoint, hidden_states, keys, tensors
                )
        else:
            raise ValueError("Invalid activation recompute method.")

        # Hand the final entries (last H_pre for the head / pipeline handoff) to the outer state.
        outer_state.load(keys, tensors, ratios)
        return hidden_states

    # ---- Engram hashing -----------------------------------------------------------------------

    def _hash_engram_rows(
        self, input_ids: Tensor, packed_seq_params: Optional[PackedSeqParams]
    ) -> Tensor:
        """Row ids for the local tokens; packed layouts stop n-grams at segment starts and, under
        context parallelism, see the tokens preceding the local block via an id all-gather."""
        if packed_seq_params is None or packed_seq_params.qkv_format != 'thd':
            return self.engram_hasher(input_ids)
        from megatron.core.transformer.experimental_attention_variant.csa2 import thd

        cu, seq_lens = thd.packed_layout(packed_seq_params)
        cp_group = packed_seq_params.cp_group or self.pg_collection.cp
        cp_size = cp_group.size() if cp_group is not None else 1
        if cp_size > 1 and packed_seq_params.cp_partition_mode != "contiguous":
            raise ValueError(
                "DeepSeek-V4.1 Engram under CP requires cp_partition_mode='contiguous'"
            )
        if cp_size == 1:
            full_ids = input_ids
        else:
            # Token ids are cheap: gather the whole packed layout, hash once, keep the local
            # block.
            chunks = [torch.empty_like(input_ids) for _ in range(cp_size)]
            torch.distributed.all_gather(chunks, input_ids.contiguous(), group=cp_group)
            full_ids = torch.cat(chunks, dim=1)
        # Padding rows of a padded pack hash as dead tokens: no n-gram may read them.
        valid = thd.row_metadata(cu, full_ids.size(1), 0, seq_lens).valid.unsqueeze(0)
        rows = self.engram_hasher(full_ids, token_mask=valid, cu_seqlens=cu)  # [s, 1, L, cols]
        if cp_size == 1:
            return rows
        local = input_ids.size(1)
        start = cp_group.rank() * local
        return rows[start : start + local]

    # ---- pipeline handoff of H_pre --------------------------------------------------------------

    @property
    def _handoff_channels(self) -> int:
        return self.config.num_residual_streams

    def _split_incoming_handoff(self, tensor: Tensor, state: DSv41SharedState) -> Tensor:
        """Non-first stages receive ``[s, b, n*C + n]``: strip and publish the predecessor H_pre."""
        n = self._handoff_channels
        streams, h_pre = tensor[..., :-n], tensor[..., -n:]
        state.publish_h_pre(self._first_layer_number - 1, h_pre)
        return streams.contiguous()

    def _append_outgoing_handoff(self, hidden_states: Tensor, state: DSv41SharedState) -> Tensor:
        """Non-last stages send ``[s, b, n*C + n]`` with the last layer's H_pre appended."""
        h_pre = state.final_h_pre(self.layers[-1].layer_number).to(hidden_states.dtype)
        return torch.cat([hidden_states, h_pre], dim=-1)

    # ---- forward ------------------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask=None,
        input_ids: Optional[Tensor] = None,
    ):
        state = DSv41SharedState()
        if self.engram_hasher is not None:
            if input_ids is None:
                raise RuntimeError("DeepSeek-V4.1 Engram layers need input_ids in the decoder call")
            state.engram_hash_ids = self._hash_engram_rows(input_ids, packed_seq_params)

        restore_input_tensor = None
        if not self.pre_process:
            restore_input_tensor = self.input_tensor
            self.input_tensor = self._split_incoming_handoff(self.input_tensor, state)

        self._current_state = state
        try:
            output = super().forward(
                hidden_states,
                attention_mask,
                inference_context,
                rotary_pos_emb,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
                input_ids=input_ids,
            )
            if not self.post_process:
                output = self._append_outgoing_handoff(output, state)
        finally:
            self._current_state = None
            if restore_input_tensor is not None:
                self.input_tensor = restore_input_tensor
        return output

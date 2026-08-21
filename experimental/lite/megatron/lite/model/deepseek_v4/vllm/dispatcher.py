"""DS4 vLLM route-preserving normal-DeepEP dispatcher."""

from __future__ import annotations

import torch

from megatron.lite.model.deepseek_v4.vllm.deepep_route import (
    _VLLMEPGatherWithBF16Backward,
    _compact_route_preserving_metadata_inputs,
    _deepep_route_handle_received_rows,
    _scatter_deepep_routes_with_padding,
    _validate_and_order_route_preserving_outputs,
)
from megatron.lite.primitive.modules.dispatcher import (
    TokenDispatcher,
    _DeepEPCombine,
    _DeepEPDispatch,
    _dispatch_deepep_raw,
)


class VLLMAlignedNormalDeepEPDispatcher(TokenDispatcher):
    """Match vLLM route identity while using only normal DeepEP transport."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ep_size > 1 and not self.use_deepep:
            raise RuntimeError("vLLM alignment at EP>1 requires normal DeepEP")

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._dispatch_aligned(hidden_states, topk_scores, topk_indices)

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        route_outputs = expert_output.index_select(0, self._metadata_route_rows)
        if self.ep_size > 1:
            source_routes = _DeepEPCombine.apply(
                self._deepep_group,
                route_outputs,
                self._route_handle,
                True,
                False,
            )
        else:
            source_routes = route_outputs
        output = _VLLMEPGatherWithBF16Backward.apply(
            source_routes,
            self._source_indices,
            self._source_weights,
            self._source_output_index,
            True,
        )
        for name in (
            "_metadata_route_rows",
            "_route_handle",
            "_source_indices",
            "_source_weights",
            "_source_output_index",
        ):
            delattr(self, name)
        self._local_tpe_list = None
        return output

    def _dispatch_aligned(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_deepep:
            self._ensure_deepep_buffer(hidden_states)
        if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
            raise TypeError("aligned DeepEP requires BF16 [tokens, hidden]")
        if hidden_states.shape[1] < 16:
            raise ValueError("aligned DeepEP requires hidden size >= 16")
        if topk_indices.shape != topk_scores.shape:
            raise ValueError("top-k IDs and scores must have identical shapes")
        topk_indices = topk_indices.long().contiguous()
        topk_scores = topk_scores.float().contiguous()

        if self.ep_size > 1:
            (
                received_hidden,
                received_indices,
                received_weights,
                received_per_expert,
                _,
            ) = _DeepEPDispatch.apply(
                self._deepep_group,
                hidden_states,
                topk_indices,
                topk_scores,
                self.num_experts,
                False,
                False,
            )
            (
                route_indices,
                route_weights,
                route_fingerprints,
                source_output_index,
            ) = _compact_route_preserving_metadata_inputs(
                hidden_states,
                topk_indices,
                topk_scores,
            )
            (
                received_fingerprints,
                received_route_indices,
                received_route_weights,
                _,
                route_handle,
                _,
            ) = _dispatch_deepep_raw(
                self._deepep_group,
                route_fingerprints,
                route_indices,
                route_weights,
                self.num_experts,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
        else:
            received_hidden = hidden_states
            received_indices = topk_indices
            received_weights = topk_scores
            positions = torch.nonzero(topk_indices >= 0, as_tuple=False)
            token_rows = positions[:, 0]
            topk_slots = positions[:, 1]
            received_fingerprints = (
                hidden_states.detach().narrow(1, 0, 16).index_select(0, token_rows)
            )
            received_route_indices = topk_indices[token_rows, topk_slots]
            received_route_weights = topk_scores[token_rows, topk_slots]
            route_handle = None
            source_output_index = torch.arange(
                topk_indices.numel(), device=topk_indices.device, dtype=torch.long
            ).reshape_as(topk_indices)
            received_per_expert = torch.bincount(
                received_route_indices.reshape(-1).long(),
                minlength=self.num_local_experts,
            )

        expected_route_count = (
            _deepep_route_handle_received_rows(route_handle)
            if route_handle is not None
            else int((received_indices >= 0).sum().item())
        )
        (
            expert_hidden,
            expert_probs,
            output_index,
            sanitized_indices,
            _,
            _,
            _,
            positions,
        ) = _scatter_deepep_routes_with_padding(
            received_hidden,
            received_indices,
            received_weights,
            received_per_expert,
            return_route_positions=True,
            expected_route_count=expected_route_count,
        )
        self._metadata_route_rows = _validate_and_order_route_preserving_outputs(
            expert_hidden,
            received_hidden,
            sanitized_indices,
            received_weights,
            output_index,
            received_fingerprints,
            received_route_indices.reshape(-1),
            received_route_weights.reshape(-1),
            order_outputs=False,
            route_positions=positions,
            return_route_rows=True,
        )
        self._route_handle = route_handle
        self._source_indices = topk_indices
        self._source_weights = topk_scores
        self._source_output_index = source_output_index
        self._local_tpe_list = received_per_expert.tolist()
        return expert_hidden, received_per_expert, expert_probs

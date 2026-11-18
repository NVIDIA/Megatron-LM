# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoE layer mixins and helpers for TE CUDA graph partial capture."""

import functools
from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.typed_torch import apply_module
from megatron.core.utils import internal_api, nvtx_range_pop, nvtx_range_push


class TEPartialCaptureMoEMixin:
    """Resume MoE work after a parent graph captured the router or preprocessing.

    The parent retains graph ownership; this mixin restores the MoE intermediates
    and resumes its eager operations without registering a second graph.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config.cuda_graph_impl == "transformer_engine":
            # Cudagraph tensor store for resuming the forward pass from the end of the cudagraph.
            self.te_cuda_graph_tensor_store = TEMoECudaGraphTensorStore()
            # Keep capture validation separate for each training/inference dispatcher.
            self.te_cuda_graph_valid_dispatcher_attrs = {}
            if {CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess} & set(
                self.config.cuda_graph_modules
            ):
                self.run_shared_experts_after_router = False

    def _te_cuda_graph_resume(self, cuda_graph_output, padding_mask, packed_seq_params=None):
        """Restore captured intermediates and resume the eager MoE steps.

        The parent retains the residual. With EP overlap, the schedule owns the
        remaining work and receives (hidden_states, probs, shared_expert_output).
        """
        shared_expert_output, routing_map = None, None
        if (
            self.config.moe_shared_expert_intermediate_size is not None
            and not self.config.moe_shared_expert_overlap
        ):
            # The shared expert output is the last second element in the CUDA graph output.
            shared_expert_output = cuda_graph_output.pop()

        if CudaGraphModule.moe_preprocess in self.config.cuda_graph_modules:
            # CUDA graph output is [hidden_states, probs] + attributes outputs.
            (hidden_states, probs), attr_outputs = cuda_graph_output[:2], cuda_graph_output[2:]
            valid_cudagraph_attrs = self.te_cuda_graph_valid_dispatcher_attrs.get(
                self.token_dispatcher
            )
            assert len(attr_outputs) == len(
                valid_cudagraph_attrs
            ), f"attr_outputs: {len(attr_outputs)} != {len(valid_cudagraph_attrs)}"
            for i, attr_name in enumerate(valid_cudagraph_attrs):
                hier_attr_name = attr_name.split('.')
                attr = self.token_dispatcher
                for name in hier_attr_name[:-1]:
                    attr = getattr(attr, name)
                setattr(attr, hier_attr_name[-1], attr_outputs[i])
        else:
            # CUDA graph output is [hidden_states, probs, routing_map].
            assert len(cuda_graph_output) == 3, (
                "CUDA graph output should be [hidden_states, probs, routing_map], "
                f"but got {len(cuda_graph_output)} elements"
            )
            hidden_states, probs, routing_map = cuda_graph_output

        # Resume the MoELayer forward pass from the end of the CUDA graph scope.
        # The MoE layer will skip redundant computations when we pass in the calculated values
        # through the keyword arguments. See MoELayer.forward docstring for more details.
        nvtx_range_push(suffix="mlp")
        self.te_cuda_graph_tensor_store.set(
            hidden_states=hidden_states,
            probs=probs,
            routing_map=routing_map,
            shared_expert_output=shared_expert_output,
        )
        # If EP overlap is enabled, remaining of mlp will be called as fine_grained_callables
        # and should be skipped here.
        if self.config.overlap_moe_expert_parallel_comm:
            probs, routing_map = self.route(hidden_states, padding_mask)
            hidden_states, probs = self.preprocess(
                hidden_states, probs, routing_map, padding_mask, packed_seq_params=packed_seq_params
            )
            nvtx_range_pop(suffix="mlp")
            return hidden_states, probs, shared_expert_output
        # The cached router outputs are unmasked; eager preprocess needs the replay mask
        # so the dispatcher can exclude padded rows.
        mlp_output_with_bias = apply_module(self)(hidden_states, padding_mask=padding_mask)
        self.te_cuda_graph_tensor_store.clear()
        nvtx_range_pop(suffix="mlp")

        return mlp_output_with_bias

    def _te_cuda_graph_pre_dispatch(self, hidden_states, padding_mask, packed_seq_params=None):
        """Run the eager MoE prefix after an attention-only graph for EP overlap."""
        shared_expert_output = self.shared_experts_compute(hidden_states)
        probs, routing_map = self.route(hidden_states, padding_mask)
        hidden_states, probs = self.preprocess(
            hidden_states, probs, routing_map, padding_mask, packed_seq_params=packed_seq_params
        )
        return hidden_states, probs, shared_expert_output


class TEMoECudaGraphPartialCaptureSignal(Exception):
    """
    Used to early-return from a MoE layer forward pass in CUDA graph capture.
    This signal is raised when we are partially capturing the CUDA graph of the MoE layer,
    and the related intermediate tensors are recorded in self.kwargs.
    Call self.get_early_return_outputs() to collect the CUDA graph outputs.
    """

    def __init__(self, moe_layer, return_step: str, **kwargs):
        self.moe_layer = moe_layer
        self.return_step = return_step
        self.kwargs = kwargs

    def get_early_return_outputs(
        self, hidden_states: torch.Tensor, shared_expert_output: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get the CUDA graph early return outputs for the MoE layer, including the intermediate
        tensors and the intermediate attributes of the token dispatcher.

        The returned output tensors are in the order of:
        - routed experts path outputs
          - hidden states, probs, and routing map for capturing router
          - hidden states and probs for capturing router and preprocess
        - intermediate attributes of the token dispatcher (if capturing the preprocess step)
        - shared expert path output (if exists)
        """
        if self.return_step == "route":
            # Capturing the router step returns three intermediate tensors:
            # hidden states, routing probabilities, and routing map.
            outputs = [hidden_states, self.kwargs['probs'], self.kwargs['routing_map']]
        elif self.return_step == "preprocess":
            # Capturing the preprocess step returns two intermediate tensors:
            # hidden states and routing probabilities.
            # It also returns the intermediate attributes of the token dispatcher, recorded in
            # "token_dispatcher.cudagraph_attrs".
            outputs = [self.kwargs['hidden_states'], self.kwargs['probs']]
            valid_cudagraph_attrs = []
            for attr_name in self.moe_layer.token_dispatcher.cudagraph_attrs:
                hier_attr_name = attr_name.split('.')
                attr = self.moe_layer.token_dispatcher
                for name in hier_attr_name:
                    attr = getattr(attr, name, None)
                    if attr is None:
                        break
                if isinstance(attr, torch.Tensor):
                    outputs.append(attr)
                    valid_cudagraph_attrs.append(attr_name)
            dispatcher = self.moe_layer.token_dispatcher
            attrs_by_dispatcher = self.moe_layer.te_cuda_graph_valid_dispatcher_attrs
            if dispatcher not in attrs_by_dispatcher:
                attrs_by_dispatcher[dispatcher] = valid_cudagraph_attrs
            else:
                assert attrs_by_dispatcher[dispatcher] == valid_cudagraph_attrs, (
                    "valid_cudagraph_attrs mismatch: "
                    f"{attrs_by_dispatcher[dispatcher]} != "
                    f"{valid_cudagraph_attrs}"
                )
        # Also return the shared expert output, if it is not None.
        if shared_expert_output is not None:
            outputs.append(shared_expert_output)
        return outputs


@internal_api
@dataclass
class TEMoECudaGraphTensorStore:
    """Storage for tensors used in CUDA graph replay for MoE layers.

    This dataclass stores intermediate tensors computed during CUDA graph replay
    that need to be resumed from the end of the CUDA graph scope to skip redundant computations.

    Attributes:
        hidden_states (Optional[torch.Tensor]): The hidden states output from the CUDA graph replay.
        probs (Optional[torch.Tensor]): The routing probabilities for each token-expert pair.
        routing_map (Optional[torch.Tensor]): The sparse mapping indicating which experts
            were selected for each token. Used to skip the normal router step.
        shared_expert_output (Optional[torch.Tensor]): The output from shared experts
            computation. Used to skip the normal shared expert computation step.
    """

    hidden_states: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    routing_map: Optional[torch.Tensor] = None
    shared_expert_output: Optional[torch.Tensor] = None

    def is_empty(self) -> bool:
        """Check if the store has any non-None tensors.

        Returns:
            bool: True if all fields are None, False otherwise.
        """
        return all(
            getattr(self, field_name) is None
            for field_name in ['hidden_states', 'probs', 'routing_map', 'shared_expert_output']
        )

    def set(self, **kwargs):
        """Set the tensors in the store from keyword arguments."""
        for field_name, value in kwargs.items():
            assert field_name in [
                'hidden_states',
                'probs',
                'routing_map',
                'shared_expert_output',
            ], f"Invalid field name: {field_name}"
            if value is not None:
                assert isinstance(
                    value, torch.Tensor
                ), f"Value must be a torch.Tensor, got {type(value)} for field {field_name}"
                setattr(self, field_name, value)

    def clear(self):
        """Reset all stored tensors to None."""
        for field_name in ['hidden_states', 'probs', 'routing_map', 'shared_expert_output']:
            setattr(self, field_name, None)


def maybe_skip_or_early_return_by_te_cuda_graph(step_condition):
    """
    Decorator to skip certain codepaths in the MoE layer forward pass in CUDA graph replay,
    or early return from the MoE layer forward pass in CUDA graph capture.

    Args:
        step_condition: The step condition to check. Can be "shared_experts_compute", "route",
        or "preprocess". If "shared_experts_compute", the shared experts computation will be
        skipped in replay if it is in the CUDA graph scope. If "route" or "preprocess", the
        router or preprocess will be skipped in replay if it is in the CUDA graph scope, or
        early return from the MoE layer forward pass if it is in CUDA graph capturing mode.

    Returns:
        A decorator function that wraps the MoE layer forward pass.
    """

    def maybe_raise_signal(moe_layer, **kwargs):
        """
        Check if the MoE layer should early return for CUDA graph capture.
        If so, raise a TEMoECudaGraphPartialCaptureSignal.
        """
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        if (
            moe_layer.config.cuda_graph_impl == "transformer_engine"
            and moe_layer.training
            and is_graph_capturing()
        ):
            if (
                step_condition == "route"
                and CudaGraphModule.moe_router in moe_layer.config.cuda_graph_modules
                and CudaGraphModule.moe_preprocess not in moe_layer.config.cuda_graph_modules
            ):
                raise TEMoECudaGraphPartialCaptureSignal(moe_layer, "route", **kwargs)
            elif (
                step_condition == "preprocess"
                and CudaGraphModule.moe_preprocess in moe_layer.config.cuda_graph_modules
            ):
                raise TEMoECudaGraphPartialCaptureSignal(moe_layer, "preprocess", **kwargs)

    def decorator(func):

        @functools.wraps(func)
        def wrapped_func(moe_layer, *args, **kwargs):
            """
            Check if we should skip executing the original function based on the current
            step condition and the tensor store status. If the tensor can be found in the store,
            it indicates that it is already computed by the CUDA graph replay, so we can skip it.
            Otherwise, we execute the original function and check if we should raise a signal to
            early return in CUDA graph capture.
            """

            if moe_layer.config.cuda_graph_impl != "transformer_engine":
                return func(moe_layer, *args, **kwargs)

            from megatron.core.transformer.cuda_graphs import is_graph_capturing

            # The non-cudagraph codepath just calls the original function.
            if not is_graph_capturing() and moe_layer.te_cuda_graph_tensor_store.is_empty():
                return func(moe_layer, *args, **kwargs)

            assert (
                not is_graph_capturing() or moe_layer.te_cuda_graph_tensor_store.is_empty()
            ), "te_cuda_graph_tensor_store cannot be used when it is capturing cuda graph."
            if step_condition == "shared_experts_compute":
                if moe_layer.te_cuda_graph_tensor_store.shared_expert_output is None:
                    # Don't skip the shared expert computation.
                    shared_expert_output = func(moe_layer, *args, **kwargs)
                else:
                    # Skip the shared expert computation and get value from store.
                    shared_expert_output = moe_layer.te_cuda_graph_tensor_store.shared_expert_output
                return shared_expert_output
            elif step_condition == "route":
                if moe_layer.te_cuda_graph_tensor_store.probs is None:
                    # Don't skip the router.
                    assert (
                        moe_layer.te_cuda_graph_tensor_store.routing_map is None
                    ), "routing_map must be None if probs is None"
                    probs, routing_map = func(moe_layer, *args, **kwargs)

                    # Maybe early return after the router.
                    maybe_raise_signal(moe_layer, probs=probs, routing_map=routing_map)
                else:
                    # Skip the router and get value from store.
                    probs, routing_map = (
                        moe_layer.te_cuda_graph_tensor_store.probs,
                        moe_layer.te_cuda_graph_tensor_store.routing_map,
                    )
                return probs, routing_map
            elif step_condition == "preprocess":
                if (
                    moe_layer.te_cuda_graph_tensor_store.is_empty()
                    or moe_layer.te_cuda_graph_tensor_store.routing_map is not None
                ):
                    # Don't skip the preprocess.
                    hidden_states, probs = func(moe_layer, *args, **kwargs)

                    # Maybe early return after the preprocess.
                    maybe_raise_signal(moe_layer, hidden_states=hidden_states, probs=probs)
                else:
                    # Skip the preprocess and get value from store.
                    assert (
                        moe_layer.te_cuda_graph_tensor_store.hidden_states is not None
                        and moe_layer.te_cuda_graph_tensor_store.probs is not None
                    ), "hidden_states and probs must be given in moe_preprocess cudagraph replay"
                    hidden_states, probs = (
                        moe_layer.te_cuda_graph_tensor_store.hidden_states,
                        moe_layer.te_cuda_graph_tensor_store.probs,
                    )
                return hidden_states, probs

        return wrapped_func

    return decorator

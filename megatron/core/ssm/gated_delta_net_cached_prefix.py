# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Callable, List

import torch

from megatron.core.cached_prefix_utils import CachedPrefixParams, KVCache
from megatron.core.transformer import TransformerConfig


class ChunkedKVCacheForGatedDeltaNet(KVCache):
    """Chunked KV cache for gated delta net."""

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.conv1d_states_cache = []
        self.conv1d_grad_states_cache = []
        self.gated_delta_rule_states_cache = []
        self.gated_delta_rule_grad_states_cache = []

    def forward_conv1d_with_kv_cache(
        self,
        module: Callable,
        cached_prefix_params: CachedPrefixParams,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
    ):
        """Forward pass of conv1D with chunked KV cache."""
        n_prefix = len(cached_prefix_params.prefix_seqlens)
        assert len(self.conv1d_states_cache) == n_prefix, (
            f"Expected {n_prefix} prefix states, but got {len(self.conv1d_states_cache)=}. "
            "Please check the cached prefix parameters."
        )

        out = Conv1DFunctionWithChunkedKVCache.apply(
            module,
            x,
            weight,
            bias,
            activation,
            self.conv1d_states_cache,
            self.conv1d_grad_states_cache,
        )
        return out

    def forward_gated_delta_rule_with_kv_cache(
        self,
        module: Callable,
        cached_prefix_params: CachedPrefixParams,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        use_qk_l2norm_in_kernel: bool,
    ):
        """Forward pass of gated delta rule with chunked KV cache."""
        n_prefix = len(cached_prefix_params.prefix_seqlens)
        assert len(self.gated_delta_rule_states_cache) == n_prefix, (
            f"Expected {n_prefix} prefix states, but got {len(self.gated_delta_rule_states_cache)=}. "
            "Please check the cached prefix parameters."
        )

        out = GatedDeltaRuleFuncionWithChunkedKVCache.apply(
            module,
            query,
            key,
            value,
            g,
            beta,
            use_qk_l2norm_in_kernel,
            self.gated_delta_rule_states_cache,
            self.gated_delta_rule_grad_states_cache,
        )
        return out


class Conv1DFunctionWithChunkedKVCache(torch.autograd.Function):
    """Conv1D function with chunked KV cache."""

    @staticmethod
    def forward(
        ctx,
        module: Callable,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
        state_cache: List[torch.Tensor],
        grad_state_cache: List[torch.Tensor],
    ):
        if len(state_cache) == 0:
            initial_state = None
        else:
            initial_state = state_cache[-1]

        with torch.autograd.set_grad_enabled(True):
            out, final_state = module(
                x=x,
                weight=weight,
                bias=bias,
                activation=activation,
                initial_state=initial_state,
                output_final_state=True,
            )
        state_cache.append(final_state)
        grad_state_cache.append(None)

        ctx.save_for_backward(x, weight, bias, initial_state, out, final_state)
        ctx.state_cache = state_cache
        ctx.grad_state_cache = grad_state_cache

        out = out.clone()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, initial_state, out, final_state = ctx.saved_tensors
        state_cache = ctx.state_cache
        grad_state_cache = ctx.grad_state_cache

        final_state_grad = grad_state_cache.pop(-1)
        state_cache.pop(-1)

        # Prepare inputs and outputs for autograd.grad
        _outputs = (out,)
        _inputs = (x, weight)
        _grad_outputs = (grad_output,)
        if final_state_grad is not None:
            _outputs += (final_state,)
            _grad_outputs += (final_state_grad,)
        if bias is not None:
            _inputs += (bias,)
        if initial_state is not None:
            _inputs += (initial_state,)
        # Calculate gradients
        grads = torch.autograd.grad(
            outputs=_outputs, inputs=_inputs, grad_outputs=_grad_outputs, allow_unused=True
        )
        # Unpack gradients
        dx, dweight = grads[:2]
        grads = grads[2:]
        if bias is not None:
            dbias = grads[0]
            grads = grads[1:]
        else:
            dbias = None
        if initial_state is not None:
            dinitial_state = grads[0]
            if grad_state_cache[-1] is None:
                grad_state_cache[-1] = dinitial_state
            else:
                grad_state_cache[-1].add_(dinitial_state)

        return (
            None,  # module
            dx,
            dweight,  # weight
            dbias,  # bias
            None,  # activation
            None,  # state_cache
            None,  # grad_state_cache
        )


class GatedDeltaRuleFuncionWithChunkedKVCache(torch.autograd.Function):
    """Gated delta rule function with chunked KV cache."""

    @staticmethod
    def forward(
        ctx,
        module: Callable,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        use_qk_l2norm_in_kernel: bool,
        state_cache: List[torch.Tensor],
        grad_state_cache: List[torch.Tensor],
    ):
        if len(state_cache) == 0:
            initial_state = None
        else:
            initial_state = state_cache[-1]

        with torch.autograd.set_grad_enabled(True):
            out, final_state = module(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        state_cache.append(final_state)
        grad_state_cache.append(None)

        ctx.save_for_backward(query, key, value, g, beta, initial_state, out, final_state)
        ctx.state_cache = state_cache
        ctx.grad_state_cache = grad_state_cache

        out = out.clone()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, g, beta, initial_state, out, final_state = ctx.saved_tensors
        state_cache = ctx.state_cache
        grad_state_cache = ctx.grad_state_cache

        final_state_grad = grad_state_cache.pop(-1)
        state_cache.pop(-1)

        # Prepare inputs and outputs for autograd.grad
        _outputs = (out,)
        _inputs = (query, key, value, g, beta)
        _grad_outputs = (grad_output,)
        if final_state_grad is not None:
            _outputs += (final_state,)
            _grad_outputs += (final_state_grad,)
        if initial_state is not None:
            _inputs += (initial_state,)

        # Calculate gradients
        grads = torch.autograd.grad(
            outputs=_outputs, inputs=_inputs, grad_outputs=_grad_outputs, allow_unused=True
        )

        # Unpack gradients
        dquery, dkey, dvalue, dg, dbeta = grads[:5]
        grads = grads[5:]
        if initial_state is not None:
            dinitial_state = grads[0]
            if grad_state_cache[-1] is None:
                grad_state_cache[-1] = dinitial_state
            else:
                grad_state_cache[-1].add_(dinitial_state)

        return (
            None,  # module
            dquery,  # query
            dkey,  # key
            dvalue,  # value
            dg,  # g
            dbeta,  # beta
            None,  # use_qk_l2norm_in_kernel
            None,  # state_cache
            None,  # grad_state_cache
        )

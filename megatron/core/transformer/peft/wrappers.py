# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from NVIDIA-NeMo/Megatron-Bridge (Apache-2.0).
# Source: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/
# c031e61d82dafbe4655a2e2d7e742b88f7e81c3a/src/megatron/bridge/peft/adapter_wrapper.py

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core.utils import make_sharded_tensor_for_checkpoint

from .adapters import ParallelLinearAdapter

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict


def _plain_module_sharded_state_dict(
    module: nn.Module,
    prefix: str,
    sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
    pg_collection=None,
) -> "ShardedStateDict":
    """Build a sharded state dict for a module that has no ``sharded_state_dict``.

    Used as a fallback for plain ``nn.Linear`` / delta-only ``LinearAdapter`` leaves on
    the simple (non-parallel) path: each parameter is wrapped as a replicated
    :class:`ShardedTensor` so distributed checkpointing preserves the trained weights
    instead of silently dropping them.

    Args:
        module: The plain module whose parameters to shard.
        prefix: Key prefix applied to every parameter.
        sharded_offsets: Offsets prepended to each sharded tensor.

    Returns:
        A sharded state dict mapping ``f"{prefix}{param_name}"`` to ShardedTensor.
    """
    if pg_collection is None or pg_collection.tp is None or pg_collection.dp_cp is None:
        raise ValueError("Plain adapter checkpointing requires explicit TP and DP/CP groups")
    sharded_state_dict: Dict[str, Any] = {}
    for name, param in module.state_dict(prefix="", keep_vars=True).items():
        key = f"{prefix}{name}"
        sharded_state_dict[key] = make_sharded_tensor_for_checkpoint(
            param,
            key,
            prepend_offsets=sharded_offsets,
            tp_group=pg_collection.tp,
            dp_cp_group=pg_collection.dp_cp,
        )
    return sharded_state_dict


def _compute_mamba_dim_info(wrapped_module: nn.Module) -> Dict[str, int]:
    """Compute Mamba dimension information from a wrapped module's config.

    This follows the same logic as mamba_mixer.py to derive local tensor parallel
    dimensions from the TransformerConfig.

    Args:
        wrapped_module: The wrapped module (typically a linear projection in a Mamba layer).

    Returns:
        Dictionary containing d_inner_local_tp, ngroups_local_tp, d_state, and nheads_local_tp.
    """
    config = wrapped_module.config

    # Get base dimensions from config
    d_state = config.mamba_state_dim
    headdim = config.mamba_head_dim
    ngroups = config.mamba_num_groups

    # Compute nheads and d_inner
    if config.mamba_num_heads is not None:
        nheads = config.mamba_num_heads
        d_inner = nheads * headdim
    else:
        d_inner = wrapped_module.d_inner
        nheads = d_inner // headdim

    # Get tensor parallel size and compute local dimensions
    tp_size = wrapped_module.tp_size

    return {
        "d_inner_local_tp": d_inner // tp_size,
        "ngroups_local_tp": ngroups // tp_size,
        "d_state": d_state,
        "nheads_local_tp": nheads // tp_size,
    }


class AdapterWrapper(nn.Module):
    """Abstract base class for wrapping modules with adapters in Parameter-Efficient Fine-Tuning (PEFT).

    This class wraps a module and its associated adapter, providing methods for
    managing the state dictionaries of both the main module and the adapter. It does not
    implement the forward method, which must be implemented by concrete subclasses.

    Attributes:
        to_wrap (nn.Module): The main module to be wrapped.
        adapter (nn.Module): The adapter module to be applied.

    Note:
        This class is abstract and cannot be instantiated directly. Subclasses must
        implement the forward method.

    Example:
        class LoRALinear(AdapterWrapper):
            def __init__(self, to_wrap, adapter):
                super().__init__(to_wrap, adapter)

            def forward(self, x):
                return self.to_wrap(x) + self.adapter(x)

        main_module = nn.Linear(100, 100)
        adapter = nn.Linear(100, 100)
        parallel_adapter = LoRALinear(main_module, adapter)
    """

    def __init__(self, to_wrap: nn.Module, adapter: nn.Module, *, pg_collection=None) -> None:
        """Initialize the AdapterWrapper with a main module and adapter.

        Args:
            to_wrap: The main module to be wrapped.
            adapter: The adapter module to be applied.
        """
        super(AdapterWrapper, self).__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter
        self.pg_collection = pg_collection
        self._adapter_enabled = True
        self._base_returns_tuple = True

    def enable_adapter_layers(self) -> None:
        """Enable the adapter layers, allowing them to contribute to the forward pass output."""
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        """Disable the adapter layers, making the forward pass return only the base module output."""
        self._adapter_enabled = False

    @property
    def gather_output(self) -> bool:
        """Report whether the wrapped layer gathers its tensor-parallel output.

        ``GPTModel._postprocess`` and ``HybridModel`` read ``output_layer.gather_output``
        to describe the logits layout to the tensor observation hooks
        (NVIDIA/Megatron-LM#6648). Any wrapper placed in that slot must answer the way
        the wrapped module does. A wrapped layer without the attribute is not
        tensor-parallel, so its output is already replicated.
        """
        return getattr(self.to_wrap, "gather_output", True)

    def base_linear_forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Run the forward method of the linear module `to_wrap`.

        This method handles the complex return patterns of Megatron's linear layers,
        which can return different combinations of outputs, biases, and layernorm outputs.

        The flow is: x -> [layernorm/identity] -> layernorm_output -> [linear] -> linear_output, bias

        Args:
            x: Input tensor.
            *args: Additional positional arguments for the wrapped module.
            **kwargs: Additional keyword arguments for the wrapped module.

        Returns:
            A tuple containing:
                - linear_output: The output from the linear layer
                - bias: The bias term (if present, otherwise None)
                - layernorm_output: The output from layernorm (differs from x only for
                  LayerNormColumnParallelLinear, otherwise equals x)

        Note:
            The wrapped module can return values in four different patterns:
            1. nothing: (out, None)
            2. return_bias: (out, bias)
            3. return_layernorm_output: ((out, ln_out), None)
            4. both: (out, bias, ln_out)
        """
        linear_output = self.to_wrap(x, *args, **kwargs)

        bias = None
        layernorm_output = x

        # Plain nn.Linear (and similar modules) return a bare tensor rather than the
        # Megatron-style tuple. Handle that case without asserting, so the wrapper stays
        # a drop-in for nn.Linear in simple (non-parallel) models. Record the return
        # shape so ``forward`` implementations can mirror it (tensor vs. tuple).
        if isinstance(linear_output, torch.Tensor):
            self._base_returns_tuple = False
            bias = getattr(self.to_wrap, "bias", None)
            return linear_output, bias, layernorm_output

        assert isinstance(
            linear_output, tuple
        ), f"{self.to_wrap} should return a tensor or a tuple but instead returns {type(linear_output).__name__}"
        self._base_returns_tuple = True

        if len(linear_output) == 2:
            linear_output, bias = linear_output
            if isinstance(linear_output, tuple) and len(linear_output) == 2:
                linear_output, layernorm_output = linear_output
        elif len(linear_output) == 3:
            linear_output, bias, layernorm_output = linear_output

        return linear_output, bias, layernorm_output

    def adapter_forward(
        self, adapter: nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Run an adapter with the wrapped module's forwarded arguments."""

        parameter = next(adapter.parameters())
        return adapter(x.to(dtype=parameter.dtype), *args, **kwargs).to(dtype=x.dtype)

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve the state dictionary of the wrapped module and adapter.

        This method overrides the default state_dict behavior to include both
        the main module's state and the adapter's state under a special 'adapter' prefix.

        Args:
            destination: A dictionary to store the state. If None, a new
                        dictionary is created. Defaults to None.
            prefix: A prefix added to parameter and buffer names. Defaults to ''.
            keep_vars: If True, returns variables instead of tensor values.
                      Defaults to False.

        Returns:
            The state dictionary containing both the main module and adapter states.
        """
        if destination is None:
            destination = {}

        # Get state dict of the main module
        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Store adapter state dict under the "adapter" prefix in the destination dict
        self.adapter.state_dict(
            destination=destination, prefix=f"{prefix}adapter.", keep_vars=keep_vars
        )
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ShardedStateDict":
        """Retrieve the sharded state dictionary of the wrapped module and adapter.

        This method is used for distributed checkpointing, combining the sharded states
        of both the main module and the adapter.

        Args:
            prefix: A prefix added to parameter and buffer names. Defaults to ''.
            sharded_offsets: Offsets for sharded parameters. Defaults to an empty tuple.
            metadata: Additional metadata for the sharded state. Defaults to None.

        Returns:
            The combined sharded state dictionary.
        """
        adapter_sharded_state_dict_kwargs = {}
        if (
            isinstance(self.adapter, ParallelLinearAdapter)
            and "mixer.in_proj" in self.adapter.base_linear_name
        ):
            adapter_sharded_state_dict_kwargs["mamba_dim_info"] = _compute_mamba_dim_info(
                self.to_wrap
            )

        sharded_state_dict = {}
        # The wrapped module may be a plain nn.Linear (simple, non-parallel path) that
        # does not implement distributed sharding. Fall back to wrapping each parameter
        # as a replicated ShardedTensor so the trained weights are preserved.
        if hasattr(self.to_wrap, "sharded_state_dict"):
            sharded_state_dict.update(
                self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata)
            )
        else:
            sharded_state_dict.update(
                _plain_module_sharded_state_dict(
                    self.to_wrap, prefix, sharded_offsets, self.pg_collection
                )
            )
        # Likewise, a delta-only LinearAdapter has no sharded layout; only parallel adapters
        # carry distributed-sharding information. Fall back to replicated ShardedTensors for
        # the plain-adapter case so the trained LoRA matrices are not dropped.
        if hasattr(self.adapter, "sharded_state_dict"):
            sharded_state_dict.update(
                self.adapter.sharded_state_dict(
                    f"{prefix}adapter.",
                    sharded_offsets,
                    metadata,
                    **adapter_sharded_state_dict_kwargs,
                )
            )
        else:
            sharded_state_dict.update(
                _plain_module_sharded_state_dict(
                    self.adapter, f"{prefix}adapter.", sharded_offsets, self.pg_collection
                )
            )
        return sharded_state_dict

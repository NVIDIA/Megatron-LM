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
# c031e61d82dafbe4655a2e2d7e742b88f7e81c3a/src/megatron/bridge/peft/lora_layers.py

# Canonical projection wrappers: canonical_lora.py at the same revision.
"""Apply low-rank deltas without changing the base layer's output convention."""

from typing import Any, Optional, Tuple

import torch
from torch import nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict

from .merge import LoRAMerge
from .wrappers import AdapterWrapper


class LoRALinear(AdapterWrapper):
    """An adapter wrapper that adds the output of the adapter to the output of the wrapped module.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    @property
    def weight(self) -> torch.Tensor:
        """Return the effective base weight, including the LoRA delta when enabled."""
        base_weight = self.to_wrap.weight
        if not self._adapter_enabled:
            return base_weight

        linear_in_weight = self.adapter.linear_in.weight
        linear_out_weight = self.adapter.linear_out.weight

        merged_weight = LoRAMerge().merge(
            base_weight,
            linear_out_weight,
            linear_in_weight,
            self.adapter.alpha,
            self.adapter.dim,
            tp_group=getattr(self.adapter, "tp_group", None),
        )
        if merged_weight.shape != base_weight.shape:
            raise RuntimeError(
                "LoRA effective weight shape mismatch: "
                f"base={tuple(base_weight.shape)}, merged={tuple(merged_weight.shape)}"
            )
        return merged_weight.to(device=base_weight.device, dtype=base_weight.dtype)

    @property
    def bias(self) -> torch.Tensor | None:
        """Return the wrapped linear bias."""
        return getattr(self.to_wrap, "bias", None)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        """Forward pass that combines the wrapped module output with the adapter output.

        Args:
            x: Input tensor.
            *args: Additional positional arguments for the wrapped module.
            **kwargs: Additional keyword arguments for the wrapped module.

        Returns:
            When the wrapped module returns a Megatron-style tuple, a
            ``(combined_output, bias)`` tuple; when it returns a bare tensor
            (e.g. a plain ``nn.Linear``), a bare tensor so the wrapper stays a
            drop-in replacement in simple (non-parallel) models.
        """
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output if not self._base_returns_tuple else (linear_output, bias)
        # Serialize full-width branch buffers to keep long-context peak at two outputs.
        combined_output = linear_output.clone()
        del linear_output
        adapter_output = self.adapter_forward(
            self.adapter, layernorm_output.contiguous(), *args, **kwargs
        )
        adapter_output = adapter_output.reshape(combined_output.shape)
        combined_output.add_(adapter_output)
        if not self._base_returns_tuple:
            return combined_output
        return combined_output, bias


class ModuleDict(nn.ModuleDict):
    """
    nn.ModuleDict with a sharded_state_dict implementation for checkpointing
    """

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> "ShardedStateDict":
        """Retrieve the sharded state dictionary of the wrapped module and adapter.

        This method is used for distributed checkpointing, combining the sharded states
        of both the main module and the adapter.

        Args:
            prefix (str): A prefix added to parameter and buffer names. Defaults to ''.
            sharded_offsets (Tuple[Tuple[int, int, int]]): Offsets for sharded parameters.
                                                           Defaults to an empty tuple.
            metadata (Optional[dict]): Additional metadata for the sharded state.
                                       Defaults to None.

        Returns:
            ShardedStateDict: The combined sharded state dictionary.
        """
        sharded_state_dict = {}
        for key, layer in self.items():
            if layer is None:
                continue
            sharded_state_dict.update(
                layer.sharded_state_dict(f"{prefix}{key}.", sharded_offsets, metadata)
            )
        return sharded_state_dict


class LoRALinearSplitQKV(AdapterWrapper):
    """An adapter wrapper for `linear_qkv` where q, k, v are three separate adapters.
    This module that adds the output of the adapters to the output of the wrapped module while taking care of shape.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def _interleave_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Interleave QKV outputs to match Megatron's packed ordering."""

        config = self.to_wrap.config
        head_num = getattr(config, "num_attention_heads", None)
        num_query_groups = getattr(config, "num_query_groups", None)
        head_size = getattr(config, "kv_channels", None)

        if head_size is None:
            hidden_size = getattr(config, "hidden_size", None)
            if head_num is not None and hidden_size is not None:
                head_size = hidden_size // head_num
            elif num_query_groups:
                if key.size(-1) % num_query_groups != 0:
                    raise ValueError("Key projection size must be divisible by num_query_groups.")
                head_size = key.size(-1) // num_query_groups
            elif head_num is not None:
                if query.size(-1) % head_num != 0:
                    raise ValueError(
                        "Query projection size must be divisible by num_attention_heads."
                    )
                head_size = query.size(-1) // head_num
            else:
                raise ValueError(
                    "Cannot infer head size without kv_channels or hidden_size/num_attention_heads or num_query_groups."
                )

        if head_num is None:
            if query.size(-1) % head_size != 0:
                raise ValueError("Query projection size must be divisible by head_size.")
            head_num = query.size(-1) // head_size

        if not num_query_groups:
            if key.size(-1) % head_size != 0:
                raise ValueError("Key projection size must be divisible by head_size.")
            num_query_groups = key.size(-1) // head_size

        if head_num % num_query_groups != 0:
            raise ValueError("num_attention_heads must be divisible by num_query_groups.")

        # Adapter outputs carry only this TP rank's heads and query groups.
        head_num = query.size(-1) // head_size
        num_query_groups = key.size(-1) // head_size
        heads_per_group = head_num // num_query_groups

        leading_shape = query.shape[:-1]
        query = query.reshape(-1, head_num, head_size)
        key = key.reshape(-1, num_query_groups, head_size)
        value = value.reshape(-1, num_query_groups, head_size)

        qkv_chunks = []
        for i in range(num_query_groups):
            q_group = query[:, i * heads_per_group : (i + 1) * heads_per_group, :]
            k_group = key[:, i : i + 1, :]
            v_group = value[:, i : i + 1, :]
            qkv_chunks.extend([q_group, k_group, v_group])

        qkv = torch.cat(qkv_chunks, dim=1)
        return qkv.reshape(*leading_shape, -1)

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output, bias
        query = (
            self.adapter_forward(self.adapter.adapter_q, layernorm_output, *args, **kwargs)
            if self.adapter.adapter_q is not None
            else None
        )
        key = (
            self.adapter_forward(self.adapter.adapter_k, layernorm_output, *args, **kwargs)
            if self.adapter.adapter_k is not None
            else None
        )
        value = (
            self.adapter_forward(self.adapter.adapter_v, layernorm_output, *args, **kwargs)
            if self.adapter.adapter_v is not None
            else None
        )

        if key is None and value is None:
            kv_size = (linear_output.size(-1) - query.size(-1)) // 2
            key = value = query.new_zeros(*query.shape[:-1], kv_size)
        elif key is None:
            key = torch.zeros_like(value)
        elif value is None:
            value = torch.zeros_like(key)

        if query is None:
            query_size = linear_output.size(-1) - 2 * key.size(-1)
            query = key.new_zeros(*key.shape[:-1], query_size)

        adapter_output = self._interleave_qkv(query, key, value)

        return linear_output + adapter_output, bias


class LoRALinearSplitFC1UpGate(AdapterWrapper):
    """An adapter wrapper for `linear_fc1` where up_proj and gate_proj are two separate adapters.
    This module that adds the output of the adapters to the output of the wrapped module while taking care of shape.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output, bias
        adapter_output_gate = (
            self.adapter_forward(self.adapter.adapter_gate, layernorm_output, *args, **kwargs)
            if self.adapter.adapter_gate is not None
            else None
        )
        adapter_output_up = (
            self.adapter_forward(self.adapter.adapter_up, layernorm_output, *args, **kwargs)
            if self.adapter.adapter_up is not None
            else None
        )
        if adapter_output_gate is None:
            adapter_output_gate = torch.zeros_like(adapter_output_up)
        if adapter_output_up is None:
            adapter_output_up = torch.zeros_like(adapter_output_gate)
        adapter_output = torch.cat([adapter_output_gate, adapter_output_up], dim=-1)
        return linear_output + adapter_output, bias

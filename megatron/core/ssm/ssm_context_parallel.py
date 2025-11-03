# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional

import torch

from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)


class SSMContextParallel:
    """
    This class provides the following functionality related to SSM "all-to-all" context parallel:
    1. Error checking, and creation of, relevant parameters (e.g. nheads_local_tpcp)
    2. Collective operations on activations, on each context parallel rank, before and after the
       convolution and SSM
    3. A convolution operator that uses the correct slices of trainable variables on the current
       context parallel rank
    4. Sliced views of relevant trainable variables for the current context parallel rank
    """

    def __init__(self, cp_group: torch.distributed.ProcessGroup):
        self.cp_group = cp_group
        self.cp_size = cp_group.size()
        self.cp_rank = cp_group.rank()

    def get_parameter_local_cp(
        self, param: torch.Tensor, dim: int, split_size_or_sections: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Get the local parameter for the current context parallel rank."""

        # No need to split if CP size is 1.
        if self.cp_size == 1:
            return param

        # Split first if needed.
        if split_size_or_sections is not None:
            inputs = torch.split(param, split_size_or_sections, dim=dim)
            outputs = []
            for p in inputs:
                p = self.get_parameter_local_cp(p, dim)
                outputs.append(p)
            return torch.cat(outputs, dim=dim)

        # Slice the parameter.
        slices = [slice(None)] * param.dim()
        dim_size = param.size(dim=dim)
        slices[dim] = slice(
            self.cp_rank * dim_size // self.cp_size, (self.cp_rank + 1) * dim_size // self.cp_size
        )
        param = param[slices]
        return param

    def tensor_a2a_cp2hp(
        self,
        tensor: torch.Tensor,
        seq_dim: int,
        head_dim: int,
        split_size_or_sections: Optional[List[int]] = None,
        undo_attention_load_balancing: bool = True,
    ):
        """All-to-all context parallel to hidden parallel."""

        # No need to all-to-all if CP size is 1.
        if self.cp_size == 1:
            return tensor

        # Limitations of mamba_context_parallel._all_to_all_cp2hp.
        assert (
            seq_dim == 0
        ), f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
        assert (
            head_dim == -1 or head_dim == 2
        ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
        assert (
            tensor.dim() == 3
        ), f"tensor_a2a_cp2hp only supports 3-d input tensor for now, but got {tensor.dim()=}"

        # Split first if needed.
        if split_size_or_sections is not None:
            inputs = torch.split(tensor, split_size_or_sections, dim=head_dim)
            outputs = []
            for x in inputs:
                x = self.tensor_a2a_cp2hp(
                    x, seq_dim=seq_dim, head_dim=head_dim, undo_attention_load_balancing=False
                )
                outputs.append(x)
            tensor = torch.cat(outputs, dim=head_dim)
        else:
            tensor = _all_to_all_cp2hp(tensor, self.cp_group)

        # Undo attention load balancing last if needed.
        if undo_attention_load_balancing:
            tensor = _undo_attention_load_balancing(tensor, self.cp_size)
        return tensor

    def tensor_a2a_hp2cp(
        self,
        tensor: torch.Tensor,
        seq_dim: int,
        head_dim: int,
        split_size_or_sections: Optional[List[int]] = None,
        redo_attention_load_balancing: bool = True,
    ):
        """All-to-all hidden parallel to context parallel."""

        # No need to all-to-all if CP size is 1.
        if self.cp_size == 1:
            return tensor

        # Limitations of mamba_context_parallel._all_to_all_hp2cp.
        assert (
            seq_dim == 0
        ), f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
        assert (
            head_dim == -1 or head_dim == 2
        ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
        assert (
            tensor.dim() == 3
        ), f"tensor_a2a_cp2hp only supports 3-d input tensor for now, but got {tensor.dim()=}"

        # Redo attention load balancing first if needed.
        if redo_attention_load_balancing:
            tensor = _redo_attention_load_balancing(tensor, self.cp_size)

        # Split first if needed.
        if split_size_or_sections is not None:
            inputs = torch.split(tensor, split_size_or_sections, dim=head_dim)
            outputs = []
            for x in inputs:
                x = self.tensor_a2a_hp2cp(
                    x, seq_dim=seq_dim, head_dim=head_dim, redo_attention_load_balancing=False
                )
                outputs.append(x)
            tensor = torch.cat(outputs, dim=head_dim)
        else:
            tensor = _all_to_all_hp2cp(tensor, self.cp_group)

        return tensor

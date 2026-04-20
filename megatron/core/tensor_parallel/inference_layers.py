# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.inference.communication.torch_symm_triton import (
    are_tensors_nvls_eligible,
    fused_multimem_rs_add_norm_ag,
    multimem_all_gather,
    multimem_reduce_scatter,
)
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.inference.quantization.utils import mm_mxfp8
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.utils import get_tensor_model_parallel_group_if_none

try:
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.distributed import (
        gather_along_first_dim,
        reduce_scatter_along_first_dim,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def _te_rms_norm_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float):
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    out, _, _ = tex.rmsnorm_fwd(
        x, weight, eps, None, None, TE_DType[x.dtype], 16, False  # sm-margin  # zero centered gamma
    )
    out = out.view(*x_shape[:-1], -1)
    return out.to(x.dtype)


def _apply_linear(
    x: torch.Tensor,
    weight: Union[torch.Tensor, MXFP8Tensor],
    config: TransformerConfig,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Helper to apply either MXFP8 or standard GEMM based on the configuration.
    """
    kwargs = {"out": out} if out is not None else {}
    if isinstance(weight, MXFP8Tensor):
        return mm_mxfp8(x, weight, **kwargs)
    return torch.matmul(x, weight.t(), **kwargs)


class InferenceLinear(TELinear):
    """Inference optimized version of TELinear."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: Optional[str] = None,
        is_expert: bool = False,
        symmetric_ar_type: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
            symmetric_ar_type=symmetric_ar_type,
            tp_group=tp_group,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Forward pass."""
        if self.training:
            return super().forward(x)

        x = _apply_linear(x, self.weight, self.config)
        return x, None


class InferenceLayerNormColumnParallelLinear(TELayerNormColumnParallelLinear):
    """
    Inference optimized version of TELayerNormColumnParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        stride: int = 1,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            stride=stride,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_size = dist.get_world_size(self.tp_group)

        assert (
            output_size % self.tp_size == 0
        ), f"output_size ({output_size}) must be divisible by tp_size ({self.tp_size})"

        self.eps = config.layernorm_epsilon

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--transformer-impl=inference_optimized requires --sequence-parallel"

        self.triton_nvls_kernels_allowed = not config.inference_disable_triton_nvls_kernels

        # Boolean to be toggled externally for skipping norm and all-gather.
        # This is used when enabling fused reduce-scatter + add + rms-norm + all-gather
        # in tensor parallelism. In this case, the preceeding RowParallelLinear layer
        # has already applied the rms-norm and all-gather.
        self.skip_norm_and_all_gather = False

    def _maybe_allocate_symmetric_buffer(self, x: torch.Tensor):
        """
        Attempt to allocate symmetric memory buffer for all-gather.
        """
        symm_mem_buffer_dims = list(x.size())
        symm_mem_buffer_dims[0] *= self.tp_size
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=self.tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(symm_mem_buffer_dims, dtype=x.dtype)
        return symm_mem_buffer

    def _all_gather(self, x: torch.Tensor, symm_mem_buffer: dict) -> None:
        """
        Attempt an NVLS all-gather into symmetric memory. If not possible,
        revert to torch dist (NCCL) all-gather.
        """
        if self.tp_size == 1:
            return x

        # Check input only: if input is 16-byte divisible, the output
        # (world_size * input) is too.
        can_use_nvls = (
            self.triton_nvls_kernels_allowed
            and are_tensors_nvls_eligible(x)
            and symm_mem_buffer["handle"] is not None
        )
        if can_use_nvls:
            # do multimem all gather
            multimem_all_gather(symm_mem_buffer["tensor"], x, symm_mem_buffer["handle"])
            return symm_mem_buffer["tensor"]
        else:
            # revert to torch dist (NCCL) all gather
            x, _ = gather_along_first_dim(x, process_group=self.tp_group)
            return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass.
        """
        # Necessary conditions to ensure we are executing the fused rs-add-rmsnorm-ag
        # in the preceeding RowParallelLinear layer.
        # 1. skip_norm_and_all_gather is True
        # 2. tp_size > 1
        # 3. enough symmetric memory is available - if available it already has the output

        if self.training:
            return super().forward(x)

        if self.tp_size == 1:
            x = _te_rms_norm_kernel(x=x, weight=self.layer_norm_weight, eps=self.eps)
            x = _apply_linear(x, self.weight, self.config)
            return x, None

        symm_mem_buffer = self._maybe_allocate_symmetric_buffer(x)
        is_in_fused_mode = (
            self.skip_norm_and_all_gather
            and self.tp_size > 1
            and symm_mem_buffer["handle"] is not None
        )
        if is_in_fused_mode:
            x = symm_mem_buffer["tensor"]
        else:
            x = _te_rms_norm_kernel(x=x, weight=self.layer_norm_weight, eps=self.eps)
            x = self._all_gather(x, symm_mem_buffer)

        x = _apply_linear(x, self.weight, self.config)

        return x, None


class InferenceColumnParallelLinear(TEColumnParallelLinear):
    """
    Inference optimized version of TEColumnParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        stride: int = 1,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            stride=stride,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_size = dist.get_world_size(self.tp_group)

        assert (
            output_size % self.tp_size == 0
        ), f"output_size ({output_size}) must be divisible by tp_size ({self.tp_size})"

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--transformer-impl=inference_optimized requires --sequence-parallel"

        self.triton_nvls_kernels_allowed = not config.inference_disable_triton_nvls_kernels

    def _maybe_allocate_symmetric_buffer(self, x: torch.Tensor):
        """
        Attempt to allocate symmetric memory buffer for all-gather.
        """
        symm_mem_buffer_dims = list(x.size())
        symm_mem_buffer_dims[0] *= self.tp_size
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=self.tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(symm_mem_buffer_dims, dtype=x.dtype)
        return symm_mem_buffer

    def _all_gather(self, x: torch.Tensor, symm_mem_buffer: dict) -> None:
        """
        Attempt an NVLS all-gather into symmetric memory. If not possible,
        revert to torch dist (NCCL) all-gather.
        """
        if self.tp_size == 1:
            return x

        can_use_nvls = (
            self.triton_nvls_kernels_allowed
            and are_tensors_nvls_eligible(x)
            and symm_mem_buffer["handle"] is not None
        )
        if can_use_nvls:
            multimem_all_gather(symm_mem_buffer["tensor"], x, symm_mem_buffer["handle"])
            return symm_mem_buffer["tensor"]
        else:
            x, _ = gather_along_first_dim(x, process_group=self.tp_group)
            return x

    def _nvls_gather_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        """NVLS all-gather along last dim, with NCCL fallback."""
        ag_buffer_dims = list(x.size())
        ag_buffer_dims[0] *= self.tp_size
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=self.tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(ag_buffer_dims, dtype=x.dtype)

        can_use_nvls = (
            self.triton_nvls_kernels_allowed
            and are_tensors_nvls_eligible(x)
            and symm_mem_buffer["handle"] is not None
        )
        if can_use_nvls:
            multimem_all_gather(symm_mem_buffer["tensor"], x, symm_mem_buffer["handle"])
            tensor_list = symm_mem_buffer["tensor"].chunk(self.tp_size, dim=0)
            return torch.cat(tensor_list, dim=-1).contiguous()

        return gather_from_tensor_model_parallel_region(x, group=self.tp_group)

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass."""
        if self.training:
            return super().forward(x, weight=weight, runtime_gather_output=runtime_gather_output)

        if weight is None:
            weight = self.weight

        if self.tp_size == 1:
            x = _apply_linear(x, weight, self.config)
            return x, None

        if self.sequence_parallel:
            symm_mem_buffer = self._maybe_allocate_symmetric_buffer(x)
            x = self._all_gather(x, symm_mem_buffer)

        x = _apply_linear(x, weight, self.config)

        gather_output = self.gather_output
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output
        if gather_output:
            x = self._nvls_gather_last_dim(x)

        return x, None


class InferenceRowParallelLinear(TERowParallelLinear):
    """
    Inference optimized version of TERowParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_size = dist.get_world_size(self.tp_group)
        assert (
            input_size % self.tp_size == 0
        ), f"input_size ({input_size}) must be divisible by tp_size ({self.tp_size})"

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--transformer-impl=inference_optimized requires --sequence-parallel"

        self.triton_nvls_kernels_allowed = not getattr(
            config, 'inference_disable_triton_nvls_kernels', False
        )

        # Placeholder for next layer norm weights for fused
        # reduce-scatter + add + rms-norm + all-gather
        self.next_layer_norm_weights = None
        self.config = config

    def _matmul_reduce_scatter(self, x, residual=None):
        """
        Multiplies x by the weight matrix and performs a reduce-scatter.
        It will first try to write the matmul output to symmetric memory
        and perform an NVLS multicast reduce-scatter. If that is not possible,
        it will revert to torch.dist (NCCL) reduce-scatter.
        """
        use_mxfp8 = isinstance(self.weight, MXFP8Tensor)
        symm_mem_buffer_dims = list(x.size())
        if use_mxfp8:
            # Remove seq_len dimension for MXFP8 (mm_mxfp8 squeezes internally)
            del symm_mem_buffer_dims[1]
        symm_mem_buffer_dims[-1] = self.weight.size(0)
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=self.tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(symm_mem_buffer_dims, dtype=x.dtype)

        # RS requires bf16 (hardware multimem reduce is bf16-only).
        # Check the matmul output shape: if it is NVLS-eligible, the RS output
        # (world_size times smaller on dim 0) is too.
        can_use_nvls = (
            self.triton_nvls_kernels_allowed
            and x.dtype == torch.bfloat16
            and are_tensors_nvls_eligible(x)
            and symm_mem_buffer["handle"] is not None
        )

        if can_use_nvls:
            # Write output of matmul directly onto the symmetric memory buffer

            x = _apply_linear(x, self.weight, self.config, out=symm_mem_buffer["tensor"])

            # perform nvls reduce-scatter
            if self.next_layer_norm_weights is None:
                output_dims = list(x.size())
                output_dims[0] = x.size(0) // self.tp_size
                output = torch.empty(output_dims, dtype=x.dtype, device=x.device)
                multimem_reduce_scatter(output, x, symm_mem_buffer["handle"])
                return output
            else:
                assert hasattr(self, "residual"), (
                    "For fused reduce-scatter + add + rms-norm + all-gather, "
                    "residual must be set via _set_residual()"
                )
                residual = self.residual
                fused_multimem_rs_add_norm_ag(
                    residual,
                    symm_mem_buffer["tensor"],
                    symm_mem_buffer["handle"],
                    residual,
                    self.next_layer_norm_weights,
                    self.config.layernorm_epsilon,
                )
                # 1. Residual has the output of the reduce-scatter + residual add
                #    Care must be taken in the model definition, so as to not apply the
                #    residual again.
                # 2. The output of the full reduce-scatter + add + rms-norm + all-gather is
                #    written into symm_mem_buffer["tensor"] and will be accessible there.
                return residual
        else:
            # revert to torch dist (NCCL) reduce-scatter
            x = _apply_linear(x, self.weight, self.config)
            x, _ = reduce_scatter_along_first_dim(x, tp_group=self.tp_group)
        return x

    def _set_next_layer_norm_weights(self, weights: torch.Tensor):
        """
        Set next layer norm weights for fused reduce-scatter + add + rms-norm + all-gather.
        """
        self.next_layer_norm_weights = weights

    def _set_residual(self, residual: torch.Tensor):
        """
        Set residual for fused reduce-scatter + add + rms-norm + all-gather.
        """
        self.residual = residual

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, None]:
        """
        Forward pass.
        """
        if self.training:
            return super().forward(x)

        if self.tp_size == 1:
            x = _apply_linear(x, self.weight, self.config)
            return x, None
        else:
            x = self._matmul_reduce_scatter(x)
            return x, None


def inference_all_gather_last_dim(
    x: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
    config: TransformerConfig,
) -> torch.Tensor:
    """NVLS-optimized all-gather along the last dimension, with NCCL fallback.

    Replaces ``gather_from_tensor_model_parallel_region`` in inference paths
    where autograd is not needed and NVLS symmetric-memory is available.

    The NVLS path performs a flat all-gather into symmetric memory (concatenating
    along dim-0), then rearranges the result to the last dimension — the same
    semantics as ``_gather_along_last_dim`` but using hardware multicast when
    possible.
    """
    tp_size = dist.get_world_size(tp_group)
    if tp_size == 1:
        return x

    triton_nvls_kernels_allowed = not getattr(
        config, 'inference_disable_triton_nvls_kernels', False
    )

    if triton_nvls_kernels_allowed and SymmetricMemoryManager.is_initialized("tp"):
        ag_buffer_dims = list(x.size())
        ag_buffer_dims[0] *= tp_size
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(ag_buffer_dims, dtype=x.dtype)

        if are_tensors_nvls_eligible(x) and symm_mem_buffer["handle"] is not None:
            multimem_all_gather(symm_mem_buffer["tensor"], x, symm_mem_buffer["handle"])
            tensor_list = symm_mem_buffer["tensor"].chunk(tp_size, dim=0)
            return torch.cat(tensor_list, dim=-1).contiguous()

    return gather_from_tensor_model_parallel_region(x, group=tp_group)

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# === save_tensor 插桩 ===
from megatron.core.align_dump_utils import _mg_tensor_info, _mg_grad_info, save_tensor, save_tensor_grad, is_log_enabled as _is_log_enabled
_MLP_CALL_COUNT = [0]
_MLP_DEBUG_EXPERT = [False]  # 由外部 SequentialMLP 设置，True 时打印中间结果
# === save_tensor 插桩结束 ===

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


# pylint: disable=missing-class-docstring
@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: int = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError("MoE MLP requires `ffn_hidden_size`, but it was not provided.")
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = self.config.ffn_hidden_size

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
        )

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        _cur_layer = _MLP_CALL_COUNT[0]
        _MLP_CALL_COUNT[0] += 1
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        # === 插桩: CP12a mlp_fc1_out (只保存第一层 dense MLP) ===
        if _cur_layer == 0:
            _mg_tensor_info("cp12a_mlp_fc1_out", intermediate_parallel, layer_num=0, prefix="MG MLP")
        # === 插桩结束 ===
        # === DEBUG: expert0 内部中间值 ===
        if _MLP_DEBUG_EXPERT[0]:
            _mg_tensor_info("DEBUG_expert0_fc1_out", intermediate_parallel, None, prefix="DEBUG MG MLP")
            if intermediate_parallel.requires_grad:
                intermediate_parallel.register_hook(_mg_grad_info("expert0_fc1_out", layer_num=None, prefix="GRAD MG MoE"))
        # === DEBUG END ===

        nvtx_range_push(suffix="activation")
        if self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                    # === GRAD DEBUG: expert0 act_out (fused swiglu+probs) ===
                    if _MLP_DEBUG_EXPERT[0] and intermediate_parallel.requires_grad:
                        intermediate_parallel.register_hook(_mg_grad_info("expert0_act_out(after_weighted_swiglu)", layer_num=None, prefix="GRAD MG MoE"))
                    # === GRAD DEBUG END ===
                else:
                    raise ValueError("Only support fusion of swiglu with per_token_scale in MLP.")
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
                if _MLP_DEBUG_EXPERT[0] and not hasattr(MLP, '_path_logged'):
                    MLP._path_logged = True
                    print(f"[DEBUG MG MLP] BACKWARD PATH: unfused glu (autograd), NOT SwiGLUFunction")
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            # === DEBUG: expert0 纯 swiglu 输出（probs 乘之前）===
            if _MLP_DEBUG_EXPERT[0]:
                _mg_tensor_info("DEBUG_expert0_act_out", intermediate_parallel, None, prefix="DEBUG MG MLP")
            # === DEBUG END ===
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
                # === GRAD DEBUG: expert0 act_out (unfused: after probs * cast) ===
                if _MLP_DEBUG_EXPERT[0] and intermediate_parallel.requires_grad:
                    intermediate_parallel.register_hook(_mg_grad_info("expert0_act_out(after_probs_cast)", layer_num=None, prefix="GRAD MG MoE"))
                # === GRAD DEBUG END ===
        nvtx_range_pop(suffix="activation")

        # === 插桩: CP12b mlp_swiglu_out ===
        if _cur_layer == 0:
            _mg_tensor_info("cp12b_mlp_swiglu_out", intermediate_parallel, layer_num=0, prefix="MG MLP")
        # === 插桩结束 ===
        # === DEBUG: expert0 swiglu 后（含 per_token_scale）===
        if _MLP_DEBUG_EXPERT[0]:
            _mg_tensor_info("DEBUG_expert0_swiglu_out", intermediate_parallel, None, prefix="DEBUG MG MLP")
        # === DEBUG END ===

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")
        output, output_bias = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        # === 插桩: CP12c mlp_fc2_out ===
        if _cur_layer == 0:
            _mg_tensor_info("cp12c_mlp_fc2_out", output, layer_num=0, prefix="MG MLP")
        # === 插桩结束 ===
        # === DEBUG: expert0 fc2 输出 ===
        if _MLP_DEBUG_EXPERT[0]:
            _mg_tensor_info("DEBUG_expert0_fc2_out", output, None, prefix="DEBUG MG MLP")
            if output.requires_grad:
                output.register_hook(_mg_grad_info("expert0_fc2_out(inside_mlp)", layer_num=None, prefix="GRAD MG MoE"))
            _MLP_DEBUG_EXPERT[0] = False
        # === DEBUG END ===

        if per_token_scale is not None:
            assert output_bias is None, "Bias is not supported with per_token_scale"

        return output, output_bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == "linear_fc1":
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict


# pylint: disable=missing-function-docstring
def apply_swiglu_sharded_factory(original_sh_ten, sharded_offsets):
    # We must split the tensor into 2 parts, each sharded separately.
    # This requires a ShardedTensorFactory which `chunk`s during saving
    # and `cat`s during loading

    swiglu_shard_axis = 0
    prepend_axis_num = len(sharded_offsets)
    original_shape = original_sh_ten.local_shape
    original_numel = int(np.prod(original_shape))
    local_axis_size = original_shape[swiglu_shard_axis]
    assert (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] % local_axis_size == 0
    )
    rank_offset = (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
    )
    axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
        offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset + axis_frag, axis_frag * 2)
        if flattened_range is None:
            tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
            return [
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_w,
                    *sharded_offsets,
                    offset_w,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_v,
                    *sharded_offsets,
                    offset_v,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
            ]
        else:
            # Here we need to map a slice `t` (`flattened_range` specifies slice start and stop)
            # of the *original* flattened tensor into slices `w` and `v` of chunked
            # and flattened tensor.
            # Example:
            # If original tensor has (16, 5) shape and flattened_range is `slice(8, 64)`,
            # then `t` has shape `(56,)` and we need to create 2 tensors:
            # w: first 32 elements of `t` with flattened_range slice(8, 40)
            # v: last 24 elements of `t` with flattened_range slice(0, 24)
            # Global offsets are the same as in the non-flattened case
            assert t.ndim == 1, (key, t.shape)
            non_flat_local_shape = (original_shape[0] // 2, *original_shape[1:])
            chunk_numel = original_numel // 2
            result = []
            if flattened_range.start < chunk_numel:
                # Non-empty `w` chunk
                tensor_w = t[: chunk_numel - flattened_range.start]
                flattened_range_w = slice(
                    flattened_range.start, min(chunk_numel, flattened_range.stop)
                )
                assert len(tensor_w) == flattened_range_w.stop - flattened_range_w.start
                result.append(
                    ShardedTensor.from_rank_offsets_flat(
                        key,
                        tensor_w,
                        non_flat_local_shape,
                        *sharded_offsets,
                        offset_w,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                        flattened_range=flattened_range_w,
                    )
                )
            if flattened_range.stop > chunk_numel:
                # Non-empty `v` chunk
                tensor_v = t[-(flattened_range.stop - chunk_numel) :]
                flattened_range_v = slice(
                    max(chunk_numel, flattened_range.start) - chunk_numel,
                    flattened_range.stop - chunk_numel,
                )
                assert len(tensor_v) == flattened_range_v.stop - flattened_range_v.start, (
                    len(tensor_v),
                    flattened_range_v,
                )

                result.append(
                    ShardedTensor.from_rank_offsets_flat(
                        key,
                        tensor_v,
                        non_flat_local_shape,
                        *sharded_offsets,
                        offset_v,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                        flattened_range=flattened_range_v,
                    )
                )
            assert sum(sh_ten.data.numel() for sh_ten in result) == t.numel(), (result, t.shape)
            return result

    def sh_ten_merge_fn(sub_state_dict):
        with torch.no_grad():
            return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )

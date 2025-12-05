# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass(frozen=True)
class MoEModelConfig:
    seq_length: int
    micro_batch_size: int
    hidden_size: int
    moe_ffn_hidden_size: int
    num_experts: int
    router_topk: int
    num_attention_heads: int = 8
    moe_shared_expert_intermediate_size: Optional[int] = None

    # Router related
    moe_router_load_balancing_type: str = "aux_loss"
    moe_router_num_groups: Optional[int] = None
    moe_router_group_topk: Optional[int] = None
    moe_router_score_function: str = "softmax"
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = False


@dataclass(frozen=True)
class MoEPerformanceCase:
    """Describes a single MoE performance configuration to exercise."""

    name: str
    model: MoEModelConfig

    # Token dispatcher related
    token_dispatcher: str
    moe_flex_dispatcher_backend: str = "deepep"

    # FP8 related
    fp8: Optional[str] = None
    fp8_recipe: Optional[str] = None

    # Tested GPU platform
    gpu_platform: str = "H100"

    # Parallelism related
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    # kernel fusion related
    moe_permute_fusion: bool = True
    moe_router_fusion: bool = True

    # Performance stability related
    moe_router_force_load_balancing: bool = True
    manual_gc: bool = True

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def is_current_platform(self) -> bool:
        if self.gpu_platform is None:
            return True
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return self.gpu_platform.lower() in device_name.lower()


MIXTRAL_PROXY = MoEModelConfig(
    seq_length=4096,
    micro_batch_size=1,
    hidden_size=4096,
    moe_ffn_hidden_size=14336,
    num_experts=8,
    router_topk=2,
    moe_router_load_balancing_type="aux_loss",
)

DEEPSEEK_PROXY = MoEModelConfig(
    seq_length=4096,
    micro_batch_size=1,
    hidden_size=7168,
    moe_ffn_hidden_size=2048,
    num_experts=32,
    router_topk=8,
    moe_router_load_balancing_type="seq_aux_loss",
    moe_router_num_groups=8,
    moe_router_group_topk=4,
    moe_router_score_function="sigmoid",
    moe_router_dtype="fp32",
    moe_router_enable_expert_bias=True,
    moe_shared_expert_intermediate_size=2048,
)


PERFORMANCE_CASES: Iterable[MoEPerformanceCase] = (
    MoEPerformanceCase(
        name="mixtral_a2a_tp1ep8_fp8",
        token_dispatcher="alltoall",
        model=MIXTRAL_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
        fp8="e4m3",
        fp8_recipe="blockwise",
    ),
    MoEPerformanceCase(
        name="mixtral_deepep_tp1ep8_fp8",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="deepep",
        model=MIXTRAL_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
        fp8="e4m3",
        fp8_recipe="blockwise",
    ),
    MoEPerformanceCase(
        name="mixtral_hybridep_tp1ep8_fp8",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="hybridep",
        model=MIXTRAL_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
        fp8="e4m3",
        fp8_recipe="blockwise",
    ),
    MoEPerformanceCase(
        name="deepseek_a2a_tp1ep8_fp8",
        token_dispatcher="alltoall",
        model=DEEPSEEK_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
        fp8="e4m3",
        fp8_recipe="blockwise",
    ),
    MoEPerformanceCase(
        name="deepseek_hybridep_tp1ep8_fp8",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="hybridep",
        model=DEEPSEEK_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
        fp8="e4m3",
        fp8_recipe="blockwise",
    ),
    MoEPerformanceCase(
        name="deepseek_deepep_tp1ep8_fp8",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="deepep",
        model=DEEPSEEK_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
        fp8="e4m3",
        fp8_recipe="blockwise",
    ),
    MoEPerformanceCase(
        name="mixtral_a2a_tp1ep8_bf16",
        token_dispatcher="alltoall",
        model=MIXTRAL_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
    ),
    MoEPerformanceCase(
        name="mixtral_deepep_tp1ep8_bf16",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="deepep",
        model=MIXTRAL_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
    ),
    MoEPerformanceCase(
        name="mixtral_hybridep_tp1ep8_bf16",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="hybridep",
        model=MIXTRAL_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
    ),
    MoEPerformanceCase(
        name="deepseek_a2a_tp1ep8_bf16",
        token_dispatcher="alltoall",
        model=DEEPSEEK_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
    ),
    MoEPerformanceCase(
        name="deepseek_deepep_tp1ep8_bf16",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="deepep",
        model=DEEPSEEK_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
    ),
    MoEPerformanceCase(
        name="deepseek_hybridep_tp1ep8_bf16",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="hybridep",
        model=DEEPSEEK_PROXY,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
    ),
)

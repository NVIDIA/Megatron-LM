#!/usr/bin/env python3
"""
BF16算子模块
提供带有tensor保存功能的BF16矩阵乘法算子
"""

import torch
from torch.autograd import Function
from typing import Optional, Dict, Any


class BF16MatMul(Function):
    """BF16矩阵乘法算子，集成tensor保存功能"""
    
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor,
                layer_type: Optional[str] = None, layer_idx: Optional[int] = None,
                operation: str = "forward", phase: str = "pre", component: str = "linear",
                rank: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        BF16矩阵乘法前向传播
        
        Args:
            A: 输入tensor A (BF16)
            B: 输入tensor B (BF16)
            layer_type: 层类型 ("attention", "linear", etc.)
            layer_idx: 层索引
            operation: 操作类型 ("forward", "backward")
            phase: 阶段 ("pre", "post")
            component: 组件类型 ("linear", "FA", etc.)
            rank: GPU rank信息
            metadata: 额外的元数据
        """
        # 保存tensor和参数到ctx
        ctx.save_for_backward(A, B)
        ctx.layer_type = layer_type
        ctx.layer_idx = layer_idx
        ctx.operation = operation
        ctx.phase = phase
        ctx.component = component
        ctx.rank = rank
        ctx.metadata = metadata
        
        # 确保tensor是BF16格式
        if A.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
        if B.dtype != torch.bfloat16:
            B = B.to(torch.bfloat16)
        
        # 执行矩阵乘法
        output = torch.matmul(A, B)
        
        # 自动保存forward阶段的tensor
        if layer_type is not None:
            try:
                from megatron.core.tensor_saver import save_tensor
                
                # 保存输入tensor A
                save_tensor(
                    tensor=A,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name="input_A",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存输入tensor B
                save_tensor(
                    tensor=B,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name="input_B",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存输出tensor
                save_tensor(
                    tensor=output,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name="output",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                print(f"[BF16MatMul] 保存tensor时出错: {e}")
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        BF16矩阵乘法反向传播
        """
        A, B = ctx.saved_tensors
        grad_A = grad_B = None
        
        # 计算梯度
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output, B.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A.transpose(-2, -1), grad_output)
        
        # 自动保存backward阶段的tensor
        if ctx.layer_type is not None:
            try:
                from megatron.core.tensor_saver import save_tensor
                
                # 保存梯度输出
                save_tensor(
                    tensor=grad_output,
                    layer_type=ctx.layer_type,
                    operation="backward",
                    quant_type="bf16",
                    tensor_name="grad_output",
                    layer_idx=ctx.layer_idx,
                    phase="post",
                    component=ctx.component,
                    rank=ctx.rank,
                    metadata=ctx.metadata
                )
                
                # 保存梯度A
                if grad_A is not None:
                    save_tensor(
                        tensor=grad_A,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name="grad_input_A",
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx.metadata
                    )
                
                # 保存梯度B
                if grad_B is not None:
                    save_tensor(
                        tensor=grad_B,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name="grad_input_B",
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx.metadata
                    )
                    
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                print(f"[BF16MatMul] 保存backward tensor时出错: {e}")
        
        return grad_A, grad_B, None, None, None, None, None, None  # None对应所有额外参数


class BF16Linear(Function):
    """BF16线性层算子，集成tensor保存功能"""
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                layer_type: Optional[str] = None, layer_idx: Optional[int] = None,
                operation: str = "forward", phase: str = "pre", component: str = "linear",
                rank: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        BF16线性层前向传播
        
        Args:
            input_tensor: 输入tensor (BF16)
            weight: 权重tensor (BF16)
            bias: 偏置tensor (BF16, 可选)
            layer_type: 层类型
            layer_idx: 层索引
            operation: 操作类型
            phase: 阶段
            component: 组件类型
            rank: GPU rank信息
            metadata: 额外的元数据
        """
        # 保存tensor和参数到ctx
        ctx.save_for_backward(input_tensor, weight, bias)
        ctx.has_bias = bias is not None
        ctx.layer_type = layer_type
        ctx.layer_idx = layer_idx
        ctx.operation = operation
        ctx.phase = phase
        ctx.component = component
        ctx.rank = rank
        ctx.metadata = metadata
        
        # 确保tensor是BF16格式
        if input_tensor.dtype != torch.bfloat16:
            input_tensor = input_tensor.to(torch.bfloat16)
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)
        if bias is not None and bias.dtype != torch.bfloat16:
            bias = bias.to(torch.bfloat16)
        
        # 执行线性变换
        output = torch.nn.functional.linear(input_tensor, weight, bias)
        
        # 自动保存forward阶段的tensor
        if layer_type is not None:
            try:
                from megatron.core.tensor_saver import save_tensor
                
                # 保存输入tensor
                save_tensor(
                    tensor=input_tensor,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name="input",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存权重tensor
                save_tensor(
                    tensor=weight,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name="weight",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存偏置tensor（如果存在）
                if bias is not None:
                    save_tensor(
                        tensor=bias,
                        layer_type=layer_type,
                        operation=operation,
                        quant_type="bf16",
                        tensor_name="bias",
                        layer_idx=layer_idx,
                        phase=phase,
                        component=component,
                        rank=rank,
                        metadata=metadata
                    )
                
                # 保存输出tensor
                save_tensor(
                    tensor=output,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name="output",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                print(f"[BF16Linear] 保存tensor时出错: {e}")
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        BF16线性层反向传播
        """
        input_tensor, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        # 计算梯度
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.linear(grad_output, weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.t(), input_tensor).t()
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        # 自动保存backward阶段的tensor
        if ctx.layer_type is not None:
            try:
                from megatron.core.tensor_saver import save_tensor
                
                # 保存梯度输出
                save_tensor(
                    tensor=grad_output,
                    layer_type=ctx.layer_type,
                    operation="backward",
                    quant_type="bf16",
                    tensor_name="grad_output",
                    layer_idx=ctx.layer_idx,
                    phase="post",
                    component=ctx.component,
                    rank=ctx.rank,
                    metadata=ctx.metadata
                )
                
                # 保存输入梯度
                if grad_input is not None:
                    save_tensor(
                        tensor=grad_input,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name="grad_input",
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx.metadata
                    )
                
                # 保存权重梯度
                if grad_weight is not None:
                    save_tensor(
                        tensor=grad_weight,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name="grad_weight",
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx.metadata
                    )
                
                # 保存偏置梯度
                if grad_bias is not None:
                    save_tensor(
                        tensor=grad_bias,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name="grad_bias",
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx.metadata
                    )
                    
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                print(f"[BF16Linear] 保存backward tensor时出错: {e}")
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None  # None对应所有额外参数


# 便捷函数
def bf16_matmul(A: torch.Tensor, B: torch.Tensor, **tensor_save_kwargs) -> torch.Tensor:
    """
    BF16矩阵乘法便捷函数，支持tensor保存
    
    Args:
        A, B: 输入tensor
        **tensor_save_kwargs: tensor保存相关参数
            - layer_type: 层类型
            - layer_idx: 层索引
            - operation: 操作类型
            - phase: 阶段
            - component: 组件类型
            - rank: GPU rank
            - metadata: 元数据
    """
    # 如果有tensor保存参数，使用集成算子
    if tensor_save_kwargs and any(key in tensor_save_kwargs for key in 
                                 ['layer_type', 'layer_idx', 'operation', 'phase', 'component', 'rank', 'metadata']):
        return BF16MatMul.apply(
            A, B,
            tensor_save_kwargs.get('layer_type'),
            tensor_save_kwargs.get('layer_idx'),
            tensor_save_kwargs.get('operation', 'forward'),
            tensor_save_kwargs.get('phase', 'pre'),
            tensor_save_kwargs.get('component', 'linear'),
            tensor_save_kwargs.get('rank'),
            tensor_save_kwargs.get('metadata')
        )
    else:
        # 否则使用原始调用方式
        return BF16MatMul.apply(A, B)


def bf16_linear(input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **tensor_save_kwargs) -> torch.Tensor:
    """
    BF16线性层便捷函数，支持tensor保存
    
    Args:
        input_tensor: 输入tensor
        weight: 权重tensor
        bias: 偏置tensor (可选)
        **tensor_save_kwargs: tensor保存相关参数
            - layer_type: 层类型
            - layer_idx: 层索引
            - operation: 操作类型
            - phase: 阶段
            - component: 组件类型
            - rank: GPU rank
            - metadata: 元数据
    """
    # 如果有tensor保存参数，使用集成算子
    if tensor_save_kwargs and any(key in tensor_save_kwargs for key in 
                                 ['layer_type', 'layer_idx', 'operation', 'phase', 'component', 'rank', 'metadata']):
        return BF16Linear.apply(
            input_tensor, weight, bias,
            tensor_save_kwargs.get('layer_type'),
            tensor_save_kwargs.get('layer_idx'),
            tensor_save_kwargs.get('operation', 'forward'),
            tensor_save_kwargs.get('phase', 'pre'),
            tensor_save_kwargs.get('component', 'linear'),
            tensor_save_kwargs.get('rank'),
            tensor_save_kwargs.get('metadata')
        )
    else:
        # 否则使用原始调用方式
        return BF16Linear.apply(input_tensor, weight, bias)

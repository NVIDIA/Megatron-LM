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
        # 使用私有属性名保存metadata，避免属性冲突
        ctx._metadata = metadata
        
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
                
                # 根据component类型确定tensor名称
                if component == "FA" or component == "attention":
                    # attention操作：A是attention_probs，B是value
                    tensor_name_A = "attention_probs"
                    tensor_name_B = "value"
                else:
                    # linear操作：使用通用名称
                    tensor_name_A = "input_A"
                    tensor_name_B = "input_B"
                
                # 保存输入tensor A
                save_tensor(
                    tensor=A,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name=tensor_name_A,
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
                    tensor_name=tensor_name_B,
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
                    metadata=ctx._metadata
                )
                
                # 根据component类型确定backward tensor名称
                if ctx.component == "FA" or ctx.component == "attention":
                    # attention操作：grad_A是grad_attention_probs，grad_B是grad_value
                    grad_tensor_name_A = "grad_attention_probs"
                    grad_tensor_name_B = "grad_value"
                else:
                    # linear操作：使用通用名称
                    grad_tensor_name_A = "grad_input_A"
                    grad_tensor_name_B = "grad_input_B"
                
                # 保存梯度A
                if grad_A is not None:
                    save_tensor(
                        tensor=grad_A,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name=grad_tensor_name_A,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                
                # 保存梯度B
                if grad_B is not None:
                    save_tensor(
                        tensor=grad_B,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name=grad_tensor_name_B,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                    
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                print(f"[BF16MatMul] 保存backward tensor时出错: {e}")
        
        return grad_A, grad_B, None, None, None, None, None, None, None  # None对应所有额外参数（9个）


class BF16BAddBmm(Function):
    """BF16 Batch Add Batch Matrix Multiplication算子，集成tensor保存功能"""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor,
                beta: float = 1.0, alpha: float = 1.0,
                layer_type: Optional[str] = None, layer_idx: Optional[int] = None,
                operation: str = "forward", phase: str = "pre", component: str = "attention",
                rank: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        BF16 Batch Add Batch Matrix Multiplication前向传播
        
        Args:
            input: 输入tensor
            batch1: 第一个batch tensor
            batch2: 第二个batch tensor
            beta: beta参数
            alpha: alpha参数
            layer_type: 层类型 ("attention", "linear", etc.)
            layer_idx: 层索引
            operation: 操作类型 ("forward", "backward")
            phase: 阶段 ("pre", "post")
            component: 组件类型 ("attention", "linear", etc.)
            rank: GPU rank信息
            metadata: 额外的元数据
        """
        # 保存tensor和参数到ctx
        ctx.save_for_backward(input, batch1, batch2)
        ctx.beta = beta
        ctx.alpha = alpha
        ctx.layer_type = layer_type
        ctx.layer_idx = layer_idx
        ctx.operation = operation
        ctx.phase = phase
        ctx.component = component
        ctx.rank = rank
        ctx._metadata = metadata
        
        # 确保tensor是BF16格式
        if input.dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)
        if batch1.dtype != torch.bfloat16:
            batch1 = batch1.to(torch.bfloat16)
        if batch2.dtype != torch.bfloat16:
            batch2 = batch2.to(torch.bfloat16)
        
        # 执行batch matrix multiplication
        mm_out = torch.bmm(batch1, batch2)
        output = beta * input + alpha * mm_out
        
        # 自动保存forward阶段的tensor
        if layer_type is not None:
            try:
                from megatron.core.tensor_saver import save_tensor
                
                # 根据component类型确定tensor名称
                if component == "FA" or component == "attention":
                    # attention操作：input是matmul_input_buffer，batch1是query，batch2是key
                    tensor_name_input = "matmul_input_buffer"
                    tensor_name_batch1 = "query"
                    tensor_name_batch2 = "key"
                else:
                    # 其他操作：使用通用名称
                    tensor_name_input = "input"
                    tensor_name_batch1 = "batch1"
                    tensor_name_batch2 = "batch2"
                
                # 保存输入tensor
                save_tensor(
                    tensor=input,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name=tensor_name_input,
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存batch1 tensor
                save_tensor(
                    tensor=batch1,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name=tensor_name_batch1,
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存batch2 tensor
                save_tensor(
                    tensor=batch2,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name=tensor_name_batch2,
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存矩阵乘法结果
                save_tensor(
                    tensor=mm_out,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="bf16",
                    tensor_name="mm_output",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存最终输出
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
                print(f"[BF16BAddBmm] 保存tensor时出错: {e}")
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, batch1, batch2 = ctx.saved_tensors
        beta, alpha = ctx.beta, ctx.alpha
        
        grad_input = grad_batch1 = grad_batch2 = None
        
        # 计算梯度
        if ctx.needs_input_grad[0]:
            grad_input = beta * grad_output
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            mm_grad = alpha * grad_output
            grad_batch1 = torch.bmm(mm_grad, batch2.transpose(-2, -1))
            grad_batch2 = torch.bmm(batch1.transpose(-2, -1), mm_grad)
        
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
                    metadata=ctx._metadata
                )
                
                # 根据component类型确定backward tensor名称
                if ctx.component == "FA" or ctx.component == "attention":
                    # attention操作：grad_input是grad_matmul_input_buffer，grad_batch1是grad_query，grad_batch2是grad_key
                    grad_tensor_name_input = "grad_matmul_input_buffer"
                    grad_tensor_name_batch1 = "grad_query"
                    grad_tensor_name_batch2 = "grad_key"
                else:
                    # 其他操作：使用通用名称
                    grad_tensor_name_input = "grad_input"
                    grad_tensor_name_batch1 = "grad_batch1"
                    grad_tensor_name_batch2 = "grad_batch2"
                
                # 保存梯度input
                if grad_input is not None:
                    save_tensor(
                        tensor=grad_input,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name=grad_tensor_name_input,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                
                # 保存梯度batch1
                if grad_batch1 is not None:
                    save_tensor(
                        tensor=grad_batch1,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name=grad_tensor_name_batch1,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                
                # 保存梯度batch2
                if grad_batch2 is not None:
                    save_tensor(
                        tensor=grad_batch2,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="bf16",
                        tensor_name=grad_tensor_name_batch2,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                    
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                print(f"[BF16BAddBmm] 保存backward tensor时出错: {e}")
        
        return grad_input, grad_batch1, grad_batch2, None, None, None, None, None, None, None, None, None  # None对应所有额外参数（12个）




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


def bf16_baddbmm(input: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor, 
                 beta: float = 1.0, alpha: float = 1.0, **tensor_save_kwargs) -> torch.Tensor:
    """
    BF16 Batch Add Batch Matrix Multiplication便捷函数，支持tensor保存
    
    Args:
        input: 输入tensor
        batch1: 第一个batch tensor
        batch2: 第二个batch tensor
        beta: beta参数
        alpha: alpha参数
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
        return BF16BAddBmm.apply(
            input, batch1, batch2, beta, alpha,
            tensor_save_kwargs.get('layer_type'),
            tensor_save_kwargs.get('layer_idx'),
            tensor_save_kwargs.get('operation', 'forward'),
            tensor_save_kwargs.get('phase', 'pre'),
            tensor_save_kwargs.get('component', 'attention'),
            tensor_save_kwargs.get('rank'),
            tensor_save_kwargs.get('metadata')
        )
    else:
        # 否则使用原始调用方式
        return BF16BAddBmm.apply(input, batch1, batch2, beta, alpha)



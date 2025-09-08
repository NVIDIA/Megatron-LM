#!/usr/bin/env python3
"""
Tensor保存工具模块
用于保存attention和linear层的forward/backward输入tensor
"""

import os
import torch
import time
from typing import Optional, Dict, Any
from pathlib import Path


class TensorSaver:
    """Tensor保存器，用于保存量化前后的tensor数据"""
    
    def __init__(self, save_dir: str = "./tensor_logs", enabled: bool = True):
        """
        初始化Tensor保存器
        
        Args:
            save_dir: 保存目录
            enabled: 是否启用保存功能
        """
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.tensor_counter = 0
        
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[TensorSaver] 初始化完成，保存目录: {self.save_dir}")
    
    def _get_tensor_info(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """获取tensor的基本信息"""
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "min": float(tensor.min().item()) if tensor.numel() > 0 else 0.0,
            "max": float(tensor.max().item()) if tensor.numel() > 0 else 0.0,
            "mean": float(tensor.mean().item()) if tensor.numel() > 0 else 0.0,
            "std": float(tensor.std().item()) if tensor.numel() > 0 else 0.0,
        }
    
    def _generate_filename(self, 
                          layer_type: str,
                          operation: str,
                          quant_type: str, 
                          tensor_name: str,
                          layer_idx: Optional[int] = None,
                          phase: str = "unknown",
                          component: str = "unknown") -> str:
        """生成文件名"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.tensor_counter += 1
        
        if layer_idx is not None:
            filename = f"{timestamp}_{self.tensor_counter:04d}_{layer_type}_L{layer_idx}_{operation}_{phase}_{component}_{quant_type}_{tensor_name}.pt"
        else:
            filename = f"{timestamp}_{self.tensor_counter:04d}_{layer_type}_{operation}_{phase}_{component}_{quant_type}_{tensor_name}.pt"
        
        return filename
    
    def save_tensor(self, 
                   tensor: torch.Tensor,
                   layer_type: str,
                   operation: str,  # "forward" or "backward"
                   quant_type: str,
                   tensor_name: str,
                   layer_idx: Optional[int] = None,
                   phase: str = "unknown",  # "pre" or "post" for forward/backward phases
                   component: str = "unknown",  # "linear" or "FA" for component type
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        保存tensor到文件
        
        Args:
            tensor: 要保存的tensor
            layer_type: 层类型 ("attention" or "linear")
            operation: 操作类型 ("forward" or "backward")
            quant_type: 量化类型 ("hifp8", "mxfp8", "mxfp4", "bf16", etc.)
            tensor_name: tensor名称 ("input", "output", "grad_input", etc.)
            layer_idx: 层索引
            phase: 阶段 ("pre" or "post" for forward/backward phases)
            component: 组件类型 ("linear" or "FA" for component type)
            metadata: 额外的元数据
            
        Returns:
            保存的文件路径，如果未启用则返回None
        """
        if not self.enabled:
            return None
        
        try:
            # 生成文件名
            filename = self._generate_filename(layer_type, operation, quant_type, tensor_name, layer_idx, phase, component)
            filepath = self.save_dir / filename
            
            # 准备保存数据
            save_data = {
                "tensor": tensor.detach().cpu(),  # 移动到CPU并分离梯度
                "tensor_info": self._get_tensor_info(tensor),
                "metadata": {
                    "layer_type": layer_type,
                    "operation": operation,
                    "quant_type": quant_type,
                    "tensor_name": tensor_name,
                    "layer_idx": layer_idx,
                    "phase": phase,
                    "component": component,
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **(metadata or {})
                }
            }
            
            # 保存到文件
            torch.save(save_data, filepath)
            
            print(f"[TensorSaver] 已保存: {filename} "
                  f"(shape={tensor.shape}, dtype={tensor.dtype}, "
                  f"range=[{save_data['tensor_info']['min']:.4f}, {save_data['tensor_info']['max']:.4f}])")
            
            return str(filepath)
            
        except Exception as e:
            print(f"[TensorSaver] 保存tensor失败: {e}")
            return None
    
    def save_attention_tensors(self,
                              query: torch.Tensor,
                              key: torch.Tensor, 
                              value: torch.Tensor,
                              quant_type: str,
                              operation: str = "forward",
                              layer_idx: Optional[int] = None,
                              phase: str = "pre",
                              component: str = "FA",
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
        """
        保存attention层的输入tensor
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            quant_type: 量化类型
            operation: 操作类型
            layer_idx: 层索引
            metadata: 额外元数据
            
        Returns:
            保存的文件路径字典
        """
        results = {}
        
        # 保存query tensor
        if query is not None:
            results["query"] = self.save_tensor(
                query, "attention", operation, quant_type, "query", layer_idx, phase, component, metadata
            )
        
        # 保存key tensor
        if key is not None:
            results["key"] = self.save_tensor(
                key, "attention", operation, quant_type, "key", layer_idx, phase, component, metadata
            )
        
        # 保存value tensor
        if value is not None:
            results["value"] = self.save_tensor(
                value, "attention", operation, quant_type, "value", layer_idx, phase, component, metadata
            )
        
        return results
    
    def save_linear_tensors(self,
                           input_tensor: torch.Tensor,
                           weight: torch.Tensor,
                           quant_type: str,
                           operation: str = "forward",
                           layer_idx: Optional[int] = None,
                           phase: str = "pre",
                           component: str = "linear",
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
        """
        保存linear层的输入tensor
        
        Args:
            input_tensor: 输入tensor
            weight: 权重tensor
            quant_type: 量化类型
            operation: 操作类型
            layer_idx: 层索引
            metadata: 额外元数据
            
        Returns:
            保存的文件路径字典
        """
        results = {}
        
        # 保存input tensor
        if input_tensor is not None:
            results["input"] = self.save_tensor(
                input_tensor, "linear", operation, quant_type, "input", layer_idx, phase, component, metadata
            )
        
        # 保存weight tensor
        if weight is not None:
            results["weight"] = self.save_tensor(
                weight, "linear", operation, quant_type, "weight", layer_idx, phase, component, metadata
            )
        
        return results


# 全局tensor保存器实例
_global_tensor_saver = None


def get_tensor_saver() -> TensorSaver:
    """获取全局tensor保存器实例"""
    global _global_tensor_saver
    if _global_tensor_saver is None:
        # 从环境变量获取配置
        save_dir = os.environ.get("TENSOR_SAVE_DIR", "./tensor_logs")
        enabled = os.environ.get("TENSOR_SAVE_ENABLED", "true").lower() == "true"
        _global_tensor_saver = TensorSaver(save_dir=save_dir, enabled=enabled)
    return _global_tensor_saver


def save_attention_tensors(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          quant_type: str,
                          operation: str = "forward",
                          layer_idx: Optional[int] = None,
                          phase: str = "pre",
                          component: str = "FA",
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """保存attention层tensor的便捷函数"""
    saver = get_tensor_saver()
    return saver.save_attention_tensors(query, key, value, quant_type, operation, layer_idx, phase, component, metadata)


def save_linear_tensors(input_tensor: torch.Tensor,
                       weight: torch.Tensor,
                       quant_type: str,
                       operation: str = "forward",
                       layer_idx: Optional[int] = None,
                       phase: str = "pre",
                       component: str = "linear",
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """保存linear层tensor的便捷函数"""
    saver = get_tensor_saver()
    return saver.save_linear_tensors(input_tensor, weight, quant_type, operation, layer_idx, phase, component, metadata)


def save_tensor(tensor: torch.Tensor,
                layer_type: str,
                operation: str,
                quant_type: str,
                tensor_name: str,
                layer_idx: Optional[int] = None,
                phase: str = "unknown",
                component: str = "unknown",
                metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """保存单个tensor的便捷函数"""
    saver = get_tensor_saver()
    return saver.save_tensor(tensor, layer_type, operation, quant_type, tensor_name, layer_idx, phase, component, metadata)

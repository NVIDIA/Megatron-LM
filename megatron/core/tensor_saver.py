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


# 全局状态管理
class TensorCollectionState:
    """Tensor收集状态管理器"""
    def __init__(self):
        self.current_rank = None
        self.current_sample_idx = None
        self.current_iteration = 0
        self.batch_idx = 0
        self.sequence_idx = 0
    
    def set_rank(self, rank: int):
        """设置当前rank"""
        self.current_rank = rank
        print(f"[TensorCollectionState] 设置rank: {rank}")
    
    def set_sample_idx(self, sample_idx: int):
        """设置当前sample索引"""
        self.current_sample_idx = sample_idx
        print(f"[TensorCollectionState] 设置sample_idx: {sample_idx}")
    
    def set_iteration(self, iteration: int):
        """设置当前iteration"""
        self.current_iteration = iteration
        print(f"[TensorCollectionState] 设置iteration: {iteration}")
    
    def set_batch_idx(self, batch_idx: int):
        """设置当前batch索引"""
        self.batch_idx = batch_idx
        print(f"[TensorCollectionState] 设置batch_idx: {batch_idx}")
    
    def set_sequence_idx(self, sequence_idx: int):
        """设置当前sequence索引"""
        self.sequence_idx = sequence_idx
        print(f"[TensorCollectionState] 设置sequence_idx: {sequence_idx}")
    
    def get_rank(self) -> Optional[int]:
        """获取当前rank"""
        if self.current_rank is not None:
            return self.current_rank
        
        # 尝试从分布式环境获取
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                self.current_rank = rank
                return rank
        except:
            pass
        
        # 尝试从环境变量获取
        rank_env = os.environ.get("LOCAL_RANK")
        if rank_env is not None:
            try:
                rank = int(rank_env)
                self.current_rank = rank
                return rank
            except ValueError:
                pass
        
        return None
    
    def get_sample_idx(self) -> Optional[int]:
        """获取当前sample索引"""
        if self.current_sample_idx is not None:
            return self.current_sample_idx
        
        # 尝试从环境变量获取
        sample_env = os.environ.get("CURRENT_SAMPLE_IDX")
        if sample_env is not None:
            try:
                sample_idx = int(sample_env)
                self.current_sample_idx = sample_idx
                return sample_idx
            except ValueError:
                pass
        
        return None
    
    def get_iteration(self) -> int:
        """获取当前iteration"""
        return self.current_iteration
    
    def get_batch_idx(self) -> int:
        """获取当前batch索引"""
        return self.batch_idx
    
    def get_sequence_idx(self) -> int:
        """获取当前sequence索引"""
        return self.sequence_idx

# 全局状态实例
_global_tensor_state = TensorCollectionState()

# 全局tensor索引管理器
class TensorIndexManager:
    """Tensor索引管理器，确保同一层的不同tensor使用相同的索引"""
    def __init__(self):
        self.layer_tensor_counters = {}  # {layer_key: counter}
        self.current_layer_key = None
        self.current_tensor_group = None
    
    def get_layer_key(self, layer_type: str, layer_idx: Optional[int], operation: str) -> str:
        """生成层标识键"""
        if layer_idx is not None:
            return f"{layer_type}_L{layer_idx}_{operation}"
        else:
            return f"{layer_type}_unknown_{operation}"
    
    def get_tensor_group_key(self, layer_type: str, layer_idx: Optional[int], operation: str) -> str:
        """生成tensor组标识键"""
        return self.get_layer_key(layer_type, layer_idx, operation)
    
    def get_tensor_index(self, layer_type: str, layer_idx: Optional[int], operation: str) -> int:
        """获取tensor索引"""
        layer_key = self.get_layer_key(layer_type, layer_idx, operation)
        
        if layer_key not in self.layer_tensor_counters:
            self.layer_tensor_counters[layer_key] = 0
        
        # 对于同一层的不同tensor，使用相同的索引
        return self.layer_tensor_counters[layer_key]
    
    def increment_layer_counter(self, layer_type: str, layer_idx: Optional[int], operation: str):
        """增加层计数器（当该层的所有tensor都保存完毕后调用）"""
        layer_key = self.get_layer_key(layer_type, layer_idx, operation)
        if layer_key in self.layer_tensor_counters:
            self.layer_tensor_counters[layer_key] += 1
            print(f"[TensorIndexManager] 层 {layer_key} 索引递增到 {self.layer_tensor_counters[layer_key]}")
    
    def reset_layer_counter(self, layer_type: str, layer_idx: Optional[int], operation: str):
        """重置层计数器"""
        layer_key = self.get_layer_key(layer_type, layer_idx, operation)
        if layer_key in self.layer_tensor_counters:
            self.layer_tensor_counters[layer_key] = 0
            print(f"[TensorIndexManager] 层 {layer_key} 索引重置为 0")

# 全局tensor索引管理器实例
_global_tensor_index_manager = TensorIndexManager()

def get_tensor_index_manager() -> TensorIndexManager:
    """获取全局tensor索引管理器"""
    return _global_tensor_index_manager

def get_tensor_collection_state() -> TensorCollectionState:
    """获取全局tensor收集状态"""
    return _global_tensor_state

def set_global_rank(rank: int):
    """设置全局rank"""
    _global_tensor_state.set_rank(rank)

def set_global_sample_idx(sample_idx: int):
    """设置全局sample索引"""
    _global_tensor_state.set_sample_idx(sample_idx)

def set_global_iteration(iteration: int):
    """设置全局iteration"""
    _global_tensor_state.set_iteration(iteration)

def set_global_batch_idx(batch_idx: int):
    """设置全局batch索引"""
    _global_tensor_state.set_batch_idx(batch_idx)

def set_global_sequence_idx(sequence_idx: int):
    """设置全局sequence索引"""
    _global_tensor_state.set_sequence_idx(sequence_idx)

def get_rank_from_tensor_device(tensor: torch.Tensor) -> Optional[int]:
    """尝试从tensor设备信息推断rank"""
    try:
        if tensor.is_cuda:
            device_id = tensor.device.index
            if device_id is not None:
                # 在某些情况下，device_id可能对应rank
                return device_id
    except:
        pass
    return None

def get_current_rank() -> Optional[int]:
    """获取当前的rank信息"""
    state = get_tensor_collection_state()
    
    # 首先尝试从全局状态获取
    rank = state.get_rank()
    
    # 如果全局状态中没有，尝试直接从分布式环境获取
    if rank is None:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                # 更新全局状态
                state.set_rank(rank)
                print(f"[TensorSaver] 从分布式环境获取rank: {rank}")
        except Exception as e:
            print(f"[TensorSaver] 无法从分布式环境获取rank: {e}")
    
    # 如果仍然没有，尝试从环境变量获取
    if rank is None:
        rank_env = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
        if rank_env is not None:
            try:
                rank = int(rank_env)
                state.set_rank(rank)
                print(f"[TensorSaver] 从环境变量获取rank: {rank}")
            except ValueError:
                pass
    
    return rank

def initialize_tensor_collection(rank: Optional[int] = None, 
                               sample_idx: Optional[int] = None, 
                               iteration: int = 0,
                               batch_idx: int = 0,
                               sequence_idx: int = 0):
    """初始化tensor收集状态"""
    state = get_tensor_collection_state()
    
    if rank is not None:
        state.set_rank(rank)
    else:
        # 尝试自动检测rank
        auto_rank = state.get_rank()
        if auto_rank is None:
            state.set_rank(0)  # 默认值
    
    if sample_idx is not None:
        state.set_sample_idx(sample_idx)
    else:
        # 尝试自动检测sample_idx
        auto_sample_idx = state.get_sample_idx()
        if auto_sample_idx is None:
            state.set_sample_idx(0)  # 默认值
    
    state.set_iteration(iteration)
    state.set_batch_idx(batch_idx)
    state.set_sequence_idx(sequence_idx)
    
    print(f"[TensorCollection] 初始化完成 - Rank: {state.get_rank()}, Sample: {state.get_sample_idx()}, Iteration: {state.get_iteration()}")


class TensorSaver:
    """Tensor保存器，用于保存量化前后的tensor数据"""
    
    def __init__(self, save_dir: str = "./enhanced_tensor_logs", enabled: bool = True):
        """
        初始化Tensor保存器
        
        Args:
            save_dir: 保存目录
            enabled: 是否启用保存功能
        """
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.tensor_counter = 0
        self.current_iteration = 0
        self.micro_batch_count = 0
        self.control_micro_batches = 1  # 固定为1，进行一次完整forward后跳出
        self.collection_completed = False  # 标记是否已完成收集
        self.tensor_collected_in_warmup = False  # 标记是否已在warmup阶段收集过tensor
        
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[TensorSaver] 保存目录: {self.save_dir}")
    
    def set_iteration(self, iteration: int):
        """设置当前iteration"""
        self.current_iteration = iteration
        self.micro_batch_count = 0  # 重置micro_batch计数
        # 同时更新全局状态
        set_global_iteration(iteration)
        print(f"[TensorSaver] 设置当前iteration: {iteration}")
    
    def mark_collection_completed(self):
        """标记tensor收集已完成"""
        self.collection_completed = True
    
    def should_exit_after_forward(self) -> bool:
        """检查是否应该在forward后退出"""
        # 只有在启用tensor保存且已完成收集时才退出
        return self.enabled and self.collection_completed
    
    def should_collect_tensor(self) -> bool:
        """检查是否应该收集tensor"""
        # 只有在启用tensor保存且未在warmup阶段收集过时才收集
        return self.enabled and not self.tensor_collected_in_warmup
    
    def mark_warmup_collection(self):
        """标记已在warmup阶段收集过tensor"""
        self.tensor_collected_in_warmup = True
    
    def should_collect_in_steady_state(self) -> bool:
        """检查是否应该在steady state阶段收集tensor"""
        return self.enabled and not self.tensor_collected_in_warmup
    
    def _get_tensor_info(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """获取tensor的基本信息"""
        if tensor.numel() == 0:
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "requires_grad": tensor.requires_grad,
                "is_leaf": tensor.is_leaf,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "overflow_info": {
                    "upper_overflow_count": 0,
                    "lower_overflow_count": 0,
                    "upper_overflow_ratio": 0.0,
                    "lower_overflow_ratio": 0.0,
                    "total_overflow_ratio": 0.0
                }
            }
        
        # 计算基本统计信息
        tensor_flat = tensor.float().flatten()
        min_val = float(tensor_flat.min().item())
        max_val = float(tensor_flat.max().item())
        mean_val = float(tensor_flat.mean().item())
        std_val = float(tensor_flat.std().item())
        
        # 计算溢出信息
        overflow_info = self._calculate_overflow_info(tensor_flat)
        
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "overflow_info": overflow_info
        }
    
    def _calculate_overflow_info(self, tensor_flat: torch.Tensor) -> Dict[str, Any]:
        """计算tensor的溢出信息"""
        total_elements = tensor_flat.numel()
        
        # 定义不同数据类型的溢出阈值
        dtype_thresholds = {
            'torch.float16': {'max': 65504.0, 'min': -65504.0},
            'torch.bfloat16': {'max': 3.3895313892515355e+38, 'min': -3.3895313892515355e+38},
            'torch.float32': {'max': 3.4028235e+38, 'min': -3.4028235e+38},
            'torch.float64': {'max': 1.7976931348623157e+308, 'min': -1.7976931348623157e+308},
        }
        
        # 获取当前tensor的阈值
        tensor_dtype = str(tensor_flat.dtype)
        if tensor_dtype in dtype_thresholds:
            max_threshold = dtype_thresholds[tensor_dtype]['max']
            min_threshold = dtype_thresholds[tensor_dtype]['min']
        else:
            # 默认使用float32阈值
            max_threshold = dtype_thresholds['torch.float32']['max']
            min_threshold = dtype_thresholds['torch.float32']['min']
        
        # 计算上溢出和下溢出
        upper_overflow_mask = tensor_flat > max_threshold
        lower_overflow_mask = tensor_flat < min_threshold
        
        upper_overflow_count = int(upper_overflow_mask.sum().item())
        lower_overflow_count = int(lower_overflow_mask.sum().item())
        
        upper_overflow_ratio = upper_overflow_count / total_elements if total_elements > 0 else 0.0
        lower_overflow_ratio = lower_overflow_count / total_elements if total_elements > 0 else 0.0
        total_overflow_ratio = (upper_overflow_count + lower_overflow_count) / total_elements if total_elements > 0 else 0.0
        
        return {
            "upper_overflow_count": upper_overflow_count,
            "lower_overflow_count": lower_overflow_count,
            "upper_overflow_ratio": upper_overflow_ratio,
            "lower_overflow_ratio": lower_overflow_ratio,
            "total_overflow_ratio": total_overflow_ratio,
            "max_threshold": max_threshold,
            "min_threshold": min_threshold
        }
    
    def _generate_filename(self, 
                          layer_type: str,
                          operation: str,
                          quant_type: str, 
                          tensor_name: str,
                          layer_idx: Optional[int] = None,
                          phase: str = "unknown",
                          component: str = "unknown",
                          rank: Optional[int] = None,
                          tensor_group_idx: Optional[int] = None) -> str:
        """生成文件名"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.tensor_counter += 1
        
        # 构建文件名组件
        parts = [
            timestamp,
            f"{self.tensor_counter:04d}",
            f"iter{self.current_iteration:03d}",
            layer_type
        ]
        
        if layer_idx is not None:
            parts.append(f"L{layer_idx}")
        
        parts.extend([operation, phase, component, quant_type])
        
        if rank is not None:
            parts.append(f"rank{rank:02d}")
        
        # 添加tensor组索引（同一层的不同tensor使用相同索引）
        if tensor_group_idx is not None:
            parts.append(f"group{tensor_group_idx:03d}")
        
        parts.append(tensor_name)
        
        filename = "_".join(parts) + ".pt"
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
                   rank: Optional[int] = None,
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
            rank: GPU rank信息
            metadata: 额外的元数据
            
        Returns:
            保存的文件路径，如果未启用则返回None
        """
        if not self.should_collect_tensor():
            return None
        
        # 当启用tensor保存时，会在一次forward后自动退出，无需额外检查
        
        # 自动获取rank信息（如果未提供）
        if rank is None:
            rank = get_current_rank()
        
        # 如果仍然无法获取rank，尝试从tensor设备信息推断
        if rank is None:
            rank = get_rank_from_tensor_device(tensor)
            if rank is not None:
                print(f"[TensorSaver] 从tensor设备信息推断rank: {rank}")
                # 更新全局状态
                state = get_tensor_collection_state()
                state.set_rank(rank)
        
        # 如果仍然无法获取，使用默认值并打印警告
        if rank is None:
            rank = 0  # 默认rank为0
            print(f"[TensorSaver] 警告: 无法获取rank信息，使用默认值 {rank}")
        
        # 获取tensor组索引（同一层的不同tensor使用相同索引）
        index_manager = get_tensor_index_manager()
        tensor_group_idx = index_manager.get_tensor_index(layer_type, layer_idx, operation)
        
        try:
            # 生成文件名
            filename = self._generate_filename(layer_type, operation, quant_type, tensor_name, 
                                            layer_idx, phase, component, rank, tensor_group_idx)
            filepath = self.save_dir / filename
            
            # iteration数据计数已简化，无需手动增加
            
            # 准备保存数据 - 添加更安全的tensor处理
            try:
                # 先获取tensor信息（在移动之前）
                tensor_info = self._get_tensor_info(tensor)
                
                # 安全地处理tensor
                if tensor.is_cuda:
                    tensor_cpu = tensor.detach().cpu()
                else:
                    tensor_cpu = tensor.detach().clone()
                
                # 确保tensor是连续的
                if not tensor_cpu.is_contiguous():
                    tensor_cpu = tensor_cpu.contiguous()
                
                save_data = {
                    "tensor": tensor_cpu,
                    "tensor_info": tensor_info,
                    "metadata": {
                        "layer_type": layer_type,
                        "operation": operation,
                        "quant_type": quant_type,
                        "tensor_name": tensor_name,
                        "layer_idx": layer_idx,
                        "phase": phase,
                        "component": component,
                        "rank": rank,
                        "iteration": self.current_iteration,
                        "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        **(metadata or {})
                    }
                }
            except Exception as tensor_error:
                print(f"[TensorSaver] 处理tensor时出错: {tensor_error}")
                # 如果tensor处理失败，尝试更简单的方式
                save_data = {
                    "tensor": tensor.detach().cpu().contiguous(),
                    "tensor_info": {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
                    "metadata": {
                        "layer_type": layer_type,
                        "operation": operation,
                        "quant_type": quant_type,
                        "tensor_name": tensor_name,
                        "layer_idx": layer_idx,
                        "phase": phase,
                        "component": component,
                        "rank": rank,
                        "iteration": self.current_iteration,
                        "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        **(metadata or {})
                    }
                }
            
            # 保存到文件 - 添加更安全的保存过程
            try:
                # 确保目录存在
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # 使用更安全的保存方式
                torch.save(save_data, filepath, _use_new_zipfile_serialization=False)
                
                # 验证文件是否保存成功
                if filepath.exists() and filepath.stat().st_size > 0:
                    print(f"[TensorSaver] 已保存: {filename}")
                    return str(filepath)
                else:
                    print(f"[TensorSaver] 保存失败: 文件为空或不存在")
                    return None
                    
            except Exception as save_error:
                print(f"[TensorSaver] 保存文件时出错: {save_error}")
                # 尝试使用pickle保存
                try:
                    import pickle
                    with open(filepath.with_suffix('.pkl'), 'wb') as f:
                        pickle.dump(save_data, f)
                    print(f"[TensorSaver] 使用pickle保存成功: {filename}")
                    return str(filepath.with_suffix('.pkl'))
                except Exception as pickle_error:
                    print(f"[TensorSaver] pickle保存也失败: {pickle_error}")
                    return None
            
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
                              rank: Optional[int] = None,
                              attention_weights: Optional[torch.Tensor] = None,
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
            phase: 阶段
            component: 组件类型
            rank: GPU rank信息
            attention_weights: Attention权重矩阵（P分布）
            metadata: 额外元数据
            
        Returns:
            保存的文件路径字典
        """
        results = {}
        
        # 保存query tensor
        if query is not None:
            results["query"] = self.save_tensor(
                query, "attention", operation, quant_type, "query", layer_idx, phase, component, rank, metadata
            )
        
        # 保存key tensor
        if key is not None:
            results["key"] = self.save_tensor(
                key, "attention", operation, quant_type, "key", layer_idx, phase, component, rank, metadata
            )
        
        # 保存value tensor
        if value is not None:
            results["value"] = self.save_tensor(
                value, "attention", operation, quant_type, "value", layer_idx, phase, component, rank, metadata
            )
        
        # 保存attention权重（P分布）
        if attention_weights is not None:
            results["attention_weights"] = self.save_tensor(
                attention_weights, "attention", operation, quant_type, "attention_weights", layer_idx, phase, component, rank, metadata
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
                           rank: Optional[int] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
        """
        保存linear层的输入tensor
        
        Args:
            input_tensor: 输入tensor
            weight: 权重tensor
            quant_type: 量化类型
            operation: 操作类型
            layer_idx: 层索引
            phase: 阶段
            component: 组件类型
            rank: GPU rank信息
            metadata: 额外元数据
            
        Returns:
            保存的文件路径字典
        """
        results = {}
        
        # 保存input tensor
        if input_tensor is not None:
            results["input"] = self.save_tensor(
                input_tensor, "linear", operation, quant_type, "input", layer_idx, phase, component, rank, metadata
            )
        
        # 保存weight tensor
        if weight is not None:
            results["weight"] = self.save_tensor(
                weight, "linear", operation, quant_type, "weight", layer_idx, phase, component, rank, metadata
            )
        
        return results


# 全局tensor保存器实例
_global_tensor_saver = None


def get_tensor_saver() -> TensorSaver:
    """获取全局tensor保存器实例"""
    global _global_tensor_saver
    if _global_tensor_saver is None:
        # 从环境变量和命令行参数获取配置
        save_dir = os.environ.get("TENSOR_SAVE_DIR", "./enhanced_tensor_logs")
        enabled = os.environ.get("TENSOR_SAVE_ENABLED", "false").lower() == "true"
        
        # 尝试从命令行参数获取配置（如果可用）
        try:
            from megatron.training.global_vars import get_args
            args = get_args()
            if hasattr(args, 'tensor_save_dir') and args.tensor_save_dir:
                save_dir = args.tensor_save_dir
            if hasattr(args, 'save_tensors'):
                enabled = args.save_tensors or enabled
        except Exception as e:
            pass
        
        print(f"[TensorSaver] 初始化 - 保存目录: {save_dir}, 启用: {enabled}")
        _global_tensor_saver = TensorSaver(save_dir=save_dir, enabled=enabled)
        
        # 从环境变量设置iteration
        iteration = os.environ.get("TENSOR_SAVER_ITERATION")
        if iteration is not None:
            try:
                _global_tensor_saver.set_iteration(int(iteration))
            except ValueError:
                print(f"[TensorSaver] 无效的iteration值: {iteration}")
        
        # 初始化tensor收集状态
        initialize_tensor_collection()
    
    return _global_tensor_saver


def save_attention_tensors(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          quant_type: str,
                          operation: str = "forward",
                          layer_idx: Optional[int] = None,
                          phase: str = "pre",
                          component: str = "FA",
                          rank: Optional[int] = None,
                          attention_weights: Optional[torch.Tensor] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """保存attention层tensor的便捷函数"""
    saver = get_tensor_saver()
    return saver.save_attention_tensors(query, key, value, quant_type, operation, layer_idx, phase, component, rank, attention_weights, metadata)


def save_linear_tensors(input_tensor: torch.Tensor,
                       weight: torch.Tensor,
                       quant_type: str,
                       operation: str = "forward",
                       layer_idx: Optional[int] = None,
                       phase: str = "pre",
                       component: str = "linear",
                       rank: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """保存linear层tensor的便捷函数"""
    saver = get_tensor_saver()
    return saver.save_linear_tensors(input_tensor, weight, quant_type, operation, layer_idx, phase, component, rank, metadata)


def save_tensor(tensor: torch.Tensor,
                layer_type: str,
                operation: str,
                quant_type: str,
                tensor_name: str,
                layer_idx: Optional[int] = None,
                phase: str = "unknown",
                component: str = "unknown",
                rank: Optional[int] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """保存单个tensor的便捷函数"""
    saver = get_tensor_saver()
    return saver.save_tensor(tensor, layer_type, operation, quant_type, tensor_name, layer_idx, phase, component, rank, metadata)

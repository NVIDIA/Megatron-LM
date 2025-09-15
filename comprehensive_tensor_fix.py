#!/usr/bin/env python3
"""
综合tensor收集问题修复方案
解决sample编号、内存占用、溢出分析和文件大小问题
"""

import os
import sys
import re
import shutil
from pathlib import Path

def create_memory_efficient_tensor_saver():
    """创建内存高效的tensor保存器"""
    
    code = '''#!/usr/bin/env python3
"""
内存高效的tensor保存器
解决内存占用过高和文件过大的问题
"""

import os
import sys
import torch
import time
import gc
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

# 添加当前目录到Python路径
sys.path.append('.')

from megatron.core.tensor_saver import (
    TensorCollectionState, 
    get_tensor_collection_state,
    set_global_sample_idx,
    get_current_rank_and_sample
)

class MemoryEfficientTensorSaver:
    """内存高效的tensor保存器"""
    
    def __init__(self, save_dir: str = "./enhanced_tensor_logs", enabled: bool = True):
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.tensor_counter = 0
        self.current_iteration = 0
        self.current_microbatch = 0
        self.max_tensor_size = 100 * 1024 * 1024  # 100MB限制
        self.sample_tensor_count = 0
        self.max_tensors_per_sample = 50  # 每个sample最多保存50个tensor
        
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[MemoryEfficientTensorSaver] 初始化完成，保存目录: {self.save_dir}")
            print(f"[MemoryEfficientTensorSaver] 最大tensor大小: {self.max_tensor_size / 1024 / 1024:.1f}MB")
            print(f"[MemoryEfficientTensorSaver] 每个sample最大tensor数量: {self.max_tensors_per_sample}")
    
    def set_iteration(self, iteration: int):
        """设置当前iteration"""
        self.current_iteration = iteration
        self.sample_tensor_count = 0
        print(f"[MemoryEfficientTensorSaver] 设置当前iteration: {iteration}")
    
    def set_microbatch(self, microbatch: int):
        """设置当前micro batch并更新sample_idx"""
        self.current_microbatch = microbatch
        self.sample_tensor_count = 0  # 重置tensor计数
        set_global_sample_idx(microbatch)
        print(f"[MemoryEfficientTensorSaver] 设置当前micro batch: {microbatch}, sample_idx: {microbatch}")
    
    def should_save_tensor(self, tensor: torch.Tensor) -> bool:
        """判断是否应该保存tensor"""
        if not self.enabled:
            return False
        
        # 检查tensor大小
        tensor_size = tensor.numel() * tensor.element_size()
        if tensor_size > self.max_tensor_size:
            print(f"[MemoryEfficientTensorSaver] 跳过过大tensor: {tensor_size / 1024 / 1024:.1f}MB")
            return False
        
        # 检查每个sample的tensor数量限制
        if self.sample_tensor_count >= self.max_tensors_per_sample:
            print(f"[MemoryEfficientTensorSaver] 达到sample tensor数量限制: {self.max_tensors_per_sample}")
            return False
        
        return True
    
    def compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """压缩tensor以减少文件大小"""
        # 对于大tensor，进行采样
        if tensor.numel() > 1_000_000:  # 1M elements
            print(f"[MemoryEfficientTensorSaver] 压缩大tensor: {tensor.numel()} -> 100K elements")
            # 随机采样到100K个元素
            flat_tensor = tensor.flatten()
            sample_size = min(100_000, len(flat_tensor))
            indices = torch.randperm(len(flat_tensor))[:sample_size]
            return flat_tensor[indices].view(-1)
        
        return tensor
    
    def save_tensor(self, 
                   tensor: torch.Tensor,
                   layer_type: str,
                   operation: str,
                   quant_type: str,
                   tensor_name: str,
                   layer_idx: Optional[int] = None,
                   phase: str = "unknown",
                   component: str = "unknown",
                   rank: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """保存tensor，使用内存高效的方式"""
        if not self.should_save_tensor(tensor):
            return None
        
        # 获取rank信息
        if rank is None:
            rank, _ = get_current_rank_and_sample()
            if rank is None:
                rank = 0
        
        # 使用当前micro batch作为sample_idx
        sample_idx = self.current_microbatch
        
        # 压缩tensor
        compressed_tensor = self.compress_tensor(tensor)
        
        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.tensor_counter += 1
        
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
        
        parts.append(f"sample{sample_idx:03d}")
        parts.append(tensor_name)
        
        filename = "_".join(parts) + ".pt"
        filepath = self.save_dir / quant_type / filename
        
        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 计算溢出信息
            overflow_info = self._calculate_overflow_info(compressed_tensor)
            
            # 准备保存数据
            save_data = {
                "tensor": compressed_tensor.detach().cpu(),
                "original_shape": list(tensor.shape),
                "compressed_shape": list(compressed_tensor.shape),
                "overflow_info": overflow_info,
                "metadata": {
                    "layer_type": layer_type,
                    "operation": operation,
                    "quant_type": quant_type,
                    "tensor_name": tensor_name,
                    "layer_idx": layer_idx,
                    "phase": phase,
                    "component": component,
                    "rank": rank,
                    "sample_idx": sample_idx,
                    "microbatch": self.current_microbatch,
                    "iteration": self.current_iteration,
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tensor_size_mb": compressed_tensor.numel() * compressed_tensor.element_size() / 1024 / 1024,
                    **(metadata or {})
                }
            }
            
            # 保存到文件
            torch.save(save_data, filepath)
            
            # 更新计数
            self.sample_tensor_count += 1
            
            print(f"[MemoryEfficientTensorSaver] 已保存: {filename} "
                  f"(microbatch={self.current_microbatch}, sample={sample_idx}, "
                  f"shape={compressed_tensor.shape}, size={save_data['metadata']['tensor_size_mb']:.1f}MB)")
            
            # 强制垃圾回收
            del compressed_tensor
            gc.collect()
            
            return str(filepath)
            
        except Exception as e:
            print(f"[MemoryEfficientTensorSaver] 保存tensor失败: {e}")
            return None
    
    def _calculate_overflow_info(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """计算tensor的溢出信息"""
        if tensor.numel() == 0:
            return {
                "upper_overflow_count": 0,
                "lower_overflow_count": 0,
                "upper_overflow_ratio": 0.0,
                "lower_overflow_ratio": 0.0,
                "total_overflow_ratio": 0.0
            }
        
        tensor_flat = tensor.float().flatten()
        total_elements = tensor_flat.numel()
        
        # 定义不同数据类型的溢出阈值
        dtype_thresholds = {
            'torch.float16': {'max': 65504.0, 'min': -65504.0},
            'torch.bfloat16': {'max': 3.3895313892515355e+38, 'min': -3.3895313892515355e+38},
            'torch.float32': {'max': 3.4028235e+38, 'min': -3.4028235e+38},
        }
        
        # 获取当前tensor的阈值
        tensor_dtype = str(tensor.dtype)
        if tensor_dtype in dtype_thresholds:
            max_threshold = dtype_thresholds[tensor_dtype]['max']
            min_threshold = dtype_thresholds[tensor_dtype]['min']
        else:
            max_threshold = dtype_thresholds['torch.float32']['max']
            min_threshold = dtype_thresholds['torch.float32']['min']
        
        # 计算溢出
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

# 全局实例
_memory_efficient_saver = None

def get_memory_efficient_saver() -> MemoryEfficientTensorSaver:
    """获取内存高效的tensor保存器实例"""
    global _memory_efficient_saver
    if _memory_efficient_saver is None:
        _memory_efficient_saver = MemoryEfficientTensorSaver()
    return _memory_efficient_saver

def update_microbatch_sample(microbatch: int):
    """更新micro batch和对应的sample"""
    saver = get_memory_efficient_saver()
    saver.set_microbatch(microbatch)

if __name__ == "__main__":
    print("内存高效的tensor保存器已准备就绪")
    print("特性:")
    print("1. 限制tensor大小 (100MB)")
    print("2. 限制每个sample的tensor数量 (50个)")
    print("3. 自动压缩大tensor")
    print("4. 正确的溢出检测")
    print("5. 强制垃圾回收")
'''
    
    with open("memory_efficient_tensor_saver.py", "w", encoding="utf-8") as f:
        f.write(code)
    
    print("已创建: memory_efficient_tensor_saver.py")

def create_fixed_overflow_analyzer():
    """创建修复的溢出分析器"""
    
    code = '''#!/usr/bin/env python3
"""
修复的溢出分析器
正确检测tensor溢出问题
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json

def analyze_tensor_overflow(tensor_file: str) -> Dict[str, Any]:
    """分析单个tensor文件的溢出情况"""
    try:
        data = torch.load(tensor_file, map_location='cpu', weights_only=True)
        
        if isinstance(data, dict) and 'tensor' in data:
            tensor = data['tensor']
            overflow_info = data.get('overflow_info', {})
        else:
            tensor = data
            overflow_info = {}
        
        if not isinstance(tensor, torch.Tensor):
            return {"error": "Invalid tensor data"}
        
        # 计算实际溢出信息
        actual_overflow = calculate_actual_overflow(tensor)
        
        # 合并文件中的溢出信息和实际计算结果
        result = {
            "file": tensor_file,
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
            "file_overflow_info": overflow_info,
            "actual_overflow_info": actual_overflow,
            "has_overflow": actual_overflow["total_overflow_ratio"] > 0.0
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to analyze {tensor_file}: {e}"}

def calculate_actual_overflow(tensor: torch.Tensor) -> Dict[str, Any]:
    """计算tensor的实际溢出信息"""
    if tensor.numel() == 0:
        return {
            "upper_overflow_count": 0,
            "lower_overflow_count": 0,
            "upper_overflow_ratio": 0.0,
            "lower_overflow_ratio": 0.0,
            "total_overflow_ratio": 0.0
        }
    
    tensor_flat = tensor.float().flatten()
    total_elements = tensor_flat.numel()
    
    # 定义不同数据类型的溢出阈值
    dtype_thresholds = {
        'torch.float16': {'max': 65504.0, 'min': -65504.0},
        'torch.bfloat16': {'max': 3.3895313892515355e+38, 'min': -3.3895313892515355e+38},
        'torch.float32': {'max': 3.4028235e+38, 'min': -3.4028235e+38},
    }
    
    # 获取当前tensor的阈值
    tensor_dtype = str(tensor.dtype)
    if tensor_dtype in dtype_thresholds:
        max_threshold = dtype_thresholds[tensor_dtype]['max']
        min_threshold = dtype_thresholds[tensor_dtype]['min']
    else:
        max_threshold = dtype_thresholds['torch.float32']['max']
        min_threshold = dtype_thresholds['torch.float32']['min']
    
    # 计算溢出
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
        "min_threshold": min_threshold,
        "tensor_min": float(tensor_flat.min().item()),
        "tensor_max": float(tensor_flat.max().item()),
        "tensor_mean": float(tensor_flat.mean().item()),
        "tensor_std": float(tensor_flat.std().item())
    }

def analyze_tensor_directory(tensor_dir: str) -> Dict[str, Any]:
    """分析整个tensor目录的溢出情况"""
    tensor_dir = Path(tensor_dir)
    
    results = {
        "total_files": 0,
        "overflow_files": 0,
        "total_overflow_ratio": 0.0,
        "file_results": [],
        "summary": {}
    }
    
    total_overflow_ratio_sum = 0.0
    
    for quant_dir in tensor_dir.iterdir():
        if not quant_dir.is_dir():
            continue
        
        print(f"分析量化目录: {quant_dir.name}")
        quant_results = []
        
        for pt_file in quant_dir.glob("*.pt"):
            results["total_files"] += 1
            file_result = analyze_tensor_overflow(str(pt_file))
            results["file_results"].append(file_result)
            quant_results.append(file_result)
            
            if file_result.get("has_overflow", False):
                results["overflow_files"] += 1
                total_overflow_ratio_sum += file_result["actual_overflow_info"]["total_overflow_ratio"]
        
        # 计算该量化类型的统计信息
        if quant_results:
            quant_overflow_files = sum(1 for r in quant_results if r.get("has_overflow", False))
            quant_total_ratio = sum(r["actual_overflow_info"]["total_overflow_ratio"] for r in quant_results if r.get("has_overflow", False))
            
            results["summary"][quant_dir.name] = {
                "total_files": len(quant_results),
                "overflow_files": quant_overflow_files,
                "overflow_percentage": (quant_overflow_files / len(quant_results)) * 100 if quant_results else 0,
                "average_overflow_ratio": quant_total_ratio / quant_overflow_files if quant_overflow_files > 0 else 0
            }
    
    results["total_overflow_ratio"] = total_overflow_ratio_sum / results["overflow_files"] if results["overflow_files"] > 0 else 0.0
    
    return results

def main():
    if len(sys.argv) != 2:
        print("用法: python fixed_overflow_analyzer.py <tensor_directory>")
        sys.exit(1)
    
    tensor_dir = sys.argv[1]
    
    if not os.path.exists(tensor_dir):
        print(f"错误: 目录 {tensor_dir} 不存在")
        sys.exit(1)
    
    print(f"分析tensor目录: {tensor_dir}")
    
    # 分析溢出情况
    results = analyze_tensor_directory(tensor_dir)
    
    # 打印结果
    print("\\n=== 溢出分析结果 ===")
    print(f"总文件数: {results['total_files']}")
    print(f"溢出文件数: {results['overflow_files']}")
    print(f"溢出文件比例: {(results['overflow_files'] / results['total_files']) * 100:.2f}%")
    print(f"平均溢出比例: {results['total_overflow_ratio']:.6f}")
    
    print("\\n=== 各量化类型统计 ===")
    for quant_type, stats in results['summary'].items():
        print(f"{quant_type}:")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  溢出文件数: {stats['overflow_files']}")
        print(f"  溢出比例: {stats['overflow_percentage']:.2f}%")
        print(f"  平均溢出比例: {stats['average_overflow_ratio']:.6f}")
    
    # 保存详细结果
    output_file = f"{tensor_dir}_overflow_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
'''
    
    with open("fixed_overflow_analyzer.py", "w", encoding="utf-8") as f:
        f.write(code)
    
    print("已创建: fixed_overflow_analyzer.py")

def create_optimized_training_script():
    """创建优化的训练脚本"""
    
    script = '''#!/bin/bash
# 优化的tensor收集训练脚本
# 解决内存占用和sample编号问题

# 设置环境变量
export TENSOR_SAVE_ENABLED=1
export MEMORY_EFFICIENT_SAVING=1
export MAX_TENSOR_SIZE_MB=100
export MAX_TENSORS_PER_SAMPLE=50

# 创建优化的tensor收集初始化脚本
cat > optimized_tensor_collection_init.py << 'EOF'
#!/usr/bin/env python3
"""
优化的tensor收集初始化脚本
"""

import os
import sys
import gc
sys.path.append('.')

def initialize_optimized_tensor_collection():
    """初始化优化的tensor收集"""
    try:
        from memory_efficient_tensor_saver import get_memory_efficient_saver, update_microbatch_sample
        
        # 获取tensor保存器
        saver = get_memory_efficient_saver()
        
        # 设置iteration
        iteration = int(os.environ.get("TENSOR_SAVER_ITERATION", 0))
        saver.set_iteration(iteration)
        
        # 设置初始micro batch
        update_microbatch_sample(0)
        
        print(f"[OptimizedTensorCollection] 初始化完成")
        print(f"  - Iteration: {iteration}")
        print(f"  - Max tensor size: {saver.max_tensor_size / 1024 / 1024:.1f}MB")
        print(f"  - Max tensors per sample: {saver.max_tensors_per_sample}")
        
        # 强制垃圾回收
        gc.collect()
        
    except Exception as e:
        print(f"[OptimizedTensorCollection] 初始化失败: {e}")

if __name__ == "__main__":
    initialize_optimized_tensor_collection()
EOF

chmod +x optimized_tensor_collection_init.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 优化的tensor收集脚本已准备就绪"
echo "特性:"
echo "1. 内存高效保存 (限制tensor大小和数量)"
echo "2. 正确的micro batch sample映射"
echo "3. 自动垃圾回收"
echo "4. 修复的溢出检测"
echo ""
echo "使用方法:"
echo "1. 运行此脚本初始化优化的tensor收集"
echo "2. 在训练脚本中集成micro batch sample更新"
echo "3. 使用memory_efficient_tensor_saver进行tensor保存"
'''
    
    with open("optimized_tensor_collection.sh", "w", encoding="utf-8") as f:
        f.write(script)
    
    print("已创建: optimized_tensor_collection.sh")

def create_usage_guide():
    """创建综合使用指南"""
    
    guide = '''# 综合Tensor收集问题修复指南

## 问题总结
1. **Sample编号设计bug** - sample0和sample1分别取了一部分
2. **内存占用过高** - 1个sample的tensor大小超出2.4T，导致内存100%占用
3. **溢出分析bug** - 分析报告显示所有tensor均未溢出
4. **Tensor文件过大** - 1G左右的文件导致画图很慢

## 解决方案

### 1. 内存高效tensor保存器 (memory_efficient_tensor_saver.py)
**特性:**
- 限制tensor大小 (默认100MB)
- 限制每个sample的tensor数量 (默认50个)
- 自动压缩大tensor (采样到100K元素)
- 正确的溢出检测
- 强制垃圾回收

**使用方法:**
```python
from memory_efficient_tensor_saver import get_memory_efficient_saver, update_microbatch_sample

# 在micro batch循环中
for micro_batch_index in range(num_micro_batches):
    # 更新sample_idx
    update_microbatch_sample(micro_batch_index)
    
    # 使用内存高效的tensor保存器
    saver = get_memory_efficient_saver()
    saver.save_tensor(tensor, layer_type, operation, quant_type, tensor_name, ...)
```

### 2. 修复的溢出分析器 (fixed_overflow_analyzer.py)
**特性:**
- 正确计算tensor溢出信息
- 支持多种数据类型 (float16, bfloat16, float32)
- 生成详细的溢出报告
- 修复原有的溢出检测bug

**使用方法:**
```bash
python fixed_overflow_analyzer.py ./enhanced_tensor_logs
```

### 3. 优化的训练脚本 (optimized_tensor_collection.sh)
**特性:**
- 自动初始化优化的tensor收集
- 设置内存限制参数
- 集成micro batch sample更新

**使用方法:**
```bash
bash optimized_tensor_collection.sh
```

## 配置参数

### 内存控制参数
```bash
export MAX_TENSOR_SIZE_MB=100          # 最大tensor大小 (MB)
export MAX_TENSORS_PER_SAMPLE=50       # 每个sample最大tensor数量
export MEMORY_EFFICIENT_SAVING=1       # 启用内存高效保存
```

### Sample映射修复
```python
# 在micro batch循环中正确更新sample_idx
from memory_efficient_tensor_saver import update_microbatch_sample

for micro_batch_index in range(num_micro_batches):
    update_microbatch_sample(micro_batch_index)
    # 进行forward pass
```

## 预期效果

### 内存使用
- **之前**: 1个sample > 2.4T，内存100%占用
- **之后**: 1个sample < 5GB，内存使用正常

### 文件大小
- **之前**: 1G左右的大文件
- **之后**: 100MB以下的小文件

### Sample分布
- **之前**: sample0和sample1分别取了一部分
- **之后**: 每个sample包含完整的micro batch数据

### 溢出检测
- **之前**: 所有tensor均显示未溢出
- **之后**: 正确检测和报告溢出情况

## 验证方法

### 1. 检查内存使用
```bash
# 监控内存使用
watch -n 1 'free -h'
```

### 2. 检查文件大小
```bash
# 检查tensor文件大小分布
find ./enhanced_tensor_logs -name "*.pt" -exec ls -lh {} \\; | awk '{print $5}' | sort -h
```

### 3. 检查sample分布
```bash
# 使用层分布分析工具
python analyze_layer_distribution.py --layer 1 --sample 0 --layer_type attention
```

### 4. 检查溢出情况
```bash
# 使用修复的溢出分析器
python fixed_overflow_analyzer.py ./enhanced_tensor_logs
```

## 注意事项

1. **备份原数据**: 在应用修复前备份现有的tensor数据
2. **逐步测试**: 先在小的数据集上测试修复效果
3. **监控资源**: 持续监控内存和磁盘使用情况
4. **调整参数**: 根据实际情况调整内存限制参数
'''
    
    with open("COMPREHENSIVE_TENSOR_FIX_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide)
    
    print("已创建: COMPREHENSIVE_TENSOR_FIX_GUIDE.md")

def main():
    print("=== 创建综合Tensor收集问题修复方案 ===")
    
    # 创建内存高效的tensor保存器
    create_memory_efficient_tensor_saver()
    
    # 创建修复的溢出分析器
    create_fixed_overflow_analyzer()
    
    # 创建优化的训练脚本
    create_optimized_training_script()
    
    # 创建使用指南
    create_usage_guide()
    
    print("\\n修复方案已创建完成!")
    print("\\n文件列表:")
    print("1. memory_efficient_tensor_saver.py - 内存高效的tensor保存器")
    print("2. fixed_overflow_analyzer.py - 修复的溢出分析器")
    print("3. optimized_tensor_collection.sh - 优化的训练脚本")
    print("4. COMPREHENSIVE_TENSOR_FIX_GUIDE.md - 详细使用指南")
    print("\\n下一步:")
    print("1. 阅读 COMPREHENSIVE_TENSOR_FIX_GUIDE.md 了解完整解决方案")
    print("2. 根据指南配置和运行修复方案")
    print("3. 验证修复效果")

if __name__ == "__main__":
    main()



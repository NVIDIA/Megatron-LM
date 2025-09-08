#!/usr/bin/env python3
"""
数据处理工具函数
提供常用的数据处理辅助功能
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import shutil

class DataProcessor:
    """数据处理器类"""
    
    def __init__(self, base_dir: str = "."):
        """
        初始化数据处理器
        
        Args:
            base_dir: 基础目录路径
        """
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "dataset"
        self.model_dir = self.base_dir / "model"
        self.tools_dir = self.base_dir / "tools"
        
    def check_environment(self) -> bool:
        """检查环境是否满足要求"""
        print("=== 检查环境 ===")
        
        # 检查必要目录
        required_dirs = [self.dataset_dir, self.model_dir, self.tools_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"❌ 目录不存在: {dir_path}")
                return False
            print(f"✅ 目录存在: {dir_path}")
        
        # 检查preprocess_data.py
        preprocess_script = self.tools_dir / "preprocess_data.py"
        if not preprocess_script.exists():
            print(f"❌ 预处理脚本不存在: {preprocess_script}")
            return False
        print(f"✅ 预处理脚本存在: {preprocess_script}")
        
        return True
    
    def list_available_datasets(self) -> List[str]:
        """列出可用的数据集"""
        print("=== 可用数据集 ===")
        
        if not self.dataset_dir.exists():
            print("数据集目录不存在")
            return []
        
        datasets = []
        for item in self.dataset_dir.iterdir():
            if item.is_dir():
                # 检查是否包含数据文件
                data_files = list(item.glob("**/*.json*")) + list(item.glob("**/*.txt*"))
                if data_files:
                    datasets.append(item.name)
                    print(f"✅ {item.name} ({len(data_files)} 个文件)")
                else:
                    print(f"⚠️  {item.name} (无数据文件)")
        
        return datasets
    
    def list_available_models(self) -> List[str]:
        """列出可用的模型"""
        print("=== 可用模型 ===")
        
        if not self.model_dir.exists():
            print("模型目录不存在")
            return []
        
        models = []
        for item in self.model_dir.iterdir():
            if item.is_dir():
                # 检查是否包含tokenizer文件
                tokenizer_files = list(item.glob("tokenizer*")) + list(item.glob("*.json"))
                if tokenizer_files:
                    models.append(item.name)
                    print(f"✅ {item.name}")
                else:
                    print(f"⚠️  {item.name} (无tokenizer文件)")
        
        return models
    
    def estimate_processing_time(self, input_path: str, workers: int = 16) -> str:
        """估算处理时间"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            return "无法估算：输入路径不存在"
        
        # 计算文件大小
        total_size = 0
        file_count = 0
        
        if input_path.is_file():
            total_size = input_path.stat().st_size
            file_count = 1
        else:
            for file_path in input_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
        
        # 估算处理时间（基于经验值）
        # 假设每GB数据需要约10-30分钟，取决于硬件配置
        size_gb = total_size / (1024**3)
        estimated_minutes = size_gb * 20  # 20分钟/GB
        
        return f"估算处理时间: {estimated_minutes:.1f}分钟 (基于{size_gb:.2f}GB数据, {file_count}个文件)"
    
    def get_optimal_workers(self, input_path: str) -> int:
        """获取最优的工作进程数"""
        import multiprocessing
        
        # 获取CPU核心数
        cpu_count = multiprocessing.cpu_count()
        
        # 检查输入文件数量
        input_path = Path(input_path)
        file_count = 0
        
        if input_path.is_file():
            file_count = 1
        else:
            file_count = len(list(input_path.rglob("*")))
        
        # 计算最优进程数
        optimal_workers = min(cpu_count, max(1, file_count // 4))
        
        return optimal_workers
    
    def validate_input_data(self, input_path: str) -> Tuple[bool, str]:
        """验证输入数据"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            return False, "输入路径不存在"
        
        # 检查文件格式
        if input_path.is_file():
            if input_path.suffix in ['.json', '.jsonl', '.txt']:
                return True, "文件格式正确"
            else:
                return False, f"不支持的文件格式: {input_path.suffix}"
        else:
            # 检查目录中的文件
            data_files = list(input_path.rglob("*.json*")) + list(input_path.rglob("*.txt*"))
            if not data_files:
                return False, "目录中没有找到数据文件"
            
            return True, f"找到 {len(data_files)} 个数据文件"
    
    def create_processing_script(self, 
                                dataset_name: str,
                                input_path: str,
                                output_prefix: str,
                                tokenizer_model: str,
                                **kwargs) -> str:
        """创建处理脚本"""
        
        script_content = f"""#!/bin/bash
# 自动生成的数据处理脚本 - {dataset_name}
# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

# 设置参数
INPUT_PATH="{input_path}"
OUTPUT_PREFIX="{output_prefix}"
TOKENIZER_MODEL="{tokenizer_model}"
WORKERS={kwargs.get('workers', 16)}
PARTITIONS={kwargs.get('partitions', 4)}
TOKENIZER_TYPE="{kwargs.get('tokenizer_type', 'HuggingFaceTokenizer')}"
APPEND_EOD="{kwargs.get('append_eod', 'true')}"
SEQUENCE_LENGTH={kwargs.get('sequence_length', 2048)}
OVERWRITE="{kwargs.get('overwrite', 'false')}"

echo "=== {dataset_name}数据处理 ==="
echo "输入路径: $INPUT_PATH"
echo "输出前缀: $OUTPUT_PREFIX"
echo "分词器模型: $TOKENIZER_MODEL"
echo "工作进程数: $WORKERS"
echo "分区数: $PARTITIONS"

# 构建命令
CMD="python tools/preprocess_data.py"
CMD="$CMD --input '$INPUT_PATH'"
CMD="$CMD --workers $WORKERS"
CMD="$CMD --partitions $PARTITIONS"
CMD="$CMD --output-prefix $OUTPUT_PREFIX"
CMD="$CMD --tokenizer-type $TOKENIZER_TYPE"
CMD="$CMD --tokenizer-model $TOKENIZER_MODEL"

if [ "$APPEND_EOD" = "true" ]; then
    CMD="$CMD --append-eod"
fi

if [ "$SEQUENCE_LENGTH" != "2048" ]; then
    CMD="$CMD --seq-length $SEQUENCE_LENGTH"
fi

if [ "$OVERWRITE" = "true" ]; then
    CMD="$CMD --overwrite"
fi

echo "执行命令: $CMD"
echo "开始处理时间: $(date)"

# 执行命令
eval $CMD

if [ $? -eq 0 ]; then
    echo "✅ 处理完成: $(date)"
    echo "输出文件:"
    ls -lh "$OUTPUT_PREFIX"*
else
    echo "❌ 处理失败"
    exit 1
fi
"""
        
        # 保存脚本
        script_path = self.base_dir / "script" / f"process_{dataset_name}_auto.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 设置执行权限
        os.chmod(script_path, 0o755)
        
        return str(script_path)
    
    def run_processing(self, 
                      input_path: str,
                      output_prefix: str,
                      tokenizer_model: str,
                      **kwargs) -> bool:
        """运行数据处理"""
        
        # 验证输入
        is_valid, message = self.validate_input_data(input_path)
        if not is_valid:
            print(f"❌ 输入验证失败: {message}")
            return False
        
        print(f"✅ 输入验证通过: {message}")
        
        # 获取最优参数
        optimal_workers = self.get_optimal_workers(input_path)
        if 'workers' not in kwargs:
            kwargs['workers'] = optimal_workers
            print(f"💡 使用最优工作进程数: {optimal_workers}")
        
        # 估算处理时间
        time_estimate = self.estimate_processing_time(input_path, kwargs['workers'])
        print(f"⏱️  {time_estimate}")
        
        # 构建命令
        cmd = [
            "python", "tools/preprocess_data.py",
            "--input", input_path,
            "--workers", str(kwargs.get('workers', 16)),
            "--partitions", str(kwargs.get('partitions', 4)),
            "--output-prefix", output_prefix,
            "--tokenizer-type", kwargs.get('tokenizer_type', 'HuggingFaceTokenizer'),
            "--tokenizer-model", tokenizer_model
        ]
        
        if kwargs.get('append_eod', True):
            cmd.append("--append-eod")
        
        if kwargs.get('sequence_length', 2048) != 2048:
            cmd.extend(["--seq-length", str(kwargs['sequence_length'])])
        
        if kwargs.get('overwrite', False):
            cmd.append("--overwrite")
        
        print(f"🚀 执行命令: {' '.join(cmd)}")
        
        # 记录开始时间
        start_time = time.time()
        print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 执行命令
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # 计算处理时间
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ 处理完成!")
            print(f"⏰ 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
            
            # 显示输出文件
            output_path = Path(output_prefix)
            if output_path.with_suffix('.bin').exists():
                bin_size = output_path.with_suffix('.bin').stat().st_size / (1024**2)
                idx_size = output_path.with_suffix('.idx').stat().st_size / (1024**2)
                print(f"📁 输出文件大小: .bin={bin_size:.1f}MB, .idx={idx_size:.1f}MB")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 处理失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据处理工具')
    parser.add_argument('--action', choices=['check', 'list', 'process', 'estimate'], 
                       default='check', help='执行的操作')
    parser.add_argument('--input', type=str, help='输入数据路径')
    parser.add_argument('--output', type=str, help='输出前缀')
    parser.add_argument('--tokenizer', type=str, help='分词器模型路径')
    parser.add_argument('--workers', type=int, default=16, help='工作进程数')
    parser.add_argument('--partitions', type=int, default=4, help='分区数')
    parser.add_argument('--seq-length', type=int, default=2048, help='序列长度')
    parser.add_argument('--no-eod', action='store_true', help='不追加EOD')
    parser.add_argument('--overwrite', action='store_true', help='覆盖输出文件')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = DataProcessor()
    
    if args.action == 'check':
        processor.check_environment()
    
    elif args.action == 'list':
        processor.list_available_datasets()
        processor.list_available_models()
    
    elif args.action == 'estimate':
        if not args.input:
            print("错误: 需要指定 --input 参数")
            return
        estimate = processor.estimate_processing_time(args.input, args.workers)
        print(estimate)
    
    elif args.action == 'process':
        if not all([args.input, args.output, args.tokenizer]):
            print("错误: 需要指定 --input, --output, --tokenizer 参数")
            return
        
        kwargs = {
            'workers': args.workers,
            'partitions': args.partitions,
            'sequence_length': args.seq_length,
            'append_eod': not args.no_eod,
            'overwrite': args.overwrite
        }
        
        success = processor.run_processing(
            args.input, args.output, args.tokenizer, **kwargs
        )
        
        if not success:
            exit(1)


if __name__ == "__main__":
    main()

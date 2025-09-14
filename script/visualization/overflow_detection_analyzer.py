#!/usr/bin/env python3
"""
Overflow Detection Analyzer
基于量化类型特征值检测tensor溢出情况
支持bf16, mxfp8, mxfp4, hifp8四种量化类型
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, NamedTuple
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端
plt.switch_backend('Agg')

class QuantizationLimits(NamedTuple):
    """量化类型限制值"""
    max_positive_normal: float
    min_positive_normal: float
    max_positive_denormal: float
    min_positive_denormal: float
    exponent_range: Tuple[int, int]
    exponent_range_with_denormal: Tuple[int, int]
    supports_infinity: bool
    supports_nan: bool
    supports_zero: bool

class OverflowDetectionAnalyzer:
    def __init__(self, tensor_dir: str, output_dir: str, max_workers: int = 4):
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # 支持的量化类型
        self.quant_types = ['bf16', 'mxfp8', 'mxfp4', 'hifp8']
        
        # 支持的样本和层
        self.samples = [0, 1, 2]
        self.layers = list(range(1, 17))  # 1-16层
        
        # 定义各量化类型的限制值（基于图中的数据）
        self.quantization_limits = {
            'bf16': QuantizationLimits(
                max_positive_normal=2**15 * (2 - 2**-10),  # 65504
                min_positive_normal=2**-14,  # 6.103515625e-05
                max_positive_denormal=2**-14 * (1 - 2**-10),  # 6.095551e-05
                min_positive_denormal=2**-24,  # 5.960464477539063e-08
                exponent_range=(-14, 15),
                exponent_range_with_denormal=(-24, 15),
                supports_infinity=True,
                supports_nan=True,
                supports_zero=True
            ),
            'hifp8': QuantizationLimits(
                max_positive_normal=2**15,  # 32768
                min_positive_normal=2**-15,  # 3.0517578125e-05
                max_positive_denormal=2**-16,  # 1.52587890625e-05
                min_positive_denormal=2**-22,  # 2.384185791015625e-07
                exponent_range=(-15, 15),
                exponent_range_with_denormal=(-22, 15),
                supports_infinity=True,
                supports_nan=True,
                supports_zero=True
            ),
            'mxfp8': QuantizationLimits(  # FP8-E4M3
                max_positive_normal=1.75 * 2**8,  # 448
                min_positive_normal=2**-6,  # 0.015625
                max_positive_denormal=1.75 * 2**-7,  # 0.0013671875
                min_positive_denormal=2**-9,  # 0.001953125
                exponent_range=(-6, 8),
                exponent_range_with_denormal=(-9, 8),
                supports_infinity=False,
                supports_nan=True,
                supports_zero=True
            ),
            'mxfp4': QuantizationLimits(  # FP4-E2M1: 1位符号 + 2位指数 + 1位尾数
                max_positive_normal=1.5 * 2**3,  # 12 (最大指数=3, 尾数=1.5)
                min_positive_normal=2**-2,  # 0.25 (最小指数=-2)
                max_positive_denormal=1.5 * 2**-3,  # 0.1875 (非正常数最大值)
                min_positive_denormal=2**-4,  # 0.0625 (非正常数最小值)
                exponent_range=(-2, 3),
                exponent_range_with_denormal=(-4, 3),
                supports_infinity=False,
                supports_nan=False,
                supports_zero=True
            )
        }
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.subdirs = {
            'overflow_analysis': self.output_dir / 'overflow_analysis',
            'statistics': self.output_dir / 'statistics',
            'detailed_reports': self.output_dir / 'detailed_reports'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def parse_filename(self, filename: str) -> Optional[Dict]:
        """解析文件名，提取量化类型、层数、样本等信息"""
        try:
            parts = filename.split('_')
            
            if len(parts) < 8:
                return None
            
            # 查找量化类型
            quant_type = None
            for qtype in self.quant_types:
                if qtype in parts:
                    quant_type = qtype
                    break
            
            if not quant_type:
                return None
            
            # 查找层数
            layer_match = re.search(r'L(\d+)', filename)
            layer = int(layer_match.group(1)) if layer_match else None
            
            # 查找样本
            sample_match = re.search(r'sample(\d+)', filename)
            sample = int(sample_match.group(1)) if sample_match else None
            
            # 查找层类型
            layer_type = None
            if 'attention' in filename:
                layer_type = 'attention'
            elif 'linear' in filename:
                layer_type = 'linear'
            
            # 查找操作类型
            operation = None
            if 'forward' in filename:
                operation = 'forward'
            elif 'backward' in filename:
                operation = 'backward'
            
            # 查找tensor名称
            tensor_name = parts[-1].replace('.pt', '')
            
            return {
                'quant_type': quant_type,
                'layer': layer,
                'sample': sample,
                'layer_type': layer_type,
                'operation': operation,
                'tensor_name': tensor_name,
                'filename': filename
            }
        except Exception as e:
            print(f"解析文件名失败: {filename}, 错误: {e}")
            return None
    
    def load_tensor_values(self, file_info: Dict) -> Optional[np.ndarray]:
        """加载tensor数值"""
        try:
            tensor = torch.load(file_info['file_path'], map_location='cpu')
            if isinstance(tensor, torch.Tensor):
                return tensor.numpy()
            return None
        except Exception as e:
            print(f"加载tensor失败: {file_info['filename']}, 错误: {e}")
            return None
    
    def detect_overflow(self, tensor_values: np.ndarray, quant_type: str) -> Dict:
        """检测tensor溢出情况"""
        if quant_type not in self.quantization_limits:
            return {'error': f'不支持的量化类型: {quant_type}'}
        
        limits = self.quantization_limits[quant_type]
        
        # 移除NaN和Inf值进行统计
        finite_values = tensor_values[np.isfinite(tensor_values)]
        
        if len(finite_values) == 0:
            return {
                'quant_type': quant_type,
                'total_values': len(tensor_values),
                'finite_values': 0,
                'nan_count': np.sum(np.isnan(tensor_values)),
                'inf_count': np.sum(np.isinf(tensor_values)),
                'min_value': np.nan,
                'max_value': np.nan,
                'mean_value': np.nan,
                'std_value': np.nan,
                'overflow_upper': 0,
                'overflow_lower': 0,
                'underflow_upper': 0,
                'underflow_lower': 0,
                'overflow_percentage': 0.0,
                'underflow_percentage': 0.0
            }
        
        # 计算基本统计信息
        min_val = np.min(finite_values)
        max_val = np.max(finite_values)
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values)
        
        # 检测上溢出（超过最大正常值）
        overflow_upper = np.sum(finite_values > limits.max_positive_normal)
        
        # 检测下溢出（小于最小正常值）
        underflow_upper = np.sum(finite_values < limits.min_positive_normal)
        
        # 检测极值溢出（超过最大非正常值）
        overflow_lower = np.sum(finite_values > limits.max_positive_denormal)
        
        # 检测极值下溢出（小于最小非正常值）
        underflow_lower = np.sum(finite_values < limits.min_positive_denormal)
        
        # 计算溢出百分比
        total_finite = len(finite_values)
        overflow_percentage = (overflow_upper / total_finite) * 100 if total_finite > 0 else 0.0
        underflow_percentage = (underflow_upper / total_finite) * 100 if total_finite > 0 else 0.0
        
        return {
            'quant_type': quant_type,
            'total_values': len(tensor_values),
            'finite_values': len(finite_values),
            'nan_count': np.sum(np.isnan(tensor_values)),
            'inf_count': np.sum(np.isinf(tensor_values)),
            'min_value': min_val,
            'max_value': max_val,
            'mean_value': mean_val,
            'std_value': std_val,
            'overflow_upper': overflow_upper,
            'overflow_lower': overflow_lower,
            'underflow_upper': underflow_upper,
            'underflow_lower': underflow_lower,
            'overflow_percentage': overflow_percentage,
            'underflow_percentage': underflow_percentage,
            'limits': {
                'max_positive_normal': limits.max_positive_normal,
                'min_positive_normal': limits.min_positive_normal,
                'max_positive_denormal': limits.max_positive_denormal,
                'min_positive_denormal': limits.min_positive_denormal
            }
        }
    
    def analyze_tensor_file(self, file_info: Dict) -> Optional[Dict]:
        """分析单个tensor文件"""
        try:
            tensor_values = self.load_tensor_values(file_info)
            if tensor_values is None:
                return None
            
            overflow_info = self.detect_overflow(tensor_values, file_info['quant_type'])
            overflow_info.update(file_info)
            
            return overflow_info
        except Exception as e:
            print(f"分析文件失败: {file_info['filename']}, 错误: {e}")
            return None
    
    def load_all_tensor_data(self) -> List[Dict]:
        """加载所有tensor数据"""
        print("正在扫描tensor文件...")
        
        all_files = []
        
        # 扫描所有tensor文件
        for quant_type in self.quant_types:
            quant_dir = self.tensor_dir / quant_type
            if not quant_dir.exists():
                print(f"警告: 量化类型目录不存在: {quant_dir}")
                continue
            
            pt_files = list(quant_dir.glob('*.pt'))
            print(f"找到 {len(pt_files)} 个 {quant_type} 文件")
            
            for file_path in pt_files:
                file_info = self.parse_filename(file_path.name)
                if file_info:
                    file_info['file_path'] = file_path
                    all_files.append(file_info)
        
        print(f"总共找到 {len(all_files)} 个tensor文件")
        return all_files
    
    def analyze_all_tensors(self, file_list: List[Dict]) -> List[Dict]:
        """分析所有tensor文件"""
        print("开始分析tensor溢出情况...")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.analyze_tensor_file, file_info): file_info 
                for file_info in file_list
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        if len(results) % 100 == 0:
                            print(f"已分析 {len(results)} 个文件...")
                except Exception as e:
                    print(f"分析文件失败: {file_info['filename']}, 错误: {e}")
        
        print(f"分析完成，共处理 {len(results)} 个文件")
        return results
    
    def generate_overflow_summary(self, results: List[Dict]) -> Dict:
        """生成溢出情况汇总"""
        print("生成溢出情况汇总...")
        
        summary = {
            'total_files': len(results),
            'by_quant_type': {},
            'by_sample': {},
            'by_layer': {},
            'by_layer_type': {},
            'overall_stats': {
                'total_overflow_upper': 0,
                'total_overflow_lower': 0,
                'total_underflow_upper': 0,
                'total_underflow_lower': 0,
                'total_values': 0,
                'total_finite_values': 0,
                'total_nan_count': 0,
                'total_inf_count': 0
            }
        }
        
        # 按量化类型汇总
        for quant_type in self.quant_types:
            quant_results = [r for r in results if r['quant_type'] == quant_type]
            if quant_results:
                summary['by_quant_type'][quant_type] = self._summarize_results(quant_results)
        
        # 按样本汇总
        for sample in self.samples:
            sample_results = [r for r in results if r.get('sample') == sample]
            if sample_results:
                summary['by_sample'][sample] = self._summarize_results(sample_results)
        
        # 按层汇总
        for layer in self.layers:
            layer_results = [r for r in results if r.get('layer') == layer]
            if layer_results:
                summary['by_layer'][layer] = self._summarize_results(layer_results)
        
        # 按层类型汇总
        for layer_type in ['attention', 'linear']:
            layer_type_results = [r for r in results if r.get('layer_type') == layer_type]
            if layer_type_results:
                summary['by_layer_type'][layer_type] = self._summarize_results(layer_type_results)
        
        # 总体统计
        summary['overall_stats'] = self._summarize_results(results)
        
        return summary
    
    def _summarize_results(self, results: List[Dict]) -> Dict:
        """汇总结果"""
        if not results:
            return {}
        
        total_files = len(results)
        total_values = sum(r['total_values'] for r in results)
        total_finite = sum(r['finite_values'] for r in results)
        total_nan = sum(r['nan_count'] for r in results)
        total_inf = sum(r['inf_count'] for r in results)
        
        total_overflow_upper = sum(r['overflow_upper'] for r in results)
        total_overflow_lower = sum(r['overflow_lower'] for r in results)
        total_underflow_upper = sum(r['underflow_upper'] for r in results)
        total_underflow_lower = sum(r['underflow_lower'] for r in results)
        
        # 计算平均值
        avg_min = np.mean([r['min_value'] for r in results if not np.isnan(r['min_value'])])
        avg_max = np.mean([r['max_value'] for r in results if not np.isnan(r['max_value'])])
        avg_mean = np.mean([r['mean_value'] for r in results if not np.isnan(r['mean_value'])])
        avg_std = np.mean([r['std_value'] for r in results if not np.isnan(r['std_value'])])
        
        return {
            'total_files': total_files,
            'total_values': total_values,
            'total_finite_values': total_finite,
            'total_nan_count': total_nan,
            'total_inf_count': total_inf,
            'total_overflow_upper': total_overflow_upper,
            'total_overflow_lower': total_overflow_lower,
            'total_underflow_upper': total_underflow_upper,
            'total_underflow_lower': total_underflow_lower,
            'overflow_upper_percentage': (total_overflow_upper / total_finite * 100) if total_finite > 0 else 0,
            'overflow_lower_percentage': (total_overflow_lower / total_finite * 100) if total_finite > 0 else 0,
            'underflow_upper_percentage': (total_underflow_upper / total_finite * 100) if total_finite > 0 else 0,
            'underflow_lower_percentage': (total_underflow_lower / total_finite * 100) if total_finite > 0 else 0,
            'avg_min_value': avg_min,
            'avg_max_value': avg_max,
            'avg_mean_value': avg_mean,
            'avg_std_value': avg_std
        }
    
    def plot_overflow_analysis(self, results: List[Dict], summary: Dict):
        """绘制溢出分析图"""
        print("绘制溢出分析图...")
        
        # 创建大图
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 量化类型溢出比较
        ax1 = fig.add_subplot(gs[0, 0])
        quant_types = list(summary['by_quant_type'].keys())
        overflow_upper = [summary['by_quant_type'][qt]['overflow_upper_percentage'] for qt in quant_types]
        underflow_upper = [summary['by_quant_type'][qt]['underflow_upper_percentage'] for qt in quant_types]
        
        x = np.arange(len(quant_types))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, overflow_upper, width, label='上溢出 (%)', alpha=0.8, color='red')
        bars2 = ax1.bar(x + width/2, underflow_upper, width, label='下溢出 (%)', alpha=0.8, color='blue')
        
        ax1.set_title('各量化类型溢出情况', fontweight='bold')
        ax1.set_ylabel('溢出百分比 (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(quant_types)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars1, overflow_upper):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
        for bar, val in zip(bars2, underflow_upper):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. 样本溢出比较
        ax2 = fig.add_subplot(gs[0, 1])
        samples = list(summary['by_sample'].keys())
        if samples:
            sample_overflow = [summary['by_sample'][s]['overflow_upper_percentage'] for s in samples]
            sample_underflow = [summary['by_sample'][s]['underflow_upper_percentage'] for s in samples]
            
            x = np.arange(len(samples))
            bars3 = ax2.bar(x - width/2, sample_overflow, width, label='上溢出 (%)', alpha=0.8, color='red')
            bars4 = ax2.bar(x + width/2, sample_underflow, width, label='下溢出 (%)', alpha=0.8, color='blue')
            
            ax2.set_title('各样本溢出情况', fontweight='bold')
            ax2.set_ylabel('溢出百分比 (%)')
            ax2.set_xlabel('样本编号')
            ax2.set_xticks(x)
            ax2.set_xticklabels(samples)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 层溢出比较
        ax3 = fig.add_subplot(gs[0, 2])
        layers = list(summary['by_layer'].keys())
        if layers:
            layer_overflow = [summary['by_layer'][l]['overflow_upper_percentage'] for l in layers]
            layer_underflow = [summary['by_layer'][l]['underflow_upper_percentage'] for l in layers]
            
            ax3.plot(layers, layer_overflow, marker='o', label='上溢出 (%)', color='red', linewidth=2)
            ax3.plot(layers, layer_underflow, marker='s', label='下溢出 (%)', color='blue', linewidth=2)
            
            ax3.set_title('各层溢出情况', fontweight='bold')
            ax3.set_ylabel('溢出百分比 (%)')
            ax3.set_xlabel('层编号')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 量化类型数值范围比较
        ax4 = fig.add_subplot(gs[1, 0])
        quant_ranges = []
        quant_labels = []
        for quant_type in quant_types:
            if quant_type in summary['by_quant_type']:
                stats = summary['by_quant_type'][quant_type]
                quant_ranges.append([stats['avg_min_value'], stats['avg_max_value']])
                quant_labels.append(quant_type)
        
        if quant_ranges:
            for i, (ranges, label) in enumerate(zip(quant_ranges, quant_labels)):
                ax4.barh(i, ranges[1] - ranges[0], left=ranges[0], alpha=0.7, label=label)
                ax4.text(ranges[1], i, f'{ranges[1]:.2e}', ha='left', va='center')
                ax4.text(ranges[0], i, f'{ranges[0]:.2e}', ha='right', va='center')
        
        ax4.set_title('各量化类型数值范围', fontweight='bold')
        ax4.set_xlabel('数值范围')
        ax4.set_yticks(range(len(quant_labels)))
        ax4.set_yticklabels(quant_labels)
        ax4.set_xscale('log')
        
        # 5. 溢出分布热力图
        ax5 = fig.add_subplot(gs[1, 1])
        
        # 创建量化类型-样本溢出矩阵
        overflow_matrix = np.zeros((len(quant_types), len(samples)))
        for i, quant_type in enumerate(quant_types):
            for j, sample in enumerate(samples):
                if quant_type in summary['by_quant_type'] and sample in summary['by_sample']:
                    # 计算该量化类型和样本组合的溢出率
                    quant_sample_results = [r for r in results 
                                          if r['quant_type'] == quant_type and r.get('sample') == sample]
                    if quant_sample_results:
                        total_overflow = sum(r['overflow_upper_percentage'] for r in quant_sample_results)
                        overflow_matrix[i, j] = total_overflow / len(quant_sample_results)
        
        im = ax5.imshow(overflow_matrix, cmap='Reds', aspect='auto')
        ax5.set_title('量化类型-样本溢出热力图', fontweight='bold')
        ax5.set_xlabel('样本编号')
        ax5.set_ylabel('量化类型')
        ax5.set_xticks(range(len(samples)))
        ax5.set_xticklabels(samples)
        ax5.set_yticks(range(len(quant_types)))
        ax5.set_yticklabels(quant_types)
        
        # 添加数值标签
        for i in range(len(quant_types)):
            for j in range(len(samples)):
                text = ax5.text(j, i, f'{overflow_matrix[i, j]:.1f}%',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax5, label='平均溢出率 (%)')
        
        # 6. 层类型溢出比较
        ax6 = fig.add_subplot(gs[1, 2])
        layer_types = list(summary['by_layer_type'].keys())
        if layer_types:
            lt_overflow = [summary['by_layer_type'][lt]['overflow_upper_percentage'] for lt in layer_types]
            lt_underflow = [summary['by_layer_type'][lt]['underflow_upper_percentage'] for lt in layer_types]
            
            x = np.arange(len(layer_types))
            bars5 = ax6.bar(x - width/2, lt_overflow, width, label='上溢出 (%)', alpha=0.8, color='red')
            bars6 = ax6.bar(x + width/2, lt_underflow, width, label='下溢出 (%)', alpha=0.8, color='blue')
            
            ax6.set_title('各层类型溢出情况', fontweight='bold')
            ax6.set_ylabel('溢出百分比 (%)')
            ax6.set_xticks(x)
            ax6.set_xticklabels(layer_types)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. 溢出统计汇总
        ax7 = fig.add_subplot(gs[2, :])
        
        # 创建溢出统计表格
        overflow_stats = []
        for quant_type in quant_types:
            if quant_type in summary['by_quant_type']:
                stats = summary['by_quant_type'][quant_type]
                overflow_stats.append([
                    quant_type,
                    f"{stats['total_files']:,}",
                    f"{stats['total_values']:,}",
                    f"{stats['overflow_upper_percentage']:.2f}%",
                    f"{stats['underflow_upper_percentage']:.2f}%",
                    f"{stats['avg_min_value']:.2e}",
                    f"{stats['avg_max_value']:.2e}"
                ])
        
        # 创建表格
        table_data = [['量化类型', '文件数', '总数值数', '上溢出率', '下溢出率', '平均最小值', '平均最大值']]
        table_data.extend(overflow_stats)
        
        ax7.axis('tight')
        ax7.axis('off')
        table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置表头样式
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax7.set_title('溢出统计汇总表', fontweight='bold', pad=20)
        
        plt.suptitle('Tensor溢出检测分析报告', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(self.subdirs['overflow_analysis'] / 'overflow_analysis_report.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"溢出分析图已保存: {self.subdirs['overflow_analysis'] / 'overflow_analysis_report.png'}")
    
    def save_detailed_report(self, results: List[Dict], summary: Dict):
        """保存详细报告"""
        print("保存详细报告...")
        
        report_path = self.subdirs['detailed_reports'] / 'overflow_detection_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Tensor溢出检测详细报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 量化类型限制值
            f.write("量化类型限制值:\n")
            f.write("-" * 50 + "\n")
            for quant_type, limits in self.quantization_limits.items():
                f.write(f"{quant_type}:\n")
                f.write(f"  最大正常值: {limits.max_positive_normal:.6e}\n")
                f.write(f"  最小正常值: {limits.min_positive_normal:.6e}\n")
                f.write(f"  最大非正常值: {limits.max_positive_denormal:.6e}\n")
                f.write(f"  最小非正常值: {limits.min_positive_denormal:.6e}\n")
                f.write(f"  指数范围: {limits.exponent_range}\n")
                f.write(f"  支持无穷大: {limits.supports_infinity}\n")
                f.write(f"  支持NaN: {limits.supports_nan}\n\n")
            
            # 总体统计
            f.write("总体统计:\n")
            f.write("-" * 50 + "\n")
            overall = summary['overall_stats']
            f.write(f"总文件数: {overall['total_files']:,}\n")
            f.write(f"总数值数: {overall['total_values']:,}\n")
            f.write(f"有限数值数: {overall['total_finite_values']:,}\n")
            f.write(f"NaN数量: {overall['total_nan_count']:,}\n")
            f.write(f"无穷大数量: {overall['total_inf_count']:,}\n")
            f.write(f"上溢出数量: {overall['total_overflow_upper']:,}\n")
            f.write(f"下溢出数量: {overall['total_underflow_upper']:,}\n")
            f.write(f"上溢出率: {overall['overflow_upper_percentage']:.2f}%\n")
            f.write(f"下溢出率: {overall['underflow_upper_percentage']:.2f}%\n\n")
            
            # 按量化类型统计
            f.write("按量化类型统计:\n")
            f.write("-" * 50 + "\n")
            for quant_type, stats in summary['by_quant_type'].items():
                f.write(f"{quant_type}:\n")
                f.write(f"  文件数: {stats['total_files']:,}\n")
                f.write(f"  总数值数: {stats['total_values']:,}\n")
                f.write(f"  上溢出率: {stats['overflow_upper_percentage']:.2f}%\n")
                f.write(f"  下溢出率: {stats['underflow_upper_percentage']:.2f}%\n")
                f.write(f"  平均最小值: {stats['avg_min_value']:.6e}\n")
                f.write(f"  平均最大值: {stats['avg_max_value']:.6e}\n\n")
            
            # 按样本统计
            f.write("按样本统计:\n")
            f.write("-" * 50 + "\n")
            for sample, stats in summary['by_sample'].items():
                f.write(f"Sample {sample}:\n")
                f.write(f"  文件数: {stats['total_files']:,}\n")
                f.write(f"  上溢出率: {stats['overflow_upper_percentage']:.2f}%\n")
                f.write(f"  下溢出率: {stats['underflow_upper_percentage']:.2f}%\n\n")
            
            # 按层统计
            f.write("按层统计:\n")
            f.write("-" * 50 + "\n")
            for layer, stats in summary['by_layer'].items():
                f.write(f"Layer {layer}:\n")
                f.write(f"  文件数: {stats['total_files']:,}\n")
                f.write(f"  上溢出率: {stats['overflow_upper_percentage']:.2f}%\n")
                f.write(f"  下溢出率: {stats['underflow_upper_percentage']:.2f}%\n\n")
        
        print(f"详细报告已保存: {report_path}")
    
    def run_analysis(self):
        """运行溢出检测分析"""
        print("开始Tensor溢出检测分析...")
        print(f"Tensor目录: {self.tensor_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"最大线程数: {self.max_workers}")
        
        # 加载数据
        file_list = self.load_all_tensor_data()
        if not file_list:
            print("错误: 没有找到tensor文件")
            return
        
        # 分析所有tensor
        results = self.analyze_all_tensors(file_list)
        if not results:
            print("错误: 没有成功分析任何文件")
            return
        
        # 生成汇总
        summary = self.generate_overflow_summary(results)
        
        # 绘制分析图
        self.plot_overflow_analysis(results, summary)
        
        # 保存详细报告
        self.save_detailed_report(results, summary)
        
        print("\n" + "=" * 60)
        print("Tensor溢出检测分析完成!")
        print("=" * 60)
        print(f"输出目录: {self.output_dir}")
        print("生成的文件:")
        print(f"  - 溢出分析图: {self.subdirs['overflow_analysis'] / 'overflow_analysis_report.png'}")
        print(f"  - 详细报告: {self.subdirs['detailed_reports'] / 'overflow_detection_report.txt'}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Tensor溢出检测分析工具')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor文件目录')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='输出目录')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='最大线程数')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = OverflowDetectionAnalyzer(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # 运行分析
    analyzer.run_analysis()

if __name__ == "__main__":
    main()

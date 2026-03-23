"""
LLM Training Profile Analyzer

Analyzes torch profiler timeline traces to extract module-level timing statistics
for transformer layers, including attention, MLP, loss, and MTP modules.
"""

import sys
import os
import glob as glob_module
import argparse
import json
import re
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import pandas as pd
import numpy as np

# Import perfetto directly
from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig

from module_flops import (
    mla_layer_flops,
    lighting_linear_layer_flops,
    KDA_linear_layer_flops,
    moe_layer_flops,
    mlp_layer_flops,
    moe_expert_flops,
    shared_expert_flops,
    router_flops,
    loss_flops,
)
from module_bandwidth import (
    token_dispatch_comm_volume,
    token_combine_comm_volume,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ModuleInfo:
    """Information about a module extracted from event name"""
    module_type: str  # 'attention', 'mlp', 'loss', 'mtp', 'unknown'
    subtype: str      # 'kda', 'mla', 'dense', 'moe', etc.
    layer_num: Optional[int]
    is_backward: bool


@dataclass
class TimeInfo:
    """Timing information for a module"""
    wall_time_ms: float
    launch_delay_ms: float
    kernel_times_ms: List[float]
    stream_ids: List[int]
    stream_execution_times: Dict[int, float]
    num_kernels: int


@dataclass
class ModuleStat:
    """Statistics for a single module instance"""
    module_type: str
    subtype: str
    layer_num: Optional[int]
    is_backward: bool
    wall_time_ms: float
    launch_delay_ms: float
    num_kernels: int
    ts: float
    dur: float
    microbatch_id: int = 0
    iteration_id: int = 0


@dataclass
class ModuleConfig:
    """Configuration for a specific module type"""
    name: str                    # Internal name (e.g., 'KimiDeltaAttention')
    sql_pattern: str             # SQL LIKE pattern (e.g., '%::KimiDeltaAttention')
    display_name: str = None     # Optional display name
    category: str = 'user_annotation'  # Trace event category to search in

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.name


@dataclass
class AnalysisConfig:
    """Centralized configuration for all analysis modules"""

    # Module-level analysis targets
    module_configs: List[ModuleConfig] = None

    # VPP schedule analysis targets
    vpp_event_configs: List[ModuleConfig] = None

    def __post_init__(self):
        if self.module_configs is None:
            # Order matters: more specific sub-modules must come before their
            # parent modules, because classify_module returns the first match.
            self.module_configs = [
                # MoE sub-modules (must precede MoELayer)
                ModuleConfig('TEGroupedMLP', '%mlp.experts::TEGroupedMLP'),
                ModuleConfig('SharedExpertMLP', '%shared_experts::SharedExpertMLP'),
                ModuleConfig('TopKRouter', '%router::TopKRouter'),
                ModuleConfig('FusedDispatch', 'FusedDispatch', category='cpu_op'),
                ModuleConfig('FusedCombine', 'FusedCombine', category='cpu_op'),
                # Top-level modules
                ModuleConfig('KimiDeltaAttention', '%::KimiDeltaAttention'),
                ModuleConfig('MLASelfAttention', '%::MLASelfAttention'),
                ModuleConfig('MoELayer', '%::MoELayer'),
                ModuleConfig('MLP', '%::MLP'),
                ModuleConfig('MultiTokenPredictionBlock', '%::MultiTokenPredictionBlock'),
                #ModuleConfig('loss_func', '%loss_func%', 'loss_function'),
            ]

        if self.vpp_event_configs is None:
            self.vpp_event_configs = [
                ModuleConfig('forward_step', '%forward_step%'),
                ModuleConfig('backward_step', '%backward_step%'),
                ModuleConfig('p2p_send', '%p2p%'),  # Will need special handling
                ModuleConfig('p2p_recv', '%p2p%'),  # Will need special handling
                ModuleConfig('optimizer', '%ChainedOptimizer.step%'),
                ModuleConfig('loss_postprocessing', '%loss_postprocessing%'),
                ModuleConfig('training_log', '%AtorchMegatronEngine.training_log%'),
            ]

    def get_module_names(self) -> List[str]:
        """Get list of module names"""
        return [m.name for m in self.module_configs]

    def get_vpp_event_names(self) -> List[str]:
        """Get list of VPP event names"""
        return [m.name for m in self.vpp_event_configs]

    def get_module_by_name(self, name: str) -> Optional[ModuleConfig]:
        """Get module config by name"""
        for m in self.module_configs:
            if m.name == name:
                return m
        return None

    def classify_module(self, event_name: str) -> str:
        """Classify module type from event name"""
        for config in self.module_configs:
            # Convert SQL LIKE pattern to Python check
            pattern = config.sql_pattern.strip('%')
            if pattern in event_name:
                return config.name
        return 'other'

    def classify_vpp_event(self, event_name: str) -> str:
        """Classify VPP event type from event name"""
        name_lower = event_name.lower()

        # Special handling for p2p events
        if 'p2p' in name_lower:
            if 'send' in name_lower:
                return 'p2p_send'
            elif 'recv' in name_lower:
                return 'p2p_recv'
            else:
                return 'p2p'

        # Check other patterns
        for config in self.vpp_event_configs:
            if config.name in ['p2p_send', 'p2p_recv']:
                continue  # Already handled above
            pattern = config.sql_pattern.strip('%').lower()
            if pattern in name_lower:
                return config.name

        return 'other'


# ============================================================================
# Model FLOPs Calculator
# ============================================================================

class ModelFlopsCalculator:
    """Calculates theoretical forward FLOPs for each module type based on model config."""

    def __init__(self, config_path: str):
        """
        Load model config from YAML and extract parameters needed for FLOPs calculation.

        Args:
            config_path: Path to model_config.yaml
        """
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        model_cfg = raw_config.get('model', {}).get('config', {})
        extra_cfg = raw_config.get('trainer', {}).get('args', {}).get('extra_configs', {})

        # Basic model parameters
        self.hidden_size = model_cfg['hidden_size']
        self.num_attention_heads = model_cfg['num_attention_heads']
        self.num_query_groups = model_cfg.get('num_query_groups', self.num_attention_heads)
        self.seq_len = model_cfg['seq_length']
        self.batch_size = extra_cfg.get('micro_batch_size', model_cfg.get('micro_batch_size', 1))
        self.vocab_size = model_cfg.get('vocab_size', 0)

        # Attention type detection
        self.linear_attn_type = model_cfg.get('linear_attn_type', None)
        self.multi_latent_attention = extra_cfg.get('multi_latent_attention', False)

        # KDA / Linear attention parameters
        self.kv_channels = model_cfg.get('kv_channels', self.hidden_size // self.num_attention_heads)
        self.no_kda_lora = model_cfg.get('no_kda_lora', True)
        self.gqa = extra_cfg.get('group_query_attention', False)

        # MLA parameters
        self.qk_head_dim = model_cfg.get('qk_head_dim', 128)
        self.v_head_dim = model_cfg.get('v_head_dim', 128)
        self.qk_pos_emb_head_dim = model_cfg.get('qk_pos_emb_head_dim', 64)
        self.kv_lora_rank = model_cfg.get('kv_lora_rank', 512)
        self.q_lora_rank = model_cfg.get('q_lora_rank', None)

        # MoE parameters
        self.num_experts = model_cfg.get('num_experts', 1)
        self.moe_ffn_hidden_size = model_cfg.get('moe_ffn_hidden_size', 0)
        self.moe_router_topk = model_cfg.get('moe_router_topk', 1)
        self.moe_shared_expert_intermediate_size = model_cfg.get('moe_shared_expert_intermediate_size', 0)

        # MLP parameters
        self.ffn_hidden_size = model_cfg.get('ffn_hidden_size', int(self.hidden_size * 4))
        self.swiglu = extra_cfg.get('swiglu', False)

        # Dtype for bandwidth calculation
        self.dtype_bytes = 2  # bf16

        logger.info(f"ModelFlopsCalculator loaded: hidden_size={self.hidden_size}, "
                    f"seq_len={self.seq_len}, batch_size={self.batch_size}, "
                    f"attn_type={self.linear_attn_type or ('mla' if self.multi_latent_attention else 'standard')}")

    def _get_attention_flops(self) -> float:
        """Get FLOPs for the attention module based on detected type."""
        if self.multi_latent_attention:
            return mla_layer_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                qk_head_dim=self.qk_head_dim,
                v_head_dim=self.v_head_dim,
                qk_pos_emb_head_dim=self.qk_pos_emb_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
            )
        elif self.linear_attn_type == 'kda':
            return KDA_linear_layer_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                gqa=self.gqa,
                num_query_groups=self.num_query_groups,
                kv_channels=self.kv_channels,
                no_kda_lora=self.no_kda_lora,
            )
        else:
            return lighting_linear_layer_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                gqa=self.gqa,
                num_query_groups=self.num_query_groups,
                kv_channels=self.kv_channels,
            )

    def get_module_flops(self, module_name: str) -> Optional[float]:
        """
        Get theoretical forward FLOPs for a given module name.

        Args:
            module_name: Module name from AnalysisConfig (e.g., 'KimiDeltaAttention', 'MoELayer')

        Returns:
            FLOPs count (float), or None if unknown module
        """
        if module_name in ('KimiDeltaAttention', 'MLASelfAttention'):
            return self._get_attention_flops()

        elif module_name == 'MoELayer':
            return moe_layer_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                moe_ffn_hidden_size=self.moe_ffn_hidden_size,
                moe_router_topk=self.moe_router_topk,
                moe_shared_expert_intermediate_size=self.moe_shared_expert_intermediate_size,
                swiglu=self.swiglu,
            )

        elif module_name == 'MLP':
            expansion = self.ffn_hidden_size / self.hidden_size
            return mlp_layer_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                expansion=expansion,
                swiglu=self.swiglu,
            )

        elif module_name == 'MultiTokenPredictionBlock':
            # MTP = attention + MoE + logits computation
            attn_flops = self._get_attention_flops()
            moe_flops_val = moe_layer_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                moe_ffn_hidden_size=self.moe_ffn_hidden_size,
                moe_router_topk=self.moe_router_topk,
                moe_shared_expert_intermediate_size=self.moe_shared_expert_intermediate_size,
                swiglu=self.swiglu,
            )
            # logits computation: output projection 2 * B * S * H * V
            logits_flops = 2 * self.batch_size * self.seq_len * self.hidden_size * self.vocab_size
            return attn_flops + moe_flops_val + logits_flops

        elif module_name == 'TEGroupedMLP':
            return moe_expert_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                moe_ffn_hidden_size=self.moe_ffn_hidden_size,
                moe_router_topk=self.moe_router_topk,
                swiglu=self.swiglu,
            )

        elif module_name == 'SharedExpertMLP':
            return shared_expert_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                moe_shared_expert_intermediate_size=self.moe_shared_expert_intermediate_size,
                swiglu=self.swiglu,
            )

        elif module_name == 'TopKRouter':
            return router_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
            )

        elif module_name == 'loss_func':
            return loss_flops(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
            )

        else:
            return None

    def get_module_comm_volume(self, module_name: str) -> Optional[float]:
        """
        Get communication volume in bytes for bandwidth-type modules.

        Args:
            module_name: Module name (e.g., 'FusedDispatch', 'FusedCombine')

        Returns:
            Communication volume in bytes, or None if not a bandwidth module
        """
        if module_name == 'FusedDispatch':
            return token_dispatch_comm_volume(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                topk=self.moe_router_topk,
                dtype_bytes=self.dtype_bytes,
            )
        elif module_name == 'FusedCombine':
            return token_combine_comm_volume(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                topk=self.moe_router_topk,
                dtype_bytes=self.dtype_bytes,
            )
        return None


# ============================================================================
# Module Identifier
# ============================================================================

class ModuleIdentifier:
    """Identifies and classifies transformer modules from event names"""

    # Pattern matching rules for module identification
    ATTENTION_PATTERNS = {
        'kda': r'(?i)kimi.*delta.*attention|delta_attn|KimiDeltaAttention',
        'mla': r'(?i)mla|multi.*latent.*attention',
        'standard': r'(?i)self_attention|core_attention',
        'gqa': r'(?i)group.*query.*attention',
    }

    MLP_PATTERNS = {
        'moe': r'(?i)moe|mixture.*expert|router.*expert|expert',
        'dense': r'(?i)(?<!_)mlp(?!.*moe)|feed.*forward|dense.*mlp',
    }

    LOSS_PATTERNS = {
        'cross_entropy': r'(?i)cross.*entropy|nll_loss',
        'custom': r'(?i)loss.*function|compute.*loss',
    }

    MTP_PATTERNS = {
        'mtp': r'(?i)multi.*token.*prediction|mtp',
    }

    # Layer number extraction pattern
    LAYER_NUMBER_PATTERN = r'\.layers?\.(\d+)[\.\:]|layer[_\s](\d+)|layers\[(\d+)\]'

    # Backward pass indicators
    BACKWARD_INDICATORS = ['Backward::', 'autograd::engine::evaluate_function:']

    def identify_module_type(self, event_name: str) -> ModuleInfo:
        """
        Identify module type from event name

        Priority order:
        1. Detect backward pass
        2. Check MTP (specific modules)
        3. Check Loss
        4. Check Attention (with subtype)
        5. Check MLP (with subtype)
        """
        # Step 1: Detect backward pass
        is_backward = any(indicator in event_name for indicator in self.BACKWARD_INDICATORS)

        # Step 2: Extract layer number
        layer_num = self.extract_layer_number(event_name)

        # Step 3: Classify module type
        module_type = 'unknown'
        subtype = 'unknown'

        # Check each pattern category
        if self._match_any(event_name, self.MTP_PATTERNS.values()):
            module_type = 'mtp'
            subtype = self._find_matching_subtype(event_name, self.MTP_PATTERNS)
        elif self._match_any(event_name, self.LOSS_PATTERNS.values()):
            module_type = 'loss'
            subtype = self._find_matching_subtype(event_name, self.LOSS_PATTERNS)
        elif self._match_any(event_name, self.ATTENTION_PATTERNS.values()):
            module_type = 'attention'
            subtype = self._find_matching_subtype(event_name, self.ATTENTION_PATTERNS)
        elif self._match_any(event_name, self.MLP_PATTERNS.values()):
            module_type = 'mlp'
            subtype = self._find_matching_subtype(event_name, self.MLP_PATTERNS)

        return ModuleInfo(
            module_type=module_type,
            subtype=subtype,
            layer_num=layer_num,
            is_backward=is_backward
        )

    def extract_layer_number(self, event_name: str) -> Optional[int]:
        """Extract layer number from event name"""
        match = re.search(self.LAYER_NUMBER_PATTERN, event_name)
        if match:
            # Try each captured group
            for group in match.groups():
                if group is not None:
                    return int(group)
        return None

    def _match_any(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the patterns"""
        return any(re.search(pattern, text) for pattern in patterns)

    def _find_matching_subtype(self, text: str, pattern_dict: Dict[str, str]) -> str:
        """Find which subtype pattern matches the text"""
        for subtype, pattern in pattern_dict.items():
            if re.search(pattern, text):
                return subtype
        return 'unknown'


# ============================================================================
# Event Correlator
# ============================================================================

class EventCorrelator:
    """Correlates CPU and GPU events, calculates wall times"""

    def __init__(self, kernel_df: pd.DataFrame, tp=None):
        """
        Initialize with kernel dataframe from PerfettoParser

        Args:
            kernel_df: DataFrame with correlation IDs already joined (sort_kernel_df)
            tp: Optional TraceProcessor instance for hierarchical queries
        """
        self.kernel_df = kernel_df
        self.tp = tp

    def find_child_kernels(self, parent_event: pd.Series, tp=None, use_hierarchy=True) -> pd.DataFrame:
        """
        Find all GPU kernels that belong to a parent CPU module

        This method supports two matching strategies:
        1. Hierarchical matching (use_hierarchy=True): Uses track_id + depth filtering
           - Best for nested modules (attention, MLP) within forward/backward
           - Respects parent-child hierarchy in trace
        2. Time-range matching (use_hierarchy=False): Only uses time range
           - Best for top-level operations (forward_step, backward_step)
           - Backward pass may spawn kernels on different tracks (autograd engine)

        Args:
            parent_event: Series containing parent event info (ts, dur, track_id, depth, etc.)
            tp: Optional TraceProcessor instance for SQL queries (needed for hierarchy-based matching)
            use_hierarchy: Whether to use track_id+depth filtering (default: True)

        Returns:
            DataFrame of child kernels (subset of kernel_df with matching correlation_ids)
        """
        # If TraceProcessor is available, use SQL-based correlation matching
        if tp is not None and 'ts' in parent_event and 'dur' in parent_event:
            try:
                # Build SQL query based on matching strategy
                if use_hierarchy and 'track_id' in parent_event and 'depth' in parent_event:
                    # Hierarchical matching: filter by track_id + depth + time
                    child_sql = f"""
                    SELECT args_1.display_value AS correlation_id
                    FROM slice
                    LEFT JOIN args AS args_1 ON args_1.arg_set_id = slice.arg_set_id
                      AND args_1.key = 'args.correlation'
                    WHERE slice.category IN ('cuda_runtime', 'cuda_driver')
                      AND slice.track_id = {parent_event['track_id']}
                      AND slice.depth > {parent_event['depth']}
                      AND slice.ts >= {parent_event['ts']}
                      AND slice.ts < {parent_event['ts'] + parent_event['dur']}
                      AND args_1.display_value IS NOT NULL
                    """
                else:
                    # Time-range matching: only filter by time (for VPP level events)
                    child_sql = f"""
                    SELECT args_1.display_value AS correlation_id
                    FROM slice
                    LEFT JOIN args AS args_1 ON args_1.arg_set_id = slice.arg_set_id
                      AND args_1.key = 'args.correlation'
                    WHERE slice.category IN ('cuda_runtime', 'cuda_driver')
                      AND slice.ts >= {parent_event['ts']}
                      AND slice.ts < {parent_event['ts'] + parent_event['dur']}
                      AND args_1.display_value IS NOT NULL
                    """

                cuda_children = tp.query(child_sql).as_pandas_dataframe()

                if not cuda_children.empty:
                    # Convert correlation_id to int for matching (they're strings in SQL result)
                    cuda_corr_ids = cuda_children['correlation_id'].astype(int).tolist()

                    # Match with kernel_df using correlation_id
                    matched_kernels = self.kernel_df[
                        self.kernel_df['correlation_id'].isin(cuda_corr_ids)
                    ].copy()

                    if not matched_kernels.empty:
                        return matched_kernels

            except Exception as e:
                # If SQL-based matching fails, fall back to time-based filtering
                logger.warning(f"SQL-based kernel matching failed: {e}, falling back to time-based filtering")

        # Fallback: time-based filtering using kernel_df directly
        parent_start = parent_event['ts']
        parent_end = parent_event['ts'] + parent_event['dur']
        parent_external_id = parent_event.get('external_id')

        # Filter by time range overlap
        mask = (self.kernel_df['ts_kernel'] >= parent_start) & \
               (self.kernel_df['ts_kernel'] < parent_end)

        # Further filter by external_id if available
        if parent_external_id is not None and not pd.isna(parent_external_id):
            try:
                # Convert to int for comparison
                parent_ext_id = int(parent_external_id) if isinstance(parent_external_id, str) else parent_external_id
                # Child external_id should be >= parent external_id
                external_id_col = 'external_id_kernel'
                if external_id_col in self.kernel_df.columns:
                    # Filter out NaN and convert to numeric
                    ext_id_mask = self.kernel_df[external_id_col].notna()
                    mask &= ext_id_mask & (self.kernel_df[external_id_col] >= parent_ext_id)
            except (ValueError, TypeError):
                # If conversion fails, skip external_id filtering
                pass

        return self.kernel_df[mask].copy()

    def calculate_wall_time(self, kernels_df: pd.DataFrame) -> TimeInfo:
        """
        Calculate wall time for a module (from first kernel to last kernel end)

        Args:
            kernels_df: DataFrame of kernels belonging to the module

        Returns:
            TimeInfo with wall time and other statistics (in milliseconds)
        """
        if kernels_df.empty:
            return TimeInfo(
                wall_time_ms=0.0,
                launch_delay_ms=0.0,
                kernel_times_ms=[],
                stream_ids=[],
                stream_execution_times={},
                num_kernels=0
            )

        # Calculate wall time (convert from nanoseconds to milliseconds)
        # Note: Perfetto trace stores times in nanoseconds
        min_start = kernels_df['ts_kernel'].min()
        max_end = (kernels_df['ts_kernel'] + kernels_df['dur_kernel']).max()
        wall_time_ns = max_end - min_start
        wall_time_ms = wall_time_ns / 1000000.0  # ns -> ms

        # Calculate average launch delay (convert from ns to ms)
        if 'kernel_delay' in kernels_df.columns:
            avg_launch_delay_ns = kernels_df['kernel_delay'].mean()
            avg_launch_delay_ms = avg_launch_delay_ns / 1000000.0  # ns -> ms
        else:
            avg_launch_delay_ms = 0.0

        # Get kernel times (convert from ns to ms)
        kernel_times_ms = (kernels_df['dur_kernel'] / 1000000.0).tolist()

        # Get stream information (convert from ns to ms)
        if 'stream' in kernels_df.columns:
            stream_ids = kernels_df['stream'].dropna().unique().tolist()

            # Calculate execution time per stream
            stream_times = {}
            for stream_id in stream_ids:
                stream_kernels = kernels_df[kernels_df['stream'] == stream_id]
                stream_time_ns = stream_kernels['dur_kernel'].sum()
                stream_times[int(stream_id)] = float(stream_time_ns / 1000000.0)  # ns -> ms
        else:
            stream_ids = []
            stream_times = {}

        return TimeInfo(
            wall_time_ms=float(wall_time_ms),
            launch_delay_ms=float(avg_launch_delay_ms),
            kernel_times_ms=kernel_times_ms,
            stream_ids=stream_ids,
            stream_execution_times=stream_times,
            num_kernels=len(kernels_df)
        )

    def get_event_kernel_timing(self, cpu_event: pd.Series, use_hierarchy=True) -> Dict[str, Any]:
        """
        Get comprehensive GPU kernel timing for a CPU-side event

        This is a high-level method that combines find_child_kernels() and
        calculate_wall_time() with additional bubble analysis.

        Args:
            cpu_event: pandas Series with required fields:
                - ts: timestamp (nanoseconds)
                - dur: duration (nanoseconds)
                - external_id: optional external ID for correlation
                - name: event name (for logging)
                - track_id: (optional) for hierarchical matching
                - depth: (optional) for hierarchical matching
            use_hierarchy: Whether to use track_id+depth filtering (default: True)
                - True: For nested modules (attention, MLP)
                - False: For VPP level (forward_step, backward_step)

        Returns:
            Dict containing:
                - event_name: Name of the CPU event
                - kernels: DataFrame of matched GPU kernels
                - cpu_duration_ms: CPU annotation duration in ms
                - gpu_walltime_ms: GPU actual execution time (first kernel to last kernel end)
                - bubble_ms: Idle time (cpu_duration - gpu_walltime)
                - bubble_ratio: bubble_ms / cpu_duration_ms
                - num_kernels: Number of GPU kernels launched
                - stream_execution_times: Dict[stream_id -> execution_time_ms]
                - kernel_launch_delay_ms: Average kernel launch delay
                - time_info: Full TimeInfo object
        """
        # Find kernels launched during this CPU event
        kernels = self.find_child_kernels(cpu_event, tp=self.tp, use_hierarchy=use_hierarchy)

        # Calculate GPU execution timing
        time_info = self.calculate_wall_time(kernels)

        # CPU duration in milliseconds
        cpu_duration_ms = float(cpu_event['dur'] / 1000000.0)

        # GPU wall time (actual kernel execution)
        gpu_walltime_ms = time_info.wall_time_ms

        # Bubble: difference between CPU annotation and actual GPU execution
        bubble_ms = cpu_duration_ms - gpu_walltime_ms
        bubble_ratio = bubble_ms / cpu_duration_ms if cpu_duration_ms > 0 else 0.0

        return {
            'event_name': cpu_event.get('name', 'unknown'),
            'kernels': kernels,
            'cpu_duration_ms': cpu_duration_ms,
            'gpu_walltime_ms': gpu_walltime_ms,
            'bubble_ms': bubble_ms,
            'bubble_ratio': bubble_ratio,
            'num_kernels': time_info.num_kernels,
            'stream_execution_times': time_info.stream_execution_times,
            'kernel_launch_delay_ms': time_info.launch_delay_ms,
            'time_info': time_info,
        }

    def get_events_kernel_timing_batch(self, cpu_events: pd.DataFrame, use_hierarchy=True) -> pd.DataFrame:
        """
        Get kernel timing for multiple CPU events (batch processing)

        Args:
            cpu_events: DataFrame of CPU events, each row should have:
                - ts, dur, external_id (optional), name, track_id (optional), depth (optional)
            use_hierarchy: Whether to use track_id+depth filtering (default: True)
                - True: For nested modules (attention, MLP)
                - False: For VPP level (forward_step, backward_step)

        Returns:
            DataFrame with one row per input event, containing:
                - All original columns from cpu_events
                - cpu_duration_ms
                - gpu_walltime_ms
                - bubble_ms
                - bubble_ratio
                - num_kernels
                - kernel_launch_delay_ms
                - stream_execution_times (as dict)
        """
        results = []

        for idx, event in cpu_events.iterrows():
            timing = self.get_event_kernel_timing(event, use_hierarchy=use_hierarchy)

            # Combine original event data with timing results
            result = {
                **event.to_dict(),  # Original event fields
                'cpu_duration_ms': timing['cpu_duration_ms'],
                'gpu_walltime_ms': timing['gpu_walltime_ms'],
                'bubble_ms': timing['bubble_ms'],
                'bubble_ratio': timing['bubble_ratio'],
                'num_kernels': timing['num_kernels'],
                'kernel_launch_delay_ms': timing['kernel_launch_delay_ms'],
                'stream_execution_times': timing['stream_execution_times'],
            }
            results.append(result)

        return pd.DataFrame(results)


# ============================================================================
# Statistics Aggregator
# ============================================================================

class StatisticsAggregator:
    """Aggregates statistics across different dimensions"""

    def aggregate_by_layer(self, module_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate statistics by layer dimension (all times in milliseconds)

        Groups by: (layer_num, module_type, subtype, is_backward)
        """
        if module_stats_df.empty:
            return pd.DataFrame()

        grouped = module_stats_df.groupby([
            'layer_num', 'module_type', 'subtype', 'is_backward'
        ])['wall_time_ms']

        stats = grouped.agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('p50', lambda x: x.quantile(0.5)),
            ('p95', lambda x: x.quantile(0.95)),
            ('p99', lambda x: x.quantile(0.99)),
            ('sum', 'sum')
        ]).reset_index()

        return stats

    def aggregate_by_microbatch(self, module_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate statistics by micro-batch dimension (all times in milliseconds)

        Groups by: (microbatch_id, module_type, subtype, is_backward)
        """
        if module_stats_df.empty:
            return pd.DataFrame()

        grouped = module_stats_df.groupby([
            'microbatch_id', 'module_type', 'subtype', 'is_backward'
        ])['wall_time_ms']

        stats = grouped.agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('sum', 'sum')
        ]).reset_index()

        return stats

    def generate_summary(self, module_stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate global summary statistics (all times in milliseconds)"""
        if module_stats_df.empty:
            return {}

        summary = {}

        # Total time by module type (in ms)
        by_module_type = module_stats_df.groupby('module_type')['wall_time_ms'].sum()
        for module_type, total_time_ms in by_module_type.items():
            summary[f'total_{module_type}_time_ms'] = float(total_time_ms)

        # Total launch delay (in ms)
        summary['total_launch_delay_ms'] = float(module_stats_df['launch_delay_ms'].sum())

        # Total kernels
        summary['total_kernels'] = int(module_stats_df['num_kernels'].sum())

        # Total wall time (in ms)
        summary['total_wall_time_ms'] = float(module_stats_df['wall_time_ms'].sum())

        return summary


# ============================================================================
# SQL Query Builder
# ============================================================================

class SQLQueryBuilder:
    """Generate SQL queries based on analysis configuration"""

    # Static SQL queries (unchanged)
    KERNEL_SQL = """
    SELECT
      slice.id AS id,
      CAST(slice.ts AS LONG) AS ts,
      CAST(args_1.display_value AS LONG) AS external_id,
      CAST(args_2.display_value AS LONG) AS correlation_id,
      CAST(args_3.display_value AS LONG) AS stream,
      CAST(slice.dur AS LONG) AS dur,
      CAST(slice.ts AS LONG) + CAST(slice.dur AS LONG) AS end_ts,
      slice.category AS category,
      slice.name AS name
    FROM slice
    LEFT JOIN args AS args_1 ON args_1.arg_set_id = slice.arg_set_id AND args_1.key = 'args.External id'
    LEFT JOIN args AS args_2 ON args_2.arg_set_id = slice.arg_set_id AND args_2.key = 'args.correlation'
    LEFT JOIN args AS args_3 ON args_3.arg_set_id = slice.arg_set_id AND args_3.key = 'args.stream'
    WHERE slice.category IN ('kernel', 'cuda_driver', 'cuda_runtime', 'gpu_memcpy', 'gpu_memset');
    """

    ALL_EVENTS_SQL = """
    SELECT
      slice.id AS id,
      CAST(slice.ts AS LONG) AS ts,
      CAST(slice.dur AS LONG) AS dur,
      CAST(slice.ts AS LONG) + CAST(slice.dur AS LONG) AS end_ts,
      slice.category AS category,
      slice.name AS name,
      slice.depth AS depth,
      slice.track_id AS track_id,
      args_1.display_value AS external_id,
      args_2.display_value AS correlation_id,
      args_3.display_value AS stream
    FROM slice
    LEFT JOIN args AS args_1 ON args_1.arg_set_id = slice.arg_set_id AND args_1.key = 'args.External id'
    LEFT JOIN args AS args_2 ON args_2.arg_set_id = slice.arg_set_id AND args_2.key = 'args.correlation'
    LEFT JOIN args AS args_3 ON args_3.arg_set_id = slice.arg_set_id AND args_3.key = 'args.stream';
    """

    @staticmethod
    def build_module_sql(configs: List[ModuleConfig]) -> str:
        """Build SQL for module-level analysis, supporting multiple categories."""
        # Group configs by category to generate proper WHERE clause
        from collections import defaultdict
        by_category = defaultdict(list)
        for c in configs:
            by_category[c.category].append(f"slice.name LIKE '{c.sql_pattern}'")

        # Build per-category clauses: (category = X AND (pattern1 OR pattern2))
        category_clauses = []
        for category, patterns in by_category.items():
            pattern_clause = " OR ".join(patterns)
            category_clauses.append(
                f"(slice.category = '{category}' AND ({pattern_clause}))"
            )

        where_clause = "\n          OR ".join(category_clauses)

        return f"""
        SELECT
          slice.id AS id,
          slice.ts AS ts,
          slice.dur AS dur,
          args_1.display_value AS external_id,
          slice.category AS category,
          slice.name AS name,
          slice.depth AS depth,
          slice.track_id AS track_id
        FROM slice
        LEFT JOIN args AS args_1 ON args_1.arg_set_id = slice.arg_set_id
          AND args_1.key = 'args.External id'
        WHERE {where_clause};
        """

    @staticmethod
    def build_vpp_schedule_sql(configs: List[ModuleConfig]) -> str:
        """Build SQL for VPP schedule analysis"""
        patterns = [f"slice.name LIKE '{c.sql_pattern}'" for c in configs]
        pattern_clause = "\n        OR ".join(patterns)

        return f"""
        SELECT
          slice.id AS id,
          slice.ts AS ts,
          slice.dur AS dur,
          args_1.display_value AS external_id,
          slice.category AS category,
          slice.name AS name,
          slice.depth AS depth,
          slice.track_id AS track_id
        FROM slice
        LEFT JOIN args AS args_1 ON args_1.arg_set_id = slice.arg_set_id
          AND args_1.key = 'args.External id'
        WHERE slice.category = 'user_annotation'
          AND (
            {pattern_clause}
          );
        """


# ============================================================================
# Main Analyzer
# ============================================================================

class LLMProfileAnalyzer:
    """Main analyzer class orchestrating the entire analysis"""


    def __init__(self, trace_path: str, bin_path: str = None, config: AnalysisConfig = None,
                 model_config_path: str = None):
        """
        Initialize analyzer

        Args:
            trace_path: Path to the trace file (.pt.trace.json.gz)
            bin_path: Path to trace_processor_shell binary
            config: Optional AnalysisConfig for customizing analysis targets
            model_config_path: Optional path to model_config.yaml for FLOPs calculation
        """
        self.trace_path = trace_path
        self.config = config or AnalysisConfig()
        self.flops_calculator = None
        if model_config_path and os.path.exists(model_config_path):
            try:
                self.flops_calculator = ModelFlopsCalculator(model_config_path)
                logger.info(f"FLOPs calculator loaded from: {model_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load model config for FLOPs: {e}")
        logger.info(f"Loading trace from: {trace_path}")

        # Initialize TraceProcessor directly
        if bin_path is None:
            # Try to find trace_processor_shell in profiles/tools directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            profiles_dir = os.path.dirname(script_dir)  # profiles/
            default_bin = os.path.join(profiles_dir, 'tools', 'trace_processor_shell')
            if os.path.exists(default_bin):
                bin_path = default_bin
                logger.info(f"Using default trace_processor_shell: {bin_path}")

        logger.info("Initializing TraceProcessor...")
        self.tp = TraceProcessor(trace=trace_path, config=TraceProcessorConfig(bin_path=bin_path))

        # Generate dynamic SQL queries based on config
        logger.info("Generating SQL queries from configuration...")
        self.MODULE_SQL = SQLQueryBuilder.build_module_sql(self.config.module_configs)
        self.VPP_SCHEDULE_SQL = SQLQueryBuilder.build_vpp_schedule_sql(self.config.vpp_event_configs)

        # Load and join kernel data
        logger.info("Loading kernel data...")
        self.kernel_df = self._load_kernel_data()

        # Initialize components
        self.module_identifier = ModuleIdentifier()
        self.correlator = EventCorrelator(self.kernel_df, tp=self.tp)
        self.aggregator = StatisticsAggregator()

        logger.info("Analyzer initialized successfully")

    def _load_kernel_data(self) -> pd.DataFrame:
        """Load kernel data and join CPU/GPU events by correlation ID"""
        logger.info("Querying kernel events...")
        df = self.tp.query(SQLQueryBuilder.KERNEL_SQL).as_pandas_dataframe()
        logger.info(f"Found {len(df)} kernel-related events")

        # Separate cuda_runtime and kernel events
        df_cuda_runtime = df[df['category'].isin(['cuda_runtime', 'cuda_driver'])].copy()
        df_kernel = df[df['category'].isin(['kernel', 'gpu_memcpy', 'gpu_memset'])].copy()

        logger.info(f"CUDA runtime events: {len(df_cuda_runtime)}")
        logger.info(f"GPU kernel events: {len(df_kernel)}")

        # Join by correlation_id
        df_joined = pd.merge(
            df_cuda_runtime,
            df_kernel,
            on='correlation_id',
            how='left',
            suffixes=('_cuda', '_kernel')
        )

        # Calculate kernel delay
        df_delay = df_joined[df_joined['id_kernel'].notna()].copy()
        df_delay['kernel_delay'] = df_delay['ts_kernel'] - df_delay['ts_cuda'] - df_delay['dur_cuda']
        df_delay['kernel_delay'] = df_delay['kernel_delay'].clip(lower=1)

        # Sort by kernel timestamp
        df_delay_sort = df_delay.sort_values(by='ts_kernel', ascending=True).copy()

        logger.info(f"Joined {len(df_delay_sort)} CPU-GPU event pairs")
        return df_delay_sort

    def analyze_vpp_schedule(self) -> Dict[str, Any]:
        """
        Analyze VPP schedule with GPU kernel timing (REFACTORED VERSION)

        Uses configuration to determine which VPP events to analyze.
        Default events: forward_step, backward_step, p2p_send, p2p_recv, etc.

        Returns:
            Dict containing:
                - forward_step_gpu_list: List of dicts with {start_ms, end_ms, duration_ms}
                - backward_step_gpu_list: List of dicts with {start_ms, end_ms, duration_ms}
                - p2p_send_gpu_list: List of dicts with {start_ms, end_ms, duration_ms} (if pp>1)
                - p2p_recv_gpu_list: List of dicts with {start_ms, end_ms, duration_ms} (if pp>1)
                - summary: Overall statistics (mean, std, min, max of durations)
        """
        logger.info("Analyzing VPP schedule level (config-based)...")

        # Query using dynamically generated SQL
        df = self.tp.query(self.VPP_SCHEDULE_SQL).as_pandas_dataframe()
        logger.info(f"Found {len(df)} VPP schedule events")

        if df.empty:
            return self._empty_vpp_result_new()

        df = df.sort_values(by='ts', ascending=True).copy()

        # Classify events using configuration
        df['event_type'] = df['name'].apply(self.config.classify_vpp_event)

        # Separate by type
        forward_events = df[df['event_type'] == 'forward_step'].copy()
        backward_events = df[df['event_type'] == 'backward_step'].copy()
        p2p_send_events = df[df['event_type'] == 'p2p_send'].copy()
        p2p_recv_events = df[df['event_type'] == 'p2p_recv'].copy()

        # Detect PP degree
        pp_degree = 2 if (len(p2p_send_events) > 0 or len(p2p_recv_events) > 0) else 1

        logger.info(f"Detected pp_degree = {pp_degree}")
        logger.info(f"  Forward steps: {len(forward_events)}")
        logger.info(f"  Backward steps: {len(backward_events)}")
        if pp_degree > 1:
            logger.info(f"  P2P sends: {len(p2p_send_events)}")
            logger.info(f"  P2P recvs: {len(p2p_recv_events)}")

        # ===== Get GPU times for each step =====
        forward_gpu_list = []
        backward_gpu_list = []
        p2p_send_gpu_list = []
        p2p_recv_gpu_list = []

        # Process forward steps
        logger.info("Getting GPU times for forward steps...")
        for _, event in forward_events.iterrows():
            kernels = self.correlator.find_child_kernels(event, tp=self.tp, use_hierarchy=False)
            if not kernels.empty:
                gpu_start = kernels['ts_kernel'].min()
                gpu_end = (kernels['ts_kernel'] + kernels['dur_kernel']).max()
                gpu_time_ms = (gpu_end - gpu_start) / 1000000.0
                forward_gpu_list.append({
                    'start_ms': float(gpu_start / 1000000.0),
                    'end_ms': float(gpu_end / 1000000.0),
                    'duration_ms': float(gpu_time_ms)
                })

        # Process backward steps
        logger.info("Getting GPU times for backward steps...")
        for _, event in backward_events.iterrows():
            kernels = self.correlator.find_child_kernels(event, tp=self.tp, use_hierarchy=False)
            if not kernels.empty:
                gpu_start = kernels['ts_kernel'].min()
                gpu_end = (kernels['ts_kernel'] + kernels['dur_kernel']).max()
                gpu_time_ms = (gpu_end - gpu_start) / 1000000.0
                backward_gpu_list.append({
                    'start_ms': float(gpu_start / 1000000.0),
                    'end_ms': float(gpu_end / 1000000.0),
                    'duration_ms': float(gpu_time_ms)
                })

        # Process P2P events (if pp>1)
        if pp_degree > 1:
            logger.info("Getting GPU times for P2P communication...")
            for _, event in p2p_send_events.iterrows():
                kernels = self.correlator.find_child_kernels(event, tp=self.tp, use_hierarchy=False)
                if not kernels.empty:
                    gpu_start = kernels['ts_kernel'].min()
                    gpu_end = (kernels['ts_kernel'] + kernels['dur_kernel']).max()
                    gpu_time_ms = (gpu_end - gpu_start) / 1000000.0
                    p2p_info = self._extract_p2p_info(kernels)
                    p2p_send_gpu_list.append({
                        'start_ms': float(gpu_start / 1000000.0),
                        'end_ms': float(gpu_end / 1000000.0),
                        'duration_ms': float(gpu_time_ms),
                        **p2p_info,
                    })

            for _, event in p2p_recv_events.iterrows():
                kernels = self.correlator.find_child_kernels(event, tp=self.tp, use_hierarchy=False)
                if not kernels.empty:
                    gpu_start = kernels['ts_kernel'].min()
                    gpu_end = (kernels['ts_kernel'] + kernels['dur_kernel']).max()
                    gpu_time_ms = (gpu_end - gpu_start) / 1000000.0
                    p2p_info = self._extract_p2p_info(kernels)
                    p2p_recv_gpu_list.append({
                        'start_ms': float(gpu_start / 1000000.0),
                        'end_ms': float(gpu_end / 1000000.0),
                        'duration_ms': float(gpu_time_ms),
                        **p2p_info,
                    })

        # ===== Calculate summary statistics =====
        summary = {
            'pp_degree': pp_degree,
            'num_forward_steps': len(forward_gpu_list),
            'num_backward_steps': len(backward_gpu_list),
        }

        if forward_gpu_list:
            forward_durations = [item['duration_ms'] for item in forward_gpu_list]
            summary['forward_gpu_mean_ms'] = float(np.mean(forward_durations))
            summary['forward_gpu_std_ms'] = float(np.std(forward_durations))
            summary['forward_gpu_min_ms'] = float(np.min(forward_durations))
            summary['forward_gpu_max_ms'] = float(np.max(forward_durations))

        if backward_gpu_list:
            backward_durations = [item['duration_ms'] for item in backward_gpu_list]
            summary['backward_gpu_mean_ms'] = float(np.mean(backward_durations))
            summary['backward_gpu_std_ms'] = float(np.std(backward_durations))
            summary['backward_gpu_min_ms'] = float(np.min(backward_durations))
            summary['backward_gpu_max_ms'] = float(np.max(backward_durations))

        if pp_degree > 1:
            if p2p_send_gpu_list:
                p2p_send_durations = [item['duration_ms'] for item in p2p_send_gpu_list]
                summary['p2p_send_mean_ms'] = float(np.mean(p2p_send_durations))
                summary['p2p_send_total_ms'] = float(np.sum(p2p_send_durations))
            if p2p_recv_gpu_list:
                p2p_recv_durations = [item['duration_ms'] for item in p2p_recv_gpu_list]
                summary['p2p_recv_mean_ms'] = float(np.mean(p2p_recv_durations))
                summary['p2p_recv_total_ms'] = float(np.sum(p2p_recv_durations))

        # Package results
        result = {
            'summary': summary,
            'forward_step_gpu_list': forward_gpu_list,
            'backward_step_gpu_list': backward_gpu_list,
            'p2p_send_gpu_list': p2p_send_gpu_list,
            'p2p_recv_gpu_list': p2p_recv_gpu_list,
        }

        logger.info(f"VPP schedule analysis complete:")
        logger.info(f"  - PP degree: {pp_degree}")
        logger.info(f"  - Forward GPU time: {summary.get('forward_gpu_mean_ms', 0):.2f} ± {summary.get('forward_gpu_std_ms', 0):.2f} ms")
        logger.info(f"  - Backward GPU time: {summary.get('backward_gpu_mean_ms', 0):.2f} ± {summary.get('backward_gpu_std_ms', 0):.2f} ms")

        return result

    def _empty_vpp_result_new(self) -> Dict[str, Any]:
        """Return empty VPP result structure"""
        return {
            'summary': {
                'pp_degree': 1,
                'num_forward_steps': 0,
                'num_backward_steps': 0,
            },
            'forward_step_gpu_list': [],
            'backward_step_gpu_list': [],
            'p2p_send_gpu_list': [],
            'p2p_recv_gpu_list': [],
        }

    def _extract_p2p_info(self, kernels: pd.DataFrame) -> Dict[str, Any]:
        """Extract P2P direction info (collective name, dst rank, group ranks) from kernel args.

        Queries the perfetto trace for NCCL kernel args like 'Collective name', 'Dst Rank',
        'Process Group Ranks', 'Process Group Description'.

        Args:
            kernels: DataFrame of matched GPU kernels for a P2P event

        Returns:
            Dict with keys: collective, dst_rank, group_ranks, group_desc, label
        """
        result = {
            'collective': None,
            'dst_rank': None,
            'group_ranks': None,
            'group_desc': None,
            'label': '',
        }
        if kernels.empty:
            return result

        # Use the first kernel with a valid id
        kernel_id = None
        if 'id_kernel' in kernels.columns:
            valid_ids = kernels['id_kernel'].dropna()
            if not valid_ids.empty:
                kernel_id = int(valid_ids.iloc[0])

        if kernel_id is None:
            return result

        try:
            args_sql = f"""
            SELECT args.key, args.display_value
            FROM slice
            JOIN args ON args.arg_set_id = slice.arg_set_id
            WHERE slice.id = {kernel_id}
              AND args.key IN ('Collective name', 'Dst Rank',
                               'Process Group Ranks', 'Process Group Description')
            """
            args_df = self.tp.query(args_sql).as_pandas_dataframe()

            if args_df.empty:
                return result

            args_dict = dict(zip(args_df['key'], args_df['display_value']))
            collective = args_dict.get('Collective name', None)
            dst_rank = args_dict.get('Dst Rank', None)
            group_ranks = args_dict.get('Process Group Ranks', None)
            group_desc = args_dict.get('Process Group Description', None)

            result['collective'] = str(collective) if collective else None
            result['dst_rank'] = int(dst_rank) if dst_rank is not None else None
            result['group_ranks'] = str(group_ranks) if group_ranks else None
            result['group_desc'] = str(group_desc) if group_desc else None

            # Build label
            if collective and dst_rank is not None:
                direction = 'Send' if 'send' in str(collective).lower() else 'Recv'
                result['label'] = f"{direction} →Rank{dst_rank}"
            elif collective:
                result['label'] = str(collective)

        except Exception as e:
            logger.warning(f"Failed to extract P2P info from kernel args: {e}")

        return result

    def analyze_optimizer_phase(self) -> Dict[str, Any]:
        """Analyze optimizer-phase events: grad reduce-scatter, allreduce, loss postprocessing,
        optimizer step, and training log.

        Queries the trace for optimizer-related CPU ops and correlates them with GPU kernels.

        Returns:
            Dict with keys: grad_reduce_scatter, allreduce, loss_postprocessing,
            optimizer_step, training_log (each a list of event dicts), and summary.
        """
        logger.info("Analyzing optimizer phase events...")

        # SQL query for optimizer-related events
        optimizer_sql = """
        SELECT slice.id AS id, slice.ts AS ts, slice.dur AS dur,
               args_1.display_value AS external_id,
               slice.category AS category, slice.name AS name,
               slice.depth AS depth, slice.track_id AS track_id
        FROM slice
        LEFT JOIN args AS args_1 ON args_1.arg_set_id = slice.arg_set_id
          AND args_1.key = 'args.External id'
        WHERE (slice.category = 'user_annotation' AND (
            slice.name LIKE '%finish_grad_sync%'
            OR slice.name LIKE '%loss_postprocessing%'
            OR slice.name LIKE '%ChainedOptimizer.step%'
            OR slice.name LIKE '%training_log%'
          ))
          OR (slice.category = 'cpu_op' AND slice.name LIKE '%allreduce_%')
        """

        empty_result = {
            'grad_reduce_scatter': [],
            'allreduce': [],
            'loss_postprocessing': [],
            'optimizer_step': [],
            'training_log': [],
            'summary': {
                'grad_reduce_scatter_total_ms': 0.0,
                'allreduce_total_ms': 0.0,
                'loss_postprocessing_ms': 0.0,
                'optimizer_step_ms': 0.0,
                'training_log_ms': 0.0,
                'total_optimizer_ms': 0.0,
            }
        }

        try:
            df = self.tp.query(optimizer_sql).as_pandas_dataframe()
            logger.info(f"Found {len(df)} optimizer-phase events")

            if df.empty:
                return empty_result

            df = df.sort_values(by='ts', ascending=True).copy()

            # Classify events into phases
            grad_rs_events = df[df['name'].str.contains('finish_grad_sync', na=False)]
            allreduce_events = df[df['name'].str.contains('allreduce_', na=False)]
            loss_post_events = df[df['name'].str.contains('loss_postprocessing', na=False)]
            opt_step_events = df[df['name'].str.contains('ChainedOptimizer.step', na=False)]
            train_log_events = df[df['name'].str.contains('training_log', na=False)]

            logger.info(f"  finish_grad_sync: {len(grad_rs_events)}")
            logger.info(f"  allreduce_: {len(allreduce_events)}")
            logger.info(f"  loss_postprocessing: {len(loss_post_events)}")
            logger.info(f"  ChainedOptimizer.step: {len(opt_step_events)}")
            logger.info(f"  training_log: {len(train_log_events)}")

            # Helper to process events and get GPU timing
            def _process_events(events_df, label_prefix, use_index=True):
                results = []
                for idx_i, (_, event) in enumerate(events_df.iterrows()):
                    kernels = self.correlator.find_child_kernels(event, tp=self.tp, use_hierarchy=False)

                    cpu_start_ms = float(event['ts'] / 1000000.0)
                    cpu_end_ms = float((event['ts'] + event['dur']) / 1000000.0)
                    cpu_duration_ms = cpu_end_ms - cpu_start_ms

                    entry = {
                        'gpu_start_ms': None,
                        'gpu_end_ms': None,
                        'gpu_duration_ms': 0.0,
                        'cpu_start_ms': cpu_start_ms,
                        'cpu_end_ms': cpu_end_ms,
                        'cpu_duration_ms': cpu_duration_ms,
                        'label': f"{label_prefix}{idx_i}" if use_index else label_prefix,
                    }

                    if not kernels.empty:
                        gpu_start = kernels['ts_kernel'].min()
                        gpu_end = (kernels['ts_kernel'] + kernels['dur_kernel']).max()
                        entry['gpu_start_ms'] = float(gpu_start / 1000000.0)
                        entry['gpu_end_ms'] = float(gpu_end / 1000000.0)
                        entry['gpu_duration_ms'] = float((gpu_end - gpu_start) / 1000000.0)

                    results.append(entry)
                return results

            # Process each phase
            grad_rs_list = _process_events(grad_rs_events, 'GradRS')
            allreduce_list = _process_events(allreduce_events, 'AR')
            loss_post_list = _process_events(loss_post_events, 'LossPost', use_index=False)
            opt_step_list = _process_events(opt_step_events, 'OptStep', use_index=False)
            train_log_list = _process_events(train_log_events, 'TrainLog', use_index=False)

            # Compute summary
            grad_rs_total = sum(e['gpu_duration_ms'] for e in grad_rs_list)
            allreduce_total = sum(e['gpu_duration_ms'] for e in allreduce_list)
            loss_post_total = sum(e['gpu_duration_ms'] for e in loss_post_list)
            opt_step_total = sum(e['gpu_duration_ms'] for e in opt_step_list)
            train_log_total = sum(e['gpu_duration_ms'] for e in train_log_list)
            total_optimizer = grad_rs_total + allreduce_total + loss_post_total + opt_step_total + train_log_total

            result = {
                'grad_reduce_scatter': grad_rs_list,
                'allreduce': allreduce_list,
                'loss_postprocessing': loss_post_list,
                'optimizer_step': opt_step_list,
                'training_log': train_log_list,
                'summary': {
                    'grad_reduce_scatter_total_ms': round(grad_rs_total, 4),
                    'allreduce_total_ms': round(allreduce_total, 4),
                    'loss_postprocessing_ms': round(loss_post_total, 4),
                    'optimizer_step_ms': round(opt_step_total, 4),
                    'training_log_ms': round(train_log_total, 4),
                    'total_optimizer_ms': round(total_optimizer, 4),
                }
            }

            logger.info(f"Optimizer phase analysis complete:")
            logger.info(f"  Grad RS total GPU: {grad_rs_total:.2f} ms")
            logger.info(f"  Allreduce total GPU: {allreduce_total:.2f} ms")
            logger.info(f"  Loss postprocessing GPU: {loss_post_total:.2f} ms")
            logger.info(f"  Optimizer step GPU: {opt_step_total:.2f} ms")
            logger.info(f"  Training log GPU: {train_log_total:.2f} ms")
            logger.info(f"  Total optimizer GPU: {total_optimizer:.2f} ms")

            return result

        except Exception as e:
            logger.warning(f"Failed to analyze optimizer phase: {e}")
            return empty_result

    def _calculate_summary_stats(self, gpu_timings: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate summary statistics for a list of GPU timings"""
        if not gpu_timings:
            return {
                'count': 0,
                'mean_duration_ms': 0.0,
                'std_duration_ms': 0.0,
                'min_duration_ms': 0.0,
                'max_duration_ms': 0.0,
                'mean_delay_ms': 0.0,
                'std_delay_ms': 0.0,
            }

        durations = [t['duration_ms'] for t in gpu_timings]
        delays = [t.get('launch_delay_ms', 0.0) for t in gpu_timings]

        return {
            'count': len(gpu_timings),
            'mean_duration_ms': float(np.mean(durations)),
            'std_duration_ms': float(np.std(durations)),
            'min_duration_ms': float(np.min(durations)),
            'max_duration_ms': float(np.max(durations)),
            'mean_delay_ms': float(np.mean(delays)) if delays else 0.0,
            'std_delay_ms': float(np.std(delays)) if delays else 0.0,
        }

    def analyze_modules(self) -> Dict[str, Any]:
        """
        Analyze module-level timing for configured modules (REFACTORED VERSION)

        Uses configuration to determine which modules to analyze.
        Default modules: KimiDeltaAttention, MLASelfAttention, MoELayer, MLP, etc.
        Can be customized via AnalysisConfig.

        Returns:
            Dict containing per-module results:
                {
                    'module_name': {
                        'events': [{'start_ms', 'end_ms', 'duration_ms', 'launch_delay_ms'}, ...],
                        'summary': {'count', 'mean_duration_ms', 'mean_delay_ms', ...}
                    },
                    ...
                }
        """
        logger.info("Analyzing module-level timing (config-based)...")

        # Query using dynamically generated SQL
        df = self.tp.query(self.MODULE_SQL).as_pandas_dataframe()
        logger.info(f"Found {len(df)} total module events")

        # Get module names from configuration
        module_names = self.config.get_module_names()

        if df.empty:
            # Return empty results for all configured modules
            empty_result = {
                'events': [],
                'summary': {
                    'count': 0,
                    'mean_duration_ms': 0.0,
                    'mean_delay_ms': 0.0,
                }
            }
            return {name: empty_result.copy() for name in module_names}

        # Classify each event using configuration
        df['module_type'] = df['name'].apply(self.config.classify_module)

        # Process each event to get GPU timing
        logger.info("Processing events to get GPU timing...")

        df['gpu_start_ns'] = None
        df['gpu_end_ns'] = None
        df['gpu_duration_ms'] = None
        df['launch_delay_ms'] = None

        for idx, event in df.iterrows():
            kernels = self.correlator.find_child_kernels(event, tp=self.tp, use_hierarchy=False)

            if kernels.empty:
                continue

            gpu_start = kernels['ts_kernel'].min()
            gpu_end = (kernels['ts_kernel'] + kernels['dur_kernel']).max()
            gpu_duration_ms = (gpu_end - gpu_start) / 1000000.0

            cpu_start = event['ts']
            launch_delay_ms = (gpu_start - cpu_start) / 1000000.0

            df.at[idx, 'gpu_start_ns'] = gpu_start
            df.at[idx, 'gpu_end_ns'] = gpu_end
            df.at[idx, 'gpu_duration_ms'] = gpu_duration_ms
            df.at[idx, 'launch_delay_ms'] = launch_delay_ms

        # Filter events with GPU timing
        df_with_timing = df[df['gpu_start_ns'].notna()].copy()
        logger.info(f"Successfully processed {len(df_with_timing)}/{len(df)} events")

        # Group by module type and calculate statistics
        results = {}

        for module_name in module_names:
            module_events = df_with_timing[df_with_timing['module_type'] == module_name]

            if module_events.empty:
                results[module_name] = {
                    'events': [],
                    'summary': self._calculate_summary_stats([])
                }
                logger.info(f"  {module_name}: 0 events")
                continue

            # Convert to events list
            events_list = []
            for _, event in module_events.iterrows():
                events_list.append({
                    'start_ms': float(event['gpu_start_ns'] / 1000000.0),
                    'end_ms': float(event['gpu_end_ns'] / 1000000.0),
                    'duration_ms': float(event['gpu_duration_ms']),
                    'launch_delay_ms': float(event['launch_delay_ms']),
                })

            # Calculate statistics using helper method
            summary = self._calculate_summary_stats(events_list)

            # Calculate TFLOPS or bandwidth if flops calculator is available
            flops = None
            mean_tflops = None
            min_tflops = None
            max_tflops = None
            comm_volume = None
            mean_bandwidth_gbs = None
            min_bandwidth_gbs = None
            max_bandwidth_gbs = None

            if self.flops_calculator:
                # Try FLOPs first
                flops = self.flops_calculator.get_module_flops(module_name)
                if flops is not None and summary['mean_duration_ms'] > 0:
                    mean_tflops = flops / (summary['mean_duration_ms'] * 1e-3) / 1e12
                if flops is not None and summary['max_duration_ms'] > 0:
                    min_tflops = flops / (summary['max_duration_ms'] * 1e-3) / 1e12
                if flops is not None and summary['min_duration_ms'] > 0:
                    max_tflops = flops / (summary['min_duration_ms'] * 1e-3) / 1e12

                # Try bandwidth
                comm_volume = self.flops_calculator.get_module_comm_volume(module_name)
                if comm_volume is not None:
                    volume_gb = comm_volume / 1e9
                    if summary['mean_duration_ms'] > 0:
                        mean_bandwidth_gbs = volume_gb / (summary['mean_duration_ms'] * 1e-3)
                    if summary['max_duration_ms'] > 0:
                        min_bandwidth_gbs = volume_gb / (summary['max_duration_ms'] * 1e-3)
                    if summary['min_duration_ms'] > 0:
                        max_bandwidth_gbs = volume_gb / (summary['min_duration_ms'] * 1e-3)

            summary['flops'] = flops
            summary['mean_tflops'] = mean_tflops
            summary['min_tflops'] = min_tflops
            summary['max_tflops'] = max_tflops
            summary['comm_volume_bytes'] = comm_volume
            summary['mean_bandwidth_gbs'] = mean_bandwidth_gbs
            summary['min_bandwidth_gbs'] = min_bandwidth_gbs
            summary['max_bandwidth_gbs'] = max_bandwidth_gbs

            results[module_name] = {
                'events': events_list,
                'summary': summary,
            }

            perf_str = ""
            if mean_tflops:
                perf_str = f", mean TFLOPS={mean_tflops:.1f}"
            elif mean_bandwidth_gbs:
                perf_str = f", mean BW={mean_bandwidth_gbs:.1f} GB/s"
            logger.info(f"  {module_name}: {len(events_list)} events, "
                       f"mean duration={summary['mean_duration_ms']:.2f}ms"
                       f"{perf_str}, "
                       f"mean delay={summary['mean_delay_ms']:.2f}ms")

        logger.info("Module-level analysis complete")
        return results


# ============================================================================
# Multi-PP Utilities
# ============================================================================

def parse_layout_num_model_chunks(layout_str: str, pp_degree: int) -> int:
    """Parse num_model_chunks (VPP size) from pipeline_model_parallel_layout string.

    The layout string format is e.g. 'Et*4|t*5|t*5|t*2mL'.
    Each '|' separates a stage. num_stages / pp_degree = num_model_chunks.

    Args:
        layout_str: Pipeline layout string (e.g. 'Et*4|t*5|t*5|t*2mL')
        pp_degree: Pipeline parallel size

    Returns:
        num_model_chunks (virtual_pipeline_model_parallel_size)
    """
    layout_str = layout_str.replace(",", "")
    # Expand ()*n and x*n patterns
    for pattern in [r'\(([^)]+)\)\*(\d+)', r'(.)\*(\d+)']:
        layout_str = re.sub(pattern, lambda x: x.group(1) * int(x.group(2)), layout_str)
    num_stages = len(layout_str.split('|'))
    assert num_stages % pp_degree == 0, (
        f"Number of stages ({num_stages}) must be divisible by pp_degree ({pp_degree})"
    )
    return num_stages // pp_degree


def generate_schedule_table(
    num_microbatches: int,
    num_model_chunks: int,
    microbatch_group_size_per_vp_stage: int,
) -> List[Tuple[int, int]]:
    """Lightweight version of Megatron's get_schedule_table.

    Generates the mapping from virtual_microbatch_id to (microbatch_id, model_chunk_id).

    Args:
        num_microbatches: Number of microbatches
        num_model_chunks: Number of model chunks (VPP size)
        microbatch_group_size_per_vp_stage: Group size per VPP stage (typically = pp_degree)

    Returns:
        List of (microbatch_id, model_chunk_id) tuples
    """
    table = []
    for group_start in range(0, num_microbatches, microbatch_group_size_per_vp_stage):
        group_end = min(group_start + microbatch_group_size_per_vp_stage, num_microbatches)
        for chunk_id in range(num_model_chunks):
            for mb_id in range(group_start, group_end):
                table.append((mb_id, chunk_id))
    return table


def discover_trace_files(trace_dir: str) -> List[str]:
    """Discover and sort trace files in a directory by rank number.

    Supports patterns: *rank<N>*.json.gz, *rank<N>*.perfetto-trace

    Args:
        trace_dir: Directory containing trace files

    Returns:
        List of trace file paths sorted by rank number
    """
    patterns = ['*.pt.trace.json.gz', '*.perfetto-trace', '*.json.gz']
    seen = set()
    files = []
    for pat in patterns:
        for f in glob_module.glob(os.path.join(trace_dir, pat)):
            real = os.path.realpath(f)
            if real not in seen:
                seen.add(real)
                files.append(f)

    if not files:
        return []

    # Extract rank number from filename
    def extract_rank(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'rank(\d+)', basename)
        if match:
            return int(match.group(1))
        return 0

    files.sort(key=extract_rank)
    return files


# ============================================================================
# Multi-PP Analyzer
# ============================================================================

class MultiPPAnalyzer:
    """Analyze multiple PP rank traces and merge into a unified timeline.

    Loads trace files from a directory (one per PP rank), processes each with
    LLMProfileAnalyzer in parallel, then merges results with time alignment
    and schedule table mapping.
    """

    def __init__(
        self,
        trace_dir: str,
        bin_path: str = None,
        config: AnalysisConfig = None,
        model_config_path: str = None,
    ):
        """
        Args:
            trace_dir: Directory containing one trace file per PP rank
            bin_path: Path to trace_processor_shell binary
            config: Optional AnalysisConfig
            model_config_path: Path to model_config.yaml
        """
        self.trace_dir = trace_dir
        self.bin_path = bin_path
        self.config = config or AnalysisConfig()
        self.model_config_path = model_config_path

        # Discover trace files
        self.trace_files = discover_trace_files(trace_dir)
        self.pp_degree = len(self.trace_files)
        if self.pp_degree == 0:
            raise ValueError(f"No trace files found in {trace_dir}")
        logger.info(f"Discovered {self.pp_degree} trace files in {trace_dir}")
        for i, f in enumerate(self.trace_files):
            logger.info(f"  Rank {i}: {os.path.basename(f)}")

        # Load schedule params from model config
        self.num_model_chunks = 1
        self.layout_str = None
        self._load_schedule_params()

    def _load_schedule_params(self):
        """Load VPP schedule parameters from model config YAML."""
        if not self.model_config_path or not os.path.exists(self.model_config_path):
            logger.info("No model config provided, using default schedule params "
                       f"(num_model_chunks=1)")
            return

        try:
            with open(self.model_config_path, 'r') as f:
                raw_config = yaml.safe_load(f)

            extra_cfg = raw_config.get('trainer', {}).get('args', {}).get('extra_configs', {})

            # Get layout string
            layout = extra_cfg.get('pipeline_model_parallel_layout', None)
            if layout:
                self.layout_str = layout
                self.num_model_chunks = parse_layout_num_model_chunks(layout, self.pp_degree)
                logger.info(f"Parsed layout '{layout}' -> num_model_chunks={self.num_model_chunks}")
            else:
                logger.info("No pipeline_model_parallel_layout in config, num_model_chunks=1")

        except Exception as e:
            logger.warning(f"Failed to load schedule params from config: {e}")

    def analyze_all(self) -> Dict[str, Any]:
        """Process all rank traces in parallel and merge results.

        Returns:
            Merged multi-PP result dict with structure:
            {
                'pp_degree': int,
                'num_microbatches': int,
                'num_model_chunks': int,
                'schedule_table': [(mbs_id, chunk_id), ...],
                'ranks': {
                    0: {
                        'forward_steps': [{start_ms, end_ms, duration_ms, mbs_id, chunk_id, label}, ...],
                        'backward_steps': [...],
                        'p2p_sends': [{..., collective, dst_rank, label}, ...],
                        'p2p_recvs': [...],
                    },
                    ...
                },
                'summary': {
                    'total_time_ms': float,
                    'bubble_ratio': float,
                    'per_rank_stats': [{rank, compute_ms, bubble_ms, bubble_ratio}, ...],
                }
            }
        """
        # Step 1: Process each rank in parallel
        per_rank_vpp = {}
        per_rank_modules = {}
        per_rank_optimizer = {}

        def _process_rank(rank_idx):
            trace_path = self.trace_files[rank_idx]
            logger.info(f"Processing rank {rank_idx}: {os.path.basename(trace_path)}")
            analyzer = LLMProfileAnalyzer(
                trace_path=trace_path,
                bin_path=self.bin_path,
                config=self.config,
                model_config_path=self.model_config_path,
            )
            vpp_result = analyzer.analyze_vpp_schedule()
            module_result = analyzer.analyze_modules()
            optimizer_result = analyzer.analyze_optimizer_phase()
            return rank_idx, vpp_result, module_result, optimizer_result

        with ThreadPoolExecutor(max_workers=min(self.pp_degree, 4)) as executor:
            futures = {executor.submit(_process_rank, i): i for i in range(self.pp_degree)}
            for future in as_completed(futures):
                rank_idx, vpp_res, mod_res, opt_res = future.result()
                per_rank_vpp[rank_idx] = vpp_res
                per_rank_modules[rank_idx] = mod_res
                per_rank_optimizer[rank_idx] = opt_res

        logger.info(f"All {self.pp_degree} ranks processed")

        # Step 2: Merge VPP timeline results (also merges optimizer events into ranks)
        merged = self._merge_results(per_rank_vpp, per_rank_optimizer)

        # Step 3: Merge module results across ranks
        merged['module_result'] = self._merge_module_results(per_rank_modules)

        # Step 4: Merge optimizer results across ranks
        merged['optimizer'] = self._merge_optimizer_results(per_rank_optimizer, merged)

        return merged

    def _merge_results(self, per_rank_results: Dict[int, Dict],
                        per_rank_optimizer: Dict[int, Dict] = None) -> Dict[str, Any]:
        """Merge per-rank VPP results into unified timeline with schedule table mapping."""

        # Determine num_microbatches from the first rank's forward count
        rank0 = per_rank_results.get(0, {})
        num_forward = len(rank0.get('forward_step_gpu_list', []))
        total_virtual_mbs = num_forward  # per rank
        num_microbatches = total_virtual_mbs // self.num_model_chunks if self.num_model_chunks > 0 else total_virtual_mbs

        # Generate schedule table
        microbatch_group_size = self.pp_degree
        schedule_table = generate_schedule_table(
            num_microbatches, self.num_model_chunks, microbatch_group_size
        )
        mb_id_table = [t[0] for t in schedule_table]
        chunk_id_table = [t[1] for t in schedule_table]
        total_num_virtual_mbs = len(schedule_table)

        logger.info(f"Schedule table: num_microbatches={num_microbatches}, "
                    f"num_model_chunks={self.num_model_chunks}, "
                    f"group_size={microbatch_group_size}, "
                    f"total_virtual_mbs={total_num_virtual_mbs}")

        # Step 1: Time alignment — use each rank's first forward GPU start as baseline
        rank_time_offsets = {}  # rank -> offset to subtract (makes first forward start at ~0)
        global_min_start = float('inf')

        for rank_idx in range(self.pp_degree):
            result = per_rank_results.get(rank_idx, {})
            fwd_list = result.get('forward_step_gpu_list', [])
            if fwd_list:
                first_start = fwd_list[0]['start_ms']
                rank_time_offsets[rank_idx] = first_start
                global_min_start = min(global_min_start, first_start)
            else:
                rank_time_offsets[rank_idx] = 0.0

        # Normalize: all times relative to the earliest rank's first forward
        if global_min_start == float('inf'):
            global_min_start = 0.0

        # Step 2: Build per-rank data with mbs labels
        ranks_data = {}
        for rank_idx in range(self.pp_degree):
            result = per_rank_results.get(rank_idx, {})
            offset = rank_time_offsets.get(rank_idx, 0.0) - global_min_start

            # Process forward steps with schedule table mapping
            fwd_list = result.get('forward_step_gpu_list', [])
            forward_steps = []
            for i, step in enumerate(fwd_list):
                # Map to schedule table
                if i < total_num_virtual_mbs:
                    mbs_id = mb_id_table[i]
                    chunk_id = chunk_id_table[i]
                else:
                    mbs_id = i
                    chunk_id = 0

                if self.num_model_chunks > 1:
                    label = f"F{mbs_id}c{chunk_id}"
                else:
                    label = f"F{mbs_id}"

                forward_steps.append({
                    'start_ms': step['start_ms'] - rank_time_offsets[rank_idx] + offset,
                    'end_ms': step['end_ms'] - rank_time_offsets[rank_idx] + offset,
                    'duration_ms': step['duration_ms'],
                    'mbs_id': mbs_id,
                    'chunk_id': chunk_id,
                    'label': label,
                })

            # Process backward steps with schedule table mapping (chunk_id reversed)
            bwd_list = result.get('backward_step_gpu_list', [])
            backward_steps = []
            for i, step in enumerate(bwd_list):
                if i < total_num_virtual_mbs:
                    mbs_id = mb_id_table[i]
                    chunk_id = self.num_model_chunks - 1 - chunk_id_table[i]
                else:
                    mbs_id = i
                    chunk_id = 0

                if self.num_model_chunks > 1:
                    label = f"B{mbs_id}c{chunk_id}"
                else:
                    label = f"B{mbs_id}"

                backward_steps.append({
                    'start_ms': step['start_ms'] - rank_time_offsets[rank_idx] + offset,
                    'end_ms': step['end_ms'] - rank_time_offsets[rank_idx] + offset,
                    'duration_ms': step['duration_ms'],
                    'mbs_id': mbs_id,
                    'chunk_id': chunk_id,
                    'label': label,
                })

            # Process P2P events (keep direction info)
            p2p_sends = []
            for step in result.get('p2p_send_gpu_list', []):
                entry = {
                    'start_ms': step['start_ms'] - rank_time_offsets[rank_idx] + offset,
                    'end_ms': step['end_ms'] - rank_time_offsets[rank_idx] + offset,
                    'duration_ms': step['duration_ms'],
                    'collective': step.get('collective'),
                    'dst_rank': step.get('dst_rank'),
                    'group_ranks': step.get('group_ranks'),
                    'label': step.get('label', 'Send'),
                }
                p2p_sends.append(entry)

            p2p_recvs = []
            for step in result.get('p2p_recv_gpu_list', []):
                entry = {
                    'start_ms': step['start_ms'] - rank_time_offsets[rank_idx] + offset,
                    'end_ms': step['end_ms'] - rank_time_offsets[rank_idx] + offset,
                    'duration_ms': step['duration_ms'],
                    'collective': step.get('collective'),
                    'dst_rank': step.get('dst_rank'),
                    'group_ranks': step.get('group_ranks'),
                    'label': step.get('label', 'Recv'),
                }
                p2p_recvs.append(entry)

            # Process optimizer events (time-aligned)
            optimizer_events = []
            if per_rank_optimizer:
                opt_data = per_rank_optimizer.get(rank_idx, {})
                phase_map = {
                    'grad_reduce_scatter': 'grad_reduce_scatter',
                    'allreduce': 'allreduce',
                    'loss_postprocessing': 'loss_postprocessing',
                    'optimizer_step': 'optimizer_step',
                    'training_log': 'training_log',
                }
                for phase_key, phase_name in phase_map.items():
                    for evt in opt_data.get(phase_key, []):
                        entry = {
                            'phase': phase_name,
                            'label': evt['label'],
                            'gpu_duration_ms': evt['gpu_duration_ms'],
                            'cpu_start_ms': evt['cpu_start_ms'] - rank_time_offsets[rank_idx] + offset,
                            'cpu_end_ms': evt['cpu_end_ms'] - rank_time_offsets[rank_idx] + offset,
                            'cpu_duration_ms': evt['cpu_duration_ms'],
                        }
                        if evt.get('gpu_start_ms') is not None:
                            entry['gpu_start_ms'] = evt['gpu_start_ms'] - rank_time_offsets[rank_idx] + offset
                            entry['gpu_end_ms'] = evt['gpu_end_ms'] - rank_time_offsets[rank_idx] + offset
                            # Use gpu_start_ms/gpu_end_ms as canonical start/end
                            entry['start_ms'] = entry['gpu_start_ms']
                            entry['end_ms'] = entry['gpu_end_ms']
                            entry['duration_ms'] = evt['gpu_duration_ms']
                        else:
                            entry['gpu_start_ms'] = None
                            entry['gpu_end_ms'] = None
                            # Fall back to CPU timing for span calculation
                            entry['start_ms'] = entry['cpu_start_ms']
                            entry['end_ms'] = entry['cpu_end_ms']
                            entry['duration_ms'] = evt['cpu_duration_ms']
                        optimizer_events.append(entry)

            ranks_data[rank_idx] = {
                'forward_steps': forward_steps,
                'backward_steps': backward_steps,
                'p2p_sends': p2p_sends,
                'p2p_recvs': p2p_recvs,
                'optimizer_events': optimizer_events,
            }

        # Step 3: Compute summary stats
        total_time_ms = 0.0
        per_rank_stats = []
        for rank_idx in range(self.pp_degree):
            rd = ranks_data[rank_idx]
            all_events = (rd['forward_steps'] + rd['backward_steps']
                          + rd['p2p_sends'] + rd['p2p_recvs']
                          + [e for e in rd['optimizer_events'] if e.get('start_ms') is not None])
            if not all_events:
                per_rank_stats.append({
                    'rank': rank_idx, 'compute_ms': 0.0, 'bubble_ms': 0.0, 'bubble_ratio': 0.0,
                    'fwd_mean_ms': 0.0, 'bwd_mean_ms': 0.0,
                })
                continue

            rank_start = min(e['start_ms'] for e in all_events)
            rank_end = max(e['end_ms'] for e in all_events)
            rank_span = rank_end - rank_start
            total_time_ms = max(total_time_ms, rank_span)

            compute_ms = sum(e['duration_ms'] for e in all_events)
            bubble_ms = rank_span - compute_ms
            bubble_ratio = bubble_ms / rank_span if rank_span > 0 else 0.0

            fwd_durations = [s['duration_ms'] for s in rd['forward_steps']]
            bwd_durations = [s['duration_ms'] for s in rd['backward_steps']]

            per_rank_stats.append({
                'rank': rank_idx,
                'compute_ms': round(compute_ms, 2),
                'bubble_ms': round(max(0, bubble_ms), 2),
                'bubble_ratio': round(max(0, bubble_ratio), 4),
                'fwd_mean_ms': round(float(np.mean(fwd_durations)), 2) if fwd_durations else 0.0,
                'bwd_mean_ms': round(float(np.mean(bwd_durations)), 2) if bwd_durations else 0.0,
            })

        avg_bubble_ratio = float(np.mean([s['bubble_ratio'] for s in per_rank_stats])) if per_rank_stats else 0.0

        return {
            'pp_degree': self.pp_degree,
            'num_microbatches': num_microbatches,
            'num_model_chunks': self.num_model_chunks,
            'schedule_table': schedule_table,
            'ranks': ranks_data,
            'summary': {
                'total_time_ms': round(total_time_ms, 2),
                'bubble_ratio': round(avg_bubble_ratio, 4),
                'per_rank_stats': per_rank_stats,
            },
        }

    def _merge_optimizer_results(self, per_rank_optimizer: Dict[int, Dict],
                                    merged_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge optimizer analysis results across all PP ranks.

        Uses the time-aligned optimizer events already stored in merged_result['ranks']
        and computes per-phase summary statistics.

        Args:
            per_rank_optimizer: {rank_idx: optimizer_result_dict} from analyze_optimizer_phase()
            merged_result: The merged result dict (with ranks data containing optimizer_events)

        Returns:
            Dict with per-rank optimizer detail and cross-rank summary.
        """
        phase_keys = ['grad_reduce_scatter', 'allreduce', 'loss_postprocessing',
                       'optimizer_step', 'training_log']
        phase_display = {
            'grad_reduce_scatter': 'Grad Reduce-Scatter',
            'allreduce': 'Allreduce',
            'loss_postprocessing': 'Loss Postprocessing',
            'optimizer_step': 'Optimizer Step',
            'training_log': 'Training Log',
        }

        ranks_data = merged_result.get('ranks', {})
        per_rank_detail = {}

        # Build per-rank detail from time-aligned optimizer events
        for rank_idx in sorted(ranks_data.keys()):
            rd = ranks_data[rank_idx]
            opt_events = rd.get('optimizer_events', [])

            rank_detail = {pk: [] for pk in phase_keys}
            for evt in opt_events:
                phase = evt.get('phase')
                if phase in rank_detail:
                    rank_detail[phase].append(evt)

            per_rank_detail[rank_idx] = rank_detail

        # Compute per-phase summary across ranks
        per_phase_summary = []
        for pk in phase_keys:
            row = {'phase': phase_display[pk]}
            rank_totals = []
            for rank_idx in sorted(per_rank_detail.keys()):
                events = per_rank_detail[rank_idx][pk]
                total_ms = sum(e['gpu_duration_ms'] for e in events)
                row[f'rank_{rank_idx}_ms'] = round(total_ms, 4)
                rank_totals.append(total_ms)
            row['mean_ms'] = round(float(np.mean(rank_totals)), 4) if rank_totals else 0.0
            per_phase_summary.append(row)

        # Total optimizer time averaged across ranks
        rank_total_opts = []
        for rank_idx in sorted(per_rank_detail.keys()):
            total = 0.0
            for pk in phase_keys:
                total += sum(e['gpu_duration_ms'] for e in per_rank_detail[rank_idx][pk])
            rank_total_opts.append(total)
        avg_total_optimizer = float(np.mean(rank_total_opts)) if rank_total_opts else 0.0

        return {
            'ranks': per_rank_detail,
            'summary': {
                'per_phase': per_phase_summary,
                'total_optimizer_ms': round(avg_total_optimizer, 4),
            }
        }

    def _merge_module_results(self, per_rank_modules: Dict[int, Dict]) -> Dict[str, Any]:
        """Merge module analysis results across all PP ranks.

        For each module, pool all events from all ranks together and recompute
        summary statistics. This treats the same module on different ranks as
        equivalent samples.

        Args:
            per_rank_modules: {rank_idx: module_result_dict} from analyze_modules()

        Returns:
            Merged module result dict with the same structure as analyze_modules()
        """
        # Collect all module names across ranks
        all_module_names = set()
        for mod_result in per_rank_modules.values():
            all_module_names.update(mod_result.keys())

        merged = {}
        for module_name in all_module_names:
            all_events = []
            # Pool events from all ranks
            for rank_idx in sorted(per_rank_modules.keys()):
                mod_data = per_rank_modules[rank_idx].get(module_name, {})
                events = mod_data.get('events', [])
                all_events.extend(events)

            if not all_events:
                merged[module_name] = {
                    'events': [],
                    'summary': {
                        'count': 0,
                        'mean_duration_ms': 0.0,
                        'std_duration_ms': 0.0,
                        'min_duration_ms': 0.0,
                        'max_duration_ms': 0.0,
                        'mean_delay_ms': 0.0,
                        'std_delay_ms': 0.0,
                    }
                }
                continue

            durations = [e['duration_ms'] for e in all_events]
            delays = [e.get('launch_delay_ms', 0.0) for e in all_events]

            summary = {
                'count': len(all_events),
                'mean_duration_ms': float(np.mean(durations)),
                'std_duration_ms': float(np.std(durations)),
                'min_duration_ms': float(np.min(durations)),
                'max_duration_ms': float(np.max(durations)),
                'mean_delay_ms': float(np.mean(delays)),
                'std_delay_ms': float(np.std(delays)),
            }

            # Carry over FLOPs/bandwidth from any rank that has it
            # (they should all be the same since model config is shared)
            sample_summary = None
            for rank_idx in sorted(per_rank_modules.keys()):
                mod_data = per_rank_modules[rank_idx].get(module_name, {})
                s = mod_data.get('summary', {})
                if s.get('flops') is not None or s.get('comm_volume_bytes') is not None:
                    sample_summary = s
                    break

            if sample_summary:
                flops = sample_summary.get('flops')
                comm_volume = sample_summary.get('comm_volume_bytes')

                summary['flops'] = flops
                if flops is not None and summary['mean_duration_ms'] > 0:
                    summary['mean_tflops'] = flops / (summary['mean_duration_ms'] * 1e-3) / 1e12
                else:
                    summary['mean_tflops'] = None
                if flops is not None and summary['max_duration_ms'] > 0:
                    summary['min_tflops'] = flops / (summary['max_duration_ms'] * 1e-3) / 1e12
                else:
                    summary['min_tflops'] = None
                if flops is not None and summary['min_duration_ms'] > 0:
                    summary['max_tflops'] = flops / (summary['min_duration_ms'] * 1e-3) / 1e12
                else:
                    summary['max_tflops'] = None

                summary['comm_volume_bytes'] = comm_volume
                if comm_volume is not None:
                    volume_gb = comm_volume / 1e9
                    summary['mean_bandwidth_gbs'] = volume_gb / (summary['mean_duration_ms'] * 1e-3) if summary['mean_duration_ms'] > 0 else None
                    summary['min_bandwidth_gbs'] = volume_gb / (summary['max_duration_ms'] * 1e-3) if summary['max_duration_ms'] > 0 else None
                    summary['max_bandwidth_gbs'] = volume_gb / (summary['min_duration_ms'] * 1e-3) if summary['min_duration_ms'] > 0 else None
                else:
                    summary['mean_bandwidth_gbs'] = None
                    summary['min_bandwidth_gbs'] = None
                    summary['max_bandwidth_gbs'] = None
            else:
                summary['flops'] = None
                summary['mean_tflops'] = None
                summary['min_tflops'] = None
                summary['max_tflops'] = None
                summary['comm_volume_bytes'] = None
                summary['mean_bandwidth_gbs'] = None
                summary['min_bandwidth_gbs'] = None
                summary['max_bandwidth_gbs'] = None

            merged[module_name] = {
                'events': all_events,
                'summary': summary,
            }

        return merged


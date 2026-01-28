# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for HyperConnection latency benchmarks.

Tests the latency of:
1. _projection_and_rms: Project input hidden states and apply RMS normalization
2. _compute_h: Compute h from projected hidden states and scaling factors
3. projection_rms + compute_h combined: Full mapping computation (excluding Sinkhorn)

Benchmarks are run with various tensor shapes to understand performance characteristics.
"""

import pytest
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestHyperConnectionLatency:
    """Benchmark latency tests for HyperConnection functions."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_hyper_connection_module(
        self, hidden_size: int, num_residual_streams: int
    ) -> HyperConnectionModule:
        """Create a HyperConnectionModule for testing."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=max(4, hidden_size // 64),  # Ensure valid num_heads
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_residual_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
        )
        module = HyperConnectionModule(config=config, layer_number=1)
        module.cuda()
        return module

    def _benchmark_function(
        self,
        func,
        inputs: Tuple,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> float:
        """
        Benchmark a function using CUDA events for accurate GPU timing.
        
        Args:
            func: Function to benchmark
            inputs: Tuple of input tensors
            num_warmup: Number of warmup iterations
            num_iterations: Number of measurement iterations
        
        Returns:
            Average latency in milliseconds
        """
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        for _ in range(num_warmup):
            _ = func(*inputs)
        
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start_event.record()
            _ = func(*inputs)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        
        return sum(latencies) / len(latencies)

    def _get_test_shapes(self) -> List[Dict]:
        """
        Get a list of test shapes covering various use cases.
        
        Returns:
            List of dicts with keys: seq_len, batch_size, hidden_size, num_streams
        """
        shapes = []
        
        # Vary sequence length
        for seq_len in [128, 256, 512, 1024, 2048, 4096]:
            shapes.append({
                'seq_len': seq_len,
                'batch_size': 2,
                'hidden_size': 4096,
                'num_streams': 4,
            })
        
        # Vary batch size
        for batch_size in [1, 2, 4, 8]:
            shapes.append({
                'seq_len': 1024,
                'batch_size': batch_size,
                'hidden_size': 4096,
                'num_streams': 4,
            })
        
        # Vary hidden size
        for hidden_size in [1024, 2048, 4096, 8192]:
            shapes.append({
                'seq_len': 1024,
                'batch_size': 2,
                'hidden_size': hidden_size,
                'num_streams': 4,
            })
        
        # Vary num_residual_streams
        for num_streams in [2, 4, 8]:
            shapes.append({
                'seq_len': 1024,
                'batch_size': 2,
                'hidden_size': 4096,
                'num_streams': num_streams,
            })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_shapes = []
        for shape in shapes:
            key = (shape['seq_len'], shape['batch_size'], shape['hidden_size'], shape['num_streams'])
            if key not in seen:
                seen.add(key)
                unique_shapes.append(shape)
        
        return unique_shapes

    def test_projection_rms_latency(self):
        """
        Benchmark _projection_and_rms function latency across various shapes.
        """
        results = []
        test_shapes = self._get_test_shapes()
        
        for shape in test_shapes:
            seq_len = shape['seq_len']
            batch_size = shape['batch_size']
            hidden_size = shape['hidden_size']
            num_streams = shape['num_streams']
            
            module = self._create_hyper_connection_module(hidden_size, num_streams)
            
            # Create input tensor: [s, b, n*C]
            x = torch.randn(
                seq_len, batch_size, num_streams * hidden_size,
                device='cuda', dtype=torch.float32
            )
            
            # Benchmark
            latency_ms = self._benchmark_function(
                module._projection_and_rms,
                (x,),
                num_warmup=10,
                num_iterations=100,
            )
            
            results.append({
                'shape': f"s={seq_len}, b={batch_size}, C={hidden_size}, n={num_streams}",
                'seq_len': seq_len,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'num_streams': num_streams,
                'latency_ms': latency_ms,
            })
            
            # Clean up
            del module, x
            torch.cuda.empty_cache()
        
        # Print results table
        self._print_results_table("projection_rms", results)
        
        # Store results as class attribute for combined reporting
        self._projection_rms_results = results

    def test_compute_h_latency(self):
        """
        Benchmark _compute_h function latency across various shapes.
        """
        results = []
        test_shapes = self._get_test_shapes()
        
        for shape in test_shapes:
            seq_len = shape['seq_len']
            batch_size = shape['batch_size']
            hidden_size = shape['hidden_size']
            num_streams = shape['num_streams']
            
            module = self._create_hyper_connection_module(hidden_size, num_streams)
            
            # Create input tensors matching _compute_h signature
            # proj: [s, b, n^2 + 2n]
            proj = torch.randn(
                seq_len, batch_size, num_streams * num_streams + 2 * num_streams,
                device='cuda', dtype=torch.float32
            )
            # r: [s, b, 1]
            r = torch.randn(
                seq_len, batch_size, 1,
                device='cuda', dtype=torch.float32
            )
            
            # Benchmark
            latency_ms = self._benchmark_function(
                module._compute_h,
                (proj, r),
                num_warmup=10,
                num_iterations=100,
            )
            
            results.append({
                'shape': f"s={seq_len}, b={batch_size}, C={hidden_size}, n={num_streams}",
                'seq_len': seq_len,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'num_streams': num_streams,
                'latency_ms': latency_ms,
            })
            
            # Clean up
            del module, proj, r
            torch.cuda.empty_cache()
        
        # Print results table
        self._print_results_table("compute_h", results)
        
        # Store results as class attribute for combined reporting
        self._compute_h_results = results

    def test_combined_projection_and_compute_h_latency(self):
        """
        Benchmark combined _projection_and_rms + _compute_h function latency.
        This represents the full mapping computation (excluding Sinkhorn).
        """
        results = []
        test_shapes = self._get_test_shapes()
        
        for shape in test_shapes:
            seq_len = shape['seq_len']
            batch_size = shape['batch_size']
            hidden_size = shape['hidden_size']
            num_streams = shape['num_streams']
            
            module = self._create_hyper_connection_module(hidden_size, num_streams)
            
            # Create input tensor: [s, b, n*C]
            x = torch.randn(
                seq_len, batch_size, num_streams * hidden_size,
                device='cuda', dtype=torch.float32
            )
            
            # Define combined function
            def combined_func(x):
                proj, r = module._projection_and_rms(x)
                h_pre, h_post, h_res = module._compute_h(proj, r)
                return h_pre, h_post, h_res
            
            # Benchmark
            latency_ms = self._benchmark_function(
                combined_func,
                (x,),
                num_warmup=10,
                num_iterations=100,
            )
            
            results.append({
                'shape': f"s={seq_len}, b={batch_size}, C={hidden_size}, n={num_streams}",
                'seq_len': seq_len,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'num_streams': num_streams,
                'latency_ms': latency_ms,
            })
            
            # Clean up
            del module, x
            torch.cuda.empty_cache()
        
        # Print results table
        self._print_results_table("combined (projection_rms + compute_h)", results)
        
        # Store results as class attribute for combined reporting
        self._combined_results = results

    def test_all_scenarios_summary(self):
        """
        Run all three scenarios and print a combined summary table.
        """
        test_shapes = self._get_test_shapes()
        
        projection_rms_results = []
        compute_h_results = []
        combined_results = []
        
        for shape in test_shapes:
            seq_len = shape['seq_len']
            batch_size = shape['batch_size']
            hidden_size = shape['hidden_size']
            num_streams = shape['num_streams']
            
            module = self._create_hyper_connection_module(hidden_size, num_streams)
            
            # Create input tensor: [s, b, n*C]
            x = torch.randn(
                seq_len, batch_size, num_streams * hidden_size,
                device='cuda', dtype=torch.float32
            )
            
            # Benchmark projection_rms
            proj_rms_latency = self._benchmark_function(
                module._projection_and_rms,
                (x,),
                num_warmup=10,
                num_iterations=100,
            )
            
            # Create inputs for compute_h
            proj = torch.randn(
                seq_len, batch_size, num_streams * num_streams + 2 * num_streams,
                device='cuda', dtype=torch.float32
            )
            r = torch.randn(
                seq_len, batch_size, 1,
                device='cuda', dtype=torch.float32
            )
            
            # Benchmark compute_h
            compute_h_latency = self._benchmark_function(
                module._compute_h,
                (proj, r),
                num_warmup=10,
                num_iterations=100,
            )
            
            # Benchmark combined
            def combined_func(x):
                proj_out, r_out = module._projection_and_rms(x)
                h_pre, h_post, h_res = module._compute_h(proj_out, r_out)
                return h_pre, h_post, h_res
            
            combined_latency = self._benchmark_function(
                combined_func,
                (x,),
                num_warmup=10,
                num_iterations=100,
            )
            
            shape_str = f"s={seq_len}, b={batch_size}, C={hidden_size}, n={num_streams}"
            
            projection_rms_results.append({
                'shape': shape_str,
                'latency_ms': proj_rms_latency,
            })
            compute_h_results.append({
                'shape': shape_str,
                'latency_ms': compute_h_latency,
            })
            combined_results.append({
                'shape': shape_str,
                'latency_ms': combined_latency,
            })
            
            # Clean up
            del module, x, proj, r
            torch.cuda.empty_cache()
        
        # Print combined summary table
        self._print_combined_summary(
            projection_rms_results, compute_h_results, combined_results
        )

    def _print_results_table(self, scenario_name: str, results: List[Dict]):
        """Print a formatted results table for a single scenario."""
        print(f"\n{'=' * 80}")
        print(f"Latency Results for: {scenario_name}")
        print(f"{'=' * 80}")
        print(f"{'Shape':<45} | {'Latency (ms)':>15}")
        print(f"{'-' * 45}-+-{'-' * 15}")
        
        for result in results:
            print(f"{result['shape']:<45} | {result['latency_ms']:>15.4f}")
        
        print(f"{'=' * 80}\n")

    def _print_combined_summary(
        self,
        projection_rms_results: List[Dict],
        compute_h_results: List[Dict],
        combined_results: List[Dict],
    ):
        """Print a combined summary table for all three scenarios."""
        print(f"\n{'=' * 100}")
        print("COMBINED LATENCY SUMMARY")
        print(f"{'=' * 100}")
        print(
            f"{'Shape':<40} | {'projection_rms':>15} | {'compute_h':>15} | {'combined':>15}"
        )
        print(f"{'-' * 40}-+-{'-' * 15}-+-{'-' * 15}-+-{'-' * 15}")
        
        for proj, comp, comb in zip(projection_rms_results, compute_h_results, combined_results):
            print(
                f"{proj['shape']:<40} | "
                f"{proj['latency_ms']:>12.4f} ms | "
                f"{comp['latency_ms']:>12.4f} ms | "
                f"{comb['latency_ms']:>12.4f} ms"
            )
        
        print(f"{'=' * 100}")
        
        # Calculate and print statistics
        print(f"\n{'=' * 60}")
        print("STATISTICS")
        print(f"{'=' * 60}")
        
        proj_latencies = [r['latency_ms'] for r in projection_rms_results]
        comp_latencies = [r['latency_ms'] for r in compute_h_results]
        comb_latencies = [r['latency_ms'] for r in combined_results]
        
        print(f"projection_rms:  min={min(proj_latencies):.4f}ms, max={max(proj_latencies):.4f}ms, avg={sum(proj_latencies)/len(proj_latencies):.4f}ms")
        print(f"compute_h:       min={min(comp_latencies):.4f}ms, max={max(comp_latencies):.4f}ms, avg={sum(comp_latencies)/len(comp_latencies):.4f}ms")
        print(f"combined:        min={min(comb_latencies):.4f}ms, max={max(comb_latencies):.4f}ms, avg={sum(comb_latencies)/len(comb_latencies):.4f}ms")
        print(f"{'=' * 60}\n")


class TestHyperConnectionLatencyParameterized:
    """
    Parameterized latency tests for more granular control over test shapes.
    Run with: pytest test_hyper_connection_latency.py -v -k "Parameterized" -s
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_hyper_connection_module(
        self, hidden_size: int, num_residual_streams: int
    ) -> HyperConnectionModule:
        """Create a HyperConnectionModule for testing."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=max(4, hidden_size // 64),
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_residual_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
        )
        module = HyperConnectionModule(config=config, layer_number=1)
        module.cuda()
        return module

    @pytest.mark.parametrize("seq_len", [128, 512, 2048, 4096])
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("hidden_size", [2048, 4096])
    @pytest.mark.parametrize("num_streams", [4])
    def test_parameterized_projection_rms(
        self, seq_len: int, batch_size: int, hidden_size: int, num_streams: int
    ):
        """Parameterized test for projection_rms latency."""
        module = self._create_hyper_connection_module(hidden_size, num_streams)
        
        x = torch.randn(
            seq_len, batch_size, num_streams * hidden_size,
            device='cuda', dtype=torch.float32
        )
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        for _ in range(10):
            _ = module._projection_and_rms(x)
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(100):
            start_event.record()
            _ = module._projection_and_rms(x)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n[projection_rms] s={seq_len}, b={batch_size}, C={hidden_size}, n={num_streams}: {avg_latency:.4f} ms")


def run_latency_benchmark():
    """
    Standalone function to run the latency benchmark.
    Can be called directly without pytest for quick testing.
    
    Usage:
        python -c "from tests.unit_tests.transformer.test_hyper_connection_latency import run_latency_benchmark; run_latency_benchmark()"
    """
    import sys
    sys.path.insert(0, '/Users/jingqiny/Desktop/Projects/Megatron-LM')
    
    test = TestHyperConnectionLatency()
    test.setup_method(None)
    try:
        test.test_all_scenarios_summary()
    finally:
        test.teardown_method(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-k", "test_all_scenarios_summary"])

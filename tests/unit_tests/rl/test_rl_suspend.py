# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for RL suspend/resume functionality.

These tests cover:
1. DynamicInferenceContext allocation/deallocation with various KV cache modes
2. DynamicInferenceEngine suspend/resume with different CUDA graph configurations
3. GPU memory behavior during suspend/resume cycles
4. Request recompute marking for partial rollouts
"""

import asyncio
import gc
import itertools
import random
import types
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils


def set_rounder(value):
    """Utility function to set the DynamicInferenceContext rounder."""
    DynamicInferenceContext.ROUNDER = value
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


def get_gpu_memory_allocated() -> int:
    """Get current GPU memory allocated in bytes."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def get_gpu_memory_reserved() -> int:
    """Get current GPU memory reserved in bytes."""
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved()


def force_gc():
    """Force garbage collection and empty CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@dataclass
class SuspendTestConfig:
    """Test configuration for suspend/resume tests."""

    random_seed: int = 123
    vocab_size: int = 100
    num_requests: int = 4
    min_prompt_length: int = 4
    max_prompt_length: int = 16
    num_tokens_to_generate: int = 8
    max_sequence_length: Optional[int] = None

    # Context configuration
    context_buffer_size_gb: float = 0.05
    context_block_size_tokens: int = 64
    context_max_requests: Optional[int] = None
    context_max_tokens: Optional[int] = None

    # Model configuration
    num_layers: int = 2
    hidden_size: int = 32
    num_attention_heads: int = 4
    kv_channels: int = 8

    # Suspend/resume configuration
    persist_cuda_graphs: bool = False
    offload_kv_cache: bool = False
    remove_kv_cache: bool = False
    unified_memory_level: int = 0
    num_cuda_graphs: Optional[int] = None

    def __post_init__(self):
        if self.max_sequence_length is None:
            self.max_sequence_length = self.max_prompt_length + self.num_tokens_to_generate

        set_rounder(4)


class TestDynamicContextSuspend:
    """Tests for DynamicInferenceContext suspend/resume (allocate/deallocate tensors)."""

    def setup_method(self, method):
        """Setup model parallel for each test."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        set_rounder(4)

    def teardown_method(self, method):
        """Teardown model parallel after each test."""
        set_rounder(64)
        Utils.destroy_model_parallel()
        force_gc()

    def _create_context(
        self,
        config: SuspendTestConfig,
    ) -> DynamicInferenceContext:
        """Create a DynamicInferenceContext with the given configuration."""
        context = DynamicInferenceContext(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=config.num_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            max_requests=config.context_max_requests,
            max_tokens=config.context_max_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            mamba_inference_state_config=None,
            materialize_only_last_token_logits=True,
            unified_memory_level=config.unified_memory_level,
            persist_cuda_graphs=config.persist_cuda_graphs,
            offload_kv_cache=config.offload_kv_cache,
            remove_kv_cache=config.remove_kv_cache,
        )
        return context

    def _add_test_requests(
        self, context: DynamicInferenceContext, config: SuspendTestConfig
    ) -> List[DynamicInferenceRequest]:
        """Add test requests to the context."""
        random.seed(config.random_seed)
        requests = []

        for request_id in range(config.num_requests):
            prompt_length = random.randint(config.min_prompt_length, config.max_prompt_length)
            prompt_tokens = torch.randint(
                0,
                config.vocab_size - 1,
                (prompt_length,),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            request = DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=config.num_tokens_to_generate,
                    termination_id=config.vocab_size - 1,
                ),
            )
            context.add_request(request)
            requests.append(request)

        return requests

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_context_basic_allocate_deallocate(self):
        """Test basic tensor allocation and deallocation without any special modes."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=False,
            unified_memory_level=0,
        )
        context = self._create_context(config)

        # Verify tensors are allocated after init
        assert context.is_tensor_state_allocated
        assert hasattr(context, 'memory_buffer')
        assert context.memory_buffer is not None

        initial_memory = get_gpu_memory_allocated()

        # Deallocate tensors
        context.deallocate_all_tensors()

        assert not context.is_tensor_state_allocated

        # Memory should not change because deallocate is a no-op without remove_kv_cache
        # (when not using UVM or torch_memory_saver and not explicitly removing)
        current_memory = get_gpu_memory_allocated()
        # With basic config (no offload, no remove), memory_buffer stays allocated
        # since we're not in any special mode

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_context_remove_kv_cache_mode(self):
        """Test that remove_kv_cache mode properly deletes tensors on deallocate."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )
        context = self._create_context(config)

        # Add some requests so we have meaningful state
        requests = self._add_test_requests(context, config)
        context.initialize_attention_state()

        # Record memory with tensors allocated
        force_gc()
        memory_with_tensors = get_gpu_memory_allocated()

        # Verify tensors are allocated
        assert context.is_tensor_state_allocated
        assert hasattr(context, 'memory_buffer')

        # Deallocate tensors
        context.deallocate_all_tensors()

        assert not context.is_tensor_state_allocated

        # Force garbage collection
        force_gc()
        memory_after_deallocate = get_gpu_memory_allocated()

        # Memory should decrease after deallocating in remove_kv_cache mode
        assert memory_after_deallocate < memory_with_tensors, (
            f"Memory should decrease after deallocate in remove_kv_cache mode. "
            f"Before: {memory_with_tensors / 1e6:.2f} MB, After: {memory_after_deallocate / 1e6:.2f} MB"
        )

        # memory_buffer should be deleted
        assert not hasattr(context, 'memory_buffer'), "memory_buffer should be deleted"

        # Reallocate tensors
        context.allocate_all_tensors(is_init=False)

        assert context.is_tensor_state_allocated
        assert hasattr(context, 'memory_buffer')
        assert context.memory_buffer is not None

        # Memory should be restored
        force_gc()
        memory_after_allocate = get_gpu_memory_allocated()
        # Memory should increase back to approximately original level
        assert memory_after_allocate > memory_after_deallocate

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @patch('megatron.core.inference.contexts.dynamic_context.HAVE_TORCH_MEMORY_SAVER', False)
    def test_context_offload_kv_cache_mode(self):
        """Test that offload_kv_cache mode properly manages tensor storage."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=True,
            remove_kv_cache=False,
            unified_memory_level=0,
        )
        context = self._create_context(config)

        # Add some requests so we have meaningful state
        requests = self._add_test_requests(context, config)
        context.initialize_attention_state()

        force_gc()
        memory_with_tensors = get_gpu_memory_allocated()

        # Verify tensors are allocated
        assert context.is_tensor_state_allocated

        # Get list of tracked tensor names (these should be offloaded)
        tracked_names = context._offloadable_tensor_names.copy()
        assert len(tracked_names) > 0, "Should have tracked tensors in offload mode"

        # Deallocate (offload) tensors
        context.deallocate_all_tensors()

        assert not context.is_tensor_state_allocated

        # Check that CPU backups were created
        assert len(context._offloadable_cpu_backups) > 0, "Should have CPU backups"
        assert len(context._offloadable_storage_sizes) > 0, "Should have storage sizes recorded"

        # GPU memory should decrease (tracked tensors' storage resized to 0)
        force_gc()
        memory_after_offload = get_gpu_memory_allocated()
        assert memory_after_offload < memory_with_tensors, (
            f"Memory should decrease after offload. "
            f"Before: {memory_with_tensors / 1e6:.2f} MB, After: {memory_after_offload / 1e6:.2f} MB"
        )

        # Verify the tracked tensors have storage size 0
        for name in tracked_names:
            tensor = getattr(context, name, None)
            if tensor is not None:
                assert tensor.storage().size() == 0, f"Tensor {name} storage should be 0"

        # Reallocate (restore) tensors
        context.allocate_all_tensors(is_init=False)

        assert context.is_tensor_state_allocated

        # Verify tracked tensors are restored
        for name in tracked_names:
            tensor = getattr(context, name, None)
            if tensor is not None:
                assert tensor.storage().size() > 0, f"Tensor {name} should have restored storage"

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_context_multiple_suspend_resume_cycles(self):
        """Test multiple suspend/resume cycles to ensure consistency."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )
        context = self._create_context(config)

        # Initial state
        force_gc()
        initial_memory = get_gpu_memory_allocated()

        for cycle in range(3):
            # Add requests
            requests = self._add_test_requests(context, config)
            context.initialize_attention_state()

            force_gc()
            memory_with_requests = get_gpu_memory_allocated()
            assert context.is_tensor_state_allocated

            # Deallocate
            context.deallocate_all_tensors()
            assert not context.is_tensor_state_allocated

            force_gc()
            memory_deallocated = get_gpu_memory_allocated()
            assert memory_deallocated < memory_with_requests, f"Cycle {cycle}: Memory should decrease"

            # Reallocate
            context.allocate_all_tensors(is_init=False)
            assert context.is_tensor_state_allocated

            # Reset context for next cycle
            context.reset()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_context_deallocate_idempotent(self):
        """Test that multiple deallocate calls are safe (idempotent)."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )
        context = self._create_context(config)

        # First deallocate
        context.deallocate_all_tensors()
        assert not context.is_tensor_state_allocated

        # Second deallocate should be a no-op
        context.deallocate_all_tensors()
        assert not context.is_tensor_state_allocated

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_context_allocate_idempotent(self):
        """Test that multiple allocate calls are safe (idempotent)."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )
        context = self._create_context(config)

        # Initial allocation happens in __init__
        assert context.is_tensor_state_allocated
        initial_buffer_id = id(context.memory_buffer)

        # Second allocate should be a no-op
        context.allocate_all_tensors(is_init=False)
        assert context.is_tensor_state_allocated
        assert id(context.memory_buffer) == initial_buffer_id, "Buffer should not be reallocated"


class TestDynamicEngineSuspend:
    """Tests for DynamicInferenceEngine suspend/resume."""

    def setup_method(self, method):
        """Setup model parallel for each test."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        set_rounder(4)

        # Reset global cuda graph state
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None

    def teardown_method(self, method):
        """Teardown model parallel after each test."""
        set_rounder(64)
        Utils.destroy_model_parallel()
        force_gc()

    def _build_engine(
        self,
        config: SuspendTestConfig,
        enable_cuda_graph: bool = False,
    ) -> DynamicInferenceEngine:
        """Build a DynamicInferenceEngine with the given configuration."""
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        model_parallel_cuda_manual_seed(
            seed=config.random_seed, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

        # Transformer config
        transformer_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="local" if enable_cuda_graph else "none",
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=True,
        )

        # GPT model
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            parallel_output=True,
            pre_process=True,
            post_process=True,
        ).cuda()

        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16)
        model.eval()

        # Inference config
        inference_config = InferenceWrapperConfig(
            hidden_size=transformer_config.hidden_size,
            inference_batch_times_seqlen_threshold=400,
            fp32_residual_connection=False,
            params_dtype=transformer_config.params_dtype,
            padded_vocab_size=config.vocab_size,
        )

        # Inference context
        context = DynamicInferenceContext(
            params_dtype=transformer_config.params_dtype,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=config.num_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            max_requests=config.context_max_requests,
            max_tokens=config.context_max_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=config.unified_memory_level,
            persist_cuda_graphs=config.persist_cuda_graphs,
            offload_kv_cache=config.offload_kv_cache,
            remove_kv_cache=config.remove_kv_cache,
        )

        # Inference model wrapper
        inference_wrapped_model = GPTInferenceWrapper(model, inference_config, context)
        inference_wrapped_model.model_is_pipeline_parallel = False

        # Text generation controller
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=types.SimpleNamespace(
                vocab_size=config.vocab_size, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )

        # Inference engine
        engine = DynamicInferenceEngine(
            text_generation_controller,
            context,
            random_seed=config.random_seed,
            enable_cuda_graph=enable_cuda_graph,
            persist_cuda_graphs=config.persist_cuda_graphs,
        )

        return engine

    def _add_requests_to_engine(
        self, engine: DynamicInferenceEngine, config: SuspendTestConfig
    ) -> List[DynamicInferenceRequest]:
        """Add test requests to the engine."""
        random.seed(config.random_seed)
        requests = []

        for request_id in range(config.num_requests):
            prompt_length = random.randint(config.min_prompt_length, config.max_prompt_length)
            prompt_tokens = torch.randint(
                0,
                config.vocab_size - 1,
                (prompt_length,),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            request = DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=config.num_tokens_to_generate,
                    termination_id=config.vocab_size - 1,
                ),
            )
            engine._add_request(request)
            requests.append(request)

        return requests

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_engine_basic_suspend_resume(self):
        """Test basic engine suspend/resume without CUDA graphs."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )
        engine = self._build_engine(config, enable_cuda_graph=False)

        # Add requests
        requests = self._add_requests_to_engine(engine, config)

        assert not engine.is_suspended
        assert engine.context.is_tensor_state_allocated

        force_gc()
        memory_before_suspend = get_gpu_memory_allocated()

        # Suspend
        engine.suspend()

        assert engine.is_suspended
        assert not engine.context.is_tensor_state_allocated

        force_gc()
        memory_after_suspend = get_gpu_memory_allocated()
        assert memory_after_suspend < memory_before_suspend, (
            f"Memory should decrease after suspend. "
            f"Before: {memory_before_suspend / 1e6:.2f} MB, After: {memory_after_suspend / 1e6:.2f} MB"
        )

        # Resume
        engine.resume()

        assert not engine.is_suspended
        assert engine.context.is_tensor_state_allocated

        force_gc()
        memory_after_resume = get_gpu_memory_allocated()
        assert memory_after_resume > memory_after_suspend

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_engine_suspend_resume_idempotent(self):
        """Test that multiple suspend/resume calls are safe."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )
        engine = self._build_engine(config, enable_cuda_graph=False)

        # Multiple suspends should be safe
        engine.suspend()
        assert engine.is_suspended
        engine.suspend()  # Should be no-op
        assert engine.is_suspended

        # Multiple resumes should be safe
        engine.resume()
        assert not engine.is_suspended
        engine.resume()  # Should be no-op
        assert not engine.is_suspended

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_engine_suspend_preserves_request_ids(self):
        """Test that suspend preserves request IDs for resume."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )
        engine = self._build_engine(config, enable_cuda_graph=False)

        # Add requests
        requests = self._add_requests_to_engine(engine, config)
        original_request_ids = set(engine.requests.keys())

        # Suspend
        engine.suspend()

        # Check resume_request_ids contains all original requests
        assert set(engine.resume_request_ids) == original_request_ids

        # Resume
        engine.resume()

        # Requests should still be accessible
        assert set(engine.requests.keys()) == original_request_ids

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_engine_recompute_soon_flag(self):
        """Test that recompute_soon flag is properly handled during suspend/resume."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )
        engine = self._build_engine(config, enable_cuda_graph=False)

        # Add requests
        requests = self._add_requests_to_engine(engine, config)

        # Mark some requests for recompute
        request_ids = list(engine.requests.keys())
        for i, request_id in enumerate(request_ids):
            # Mark every other request for recompute
            engine.requests[request_id].recompute_soon = (i % 2 == 0)

        # Record which requests were marked
        marked_for_recompute = {
            rid for rid in request_ids if engine.requests[rid].recompute_soon
        }

        # Suspend
        engine.suspend()

        # Check that marked requests were checkpointed
        for request_id in marked_for_recompute:
            # Verify the request record has a checkpoint
            record = engine.requests[request_id].record
            assert len(record) >= 1, f"Request {request_id} should have a record"

        # Resume
        engine.resume()

        # Check that recompute_soon flags are cleared for marked requests
        for request_id in marked_for_recompute:
            assert not engine.requests[request_id].recompute_soon, (
                f"Request {request_id} recompute_soon should be False after resume"
            )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_engine_multiple_cycles(self):
        """Test multiple suspend/resume cycles."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )
        engine = self._build_engine(config, enable_cuda_graph=False)

        for cycle in range(3):
            # Add requests if first cycle
            if cycle == 0:
                requests = self._add_requests_to_engine(engine, config)
            else:
                # Mark all requests for recompute
                for request_entry in engine.requests.values():
                    request_entry.recompute_soon = True

            assert not engine.is_suspended

            # Suspend
            engine.suspend()
            assert engine.is_suspended

            # Resume
            engine.resume()
            assert not engine.is_suspended

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("num_cuda_graphs", [None, 1, 2])
    def test_engine_cuda_graph_handling_no_persist(self, num_cuda_graphs):
        """Test CUDA graph deletion/recreation when persist_cuda_graphs=False."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
            num_cuda_graphs=num_cuda_graphs,
        )
        enable_cuda_graph = num_cuda_graphs is not None and num_cuda_graphs > 0

        engine = self._build_engine(config, enable_cuda_graph=enable_cuda_graph)

        # Check initial CUDA graph state
        initial_cuda_graph_count = len(_CudagraphGlobalRecord.cudagraph_record)

        # Suspend should delete CUDA graphs when not persisting
        engine.suspend()

        if enable_cuda_graph:
            # CUDA graphs should be deleted
            assert len(_CudagraphGlobalRecord.cudagraph_record) == 0, (
                "CUDA graphs should be deleted after suspend when not persisting"
            )

        # Resume should recreate CUDA graphs
        engine.resume()

        # Note: CUDA graphs are created in create_cuda_graphs() which happens during resume


class TestRequestRecompute:
    """Tests for request recompute marking functionality."""

    def setup_method(self, method):
        """Setup model parallel for each test."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        set_rounder(4)

    def teardown_method(self, method):
        """Teardown model parallel after each test."""
        set_rounder(64)
        Utils.destroy_model_parallel()
        force_gc()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_mark_for_recompute_basic(self):
        """Test basic mark_for_recompute functionality on request entries."""
        from megatron.core.inference.engines.dynamic_engine import RequestEntry
        from megatron.core.inference.inference_request import DynamicInferenceRequestRecord

        # Create a mock request and entry
        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.tensor([1, 2, 3], device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=5),
        )
        record = DynamicInferenceRequestRecord.from_request(request)
        entry = RequestEntry(record=record, future=asyncio.Future())

        # Initially recompute_soon should be False
        assert not entry.recompute_soon

        # Mark for recompute
        entry.recompute_soon = True
        assert entry.recompute_soon

        # Unmark
        entry.recompute_soon = False
        assert not entry.recompute_soon


class TestArgumentCombinations:
    """Tests for various argument combinations for suspend/resume."""

    def setup_method(self, method):
        """Setup model parallel for each test."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        set_rounder(4)

        # Reset global cuda graph state
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None

    def teardown_method(self, method):
        """Teardown model parallel after each test."""
        set_rounder(64)
        Utils.destroy_model_parallel()
        force_gc()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "remove_kv_cache,offload_kv_cache",
        [
            (True, False),
            (False, False),
            # Note: (True, True) is invalid and (False, True) requires torch_memory_saver
        ],
    )
    def test_kv_cache_modes_context(self, remove_kv_cache, offload_kv_cache):
        """Test context with different KV cache modes."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=offload_kv_cache,
            remove_kv_cache=remove_kv_cache,
            unified_memory_level=0,
        )

        context = DynamicInferenceContext(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=None,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            max_requests=config.context_max_requests,
            max_tokens=config.context_max_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=0,
            persist_cuda_graphs=False,
            offload_kv_cache=offload_kv_cache,
            remove_kv_cache=remove_kv_cache,
        )

        # Verify initial state
        assert context.is_tensor_state_allocated
        assert context.offload_kv_cache == offload_kv_cache
        assert context.remove_kv_cache == remove_kv_cache

        # Deallocate
        context.deallocate_all_tensors()
        assert not context.is_tensor_state_allocated

        # Allocate
        context.allocate_all_tensors(is_init=False)
        assert context.is_tensor_state_allocated

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_invalid_offload_and_remove_combination(self):
        """Test that offload and remove KV cache cannot both be True."""
        config = SuspendTestConfig()

        with pytest.raises(AssertionError, match="Cannot both offload and remove"):
            DynamicInferenceContext(
                params_dtype=torch.bfloat16,
                num_layers=config.num_layers,
                kv_channels=config.kv_channels,
                num_attention_heads=config.num_attention_heads,
                max_sequence_length=config.max_sequence_length,
                num_cuda_graphs=None,
                use_cuda_graphs_for_non_decode_steps=True,
                buffer_size_gb=config.context_buffer_size_gb,
                paused_buffer_size_gb=0.01,
                block_size_tokens=config.context_block_size_tokens,
                max_requests=config.context_max_requests,
                max_tokens=config.context_max_tokens,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                unified_memory_level=0,
                persist_cuda_graphs=False,
                offload_kv_cache=True,
                remove_kv_cache=True,
            )


class TestEngineStepWithSuspend:
    """Tests that run engine steps before and after suspend/resume."""

    def setup_method(self, method):
        """Setup model parallel for each test."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        set_rounder(4)

        # Reset global cuda graph state
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None

    def teardown_method(self, method):
        """Teardown model parallel after each test."""
        set_rounder(64)
        Utils.destroy_model_parallel()
        force_gc()

    def _build_engine(self, config: SuspendTestConfig) -> DynamicInferenceEngine:
        """Build a DynamicInferenceEngine with the given configuration."""
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        model_parallel_cuda_manual_seed(
            seed=config.random_seed, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

        # Transformer config
        transformer_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="none",
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=True,
        )

        # GPT model
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            parallel_output=True,
            pre_process=True,
            post_process=True,
        ).cuda()

        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16)
        model.eval()

        # Inference config
        inference_config = InferenceWrapperConfig(
            hidden_size=transformer_config.hidden_size,
            inference_batch_times_seqlen_threshold=400,
            fp32_residual_connection=False,
            params_dtype=transformer_config.params_dtype,
            padded_vocab_size=config.vocab_size,
        )

        # Inference context
        context = DynamicInferenceContext(
            params_dtype=transformer_config.params_dtype,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=config.num_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            max_requests=config.context_max_requests,
            max_tokens=config.context_max_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=config.unified_memory_level,
            persist_cuda_graphs=config.persist_cuda_graphs,
            offload_kv_cache=config.offload_kv_cache,
            remove_kv_cache=config.remove_kv_cache,
        )

        # Inference model wrapper
        inference_wrapped_model = GPTInferenceWrapper(model, inference_config, context)
        inference_wrapped_model.model_is_pipeline_parallel = False

        # Text generation controller
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=types.SimpleNamespace(
                vocab_size=config.vocab_size, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )

        # Inference engine
        engine = DynamicInferenceEngine(
            text_generation_controller,
            context,
            random_seed=config.random_seed,
            enable_cuda_graph=False,
            persist_cuda_graphs=config.persist_cuda_graphs,
        )

        return engine

    def _run_engine_step(self, engine: DynamicInferenceEngine) -> dict:
        """Run one step of the engine."""
        return engine.step_modern()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_step_suspend_step(self):
        """Test: add requests -> step -> suspend -> resume -> step."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
            num_requests=2,
            num_tokens_to_generate=4,
        )
        engine = self._build_engine(config)

        # Add requests
        random.seed(config.random_seed)
        for request_id in range(config.num_requests):
            prompt_length = random.randint(config.min_prompt_length, config.max_prompt_length)
            prompt_tokens = torch.randint(
                0, config.vocab_size - 1, (prompt_length,), dtype=torch.int64, device='cuda'
            )
            request = DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=config.num_tokens_to_generate,
                    termination_id=config.vocab_size - 1,
                ),
            )
            engine._add_request(request)

        # Run a few steps
        for _ in range(2):
            self._run_engine_step(engine)

        # Record state before suspend
        requests_before = set(engine.requests.keys())
        active_count_before = engine.context.total_request_count

        # Suspend
        engine.suspend()
        assert engine.is_suspended

        # Simulate some "training" work (just clear memory tracking)
        force_gc()

        # Mark all requests for recompute (simulating partial rollout scenario)
        for entry in engine.requests.values():
            entry.recompute_soon = True

        # Resume
        engine.resume()
        assert not engine.is_suspended

        # Verify requests are restored
        assert set(engine.requests.keys()) == requests_before

        # Run more steps until completion
        while engine.has_unfinished_requests():
            self._run_engine_step(engine)

        # Verify all requests completed
        for request_entry in engine.requests.values():
            request = request_entry.record[-1]
            assert request.status in (Status.COMPLETED, Status.FAILED)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_multiple_suspend_resume_with_steps(self):
        """Test multiple suspend/resume cycles with engine steps in between."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
            num_requests=2,
            num_tokens_to_generate=8,
        )
        engine = self._build_engine(config)

        # Add requests
        random.seed(config.random_seed)
        for request_id in range(config.num_requests):
            prompt_length = random.randint(config.min_prompt_length, config.max_prompt_length)
            prompt_tokens = torch.randint(
                0, config.vocab_size - 1, (prompt_length,), dtype=torch.int64, device='cuda'
            )
            request = DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=config.num_tokens_to_generate,
                    termination_id=config.vocab_size - 1,
                ),
            )
            engine._add_request(request)

        # Multiple cycles of step -> suspend -> resume
        for cycle in range(3):
            # Run a step
            if engine.has_unfinished_requests():
                self._run_engine_step(engine)

            # Suspend
            engine.suspend()
            assert engine.is_suspended

            # Mark for recompute
            for entry in engine.requests.values():
                entry.recompute_soon = True

            # Resume
            engine.resume()
            assert not engine.is_suspended

        # Complete all requests
        while engine.has_unfinished_requests():
            self._run_engine_step(engine)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_suspend_without_recompute_flag(self):
        """Test suspend/resume where requests are NOT marked for recompute."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
            num_requests=2,
            num_tokens_to_generate=4,
        )
        engine = self._build_engine(config)

        # Add requests
        random.seed(config.random_seed)
        for request_id in range(config.num_requests):
            prompt_length = random.randint(config.min_prompt_length, config.max_prompt_length)
            prompt_tokens = torch.randint(
                0, config.vocab_size - 1, (prompt_length,), dtype=torch.int64, device='cuda'
            )
            request = DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=config.num_tokens_to_generate,
                    termination_id=config.vocab_size - 1,
                ),
            )
            engine._add_request(request)

        # Run a step
        self._run_engine_step(engine)

        # Suspend WITHOUT marking for recompute
        engine.suspend()

        # Resume (requests should not be re-added to context)
        engine.resume()

        # Context should have 0 requests since none were marked for recompute
        assert engine.context.total_request_count == 0


class TestMemoryBehavior:
    """Tests specifically focused on GPU memory behavior during suspend/resume."""

    def setup_method(self, method):
        """Setup model parallel for each test."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        set_rounder(4)

    def teardown_method(self, method):
        """Teardown model parallel after each test."""
        set_rounder(64)
        Utils.destroy_model_parallel()
        force_gc()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_memory_tracking_remove_kv_cache(self):
        """Test detailed memory tracking with remove_kv_cache mode."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
            context_buffer_size_gb=0.1,  # Larger buffer for more visible memory change
        )

        context = DynamicInferenceContext(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=None,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=0,
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
        )

        # Track memory at each stage
        memory_stages = {}

        force_gc()
        memory_stages['initial'] = get_gpu_memory_allocated()

        # Add requests
        random.seed(config.random_seed)
        for request_id in range(config.num_requests):
            prompt_length = random.randint(config.min_prompt_length, config.max_prompt_length)
            prompt_tokens = torch.randint(
                0, config.vocab_size - 1, (prompt_length,), dtype=torch.int64, device='cuda'
            )
            request = DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                sampling_params=SamplingParams(num_tokens_to_generate=config.num_tokens_to_generate),
            )
            context.add_request(request)

        force_gc()
        memory_stages['with_requests'] = get_gpu_memory_allocated()

        # Deallocate
        context.deallocate_all_tensors()
        force_gc()
        memory_stages['deallocated'] = get_gpu_memory_allocated()

        # Allocate
        context.allocate_all_tensors(is_init=False)
        force_gc()
        memory_stages['reallocated'] = get_gpu_memory_allocated()

        # Verify expected memory behavior
        assert memory_stages['deallocated'] < memory_stages['with_requests'], (
            f"Memory should decrease after deallocation. "
            f"With requests: {memory_stages['with_requests'] / 1e6:.2f} MB, "
            f"Deallocated: {memory_stages['deallocated'] / 1e6:.2f} MB"
        )

        assert memory_stages['reallocated'] > memory_stages['deallocated'], (
            f"Memory should increase after reallocation. "
            f"Deallocated: {memory_stages['deallocated'] / 1e6:.2f} MB, "
            f"Reallocated: {memory_stages['reallocated'] / 1e6:.2f} MB"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_memory_consistency_across_cycles(self):
        """Test that memory usage is consistent across multiple suspend/resume cycles."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
            context_buffer_size_gb=0.1,
        )

        context = DynamicInferenceContext(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=None,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=0,
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
        )

        # Record memory at each cycle
        cycle_memories = []

        for cycle in range(5):
            # Add requests
            random.seed(config.random_seed)  # Same seed for consistent requests
            for request_id in range(config.num_requests):
                prompt_length = random.randint(config.min_prompt_length, config.max_prompt_length)
                prompt_tokens = torch.randint(
                    0, config.vocab_size - 1, (prompt_length,), dtype=torch.int64, device='cuda'
                )
                request = DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    sampling_params=SamplingParams(num_tokens_to_generate=config.num_tokens_to_generate),
                )
                context.add_request(request)

            force_gc()
            memory_with_requests = get_gpu_memory_allocated()

            # Deallocate and reallocate
            context.deallocate_all_tensors()
            context.allocate_all_tensors(is_init=False)
            context.reset()

            force_gc()
            memory_after_cycle = get_gpu_memory_allocated()

            cycle_memories.append({
                'with_requests': memory_with_requests,
                'after_cycle': memory_after_cycle,
            })

        # Check that memory is roughly consistent across cycles (allowing 10% variance)
        for i in range(1, len(cycle_memories)):
            prev = cycle_memories[i - 1]['after_cycle']
            curr = cycle_memories[i]['after_cycle']
            variance = abs(curr - prev) / max(prev, 1)
            assert variance < 0.1, (
                f"Memory should be consistent across cycles. "
                f"Cycle {i-1}: {prev / 1e6:.2f} MB, Cycle {i}: {curr / 1e6:.2f} MB, "
                f"Variance: {variance * 100:.1f}%"
            )


class TestTorchMemorySaverIntegration:
    """Tests for torch_memory_saver integration if available."""

    def setup_method(self, method):
        """Setup model parallel for each test."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        set_rounder(4)

    def teardown_method(self, method):
        """Teardown model parallel after each test."""
        set_rounder(64)
        Utils.destroy_model_parallel()
        force_gc()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_torch_memory_saver_availability(self):
        """Test that torch_memory_saver availability is properly detected."""
        try:
            from torch_memory_saver import torch_memory_saver
            has_tms = True
        except ImportError:
            has_tms = False

        from megatron.core.inference.contexts.dynamic_context import HAVE_TORCH_MEMORY_SAVER
        assert HAVE_TORCH_MEMORY_SAVER == has_tms

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_persist_cuda_graphs_requires_tms_or_uvm(self):
        """Test that persist_cuda_graphs without UVM requires torch_memory_saver."""
        try:
            from torch_memory_saver import torch_memory_saver
            # If torch_memory_saver is available, this should work
            config = SuspendTestConfig(
                persist_cuda_graphs=True,
                offload_kv_cache=False,
                remove_kv_cache=False,
                unified_memory_level=0,
            )
            context = DynamicInferenceContext(
                params_dtype=torch.bfloat16,
                num_layers=config.num_layers,
                kv_channels=config.kv_channels,
                num_attention_heads=config.num_attention_heads,
                max_sequence_length=config.max_sequence_length,
                num_cuda_graphs=None,
                use_cuda_graphs_for_non_decode_steps=True,
                buffer_size_gb=config.context_buffer_size_gb,
                paused_buffer_size_gb=0.01,
                block_size_tokens=config.context_block_size_tokens,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                unified_memory_level=0,
                persist_cuda_graphs=True,
                offload_kv_cache=False,
                remove_kv_cache=False,
            )
            # If we get here, the context was created successfully
            assert context.persist_cuda_graphs
        except ImportError:
            # If torch_memory_saver is not available, persist_cuda_graphs without UVM should fail
            with pytest.raises(AssertionError, match="torch_memory_saver"):
                config = SuspendTestConfig(
                    persist_cuda_graphs=True,
                    offload_kv_cache=False,
                    remove_kv_cache=False,
                    unified_memory_level=0,
                )
                DynamicInferenceContext(
                    params_dtype=torch.bfloat16,
                    num_layers=config.num_layers,
                    kv_channels=config.kv_channels,
                    num_attention_heads=config.num_attention_heads,
                    max_sequence_length=config.max_sequence_length,
                    num_cuda_graphs=None,
                    use_cuda_graphs_for_non_decode_steps=True,
                    buffer_size_gb=config.context_buffer_size_gb,
                    paused_buffer_size_gb=0.01,
                    block_size_tokens=config.context_block_size_tokens,
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    unified_memory_level=0,
                    persist_cuda_graphs=True,
                    offload_kv_cache=False,
                    remove_kv_cache=False,
                )


class TestEdgeCases:
    """Tests for edge cases in suspend/resume functionality."""

    def setup_method(self, method):
        """Setup model parallel for each test."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        set_rounder(4)

        # Reset global cuda graph state
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None

    def teardown_method(self, method):
        """Teardown model parallel after each test."""
        set_rounder(64)
        Utils.destroy_model_parallel()
        force_gc()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_context_empty_offload_list(self):
        """Test offload with no tracked tensors (edge case)."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=False,
            unified_memory_level=0,
        )

        context = DynamicInferenceContext(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=None,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=0,
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=False,
        )

        # Manually call offload methods with empty lists - should not crash
        context._offloadable_tensor_names = []
        context._offload_tracked_tensors_to_cpu()
        context._restore_tracked_tensors_from_cpu()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_context_no_requests_suspend(self):
        """Test suspend/resume with no requests added."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )

        context = DynamicInferenceContext(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=None,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=0,
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
        )

        # No requests added - deallocate should still work
        context.deallocate_all_tensors()
        assert not context.is_tensor_state_allocated

        # Allocate should work
        context.allocate_all_tensors(is_init=False)
        assert context.is_tensor_state_allocated

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_context_reset_after_deallocate(self):
        """Test that reset works correctly after deallocate."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
            unified_memory_level=0,
        )

        context = DynamicInferenceContext(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=None,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=0,
            persist_cuda_graphs=False,
            offload_kv_cache=False,
            remove_kv_cache=True,
        )

        # Add a request
        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=4),
        )
        context.add_request(request)
        assert context.total_request_count == 1

        # Deallocate
        context.deallocate_all_tensors()

        # Allocate
        context.allocate_all_tensors(is_init=False)

        # Reset should work
        context.reset()
        assert context.total_request_count == 0


class TestTrackOffloadableTensors:
    """Tests for the _track_offloadable_tensors context manager."""

    def setup_method(self, method):
        """Setup model parallel for each test."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        set_rounder(4)

    def teardown_method(self, method):
        """Teardown model parallel after each test."""
        set_rounder(64)
        Utils.destroy_model_parallel()
        force_gc()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @patch('megatron.core.inference.contexts.dynamic_context.HAVE_TORCH_MEMORY_SAVER', False)
    def test_track_new_tensors(self):
        """Test that _track_offloadable_tensors correctly tracks new tensors."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=True,
            remove_kv_cache=False,
            unified_memory_level=0,
        )

        context = DynamicInferenceContext(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=None,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=0,
            persist_cuda_graphs=False,
            offload_kv_cache=True,
            remove_kv_cache=False,
        )

        # Verify that tracked tensors were recorded
        assert len(context._offloadable_tensor_names) > 0, (
            "Should have tracked tensors when using offload_kv_cache"
        )

        # Verify memory_buffer is among tracked tensors (it's the main KV cache)
        assert 'memory_buffer' in context._offloadable_tensor_names, (
            "memory_buffer should be tracked for offload"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @patch('megatron.core.inference.contexts.dynamic_context.HAVE_TORCH_MEMORY_SAVER', False)
    def test_offload_restore_data_integrity(self):
        """Test that data is preserved through offload/restore cycle."""
        config = SuspendTestConfig(
            persist_cuda_graphs=False,
            offload_kv_cache=True,
            remove_kv_cache=False,
            unified_memory_level=0,
        )

        context = DynamicInferenceContext(
            params_dtype=torch.bfloat16,
            num_layers=config.num_layers,
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_sequence_length,
            num_cuda_graphs=None,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=config.context_buffer_size_gb,
            paused_buffer_size_gb=0.01,
            block_size_tokens=config.context_block_size_tokens,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            unified_memory_level=0,
            persist_cuda_graphs=False,
            offload_kv_cache=True,
            remove_kv_cache=False,
        )

        # Write some data to the memory buffer
        if hasattr(context, 'memory_buffer') and context.memory_buffer is not None:
            # Fill with a known pattern
            context.memory_buffer.fill_(42.0)
            original_sum = context.memory_buffer.sum().item()

            # Offload
            context._offload_tracked_tensors_to_cpu()

            # Verify storage was resized
            assert context.memory_buffer.storage().size() == 0

            # Restore
            context._restore_tracked_tensors_from_cpu()

            # Verify data is preserved
            restored_sum = context.memory_buffer.sum().item()
            assert abs(original_sum - restored_sum) < 1e-3, (
                f"Data should be preserved. Original sum: {original_sum}, "
                f"Restored sum: {restored_sum}"
            )

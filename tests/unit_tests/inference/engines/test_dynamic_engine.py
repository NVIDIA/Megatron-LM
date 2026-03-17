# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import gc
import math
import random
import types
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple
from unittest import mock

import pytest
import torch
from tqdm import tqdm
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.inference.config import (
    InferenceConfig,
    KVCacheManagementMode,
    MambaInferenceStateConfig,
)
from megatron.core.inference.contexts.dynamic_context import (
    ActiveRequestCountOverflowError,
    BlockOverflowError,
    DynamicInferenceContext,
    RequestOverflowError,
    TokenOverflowError,
)
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.test_utilities import Utils

try:
    from torch_memory_saver import torch_memory_saver  # noqa: F401

    HAVE_TORCH_MEMORY_SAVER = True
except ImportError:
    HAVE_TORCH_MEMORY_SAVER = False


def skip_if_mamba_sequence_packing_not_available(model_provider: str):
    if model_provider == "mamba":
        sequence_packing_available, reason_for_no_sequence_packing = (
            _check_mamba_sequence_packing_support()
        )
        if not sequence_packing_available:
            pytest.skip(reason_for_no_sequence_packing)


def set_rounder(value):
    """Utility function to set the DynamicInferenceContext rounder."""
    DynamicInferenceContext.ROUNDER = value  # For backwards compatibility
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


def mock_forward(input_ids, position_ids, attention_mask, *args, **kwargs):
    """Mock forward function to avoid numerics issues with random inputs."""
    return torch.randn(
        input_ids.size(0),
        input_ids.size(1),
        kwargs["vocab_size"],
        device=input_ids.device,
        dtype=torch.bfloat16,
    )


@dataclass
class DynamicEngineTestConfig:
    """Test configuration args."""

    random_seed = 123
    vocab_size = 100

    set_rounder(4)
    num_requests: int = 2 * DynamicInferenceContext.round_up_requests(1, 1)
    min_prompt_length: int = 4
    max_prompt_length: int = 16
    num_tokens_to_generate: Optional[int] = 4
    num_tokens_total: Optional[int] = None
    max_sequence_length: Optional[int] = None

    num_gap_steps: int = 2

    context_buffer_size_gb: float = 0.1  # enough room for all tokens.
    context_paused_buffer_size_gb: float | None = None
    context_block_size_tokens: int = 256
    context_max_requests: Optional[int] = None
    context_max_tokens: Optional[int] = None
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    sequence_parallel: bool = False

    use_fixed_output_lengths: bool = False
    num_cuda_graphs: int = None
    use_cuda_graphs_for_non_decode_steps: bool = True
    fp8: bool = False
    model_provider: str = "gpt"
    return_log_probs: bool = False
    materialize_only_last_token_logits: bool = True
    skip_prompt_log_probs: bool = False
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    cuda_graph_scope: List[CudaGraphScope] = field(
        default_factory=lambda: [CudaGraphScope.full_iteration_inference]
    )
    force_build_cuda_graphs: bool = False
    transformer_impl: str = "local"
    # If False, do not build cuda graphs in the tests, even if
    # num_cuda_graphs is set.
    # For tests concerning cuda-graph warmups, we set this to False
    # to avoid the overhead of building the graphs, which is not
    # relevant to the test. The tests only check if the required
    # context attributes are set correctly.
    suspend_resume_interval: Optional[int] = None
    kv_cache_management_mode: str = "persist"
    static_kv_memory_pointers: bool = True
    track_generated_token_events: bool = False
    num_speculative_tokens: int = 0

    def __post_init__(self):

        # Compute max_sequence_length.
        if self.max_sequence_length is None:
            assert self.num_tokens_to_generate is None or self.num_tokens_total is None
            if self.num_tokens_to_generate is not None:
                self.max_sequence_length = (
                    self.max_prompt_length
                    + self.num_tokens_to_generate
                    + self.num_speculative_tokens
                )
            else:
                assert self.num_tokens_total is not None
                self.max_sequence_length = self.num_tokens_total + self.num_speculative_tokens

        # Default paused buffer size.
        if self.context_paused_buffer_size_gb is None:
            self.context_paused_buffer_size_gb = 0.2 * self.context_buffer_size_gb


@dataclass
class DynamicEngineTestEnv:
    """Test environment, including requests and engine."""

    config: DynamicEngineTestConfig
    requests: List[DynamicInferenceRequest]
    engine: DynamicInferenceEngine
    mem_usage: dict = field(
        default_factory=lambda: {"start": None, "end": None, "suspend_resume": {}}
    )


class TestDynamicInferenceEngine:

    @classmethod
    def _build_requests(cls, test_config: DynamicEngineTestConfig) -> List[DynamicInferenceRequest]:

        requests = []
        for request_id in range(test_config.num_requests):

            # Prompt length.
            if test_config.min_prompt_length == test_config.max_prompt_length:
                prompt_length = test_config.min_prompt_length
            else:
                prompt_length = random.randint(
                    test_config.min_prompt_length, test_config.max_prompt_length
                )

            # Num tokens to generate.
            num_tokens_to_generate = test_config.num_tokens_to_generate
            num_tokens_total = test_config.num_tokens_total

            if test_config.use_fixed_output_lengths:
                if num_tokens_to_generate is not None:
                    num_tokens_to_generate = random.randint(
                        1, test_config.max_sequence_length - prompt_length
                    )
                else:
                    num_tokens_total = random.randint(
                        prompt_length + 1, test_config.max_sequence_length
                    )

            # Sampling params.
            sampling_params = SamplingParams(
                num_tokens_to_generate=num_tokens_to_generate,
                termination_id=(
                    -1 if test_config.use_fixed_output_lengths else test_config.vocab_size - 1
                ),
                return_log_probs=test_config.return_log_probs,
                skip_prompt_log_probs=test_config.skip_prompt_log_probs,
            )
            if not hasattr(sampling_params, "num_tokens_total"):
                # Remove this if statement branch in megatron-core 0.16
                sampling_params.add_attributes({"num_tokens_total": num_tokens_total})
            else:
                sampling_params.num_tokens_total = num_tokens_total

            # Request.
            prompt_tokens = torch.randint(
                0,
                test_config.vocab_size - 1,
                (prompt_length,),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            request = DynamicInferenceRequest(
                request_id=request_id, prompt_tokens=prompt_tokens, sampling_params=sampling_params
            )
            requests.append(request)

        return requests

    @classmethod
    def _build_inference_context(
        cls,
        test_config: DynamicEngineTestConfig,
        transformer_config: TransformerConfig,
        requests: List[DynamicInferenceRequest],
        mamba_inference_state_config: Optional[MambaInferenceStateConfig] = None,
    ):
        """The inference context manages the KV cache and other inference state."""

        # Inference context.
        context = DynamicInferenceContext(
            model_config=transformer_config,
            inference_config=InferenceConfig(
                max_sequence_length=test_config.max_sequence_length,
                num_cuda_graphs=test_config.num_cuda_graphs,
                use_cuda_graphs_for_non_decode_steps=True,
                buffer_size_gb=test_config.context_buffer_size_gb,
                paused_buffer_size_gb=test_config.context_paused_buffer_size_gb,
                block_size_tokens=test_config.context_block_size_tokens,
                max_requests=test_config.context_max_requests,
                max_tokens=test_config.context_max_tokens,
                mamba_inference_state_config=mamba_inference_state_config,
                materialize_only_last_token_logits=test_config.materialize_only_last_token_logits,
                kv_cache_management_mode=KVCacheManagementMode(
                    test_config.kv_cache_management_mode
                ),
                static_kv_memory_pointers=test_config.static_kv_memory_pointers,
                enable_chunked_prefill=test_config.enable_chunked_prefill,
                enable_prefix_caching=test_config.enable_prefix_caching,
                use_flashinfer_fused_rope=None,  # default to using flash-infer if available
                # this is for compatibility with the LTS environment
                unified_memory_level=0,  # unit tests currently broken with UVM
                track_generated_token_events=test_config.track_generated_token_events,
                num_speculative_tokens=test_config.num_speculative_tokens,
            ),
        )

        return context

    @classmethod
    @torch.inference_mode()
    def _build_test_env(cls, test_config):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=test_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=test_config.pipeline_model_parallel_size,
        )

        set_rounder(4)

        # Random state.
        random.seed(test_config.random_seed)
        torch.manual_seed(test_config.random_seed)
        model_parallel_cuda_manual_seed(
            seed=test_config.random_seed,
            inference_rng_tracker=True,
            use_cudagraphable_rng=False,
            force_reset_rng=True,
        )

        # Requests.
        requests = cls._build_requests(test_config)

        if test_config.model_provider == "gpt":
            # Transformer config.
            transformer_config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=4,
                mtp_num_layers=test_config.num_speculative_tokens,
                hidden_size=128 if test_config.fp8 else 32,
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl=(
                    "local"
                    if test_config.num_cuda_graphs is not None
                    and test_config.force_build_cuda_graphs
                    else "none"
                ),
                inference_rng_tracker=True,
                tensor_model_parallel_size=test_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=test_config.pipeline_model_parallel_size,
                expert_model_parallel_size=test_config.expert_model_parallel_size,
                num_moe_experts=(
                    None
                    if test_config.expert_model_parallel_size == 1
                    else test_config.expert_model_parallel_size
                ),
                sequence_parallel=test_config.sequence_parallel,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=test_config.expert_model_parallel_size == 1
                and not (test_config.transformer_impl == "inference_optimized"),
                fp8="hybrid" if test_config.fp8 else None,
                fp8_recipe="tensorwise" if test_config.fp8 else None,
                inference_sampling_seed=test_config.random_seed,
                cuda_graph_scope=test_config.cuda_graph_scope,
                transformer_impl=test_config.transformer_impl,
                normalization=(
                    "RMSNorm"
                    if test_config.transformer_impl == "inference_optimized"
                    else "LayerNorm"
                ),
                # inference optimized currently only supports RMS Norm
            )
            if test_config.fp8 or test_config.transformer_impl == "transformer_engine":
                layer_spec = get_gpt_layer_with_transformer_engine_spec()
            elif test_config.transformer_impl == "local":
                layer_spec = get_gpt_layer_local_spec()
            elif test_config.transformer_impl == "inference_optimized":
                layer_spec = get_gpt_layer_with_inference_spec()

            # MTP block spec (needed for speculative decoding).
            mtp_block_spec = None
            if test_config.num_speculative_tokens > 0:
                use_te = test_config.fp8 or test_config.transformer_impl == "transformer_engine"
                mtp_block_spec = get_gpt_mtp_block_spec(
                    config=transformer_config, spec=layer_spec, use_transformer_engine=use_te
                )

            # GPT model.
            model = GPTModel(
                config=transformer_config,
                transformer_layer_spec=layer_spec,
                vocab_size=test_config.vocab_size,
                max_sequence_length=test_config.max_sequence_length,
                parallel_output=True,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
                mtp_block_spec=mtp_block_spec,
            ).cuda()
        elif test_config.model_provider == "mamba":
            pp_size = test_config.pipeline_model_parallel_size
            # Transformer config.
            transformer_config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=(
                    3 if pp_size == 1 else 6
                ),  # 1 Mamba layer, 1 attention layer, 1 MLP layer
                mtp_num_layers=test_config.num_speculative_tokens,
                hidden_size=256,  # The Mamba layer places several constraints on this
                mamba_num_heads=16,
                num_attention_heads=16,
                use_cpu_initialization=True,
                cuda_graph_impl=(
                    "local"
                    if test_config.num_cuda_graphs is not None
                    and test_config.force_build_cuda_graphs
                    else "none"
                ),
                inference_rng_tracker=True,
                tensor_model_parallel_size=test_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=pp_size,
                expert_model_parallel_size=test_config.expert_model_parallel_size,
                num_moe_experts=(
                    None
                    if test_config.expert_model_parallel_size == 1
                    else test_config.expert_model_parallel_size
                ),
                sequence_parallel=test_config.sequence_parallel,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=test_config.expert_model_parallel_size == 1,
                fp8="hybrid" if test_config.fp8 else None,
                fp8_recipe="tensorwise" if test_config.fp8 else None,
                inference_sampling_seed=test_config.random_seed,
                cuda_graph_scope=test_config.cuda_graph_scope,
                transformer_impl=test_config.transformer_impl,
                is_hybrid_model=True,  # Needs to be set for correct out_proj init
            )

            # Mamba model.
            model = MambaModel(
                config=transformer_config,
                mamba_stack_spec=mamba_stack_spec,
                vocab_size=test_config.vocab_size,
                max_sequence_length=test_config.max_sequence_length,
                parallel_output=True,
                hybrid_layer_pattern=(
                    "M*-" if pp_size == 1 else "M*-|M*-"
                ),  # 3 or 6 layers (2 PP stages)
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
        else:
            raise ValueError(f"Invalid model provider {test_config.model_provider}")

        for param in model.parameters():
            param.data = param.data.to(transformer_config.params_dtype)

        model.eval()

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(model)

        # Inference context.
        inference_context = cls._build_inference_context(
            test_config=test_config,
            transformer_config=transformer_config,
            requests=requests,
            mamba_inference_state_config=mamba_inference_state_config,
        )

        # Inference model wrapper.
        inference_wrapped_model = GPTInferenceWrapper(model, inference_context)

        # Note: the following is taken from AbstractModelInferenceWrapper.prep_model_for_inference().
        inference_wrapped_model.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

        # Text generation controller.
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=types.SimpleNamespace(
                vocab_size=test_config.vocab_size, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )

        # Reset global cuda graph state.
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

        # Inference engine.
        engine = DynamicInferenceEngine(text_generation_controller, inference_context)

        # Test env.
        env = DynamicEngineTestEnv(config=test_config, requests=requests, engine=engine)

        return env

    @classmethod
    @torch.inference_mode()
    def _run_step(cls, env):
        set_rounder(4)
        # Step inference engine (i.e., generate one token per request).
        # It's safe to use request 0's sampling params here because
        # the only thing that differs between requests is num_tokens_to_generate,
        # and engine.async_step() doesn't use this sampling param's
        # num_tokens_to_generate.
        result = env.engine.step_modern()

        # Suspend + resume.
        if (
            env.config.suspend_resume_interval is not None
            and env.engine.context.step_count % env.config.suspend_resume_interval == 0
        ):
            suspend_resume_mems = {}
            suspend_resume_mems["start"] = torch.cuda.memory_stats()
            env.engine.suspend()  # suspend.
            suspend_resume_mems["mid"] = torch.cuda.memory_stats()
            env.engine.resume()  # resume.
            suspend_resume_mems["end"] = torch.cuda.memory_stats()
            env.mem_usage["suspend_resume"][env.engine.context.step_count] = suspend_resume_mems

        # Nothing done?
        finished_request_records = result["finished_request_records"]
        if len(finished_request_records) == 0:
            return

        # Append output tokens.
        for finished_request_record in finished_request_records:
            finished_request = finished_request_record.merge()
            request = env.requests[finished_request.request_id]
            request.output = finished_request.generated_tokens
            request.status = finished_request.status

    @classmethod
    @torch.inference_mode()
    def _run_test(cls, **test_config_kwargs):
        # Test environment.
        test_config = DynamicEngineTestConfig(**test_config_kwargs)
        env = cls._build_test_env(test_config)

        # Add requests to engine.
        env.mem_usage["start"] = torch.cuda.memory_stats()
        for request in tqdm(env.requests, "add requests"):

            # Add request.
            env.engine._add_request(request)
            request.state = "pending"

            # Insert gap steps between adding requests.
            for _ in range(test_config.num_gap_steps):
                cls._run_step(env)

        # Step engine until finished.
        while True:
            # Run at least one step to collect failed requests.
            cls._run_step(env)
            if not env.engine.has_unfinished_requests():
                break

        # Validate all requests finished.
        for request in env.requests:
            assert request.status in (
                Status.COMPLETED,
                Status.FAILED,
            ), f"request.status == '{request.status}'."

            num_tokens_to_generate = request.sampling_params.num_tokens_to_generate
            num_tokens_total = request.sampling_params.num_tokens_total
            num_tokens_expected = (
                num_tokens_to_generate
                if num_tokens_total is None
                else num_tokens_total - len(request.prompt_tokens)
            )

            # Validate the output length only if suspend_resume_interval is None.
            # If it is not None, then the output length could be anything in the
            # range [1, num_tokens_to_generate].
            if test_config.suspend_resume_interval is None:
                assert (
                    (num_tokens_to_generate is None and num_tokens_total is None)
                    or len(request.generated_tokens) <= num_tokens_expected
                    or request.status == Status.FAILED
                ), (
                    f"Request {request.request_id} expected to generate {num_tokens_to_generate} "
                    f"tokens but generated {len(request.generated_tokens)}"
                )
        env.mem_usage["end"] = torch.cuda.memory_stats()

        return env

    def teardown_method(self, method):
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "mamba"])
    @pytest.mark.parametrize("num_cuda_graphs", [None, 1, 4, -1])
    @pytest.mark.parametrize("cuda_graph_scope", [[], [CudaGraphScope.full_iteration_inference]])
    def test_simple(self, model_provider, num_cuda_graphs, cuda_graph_scope) -> None:
        """Simple test that runs without errors, and validates output."""
        skip_if_mamba_sequence_packing_not_available(model_provider)
        num_tokens_to_generate = 16

        # Run test.
        env = self._run_test(
            num_tokens_to_generate=num_tokens_to_generate,
            model_provider=model_provider,
            num_cuda_graphs=num_cuda_graphs,
            cuda_graph_scope=cuda_graph_scope,
            force_build_cuda_graphs=True,
            context_max_requests=128,
        )

        # Validate max_requests, max_tokens.
        assert env.engine.context.max_tokens == DynamicInferenceContext.DEFAULT_MAX_TOKENS

        if num_cuda_graphs is not None:
            assert env.engine.context.cuda_graph_token_counts is not None
            assert env.engine.context.cuda_graph_batch_dimensions_list
            model = env.engine.controller.inference_wrapped_model.model
            if cuda_graph_scope == [CudaGraphScope.full_iteration_inference]:
                # check if cudagraph runners are created at the decoder level
                assert model.decoder.cudagraph_manager.cudagraph_runners
            else:
                # check if cudagraph runners are created at the layer level
                for layer in model.decoder.layers:
                    assert layer.cudagraph_manager.cudagraph_runners

        # Validate generated tokens.
        gpt_expected_generated_tokens = [
            [69, 85, 55, 74, 56, 89, 64, 59, 55, 67, 15, 58, 6, 37, 54, 47],
            [29, 54, 33, 72, 45, 76, 41, 56, 28, 25, 17, 2, 61, 6, 98, 76],
            [35, 78, 54, 16, 79, 98, 22, 5, 60, 0, 1, 76, 77, 11, 25, 7],
            [25, 75, 57, 85, 81, 37, 88, 17, 71, 15, 70, 64, 50, 0, 64, 45],
            [32, 5, 85, 75, 30, 68, 23, 33, 20, 26, 89, 20, 49, 28, 38, 81],
            [33, 69, 32, 49, 93, 24, 33, 6, 54, 89, 92, 97, 42, 80, 50, 53],
            [82, 78, 78, 65, 26, 5, 69, 36, 37, 99],
            [51, 70, 22, 1, 87, 42, 36, 26, 27, 56, 82, 32, 8, 80, 20, 43],
        ]

        mamba_expected_generated_tokens = [
            [69, 85, 55, 74, 85, 89, 64, 59, 55, 67, 15, 58, 6, 37, 34, 47],
            [29, 16, 33, 30, 45, 76, 41, 46, 82, 17, 17, 2, 61, 6, 98, 76],
            [35, 78, 54, 16, 79, 98, 22, 5, 37, 30, 1, 76, 5, 11, 25, 86],
            [25, 75, 57, 85, 81, 59, 88, 38, 71, 15, 70, 64, 50, 0, 64, 45],
            [32, 5, 85, 75, 30, 68, 23, 33, 20, 26, 35, 20, 49, 28, 34, 81],
            [87, 69, 32, 49, 93, 24, 33, 6, 54, 89, 92, 97, 42, 80, 50, 53],
            [82, 78, 78, 19, 70, 5, 97, 36, 37, 99],
            [51, 70, 22, 1, 87, 42, 36, 26, 27, 56, 82, 32, 8, 20, 20, 43],
        ]

        if model_provider == "gpt":
            expected_generated_tokens_list = gpt_expected_generated_tokens
        elif model_provider == "mamba":
            expected_generated_tokens_list = mamba_expected_generated_tokens
        else:
            raise ValueError(f"Invalid model_provider {model_provider}")

        print(f"Validating {len(env.requests)} requests.")
        print(f"Expected generated tokens: {expected_generated_tokens_list}")
        print(f"Actual generated tokens: {[request.generated_tokens for request in env.requests]}")

        assert len(env.requests) == len(expected_generated_tokens_list)

        for request, expected_generated_tokens in zip(env.requests, expected_generated_tokens_list):
            assert request.generated_tokens == expected_generated_tokens, (
                f"request {request.request_id}, "
                f"result ({request.generated_tokens}) != "
                f"expected ({expected_generated_tokens})."
            )

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_token_overflow_transient(self) -> None:
        """Test token overflow."""
        test_config = DynamicEngineTestConfig(
            num_requests=2,
            min_prompt_length=512,
            max_prompt_length=512,
            num_tokens_to_generate=2,
            context_max_tokens=900,
        )
        env = self._build_test_env(test_config)
        env.engine._add_request(env.requests[0])
        env.engine._add_request(env.requests[1])
        env.engine.schedule_waiting_requests()
        assert list(env.engine.waiting_request_ids) == [1]

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.skip(
        reason="activate for `megatron-core >= 0.16`, after fixing "
        "`raise TokenOverflowError(is_transient=False)` compatibility with "
        "legacy tests."
    )
    def test_token_overflow_nontransient(self) -> None:
        """Test token overflow (non-transient)."""
        test_config = DynamicEngineTestConfig(context_max_tokens=8)
        env = self._build_test_env(test_config)
        try:
            env.engine._add_request(env.requests[0])
        except TokenOverflowError as e:
            assert e.is_transient == False
        else:
            raise Exception("should have raised TokenOverflowError(is_transient=False).")

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "mamba"])
    def test_block_overflow(self, model_provider: str) -> None:
        """Test block overflow."""
        skip_if_mamba_sequence_packing_not_available(model_provider)
        env = self._build_test_env(DynamicEngineTestConfig(model_provider=model_provider))
        context = env.engine.context
        block_size_bytes = context.block_size_bytes
        buffer_size_gb = (block_size_bytes + 1) / 1024**3
        test_config = DynamicEngineTestConfig(
            context_buffer_size_gb=buffer_size_gb, model_provider=model_provider
        )
        env = self._build_test_env(test_config)
        env.engine._add_request(env.requests[0])
        assert list(env.engine.waiting_request_ids) == [0]

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "mamba"])
    def test_multi_add(self, model_provider: str) -> None:
        """Test adding multiple requests simultaneously."""
        skip_if_mamba_sequence_packing_not_available(model_provider)
        self._run_test(num_gap_steps=0, model_provider=model_provider)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "mamba"])
    def test_fixed_output_lengths(self, model_provider: str) -> None:
        """Test generating a fixed number of output tokens."""
        skip_if_mamba_sequence_packing_not_available(model_provider)
        self._run_test(use_fixed_output_lengths=True, model_provider=model_provider)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_cuda_graph_token_counts(self) -> None:
        """Test initialization of `cuda_graph_token_counts` in dynamic context."""

        # Test num_cuda_graphs.
        for num_cuda_graphs, expected_cuda_graph_token_counts in [
            (0, [80]),
            (1, [80]),
            (2, [80, 40]),
            (4, [80, 72, 48, 24]),
            (8, [80, 64, 48, 32, 16]),
            (16, [80, 72, 64, 56, 48, 40, 32, 24, 16, 8]),
            (32, [80, 72, 64, 56, 48, 40, 32, 24, 16, 8]),
        ]:

            # Build cuda graphs (inside dynamic engine).
            env = self._build_test_env(
                DynamicEngineTestConfig(
                    context_buffer_size_gb=0.01, num_cuda_graphs=num_cuda_graphs
                )
            )
            actual_cuda_graph_token_counts = env.engine.context.cuda_graph_token_counts
            assert (
                actual_cuda_graph_token_counts == expected_cuda_graph_token_counts
            ), "num_cuda_graphs %d ... cuda_graph_token_counts: expected %s, found %s." % (
                num_cuda_graphs,
                expected_cuda_graph_token_counts,
                actual_cuda_graph_token_counts,
            )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "mamba"])
    @torch.inference_mode()
    def test_generate_function(self, model_provider: str) -> None:
        """Test the generate function that processes multiple prompts at once."""
        skip_if_mamba_sequence_packing_not_available(model_provider)

        # Set up test environment
        test_config = DynamicEngineTestConfig(
            num_requests=4,
            max_prompt_length=8,
            num_tokens_to_generate=4,
            model_provider=model_provider,
        )
        env = self._build_test_env(test_config)

        # Create string prompts (just mock strings, since the test environment mocks the tokenizer)
        prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]

        # Mock the tokenize_prompt method to return predictable token sequences
        def mock_tokenize_prompt(prompt, add_BOS=False):
            # Return a token sequence based on the prompt number
            prompt_num = int(prompt[-1])
            return [10 + i for i in range(prompt_num + 2)]

        env.engine.controller.tokenize_prompt = mock_tokenize_prompt

        # Call the generate function.
        # It's safe to use request 0's sampling params here because all sampling
        # params are identical as long as use_fixed_output_lengths == False.
        finished_request_records = env.engine.generate(prompts, env.requests[0].sampling_params)
        finished_requests = [r.merge() for r in finished_request_records]

        # Verify results
        assert len(finished_requests) == len(
            prompts
        ), "Should return same number of finished requests as prompts"

        request_ids = [r.request_id for r in finished_requests]
        assert request_ids == sorted(
            request_ids
        ), f"Request ids are not in sorted order: {request_ids}"

        # Check each request was processed
        for i, request in enumerate(finished_requests):
            # Verify each request has generated tokens
            assert len(request.generated_tokens) > 0, f"Request {i} should have generated tokens"
            assert request.status == Status.COMPLETED, f"Request {i} should be completed"

    @pytest.mark.internal
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    async def test_run_engine(self):
        """
        Test asynchronously adding and waiting for requests while the engine is
        running continuously.
        """
        # Have to wrap inference mode in-line because async functions are not supported
        with torch.inference_mode():
            # Test environment.
            test_config = DynamicEngineTestConfig(num_requests=8, use_fixed_output_lengths=True)
            env = self._build_test_env(test_config)

            engine_task = asyncio.create_task(env.engine.run_engine())

            request_completion_futures: Dict[int, asyncio.Future[DynamicInferenceRequest]] = {}

            # Add requests to engine.
            for request in tqdm(env.requests, "add requests"):
                request_completion_futures[request.request_id] = env.engine._add_request(request)

            # Wait for all requests to complete.
            await asyncio.gather(*request_completion_futures.values())

            # Verify that all request outputs were set.
            for request_id, fut in request_completion_futures.items():
                num_tokens_to_generate = env.requests[
                    request_id
                ].sampling_params.num_tokens_to_generate
                request_record = fut.result()
                request = request_record.merge()
                assert request.generated_length == num_tokens_to_generate, (
                    f"Request {request_id} expected to generate {num_tokens_to_generate} "
                    f"tokens but generated {request.generated_length}"
                )

            engine_task.cancel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("model_provider", ["gpt", "mamba"])
    def test_fp8_inference(self, model_provider: str):
        skip_if_mamba_sequence_packing_not_available(model_provider)

        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)

        self._run_test(model_provider=model_provider, fp8=True)

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_return_log_probs(self):
        """Verify that log probs are returned and computed correctly."""
        # Returning log probs requires materializing the full prompt logits or
        # explicitly disabling prompt logits.
        with pytest.raises(AssertionError):
            env = self._run_test(return_log_probs=True, materialize_only_last_token_logits=True)

        # Test with full logits materialization
        env = self._run_test(
            return_log_probs=True,
            materialize_only_last_token_logits=False,
            num_tokens_to_generate=5,
        )

        # Validate log probs for each completed request
        for request in env.requests:
            if request.status != Status.COMPLETED:
                continue

            # Validate prompt log probs
            if request.prompt_log_probs is not None and len(request.prompt_log_probs) > 0:
                prompt_len = len(request.prompt_tokens)
                # Should have log probs for all tokens except the first one
                assert len(request.prompt_log_probs) == prompt_len - 1, (
                    f"Request {request.request_id}: Expected {prompt_len - 1} prompt log probs, "
                    f"got {len(request.prompt_log_probs)}"
                )

                # Validate each prompt log prob
                for i, log_prob in enumerate(request.prompt_log_probs):
                    assert not math.isnan(
                        log_prob
                    ), f"Request {request.request_id}, prompt token {i}: log_prob is NaN"
                    assert not math.isinf(
                        log_prob
                    ), f"Request {request.request_id}, prompt token {i}: log_prob is inf"
                    assert log_prob <= 0.0, (
                        f"Request {request.request_id}, prompt token {i}: "
                        f"log_prob {log_prob} should be <= 0"
                    )
                    assert log_prob >= -50.0, (
                        f"Request {request.request_id}, prompt token {i}: "
                        f"log_prob {log_prob} is unreasonably small"
                    )

            # Validate generated log probs
            assert (
                request.generated_log_probs is not None
            ), f"Request {request.request_id}: generated_log_probs should not be None"
            assert len(request.generated_log_probs) == len(request.generated_tokens), (
                f"Request {request.request_id}: Expected {len(request.generated_tokens)} "
                f"generated log probs, got {len(request.generated_log_probs)}"
            )

            # Validate each generated log prob
            for i, log_prob in enumerate(request.generated_log_probs):
                assert not math.isnan(
                    log_prob
                ), f"Request {request.request_id}, generated token {i}: log_prob is NaN"
                assert not math.isinf(
                    log_prob
                ), f"Request {request.request_id}, generated token {i}: log_prob is inf"
                assert log_prob <= 0.0, (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} should be <= 0"
                )
                assert log_prob >= -50.0, (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is unreasonably small"
                )

            # Validate that all generated tokens are valid
            for i, token_id in enumerate(request.generated_tokens):
                assert 0 <= token_id < env.config.vocab_size, (
                    f"Request {request.request_id}, token {i}: token_id {token_id} "
                    f"is out of valid range [0, {env.config.vocab_size})"
                )

        # Test with skipping prompt log probs
        env = self._run_test(
            return_log_probs=True,
            materialize_only_last_token_logits=True,
            skip_prompt_log_probs=True,
            num_tokens_to_generate=5,
        )

        # Validate that prompt log probs are empty/None when skipped
        for request in env.requests:
            if request.status != Status.COMPLETED:
                continue

            # When skip_prompt_log_probs is True, prompt_log_probs should be empty
            assert request.prompt_log_probs is None or len(request.prompt_log_probs) == 0, (
                f"Request {request.request_id}: prompt_log_probs should be empty when "
                f"skip_prompt_log_probs=True, but got {len(request.prompt_log_probs)} items"
            )

            # Generated log probs should still be present
            assert (
                request.generated_log_probs is not None and len(request.generated_log_probs) > 0
            ), f"Request {request.request_id}: generated_log_probs should be present"

            # Validate generated log probs are still valid
            for i, log_prob in enumerate(request.generated_log_probs):
                assert not math.isnan(log_prob) and not math.isinf(log_prob), (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is invalid"
                )
                assert -50.0 <= log_prob <= 0.0, (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is out of expected range [-50.0, 0.0]"
                )

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_log_probs_token_correspondence(self):
        """
        Verify that log probabilities correspond to the actual sampled tokens.
        This test checks that the log probability reported for each token actually
        corresponds to that token's probability in the distribution.
        """
        # Run test with log probs enabled
        env = self._run_test(
            return_log_probs=True,
            materialize_only_last_token_logits=False,
            num_tokens_to_generate=5,
            num_requests=4,
        )

        # For each completed request
        for request in env.requests:
            if request.status != Status.COMPLETED:
                continue

            # Check that we have log probs for generated tokens
            assert request.generated_log_probs is not None
            assert len(request.generated_log_probs) == len(request.generated_tokens)

            # Verify log probs are valid and in reasonable range
            for i, (token_id, log_prob) in enumerate(
                zip(request.generated_tokens, request.generated_log_probs)
            ):
                # Basic validity checks
                assert not math.isnan(
                    log_prob
                ), f"Request {request.request_id}, token {i}: log_prob is NaN"
                assert not math.isinf(
                    log_prob
                ), f"Request {request.request_id}, token {i}: log_prob is inf"

                # Log probabilities should be <= 0 (since prob <= 1)
                assert log_prob <= 0.0, (
                    f"Request {request.request_id}, token {i}: "
                    f"log_prob {log_prob} should be <= 0"
                )

                # Check reasonable range (not too negative)
                # Using a more lenient threshold since actual model outputs can vary
                assert log_prob >= -100.0, (
                    f"Request {request.request_id}, token {i}: "
                    f"log_prob {log_prob} is unreasonably small"
                )

                # Token ID should be valid
                assert 0 <= token_id < env.config.vocab_size, (
                    f"Request {request.request_id}, token {i}: "
                    f"token_id {token_id} is out of range [0, {env.config.vocab_size})"
                )

            # Check prompt log probs if available
            if request.prompt_log_probs is not None and len(request.prompt_log_probs) > 0:
                expected_prompt_log_probs = len(request.prompt_tokens) - 1
                assert len(request.prompt_log_probs) == expected_prompt_log_probs, (
                    f"Request {request.request_id}: Expected {expected_prompt_log_probs} "
                    f"prompt log probs, got {len(request.prompt_log_probs)}"
                )

                for i, log_prob in enumerate(request.prompt_log_probs):
                    assert not math.isnan(log_prob) and not math.isinf(log_prob)
                    assert -100.0 <= log_prob <= 0.0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("materialize_only_last_token_logits", [False, True])
    @pytest.mark.parametrize("sequence_parallel", [False, True])
    @pytest.mark.parametrize("ep_size", [1, 2])
    @pytest.mark.parametrize("pp_size", [1, 2])
    @pytest.mark.parametrize("tp_size", [1, 2])
    @pytest.mark.parametrize("model_provider", ["gpt", "mamba"])
    @pytest.mark.parametrize("transformer_impl", ["local", "inference_optimized"])
    @torch.inference_mode()
    def test_parallel_inference(
        self,
        model_provider,
        tp_size,
        pp_size,
        ep_size,
        sequence_parallel,
        materialize_only_last_token_logits,
        transformer_impl,
    ):
        skip_if_mamba_sequence_packing_not_available(model_provider)

        if tp_size == 1 and pp_size == 1 and ep_size == 1:
            pytest.skip(reason="Test requires tp_size > 1 or pp_size > 1 or ep_size > 1")
        elif not torch.distributed.is_initialized():
            pytest.skip("Distributed not initialized")
        world_size = torch.distributed.get_world_size()
        min_world_size = tp_size * pp_size * ep_size
        if world_size < min_world_size:
            pytest.skip(f"Test requires at least {min_world_size} GPUs")
        elif tp_size == 1 and sequence_parallel:
            pytest.skip(reason="Sequence parallelism requires tp_size > 1")
        elif tp_size > 1 and ep_size > 1 and not sequence_parallel:
            pytest.skip(reason="Sequence parallelism must be used with tp_size > 1 and ep_size > 1")
        elif transformer_impl == "inference_optimized":
            if ep_size > 1:
                pytest.skip(
                    reason="MoE models are not supported with the inference optimized transformer."
                )
            if tp_size > 1 and not sequence_parallel:
                pytest.skip(
                    reason=(
                        "The inference optimized transformer requires sequence parallelism "
                        "when tp_size > 1."
                    )
                )
            if model_provider == "mamba":
                pytest.skip(
                    reason="Mamba model is not supported with the inference optimized transformer."
                )

        env = self._run_test(
            model_provider=model_provider,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=sequence_parallel,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            transformer_impl=transformer_impl,
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("materialize_only_last_token_logits", [False, True])
    def test_sequence_parallel_fp8_inference(self, materialize_only_last_token_logits: bool):
        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)

        self._run_test(
            min_prompt_length=19,
            max_prompt_length=19,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            materialize_only_last_token_logits=True,
            fp8=True,
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_num_tokens_total(self):
        """Simple test, but using num_tokens_total instead of num_tokens_to_generate."""
        # Run test.
        env = self._run_test(
            num_tokens_to_generate=None, num_tokens_total=20, use_fixed_output_lengths=True
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.skip(
        reason="activate for `megatron-core >= 0.16`, after fixing "
        "`raise TokenOverflowError(is_transient=False)` compatibility with "
        "legacy tests."
    )
    def test_events(self):
        """Test events."""
        env = self._run_test(
            num_requests=16,
            max_prompt_length=10,
            num_tokens_to_generate=32,
            context_buffer_size_gb=0.001,  # 0.001, # 8 blocks
            context_max_tokens=8,
            num_gap_steps=1,
        )

        expected_event_types = [
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
        ]
        result_event_types = [
            [e.type.name for e in r.events if e.type.name != 'GENERATED_TOKEN']
            for r in env.requests
        ]
        assert result_event_types == expected_event_types

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_event_timestamps(self):
        """Test that events are recorded with sensical timestamps.

        Verifies:
        1. Completed requests have ADD_ENGINE, ADD_CONTEXT, GENERATED_TOKEN(s), FINISH events
        2. Event timestamps are monotonically increasing
        3. TTFT (time-to-first-token) can be computed as first GENERATED_TOKEN - ADD_ENGINE
        """
        num_tokens_to_generate = 8
        env = self._run_test(
            num_requests=4,
            max_prompt_length=16,
            num_tokens_to_generate=num_tokens_to_generate,
            context_buffer_size_gb=0.1,
            num_gap_steps=0,
            track_generated_token_events=True,
        )

        # All requests should complete with this generous config (large buffer, no gap steps).
        assert all(r.status == Status.COMPLETED for r in env.requests)
        for request in env.requests:

            # Verify event types for completed requests
            event_types = [e.type.name for e in request.events]
            # Should be: ADD_ENGINE, ADD_CONTEXT, GENERATED_TOKEN (repeated), FINISH
            assert (
                event_types[0] == 'ADD_ENGINE'
            ), f"Request {request.request_id}: first event should be ADD_ENGINE, got {event_types[0]}"
            assert (
                event_types[1] == 'ADD_CONTEXT'
            ), f"Request {request.request_id}: second event should be ADD_CONTEXT, got {event_types[1]}"
            assert (
                event_types[-1] == 'FINISH'
            ), f"Request {request.request_id}: last event should be FINISH, got {event_types[-1]}"
            # Check that GENERATED_TOKEN events are in the middle
            gen_token_count = event_types.count('GENERATED_TOKEN')
            assert gen_token_count == len(request.generated_tokens), (
                f"Request {request.request_id}: GENERATED_TOKEN count ({gen_token_count}) != "
                f"generated_tokens length ({len(request.generated_tokens)})"
            )

            # Verify timestamps are monotonically increasing
            timestamps = [e.timestamp for e in request.events]
            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i - 1], (
                    f"Request {request.request_id}: timestamp[{i}] ({timestamps[i]}) < "
                    f"timestamp[{i-1}] ({timestamps[i-1]})"
                )

            # Verify TTFT is positive and sensical (first GENERATED_TOKEN - ADD_ENGINE)
            add_engine_ts = request.events[0].timestamp
            first_token_ts = request.events[2].timestamp  # First GENERATED_TOKEN event
            assert (
                request.events[2].type.name == 'GENERATED_TOKEN'
            ), f"Request {request.request_id}: event[2] should be GENERATED_TOKEN"
            ttft = first_token_ts - add_engine_ts
            assert ttft >= 0, f"Request {request.request_id}: TTFT is negative ({ttft})"

            # Verify total request time is positive
            finish_ts = request.events[-1].timestamp
            total_time = finish_ts - add_engine_ts
            assert (
                total_time >= ttft
            ), f"Request {request.request_id}: total_time ({total_time}) < TTFT ({ttft})"

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_mamba_chunked_prefill(self):
        """
        Test chunked prefill with a Mamba model.
        """
        skip_if_mamba_sequence_packing_not_available("mamba")

        # Context max tokens = 50.
        test_config = DynamicEngineTestConfig(
            model_provider="mamba",
            num_requests=0,
            num_tokens_to_generate=None,
            num_tokens_total=200,
            context_max_tokens=52,
            context_max_requests=5,
            context_block_size_tokens=256,
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        env = self._build_test_env(test_config)
        ctx = env.engine.context

        # Mock the model forward function to avoid possible numerics issues
        # caused by random inputs
        model_instance = env.engine.controller.inference_wrapped_model.model
        model_instance.forward = partial(mock_forward, vocab_size=test_config.vocab_size)

        # Request 1: 150 tokens
        req1_tokens = torch.randint(0, test_config.vocab_size, (130,), device='cuda')
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=req1_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=3),
        )

        # Request 2: 160 tokens
        req2_tokens = torch.randint(0, test_config.vocab_size, (160,), device='cuda')
        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=req2_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )

        # Request 3: 24 tokens
        req3_tokens = torch.randint(0, test_config.vocab_size, (24,), device='cuda')
        req3 = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=req3_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )

        # Add requests 1-3
        env.engine._add_request(req1)
        env.engine._add_request(req2)
        env.engine._add_request(req3)

        # Run step 1
        env.engine.schedule_waiting_requests()

        env.engine.step_modern()
        assert req1.finished_chunk_token_count == 52

        # Prepare for step 2
        env.engine.schedule_waiting_requests()

        # Verify that requests 2 and 3 are queued because request 1 is still running
        assert ctx.num_prefill_requests == 1
        active_ids = ctx.request_ids[: ctx.total_request_count].tolist()
        assert 1 in active_ids
        assert 2 not in active_ids
        assert 3 not in active_ids
        assert list(env.engine.waiting_request_ids) == [1, 2, 3]

        # Verify that active token count == max tokens
        assert ctx.active_token_count == 52

        # Verify that request 1 is the designated chunked prefill request
        assert ctx.chunked_prefill_request_id == 1

        # Run step 2
        env.engine.step_modern()
        assert req1.finished_chunk_token_count == 104

        # Prepare for step 3
        env.engine.schedule_waiting_requests()

        # Verify that request 2 got partially scheduled and is now
        # the designated chunked prefill request
        req2_idx = ctx.request_ids.tolist().index(2)
        assert req2_idx == 1
        assert ctx.num_prefill_requests == 2
        assert ctx.chunked_prefill_request_id == 2
        assert ctx.get_index_of_chunked_prefill_request() == req2_idx
        active_ids = ctx.request_ids[: ctx.total_request_count].tolist()
        assert 1 in active_ids
        assert 2 in active_ids
        assert 3 not in active_ids

        # Store the Mamba state tensor idx for request 2
        req2_mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[req2_idx].item()

        # Verify that the active token count is the maximum token count
        assert ctx.active_token_count == 52

        # Run step 3
        env.engine.step_modern()
        assert req1.finished_chunk_token_count == 104

        # Prepare for step 4
        env.engine.schedule_waiting_requests()

        # Verify that request 2 is still the first prefill request
        assert ctx.request_ids.tolist().index(2) == 1
        assert ctx.mamba_metadata.request_to_mamba_state_idx[1] == req2_mamba_idx

        # Verify that request 1 is running decode
        active_ids = ctx.request_ids[: ctx.total_request_count].tolist()
        assert ctx.num_decode_requests == 1
        assert 1 in active_ids

        # Verify that request 2 is still running prefill as the designated chunked prefill request
        assert ctx.num_prefill_requests == 1
        assert ctx.chunked_prefill_request_id == 2
        assert ctx.get_index_of_chunked_prefill_request() == 1

        # Verify that request 3 is still waiting
        assert 3 not in active_ids
        assert 3 in env.engine.waiting_request_ids

        # Verify that active token count == max tokens
        assert ctx.active_token_count == 52

        # Run step 4
        env.engine.step_modern()

        assert req2.finished_chunk_token_count == 77

        # Prepare for step 5
        env.engine.schedule_waiting_requests()

        # Verify that request 2 is still the first prefill request
        assert ctx.request_ids.tolist().index(2) == 1
        assert ctx.mamba_metadata.request_to_mamba_state_idx[1] == req2_mamba_idx

        # Run step 5
        env.engine.step_modern()
        assert req2.finished_chunk_token_count == 128

        # Prepare for step 6
        env.engine.schedule_waiting_requests()

        # Verify that request 1 has completed
        assert req1.status == Status.COMPLETED

        # Verify that request 2 is still the first prefill request
        assert ctx.request_ids.tolist().index(2) == 0
        assert ctx.mamba_metadata.request_to_mamba_state_idx[0] == req2_mamba_idx

        # Verify that request 3 is now scheduled as the chunked prefill request
        active_ids = ctx.request_ids[: ctx.total_request_count].tolist()
        assert 2 in active_ids
        assert 3 in active_ids
        assert ctx.chunked_prefill_request_id == 3
        req3_idx = active_ids.index(3)
        assert req3_idx == 1

        # Store the Mamba state tensor idx for request 3
        req3_mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[req3_idx].item()

        # Run step 6
        env.engine.step_modern()

        # Verify that request 2 has finished
        assert req2.status == Status.COMPLETED
        assert req3.finished_chunk_token_count == 20

        # Prepare for step 7
        env.engine.schedule_waiting_requests()

        # Verify that request 3 is now the first prefill request
        req3_idx = ctx.request_ids.tolist().index(3)
        assert req3_idx == 0
        assert ctx.mamba_metadata.request_to_mamba_state_idx[0] == req3_mamba_idx

        # Run step 7
        env.engine.step_modern()

        # Verify that request 3 has finished
        assert req3.status == Status.COMPLETED

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_chunked_prefill_avoid_single_token_chunk(self):
        """
        Test that chunked prefill scheduling avoids leaving exactly 1 token for the final chunk.
        This leads to a known bug in the Flash Attention kernel:
        https://github.com/Dao-AILab/flash-attention/issues/1537

        Scenario:
            - Max tokens per step (Chunk Size): 256
            - Request prompt length: 513

        Default scheduling would do:
            1. Chunk 256 (Remaining 257)
            2. Chunk 256 (Remaining 1) -> max_seqlen_q=1 triggers decode path in kernel
            3. Chunk 1

        Fixed scheduling should do:
            1. Chunk 256 (Remaining 257) -> 513 - 256 == 257. Schedule full 256.
            2. Chunk 255 (Remaining 2)   -> 257 tokens left. If we take 256, 1 remains.
                                            So we reduce chunk to 255.
            3. Chunk 2   (Remaining 0)
        """
        chunk_size = 256
        # Prompt length designed to trigger the edge case: Chunk + (Chunk + 1)
        # 256 + 255 + 2 = 513
        prompt_len = 513

        test_config = DynamicEngineTestConfig(
            model_provider="gpt",
            num_requests=0,
            num_tokens_to_generate=None,
            num_tokens_total=prompt_len + 1,
            context_max_tokens=chunk_size,
            context_max_requests=1,
            context_block_size_tokens=256,
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        env = self._build_test_env(test_config)
        ctx = env.engine.context

        # Mock the model forward function to avoid possible numerics issues
        # caused by random inputs
        model_instance = env.engine.controller.inference_wrapped_model.model
        model_instance.forward = partial(mock_forward, vocab_size=test_config.vocab_size)

        # Create a request with length 513
        req_tokens = torch.randint(0, test_config.vocab_size, (prompt_len,), device='cuda')
        req = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=req_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )

        env.engine._add_request(req)

        assert req.status == Status.ACTIVE_AND_GENERATING_TOKENS

        # --- Step 1 ---
        # Available: 256. Remaining: 513.
        # Logic: 513 - 256 = 257. Not 1. Schedule full 256.
        env.engine.step_modern()

        assert env.engine.context.total_request_count == 0, env.engine.context.total_request_count

        assert (
            req.finished_chunk_token_count == 256
        ), f"Step 1: Expected 256 tokens processed, got {req.finished_chunk_token_count}"

        # --- Step 2 ---
        # Available: 256. Remaining un-prefilled: 257.
        # Logic: 257 - 256 = 1. This is the edge case!
        # Fix should reduce chunk size by 1 (to 255).
        env.engine.step_modern()

        assert env.engine.context.total_request_count == 0, env.engine.context.total_request_count

        # 256 (previous) + 255 (this step) = 511
        assert req.finished_chunk_token_count == 511, (
            "Step 2: Expected 511 tokens processed (256+255), "
            f"got {req.finished_chunk_token_count}. "
        )

        # --- Step 3 ---
        # Remaining un-prefilled: 2. Available: 256.
        # Logic: 2 <= 256. Schedule 2.
        env.engine.schedule_waiting_requests()
        env.engine.step_modern()

        # Verify request finishes prefill and completes
        assert ctx.num_prefill_requests == 0
        assert req.status == Status.COMPLETED

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_chunked_prefill_delay_scheduling_for_unavoidable_single_token_chunk(self):
        """
        Test that chunked prefill scheduling delays execution when the only available
        option is to schedule a chunk of size 1 that leaves exactly 1 token remaining.

        Scenario:
            - Max tokens per step: 256
            - Request A: 254 token prompt
            - Request B: 2 token prompt

        Sequence:
            1. Step 1 scheduling:
               - Request A is scheduled (255 tokens).
               - Context has 1 token available (256 - 255).
               - Request B has 2 tokens remaining.
               - If we schedule 1 token for B, it leaves exactly 1 token for its final chunk,
                 crashing FA3. Since chunk_length is 1, we can't safely reduce it.
                 The engine MUST delay scheduling Request B.
            2. Step 1 executes prefill for Request A only.
            3. Step 2 scheduling:
               - Request A enters decode phase (takes 1 active token).
               - Context has 255 tokens available (256 - 1).
               - Request B is now safely scheduled for its full 2 tokens.
        """
        test_config = DynamicEngineTestConfig(
            model_provider="gpt",
            num_requests=0,
            num_tokens_to_generate=None,
            num_tokens_total=256,
            context_max_tokens=256,
            context_max_requests=2,
            context_block_size_tokens=256,
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        env = self._build_test_env(test_config)
        ctx = env.engine.context

        # Mock the model forward function to avoid possible numerics issues
        model_instance = env.engine.controller.inference_wrapped_model.model
        model_instance.forward = partial(mock_forward, vocab_size=test_config.vocab_size)

        # Add Request A (Length 255)
        req_a_tokens = torch.randint(0, test_config.vocab_size, (255,), device='cuda')
        req_a = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=req_a_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        env.engine._add_request(req_a)

        # Add Request B (Length 2)
        req_b_tokens = torch.randint(0, test_config.vocab_size, (2,), device='cuda')
        req_b = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=req_b_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        env.engine._add_request(req_b)

        # --- Step 1 ---
        # Should schedule Request A fully (255), but delay Request B
        env.engine.step_modern()

        assert req_a.status == Status.COMPLETED

        # Request B MUST be delayed (0 tokens processed) to avoid the FA3 bug
        assert (
            req_b.finished_chunk_token_count == 0
        ), "Request B should have been delayed to avoid leaving a 1-token chunk"
        assert len(env.engine.waiting_request_ids) == 1
        assert env.engine.waiting_request_ids[0] == 2

        # --- Step 2 ---
        # Request A has completed. Context has 256 tokens available.
        # Request B can now schedule its full 2 tokens safely.
        env.engine.step_modern()

        assert req_b.status == Status.COMPLETED
        assert len(env.engine.waiting_request_ids) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    @pytest.mark.parametrize("skip_prompt_log_probs", [True, False])
    @torch.inference_mode()
    def test_chunked_prefill_with_log_probs(
        self, materialize_only_last_token_logits: bool, skip_prompt_log_probs: bool
    ):
        """
        Test that chunked prefill correctly handles log probs across all branches
        of the log-prob accumulation logic.
        When materialize_only_last_token_logits=True, skip_prompt_log_probs must be True.
        """
        if materialize_only_last_token_logits and not skip_prompt_log_probs:
            with pytest.raises(AssertionError, match="only last token logits are materialized"):
                self._run_test(
                    num_requests=1,
                    min_prompt_length=1200,
                    max_prompt_length=1200,
                    num_tokens_to_generate=8,
                    materialize_only_last_token_logits=True,
                    return_log_probs=True,
                    skip_prompt_log_probs=False,
                    model_provider="gpt",
                    context_block_size_tokens=256,
                    context_max_tokens=1000,
                    enable_chunked_prefill=True,
                )
            return

        prompt_length = 1200
        num_tokens_to_generate = 8

        env = self._run_test(
            num_requests=1,
            min_prompt_length=prompt_length,
            max_prompt_length=prompt_length,
            num_tokens_to_generate=num_tokens_to_generate,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            return_log_probs=True,
            skip_prompt_log_probs=skip_prompt_log_probs,
            model_provider="gpt",
            context_block_size_tokens=256,
            context_max_tokens=1000,
            enable_chunked_prefill=True,
        )

        # Validate results
        for request in env.requests:
            if request.status != Status.COMPLETED:
                continue

            # Validate generated log probs
            assert (
                request.generated_log_probs is not None
            ), f"Request {request.request_id}: generated_log_probs should not be None"
            assert len(request.generated_log_probs) == len(request.generated_tokens), (
                f"Request {request.request_id}: Expected {len(request.generated_tokens)} "
                f"generated log probs, got {len(request.generated_log_probs)}"
            )

            if skip_prompt_log_probs:
                assert request.prompt_log_probs is None or len(request.prompt_log_probs) == 0, (
                    f"Request {request.request_id}: prompt_log_probs should be empty when "
                    f"skip_prompt_log_probs=True, but got "
                    f"{len(request.prompt_log_probs) if request.prompt_log_probs else 0} items"
                )
            else:
                assert len(request.prompt_log_probs) == prompt_length - 1, (
                    f"Request {request.request_id}: Expected {prompt_length - 1} "
                    f"prompt log probs, got {len(request.prompt_log_probs)}"
                )

            # Validate each generated log prob
            for i, log_prob in enumerate(request.generated_log_probs):
                assert not math.isnan(log_prob) and not math.isinf(log_prob), (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is invalid"
                )
                assert -50.0 <= log_prob <= 0.0, (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is out of expected range [-50.0, 0.0]"
                )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_chunked_prefill_log_probs_match_baseline(self):
        """
        Verify that chunked prefill computes the exact same prompt log probabilities
        as non-chunked prefill. This explicitly catches the bug where garbage
        sampled tokens corrupt the prompt log probabilities at chunk boundaries.
        """
        prompt_length = 512
        num_tokens_to_generate = 4

        # Create a deterministic mock forward pass that returns logits
        # dependent ONLY on position_ids. This guarantees the same logits
        # whether processed in one giant chunk or split across multiple chunks.
        def deterministic_mock_forward(input_ids, position_ids, attention_mask, *args, **kwargs):
            vocab_size = kwargs["vocab_size"]
            # Use torch.linspace to generate varying but 100% deterministic logits per position
            static_logits = torch.linspace(
                -50, 50, 4096 * vocab_size, device=input_ids.device, dtype=torch.bfloat16
            ).view(4096, vocab_size)

            return static_logits[position_ids]

        def get_log_probs(chunked: bool, max_tokens: int):
            test_config = DynamicEngineTestConfig(
                num_requests=0,  # Added manually below
                num_tokens_to_generate=num_tokens_to_generate,
                materialize_only_last_token_logits=False,
                return_log_probs=True,
                skip_prompt_log_probs=False,
                model_provider="gpt",
                context_block_size_tokens=256,
                context_max_requests=1,
                context_max_tokens=max_tokens,
                max_sequence_length=1024,
                enable_chunked_prefill=chunked,
                use_cuda_graphs_for_non_decode_steps=False,
            )
            env = self._build_test_env(test_config)

            # Patch the mock forward to be deterministic
            model_instance = env.engine.controller.inference_wrapped_model.model
            model_instance.forward = partial(
                deterministic_mock_forward, vocab_size=test_config.vocab_size
            )

            # Ensure identical prompt tokens for both runs
            torch.manual_seed(42)
            req_tokens = torch.randint(0, test_config.vocab_size, (prompt_length,), device='cuda')
            req = DynamicInferenceRequest(
                request_id=1,
                prompt_tokens=req_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    return_log_probs=True,
                    skip_prompt_log_probs=False,
                    termination_id=-1,
                ),
            )

            env.engine._add_request(req)

            # Drive the engine until the request finishes
            while env.engine.has_unfinished_requests():
                env.engine.schedule_waiting_requests()
                env.engine.step_modern()

            return req.prompt_log_probs

        # Run non-chunked baseline (all 512 tokens in one pass)
        baseline_log_probs = get_log_probs(chunked=False, max_tokens=1000)

        # Run chunked (512 tokens split across 256-token boundaries)
        chunked_log_probs = get_log_probs(chunked=True, max_tokens=256)

        assert baseline_log_probs is not None, "Baseline prompt_log_probs is missing"
        assert chunked_log_probs is not None, "Chunked prompt_log_probs is missing"

        assert len(baseline_log_probs) == prompt_length - 1
        assert len(chunked_log_probs) == prompt_length - 1

        # Compare element-wise using math.isclose to handle minor floating point rounding
        for i, (base_lp, chunk_lp) in enumerate(zip(baseline_log_probs, chunked_log_probs)):
            assert math.isclose(base_lp, chunk_lp, rel_tol=1e-3, abs_tol=1e-3), (
                f"Log prob mismatch at prompt token index {i}: "
                f"Baseline={base_lp:.4f}, Chunked={chunk_lp:.4f}. "
                "This indicates log prob corruption at chunk boundaries!"
            )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("skip_prompt_log_probs", [True, False])
    @torch.inference_mode()
    def test_top_n_logprobs_dynamic(self, skip_prompt_log_probs: bool):
        """
        Test that top_n_logprobs are computed correctly in dynamic batching mode.
        Verifies:
        1. top_n_logprobs are returned for generated tokens
        2. skip_prompt_log_probs controls whether prompt top-n logprobs are skipped
        3. The top-n values are consistent with the selected token's log prob
        """
        # Build test environment with multiple requests of varying lengths
        test_config = DynamicEngineTestConfig(
            num_requests=4,
            min_prompt_length=4,
            max_prompt_length=12,
            num_tokens_to_generate=4,
            materialize_only_last_token_logits=False,
        )
        env = self._build_test_env(test_config)

        # Create requests with top_n_logprobs enabled
        top_n = 5
        requests_to_add = []
        for request in env.requests:
            # Update sampling params to include top_n_logprobs
            request.sampling_params = SamplingParams(
                num_tokens_to_generate=test_config.num_tokens_to_generate,
                termination_id=test_config.vocab_size - 1,
                return_log_probs=True,
                top_n_logprobs=top_n,
                skip_prompt_log_probs=skip_prompt_log_probs,
                top_k=10,  # Add some sampling randomness
            )
            requests_to_add.append(request)

        # Add requests and run inference
        for request in requests_to_add:
            env.engine._add_request(request)

        # Step engine until all requests are finished
        while env.engine.has_unfinished_requests():
            result = env.engine.step_modern()

        # Validate results
        for request in requests_to_add:
            assert request.status == Status.COMPLETED, f"Request {request.request_id} not completed"

            # Validate generated top-n logprobs
            assert hasattr(
                request, 'generated_top_n_logprobs'
            ), f"Request {request.request_id} missing generated_top_n_logprobs"
            assert (
                request.generated_top_n_logprobs is not None
            ), f"Request {request.request_id} has None generated_top_n_logprobs"
            assert len(request.generated_top_n_logprobs) == len(
                request.generated_tokens
            ), f"Request {request.request_id}: generated_top_n_logprobs length mismatch"

            # Validate each top-n dict
            for i, top_n_dict in enumerate(request.generated_top_n_logprobs):
                assert isinstance(
                    top_n_dict, dict
                ), f"Request {request.request_id}, token {i}: top_n_dict is not a dict"
                assert (
                    len(top_n_dict) <= top_n
                ), f"Request {request.request_id}, token {i}: too many top-n entries"
                assert (
                    len(top_n_dict) > 0
                ), f"Request {request.request_id}, token {i}: empty top-n dict"

            # Validate prompt top-n logprobs based on skip_prompt_log_probs flag
            if not skip_prompt_log_probs:
                assert hasattr(
                    request, 'prompt_top_n_logprobs'
                ), f"Request {request.request_id} missing prompt_top_n_logprobs"
                assert (
                    request.prompt_top_n_logprobs is not None
                ), f"Request {request.request_id} has None prompt_top_n_logprobs"
                # Prompt top-n should have N-1 entries (excluding first token)
                expected_prompt_top_n_len = len(request.prompt_tokens) - 1
                assert (
                    len(request.prompt_top_n_logprobs) == expected_prompt_top_n_len
                ), f"Request {request.request_id}: prompt_top_n_logprobs length {len(request.prompt_top_n_logprobs)} != expected {expected_prompt_top_n_len}"

                # Validate each prompt top-n dict
                for i, top_n_dict in enumerate(request.prompt_top_n_logprobs):
                    assert isinstance(
                        top_n_dict, dict
                    ), f"Request {request.request_id}, prompt token {i}: top_n_dict is not a dict"
                    assert (
                        len(top_n_dict) <= top_n
                    ), f"Request {request.request_id}, prompt token {i}: too many top-n entries"
                    assert (
                        len(top_n_dict) > 0
                    ), f"Request {request.request_id}, prompt token {i}: empty top-n dict"
            else:
                # When skip_prompt_log_probs is True, prompt_top_n_logprobs should be None or empty
                if hasattr(request, 'prompt_top_n_logprobs'):
                    assert (
                        request.prompt_top_n_logprobs is None
                        or len(request.prompt_top_n_logprobs) == 0
                    ), f"Request {request.request_id}: prompt_top_n_logprobs should be None or empty when skip_prompt_log_probs is True"

            # Validate consistency between log_probs and top_n_logprobs
            if hasattr(request, 'generated_log_probs') and request.generated_log_probs is not None:
                assert len(request.generated_log_probs) == len(
                    request.generated_top_n_logprobs
                ), f"Request {request.request_id}: generated_log_probs and generated_top_n_logprobs length mismatch"

                # Check that the selected token's log prob appears in the top-n
                for i, (log_prob, top_n_dict, token_id) in enumerate(
                    zip(
                        request.generated_log_probs,
                        request.generated_top_n_logprobs,
                        request.generated_tokens,
                    )
                ):
                    # Get the token string for this token_id
                    token_str = env.engine.controller.tokenizer.detokenize([token_id])
                    # The selected token should be in the top-n
                    assert (
                        token_str in top_n_dict
                    ), f"Request {request.request_id}, token {i}: selected token '{token_str}' not in top-n"
                    # The log prob should match (with some tolerance for floating point precision)
                    # Using 0.1 tolerance to account for FP16/BF16 precision in mixed precision training
                    assert (
                        abs(log_prob - top_n_dict[token_str]) < 0.1
                    ), f"Request {request.request_id}, token {i}: log_prob mismatch {log_prob} vs {top_n_dict[token_str]}"

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("max_requests", [None, 4])
    @torch.inference_mode()
    def test_max_requests(self, max_requests: int | None):
        """Test max requests."""
        env = self._run_test(
            context_max_requests=max_requests, num_tokens_to_generate=16, num_gap_steps=1
        )
        step_count = env.engine.context.step_count
        context = env.engine.context
        if max_requests is None:
            assert context.max_requests == 816
            assert step_count == 23
        else:
            assert max_requests < len(env.requests), (
                f"Test is only useful if max_requests ({max_requests}) < "
                f"num_requests ({len(env.requests)})."
            )
            assert context.max_requests == 4
            assert step_count == 35
        assert context.block_allocator.active_count == 655

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("static_kv_memory_pointers", [True, False])
    @pytest.mark.parametrize("kv_cache_management_mode", ["persist", "offload", "recompute"])
    @torch.inference_mode()
    def test_suspend_resume_cycle(self, kv_cache_management_mode, static_kv_memory_pointers):
        """Full suspend -> resume cycle with memory, data, and address checks."""
        needs_tms = static_kv_memory_pointers and kv_cache_management_mode != "persist"

        test_config = DynamicEngineTestConfig(
            kv_cache_management_mode=kv_cache_management_mode,
            static_kv_memory_pointers=static_kv_memory_pointers,
        )

        # Without TMS, these combos must assert on construction.
        if needs_tms and not HAVE_TORCH_MEMORY_SAVER:
            with pytest.raises(AssertionError, match="Static KV memory pointers"):
                self._build_test_env(test_config)
            return

        env = self._build_test_env(test_config)
        engine = env.engine
        context = engine.context

        assert engine.state != EngineState.SUSPENDED
        assert context.is_tensor_state_allocated

        deallocates = kv_cache_management_mode != "persist"
        uses_tms = context._uses_torch_memory_saver
        preserves_data = kv_cache_management_mode != "recompute"

        # Write a deterministic pattern for data integrity check.
        if preserves_data:
            context.memory_buffer.copy_(torch.randn_like(context.memory_buffer))
            expected = context.memory_buffer.clone()

        addr_before = context.memory_buffer.data_ptr()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        if uses_tms:
            phys_mem_before = torch.cuda.mem_get_info()[0]

        # Suspend.
        engine.suspend()
        assert engine.state == EngineState.SUSPENDED
        assert not context.is_tensor_state_allocated

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_suspended = torch.cuda.memory_allocated()
        if uses_tms:
            phys_mem_suspended = torch.cuda.mem_get_info()[0]

        if deallocates and not uses_tms:
            assert mem_suspended < mem_before, (
                f"GPU memory should decrease after suspend "
                f"(mode={kv_cache_management_mode}). "
                f"Before: {mem_before}, After: {mem_suspended}"
            )
        else:
            assert mem_suspended == mem_before, (
                f"Memory should not change on suspend. "
                f"Before: {mem_before}, Suspended: {mem_suspended}"
            )

        if uses_tms:
            assert phys_mem_suspended > phys_mem_before, (
                f"torch_memory_saver should free physical GPU memory after suspend. "
                f"Before: {phys_mem_before}, After: {phys_mem_suspended}"
            )

        # Resume.
        engine.resume()
        assert engine.state != EngineState.SUSPENDED
        assert context.is_tensor_state_allocated

        if deallocates and not uses_tms:
            torch.cuda.synchronize()
            mem_resumed = torch.cuda.memory_allocated()
            assert mem_resumed > mem_suspended, (
                f"GPU memory should increase after resume. "
                f"Suspended: {mem_suspended}, Resumed: {mem_resumed}"
            )

        if uses_tms:
            torch.cuda.synchronize()
            phys_mem_resumed = torch.cuda.mem_get_info()[0]
            assert phys_mem_resumed < phys_mem_suspended, (
                f"torch_memory_saver should re-allocate physical GPU memory after resume. "
                f"Suspended: {phys_mem_suspended}, Resumed: {phys_mem_resumed}"
            )

        # Data integrity.
        if preserves_data:
            torch.testing.assert_close(
                context.memory_buffer,
                expected,
                msg="memory_buffer data must be identical after suspend/resume",
            )

        # Address stability when CUDA graphs persist.
        if static_kv_memory_pointers:
            addr_after = context.memory_buffer.data_ptr()
            assert addr_before == addr_after, (
                f"Tensor address must be stable when static_kv_memory_pointers is set. "
                f"Before: {addr_before:#x}, After: {addr_after:#x}"
            )

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("use_checkpoint", [False, True], ids=["persist", "recompute"])
    @torch.inference_mode()
    def test_staleness_tracking(self, use_checkpoint):
        """Test that training-iteration stamps are correctly tracked.
        The use_checkpoint parameter simulates the behavior of different kv_cache_management_mode.
        """
        PROMPT_LEN = 8
        NUM_TOKENS = 8

        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=PROMPT_LEN,
            max_prompt_length=PROMPT_LEN,
            num_tokens_to_generate=NUM_TOKENS,
        )
        env = self._build_test_env(test_config)
        engine = env.engine

        for i in range(2):
            prompt_tokens = torch.randint(
                0,
                test_config.vocab_size - 1,
                (PROMPT_LEN,),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            engine._add_request(
                DynamicInferenceRequest(
                    request_id=i,
                    prompt_tokens=prompt_tokens,
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=NUM_TOKENS, termination_id=-1
                    ),
                )
            )

        def set_epoch(epoch):
            """Simulate receiving a SET_GENERATION_EPOCH signal."""
            engine._generation_epoch = epoch
            for entry in engine.requests.values():
                request = entry.record[-1]
                total = len(request.prompt_tokens) + len(request.generated_tokens)
                if total > 0:
                    boundary = (total - 1, epoch)
                    if request.policy_epoch is None:
                        request.policy_epoch = [(0, epoch)]
                    else:
                        request.policy_epoch.append(boundary)
                    if request.kv_cache_epoch is None:
                        request.kv_cache_epoch = [(0, epoch)]
                    else:
                        request.kv_cache_epoch.append(boundary)

        # Steps without a generation epoch set — no stamps.
        engine.step_modern()
        for entry in engine.requests.values():
            assert entry.record[-1].policy_epoch is None
            assert entry.record[-1].kv_cache_epoch is None

        # Generation epoch 0: stamps all active requests at their current length.
        set_epoch(0)
        for _ in range(2):
            engine.step_modern()

        for entry in engine.requests.values():
            ps = entry.record[-1].policy_epoch
            ks = entry.record[-1].kv_cache_epoch
            assert ps == ks == [(0, 0)]

        # Generation epoch 1: boundary at current length, before next step.
        set_epoch(1)
        for _ in range(3):
            engine.step_modern()

        for entry in engine.requests.values():
            ps = entry.record[-1].policy_epoch
            ks = entry.record[-1].kv_cache_epoch
            assert ps == ks == [(0, 0), (PROMPT_LEN + 2, 1)]

        # Simulate RECOMPUTE — checkpoint clears kv_cache so the engine's
        # stamping logic will recreate it fresh on the next epoch signal.
        if use_checkpoint:
            for entry in engine.requests.values():
                old_req = entry.record[-1]
                event_add_engine = old_req.event_add_engine
                entry.record.checkpoint()
                # Prevent TTFT crash due to missing _add_request in test.
                entry.record[-1].event_add_engine = event_add_engine

            for entry in engine.requests.values():
                assert entry.record[-1].kv_cache_epoch is None

        # Generation epoch 2: stamp then generate remaining tokens.
        set_epoch(2)

        finished_records = []
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            finished_records.extend(result["finished_request_records"])

        for record in finished_records:
            merged = record.merge()

            assert merged.policy_epoch == [(0, 0), (PROMPT_LEN + 2, 1), (PROMPT_LEN + 5, 2)]

            if use_checkpoint:
                # KV cache was cleared by checkpoint; stamping logic recreated it at epoch 2.
                assert merged.kv_cache_epoch == [(0, 2)]
            else:
                assert merged.kv_cache_epoch == [(0, 0), (PROMPT_LEN + 2, 1), (PROMPT_LEN + 5, 2)]

        # Verify checkpoint clears kv_cache_epoch and preserves policy.
        record = finished_records[0]
        record.checkpoint()
        assert record[-1].policy_epoch == merged.policy_epoch
        assert record[-1].kv_cache_epoch is None

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_with_early_termination(self):
        """Test that speculative decoding handles premature request termination safely
        (e.g. hitting max_sequence_length mid-speculative-batch)."""

        # Set max_sequence_length tight so it terminates during a speculative step
        test_config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=3,  # Prompt (4) + Gen (3) = 7
            max_sequence_length=7,  # Will force termination after 3 tokens
            model_provider="gpt",
            num_speculative_tokens=3,
            materialize_only_last_token_logits=False,
        )

        env = self._build_test_env(test_config)
        unwrapped_model = env.engine.controller.inference_wrapped_model.model

        # Mock forward to return deterministic data so speculative tokens are always accepted
        hidden_size = unwrapped_model.config.hidden_size

        def mock_mtp_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))

            base_logits = torch.zeros(
                tokens.size(0),
                tokens.size(1),
                test_config.vocab_size,
                device=tokens.device,
                dtype=torch.bfloat16,
            )
            base_logits[:, :, 0] = 100.0  # High probability for token 0

            # Cache hidden states for serial MTP computation
            unwrapped_model._decoder_hidden_states_cache = torch.zeros(
                tokens.size(1), 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth, runtime_gather_output=True
        ):
            n = hidden_states.size(0)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits[:, :, 0] = 100.0  # High probability for token 0
            return hidden_states, logits

        unwrapped_model.forward = mock_mtp_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        env.engine._add_request(env.requests[0])
        env.engine.schedule_waiting_requests()

        # Step engine until finished naturally
        # This allows the bookkeeping logic to gracefully truncate the
        # speculative tokens to the max_sequence_length boundary.
        while env.engine.has_unfinished_requests():
            env.engine.step_modern()

        assert env.requests[0].status == Status.COMPLETED

        # It should trim the output to the max_sequence_length boundary
        # Prompt was 4, Max was 7, so it should have generated exactly 3 tokens.
        assert len(env.requests[0].generated_tokens) == 3

        # Validate the engine's tracking state is clean
        assert env.engine.context.active_token_count == 0
        assert env.engine.context.total_request_count == 0

    @pytest.mark.internal
    @torch.inference_mode()
    def test_speculative_block_boundary_crossing(self):
        """Test to verify KV cache block boundary crossing logic.

        When a request fills exactly one block and speculative decoding generates
        multiple tokens, the first new token shouldn't incorrectly overwrite the old block.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=256,
            max_prompt_length=256,
            num_tokens_to_generate=3,
            num_speculative_tokens=2,
            context_block_size_tokens=256,  # Exactly matches prompt length
            context_max_requests=16,
            model_provider="gpt",
            materialize_only_last_token_logits=False,
            use_fixed_output_lengths=True,
        )
        env = self._build_test_env(test_config)

        req = env.requests[0]
        req.sampling_params.num_tokens_to_generate = 3
        env.engine._add_request(req)
        env.engine.schedule_waiting_requests()

        # Step 1: Prefill. Processes the 4 prompt tokens.
        # At the end of this step, `update_requests` prepares the token indices for Step 2.
        # It assigns block indices for the 3 upcoming tokens (1 base + 2 spec).
        env.engine.step_modern()

        context = env.engine.context

        # The request has 2 blocks allocated now (1 for prompt, 1 for the new 3 tokens)
        assigned_blocks = context.request_to_kv_block_ids[0]
        first_block = assigned_blocks[0].item()
        second_block = assigned_blocks[1].item()

        # The active_token_count for the next step should be 3
        assert context.active_token_count == 3

        # Check which blocks the 3 new tokens are assigned to.
        # Because the prompt exactly filled the first block, ALL 3 new tokens
        # MUST go to the second block.
        token_blocks = context.token_to_block_idx[: context.active_token_count].tolist()

        assert token_blocks == [
            second_block,
            second_block,
            second_block,
        ], f"Expected all new tokens to go to block {second_block}, but got {token_blocks}."

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_stop_word_hit(self):
        """Test that if an accepted speculative token completes a stop word,
        the request correctly triggers the stop logic without crashing."""

        test_config = DynamicEngineTestConfig(
            num_requests=0,  # We will manually add our request cleanly
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=10,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward to deterministically output an ascending sequence (1->2->3...)
        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape

            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
            base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)

            # Cache hidden states for serial MTP computation
            unwrapped_model._decoder_hidden_states_cache = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth, runtime_gather_output=True
        ):
            n = hidden_states.size(0)
            # Predict next_token_ids + 1 (continuing the ascending sequence)
            pred_toks = (next_token_ids + 1).clamp(max=test_config.vocab_size - 1)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits.scatter_(2, pred_toks.transpose(0, 1).unsqueeze(-1), 100.0)
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        # Add the request formally to ensure all internal state tensors align
        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=10, termination_id=99),
        )

        # Inject the parsed stop word IDs
        tracked_req = env.engine.get_request(0)
        tracked_req.stop_word_ids = [[8, 9]]  # The sequence will generate 5, 6, 7, 8, 9, ...

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        # Retrieve the finalized request from the engine's output
        finished_req = finished_records[0].merge()

        assert finished_req.status == Status.COMPLETED
        # Since num_tokens_to_generate=10, output should stop early at ~7 tokens
        assert len(finished_req.generated_tokens) < 10
        # Verify the stop word was actually generated and caused the termination
        token_pairs = [
            finished_req.generated_tokens[i : i + 2]
            for i in range(len(finished_req.generated_tokens) - 1)
        ]
        assert [8, 9] in token_pairs

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_long_stop_word_hit(self):
        """Test that if an accepted speculative token completes a long stop word
        (length > num_speculative_tokens), it is correctly detected."""

        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=10,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward to deterministically output an ascending sequence
        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape

            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
            base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)

            # Cache hidden states for serial MTP computation
            unwrapped_model._decoder_hidden_states_cache = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth, runtime_gather_output=True
        ):
            n = hidden_states.size(0)
            # Predict next_token_ids + 1 (continuing the ascending sequence)
            pred_toks = (next_token_ids + 1).clamp(max=test_config.vocab_size - 1)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits.scatter_(2, pred_toks.transpose(0, 1).unsqueeze(-1), 100.0)
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=10, termination_id=99),
        )

        # Stop word length 3 > num_speculative_tokens (2)
        tracked_req = env.engine.get_request(0)
        tracked_req.stop_word_ids = [[7, 8, 9]]

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        finished_req = finished_records[0].merge()

        assert finished_req.status == Status.COMPLETED
        assert len(finished_req.generated_tokens) < 10
        token_triplets = [
            finished_req.generated_tokens[i : i + 3]
            for i in range(len(finished_req.generated_tokens) - 2)
        ]
        assert [7, 8, 9] in token_triplets

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_stop_word_truncates_trailing_tokens(self):
        """Test that when a stop word lands in the middle of speculative tokens,
        the extra tokens generated after the stop word are removed.

        With num_speculative_tokens=2, each step produces up to 3 tokens
        (1 base + 2 speculative). If the stop word is [6] and the engine
        generates [5, 6, 7] in one step, token 7 must be truncated so the
        output ends with the stop word [6]."""

        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=10,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward to deterministically output an ascending sequence (1->2->3...)
        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape

            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
            base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)

            # Cache hidden states for serial MTP computation
            unwrapped_model._decoder_hidden_states_cache = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth, runtime_gather_output=True
        ):
            n = hidden_states.size(0)
            # Predict next_token_ids + 1 (continuing the ascending sequence)
            pred_toks = (next_token_ids + 1).clamp(max=test_config.vocab_size - 1)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits.scatter_(2, pred_toks.transpose(0, 1).unsqueeze(-1), 100.0)
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=10, termination_id=99),
        )

        # Stop word [6] will land in the middle of a speculative batch [5, 6, 7].
        # Token 7 should be truncated from the output.
        tracked_req = env.engine.get_request(0)
        tracked_req.stop_word_ids = [[6]]

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        finished_req = finished_records[0].merge()

        assert finished_req.status == Status.COMPLETED
        # The output should end exactly at the stop word, with no trailing tokens.
        assert finished_req.generated_tokens[-1] == 6, (
            f"Expected last token to be stop word 6, "
            f"got {finished_req.generated_tokens[-1]}. "
            f"Trailing tokens after stop word were not truncated. "
            f"Full output: {finished_req.generated_tokens}"
        )
        # Verify no tokens after the stop word exist
        assert 7 not in finished_req.generated_tokens, (
            f"Token 7 should have been truncated after stop word 6. "
            f"Full output: {finished_req.generated_tokens}"
        )

    @pytest.mark.internal
    @torch.inference_mode()
    def test_speculative_sequence_length_double_counting(self):
        """Test to verify active_sequence_lengths is not double-counted.

        If active sequence length is double-counted during speculative decoding,
        the request will terminate prematurely before generating the requested tokens.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=6,
            max_sequence_length=10,  # Exactly prompt (4) + generate (6)
            context_max_requests=16,
            num_speculative_tokens=2,
            model_provider="gpt",
            materialize_only_last_token_logits=False,
            use_fixed_output_lengths=False,
            context_max_tokens=512,
        )
        env = self._build_test_env(test_config)

        # Mock forward pass to return deterministic base logits.
        # Speculative tokens will be wrong (predicted by MTP as tokens + 5)
        # to guarantee rejection every time.
        model = env.engine.controller.inference_wrapped_model.model
        hidden_size = model.config.hidden_size

        def mock_mtp_forward_reject(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape

            # Base model correctly predicts tokens + 1
            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
            base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)

            # Cache hidden states for serial MTP computation
            model._decoder_hidden_states_cache = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth, runtime_gather_output=True
        ):
            n = hidden_states.size(0)
            # Predict wildly wrong tokens (+ 5) to guarantee rejection
            wrong_toks = (next_token_ids + 5).clamp(max=test_config.vocab_size - 1)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits.scatter_(2, wrong_toks.transpose(0, 1).unsqueeze(-1), 100.0)
            return hidden_states, logits

        model.forward = mock_mtp_forward_reject
        model.compute_mtp_single_step = mock_compute_mtp_single_step

        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=6, termination_id=99),
        )

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        finished_req = finished_records[0].merge()

        # If there is double counting, the tracked active length will outpace the actual
        # generated tokens, causing premature termination when it thinks it hit max_sequence_length.
        assert finished_req.status == Status.COMPLETED
        assert (
            len(finished_req.generated_tokens) == 6
        ), f"Expected 6 tokens, got {len(finished_req.generated_tokens)}. Double counting occurred."

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_with_eviction_and_swapping(self):
        """Test that speculative decoding works correctly when requests are paused and evicted.

        This exercises the `_swap_book_keeping_tensors` logic with the 2D `new_speculative_tokens`
        tensor, ensuring no dimensional mismatch or index errors occur during tensor swapping.
        """
        # Very constrained memory environment to force pausing and eviction
        test_config = DynamicEngineTestConfig(
            num_requests=3,
            min_prompt_length=256,
            max_prompt_length=256,
            num_tokens_to_generate=512,
            context_block_size_tokens=256,
            num_speculative_tokens=2,
            context_buffer_size_gb=0.00064,  # 640 KB
            context_paused_buffer_size_gb=0.0,  # 0 paused buffer forces immediate eviction
            model_provider="gpt",
            materialize_only_last_token_logits=False,
            use_fixed_output_lengths=True,
        )

        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward pass to return safe, deterministic logits to avoid NaN/Inf crashes
        # in torch.multinomial caused by randomly initialized weights.
        def mock_safe_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape

            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            base_logits[:, :, 0] = 100.0  # Force model to deterministically pick token 0

            # Cache hidden states for serial MTP computation
            unwrapped_model._decoder_hidden_states_cache = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth, runtime_gather_output=True
        ):
            n = hidden_states.size(0)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits[:, :, 0] = 100.0  # Force speculative heads to also pick token 0
            return hidden_states, logits

        unwrapped_model.forward = mock_safe_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        # Add all requests at once. They will all start prefill, but as they generate
        # and request more blocks, the engine will run out of active blocks.
        # Since paused_buffer_size is 0, any request that pauses will immediately
        # overflow the paused buffer and trigger an eviction.
        for request in env.requests:
            request.sampling_params.num_tokens_to_generate = 512
            env.engine._add_request(request)

        eviction_occurred = False

        # Step the engine manually until all requests finish.
        while env.engine.has_unfinished_requests():
            # Record the number of evicted requests before the step
            evicted_before = env.engine.evicted_request_count

            # Step the engine
            env.engine.schedule_waiting_requests()
            env.engine.step_modern()

            # Check if any request was evicted during this step
            if env.engine.evicted_request_count > evicted_before:
                eviction_occurred = True

        # Assert that our constrained memory actually caused an eviction,
        # proving we exercised the evict_overflow_paused_requests path with spec tokens.
        assert (
            eviction_occurred
        ), "Test failed to trigger an eviction. The test environment memory wasn't tight enough."

        # Verify all requests successfully went back through the queue and finished cleanly.
        # We MUST check the merged records from the engine, because eviction checkpoints
        # the requests, leaving the original instances in env.requests permanently active.
        for request_id, entry in env.engine.requests.items():
            merged_req = entry.record.merge()
            assert (
                merged_req.status == Status.COMPLETED
            ), f"Request {request_id} failed to complete."
            assert (
                len(merged_req.generated_tokens) == 511
            ), f"Request {request_id} didn't generate expected tokens."

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_with_prefix_caching(self):
        """Test that speculative decoding works correctly when prefix caching is enabled.

        Two requests share the same prompt prefix. The second request should reuse
        cached KV blocks from the first and still generate correctly with spec decoding.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,  # Added manually below
            min_prompt_length=256,
            max_prompt_length=256,
            num_tokens_to_generate=4,
            num_speculative_tokens=2,
            enable_prefix_caching=True,  # Set at config level
            context_block_size_tokens=256,  # Ensure exact 1 block per prompt
            materialize_only_last_token_logits=False,
            model_provider="gpt",
            context_max_tokens=4096,
            context_max_requests=512,
            max_sequence_length=1024,
        )
        env = self._build_test_env(test_config)

        # Create two pairs of requests with identical shared prefixes.
        shared_prompt_a = torch.randint(
            0, test_config.vocab_size - 1, (256,), dtype=torch.int64, device='cuda'
        )
        shared_prompt_b = torch.randint(
            0, test_config.vocab_size - 1, (256,), dtype=torch.int64, device='cuda'
        )

        prompts = [shared_prompt_a, shared_prompt_a, shared_prompt_b, shared_prompt_b]

        for i, prompt in enumerate(prompts):
            # Using the clean public API guarantees correct hashing and dataclass creation
            env.engine.add_request(
                request_id=i,
                prompt=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=128, termination_id=99),
            )

        # First, run schedule_waiting_requests and ONE step to allocate the prefill blocks.
        # Req 0 and 2 will schedule immediately. Req 1 and 3 will defer because their hashes
        # are currently pending (being registered by 0 and 2).
        env.engine.schedule_waiting_requests()
        env.engine.step_modern()

        # After step 1, Req 0 and 2 have completely registered their cached blocks.
        # Now, schedule the deferred ones (Req 1 and 3). They will find the registered blocks!
        env.engine.schedule_waiting_requests()
        env.engine.step_modern()

        # 4 requests. 2 unique prefixes (1 block each).
        # Without sharing, we'd need 8 blocks + 1 dummy = 9 active_used.
        # With sharing, we need 2 shared blocks + 4 generation blocks + 1 dummy = 7 active_used.
        active_used = env.engine.context.block_allocator.get_active_used()
        assert (
            active_used <= 7
        ), f"Prefix caching failed, expected <= 7 active blocks but got {active_used}"

        while env.engine.has_unfinished_requests():
            env.engine.step_modern()

        # Context should be clean after all requests finish.
        assert env.engine.context.active_token_count == 0
        assert env.engine.context.total_request_count == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_chunked_prefill_and_prefix_caching(self):
        """End-to-end test combining speculative decoding, chunked prefill, and prefix caching.

        Verifies that all three features interact correctly:
        - Prefix caching shares KV blocks between requests with common prompts
        - Chunked prefill processes long prompts in chunks
        - Speculative decoding generates multiple tokens per step
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=512,
            max_prompt_length=512,
            num_tokens_to_generate=128,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,  # Set at config level
            context_block_size_tokens=256,
            model_provider="gpt",
            context_max_tokens=1536,  # Force chunking
            context_max_requests=48,
        )
        env = self._build_test_env(test_config)

        # Create identical prompts for all 4 requests
        shared_prompt = torch.randint(
            0, test_config.vocab_size - 1, (512,), dtype=torch.int64, device='cuda'
        )

        for i in range(4):
            env.engine.add_request(
                request_id=i,
                prompt=shared_prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=128, termination_id=99),
            )

        while env.engine.has_unfinished_requests():
            env.engine.step_modern()

        assert env.engine.context.active_token_count == 0
        assert env.engine.context.total_request_count == 0

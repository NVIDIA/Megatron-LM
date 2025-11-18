# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import math
import random
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from tqdm import tqdm
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.inference.contexts.dynamic_context import (
    ActiveRequestCountOverflowError,
    BlockOverflowError,
    DynamicInferenceContext,
    RequestOverflowError,
    TokenOverflowError,
    WarmupEngineMode,
)
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
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    check_mamba_sequence_packing_support,
    get_attr_wrapped_model,
    is_fa_min_version,
    is_te_min_version,
)
from tests.unit_tests.test_utilities import Utils


def skip_if_mamba_sequence_packing_not_available(model_provider: str):
    if model_provider == "mamba":
        sequence_packing_available, reason_for_no_sequence_packing = (
            check_mamba_sequence_packing_support()
        )
        if not sequence_packing_available:
            pytest.skip(reason_for_no_sequence_packing)


def set_rounder(value):
    """Utility function to set the DynamicInferenceContext rounder."""
    DynamicInferenceContext.ROUNDER = value  # For backwards compatibility
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


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

    context_active_buffer_size_gb: float = 0.1  # enough room for all tokens.
    context_block_size_tokens: int = 256
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
    cuda_graph_scope: str = "full_iteration"
    force_build_cuda_graphs: bool = False
    transformer_impl: str = "local"
    # If False, do not build cuda graphs in the tests, even if
    # num_cuda_graphs is set.
    # For tests concerning cuda-graph warmups, we set this to False
    # to avoid the overhead of building the graphs, which is not
    # relevant to the test. The tests only check if the required
    # context attributes are set correctly.
    suspend_resume_interval: Optional[int] = None

    fp8: bool = False

    def __post_init__(self):

        # Compute max_sequence_length.
        assert self.max_sequence_length is None
        assert self.num_tokens_to_generate is None or self.num_tokens_total is None
        if self.num_tokens_to_generate is not None:
            self.max_sequence_length = self.max_prompt_length + self.num_tokens_to_generate
        else:
            assert self.num_tokens_total is not None
            self.max_sequence_length = self.num_tokens_total


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
                    test_config.min_prompt_length, test_config.max_prompt_length + 1
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
        layer_type_list: Optional[List[str]],
        mamba_conv_states_shape: Optional[Tuple[int]] = None,
        mamba_ssm_states_shape: Optional[Tuple[int]] = None,
    ):
        """The inference context manages the KV cache and other inference state."""

        # Inference context.
        context = DynamicInferenceContext(
            params_dtype=transformer_config.params_dtype,
            num_layers=transformer_config.num_layers,
            kv_channels=transformer_config.kv_channels,
            num_attention_heads=transformer_config.num_query_groups,
            max_sequence_length=test_config.max_sequence_length,
            num_cuda_graphs=test_config.num_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=not test_config.model_provider == "mamba",
            active_buffer_size_gb=test_config.context_active_buffer_size_gb,
            block_size_tokens=test_config.context_block_size_tokens,
            max_tokens=test_config.context_max_tokens,
            tensor_model_parallel_size=transformer_config.tensor_model_parallel_size,
            layer_type_list=layer_type_list,
            mamba_conv_states_shape=mamba_conv_states_shape,
            mamba_ssm_states_shape=mamba_ssm_states_shape,
            materialize_only_last_token_logits=test_config.materialize_only_last_token_logits,
            use_flashinfer_fused_rope=None,  # default to using flash-infer if available
            # this is for compatibility with the LTS environment
            unified_memory_level=0,  # unit tests currently broken with UVM
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

            # GPT model.
            model = GPTModel(
                config=transformer_config,
                transformer_layer_spec=layer_spec,
                vocab_size=test_config.vocab_size,
                max_sequence_length=test_config.max_sequence_length,
                parallel_output=True,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
        elif test_config.model_provider == "mamba":
            # Transformer config.
            transformer_config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=3,  # 1 Mamba layer, 1 attention layer, 1 MLP layer
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
                pipeline_model_parallel_size=test_config.pipeline_model_parallel_size,
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
                cuda_graph_scope=test_config.cuda_graph_scope,
            )

            # Mamba model.
            model = MambaModel(
                config=transformer_config,
                mamba_stack_spec=mamba_stack_spec,
                vocab_size=test_config.vocab_size,
                max_sequence_length=test_config.max_sequence_length,
                parallel_output=True,
                hybrid_attention_ratio=0.3,
                hybrid_mlp_ratio=0.3,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
        else:
            raise ValueError(f"Invalid model provider {test_config.model_provider}")

        for param in model.parameters():
            param.data = param.data.to(transformer_config.params_dtype)

        model.eval()

        # Layer type list for hybrid models
        decoder = get_attr_wrapped_model(model, "decoder")
        layer_type_list = getattr(decoder, "layer_type_list", None)
        if test_config.model_provider == "mamba":
            mamba_states_shapes = decoder.mamba_state_shapes_per_request()
            if mamba_states_shapes is not None:
                (mamba_conv_states_shape, mamba_ssm_states_shape) = mamba_states_shapes
            else:
                # A `MambaBlock` can only not have a `MambaLayer` if using pipeline parallelism
                # and a particular pipeline stage was not assigned a `MambaLayer`.
                assert test_config.pipeline_model_parallel_size > 1
                mamba_conv_states_shape = None
                mamba_ssm_states_shape = None
        else:
            mamba_conv_states_shape = None
            mamba_ssm_states_shape = None

        # Inference config.
        inference_config = InferenceWrapperConfig(
            hidden_size=transformer_config.hidden_size,
            inference_batch_times_seqlen_threshold=400,
            fp32_residual_connection=False,
            params_dtype=transformer_config.params_dtype,
            fp8=transformer_config.fp8,
            padded_vocab_size=test_config.vocab_size,
        )

        # Inference context.
        inference_context = cls._build_inference_context(
            test_config=test_config,
            transformer_config=transformer_config,
            requests=requests,
            layer_type_list=layer_type_list,
            mamba_conv_states_shape=mamba_conv_states_shape,
            mamba_ssm_states_shape=mamba_ssm_states_shape,
        )

        # Inference model wrapper.
        inference_wrapped_model = GPTInferenceWrapper(model, inference_config, inference_context)

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
        engine = DynamicInferenceEngine(
            text_generation_controller,
            inference_context,
            random_seed=test_config.random_seed,
            enable_cuda_graph=transformer_config.cuda_graph_impl == "local",
        )

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
        result = env.engine.step_modern(verbose=False)

        # Suspend + resume.
        if (
            env.config.suspend_resume_interval is not None
            and env.engine.step_count % env.config.suspend_resume_interval == 0
        ):
            suspend_resume_mems = {}
            suspend_resume_mems["start"] = torch.cuda.memory_stats()
            env.engine.suspend()  # suspend.
            suspend_resume_mems["mid"] = torch.cuda.memory_stats()
            env.engine.resume()  # resume.
            suspend_resume_mems["end"] = torch.cuda.memory_stats()
            env.mem_usage["suspend_resume"][env.engine.step_count] = suspend_resume_mems

        # Nothing done?
        finished_request_records = result["finished_request_records"]
        if len(finished_request_records) == 0:
            return

        # Append output tokens.
        for finished_request_record in finished_request_records:
            finished_request = finished_request_record.merge(env.engine.controller.tokenizer)
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
    @pytest.mark.parametrize("num_cuda_graphs", [None, 1, 4])
    @pytest.mark.parametrize("cuda_graph_scope", ["full", "full_iteration"])
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
        )

        # Validate max_requests, max_tokens.
        assert env.engine.context.max_tokens == DynamicInferenceContext.DEFAULT_MAX_TOKENS

        # Validate generated tokens.
        gpt_expected_generated_tokens = [
            [69, 85, 55, 74, 56, 89, 64, 59, 55, 67, 15, 58, 6, 37, 54, 47],
            [29, 54, 33, 72, 45, 76, 41, 56, 28, 25, 17, 2, 61, 6, 98, 76],
            [35, 78, 54, 16, 79, 98, 22, 5, 60, 0, 1, 76, 77, 11, 25, 7],
            [25, 75, 57, 85, 81, 37, 88, 17, 71, 15, 70, 64, 50, 0, 64, 45],
            [32, 5, 85, 75, 30, 68, 23, 33, 20, 26, 89, 20, 92, 97, 38, 81],
            [33, 69, 32, 49, 93, 24, 33, 6, 97, 36, 37, 99],
            [82, 78, 78, 65, 22, 1, 87, 42, 36, 26, 27, 56, 82, 32, 8, 80],
            [],
        ]

        mamba_expected_generated_tokens = [
            [74, 72, 83, 59, 1, 70, 15, 89, 30, 52, 82, 70, 64, 16, 83, 5],
            [25, 54, 42, 57, 33, 64, 60, 13, 28, 74, 8, 4, 56, 68, 87, 82],
            [31, 55, 77, 25, 96, 13, 32, 49, 40, 54, 73, 10, 50, 2, 64, 96],
            [72, 80, 35, 72, 77, 85, 98, 36, 4, 97, 37, 46, 79, 95, 83, 85],
            [8, 80, 56, 4, 87, 1, 15, 98, 85, 7, 31, 38, 91, 28, 18, 80],
            [9, 94, 48, 60, 87, 57, 25, 76, 91, 34, 69, 86, 73, 24, 63, 97],
            [17, 5, 62, 66, 15, 52, 32, 75, 66, 18, 69, 5, 67, 37, 94, 51],
            [],
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
        active_buffer_size_gb = (block_size_bytes + 1) / 1024**3
        test_config = DynamicEngineTestConfig(
            context_active_buffer_size_gb=active_buffer_size_gb, model_provider=model_provider
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
            (64, [80, 72, 64, 56, 48, 40, 32, 24, 16, 8]),
            (1024, [80, 72, 64, 56, 48, 40, 32, 24, 16, 8]),
        ]:

            # Build cuda graphs (inside dynamic engine).
            env = self._build_test_env(
                DynamicEngineTestConfig(
                    context_active_buffer_size_gb=0.01, num_cuda_graphs=num_cuda_graphs
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
    @pytest.mark.parametrize(
        "warmup_engine_mode", [WarmupEngineMode.DECODE, WarmupEngineMode.NON_DECODE]
    )
    @pytest.mark.parametrize(
        "num_warmup_tokens, expected_cuda_graph_token_count",
        [
            (1, 8),
            (2, 8),
            (4, 8),
            (8, 8),
            (10, 16),
            (12, 16),
            (16, 16),
            (20, 24),
            (24, 24),
            (28, 32),
            (32, 32),
        ],
    )
    @torch.inference_mode()
    def test_cuda_graph_warmup(
        self,
        warmup_engine_mode: WarmupEngineMode,
        num_warmup_tokens: int,
        expected_cuda_graph_token_count: int,
    ) -> None:
        """Test initialization during cuda graph warmup."""
        if num_warmup_tokens == 1 and warmup_engine_mode == WarmupEngineMode.NON_DECODE:
            pytest.skip("WarmupEngineMode.NON_DECODE with num_warmup_tokens=1 is not supported.")

        # Initialize context.
        env = self._build_test_env(
            DynamicEngineTestConfig(
                context_active_buffer_size_gb=0.0041, num_cuda_graphs=8, num_tokens_to_generate=1
            )
        )

        context = env.engine.context
        assert context.is_decode_only()
        assert context.cuda_graph_token_counts == [
            32,
            24,
            16,
            8,
        ], "cuda_graph_token_counts: %s." % str(context.cuda_graph_token_counts)

        context.initialize_attention_state(
            num_warmup_tokens=num_warmup_tokens, warmup_engine_mode=warmup_engine_mode
        )

        # Validate request & token counts.

        assert (
            expected_cuda_graph_token_count
            == context.padded_active_request_count
            == context.padded_active_token_count
        ), (
            "failed ... num_warmup_tokens (%d) ... expected_cuda_graph_request_count (%d) == context.padded_active_request_count (%d) == context.padded_active_token_count (%d)"
            % (
                num_warmup_tokens,
                expected_cuda_graph_token_count,
                context.padded_active_request_count,
                context.padded_active_token_count,
            )
        )

        # Validate input/position dimensions.
        input_ids, pos_ids = context.current_input_and_position_ids()
        assert input_ids.shape[1] == pos_ids.shape[1] == expected_cuda_graph_token_count
        assert context.using_cuda_graph_this_step, (
            "expected `using_cuda_graph_this_step` to be True for decode step with "
            "num_warmup_tokens <= max_requests."
        )
        context.reset()

        # Test active request count overflow
        for num_warmup_tokens in (64, 128, 1024):
            try:
                context.initialize_attention_state(
                    num_warmup_tokens=num_warmup_tokens, warmup_engine_mode=warmup_engine_mode
                )
            except ActiveRequestCountOverflowError as e:
                continue
            raise Exception("`ActiveRequestCountOverflowError should have been raised.")

        context.reset()

        # test the case where the active token count exceeds max requests.
        # expectation: we should be in non-decode mode and not using cuda graphs

        # add all requests to the context.
        for request in tqdm(env.requests, "add requests"):
            env.engine._add_request(request)
        env.engine.schedule_waiting_requests()

        # we should now have more active tokens than max requests.
        context.initialize_attention_state()
        assert not context.is_decode_only()
        assert not context.using_cuda_graph_this_step(), (
            "expected `using_cuda_graph_this_step` to be False for non-decode step where "
            "the active token count exceeds max requests"
        )
        context.reset()

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
        finished_requests = [
            r.merge(env.engine.controller.tokenizer) for r in finished_request_records
        ]

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

            engine_task = asyncio.create_task(env.engine.run_engine(verbose=False))

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
                request = request_record.merge(env.engine.controller.tokenizer)
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
        """Verify that returning log probs does not raise any error."""
        # Returning log probs requires materializing the full prompt logits or
        # explicitly disabling prompt logits.
        with pytest.raises(AssertionError):
            env = self._run_test(return_log_probs=True, materialize_only_last_token_logits=True)
        env = self._run_test(return_log_probs=True, materialize_only_last_token_logits=False)
        env = self._run_test(
            return_log_probs=True,
            materialize_only_last_token_logits=True,
            skip_prompt_log_probs=True,
        )

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
        elif pp_size > 1 and model_provider == "mamba":
            pytest.skip(
                reason=(
                    "Running hybrid models with pp_size > 1 and no attention on some "
                    "pipeline stages is not supported yet."
                )
            )
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
            context_active_buffer_size_gb=0.001,  # 0.001, # 8 blocks
            context_max_tokens=8,
            num_gap_steps=1,
        )

        expected_event_types = [
            ['ADD', 'FINISH'],
            ['ADD', 'FINISH'],
            ['ADD', 'FINISH'],
            ['ADD', 'FINISH'],
            ['ERROR_TRANSIENT', 'ADD', 'FINISH'],
            ['ERROR_TRANSIENT', 'ADD', 'FINISH'],
            ['ADD', 'FINISH'],
            ['ERROR_NONTRANSIENT', 'FAIL'],
            ['ERROR_NONTRANSIENT', 'FAIL'],
            ['ERROR_TRANSIENT', 'ADD', 'FINISH'],
            ['ERROR_NONTRANSIENT', 'FAIL'],
            ['ERROR_TRANSIENT', 'ADD', 'FINISH'],
            ['ADD', 'FINISH'],
            ['ERROR_TRANSIENT', 'ADD', 'FINISH'],
            ['ERROR_TRANSIENT', 'ADD', 'FINISH'],
            ['ERROR_TRANSIENT', 'ADD', 'FINISH'],
        ]
        result_event_types = [[e.type.name for e in r.events] for r in env.requests]

        assert result_event_types == expected_event_types

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "mamba"])
    @torch.inference_mode()
    def test_chunked_prefill(self, model_provider: str):
        """Verify that chunked prefill output is equivalent to regular prefill."""
        skip_if_mamba_sequence_packing_not_available(model_provider)

        prompt_length = 1200
        num_tokens_to_generate = 16
        max_sequence_length = prompt_length + num_tokens_to_generate

        # Configure context to force chunking (chunked prefill is enabled by default)
        env = self._run_test(
            num_requests=1,
            min_prompt_length=prompt_length,
            max_prompt_length=prompt_length,
            num_tokens_to_generate=num_tokens_to_generate,
            materialize_only_last_token_logits=False,
            model_provider=model_provider,
            context_block_size_tokens=256,
            context_max_tokens=1000,
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.skip(
        reason="test works in isolation, but memory dynamics change when run "
        "within unt test suite."
    )
    def test_suspend_resume_memory(self):

        # Run tests.
        mem_usages = {}
        for suspend_resume_interval in None, 8, 4, 2:  # interval 1 acts funny.

            # Run test.
            env = self._run_test(suspend_resume_interval=suspend_resume_interval, num_gap_steps=1)

            # Record memory usage.
            mem_usages[suspend_resume_interval] = env.mem_usage

            # Clear memory to make recorded memories consistent between tests.
            # TODO(@lmcafee): why is memory not automatically cleared?
            # env.engine.suspend() # TODO(@lmcafee): useful?
            del env

        # Utility methods.
        get_alloc = lambda mem_stats: mem_stats["allocated_bytes.all.current"]

        # Validate overall 'end' memory usage.
        golden_end_bytes = get_alloc(mem_usages[None]["end"])
        for interval, mem_usage in mem_usages.items():
            current_end_bytes = get_alloc(mem_usage["end"])
            assert math.isclose(
                golden_end_bytes, current_end_bytes, rel_tol=0.01
            ), f"{current_end_bytes} != {golden_end_bytes}."

        # Validate 'suspend/resume' memory usage.
        get_suspend_resume_bytes = lambda key: list(
            get_alloc(list(d["suspend_resume"].values())[-1][key])
            for i, d in mem_usages.items()
            if i is not None
        )
        suspend_resume_mid_bytes = get_suspend_resume_bytes("mid")
        suspend_resume_end_bytes = get_suspend_resume_bytes("end")
        for mid_bytes in suspend_resume_mid_bytes:
            assert math.isclose(
                suspend_resume_mid_bytes[0], mid_bytes, rel_tol=0.01
            ), f"{mid_bytes} != {suspend_resume_mid_bytes[0]}."
        for end_bytes in suspend_resume_end_bytes:
            assert math.isclose(
                suspend_resume_end_bytes[0], end_bytes, rel_tol=0.01
            ), f"{end_bytes} != {suspend_resume_end_bytes[0]}."

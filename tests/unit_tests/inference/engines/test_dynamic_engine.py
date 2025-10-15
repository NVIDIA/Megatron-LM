# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import random
import types
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest
import torch
from tqdm import tqdm

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
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils


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

    context_buffer_size_gb: float = 0.1  # enough room for all tokens.
    context_block_size_tokens: int = 256
    context_buffer_guaranteed_fraction: float = 0.01
    context_buffer_overflow_factor: Optional[float] = None
    context_max_requests_override: Optional[int] = None
    context_max_tokens_override: Optional[int] = None
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    sequence_parallel: bool = False

    use_fixed_output_lengths: bool = False
    num_cuda_graphs: int = None
    return_log_probs: bool = False
    materialize_only_last_token_logits: bool = True
    skip_prompt_log_probs_for_dynamic_inference: bool = False
    cuda_graph_scope: List[str] = None
    force_build_cuda_graphs: bool = False
    # If False, do not build cuda graphs in the tests, even if
    # num_cuda_graphs is set.
    # For tests concerning cuda-graph warmups, we set this to False
    # to avoid the overhead of building the graphs, which is not
    # relevant to the test. The tests only check if the required
    # context attributes are set correctly.

    def __post_init__(self):

        # Compute max_sequence_length.
        assert self.max_sequence_length is None
        assert self.num_tokens_to_generate is None or self.num_tokens_total is None
        if self.num_tokens_to_generate is not None:
            self.max_sequence_length = self.max_prompt_length + self.num_tokens_to_generate
        else:
            assert self.num_tokens_total is not None
            self.max_sequence_length = self.num_tokens_total

        # Update overrides if not using overflow factor.
        if self.context_buffer_overflow_factor is None:

            # Enough room for all requests.
            if self.context_max_requests_override is None:
                self.context_max_requests_override = self.num_requests

            # Enough room for all tokens.
            if self.context_max_tokens_override is None:
                self.context_max_tokens_override = self.num_requests * self.max_sequence_length

        if self.cuda_graph_scope is None:
            self.cuda_graph_scope = ["full_iteration"]


@dataclass
class DynamicEngineTestEnv:
    """Test environment, including requests and engine."""

    config: DynamicEngineTestConfig
    requests: List[DynamicInferenceRequest]
    engine: DynamicInferenceEngine


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
                return_log_probs=test_config.return_log_probs,
            )
            if not hasattr(sampling_params, "num_tokens_total"):
                # Remove this if statement branch in megatron-core 0.16
                sampling_params.add_attributes({"num_tokens_total": num_tokens_total})
            else:
                sampling_params.num_tokens_total = num_tokens_total
            sampling_params.add_attributes(
                {
                    "skip_prompt_log_probs_for_dynamic_inference": test_config.skip_prompt_log_probs_for_dynamic_inference
                }
            )

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
            buffer_size_gb=test_config.context_buffer_size_gb,
            buffer_guaranteed_fraction=test_config.context_buffer_guaranteed_fraction,
            block_size_tokens=test_config.context_block_size_tokens,
            buffer_overflow_factor=test_config.context_buffer_overflow_factor,
            max_requests_override=test_config.context_max_requests_override,
            max_tokens_override=test_config.context_max_tokens_override,
            tensor_model_parallel_size=transformer_config.tensor_model_parallel_size,
            materialize_only_last_token_logits=test_config.materialize_only_last_token_logits,
            use_flashinfer_fused_rope=None,  # default to using flash-infer if available
            # this is for compatibility with the LTS environment
        )

        return context

    @classmethod
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

        # Transformer config.
        transformer_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=4,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl=(
                "local"
                if test_config.num_cuda_graphs is not None and test_config.force_build_cuda_graphs
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
            inference_sampling_seed=test_config.random_seed,
            cuda_graph_scope=test_config.cuda_graph_scope,
        )

        # Requests.
        requests = cls._build_requests(test_config)

        # GPT model.
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=test_config.vocab_size,
            max_sequence_length=test_config.max_sequence_length,
            parallel_output=True,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()

        for param in model.parameters():
            param.data = param.data.to(transformer_config.params_dtype)

        model.eval()

        # Inference config.
        inference_config = InferenceWrapperConfig(
            hidden_size=transformer_config.hidden_size,
            inference_batch_times_seqlen_threshold=400,
            fp32_residual_connection=False,
            params_dtype=transformer_config.params_dtype,
            padded_vocab_size=test_config.vocab_size,
        )

        # Inference context.
        inference_context = cls._build_inference_context(
            test_config=test_config, transformer_config=transformer_config, requests=requests
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
            tokenizer=types.SimpleNamespace(vocab_size=test_config.vocab_size),
        )

        # Reset global cuda graph state.
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

        # Inference engine.
        engine = DynamicInferenceEngine(
            text_generation_controller,
            inference_context,
            termination_id=(
                -1 if test_config.use_fixed_output_lengths else test_config.vocab_size - 1
            ),
            random_seed=test_config.random_seed,
            enable_cuda_graph=transformer_config.cuda_graph_impl == "local",
        )

        # Test env.
        env = DynamicEngineTestEnv(config=test_config, requests=requests, engine=engine)

        # Mock the detokenize method to return predictable result
        def mock_detokenize_prompt(tokens):
            return "tokenized_prompt"

        env.engine.controller.tokenizer.detokenize = mock_detokenize_prompt

        return env

    @classmethod
    def _run_step(cls, env):
        set_rounder(4)
        # Step inference engine (i.e., generate one token per request).
        # It's safe to use request 0's sampling params here because
        # the only thing that differs between requests is num_tokens_to_generate,
        # and engine.async_step() doesn't use this sampling param's
        # num_tokens_to_generate.
        result = env.engine.step_modern(env.requests[0].sampling_params, verbose=False)
        finished_requests = result["finished_requests"]

    @classmethod
    def _run_test(cls, **test_config_kwargs):

        # Test environment.
        test_config = DynamicEngineTestConfig(**test_config_kwargs)
        env = cls._build_test_env(test_config)

        # Add requests to engine.
        for request in tqdm(env.requests, "add requests"):

            # Add request.
            env.engine._add_request(request)

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
            assert (
                (num_tokens_to_generate is None and num_tokens_total is None)
                or len(request.generated_tokens) == num_tokens_expected
                or request.status == Status.FAILED
            ), (
                f"Request {request.request_id} expected to generate {num_tokens_to_generate} "
                f"tokens but generated {len(request.generated_tokens)}"
            )

        return env

    def teardown_method(self, method):
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("num_cuda_graphs", [None, 1, 4])
    @pytest.mark.parametrize("cuda_graph_scope", [[], ["full_iteration"]])
    def test_simple(self, num_cuda_graphs, cuda_graph_scope) -> None:
        """Simple test that runs without errors, and validates output."""

        # Run test.
        env = self._run_test(
            num_cuda_graphs=num_cuda_graphs,
            context_max_requests_override=32,
            cuda_graph_scope=cuda_graph_scope,
            force_build_cuda_graphs=True,
        )

        # Validate max_requests, max_tokens.
        assert env.engine.context.max_requests == 32
        assert env.engine.context.max_tokens == 160

        # Validate generated tokens.
        expected_generated_tokens_list = [
            [69, 85, 55, 74],
            [29, 54, 85, 89],
            [33, 30, 64, 59],
            [45, 76, 33, 67],
            [41, 56, 15, 58],
            [28, 17, 6, 37],
            [17, 2, 54, 47],
            [],  # this request is failed due to max sequence length overflow
        ]

        assert len(env.requests) == len(expected_generated_tokens_list)
        for request, expected_generated_tokens in zip(env.requests, expected_generated_tokens_list):
            assert request.generated_tokens == expected_generated_tokens, (
                f"request {request.request_id}, "
                f"result ({request.generated_tokens}) != "
                f"expected ({expected_generated_tokens})."
            )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_overflow_factor(self) -> None:
        """Test overflow factor arg."""
        # Run test.
        env = self._run_test(
            context_buffer_overflow_factor=0.1,
            context_max_requests_override=None,
            context_max_tokens_override=None,
        )

        # Validate max_requests, max_tokens.
        assert env.engine.context.max_requests == 420
        assert env.engine.context.max_tokens == 420

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_request_overflow(self) -> None:
        """Test request overflow."""
        self._run_test(context_max_requests_override=4)

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_token_overflow_transient(self) -> None:
        """Test token overflow."""
        test_config = DynamicEngineTestConfig(
            num_requests=2, min_prompt_length=8, max_prompt_length=8, context_max_tokens_override=12
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
        test_config = DynamicEngineTestConfig(context_max_tokens_override=8)
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
    def test_block_overflow(self) -> None:
        """Test block overflow."""
        env = self._build_test_env(DynamicEngineTestConfig())
        context = env.engine.context
        block_size_bytes = context.block_size_bytes
        buffer_size_gb = (block_size_bytes + 1) / 1024**3
        test_config = DynamicEngineTestConfig(context_buffer_size_gb=buffer_size_gb)
        env = self._build_test_env(test_config)
        env.engine._add_request(env.requests[0])
        assert list(env.engine.waiting_request_ids) == [0]

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_multi_add(self) -> None:
        """Test adding multiple requests simultaneously."""
        self._run_test(num_gap_steps=0)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_fixed_output_lengths(self) -> None:
        """Test generating a fixed number of output tokens."""
        self._run_test(use_fixed_output_lengths=True)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_cuda_graph_token_counts(self) -> None:
        """Test initialization of `cuda_graph_token_counts` in dynamic context."""

        # Test num_cuda_graphs.
        for num_cuda_graphs, expected_cuda_graph_token_counts in [
            (0, [64]),
            (1, [64]),
            (2, [64, 32]),
            (4, [64, 48, 32, 16]),
            (8, [64, 56, 48, 40, 32, 24, 16, 8]),
            (16, [64, 56, 48, 40, 32, 24, 16, 8]),
            (64, [64, 56, 48, 40, 32, 24, 16, 8]),
            (1024, [64, 56, 48, 40, 32, 24, 16, 8]),
        ]:

            # Build cuda graphs (inside dynamic engine).
            env = self._build_test_env(
                DynamicEngineTestConfig(num_requests=64, num_cuda_graphs=num_cuda_graphs)
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
            DynamicEngineTestConfig(num_requests=32, num_cuda_graphs=8, num_tokens_to_generate=1)
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
    def test_generate_function(self) -> None:
        """Test the generate function that processes multiple prompts at once."""
        # Set up test environment
        test_config = DynamicEngineTestConfig(
            num_requests=4, max_prompt_length=8, num_tokens_to_generate=4
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
        finished_requests = env.engine.generate(prompts, env.requests[0].sampling_params)

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
        # Test environment.
        test_config = DynamicEngineTestConfig(use_fixed_output_lengths=True)
        env = self._build_test_env(test_config)

        # It's safe to use request 0's sampling params here because all sampling
        # params are identical as long as use_fixed_output_lengths == False.
        engine_task = asyncio.create_task(
            env.engine.run_engine(sampling_params=env.requests[0].sampling_params, verbose=False)
        )

        request_completion_futures: Dict[int, asyncio.Future[DynamicInferenceRequest]] = {}

        # Add requests to engine.
        for request in tqdm(env.requests, "add requests"):
            request_completion_futures[request.request_id] = env.engine._add_request(request)

        # Wait for all requests to complete.
        await asyncio.gather(*request_completion_futures.values())

        # Verify that all request outputs were set.
        for request_id, fut in request_completion_futures.items():
            num_tokens_to_generate = env.requests[request_id].sampling_params.num_tokens_to_generate
            result = fut.result()
            assert result.generated_length == num_tokens_to_generate, (
                f"Request {request_id} expected to generate {num_tokens_to_generate} "
                f"tokens but generated {result.generated_length}"
            )

        engine_task.cancel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
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
            skip_prompt_log_probs_for_dynamic_inference=True,
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
    def test_parallel_inference(
        self, tp_size, pp_size, ep_size, sequence_parallel, materialize_only_last_token_logits
    ):
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
        env = self._run_test(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=sequence_parallel,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
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
            context_max_requests_override=8,
            context_max_tokens_override=8,
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


if __name__ == "__main__":
    test = TestDynamicInferenceEngine()
    test.test_simple(4)
    test.test_overflow_factor()
    test.test_request_overflow()
    test.test_token_overflow_transient()
    # test.test_token_overflow_nontransient() # uncomment in megatron-core 0.16
    test.test_block_overflow()
    test.test_multi_add()
    test.test_fixed_output_lengths()
    test.test_cuda_graph_request_counts()
    test.test_cuda_graph_warmup(WarmupEngineMode.DECODE, 1, 8)
    test.test_generate_function()
    asyncio.run(test.test_run_engine())
    test.test_return_log_probs()
    test.test_parallel_inference()
    # test.test_events() # uncomment in megatron-core 0.16
    test.teardown_method(None)
    print("~~~")
    print("success.")

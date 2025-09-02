# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import random
import types
from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch
from tqdm import tqdm

from megatron.core import parallel_state
from megatron.core.inference.contexts.dynamic_context import (
    ActiveRequestCountOverflowError,
    ChunkOverflowError,
    DynamicInferenceContext,
    RequestOverflowError,
    TokenOverflowError,
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


class Request:
    """Simple class to hold prompt tokens and output tokens."""

    def __init__(self, prompt: List[int], num_tokens_to_generate: Optional[int] = None):
        self.prompt = prompt
        self.num_tokens_to_generate = num_tokens_to_generate
        self.output = []
        self.state = "queued"

    def __str__(self) -> str:
        return "[%s]; prompt len %d; output len %d" % (
            self.state,
            len(self.prompt),
            len(self.output),
        )


@dataclass
class DynamicEngineTestConfig:
    """Test configuration args."""

    set_rounder(4)
    num_requests: int = 2 * DynamicInferenceContext.round_up_requests(1, 1)
    min_prompt_length: int = 4
    max_prompt_length: int = 16
    max_output_length: int = 4
    max_sequence_length: Optional[int] = None

    num_gap_steps: int = 2

    context_buffer_size_gb: float = 0.1  # enough room for all tokens.
    context_chunk_size_tokens: int = 256
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

    def __post_init__(self):

        # Compute max_sequence_length.
        assert self.max_sequence_length is None
        self.max_sequence_length = self.max_prompt_length + self.max_output_length

        # Update overrides if not using overflow factor.
        if self.context_buffer_overflow_factor is None:

            # Enough room for all requests.
            if self.context_max_requests_override is None:
                self.context_max_requests_override = self.num_requests

            # Enough room for all tokens.
            if self.context_max_tokens_override is None:
                self.context_max_tokens_override = self.num_requests * self.max_sequence_length


@dataclass
class DynamicEngineTestEnv:
    """Test environment, including requests and engine."""

    config: DynamicEngineTestConfig
    sampling_params: SamplingParams
    requests: List[Request]
    engine: DynamicInferenceEngine


class TestDynamicInferenceEngine:

    @classmethod
    def _build_requests(
        cls,
        num_requests: int,
        min_prompt_length: int,
        max_prompt_length: int,
        max_sequence_length: int,
        vocab_size: int,
        use_fixed_output_lengths: bool = False,
    ) -> List[Request]:
        prompt_lengths = torch.randint(
            min_prompt_length, max_prompt_length + 1, (num_requests,)
        ).tolist()
        num_tokens_to_generate: List[Optional[int]]
        if use_fixed_output_lengths:
            num_tokens_to_generate = [
                random.randint(1, max_sequence_length - p) for p in prompt_lengths
            ]
        else:
            num_tokens_to_generate = [None for _ in range(num_requests)]
        prompts = [
            torch.randint(0, vocab_size - 1, (length,)).tolist() for length in prompt_lengths
        ]
        requests = [
            Request(prompt=p, num_tokens_to_generate=n)
            for (p, n) in zip(prompts, num_tokens_to_generate)
        ]
        return requests

    @classmethod
    def _build_inference_context(
        cls,
        test_config: DynamicEngineTestConfig,
        transformer_config: TransformerConfig,
        requests: List[Request],
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
            chunk_size_tokens=test_config.context_chunk_size_tokens,
            buffer_overflow_factor=test_config.context_buffer_overflow_factor,
            max_requests_override=test_config.context_max_requests_override,
            max_tokens_override=test_config.context_max_tokens_override,
            tensor_model_parallel_size=transformer_config.tensor_model_parallel_size,
            materialize_only_last_token_logits=test_config.materialize_only_last_token_logits,
        )

        return context

    @classmethod
    def _build_test_env(cls, test_config):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=test_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=test_config.pipeline_model_parallel_size,
        )

        set_rounder(4)

        random_seed = 123
        vocab_size = 100

        # Random state.
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        model_parallel_cuda_manual_seed(
            seed=random_seed,
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
            enable_cuda_graph=test_config.num_cuda_graphs is not None,
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
        )

        # Requests.
        requests = cls._build_requests(
            num_requests=test_config.num_requests,
            min_prompt_length=test_config.min_prompt_length,
            max_prompt_length=test_config.max_prompt_length,
            max_sequence_length=test_config.max_sequence_length,
            vocab_size=vocab_size,
            use_fixed_output_lengths=test_config.use_fixed_output_lengths,
        )

        # Sampling params.
        sampling_params = SamplingParams(
            num_tokens_to_generate=test_config.max_output_length,
            return_log_probs=test_config.return_log_probs,
        )
        sampling_params.add_attributes(
            {
                "skip_prompt_log_probs_for_dynamic_inference": test_config.skip_prompt_log_probs_for_dynamic_inference
            }
        )

        # GPT model.
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=vocab_size,
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
            padded_vocab_size=vocab_size,
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
            tokenizer=types.SimpleNamespace(vocab_size=vocab_size),
        )

        # Reset global cuda graph state.
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

        # Inference engine.
        engine = DynamicInferenceEngine(
            text_generation_controller,
            inference_context,
            termination_id=vocab_size - 1,
            random_seed=random_seed,
            enable_cuda_graph=False,
        )

        # Test env.
        env = DynamicEngineTestEnv(
            config=test_config, sampling_params=sampling_params, requests=requests, engine=engine
        )

        # Mock the detokenize method to return predictable result
        def mock_detokenize_prompt(tokens):
            return "tokenized_prompt"

        env.engine.controller.tokenizer.detokenize = mock_detokenize_prompt

        return env

    @classmethod
    def _run_step(cls, env):
        set_rounder(4)
        # Step inference engine (i.e., generate one token per request).
        active_requests, finished_requests, step_time = env.engine.step(
            env.sampling_params, verbose=False
        )

        # Nothing done?
        if len(finished_requests) == 0:
            return

        # Append output tokens.
        for finished_request in finished_requests:
            request = env.requests[finished_request.request_id]
            request.output = finished_request.generated_tokens
            request.state = "finished"

    @classmethod
    def _run_test(cls, **test_config_kwargs):

        # Test environment.
        test_config = DynamicEngineTestConfig(**test_config_kwargs)
        env = cls._build_test_env(test_config)

        # Add requests to engine.
        for request_id in tqdm(range(len(env.requests)), "add requests"):

            # Add request.
            num_tokens_to_generate = env.requests[request_id].num_tokens_to_generate
            env.engine.add_request(
                request_id,
                env.requests[request_id].prompt,
                num_tokens_to_generate=num_tokens_to_generate,
            )
            env.requests[request_id].state = "pending"

            # Insert gap steps between adding requests.
            for _ in range(test_config.num_gap_steps):
                cls._run_step(env)

        # Step engine until finished.
        while True:
            cls._run_step(env)
            if not env.engine.has_unfinished_requests():
                break

        # Validate all requests finished.
        for request_id, request in enumerate(env.requests):
            assert request.state == "finished", f"request.state == '{request.state}'."

            num_tokens_to_generate = env.requests[request_id].num_tokens_to_generate
            assert (
                num_tokens_to_generate is None or len(request.output) == num_tokens_to_generate
            ), (
                f"Request {request_id} expected to generate {num_tokens_to_generate} "
                f"tokens but generated {len(request.output)}"
            )

        return env

    def teardown_method(self, method):
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.experimental
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_simple(self) -> None:
        """Simple test that runs without errors, and validates output."""

        # Run test.
        env = self._run_test()

        # Validate max_requests, max_tokens.
        assert env.engine.context.max_requests == 8
        assert env.engine.context.max_tokens == 160

        # Validate output tokens.
        expected_outputs = [
            [69, 85, 55, 74, 85, 89, 64, 59, 55, 67],
            [29, 54, 33, 30, 45, 76, 41, 56, 28, 25, 94, 2, 61, 6, 98],
            [35, 78, 54, 32, 79, 98, 22, 5, 60],
            [25, 75, 57, 85, 81],
            [32, 5, 15, 58, 6, 37, 54, 47, 22, 1, 87, 42, 36, 26, 27, 56],
            [85, 51, 88, 62, 71],
            [30, 0, 1, 76, 77, 11, 25],
            [23, 15, 70, 76, 97, 36, 37, 99],
        ]

        assert len(env.requests) == len(expected_outputs)
        for request, expected_output in zip(env.requests, expected_outputs):
            assert request.output == expected_output

    @pytest.mark.experimental
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

    @pytest.mark.experimental
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_request_overflow(self) -> None:
        """Test request overflow."""
        self._run_test(context_max_requests_override=1)

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_token_overflow_transient(self) -> None:
        """Test token overflow (transient)."""
        test_config = DynamicEngineTestConfig(
            min_prompt_length=6,
            max_prompt_length=6,
            max_output_length=2,
            context_max_tokens_override=8,
        )
        env = self._build_test_env(test_config)
        for request_id, request in enumerate(env.requests):
            env.engine.add_request(request_id, request.prompt, request.num_tokens_to_generate)
        assert list(env.engine.waiting_request_ids) == [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ], f"waiting_request_ids: {list(env.engine.waiting_request_ids)}."

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.skip(
        reason="activate for `megatron-core >= 0.15`, after fixing "
        "`raise TokenOverflowError(is_transient=False)` compatibility with "
        "legacy tests."
    )
    def test_token_overflow_nontransient(self) -> None:
        """Test token overflow (non-transient)."""
        test_config = DynamicEngineTestConfig(context_max_tokens_override=8)
        env = self._build_test_env(test_config)
        try:
            env.engine.add_request(
                0, env.requests[0].prompt, env.requests[0].num_tokens_to_generate
            )
        except TokenOverflowError as e:
            assert e.is_transient == False
        else:
            raise Exception("should have raised TokenOverflowError(is_transient=False).")

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_chunk_overflow(self) -> None:
        """Test token overflow."""
        env = self._build_test_env(DynamicEngineTestConfig())
        context = env.engine.context
        chunk_size_bytes = context.chunk_size_bytes
        buffer_size_gb = (chunk_size_bytes + 1) / 1024**3
        test_config = DynamicEngineTestConfig(context_buffer_size_gb=buffer_size_gb)
        env = self._build_test_env(test_config)
        env.engine.add_request(0, env.requests[0].prompt, env.requests[0].num_tokens_to_generate)
        assert list(env.engine.waiting_request_ids) == [0]

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_multi_add(self) -> None:
        """Test adding multiple requests simultaneously."""
        self._run_test(num_gap_steps=0)

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_fixed_output_lengths(self) -> None:
        """Test generating a fixed number of output tokens."""
        self._run_test(use_fixed_output_lengths=True)

    def test_cuda_graph_request_counts(self) -> None:
        """Test initialization of `cuda_graph_request_counts` in dynamic context."""

        # Test num_cuda_graphs.
        for num_cuda_graphs, expected_cuda_graph_request_counts in [
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
            actual_cuda_graph_request_counts = env.engine.context.cuda_graph_request_counts
            assert (
                actual_cuda_graph_request_counts == expected_cuda_graph_request_counts
            ), "num_cuda_graphs %d ... cuda_graph_request_counts: expected %s, found %s." % (
                num_cuda_graphs,
                expected_cuda_graph_request_counts,
                actual_cuda_graph_request_counts,
            )

    def test_cuda_graph_warmup(self) -> None:
        """Test initialization during cuda graph warmup."""

        # Initialize context.
        env = self._build_test_env(DynamicEngineTestConfig(num_requests=32, num_cuda_graphs=8))

        context = env.engine.context
        assert context.is_decode_only()
        assert context.cuda_graph_request_counts == [
            32,
            24,
            16,
            8,
        ], "cuda_graph_request_counts: %s." % str(context.cuda_graph_request_counts)

        # Iterate request counts.
        for num_warmup_requests, expected_cuda_graph_request_count in [
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
        ]:

            # Initialize attention state.
            context.initialize_attention_state(num_warmup_requests=num_warmup_requests)

            # Validate request & token counts.
            assert (
                expected_cuda_graph_request_count
                == context.padded_active_request_count
                == context.padded_active_token_count
            ), (
                "failed ... num_warmup_requests (%d) ... expected_cuda_graph_request_count (%d) == context.padded_active_request_count (%d) == context.padded_active_token_count (%d)"
                % (
                    num_warmup_requests,
                    expected_cuda_graph_request_count,
                    context.padded_active_request_count,
                    context.padded_active_token_count,
                )
            )

            # Validate input/position dimensions.
            input_ids, pos_ids = context.current_input_and_position_ids()
            assert input_ids.shape[1] == pos_ids.shape[1] == expected_cuda_graph_request_count

        # Test active request count overflow.
        for num_warmup_requests in (64, 128, 1024):
            try:
                context.initialize_attention_state(num_warmup_requests=num_warmup_requests)
            except ActiveRequestCountOverflowError as e:
                continue
            raise Exception("`ActiveRequestCountOverflowError should have been raised.")

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_generate_function(self) -> None:
        """Test the generate function that processes multiple prompts at once."""
        # Set up test environment
        test_config = DynamicEngineTestConfig(
            num_requests=4, max_prompt_length=8, max_output_length=4
        )
        env = self._build_test_env(test_config)

        # Create string prompts (just mock strings, since the test environment mocks the tokenizer)
        prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]

        # Mock the tokenize_prompt method to return predictable token sequences
        def mock_tokenize_prompt(prompt):
            # Return a token sequence based on the prompt number
            prompt_num = int(prompt[-1])
            return [10 + i for i in range(prompt_num + 2)]

        env.engine.controller.tokenize_prompt = mock_tokenize_prompt

        # Call the generate function
        finished_requests = env.engine.generate(prompts, env.sampling_params)

        # Verify results
        assert len(finished_requests) == len(
            prompts
        ), "Should return same number of finished requests as prompts"
        print()
        # Check each request was processed
        for i, request in enumerate(finished_requests):
            # Verify each request has generated tokens
            assert len(request.generated_tokens) > 0, f"Request {i} should have generated tokens"
            assert request.status == Status.COMPLETED, f"Request {i} should be completed"

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

        engine_task = asyncio.create_task(
            env.engine.run_engine(sampling_params=env.sampling_params, verbose=False)
        )

        request_completion_futures: Dict[int, asyncio.Future[DynamicInferenceRequest]] = {}

        # Add requests to engine.
        for request_id in tqdm(range(len(env.requests)), "add requests"):

            # Add request.
            num_tokens_to_generate = env.requests[request_id].num_tokens_to_generate
            request_completion_futures[request_id] = env.engine.add_request(
                request_id,
                env.requests[request_id].prompt,
                num_tokens_to_generate=num_tokens_to_generate,
            )
            env.requests[request_id].state = "pending"

        # Wait for all requests to complete.
        await asyncio.gather(*request_completion_futures.values())

        # Verify that all request outputs were set.
        for request_id, fut in request_completion_futures.items():
            num_tokens_to_generate = env.requests[request_id].num_tokens_to_generate
            result = fut.result()
            assert result.generated_length == num_tokens_to_generate, (
                f"Request {request_id} expected to generate {num_tokens_to_generate} "
                f"tokens but generated {result.generated_length}"
            )

        engine_task.cancel()

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


if __name__ == "__main__":
    test = TestDynamicInferenceEngine()
    test.test_simple()
    test.test_overflow_factor()
    test.test_request_overflow()
    test.test_token_overflow_transient()
    test.test_token_overflow_nontransient()
    test.test_chunk_overflow()
    test.test_multi_add()
    test.test_fixed_output_lengths()
    test.test_cuda_graph_request_counts()
    test.test_cuda_graph_warmup()
    test.test_generate_function()
    asyncio.run(test.test_run_engine())
    test.test_return_log_probs()
    test.teardown_method(None)
    print("~~~")
    print("success.")

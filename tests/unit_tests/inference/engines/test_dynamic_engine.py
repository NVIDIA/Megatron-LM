# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import random
import types
from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch
from tqdm import tqdm

from megatron.core import parallel_state
from megatron.core.device_utils import get_current_device
from megatron.core.inference.contexts.dynamic_context import (
    ChunkOverflowError,
    DynamicInferenceContext,
    RequestOverflowError,
    TokenOverflowError,
)
from megatron.core.inference.engines import DynamicInferenceEngine
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
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils


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
class TestConfig:
    """Test configuration args."""

    DynamicInferenceContext.ROUNDER = 4
    num_requests: int = 2 * DynamicInferenceContext.round_up(1)
    max_prompt_length: int = 16
    max_output_length: int = 4
    num_gap_steps: int = 2

    context_buffer_size_gb: float = 0.1  # enough room for all tokens.
    context_chunk_size_tokens: int = 256
    context_buffer_guaranteed_fraction: float = 0.01
    context_buffer_overflow_factor: Optional[float] = None
    context_max_requests_override: Optional[int] = None
    context_max_tokens_override: Optional[int] = None

    use_fixed_output_lengths: bool = False

    def __post_init__(self):

        # Update overrides if not using overflow factor.
        if self.context_buffer_overflow_factor is None:

            # Enough room for all requests.
            if self.context_max_requests_override is None:
                self.context_max_requests_override = self.num_requests

            # Enough room for all tokens.
            if self.context_max_tokens_override is None:
                self.context_max_tokens_override = self.num_requests * (
                    self.max_prompt_length + self.max_output_length
                )


@dataclass
class TestEnv:
    """Test environment, including requests and engine."""

    config: TestConfig
    sampling_params: SamplingParams
    requests: List[Request]
    engine: DynamicInferenceEngine

@pytest.mark.skip("not supported")
class TestDynamicInferenceEngine:

    @classmethod
    def _build_requests(
        cls,
        num_requests: int,
        max_prompt_length: int,
        max_sequence_length: int,
        vocab_size: int,
        use_fixed_output_lengths: bool = False,
    ) -> List[Request]:
        prompt_lengths = torch.randint(4, max_prompt_length, (num_requests,)).tolist()
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
        cls, test_config: TestConfig, transformer_config: TransformerConfig, requests: List[Request]
    ):
        """The inference context manages the KV cache and other inference state."""

        # Max sequence length.
        max_prompt_length = max(len(r.prompt) for r in requests)
        max_sequence_length = test_config.max_prompt_length + test_config.max_output_length

        # Inference context.
        context = DynamicInferenceContext(
            params_dtype=transformer_config.params_dtype,
            num_layers=transformer_config.num_layers,
            kv_channels=transformer_config.kv_channels,
            num_attention_heads=transformer_config.num_query_groups,
            max_sequence_length=max_sequence_length,
            buffer_size_gb=test_config.context_buffer_size_gb,
            buffer_guaranteed_fraction=test_config.context_buffer_guaranteed_fraction,
            chunk_size_tokens=test_config.context_chunk_size_tokens,
            buffer_overflow_factor=test_config.context_buffer_overflow_factor,
            max_requests_override=test_config.context_max_requests_override,
            max_tokens_override=test_config.context_max_tokens_override,
        )

        return context

    @classmethod
    def _build_test_env(cls, test_config):
        DynamicInferenceContext.ROUNDER = 4
        random_seed = 123
        vocab_size = 100

        # Random state.
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        model_parallel_device_manual_seed(random_seed)

        # Transformer config.
        transformer_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=4,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        # Requests.
        requests = cls._build_requests(
            num_requests=test_config.num_requests,
            max_prompt_length=test_config.max_prompt_length,
            max_sequence_length=test_config.max_prompt_length + test_config.max_output_length,
            vocab_size=vocab_size,
            use_fixed_output_lengths=test_config.use_fixed_output_lengths,
        )

        # Sampling params.
        sampling_params = SamplingParams(num_tokens_to_generate=test_config.max_output_length)

        # GPT model.
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=vocab_size,
            max_sequence_length=test_config.max_prompt_length + test_config.max_output_length,
            parallel_output=True,
        ).to(device=get_current_device())

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

        # Inference engine.
        engine = DynamicInferenceEngine(
            text_generation_controller,
            inference_context,
            termination_id=vocab_size - 1,
            random_seed=random_seed,
            enable_cuda_graph=False,
        )

        # Test env.
        env = TestEnv(
            config=test_config, sampling_params=sampling_params, requests=requests, engine=engine
        )

        return env

    @classmethod
    def _run_step(cls, env):
        DynamicInferenceContext.ROUNDER = 4
        # Step inference engine (i.e., generate one token per request).
        result, step_time = env.engine.step(env.sampling_params, verbose=False)

        # Nothing done?
        if result is None:
            return

        # Append output tokens.
        request_ids, finished_request_ids, sample = result
        request_ids = request_ids.tolist()
        sample = sample.tolist()
        for request_id, token in zip(request_ids, sample):
            request = env.requests[request_id]
            request.output.append(token)
            if request_id in finished_request_ids:
                request.state = "finished"

    @classmethod
    def _run_test(cls, **test_config_kwargs):

        # Test environment.
        test_config = TestConfig(**test_config_kwargs)
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

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

    def teardown_method(self, method):
        DynamicInferenceContext.ROUNDER = 64
        Utils.destroy_model_parallel()

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
            [69, 85, 55, 74, 85, 78],
            [29, 16, 33, 30, 45, 76, 41, 56, 28, 17, 17, 2, 61, 6, 20],
            [35, 78, 64, 59, 33, 67, 15, 58, 6, 49],
            [54, 16, 79, 98, 22, 5, 60, 0, 1, 24],
            [57, 85, 81, 37, 88, 17, 71, 15, 70, 64, 50, 0],
            [85, 75, 30, 68, 23, 33, 20, 76, 69, 36, 37, 99],
            [32, 49, 54, 47, 22, 1, 87, 30, 36, 97],
            [93, 24, 77, 11, 25, 7, 92, 97, 27, 56, 82],
        ]

        assert len(env.requests) == len(expected_outputs)
        for request, expected_output in zip(env.requests, expected_outputs):
            assert request.output == expected_output

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
        assert env.engine.context.max_requests == 1120
        assert env.engine.context.max_tokens == 1120

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_request_overflow(self) -> None:
        """Test request overflow."""
        try:
            env = self._run_test(context_max_requests_override=1)
        except RequestOverflowError as e:
            return
        raise Exception("failed.")

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_token_overflow(self) -> None:
        """Test token overflow."""
        try:
            self._run_test(context_max_tokens_override=8)
        except TokenOverflowError as e:
            return
        raise Exception("failed.")

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_chunk_overflow(self) -> None:
        """Test chunk overflow."""
        env = self._build_test_env(TestConfig())
        context = env.engine.context
        chunk_size_bytes = context.chunk_size_bytes
        buffer_size_gb = (chunk_size_bytes + 1) / 1024**3  # +1 for rounding error.
        try:
            self._run_test(context_buffer_size_gb=buffer_size_gb)
        except ChunkOverflowError as e:
            return
        raise Exception("failed.")

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

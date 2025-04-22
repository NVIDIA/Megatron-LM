import asyncio
import random
import string
from typing import AsyncGenerator, List, Union
from unittest import mock

import pytest
import torch

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.inference_request import InferenceRequest, Status
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
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestStaticInferenceEngine:
    def setup_engine(self, engine_max_batch_size=None):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        model_parallel_cuda_manual_seed(123)
        self.batch_size = 4
        self.hidden_size = 12
        self.vocab_size = 100
        self.sequence_length = 64
        transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=self.vocab_size,
            max_sequence_length=self.sequence_length,
            parallel_output=True,
        ).cuda()

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=self.hidden_size,
            inference_batch_times_seqlen_threshold=400,
            inference_max_requests=self.batch_size,
            fp32_residual_connection=False,
            params_dtype=torch.float,
            padded_vocab_size=self.vocab_size,
        )

        inference_context = StaticInferenceContext.from_config(inference_wrapper_config)

        inference_wrapped_model = GPTInferenceWrapper(
            gpt_model, inference_wrapper_config, inference_context
        )
        self.mock_tokenizer = mock.Mock()
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=self.mock_tokenizer
        )

        if engine_max_batch_size is not None and engine_max_batch_size > self.batch_size:
            with pytest.warns(UserWarning):
                self.static_engine = StaticInferenceEngine(
                    text_generation_controller=text_generation_controller,
                    max_batch_size=engine_max_batch_size,
                )
        else:
            self.static_engine = StaticInferenceEngine(
                text_generation_controller=text_generation_controller,
                max_batch_size=engine_max_batch_size,
            )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.parametrize(
        "batch_size,num_trials,empty_prompt",
        [(4, 1, False), (4, 1, True), (4, 3, False), (2, 1, False), (8, 1, False)],
    )
    def test_generate(self, batch_size: int, num_trials: int, empty_prompt: bool):
        self.setup_engine(engine_max_batch_size=batch_size)
        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.eod = self.vocab_size - 1
        # Generating random length integer prompts
        self.mock_tokenizer.tokenize.return_value = [
            random.randint(0, self.vocab_size - 1) for _ in range(random.randint(5, 10))
        ]
        # Generates some random string
        self.mock_tokenizer.detokenize.return_value = ''.join(
            random.choices(string.ascii_letters, k=random.randint(4, 10))
        )

        for _ in range(num_trials):
            if empty_prompt:
                prompts = ["" for i in range(batch_size)]
            else:
                prompts = ["sample" * (i + 1) for i in range(batch_size)]
            results: List[InferenceRequest] = self.static_engine.generate(
                prompts, sampling_params=SamplingParams(num_tokens_to_generate=10)
            )

            assert len(results) == batch_size
            for result in results:
                assert (
                    result.status == Status.COMPLETED
                ), f"Status should be completed but its {result.status}"
                assert result.generated_length > 0, f"Generated length should be greater than zero"
                assert result.generated_text is not None, f'Generated text should not be None'

    @pytest.mark.asyncio
    async def test_streaming(self):
        self.setup_engine()

        async def collect_stream(stream_generator, num_tokens_to_generate):
            prev_log_probs = None
            prev_text = ""
            prev_idx = 0
            prev_length = 0
            num_output_tokens = 0
            async for output in stream_generator:
                num_output_tokens += 1
                assert isinstance(
                    output, InferenceRequest
                ), f"Expected InferenceRequest, got {type(output)}"
                assert output.generated_log_probs is not None, f"Expected log probs tensor"
                assert (
                    output.generated_tokens.shape[0] == output.generated_length
                ), f"Expected log probs length to match # generated tokens"
                assert (
                    len(output.generated_log_probs) == output.generated_length
                ), f"Expected log probs length to match # generated tokens"
                assert output.generated_length > prev_length, f"Expected generated length to grow"
                assert (
                    output.generated_text[:prev_idx] == prev_text
                ), f"Expected generated text to match previous text"
                assert (
                    prev_log_probs is None or prev_log_probs == output.generated_log_probs[:-1]
                ), f"Expected previous log probs to match new log probs"
                prev_length = output.generated_length
                prev_text = output.generated_text
                prev_idx = len(output.generated_text)
                prev_log_probs = output.generated_log_probs

            assert (
                num_output_tokens == num_tokens_to_generate
            ), f"Should have streamed {num_tokens_to_generate} tokens but actually streamed {num_output_tokens}"
            assert (
                len(output.generated_tokens) == num_tokens_to_generate
            ), f"Should have included {num_tokens_to_generate} tokens but actually returned {len(output.generated_tokens)}"
            assert (
                len(output.generated_log_probs) == num_tokens_to_generate
            ), f"Should have included {num_tokens_to_generate} log probs but actually returned {len(output.generated_log_probs)}"

            return output

        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.bos = self.vocab_size - 2
        # Generating random length integer prompts
        self.mock_tokenizer.tokenize.return_value = [
            random.randint(0, self.vocab_size - 1) for _ in range(random.randint(5, 10))
        ]
        # Generates some random string
        self.mock_tokenizer.detokenize.return_value = ''.join(
            random.choices(string.ascii_letters, k=random.randint(4, 10))
        )

        prompts = ["" for i in range(self.batch_size)]

        num_tokens_to_generate = 10
        sampling_params = SamplingParams(
            num_tokens_to_generate=num_tokens_to_generate, return_log_probs=True
        )
        request_ids: List[str] = [
            self.static_engine.add_request(
                prompt, add_BOS=True, sampling_params=sampling_params, streaming=True
            )
            for prompt in prompts
        ]
        stream_generators: List[AsyncGenerator[InferenceRequest, None]] = [
            self.static_engine.get_stream_generator(request_id) for request_id in request_ids
        ]
        assert all(stream_generator is not None for stream_generator in stream_generators)

        tasks = [
            asyncio.create_task(collect_stream(stream_generator, num_tokens_to_generate))
            for stream_generator in stream_generators
        ]

        await self.static_engine.run_engine_async()
        final_streamed_tokens: List[InferenceRequest] = await asyncio.gather(*tasks)
        results: List[InferenceRequest] = [
            self.static_engine.scheduler.completed_request_pool[request_id]
            for request_id in request_ids
        ]
        assert len(final_streamed_tokens) == len(results)
        for result, final_streamed_token in zip(results, final_streamed_tokens):
            assert torch.equal(
                result.generated_tokens.cpu(), final_streamed_token.generated_tokens.cpu()
            ), (
                f"result.generated_tokens={result.generated_tokens.cpu()},"
                f"final_streamed_token.generated_tokens={final_streamed_token.generated_tokens}"
            )
            assert result.generated_log_probs == final_streamed_token.generated_log_probs, (
                f"result.generated_log_probs={result.generated_log_probs}, "
                f"final_streamed_token.generated_log_probs={final_streamed_token.generated_log_probs}"
            )

import copy
import os
import random
import string
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List
from unittest import mock

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.inference.contexts import DynamicInferenceContext, StaticInferenceContext
from megatron.core.inference.contexts.dynamic_context import MaxSequenceLengthOverflowError
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
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version, is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestTextGenerationController:

    def setup_model(
        self,
        dtype,
        symmetric_ar_type=None,
        fp8: bool = False,
        tensor_model_parallel_size: int = 2,
        pipeline_model_parallel_size: int = 1,
        static: bool = True,
        use_training_random_init: bool = False,
    ):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
        )
        if use_training_random_init:
            # This is necessary to induce the training behavior which permutes the random seed
            # for every rank; otherwise, every rank will have the same seed.
            _set_random_seed(123, inference_rng_tracker=True)
        else:
            model_parallel_cuda_manual_seed(123, inference_rng_tracker=True)
        self.batch_size = 4
        self.hidden_size = 12
        self.vocab_size = 100
        self.sequence_length = 60 if fp8 else 64  # Test padding for fp8
        transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            attention_backend=AttnBackend.local,
            params_dtype=dtype,
            symmetric_ar_type=symmetric_ar_type,
            fp8="hybrid" if fp8 else None,
            fp8_recipe="tensorwise" if fp8 else None,
            fp8_param=fp8,
        )
        if dtype == torch.bfloat16:
            transformer_config.bf16 = True

        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=self.vocab_size,
            max_sequence_length=self.sequence_length,
            parallel_output=True,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()
        gpt_model.eval()
        if dtype == torch.bfloat16:
            gpt_model = Float16Module(gpt_model.config, gpt_model)

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=self.hidden_size,
            inference_batch_times_seqlen_threshold=-1,
            inference_max_seq_length=2048,
            inference_max_requests=16 if fp8 else self.batch_size,
            fp32_residual_connection=False,
            params_dtype=dtype,
            padded_vocab_size=self.vocab_size,
        )

        if static:
            inference_context = StaticInferenceContext.from_config(inference_wrapper_config)
        else:
            inference_context = DynamicInferenceContext(
                params_dtype=dtype,
                num_layers=transformer_config.num_layers,
                kv_channels=transformer_config.kv_channels,
                num_attention_heads=transformer_config.num_attention_heads,
                max_sequence_length=2048,
                buffer_size_gb=1,
                buffer_guaranteed_fraction=0.1,
                materialize_only_last_token_logits=False,
            )

        inference_wrapped_model = GPTInferenceWrapper(
            gpt_model, inference_wrapper_config, inference_context
        )

        inference_wrapped_model.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

        self.mock_tokenizer = mock.Mock()

        self.text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=self.mock_tokenizer
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sample_from_logits(self):
        self.setup_model(torch.float32)

        with pytest.raises(AssertionError) as aerror:
            self.text_generation_controller.sample_from_logits(
                last_token_logits=None,
                sampling_params=SamplingParams(top_k=2, top_p=0.4),
                vocab_size=self.vocab_size,
            )
        assert str(aerror.value) == 'Cannot have top-p and top-k both greater than zero'

        with pytest.raises(AssertionError) as aerror:
            self.text_generation_controller.sample_from_logits(
                last_token_logits=None,
                sampling_params=SamplingParams(top_p=1.4, top_k=0),
                vocab_size=self.vocab_size,
            )
        assert str(aerror.value) == 'top-p should be in (0,1]'

        with pytest.raises(AssertionError) as aerror:
            self.text_generation_controller.sample_from_logits(
                last_token_logits=torch.randn(self.batch_size, 1),
                sampling_params=SamplingParams(top_k=self.vocab_size + 10),
                vocab_size=self.vocab_size,
            )
        assert str(aerror.value) == 'top-k is larger than logit size.'

        last_token_logits = (
            torch.arange(0, self.vocab_size).repeat(self.batch_size, 1).float().cuda()
        )
        sampled_logits = self.text_generation_controller.sample_from_logits(
            last_token_logits, SamplingParams(top_k=1), self.vocab_size
        )
        assert torch.all(
            sampled_logits.cpu() == torch.ones(self.batch_size) * self.vocab_size - 1
        ), f"The sampled logits should all be {self.vocab_size} but its {sampled_logits}"

        top_n_logprobs_dict = defaultdict(list)

        class MockTokenizer:
            def detokenize(self, inp, skip_special_tokens=False):
                return inp[0]

        self.text_generation_controller.tokenizer = MockTokenizer()
        last_token_logits_top_n_input = (
            torch.arange(0, self.vocab_size).repeat(self.batch_size, 1).float().cuda() / 10
        )
        sampled_logits = self.text_generation_controller.sample_from_logits(
            last_token_logits_top_n_input,
            SamplingParams(top_k=1, top_n_logprobs=3),
            self.vocab_size,
            generation_started=torch.tensor([True] * self.batch_size),
            top_n_logprobs_dict=top_n_logprobs_dict,
        )

        assert list(top_n_logprobs_dict[0][0].values()) == pytest.approx(
            [-2.3521223068237305, -2.452122688293457, -2.5521230697631836], abs=1e-3
        )

        sampled_logits = self.text_generation_controller.sample_from_logits(
            last_token_logits, SamplingParams(top_k=2), self.vocab_size
        )
        assert torch.all(
            sampled_logits >= self.vocab_size - 2
        ), f"The sampled logits should all be greater than {self.vocab_size-2} but its {sampled_logits}"

        l = last_token_logits[0]
        top_p = 0.3
        expected_min_value = l[l.softmax(dim=-1).cumsum(dim=-1) > top_p][0].item()
        sampled_logits = self.text_generation_controller.sample_from_logits(
            last_token_logits, SamplingParams(top_p=top_p, top_k=0), self.vocab_size
        )
        assert torch.all(
            sampled_logits >= expected_min_value
        ), f"The sampled logits should all be greater than {expected_min_value} but its {sampled_logits}"

        top_p = 0.95
        temperature = 2
        expected_min_value = l[l.div_(temperature).softmax(dim=-1).cumsum(dim=-1) > top_p][0].item()
        sampled_logits = self.text_generation_controller.sample_from_logits(
            last_token_logits,
            SamplingParams(top_p=top_p, temperature=temperature, top_k=0),
            self.vocab_size,
        )
        assert torch.all(
            sampled_logits >= expected_min_value
        ), f"The sampled logits should all be greater than {expected_min_value} but its {sampled_logits}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "symmetric_ar_type",
        [
            None,
            pytest.param(
                "multimem_all_reduce",
                marks=pytest.mark.skipif(
                    not is_te_min_version("2.3"),
                    reason="multimem_all_reduce requires Transformer Engine >= 2.3",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("fp8", [False, True])
    def test_generate_all_output_tokens_static_batch(self, dtype, symmetric_ar_type, fp8):
        if fp8:
            fp8_available, reason_for_no_fp8 = check_fp8_support()
            if not fp8_available:
                pytest.skip(reason_for_no_fp8)
            elif not is_te_min_version("2.2.0"):
                pytest.skip(reason="TE 2.2.0 is required")
            elif dtype != torch.bfloat16:
                pytest.skip("Only testing fp8 inference with bf16 params")

        self.setup_model(dtype, symmetric_ar_type, fp8)
        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.detokenize.side_effect = lambda x, skip_special_tokens=False: ' '.join(
            [
                ''.join(random.choices(string.ascii_letters, k=random.randint(4, 10)))
                for _ in range(len(x))
            ]
        )
        self.mock_tokenizer.offsets.side_effect = lambda _, s: [
            i for i, c in enumerate(s) if c == ' '
        ] + [len(s)]

        active_requests: Dict[str, InferenceRequest] = OrderedDict()
        all_prompt_tokens: Dict[str, List[int]] = OrderedDict()
        for i in range(self.batch_size):
            prompt = "sample" * (i + 1)
            self.mock_tokenizer.tokenize.return_value = torch.randn(
                self.batch_size, self.vocab_size
            ).cuda()
            prompt_tokens = torch.randint(
                low=0, high=self.vocab_size - 1, size=(len(prompt),)
            ).tolist()

            request_id = str(i)
            inference_request = InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=10, return_log_probs=True, return_segments=True
                ),
                arrival_time=time.time(),
                prompt_tokens=prompt_tokens,
                status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS,
            )
            active_requests[request_id] = inference_request
            all_prompt_tokens[request_id] = copy.deepcopy(prompt_tokens)

        requests = self.text_generation_controller.generate_all_output_tokens_static_batch(
            active_requests
        )

        for request_id, request in requests.items():
            assert (
                request.status == Status.COMPLETED
            ), f"Status should be completed but its {request.status}"
            assert request.generated_length > 0, f"Generated length should be greater than zero"
            assert request.generated_text is not None, "Generated text should not be None"
            assert (
                all_prompt_tokens[request_id] == request.prompt_tokens
            ), "Prompt tokens should not have changed during generation"
            # Log probabilities are calculated based on the likelihood of a token given the
            # preceding context. The first token lacks this dependency and is excluded from
            # the logprobs output, which is why the +1 is necessary
            assert (
                len(request.segments)
                == len(request.prompt_log_probs) + len(request.generated_log_probs) + 1
            ), "Segments should be returned for both prompt and generated tokens"
            assert len(request.prompt) + len(request.generated_text) == len(
                request.text
            ), "Output text should include prompts and generations"
            assert (
                request.tpot is not None
                and isinstance(request.tpot, list)
                and len(request.tpot) == request.generated_length
            )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_output_log_probs(self, dtype):
        self.setup_model(dtype)

        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.bos = 0
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.detokenize.side_effect = lambda x, skip_special_tokens=False: ' '.join(
            [
                ''.join(random.choices(string.ascii_letters, k=random.randint(4, 10)))
                for _ in range(len(x))
            ]
        )
        self.mock_tokenizer.offsets.side_effect = lambda _, s: [
            i for i, c in enumerate(s) if c == ' '
        ] + [len(s)]

        prompt = ""
        active_requests: Dict[int, InferenceRequest] = OrderedDict()
        for i in range(self.batch_size):
            self.mock_tokenizer.tokenize.return_value = torch.randn(
                self.batch_size, self.vocab_size
            ).cuda()
            inference_request = InferenceRequest(
                request_id=i,
                prompt=prompt,
                sampling_params=SamplingParams(num_tokens_to_generate=1, return_log_probs=True),
                arrival_time=time.time(),
                prompt_tokens=[self.mock_tokenizer.bos],
                status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS,
            )
            active_requests[i] = inference_request

        requests = self.text_generation_controller.generate_all_output_tokens_static_batch(
            active_requests
        )

        for request_id, request in requests.items():
            assert (
                request.status == Status.COMPLETED
            ), f"Status should be completed but its {request.status}"
            assert request.generated_length > 0, f"Generated length should be greater than zero"
            assert request.generated_text is not None, "Generated text should not be None"
            assert len(request.generated_log_probs) == request.generated_length

    @pytest.mark.parametrize("num_tokens_to_generate", [0, 4])
    @pytest.mark.parametrize("return_prompt_top_n_logprobs", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_logprobs_and_topn_consistency(
        self, num_tokens_to_generate, return_prompt_top_n_logprobs, dtype
    ):
        """
        1.  Ensures that a batch request containing prompts of
            *different* lengths still returns the correct number of log‑probs for
            every request.
        2.  Verifies that, for every token whose log prob is returned, the value
            exactly matches the log prob reported for that same token in the
            `top_n_logprobs` payload.
        """
        self.setup_model(dtype)

        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.bos = 0
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.detokenize.side_effect = lambda toks, **_: " ".join(
            f"T{t}" for t in toks
        )  # unique, deterministic
        self.mock_tokenizer.offsets.side_effect = lambda _, s: [
            i for i, c in enumerate(s) if c == " "
        ] + [len(s)]

        prompts = ["a", "foo", "foobar", "lorem ipsum"]
        active_reqs: Dict[str, InferenceRequest] = OrderedDict()

        for rid, p in enumerate(prompts):
            prompt_tokens = torch.randint(1, self.vocab_size - 2, (len(p) + 1,)).tolist()  # +bos
            prompt_tokens[0] = self.mock_tokenizer.bos  # ensure BOS

            self.mock_tokenizer.tokenize.return_value = torch.randn(
                self.batch_size, self.vocab_size
            ).cuda()

            active_reqs[str(rid)] = InferenceRequest(
                request_id=str(rid),
                prompt=p,
                prompt_tokens=prompt_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    top_k=1,
                    top_p=0.0,
                    temperature=0.0,
                    return_log_probs=True,
                    top_n_logprobs=5,
                    return_prompt_top_n_logprobs=return_prompt_top_n_logprobs,
                ),
                arrival_time=time.time(),
                status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS,
            )

        completed = self.text_generation_controller.generate_all_output_tokens_static_batch(
            active_reqs
        )

        for request_id, request in completed.items():
            prompt_log_probs = request.prompt_log_probs
            generated_log_probs = request.generated_log_probs
            prompt_top_n_logprobs = request.prompt_top_n_logprobs
            generated_top_n_logprobs = request.generated_top_n_logprobs
            generated_tokens = request.generated_tokens

            assert len(prompt_log_probs) == len(request.prompt_tokens) - 1, (
                f"{request_id}: Expected {len(request.prompt_tokens)-1} prompt log probs, "
                f"got {len(prompt_log_probs)}"
            )
            assert len(generated_log_probs) == request.generated_length, (
                f"{request_id}: Expected {request.generated_length} generated log probs, "
                f"got {len(generated_log_probs)}"
            )

            assert (not return_prompt_top_n_logprobs and prompt_top_n_logprobs is None) or (
                return_prompt_top_n_logprobs
                and prompt_top_n_logprobs is not None
                and len(prompt_top_n_logprobs) == len(prompt_log_probs)
            )
            assert len(generated_top_n_logprobs) == request.generated_length, (
                f"{request_id}: Expected {request.generated_length} generated log probs, "
                f"got {len(generated_top_n_logprobs)}"
            )
            assert (
                request.tpot is not None
                and isinstance(request.tpot, list)
                and len(request.tpot) == request.generated_length
            )

            # Verify that the generated log probs match what is returned
            # in the top-N log probs dict
            for k, log_probs in enumerate(generated_log_probs):
                token_id = generated_tokens[k]
                top_n = generated_top_n_logprobs[k]
                token = self.mock_tokenizer.detokenize([token_id])

                assert token in top_n, f"{request_id}: Generated token {token} missing in top‑N"
                assert (
                    pytest.approx(log_probs, rel=1e-6) == top_n[token]
                ), f"{request_id}: mismatch @ generated token {k}: {log_probs} vs {top_n[token]}"

    def test_token_overflow(self):
        self.setup_model(torch.float32)

        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.bos = 0
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.detokenize.side_effect = lambda x: ' '.join(
            [
                ''.join(random.choices(string.ascii_letters, k=random.randint(4, 10)))
                for _ in range(len(x))
            ]
        )
        self.mock_tokenizer.offsets.side_effect = lambda _, s: [
            i for i, c in enumerate(s) if c == ' '
        ] + [len(s)]

        prompt = ""
        active_requests: Dict[int, InferenceRequest] = OrderedDict()
        for i in range(self.batch_size):
            self.mock_tokenizer.tokenize.return_value = torch.randn(
                self.batch_size, self.vocab_size
            ).cuda()
            inference_request = InferenceRequest(
                request_id=i,
                prompt=prompt,
                sampling_params=SamplingParams(num_tokens_to_generate=4096, return_log_probs=True),
                arrival_time=time.time(),
                prompt_tokens=[self.mock_tokenizer.bos],
                status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS,
            )
            active_requests[i] = inference_request

        with pytest.raises(MaxSequenceLengthOverflowError):
            requests = self.text_generation_controller.generate_all_output_tokens_static_batch(
                active_requests
            )

    def test_zero_tokens_generated_batch_vs_single(self):
        """
        Verifies that when `num_tokens_to_generate=0`, the outputs from batched inference
        match the outputs from single-request inference for prompt-related fields.
        """
        self.setup_model(dtype=torch.bfloat16)

        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.bos = 0
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.detokenize.side_effect = lambda toks, **_: " ".join(
            f"T{t}" for t in toks
        )  # unique, deterministic
        self.mock_tokenizer.offsets.side_effect = lambda _, s: [
            i for i, c in enumerate(s) if c == " "
        ] + [len(s)]

        prompts = [
            "a short prompt",
            "a slightly longer prompt that still fits",
            "an even longer prompt to test prompt length variability",
        ]
        batch_size_test = len(prompts)
        active_requests_batched: Dict[str, InferenceRequest] = OrderedDict()
        expected_single_requests: Dict[str, InferenceRequest] = OrderedDict()

        for rid, p in enumerate(prompts):
            prompt_tokens = torch.randint(1, self.vocab_size - 2, (len(p) + 1,)).tolist()
            prompt_tokens[0] = self.mock_tokenizer.bos

            # Mock tokenize for consistency across batch and single
            self.mock_tokenizer.tokenize.return_value = torch.randn(
                batch_size_test, self.vocab_size
            ).cuda()

            sampling_params = SamplingParams(
                num_tokens_to_generate=0,
                temperature=0.0,
                top_k=1,
                return_log_probs=True,
                top_n_logprobs=5,
                return_prompt_top_n_logprobs=True,
            )

            inference_request = InferenceRequest(
                request_id=str(rid),
                prompt=p,
                prompt_tokens=prompt_tokens,
                sampling_params=copy.deepcopy(sampling_params),
                arrival_time=time.time(),
                status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS,
            )
            active_requests_batched[str(rid)] = copy.deepcopy(inference_request)
            expected_single_requests[str(rid)] = copy.deepcopy(inference_request)

        # Perform batched inference
        completed_batched = self.text_generation_controller.generate_all_output_tokens_static_batch(
            active_requests_batched
        )

        # Perform single-request inference for comparison
        completed_single: Dict[str, InferenceRequest] = OrderedDict()
        for request_id, req in expected_single_requests.items():
            single_request_dict = {request_id: req}
            result = self.text_generation_controller.generate_all_output_tokens_static_batch(
                single_request_dict
            )
            completed_single.update(result)

        # Compare results
        for request_id in completed_batched.keys():
            request_batched = completed_batched[request_id]
            request_single = completed_single[request_id]

            assert request_batched.status == Status.COMPLETED
            assert request_single.status == Status.COMPLETED

            assert request_batched.generated_length == 0
            assert request_single.generated_length == 0

            assert request_batched.prompt_tokens == request_single.prompt_tokens
            assert request_batched.prompt_log_probs == pytest.approx(
                request_single.prompt_log_probs
            )

            # Assert prompt_top_n_logprobs for consistency
            assert request_batched.prompt_top_n_logprobs is not None
            assert request_single.prompt_top_n_logprobs is not None
            assert len(request_batched.prompt_top_n_logprobs) == len(
                request_single.prompt_top_n_logprobs
            )
            for i in range(len(request_batched.prompt_top_n_logprobs)):
                assert (
                    request_batched.prompt_top_n_logprobs[i].keys()
                    == request_single.prompt_top_n_logprobs[i].keys()
                )
                for token_str in request_batched.prompt_top_n_logprobs[i]:
                    assert (
                        pytest.approx(request_batched.prompt_top_n_logprobs[i][token_str], rel=1e-6)
                        == request_single.prompt_top_n_logprobs[i][token_str]
                    )

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("tp_size", [1, 2])
    @pytest.mark.parametrize("pp_size", [1, 2])
    def test_sampled_tokens_match_with_parallelism(self, static, tp_size, pp_size):
        """
        Verify that sampled tokens match across all parallel ranks.
        """
        if tp_size == 1 and pp_size == 1:
            pytest.skip(reason="Test requires model parallel size > 1.")

        if not static and not is_fa_min_version("2.7.3"):
            pytest.skip(reason="Need latest flash attn for dynamic batching")

        # Ensure that we are using the training setup for random seed initialization
        # so that every rank has a different seed
        self.setup_model(
            dtype=torch.bfloat16,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            static=static,
            use_training_random_init=True,
        )

        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.detokenize.side_effect = lambda x, skip_special_tokens=False: ' '.join(
            [
                ''.join(random.choices(string.ascii_letters, k=random.randint(4, 10)))
                for _ in range(len(x))
            ]
        )
        self.mock_tokenizer.offsets.side_effect = lambda _, s: [
            i for i, c in enumerate(s) if c == ' '
        ] + [len(s)]

        # Prepare requests.
        active_requests: Dict[str, InferenceRequest] = OrderedDict()
        for i in range(self.batch_size):
            prompt = "sample" * (i + 1)
            prompt_tokens = torch.randint(
                low=0, high=self.vocab_size - 1, size=(len(prompt),)
            ).tolist()
            request_id = str(i)
            inference_request = InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=SamplingParams(
                    top_k=10, num_tokens_to_generate=25, return_log_probs=True
                ),
                arrival_time=time.time(),
                prompt_tokens=prompt_tokens,
                status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS,
            )
            active_requests[request_id] = inference_request

        # Generate tokens.
        if static:
            requests = self.text_generation_controller.generate_all_output_tokens_static_batch(
                active_requests
            )
            all_generated_tokens = [req.generated_tokens.tolist() for req in requests.values()]
        else:
            all_generated_tokens = [[] for _ in range(len(active_requests))]
            context = self.text_generation_controller.inference_wrapped_model.inference_context
            for request_id, request in active_requests.items():
                context.add_request(
                    request_id=int(request_id),
                    tokens=torch.tensor(
                        request.prompt_tokens, dtype=torch.long, device=torch.cuda.current_device()
                    ),
                    num_tokens_to_generate=25,
                )
            sampling_params = SamplingParams(top_k=10, return_log_probs=True)
            while context.has_unfinished_requests():
                result = self.text_generation_controller.generate_output_tokens_dynamic_batch(
                    sampling_params=sampling_params, termination_id=-1
                )
                new_tokens = result["sample"]
                assert len(new_tokens) == len(active_requests)
                for i, token in enumerate(new_tokens.tolist()):
                    all_generated_tokens[i].append(token)

        # Wait for all communication to complete before proceeding.
        torch.distributed.barrier()

        # Collect all the generated tokens for each request from each rank in the
        # model parallel group.
        mp_group = parallel_state.get_model_parallel_group()
        mp_ranks = torch.distributed.get_process_group_ranks(mp_group)
        local_rank = torch.distributed.get_rank()
        tokens_per_rank = {}
        tokens_per_rank[local_rank] = all_generated_tokens

        for i in mp_ranks:
            # Start by communicating the batch size so each rank knows how many requests to expect.
            if i == local_rank:
                batch_size = torch.tensor(
                    len(tokens_per_rank[local_rank]),
                    dtype=torch.long,
                    device=torch.cuda.current_device(),
                )
            else:
                tokens_per_rank[i] = []
                batch_size = torch.empty(1, dtype=torch.long, device=torch.cuda.current_device())
            torch.distributed.broadcast(batch_size, group=mp_group, src=i)

            for j in range(batch_size.item()):
                # For each request, communicate the sequence length followed by the actual tokens.
                if i == local_rank:
                    sequence_length = torch.tensor(
                        len(tokens_per_rank[local_rank][j]),
                        dtype=torch.int32,
                        device=torch.cuda.current_device(),
                    )
                else:
                    sequence_length = torch.empty(
                        1, dtype=torch.int32, device=torch.cuda.current_device()
                    )
                torch.distributed.broadcast(sequence_length, group=mp_group, src=i)

                if i == local_rank:
                    generated_tokens = torch.tensor(
                        tokens_per_rank[local_rank][j],
                        dtype=torch.long,
                        device=torch.cuda.current_device(),
                    )
                else:
                    generated_tokens = torch.empty(
                        sequence_length.item(), dtype=torch.long, device=torch.cuda.current_device()
                    )
                torch.distributed.broadcast(generated_tokens, group=mp_group, src=i)

                if i != local_rank:
                    tokens_per_rank[i].append(generated_tokens.tolist())

        # Ensure that every rank in the model parallel group produced the same tokens.
        for i in mp_ranks:
            if i == local_rank:
                continue
            for j, (expected, actual) in enumerate(
                zip(tokens_per_rank[local_rank], tokens_per_rank[i])
            ):
                assert (
                    expected == actual
                ), f"Rank {i} tokens differ from rank {local_rank} tokens for request {j}"

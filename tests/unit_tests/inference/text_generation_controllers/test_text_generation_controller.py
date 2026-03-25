# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext, StaticInferenceContext
from megatron.core.inference.contexts.dynamic_context import MaxSequenceLengthOverflowError
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    InferenceRequest,
    Status,
)
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
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
        batch_size: int = 4,
        static: bool = True,
        use_training_random_init: bool = False,
        materialize_only_last_token_logits: bool = False,
        num_speculative_tokens: int = 0,
        block_size_tokens: int = 256,
        enable_prefix_caching: bool = False,
        max_requests: int = None,
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
        self.batch_size = batch_size
        self.hidden_size = 32
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
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pipeline_dtype=dtype,
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

        if static:
            inference_context = StaticInferenceContext(
                max_batch_size=16 if fp8 else self.batch_size, max_sequence_length=2048
            )
        else:
            inference_context = DynamicInferenceContext(
                model_config=transformer_config,
                inference_config=InferenceConfig(
                    max_sequence_length=2048,
                    buffer_size_gb=0.2,
                    materialize_only_last_token_logits=materialize_only_last_token_logits,
                    use_flashinfer_fused_rope=None,  # default to using flash-infer if available
                    # this is for compatibility with the LTS environment
                    unified_memory_level=0,  # unit tests currently broken with UVM
                    num_speculative_tokens=num_speculative_tokens,
                    block_size_tokens=block_size_tokens,
                    enable_prefix_caching=enable_prefix_caching,
                    max_requests=max_requests,
                ),
            )

        inference_wrapped_model = GPTInferenceWrapper(gpt_model, inference_context)

        inference_wrapped_model.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

        self.mock_tokenizer = mock.Mock()

        self.text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=self.mock_tokenizer
        )

    @classmethod
    def teardown_class(cls):
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

    @pytest.mark.parametrize("backend", ["torch"])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_sample_from_dynamic_logits(
        self, backend: str, materialize_only_last_token_logits: bool
    ):
        batch_size = 12
        self.setup_model(
            torch.float32,
            batch_size=batch_size,
            static=False,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
        )
        self.mock_tokenizer.eod = self.vocab_size

        context = self.text_generation_controller.inference_wrapped_model.inference_context

        # Prepare sampling params in human-readable format, to aid with test maintenance.
        sampling_test_cases: List[Tuple[SamplingParams, List[int]]] = [
            (SamplingParams(temperature=0.1, top_p=0.01), [9, 6, 10]),
            (SamplingParams(temperature=5.0, top_k=15), [0, 3, 2]),
            (SamplingParams(top_p=0.8), [4, 1, 7]),
            (SamplingParams(temperature=10.0, top_k=5), [11, 5, 8]),
        ]
        # For non-torch backends, test simultaneous top_k and top_p sampling.
        if backend != "torch":
            sampling_test_cases[3][0].top_p = 0.8

        # Convert sampling params to non-readable format.
        rev_sampling_dict: List[SamplingParams] = [None] * batch_size
        for sampling_params, indices in sampling_test_cases:
            for idx in indices:
                rev_sampling_dict[idx] = sampling_params

        # Prepare metadata for sample bookkeeping.
        temp_values = torch.Tensor([s.temperature for s in rev_sampling_dict])
        top_k_values = torch.Tensor([s.top_k for s in rev_sampling_dict]).to(torch.int32)
        top_p_values = torch.Tensor([s.top_p for s in rev_sampling_dict])
        request_metadata = {
            "temperature": temp_values,
            "top_k": top_k_values,
            "top_p": top_p_values,
        }
        self.text_generation_controller._request_metadata = request_metadata
        self.text_generation_controller._sampling_backend = backend

        context.padded_active_token_count = batch_size
        context.request_query_lengths = torch.ones(batch_size, dtype=torch.int32)
        context.paused_request_count = 0
        context.total_request_count = batch_size

        # Bookkeeping.
        self.text_generation_controller._dynamic_step_sample_bookkeeping()

        # Sampling.
        logits = torch.arange(0, self.vocab_size).repeat(batch_size, 1).unsqueeze(0).float().cuda()
        self.text_generation_controller._dynamic_step_sample_logits(logits)
        sampled_logits = self.text_generation_controller._sampled_tokens_cuda[:batch_size]
        vocab_indices = torch.arange(self.vocab_size).cuda()

        # Move tensors to GPU for assertion checks.
        temp_values = temp_values.cuda()
        top_k_values = top_k_values.cuda()
        top_p_values = top_p_values.cuda()

        # Assert correct sampled values.
        top_k_values[top_k_values == 0] = self.vocab_size
        assert torch.all(
            sampled_logits >= self.vocab_size - top_k_values
        ), f"The sampled logits should all be greater than {self.vocab_size - top_k_values} but its {sampled_logits}"
        l = logits.squeeze(0)
        sampled_l = l.div(temp_values.unsqueeze(1)).softmax(dim=-1)
        top_k_mask = vocab_indices.unsqueeze(0) < (self.vocab_size - top_k_values.unsqueeze(1))
        sampled_l.masked_fill_(top_k_mask, 0.0)
        top_p_mask = sampled_l.cumsum(dim=-1) > top_p_values.unsqueeze(1)

        first_excluded = torch.where(
            top_p_mask.any(dim=-1),
            top_p_mask.float().argmax(dim=-1),
            torch.full((batch_size,), self.vocab_size, device=top_p_mask.device),
        )
        last_included = torch.clamp(first_excluded - 1, min=0)
        start_idx = torch.clamp(self.vocab_size - top_k_values, min=0).long()
        last_included = torch.max(last_included, start_idx)
        expected_min_values = l.gather(1, last_included.unsqueeze(1)).squeeze(1)
        assert torch.all(
            sampled_logits >= expected_min_values
        ), f"The sampled logits should all be greater than {expected_min_values} but its {sampled_logits}"

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
    @pytest.mark.parametrize("skip_prompt_log_probs", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_logprobs_and_topn_consistency(
        self, num_tokens_to_generate, skip_prompt_log_probs, dtype
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
                    skip_prompt_log_probs=skip_prompt_log_probs,
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

            assert (skip_prompt_log_probs and prompt_top_n_logprobs is None) or (
                not skip_prompt_log_probs
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

    def test_add_bos_token(self):
        self.setup_model(torch.float32)

        prompt = "sample prompt"

        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.bos = 0
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.detokenize.side_effect = lambda x, **_: ' '.join(
            [
                ''.join(random.choices(string.ascii_letters, k=random.randint(1, len(prompt))))
                for _ in range(len(x))
            ]
        )
        self.mock_tokenizer.offsets.side_effect = lambda _, s: [
            i for i, c in enumerate(s) if c == ' '
        ] + [len(s)]
        self.mock_tokenizer.tokenize.return_value = [
            random.randint(0, self.vocab_size - 1) for _ in range(len(prompt))
        ]

        tokenizer = self.mock_tokenizer

        # Test on a tokenizer that does not add BOS by default
        no_bos_to_no_bos = TextGenerationController.tokenize_prompt(
            tokenizer, prompt, add_BOS=False
        )
        assert no_bos_to_no_bos[0] != tokenizer.bos
        no_bos_to_yes_bos = TextGenerationController.tokenize_prompt(
            tokenizer, prompt, add_BOS=True
        )
        assert no_bos_to_yes_bos[0] == tokenizer.bos
        assert no_bos_to_yes_bos[1] != tokenizer.bos

        # Force the first token to be BOS to emulate a tokenizer that does add BOS by default
        tokenizer.tokenize.return_value[0] = tokenizer.bos

        yes_bos_to_no_bos = TextGenerationController.tokenize_prompt(
            tokenizer, prompt, add_BOS=False
        )
        assert yes_bos_to_no_bos[0] != tokenizer.bos
        yes_bos_to_yes_bos = TextGenerationController.tokenize_prompt(
            tokenizer, prompt, add_BOS=True
        )
        assert yes_bos_to_yes_bos[0] == tokenizer.bos
        assert yes_bos_to_yes_bos[1] != tokenizer.bos

        # Test on an input that has had multiple BOS added
        tokenizer.tokenize.return_value[1] = tokenizer.bos

        many_bos_to_no_bos = TextGenerationController.tokenize_prompt(
            tokenizer, prompt, add_BOS=False
        )
        assert many_bos_to_no_bos[0] != tokenizer.bos
        many_bos_to_yes_bos = TextGenerationController.tokenize_prompt(
            tokenizer, prompt, add_BOS=True
        )
        assert many_bos_to_yes_bos[0] == tokenizer.bos
        assert many_bos_to_yes_bos[1] != tokenizer.bos

        # Test the assert triggered when the tokenizer has no bos
        tokenizer.bos = None
        with pytest.raises(AssertionError):
            TextGenerationController.tokenize_prompt(tokenizer, prompt, add_BOS=True)

    @pytest.mark.parametrize("remove_EOD", [True, False])
    def test_remove_eod_token(self, remove_EOD):
        self.setup_model(torch.float32)

        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.bos = 0
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.detokenize.side_effect = lambda x, **_: ' '.join(f"T{t}" for t in x)

        tokenizer = self.mock_tokenizer
        eod = tokenizer.eod
        detok = TextGenerationController.detokenize

        # No trailing EOD.
        assert detok(tokenizer, [1, 2, 3], remove_EOD=remove_EOD) == "T1 T2 T3"

        # Single trailing EOD.
        result = detok(tokenizer, [1, 2, eod], remove_EOD=remove_EOD)
        assert result == ("T1 T2" if remove_EOD else f"T1 T2 T{eod}")

        # Multiple trailing EOD.
        result = detok(tokenizer, [1, eod, eod, eod], remove_EOD=remove_EOD)
        assert result == ("T1" if remove_EOD else f"T1 T{eod} T{eod} T{eod}")

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
                skip_prompt_log_probs=False,
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

    @pytest.mark.parametrize("skip_prompt_log_probs", [True, False])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_dynamic_top_n_logprobs_calculation(
        self, skip_prompt_log_probs: bool, materialize_only_last_token_logits: bool
    ):
        """
        Test the _dynamic_step_calculate_top_n_logprobs function directly.
        Verifies:
        1. top_n_logprobs are computed for all requests
        2. skip_prompt_log_probs controls computation for prompt tokens
        3. Correct number of tokens are returned for each request
        """
        batch_size = 4
        self.setup_model(
            torch.bfloat16,
            batch_size=batch_size,
            static=False,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
        )
        self.mock_tokenizer.eod = self.vocab_size

        context = self.text_generation_controller.inference_wrapped_model.inference_context

        # Prepare sampling params
        top_n = 5
        request_metadata = {
            "top_n_logprobs": torch.full((batch_size,), top_n, dtype=torch.int32).cuda(),
            "skip_prompt_log_probs": torch.full(
                (batch_size,), float(skip_prompt_log_probs), dtype=torch.float32
            ).cuda(),
        }
        self.text_generation_controller._request_metadata = request_metadata
        self.text_generation_controller._active_request_count = batch_size
        self.text_generation_controller._active_request_slice = slice(0, batch_size)

        if materialize_only_last_token_logits:
            # Decode mode: logits for last tokens only
            logits = torch.randn(1, batch_size, self.vocab_size).cuda()

            # Set up context state for decode mode
            context.paused_request_count = 0
            context.total_request_count = batch_size

            # Compute log probabilities (required by _dynamic_step_calculate_top_n_logprobs)
            # Note: squeeze(0) to match what calculate_log_probs does in dynamic_context.py
            log_probs_tensor = torch.nn.functional.log_softmax(logits.squeeze(0), dim=-1)

            # Calculate top-n logprobs
            top_n_results = self.text_generation_controller._dynamic_step_calculate_top_n_logprobs(
                logits, log_probs_tensor
            )

            # Validate results
            assert top_n_results is not None, "top_n_results should not be None"
            assert (
                len(top_n_results) == batch_size
            ), f"Expected {batch_size} requests, got {len(top_n_results)}"

            for req_idx in range(batch_size):
                assert req_idx in top_n_results, f"Request {req_idx} missing from results"
                top_n_list = top_n_results[req_idx]

                # In decode mode, should have exactly 1 token per request
                assert (
                    len(top_n_list) == 1
                ), f"Request {req_idx}: expected 1 token, got {len(top_n_list)}"

                top_n_values, top_n_indices = top_n_list[0]
                assert top_n_values.shape[0] == top_n, f"Expected {top_n} values"
                assert top_n_indices.shape[0] == top_n, f"Expected {top_n} indices"
        else:
            # Prefill mode: logits for all tokens
            # Simulate different prompt lengths
            query_lengths = [4, 6, 5, 7]  # Different lengths for each request
            total_tokens = sum(query_lengths)

            # Set up context state
            context.paused_request_count = 0
            context.total_request_count = batch_size
            context.active_token_count = total_tokens
            context.num_prefill_requests = batch_size
            context.request_query_lengths = torch.tensor(
                [0] * context.paused_request_count + query_lengths, dtype=torch.int32, device='cuda'
            )

            # Create logits for all tokens
            logits = torch.randn(1, total_tokens, self.vocab_size).cuda()

            # Compute log probabilities (required by _dynamic_step_calculate_top_n_logprobs)
            # Note: squeeze(0) to match what calculate_log_probs does in dynamic_context.py
            log_probs_tensor = torch.nn.functional.log_softmax(logits.squeeze(0), dim=-1)

            # Calculate top-n logprobs
            top_n_results = self.text_generation_controller._dynamic_step_calculate_top_n_logprobs(
                logits, log_probs_tensor
            )

            # Validate results
            assert top_n_results is not None, "top_n_results should not be None"
            assert (
                len(top_n_results) == batch_size
            ), f"Expected {batch_size} requests, got {len(top_n_results)}"

            for req_idx in range(batch_size):
                assert req_idx in top_n_results, f"Request {req_idx} missing from results"
                top_n_list = top_n_results[req_idx]

                if not skip_prompt_log_probs:
                    # Should have top-n for all tokens
                    expected_count = query_lengths[req_idx]
                    assert (
                        len(top_n_list) == expected_count
                    ), f"Request {req_idx}: expected {expected_count} tokens, got {len(top_n_list)}"
                else:
                    # Should have top-n for only the last token (first generated token)
                    assert (
                        len(top_n_list) == 1
                    ), f"Request {req_idx}: expected 1 token when skip_prompt_log_probs=True, got {len(top_n_list)}"

                # Validate each token's top-n
                for token_idx, (top_n_values, top_n_indices) in enumerate(top_n_list):
                    assert (
                        top_n_values.shape[0] == top_n
                    ), f"Request {req_idx}, token {token_idx}: expected {top_n} values"
                    assert (
                        top_n_indices.shape[0] == top_n
                    ), f"Request {req_idx}, token {token_idx}: expected {top_n} indices"

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
                    DynamicInferenceRequest(
                        request_id=int(request_id),
                        prompt_tokens=torch.tensor(
                            request.prompt_tokens,
                            dtype=torch.long,
                            device=torch.cuda.current_device(),
                        ),
                        sampling_params=SamplingParams(
                            top_k=10, return_log_probs=True, num_tokens_to_generate=25
                        ),
                    )
                )
            expected_active_requests = set(int(x) for x in active_requests.keys())
            while context.has_unfinished_requests():
                result = self.text_generation_controller.generate_output_tokens_dynamic_batch()
                new_tokens = result["sample"]
                active_ids = result["active_request_ids"].tolist()
                finished_ids = result["finished_request_ids"].tolist()
                assert len(new_tokens) == len(expected_active_requests)
                assert set(active_ids) == expected_active_requests
                expected_active_requests -= set(finished_ids)
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

    @pytest.mark.internal
    def test_speculative_verify_tokens(self):
        """Test consecutive token acceptance logic for speculative decoding."""
        self.setup_model(torch.float32, static=False, num_speculative_tokens=2, max_requests=2)

        # Enable speculative decoding
        self.text_generation_controller.num_speculative_tokens = 2
        ctx = self.text_generation_controller.inference_wrapped_model.inference_context
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.request_in_prefill_status_tensor = torch.tensor(
            [0, 0], device='cuda'
        )  # Decode requests
        ctx.request_query_lengths = torch.tensor(
            [3, 3], dtype=torch.int32, device='cuda'
        )  # 1 sampled + 2 spec

        # Init accepted tokens tensors
        self.text_generation_controller._init_mtp_sampling_tensor()

        # Mock inputs: [Req 1 sampled, Req 1 spec1, Req 1 spec2, Req 2 sampled, Req 2 spec1, Req 2 spec2]
        # Target tokens (what the model was fed): [T0, T1, T2, T3, T4, T5]
        input_ids = torch.tensor([[10, 11, 12, 20, 21, 22]], device='cuda')

        # We need the sampling function to return a 1D tensor for base logits,
        # and a 1D tensor for the flattened MTP logits.
        def mock_sampling_func(logits, *args, **kwargs):
            if logits.shape[0] == 6:
                # Base logits -> return 1D tensor of shape [6]
                # Req 1: Predicts [11, 12, 99]. Matches T1, T2. Rejects T3. -> Accepts 2 spec tokens.
                # Req 2: Predicts [99, 22, 23]. Fails at first spec token (99 != 21). -> Accepts 0 spec tokens.
                return torch.tensor([11, 12, 99, 99, 22, 23], dtype=torch.long, device='cuda')
            else:
                # MTP logits -> return 1D tensor of shape [12]
                # The verification logic only uses base tokens, so we can return zeros here.
                return torch.zeros((12,), dtype=torch.long, device='cuda')

        # Override sampling to return our predictable mock outputs
        self.text_generation_controller._torch_sampling_buckets = [([0, 1], 1.0, 1, 0.0)]
        self.text_generation_controller._torch_sampling_func = mock.MagicMock(
            side_effect=mock_sampling_func
        )

        # Mock logits matching input shape
        logits = torch.randn(1, 6, self.vocab_size, device='cuda')

        self.text_generation_controller._dynamic_step_sample_logits_and_verify_tokens(
            logits, input_ids
        )

        # Verify acceptance counts
        accepted_counts = self.text_generation_controller._accepted_token_counts_per_request[:2]
        assert torch.equal(accepted_counts, torch.tensor([2, 0], device='cuda'))

        # Verify accepted tokens tensor
        accepted_tokens = self.text_generation_controller._accepted_tokens_per_request[:2]
        # Req 1 accepted 2 tokens: 11, 12
        assert torch.equal(accepted_tokens[0], torch.tensor([11, 12], device='cuda'))
        # Req 2 accepted 0 tokens, should remain -1
        assert torch.equal(accepted_tokens[1], torch.tensor([-1, -1], device='cuda'))

    @pytest.mark.internal
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_rewind_kv_cache(self, is_hybrid_model):
        """Test KV cache state is properly rewound for rejected speculative tokens."""
        self.setup_model(
            torch.float32,
            static=False,
            num_speculative_tokens=3,
            block_size_tokens=4,
            max_requests=16,
        )
        self.text_generation_controller.num_speculative_tokens = 3
        ctx = self.text_generation_controller.inference_wrapped_model.inference_context
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.request_in_prefill_status_tensor = torch.tensor([0, 0], device='cuda')

        # Initialize allocator and states
        ctx.kv_block_allocator.total_avail = 100
        ctx.request_kv_length_offsets[:2] = torch.tensor([10, 15], device='cuda')
        ctx.request_kv_block_counts[:2] = torch.tensor([3, 4], device='cuda')

        # Req 0: offset 2. Rewinding 2 tokens -> offset 0. No block released.
        # Req 1: offset 1. Rewinding 3 tokens -> offset 2 (prev block). 1 block released.
        ctx.request_last_kv_block_offset[:2] = torch.tensor([2, 1], device='cuda')
        ctx.request_last_kv_block_id[:2] = torch.tensor([50, 60], device='cuda')
        ctx.request_to_kv_block_ids[:2, :4] = torch.tensor(
            [[48, 49, 50, -1], [57, 58, 59, 60]], dtype=torch.int, device='cuda'
        )

        if is_hybrid_model:
            ctx.is_hybrid_model = True
            ctx.mamba_metadata = mock.MagicMock()
            ctx.mamba_metadata.request_to_mamba_state_idx = torch.tensor([0, 1], device='cuda')
            ctx.mamba_ssm_states = torch.zeros((1, 2, 16), device='cuda')
            ctx.mamba_intermediate_ssm_states = torch.ones((1, 2, 4, 16), device='cuda') * 99
            ctx.mamba_conv_states = torch.zeros((1, 2, 8), device='cuda')
            ctx.mamba_intermediate_conv_states = torch.ones((1, 2, 4, 8), device='cuda') * 77

        # Mock accepted token counts: Req 0 accepts 1 (rejects 2), Req 1 accepts 0 (rejects 3)
        self.text_generation_controller._init_mtp_sampling_tensor()
        self.text_generation_controller._accepted_token_counts_per_request = torch.tensor(
            [1, 0], device='cuda'
        )

        self.text_generation_controller._rewind_kv_cache()

        # Assert offsets updated
        assert torch.equal(
            ctx.request_last_kv_block_offset[:2],
            torch.tensor([0, 2], dtype=torch.int, device='cuda'),
        )
        assert torch.equal(
            ctx.request_kv_length_offsets[:2], torch.tensor([8, 12], dtype=torch.int, device='cuda')
        )

        # Assert block counts and IDs updated for boundary crossing
        assert torch.equal(
            ctx.request_kv_block_counts[:2], torch.tensor([3, 3], dtype=torch.int, device='cuda')
        )
        assert torch.equal(
            ctx.request_last_kv_block_id[:2], torch.tensor([50, 59], dtype=torch.int, device='cuda')
        )

        # Assert released block is cleared
        assert ctx.request_to_kv_block_ids[1, 3].item() == -1
        assert ctx.kv_block_allocator.total_avail == 101  # 1 block released

        if is_hybrid_model:
            # Check Mamba state was restored from intermediate cache based on accepted counts
            assert torch.all(ctx.mamba_ssm_states[:, 0] == 99)  # Req 0 accepted 1, loaded index 1
            assert torch.all(ctx.mamba_ssm_states[:, 1] == 99)  # Req 1 accepted 0, loaded index 0
            assert torch.all(ctx.mamba_conv_states[:, 0] == 77)  # Req 0 accepted 1, loaded index 1
            assert torch.all(ctx.mamba_conv_states[:, 1] == 77)  # Req 1 accepted 0, loaded index 0

    @pytest.mark.internal
    def test_speculative_multinomial_sampling(self):
        """Test that speculative decoding can successfully use non-greedy sampling
        (top_k > 1, top_p > 0) by flattening 3D MTP logits for torch.multinomial."""
        num_spec = 3
        self.setup_model(
            torch.float32, static=False, num_speculative_tokens=num_spec, max_requests=2
        )

        # Enable speculative decoding
        self.text_generation_controller.num_speculative_tokens = num_spec
        ctx = self.text_generation_controller.inference_wrapped_model.inference_context
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.request_in_prefill_status_tensor = torch.tensor(
            [0, 0], device='cuda'
        )  # Decode requests
        # query lengths for decode with spec tokens is (1 + num_spec) = 4
        ctx.request_query_lengths = torch.tensor([4, 4], dtype=torch.int32, device='cuda')

        # Setup inputs
        input_ids = torch.randint(0, self.vocab_size, (1, 8), device='cuda')

        # Create random logits
        # Base logits shape: [1, 8, vocab_size]
        logits = torch.randn(1, 8, self.vocab_size, device='cuda')

        # Set up a bucket that forces multinomial sampling (top_p = 0.9, top_k = 0)
        # _torch_sampling_buckets format: (indices, temp, top_k, top_p)
        self.text_generation_controller._torch_sampling_buckets = [([0, 1], 1.0, 0, 0.9)]

        # Since we are actually testing the internal math of `_torch_sampling_func` handling the shapes,
        # we DO NOT mock `_torch_sampling_func` here. We want it to run natively to prove it doesn't crash.

        try:
            self.text_generation_controller._dynamic_step_sample_logits_and_verify_tokens(
                logits, input_ids
            )
        except RuntimeError as e:
            if "prob_dist must be 1 or 2 dim" in str(e):
                pytest.fail("MTP logits were not flattened before calling multinomial sampling.")
            else:
                raise e

        # Validate that sampling produced output arrays of the correct sizes
        active_request_count = ctx.total_request_count
        sampled_tokens = self.text_generation_controller._sampled_tokens_cuda[:active_request_count]
        sampled_mtp_tokens = self.text_generation_controller._sampled_mtp_tokens_cuda[
            :, :active_request_count
        ]

        assert sampled_tokens.shape == (2,)
        assert sampled_mtp_tokens.shape == (num_spec, 2)

    @pytest.mark.internal
    def test_rewind_kv_cache_with_prefix_caching_ref_counts(self):
        """Test that _rewind_kv_cache correctly decrements ref counts on shared blocks
        when speculative token rejection causes a block boundary crossing."""
        self.setup_model(
            torch.float32,
            static=False,
            num_speculative_tokens=2,
            block_size_tokens=4,
            enable_prefix_caching=True,
            max_requests=16,
        )

        ctx = self.text_generation_controller.inference_wrapped_model.inference_context
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.request_in_prefill_status_tensor = torch.tensor([0, 0], device='cuda')

        # Req 0: 3 blocks, offset 1 in last block. Rewinding 1 token -> no block release.
        # Req 1: 3 blocks, offset 0 in last block. Rewinding 2 tokens -> crosses back, release block.
        ctx.request_kv_length_offsets[:2] = torch.tensor([9, 9], device='cuda')
        ctx.request_kv_block_counts[:2] = torch.tensor([3, 3], device='cuda')
        ctx.request_last_kv_block_offset[:2] = torch.tensor([1, 0], device='cuda')
        ctx.request_last_kv_block_id[:2] = torch.tensor([10, 20], device='cuda')
        ctx.request_to_kv_block_ids[:2, :3] = torch.tensor(
            [[8, 9, 10], [18, 19, 20]], dtype=torch.int, device='cuda'
        )

        # Set ref counts: block 20 is shared (ref=2), block 10 is exclusive (ref=1).
        ctx.kv_block_allocator.block_ref_counts[20] = 2
        ctx.kv_block_allocator.block_ref_counts[10] = 1

        initial_avail = ctx.kv_block_allocator.total_avail

        # Req 0 accepts 1 (rewinds 1), Req 1 accepts 0 (rewinds 2, crosses boundary).
        self.text_generation_controller._init_mtp_sampling_tensor()
        self.text_generation_controller._accepted_token_counts_per_request = torch.tensor(
            [1, 0], device='cuda'
        )

        self.text_generation_controller._rewind_kv_cache()

        # Req 1 should have released block 20 (ref count decremented).
        assert ctx.kv_block_allocator.block_ref_counts[20].item() == 1
        # Block 10 should be untouched.
        assert ctx.kv_block_allocator.block_ref_counts[10].item() == 1

    @pytest.mark.internal
    def test_rewind_kv_cache_does_not_release_shared_prefix_blocks(self):
        """Test that rewinding only releases the last block, never shared prefix blocks."""
        self.setup_model(
            torch.float32,
            static=False,
            num_speculative_tokens=3,
            block_size_tokens=4,
            max_requests=16,
        )

        ctx = self.text_generation_controller.inference_wrapped_model.inference_context
        ctx.total_request_count = 1
        ctx.paused_request_count = 0
        ctx.request_in_prefill_status_tensor = torch.tensor([0], device='cuda')

        # 4 blocks. Offset 2 in last block. Rewinding 3 crosses into previous block.
        ctx.request_kv_length_offsets[:1] = torch.tensor([14], device='cuda')
        ctx.request_kv_block_counts[:1] = torch.tensor([4], device='cuda')
        ctx.request_last_kv_block_offset[:1] = torch.tensor([2], device='cuda')
        ctx.request_last_kv_block_id[:1] = torch.tensor([40], device='cuda')
        ctx.request_to_kv_block_ids[0, :4] = torch.tensor(
            [10, 20, 30, 40], dtype=torch.int, device='cuda'
        )

        # Blocks 10, 20 are shared prefix blocks. Block 30, 40 are exclusive.
        ctx.kv_block_allocator.total_avail = 50

        self.text_generation_controller._init_mtp_sampling_tensor()
        self.text_generation_controller._accepted_token_counts_per_request = torch.tensor(
            [0], device='cuda'
        )

        self.text_generation_controller._rewind_kv_cache()

        # Only block 40 should be released, not blocks 10, 20, or 30.
        assert ctx.request_kv_block_counts[0].item() == 3
        assert ctx.request_last_kv_block_id[0].item() == 30
        assert ctx.request_to_kv_block_ids[0, 3].item() == -1
        assert ctx.kv_block_allocator.total_avail == 51  # exactly 1 block released

        # Prefix blocks remain in request_to_kv_block_ids.
        assert ctx.request_to_kv_block_ids[0, 0].item() == 10
        assert ctx.request_to_kv_block_ids[0, 1].item() == 20
        assert ctx.request_to_kv_block_ids[0, 2].item() == 30

    @pytest.mark.internal
    def test_speculative_mtp_position_ids_with_prefill(self):
        """Test that _compute_serial_mtp_and_sample uses the correct position IDs
        for a mixed batch of prefill and decode requests."""
        self.setup_model(torch.float32, static=False, num_speculative_tokens=2, max_requests=2)

        self.text_generation_controller.num_speculative_tokens = 2
        self.text_generation_controller.num_mtp_heads = 2
        ctx = self.text_generation_controller.inference_wrapped_model.inference_context
        ctx.total_request_count = 2
        ctx.paused_request_count = 0

        # Req 0: Decode, Req 1: Prefill
        ctx.request_in_prefill_status_tensor = torch.tensor([0, 1], device='cuda')

        # Req 0 has 10 previous tokens, just processed 3 (1 base + 2 spec)
        # Req 1 has 0 previous tokens, just processed 15 (prefill)
        ctx.request_kv_length_offsets[:2] = torch.tensor([10, 0], dtype=torch.int32, device='cuda')
        ctx.request_query_lengths[:2] = torch.tensor([3, 15], dtype=torch.int32, device='cuda')

        self.text_generation_controller._init_mtp_sampling_tensor()
        # Mock base token sampling (the first tokens fed into MTP)
        self.text_generation_controller._sampled_tokens_cuda[:2] = torch.tensor(
            [100, 200], device='cuda'
        )

        # Mock the MTP computation to record the position_ids it receives
        unwrapped_model = self.text_generation_controller.inference_wrapped_model.model
        unwrapped_model._decoder_hidden_states_cache = torch.randn(2, 1, 32, device='cuda')
        self.text_generation_controller._last_accepted_seq_indices = torch.tensor(
            [0, 1], device='cuda'
        )

        captured_position_ids = []

        def mock_compute_mtp_single_step(
            hidden_states,
            next_token_ids,
            position_ids,
            depth,
            scatter_hidden_input=True,
            gather_hidden_output=True,
        ):
            captured_position_ids.append(position_ids.clone())
            return hidden_states, torch.randn(2, 1, self.vocab_size, device='cuda')

        unwrapped_model.compute_mtp_single_step = mock.MagicMock(
            side_effect=mock_compute_mtp_single_step
        )

        # Mock _sample_from_logits_2d to return arbitrary dummy tokens
        self.text_generation_controller._sample_from_logits_2d = mock.MagicMock(
            return_value=torch.tensor([101, 201], device='cuda')
        )

        self.text_generation_controller._compute_serial_mtp_and_sample()

        # The base_position for Req 0 should be 10 + 3 = 13
        # The base_position for Req 1 should be 0 + 15 = 15
        assert len(captured_position_ids) == 2
        # Depth 0:
        assert torch.equal(
            captured_position_ids[0].squeeze(0), torch.tensor([13, 15], device='cuda')
        )
        # Depth 1:
        assert torch.equal(
            captured_position_ids[1].squeeze(0), torch.tensor([14, 16], device='cuda')
        )

    @pytest.mark.parametrize("active_request_count", [1, 3, 5])
    def test_mtp_sp_padding_real_ranks(self, active_request_count):
        """Test that _compute_serial_mtp_and_sample pads inputs to a multiple of tp_size
        and strips the padding from outputs so only real requests are sampled."""
        tp_size = 2
        num_spec = 2
        self.setup_model(
            torch.float32,
            static=False,
            num_speculative_tokens=num_spec,
            max_requests=(active_request_count % tp_size) + active_request_count,
        )

        ctrl = self.text_generation_controller
        ctrl.num_speculative_tokens = num_spec
        ctrl.num_mtp_heads = num_spec
        ctx = ctrl.inference_wrapped_model.inference_context
        ctx.total_request_count = active_request_count
        ctx.paused_request_count = 0

        ctx.request_kv_length_offsets[:active_request_count] = torch.arange(
            active_request_count, dtype=torch.int32, device='cuda'
        )
        ctx.request_query_lengths[:active_request_count] = torch.ones(
            active_request_count, dtype=torch.int32, device='cuda'
        )

        ctrl._init_mtp_sampling_tensor()
        ctrl._sampled_tokens_cuda[:active_request_count] = torch.arange(
            active_request_count, device='cuda'
        )

        unwrapped_model = ctrl.inference_wrapped_model.model
        unwrapped_model._decoder_hidden_states_cache = torch.randn(
            active_request_count, 1, self.hidden_size, device='cuda'
        )
        ctrl._last_accepted_seq_indices = torch.arange(active_request_count, device='cuda')

        expected_pad = (tp_size - active_request_count % tp_size) % tp_size
        expected_padded = active_request_count + expected_pad
        captured_shapes = []

        def mock_mtp(
            hidden_states,
            next_token_ids,
            position_ids,
            depth,
            scatter_hidden_input=True,
            gather_hidden_output=True,
        ):
            captured_shapes.append(
                {
                    'hidden': hidden_states.shape,
                    'tokens': next_token_ids.shape,
                    'positions': position_ids.shape,
                }
            )
            return hidden_states, torch.randn(
                hidden_states.shape[0], 1, self.vocab_size, device='cuda'
            )

        unwrapped_model.compute_mtp_single_step = mock.MagicMock(side_effect=mock_mtp)
        ctrl._sample_from_logits_2d = mock.MagicMock(
            return_value=torch.zeros(active_request_count, dtype=torch.int64, device='cuda')
        )

        # Mock get_pg_size to return tp_size regardless of actual process group.
        with mock.patch(
            'megatron.core.inference.text_generation_controllers.text_generation_controller.get_pg_size',
            return_value=tp_size,
        ):
            ctrl._compute_serial_mtp_and_sample()

        # Verify inputs to compute_mtp_single_step were padded.
        assert len(captured_shapes) == num_spec
        for shapes in captured_shapes:
            assert shapes['hidden'] == (expected_padded, 1, self.hidden_size)
            assert shapes['tokens'] == (1, expected_padded)
            assert shapes['positions'] == (1, expected_padded)

        # Verify sampling received only real (unpadded) logits.
        for call_args in ctrl._sample_from_logits_2d.call_args_list:
            logits_arg = call_args[0][0]
            assert logits_arg.shape[0] == active_request_count

        # Verify sampled tokens were written only for real requests.
        for depth in range(num_spec):
            assert ctrl._sampled_mtp_tokens_cuda[depth, :active_request_count].shape == (
                active_request_count,
            )

    @pytest.mark.parametrize("active_request_count", [1, 3, 5])
    def test_mtp_sp_padding_dummy_ranks(self, active_request_count):
        """Test that _dummy_serial_mtp_forward pads dummy tensors to a multiple of tp_size."""
        tp_size = 2
        num_spec = 2
        self.setup_model(
            torch.float32,
            static=False,
            num_speculative_tokens=num_spec,
            max_requests=(active_request_count % tp_size) + active_request_count,
        )

        ctrl = self.text_generation_controller
        ctrl.num_speculative_tokens = num_spec
        ctrl.num_mtp_heads = num_spec
        # Simulate EP > 1 so the dummy path is not short-circuited.
        ctrl.model_config.expert_model_parallel_size = 2

        unwrapped_model = ctrl.inference_wrapped_model.model
        unwrapped_model._decoder_hidden_states_cache = True  # trigger has_mtp=True

        captured_shapes = []

        def mock_mtp(
            hidden_states,
            next_token_ids,
            position_ids,
            depth,
            scatter_hidden_input=True,
            gather_hidden_output=True,
        ):
            captured_shapes.append(
                {
                    'hidden': hidden_states.shape,
                    'tokens': next_token_ids.shape,
                    'positions': position_ids.shape,
                }
            )
            return hidden_states, torch.randn(
                hidden_states.shape[0], 1, self.vocab_size, device='cuda'
            )

        unwrapped_model.compute_mtp_single_step = mock.MagicMock(side_effect=mock_mtp)

        with mock.patch(
            'megatron.core.inference.text_generation_controllers.text_generation_controller.get_pg_size',
            return_value=tp_size,
        ):
            ctrl._dummy_serial_mtp_forward()

        # Dummy tensors should be padded to tp_size (the smallest valid multiple).
        assert len(captured_shapes) == num_spec
        for shapes in captured_shapes:
            assert shapes['hidden'][0] == tp_size
            assert shapes['tokens'] == (1, tp_size)
            assert shapes['positions'] == (1, tp_size)

    def test_mtp_sp_dummy_hidden_uses_full_seq_len(self):
        """Test that _dummy_serial_mtp_forward passes correctly-sized hidden states.

        Depth 0 receives full-format hidden [padded_count, 1, H] and scatters
        internally.  Subsequent depths receive SP-format hidden from the previous
        depth's output.  Since the mock returns hidden_states unchanged, all
        depths see the same shape in this test.
        """
        tp_size = 4
        num_spec = 2
        self.setup_model(
            torch.float32, static=False, num_speculative_tokens=num_spec, max_requests=4
        )

        ctrl = self.text_generation_controller
        ctrl.num_speculative_tokens = num_spec
        ctrl.num_mtp_heads = num_spec
        ctrl.model_config.expert_model_parallel_size = 2
        # Simulate the SP flag being left on (as it would be on dummy ranks).
        ctrl.model_config.sequence_parallel = True

        unwrapped_model = ctrl.inference_wrapped_model.model
        unwrapped_model._decoder_hidden_states_cache = True

        captured_shapes = []

        def mock_mtp(
            hidden_states,
            next_token_ids,
            position_ids,
            depth,
            scatter_hidden_input=True,
            gather_hidden_output=True,
        ):
            captured_shapes.append(
                {
                    'hidden': hidden_states.shape,
                    'tokens': next_token_ids.shape,
                    'positions': position_ids.shape,
                }
            )
            return hidden_states, torch.randn(
                hidden_states.shape[0], 1, self.vocab_size, device='cuda'
            )

        unwrapped_model.compute_mtp_single_step = mock.MagicMock(side_effect=mock_mtp)

        with mock.patch(
            'megatron.core.inference.text_generation_controllers.text_generation_controller.get_pg_size',
            return_value=tp_size,
        ):
            ctrl._dummy_serial_mtp_forward()

        assert len(captured_shapes) == num_spec
        for shapes in captured_shapes:
            # The mock returns hidden_states unchanged, so all depths see
            # the initial full padded_count.  In real code, depth 1+ would
            # receive SP-format [padded_count/tp_size, 1, H].
            assert shapes['hidden'][0] == tp_size
            assert shapes['tokens'] == (1, tp_size)
            assert shapes['positions'] == (1, tp_size)

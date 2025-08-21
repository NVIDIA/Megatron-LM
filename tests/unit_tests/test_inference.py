import argparse
import unittest.mock

import numpy as np
import pytest
import torch

from megatron.inference.text_generation_server import MegatronServer
from megatron.training import tokenizer
from tests.unit_tests.inference.engines.test_static_engine import TestStaticInferenceEngine
from tests.unit_tests.test_tokenizer import GPT2_VOCAB_SIZE, gpt2_tiktok_vocab
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module")
def gpt2_tiktoken_tokenizer(gpt2_tiktok_vocab):
    return tokenizer.build_tokenizer(gpt2_tiktok_vocab)


@pytest.fixture(scope="module")
def static_inference_engine(gpt2_tiktoken_tokenizer):
    engine_wrapper = TestStaticInferenceEngine()
    engine_wrapper.setup_engine(vocab_size=gpt2_tiktoken_tokenizer.vocab_size)

    controller = engine_wrapper.static_engine.text_generation_controller
    controller.tokenizer = gpt2_tiktoken_tokenizer

    def mock_forward(*args, **kwargs):
        tokens = args[0]
        B, L = tokens.shape
        assert B == 1, "Test assumes batch_size == 1"
        V = gpt2_tiktoken_tokenizer.vocab_size
        next_token_idxs = tokens[0, 1:]
        logits = torch.zeros(1, L, V, dtype=torch.float32, device=tokens.device)
        logits[0, torch.arange(L - 1), next_token_idxs] = 100
        logits[0, -1, gpt2_tiktoken_tokenizer.eos] = 100
        return logits

    controller.inference_wrapped_model.model.forward = mock_forward
    yield engine_wrapper.static_engine


@pytest.fixture(scope="module")
def app(static_inference_engine):
    return MegatronServer(static_inference_engine).app


@pytest.fixture()
def client(app):
    return app.test_client()


@unittest.mock.patch('megatron.inference.endpoints.completions.send_do_generate')
@unittest.mock.patch("megatron.inference.text_generation.tokenization.get_tokenizer")
@unittest.mock.patch("megatron.inference.endpoints.completions.get_tokenizer")
def test_completions_endpoint(
    mock_get_tokenizer1, mock_get_tokenizer2, mock_send_do_generate, client, gpt2_tiktoken_tokenizer
):
    Utils.initialize_distributed()

    mock_get_tokenizer1.return_value = gpt2_tiktoken_tokenizer
    mock_get_tokenizer2.return_value = gpt2_tiktoken_tokenizer

    twinkle = ("twinkle twinkle little star,", " how I wonder what you are")
    request_data = {"prompt": twinkle[0] + twinkle[1], "max_tokens": 0, "logprobs": 5, "echo": True}

    response = client.post('/completions', json=request_data)

    assert response.status_code == 200
    assert response.is_json

    json_data = response.get_json()
    assert 'choices' in json_data
    assert len(json_data['choices']) > 0
    assert 'text' in json_data['choices'][0]
    assert 'logprobs' in json_data['choices'][0]

    # whats up with the reconstruction of the prompt?
    # we are replicating what lm-eval-harness::TemplateLM::_encode_pair does
    # it encodes prompt, then prompt+suffix, and then infers the suffix tokens
    # from the combined encoding.
    logprobs = json_data["choices"][0]["logprobs"]
    num_reconstructed_prompt_tokens = np.searchsorted(logprobs["text_offset"], len(twinkle[0]))
    assert num_reconstructed_prompt_tokens == len(gpt2_tiktoken_tokenizer.tokenize(twinkle[0]))
    suffix_logprob = logprobs["token_logprobs"][num_reconstructed_prompt_tokens:]

    # we mock logits to be 0 everywhere, and 100 at gt tokens, so logprob should be 0 for gt tokens
    assert sum(suffix_logprob) == 0, f"{suffix_logprob} != [0, .... 0]"

    # Test for unsupported HTTP methods
    response = client.put('/completions', json=request_data)
    assert response.status_code == 405  # Method Not Allowed

    mock_send_do_generate.assert_called_once()

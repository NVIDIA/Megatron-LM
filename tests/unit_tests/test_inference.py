import argparse
import unittest.mock

import numpy as np
import pytest
import torch

from megatron.inference.text_generation_server import MegatronServer
from megatron.training import tokenizer
from tests.unit_tests.test_tokenizer import GPT2_VOCAB_SIZE, gpt2_tiktok_vocab
from tests.unit_tests.test_utilities import Utils

logitsT = torch.Tensor


@pytest.fixture
def gpt2_tiktoken_tokenizer(gpt2_tiktok_vocab):
    return tokenizer.build_tokenizer(gpt2_tiktok_vocab)


def forward_step_wrapper(gpt2_tiktoken_tokenizer):
    assert gpt2_tiktoken_tokenizer.vocab_size == GPT2_VOCAB_SIZE

    def mock_forward_step_fn(tokens, position_ids, attention_mask) -> logitsT:
        B, L = tokens.shape
        assert B == 1, "Test assumes batch_size == 1"
        V = gpt2_tiktoken_tokenizer.vocab_size
        next_token_idxs = tokens[0, 1:]
        logits = torch.zeros(1, L, V, dtype=torch.float32, device=tokens.device)
        logits[0, torch.arange(L - 1), next_token_idxs] = 100
        logits[0, -1, gpt2_tiktoken_tokenizer.eos] = 100
        return logits

    return mock_forward_step_fn


@pytest.fixture
def app():
    server = MegatronServer(None)
    return server.app


@pytest.fixture
def client(app):
    return app.test_client()


@unittest.mock.patch('megatron.inference.endpoints.completions.get_tokenizer')
@unittest.mock.patch('megatron.inference.endpoints.completions.send_do_generate')
@unittest.mock.patch('megatron.inference.text_generation.generation.get_args')
@unittest.mock.patch('megatron.inference.text_generation.api.mpu')
@unittest.mock.patch('megatron.inference.text_generation.generation.mpu')
@unittest.mock.patch('megatron.inference.text_generation.communication.mpu')
@unittest.mock.patch('megatron.inference.text_generation.generation.ForwardStep')
@unittest.mock.patch('megatron.inference.text_generation.tokenization.get_tokenizer')
def test_completions(
    mock_get_tokenizer1,
    mock_forward_step,
    mock_mpu_2,
    mock_mpu_1,
    mock_mpu_0,
    mock_get_args_1,
    mock_send_do_generate,
    mock_get_tokenizer2,
    client,
    gpt2_tiktoken_tokenizer,
):
    Utils.initialize_distributed()

    # set up the mocks
    args = argparse.Namespace(
        max_position_embeddings=1024, max_tokens_to_oom=1_000_000, inference_max_seq_length=1024
    )
    mock_get_args_1.return_value = args
    mock_get_tokenizer1.return_value = gpt2_tiktoken_tokenizer
    mock_get_tokenizer2.return_value = gpt2_tiktoken_tokenizer
    mock_forward_step.return_value = forward_step_wrapper(gpt2_tiktoken_tokenizer)
    mock_mpu_0.is_pipeline_last_stage.return_value = True
    mock_mpu_1.is_pipeline_last_stage.return_value = True
    mock_mpu_2.is_pipeline_last_stage.return_value = True

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

    mock_get_tokenizer1.assert_called()
    mock_send_do_generate.assert_called_once()

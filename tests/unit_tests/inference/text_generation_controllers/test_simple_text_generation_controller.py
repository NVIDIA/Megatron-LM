
from collections import OrderedDict
from typing import Dict
import torch
from argparse import Namespace
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.inference_model_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import SimpleTextGenerationController
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from unittest import mock
import pytest
import time

from tests.unit_tests.test_utilities import Utils 

class TestTextGenerationController:

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)        
        self.batch_size = 4
        self.hidden_size = 12
        self.vocab_size = 100
        self.sequence_length = 64
        transformer_config = TransformerConfig(num_layers=4, hidden_size=self.hidden_size, num_attention_heads=4, use_cpu_initialization=True)
                                                    
        gpt_model = GPTModel(
            config=transformer_config, 
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(), 
            vocab_size=self.vocab_size, 
            max_sequence_length=self.sequence_length, 
            parallel_output = False).cuda()
        
        args = Namespace()
        args.hidden_size = self.hidden_size
        args.fp32_residual_connection = False
        args.params_dtype = torch.float
        args.inference_batch_times_seqlen_threshold = 400
        args.padded_vocab_size = self.vocab_size

        inference_wrapped_model = GPTInferenceWrapper(gpt_model, args)

        self.mock_tokenizer = mock.Mock()

        self.text_generation_controller = SimpleTextGenerationController(inference_wrapped_model=inference_wrapped_model, tokenizer=self.mock_tokenizer)


    """
    def test_sample_from_logits(self):
        with pytest.raises(AssertionError) as aerror:
            self.text_generation_controller.sample_from_logits(last_token_logits=None, common_inference_params=CommonInferenceParams(top_k=2, top_p=0.4), vocab_size=self.vocab_size )
        assert str(aerror.value) == 'Cannot have top-p and top-k both greater than zero'

        with pytest.raises(AssertionError) as aerror:
            self.text_generation_controller.sample_from_logits(last_token_logits=None, common_inference_params=CommonInferenceParams(top_p=1.4, top_k=0), vocab_size=self.vocab_size )
        assert str(aerror.value) == 'top-p should be in (0,1]'

        with pytest.raises(AssertionError) as aerror:
            self.text_generation_controller.sample_from_logits(last_token_logits=torch.randn(self.batch_size, 1), common_inference_params=CommonInferenceParams(top_k = self.vocab_size + 10), vocab_size=self.vocab_size)
        assert str(aerror.value) == 'top-k is larger than logit size.'

    
        last_token_logits = torch.arange(0, self.vocab_size).repeat(self.batch_size,1).float().cuda()
        sampled_logits = self.text_generation_controller.sample_from_logits(last_token_logits, CommonInferenceParams(), self.vocab_size)
        assert torch.all(sampled_logits.cpu() == torch.ones(self.batch_size) * self.vocab_size - 1), f"The sampled logits should all be {self.vocab_size} but its {sampled_logits}"

        sampled_logits = self.text_generation_controller.sample_from_logits(last_token_logits, CommonInferenceParams(top_k=2), self.vocab_size)
        assert torch.all(sampled_logits >= self.vocab_size - 2), f"The sampled logits should all be greater than {self.vocab_size-2} but its {sampled_logits}"

        l = last_token_logits[0]
        top_p = 0.3
        expected_min_value = l[l.softmax(dim=-1).cumsum(dim=-1) > top_p][0].item()
        sampled_logits = self.text_generation_controller.sample_from_logits(last_token_logits, CommonInferenceParams(top_p=top_p, top_k=0), self.vocab_size)
        assert torch.all(sampled_logits >= expected_min_value), f"The sampled logits should all be greater than {expected_min_value} but its {sampled_logits}"

        top_p = 0.95
        temperature=2
        expected_min_value = l[l.div_(temperature).softmax(dim=-1).cumsum(dim=-1) > top_p][0].item()
        sampled_logits = self.text_generation_controller.sample_from_logits(last_token_logits, CommonInferenceParams(top_p=top_p, temperature=temperature, top_k=0), self.vocab_size)
        assert torch.all(sampled_logits >= expected_min_value), f"The sampled logits should all be greater than {expected_min_value} but its {sampled_logits}"
    """ 
    def test_generate_all_output_tokens_static_batch(self):
        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.eod = self.vocab_size - 1

        active_requests: Dict[int, InferenceRequest] = OrderedDict()
        for i in range(self.batch_size):
            prompt = "sample" * (i+1)
            self.mock_tokenizer.tokenize.return_value = torch.randn(self.batch_size, self.vocab_size).cuda()   
            inference_request = InferenceRequest(
                request_id=i,
                prompt=prompt,
                inference_parameters=CommonInferenceParams(num_tokens_to_generate=10),
                arrival_time=time.time(),
                prompt_tokens=torch.randint(low=0, high=self.vocab_size - 1, size=(len(prompt),)).tolist(),
                status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS
            )
            active_requests[i] = inference_request

        requests = self.text_generation_controller.generate_all_output_tokens_static_batch(active_requests)
        
        for request_id, request in requests.items():
            assert request.status == Status.COMPLETED, f"Status should be completed but its {request.status}"
            assert request.generated_length > 0 , f"Generated length should be greater than zero"


        
    
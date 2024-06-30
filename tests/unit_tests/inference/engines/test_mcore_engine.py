from argparse import Namespace
from typing import List
import torch
import random 
import string

from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import SimpleTextGenerationController
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from unittest import mock

class TestMCoreEngine:
    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)          
        self.batch_size = 4
        self.hidden_size = 12
        self.vocab_size = 100
        self.sequence_length = 64
        transformer_config = TransformerConfig(num_layers=4, hidden_size=self.hidden_size, num_attention_heads=4, use_cpu_initialization=True)
                                                    
        gpt_model = GPTModel(
            config=transformer_config, 
            transformer_layer_spec=get_gpt_layer_local_spec(), 
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
        text_generation_controller = SimpleTextGenerationController(inference_wrapped_model=inference_wrapped_model, tokenizer=self.mock_tokenizer)       

        self.mcore_engine = MCoreEngine(text_generation_controller=text_generation_controller, max_batch_size=4)

    def test_generate(self):
        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.eod = self.vocab_size - 1
        # Generating random length integer prompts
        self.mock_tokenizer.tokenize.return_value = [random.randint(0, self.vocab_size -1) for _ in range(random.randint(5,10))]
        # Generates some random string
        self.mock_tokenizer.detokenize.return_value = ''.join(random.choices(string.ascii_letters, k=random.randint(4,10)))

        prompts = ["sample"*(i+1) for i in range(self.batch_size)]
        results : List[InferenceRequest] = self.mcore_engine.generate(prompts, common_inference_params=CommonInferenceParams(num_tokens_to_generate=10))

        for result in results:
            assert result.status == Status.COMPLETED, f"Status should be completed but its {result.status}"
            assert result.generated_length > 0 , f"Generated length should be greater than zero"
            assert result.generated_text is not None , f'Generated text should not be None'

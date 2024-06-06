from argparse import Namespace
from typing import List

import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.inference_model_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import SimpleTextGenerationController
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

class TestMCoreEngine:
    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2,pipeline_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)          
        self.batch_size = 4
        self.hidden_size = 12
        self.vocab_size = 100
        self.sequence_length = 32
        transformer_config = TransformerConfig(num_layers=4, hidden_size=self.hidden_size, num_attention_heads=4, use_cpu_initialization=True)
                                                    
        gpt_model = GPTModel(
            config=transformer_config, 
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(), 
            vocab_size=self.vocab_size, 
            max_sequence_length=self.sequence_length)
        
        args = Namespace()
        args.hidden_size = self.hidden_size
        args.fp32_residual_connection = False
        args.params_dtype = torch.float
        args.inference_batch_times_seqlen_threshold = 20

        inference_wrapped_model = GPTInferenceWrapper(gpt_model, args)
        tokenizer = None

        text_generation_controller = SimpleTextGenerationController(inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer)        
        self.mcore_engine = MCoreEngine(text_generation_controller=text_generation_controller, max_batch_size=4)

    def test_generate(self):
        prompts = ["random prompt"*i for i in range(self.batch_size)]
        results : List[InferenceRequest] = self.mcore_engine.generate(prompts, common_inference_params=CommonInferenceParams())

        for result in results:
            assert result.status == Status.COMPLETED, f"Status should be completed but its {result.status}"
            assert result.generated_length > 0 , f"Generated length should be greater than zero"

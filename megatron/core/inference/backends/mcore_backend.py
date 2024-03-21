from typing import List
from megatron.core.inference.backends.abstract_backend import AbstractBackend
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.text_generation_strategies.simple_text_generation_strategy import SimpleTextGenerationStrategy
import torch
from megatron.core import parallel_state

class MCoreBackend(AbstractBackend):
    
    def __init__(self, text_generation_strategy:SimpleTextGenerationStrategy, random_seed:int = None):
        """The Megatron core backend constructor

        This is the backend that does a simple forward pass on the model. Supports any model that is callable (Accepts the inputs and outputs the tensor)

        Args:
            text_generation_strategy (SimpleTextGenerationStrategy): A text generation strategy that will be used to define how to preprocess prompts, generate outputs and detokenizer the output tokens.
            random_seed (int, optional): Use a random seed if you want dterministic results. Defaults to None.
        """

        self.text_generation_strategy = text_generation_strategy
        self.random_seed = random_seed

    def generate(self, prompts:List[str], common_inference_params: CommonInferenceParams):
        
        # TODO :M core- get rng state tracker 
        if self.random_seed :
            torch.random.manual_seed(self.random_seed)
         
        prompts_tokens, prompts_lengths = self.text_generation_strategy.tokenize_and_pad_input_prompts(prompts, common_inference_params.num_tokens_to_generate)

        prompts_tokens_with_generations, generated_sequence_lengths, output_log_probs= self.text_generation_strategy.generate_output_tokens(prompts_tokens, prompts_lengths, common_inference_params)

        # Returns true for both if model is not PP (TODO: Maybe should move this into parallel state ?)
        model_is_not_pipeline_parallel = parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        # Returns the output in the first stage or in all GPUS for TP only models
        if model_is_not_pipeline_parallel or parallel_state.is_pipeline_first_stage():
            prompts_plus_generations_detokenized = self.text_generation_strategy.detokenize_generations(prompts_tokens_with_generations, generated_sequence_lengths)
            output_log_probs = None
            if common_inference_params.return_log_probs:
                output_log_probs = output_log_probs.cpu().numpy().tolist() #TODO: Need to change this
            return prompts_tokens_with_generations, prompts_plus_generations_detokenized, output_log_probs # TODO : Return dictionary 

        else:
            return None, None, None
        
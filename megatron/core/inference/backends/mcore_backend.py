from typing import List
from megatron.core.inference.backends.abstract_backend import AbstractBackend
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.communication_utils import synchronize_params_across_all_ranks
from megatron.core.inference.inference_model_wrappers.abstract_model_inference_wrapper import AbstractModelInferenceWrapper
from megatron.core.inference.text_generation_strategies.abstract_text_generation_strategy import AbstractTextGenerationStrategy
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.inference.text_generation_strategies.simple_text_generation_strategy import SimpleTextGenerationStrategy
import torch
from megatron.core import parallel_state

class MCoreBackend(AbstractBackend):
    def __init__(self, inference_wrapped_model: AbstractModelInferenceWrapper, tokenizer = None, text_generation_strategy:AbstractTextGenerationStrategy = None, random_seed:int = None):
        """The Megatron core backend constructor

        This is the backend that does a simple forward pass on the model. Supports any model that is callable (Accepts the inputs and outputs the tensor)

        Args:
            inference_wrapped_model (callable): A callable instance which returns the output logits
            tokenizer (_type_, optional): The tokenizer used to tokenize and detokenize the prompts. Defaults to None.
            text_generation_strategy (AbstractTextGenerationStrategy, optional): A text generation strategy that will be used to define how to generate the prompts. Defaults to None.
            random_seed (int, optional): Use a random seed if you want dterministic results. Defaults to None.
        """

        self.inference_wrapped_model = inference_wrapped_model
        self.tokenizer = tokenizer
        self.text_generation_strategy = SimpleTextGenerationStrategy(inference_wrapped_model, tokenizer) if text_generation_strategy is None else text_generation_strategy
        self.random_seed = random_seed

    def generate(self, prompts:List[str], common_inference_params: CommonInferenceParams):
        
        #TODO: Maybe can pass this to all gpus? instead of this synchronize ?
        common_inference_params = synchronize_params_across_all_ranks(common_inference_params)

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
        
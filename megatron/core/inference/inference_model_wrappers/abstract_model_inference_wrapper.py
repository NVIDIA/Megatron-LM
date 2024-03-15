from argparse import Namespace
from typing import Iterable, List
import abc

import torch

from megatron.core.inference_params import InferenceParams

class AbstractModelInferenceWrapper:
    def __init__(self, model , args: Namespace):
        """Constructor for the model inference wrapper

        The wrapper is in charge of preparing the model for inference, providing the required in put data and running the forward pass

        Args:
            model (Union[GPTModel, megatron.model.GPTModel]): The actual GPT model (MCore or MLM)
            args (Namespace): The commadline arguments that were passed
        """
        assert not isinstance(model, Iterable), 'interleaving schedule is not supported for inference'
        self.model = model
        self.args = args

    @abc.abstractclassmethod
    def prep_model_for_inference(self,  prompts_tokens: torch.Tensor):
        """A utility function for preparing model for inference

        The function gets called before you get the inference data and running forward pass. Use it to put the model in eval mode, build position ids ,attention mask etc, so that required slices can be extracted during the forward pass. 

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]
        """
        pass

    @abc.abstractclassmethod
    def get_batch_for_context_window(self, context_start_position:int, context_end_position:int) -> List:
        """Returns the inference data given context window

        This function gets called iteratively in a loop . Given the start and end context positions , it extracts the appropriate data. 

        Args:
            context_start_position (int): Start of the context window. During the first inference step it is mostly 0
            context_end_position (int): End of the context window. During the last inference step it will mostly be the max generated sequence length. 

        Returns:
            List: A list of inputs that will be used by your model in the forward step
        """
        pass
    
 
    #TODO : Should maybe use the parallel schedules to do this instead of doing manually
    def __call__(self , inference_input:List) -> torch.Tensor:
        """The forward pass of the model for inference

        Appropriate utility is called for the forward pass depending on the type of model parallelism used

        Args:
            inference_input (List): A list containg the inputs for the gpt model [tokens, position ids, attention mask]
            
        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]. The logits are returned only in the last pipeline stage for PP models. 
        """
        pass
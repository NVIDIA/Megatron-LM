from dataclasses import dataclass

import torch


@dataclass
class InferenceWrapperConfig:
    """Config for the model inference wrapper   

    NOTE : All the arguments here are obtained from arguments.py file
    """

    hidden_size: int
    """Receive happens between the layers during PP with size [seq_len, batch_size, hidden_size]"""

    params_dtype: torch.dtype
    """Can be torch.float or torch.half if --fp16 is used, or torch.bfloat16 if --bf16 is used"""

    inference_batch_times_seqlen_threshold: int
    """if batch-size times sequence-length is smaller than this threshold then we will not use pipelining, otherwise we will."""
    
    padded_vocab_size: int
    """The final padded vocab size (Padded to make it divisible by --make-vocab-size-divisible-by value)"""    
    
    fp32_residual_connection: bool = False
    """Move residual connections to fp32. Obtained from arguments.py"""

    def add_attributes(self, attribute_value_pair: dict):
        """Utility to add more attributes to inference params

        Use this method to pass in a custom dictonary to add more config to the instance you created. Use as follows
        c = InferenceWrapperConfig
        c.add_attributes({'precision':'fp32'})

        Args:
            attribute_value_pair (dict): A dictionary containing attributes as the key names and their values as the values.
        """
        for key, value in attribute_value_pair.items():
            setattr(self, key, value)

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
    """if (batch-size * sequence-length) is smaller than this threshold then we will not pipeline 
    the batch."""

    padded_vocab_size: int
    """The final padded vocab size (Padded to make it divisible by 
    --make-vocab-size-divisible-by value)"""

    inference_max_requests: int = 8
    """ Maximum number of requests for inference (prefill & decode). Necessary for CUDA graphs. """

    inference_max_seq_length: int = 2560
    """ Maximum sequence length for inference (prefill & decode). Necessary for CUDA graphs. """

    fp32_residual_connection: bool = False
    """Move residual connections to fp32. Obtained from arguments.py"""

    nccl_all_reduce_for_prefill: bool = False
    """When using symmetric all reduce kernels we keep the default all reduces for nccl. 
    This can be more effecient for large prefill sizes"""

    def add_attributes(self, attribute_value_pair: dict):
        """Utility to add more attributes to inference params

        Use this method to pass in a custom dictionary to add more configs to the instance created.
        Use as follows:
        c = InferenceWrapperConfig
        c.add_attributes({'precision':'fp32'})

        Args:
            attribute_value_pair (dict): A dictionary containing attributes as the key names and
            corresponding values.
        """
        for key, value in attribute_value_pair.items():
            setattr(self, key, value)

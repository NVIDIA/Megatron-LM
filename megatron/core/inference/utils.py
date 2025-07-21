# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch


class Counter:
    """A simple counter class

    This class is responsible for assigning request ids to incoming requests
    """

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        """Reset counter"""
        self.counter = 0


def get_attention_mask(seq_length: int) -> torch.Tensor:
    """Constructs an attention mask given the input sequence length."""
    attention_mask = torch.tril(
        torch.ones((1, seq_length, seq_length), device=torch.cuda.current_device())
    ).view(1, 1, seq_length, seq_length)

    # Convert to boolean
    attention_mask = attention_mask < 0.5

    return attention_mask

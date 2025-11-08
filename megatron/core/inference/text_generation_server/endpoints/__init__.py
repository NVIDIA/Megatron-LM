# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from .chat_completions import bp as ChatCompletions
from .completions import bp as Completions

__all__ = [Completions, ChatCompletions]

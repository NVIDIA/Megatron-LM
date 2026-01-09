# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


try:
    from .chat_completions import bp as ChatCompletions
    from .completions import bp as Completions

    __all__ = [Completions, ChatCompletions]
except ImportError:
    __all__ = []

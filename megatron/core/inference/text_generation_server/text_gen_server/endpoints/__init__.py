# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


try:
    from .chat_completions import bp as ChatCompletions
    from .completions import bp as Completions
    from .health import bp as Health

    __all__ = [Completions, ChatCompletions, Health]
except ImportError:
    __all__ = []

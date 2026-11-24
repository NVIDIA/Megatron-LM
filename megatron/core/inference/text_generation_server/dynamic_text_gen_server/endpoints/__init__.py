# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


try:
    from .chat_completions import bp as ChatCompletions
    from .completions import bp as Completions
    from .embeddings import bp as Embeddings
    from .health import bp as Health
    from .profile import bp as Profile

    __all__ = [Completions, ChatCompletions, Embeddings, Health, Profile]
except ImportError:
    __all__ = []

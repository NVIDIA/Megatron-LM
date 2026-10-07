# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.inference.utils import Counter, get_language_model


class _Wrapper(torch.nn.Module):
    """Stands in for a multimodal wrapper such as LLaVAModel."""

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model


class TestInferenceUtils:

    def test_counter(self):
        counter = Counter()
        r = next(counter)
        assert r == 0, f'Counter return value should be 0 but it is {r}'
        assert counter.counter == 1, f'Counter should be 1 but it is {counter.counter}'
        counter.reset()
        assert counter.counter == 0, f'Counter should be 0 but it is {counter.counter}'

    def test_get_language_model_returns_a_plain_model_itself(self):
        model = torch.nn.Linear(2, 2)
        assert get_language_model(model) is model

    def test_get_language_model_resolves_a_multimodal_wrapper(self):
        language_model = torch.nn.Linear(2, 2)
        assert get_language_model(_Wrapper(language_model)) is language_model

    def test_get_language_model_keeps_a_wrapper_without_a_language_model(self):
        # e.g. a pipeline stage that builds no decoder.
        wrapper = _Wrapper(None)
        assert get_language_model(wrapper) is wrapper

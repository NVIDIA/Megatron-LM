# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MTP inference finds the MTP block on the language model a multimodal wrapper holds."""

import torch

from megatron.core.inference.text_generation_controllers.mtp_controller_mixin import get_mtp_model


class _LanguageModel(torch.nn.Module):
    def __init__(self, with_mtp: bool):
        super().__init__()
        self.embedding = torch.nn.Identity()
        if with_mtp:
            self.mtp = torch.nn.Identity()


class _Wrapper(torch.nn.Module):
    """Stands in for a multimodal wrapper such as LLaVAModel."""

    def __init__(self, language_model: torch.nn.Module):
        super().__init__()
        self.language_model = language_model


def test_model_with_its_own_mtp_block_is_returned():
    model = _LanguageModel(with_mtp=True)
    assert get_mtp_model(model) is model


def test_wrapper_resolves_to_its_language_model():
    language_model = _LanguageModel(with_mtp=True)
    assert get_mtp_model(_Wrapper(language_model)) is language_model


def test_model_without_an_mtp_block_is_returned_unchanged():
    wrapper = _Wrapper(_LanguageModel(with_mtp=False))
    assert get_mtp_model(wrapper) is wrapper
    model = _LanguageModel(with_mtp=False)
    assert get_mtp_model(model) is model

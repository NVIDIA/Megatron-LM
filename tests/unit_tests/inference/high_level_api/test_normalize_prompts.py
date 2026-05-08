# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.inference._llm_base import _MegatronLLMBase


def _normalize(prompts):
    """Bypass __init__ (which builds the engine pipeline) and call the method
    on a bare instance."""
    obj = _MegatronLLMBase.__new__(_MegatronLLMBase)
    return obj._normalize_prompts(prompts)


class TestNormalizePrompts:
    def test_single_string(self):
        assert _normalize("abc") == (["abc"], False)

    def test_single_token_id_list(self):
        assert _normalize([1, 2, 3]) == ([[1, 2, 3]], False)

    def test_batch_of_strings(self):
        assert _normalize(["a", "b"]) == (["a", "b"], True)

    def test_batch_of_token_id_lists(self):
        assert _normalize([[1, 2], [3, 4]]) == ([[1, 2], [3, 4]], True)

    def test_empty_list_is_batch(self):
        assert _normalize([]) == ([], True)

    @pytest.mark.parametrize(
        "bad_input",
        [
            {1, 2},          # set
            1.5,             # float
            [1.5],           # list of floats (first elem is float)
            {"k": "v"},      # dict
        ],
    )
    def test_unsupported_inputs_raise_typeerror(self, bad_input):
        with pytest.raises(TypeError):
            _normalize(bad_input)

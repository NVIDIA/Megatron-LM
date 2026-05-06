# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings

import pytest

from megatron.core.inference.sampling_params import SamplingParams


class TestSamplingParams:

    def test_default_values(self):
        """Default sampling params match documented defaults."""
        sp = SamplingParams()
        assert sp.temperature == 1.0
        assert sp.top_k == 0
        assert sp.top_p == 0.0
        assert sp.return_log_probs is False
        assert sp.skip_prompt_log_probs is False
        assert sp.return_segments is False
        assert sp.num_tokens_to_generate is None
        assert sp.num_tokens_total is None
        assert sp.termination_id is None
        assert sp.top_n_logprobs == 0
        assert sp.return_prompt_top_n_logprobs is False
        assert sp.add_BOS is False
        assert sp.stop_words is None
        assert sp.detokenize_stop_sequence is False

    def test_custom_init_values_preserved(self):
        """Constructor accepts explicit values for every field."""
        sp = SamplingParams(
            temperature=0.5,
            top_k=10,
            top_p=0.9,
            return_log_probs=True,
            num_tokens_to_generate=20,
            termination_id=42,
            stop_words=["hello"],
            detokenize_stop_sequence=True,
        )
        assert sp.temperature == 0.5
        assert sp.top_k == 10
        assert sp.top_p == 0.9
        assert sp.return_log_probs is True
        assert sp.num_tokens_to_generate == 20
        assert sp.termination_id == 42
        assert sp.stop_words == ["hello"]
        assert sp.detokenize_stop_sequence is True

    def test_top_n_logprobs_enables_prompt_top_n_logprobs(self):
        """When top_n_logprobs > 0 and skip_prompt_log_probs is False, return_prompt_top_n_logprobs becomes True."""
        sp = SamplingParams(top_n_logprobs=5)
        assert sp.return_prompt_top_n_logprobs is True

    def test_top_n_logprobs_with_skip_prompt_disables_prompt_top_n(self):
        """When skip_prompt_log_probs=True, return_prompt_top_n_logprobs stays False."""
        sp = SamplingParams(top_n_logprobs=5, skip_prompt_log_probs=True)
        assert sp.return_prompt_top_n_logprobs is False

    def test_top_n_logprobs_zero_resets_return_prompt_flag(self):
        """When top_n_logprobs == 0, return_prompt_top_n_logprobs is forced to False."""
        sp = SamplingParams(top_n_logprobs=0, return_prompt_top_n_logprobs=False)
        assert sp.return_prompt_top_n_logprobs is False

    def test_deprecated_return_prompt_top_n_logprobs_warns(self):
        """Setting return_prompt_top_n_logprobs=True emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SamplingParams(return_prompt_top_n_logprobs=True, top_n_logprobs=3)
            assert any(issubclass(item.category, DeprecationWarning) for item in w)

    def test_deprecated_field_with_skip_prompt_log_probs_asserts(self):
        """Combining return_prompt_top_n_logprobs=True with skip_prompt_log_probs=True raises an AssertionError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(AssertionError):
                SamplingParams(return_prompt_top_n_logprobs=True, skip_prompt_log_probs=True)

    def test_add_attributes_sets_arbitrary_keys(self):
        """add_attributes adds new attributes to the instance dynamically."""
        sp = SamplingParams()
        sp.add_attributes({"min_tokens": 45, "custom_field": "hello"})
        assert sp.min_tokens == 45
        assert sp.custom_field == "hello"

    def test_add_attributes_resyncs_logprob_flags(self):
        """add_attributes triggers sync of return_prompt_top_n_logprobs."""
        sp = SamplingParams()
        sp.add_attributes({"top_n_logprobs": 4, "skip_prompt_log_probs": False})
        assert sp.return_prompt_top_n_logprobs is True

    def test_serialize_returns_dict_with_all_fields(self):
        """serialize() produces a dict with all dataclass fields."""
        sp = SamplingParams(temperature=0.7, top_k=5)
        data = sp.serialize()
        assert isinstance(data, dict)
        assert data["temperature"] == 0.7
        assert data["top_k"] == 5

    def test_serialize_returns_a_copy(self):
        """Mutating the serialized dict does not change the params."""
        sp = SamplingParams(temperature=0.5)
        data = sp.serialize()
        data["temperature"] = 9.9
        assert sp.temperature == 0.5

    def test_deserialize_round_trip(self):
        """deserialize(serialize(x)) returns a SamplingParams with equivalent fields."""
        sp = SamplingParams(temperature=0.3, top_k=2, top_p=0.7, num_tokens_to_generate=10)
        data = sp.serialize()
        sp2 = SamplingParams.deserialize(data)
        assert sp2.temperature == sp.temperature
        assert sp2.top_k == sp.top_k
        assert sp2.top_p == sp.top_p
        assert sp2.num_tokens_to_generate == sp.num_tokens_to_generate

    def test_deserialize_with_extra_keys_is_supported(self):
        """deserialize accepts dictionaries containing fields beyond the dataclass schema."""
        data = SamplingParams().serialize()
        data["extra_key"] = 123
        sp = SamplingParams.deserialize(data)
        assert sp.extra_key == 123

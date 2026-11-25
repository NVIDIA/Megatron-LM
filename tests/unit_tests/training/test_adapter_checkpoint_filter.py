# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for adapter-only (PEFT) checkpoint filtering. Pure Python, runs on CPU."""

import pytest

from megatron.training.checkpointing import (
    _is_adapter_resume,
    apply_peft_adapter_filter_to_state_dict,
)


def _is_adapter(key: str) -> bool:
    return ".adapter." in key


class TestApplyPeftAdapterFilter:
    def test_model_sections_keep_only_adapter_keys(self):
        state_dict = {
            "model": {"layer.base.weight": 1, "layer.adapter.lora_a": 2},
            "model0": {"layer.base.weight": 3, "layer.adapter.lora_b": 4},
            "model12": {"layer.base.weight": 5, "layer.adapter.lora_c": 6},
        }

        filtered = apply_peft_adapter_filter_to_state_dict(state_dict, _is_adapter)

        assert filtered == {
            "model": {"layer.adapter.lora_a": 2},
            "model0": {"layer.adapter.lora_b": 4},
            "model12": {"layer.adapter.lora_c": 6},
        }

    def test_non_model_sections_are_unchanged(self):
        optimizer = {"state": {0: "moments"}}
        # Names that merely start with "model" are not model sections.
        other = {"layer.base.weight": 1}
        state_dict = {
            "iteration": 7,
            "checkpoint_version": 3.1,
            "optimizer": optimizer,
            "model_extra": other,
            "model": {"layer.base.weight": 1},
        }

        filtered = apply_peft_adapter_filter_to_state_dict(state_dict, _is_adapter)

        assert filtered["iteration"] == 7
        assert filtered["checkpoint_version"] == 3.1
        assert filtered["optimizer"] is optimizer
        assert filtered["model_extra"] is other
        assert filtered["model"] == {}

    def test_router_expert_bias_is_kept_without_adapter_match(self):
        state_dict = {
            "model": {
                "router.expert_bias": 1,
                "decoder.layers.0.mlp.router.expert_bias": 2,
                "decoder.layers.0.mlp.router.weight": 3,
            }
        }

        filtered = apply_peft_adapter_filter_to_state_dict(state_dict, _is_adapter)

        assert filtered["model"] == {
            "router.expert_bias": 1,
            "decoder.layers.0.mlp.router.expert_bias": 2,
        }

    def test_extra_state_is_dropped_even_under_an_adapter(self):
        state_dict = {
            "model": {
                "layer.adapter.lora_a": 1,
                "layer.adapter.linear_in._extra_state": 2,
                "mlp.router.expert_bias._extra_state": 3,
            }
        }

        filtered = apply_peft_adapter_filter_to_state_dict(state_dict, _is_adapter)

        assert filtered["model"] == {"layer.adapter.lora_a": 1}

    def test_input_state_dict_is_not_modified(self):
        model_section = {"layer.base.weight": 1, "layer.adapter.lora_a": 2}
        state_dict = {"model": model_section}

        apply_peft_adapter_filter_to_state_dict(state_dict, _is_adapter)

        assert state_dict == {"model": {"layer.base.weight": 1, "layer.adapter.lora_a": 2}}
        assert state_dict["model"] is model_section


@pytest.mark.parametrize(
    "key_filter, finetune, loading_pretrained, expected",
    [
        (_is_adapter, False, False, True),  # resuming the run's own adapter checkpoint
        (None, False, False, False),  # no adapter filter configured
        (_is_adapter, True, False, False),  # finetune loads weights only
        (_is_adapter, False, True, False),  # loading the pretrained base checkpoint
    ],
)
def test_is_adapter_resume(key_filter, finetune, loading_pretrained, expected):
    assert _is_adapter_resume(key_filter, finetune, loading_pretrained) is expected

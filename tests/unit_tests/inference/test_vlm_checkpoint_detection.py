# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    vlm_dynamic_inference,
)


def test_text_checkpoint_detection_preserves_explicit_model_config(monkeypatch):
    args = SimpleNamespace(
        spec=["dsa_layer_specs", "dsa_stack_spec"],
        tokenizer_model="/local/tokenizer.json",
        use_checkpoint_args=False,
    )

    def load_checkpoint_args(probe):
        probe.spec = ["old_mamba_specs", "mamba_stack_spec"]
        probe.tokenizer_model = "/old_cluster/tokenizer.json"
        return probe, SimpleNamespace(language_model_type=None)

    monkeypatch.setattr(vlm_dynamic_inference, "load_args_from_checkpoint", load_checkpoint_args)
    assert not vlm_dynamic_inference._detect_vlm_from_checkpoint(
        args, user_passed_attrs={"spec", "tokenizer_model"}
    )
    assert args.spec == ["dsa_layer_specs", "dsa_stack_spec"]
    assert args.tokenizer_model == "/local/tokenizer.json"
    assert args.use_checkpoint_args is False


def test_vlm_checkpoint_detection_preserves_cli_precedence(monkeypatch):
    args = SimpleNamespace(spec=["cli", "spec"], language_model_type="cli-language")

    def load_checkpoint_args(probe):
        probe.spec = ["checkpoint", "spec"]
        probe.language_model_type = "checkpoint-language"
        return probe, SimpleNamespace(
            language_model_type="checkpoint-language",
            spec=["checkpoint", "spec"],
            vision_model_type="checkpoint-vision",
        )

    monkeypatch.setattr(vlm_dynamic_inference, "load_args_from_checkpoint", load_checkpoint_args)
    monkeypatch.setattr(vlm_dynamic_inference, "print_rank_0", lambda *a, **kw: None)
    assert vlm_dynamic_inference._detect_vlm_from_checkpoint(
        args, user_passed_attrs={"spec", "language_model_type"}
    )
    assert args.spec == ["cli", "spec"]
    assert args.language_model_type == "cli-language"
    assert args.vision_model_type == "checkpoint-vision"

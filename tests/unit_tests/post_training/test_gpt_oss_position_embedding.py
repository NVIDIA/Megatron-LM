# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPT-OSS uses YaRN only. It must not also allocate a learned position table."""

from argparse import Namespace
from types import SimpleNamespace

import torch
from torch import nn

from megatron.post_training import model_builder as mb


def _gpt_oss_args(**overrides):
    """Args matching examples/post_training/modelopt/conf/openai/gpt-oss-20b.sh.

    That script passes ``--enable-gpt-oss`` and does not pass
    ``--position-embedding-type``, so the CLI default ``learned_absolute`` remains.
    Sequence length is shrunk so the probe embedding stays small.
    """
    args = Namespace(
        enable_gpt_oss=True,
        position_embedding_type="learned_absolute",
        export_model_type="GPTModel",
        export_offline_model=False,
        export_default_te_spec=False,
        export_force_local_attention=True,
        export_te_mcore_model=False,
        export_real_quant_cfg="None",
        export_qk_l2_norm=False,
        export_moe_apply_probs_on_input=False,
        spec=None,
        mtp_num_layers=None,
        padded_vocab_size=128,
        max_position_embeddings=16,
        fp16_lm_cross_entropy=False,
        logit_dtype=None,
        untie_embeddings_and_output_weights=True,
        rotary_percent=1.0,
        rotary_base=150000,
        use_rope_scaling=False,
        rope_scaling_factor=1.0,
        transformer_impl="local",
        load=None,
        export_kd_teacher_load=None,
        freeze_base_for_mtp=False,
        qad_train_target=None,
        hybrid_layer_pattern=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _config():
    return SimpleNamespace(
        heterogeneous_block_specs=False,
        context_parallel_size=1,
        num_layers=2,
        hidden_size=8,
        sequence_parallel=True,
        num_moe_experts=None,
        qk_layernorm=False,
        multi_latent_attention=False,
        experimental_attention_variant=None,
    )


class _ProbeGPT(nn.Module):
    """Reproduce the GPTModel split between rotary type and the embedding table.

    ``GPTModel`` copies ``config.position_embedding_type`` onto itself, then passes
    the constructor argument through to ``LanguageModelEmbedding``. That embedding
    allocates ``position_embeddings`` only when the argument is ``learned_absolute``.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        ctor_type = kwargs["position_embedding_type"]
        if hasattr(config, "position_embedding_type"):
            self.position_embedding_type = config.position_embedding_type
        else:
            self.position_embedding_type = ctor_type
        self.embedding = nn.Module()
        self.embedding.add_position_embedding = ctor_type == "learned_absolute"
        if self.embedding.add_position_embedding:
            self.embedding.position_embeddings = nn.Embedding(
                kwargs["max_sequence_length"], config.hidden_size
            )
        self.constructor_position_embedding_type = ctor_type


class _ProbeHybrid(nn.Module):
    """Reproduce HybridModel: the constructor argument is the only position type."""

    def __init__(self, config, **kwargs):
        super().__init__()
        ctor_type = kwargs["position_embedding_type"]
        self.position_embedding_type = ctor_type
        self.embedding = nn.Module()
        self.embedding.add_position_embedding = ctor_type == "learned_absolute"
        if self.embedding.add_position_embedding:
            self.embedding.position_embeddings = nn.Embedding(
                kwargs["max_sequence_length"], config.hidden_size
            )
        self.constructor_position_embedding_type = ctor_type
        self.decoder = SimpleNamespace(num_layers_per_pipeline_rank=0)


def _patch_builder(monkeypatch, config, model_cls):
    monkeypatch.setattr(mb, "core_transformer_config_from_args", lambda _args: config)
    monkeypatch.setattr(mb, "get_gpt_modelopt_spec", lambda **_kwargs: object())
    monkeypatch.setattr(mb, "get_hybrid_stack_modelopt_spec", lambda **_kwargs: object())
    monkeypatch.setattr(mb, "print_rank_0", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mb, "get_args", lambda: Namespace(export_te_mcore_model=False))
    monkeypatch.setattr(mb, "MCoreGPTModel", model_cls)
    monkeypatch.setattr(mb, "MCoreHybridModel", model_cls)


def _assert_yarn_without_learned_table(model, config):
    assert model.position_embedding_type == "yarn"
    assert model.constructor_position_embedding_type == "yarn"
    assert config.position_embedding_type == "yarn"
    assert config.yarn_rotary_scaling_factor == 32.0
    assert not model.embedding.add_position_embedding
    assert getattr(model.embedding, "position_embeddings", None) is None
    assert not any("position_embeddings" in name for name, _ in model.named_parameters())


def test_gpt_oss_gpt_model_has_no_learned_position_table(monkeypatch):
    """``--enable-gpt-oss`` with the CLI default must not build a position table."""
    config = _config()
    args = _gpt_oss_args()
    _patch_builder(monkeypatch, config, _ProbeGPT)

    model = mb.modelopt_gpt_hybrid_builder(args, pre_process=True, post_process=True)

    _assert_yarn_without_learned_table(model, config)
    # The CLI value stays put. Only the model constructor is moved onto YaRN.
    assert args.position_embedding_type == "learned_absolute"


def test_gpt_oss_hybrid_model_uses_yarn(monkeypatch):
    """HybridModel ignores ``config.position_embedding_type``, so the ctor must be yarn."""
    config = _config()
    args = _gpt_oss_args(export_model_type="HybridModel", hybrid_layer_pattern="M")
    _patch_builder(monkeypatch, config, _ProbeHybrid)

    model = mb.modelopt_gpt_hybrid_builder(args, pre_process=True, post_process=True)

    _assert_yarn_without_learned_table(model, config)


def test_learned_absolute_without_gpt_oss_still_allocates_the_table(monkeypatch):
    """The table remains for a normal learned-absolute export."""
    config = _config()
    args = _gpt_oss_args(enable_gpt_oss=False)
    _patch_builder(monkeypatch, config, _ProbeGPT)

    model = mb.modelopt_gpt_hybrid_builder(args, pre_process=True, post_process=True)

    assert model.constructor_position_embedding_type == "learned_absolute"
    assert model.embedding.add_position_embedding
    assert isinstance(model.embedding.position_embeddings, torch.nn.Embedding)
    weight = model.embedding.position_embeddings.weight
    assert tuple(weight.shape) == (args.max_position_embeddings, config.hidden_size)

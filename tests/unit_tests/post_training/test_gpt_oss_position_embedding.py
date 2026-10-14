# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPT-OSS uses YaRN only. It must not also allocate a learned position table."""

import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.post_training import model_builder as mb

# Published Hugging Face GPT-OSS YaRN factors. Generic YaRN defaults are
# scaling 1, original max 4096, and correction rounding on. These three differ.
_GPT_OSS_YARN = {
    "yarn_rotary_scaling_factor": 32.0,
    "yarn_original_max_position_embeddings": 131072,
    "yarn_beta_fast": 32.0,
    "yarn_beta_slow": 1.0,
    "yarn_mscale": 1.0,
    "yarn_mscale_all_dim": 0.0,
    "yarn_correction_range_round_to_int": False,
}


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
        # CLI default for --rotary-scaling-factor. Not the GPT-OSS factor of 32.
        rotary_scaling_factor=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
        transformer_impl="local",
        load=None,
        export_kd_teacher_load=None,
        export_kd_teacher_model_config=None,
        export_kd_cfg=None,
        manual_gc=False,
        tp_comm_overlap=False,
        cross_entropy_fusion_impl="native",
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        kv_channels=64,
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
        is_hybrid_model=False,
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
        self.config = config
        self._consume_yarn(config)

    def _consume_yarn(self, config):
        """Same unguarded ``yarn_*`` reads as ``GPTModel`` / ``HybridModel``."""
        if self.position_embedding_type != "yarn":
            return
        self.yarn_factors = {name: getattr(config, name) for name in _GPT_OSS_YARN}


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
        self.config = config
        self.decoder = SimpleNamespace(num_layers_per_pipeline_rank=0)
        self._consume_yarn(config)

    def _consume_yarn(self, config):
        if self.position_embedding_type != "yarn":
            return
        self.yarn_factors = {name: getattr(config, name) for name in _GPT_OSS_YARN}


def _patch_imports(monkeypatch, model_cls):
    monkeypatch.setattr(mb, "get_gpt_modelopt_spec", lambda **_kwargs: object())
    monkeypatch.setattr(mb, "get_hybrid_stack_modelopt_spec", lambda **_kwargs: object())
    monkeypatch.setattr(mb, "print_rank_0", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mb, "MCoreGPTModel", model_cls)
    monkeypatch.setattr(mb, "MCoreHybridModel", model_cls)


def _patch_builder(monkeypatch, config, model_cls):
    monkeypatch.setattr(mb, "core_transformer_config_from_args", lambda _args: config)
    _patch_imports(monkeypatch, model_cls)
    monkeypatch.setattr(mb, "get_args", lambda: Namespace(export_te_mcore_model=False))


def _assert_yarn_without_learned_table(model, config):
    assert model.position_embedding_type == "yarn"
    assert model.constructor_position_embedding_type == "yarn"
    assert config.position_embedding_type == "yarn"
    for name, value in _GPT_OSS_YARN.items():
        assert getattr(config, name) == value, name
        assert model.yarn_factors[name] == value, name
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


def _recipe_config_from_args(raw_args, config_class=None):
    """Mirror ``core_transformer_config_from_args`` for the position fields.

    ``TransformerConfig`` has no ``position_embedding_type`` field. Yarn factors
    are attached only when the args already say ``yarn``. The GPT-OSS recipe
    never does, so a teacher rebuilt from those args has neither.
    """
    cfg = _config()
    if getattr(raw_args, "hybrid_layer_pattern", None) is not None:
        cfg.is_hybrid_model = True
    if getattr(raw_args, "position_embedding_type", None) != "yarn":
        return cfg
    if getattr(raw_args, "multi_latent_attention", False):
        return cfg
    cfg.position_embedding_type = "yarn"
    cfg.yarn_rotary_scaling_factor = getattr(raw_args, "rotary_scaling_factor", None) or 1.0
    cfg.yarn_original_max_position_embeddings = (
        getattr(raw_args, "yarn_original_max_position_embeddings", None) or 4096
    )
    cfg.yarn_beta_fast = getattr(raw_args, "yarn_beta_fast", None) or 32.0
    cfg.yarn_beta_slow = getattr(raw_args, "yarn_beta_slow", None) or 1.0
    cfg.yarn_mscale = getattr(raw_args, "mscale", None) or 1.0
    cfg.yarn_mscale_all_dim = getattr(raw_args, "mscale_all_dim", None)
    if cfg.yarn_mscale_all_dim is None:
        cfg.yarn_mscale_all_dim = 0.0
    round_to_int = getattr(raw_args, "yarn_correction_range_round_to_int", None)
    cfg.yarn_correction_range_round_to_int = True if round_to_int is None else round_to_int
    return cfg


def _install_distill_stubs(monkeypatch, seen):
    def _setup(_cfg, student_cfg=None, teacher_cfg=None):
        seen["student_cfg"] = student_cfg
        seen["teacher_cfg"] = teacher_cfg
        return SimpleNamespace(criterion="kl", loss_balancer=None)

    def _convert(model, mode):
        model.teacher_model = mode[0][1]["teacher_model"]
        return model

    monkeypatch.setattr(mb.mtd_mcore, "setup_distillation_config", _setup)
    monkeypatch.setattr(mb.mtd_mcore, "adjust_distillation_model_for_mcore", lambda *_a, **_k: None)
    monkeypatch.setattr(mb.mtd, "convert", _convert)


def _run_distill(
    monkeypatch, root, *, enable_gpt_oss, export_model_type, hybrid_layer_pattern, probe, yaml_text
):
    """Build a student and teacher with ``--export-kd-teacher-load`` set."""
    teacher_dir = Path(root) / "teacher"
    teacher_dir.mkdir()
    if yaml_text is not None:
        (teacher_dir / "model_config.yaml").write_text(yaml_text)
    args = _gpt_oss_args(
        enable_gpt_oss=enable_gpt_oss,
        export_model_type=export_model_type,
        hybrid_layer_pattern=hybrid_layer_pattern,
        export_kd_teacher_load=str(teacher_dir),
    )
    seen = {}
    loaded = {}
    real_load = mb._load_teacher_model_config

    def _spy(path):
        raw = real_load(path)
        loaded["raw"] = raw
        return raw

    monkeypatch.setattr(mb, "core_transformer_config_from_args", _recipe_config_from_args)
    _patch_imports(monkeypatch, probe)
    monkeypatch.setattr(mb, "get_args", lambda: args)
    monkeypatch.setattr(mb, "_load_teacher_model_config", _spy)
    _install_distill_stubs(monkeypatch, seen)
    model = mb.modelopt_gpt_hybrid_builder(args, pre_process=True, post_process=True)
    return model, args, seen, loaded


def _assert_loaded_teacher_still_says_learned_absolute(loaded):
    raw = loaded["raw"]
    assert raw.position_embedding_type == "learned_absolute"
    assert not hasattr(raw, "yarn_rotary_scaling_factor")


def test_gpt_oss_distillation_teacher_gets_yarn_from_recipe_defaults(monkeypatch):
    """Teacher args stay ``learned_absolute`` and still have to build YaRN.

    ``--export-kd-teacher-load`` rebuilds ``teacher_config`` from the checkpoint
    args. The recipe default recorded there is ``learned_absolute`` and the file
    has no ``yarn_*`` fields. The teacher is then constructed with the student's
    kwargs, which request yarn.
    """
    yaml_text = "position_embedding_type: learned_absolute\n"
    with tempfile.TemporaryDirectory() as root:
        model, args, seen, loaded = _run_distill(
            monkeypatch,
            root,
            enable_gpt_oss=True,
            export_model_type="GPTModel",
            hybrid_layer_pattern=None,
            probe=_ProbeGPT,
            yaml_text=yaml_text,
        )

    _assert_loaded_teacher_still_says_learned_absolute(loaded)
    assert args.position_embedding_type == "learned_absolute"
    _assert_yarn_without_learned_table(model, seen["student_cfg"])
    _assert_yarn_without_learned_table(model.teacher_model, seen["teacher_cfg"])
    assert seen["teacher_cfg"] is not seen["student_cfg"]
    assert model.teacher_model.config is seen["teacher_cfg"]


def test_gpt_oss_distillation_teacher_without_config_file_gets_yarn(monkeypatch):
    """No ``model_config.yaml`` means the teacher inherits the student args."""
    with tempfile.TemporaryDirectory() as root:
        model, args, seen, loaded = _run_distill(
            monkeypatch,
            root,
            enable_gpt_oss=True,
            export_model_type="GPTModel",
            hybrid_layer_pattern=None,
            probe=_ProbeGPT,
            yaml_text=None,
        )

    _assert_loaded_teacher_still_says_learned_absolute(loaded)
    assert args.position_embedding_type == "learned_absolute"
    _assert_yarn_without_learned_table(model.teacher_model, seen["teacher_cfg"])
    assert seen["teacher_cfg"] is not seen["student_cfg"]


def test_gpt_oss_hybrid_distillation_teacher_gets_yarn(monkeypatch):
    """HybridModel reads the constructor type, so the teacher kwargs must be yarn."""
    yaml_text = "position_embedding_type: learned_absolute\n"
    with tempfile.TemporaryDirectory() as root:
        model, _args, seen, loaded = _run_distill(
            monkeypatch,
            root,
            enable_gpt_oss=True,
            export_model_type="HybridModel",
            hybrid_layer_pattern="M",
            probe=_ProbeHybrid,
            yaml_text=yaml_text,
        )

    _assert_loaded_teacher_still_says_learned_absolute(loaded)
    assert loaded["raw"].hybrid_layer_pattern == "M"
    assert seen["teacher_cfg"].is_hybrid_model
    _assert_yarn_without_learned_table(model, seen["student_cfg"])
    _assert_yarn_without_learned_table(model.teacher_model, seen["teacher_cfg"])


def test_teacher_gpt_oss_flag_does_not_put_yarn_on_the_student(monkeypatch):
    """A teacher-only ``enable_gpt_oss`` must not change the student embedding."""
    yaml_text = "enable_gpt_oss: true\nposition_embedding_type: learned_absolute\n"
    with tempfile.TemporaryDirectory() as root:
        model, args, seen, loaded = _run_distill(
            monkeypatch,
            root,
            enable_gpt_oss=False,
            export_model_type="GPTModel",
            hybrid_layer_pattern=None,
            probe=_ProbeGPT,
            yaml_text=yaml_text,
        )

    assert args.enable_gpt_oss is False
    assert loaded["raw"].enable_gpt_oss is True
    assert args.position_embedding_type == "learned_absolute"
    assert model.constructor_position_embedding_type == "learned_absolute"
    assert model.embedding.add_position_embedding
    assert not hasattr(seen["student_cfg"], "yarn_rotary_scaling_factor")
    _assert_yarn_without_learned_table(model.teacher_model, seen["teacher_cfg"])


def test_learned_absolute_distillation_teacher_keeps_the_table(monkeypatch):
    """Distillation without GPT-OSS still allocates the learned position table."""
    yaml_text = "position_embedding_type: learned_absolute\n"
    with tempfile.TemporaryDirectory() as root:
        model, _args, seen, loaded = _run_distill(
            monkeypatch,
            root,
            enable_gpt_oss=False,
            export_model_type="GPTModel",
            hybrid_layer_pattern=None,
            probe=_ProbeGPT,
            yaml_text=yaml_text,
        )

    _assert_loaded_teacher_still_says_learned_absolute(loaded)
    assert model.constructor_position_embedding_type == "learned_absolute"
    assert model.teacher_model.constructor_position_embedding_type == "learned_absolute"
    assert model.embedding.add_position_embedding
    assert model.teacher_model.embedding.add_position_embedding
    assert not hasattr(seen["teacher_cfg"], "position_embedding_type")
    assert not hasattr(seen["teacher_cfg"], "yarn_rotary_scaling_factor")


def _forward_logits(model):
    sequence_length = model.max_sequence_length
    batch = 2
    data = list(range(sequence_length))
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((batch, 1)).cuda()
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((batch, 1)).cuda()
    attention_mask = torch.ones(
        (batch, 1, sequence_length, sequence_length), dtype=torch.bool
    ).cuda()
    logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
    return logits


def test_gpt_oss_distillation_teacher_builds_real_gpt_model(monkeypatch):
    """The teacher path must construct a real GPTModel, not only the probe.

    ``YarnRotaryEmbedding`` reads ``config.yarn_rotary_scaling_factor`` with no
    default. A teacher config rebuilt from ``learned_absolute`` args used to die
    there. This runs that constructor and a forward/backward on CUDA.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to build YarnRotaryEmbedding")

    from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    created = []

    def _tiny_config(raw_args, config_class=None):
        assert raw_args.position_embedding_type == "learned_absolute"
        cfg = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, use_cpu_initialization=True
        )
        assert not hasattr(cfg, "yarn_rotary_scaling_factor")
        created.append(cfg)
        return cfg

    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        with tempfile.TemporaryDirectory() as root:
            teacher_dir = Path(root) / "teacher"
            teacher_dir.mkdir()
            (teacher_dir / "model_config.yaml").write_text(
                "position_embedding_type: learned_absolute\n"
            )
            args = _gpt_oss_args(
                export_kd_teacher_load=str(teacher_dir),
                max_position_embeddings=8,
                padded_vocab_size=128,
            )
            seen = {}
            monkeypatch.setattr(mb, "core_transformer_config_from_args", _tiny_config)
            monkeypatch.setattr(mb, "get_args", lambda: args)
            monkeypatch.setattr(mb, "print_rank_0", lambda *_a, **_k: None)
            _install_distill_stubs(monkeypatch, seen)
            model = mb.modelopt_gpt_hybrid_builder(args, pre_process=True, post_process=True)

        teacher = model.teacher_model
        assert len(created) == 2
        assert created[0] is model.config
        assert created[1] is teacher.config
        assert created[0] is not created[1]
        for built in (model, teacher):
            assert built.position_embedding_type == "yarn"
            assert isinstance(built.rotary_pos_emb, YarnRotaryEmbedding)
            rope = built.rotary_pos_emb
            assert rope.scaling_factor == 32.0
            assert rope.original_max_position_embeddings == 131072
            assert rope.beta_fast == 32.0
            assert rope.beta_slow == 1.0
            assert rope.mscale == 1.0
            assert rope.mscale_all_dim == 0.0
            assert rope.correction_range_round_to_int is False
            assert rope.rotary_base == 150000
            assert not hasattr(built.embedding, "position_embeddings")
            built.cuda()
            logits = _forward_logits(built)
            assert logits.shape[0] == 2
            assert logits.shape[1] == 8
            assert logits.shape[2] == 128
            built.zero_grad(set_to_none=True)
            logits.float().sum().backward()
            weight = built.embedding.word_embeddings.weight
            grad = weight.grad
            if grad is None:
                grad = getattr(weight, "main_grad", None)
            assert grad is not None
            assert torch.isfinite(grad).all()
    finally:
        Utils.destroy_model_parallel()

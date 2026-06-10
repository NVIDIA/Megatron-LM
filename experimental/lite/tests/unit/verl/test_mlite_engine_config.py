from types import SimpleNamespace

from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import MegatronLiteEngine


def _optimizer_config(**override_optimizer_config) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1e-6,
        min_lr=None,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.1,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=10,
        lr_warmup_steps=0,
        lr_warmup_init=0.0,
        lr_decay_steps=None,
        lr_decay_style="constant",
        weight_decay_incr_style="constant",
        lr_wsd_decay_style="exponential",
        lr_wsd_decay_steps=None,
        use_checkpoint_opt_param_scheduler=False,
        betas=(0.9, 0.95),
        override_optimizer_config=override_optimizer_config,
    )


def _engine(
    *, engine_config: MegatronLiteEngineConfig, optimizer_config: SimpleNamespace | None = None
) -> MegatronLiteEngine:
    return MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=engine_config,
        optimizer_config=optimizer_config or _optimizer_config(),
        checkpoint_config={},
    )


def _engine_config(**kwargs) -> MegatronLiteEngineConfig:
    values = {"custom_backend_module": None, "impl_cfg": {"use_thd": True}}
    values.update(kwargs)
    return MegatronLiteEngineConfig(**values)


def test_optimizer_offload_enables_full_optimizer_state_offload_by_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(
            use_precision_aware_optimizer=True, decoupled_weight_decay=True
        ),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0
    assert optimizer.use_precision_aware_optimizer is True
    assert optimizer.decoupled_weight_decay is True
    assert optimizer.adam_beta1 == 0.9
    assert optimizer.adam_beta2 == 0.95


def test_explicit_optimizer_offload_fraction_overrides_engine_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(offload_fraction=0.25),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 0.25


def test_optimizer_cpu_offload_alias_maps_to_full_offload_fraction() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=False),
        optimizer_config=_optimizer_config(optimizer_cpu_offload=True),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0


def test_mlite_config_threads_rl_parallel_and_impl_settings() -> None:
    engine = _engine(
        engine_config=_engine_config(
            tp=2,
            ep=8,
            etp=1,
            pp=1,
            cp=1,
            optimizer_offload=True,
            attention_backend_override="flash",
            impl_cfg={"use_thd": True, "deterministic": False},
        )
    )

    config = engine._build_mlite_config()

    assert config.model_name == "qwen3_5"
    assert config.impl == "lite"
    assert config.parallel.tp == 2
    assert config.parallel.ep == 8
    assert config.parallel.etp == 1
    assert config.optimizer.offload_fraction == 1.0
    assert config.attention_backend_override == "flash"
    assert config.impl_cfg["use_thd"] is True
    assert config.impl_cfg["deterministic"] is False

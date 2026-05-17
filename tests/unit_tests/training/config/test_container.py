# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for PretrainConfigContainer validation and sub-config finalize() methods."""

from typing import Any
from unittest.mock import patch

import pytest
import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend, CudaGraphScope
from megatron.training.config.common_config import DistributedInitConfig, ProfilingConfig, RNGConfig
from megatron.training.config.container import (
    PretrainConfigContainer,
    validate_flex_dispatcher_backend,
)
from megatron.training.config.resilience_config import RerunStateMachineConfig
from megatron.training.config.training_config import (
    CheckpointConfig,
    LoggerConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.training.models.hybrid import HybridModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_world_size(monkeypatch, world_size: int) -> None:
    """Replace the imported safe_get_world_size in the container module with a stub."""
    monkeypatch.setattr(
        "megatron.training.config.container.safe_get_world_size", lambda: world_size
    )


def create_test_transformer_config(**kwargs: Any) -> TransformerConfig:
    """Minimal TransformerConfig that survives finalize()."""
    defaults: dict[str, Any] = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "apply_rope_fusion": False,
    }
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def create_test_hybrid_model_config(
    transformer: TransformerConfig | None = None, **kwargs: Any
) -> HybridModelConfig:
    """Minimal HybridModelConfig for container tests.

    Any kwarg matching a TransformerConfig field is forwarded to the embedded
    transformer config (HybridModelConfig proxies attribute access to it).
    """
    transformer_kwargs: dict[str, Any] = {}
    hybrid_kwargs: dict[str, Any] = {}
    transformer_field_names = {f for f in TransformerConfig.__dataclass_fields__}
    # ModelParallelConfig fields (parent of TransformerConfig) also live on transformer
    from megatron.core.model_parallel_config import ModelParallelConfig

    transformer_field_names |= {f for f in ModelParallelConfig.__dataclass_fields__}
    for k, v in kwargs.items():
        if k in transformer_field_names:
            transformer_kwargs[k] = v
        else:
            hybrid_kwargs[k] = v

    if transformer is None:
        transformer = create_test_transformer_config(**transformer_kwargs)
    elif transformer_kwargs:
        for k, v in transformer_kwargs.items():
            setattr(transformer, k, v)

    return HybridModelConfig(transformer=transformer, **hybrid_kwargs)


def create_test_training_config(**kwargs: Any) -> TrainingConfig:
    defaults: dict[str, Any] = {"global_batch_size": 32, "micro_batch_size": 1, "train_iters": 1000}
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def create_test_optimizer_config(**kwargs: Any) -> OptimizerConfig:
    defaults: dict[str, Any] = {"lr": 0.0001, "use_distributed_optimizer": False}
    defaults.update(kwargs)
    return OptimizerConfig(**defaults)


def create_test_scheduler_config(**kwargs: Any) -> SchedulerConfig:
    defaults: dict[str, Any] = {"lr_decay_style": "linear", "lr_warmup_iters": 0}
    defaults.update(kwargs)
    return SchedulerConfig(**defaults)


def create_test_logger_config(**kwargs: Any) -> LoggerConfig:
    return LoggerConfig(**kwargs)


def create_test_checkpoint_config(**kwargs: Any) -> CheckpointConfig:
    defaults: dict[str, Any] = {"ckpt_format": "torch_dist"}
    defaults.update(kwargs)
    return CheckpointConfig(**defaults)


def create_test_distributed_init_config(**kwargs: Any) -> DistributedInitConfig:
    defaults: dict[str, Any] = {"use_gloo_process_groups": True, "lazy_mpu_init": False}
    defaults.update(kwargs)
    return DistributedInitConfig(**defaults)


def create_test_ddp_config(**kwargs: Any) -> DistributedDataParallelConfig:
    return DistributedDataParallelConfig(**kwargs)


def create_test_pretrain_container(
    *,
    model: HybridModelConfig | None = None,
    train: TrainingConfig | None = None,
    optimizer: OptimizerConfig | None = None,
    scheduler: SchedulerConfig | None = None,
    logger: LoggerConfig | None = None,
    checkpoint: CheckpointConfig | None = None,
    dist: DistributedInitConfig | None = None,
    ddp: DistributedDataParallelConfig | None = None,
    rng: RNGConfig | None = None,
    profiling: ProfilingConfig | None = None,
    validation: ValidationConfig | None = None,
    rerun_state_machine: RerunStateMachineConfig | None = None,
) -> PretrainConfigContainer:
    """Construct a PretrainConfigContainer that survives a default validate() call."""
    return PretrainConfigContainer(
        train=train or create_test_training_config(),
        validation=validation or ValidationConfig(),
        model=model or create_test_hybrid_model_config(),
        optimizer=optimizer or create_test_optimizer_config(),
        scheduler=scheduler or create_test_scheduler_config(),
        ddp=ddp or create_test_ddp_config(),
        dist=dist or create_test_distributed_init_config(),
        rng=rng or RNGConfig(),
        logger=logger or create_test_logger_config(),
        checkpoint=checkpoint or create_test_checkpoint_config(),
        profiling=profiling if profiling is not None else ProfilingConfig(),
        rerun_state_machine=rerun_state_machine or RerunStateMachineConfig(),
    )


# ---------------------------------------------------------------------------
# Sub-config finalize() tests
# ---------------------------------------------------------------------------


class TestProfilingFinalize:
    def test_default_finalize_passes(self):
        ProfilingConfig().finalize()

    def test_pytorch_and_nsys_mutually_exclusive(self):
        cfg = ProfilingConfig(use_pytorch_profiler=True, use_nsys_profiler=True)
        with pytest.raises(AssertionError, match="Exactly one of pytorch or nsys"):
            cfg.finalize()

    def test_negative_step_start_rejected(self):
        cfg = ProfilingConfig(profile_step_start=-1)
        with pytest.raises(AssertionError, match="profile_step_start must be >= 0"):
            cfg.finalize()

    def test_negative_step_end_rejected(self):
        cfg = ProfilingConfig(profile_step_end=-1)
        with pytest.raises(AssertionError, match="profile_step_end must be >= 0"):
            cfg.finalize()

    def test_step_end_before_start_rejected(self):
        cfg = ProfilingConfig(profile_step_start=10, profile_step_end=5)
        with pytest.raises(
            AssertionError, match="profile_step_end .* must be >= profile_step_start"
        ):
            cfg.finalize()


class TestTrainingFinalize:
    def test_iters_only_passes(self):
        cfg = create_test_training_config(train_iters=10, train_samples=None)
        cfg.finalize()
        assert cfg.train_iters == 10

    def test_neither_iters_nor_samples_rejected(self):
        cfg = TrainingConfig(global_batch_size=32, micro_batch_size=1)
        with pytest.raises(AssertionError, match="Either train_iters or train_samples"):
            cfg.finalize()

    def test_both_iters_and_samples_rejected(self):
        cfg = TrainingConfig(
            global_batch_size=32, micro_batch_size=1, train_iters=100, train_samples=1000
        )
        with pytest.raises(AssertionError, match="Cannot specify both"):
            cfg.finalize()

    def test_samples_derives_train_iters(self):
        cfg = TrainingConfig(global_batch_size=32, micro_batch_size=1, train_samples=320)
        cfg.finalize()
        # 320 / 32 == 10
        assert cfg.train_iters == 10

    def test_samples_must_be_positive(self):
        cfg = TrainingConfig(global_batch_size=32, micro_batch_size=1, train_samples=0)
        with pytest.raises(AssertionError, match="train_samples must be positive"):
            cfg.finalize()

    def test_samples_disallows_rampup(self):
        cfg = TrainingConfig(
            global_batch_size=32,
            micro_batch_size=1,
            train_samples=1000,
            rampup_batch_size=[8, 8, 100],
        )
        with pytest.raises(AssertionError, match="Batch size rampup not supported"):
            cfg.finalize()

    def test_samples_requires_global_batch_size(self):
        cfg = TrainingConfig(global_batch_size=None, micro_batch_size=1, train_samples=1000)
        with pytest.raises(AssertionError, match="global_batch_size must be set"):
            cfg.finalize()


class TestSchedulerFinalize:
    def test_default_finalize_passes(self):
        create_test_scheduler_config().finalize()

    def test_negative_start_weight_decay_rejected(self):
        cfg = create_test_scheduler_config(start_weight_decay=-0.1, end_weight_decay=0.0)
        with pytest.raises(AssertionError, match="start_weight_decay should be positive"):
            cfg.finalize()

    def test_end_less_than_start_weight_decay_rejected(self):
        cfg = create_test_scheduler_config(start_weight_decay=0.2, end_weight_decay=0.1)
        with pytest.raises(AssertionError):
            cfg.finalize()

    def test_override_and_use_checkpoint_mutex(self):
        cfg = create_test_scheduler_config(
            override_opt_param_scheduler=True, use_checkpoint_opt_param_scheduler=True
        )
        with pytest.raises(AssertionError, match="both override and use-checkpoint"):
            cfg.finalize()

    def test_iter_and_sample_fields_mutex(self):
        cfg = create_test_scheduler_config(lr_decay_iters=100, lr_decay_samples=1000)
        with pytest.raises(AssertionError, match="Cannot mix iteration-based and sample-based"):
            cfg.finalize()

    def test_lr_warmup_fraction_with_iters_mutex(self):
        cfg = create_test_scheduler_config(lr_warmup_fraction=0.1, lr_warmup_iters=10)
        with pytest.raises(AssertionError, match="Cannot specify lr_warmup_fraction"):
            cfg.finalize()

    def test_lr_warmup_fraction_with_samples_mutex(self):
        cfg = create_test_scheduler_config(
            lr_warmup_fraction=0.1, lr_warmup_iters=0, lr_warmup_samples=10
        )
        with pytest.raises(AssertionError, match="Cannot specify lr_warmup_fraction"):
            cfg.finalize()


class TestCheckpointFinalize:
    def test_default_finalize_passes_and_sets_mcore(self):
        cfg = create_test_checkpoint_config()
        cfg.finalize()
        # async_save defaults to False, so finalize() forces async_strategy="mcore"
        assert cfg.async_strategy == "mcore"

    def test_invalid_async_strategy_rejected(self):
        cfg = create_test_checkpoint_config(async_strategy="invalid")
        with pytest.raises(AssertionError, match="async_strategy invalid is not supported"):
            cfg.finalize()

    def test_pretrained_checkpoint_must_exist(self):
        cfg = create_test_checkpoint_config(pretrained_checkpoint="/does/not/exist/at/all")
        with pytest.raises(AssertionError, match="does not exist"):
            cfg.finalize()

    def test_pretrained_checkpoint_existing_path_passes(self, tmp_path):
        existing = tmp_path / "fake_ckpt"
        existing.touch()
        cfg = create_test_checkpoint_config(pretrained_checkpoint=str(existing))
        cfg.finalize()

    def test_load_main_params_requires_no_load_optim(self):
        cfg = create_test_checkpoint_config(load_main_params_from_ckpt=True, load_optim=True)
        with pytest.raises(AssertionError, match="load_main_params_from_ckpt must be used with"):
            cfg.finalize()

    def test_async_save_requires_save_path(self):
        cfg = create_test_checkpoint_config(
            async_save=True, save=None, use_persistent_ckpt_worker=True
        )
        with pytest.raises(AssertionError, match="async_save is enabled, but save is not set"):
            cfg.finalize()

    def test_async_save_requires_persistent_worker(self, tmp_path):
        cfg = create_test_checkpoint_config(
            async_save=True, save=str(tmp_path), use_persistent_ckpt_worker=False
        )
        with pytest.raises(AssertionError, match="async_save requires use_persistent_ckpt_worker"):
            cfg.finalize()

    def test_async_save_requires_compatible_format(self, tmp_path):
        cfg = create_test_checkpoint_config(
            async_save=True,
            save=str(tmp_path),
            use_persistent_ckpt_worker=True,
            ckpt_format="torch",
        )
        with pytest.raises(AssertionError, match="async_save is only supported with"):
            cfg.finalize()

    def test_verify_integrity_requires_torch_dist(self):
        cfg = create_test_checkpoint_config(verify_integrity=True, ckpt_format="torch")
        with pytest.raises(
            AssertionError, match="verify_integrity.* only supported with torch_dist"
        ):
            cfg.finalize()

    def test_ckpt_step_requires_load(self):
        cfg = create_test_checkpoint_config(ckpt_step=42, load=None)
        with pytest.raises(ValueError, match="ckpt_step=42 specified but checkpoint.load is None"):
            cfg.finalize()

    def test_dist_ckpt_optim_fully_reshardable_excludes_mem_efficient(self):
        cfg = create_test_checkpoint_config(
            dist_ckpt_optim_fully_reshardable=True,
            distrib_optim_fully_reshardable_mem_efficient=True,
        )
        with pytest.raises(AssertionError, match="distrib_optim_fully_reshardable_mem_efficient"):
            cfg.finalize()


# ---------------------------------------------------------------------------
# get_data_parallel_size / set_data_parallel_size
# ---------------------------------------------------------------------------


class TestGetDataParallelSize:
    @pytest.mark.parametrize(
        "world_size, tp, pp, cp, expected",
        [(8, 1, 1, 1, 8), (8, 2, 2, 1, 2), (16, 2, 2, 2, 2), (1, 1, 1, 1, 1)],
    )
    def test_divides_world_by_total_model_size(self, world_size, tp, pp, cp, expected):
        model = create_test_hybrid_model_config(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            num_attention_heads=4,
        )
        container = create_test_pretrain_container(model=model)
        assert container.get_data_parallel_size(world_size) == expected

    def test_world_size_not_divisible_raises(self):
        model = create_test_hybrid_model_config(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(AssertionError, match="not divisible by"):
            container.get_data_parallel_size(7)

    def test_set_data_parallel_size_uses_safe_get_world_size(self, monkeypatch):
        model = create_test_hybrid_model_config()
        container = create_test_pretrain_container(model=model)
        _patch_world_size(monkeypatch, 4)
        container.set_data_parallel_size()
        assert container.data_parallel_size == 4


# ---------------------------------------------------------------------------
# Top-level validate()
# ---------------------------------------------------------------------------


class TestValidateBasic:
    def test_default_validate_succeeds(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        container = create_test_pretrain_container()
        container.validate()
        assert container.data_parallel_size == 1

    def test_te_rng_tracker_sync_from_rng(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        rng = RNGConfig(te_rng_tracker=True)
        container = create_test_pretrain_container(rng=rng)
        container.validate()
        assert container.model.use_te_rng_tracker is True
        assert container.rng.te_rng_tracker is True

    def test_te_rng_tracker_sync_from_model(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        model = create_test_hybrid_model_config(use_te_rng_tracker=True)
        container = create_test_pretrain_container(model=model)
        container.validate()
        assert container.model.use_te_rng_tracker is True
        assert container.rng.te_rng_tracker is True

    def test_lazy_mpu_init_forces_cpu_initialization(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        dist = create_test_distributed_init_config(lazy_mpu_init=True)
        model = create_test_hybrid_model_config(use_cpu_initialization=False)
        container = create_test_pretrain_container(model=model, dist=dist)
        container.validate()
        assert container.model.use_cpu_initialization is True


# ---------------------------------------------------------------------------
# Eval batch size resolution
# ---------------------------------------------------------------------------


class TestEvalBatchSize:
    def test_defaults_resolved_from_train_config(self, monkeypatch):
        _patch_world_size(monkeypatch, 8)
        train = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        container = create_test_pretrain_container(train=train)
        container.validate()
        assert container.validation.eval_global_batch_size == 64
        assert container.validation.eval_micro_batch_size == 4

    def test_explicit_overrides_preserved(self, monkeypatch):
        _patch_world_size(monkeypatch, 8)
        train = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        validation = ValidationConfig(eval_global_batch_size=16, eval_micro_batch_size=2)
        container = create_test_pretrain_container(train=train, validation=validation)
        container.validate()
        assert container.validation.eval_global_batch_size == 16
        assert container.validation.eval_micro_batch_size == 2

    @pytest.mark.parametrize(
        "eval_gbs, eval_mbs, world_size, expect_error",
        [
            (32, 4, 8, False),  # 32 / (4*8) = 1
            (64, 8, 8, False),  # 64 / (8*8) = 1
            (32, 3, 8, True),  # 32 / (3*8) not integer
            (15, 2, 8, True),  # 15 / (2*8) not integer
        ],
    )
    def test_divisibility(self, monkeypatch, eval_gbs, eval_mbs, world_size, expect_error):
        _patch_world_size(monkeypatch, world_size)
        train = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        validation = ValidationConfig(
            eval_global_batch_size=eval_gbs, eval_micro_batch_size=eval_mbs
        )
        container = create_test_pretrain_container(train=train, validation=validation)
        if expect_error:
            with pytest.raises(AssertionError, match="must be divisible by"):
                container.validate()
        else:
            container.validate()


# ---------------------------------------------------------------------------
# Megatron-FSDP validation
# ---------------------------------------------------------------------------


class TestMegatronFsdpValidation:
    def _container_with_fsdp(self, **kwargs):
        ddp = create_test_ddp_config(use_megatron_fsdp=True, **kwargs)
        dist = create_test_distributed_init_config(use_megatron_fsdp=True)
        return create_test_pretrain_container(ddp=ddp, dist=dist)

    def test_fsdp_and_torch_fsdp2_mutex(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        dist = create_test_distributed_init_config(use_megatron_fsdp=True, use_torch_fsdp2=True)
        container = create_test_pretrain_container(dist=dist)
        with pytest.raises(
            ValueError, match="use_megatron_fsdp and use_torch_fsdp2 are mutually exclusive"
        ):
            container.validate()

    def test_fsdp_auto_enables_distributed_optimizer(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
        opt = create_test_optimizer_config(use_distributed_optimizer=False)
        chkpt = create_test_checkpoint_config(ckpt_format="fsdp_dtensor")
        container = self._container_with_fsdp()
        container.optimizer = opt
        container.checkpoint = chkpt
        container.validate()
        assert container.ddp.use_distributed_optimizer is True
        assert container.optimizer.use_distributed_optimizer is True

    def test_fsdp_requires_fsdp_dtensor_when_saving(self, monkeypatch, tmp_path):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
        chkpt = create_test_checkpoint_config(ckpt_format="torch_dist", save=str(tmp_path))
        container = self._container_with_fsdp()
        container.checkpoint = chkpt
        with pytest.raises(AssertionError, match="Megatron-FSDP requires the fsdp_dtensor"):
            container.validate()

    def test_fsdp_skips_ckpt_format_check_when_no_save_or_load(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
        # save=None and load=None => skip ckpt_format guard
        chkpt = create_test_checkpoint_config(ckpt_format="torch_dist", save=None, load=None)
        container = self._container_with_fsdp()
        container.checkpoint = chkpt
        container.validate()

    def test_fsdp_rejects_cuda_device_max_connections_one(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        chkpt = create_test_checkpoint_config(ckpt_format="fsdp_dtensor")
        container = self._container_with_fsdp()
        container.checkpoint = chkpt
        with pytest.raises(AssertionError, match="CUDA_DEVICE_MAX_CONNECTIONS"):
            container.validate()

    def test_fsdp_nccl_ub_enables_manual_registration(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
        chkpt = create_test_checkpoint_config(ckpt_format="fsdp_dtensor")
        container = self._container_with_fsdp(nccl_ub=True, fsdp_manual_registration=False)
        container.checkpoint = chkpt
        container.validate()
        assert container.ddp.fsdp_manual_registration is True

    def test_fsdp_manual_registration_requires_nccl_ub(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
        chkpt = create_test_checkpoint_config(ckpt_format="fsdp_dtensor")
        container = self._container_with_fsdp(nccl_ub=False, fsdp_manual_registration=True)
        container.checkpoint = chkpt
        with pytest.raises(AssertionError, match="fsdp_manual_registration requires DDP.nccl_ub"):
            container.validate()

    def test_fsdp_optim_grads_params_blocks_check_weight_hash(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
        chkpt = create_test_checkpoint_config(ckpt_format="fsdp_dtensor")
        train = create_test_training_config(check_weight_hash_across_dp_replicas_interval=10)
        container = self._container_with_fsdp(data_parallel_sharding_strategy="optim_grads_params")
        container.checkpoint = chkpt
        container.train = train
        with pytest.raises(AssertionError, match="check_weight_hash_across_dp_replicas_interval"):
            container.validate()

    def test_fsdp_blocks_tp_pp_dp_mapping(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
        chkpt = create_test_checkpoint_config(ckpt_format="fsdp_dtensor")
        dist = create_test_distributed_init_config(
            use_megatron_fsdp=True, use_tp_pp_dp_mapping=True
        )
        container = self._container_with_fsdp()
        container.dist = dist
        container.checkpoint = chkpt
        with pytest.raises(
            AssertionError, match="use_tp_pp_dp_mapping is not supported with Megatron FSDP"
        ):
            container.validate()


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------


class TestDeterministicMode:
    def test_flash_attention_rejected(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.setenv("NCCL_ALGO", "Tree")
        model = create_test_hybrid_model_config(
            deterministic_mode=True, attention_backend=AttnBackend.flash
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(
            AssertionError, match="Flash attention can not be used in deterministic mode"
        ):
            container.validate()

    def test_cross_entropy_loss_fusion_rejected(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.setenv("NCCL_ALGO", "Tree")
        model = create_test_hybrid_model_config(
            deterministic_mode=True,
            attention_backend=AttnBackend.unfused,
            cross_entropy_loss_fusion=True,
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(
            AssertionError, match="Cross Entropy Fusion is currently not deterministic"
        ):
            container.validate()

    def test_missing_nccl_algo_rejected(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.delenv("NCCL_ALGO", raising=False)
        model = create_test_hybrid_model_config(
            deterministic_mode=True,
            attention_backend=AttnBackend.unfused,
            cross_entropy_loss_fusion=False,
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(AssertionError, match="NCCL_ALGO must be one of"):
            container.validate()

    def test_invalid_nccl_algo_rejected(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.setenv("NCCL_ALGO", "AllReduce")
        model = create_test_hybrid_model_config(
            deterministic_mode=True,
            attention_backend=AttnBackend.unfused,
            cross_entropy_loss_fusion=False,
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(AssertionError, match="NCCL_ALGO must be one of"):
            container.validate()

    def test_valid_config_calls_torch_use_deterministic(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        monkeypatch.setenv("NCCL_ALGO", "Ring")
        model = create_test_hybrid_model_config(
            deterministic_mode=True,
            attention_backend=AttnBackend.unfused,
            cross_entropy_loss_fusion=False,
        )
        container = create_test_pretrain_container(model=model)

        observed = {"flag": None}

        def _mock_use_deterministic(flag):
            observed["flag"] = flag

        with patch.object(
            torch, "use_deterministic_algorithms", side_effect=_mock_use_deterministic
        ):
            container.validate()
        assert observed["flag"] is True

    def test_skipped_when_deterministic_mode_off(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        # No NCCL_ALGO set, but deterministic_mode is False, so the helper should no-op
        monkeypatch.delenv("NCCL_ALGO", raising=False)
        model = create_test_hybrid_model_config(deterministic_mode=False)
        container = create_test_pretrain_container(model=model)
        container.validate()


# ---------------------------------------------------------------------------
# Distributed optimizer sync
# ---------------------------------------------------------------------------


class TestDistributedOptimizerSync:
    def test_both_off_unchanged(self):
        container = create_test_pretrain_container(
            ddp=create_test_ddp_config(use_distributed_optimizer=False),
            optimizer=create_test_optimizer_config(use_distributed_optimizer=False),
        )
        container._validate_and_sync_distributed_optimizer_settings()
        assert container.ddp.use_distributed_optimizer is False
        assert container.optimizer.use_distributed_optimizer is False

    def test_both_on_unchanged(self):
        container = create_test_pretrain_container(
            ddp=create_test_ddp_config(use_distributed_optimizer=True),
            optimizer=create_test_optimizer_config(use_distributed_optimizer=True),
        )
        container._validate_and_sync_distributed_optimizer_settings()
        assert container.ddp.use_distributed_optimizer is True
        assert container.optimizer.use_distributed_optimizer is True

    def test_ddp_only_syncs_to_optimizer(self):
        container = create_test_pretrain_container(
            ddp=create_test_ddp_config(use_distributed_optimizer=True),
            optimizer=create_test_optimizer_config(use_distributed_optimizer=False),
        )
        container._validate_and_sync_distributed_optimizer_settings()
        assert container.ddp.use_distributed_optimizer is True
        assert container.optimizer.use_distributed_optimizer is True

    def test_optimizer_only_syncs_to_ddp(self):
        container = create_test_pretrain_container(
            ddp=create_test_ddp_config(use_distributed_optimizer=False),
            optimizer=create_test_optimizer_config(use_distributed_optimizer=True),
        )
        container._validate_and_sync_distributed_optimizer_settings()
        assert container.ddp.use_distributed_optimizer is True
        assert container.optimizer.use_distributed_optimizer is True


# ---------------------------------------------------------------------------
# Mixed precision consistency
# ---------------------------------------------------------------------------


class TestMixedPrecisionConsistency:
    def test_fp32_passes(self):
        container = create_test_pretrain_container()
        container._validate_mixed_precision_consistency()

    def test_model_bf16_and_fp16_mutex(self):
        model = create_test_hybrid_model_config(bf16=True, fp16=True)
        container = create_test_pretrain_container(model=model)
        with pytest.raises(
            AssertionError, match="Model config cannot have both bf16=True and fp16=True"
        ):
            container._validate_mixed_precision_consistency()

    def test_optimizer_bf16_and_fp16_mutex(self):
        opt = create_test_optimizer_config(bf16=True, fp16=True)
        container = create_test_pretrain_container(optimizer=opt)
        with pytest.raises(
            AssertionError, match="Optimizer config cannot have both bf16=True and fp16=True"
        ):
            container._validate_mixed_precision_consistency()

    def test_precision_aware_bf16_alignment(self):
        model = create_test_hybrid_model_config(bf16=True)
        opt = create_test_optimizer_config(use_precision_aware_optimizer=True, bf16=False)
        container = create_test_pretrain_container(model=model, optimizer=opt)
        with pytest.raises(AssertionError, match="optimizer.bf16=True must be set"):
            container._validate_mixed_precision_consistency()

    def test_precision_aware_fp16_alignment(self):
        model = create_test_hybrid_model_config(fp16=True)
        opt = create_test_optimizer_config(use_precision_aware_optimizer=True, fp16=False)
        container = create_test_pretrain_container(model=model, optimizer=opt)
        with pytest.raises(AssertionError, match="optimizer.fp16=True must be set"):
            container._validate_mixed_precision_consistency()

    def test_precision_aware_fp32_rejects_optimizer_bf16(self):
        opt = create_test_optimizer_config(use_precision_aware_optimizer=True, bf16=True)
        container = create_test_pretrain_container(optimizer=opt)
        # model is fp32 by default
        with pytest.raises(AssertionError, match="must both be False"):
            container._validate_mixed_precision_consistency()

    def test_precision_aware_aligned_bf16_passes(self):
        model = create_test_hybrid_model_config(bf16=True)
        opt = create_test_optimizer_config(use_precision_aware_optimizer=True, bf16=True)
        container = create_test_pretrain_container(model=model, optimizer=opt)
        container._validate_mixed_precision_consistency()


# ---------------------------------------------------------------------------
# Fine-grained activation offloading
# ---------------------------------------------------------------------------


class TestFineGrainedActivationOffloading:
    def test_disabled_is_no_op(self, monkeypatch):
        # Even with a non-TE transformer impl + bad env, offloading=False should pass.
        _patch_world_size(monkeypatch, 1)
        model = create_test_hybrid_model_config(
            fine_grained_activation_offloading=False, transformer_impl="local"
        )
        container = create_test_pretrain_container(model=model)
        container._validate_fine_grained_activation_offloading()

    def test_requires_transformer_engine(self):
        model = create_test_hybrid_model_config(
            fine_grained_activation_offloading=True, transformer_impl="local"
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(ValueError, match="only supported with transformer_engine"):
            container._validate_fine_grained_activation_offloading()

    def test_te_2_10_requires_env_var(self, monkeypatch):
        monkeypatch.delenv("NVTE_CPU_OFFLOAD_V1", raising=False)
        model = create_test_hybrid_model_config(
            fine_grained_activation_offloading=True, transformer_impl="transformer_engine"
        )
        container = create_test_pretrain_container(model=model)
        with patch("megatron.core.utils.is_te_min_version", return_value=True):
            with pytest.raises(ValueError, match="NVTE_CPU_OFFLOAD_V1"):
                container._validate_fine_grained_activation_offloading()

    def test_te_2_10_with_env_var_passes(self, monkeypatch):
        monkeypatch.setenv("NVTE_CPU_OFFLOAD_V1", "1")
        model = create_test_hybrid_model_config(
            fine_grained_activation_offloading=True, transformer_impl="transformer_engine"
        )
        container = create_test_pretrain_container(model=model)
        with patch("megatron.core.utils.is_te_min_version", return_value=True):
            container._validate_fine_grained_activation_offloading()

    def test_pre_te_2_10_skips_env_var_check(self, monkeypatch):
        monkeypatch.delenv("NVTE_CPU_OFFLOAD_V1", raising=False)
        model = create_test_hybrid_model_config(
            fine_grained_activation_offloading=True, transformer_impl="transformer_engine"
        )
        container = create_test_pretrain_container(model=model)
        with patch("megatron.core.utils.is_te_min_version", return_value=False):
            container._validate_fine_grained_activation_offloading()


# ---------------------------------------------------------------------------
# CUDA graph + check_for_nan_in_loss interplay
# ---------------------------------------------------------------------------


class TestCudaGraphScope:
    def test_full_iteration_blocks_nan_check(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        model = create_test_hybrid_model_config(
            cuda_graph_impl="local", cuda_graph_scope=[CudaGraphScope.full_iteration]
        )
        rerun = RerunStateMachineConfig(check_for_nan_in_loss=True)
        container = create_test_pretrain_container(model=model, rerun_state_machine=rerun)
        with pytest.raises(AssertionError, match="check_for_nan_in_loss must be disabled"):
            container.validate()

    def test_cuda_graph_none_clears_scope(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        model = create_test_hybrid_model_config(
            cuda_graph_impl="none", cuda_graph_scope=[CudaGraphScope.attn]
        )
        container = create_test_pretrain_container(model=model)
        container.validate()
        assert container.model.cuda_graph_scope == []


# ---------------------------------------------------------------------------
# ModelOpt / quantization
# ---------------------------------------------------------------------------


class TestModelOptValidation:
    def test_restore_modelopt_state_blocks_grad_accum_fusion(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        # restore_modelopt_state is a HybridModelConfig (ModelConfig) field;
        # gradient_accumulation_fusion lives on the embedded TransformerConfig.
        model = create_test_hybrid_model_config(
            restore_modelopt_state=True, gradient_accumulation_fusion=True
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(
            AssertionError, match="Gradient accumulation fusion is not supported with ModelOpt"
        ):
            container.validate()


# ---------------------------------------------------------------------------
# Gloo + distributed optimizer + checkpoint format
# ---------------------------------------------------------------------------


class TestGlooDistOpt:
    def test_no_gloo_with_dist_opt_requires_torch_dist(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        dist = create_test_distributed_init_config(use_gloo_process_groups=False)
        opt = create_test_optimizer_config(use_distributed_optimizer=True)
        ddp = create_test_ddp_config(use_distributed_optimizer=True)
        chkpt = create_test_checkpoint_config(ckpt_format="torch")
        container = create_test_pretrain_container(
            dist=dist, optimizer=opt, ddp=ddp, checkpoint=chkpt
        )
        with pytest.raises(AssertionError):
            container.validate()

    def test_no_gloo_with_dist_opt_torch_dist_passes(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        dist = create_test_distributed_init_config(use_gloo_process_groups=False)
        opt = create_test_optimizer_config(use_distributed_optimizer=True)
        ddp = create_test_ddp_config(use_distributed_optimizer=True)
        chkpt = create_test_checkpoint_config(ckpt_format="torch_dist")
        container = create_test_pretrain_container(
            dist=dist, optimizer=opt, ddp=ddp, checkpoint=chkpt
        )
        container.validate()


# ---------------------------------------------------------------------------
# Training-scheduler cross-validation
# ---------------------------------------------------------------------------


class TestTrainingSchedulerCompatibility:
    def test_sample_based_blocks_lr_decay_iters(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = TrainingConfig(global_batch_size=32, micro_batch_size=1, train_samples=320)
        sched = create_test_scheduler_config(lr_decay_iters=100)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        with pytest.raises(AssertionError, match="Use lr_decay_samples for sample-based"):
            container.validate()

    def test_sample_based_blocks_lr_warmup_iters(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = TrainingConfig(global_batch_size=32, micro_batch_size=1, train_samples=320)
        # SchedulerConfig.finalize() also rejects mixing fields; bypass scheduler.finalize
        # by patching it for this cross-validation test
        sched = create_test_scheduler_config()
        container = create_test_pretrain_container(train=train, scheduler=sched)
        # Set lr_warmup_iters after construction so SchedulerConfig.finalize doesn't fire first
        # SchedulerConfig.finalize is called during validate(), so we patch the field around it.
        with patch.object(SchedulerConfig, "finalize", lambda self: None):
            container.scheduler.lr_warmup_iters = 5
            with pytest.raises(AssertionError, match="Use lr_warmup_samples for sample-based"):
                container.validate()

    def test_iter_based_blocks_lr_decay_samples(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        sched = create_test_scheduler_config()
        container = create_test_pretrain_container(scheduler=sched)
        with patch.object(SchedulerConfig, "finalize", lambda self: None):
            container.scheduler.lr_decay_samples = 1000
            with pytest.raises(AssertionError, match="Use lr_decay_iters for iteration-based"):
                container.validate()

    def test_iter_based_blocks_lr_warmup_samples(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        sched = create_test_scheduler_config()
        container = create_test_pretrain_container(scheduler=sched)
        with patch.object(SchedulerConfig, "finalize", lambda self: None):
            container.scheduler.lr_warmup_samples = 5
            with pytest.raises(AssertionError, match="Use lr_warmup_iters for iteration-based"):
                container.validate()


# ---------------------------------------------------------------------------
# Scheduler step calculation
# ---------------------------------------------------------------------------


class TestSchedulerStepCalculation:
    def test_iter_based_default_lr_decay_iters(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = create_test_training_config(train_iters=2000, global_batch_size=32)
        sched = create_test_scheduler_config(lr_decay_iters=None)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        container.validate()
        assert container.scheduler.lr_decay_iters == 2000
        assert container.scheduler.lr_decay_steps == 2000 * 32

    def test_iter_based_custom_lr_decay_iters(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = create_test_training_config(train_iters=2000, global_batch_size=32)
        sched = create_test_scheduler_config(lr_decay_iters=1500)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        container.validate()
        assert container.scheduler.lr_decay_iters == 1500
        assert container.scheduler.lr_decay_steps == 1500 * 32

    def test_iter_based_wd_incr_steps(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = create_test_training_config(train_iters=500, global_batch_size=16)
        container = create_test_pretrain_container(train=train)
        container.validate()
        assert container.scheduler.wd_incr_steps == 500 * 16

    def test_iter_based_warmup_from_fraction(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = create_test_training_config(train_iters=1000, global_batch_size=32)
        sched = create_test_scheduler_config(lr_warmup_iters=0, lr_warmup_fraction=0.1)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        container.validate()
        assert container.scheduler.lr_warmup_steps == 0.1 * (1000 * 32)

    def test_iter_based_warmup_from_iters(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = create_test_training_config(train_iters=1000, global_batch_size=10)
        sched = create_test_scheduler_config(lr_warmup_iters=50, lr_warmup_fraction=None)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        container.validate()
        assert container.scheduler.lr_warmup_steps == 50 * 10

    def test_iter_based_wsd_decay_steps(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = create_test_training_config(train_iters=100, global_batch_size=8)
        sched = create_test_scheduler_config(lr_wsd_decay_iters=100)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        container.validate()
        assert container.scheduler.wsd_decay_steps == 100 * 8

    def test_iter_based_wsd_decay_steps_default_for_wsd_style(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = create_test_training_config(train_iters=200, global_batch_size=4)
        sched = create_test_scheduler_config(lr_decay_style="WSD", lr_wsd_decay_iters=None)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        container.validate()
        # lr_wsd_decay_iters should default to lr_decay_iters when style is WSD
        assert container.scheduler.lr_wsd_decay_iters == 200
        assert container.scheduler.wsd_decay_steps == 200 * 4

    def test_sample_based_lr_decay_steps(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = TrainingConfig(global_batch_size=32, micro_batch_size=1, train_samples=3200)
        sched = create_test_scheduler_config(lr_decay_samples=None)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        container.validate()
        # lr_decay_samples defaults to train_samples
        assert container.scheduler.lr_decay_samples == 3200
        assert container.scheduler.lr_decay_steps == 3200
        assert container.scheduler.wd_incr_steps == 3200

    def test_sample_based_warmup_from_samples(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = TrainingConfig(global_batch_size=32, micro_batch_size=1, train_samples=3200)
        sched = create_test_scheduler_config(
            lr_warmup_iters=0, lr_warmup_samples=320, lr_warmup_fraction=None
        )
        container = create_test_pretrain_container(train=train, scheduler=sched)
        container.validate()
        assert container.scheduler.lr_warmup_steps == 320

    def test_sample_based_warmup_from_fraction(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = TrainingConfig(global_batch_size=32, micro_batch_size=1, train_samples=3200)
        sched = create_test_scheduler_config(lr_warmup_iters=0, lr_warmup_fraction=0.2)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        container.validate()
        assert container.scheduler.lr_warmup_steps == 0.2 * 3200

    def test_lr_decay_steps_zero_raises(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        train = create_test_training_config(train_iters=0, global_batch_size=32)
        # global_batch_size > 0 so lr_decay_steps == 0 * 32 == 0
        sched = create_test_scheduler_config()
        container = create_test_pretrain_container(train=train, scheduler=sched)
        with pytest.raises(ValueError, match="lr_decay_steps must be > 0"):
            container.validate()

    def test_warmup_capped_when_exceeds_decay(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        # train_iters=10, gbs=1 => lr_decay_steps=10
        # lr_warmup_iters=20, gbs=1 => lr_warmup_steps=20 (>= 10)
        train = create_test_training_config(train_iters=10, global_batch_size=1)
        sched = create_test_scheduler_config(lr_warmup_iters=20)
        container = create_test_pretrain_container(train=train, scheduler=sched)
        with pytest.warns(UserWarning, match="capping lr_warmup_steps"):
            container.validate()
        decay_steps = container.scheduler.lr_decay_steps
        assert decay_steps is not None
        assert container.scheduler.lr_warmup_steps == decay_steps - 1


# ---------------------------------------------------------------------------
# Context parallel: seq_length divisibility
# ---------------------------------------------------------------------------


class TestCpSeqLengthDivisibility:
    def test_cp_size_one_skips_check(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        # seq_length not divisible by anything special, but cp=1 -> no check
        model = create_test_hybrid_model_config(context_parallel_size=1, seq_length=11)
        container = create_test_pretrain_container(model=model)
        container.validate()

    def test_seq_length_divisible_passes(self, monkeypatch):
        _patch_world_size(monkeypatch, 4)
        # seq_length=16, cp_size=2 => 2*cp_size=4 ; 16 % 4 == 0
        model = create_test_hybrid_model_config(
            context_parallel_size=2, seq_length=16, num_attention_heads=4
        )
        container = create_test_pretrain_container(model=model)
        container.validate()

    def test_seq_length_indivisible_rejected(self, monkeypatch):
        _patch_world_size(monkeypatch, 4)
        # seq_length=17, cp_size=2 => 2*cp_size=4; 17 % 4 != 0
        model = create_test_hybrid_model_config(
            context_parallel_size=2, seq_length=17, num_attention_heads=4
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(AssertionError, match="Sequence length must be divisible"):
            container.validate()


# ---------------------------------------------------------------------------
# Context parallel: cp_comm_type / hierarchical sizes
# ---------------------------------------------------------------------------


class TestCpCommType:
    def test_a2a_p2p_requires_hcp_sizes(self, monkeypatch):
        _patch_world_size(monkeypatch, 1)
        model = create_test_hybrid_model_config(
            context_parallel_size=1,
            cp_comm_type="a2a+p2p",
            hierarchical_context_parallel_sizes=None,
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(AssertionError, match="hierarchical_context_parallel_sizes must be set"):
            container.validate()

    def test_hcp_sizes_product_must_equal_cp_size(self, monkeypatch):
        _patch_world_size(monkeypatch, 8)
        # cp_size=4 but prod([2,3])=6 != 4
        model = create_test_hybrid_model_config(
            context_parallel_size=4,
            num_attention_heads=4,
            seq_length=16,
            hierarchical_context_parallel_sizes=[2, 3],
            cp_comm_type="a2a+p2p",
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(AssertionError, match="must equal context_parallel_size"):
            container.validate()

    def test_hcp_sizes_product_matches_cp_size_passes(self, monkeypatch):
        _patch_world_size(monkeypatch, 8)
        model = create_test_hybrid_model_config(
            context_parallel_size=4,
            num_attention_heads=4,
            seq_length=16,
            hierarchical_context_parallel_sizes=[2, 2],
            cp_comm_type="a2a+p2p",
        )
        container = create_test_pretrain_container(model=model)
        container.validate()

    def test_cp_comm_type_list_must_match_num_layers(self, monkeypatch):
        _patch_world_size(monkeypatch, 4)
        # num_layers=1 but cp_comm_type list has 2 entries
        model = create_test_hybrid_model_config(
            num_layers=1,
            num_attention_heads=4,
            context_parallel_size=2,
            seq_length=16,
            cp_comm_type=["p2p", "p2p"],
        )
        container = create_test_pretrain_container(model=model)
        with pytest.raises(AssertionError, match="Length of cp_comm_type"):
            container.validate()


# ---------------------------------------------------------------------------
# Flex dispatcher backend (GPU-property gated)
# ---------------------------------------------------------------------------


class TestFlexDispatcherBackend:
    def _make_transformer(self, dispatcher: str, backend: str) -> TransformerConfig:
        return create_test_transformer_config(
            moe_token_dispatcher_type=dispatcher, moe_flex_dispatcher_backend=backend
        )

    def test_non_flex_dispatcher_skips(self):
        # When dispatcher != "flex", validate_flex_dispatcher_backend never inspects the GPU.
        cfg = self._make_transformer("allgather", "deepep")
        validate_flex_dispatcher_backend(cfg)

    def test_deepep_passes_on_hopper(self):
        cfg = self._make_transformer("flex", "deepep")
        props = type("P", (), {"major": 9, "minor": 0, "name": "NVIDIA H100"})()
        with patch("torch.cuda.get_device_properties", return_value=props):
            validate_flex_dispatcher_backend(cfg)

    def test_deepep_passes_on_blackwell_b200(self):
        cfg = self._make_transformer("flex", "deepep")
        # major=10 isn't whitelisted but the name prefix is
        props = type("P", (), {"major": 10, "minor": 0, "name": "NVIDIA B200"})()
        with patch("torch.cuda.get_device_properties", return_value=props):
            validate_flex_dispatcher_backend(cfg)

    def test_deepep_rejects_unsupported_major(self):
        cfg = self._make_transformer("flex", "deepep")
        props = type("P", (), {"major": 7, "minor": 5, "name": "NVIDIA T4"})()
        with patch("torch.cuda.get_device_properties", return_value=props):
            with pytest.raises(ValueError, match="DeepEP is supported"):
                validate_flex_dispatcher_backend(cfg)

    def test_hybridep_passes_on_blackwell_major_10(self):
        cfg = self._make_transformer("flex", "hybridep")
        props = type("P", (), {"major": 10, "minor": 0, "name": "NVIDIA B200"})()
        with patch("torch.cuda.get_device_properties", return_value=props):
            validate_flex_dispatcher_backend(cfg)

    def test_hybridep_rejects_unsupported_major(self):
        cfg = self._make_transformer("flex", "hybridep")
        props = type("P", (), {"major": 7, "minor": 5, "name": "NVIDIA T4"})()
        with patch("torch.cuda.get_device_properties", return_value=props):
            with pytest.raises(ValueError, match="HybridEP is supported"):
                validate_flex_dispatcher_backend(cfg)

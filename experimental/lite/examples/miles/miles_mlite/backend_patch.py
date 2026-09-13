# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Monkeypatch the miles Megatron train actor to use Megatron Lite.

miles chooses its train actor by a function-local import of
``miles.backends.megatron_utils.actor.MegatronTrainRayActor``. Replacing that
source symbol before actor-group construction lets the example use the existing
``--train-backend megatron`` slot without changing miles code or CLI choices.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
from miles.ray.train_actor import TrainRayActor as _MilesTrainRayActor

from .arguments import optimizer_backend_to_impl, validate_mlite_args
from .data import build_runtime_microbatches
from .loss import make_runtime_loss_fn
from .weight_update import RawHFWeightUpdater

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _iter_local_parameters(handle):
    model_chunks = handle._extras.get("model_chunks", [handle._model])
    for chunk_id, chunk in enumerate(model_chunks):
        for name, param in chunk.named_parameters():
            yield f"{chunk_id}:{name}", param


def _install_set_input_tensor_proxy(handle) -> None:
    for chunk in handle._extras.get("model_chunks", [handle._model]):
        if hasattr(chunk, "set_input_tensor"):
            continue
        module = getattr(chunk, "module", None)
        setter = getattr(module, "set_input_tensor", None)
        if not callable(setter):
            continue
        try:
            setattr(chunk, "set_input_tensor", setter)
        except Exception:
            object.__setattr__(chunk, "set_input_tensor", setter)


def _capture_checkpoint_probe(handle):
    fallback = None
    for key, param in _iter_local_parameters(handle):
        if fallback is None:
            fallback = (key, param)
        if "norm" in key and param.numel() <= 1_000_000:
            return key, param.detach().float().cpu().clone()
    if fallback is None:
        return None, None
    key, param = fallback
    return key, param.detach().float().cpu().clone()


def _find_local_parameter(handle, key: str):
    for candidate_key, param in _iter_local_parameters(handle):
        if candidate_key == key:
            return param
    return None


def _clear_checkpoint_contents(path: str) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.name == "ray_done":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _looks_like_mlite_checkpoint(path: str) -> bool:
    root = Path(path)
    if root.name.startswith("step_"):
        return (root / "metadata.json").exists() or (root / "common.pt").exists()
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if child.name.startswith("step_") and child.is_dir():
            return True
    return (root / "metadata.json").exists() or (root / "common.pt").exists()


def _group(rank: int, size: int, group):
    return SimpleNamespace(rank=rank, size=size, group=group, gloo_group=None)


def _install_miles_parallel_state(ps) -> None:
    try:
        parallel_mod = importlib.import_module("miles.backends.training_utils.parallel")
    except ImportError:
        return
    state = parallel_mod.ParallelState(
        intra_dp=_group(ps.dp_rank, ps.dp_size, ps.dp_group),
        intra_dp_cp=_group(getattr(ps, "dp_cp_rank", ps.dp_rank), getattr(ps, "dp_cp_size", ps.dp_size), getattr(ps, "dp_cp_group", ps.dp_group)),
        cp=_group(ps.cp_rank, ps.cp_size, ps.cp_group),
        tp=_group(ps.tp_rank, ps.tp_size, ps.tp_group),
        pp=_group(ps.pp_rank, ps.pp_size, ps.pp_group),
        ep=_group(getattr(ps, "ep_rank", 0), getattr(ps, "ep_size", 1), getattr(ps, "ep_group", None)),
        etp=_group(getattr(ps, "etp_rank", 0), getattr(ps, "etp_size", 1), getattr(ps, "etp_group", None)),
        cp_comm_type=getattr(ps, "cp_comm_type", None),
        is_pp_last_stage=ps.pp_rank == ps.pp_size - 1,
        vpp_size=1,
        microbatch_group_size_per_vp_stage=1,
    )
    parallel_mod.set_parallel_state(state)


class _MLiteTrainRayActorMixin:
    def _build_mlite_config(self, args: Namespace):
        from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
        from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

        validate_mlite_args(args)
        parallel = ParallelConfig(
            tp=args.tensor_model_parallel_size,
            etp=getattr(args, "expert_tensor_parallel_size", None),
            ep=getattr(args, "expert_model_parallel_size", 1),
            pp=args.pipeline_model_parallel_size,
            vpp=getattr(args, "virtual_pipeline_model_parallel_size", None) or 1,
            cp=args.context_parallel_size,
        )
        optimizer = OptimizerConfig(
            optimizer=args.optimizer,
            lr=args.lr,
            min_lr=args.min_lr if args.min_lr is not None else 0.0,
            clip_grad=args.clip_grad,
            weight_decay=args.weight_decay,
            lr_decay_style=args.lr_decay_style,
            adam_beta1=getattr(args, "adam_beta1", None),
            adam_beta2=getattr(args, "adam_beta2", None),
            adam_eps=getattr(args, "adam_eps", None),
        )
        if getattr(args, "mlite_optimizer_offload", False):
            optimizer.offload_fraction = 1.0
            optimizer.use_precision_aware_optimizer = True
            optimizer.decoupled_weight_decay = True

        attention_backend = args.mlite_attention_backend
        if attention_backend is None:
            raw = getattr(args, "attention_backend", None)
            attention_backend = getattr(raw, "name", raw)
        attention_backend = attention_backend or "flash"

        return MegatronLiteConfig(
            model_name=args.mlite_model_name,
            impl=args.mlite_impl,
            hf_path=args.hf_checkpoint,
            parallel=parallel,
            optimizer=optimizer,
            attention_backend_override=attention_backend,
            load_hf_weights=True,
            impl_cfg={
                "use_thd": True,
                "optimizer": optimizer_backend_to_impl(args.mlite_optimizer_backend),
            },
        )

    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)
        if role != "actor":
            raise NotImplementedError("Megatron Lite miles backend supports actor training only.")
        if with_ref or with_opd_teacher:
            raise NotImplementedError("Reference/teacher model swapping is not implemented for the MLite patch.")

        if args.debug_rollout_only:
            self.args = args
            return 0

        from megatron.lite.runtime import RuntimeConfig, create_runtime

        self._cfg = self._build_mlite_config(args)
        self.runtime = create_runtime(
            RuntimeConfig(backend="mlite", hf_path=args.hf_checkpoint, backend_cfg=self._cfg)
        )
        self.handle = self.runtime.build_model()
        _install_set_input_tensor_proxy(self.handle)

        ps = self.handle._parallel_state
        _install_miles_parallel_state(ps)
        self.train_parallel_config = {
            "dp_size": ps.dp_size,
            "cp_size": ps.cp_size,
            "vpp_size": self._cfg.parallel.vpp or 1,
            "microbatch_group_size_per_vp_stage": 1,
        }
        self.weight_updater = RawHFWeightUpdater(args, self.runtime, self.handle)

        start_rollout_id = 0
        if getattr(args, "load", None):
            if _looks_like_mlite_checkpoint(args.load):
                loaded = self.runtime.load_checkpoint(
                    self.handle,
                    args.load,
                    load_optimizer=_env_flag("MLITE_LOAD_OPTIMIZER_CHECKPOINT", False),
                    load_rng=_env_flag("MLITE_LOAD_RNG_CHECKPOINT", False),
                )
                if _env_flag("MLITE_RESET_ROLLOUT_AFTER_LOAD", False):
                    start_rollout_id = 0
                else:
                    start_rollout_id = int(loaded) + 1
                if _rank() == 0:
                    logger.info(
                        "MLITE_MILES_GRPO_INITIAL_LOAD_DONE path=%s loaded_step=%s start_rollout_id=%s",
                        args.load,
                        loaded,
                        start_rollout_id,
                    )
            else:
                if _rank() == 0:
                    logger.info(
                        "MLITE_MILES_GRPO_INITIAL_LOAD_SKIPPED path=%s reason=not_mlite_checkpoint",
                        args.load,
                    )

        if getattr(args, "offload_train", False) or getattr(args, "mlite_param_offload", False):
            self.sleep()
        return start_rollout_id

    def _process_rollout_data(self, rollout_data_ref):
        data_mod = importlib.import_module("miles.utils.data")
        ps = self.handle._parallel_state
        rollout_data = data_mod.process_rollout_data(self.args, rollout_data_ref, ps.dp_rank, ps.dp_size)
        rollout_data["tokens"] = [torch.as_tensor(t, dtype=torch.long) for t in rollout_data["tokens"]]
        rollout_data["loss_masks"] = [torch.as_tensor(t, dtype=torch.float32) for t in rollout_data["loss_masks"]]
        device = torch.device("cuda", torch.cuda.current_device())
        for key in ("rollout_log_probs", "log_probs", "ref_log_probs", "advantages", "returns"):
            if key in rollout_data and rollout_data[key] is not None:
                rollout_data[key] = [
                    torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1)
                    for t in rollout_data[key]
                ]
        return rollout_data

    def _compute_advantages_and_returns(self, rollout_data) -> None:
        loss_mod = importlib.import_module("miles.backends.training_utils.loss")
        loss_mod.compute_advantages_and_returns(self.args, rollout_data)

    def _build_microbatches(self, rollout_data, *, calculate_entropy: bool = False):
        return build_runtime_microbatches(
            rollout_data,
            micro_batch_size=getattr(self.args, "micro_batch_size", 1) or 1,
            use_dynamic_batch_size=getattr(self.args, "use_dynamic_batch_size", False),
            max_tokens_per_gpu=getattr(self.args, "max_tokens_per_gpu", 0) or 0,
            calculate_entropy=calculate_entropy,
            temperature=float(getattr(self.args, "rollout_temperature", 1.0) or 1.0),
        )

    def _compute_log_probs(self, rollout_data) -> list[torch.Tensor] | None:
        microbatches = self._build_microbatches(
            rollout_data,
            calculate_entropy=bool(getattr(self.args, "use_rollout_entropy", False)),
        )
        if not microbatches:
            return []
        store: list[dict[str, list[torch.Tensor]]] = []
        with self.runtime.eval_mode(self.handle):
            self.runtime.forward_backward(
                self.handle,
                (mb.as_runtime_item() for mb in microbatches),
                loss_fn=make_runtime_loss_fn(
                    self.args,
                    self.handle,
                    forward_store=store,
                    loss_context_iter=(mb.loss_context() for mb in microbatches),
                ),
                num_microbatches=len(microbatches),
                forward_only=True,
            )
        if not store and not self.runtime.is_mp_src_rank_with_outputs(self.handle):
            return None
        return [item for micro in store for item in micro["log_probs"]]

    def train(self, rollout_id: int, rollout_data_ref) -> None:
        self._last_rollout_id = rollout_id
        if getattr(self.args, "offload_train", False) or getattr(self.args, "mlite_param_offload", False):
            self.wake_up()

        rollout_data = self._process_rollout_data(rollout_data_ref)
        if self.args.debug_rollout_only:
            return None

        loss_type = getattr(self.args, "loss_type", "sft_loss")
        if loss_type == "policy_loss" and getattr(self.args, "compute_advantages_and_returns", True):
            if not getattr(self.args, "use_rollout_logprobs", False) or getattr(self.args, "get_mismatch_metrics", False):
                log_probs = self._compute_log_probs(rollout_data)
                if log_probs is not None:
                    rollout_data["log_probs"] = log_probs
            elif "rollout_log_probs" not in rollout_data:
                log_probs = self._compute_log_probs(rollout_data)
                if log_probs is not None:
                    rollout_data["log_probs"] = log_probs
            elif _rank() == 0:
                logger.info(
                    "MLITE_MILES_GRPO_USING_ROLLOUT_LOGPROBS rollout=%s count=%s",
                    rollout_id,
                    len(rollout_data["rollout_log_probs"]),
                )
            self._compute_advantages_and_returns(rollout_data)

        microbatches = self._build_microbatches(rollout_data)
        if not microbatches:
            logger.warning("rollout %s: empty data shard", rollout_id)
            return None

        with self.runtime.train_mode(self.handle):
            self.runtime.zero_grad(self.handle)
            result = self.runtime.forward_backward(
                self.handle,
                (mb.as_runtime_item() for mb in microbatches),
                loss_fn=make_runtime_loss_fn(
                    self.args,
                    self.handle,
                    loss_context_iter=(mb.loss_context() for mb in microbatches),
                ),
                num_microbatches=len(microbatches),
                forward_only=False,
            )
            _, grad_norm, _ = self.runtime.optimizer_step(self.handle)
            lr = self.runtime.lr_scheduler_step(self.handle)

        logger.info(
            "rollout %s | train/loss %s | grad_norm %.4f | lr %s | num_microbatches %s",
            rollout_id,
            result.metrics.get("loss"),
            float(grad_norm),
            lr,
            len(microbatches),
        )
        return None

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return
        save_dir = getattr(self.args, "save", None)
        if not save_dir:
            return
        save_optimizer = _env_flag("MLITE_SAVE_OPTIMIZER_CHECKPOINT", False)
        save_rng = _env_flag("MLITE_SAVE_RNG_CHECKPOINT", False)
        verify_load = _env_flag("MLITE_VERIFY_CHECKPOINT_LOAD", True)
        delete_after_load = _env_flag("MLITE_DELETE_CHECKPOINT_AFTER_LOAD", True)

        probe_key, before = _capture_checkpoint_probe(self.handle)
        self.runtime.save_checkpoint(
            self.handle,
            save_dir,
            step=rollout_id,
            save_optimizer=save_optimizer,
            save_rng=save_rng,
        )

        max_abs = None
        loaded = None
        if verify_load:
            loaded = self.runtime.load_checkpoint(
                self.handle,
                save_dir,
                load_optimizer=save_optimizer,
                load_rng=save_rng,
            )
            if probe_key is not None and before is not None:
                after_param = _find_local_parameter(self.handle, probe_key)
                if after_param is None:
                    raise RuntimeError(f"checkpoint load probe parameter missing after load: {probe_key}")
                after = after_param.detach().float().cpu()
                max_abs = float(torch.max(torch.abs(after - before)).item())
                if max_abs != 0.0:
                    raise RuntimeError(
                        f"checkpoint load probe mismatch for {probe_key}: max_abs={max_abs}"
                    )

        if _rank() == 0:
            logger.info(
                "MLITE_MILES_GRPO_SAVE_LOAD_DONE rollout=%s path=%s loaded_step=%s "
                "probe=%s max_abs=%s save_optimizer=%s save_rng=%s",
                rollout_id,
                save_dir,
                loaded,
                probe_key,
                max_abs,
                save_optimizer,
                save_rng,
            )

        _barrier()
        if delete_after_load and _rank() == 0:
            _clear_checkpoint_contents(save_dir)
            logger.info("MLITE_MILES_GRPO_CHECKPOINT_CLEANED path=%s", save_dir)
        _barrier()

    def update_weights(self, info=None) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        if info is None:
            raise ValueError("update_weights requires rollout engine info from the miles rollout manager.")
        if getattr(self.args, "offload_train", False) or getattr(self.args, "mlite_param_offload", False):
            self.wake_up()

        rollout_engines = info.rollout_engines
        rollout_engine_lock = info.rollout_engine_lock
        has_new_engines = info.has_new_engines
        engine_gpu_counts = getattr(info, "engine_gpu_counts", None)
        engine_gpu_offsets = getattr(info, "engine_gpu_offsets", None)
        del info

        if has_new_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            import ray

            if torch.distributed.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_has_new_engines.remote())

        if getattr(self.args, "debug_skip_weight_update", False):
            logger.warning("Skipping MLite actor-to-rollout weight update because --debug-skip-weight-update is set.")
            return
        self.weight_updater.update_weights()

    def sleep(self, *args, **kwargs) -> None:
        if not (getattr(self.args, "offload_train", False) or getattr(self.args, "mlite_param_offload", False)):
            return
        self.runtime.to(self.handle, "cpu")

    def wake_up(self, *args, **kwargs) -> None:
        if not (getattr(self.args, "offload_train", False) or getattr(self.args, "mlite_param_offload", False)):
            return
        self.runtime.to(self.handle, "cuda")

    def connect_actor_critic(self, critic_group=None, **kwargs):
        raise NotImplementedError("Megatron Lite miles backend does not support critic training yet.")

    def _get_parallel_config(self):
        return self.train_parallel_config


class MLiteTrainRayActor(_MLiteTrainRayActorMixin, _MilesTrainRayActor):
    """Megatron Lite TrainRayActor patched into miles."""


def _load_or_synthesize_actor_module(import_error: ImportError) -> ModuleType:
    module_name = "miles.backends.megatron_utils.actor"
    actor_mod = ModuleType(module_name)
    actor_mod.__package__ = "miles.backends.megatron_utils"
    actor_mod.__doc__ = "Synthetic MLite actor patch module for miles."
    sys.modules[module_name] = actor_mod

    parent_mod = importlib.import_module(actor_mod.__package__)
    setattr(parent_mod, "actor", actor_mod)
    logger.warning("Using synthetic %s because the original module failed to import: %s", module_name, import_error)
    return actor_mod


def patch_miles_backend() -> type:
    """Patch installed miles and return the patched actor class."""
    importlib.import_module("miles")
    try:
        actor_mod = importlib.import_module("miles.backends.megatron_utils.actor")
    except ImportError as exc:
        actor_mod = _load_or_synthesize_actor_module(exc)
    actor_mod.MegatronTrainRayActor = MLiteTrainRayActor
    logger.info("Patched miles MegatronTrainRayActor with Megatron Lite.")
    return MLiteTrainRayActor

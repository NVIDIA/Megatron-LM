# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small compatibility patches for dependency-version gaps in examples."""

from __future__ import annotations

import gc
import importlib.abc
import importlib.machinery
import importlib.util
import os
import queue
import sys
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import wraps
from pathlib import Path
from typing import Any

_VLLM_ASYNC_SERVER_MODULE = (
    "verl.workers.rollout.vllm_rollout.vllm_async_server"
)
_VLLM_ROLLOUT_CONSUMER_MODULE = (
    "verl.workers.rollout.vllm_rollout.vllm_rollout"
)
_REGISTERED_HF_CONFIG_TYPES: set[str] = set()

_VLLM_IMPORTABLE: bool | None = None


_BUCKETED_SENDER_MODULE = "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"

def _vllm_importable() -> bool:
    """Whether ``import vllm`` succeeds in THIS process.

    A fat rollout overlay (native vLLM 0.25 = torch 2.11+cu130, Transformers v5)
    can only be imported inside the vLLM rollout Ray actor, which is scoped to
    the overlay's torch + CUDA libs via the compat vLLM-server profile. The
    training driver and worker processes keep the container/SM90 stack, whose
    vllm build is ABI/CUDA-incompatible with this container — importing it there
    dies with e.g. ``libcudart.so.12: cannot open shared object file`` (SM90's
    CUDA-12 vllm in a CUDA-13 container) or the Transformers-v4/v5 mismatch.
    Every vllm-touching patch below is therefore a no-op wherever vllm is
    unimportable, so importing verl_mlite on the driver (hydra config
    validation instantiates the engine module tree, which used to eagerly import
    the rollout vllm utils) and running apply_runtime_patches there no longer
    crash on the rollout engine's vllm. The result is cached because the import
    outcome is fixed per process and a failed attempt is slow to retry.
    """
    global _VLLM_IMPORTABLE
    if _VLLM_IMPORTABLE is None:
        try:
            # A bare ``import vllm`` only runs the lazy PEP-562 package shell and
            # succeeds even where the real engine is unusable, so force the same
            # entrypoint VERL pulls (``from vllm import LLM``). Accessing ``LLM``
            # triggers the lazy load of vllm.entrypoints -> vllm.config ->
            # vllm.platforms -> ``vllm._C``, i.e. the exact chain that dies on the
            # driver with libcudart.so.12 / the torch-ABI mismatch.
            getattr(importlib.import_module("vllm"), "LLM")
            _VLLM_IMPORTABLE = True
        except Exception:
            _VLLM_IMPORTABLE = False
    return _VLLM_IMPORTABLE


def _register_opaque_hf_config() -> bool:
    """Let VERL preserve config fields for an MLite-owned model type."""
    model_type = os.environ.get("VERL_MLITE_HF_CONFIG_MODEL_TYPE", "").strip()
    if not model_type or model_type in _REGISTERED_HF_CONFIG_TYPES:
        return False

    from transformers import AutoConfig, PretrainedConfig

    config_cls = type(
        "MLiteOpaqueConfig",
        (PretrainedConfig,),
        {"model_type": model_type},
    )
    try:
        AutoConfig.register(model_type, config_cls)
    except ValueError:
        # A newer Transformers already owns this model type.
        _REGISTERED_HF_CONFIG_TYPES.add(model_type)
        return False
    _REGISTERED_HF_CONFIG_TYPES.add(model_type)
    return True


class _VllmThinFinder(importlib.abc.MetaPathFinder):
    """Resolve only the top-level ``vllm`` package from a rollout site."""

    _verl_mlite_vllm_thin_finder = True

    def __init__(self, site: str):
        self._site = site

    def find_spec(self, fullname, path, target=None):
        if fullname != "vllm":
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, [self._site], target)


def _install_vllm_thin_finder() -> bool:
    site = os.environ.get("VERL_MLITE_VLLM_SITE", "").strip()
    if not site:
        return False
    # A THIN overlay ships vllm but no torch: the container/training torch is
    # shared, so this global meta-path finder can safely redirect EVERY process's
    # ``import vllm`` (driver + rollout) to the overlay. A FAT overlay instead
    # bundles its own torch (native vLLM 0.25 = torch 2.11+cu130) and a newer
    # vllm whose hard Transformers-v5 requirement mismatches the training
    # driver's Transformers-v4 stack. It must reach ONLY the rollout Ray actor,
    # which acquires it through the scoped verl_mlite.compat vLLM-server profile
    # (its site is prepended to that actor's PYTHONPATH). Installing a global
    # finder for a fat overlay forces the driver's incidental ``import vllm``
    # (verl config validation instantiates the rollout module tree) onto the fat
    # vllm 0.25 and dies with "Support for Transformers v4 ... removed in vLLM
    # v0.24.0" (job 13956329). The presence of bundled CUDA libs distinguishes a
    # fat overlay, so skip the global finder for it and let the driver keep its
    # own (container/SM90) vllm.
    if _vllm_site_ld_library_path(site):
        return False
    if any(
        getattr(finder, "_verl_mlite_vllm_thin_finder", False)
        for finder in sys.meta_path
    ):
        return False
    sys.meta_path.insert(0, _VllmThinFinder(site))
    return True


def _patch_transformers_vision2seq_alias() -> bool:
    """Restore the Transformers 4 vision auto-class name removed in v5.

    VERL's ``verl.utils.model`` does a top-level ``from transformers import
    AutoModelForVision2Seq`` that every training/rollout worker hits regardless
    of the rollout overlay. Under the single torch2.12/cu13 stack the alias must
    apply unconditionally (transformers v5 is the only world now), so this is no
    longer gated on ``VERL_MLITE_VLLM_SITE`` (the retired fat/thin split env).

    A one-shot attribute set on the top-level module does NOT survive: importing
    VERL's vLLM utilities re-execs Transformers' ``_LazyModule`` and rebuilds a
    fresh instance whose ``_class_to_module`` map has no ``AutoModelForVision2Seq``
    entry, so the injected attribute is dropped and the ``from transformers
    import`` line fails again. We therefore patch the ``_LazyModule`` *class*'s
    ``__getattr__`` so every instance -- current and any rebuilt one -- resolves
    the removed name to ``AutoModelForImageTextToText``.
    """
    import transformers

    try:
        from transformers.utils import import_utils as _iu

        lazy_cls = _iu._LazyModule
    except Exception:  # pragma: no cover - transformers internals moved
        lazy_cls = None

    if lazy_cls is not None and not getattr(
        lazy_cls, "_mlite_vision2seq_patched", False
    ):
        _orig_getattr = lazy_cls.__getattr__

        def _getattr_with_vision2seq_alias(self, name):
            if name == "AutoModelForVision2Seq":
                return _orig_getattr(self, "AutoModelForImageTextToText")
            return _orig_getattr(self, name)

        lazy_cls.__getattr__ = _getattr_with_vision2seq_alias
        lazy_cls._mlite_vision2seq_patched = True

    # Belt-and-suspenders: also expose it on the current module object so code
    # that does ``hasattr(transformers, "AutoModelForVision2Seq")`` short-circuits.
    if not hasattr(transformers, "AutoModelForVision2Seq"):
        replacement = getattr(transformers, "AutoModelForImageTextToText", None)
        if replacement is not None:
            transformers.AutoModelForVision2Seq = replacement
            return True
    return lazy_cls is not None


def _install_vllm_triton_kernels_alias() -> bool:
    """Prefer the rollout vLLM's complete vendored Triton kernel package."""
    if not os.environ.get("VERL_MLITE_VLLM_SITE", "").strip():
        return False
    if not _vllm_importable():
        return False
    vendored = importlib.import_module("vllm.third_party.triton_kernels")
    if sys.modules.get("triton_kernels") is vendored:
        return False
    sys.modules["triton_kernels"] = vendored
    return True


def _vllm_site_pythonpath_prefixes(site: str) -> list[str]:
    """Extra PYTHONPATH entries a rollout overlay needs ahead of its site root.

    Fat overlays (e.g. the native vLLM 0.25 DS4 closure) ship cutlass-dsl under
    ``nvidia_cutlass_dsl/python_packages`` via a ``.pth`` that is NOT processed
    when the site is reached through ``PYTHONPATH`` rather than site-packages
    import. Without an explicit prefix a stray conda cutlass shadows it and
    flashinfer dies with ``cute.nvgpu.OperandMajorMode``. Thin overlays that lack
    the directory contribute nothing, so this stays a no-op for them.
    """
    prefixes: list[str] = []
    cutlass = os.path.join(site, "nvidia_cutlass_dsl", "python_packages")
    if os.path.isdir(cutlass):
        prefixes.append(cutlass)
    return prefixes


def _vllm_site_ld_library_path(site: str) -> list[str]:
    """CUDA runtime lib dirs a rollout overlay bundles for its own torch build.

    A native-CUDA overlay (torch 2.11+cu130) colocated inside a cu128 training
    container must expose its bundled ``nvidia/*/lib`` and ``torch/lib`` to the
    rollout actor's loader, but ONLY to that actor — the training driver keeps
    the container's cu128 stack. Scoping this to the vLLM Ray-actor runtime env
    (rather than a process-wide LD_LIBRARY_PATH) is what keeps the two CUDA
    majors from colliding. Overlays whose torch matches the container ship no
    such dirs, so this returns nothing and stays a no-op.
    """
    import glob

    lib_dirs = sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib")))
    torch_lib = os.path.join(site, "torch", "lib")
    if os.path.isdir(torch_lib):
        lib_dirs.append(torch_lib)
    return [d for d in lib_dirs if os.path.isdir(d)]


def _vllm_server_profile_env() -> dict[str, str]:
    """Build the dependency profile applied only to vLLM server Ray actors."""
    site = os.environ.get("VERL_MLITE_VLLM_SITE", "").strip()
    if not site:
        return {}
    pythonpath = os.environ.get("PYTHONPATH", "").strip()
    # Keep vLLM's dependency closure on the rollout site. The scoped compatibility
    # alias above lets VERL import against that site's Transformers v5 build. Fat
    # overlays additionally need their bundled cutlass ahead of the site root.
    pythonpath_entries = _vllm_site_pythonpath_prefixes(site) + [site]
    if pythonpath:
        pythonpath_entries.append(pythonpath)
    result = {
        "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        "PYTHONNOUSERSITE": "1",
    }
    ld_library_entries = _vllm_site_ld_library_path(site)
    if ld_library_entries:
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "").strip()
        if existing_ld:
            ld_library_entries.append(existing_ld)
        result["LD_LIBRARY_PATH"] = os.pathsep.join(ld_library_entries)
    shim = os.environ.get("VERL_MLITE_VLLM_LD_PRELOAD", "").strip()
    existing_preload = os.environ.get("LD_PRELOAD", "").strip()
    if shim:
        result["LD_PRELOAD"] = f"{shim}:{existing_preload}" if existing_preload else shim
    elif existing_preload:
        result["LD_PRELOAD"] = existing_preload
    return result


class _RayActorClassProfile:
    """Merge a process profile into one Ray actor class's ``runtime_env``."""

    def __init__(self, actor_class: Any, env_vars: dict[str, str]):
        self._actor_class = actor_class
        self._env_vars = dict(env_vars)

    def options(self, **kwargs: Any) -> Any:
        runtime_env = dict(kwargs.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars.update(self._env_vars)
        runtime_env["env_vars"] = env_vars
        kwargs["runtime_env"] = runtime_env
        return self._actor_class.options(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._actor_class, name)


def _patch_verl_vllm_headless_api_server_count() -> bool:
    """Normalize vLLM's headless API server count for VERL's direct caller."""
    if not os.environ.get("VERL_MLITE_VLLM_SITE", "").strip():
        return False
    if not _vllm_importable():
        return False

    server_module = importlib.import_module(_VLLM_ASYNC_SERVER_MODULE)
    original_run_headless = server_module.run_headless
    if getattr(
        original_run_headless,
        "_verl_mlite_api_server_count_patch",
        False,
    ):
        return False

    @wraps(original_run_headless)
    def patched_run_headless(args: Any) -> Any:
        if getattr(args, "api_server_count", None) is None:
            args.api_server_count = 0
        return original_run_headless(args)

    patched_run_headless._verl_mlite_api_server_count_patch = True
    server_module.run_headless = patched_run_headless
    return True


def _patch_vllm_server_profile() -> bool:
    profile = _vllm_server_profile_env()
    if not profile:
        return False
    if not _vllm_importable():
        return False
    changed = _patch_verl_vllm_headless_api_server_count()
    server_module = importlib.import_module(_VLLM_ASYNC_SERVER_MODULE)
    vLLMReplica = server_module.vLLMReplica

    if getattr(vLLMReplica, "_verl_mlite_server_profile_patch", False):
        return changed
    original_init = vLLMReplica.__init__

    @wraps(original_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.server_class = _RayActorClassProfile(self.server_class, profile)

    vLLMReplica.__init__ = patched_init
    vLLMReplica._verl_mlite_server_profile_patch = True
    return True


def _normalize_vllm_visible_device_id(device_id: int) -> int:
    """Translate a leaked physical CUDA id back to vLLM's visible-list index."""
    visible_devices = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if device_id < 0 or not visible_devices or device_id < len(visible_devices):
        return device_id

    physical_id = str(device_id)
    if physical_id in visible_devices:
        return visible_devices.index(physical_id)
    return device_id


def _patch_verl_vllm_device_uuid() -> bool:
    """Keep VERL/vLLM UUID lookup on the Ray actor's visible CUDA device."""
    if not os.environ.get("VERL_MLITE_VLLM_SITE", "").strip():
        return False
    if not _vllm_importable():
        return False

    utils = importlib.import_module("verl.workers.rollout.vllm_rollout.utils")
    original_get_device_uuid = utils.get_device_uuid
    changed = False
    if getattr(original_get_device_uuid, "_verl_mlite_visible_device_patch", False):
        patched_get_device_uuid = original_get_device_uuid
    else:
        @wraps(original_get_device_uuid)
        def patched_get_device_uuid(device_id: int) -> str:
            return original_get_device_uuid(
                _normalize_vllm_visible_device_id(device_id)
            )

        patched_get_device_uuid._verl_mlite_visible_device_patch = True
        utils.get_device_uuid = patched_get_device_uuid
        changed = True

    # Importing the leaf ``utils`` module first executes the package __init__,
    # which can bind the original helper into this consumer before we replace it.
    consumer = sys.modules.get(_VLLM_ROLLOUT_CONSUMER_MODULE)
    if (
        consumer is not None
        and getattr(consumer, "get_device_uuid", None) is not patched_get_device_uuid
    ):
        consumer.get_device_uuid = patched_get_device_uuid
        changed = True
    return changed


def _patch_transformers_rope_ignore_keys() -> None:
    try:
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        return

    for cls in vars(rope_utils).values():
        if not isinstance(cls, type):
            continue
        if getattr(cls, "_verl_mlite_rope_ignore_keys_patch", False):
            continue
        descriptor = vars(cls).get("_check_received_keys")
        if descriptor is None:
            continue

        is_staticmethod = isinstance(descriptor, staticmethod)
        is_classmethod = isinstance(descriptor, classmethod)
        original = descriptor.__func__ if is_staticmethod or is_classmethod else descriptor

        def build_wrapper(check_received_keys: Any) -> Any:
            @wraps(check_received_keys)
            def patched(*args: Any, **kwargs: Any) -> Any:
                ignore_keys = kwargs.get("ignore_keys")
                if isinstance(ignore_keys, list):
                    kwargs["ignore_keys"] = set(ignore_keys)
                elif ignore_keys is not None and not isinstance(ignore_keys, set):
                    if isinstance(ignore_keys, Iterable) and not isinstance(
                        ignore_keys, (str, bytes)
                    ):
                        kwargs["ignore_keys"] = set(ignore_keys)
                return check_received_keys(*args, **kwargs)

            return patched

        patched = build_wrapper(original)
        if is_staticmethod:
            cls._check_received_keys = staticmethod(patched)
        elif is_classmethod:
            cls._check_received_keys = classmethod(patched)
        else:
            cls._check_received_keys = patched
        cls._verl_mlite_rope_ignore_keys_patch = True


def _patch_transformers_apply_chat_template_return_dict() -> bool:
    """Restore Transformers v4's ``apply_chat_template`` list-of-ids return type.

    In Transformers v5 the ``return_dict`` default of
    ``PreTrainedTokenizerBase.apply_chat_template`` flipped from ``False`` to
    ``True``, so a ``tokenize=True`` call now yields a ``BatchEncoding`` mapping
    (``{"input_ids": [...], "attention_mask": [...]}``) instead of a bare
    ``list[int]``. VERL's agent loop
    (``verl.experimental.agent_loop.agent_loop.AgentLoop.apply_chat_template``)
    was written against v4 and treats the result as a token-id list: it forwards
    the value straight into ``TokensPrompt(prompt_token_ids=...)``. vLLM's input
    validator then evaluates ``max(prompt_ids)`` over the mapping's *keys*,
    yielding the string ``"input_ids"`` and crashing the rollout with
    ``TypeError: '>' not supported between instances of 'str' and 'int'``
    (job 13961728, single torch2.12/cu13 stack). We default ``return_dict`` back
    to ``False`` whenever the caller does not set it explicitly, restoring the v4
    contract while leaving callers that request a dict untouched. When
    ``tokenize=False`` Transformers already forces ``return_dict=False``, so this
    only affects the tokenizing path.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:  # pragma: no cover - transformers internals moved
        return False

    original = PreTrainedTokenizerBase.apply_chat_template
    if getattr(original, "_verl_mlite_return_dict_default_patch", False):
        return False

    @wraps(original)
    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        if "return_dict" not in kwargs:
            kwargs["return_dict"] = False
        return original(self, *args, **kwargs)

    patched._verl_mlite_return_dict_default_patch = True
    PreTrainedTokenizerBase.apply_chat_template = patched
    return True


def _trace_runtime_patch(stage: str, result: Any = None) -> None:
    """Report patch ordering only for an explicitly traced startup."""
    if os.environ.get("VERL_MLITE_RUNTIME_PATCH_TRACE") != "1":
        return

    import json

    transformers = sys.modules.get("transformers")
    module_vars = vars(transformers) if transformers is not None else {}
    objects = module_vars.get("_objects")
    missing = object()

    def raw_binding(name: str) -> tuple[Any, str]:
        if name in module_vars:
            return module_vars[name], "namespace"
        if isinstance(objects, dict) and name in objects:
            return objects[name], "_objects"
        return missing, "absent"

    alias, alias_source = raw_binding("AutoModelForVision2Seq")
    replacement, replacement_source = raw_binding(
        "AutoModelForImageTextToText"
    )
    payload = {
        "alias_is_replacement": (
            alias is not missing
            and replacement is not missing
            and alias is replacement
        ),
        "alias_source": alias_source,
        "changed": result,
        "event": "runtime_patch",
        "pid": os.getpid(),
        "replacement_source": replacement_source,
        "step": stage,
        "transformers_file": module_vars.get("__file__"),
        "transformers_id": id(transformers) if transformers is not None else None,
        "transformers_loaded": transformers is not None,
    }
    sys.stderr.write(
        "VERL_MLITE_RUNTIME_PATCH_TRACE "
        f"{json.dumps(payload, sort_keys=True)}\n"
    )
    sys.stderr.flush()


def _patch_verl_dsv4_mxfp4_check() -> bool:
    """Make verl's DeepSeek-V4 mxfp4 MoE check factory-aware for vLLM >= 0.24.

    verl #6473 ships ``verl.utils.vllm.vllm_dsv4_fp8_utils._is_mxfp4_fused_moe_module``
    as ``isinstance(module, FusedMoE) and isinstance(module.quant_method, Mxfp4MoEMethod)``.
    That assumes ``FusedMoE`` is a class. vLLM 0.25.1 refactored ``FusedMoE`` into
    a *factory function* (it returns a ``MoERunner``), so the first ``isinstance``
    raises ``TypeError: isinstance() arg 2 must be a type``. This fires
    unconditionally during the DS4 fp8 resync weight-prep
    (``update_weights_from_ipc`` -> ``prepare_quanted_weights_for_loading`` ->
    ``prepare_deepseek_v4_weights_for_loading`` -> ``_restore_moe_params_for_loading``).

    Replace it with a version that keys off ``module.quant_method`` -- the real
    mxfp4 discriminator (``Mxfp4MoEMethod`` is a proper class in 0.25.1) -- and
    only then confirms a fused-MoE module via the real classes. DeepSeek-V4 uses
    the fp8_ds_mla block layout, not mxfp4, so this returns ``False`` for it and
    the resync proceeds. verl source is untouched; we override our side only.
    """
    if not _vllm_importable():
        return False
    try:
        module = importlib.import_module("verl.utils.vllm.vllm_dsv4_fp8_utils")
    except Exception:
        return False
    existing = getattr(module, "_is_mxfp4_fused_moe_module", None)
    if getattr(existing, "_verl_mlite_factory_aware", False):
        return True

    def _is_mxfp4_fused_moe_module(candidate: Any) -> bool:
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

        quant_method = getattr(candidate, "quant_method", None)
        is_mxfp4 = isinstance(quant_method, Mxfp4MoEMethod)
        # Diagnostic (deduped by type pair): emit the real runtime module and
        # quant_method types so we can confirm the MoE experts are fp8 (not the
        # mxfp4 4-bit fallback). ``FusedMoE`` is a factory function on vLLM
        # >= 0.24, so the original ``isinstance(module, FusedMoE)`` type guard is
        # invalid; the quant_method is the definitive discriminator.
        seen = _is_mxfp4_fused_moe_module.__dict__.setdefault("_diag_seen", set())
        diag_key = (type(candidate).__name__, type(quant_method).__name__)
        if diag_key not in seen:
            seen.add(diag_key)
            sys.stderr.write(
                "VERL_MLITE_MXFP4_DIAG "
                f"module={type(candidate).__module__}.{type(candidate).__name__} "
                f"quant_method={type(quant_method).__module__}."
                f"{type(quant_method).__name__} is_mxfp4={is_mxfp4}\n"
            )
            sys.stderr.flush()
        return is_mxfp4

    _is_mxfp4_fused_moe_module._verl_mlite_factory_aware = True
    module._is_mxfp4_fused_moe_module = _is_mxfp4_fused_moe_module
    return True


def _restore_dsv4_attn_sink_padding(model: Any) -> int:
    """Restore the ``-inf`` FlashMLA sink padding erased by dummy loading."""
    restored = 0
    for module in model.modules():
        sink = getattr(module, "attn_sink", None)
        real_heads = getattr(module, "n_local_heads", None)
        padded_heads = getattr(module, "padded_heads", None)
        if sink is None or not isinstance(real_heads, int) or not isinstance(padded_heads, int):
            continue
        if sink.ndim != 1 or sink.numel() != padded_heads:
            continue
        if not 0 <= real_heads <= padded_heads:
            continue
        if real_heads < padded_heads:
            sink.data[real_heads:].fill_(-float("inf"))
            restored += 1
    return restored


# ============================================================================
# DS4 RL weight-resync: dense FP8 linear finalize (SHORT-TERM bridge)
# ============================================================================
# WHY THIS EXISTS / THE CLEAN DESIGN WE ARE NOT YET DOING:
#   A resync should be "cold-load replayed with fresh weights": the only place
#   that knows kernel-runtime layout (deepgemm ue8m0 requant, wo_a is_bmm 2D->3D,
#   indexer wk fusion) is vLLM's native load_weights + process_weights_after_loading;
#   resync should reuse it so rollout weights == cold-load weights bit-for-bit.
#
#   Clean 3-layer contract (target end state, NOT implemented here):
#     1. producer (mlite): emit a neutral self-describing typed weight stream
#        WeightSpec{dtype: bf16|fp8_e4m3|fp4_e2m1, quant, scale, scale_dtype};
#        pick a *format* (bf16/block_fp8/block_fp4), never name a consumer.
#     2. transport: format-agnostic byte-aligned buckets, carries (name,tensor,spec).
#     3. consumer adapter: optional transcode (bf16->fp8 / fp4->fp8 = "verl does
#        arbitrary quant") -> ONE uniform reset_weights_for_reload() over every quant
#        module (MoE/dense/indexer) -> native load -> native process-all (once).
#
#   The current path instead rides verl #6473's is_fp8_model refit
#   (load_quanted_weights -> _restore_moe_params / _prepare_linear_params /
#   process_deepseek_v4_weights_after_loading), which finalizes only MoE and leaves
#   dense attn/indexer FP8 linears in checkpoint layout != cold-load (verified:
#   weight+scale sha differ, indexer.wq_b drift ~4.5%; sparse indexer amplifies into
#   decode-only rollout divergence). #6473 is itself hacky and we do not want to
#   extend it.
#
# TODO(upstream, remove this bridge once landed):
#   * verl: file an issue proposing the neutral typed-stream contract + a single
#     type-agnostic reset (not the per-quant-type _restore_moe/_prepare_linear refit).
#   * vLLM: file a PR adding reset_weights_for_reload() (or make
#     process_weights_after_loading idempotent/reversible) so RL refit can cleanly
#     reuse the cold-load path.
#
# SHORT-TERM (below): recreate dense block-FP8 linear params to checkpoint layout
# before the resync load (create_weights, uniform, incl wo_a is_bmm) so the single
# post-load process matches cold-load. This is a bridge, scoped to DS4, additive.
# ============================================================================
def _recreate_dense_fp8_linear_params(model) -> int:
    """Reset every dense block-FP8 LinearBase to fresh checkpoint-layout params so
    the single post-load process_weights_after_loading matches cold load bit-for-bit
    (see module block above). Returns the count recreated."""
    import torch
    try:
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
    except Exception:
        return 0
    recreated = 0
    for _name, layer in model.named_modules():
        if not (isinstance(layer, LinearBase)
                and isinstance(getattr(layer, "quant_method", None), Fp8LinearMethod)):
            continue
        qm = layer.quant_method
        if not getattr(qm, "block_quant", False):
            continue
        old_w = getattr(layer, "weight", None)
        wl = getattr(old_w, "weight_loader", None) if old_w is not None else None
        if wl is None:
            continue
        try:
            qm.create_weights(
                layer,
                input_size_per_partition=int(layer.input_size_per_partition),
                output_partition_sizes=list(layer.output_partition_sizes),
                input_size=int(layer.input_size),
                output_size=int(layer.output_size),
                params_dtype=getattr(layer, "orig_dtype", torch.bfloat16),
                weight_loader=wl,
            )
            recreated += 1
        except Exception as _re:
            sys.stderr.write(f"VERL_MLITE_DENSE_RECREATE_SKIP {_name}: {_re!r}\n")
            sys.stderr.flush()
            raise RuntimeError(
                f"failed to recreate dense FP8 parameters for {_name}"
            ) from _re
    return recreated


def _patch_verl_dsv4_prepare_recreates_dense() -> bool:
    """SHORT-TERM bridge: wrap prepare_quanted_weights_for_loading so DS4 dense
    block-FP8 linears are recreated to checkpoint layout before every resync load,
    pairing with the post-load dense process in _patch_verl_dsv4_fp8_process_weights.
    See the module block above for the clean design + upstream TODOs."""
    if not _vllm_importable():
        return False
    try:
        mod = importlib.import_module("verl.utils.vllm.vllm_fp8_utils")
    except Exception:
        return False
    original = getattr(mod, "prepare_quanted_weights_for_loading", None)
    if original is None or getattr(original, "_verl_mlite_dense_recreate", False):
        return True

    @wraps(original)
    def prepare_quanted_weights_for_loading(model_runner, *args, **kwargs):
        try:
            from verl.utils.vllm.vllm_dsv4_fp8_utils import is_deepseek_v4_model
            from vllm.config import set_current_vllm_config

            model = model_runner.model
            if is_deepseek_v4_model(model):
                # Fp8LinearMethod.create_weights consults vLLM's process-global
                # config. Online reload runs outside the cold model-loader
                # context, so restore that context while recreating parameters.
                with set_current_vllm_config(model_runner.vllm_config):
                    n = _recreate_dense_fp8_linear_params(model)
                if n:
                    sys.stderr.write(
                        f"VERL_MLITE_DENSE_RECREATE recreated {n} dense FP8 linear "
                        "param set(s) to checkpoint layout before resync load\n")
                    sys.stderr.flush()
        except Exception as exc:
            sys.stderr.write(f"VERL_MLITE_DENSE_RECREATE error: {exc!r}\n")
            sys.stderr.flush()
            raise
        # Ordering is part of the online-reload contract.  verl's DS4 prepare
        # step wraps the *current* parameters and attaches loaders which accept
        # already-local TP shards from IPC.  Recreating after that step replaces
        # those wrapped parameters and silently restores the cold-load loader,
        # which slices the local shard by TP a second time.  Reset first, then
        # let verl attach its online loaders to the fresh checkpoint layout.
        return original(model_runner, *args, **kwargs)

    prepare_quanted_weights_for_loading._verl_mlite_dense_recreate = True
    mod.prepare_quanted_weights_for_loading = prepare_quanted_weights_for_loading
    return True


def _patch_verl_dsv4_fp8_prepare_state() -> bool:
    """Keep pure-FP8 DS4 reload preparation outside the per-bucket loop.

    VERL uses the truthiness of ``prepare_quanted_weights_for_loading`` to tell
    ``_update_weights`` that preparation already ran once before IPC receive.
    The DS4 helper returns only whether MXFP4/MegaMoE parameters were restored,
    so a pure-FP8 model incorrectly returns ``False``. Each bucket then repeats
    prepare/process and recreates dense parameters underneath earlier loads.
    """
    if not _vllm_importable():
        return False
    try:
        fp8_utils = importlib.import_module("verl.utils.vllm.vllm_fp8_utils")
    except Exception:
        return False
    original = getattr(fp8_utils, "prepare_quanted_weights_for_loading", None)
    if original is None or getattr(original, "_verl_mlite_ds4_fp8_state", False):
        return original is not None

    @wraps(original)
    def prepare_quanted_weights_for_loading(model_runner, *args, **kwargs):
        state = original(model_runner, *args, **kwargs)
        if state:
            return state

        model = model_runner.model
        from verl.utils.vllm.vllm_dsv4_fp8_utils import is_deepseek_v4_model
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        pure_fp8 = is_deepseek_v4_model(model) and any(
            isinstance(module, RoutedExperts)
            and isinstance(getattr(module, "quant_method", None), Fp8MoEMethod)
            for module in model.modules()
        )
        if pure_fp8:
            sys.stderr.write(
                "VERL_MLITE_FP8_PREPARE_STATE promoted DS4 pure-FP8 reload "
                "state to one-shot lifecycle\n"
            )
            sys.stderr.flush()
            return True
        return state

    prepare_quanted_weights_for_loading._verl_mlite_ds4_fp8_state = True
    fp8_utils.prepare_quanted_weights_for_loading = prepare_quanted_weights_for_loading
    return True


_DSV4_LAYERWISE_RELOAD_STATE = object()


def _patch_verl_dsv4_native_layerwise_reload() -> bool:
    """Use vLLM's native layerwise reload lifecycle for DS4 FP8 resync.

    vLLM records checkpoint-layout metadata during model construction.  Its
    layerwise reload API restores that layout, buffers complete logical layers,
    runs each quant method's native finalizer exactly once, then copies results
    back into the original kernel storage (preserving cudagraph references).
    This is the supported reset/load/process contract; it replaces verl's DS4
    boolean prepare state and our former dense recreation bridge.
    """
    if not _vllm_importable():
        return False
    try:
        fp8_utils = importlib.import_module("verl.utils.vllm.vllm_fp8_utils")
        dsv4_utils = importlib.import_module("verl.utils.vllm.vllm_dsv4_fp8_utils")
        rollout_utils = importlib.import_module(
            "verl.workers.rollout.vllm_rollout.utils"
        )
    except Exception:
        return False
    original_prepare = getattr(fp8_utils, "prepare_quanted_weights_for_loading", None)
    original_process = getattr(fp8_utils, "process_quanted_weights_after_loading", None)
    original_load = getattr(fp8_utils, "load_quanted_weights", None)
    if original_prepare is None or original_process is None or original_load is None:
        return False
    if getattr(original_prepare, "_verl_mlite_ds4_layerwise", False):
        return True

    @wraps(original_prepare)
    def prepare_quanted_weights_for_loading(model_runner, *args, **kwargs):
        model = model_runner.model
        if not dsv4_utils.is_deepseek_v4_model(model):
            return original_prepare(model_runner, *args, **kwargs)
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload
        from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS

        # These DS4 buffers already live in kernel/runtime layout and are
        # updated directly by VERL's buffer path (or restored below).  Keeping
        # them out of the meta restore prevents ``copy_`` into a meta buffer
        # from silently discarding router state between IPC buckets.
        SKIP_TENSORS.update(
            {"tid2eid", "expert_bias", "e_score_correction_bias", "attn_sink"}
        )
        with set_current_vllm_config(model_runner.vllm_config):
            initialize_layerwise_reload(model)
        model._verl_mlite_ds4_layerwise_reload_active = True
        sys.stderr.write(
            "VERL_MLITE_DSV4_LAYERWISE_RELOAD initialized native vLLM reload\n"
        )
        sys.stderr.flush()
        return _DSV4_LAYERWISE_RELOAD_STATE

    @wraps(original_process)
    def process_quanted_weights_after_loading(model_runner, reload_state):
        if reload_state is not _DSV4_LAYERWISE_RELOAD_STATE:
            return original_process(model_runner, reload_state)
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import finalize_layerwise_processing

        try:
            with set_current_vllm_config(model_runner.vllm_config):
                finalize_layerwise_processing(
                    model_runner.model,
                    model_runner.vllm_config.model_config,
                )
        finally:
            model_runner.model._verl_mlite_ds4_layerwise_reload_active = False
        restored_sinks = _restore_dsv4_attn_sink_padding(model_runner.model)
        # ``load_quanted_weights`` clones IPC bucket views because vLLM's
        # layerwise loader may retain them until a complete logical layer is
        # available.  ``finalize_layerwise_processing`` drops those references,
        # but PyTorch's caching allocator otherwise keeps the temporary device
        # storage mapped.  In colocated sleep mode that storage competes with
        # vLLM's cuMem weight mappings on the next wake-up and can OOM even
        # though no live tensor owns it (observed on every rank of a 128-GPU
        # DS4 run).  Release the now-dead staging blocks at the lifecycle
        # boundary; doing this per IPC bucket would be both too early and slow.
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        records = getattr(model_runner.model, "_verl_mlite_weight_fingerprint", None)
        if records is not None:
            from megatron.lite.primitive.ckpt.weight_sync_fingerprint import (
                report_stream_fingerprint,
            )

            report_stream_fingerprint("receiver", 0, records)
            del model_runner.model._verl_mlite_weight_fingerprint
        sys.stderr.write(
            "VERL_MLITE_DSV4_LAYERWISE_RELOAD finalized native vLLM reload "
            f"attention_sinks_restored={restored_sinks}\n"
        )
        sys.stderr.flush()

    @wraps(original_load)
    def load_quanted_weights(weights, model_runner, *args, **kwargs):
        model = model_runner.model
        if getattr(model, "_verl_mlite_ds4_layerwise_reload_active", False):
            weights = list(weights)
            from megatron.lite.primitive.ckpt.weight_sync_fingerprint import (
                tensor_fingerprint_record,
                weight_sync_fingerprint_enabled,
            )

            if weight_sync_fingerprint_enabled():
                records = getattr(model, "_verl_mlite_weight_fingerprint", None)
                if records is None:
                    records = []
                    model._verl_mlite_weight_fingerprint = records
                records.extend(
                    tensor_fingerprint_record(name, tensor) for name, tensor in weights
                )
            # Layerwise reload may retain loader arguments across multiple IPC
            # callbacks.  The receiver reuses its communication buffer after
            # each callback, so persist every tensor until its logical layer is
            # finalized instead of retaining a view into overwritten storage.
            weights = [(name, tensor.clone()) for name, tensor in weights]
        return original_load(weights, model_runner, *args, **kwargs)

    prepare_quanted_weights_for_loading._verl_mlite_ds4_layerwise = True
    process_quanted_weights_after_loading._verl_mlite_ds4_layerwise = True
    load_quanted_weights._verl_mlite_ds4_layerwise = True
    fp8_utils.prepare_quanted_weights_for_loading = prepare_quanted_weights_for_loading
    fp8_utils.process_quanted_weights_after_loading = process_quanted_weights_after_loading
    fp8_utils.load_quanted_weights = load_quanted_weights
    # ``utils.py`` imports this function at module import time, so replacing the
    # defining module alone would leave the live rollout callback on the stale
    # binding and defeat cross-bucket tensor persistence.
    rollout_utils.load_quanted_weights = load_quanted_weights
    return True


def _patch_verl_dsv4_fp8_process_weights() -> bool:
    """Restore the proven pre-dense-bridge DS4 post-reload behavior.

    Pure-FP8 routed experts still need their native MoE finalize once after the
    bucket stream, and attention sink padding must be restored.  Dense FP8
    linears are intentionally *not* reset or re-finalized here: vLLM preserves
    their RL reload loaders and an extra dense finalize is not idempotent.
    """
    if not _vllm_importable():
        return False
    try:
        utils = importlib.import_module(
            "verl.workers.rollout.vllm_rollout.utils"
        )
    except Exception:
        return False
    ext_cls = getattr(utils, "vLLMColocateWorkerExtension", None)
    original = getattr(ext_cls, "update_weights_from_ipc", None)
    if original is None or getattr(original, "_verl_mlite_fp8_process", False):
        return True

    @wraps(original)
    def update_weights_from_ipc(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = self.model_runner.model
        from verl.utils.vllm.vllm_dsv4_fp8_utils import is_deepseek_v4_model

        if not is_deepseek_v4_model(model):
            return result
        restored_sinks = _restore_dsv4_attn_sink_padding(model)
        if restored_sinks:
            sys.stderr.write(
                "VERL_MLITE_ATTN_SINK_PADDING "
                f"restored -inf padding on {restored_sinks} DS4 attention module(s)\n"
            )
            sys.stderr.flush()
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        processed = 0
        for module in model.modules():
            if isinstance(module, RoutedExperts) and isinstance(
                getattr(module, "quant_method", None), Fp8MoEMethod
            ):
                module.quant_method.process_weights_after_loading(module)
                processed += 1
        if processed:
            sys.stderr.write(
                "VERL_MLITE_FP8_PROCESS "
                f"ran process_weights_after_loading on {processed} "
                "DS4 RoutedExperts+Fp8MoEMethod module(s) after resync\n"
            )
            sys.stderr.flush()
        return result

    update_weights_from_ipc._verl_mlite_fp8_process = True
    ext_cls.update_weights_from_ipc = update_weights_from_ipc
    return True


class _SyncBucketProducer:
    """Pack one sender bucket at a time into a caller-owned staging slot."""

    def __init__(self, weights, bucket_size: int):
        from verl_mlite.rollout.layer_cluster import resync_layer_cluster_key

        self._weights = iter(weights)
        self._bucket_size = bucket_size
        self._pending = None
        self._exhausted = False
        self._layer_cluster_key = resync_layer_cluster_key

    def next_bucket(self, staging):
        import torch

        if staging.device.type == "cuda":
            torch.cuda.set_device(staging.device)
        if self._exhausted:
            return "eof", None, None, 0, None, True

        offset = 0
        bucket_meta = {}
        bucket_layer_key = None
        while True:
            try:
                if self._pending is None:
                    name, weight = next(self._weights)
                else:
                    name, weight = self._pending
                    self._pending = None
            except StopIteration:
                self._exhausted = True
                if not bucket_meta:
                    return "eof", None, None, 0, None, True
                break

            layer_key = self._layer_cluster_key(name)
            if (
                bucket_meta
                and bucket_layer_key is not None
                and layer_key != bucket_layer_key
            ):
                self._pending = (name, weight)
                break

            # Fix-A (resync IPC byte-alignment): pad every tensor's start
            # offset up to an 8-byte boundary. The receiver reconstructs each
            # tensor as ``buffer[offset:offset+size].view(dtype)``, and
            # ``Tensor.view(dtype)`` requires the byte ``storage_offset`` to be
            # divisible by ``dtype.itemsize``. A pure BF16/FP32 stream keeps
            # every offset even, so it never trips (why the proxy stays green).
            # But DS4 real-weight resync ships block-FP8 tensors (itemsize 1);
            # an odd-numel FP8 tensor leaves ``offset`` odd and the *next*
            # BF16/FP32 tensor in the same bucket crashes on the view. Aligning
            # to 8 bytes covers every dtype we transport (fp8/bf16/fp16/fp32)
            # and is byte-lossless — the receiver reads the padded offset we
            # record in ``bucket_meta`` unchanged. See
            # tests/unit/verl/test_resync_bucket_byte_alignment.py.
            offset = (offset + 7) & ~7
            if offset + weight.nbytes > self._bucket_size and bucket_meta:
                self._pending = (name, weight)
                break
            if weight.nbytes > self._bucket_size:
                return "direct", name, weight, 0, None, False

            if not bucket_meta:
                bucket_layer_key = layer_key
            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
                "handle": None,
            }
            staging[offset : offset + weight.nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True
            )
            offset += weight.nbytes
            # Padding can push a full bucket a few bytes past exact equality, so
            # break on ``>=`` rather than ``==``.
            if offset >= self._bucket_size:
                break

        ready = None
        if staging.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(staging.device))
        return "bucket", bucket_meta, None, offset, ready, self._exhausted


def _install_bucketed_sender_prefetch(sender_cls: type) -> bool:
    """Overlap synchronous weight production with the receiver's bucket ACK."""
    if getattr(sender_cls, "_mlite_weight_prefetch_patch", False):
        return False

    original_async_send_weights = sender_cls.async_send_weights

    async def prefetched_async_send_weights(self, weights):
        import torch

        if not isinstance(weights, Iterable) or hasattr(weights, "__aiter__") or self.use_shm:
            return await original_async_send_weights(self, weights)

        executor = None
        stop = threading.Event()
        free_slots = None
        ready_results = None
        held_slot = None
        try:
            self._init_socket()
            self._init_buffer()
            if self.buffer.device.type != "cuda" and not getattr(
                self, "_mlite_prefetch_allow_cpu", False
            ):
                raise RuntimeError("MLite sender prefetch requires a CUDA IPC buffer")

            staging_slots = [torch.empty_like(self.buffer) for _ in range(2)]
            producer = _SyncBucketProducer(weights, self.bucket_size)
            free_slots = queue.Queue(maxsize=2)
            ready_results = queue.Queue(maxsize=2)
            for slot_index in range(2):
                free_slots.put_nowait(slot_index)
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlite-weight-prefetch")

            def put_ready(result):
                while not stop.is_set():
                    try:
                        ready_results.put(result, timeout=0.05)
                        return True
                    except queue.Full:
                        continue
                return False

            def produce():
                slot_index = None
                try:
                    while not stop.is_set():
                        try:
                            slot_index = free_slots.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        result = producer.next_bucket(staging_slots[slot_index])
                        if not put_ready((*result, slot_index)):
                            return
                        slot_index = None
                        kind, *_, is_last = result
                        if kind == "eof" or is_last:
                            return
                except BaseException as exc:
                    put_ready(("error", exc, None, 0, None, True, slot_index))

            context = copy_context()
            worker_future = executor.submit(context.run, produce)
            while True:
                try:
                    result = ready_results.get(timeout=0.1)
                except queue.Empty:
                    if worker_future.done():
                        worker_future.result()
                        raise RuntimeError("MLite weight prefetch stopped without a terminal result")
                    continue

                kind, metadata_or_name, direct_weight, used_bytes, ready, is_last, held_slot = (
                    result
                )
                if kind == "error":
                    raise metadata_or_name
                if kind == "eof":
                    free_slots.put_nowait(held_slot)
                    held_slot = None
                    self.socket.send_pyobj({"bucket_meta": {}, "is_last": True})
                    self.socket.recv()
                    break
                if kind == "direct":
                    free_slots.put_nowait(held_slot)
                    held_slot = None
                    self._direct_send_large_weight(metadata_or_name, direct_weight)
                    continue

                if ready is not None:
                    ready.synchronize()
                staging = staging_slots[held_slot]
                self.buffer[:used_bytes].copy_(staging[:used_bytes], non_blocking=True)
                if self.buffer.device.type == "cuda":
                    torch.cuda.synchronize(self.buffer.device)
                free_slots.put_nowait(held_slot)
                held_slot = None

                self.socket.send_pyobj(
                    {"bucket_meta": metadata_or_name, "is_last": is_last}
                )
                self.socket.recv()
                if is_last:
                    break
        finally:
            stop.set()
            if held_slot is not None and free_slots is not None:
                try:
                    free_slots.put_nowait(held_slot)
                except queue.Full:
                    pass
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            self._cleanup()

    sender_cls.async_send_weights = prefetched_async_send_weights
    sender_cls._mlite_weight_prefetch_patch = True
    return True


def _weight_sync_probe_enabled() -> bool:
    return os.getenv("MLITE_WEIGHT_SYNC_PROBE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _weight_sync_fingerprint_enabled() -> bool:
    return os.getenv("MLITE_WEIGHT_SYNC_FINGERPRINT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _instrument_bucketed_weight_sender(sender_cls: type) -> bool:
    """Patch veRL's sender only while the opt-in sync probe is enabled."""
    if getattr(sender_cls, "_mlite_weight_sync_probe_patch", False):
        return False

    import torch
    import torch.distributed as dist
    from torch.utils._python_dispatch import TorchDispatchMode

    from megatron.lite.primitive.ckpt.weight_sync_probe import (
        get_weight_sync_probe,
        weight_sync_probe_session,
    )

    probe = get_weight_sync_probe()
    original_init_socket = sender_cls._init_socket
    original_async_send_weights = sender_cls.async_send_weights

    class _ProfiledSocket:
        def __init__(self, socket):
            self._socket = socket

        def __getattr__(self, name):
            return getattr(self._socket, name)

        def send_pyobj(self, *args, **kwargs):
            with probe.measure("handshake"):
                return self._socket.send_pyobj(*args, **kwargs)

        def recv(self, *args, **kwargs):
            with probe.measure("handshake"):
                return self._socket.recv(*args, **kwargs)

    class _H2DCopyMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.ops.aten.copy_.default and len(args) >= 2:
                dst, src = args[:2]
                if (
                    isinstance(dst, torch.Tensor)
                    and isinstance(src, torch.Tensor)
                    and dst.device.type == "cuda"
                    and src.device.type == "cpu"
                ):
                    with probe.measure("h2d", nbytes=src.nbytes, device=dst.device):
                        return func(*args, **kwargs)
            return func(*args, **kwargs)

    def profiled_init_socket(self, *args, **kwargs):
        result = original_init_socket(self, *args, **kwargs)
        self.socket = _ProfiledSocket(self.socket)
        return result

    async def profiled_async_send_weights(self, weights):
        backend = os.getenv("MLITE_WEIGHT_SYNC_PROBE_BACKEND", "unknown")
        from megatron.lite.primitive.ckpt.weight_sync_fingerprint import (
            report_stream_fingerprint,
            tensor_fingerprint_record,
            weight_sync_fingerprint_enabled,
        )

        fingerprint_records = []

        def fingerprinted_weights():
            for name, tensor in weights:
                if weight_sync_fingerprint_enabled():
                    fingerprint_records.append(tensor_fingerprint_record(name, tensor))
                yield name, tensor

        original_all_gather_into_tensor = dist.all_gather_into_tensor

        def profiled_all_gather_into_tensor(output, tensor, *args, **kwargs):
            with probe.measure("mbridge_gather", nbytes=output.nbytes, device=tensor.device):
                return original_all_gather_into_tensor(output, tensor, *args, **kwargs)

        with weight_sync_probe_session(backend), _H2DCopyMode():
            dist.all_gather_into_tensor = profiled_all_gather_into_tensor
            try:
                result = await original_async_send_weights(self, fingerprinted_weights())
                if weight_sync_fingerprint_enabled():
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    report_stream_fingerprint("sender", rank, fingerprint_records)
                return result
            finally:
                dist.all_gather_into_tensor = original_all_gather_into_tensor

    sender_cls._init_socket = profiled_init_socket
    sender_cls.async_send_weights = profiled_async_send_weights
    sender_cls._mlite_weight_sync_probe_patch = True
    return True


class _SenderPatchLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader):
        self._loader = loader

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _install_bucketed_sender_prefetch(module.BucketedWeightSender)
        if _weight_sync_probe_enabled() or _weight_sync_fingerprint_enabled():
            _instrument_bucketed_weight_sender(module.BucketedWeightSender)


class _SenderPatchFinder(importlib.abc.MetaPathFinder):
    _mlite_weight_sync_probe_finder = True

    def __init__(self):
        self._mlite_weight_sync_probe_requested = False

    def find_spec(self, fullname, path, target=None):
        if fullname != _BUCKETED_SENDER_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _SenderPatchLoader(spec.loader)
        return spec


def _patch_bucketed_weight_transfer() -> bool:
    module = sys.modules.get(_BUCKETED_SENDER_MODULE)
    if module is not None:
        return _install_bucketed_sender_prefetch(module.BucketedWeightSender)
    if any(getattr(finder, "_mlite_weight_sync_probe_finder", False) for finder in sys.meta_path):
        return False
    sys.meta_path.insert(0, _SenderPatchFinder())
    return True


def _patch_bucketed_weight_sender() -> bool:
    """Install production prefetch plus optional probe instrumentation."""
    changed = _patch_bucketed_weight_transfer()
    if not (_weight_sync_probe_enabled() or _weight_sync_fingerprint_enabled()):
        return changed

    module = sys.modules.get(_BUCKETED_SENDER_MODULE)
    if module is not None:
        changed = _instrument_bucketed_weight_sender(module.BucketedWeightSender) or changed
    else:
        finder = next(
            finder
            for finder in sys.meta_path
            if getattr(finder, "_mlite_weight_sync_probe_finder", False)
        )
        if not finder._mlite_weight_sync_probe_requested:
            finder._mlite_weight_sync_probe_requested = True
            changed = True
    return changed


def apply_runtime_patches() -> None:
    _trace_runtime_patch("00.begin")
    result = _patch_transformers_vision2seq_alias()
    _trace_runtime_patch("01.transformers_alias", result)
    result = _register_opaque_hf_config()
    _trace_runtime_patch("02.opaque_hf_config", result)
    result = _install_vllm_thin_finder()
    _trace_runtime_patch("03.vllm_thin_finder", result)
    result = _install_vllm_triton_kernels_alias()
    _trace_runtime_patch("04.vllm_triton_kernels_alias", result)
    result = _patch_verl_vllm_device_uuid()
    _trace_runtime_patch("05.verl_vllm_device_uuid", result)
    # Importing VERL's vLLM utilities can rebuild Transformers' lazy top-level
    # module, which drops compatibility attributes installed on the old module.
    result = _patch_transformers_vision2seq_alias()
    _trace_runtime_patch("06.transformers_alias_after_uuid", result)
    result = _patch_transformers_rope_ignore_keys()
    _trace_runtime_patch("07.transformers_rope_ignore_keys", result)
    result = _patch_transformers_apply_chat_template_return_dict()
    _trace_runtime_patch("07b.transformers_apply_chat_template_return_dict", result)
    result = _patch_bucketed_weight_sender()
    _trace_runtime_patch("08.bucketed_weight_sender", result)
    result = _patch_verl_dsv4_mxfp4_check()
    _trace_runtime_patch("08b.verl_dsv4_mxfp4_check", result)
    result = _patch_verl_dsv4_native_layerwise_reload()
    _trace_runtime_patch("08c.verl_dsv4_native_layerwise_reload", result)
    result = _patch_vllm_server_profile()
    _trace_runtime_patch("09.vllm_server_profile", result)
    _trace_runtime_patch("10.end")


def _load_verl_file(relative_path: str, module_name: str):
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("No module named 'verl'")

    path = Path(next(iter(spec.submodule_search_locations))) / relative_path
    file_spec = importlib.util.spec_from_file_location(module_name, path)
    if file_spec is None or file_spec.loader is None:
        raise ImportError(f"Unable to load VERL module from {path}")

    module = importlib.util.module_from_spec(file_spec)
    sys.modules[module_name] = module
    file_spec.loader.exec_module(module)
    return module


def load_verl_engine_api():
    # Prefer the canonical package import so the MLite engine registers into the
    # SAME EngineRegistry that verl's trainers resolve against. Loading base.py as
    # a standalone module (below) creates a *duplicate* registry, which silently
    # drops the mlite backend ("Unknown backend: mlite"). The file-load path is
    # only a fallback for environments where verl isn't importable as a package.
    try:
        from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
        from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches
    except (ModuleNotFoundError, ImportError):
        base = _load_verl_file("workers/engine/base.py", "_verl_mlite_verl_engine_base")
        utils = _load_verl_file("workers/engine/utils.py", "_verl_mlite_verl_engine_utils")
        BaseEngine = base.BaseEngine
        BaseEngineCtx = base.BaseEngineCtx
        EngineRegistry = base.EngineRegistry
        postprocess_batch_func = utils.postprocess_batch_func
        prepare_micro_batches = utils.prepare_micro_batches

    return BaseEngine, BaseEngineCtx, EngineRegistry, postprocess_batch_func, prepare_micro_batches

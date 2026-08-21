# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small compatibility patches for dependency-version gaps in examples."""

from __future__ import annotations

import gc
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Iterable
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
        return isinstance(quant_method, Mxfp4MoEMethod)

    _is_mxfp4_fused_moe_module._verl_mlite_factory_aware = True
    module._is_mxfp4_fused_moe_module = _is_mxfp4_fused_moe_module
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
    """Install the MLite-aware bucket packer without moving model work off-thread.

    ``weights`` is not an ordinary host iterator: advancing it can enter FSDP
    parameter materialization and PP/EP collectives.  Those operations must run
    on the actor thread which owns the rank's CUDA device and model lifecycle.
    Only the produced tensors may be copied into the IPC staging buffer here.
    """
    if getattr(sender_cls, "_mlite_weight_prefetch_patch", False):
        return False

    original_async_send_weights = sender_cls.async_send_weights

    async def prefetched_async_send_weights(self, weights):
        import torch

        if not isinstance(weights, Iterable) or hasattr(weights, "__aiter__") or self.use_shm:
            return await original_async_send_weights(self, weights)

        try:
            self._init_socket()
            self._init_buffer()
            if self.buffer.device.type != "cuda" and not getattr(
                self, "_mlite_prefetch_allow_cpu", False
            ):
                raise RuntimeError("MLite sender prefetch requires a CUDA IPC buffer")

            staging = torch.empty_like(self.buffer)
            producer = _SyncBucketProducer(weights, self.bucket_size)
            while True:
                kind, metadata_or_name, direct_weight, used_bytes, ready, is_last = (
                    producer.next_bucket(staging)
                )
                if kind == "eof":
                    self.socket.send_pyobj({"bucket_meta": {}, "is_last": True})
                    self.socket.recv()
                    break
                if kind == "direct":
                    self._direct_send_large_weight(metadata_or_name, direct_weight)
                    continue

                if ready is not None:
                    ready.synchronize()
                self.buffer[:used_bytes].copy_(staging[:used_bytes], non_blocking=True)
                if self.buffer.device.type == "cuda":
                    torch.cuda.synchronize(self.buffer.device)

                self.socket.send_pyobj(
                    {"bucket_meta": metadata_or_name, "is_last": is_last}
                )
                self.socket.recv()
                if is_last:
                    break
        finally:
            self._cleanup()

    sender_cls.async_send_weights = prefetched_async_send_weights
    sender_cls._mlite_weight_prefetch_patch = True
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


class _SenderPatchFinder(importlib.abc.MetaPathFinder):
    _mlite_weight_sender_finder = True

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
    if any(getattr(finder, "_mlite_weight_sender_finder", False) for finder in sys.meta_path):
        return False
    sys.meta_path.insert(0, _SenderPatchFinder())
    return True


def _patch_bucketed_weight_sender() -> bool:
    """Install bounded producer prefetch for VERL's bucketed sender."""
    return _patch_bucketed_weight_transfer()


def apply_runtime_patches() -> None:
    _patch_transformers_vision2seq_alias()
    _register_opaque_hf_config()
    _install_vllm_thin_finder()
    _install_vllm_triton_kernels_alias()
    _patch_verl_vllm_device_uuid()
    # Importing VERL's vLLM utilities can rebuild Transformers' lazy top-level
    # module, which drops compatibility attributes installed on the old module.
    _patch_transformers_vision2seq_alias()
    _patch_transformers_rope_ignore_keys()
    _patch_transformers_apply_chat_template_return_dict()
    _patch_bucketed_weight_sender()
    _patch_verl_dsv4_mxfp4_check()
    _patch_vllm_server_profile()


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

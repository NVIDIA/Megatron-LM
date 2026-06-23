"""Monkey-patch Triton's `Autotuner.check_disk_cache` to keep cache hits
even when some configs have `pre_hook`.

Upstream Triton bails out of disk caching whenever *any* config in the
autotune set has a `pre_hook`:

    # We can't serialize prehooks, so just give up and run the benchmarks.
    if not tuning_key or any(cfg.pre_hook for cfg in configs):
        bench_fn()
        return False

Save side also drops pre_hook configs:

    "configs_timings": [(config.__dict__, timings)
                       for config, timings in self.configs_timings.items()
                       if not config.pre_hook],

Practical effect at scale: Triton autotuning lines on warm-cache cold
restarts stay >0 for kernels (TE/hybrid_ep) that register a pre_hook on at
least one Config. This patch:

1. Identifies each Config by a JSON-friendly dict (kwargs + num_warps/...,
   and `pre_hook.__qualname__` if any). Configs with unnamed pre_hooks
   (lambdas/closures) are excluded — they would not round-trip.
2. On load: matches cached entries against the live `configs` list by
   structural equality. If every config has a hit, skip benching. Otherwise
   fall back to upstream behavior (re-bench the whole set).
3. On save: includes pre_hook configs (keyed by `__qualname__`) instead of
   filtering them out.

Forces `cache_results=True` at construction so the disk-cache path is always
reached regardless of caller intent.

Wiring (prefer `arm()` over `install()` so we win the import-order race):
  * `arm()` registers a `MetaPathFinder` that intercepts the first
    `import triton.runtime.autotuner` and applies the patch right after the
    module body finishes executing. Survives `from triton...` and lazy imports.
  * `install()` patches in place if `triton.runtime.autotuner` is already
    loaded. Used as a follow-up after `arm()` for the case where Triton was
    pulled in before our sitecustomize ran (rare but possible via pth files).
  * Sitecustomize shim calls `arm()` then `install()` — `arm()` covers the
    forward path, `install()` covers the already-loaded path.

Env vars:
  * `TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE=1` skips both arm and install
    (kill switch).
  * `TRITON_AUTOTUNE_PATCH_DEBUG=1` emits `[TRITON_PATCH] ...` lines to
    stderr at each lifecycle point (arm, install, MetaPathFinder fire,
    apply) and prints a counter summary at process exit
    (full_hit / partial_hit / no_cache_file / no_key / unstable_prehook /
    load_error). Use this to disambiguate a null result:
    if `[TRITON_PATCH] patch applied` is missing, the patch never loaded;
    if it's present but the summary shows full_hit=0 with high
    no_cache_file, the cache is cold for these shapes (priming run needed).
"""

from __future__ import annotations

import atexit
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys

_INSTALLED = False
_ARMED = False

# Diagnostic counters for `check_disk_cache` outcomes. Without these the
# ambiguity (autotune lines unchanged — patch loaded but ineffective, or patch
# never loaded?) is unresolvable from logs. Counters
# are reported once at process exit via atexit when TRITON_AUTOTUNE_PATCH_DEBUG=1.
_DIAG_COUNTERS = {
    "no_key": 0,         # tuning_key empty → fall through to bench
    "unstable_prehook": 0,  # lambda/closure pre_hook → fall through
    "upstream_fallthrough": 0,  # no config has pre_hook → delegate to upstream
    "no_cache_file": 0,  # cache.get_file returned None → bench + save
    "full_hit": 0,       # every live config hit the cache → skip bench
    "partial_hit": 0,    # some hits, some misses → fall through (upstream-equiv)
    "load_error": 0,     # cached file existed but unreadable → bench + save
}

# Per-kernel-name dedup for the verbose cache_key breakdown logs. At 64 ranks
# × N kernels × multiple autotune cycles, naive logging would emit thousands of
# duplicate lines. We log each kernel's full breakdown the first time it's
# autotuned per process; subsequent invocations of the same kernel just bump
# the counters. Set on rank 0 is sufficient; non-zero ranks log only the
# kernels they actually hit (usually the same set).
_LOGGED_KEY_BREAKDOWNS: set = set()
_LOGGED_OUTCOMES: set = set()  # (kernel_name, outcome) — log first occurrence of each combo


def _debug() -> bool:
    return os.environ.get("TRITON_AUTOTUNE_PATCH_DEBUG", "0") == "1"


def _log(msg: str) -> None:
    r"""Emit a diagnostic line to stderr if TRITON_AUTOTUNE_PATCH_DEBUG=1.

    Prefix is fixed so logs grep cleanly: `grep '\[TRITON_PATCH\]' run.log`.
    """
    if _debug():
        sys.stderr.write(f"[TRITON_PATCH] {msg}\n")
        sys.stderr.flush()


def _report_at_exit() -> None:
    if not _debug():
        return
    total = sum(_DIAG_COUNTERS.values())
    _log(
        f"check_disk_cache summary: total={total} "
        + " ".join(f"{k}={v}" for k, v in _DIAG_COUNTERS.items())
    )


def _config_identity(cfg) -> dict:
    """JSON-friendly stable identity for a Triton Config.

    `pre_hook` is replaced by its `__qualname__` (or `"__unstable__"` for
    lambdas/closures, which signals the caller to skip this entry).

    `ir_override` is included as `None` when absent on the Config (older
    Triton versions did not expose it). Upstream Triton's save side
    serializes `config.__dict__` which includes `ir_override`, so a cached
    entry from upstream will have the key with value `None`. Including it
    in our identity ensures dict equality matches those entries.
    """
    d = {
        "kwargs": dict(cfg.kwargs) if hasattr(cfg, "kwargs") else {},
        "num_warps": getattr(cfg, "num_warps", None),
        "num_stages": getattr(cfg, "num_stages", None),
        "num_ctas": getattr(cfg, "num_ctas", None),
        "maxnreg": getattr(cfg, "maxnreg", None),
        "ir_override": getattr(cfg, "ir_override", None),
    }
    ph = getattr(cfg, "pre_hook", None)
    if ph is not None:
        qn = getattr(ph, "__qualname__", None)
        d["pre_hook"] = qn if qn else "__unstable__"
    else:
        d["pre_hook"] = None
    return d


def _has_unstable_prehook(cfg) -> bool:
    ph = getattr(cfg, "pre_hook", None)
    return ph is not None and not getattr(ph, "__qualname__", None)


def _apply_patch(Autotuner):
    """Replace `Autotuner.check_disk_cache` and `__init__` on the given class.

    Idempotent via module-level `_INSTALLED`. Called by both `install()` (when
    Triton is already imported) and the MetaPathFinder loader wrapper (right
    after the autotuner module body finishes executing).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    import builtins as _builtins

    # Save a reference to upstream's check_disk_cache so we can fall through
    # for kernels upstream handles correctly. Without this, we'd lose access
    # to the original method the moment we replace Autotuner.check_disk_cache.
    _orig_check_disk_cache = Autotuner.check_disk_cache

    def patched_check_disk_cache(self, tuning_key, configs, bench_fn):
        if not tuning_key:
            _DIAG_COUNTERS["no_key"] += 1
            bench_fn()
            return False

        # Path C: for kernels with no pre_hook on any config, upstream Triton
        # handles disk caching correctly — it falls through to the standard
        # disk-cache path and uses cached timings for whichever cached config
        # matches the live tuning_key. Upstream is more lenient than our
        # strict structural-identity matching: it does not require every live
        # config to appear in the cache; it just needs the cached "best"
        # config to still be benchable. Our patch was designed for the case
        # where upstream BAILS (`any(cfg.pre_hook ...)`), so call upstream
        # for the non-pre_hook case rather than re-implement it ourselves.
        # Empirically, the strict matching path
        # caused 12/16 mamba kernels to fall through to re-bench on partial
        # mismatches that upstream would have skipped. Falling through here
        # restores upstream's win for mamba kernels while keeping our patch
        # active for TE / hybrid_ep kernels that do have pre_hooks.
        if not any(getattr(c, "pre_hook", None) for c in configs):
            _DIAG_COUNTERS["upstream_fallthrough"] += 1
            return _orig_check_disk_cache(self, tuning_key, configs, bench_fn)

        # If any config has a lambda/closure pre_hook, we can't round-trip it.
        # Fall back to upstream behavior for safety.
        if any(_has_unstable_prehook(c) for c in configs):
            _DIAG_COUNTERS["unstable_prehook"] += 1
            bench_fn()
            return False

        import hashlib
        from triton._C.libtriton import get_cache_invalidating_env_vars
        from triton.compiler.compiler import make_backend, triton_key
        from triton.runtime.cache import get_cache_manager
        from triton.runtime.driver import driver
        from triton.runtime.jit import JITFunction

        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn

        env_vars = get_cache_invalidating_env_vars()
        cache_key_parts = [
            triton_key(),
            make_backend(driver.active.get_current_target()).hash(),
            fn.cache_key,
            str(sorted(env_vars.items())),
            str(tuning_key),
        ] + [str(c) for c in configs]
        cache_key = hashlib.sha256("-".join(cache_key_parts).encode("utf-8")).hexdigest()
        cache = get_cache_manager(cache_key)
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)

        # Diagnostic: dump the 6 cache_key components once per unique kernel
        # name when TRITON_AUTOTUNE_PATCH_DEBUG=1. Goal is to identify which
        # component differs between cache-write context and runtime context,
        # which causes a hash mismatch and re-autotune even when the cache
        # tarball contains a matching autotune.json file. Each component
        # gets a 16-char sha256 prefix so two runs' logs can be diffed
        # field-by-field; the full env_vars list is also emitted (separate
        # line) because that's the most volatile component empirically.
        if _debug() and fn.__name__ not in _LOGGED_KEY_BREAKDOWNS:
            _LOGGED_KEY_BREAKDOWNS.add(fn.__name__)
            _h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
            _cdir = getattr(cache, "cache_dir", None) or getattr(cache, "_cache_dir", "?")
            _log(
                f"cache_key for {fn.__name__[:60]}: "
                f"sha256={cache_key} "
                f"triton={_h(cache_key_parts[0])} "
                f"backend={_h(cache_key_parts[1])} "
                f"fn_src={_h(cache_key_parts[2])} "
                f"env={_h(cache_key_parts[3])} "
                f"tk={_h(cache_key_parts[4])} "
                f"cfgs={_h(''.join(cache_key_parts[5:]))} "
                f"cache_dir={_cdir} "
                f"file_name={file_name} "
                f"path_found={path is not None}"
            )
            _log(f"env_vars for {fn.__name__[:60]}: {cache_key_parts[3]}")

        if path:
            try:
                with open(path) as cached_file:
                    payload = json.load(cached_file)
                cached_entries = payload.get("configs_timings", [])
            except (OSError, ValueError):
                _DIAG_COUNTERS["load_error"] += 1
                cached_entries = []

            # Try to satisfy every live config from the cache.
            hits = {}
            for cfg in configs:
                ident = _config_identity(cfg)
                for entry_ident, timing in cached_entries:
                    if entry_ident == ident:
                        hits[cfg] = timing
                        break
            if len(hits) == len(configs):
                _DIAG_COUNTERS["full_hit"] += 1
                if _debug() and (fn.__name__, "full_hit") not in _LOGGED_OUTCOMES:
                    _LOGGED_OUTCOMES.add((fn.__name__, "full_hit"))
                    _log(f"full_hit: {fn.__name__[:60]} cache_key={cache_key[:16]}")
                self.configs_timings = hits
                self.cache[tuning_key] = _builtins.min(hits, key=hits.get)
                full_nargs = {**self.nargs, **self.cache[tuning_key].all_kwargs()}
                self.pre_hook(full_nargs, reset_only=True)
                return True
            # Partial hit — fall through to re-bench. Triton's bench_fn
            # captures `pruned_configs` by closure, so we can't easily skip
            # the cached ones. Upstream-equivalent.
            _DIAG_COUNTERS["partial_hit"] += 1
            if _debug() and (fn.__name__, "partial_hit") not in _LOGGED_OUTCOMES:
                _LOGGED_OUTCOMES.add((fn.__name__, "partial_hit"))
                # Log the first mismatched live config so we can see whether
                # kwargs or pre_hook qualname is the differing field.
                first_missing = None
                for cfg in configs:
                    if cfg not in hits:
                        first_missing = _config_identity(cfg)
                        break
                _log(
                    f"partial_hit: {fn.__name__[:60]} cache_key={cache_key[:16]} "
                    f"live_configs={len(configs)} hits={len(hits)} "
                    f"first_missing={first_missing}"
                )
        else:
            _DIAG_COUNTERS["no_cache_file"] += 1
            if _debug() and (fn.__name__, "no_cache_file") not in _LOGGED_OUTCOMES:
                _LOGGED_OUTCOMES.add((fn.__name__, "no_cache_file"))
                _cdir = getattr(cache, "cache_dir", None) or getattr(cache, "_cache_dir", "?")
                _log(
                    f"no_cache_file: {fn.__name__[:60]} cache_key={cache_key[:16]} "
                    f"looked_in={_cdir}"
                )

        bench_fn()
        serializable = []
        for cfg, t in self.configs_timings.items():
            ident = _config_identity(cfg)
            if ident.get("pre_hook") == "__unstable__":
                continue
            serializable.append((ident, t))
        try:
            cache.put(
                json.dumps({"key": tuning_key, "configs_timings": serializable}),
                file_name,
                binary=False,
            )
        except (OSError, ValueError):
            pass
        return False

    _orig_init = Autotuner.__init__

    def patched_init(self, *args, **kwargs):
        # Path E: only force cache_results=True for Autotuner instances that
        # actually need our shim — i.e. those where upstream Triton bails on
        # disk caching (any config has a pre_hook). For the common case (all
        # pre_hook=None configs, e.g. mamba kernels), keep upstream's default
        # cache_results=False — that avoids per-autotune-cycle disk I/O that
        # cost significant init wall time at scale (Path C).
        # Sniff configs before calling _orig_init (so we have them regardless
        # of where upstream stores them on `self`). Autotuner signature is
        # __init__(self, fn, arg_names, configs, key, ...) — configs is the
        # third positional arg.
        _configs = kwargs.get("configs")
        if _configs is None and len(args) >= 3:
            _configs = args[2]
        _orig_init(self, *args, **kwargs)
        if _configs and any(getattr(c, "pre_hook", None) for c in _configs):
            self.cache_results = True

    Autotuner.check_disk_cache = patched_check_disk_cache
    Autotuner.__init__ = patched_init

    # Retrofit existing Autotuner instances. Triton's `Autotuner.run`
    # gates the `check_disk_cache` call on `self.cache_results`, which is set
    # at __init__ time. Class-level __init__ patch only affects FUTURE
    # instances. When this shim runs, the worker rank has already inherited
    # Autotuner instances from the torch.distributed.run parent process (those
    # were created when mamba_ssm / torch / TE were imported and decorated
    # their @triton.autotune kernels) — those instances have
    # `cache_results=False` baked in. Without this retrofit, our
    # `patched_check_disk_cache` never fires (verified empirically:
    # total=0, full_hit=0, every counter zero).
    import gc
    _retrofitted = 0
    for _obj in gc.get_objects():
        try:
            if isinstance(_obj, Autotuner) and not getattr(_obj, "cache_results", True):
                _obj.cache_results = True
                _retrofitted += 1
        except (ReferenceError, AttributeError):
            pass
    _log(
        f"retrofitted cache_results=True on {_retrofitted} existing Autotuner "
        f"instances (gc.get_objects walk)"
    )

    _INSTALLED = True
    _log(f"patch applied to {Autotuner.__module__}.{Autotuner.__name__}")
    atexit.register(_report_at_exit)


def install():
    """In-place install if `triton.runtime.autotuner` is already importable.

    Idempotent. No-op if kill switch set or Triton not available.
    """
    if _INSTALLED:
        _log("install: already installed (no-op)")
        return
    if os.environ.get("TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE", "0") == "1":
        _log("install: skipped (TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE=1)")
        return
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        _log("install: skipped (triton.runtime.autotuner not importable)")
        return
    _log("install: triton.runtime.autotuner already loaded — patching directly")
    _apply_patch(Autotuner)


class _AutotunerFinder(importlib.abc.MetaPathFinder):
    """Intercept the first `import triton.runtime.autotuner` and patch after exec.

    The trainer pulls Triton in lazily via the kernel registration path. By the
    time the launcher's sitecustomize runs, Triton itself may not yet be
    importable cleanly (depends on torch init order on some images). Registering
    a MetaPathFinder lets us patch deterministically at the moment Triton's
    autotuner module finishes loading — independent of whether site-init or the
    trainer wins the race.
    """

    TARGET = "triton.runtime.autotuner"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except Exception:
                continue
            if spec is None or spec.loader is None:
                continue
            orig_loader = spec.loader

            class _WrapLoader(importlib.abc.Loader):
                def create_module(self_inner, spec):  # noqa: N805
                    cm = getattr(orig_loader, "create_module", None)
                    return cm(spec) if cm else None

                def exec_module(self_inner, module):  # noqa: N805
                    orig_loader.exec_module(module)
                    _log(
                        "_AutotunerFinder: intercepted triton.runtime.autotuner "
                        "exec — applying patch"
                    )
                    try:
                        _apply_patch(module.Autotuner)
                    except Exception as e:  # noqa: BLE001 — diagnostic-only
                        _log(f"_AutotunerFinder: _apply_patch raised: {e}")

            spec.loader = _WrapLoader()
            return spec
        return None


def arm():
    """Register `_AutotunerFinder` so the patch lands on first Triton import.

    Falls back to in-place `install()` if Triton's autotuner module is already
    in `sys.modules` (rare but possible).
    """
    global _ARMED
    if _ARMED:
        _log("arm: already armed (no-op)")
        return
    if os.environ.get("TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE", "0") == "1":
        _log("arm: skipped (TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE=1)")
        return
    if _AutotunerFinder.TARGET in sys.modules:
        _log("arm: triton.runtime.autotuner already loaded — installing directly")
        install()
    else:
        sys.meta_path.insert(0, _AutotunerFinder())
        _log("arm: registered _AutotunerFinder at head of sys.meta_path")
    _ARMED = True


if __name__ == "__main__":
    arm()
    install()
    print(
        "triton_autotune_disk_cache: armed="
        + str(_ARMED)
        + " installed="
        + str(_INSTALLED)
    )

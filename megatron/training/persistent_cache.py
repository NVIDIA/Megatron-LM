"""Persistent cache hooks for Megatron-LM.

Bootstrapping (seeding node-local storage from cache_read/<scope>.tar.zst,
exporting env vars) is done by tools/persistent_cache/bootstrap.sh BEFORE this Python process
starts. By the time Python runs, cache env vars have already been consumed by
torch._inductor / triton / the CUDA driver. This module is responsible only
for:

  - validating that the bootstrap actually ran (warn if not),
  - kicking the bash writeback from save_checkpoint(),
  - registering atexit for a final writeback.

See tools/persistent_cache/README.md for the architecture and bash side.
"""
from __future__ import annotations

import atexit
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# scope -> env var the scope's compiler/library reads.
# 'dataset_idx' is CLI-driven (--data-cache-path), so it is intentionally absent.
_SCOPE_ENV = {
    'triton':                 'TRITON_CACHE_DIR',
    'inductor':               'TORCHINDUCTOR_CACHE_DIR',
    'cuda_ptx':               'CUDA_CACHE_PATH',
    'hybrid_ep':              'HYBRID_EP_JIT_DIR',
    'cudnn_fe':               'CUDNN_FRONTEND_CACHE_DIR',
    'nccl_topo':              'NCCL_TOPO_DUMP_FILE',
    'megatron_fused_kernels': 'TORCH_EXTENSIONS_DIR',
}


class PersistentCacheController:
    """Thin coordinator for the bash cache pipeline.

    Owns three responsibilities:
      1. Validate at startup that the bash bootstrap left the env in a sane state.
      2. Kick the bash writeback from save_checkpoint, throttled.
      3. Register an atexit hook for final writeback on clean exit.
    """

    def __init__(self, args):
        self.read_dir = getattr(args, 'persistent_cache_read_dir', None)
        self.write_dir = getattr(args, 'persistent_cache_write_dir', None)
        self.scopes = list(getattr(args, 'persistent_cache_scopes', []) or [])
        self.writeback_every_n_saves = getattr(
            args, 'persistent_cache_writeback_every_n_saves', 8)
        self.skip_validation = getattr(
            args, 'persistent_cache_skip_validation', False)
        self.enabled = bool(self.read_dir or self.write_dir)
        self._save_count = 0
        self._writeback_in_flight: Optional[subprocess.Popen] = None
        self._writeback_script = os.environ.get(
            'MCORE_CACHE_WRITEBACK_SCRIPT',
            self._default_script_path('writeback.sh'),
        )

    @staticmethod
    def _default_script_path(name: str) -> str:
        # Resolve <repo>/tools/persistent_cache/<name> relative to this file.
        here = Path(__file__).resolve()
        return str(here.parents[2] / 'tools' / 'persistent_cache' / name)

    def validate(self) -> None:
        """Sanity-check that the bash bootstrap populated the env. Never raises.

        Local rank 0 only: the per-node bootstrap seed is identical for every rank
        on a node, so one report per node catches per-node seed failures without
        emitting the same lines from every rank at scale.
        """
        if not self.enabled or self.skip_validation:
            return
        if not _is_local_rank_zero():
            return
        for scope in self.scopes:
            env_var = _SCOPE_ENV.get(scope)
            if env_var is None:
                continue  # CLI-driven scope (dataset_idx); not our concern here.
            value = os.environ.get(env_var)
            if not value:
                logger.warning(
                    "[persistent_cache] scope %r requested but %s is unset; "
                    "did the launcher source tools/persistent_cache/bootstrap.sh?",
                    scope, env_var)
                continue
            path = Path(value)
            check_path = path.parent if env_var.endswith('_DUMP_FILE') else path
            if not check_path.exists():
                logger.warning(
                    "[persistent_cache] scope %r %s=%s does not exist",
                    scope, env_var, value)
                continue
            # If a read_dir was configured, the operator expected warm content.
            # An empty seeded dir means the bash extract silently failed.
            if self.read_dir and self._seed_looks_empty(env_var, path):
                logger.warning(
                    "[persistent_cache] scope %r %s=%s exists but is empty — "
                    "bootstrap seed may have failed (will compile cold)",
                    scope, env_var, value)
                continue
            logger.debug(
                "[persistent_cache] scope %r ready (%s=%s, size=%s)",
                scope, env_var, value, _du_sh(value))

    @staticmethod
    def _seed_looks_empty(env_var: str, path: Path) -> bool:
        if env_var.endswith('_DUMP_FILE'):
            return not (path.exists() and path.stat().st_size > 0)
        try:
            return path.is_dir() and not any(path.iterdir())
        except OSError:
            return False

    def maybe_kick_writeback(self, iteration: int) -> None:
        """Throttled writeback kick from save_checkpoint(). Local rank 0 only.

        Non-blocking: spawns the bash writeback in a new session and returns
        immediately. Save latency is unaffected.
        """
        if not self.enabled or not self.write_dir:
            return
        if not _is_local_rank_zero():
            return
        self._save_count += 1
        if self._save_count % max(1, self.writeback_every_n_saves) != 0:
            return
        if self._writeback_in_flight is not None and self._writeback_in_flight.poll() is None:
            logger.info(
                "[persistent_cache] previous writeback still running, skipping kick at iter %d",
                iteration)
            return
        try:
            self._writeback_in_flight = subprocess.Popen(
                ['bash', self._writeback_script],
                start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logger.info(
                "[persistent_cache] kicked writeback at iter %d (pid=%d)",
                iteration, self._writeback_in_flight.pid)
        except FileNotFoundError:
            logger.warning(
                "[persistent_cache] writeback script not found at %s",
                self._writeback_script)

    def register_atexit(self) -> None:
        if not self.enabled or not self.write_dir:
            return
        if not _is_local_rank_zero():
            return
        atexit.register(self._final_writeback)

    # Bounded so process exit isn't held hostage by a hung sidecar.
    _IN_FLIGHT_WAIT_SECONDS = 60
    _FINAL_WRITEBACK_TIMEOUT_SECONDS = 300

    def _final_writeback(self) -> None:
        # Wait for any in-flight writeback (bounded). If still hung, terminate it
        # so container teardown isn't blocked.
        if self._writeback_in_flight is not None:
            try:
                self._writeback_in_flight.wait(timeout=self._IN_FLIGHT_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "[persistent_cache] in-flight writeback hung at exit (>%ds); terminating",
                    self._IN_FLIGHT_WAIT_SECONDS)
                self._writeback_in_flight.terminate()
        try:
            subprocess.run(
                ['bash', self._writeback_script, '--final'],
                timeout=self._FINAL_WRITEBACK_TIMEOUT_SECONDS, check=False,
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning("[persistent_cache] final writeback failed: %s", e)


_singleton: Optional[PersistentCacheController] = None


def init(args) -> Optional[PersistentCacheController]:
    """Idempotent. Call once from training.py at startup."""
    global _singleton
    if _singleton is not None:
        return _singleton
    _singleton = PersistentCacheController(args)
    _singleton.validate()
    _singleton.register_atexit()
    return _singleton


def get() -> Optional[PersistentCacheController]:
    return _singleton


def _is_local_rank_zero() -> bool:
    local_rank = (os.environ.get('SLURM_LOCALID')
                  or os.environ.get('LOCAL_RANK')
                  or '0')
    try:
        return int(local_rank) == 0
    except ValueError:
        return False


def _du_sh(path: str) -> str:
    try:
        r = subprocess.run(
            ['du', '-sh', path], capture_output=True, text=True, timeout=5)
        return r.stdout.split()[0] if r.returncode == 0 else '?'
    except Exception:  # noqa: BLE001
        return '?'

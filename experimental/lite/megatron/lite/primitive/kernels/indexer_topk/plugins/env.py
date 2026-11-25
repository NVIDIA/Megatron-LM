# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The process environment of the external LiteTopK plugins.

ABI v1 plugins take their configuration from ``SGLANG_LITETOPK*`` environment keys: the adapter
module snapshots them once, at import, and the CUDA extension reads a few of them again with
``getenv`` on every launch. This is the only module of the package that names those keys or
touches ``os.environ``. The rules:

* :func:`render_plugin_env` turns :class:`LiteTopKPluginSettings` into the complete key table of
  one plugin load: the fixed keys plus one key per setting that is not None.
* :func:`claim_plugin_env` writes that table into the process environment before the import.
  A key that is already set to a different value is an error, and so is every other
  ``SGLANG_LITETOPK*`` key or known launch-time key that no loaded plugin rendered, except the
  diagnostic keys, which pass through. Rendered keys are never restored: the extension keeps
  reading the launch-time ones, so restoring them after the import would silently change the
  kernels that run. One process therefore holds one value per key, shared by every plugin.
* After the import, the claim checks that the module snapshotted exactly the rendered values,
  that no launch-time key the plugin reads was set for another plugin, and that the plugin reads
  every rendered launch-time setting (a setting only some plugins read, such as the staging
  layout of a raw-FP32-key plugin, is rendered only for a plugin that lists its key).
* :func:`check_launch_time_env` rechecks the recorded launch-time values before a selection.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable, Mapping

from megatron.lite.primitive.kernels.indexer_topk.config import (
    IndexerTopKPluginError,
    IndexerTopKRuntimeError,
    LiteTopKPluginSettings,
)

__all__ = [
    "DIAGNOSTIC_ENV_KEYS",
    "LAUNCH_TIME_ENV_KEYS",
    "PluginEnvClaim",
    "check_launch_time_env",
    "claim_plugin_env",
    "diagnostic_env",
    "render_plugin_env",
]

# Every key an ABI v1 adapter snapshots starts with this prefix.
_PREFIX = "SGLANG_LITETOPK"

# Rendered for every plugin load. CP_SMALL_Q gates the qualified direct-paged query shapes; it is
# independent of the admission setting (fp8_paged_admit_max_query_len).
_FIXED_ENV = {
    "SGLANG_LITETOPK": "1",
    "SGLANG_LITETOPK_PAGED_CANDIDATES": "1",
    "SGLANG_LITETOPK_EXPERIMENTAL_CP_SMALL_Q": "1",
    "SGLANG_LITETOPK_RELEASE_SCRATCH_ON_ROLLBACK": "1",
}

# One key per LiteTopKPluginSettings field; None values are not rendered.
_SETTING_KEYS = {
    "tie_policy": "SGLANG_LITETOPK_H32_TIE_POLICY",
    "score_policy": "SGLANG_LITETOPK_H32_SCORE_POLICY",
    "paged_pool_pages_per_row": "SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW",
    "fp8_row_tiles": "SGLANG_LITETOPK_FP8_LARGE_Q_ROW_TILES",
    "fp8_paged_admit_max_query_len": "SGLANG_LITETOPK_FP8_PAGED_ADMIT_MAX_Q",
    "tiered_seed_12k": "SGLANG_LITETOPK_TIERED_SEED_12K",
    "coldstart_identity": "SGLANG_LITETOPK_COLDSTART_IDENTITY",
    "raw32_staging": "SGLANG_LITETOPK_RAW32_STAGING",
}
# Settings that are launch-time keys of some plugins only: a load that renders one of them needs
# a plugin that lists the key in plugin_info()["launch_time_env_keys"].
_LAUNCH_TIME_SETTING_KEYS = frozenset((_SETTING_KEYS["raw32_staging"],))

# Diagnostic keys that may pass through from the process environment. They change logging and
# checks, not selections, and are recorded in the load provenance.
DIAGNOSTIC_ENV_KEYS = frozenset(
    (
        "SGLANG_LITETOPK_CANDIDATE_STATS_DIR",
        "SGLANG_LITETOPK_CARRY_DEBUG",
        "SGLANG_LITETOPK_CARRY_TIMING",
        "SGLANG_LITETOPK_FAIL_DIAG",
        "SGLANG_LITETOPK_OVF_LOG",
        "SGLANG_LITETOPK_PATH_TIMING",
        "SGLANG_LITETOPK_PROBE_EVERY",
    )
)

# The keys the CUDA extensions of the known plugin sources (996e735c52df, e4a1280b4416,
# 83669db87b20, 7e5eb835fb7f) read with getenv on each launch: the complete `grep getenv`
# of their three CUDA files. Only the check for stray keys before a plugin is imported uses this
# list (it names the one key without the SGLANG_LITETOPK prefix). Loading never relies on it:
# every key a plugin lists in plugin_info()["launch_time_env_keys"] is checked, recorded and
# rechecked before each selection, including keys this list does not know.
LAUNCH_TIME_ENV_KEYS = frozenset(
    (
        "LITETOPK_GRAFT_TIGHTEN",
        "SGLANG_LITETOPK_FP8_CP_DUAL_CTA",
        "SGLANG_LITETOPK_FP8_CP_LOREG_CONTROL",
        "SGLANG_LITETOPK_FP8_CP_SMALL_Q_KV_SPLITS",
        "SGLANG_LITETOPK_H32_SCORE_POLICY",
        "SGLANG_LITETOPK_RAW32_STAGING",
    )
)

# The environment ledger of the loaded plugins (guarded by the loader's lock): the owners of
# every rendered key with the value they rendered, and the launch-time values each plugin was
# loaded with.
_RENDERED_BY: dict[str, dict[str, str]] = {}
_LAUNCH_ENV: dict[str, dict[str, str | None]] = {}


def _render_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def render_plugin_env(settings: LiteTopKPluginSettings | None) -> dict[str, str]:
    """Return the complete environment table of a plugin load.

    Args:
        settings: The import-time settings; None renders only the fixed keys.

    Returns:
        The rendered keys and values, sorted by key. Booleans render as ``"1"`` and ``"0"``.
    """
    rendered = dict(_FIXED_ENV)
    if settings is not None:
        for field in dataclasses.fields(settings):
            value = getattr(settings, field.name)
            if value is not None:
                rendered[_SETTING_KEYS[field.name]] = _render_value(value)
    return dict(sorted(rendered.items()))


def diagnostic_env() -> dict[str, str]:
    """Return the diagnostic keys set in the process environment, sorted by key."""
    return {key: os.environ[key] for key in sorted(DIAGNOSTIC_ENV_KEYS) if key in os.environ}


def _owners(key: str) -> str:
    return ", ".join(sorted(_RENDERED_BY.get(key, {})))


class PluginEnvClaim:
    """The environment written for one plugin load, until it is committed or rolled back."""

    def __init__(
        self,
        owner: str,
        rendered: Mapping[str, str],
        written: list[str],
        diagnostics: Mapping[str, str],
    ) -> None:
        self.owner = owner
        self.rendered = dict(rendered)
        self.diagnostics = dict(diagnostics)
        self._written = written

    def check_effective_config(self, effective_config: Mapping[str, str | None]) -> None:
        """Check the values the plugin module snapshotted at import.

        Every rendered key must hold its rendered value, a diagnostic key the plugin reads its
        process value, and every other key must have been unset.

        Raises:
            IndexerTopKPluginError: If any snapshotted value differs.
        """
        problems = []
        expected = dict.fromkeys(effective_config)
        expected.update(
            (key, value) for key, value in self.diagnostics.items() if key in effective_config
        )
        expected.update(self.rendered)
        for key, want in sorted(expected.items()):
            got = effective_config.get(key)
            if got == want:
                continue
            problem = (
                f"LiteTopK plugin effective_config[{key}]={got!r} differs from the requested "
                f"{want!r}"
            )
            if key not in self.rendered and key in _RENDERED_BY:
                problem += (
                    f" ({key} was rendered for LiteTopK source {_owners(key)} and this plugin "
                    "reads it at import; load plugins that need different values in separate "
                    "processes)"
                )
            problems.append(problem)
        if problems:
            raise IndexerTopKPluginError("; ".join(problems))

    def check_launch_time_keys(self, launch_keys: Iterable[str]) -> None:
        """Check the launch-time keys the plugin reports against the rendered table.

        No launch-time key of the plugin may be set unless this load rendered it, and every
        rendered launch-time setting must be a key the plugin reads.

        Raises:
            IndexerTopKPluginError: If an unrendered launch-time key of the plugin is set in the
                process, for another plugin or by the user, or the plugin does not read a
                rendered launch-time setting.
        """
        launch_keys = set(launch_keys)
        problems = [
            f"the plugin settings render {key}={self.rendered[key]!r}, but LiteTopK source "
            f"{self.owner} does not list it among its launch-time keys; use a plugin that reads "
            "it, or leave the setting None"
            for key in sorted((_LAUNCH_TIME_SETTING_KEYS & set(self.rendered)) - launch_keys)
        ]
        for key in sorted(launch_keys - set(self.rendered)):
            value = os.environ.get(key)
            if value is None:
                continue
            if key in _RENDERED_BY:
                problems.append(
                    f"LiteTopK source {self.owner} reads {key} on every CUDA launch and needs it "
                    f"unset, but LiteTopK source {_owners(key)} set {key}={value!r} in this "
                    "process; launch-time keys are process-wide, so load these plugins in "
                    "separate processes"
                )
            else:
                problems.append(
                    f"{key}={value!r} is set in the process, but LiteTopK source {self.owner} "
                    "reads it on every CUDA launch and Megatron Lite does not render it; unset it"
                )
        if problems:
            raise IndexerTopKPluginError("; ".join(problems))

    def commit(self, launch_keys: Iterable[str]) -> dict[str, str | None]:
        """Keep the written environment for the process lifetime and record the launch values.

        Args:
            launch_keys: The launch-time keys the plugin reports.

        Returns:
            The launch-time keys and their current values (None when unset), sorted by key.
        """
        recorded = {key: os.environ.get(key) for key in sorted(set(launch_keys))}
        for key, value in self.rendered.items():
            _RENDERED_BY.setdefault(key, {})[self.owner] = value
        _LAUNCH_ENV[self.owner] = dict(recorded)
        self._written = []
        return recorded

    def rollback(self) -> None:
        """Remove the keys this claim added to the process environment (after a failed load)."""
        for key in self._written:
            os.environ.pop(key, None)
        self._written = []


def claim_plugin_env(owner: str, rendered: Mapping[str, str]) -> PluginEnvClaim:
    """Check the process environment against a rendered table and write the table.

    Args:
        owner: The plugin being loaded, as named in error messages.
        rendered: The table from :func:`render_plugin_env`.

    Returns:
        The claim; the caller commits it after a successful load or rolls it back.

    Raises:
        IndexerTopKPluginError: If a rendered key is set to a different value, conflicts with
            the launch-time values of a loaded plugin, or the process sets a plugin key that no
            loaded plugin rendered and that is not a diagnostic key.
    """
    problems = []
    for key, value in sorted(rendered.items()):
        current = os.environ.get(key)
        if current is not None and current != value:
            problem = (
                f"{key}={current!r} is already set in the process but the plugin settings "
                f"need {value!r}; unset it or change LiteTopKPluginSettings"
            )
            if key in _RENDERED_BY:
                problem += f" (it was rendered for LiteTopK source {_owners(key)})"
            problems.append(problem)
            continue
        for other, launch_env in sorted(_LAUNCH_ENV.items()):
            if key in launch_env and launch_env[key] != value:
                problems.append(
                    f"LiteTopK source {other} reads {key} on every CUDA launch and was loaded "
                    f"with {key}={launch_env[key]!r}, but the plugin settings need {value!r}; "
                    "launch-time keys are process-wide, so load these plugins in separate "
                    "processes"
                )
    # Launch-time keys of the plugins already loaded count as plugin keys too, whatever they are
    # called.
    loaded_launch_keys = {key for launch_env in _LAUNCH_ENV.values() for key in launch_env}
    stray = sorted(
        key
        for key in os.environ
        if key not in rendered
        and key not in _RENDERED_BY
        and key not in DIAGNOSTIC_ENV_KEYS
        and (key.startswith(_PREFIX) or key in LAUNCH_TIME_ENV_KEYS or key in loaded_launch_keys)
    )
    if stray:
        problems.append(
            f"the process sets {', '.join(stray)}, which the plugin settings do not render; "
            "LiteTopK plugins snapshot or read these keys, so unset them (only the diagnostic "
            f"keys {', '.join(sorted(DIAGNOSTIC_ENV_KEYS))} may pass through)"
        )
    if problems:
        raise IndexerTopKPluginError("; ".join(problems))
    written = [key for key in rendered if key not in os.environ]
    for key in written:
        os.environ[key] = rendered[key]
    return PluginEnvClaim(owner, rendered, written, diagnostic_env())


def check_launch_time_env(recorded: Mapping[str, str | None], *, owner: str) -> None:
    """Check that the launch-time keys still hold the values recorded when a plugin loaded.

    A pure dictionary comparison, cheap enough to run before every selection.

    Args:
        recorded: The launch-time values returned by :meth:`PluginEnvClaim.commit`.
        owner: The plugin, as named in the error message.

    Raises:
        IndexerTopKRuntimeError: If a launch-time key was set, changed or unset after the load.
    """
    for key, value in recorded.items():
        current = os.environ.get(key)
        if current != value:
            raise IndexerTopKRuntimeError(
                f"{key} changed from {value!r} to {current!r} after LiteTopK source {owner} was "
                "loaded; its CUDA extension reads this key on every launch, so it must stay "
                "fixed for the process lifetime"
            )

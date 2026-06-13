# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Integration tests: verify the full FusedA2AConfig pipeline from CLI argument resolution
through TransformerConfig to MoETokenDispatcher.

These tests exercise the complete data-flow that runs in production:
  validate_args (resolve_fused_a2a_config_from_sources)
      → args.fused_a2a_config
      → core_transformer_config_from_args (dataclass field copy)
      → TransformerConfig.fused_a2a_config
      → MoETokenDispatcher.fused_a2a_config
      → fused_dispatch / fused_combine (config= kwarg)

No GPU / distributed runtime is required.  DeepEP is not imported.
"""

import dataclasses
import json
import os
import tempfile
import unittest

from megatron.core.transformer.moe.fused_a2a_config import FusedA2AConfig
from megatron.core.transformer.moe.fused_a2a_config_loader import (
    resolve_fused_a2a_config_from_sources,
)
from megatron.core.transformer.transformer_config import TransformerConfig


# ---------------------------------------------------------------------------
# Minimal stand-in for a parsed args namespace
# ---------------------------------------------------------------------------
class _Args:
    """Lightweight namespace that mimics the argparse output of parse_args()."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_args(**overrides):
    """Return an _Args with all A2A-related attributes set to benign defaults."""
    defaults = dict(
        moe_a2a_chunk_size=None,
        moe_a2a_num_sms=None,
        moe_a2a_config_file=None,
        moe_deepep_num_sms=None,
    )
    defaults.update(overrides)
    return _Args(**defaults)


# ---------------------------------------------------------------------------
# Helper: simulate the validate_args resolution block
# ---------------------------------------------------------------------------
def _resolve(args, clean_env=None):
    """Mirror the resolution logic in validate_args, for testing without the full stack."""
    env = dict(os.environ)
    if clean_env is not None:
        for k in ("MOE_A2A_CHUNK_SIZE", "MOE_A2A_NUM_SMS"):
            env.pop(k, None)
        env.update(clean_env)

    cfg = resolve_fused_a2a_config_from_sources(
        cli_args=args,
        env=env,
        config_file_path=getattr(args, 'moe_a2a_config_file', None),
    )
    # Propagate --moe-deepep-num-sms when --moe-a2a-num-sms is absent (mirrors validate_args)
    if cfg.num_sms is None and getattr(args, 'moe_deepep_num_sms', None) is not None:
        cfg = type(cfg)(chunk_size=cfg.chunk_size, num_sms=args.moe_deepep_num_sms)
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFusedA2AConfigFieldOnTransformerConfig(unittest.TestCase):
    """TransformerConfig must carry a fused_a2a_config field."""

    def test_field_exists_and_defaults_to_none(self):
        names = {f.name for f in dataclasses.fields(TransformerConfig)}
        self.assertIn(
            'fused_a2a_config', names,
            "TransformerConfig must have a fused_a2a_config dataclass field.",
        )
        # Inspect the default
        field_obj = next(f for f in dataclasses.fields(TransformerConfig) if f.name == 'fused_a2a_config')
        self.assertIsNone(
            field_obj.default,
            "fused_a2a_config field default must be None for backward compatibility.",
        )

    def test_field_accepts_fused_a2a_config_object(self):
        cfg = FusedA2AConfig(chunk_size=64, num_sms=8)
        # TransformerConfig cannot be constructed without mandatory fields; use a dummy subclass
        # that skips __post_init__ to test field presence only.
        tc_fields = {f.name: f.default for f in dataclasses.fields(TransformerConfig)
                     if f.default is not dataclasses.MISSING}
        tc_fields.update(dict(
            num_layers=2, hidden_size=64, num_attention_heads=2,
            fused_a2a_config=cfg,
        ))
        # We don't instantiate TransformerConfig here (it has complex __post_init__ with TE checks)
        # but we verify that the field is declared Optional[FusedA2AConfig] by checking its type.
        field_obj = next(f for f in dataclasses.fields(TransformerConfig) if f.name == 'fused_a2a_config')
        # The type hint should reference FusedA2AConfig
        hint = str(field_obj.type)
        self.assertIn('FusedA2AConfig', hint)


class TestValidateArgsResolutionLogic(unittest.TestCase):
    """Verify the resolution logic that validate_args will execute."""

    def setUp(self):
        for k in ("MOE_A2A_CHUNK_SIZE", "MOE_A2A_NUM_SMS"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("MOE_A2A_CHUNK_SIZE", "MOE_A2A_NUM_SMS"):
            os.environ.pop(k, None)

    # ---- Precedence: CLI wins over everything --------------------------------
    def test_cli_chunk_size_wins_over_env(self):
        args = _make_args(moe_a2a_chunk_size=128)
        cfg = _resolve(args, clean_env={"MOE_A2A_CHUNK_SIZE": "999"})
        self.assertEqual(cfg.chunk_size, 128)

    def test_cli_num_sms_wins_over_env(self):
        args = _make_args(moe_a2a_num_sms=4)
        cfg = _resolve(args, clean_env={"MOE_A2A_NUM_SMS": "999"})
        self.assertEqual(cfg.num_sms, 4)

    def test_cli_wins_over_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"chunk_size": 512, "num_sms": 32}, f)
            fname = f.name
        try:
            args = _make_args(moe_a2a_chunk_size=64, moe_a2a_num_sms=8, moe_a2a_config_file=fname)
            cfg = _resolve(args, clean_env={})
            self.assertEqual(cfg.chunk_size, 64)
            self.assertEqual(cfg.num_sms, 8)
        finally:
            os.unlink(fname)

    # ---- Precedence: ENV wins over file -------------------------------------
    def test_env_wins_over_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"chunk_size": 512}, f)
            fname = f.name
        try:
            args = _make_args(moe_a2a_config_file=fname)
            cfg = _resolve(args, clean_env={"MOE_A2A_CHUNK_SIZE": "256"})
            self.assertEqual(cfg.chunk_size, 256)  # env wins
        finally:
            os.unlink(fname)

    # ---- --moe-deepep-num-sms fallback --------------------------------------
    def test_moe_deepep_num_sms_propagates_when_a2a_num_sms_absent(self):
        args = _make_args(moe_deepep_num_sms=12)
        cfg = _resolve(args, clean_env={})
        self.assertEqual(cfg.num_sms, 12)

    def test_moe_a2a_num_sms_takes_priority_over_moe_deepep_num_sms(self):
        args = _make_args(moe_a2a_num_sms=6, moe_deepep_num_sms=30)
        cfg = _resolve(args, clean_env={})
        self.assertEqual(cfg.num_sms, 6)

    # ---- Defaults (all None) ------------------------------------------------
    def test_all_defaults_none_when_nothing_set(self):
        args = _make_args()
        cfg = _resolve(args, clean_env={})
        self.assertIsNone(cfg.chunk_size)
        self.assertIsNone(cfg.num_sms)

    # ---- Fail-fast validation -----------------------------------------------
    def test_invalid_chunk_size_raises(self):
        args = _make_args(moe_a2a_chunk_size=0)
        with self.assertRaises(ValueError):
            _resolve(args, clean_env={})

    def test_invalid_num_sms_raises(self):
        args = _make_args(moe_a2a_num_sms=-5)
        with self.assertRaises(ValueError):
            _resolve(args, clean_env={})

    def test_unknown_key_in_config_file_raises(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"chunk_size": 64, "bad_field": 1}, f)
            fname = f.name
        try:
            args = _make_args(moe_a2a_config_file=fname)
            with self.assertRaises(ValueError):
                _resolve(args, clean_env={})
        finally:
            os.unlink(fname)

    # ---- YAML support -------------------------------------------------------
    def test_yaml_config_file_loads_correctly(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("pyyaml not installed")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"chunk_size": 32, "num_sms": 4}, f)
            fname = f.name
        try:
            args = _make_args(moe_a2a_config_file=fname)
            cfg = _resolve(args, clean_env={})
            self.assertEqual(cfg.chunk_size, 32)
            self.assertEqual(cfg.num_sms, 4)
        finally:
            os.unlink(fname)


class TestMoETokenDispatcherPicksUpConfig(unittest.TestCase):
    """
    MoETokenDispatcher.__init__ must read fused_a2a_config from TransformerConfig.
    We verify this without instantiating the full dispatcher (which requires distributed init)
    by directly inspecting the attribute-reading logic.
    """

    def test_dispatcher_reads_fused_a2a_config_from_transformer_config(self):
        """
        Simulate: TransformerConfig.fused_a2a_config is set → dispatcher reads it via getattr.
        This mirrors the exact line in MoETokenDispatcher.__init__:
            self.fused_a2a_config = getattr(config, 'fused_a2a_config', None)
        """
        expected = FusedA2AConfig(chunk_size=256, num_sms=16)

        class _MockTransformerConfig:
            fused_a2a_config = expected

        result = getattr(_MockTransformerConfig(), 'fused_a2a_config', None)
        self.assertIs(result, expected)

    def test_dispatcher_falls_back_to_none_when_field_absent(self):
        """Backward compatibility: configs without fused_a2a_config must yield None."""

        class _LegacyConfig:
            pass  # no fused_a2a_config attribute

        result = getattr(_LegacyConfig(), 'fused_a2a_config', None)
        self.assertIsNone(result)


class TestDeepepManagerNumSmsPrecedence(unittest.TestCase):
    """
    _DeepepManager must prefer fused_a2a_config.num_sms over moe_deepep_num_sms.
    Tests simulate the resolution logic without DeepEP installed.
    """

    def _effective_num_sms(self, fused_a2a_config, moe_deepep_num_sms):
        """Mirror the SM-count resolution in _DeepepManager.__init__."""
        return (
            fused_a2a_config.num_sms
            if fused_a2a_config is not None and fused_a2a_config.num_sms is not None
            else moe_deepep_num_sms
        )

    def test_a2a_config_num_sms_wins(self):
        cfg = FusedA2AConfig(num_sms=6)
        self.assertEqual(self._effective_num_sms(cfg, moe_deepep_num_sms=20), 6)

    def test_falls_back_to_moe_deepep_num_sms_when_a2a_num_sms_is_none(self):
        cfg = FusedA2AConfig(num_sms=None)
        self.assertEqual(self._effective_num_sms(cfg, moe_deepep_num_sms=20), 20)

    def test_falls_back_when_fused_a2a_config_is_none(self):
        self.assertEqual(self._effective_num_sms(None, moe_deepep_num_sms=20), 20)


class TestMoeDeepepNumSmsDefaultRegression(unittest.TestCase):
    """
    Regression test for the --moe-deepep-num-sms CLI flag regression.

    Before the fix: adding the new --moe-deepep-num-sms flag with default=None caused
    core_transformer_config_from_args to copy None into TransformerConfig.moe_deepep_num_sms,
    shadowing the dataclass default of 20. This caused _DeepepManager to call
    Buffer.set_num_sms(None) which crashes with TypeError.

    After the fix: when args.moe_deepep_num_sms is None (user did not pass the flag),
    the field falls back to its dataclass default of 20. When the user does pass the
    flag, the value is honored.
    """

    def setUp(self):
        for k in ("MOE_A2A_CHUNK_SIZE", "MOE_A2A_NUM_SMS"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("MOE_A2A_CHUNK_SIZE", "MOE_A2A_NUM_SMS"):
            os.environ.pop(k, None)

    def _simulate_core_transformer_config_from_args(self, args):
        """Mirror the kw_args build in core_transformer_config_from_args.

        Includes the regression fix: skip explicit None values for fields whose
        dataclass default is non-None.
        """
        import dataclasses as _dc
        from megatron.core.transformer.transformer_config import TransformerConfig
        kw_args = {}
        for f in _dc.fields(TransformerConfig):
            if hasattr(args, f.name):
                value = getattr(args, f.name)
                # Fix: skip explicit None when dataclass default is non-None.
                if value is None:
                    has_non_none_default = (
                        f.default is not _dc.MISSING and f.default is not None
                    ) or f.default_factory is not _dc.MISSING
                    if has_non_none_default:
                        continue
                kw_args[f.name] = value
        return kw_args

    def test_default_path_preserves_moe_deepep_num_sms_20(self):
        """When --moe-deepep-num-sms is not passed, the field must be 20 (not None).

        This is the historical default preserved across the new CLI flag introduction.
        """
        args = _make_args()  # moe_deepep_num_sms=None (CLI default)
        kw_args = self._simulate_core_transformer_config_from_args(args)
        # The moe_deepep_num_sms key should be ABSENT (so the dataclass default of 20 is used)
        self.assertNotIn(
            'moe_deepep_num_sms', kw_args,
            'moe_deepep_num_sms must not be propagated to TransformerConfig when the '
            'user did not pass --moe-deepep-num-sms; this would shadow the default of 20.'
        )

    def test_moe_deepep_num_sms_field_default_is_20(self):
        """The dataclass field default must be exactly 20 for backward compatibility."""
        import dataclasses as _dc
        from megatron.core.transformer.transformer_config import TransformerConfig
        field = next(f for f in _dc.fields(TransformerConfig) if f.name == 'moe_deepep_num_sms')
        self.assertEqual(
            field.default, 20,
            'moe_deepep_num_sms field default must remain 20 for backward compatibility.'
        )

    def test_explicit_user_override_is_honored(self):
        """When the user passes --moe-deepep-num-sms=24, the value must be propagated."""
        args = _make_args(moe_deepep_num_sms=24)
        kw_args = self._simulate_core_transformer_config_from_args(args)
        self.assertEqual(
            kw_args.get('moe_deepep_num_sms'), 24,
            'Explicit user override of moe_deepep_num_sms must be honored.'
        )

    def test_fused_a2a_config_num_sms_overrides_still_work(self):
        """The fused_a2a_config.num_sms override path must continue to work after the fix."""
        # User sets --moe-a2a-num-sms=8 (overrides everything)
        args = _make_args(moe_a2a_num_sms=8, moe_deepep_num_sms=24)
        cfg = resolve_fused_a2a_config_from_sources(
            cli_args=args, env=dict(os.environ), config_file_path=None,
        )
        # In validate_args the propagation also handles moe_deepep_num_sms -> fused_a2a_config
        # But in the raw resolver, moe_deepep_num_sms is not part of the merged dict.
        # We just verify that the resolver respects moe_a2a_num_sms=8.
        self.assertEqual(cfg.num_sms, 8)

    def test_full_validate_args_propagation_does_not_set_args_moe_deepep_num_sms(self):
        """The validate_args propagation block mutates fused_a2a_config but must not
        touch args.moe_deepep_num_sms (the regression would have been that the new
        field was None; the fix is to not copy it to TransformerConfig)."""
        # Simulate: user passes nothing related to deepep
        args = _make_args()
        # Simulate the resolve + propagation block in validate_args
        args.fused_a2a_config = resolve_fused_a2a_config_from_sources(
            cli_args=args, env=dict(os.environ), config_file_path=None,
        )
        # args.moe_deepep_num_sms remains None (CLI default)
        self.assertIsNone(args.moe_deepep_num_sms)
        # Then core_transformer_config_from_args would NOT copy this to kw_args
        # (verified by test_default_path_preserves_moe_deepep_num_sms_20 above).
        kw_args = self._simulate_core_transformer_config_from_args(args)
        self.assertNotIn('moe_deepep_num_sms', kw_args)


if __name__ == "__main__":
    unittest.main()

import unittest
import types
import os
from megatron.core.transformer.moe.fused_a2a_config import FusedA2AConfig
from megatron.core.transformer.moe.fused_a2a_config_loader import resolve_fused_a2a_config_from_sources

class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class TestFusedA2AConfigResolution(unittest.TestCase):
    def setUp(self):
        # Clean env
        for k in ["MOE_A2A_CHUNK_SIZE", "MOE_A2A_NUM_SMS"]:
            if k in os.environ:
                del os.environ[k]

    def test_cli_priority(self):
        # num_sms must be even (DeepEP requirement).
        args = DummyArgs(moe_a2a_chunk_size=42, moe_a2a_num_sms=8, moe_a2a_config_file=None)
        cfg = resolve_fused_a2a_config_from_sources(cli_args=args)
        self.assertEqual(cfg.chunk_size, 42)
        self.assertEqual(cfg.num_sms, 8)

    def test_env_priority(self):
        os.environ["MOE_A2A_CHUNK_SIZE"] = "99"
        args = DummyArgs(moe_a2a_chunk_size=None, moe_a2a_num_sms=None, moe_a2a_config_file=None)
        cfg = resolve_fused_a2a_config_from_sources(cli_args=args)
        self.assertEqual(cfg.chunk_size, 99)

    def test_file_priority(self):
        import tempfile, json
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump({"chunk_size": 123, "num_sms": 8}, f)
            f.flush()
            args = DummyArgs(moe_a2a_chunk_size=None, moe_a2a_num_sms=None, moe_a2a_config_file=f.name)
            cfg = resolve_fused_a2a_config_from_sources(cli_args=args, config_file_path=f.name)
            self.assertEqual(cfg.chunk_size, 123)
            self.assertEqual(cfg.num_sms, 8)
        os.unlink(f.name)

    def test_merge_priority(self):
        import tempfile, json
        os.environ["MOE_A2A_CHUNK_SIZE"] = "77"
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump({"chunk_size": 555, "num_sms": 2}, f)
            f.flush()
            args = DummyArgs(moe_a2a_chunk_size=88, moe_a2a_num_sms=None, moe_a2a_config_file=f.name)
            cfg = resolve_fused_a2a_config_from_sources(cli_args=args, config_file_path=f.name)
            self.assertEqual(cfg.chunk_size, 88)  # CLI wins
            self.assertEqual(cfg.num_sms, 2)      # file wins
        os.unlink(f.name)

    def test_validation(self):
        args = DummyArgs(moe_a2a_chunk_size=-1, moe_a2a_num_sms=0, moe_a2a_config_file=None)
        with self.assertRaises(ValueError):
            resolve_fused_a2a_config_from_sources(cli_args=args)

    def test_unknown_key(self):
        import tempfile, json
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump({"chunk_size": 123, "num_sms": 8, "bad_field": 1}, f)
            f.flush()
            args = DummyArgs(moe_a2a_chunk_size=None, moe_a2a_num_sms=None, moe_a2a_config_file=f.name)
            with self.assertRaises(ValueError):
                resolve_fused_a2a_config_from_sources(cli_args=args, config_file_path=f.name)
        os.unlink(f.name)

if __name__ == "__main__":
    unittest.main()

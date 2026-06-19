import unittest
from megatron.core.transformer.moe.fused_a2a_config import FusedA2AConfig

class TestFusedA2AConfigValidation(unittest.TestCase):
    def test_valid(self):
        cfg = FusedA2AConfig(chunk_size=32, num_sms=8)
        cfg.validate()  # Should not raise

    def test_valid_none(self):
        # All fields None is a valid configuration (use hardware defaults).
        cfg = FusedA2AConfig()
        cfg.validate()

    def test_valid_even_num_sms(self):
        for n in (2, 4, 6, 8, 16, 20, 24, 64):
            cfg = FusedA2AConfig(chunk_size=32, num_sms=n)
            cfg.validate()  # Should not raise

    def test_invalid_chunk_size(self):
        cfg = FusedA2AConfig(chunk_size=0, num_sms=8)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_num_sms(self):
        cfg = FusedA2AConfig(chunk_size=32, num_sms=-1)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_odd_num_sms_rejected(self):
        # DeepEP requires num_sms to be even. An odd value would crash
        # deep inside the kernel with an opaque assertion; the validator
        # must fail fast.
        for n in (1, 3, 7, 21, 33):
            cfg = FusedA2AConfig(chunk_size=32, num_sms=n)
            with self.assertRaises(ValueError) as ctx:
                cfg.validate()
            self.assertIn("even", str(ctx.exception))
            self.assertIn(str(n), str(ctx.exception))

    def test_odd_num_sms_alone_rejected(self):
        # Even chunk_size doesn't excuse an odd num_sms.
        cfg = FusedA2AConfig(chunk_size=64, num_sms=11)
        with self.assertRaises(ValueError):
            cfg.validate()

if __name__ == "__main__":
    unittest.main()

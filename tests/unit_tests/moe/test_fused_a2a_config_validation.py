import unittest
from megatron.core.transformer.moe.fused_a2a_config import FusedA2AConfig

class TestFusedA2AConfigValidation(unittest.TestCase):
    def test_valid(self):
        cfg = FusedA2AConfig(chunk_size=32, num_sms=8)
        cfg.validate()  # Should not raise

    def test_invalid_chunk_size(self):
        cfg = FusedA2AConfig(chunk_size=0, num_sms=8)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_num_sms(self):
        cfg = FusedA2AConfig(chunk_size=32, num_sms=-1)
        with self.assertRaises(ValueError):
            cfg.validate()

if __name__ == "__main__":
    unittest.main()

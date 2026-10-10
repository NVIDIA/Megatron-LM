# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest

from tools.check_golden_values import compares_deterministically


class TestGoldenQuotedEnvKeys(unittest.TestCase):
    def test_quoted_yaml_keys_use_the_same_comparison_mode(self):
        keys = ['NON_DETERMINSTIC_RESULTS', 'NVTE_ALLOW_NONDETERMINISTIC_ALGO', 'SKIP_PYTEST']
        for key in keys:
            for quote in ["'", '"']:
                with self.subTest(key=key, quote=quote):
                    text = f'ENV_VARS:\n  {quote}{key}{quote}: 1\n'
                    self.assertFalse(compares_deterministically(text))


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import tempfile
import unittest
from pathlib import Path

import check_process_group_usage as checker


class TestExpressionScopes(unittest.TestCase):
    def test_local_expression_names_do_not_resolve_to_imports(self):
        for expression in [
            'lambda parallel_state: parallel_state.get_tensor_model_parallel_group()',
            '[parallel_state.get_tensor_model_parallel_group() for parallel_state in values]',
            '{parallel_state.get_tensor_model_parallel_group() for parallel_state in values}',
            '(parallel_state.get_tensor_model_parallel_group() for parallel_state in values)',
            '{key: parallel_state.get_tensor_model_parallel_group() for parallel_state in values for key in keys}',
        ]:
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / 'sample.py'
                p.write_text(
                    'from megatron.core import parallel_state\nvalues = []\nkeys = []\nresult = '
                    + expression
                    + '\n'
                )
                self.assertEqual(checker._violations_in(p), [])

    def test_outer_import_remains_detected_in_first_iterable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'sample.py'
            p.write_text(
                'from megatron.core import parallel_state\nresult = [parallel_state for parallel_state in parallel_state.get_tensor_model_parallel_group()]\n'
            )
            self.assertEqual(len(checker._violations_in(p)), 1)


if __name__ == '__main__':
    unittest.main()

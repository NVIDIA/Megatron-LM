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

    def test_immediate_comprehension_does_not_see_later_import(self):
        for scope in ('', 'def model():\n'):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'sample.py'
                prefix = '    ' if scope else ''
                path.write_text(
                    scope
                    + prefix
                    + 'result = [parallel_state.get_tensor_model_parallel_group() for value in values]\n'
                    + prefix
                    + 'from megatron.core import parallel_state\n'
                )
                self.assertEqual(checker._violations_in(path), [])

    def test_attribute_and_subscript_targets_do_not_bind_loaded_names(self):
        for target in ('parallel_state.item', 'parallel_state.items[0]'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'sample.py'
                path.write_text(
                    'from megatron.core import parallel_state\nresult = [parallel_state.get_tensor_model_parallel_group() for '
                    + target
                    + ' in values]\n'
                )
                self.assertEqual(len(checker._violations_in(path)), 1)

    def test_deferred_expression_can_see_later_import(self):
        for expression in (
            'lambda: parallel_state.get_tensor_model_parallel_group()',
            '(parallel_state.get_tensor_model_parallel_group() for value in values)',
        ):
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'sample.py'
                path.write_text(
                    'result = ' + expression + '\nfrom megatron.core import parallel_state\n'
                )
                self.assertEqual(len(checker._violations_in(path)), 1)

    def test_outer_import_remains_detected_in_first_iterable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'sample.py'
            p.write_text(
                'from megatron.core import parallel_state\nresult = [parallel_state for parallel_state in parallel_state.get_tensor_model_parallel_group()]\n'
            )
            self.assertEqual(len(checker._violations_in(p)), 1)


if __name__ == '__main__':
    unittest.main()

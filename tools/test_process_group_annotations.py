# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import tempfile
import unittest
from pathlib import Path

import check_process_group_usage as checker


class TestAnnotations(unittest.TestCase):
    def test_calls_in_function_annotations_are_detected(self):
        for signature in [
            'def model(value: parallel_state.get_tensor_model_parallel_group()): pass',
            'def model() -> parallel_state.get_tensor_model_parallel_group(): pass',
            'async def model(*values: parallel_state.get_tensor_model_parallel_group()): pass',
            'def model(**values: parallel_state.get_tensor_model_parallel_group()): pass',
        ]:
            with self.subTest(signature=signature), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'sample.py'
                path.write_text('from megatron.core import parallel_state\n' + signature + '\n')
                self.assertEqual(len(checker._violations_in(path)), 1)


if __name__ == '__main__':
    unittest.main()

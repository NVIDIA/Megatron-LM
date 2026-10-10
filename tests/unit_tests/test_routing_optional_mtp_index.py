# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools.moe_routing import analyze_routing_concentration


class TestRoutingOptionalMtpIndex(unittest.TestCase):
    def test_sorts_mtp_records_with_and_without_optional_head_indices(self):
        records = [
            {
                'rank': 0,
                'step': 1,
                'block': 'mtp',
                'layer': 1,
                'num_tokens': 1,
                'top_indices': [[0]],
                'topk': 1,
            },
            {
                'rank': 0,
                'step': 1,
                'block': 'mtp',
                'mtp_idx': 0,
                'layer': 1,
                'num_tokens': 1,
                'top_indices': [[1]],
                'topk': 1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'router_trace_rank0.jsonl'
            trace.write_text(''.join(json.dumps(record) + '\n' for record in records))
            output = io.StringIO()
            with patch('sys.argv', ['routing', directory]), redirect_stdout(output):
                analyze_routing_concentration.main()
            self.assertIn('Layers found: 2', output.getvalue())
            self.assertIn('mNone:1', output.getvalue())
            self.assertIn('m0:1', output.getvalue())


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools.moe_routing import analyze_routing_concentration


class TestRoutingDecodeTokenCount(unittest.TestCase):
    def test_later_small_mtp_record_does_not_reclassify_prefill_as_decode(self):
        records = [
            {
                'rank': 0,
                'step': 1,
                'block': 'decoder',
                'layer': 1,
                'num_tokens': 100,
                'top_indices': [[0]] * 100,
                'topk': 1,
            },
            {
                'rank': 0,
                'step': 1,
                'block': 'mtp',
                'mtp_idx': 0,
                'layer': 1,
                'num_tokens': 1,
                'top_indices': [[0]],
                'topk': 1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'router_trace_rank0.jsonl'
            trace.write_text(''.join(json.dumps(record) + '\n' for record in records))
            output = io.StringIO()
            with (
                patch('sys.argv', ['routing', directory, '--decode-only']),
                redirect_stdout(output),
            ):
                analyze_routing_concentration.main()
            self.assertIn('No data found', output.getvalue())


if __name__ == '__main__':
    unittest.main()

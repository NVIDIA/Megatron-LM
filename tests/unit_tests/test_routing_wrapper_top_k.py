# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.moe_routing import analyze_routing


class TestRoutingWrapperTopK(unittest.TestCase):
    def test_explicit_top_k_reaches_the_concentration_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [
                {'rank': 0, 'step': 1, 'layer': layer, 'num_tokens': 1, 'top_indices': [[0, 1]]}
                for layer in [1, 2]
            ]
            trace = Path(directory) / 'router_trace_rank0.jsonl'
            trace.write_text(''.join(json.dumps(record) + '\n' for record in records))
            outputs = []

            def run_analysis(script, arguments, label):
                if script != 'analyze_routing_concentration.py':
                    return
                result = subprocess.run(
                    [sys.executable, str(Path(analyze_routing.SCRIPT_DIR) / script), *arguments],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                outputs.append(result.stdout)

            with (
                patch.object(analyze_routing, 'run', side_effect=run_analysis),
                patch('sys.argv', ['routing', directory, '--top-k', '2']),
            ):
                analyze_routing.main()
            self.assertEqual(len(outputs), 1)
            self.assertIn('Top-K (router): 2', outputs[0])


if __name__ == '__main__':
    unittest.main()

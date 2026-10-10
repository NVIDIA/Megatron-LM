# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.moe_routing import analyze_routing_concentration


class TestUniformCoverage(unittest.TestCase):
    def test_coverage_baseline_saturates_at_all_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {
                    "rank": 0,
                    "step": 1,
                    "layer": layer,
                    "num_tokens": 4,
                    "topk": 1,
                    "top_indices": [[0], [1], [2], [3]],
                }
                for layer in (1, 2)
            ]
            Path(tmp, "router_trace_rank0.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records)
            )
            output = io.StringIO()
            with (
                mock.patch(
                    "sys.argv", ["concentration", tmp, "--num-experts", "4", "--n-values", "8"]
                ),
                contextlib.redirect_stdout(output),
            ):
                analyze_routing_concentration.main()
            self.assertIn("uniform baseline: 1.000, ratio: 1.00", output.getvalue())


if __name__ == "__main__":
    unittest.main()

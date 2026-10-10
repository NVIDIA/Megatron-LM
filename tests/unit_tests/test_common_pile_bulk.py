# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.common_pile_dataset.create_common_pile_ci_dataset import download_common_pile_bulk


class TestCommonPileBulk(unittest.TestCase):
    def test_scans_past_filtered_samples_to_fill_document_quota(self):
        samples = [{'text': 'short'}] * 4 + [{'text': 'long document ' * 20}] * 3
        datasets = types.ModuleType('datasets')
        datasets.load_dataset = lambda *args, **kwargs: samples
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict('sys.modules', {'datasets': datasets}),
        ):
            output = Path(directory) / 'sample.jsonl'
            count = download_common_pile_bulk(str(output), 3, 'fixture')
            self.assertEqual(count, 3)
            self.assertEqual(len(output.read_text().splitlines()), 3)


if __name__ == '__main__':
    unittest.main()

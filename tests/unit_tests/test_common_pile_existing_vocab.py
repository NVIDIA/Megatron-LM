# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import tempfile
import unittest
from pathlib import Path

from tools.common_pile_dataset.create_common_pile_ci_dataset import copy_vocab_files


class TestCommonPileExistingVocab(unittest.TestCase):
    def test_resumes_with_existing_vocab_when_source_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            vocab_paths = [
                'bpe/vocab.json',
                'bpe/merges.txt',
                'vocab.txt',
                'bert-large-cased-vocab.txt',
            ]
            for name in vocab_paths:
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('existing vocabulary')
            copy_vocab_files(str(output), str(Path(directory) / 'unavailable-source'))
            for name in vocab_paths:
                self.assertEqual((output / name).read_text(), 'existing vocabulary')


if __name__ == '__main__':
    unittest.main()

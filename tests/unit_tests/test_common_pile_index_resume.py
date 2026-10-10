# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.common_pile_dataset import create_common_pile_ci_dataset as common_pile


class TestCommonPileIndexResume(unittest.TestCase):
    def test_rebuilds_each_binary_with_a_missing_index(self):
        prefixes = [
            'my-gpt3_00_text_document',
            'my-bert_00_text_sentence',
            'my-t5_00_text_document',
        ]
        for incomplete in prefixes:
            with self.subTest(incomplete=incomplete), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                for name in [
                    'bpe/vocab.json',
                    'bpe/merges.txt',
                    'vocab.txt',
                    'bert-large-cased-vocab.txt',
                ]:
                    path = output / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('vocabulary')
                for prefix in prefixes:
                    (output / (prefix + '.bin')).write_bytes(b'data')
                    if prefix != incomplete:
                        (output / (prefix + '.idx')).write_bytes(b'index')
                raw = output / 'input.jsonl'
                raw.write_text('{"text":"document"}\n')
                (output / 'input_ss.jsonl').write_text('{"text":["document"]}\n')

                def preprocess(**kwargs):
                    suffix = '_text_sentence' if kwargs.get('split_sentences') else '_text_document'
                    Path(kwargs['output_prefix'] + suffix + '.idx').write_bytes(b'index')

                with (
                    patch.object(common_pile, 'run_preprocess', side_effect=preprocess) as run,
                    patch(
                        'sys.argv',
                        [
                            'common-pile',
                            '--output-dir',
                            directory,
                            '--existing-jsonl',
                            str(raw),
                            '--download-vocab',
                        ],
                    ),
                ):
                    common_pile.main()
                self.assertEqual(run.call_count, 1)
                self.assertTrue((output / (incomplete + '.idx')).is_file())


if __name__ == '__main__':
    unittest.main()

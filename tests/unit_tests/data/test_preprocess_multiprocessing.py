# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Run partitioned preprocessing with real nested worker pools, without downloads."""

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

from megatron.core.datasets.indexed_dataset import IndexedDataset


@pytest.mark.parametrize('start_method', multiprocessing.get_all_start_methods())
@pytest.mark.parametrize('keep_sequential_samples', [False, True])
def test_partitioned_preprocessing(tmp_path, start_method, keep_sequential_samples):
    root = Path(__file__).resolve().parents[3]
    runner = tmp_path / 'preprocess.py'
    runner.write_text(
        'import multiprocessing\n'
        'import sys\n'
        'from tools.preprocess_data import main\n'
        'if __name__ == "__main__":\n'
        '    multiprocessing.set_start_method(sys.argv.pop(1))\n'
        '    main()\n'
    )
    documents = [[1, 2], [3], [4, 5, 6], [7, 8], [9], [10, 11]]
    input_path = tmp_path / 'input.jsonl'
    input_path.write_text(
        ''.join(json.dumps({'text': ' '.join(map(str, doc))}) + '\n' for doc in documents)
    )
    prefix = tmp_path / 'tokens'
    command = [
        sys.executable,
        str(runner),
        start_method,
        '--input',
        str(input_path),
        '--output-prefix',
        str(prefix),
        '--tokenizer-type',
        'NullTokenizer',
        '--vocab-size',
        '32',
        '--append-eod',
        '--workers',
        '4',
        '--partitions',
        '2',
    ]
    if keep_sequential_samples:
        command.append('--keep-sequential-samples')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(root), env.get('PYTHONPATH')]))
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ordered = documents if keep_sequential_samples else documents[::2] + documents[1::2]
    expected = [doc + [31] for doc in ordered]
    dataset = IndexedDataset(f'{prefix}_text_document')
    assert [dataset[index].tolist() for index in range(len(dataset))] == expected
    assert dataset.sequence_lengths.tolist() == [len(doc) for doc in expected]
    assert dataset.document_indices.tolist() == list(range(len(documents) + 1))

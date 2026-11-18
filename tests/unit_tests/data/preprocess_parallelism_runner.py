# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fresh-process runner for tokenizer initialization tests, including spawned workers."""

import importlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

PARALLELISM = "TOKENIZERS_PARALLELISM"
_INITIAL_VALUE = os.environ.get(PARALLELISM)

preprocess_data = importlib.import_module("tools.preprocess_data")

_VALUE_AFTER_IMPORT = os.environ.get(PARALLELISM)
_BUILD_TOKENIZER = preprocess_data.build_tokenizer


def _record_tokenizer(args):
    # Observe before delegating to the real builder. Assertions belong in the
    # controller: raising in a Pool initializer can cause endless respawning.
    path = Path(args.observation_dir) / f"{os.getpid()}.json"
    path.write_text(json.dumps({"parallelism": os.environ.get(PARALLELISM)}))
    return _BUILD_TOKENIZER(args)


# Install this observer in each fresh interpreter, not just in the parent.
preprocess_data.build_tokenizer = _record_tokenizer


def _read_parallelism():
    return os.environ.get(PARALLELISM)


def _check_observations(directory: Path, expected: str, count: int) -> None:
    observations = list(directory.glob("*.json"))
    assert len(observations) == count
    for path in observations:
        assert json.loads(path.read_text())["parallelism"] == expected, path.read_text()


def main(start_method: str, directory: Path, expected: str, tokenizer_args: list[str]) -> None:
    """Exercise an independent worker and a partition with a nested worker pool."""
    assert _VALUE_AFTER_IMPORT == _INITIAL_VALUE, "Import changed tokenizer parallelism"
    mp.set_start_method(start_method)
    # Prime the forkserver while the setting is still unset (or user-supplied).
    # This also proves the selected context starts workers with the initial value.
    with mp.Pool(1) as pool:
        assert pool.apply_async(_read_parallelism).get(timeout=60) == _INITIAL_VALUE

    worker_dir = directory / "worker"
    worker_dir.mkdir()
    input_path = directory / "input.jsonl"
    prefix = str(directory / "tokens")
    sys.argv = [
        "preprocess_data.py",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "32",
        "--input",
        str(input_path),
        "--output-prefix",
        prefix,
        "--workers",
        "1",
        *tokenizer_args,
    ]
    args = preprocess_data.get_args()
    assert os.environ.get(PARALLELISM) == _INITIAL_VALUE
    args.observation_dir = str(worker_dir)
    document = json.dumps({"text": "1 2"})
    encoder = preprocess_data.Encoder(args)
    # Exercise this path independently so a partition's default cannot mask
    # removal of the worker-local default under fork or spawn.
    with mp.Pool(1, initializer=encoder.initializer) as pool:
        ids, lengths, _ = pool.apply_async(encoder.encode, (document,)).get(timeout=60)
    assert ids == {"text": [1, 2]}
    assert lengths == {"text": [2]}
    _check_observations(worker_dir, expected, 1)
    assert os.environ.get(PARALLELISM) == _INITIAL_VALUE

    partition_dir = directory / "partition"
    partition_dir.mkdir()
    args.observation_dir = str(partition_dir)
    input_path.write_text(document + "\n")
    partition = preprocess_data.Partition(args, workers=1)
    process = mp.Process(target=partition.process_json_file, args=((str(input_path), prefix),))
    process.start()
    try:
        process.join(timeout=60)
        assert not process.is_alive(), "Partition did not finish"
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    _check_observations(partition_dir, expected, 2)
    assert (partition_dir / f"{process.pid}.json").exists()
    assert os.environ.get(PARALLELISM) == _INITIAL_VALUE

    dataset = preprocess_data.indexed_dataset.IndexedDataset(f"{prefix}_text_document")
    assert len(dataset) == 1
    assert dataset[0].tolist() == [1, 2]
    assert dataset.sequence_lengths.tolist() == [2]
    assert dataset.document_indices.tolist() == [0, 1]


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4:])

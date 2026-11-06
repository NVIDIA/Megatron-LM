# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib

import torch


def test_legacy_mimo_samples_keep_metadata_aliases_and_model_payload(monkeypatch):
    monkeypatch.syspath_prepend("examples/multimodal")
    helpers = importlib.import_module("examples.multimodal.dataset_helpers")
    # Import the current MIMO AVLM consumer too: it imports the legacy helper
    # module used by existing MIMO training configurations.
    importlib.import_module("examples.mimo.data.energon_avlm_task_encoder")

    metadata = {"source": "legacy", "split": "train"}
    image = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    tokens = torch.tensor([11, 12, 13])
    labels = torch.tensor([-100, 12, 13])
    constructors = [
        (
            helpers.ImageTaskSample,
            {
                "imgs": [image],
                "num_tiles": [1],
                "tokens": tokens,
                "total_len": 3,
                "total_len_padded": 3,
                "labels": labels,
            },
        ),
        (
            helpers.ImageTaskSamplePacked,
            {
                "tokens": tokens,
                "labels": labels,
                "imgs": [image],
                "num_tiles": [1],
                "max_length": 3,
                "cu_lengths": [0, 3],
                "cu_lengths_padded": [0, 3],
            },
        ),
        (
            helpers.ImageTaskBatchPacked,
            {
                "tokens": tokens.unsqueeze(0),
                "labels": labels.unsqueeze(0),
                "imgs": image.unsqueeze(0),
                "num_tiles": [[1]],
                "max_lengths": [3],
                "cu_lengths": [[0, 3]],
                "cu_lengths_padded": [[0, 3]],
                "imgs_sizes": [(2, 2)],
                "vision_max_lengths": [1],
                "vision_cu_lengths": [[0, 1]],
                "has_pad_img": False,
            },
        ),
    ]

    samples = []
    for sample_type, payload in constructors:
        original_metadata = metadata.copy()
        sample = sample_type(
            __key__="sample-0",
            __restore_key__=("dataset", 0),
            __subflavor__=metadata,
            __subflavors__=metadata,
            **payload,
        )
        assert metadata == original_metadata
        assert sample.__subflavors__ == original_metadata
        if hasattr(sample, "__tags__"):
            assert sample.__tags__ == original_metadata
        samples.append(sample)

    batch = samples[-1]
    encoder = object.__new__(helpers.TaskEncoder)
    model_payload = encoder.encode_batch(batch)

    assert torch.equal(model_payload["tokens"], tokens.unsqueeze(0))
    assert torch.equal(model_payload["labels"], labels.unsqueeze(0))
    assert torch.equal(model_payload["imgs"], image.unsqueeze(0))
    assert model_payload["__subflavor__"] == metadata  # Existing legacy payload is preserved.
    assert not {"__subflavors__", "__tags__"}.intersection(model_payload)

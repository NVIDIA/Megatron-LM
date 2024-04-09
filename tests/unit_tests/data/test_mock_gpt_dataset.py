import random
import sys
from types import SimpleNamespace

import numpy

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset


def sample_N(dataset, N, randomize):
    if randomize:
        indices = [random.randint(0, len(dataset)-1) for _ in range(N)]
    else:
        indices = list(range(N))
    samples = [dataset[index]["tokens"].numpy() for index in indices]
    return samples


def test_builder_mock_data():
    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        mock=True,
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        tokenizer=SimpleNamespace(),
    )

    datasets = BlendedMegatronDatasetBuilder(MockGPTDataset, [100, 100, 100], lambda: True, config).build()

    N = 10

    # Check iso-index split variance
    subsets = [sample_N(dataset, N, randomize=False) for dataset in datasets]
    assert not numpy.allclose(subsets[0], subsets[1])
    assert not numpy.allclose(subsets[0], subsets[2])
    assert not numpy.allclose(subsets[1], subsets[2])

    # Check iso-split / iso-index identity
    subset_1A = sample_N(datasets[0], N, randomize=False)
    subset_1B = sample_N(datasets[0], N, randomize=False)
    assert numpy.allclose(subset_1A, subset_1B)

    # Check iso-split index variance
    subset_1A = sample_N(datasets[0], N, randomize=True)
    subset_1B = sample_N(datasets[0], N, randomize=True)
    assert not numpy.allclose(subset_1A, subset_1B)


if __name__ == "__main__":
    test_builder_mock_data()

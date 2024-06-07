##
# Compile megatron.core.datasets.helpers dependencies before BlendedDataset import
##

import torch

from megatron.core.datasets.utils import compile_helpers
from tests.unit_tests.test_utilities import Utils

if torch.distributed.is_available():
    Utils.initialize_distributed()
    if torch.distributed.get_rank() == 0:
        compile_helpers()
    torch.distributed.barrier()
else:
    compile_helpers()

##
# Done
##

import os
import tempfile
from collections import defaultdict
from typing import Dict, Optional

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split, get_blend_from_list

_NUM_DATASETS = 10

_SEQUENCE_LENGTH = 10

_SIZES = {}
for split in Split:
    _SIZES[split] = []
    for i in range(_NUM_DATASETS):
        _SIZES[split].append({Split.train: 1000, Split.valid: 100, Split.test: 10}[split] * (i + 1))

_MARGIN = 0.005


def do_setup(odir):
    paths = defaultdict(list)

    for i in range(_NUM_DATASETS):
        path_to_data = os.path.join(odir, str(i))
        os.mkdir(path_to_data)

        for split in _SIZES:
            data = numpy.zeros((_SIZES[split][i], _SEQUENCE_LENGTH))
            path = os.path.join(path_to_data, f"{split.name}.npy")
            numpy.save(path, data)
            paths[split].append(path)

    return paths


def test_builder():

    # Define the class here to avoid pytest warnings

    class TestDataset(MegatronDataset):
        def __init__(
            self,
            dataset: LowLevelDataset,
            dataset_path: Optional[str],
            indices: numpy.ndarray,
            num_samples: Optional[int],
            index_split: Split,
            config: BlendedMegatronDatasetConfig,
        ) -> None:
            super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

            if self.num_samples is None:
                self.num_samples = len(self.indices)

            self.sample_index = numpy.random.choice(self.indices, size=self.num_samples)

        @staticmethod
        def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
            return len(low_level_dataset)

        @staticmethod
        def build_low_level_dataset(
            dataset_path: str, config: BlendedMegatronDatasetConfig
        ) -> LowLevelDataset:
            return numpy.load(dataset_path)

        def __len__(self) -> int:
            return len(self.sample_index)

        def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
            return {"text": self.dataset[self.sample_index[idx]]}

    with tempfile.TemporaryDirectory() as temp_dir:

        paths = do_setup(temp_dir)

        blends = {
            split: get_blend_from_list(
                [
                    weight_or_path
                    for pair in zip(list(range(1, len(paths[split]) + 1, 1)), paths[split])
                    for weight_or_path in pair
                ]
            )
            for split in Split
        }

        blends_unweighted = {split: (blends[split][0], None) for split in blends}

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[blends[Split.train], None, None,],
        )
        try:
            datasets = BlendedMegatronDatasetBuilder(
                TestDataset, [None, None, None], lambda: True, config
            ).build()
            raise RuntimeError
        except AssertionError:
            pass

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[get_blend_from_list([paths[Split.train][0]]), None, None,],
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [1000, None, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == 1000 and isinstance(datasets[0], TestDataset)
        assert datasets[1] is None
        assert datasets[2] is None

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[
                blends_unweighted[Split.train],
                blends_unweighted[Split.valid],
                blends_unweighted[Split.test],
            ],
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [1000, 1000, 1000], lambda: True, config
        ).build()
        assert len(datasets[0]) == 1000
        assert len(datasets[1]) == 1000
        assert len(datasets[2]) == sum(_SIZES[Split.test])

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[
                blends_unweighted[Split.train],
                blends_unweighted[Split.valid],
                blends_unweighted[Split.test],
            ],
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [None, None, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == sum(_SIZES[Split.train])
        assert numpy.all(
            numpy.array(datasets[0].weights)
            == numpy.unique(datasets[0].dataset_index, return_counts=True)[1]
        )
        assert len(datasets[1]) == sum(_SIZES[Split.valid])
        assert numpy.all(
            numpy.array(datasets[1].weights)
            == numpy.unique(datasets[1].dataset_index, return_counts=True)[1]
        )
        assert len(datasets[2]) == sum(_SIZES[Split.test])
        assert numpy.all(
            numpy.array(datasets[2].weights)
            == numpy.unique(datasets[2].dataset_index, return_counts=True)[1]
        )

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[blends_unweighted[Split.train], None, None,],
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [1000, None, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == 1000
        for i in range(_NUM_DATASETS):
            assert len(datasets[0].datasets[i]) == _SIZES[Split.train][i]
        assert datasets[1] is None
        assert datasets[2] is None

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[blends[Split.train], None, None],
        )
        try:
            datasets = BlendedMegatronDatasetBuilder(
                TestDataset, [1000, None, None], lambda: True, config
            ).build()
            raise RuntimeError
        except IndexError:
            ##
            #
            # The size per dataset is a function of the requested size, the weight per dataset,
            # and a constant coefficient. The sizes, and consequently the total size to request,
            # are modified such that the weights may or may not be sufficiently representative.
            # To fix this, the weights should be reset according to the new sizes:
            #
            # S := size
            # W := weights
            #
            # S = func(S, W)
            #
            # W = S / sum(S)
            #
            ##
            config = BlendedMegatronDatasetConfig(
                random_seed=1234,
                sequence_length=_SEQUENCE_LENGTH,
                blend_per_split=[blends[Split.train], blends[Split.valid], blends[Split.test],],
            )
            datasets = BlendedMegatronDatasetBuilder(
                TestDataset, [100, 100, 100], lambda: True, config
            ).build()
            assert (
                len(datasets[0]) >= 100 and len(datasets[0]) <= 100 * (1 + _MARGIN) + _NUM_DATASETS
            )
            assert (
                len(datasets[1]) >= 100 and len(datasets[1]) <= 100 * (1 + _MARGIN) + _NUM_DATASETS
            )
            assert (
                len(datasets[2]) >= 100 and len(datasets[2]) <= 100 * (1 + _MARGIN) + _NUM_DATASETS
            )

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends_unweighted[Split.train],
            split="100,0,0",
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [None, None, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == sum(_SIZES[Split.train])
        assert numpy.all(
            numpy.array(datasets[0].weights)
            == numpy.unique(datasets[0].dataset_index, return_counts=True)[1]
        )
        assert datasets[1] is None
        assert datasets[2] is None

        if torch.distributed.is_initialized():
            config = BlendedMegatronDatasetConfig(
                random_seed=1234,
                sequence_length=_SEQUENCE_LENGTH,
                blend=blends_unweighted[Split.train],
                split="100,0,0",
            )
            datasets = BlendedMegatronDatasetBuilder(
                TestDataset,
                [None, None, None],
                lambda: torch.distributed.get_rank() % 2 == 0,
                config,
            ).build()
            if torch.distributed.get_rank() % 2 == 0:
                assert len(datasets[0]) == sum(_SIZES[Split.train])
                assert numpy.all(
                    numpy.array(datasets[0].weights)
                    == numpy.unique(datasets[0].dataset_index, return_counts=True)[1]
                )
            else:
                assert datasets[0] is None
            assert datasets[1] is None
            assert datasets[2] is None

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends_unweighted[Split.train],
            split="50,50,0",
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [1000, 0, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == 1000
        assert sum(map(len, datasets[0].datasets)) == sum(_SIZES[Split.train]) / 2
        assert sum(map(len, datasets[1].datasets)) == sum(_SIZES[Split.train]) / 2
        assert datasets[1] is not None and len(datasets[1]) == 0
        assert datasets[2] is None

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends_unweighted[Split.train],
            split="50,50,0",
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset,
            [int(sum(_SIZES[Split.train]) / 4), int(sum(_SIZES[Split.train])), None],
            lambda: True,
            config,
        ).build()
        assert len(datasets[0]) == sum(_SIZES[Split.train]) / 4
        assert len(datasets[1]) == sum(_SIZES[Split.train]) / 2
        assert datasets[2] is None

        # 990 9 1
        # 100000 1000 1
        # []
        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends[Split.train],
            split="990,9,1",
        )
        try:
            # All three of 100000, 1000, and 1 result in error, yet 10000 and 100 do not
            datasets = BlendedMegatronDatasetBuilder(
                TestDataset, [100000, 1000, 1], lambda: True, config
            ).build()
        except IndexError:
            ##
            #
            # The size per dataset is a function of the requested size, the weight per dataset,
            # and a constant coefficient. The sizes, and consequently the total size to request,
            # are modified such that the weights may or may not be sufficiently representative.
            # To fix this, the weights should be reset according to the new sizes:
            #
            # S := size
            # W := weights
            #
            # S = func(S, W)
            #
            # W = S / sum(S)
            #
            ##
            config = BlendedMegatronDatasetConfig(
                random_seed=1234,
                sequence_length=_SEQUENCE_LENGTH,
                blend=blends[Split.train],
                split="990,9,1",
            )
            datasets = BlendedMegatronDatasetBuilder(
                TestDataset, [10000, 100, 0], lambda: True, config
            ).build()
            assert (
                len(datasets[0]) >= 10000
                and len(datasets[0]) <= 10000 * (1 + _MARGIN) + _NUM_DATASETS
            )
            assert (
                len(datasets[1]) >= 100 and len(datasets[1]) <= 100 * (1 + _MARGIN) + _NUM_DATASETS
            )
            assert len(datasets[2]) == 0


if __name__ == "__main__":
    test_builder()

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
from typing import Dict

import numpy
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split


_NUM_DATASETS = 10

_SEQUENCE_LENGTH = 10

_SIZES_PER_SPLIT = {
    Split.train: 900,
    Split.valid: 90,
    Split.test: 10,
}


def do_setup(odir):
    paths = defaultdict(list)

    for i in range(_NUM_DATASETS):
        path_to_data = os.path.join(odir, str(i))
        os.mkdir(path_to_data)

        for split in _SIZES_PER_SPLIT:
            data = numpy.zeros((_SIZES_PER_SPLIT[split], _SEQUENCE_LENGTH))
            path = os.path.join(path_to_data, f"{split.name}.npy")
            numpy.save(path, data)
            paths[split].append(path)

    return paths


def test_builder():

    # Define the class here to avoid pytest warnings

    class TestDataset(MegatronDataset):
        def _finalize(self) -> None:
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
            split: [
                weight_or_path
                for pair in zip(list(range(len(paths[split]))), paths[split])
                for weight_or_path in pair
            ]
            for split in Split
        }

        # one dataset, one split AND multiple datasets, one split
        config = BlendedMegatronDatasetConfig(
            is_built_on_rank=lambda: True,
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[[paths[Split.train][0]], blends[Split.valid], None,],
        )
        datasets = BlendedMegatronDatasetBuilder(TestDataset, [100, 100, 100], config).build()
        assert len(datasets[0]) == 100 and isinstance(datasets[0], TestDataset)
        assert len(datasets[1]) >= 100 and isinstance(datasets[1], BlendedDataset)
        assert datasets[2] is None

        # blend_per_split, all splits
        config = BlendedMegatronDatasetConfig(
            is_built_on_rank=lambda: True,
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[blends[Split.train], blends[Split.valid], blends[Split.test],],
        )
        datasets = BlendedMegatronDatasetBuilder(TestDataset, [100, 100, 100], config).build()
        assert len(datasets[0]) >= 100
        assert len(datasets[1]) >= 100
        assert len(datasets[2]) >= 100

        # blend_per_split, one split
        config = BlendedMegatronDatasetConfig(
            is_built_on_rank=lambda: True,
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[blends[Split.train], None, None,],
        )
        datasets = BlendedMegatronDatasetBuilder(TestDataset, [100, 100, 100], config).build()
        assert len(datasets[0]) >= 100
        assert datasets[1] is None
        assert datasets[2] is None

        # blend, 90,9,1 split
        config = BlendedMegatronDatasetConfig(
            is_built_on_rank=lambda: True,
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends[Split.train],
            split="90,9,1",
        )
        datasets = BlendedMegatronDatasetBuilder(TestDataset, [100, 100, 100], config).build()
        assert len(datasets[0]) >= 100
        assert len(datasets[1]) >= 100
        assert len(datasets[2]) >= 100

        # blend, 100,0,0 split
        config = BlendedMegatronDatasetConfig(
            is_built_on_rank=lambda: True,
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends[Split.train],
            split="100,0,0",
        )
        datasets = BlendedMegatronDatasetBuilder(TestDataset, [100, 100, 100], config).build()
        assert len(datasets[0]) >= 100
        assert datasets[1] is None
        assert datasets[2] is None


if __name__ == "__main__":
    test_builder()

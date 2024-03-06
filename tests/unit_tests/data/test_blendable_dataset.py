
import unittest
from unittest import mock
import numpy as np
import nltk
import os
from glob import glob
import tempfile
import torch
from tests.unit_tests.data.test_preprocess_data import dummy_jsonl, build_datasets, gpt2_merge, gpt2_vocab
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
import io
from contextlib import redirect_stdout, redirect_stderr

def build_dummy_data(temp_dir):
    # set the default nltk data path
    os.environ["NLTK_DATA"] = os.path.join(temp_dir, "nltk_data")
    nltk.data.path.append(os.environ["NLTK_DATA"])

    path_to_raws = os.path.join(temp_dir, "sample_raws")
    path_to_data = os.path.join(temp_dir, "sample_data")
    os.mkdir(path_to_raws)
    os.mkdir(path_to_data)
    # supress output for these calls
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        # create the dummy resources
        dummy_jsonl(path_to_raws)

        gpt_args = [
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            gpt2_vocab(temp_dir),
            "--merge-file",
            gpt2_merge(temp_dir),
            "--append-eod",
            "--workers",
            "10",
            "--log-interval",
            "1",
        ]

        # build the datasets
        build_datasets(
            path_to_raws, path_to_data, extra_args=gpt_args,
        )
    return path_to_data

def init_dist():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    # Initialize the process group
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)

def destroy_dist():
    torch.distributed.destroy_process_group()

class TestGPTDataset(unittest.TestCase):
    
    def tearDown(self) -> None:
        destroy_dist()
        self.temp_dir.cleanup()
        return super().tearDown()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path_to_data = build_dummy_data(self.temp_dir.name)
        paths = [path[:-4] for path in glob(self.path_to_data+'/*.bin')]
        self.data_path = []
        for path in paths:
            self.data_path.append('1.0')
            self.data_path.append(path)
        init_dist()

    def test_build(self):
        config = GPTDatasetConfig(lambda: True, 1234, 5, blend=self.data_path, 
                                  split="1,0,0", path_to_cache=self.temp_dir.name, filter_consumed_samples=False)
        builder = BlendedMegatronDatasetBuilder(GPTDataset, [2000], dict(), config)
        train_dataset = builder.build()[0]
        dataset = train_dataset.datasets[0]
        num_samples = len(dataset.shuffle_index)
        n=5
        for i in range(n): 
            dataset[i]
        dataset.shuffle_index = dataset._filter_shuffle_index(dataset.shuffle_index, dataset.sample_index)
        self.assertEqual(len(dataset.shuffle_index), num_samples-n, 
                         f"Filtering resulted in {len(dataset.shuffle_index)} but should be {num_samples-n}!")

    def test_build_initial(self):
        init_dict = {(208, 25): 1, (214, 0): 1, (79, 17): 1, (166, 13): 1, (62, 6): 1}
        consumed_samples_dict = {os.path.basename(self.data_path[1]): init_dict.copy()}
        config = GPTDatasetConfig(lambda: True, 1234, 5, blend=self.data_path, 
                                  split="1,0,0", path_to_cache=self.temp_dir.name, filter_consumed_samples=False)
        builder = BlendedMegatronDatasetBuilder(GPTDataset, [2000], consumed_samples_dict, config)
        train_dataset = builder.build()[0]
        dataset = train_dataset.datasets[0]
        num_samples = len(dataset.shuffle_index)
        n=5
        for i in range(n): 
            dataset[i]
        # make sure the first 5 samples have been counted twice
        for k, v in dataset.consumed_samples_dict.items():
            self.assertEqual(v, init_dict[k]+1, 
                             f"Frequency of sample {k} != {init_dict[k]+1} in consumed samples dict")
        dataset.shuffle_index = dataset._filter_shuffle_index(dataset.shuffle_index, dataset.sample_index)
        self.assertEqual(len(dataset.shuffle_index), num_samples-n, 
                         f"Filtering resulted in {len(dataset.shuffle_index)} but should be {num_samples-n}!")

    def test_build_initial_filter(self):
        init_dict = {(208, 25): 1, (214, 0): 1, (79, 17): 1, (166, 13): 1, (62, 6): 1}
        consumed_samples_dict = {os.path.basename(self.data_path[1]): init_dict}
        config = GPTDatasetConfig(lambda: True, 1234, 5, blend=self.data_path, 
                                  split="1,0,0", path_to_cache=self.temp_dir.name, filter_consumed_samples=True)
        builder = BlendedMegatronDatasetBuilder(GPTDataset, [2000], consumed_samples_dict, config)
        train_dataset = builder.build()[0]
        dataset = train_dataset.datasets[0]
        num_samples = len(dataset.shuffle_index)
        n=5
        for i in range(n): 
            dataset[i]
        dataset.shuffle_index = dataset._filter_shuffle_index(dataset.shuffle_index, dataset.sample_index)
        self.assertEqual(len(dataset.shuffle_index), num_samples-n, 
                         f"Filtering resulted in {len(dataset.shuffle_index)} but should be {num_samples-n}!")


    def test_blendable(self):
        config = GPTDatasetConfig(lambda: True, 1234, 5, blend=self.data_path, 
                                  split="1,0,0", path_to_cache=self.temp_dir.name, filter_consumed_samples=False)
        builder = BlendedMegatronDatasetBuilder(GPTDataset, [2000], dict(), config)
        train_dataset = builder.build()[0]
        num_samples = sum(len(dataset.shuffle_index) for dataset in train_dataset.datasets)
        n = 5
        for i in range(n):
            train_dataset[i]
        
        for dataset in train_dataset.datasets:
            dataset.shuffle_index = dataset._filter_shuffle_index(dataset.shuffle_index, dataset.sample_index)
        num_samples_after_filter = sum(len(dataset.shuffle_index) for dataset in train_dataset.datasets)
        self.assertEqual(num_samples_after_filter, num_samples-n, 
                         f"Filtering resulted in {num_samples_after_filter} but should be {num_samples-n}!")


if __name__ == '__main__':
    unittest.main()

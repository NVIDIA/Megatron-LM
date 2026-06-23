# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from datasets import load_dataset
from pydantic import BaseModel


class HFDatasetAgent(BaseModel):
    """
    Agent base class for loading and accessing HuggingFace datasets.

    Uses either a local dataset file (Arrow format) or downloads from the HuggingFace Hub,
    depending on which initialization argument is provided.

    Attributes:
        dataset_file (str | None): Path to a local dataset file directory. If provided, loads dataset from here.
        hf_dataset_name (str | None): Name of the HuggingFace dataset to load from the hub, if no file provided.
    """

    dataset_file: str | None = None
    hf_dataset_name: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self.dataset = self.load_hf_dataset()

    def load_hf_dataset(self):
        """
        Loads the dataset from either a local file or the HuggingFace Hub.
        """
        if self.dataset_file:
            return load_dataset("arrow", data_dir=self.dataset_file, split=self.split)
        else:
            return load_dataset(self.hf_dataset_name, split=self.split)

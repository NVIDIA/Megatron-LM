# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from pathlib import Path

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.validation import (
    save_integrity_manifest,
    verify_integrity_manifest,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def init_model_parallel():
    """Init torch distributed."""
    Utils.initialize_model_parallel(1, 1)
    yield  # Run the actual test.
    Utils.destroy_model_parallel()


class TestIntegrity:
    def test_save_verify_integrity_manifest_with_ckpt(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1, 1)
        state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(1, 1), replica_id=Utils.rank
            ),
            'rank': 0,
        }
        load_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.empty(1, 1), replica_id=Utils.rank
            )
        }

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_save_integrity_manifest', sync=True
        ) as ckpt_dir:
            save(state_dict, ckpt_dir, verify_integrity=True)

            integrity_file = Path(ckpt_dir / "integrity.json")
            assert integrity_file.is_file(), "integrity.json doesn't exist."

            with open(integrity_file, "r") as f:
                data = json.load(f)
                files = list(data["files"].keys())

            assert "__0_0.distcp" in files
            assert len(data["files"]["common.pt"]) == 64

            loaded_state_dict = load(load_state_dict, ckpt_dir, verify_integrity=True)

        Utils.destroy_model_parallel()

    def test_save_verify_integrity_manifest_directly(self, init_model_parallel, tmp_path_dist_ckpt):
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_save_integrity_manifest_directly', sync=True
        ) as ckpt_dir:
            metadata_file = Path(ckpt_dir / "metadata.json")
            with open(metadata_file, "w") as f:
                data = {"test_metadata": 1}
                json.dump(data, f)

            if torch.distributed.get_rank() == 0:
                save_integrity_manifest(ckpt_dir)
            torch.distributed.barrier()
            integrity_file = Path(ckpt_dir / "integrity.json")
            assert integrity_file.is_file(), "integrity.json doesn't exist."

            with open(integrity_file, "r") as f:
                data = json.load(f)
                files = list(data["files"].keys())

            assert len(files) == 1
            assert len(data["files"]["metadata.json"]) == 64

            verify_integrity_manifest(ckpt_dir)

    def test_save_verify_integrity_manifest_error(self, init_model_parallel, tmp_path_dist_ckpt):
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_save_integrity_manifest_error', sync=True
        ) as ckpt_dir:
            metadata_file = Path(ckpt_dir / "metadata.json")

            with open(metadata_file, "w") as f:
                data = {"test_metadata": 1}
                json.dump(data, f)

            if torch.distributed.get_rank() == 0:
                save_integrity_manifest(ckpt_dir)
            torch.distributed.barrier()

            with open(metadata_file, "w") as f:
                data = {"test_metadata": 11}
                json.dump(data, f)

            # CheckpointingException, hash mismatch
            with pytest.raises(CheckpointingException):
                verify_integrity_manifest(ckpt_dir)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise integrity verification failures with real CPU process groups."""

import json
import logging
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.validation import (
    INTEGRITY_FNAME,
    save_integrity_manifest,
    verify_integrity_manifest,
)


def _verify_manifest_worker(rank, checkpoint_dir, rendezvous, expected_error):
    """Verify that an error is collective without poisoning subsequent collectives."""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        if expected_error is None:
            verify_integrity_manifest(checkpoint_dir)
        else:
            with pytest.raises(CheckpointingException, match=expected_error) as exc_info:
                verify_integrity_manifest(checkpoint_dir)

            errors = [None, None]
            dist.all_gather_object(errors, str(exc_info.value))
            assert errors[0] == errors[1]

        # A failed verification must leave the group usable for a subsequent attempt.
        dist.barrier()
        if rank == 0:
            manifest_path = Path(checkpoint_dir) / INTEGRITY_FNAME
            if manifest_path.is_dir():
                manifest_path.rmdir()
            (Path(checkpoint_dir) / "shard.distcp").write_bytes(b"checkpoint data")
            save_integrity_manifest(checkpoint_dir)
        dist.barrier()
        verify_integrity_manifest(checkpoint_dir)
        dist.barrier()
    except Exception:
        # Preserve the original rank-zero error if spawn reports a peer's teardown error first.
        logging.exception("Integrity verification failed on rank %s", rank)
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("truncated_json", "Expecting"),
        ("invalid_utf8", "decode"),
        ("unreadable_manifest", "Is a directory"),
        ("missing_manifest", "Integrity manifest not found"),
        ("missing_shard", "file missing or unreadable"),
        ("hash_mismatch", "hash mismatch"),
        ("unsupported_algorithm", "Unsupported hash algorithm"),
        (None, None),
    ],
)
def test_integrity_verification_across_ranks(tmp_path, failure, expected_error):
    """All ranks report the same failure and can subsequently verify a good checkpoint."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    shard_path = checkpoint_dir / "shard.distcp"
    shard_path.write_bytes(b"checkpoint data")
    save_integrity_manifest(str(checkpoint_dir))
    manifest_path = checkpoint_dir / INTEGRITY_FNAME

    if failure == "truncated_json":
        manifest_path.write_text('{"algorithm": "sha256", "files":', encoding="utf-8")
    elif failure == "invalid_utf8":
        manifest_path.write_bytes(b"\xff")
    elif failure == "unreadable_manifest":
        manifest_path.unlink()
        manifest_path.mkdir()
    elif failure == "missing_manifest":
        manifest_path.unlink()
    elif failure == "missing_shard":
        shard_path.unlink()
    elif failure == "hash_mismatch":
        shard_path.write_bytes(b"corrupted checkpoint data")
    elif failure == "unsupported_algorithm":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["algorithm"] = "unsupported"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    mp.spawn(
        _verify_manifest_worker,
        args=(str(checkpoint_dir), str(tmp_path / "rendezvous"), expected_error),
        nprocs=2,
        join=True,
    )

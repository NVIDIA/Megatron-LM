# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint integrity verification via SHA-256 hashing.

Public API
----------
save_integrity_manifest   -- called at save time (rank 0 only)
verify_integrity_manifest -- called at load time (distributed-aware)
INTEGRITY_FNAME           -- filename of the manifest ('integrity.json')
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict

from megatron.core.msc_utils import MultiStorageClientFeature

from .core import CheckpointingException

logger = logging.getLogger(__name__)

INTEGRITY_FNAME = 'integrity.json'
_HASH_ALGORITHM = 'sha256'
_READ_CHUNK_SIZE = 1 << 20  # 1 MiB


def _compute_file_hash(file_path: str) -> str:
    """Return the SHA-256 hex digest of *file_path*, read in streaming chunks.

    Args:
        file_path: absolute path to the file to hash.

    Returns:
        Lowercase hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        with msc.open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(_READ_CHUNK_SIZE), b''):
                h.update(chunk)
    else:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(_READ_CHUNK_SIZE), b''):
                h.update(chunk)
    return h.hexdigest()


def save_integrity_manifest(checkpoint_dir: str) -> None:
    """Hash every file in *checkpoint_dir* and write an integrity manifest.

    The manifest is stored as ``{checkpoint_dir}/integrity.json`` and lists
    each filename (relative to *checkpoint_dir*) together with its SHA-256
    digest.  The manifest file itself is excluded from the listing.

    This function must be called **after** all checkpoint files have been
    flushed to disk (including ``metadata.json``) and must be called from a
    **single process only** (typically rank 0).

    Args:
        checkpoint_dir: directory that contains the checkpoint files.
    """
    manifest: Dict[str, str] = {}

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        ckpt_path = msc.Path(checkpoint_dir)
        for entry in sorted(ckpt_path.iterdir()):
            if entry.name != INTEGRITY_FNAME:
                manifest[entry.name] = _compute_file_hash(str(entry))
    else:
        ckpt_path = Path(checkpoint_dir)
        for entry in sorted(ckpt_path.iterdir()):
            if entry.is_file() and entry.name != INTEGRITY_FNAME:
                manifest[entry.name] = _compute_file_hash(str(entry))

    integrity_path = os.path.join(checkpoint_dir, INTEGRITY_FNAME)
    payload = {'algorithm': _HASH_ALGORITHM, 'files': manifest}

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        with msc.open(integrity_path, 'w') as f:
            json.dump(payload, f, indent=2)
    else:
        with open(integrity_path, 'w') as f:
            json.dump(payload, f, indent=2)

    logger.debug(
        "Saved integrity manifest with %d file(s) to %s", len(manifest), integrity_path
    )


def _verify_integrity_manifest_impl(checkpoint_dir: str) -> None:
    """Single-process implementation of integrity verification.

    Reads ``integrity.json``, recomputes each file's hash, and raises
    :class:`~megatron.core.dist_checkpointing.core.CheckpointingException`
    on any mismatch or missing file.

    Args:
        checkpoint_dir: checkpoint directory to verify.

    Raises:
        CheckpointingException: if the manifest is absent, uses an unsupported
            algorithm, or any file's hash does not match.
    """
    integrity_path = os.path.join(checkpoint_dir, INTEGRITY_FNAME)

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        if not msc.os.path.exists(integrity_path):
            raise CheckpointingException(
                f'Integrity manifest not found at {integrity_path}. '
                'The checkpoint must be saved with integrity verification enabled '
                '(save_integrity=True) before it can be verified on load.'
            )
        with msc.open(integrity_path) as f:
            manifest_data = json.load(f)
    else:
        if not os.path.exists(integrity_path):
            raise CheckpointingException(
                f'Integrity manifest not found at {integrity_path}. '
                'The checkpoint must be saved with integrity verification enabled '
                '(save_integrity=True) before it can be verified on load.'
            )
        with open(integrity_path) as f:
            manifest_data = json.load(f)

    algorithm = manifest_data.get('algorithm', _HASH_ALGORITHM)
    if algorithm != _HASH_ALGORITHM:
        raise CheckpointingException(
            f'Unsupported hash algorithm in integrity manifest: {algorithm!r}. '
            f'Expected: {_HASH_ALGORITHM!r}.'
        )

    manifest: Dict[str, str] = manifest_data['files']
    mismatches = []

    for filename, expected_hash in manifest.items():
        full_path = os.path.join(checkpoint_dir, filename)
        try:
            actual_hash = _compute_file_hash(full_path)
        except (FileNotFoundError, OSError) as exc:
            mismatches.append(f'  {filename}: file missing or unreadable ({exc})')
            continue
        if actual_hash != expected_hash:
            mismatches.append(
                f'  {filename}: hash mismatch '
                f'(expected {expected_hash[:16]}..., got {actual_hash[:16]}...)'
            )

    if mismatches:
        raise CheckpointingException(
            f'Checkpoint integrity verification failed for {len(mismatches)} '
            f'file(s) in {checkpoint_dir}:\n' + '\n'.join(mismatches)
        )

    logger.info(
        "Checkpoint integrity verified: %d file(s) OK in %s", len(manifest), checkpoint_dir
    )


def verify_integrity_manifest(checkpoint_dir: str) -> None:
    """Verify checkpoint files against their recorded SHA-256 hashes.

    In a distributed context (``torch.distributed`` initialised with more than
    one rank), only **rank 0** performs the I/O-intensive hash verification.
    The pass / fail result is then broadcast to every other rank so that all
    processes raise :class:`CheckpointingException` consistently on corruption.

    When ``torch.distributed`` is not initialised (single-process use), the
    verification runs directly in the calling process.

    Args:
        checkpoint_dir: checkpoint directory to verify.

    Raises:
        CheckpointingException: if ``integrity.json`` is absent or any file's
            hash no longer matches the stored value.
    """
    import torch

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        error_payload = [None]
        if torch.distributed.get_rank() == 0:
            try:
                _verify_integrity_manifest_impl(checkpoint_dir)
            except CheckpointingException as exc:
                error_payload = [str(exc)]
        torch.distributed.broadcast_object_list(error_payload, src=0)
        if error_payload[0] is not None:
            raise CheckpointingException(error_payload[0])
    else:
        _verify_integrity_manifest_impl(checkpoint_dir)

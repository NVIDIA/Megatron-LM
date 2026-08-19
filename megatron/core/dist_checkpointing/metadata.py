# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for framework content metadata stored in distributed checkpoints."""

import torch

DP_RESHARDABLE_PADDING_MANIFEST_KEY = 'distrib_optim_dp_reshardable_padding'
DP_RESHARDABLE_PADDING_MANIFEST_FORMAT = 1


def _validate_dp_reshardable_padding_manifest(manifest: dict) -> dict:
    """Validate and return a dp_reshardable sparse-padding manifest."""
    if not isinstance(manifest, dict):
        raise ValueError(f'Expected a padding manifest dict, got {type(manifest)}')
    if manifest.get('format') != DP_RESHARDABLE_PADDING_MANIFEST_FORMAT:
        raise ValueError(
            f'Unsupported dp_reshardable padding manifest format: {manifest.get("format")}'
        )
    buckets = manifest.get('buckets')
    if not isinstance(buckets, dict):
        raise ValueError('dp_reshardable padding manifest must contain a buckets dict')

    for bucket_key, bucket_metadata in buckets.items():
        if not isinstance(bucket_key, str) or not isinstance(bucket_metadata, dict):
            raise ValueError(f'Invalid padding manifest bucket entry: {bucket_key!r}')
        global_numel = bucket_metadata.get('global_numel')
        padding_ranges = bucket_metadata.get('padding_ranges')
        if not isinstance(global_numel, int) or global_numel < 0:
            raise ValueError(f'Invalid global_numel for {bucket_key}: {global_numel!r}')
        if not isinstance(padding_ranges, list):
            raise ValueError(f'Invalid padding_ranges for {bucket_key}: {padding_ranges!r}')

        previous_end = 0
        for padding_range in padding_ranges:
            if not (
                isinstance(padding_range, list)
                and len(padding_range) == 2
                and all(isinstance(offset, int) for offset in padding_range)
            ):
                raise ValueError(f'Invalid padding range for {bucket_key}: {padding_range!r}')
            start, end = padding_range
            if start < previous_end or start >= end or end > global_numel:
                raise ValueError(
                    f'Invalid or overlapping padding range for {bucket_key}: '
                    f'{padding_range!r} (global_numel={global_numel})'
                )
            previous_end = end
    return manifest


def get_dp_reshardable_padding_manifest(metadata: dict | None) -> dict | None:
    """Return the validated sparse-padding manifest from checkpoint content metadata."""
    if metadata is None or DP_RESHARDABLE_PADDING_MANIFEST_KEY not in metadata:
        return None
    return _validate_dp_reshardable_padding_manifest(metadata[DP_RESHARDABLE_PADDING_MANIFEST_KEY])


def add_dp_reshardable_padding_manifest_entry(
    metadata: dict | None, bucket_key: str, global_numel: int, padding_ranges: list[list[int]]
) -> None:
    """Add one deterministic bucket entry to a checkpoint content-metadata manifest."""
    if metadata is None:
        raise ValueError(
            'dp_reshardable checkpoints with omitted padding shards require content metadata'
        )

    manifest = metadata.setdefault(
        DP_RESHARDABLE_PADDING_MANIFEST_KEY,
        {'format': DP_RESHARDABLE_PADDING_MANIFEST_FORMAT, 'buckets': {}},
    )
    _validate_dp_reshardable_padding_manifest(manifest)
    entry = {
        'global_numel': int(global_numel),
        'padding_ranges': [[int(start), int(end)] for start, end in padding_ranges],
    }
    existing = manifest['buckets'].setdefault(bucket_key, entry)
    if existing != entry:
        raise ValueError(
            f'Conflicting dp_reshardable padding metadata for {bucket_key}: '
            f'{existing!r} != {entry!r}'
        )
    _validate_dp_reshardable_padding_manifest(manifest)


def extract_scoped_dp_reshardable_padding_metadata(metadata: dict, prefix: str) -> dict:
    """Copy metadata and expose only manifest entries belonging to ``prefix``.

    ChainedOptimizer prefixes ShardedTensor keys after each child builds its state dict. During
    load, strip that same prefix before passing the persisted manifest to the child optimizer.
    """
    scoped_metadata = dict(metadata)
    manifest = get_dp_reshardable_padding_manifest(metadata)
    if manifest is None:
        return scoped_metadata

    scoped_buckets = {
        bucket_key[len(prefix) :]: bucket_metadata
        for bucket_key, bucket_metadata in manifest['buckets'].items()
        if bucket_key.startswith(prefix)
    }
    if scoped_buckets:
        scoped_metadata[DP_RESHARDABLE_PADDING_MANIFEST_KEY] = {
            'format': DP_RESHARDABLE_PADDING_MANIFEST_FORMAT,
            'buckets': scoped_buckets,
        }
    else:
        scoped_metadata.pop(DP_RESHARDABLE_PADDING_MANIFEST_KEY, None)
    return scoped_metadata


def merge_scoped_dp_reshardable_padding_metadata(
    metadata: dict, scoped_metadata: dict, prefix: str
) -> None:
    """Merge a child optimizer's manifest into its parent using the final tensor-key prefix."""
    scoped_manifest = get_dp_reshardable_padding_manifest(scoped_metadata)
    if scoped_manifest is None:
        return
    for bucket_key, bucket_metadata in scoped_manifest['buckets'].items():
        add_dp_reshardable_padding_manifest_entry(
            metadata,
            f'{prefix}{bucket_key}',
            bucket_metadata['global_numel'],
            bucket_metadata['padding_ranges'],
        )


def merge_global_dp_reshardable_padding_metadata(metadata: dict) -> dict:
    """Merge model-parallel ranks' local manifest fragments into identical global metadata."""
    manifest = get_dp_reshardable_padding_manifest(metadata)
    local_buckets = {} if manifest is None else manifest['buckets']

    if torch.distributed.is_initialized():
        gathered_buckets = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_buckets, local_buckets)
    else:
        gathered_buckets = [local_buckets]

    merged_buckets = {}
    for buckets in gathered_buckets:
        for bucket_key, bucket_metadata in buckets.items():
            existing = merged_buckets.setdefault(bucket_key, bucket_metadata)
            if existing != bucket_metadata:
                raise ValueError(
                    f'Conflicting dp_reshardable padding metadata for {bucket_key}: '
                    f'{existing!r} != {bucket_metadata!r}'
                )

    merged_metadata = dict(metadata)
    if merged_buckets:
        merged_metadata[DP_RESHARDABLE_PADDING_MANIFEST_KEY] = {
            'format': DP_RESHARDABLE_PADDING_MANIFEST_FORMAT,
            'buckets': dict(sorted(merged_buckets.items())),
        }
    else:
        merged_metadata.pop(DP_RESHARDABLE_PADDING_MANIFEST_KEY, None)
    return merged_metadata

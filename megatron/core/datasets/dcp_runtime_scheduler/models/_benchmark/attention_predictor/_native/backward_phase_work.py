# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Source-derived GB200 backward loop descriptors, not latency predictions."""

from .cp1_matrix_work import GEOMETRIES, expected_ops

def phase_work(lengths, geometry, role):
    lengths, geometry = tuple(lengths), tuple(geometry)
    if (geometry not in GEOMETRIES or any(type(v) is not int for v in geometry)
            or not lengths or any(type(v) is not int or v < 1 for v in lengths)):
        raise ValueError('six captured geometries and positive unchanged raw lengths required')
    heads, _, dim = geometry
    if dim == 128 and role == 'backward_core_family_stage_0':
        query_rows, key_rows, weight, mask_limit = 128, 128, 10, 1
    elif dim == 256 and role in ('backward_core_family_stage_0',
            'backward_core_family_stage_1', 'backward_core_family_stage_2'):
        stage = int(role[-1])
        query_rows, key_rows = (128, 256) if stage < 2 else (256, 128)
        weight, mask_limit = (6, 4, 6)[stage], (2 if stage < 2 else 4)
    else:
        raise ValueError('an actual backward matrix program is required')
    counts = []
    for length in lengths:
        if dim == 256 and role.endswith('_2'):
            # Query-major consumer: causal checks are retained for the last
            # four K iterations, even though not every such panel is diagonal.
            current = [(min(length, start+query_rows)+key_rows-1)//key_rows
                       for start in range(0, length, query_rows)]
        else:
            # Key-major consumer starts at its diagonal Q panel. IMAD.IADD
            # with multiplier 1 computes begin-end, not begin+1-end.
            current = [(length+query_rows-1)//query_rows-start//query_rows
                       for start in range(0, length, key_rows)]
        counts.extend(current*heads)
    steps = sum(counts)
    masked = sum(min(count, mask_limit) for count in counts)
    unit = weight*dim*query_rows*key_rows
    if unit*steps != expected_ops(lengths, geometry, role):
        raise ValueError('loop descriptors must conserve independently counted matrix work')
    return dict(jobs=len(counts), steps=steps, masked_steps=masked,
                unmasked_steps=steps-masked, query_rows=query_rows,
                key_rows=key_rows, mask_limit=mask_limit,
                matrix_ops=unit*steps,
                masked_panel_elements=masked*query_rows*key_rows,
                consumer_threads_per_cta=256, ctas_per_cluster=dim//128,
                consumer_warps_per_cluster=8*(dim//128))

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Independent backward stage-service hypothesis, not an approved predictor.

The form matches the forward worker model: call and footprint service plus
the critical source-job service. Backward jobs come from backward programs,
never a multiplier of forward time. FIFO ownership and complete sequence
remapping remain explicit hypotheses; D256 GB200 uses static cluster bundles.
"""

from functools import lru_cache

from .geometry import GEOMETRIES

@lru_cache(maxsize=256)
def source_stages(lengths, geometry, hardware):
    lengths, geometry = tuple(lengths), tuple(geometry)
    if (hardware not in ('h100', 'gb200') or geometry not in GEOMETRIES
            or any(type(v) is not int for v in geometry)
            or not 1 <= len(lengths) <= 64
            or any(type(n) is not int or n < 1 for n in lengths)
            or sum(lengths) > 16384):
        raise ValueError('one original CP1 hardware/geometry and positive logical pack up to16K required')
    if hardware == 'h100':
        from .h100_causal_loops import consumer_jobs, work
        jobs = consumer_jobs(lengths, geometry, 'backward')
        counted = work(lengths, geometry, 'backward')
        if (len(jobs) != counted['source_dispatch_units']
                or sum(jobs) != counted['consumer_loop_iterations']):
            raise ValueError('every independent H100 backward dispatch unit and source loop must be conserved')
        return ((jobs, 132, 'fifo'),)

    from .native_query_backward_work import backward_work
    descriptors = [backward_work(n, n, geometry, bottom_right=False)['stages'] for n in lengths]
    # D128 traverses original IDs; D256 uses the descending-length remap.
    # Complete new remap-array contents and actual FIFO ownership are not
    # established by arithmetic counters and remain provisional.
    order = (range(len(lengths)) if geometry[2] == 128 else
             sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)))
    result = []
    for stage in range(len(descriptors[0])):
        reference = descriptors[0][stage]
        weight = reference['dense_operation_weight']
        jobs = tuple(weight * count for i in order for _ in range(geometry[0])
                     for count in descriptors[i][stage]['per_head_panel_steps'])
        wanted = sum(rows[stage]['dense_tensor_ops'] for rows in descriptors)
        if (min(jobs) <= 0
                or len(jobs) != sum(rows[stage]['candidate_panel_jobs'] for rows in descriptors)
                or sum(jobs) * geometry[2] * reference['query_tile'] * reference['key_tile'] != wanted):
            raise ValueError('each actual GB200 backward stage must conserve all independently counted dense work')
        result.append((jobs, 152 // (geometry[2] // 128),
                       'fifo' if geometry[2] == 128 else 'static_modulo'))
    return tuple(result)

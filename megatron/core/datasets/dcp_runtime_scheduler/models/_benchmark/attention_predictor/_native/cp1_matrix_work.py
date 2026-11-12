# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Zero-fit GB200 matrix-work hypothesis, not a latency or bandwidth model.

The forward256x128 macro-panel is source-backed for D128. The captured
D256 and backward paths are tested against every retained hardware count;
matching counts alone do not prove their internal dataflow or generality.
"""

GEOMETRIES = ((32, 4, 128), (64, 4, 128), (16, 2, 256),
              (32, 2, 256), (32, 8, 128), (64, 8, 128))

HELPERS = ('backward_output_dout_dot', 'backward_head_reduction_occurrence_0',
           'backward_head_reduction_occurrence_1')

def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name+' must be a positive integer')

def panel_pairs(lengths, query_rows, key_rows):
    """Count full issued macro-panels, never enlarge the logical sequences."""
    lengths = tuple(lengths)
    if not lengths:
        raise ValueError('at least one real sequence required')
    positive(query_rows, 'query panel rows')
    positive(key_rows, 'key panel rows')
    result = 0
    for length in lengths:
        positive(length, 'raw valid sequence length')
        for start in range(0, length, query_rows):
            columns = (min(length, start+query_rows)+key_rows-1)//key_rows
            result += query_rows*key_rows*columns
    return result

def expected_ops(lengths, geometry, role, *, hardware='gb200'):
    """Return candidate operation units for the captured BF16 tensor-path metric.

    This is issued tiled work, not valid FLOPs, time, DRAM bytes or a
    backward/forward latency ratio. Physical padding is not a new sequence.
    """
    lengths = tuple(lengths)
    if (hardware != 'gb200' or tuple(geometry) not in GEOMETRIES
            or any(type(v) is not int for v in geometry) or not lengths
            or any(type(v) is not int or v < 1 for v in lengths)):
        raise ValueError('only captured GB200 geometries and positive real lengths are supported')
    heads, _, dim = geometry
    if role in HELPERS or dim == 128 and role == 'backward_dq_conversion':
        return 0
    if role == 'forward_core_family':
        query_rows, key_rows, weight = 256, 128, 4
    elif dim == 128 and role == 'backward_core_family_stage_0':
        query_rows, key_rows, weight = 128, 128, 10
    elif dim == 256 and role in ('backward_core_family_stage_0',
            'backward_core_family_stage_1', 'backward_core_family_stage_2'):
        stage = int(role[-1])
        query_rows, key_rows = (128, 256) if stage < 2 else (256, 128)
        weight = (6, 4, 6)[stage]
    else:
        raise ValueError('unmeasured or impossible program role for this head dimension')
    return weight*heads*dim*panel_pairs(lengths, query_rows, key_rows)

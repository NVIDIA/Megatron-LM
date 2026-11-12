# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Source-derived matrix work for current GB200 causal/full rectangles, not time."""

from .export_cp1_program_bodies import require

from .geometry import GEOMETRIES

QUERY_TILE = 256

KEY_TILE = 128

MATRIX_WARP_INSTRUCTIONS_PER_STEP = {128: 32, 256: 24}

def forward_work(query_length, key_length, geometry, *, bottom_right, causal=True):
    geometry = tuple(geometry)
    require(geometry in GEOMETRIES and type(query_length) is int and type(key_length) is int
            and query_length >= 1 and key_length >= 1 and type(bottom_right) is bool
            and type(causal) is bool and (query_length <= key_length if causal else not bottom_right),
            'one supported geometry, unchanged positive logical lengths and explicit causal alignment')
    hq, _, dim = geometry
    offset = key_length-query_length if causal and bottom_right else 0
    counts = ([(offset+min(origin+QUERY_TILE, query_length)+KEY_TILE-1)//KEY_TILE
               for origin in range(0, query_length, QUERY_TILE)] if causal else
              [(key_length+KEY_TILE-1)//KEY_TILE for _ in range(0, query_length, QUERY_TILE)])
    steps, jobs = hq*sum(counts), hq*len(counts)
    result = dict(query_length=query_length, key_length=key_length, geometry=geometry,
        bottom_right=bottom_right, causal_offset=offset, query_tile=QUERY_TILE, key_tile=KEY_TILE,
        jobs=jobs, consumer_steps=steps, issued_matrix_warp_instructions=steps*MATRIX_WARP_INSTRUCTIONS_PER_STEP[dim],
        dense_tensor_ops=steps*4*QUERY_TILE*KEY_TILE*dim,
        logical_allowed_pairs=(query_length*(query_length+1)//2+query_length*offset if causal else query_length*key_length),
        per_query_tile_key_steps=counts, core_forward_matrix_work_only_NOT_latency=True,
        no_projection_no_communication_no_nonattention=True,
        backward_requires_independent_executed_program_analysis=True)
    if not causal:
        result.update(causal=False, mask='padding', full_mask_program_and_counter_validation_required=True)
    return result

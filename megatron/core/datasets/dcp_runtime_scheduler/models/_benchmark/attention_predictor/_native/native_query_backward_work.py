# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Causal/full backward matrix work, not a backward time multiplier."""

from .geometry import GEOMETRIES

from .native_terminal_trace import require

def backward_work(query_length, key_length, geometry, *, bottom_right, causal=True):
    geometry = tuple(geometry)
    require(geometry in GEOMETRIES and type(query_length) is type(key_length) is int
            and query_length >= 1 and key_length >= 1 and type(bottom_right) is bool
            and type(causal) is bool and (query_length <= key_length if causal else not bottom_right),
            'one supported geometry, unchanged positive lengths and explicit causal alignment')
    hq, _, dim = geometry
    offset = key_length-query_length if causal and bottom_right else 0
    # Descriptor sizes and matrix-operation weights come from the captured
    # backward phase programs, independently of the forward time model.
    descriptors = ((128, 128, 10, 'key'),) if dim == 128 else (
        (128, 256, 6, 'key'), (128, 256, 4, 'key'), (256, 128, 6, 'query'))
    stages = []
    for stage, (qtile, ktile, weight, major) in enumerate(descriptors):
        if not causal:
            counts = ([(key_length+ktile-1)//ktile]*((query_length+qtile-1)//qtile) if major == 'query' else
                      [(query_length+qtile-1)//qtile]*((key_length+ktile-1)//ktile))
        elif major == 'query':
            counts = [(offset+min(origin+qtile, query_length)+ktile-1)//ktile
                      for origin in range(0, query_length, qtile)]
        else:
            panels = (query_length+qtile-1)//qtile
            counts = [max(0, panels-max(0, (origin-offset)//qtile))
                      for origin in range(0, key_length, ktile)]
        steps = hq*sum(counts)
        stages.append(dict(stage=stage, query_tile=qtile, key_tile=ktile,
            major=major, dense_operation_weight=weight,
            per_head_panel_steps=counts, matrix_steps=steps,
            candidate_panel_jobs=hq*len(counts), zero_matrix_panel_jobs=hq*counts.count(0),
            dense_tensor_ops=steps*weight*dim*qtile*ktile))
    result = dict(query_length=query_length, key_length=key_length, geometry=geometry,
        bottom_right=bottom_right, causal_offset=offset, stages=stages,
        dense_tensor_ops=sum(s['dense_tensor_ops'] for s in stages),
        source_descriptors_NOT_empirical_fit_features=True,
        arithmetic_work_is_NOT_a_latency_prediction=True,
        no_forward_time_multiplier_no_projection_no_communication=True,
        arbitrary_rectangles_still_require_actual_program_and_counter_validation=True)
    if not causal:
        result.update(causal=False, mask='padding', native_global_LSE_backward_transfer_NOT_validated=True)
    return result

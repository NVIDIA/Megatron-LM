# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Common source-panel/job CP1 baseline, explicitly not accuracy-approved."""

from .geometry import GEOMETRIES, Pack

def work_counts(lengths,geometry,phase,hardware):
    if hardware not in ('h100','gb200') or tuple(geometry) not in GEOMETRIES or phase not in ('forward','backward'):
        raise ValueError('explicit calibrated hardware, geometry and independent direction required')
    pack=Pack.build(lengths,1,1)
    if hardware=='h100':
        from .h100_causal_loops import work
        result=work(pack.lengths,geometry,phase)
        return result['consumer_loop_iterations'],result['source_dispatch_units']
    if phase=='forward':
        from .native_query_work import forward_work
        rows=[forward_work(n,n,geometry,bottom_right=False) for n in pack.lengths]
        return sum(r['consumer_steps'] for r in rows),sum(r['jobs'] for r in rows)
    from .native_query_backward_work import backward_work
    stages=[s for n in pack.lengths for s in backward_work(n,n,geometry,bottom_right=False)['stages']]
    return sum(s['matrix_steps'] for s in stages),sum(s['candidate_panel_jobs'] for s in stages)

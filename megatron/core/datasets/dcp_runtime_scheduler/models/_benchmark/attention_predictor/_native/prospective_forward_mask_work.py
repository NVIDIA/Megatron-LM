# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Count source-selected normal/checked forward panels, without timing labels."""

from .export_cp1_program_bodies import require

from .geometry import GEOMETRIES, Pack

from .prospective_source_count_cp1 import work_counts

def panel_split(lengths,geometry,hardware):
    geometry=tuple(geometry);pack=Pack.build(lengths,1,1)
    require(geometry in GEOMETRIES and hardware in ('h100','gb200'),'captured hardware and geometry')
    panels,jobs=work_counts(pack.lengths,geometry,'forward',hardware)
    if hardware=='h100':
        checked_panels=jobs
    else:
        from .native_query_work import forward_work
        checked_panels=geometry[0]*sum(min(n,2) for length in pack.lengths
            for n in forward_work(length,length,geometry,bottom_right=False)['per_query_tile_key_steps'])
    require(0<checked_panels<=panels,'normal/checked classes conserve all source matrix iterations')
    return dict(panels=panels,jobs=jobs,checked_panels=checked_panels,normal_panels=panels-checked_panels)

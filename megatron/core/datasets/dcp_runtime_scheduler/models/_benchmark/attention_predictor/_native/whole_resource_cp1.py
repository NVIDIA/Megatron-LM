# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Provisional ordinary causal core-attention CP1 resource model, milliseconds.

Call boundary + compulsory valid/padded tensor-footprint service + native
source-job resource critical span. Footprints are not measured HBM traffic;
worker ownership and the resource envelope remain explicit hypotheses.
"""

import math

from .cp1_model import features as tensor_features

from .geometry import GEOMETRIES, PACK_LENGTHS, Pack

from .prospective_job_resource_cp1 import resource_service_span

SCHEMA = 'prospective-whole-core-native-job-resource-cp1/v1'

FEATURES = ('call_boundary', 'valid_tensor_mb', 'padding_tensor_mb', 'source_resource_span')

def features(pack, geometry, hardware, phase, coordinates):
    valid, padding = tensor_features(pack, geometry, phase)[2:4]
    return 1., valid, padding, resource_service_span(pack.lengths, tuple(geometry), hardware, phase, tuple(coordinates))

def predict_ms(model, lengths, *, sequence_alignment, pack_alignment, hardware,
               query_heads, kv_heads, head_dim, phase='forward', cp_size=1, allow_provisional=False):
    if (allow_provisional is not True or model.get('schema') != SCHEMA
            or model.get('features') != list(FEATURES) or model.get('hardware') != hardware
            or model.get('approved_for_runtime_use') is not False
            or model.get('independent_accuracy_passed') is not False
            or model.get('holdout_times_acquired') is not False
            or phase not in ('forward', 'backward') or type(cp_size) is not int or cp_size != 1):
        raise ValueError('explicit provisional CP1 prediction required; no independent accuracy or multi-CP approval')
    geometry = query_heads, kv_heads, head_dim
    if geometry not in GEOMETRIES or any(type(n) is not int for n in geometry):
        raise ValueError('one of the six captured attention geometries required')
    pack = Pack.build(lengths, sequence_alignment, pack_alignment, static_slots=64)
    if pack.storage_tokens not in PACK_LENGTHS:
        raise ValueError('captured4K/8K/16K storage required')
    matches = [h for h in model['heads'] if tuple(h['geometry']) == geometry and h['phase'] == phase]
    if len(matches) != 1:
        raise ValueError('one independently fitted hardware/geometry/direction head required')
    head = matches[0]
    prices = head['coefficients_ms']
    if len(prices) != len(FEATURES) or any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in prices):
        raise ValueError('four finite nonnegative physical service prices required')
    value = math.fsum(p * x for p, x in zip(prices, features(pack, geometry, hardware, phase,
                                 head['source_price_coordinates']), strict=True))
    if not math.isfinite(value) or value <= 0:
        raise ValueError('positive finite prediction required without clipping or residual compensation')
    return value

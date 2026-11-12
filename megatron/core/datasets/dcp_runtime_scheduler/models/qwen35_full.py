# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full Qwen3.5-35B module costs with explicitly supplied remaining work."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from ..runtime.planner import MAX_PRICED_GROUP_SEQUENCES
from ..runtime.timeline import _freeze_op
from .base import NVLINK_DOMAIN_SIZES
from .benchmark import BenchmarkTimeline
from .full35b import PARAMETERS_SHA256
from .full35b.gdn_features import features as gdn_features
from .full35b.geometry import dot, fa_features, pack_geometry
from .selection import validate_full_qwen35_config

# Values of thd_tail_padding_policy; the unset policy resolves to the first.
_TAIL_PADDING_POLICIES = ('append_dummy_seq', 'extend_last')
# Parallel geometry of the measured source: (label in messages, measured value, geometry field).
_MEASURED_GEOMETRY = (
    ('DPxCP', 32, 'world_size'),
    ('expert_model_parallel_size', 32, 'ep_size'),
    ('max_seqlen_per_dp_cp_rank', 16384, 'tokens_per_rank'),
    ('dataset_alignment', 64, 'dataset_alignment'),
)


def _duration(value, label):
    if type(value) not in (int, float):
        raise ValueError(f'{label} must be finite nonnegative milliseconds')
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{label} must be finite nonnegative milliseconds')
    return float(value)


class Qwen35FullCostModel:
    """Compose all 40 decoder layers and the MTP layer in their actual order.

    Frozen FA/GDN predictions cover the whole attention module, including its
    projections and layout work. Required user timings exclude that module and
    include MoE/EP and other remaining work. No density regression is loaded.

    The features evaluate every pack in the layout of the measured source, where the
    padding tail extends the last sequence. The tail policy the training run executes
    is not modelled; the metadata reports both.
    """

    name = 'dcp_benchmark_qwen35_full35b'
    layer_count = 41
    placement_invariant = False
    # Largest CP group this model prices; the scheduler forms no larger one.
    largest_cp_size = 32
    # Layer types a per-type ``non_attention`` mapping must name for this model; the MTP
    # layer uses the ``gqa`` value unless ``mtp`` is given.
    non_attention_layer_types = ('gdn', 'gqa')
    non_attention_optional_layer_types = ('mtp',)

    def __init__(
        self,
        geometry,
        *,
        hardware='gb200',
        non_attention,
        first_forward_dp_cp_sync=True,
        tail_padding='append_dummy_seq',
    ):
        self.geometry = geometry
        if hardware != 'gb200':
            raise ValueError(f'"hardware" must be "gb200" for full Qwen3.5, got {hardware!r}')
        self.hardware = hardware
        # All 32 ranks of the measured source lie in one NVLink domain, so a CP group has
        # the same price wherever it starts.
        self.nvlink_domain_size = NVLINK_DOMAIN_SIZES[hardware]
        if type(first_forward_dp_cp_sync) is not bool:
            raise ValueError('first_forward_dp_cp_sync must be a boolean')
        self.first_forward_dp_cp_sync = first_forward_dp_cp_sync
        if tail_padding not in _TAIL_PADDING_POLICIES:
            raise ValueError(
                f'tail_padding must be one of {list(_TAIL_PADDING_POLICIES)}, got {tail_padding!r}'
            )
        # The THD tail policy of the training run. It is reported in the metadata only: the
        # FA/GDN features keep the pack layout of the measured source (see ``metadata``).
        self.executed_tail_padding = tail_padding
        self.check_domain(geometry)
        root = Path(__file__).with_name('full35b')
        raw = (root / 'parameters.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != PARAMETERS_SHA256:
            raise ValueError('Frozen full Qwen3.5-35B parameter hash mismatch')
        parameters = json.loads(raw)
        if parameters.get('schema') != 'qwen35-full35b-fixed-input-runtime/1':
            raise ValueError('Unsupported full Qwen3.5-35B parameter schema')
        manifest_raw = (root / 'manifest.json').read_bytes()
        manifest = json.loads(manifest_raw)
        for name, digest in manifest['packaged_sha256'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Frozen full Qwen3.5-35B source hash mismatch: {name}')
        self.attention = parameters['attention']
        self.layers = parameters['layers']
        self.cache = parameters['cache']
        self.original_status = parameters['source_status']
        self.source_execution_domain = parameters['source_execution_domain']
        expected = ('gdn', 'gdn', 'gdn', 'fa') * 10 + ('fa',)
        if tuple(layer['kind'] for layer in self.layers) != expected:
            raise ValueError('Full Qwen3.5-35B requires the actual 40 decoder + 1 MTP ordering')
        self.layer_types = expected
        self.source_hashes = {
            'parameters.json': PARAMETERS_SHA256,
            'manifest.json': hashlib.sha256(manifest_raw).hexdigest(),
            'source_candidate.json': manifest['source_candidate_sha256'],
            **{'source/' + name: digest for name, digest in manifest['source_sha256'].items()},
        }
        self.non_attention = self._non_attention_inputs(non_attention)
        self.iteration_tail_ms = self.non_attention.get('iteration_tail_ms', 0.0)
        self.microbatch_ms = self.non_attention.get('microbatch_ms', 0.0)
        self._fixed = {
            phase: {layer['name']: self._layer_duration(phase, layer) for layer in self.layers}
            for phase in ('forward', 'backward')
        }
        self.source_hashes['non_attention'] = hashlib.sha256(
            json.dumps(self.non_attention, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        self.execution_program = self._execution_program()

    def _non_attention_inputs(self, values):
        if not isinstance(values, dict):
            raise ValueError('Full Qwen3.5-35B requires explicit non_attention inputs')
        allowed = {
            'forward_ms_per_layer',
            'backward_ms_per_layer',
            'source',
            'iteration_tail_ms',
            'microbatch_ms',
        }
        if set(values) - allowed:
            raise ValueError('Unknown full Qwen3.5-35B non_attention field')
        result = copy.deepcopy(values)
        result.setdefault('source', 'user-provided fixed non-attention timings')
        if not isinstance(result['source'], str) or not result['source'].strip():
            raise ValueError('non_attention source must be nonempty when supplied')
        names = {layer['name'] for layer in self.layers}
        for phase in ('forward', 'backward'):
            key = phase + '_ms_per_layer'
            if key not in result:
                raise ValueError(f'Full Qwen3.5-35B requires non_attention.{key}')
            value = result[key]
            if isinstance(value, Mapping):
                if not value or set(value) - names - {'gdn', 'fa', 'gqa', 'mtp'}:
                    raise ValueError(f'{key} needs gdn/gqa/mtp layer types or actual layer names')
                if 'fa' in value and 'gqa' in value:
                    raise ValueError(f'{key} must use either fa or its gqa alias')
                result[key] = {name: _duration(v, f'{key}.{name}') for name, v in value.items()}
            else:
                result[key] = _duration(value, key)
        for key in ('iteration_tail_ms', 'microbatch_ms'):
            if key in result:
                result[key] = _duration(result[key], key)
        return result

    def _layer_duration(self, phase, layer):
        value = self.non_attention[phase + '_ms_per_layer']
        if isinstance(value, Mapping):
            kind = layer['kind']
            if layer['name'] in value:
                return value[layer['name']]
            if layer['name'] == self.layers[-1]['name'] and 'mtp' in value:
                return value['mtp']
            if kind in value:
                return value[kind]
            if kind == 'fa' and 'gqa' in value:
                return value['gqa']
            raise ValueError(f'Missing {phase} non-attention timing for {layer["name"]}')
        return value

    def check_domain(self, geometry, config=None):
        """Check the measured parallel geometry; model and execution checks live in selection."""
        if geometry != self.geometry:
            raise ValueError('Full Qwen3.5-35B parameters were bound to a different geometry')
        # Name the first setting that differs, with the measured and the actual value.
        for label, measured, field in _MEASURED_GEOMETRY:
            actual = getattr(geometry, field)
            if actual != measured:
                raise ValueError(
                    f'Full Qwen3.5-35B parameters were measured with {label}={measured}, '
                    f'got {actual}'
                )
        allowed = tuple(geometry.allowed_cp_sizes)
        if not allowed or any(
            type(cp) is not int or not 1 <= cp <= self.largest_cp_size for cp in allowed
        ):
            raise ValueError(
                'Full Qwen3.5-35B parameters cover CP groups of 1 to '
                f'{self.largest_cp_size} ranks, got {list(allowed)}'
            )
        # The parameters cover packs of up to 64 sequences. A smaller cap is inside that
        # domain: 63 when a 64-entry THD buffer reserves one slot for a dummy tail sequence.
        sequence_cap = geometry.max_group_seqs
        if not (
            sequence_cap is None
            or (type(sequence_cap) is int and 1 <= sequence_cap <= MAX_PRICED_GROUP_SEQUENCES)
        ):
            raise ValueError(
                f'Full Qwen3.5-35B parameters cover at most {MAX_PRICED_GROUP_SEQUENCES} '
                f'sequences per CP group, got {sequence_cap}'
            )
        if config is not None:
            validate_full_qwen35_config(config)

    def _check_cp(self, cp):
        if type(cp) is not int or cp not in self.geometry.allowed_cp_sizes:
            raise ValueError('CP size is outside the full Qwen3.5-35B prediction domain')

    def _ranks(self, cp, ranks):
        self._check_cp(cp)
        if ranks is None:
            raise ValueError('Actual contiguous physical rank IDs are required')
        ranks = tuple(ranks)
        if (
            len(ranks) != cp
            or any(type(rank) is not int for rank in ranks)
            or ranks != tuple(range(ranks[0], ranks[0] + cp))
            or ranks[0] < 0
            or ranks[-1] >= 32
        ):
            raise ValueError('Actual contiguous physical ranks must be inside the 32-rank domain')
        return ranks

    @lru_cache(maxsize=65536)
    def _attention_costs(self, lengths, cp):
        self._check_cp(cp)
        point = pack_geometry(lengths, cp)
        features = fa_features(point)
        fa = {
            phase: dot(self.attention['models_fa_module'][str(cp)][phase]['coefficients'], features)
            for phase in ('forward', 'backward')
        }
        gdn = {
            phase: {
                branch: dot(
                    self.attention['models_gdn_module'][phase + '_cache' + str(branch)][
                        'coefficients'
                    ],
                    gdn_features(point, phase),
                )
                for branch in (0, 1)
            }
            for phase in ('forward', 'backward')
        }
        return fa, gdn

    def attention_components(self, lengths, cp, *, cache_pass=0, scope='module'):
        """Expose unchanged source FA/GDN module predictions for one cache branch."""
        if scope != 'module':
            raise ValueError(
                'Full Qwen3.5-35B predicts whole attention modules, not core attention'
            )
        if type(cache_pass) is not int or cache_pass not in (0, 1):
            raise ValueError('GDN cache_pass must be 0 or 1')
        fa, gdn = self._attention_costs(tuple(lengths), cp)
        return {
            'fa': {'module': dict(fa)},
            'gdn': {'module': {phase: gdn[phase][cache_pass] for phase in ('forward', 'backward')}},
        }

    @lru_cache(maxsize=65536)
    def _program_costs(self, lengths, cp, ranks):
        fa, gdn = self._attention_costs(lengths, cp)
        result = {}
        for layer in self.layers:
            name, kind = layer['name'], layer['kind']
            for phase in ('forward', 'backward'):
                branch = self.cache['by_cp'][str(cp)][name][phase] if kind == 'gdn' else 0
                result[name + ':attention:' + phase] = (
                    gdn[phase][branch] if kind == 'gdn' else fa[phase]
                )
                result[name + ':nonattention:' + phase] = self._fixed[phase][name]
        result['outer_microbatch'] = self.microbatch_ms
        return result

    def program_costs(self, lengths, cp, *, ranks=None):
        """Return all 41 named module costs and fixed user non-attention inputs."""
        ranks = self._ranks(cp, ranks)
        return dict(self._program_costs(tuple(lengths), cp, ranks))

    def forward(self, lengths, cp, *, ranks=None):
        """Return the mean whole-attention-module forward cost for search ordering."""
        values = self.program_costs(lengths, cp, ranks=ranks)
        return sum(values[layer['name'] + ':attention:forward'] for layer in self.layers) / 41

    def backward(self, lengths, cp, *, ranks=None):
        """Return the mean whole-attention-module backward cost for search ordering."""
        values = self.program_costs(lengths, cp, ranks=ranks)
        return sum(values[layer['name'] + ':attention:backward'] for layer in self.layers) / 41

    def _execution_program(self):
        forward, backward = [], []
        for layer in self.layers:
            forward.extend(
                [
                    dict(sync='cp', cost=layer['name'] + ':attention:forward'),
                    dict(sync='ep', duration_ms=self._fixed['forward'][layer['name']]),
                ]
            )
        for layer in reversed(self.layers):
            backward.extend(
                [
                    dict(sync='ep', duration_ms=self._fixed['backward'][layer['name']]),
                    dict(sync='cp', cost=layer['name'] + ':attention:backward'),
                ]
            )
        backward.append(dict(sync='local', duration_ms=self.microbatch_ms))
        return [dict(layer_count=41, forward=forward, backward=backward)]

    def timing(self):
        """Use the full model's ordered layer program and only user-supplied tails."""
        return dict(
            execution_program=copy.deepcopy(self.execution_program),
            iteration_tail_ms=self.iteration_tail_ms,
        )

    def build_timeline(self, geometry):
        """Score the complete 41-layer program with the shared benchmark E2E engine."""
        self.check_domain(geometry)
        program = self.execution_program[0]
        return BenchmarkTimeline(
            world_size=geometry.world_size,
            ep_size=geometry.ep_size,
            layer_count=41,
            iteration_tail_ms=self.iteration_tail_ms,
            first_forward_dp_cp_sync=self.first_forward_dp_cp_sync,
            program=tuple(
                tuple(_freeze_op(op) for op in program[phase]) for phase in ('forward', 'backward')
            ),
            execution_program=self.execution_program,
        )

    def clear(self):
        """Reset all per-global-batch module and program prediction caches."""
        self._attention_costs.cache_clear()
        self._program_costs.cache_clear()

    def metadata(self):
        """Report the model scope, the fixed inputs, the source domain and its accuracy gates."""
        # Held-out gate results of the source fits: the facts, without its status labels.
        accuracy = {
            gate: passed
            for gate, passed in self.original_status['accuracy'].items()
            if type(passed) is bool
        }
        return dict(
            name=self.name,
            model='Qwen3.5-35B-A3B',
            hardware=self.hardware,
            nvlink_domain_size=self.nvlink_domain_size,
            layer_count=41,
            decoder_layer_count=40,
            mtp_layer_count=1,
            attention_geometry=[16, 2, 256],
            layer_kinds=list(self.layer_types),
            attention_prediction_scope='WHOLE_MODULE_ONLY',
            whole_attention_module_includes_projections=True,
            core_prediction_available=False,
            gdn_feature_schema=self.attention['gdn_feature_schema'],
            non_attention=copy.deepcopy(self.non_attention),
            non_attention_scope='fixed user times excluding the whole FA/GDN attention module',
            non_attention_mapping_keys=[
                *self.non_attention_layer_types,
                *self.non_attention_optional_layer_types,
            ],
            non_attention_density_regression_loaded=False,
            e2e_residual_fitted=False,
            source_hashes=dict(self.source_hashes),
            source_execution_domain=copy.deepcopy(self.source_execution_domain),
            # The features evaluate every pack in the layout of the measured source, whose
            # tail policy is the first of these two; training executes the second.
            source_tail_padding=self.source_execution_domain['geometry_policy']['tail_padding'],
            executed_tail_padding=self.executed_tail_padding,
            scientific_accuracy_passed=bool(accuracy) and all(accuracy.values()),
            accuracy=accuracy,
            cache_observed_cp_sizes=list(self.cache['observed_cp_sizes']),
            full_cp_cache_observation_claimed=self.cache['fullCP_observation_claimed'],
            forward_cache_policy=self.cache['forward_policy'],
            backward_cache_policy=self.cache['backward_policy'],
        )

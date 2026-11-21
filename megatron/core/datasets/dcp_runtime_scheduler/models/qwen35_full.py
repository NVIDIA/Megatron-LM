# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full Qwen3.5-35B module costs with explicitly supplied remaining work."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from ..runtime.timeline import layer_program
from .base import NVLINK_DOMAIN_SIZES
from .full35b import PARAMETERS_SHA256
from .full35b.gdn_features import features as gdn_features
from .full35b.geometry import dot, fa_features, pack_geometry
from .selection import validate_full_qwen35_config

# Parallel geometry of the measured source: (label in messages, measured value, geometry field).
_MEASURED_GEOMETRY = (
    ('DPxCP', 32, 'world_size'),
    ('expert_model_parallel_size', 32, 'ep_size'),
    ('max_seqlen_per_dp_cp_rank', 16384, 'tokens_per_rank'),
    ('dataset_alignment', 64, 'dataset_alignment'),
)


class Qwen35FullCostModel:
    """Compose all 40 decoder layers and the MTP layer in their actual order.

    FA/GDN predictions cover the whole attention module, including its
    projections and layout work. Required user timings exclude that module and
    include MoE/EP and other remaining work. No density regression is loaded.

    The features evaluate every pack in the layout of the measured source, where the
    padding tail extends the last sequence. The tail policy the training run executes
    (append_dummy_seq, the only one GDN admits with a fixed per-rank buffer) is not modelled.
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

    def __init__(self, geometry, *, hardware='gb200', non_attention):
        self.geometry = geometry
        if hardware != 'gb200':
            raise ValueError(f'"hardware" must be "gb200" for full Qwen3.5, got {hardware!r}')
        self.hardware = hardware
        # All 32 ranks of the measured source lie in one NVLink domain, so a CP group has
        # the same price wherever it starts.
        self.nvlink_domain_size = NVLINK_DOMAIN_SIZES[hardware]
        root = Path(__file__).with_name('full35b')
        raw = (root / 'parameters.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != PARAMETERS_SHA256:
            raise ValueError('Frozen full Qwen3.5-35B parameter hash mismatch')
        parameters = json.loads(raw)
        manifest_raw = (root / 'manifest.json').read_bytes()
        manifest = json.loads(manifest_raw)
        for name, digest in manifest['packaged_sha256'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Frozen full Qwen3.5-35B source hash mismatch: {name}')
        # Coefficients by direction: one vector for every CP size (see full35b/geometry.py
        # and full35b/gdn_features.py for the features they multiply).
        self.full_attention = parameters['full_attention']
        self.gdn = parameters['gdn']
        self.source_execution_domain = parameters['source_execution_domain']
        # The actual order: three GDN layers and one FA layer, ten times, then the FA MTP layer.
        expected = ('gdn', 'gdn', 'gdn', 'fa') * 10 + ('fa',)
        names = parameters['layer_names']
        self.layers = [
            dict(kind=kind, name=names['decoder'].format(index=index))
            for index, kind in enumerate(expected[:-1])
        ]
        self.layers.append(dict(kind='fa', name=names['mtp']))
        # With CP > 1 the first GDN layer builds the CP context in its forward pass (the
        # cold pass of the measured source); every other GDN forward pass is hot.
        self._cold_layer = self.layers[0]['name']
        self.source_hashes = {
            'parameters.json': PARAMETERS_SHA256,
            'manifest.json': hashlib.sha256(manifest_raw).hexdigest(),
            'source_candidate.json': manifest['source_candidate_sha256'],
            **{'source/' + name: digest for name, digest in manifest['source_sha256'].items()},
            **{
                'points/' + name: digest
                for name, digest in manifest['measured_points_sha256'].items()
            },
        }
        self.non_attention = self._non_attention_inputs(non_attention)
        self._fixed = {
            phase: {layer['name']: self._layer_duration(phase, layer) for layer in self.layers}
            for phase in ('forward', 'backward')
        }
        self.source_hashes['non_attention'] = hashlib.sha256(
            json.dumps(self.non_attention, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        self.execution_program = self._execution_program()

    def _non_attention_inputs(self, values):
        """The times of the runtime file as floats: one per phase, or one per layer type."""
        result = dict(values)
        result.setdefault('source', 'user-provided fixed non-attention timings')
        for key in ('forward_ms_per_layer', 'backward_ms_per_layer'):
            value = result[key]
            if isinstance(value, Mapping):
                result[key] = {name: float(v) for name, v in value.items()}
            else:
                result[key] = float(value)
        return result

    def _layer_duration(self, phase, layer):
        value = self.non_attention[phase + '_ms_per_layer']
        if not isinstance(value, Mapping):
            return value
        if layer is self.layers[-1] and 'mtp' in value:
            return value['mtp']
        return value['gdn' if layer['kind'] == 'gdn' else 'gqa']

    def check_domain(self, geometry, config=None):
        """Check the measured parallel geometry; model and execution checks live in selection."""
        # Name the first setting that differs, with the measured and the actual value.
        for label, measured, field in _MEASURED_GEOMETRY:
            actual = getattr(geometry, field)
            if actual != measured:
                raise ValueError(
                    f'Full Qwen3.5-35B parameters were measured with {label}={measured}, '
                    f'got {actual}'
                )
        if config is not None:
            validate_full_qwen35_config(config)

    @lru_cache(maxsize=65536)
    def _program_costs(self, lengths, cp):
        if type(cp) is not int or cp not in self.geometry.allowed_cp_sizes:
            raise ValueError('CP size is outside the full Qwen3.5-35B prediction domain')
        point = pack_geometry(lengths, cp)
        features = fa_features(point)
        fa = {phase: dot(self.full_attention[phase], features) for phase in ('forward', 'backward')}
        # GDN forward is the hot pass; the backward price does not depend on the cache pass.
        gdn = {
            phase: dot(self.gdn[phase], gdn_features(point, phase))
            for phase in ('forward', 'backward')
        }
        result = {}
        for layer in self.layers:
            name, kind = layer['name'], layer['kind']
            for phase in ('forward', 'backward'):
                cost = gdn[phase] if kind == 'gdn' else fa[phase]
                if phase == 'forward' and cp > 1 and name == self._cold_layer:
                    cost += self.gdn['first_layer_cold_forward_ms']
                result[name + ':attention:' + phase] = cost
                result[name + ':nonattention:' + phase] = self._fixed[phase][name]
        return result

    def program_costs(self, lengths, cp, *, ranks=None):
        """Return all 41 named module costs and fixed user non-attention inputs."""
        return dict(self._program_costs(tuple(lengths), cp))

    def forward(self, lengths, cp, *, ranks=None):
        """Return the mean whole-attention-module forward cost for search ordering."""
        values = self.program_costs(lengths, cp, ranks=ranks)
        return sum(values[layer['name'] + ':attention:forward'] for layer in self.layers) / 41

    def backward(self, lengths, cp, *, ranks=None):
        """Return the mean whole-attention-module backward cost for search ordering."""
        values = self.program_costs(lengths, cp, ranks=ranks)
        return sum(values[layer['name'] + ':attention:backward'] for layer in self.layers) / 41

    def _execution_program(self):
        return [
            layer_program(
                [
                    (
                        layer['name'] + ':attention:forward',
                        self._fixed['forward'][layer['name']],
                        layer['name'] + ':attention:backward',
                        self._fixed['backward'][layer['name']],
                    )
                    for layer in self.layers
                ]
            )
        ]

    def timing(self):
        """Use the full model's ordered layer program."""
        return dict(execution_program=copy.deepcopy(self.execution_program))

    def clear(self):
        """Reset the per-global-batch module and program prediction cache."""
        self._program_costs.cache_clear()

    def metadata(self):
        """Report the fixed inputs, the file digests and the tail policy of the measured source."""
        return dict(
            name=self.name,
            hardware=self.hardware,
            nvlink_domain_size=self.nvlink_domain_size,
            non_attention=copy.deepcopy(self.non_attention),
            source_hashes=dict(self.source_hashes),
            source_tail_padding=self.source_execution_domain['geometry_policy']['tail_padding'],
        )

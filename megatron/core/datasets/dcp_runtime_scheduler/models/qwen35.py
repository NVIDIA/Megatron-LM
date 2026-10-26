# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Frozen Qwen3.5 mixed-layer proxy used for runtime planning, not GPU claims."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from functools import lru_cache

from .attention.predictor import load_parameters

FEATURES = (
    'bias',
    'cp_enabled',
    'summary_hops',
    'max_prefix_chain',
    'boundary_summary_chunks_per_64',
    'extra_local_sequences',
)
LAYER_TYPES = ('gdn', 'gdn', 'gdn', 'fa', 'fa')


class GdnPredictor:
    """Only the chunkwise GDN coefficients consumed by the selected composition."""

    def __init__(self):
        data, self.parameter_sha256 = load_parameters('qwen35.json')
        self.data = data['gdn']
        self.calibration_sha256 = self.data['source_sha256']
        if tuple(self.data['gdn_features']) != FEATURES:
            raise ValueError('GDN feature contract mismatch')

    def metadata(self):
        """Keep calibration provenance, geometry and transfer limitations visible."""
        return dict(
            profile=self.data['profile'],
            hardware='gb200',
            domain_size=72,
            transport_backend='native',
            geometry=self.data['geometry'],
            calibration_sha256=self.calibration_sha256,
            cost_scope='attention_module',
            local_tokens=16384,
            phase_split='measured_separately',
            provisional=True,
            provisional_reasons=[
                'arbitrary pack/partition transfer is not universally validated',
                'many-short-sequence packs at small odd CP are under-predicted',
            ],
            padding='GDN coefficients transferred from appended dummy to extend_last',
        )

    @staticmethod
    def _features(cu, cp, phase, recompute, local):
        difference = [0] * (cp + 1)
        chain = forward_chunks = backward_chunks = 0
        for begin, end in zip(cu, cu[1:]):
            first, last = (begin // local, (end - 1) // local)
            difference[first] += 1
            difference[last + 1] -= 1
            chain = max(chain, last - first)
            if last > first:
                forward_chunks = max(forward_chunks, (local - begin % local + 63) // 64)
                backward_chunks = max(backward_chunks, ((end - 1) % local + 64) // 64)
                if last - first > 1:
                    forward_chunks = backward_chunks = local // 64
        active = maximum = 0
        for delta in difference[:-1]:
            active += delta
            maximum = max(maximum, active)
        nf = 1 if phase == 'forward_gpu_ms' else int(recompute)
        nb = 0 if phase == 'forward_gpu_ms' else 1
        if phase == 'forward_backward_gpu_ms':
            nf += 1
        return (
            1.0,
            float(cp > 1),
            (nf + nb) * (cp - 1),
            (nf + nb) * chain,
            (nf * forward_chunks + nb * backward_chunks) / 64,
            float(maximum - 1),
        )


def load_non_attention_profile(profile: str) -> dict:
    """The paired GB200/EP4/local16K measurements of the selected proxy."""
    if profile != 'gb200-ep4-local16384-7319138':
        raise ValueError('Unsupported paired non-attention profile')
    return copy.deepcopy(load_parameters('qwen35.json')[0]['non_attention'])


def execution_timing(other, recompute):
    """Execution timing."""

    def attention(kind, phase):
        return dict(sync='cp', cost=f'{kind}_{phase}')

    def remainder(key):
        return dict(sync='ep', duration_ms=other[key])

    forward, backward = ([], [])
    for kind in LAYER_TYPES:
        forward.extend([attention(kind, 'f'), remainder('layer_forward')])
    for kind in reversed(LAYER_TYPES):
        if recompute:
            backward.extend([attention(kind, 'f'), remainder('layer_recompute_forward')])
        backward.extend([remainder('layer_backward'), attention(kind, 'b')])
    backward.append(dict(sync='local', duration_ms=other['microbatch_outer']))
    return dict(
        execution_program=[dict(layer_count=len(LAYER_TYPES), forward=forward, backward=backward)],
        iteration_tail_ms=other['iteration_tail'],
    )


class Qwen35CoreCostModel:
    """Three calibrated GDN modules + two new FA cores and measured FA remainders.

    The injected callback returns a ``forward_ms``/``backward_ms`` mapping. This
    module imports no attention predictor or scheduler. A caller must supply a
    paired measured FA remainder; model-minus-model offsets are rejected.
    """

    name = 'qwen35_gb200_new_core'
    layer_count = 5
    placement_invariant = True

    def __init__(self, core_predictor, *, fa_remainder, non_attention=None):
        if not callable(core_predictor):
            raise TypeError('core_predictor must be callable')
        if fa_remainder.get('status') != 'PAIRED_FA_REMAINDER_PASS':
            raise ValueError('a measured paired FA module-minus-core input is required')
        if (
            fa_remainder.get('hardware') != 'gb200'
            or fa_remainder.get('tokens_per_rank') != 16384
            or fa_remainder.get('geometry') != [32, 2, 256]
        ):
            raise ValueError('FA remainder scope requires GB200 / local16K / 32,2,256')
        remainder = fa_remainder.get('cost_ms', {})
        if set(remainder) != {'forward', 'backward'} or any(
            (not math.isfinite(float(v)) or v < 0 for v in remainder.values())
        ):
            raise ValueError('FA remainder needs finite nonnegative forward and backward costs')
        if not fa_remainder.get('source_hashes'):
            raise ValueError('FA remainder must identify measured source records')
        self.core_predictor = core_predictor
        self.fa_remainder = copy.deepcopy(fa_remainder)
        self.non_attention = (
            copy.deepcopy(non_attention)
            if non_attention is not None
            else load_non_attention_profile('gb200-ep4-local16384-7319138')
        )
        if self.non_attention.get('profile') != 'gb200-ep4-local16384-7319138':
            raise ValueError('this composition requires the measured GB200 EP4/local16K profile')
        if self.non_attention.get('status') != 'PAIRED_MODULE_PROFILE_PASS':
            raise ValueError('a paired measured non-attention profile is required')
        other = self.non_attention.get('cost_ms', {})
        if set(other) != {
            'layer_forward',
            'layer_backward',
            'microbatch_outer',
            'iteration_tail',
        } or any(
            (
                not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in other.values()
            )
        ):
            raise ValueError('non-attention inputs must be fixed finite nonnegative measured times')
        self.gdn = GdnPredictor()
        self.source_hashes = {
            'gdn_calibration': self.gdn.calibration_sha256,
            'non_attention_input': hashlib.sha256(
                json.dumps(self.non_attention, sort_keys=True).encode()
            ).hexdigest(),
            'fa_remainder_input': hashlib.sha256(
                json.dumps(self.fa_remainder, sort_keys=True).encode()
            ).hexdigest(),
        }
        self._guide_table = {}

    def _gdn_costs(self, lengths, cp):
        stored = [((n + 63) // 64 * 64 + 2 * cp - 1) // (2 * cp) * (2 * cp) for n in lengths]
        if sum(stored) > cp * 16384:
            raise ValueError('pack exceeds physical capacity')
        stored[-1] += cp * 16384 - sum(stored)
        cu = list(itertools.accumulate(stored, initial=0))
        costs = {}
        for phase, key in (('forward_gpu_ms', 'gdn_f'), ('backward_gpu_ms', 'gdn_b')):
            coefficients = self.gdn.data['gdn']['plain'][phase]
            features = self.gdn._features(cu, cp, phase, False, 16384)
            costs[key] = sum(
                (coefficients[name] * value for name, value in zip(FEATURES, features))
            )
        return costs

    def prepare_guide(self, cp_sizes):
        """Small predictor-query table for proposal ordering, NOT final scoring."""
        for cp in cp_sizes:
            if cp in self._guide_table:
                continue
            capacity = cp * 16384
            levels = {}
            for count in (1, 2, 3, 4, 6, 8, 16, 32, 64):
                packs = []
                for divisor in (64, 4, 2, 4 / 3, 1):
                    length = max(1, int(capacity // (divisor * count)))
                    while (
                        count * (((length + 63) // 64 * 64 + 2 * cp - 1) // (2 * cp) * (2 * cp))
                        > capacity
                    ):
                        length -= 64
                    packs.append((length,) * count)
                if count > 1:
                    short = min(1024, max(1, capacity // (8 * count)))
                    physical_short = ((short + 63) // 64 * 64 + 2 * cp - 1) // (2 * cp) * (2 * cp)
                    available = capacity - (count - 1) * physical_short
                    long = available // 64 * 64
                    while (long + 2 * cp - 1) // (2 * cp) * (2 * cp) > available:
                        long -= 64
                    packs.append((long,) + (short,) * (count - 1))
                rows = {}
                for values in packs:
                    core = self.core_predictor(values, cp, ranks=tuple(range(cp)))
                    square = sum((length**2 for length in values)) / capacity**2
                    rows[square] = (float(core['forward_ms']), float(core['backward_ms']))
                levels[count] = tuple(((x, *v) for x, v in sorted(rows.items())))
            self._guide_table[cp] = levels

    def guide_program_costs(self, lengths, cp, *, ranks=None):
        """Approximate FA interpolation plus exact cheap GDN for candidate ordering."""
        return dict(self._guide_program(tuple(lengths), cp))

    @lru_cache(maxsize=65536)
    def _guide_program(self, lengths, cp):
        levels = self._guide_table[cp]
        count = len(lengths)
        low = max((n for n in levels if n <= count))
        high = min((n for n in levels if n >= count))
        square = sum((n * n for n in lengths)) / (cp * 16384) ** 2

        def interpolate(rows, index):
            for lower, upper in zip(rows, rows[1:]):
                if square <= upper[0]:
                    fraction = max(0.0, min(1.0, (square - lower[0]) / (upper[0] - lower[0])))
                    return lower[index] + fraction * (upper[index] - lower[index])
            return rows[-1][index]

        result = self._gdn_costs(lengths, cp)
        for index, phase, key in ((1, 'forward', 'fa_f'), (2, 'backward', 'fa_b')):
            first = interpolate(levels[low], index)
            weight = (count - low) / (high - low) if high != low else 0.0
            second = interpolate(levels[high], index) if high != low else first
            result[key] = first + weight * (second - first) + self.fa_remainder['cost_ms'][phase]
        return result

    @lru_cache(maxsize=65536)
    def _program(self, lengths, cp):
        costs = self._gdn_costs(lengths, cp)
        core = self.core_predictor(lengths, cp, ranks=tuple(range(cp)))
        costs['fa_f'] = float(core['forward_ms']) + self.fa_remainder['cost_ms']['forward']
        costs['fa_b'] = float(core['backward_ms']) + self.fa_remainder['cost_ms']['backward']
        if any((not math.isfinite(v) or v <= 0 for v in costs.values())):
            raise ValueError('invalid predicted mixed-layer cost')
        return costs

    def program_costs(self, lengths, cp, *, ranks=None):
        values = tuple(lengths)
        if not values or len(values) > 64 or any((type(n) is not int or n <= 0 for n in values)):
            raise ValueError('one to 64 positive integer logical lengths required')
        if type(cp) is not int or not 1 <= cp <= 32:
            raise ValueError('the proxy covers CP1..32')
        if ranks is not None:
            ranks = tuple(ranks)
            if (
                len(ranks) != cp
                or any((type(r) is not int for r in ranks))
                or ranks != tuple(range(ranks[0], ranks[0] + cp))
                or (ranks[0] < 0)
                or (ranks[-1] >= 32)
            ):
                raise ValueError('the GB200 proxy requires contiguous physical ranks inside 0..31')
        return dict(self._program(values, cp))

    def forward(self, lengths, cp, *, ranks=None):
        p = self.program_costs(lengths, cp, ranks=ranks)
        return (3 * p['gdn_f'] + 2 * p['fa_f']) / 5

    def backward(self, lengths, cp, *, ranks=None):
        p = self.program_costs(lengths, cp, ranks=ranks)
        return (3 * p['gdn_b'] + 2 * p['fa_b']) / 5

    def timing(self):
        return execution_timing(self.non_attention['cost_ms'], False)

    def clear(self):
        self._program.cache_clear()
        self._guide_program.cache_clear()

    def check_domain(self, geometry, config=None):
        if (
            geometry.world_size,
            geometry.ep_size,
            geometry.tokens_per_rank,
            geometry.dataset_alignment,
        ) != (32, 4, 16384, 64):
            raise ValueError('the composed proxy requires W32/EP4/local16K/dataset64')

    def metadata(self):
        return dict(
            name=self.name,
            source_hashes=self.source_hashes,
            core=getattr(self.core_predictor, 'metadata', lambda: None)(),
            proposal_guide=(
                'frozen-predictor queries indexed by CP, sequence count '
                'and sum of squared lengths; final scores exact'
            ),
            non_attention=self.non_attention,
            fa_remainder=self.fa_remainder,
            gdn=self.gdn.metadata(),
            tail_padding='extend_last',
            dataset_alignment=64,
            provisional=True,
            provisional_reasons=[
                'mixed_eager_native_gdn_and_cuda_graph_nccl_fa',
                'gdn_coefficients_transferred_to_extend_last_padding',
                'measured_fixed_non_attention_transferred_to_variable_packs',
                'cp1_fa_remainder_transferred_to_cp2_32',
                'cp1_remainder_does_not_measure_multicp_contiguous_zigzag_conversion',
            ],
            scope=(
                'attention modules plus measured fixed non-attention; '
                'prediction, not GPU E2E measurement'
            ),
        )

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only attention costs from the selected frozen GB200/GB300 formulas.

Calibration used CUDA Graph/NCCL; transfer to eager/native remains provisional.
This module does not import PyTorch or execute CUDA/communication operations.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

TILE = 128
BLOCK_ROWS = 256
HEAD_DIM_HALF_OCCUPANCY = 256
MIB = float(2**20)
MASKS = ('causal', 'rectangle')
SECTION_MASK = {'diagonal': 'causal', 'lower': 'rectangle', 'upper': 'rectangle'}
DEFAULT_CLOCK_LAW = 'alpha'

PARAMETER_HASHES = {
    'attention/gb200.json': '5becc0bb6bf737b6ab8bfacf41e78930cba0e9c74d9c07c838ac1a3c123629c1',
    'attention/gb300.json': 'c6ea0650b7dcb37e76c4a56452d7eee85e8bef5d985480ad4e09a5c9ad45d73a',
    'qwen35.json': 'c4a93f1a6221c53d68dacf2fb8660f63ec85a3e3aaaabfe1ac0615596fe51ffc',
}


@lru_cache(maxsize=3)
def load_parameters(name: str) -> tuple[dict, str]:
    """Load one immutable runtime parameter file with its pinned SHA256."""
    if name not in PARAMETER_HASHES:
        raise ValueError(f'Unsupported parameter file: {name}')
    raw = (Path(__file__).parent.parent / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PARAMETER_HASHES[name]:
        raise ValueError(f'Frozen runtime parameter hash mismatch: {name}')
    payload = json.loads(raw)
    if payload.get('schema') != 1:
        raise ValueError('Unsupported runtime parameter schema')
    return payload, digest


def chunk_pair(valid_length: int, padded_length: int, cp_size: int, rank: int) -> tuple[int, int]:
    """Valid tokens in a rank's first and mirrored second zigzag chunks."""
    if cp_size <= 0 or not 0 <= rank < cp_size:
        raise ValueError(f'Invalid CP rank {rank} for CP={cp_size}')
    if padded_length < valid_length or padded_length % (2 * cp_size):
        raise ValueError('Padded length must cover valid tokens and be divisible by 2CP')
    chunk = padded_length // (2 * cp_size)
    first = min(max(valid_length - rank * chunk, 0), chunk)
    second_chunk_index = 2 * cp_size - rank - 1
    second = min(max(valid_length - second_chunk_index * chunk, 0), chunk)
    return first, second


def layout_extend_last(lengths, cp_size, storage_tokens, alignment, dataset_alignment=1):
    """Round each length to dataset then CP alignment; extend the last to S x CP."""
    values = [int(v) for v in lengths]
    if not values or min(values) <= 0:
        raise ValueError('lengths must contain positive sequence lengths')
    cp, storage, align = (int(cp_size), int(storage_tokens), int(alignment))
    dataset_align = int(dataset_alignment)
    if cp <= 0 or storage <= 0 or align <= 0 or (dataset_align <= 0):
        raise ValueError('cp_size, storage_tokens and alignment must be positive')
    dataset_padded = [-(-v // dataset_align) * dataset_align for v in values]
    padded = [-(-v // align) * align for v in dataset_padded]
    capacity = storage * cp
    if sum(padded) > capacity:
        raise ValueError(
            f'aligned storage {sum(padded)} exceeds the pack capacity {capacity} '
            f'(storage {storage} tokens per rank x cp {cp}, alignment {align})'
        )
    padded[-1] += capacity - sum(padded)
    if cp > 1 and any((v % (2 * cp) for v in padded)):
        raise ValueError(
            f'padded lengths {padded} are not divisible by 2 x cp = {2 * cp}: '
            f'alignment {align} does not support the zigzag layout at cp {cp}'
        )
    return padded


@lru_cache(maxsize=256)
def _round_pieces(lengths, padded, cp_size):
    if cp_size <= 1 or len(lengths) != len(padded) or (not lengths) or (min(lengths) <= 0):
        raise ValueError('round geometry requires CP > 1 and matching positive lengths')
    chunks = [
        [chunk_pair(n, p, cp_size, rank) for n, p in zip(lengths, padded)]
        for rank in range(cp_size)
    ]
    first = [tuple((a for a, _ in row)) for row in chunks]
    second = [tuple((b for _, b in row)) for row in chunks]
    full = [tuple((a + b for a, b in row)) for row in chunks]
    physical = sum(padded) // cp_size
    out, piece_cache = ([], {})
    for rank in range(cp_size):
        for step in range(cp_size):
            kv_rank = (rank - step) % cp_size
            if step == 0:
                section, q, kv, pq, pk = ('diagonal', full[rank], full[kv_rank], physical, physical)
            elif step <= rank:
                section, q, kv, pq, pk = (
                    'lower',
                    full[rank],
                    first[kv_rank],
                    physical,
                    physical // 2,
                )
            else:
                section, q, kv, pq, pk = (
                    'upper',
                    second[rank],
                    full[kv_rank],
                    physical // 2,
                    physical,
                )
            key = (q, kv)
            pieces = piece_cache.get(key)
            if pieces is None:
                pieces = piece_cache[key] = tuple(
                    ((a, b) for a, b in zip(q, kv) if a > 0 and b > 0)
                )
            out.append(
                dict(
                    rank=rank,
                    step=step,
                    kv_rank=kv_rank,
                    section=section,
                    pieces=pieces,
                    q_physical=pq,
                    kv_physical=pk,
                )
            )
    return tuple(out)


def round_pieces(lengths, padded, cp_size):
    """Valid-token pieces for each rank and causal zigzag ring step."""
    return _round_pieces(
        tuple((int(v) for v in lengths)), tuple((int(v) for v in padded)), int(cp_size)
    )


def ceil_div(a, b):
    return -(-int(a) // int(b))


def tile_of(tokens, tile=TILE):
    """Valid tokens rounded to kernel tiles, with one tile for a nonempty piece."""
    return max(1, ceil_div(tokens, tile))


def slots_for(head_dim, p_hw, half_occupancy_head_dim=HEAD_DIM_HALF_OCCUPANCY):
    """Resident kernel slots; large head dimensions consume twice the resources."""
    if half_occupancy_head_dim is None:
        return int(p_hw)
    return int(p_hw) if int(head_dim) < int(half_occupancy_head_dim) else max(1, int(p_hw) // 2)


def active_pieces(pieces):
    """(q_tokens, kv_tokens) pieces with q > 0 and kv > 0, pack order kept."""
    return [(int(q), int(kv)) for q, kv in pieces if int(q) > 0 and int(kv) > 0]


def units_kv(pieces, mask, q_heads, tile=TILE, block_rows=BLOCK_ROWS):
    """KV tiles per unit in launch order: pack order x heads x Q blocks ascending."""
    if mask not in MASKS:
        raise ValueError(f'mask must be one of {MASKS}, got {mask!r}')
    out = []
    for q, kv in active_pieces(pieces):
        tq, tkv = (tile_of(q, tile), tile_of(kv, tile))
        blocks = ceil_div(tq, block_rows // tile)
        per_piece = (
            [min(2 * b, tq) for b in range(1, blocks + 1)] if mask == 'causal' else [tkv] * blocks
        )
        out.extend(per_piece * int(q_heads))
    return out


def makespan(kv_tiles, P, rho):
    """Greedy kernel-wave duration in rho-plus-tile units, in dispatch order."""
    from .dispatch import dispatch_chain

    if int(P) <= 0:
        raise ValueError('P must be positive')
    return dispatch_chain(kv_tiles, P, rho)[0]


def static_ms(n_active, us_per_slot, thd_max):
    """Fixed static-batch term: each empty slot of a thd_max-sized batch costs us_per_slot."""
    return float(us_per_slot) * 0.001 * (int(thd_max) - int(n_active))


def topology_class_for_cp(topology, cp_size):
    """Recorded communication class, otherwise the calibration's node-size rule."""
    cp = int(cp_size)
    if cp < 2:
        raise ValueError('the ring layer needs cp >= 2')
    classes = topology.get('classes') or {}
    for name, record in classes.items():
        if cp in [int(c) for c in record.get('cps') or []]:
            return (name, record.get('role'))
    by_role = topology.get('class_by_role') or {}
    gpus = int(topology.get('gpus_per_node') or 4)
    role = 'pair' if cp == 2 else 'intra_node' if cp <= gpus else 'inter_node'
    name = by_role.get(role)
    if name is None:
        raise ValueError(f'the topology block has no {role!r} class for cp {cp}')
    return (name, role)


class AttentionModel:
    """Shared frozen-parameter, geometry and physical-layout contract."""

    def __init__(self, hardware, phase):
        self.data, self.parameter_sha256 = load_parameters(f'attention/{hardware}.json')
        if self.data['hardware'] != hardware:
            raise ValueError('Calibration hardware mismatch')
        self.fit = self.data[phase]
        self.phase = phase
        self.data_sha256 = self.data['source_sha256']
        self.head, self.head_status = self.fit['head'], self.fit['head_status']
        self.known_limits = self.fit['known_limits']
        self.topology = self.data['topology']
        self.alignment_default = int(self.data['layout']['alignment_default'])
        self.production_default_clock_mhz = self.data['production_clock_mhz']
        cp1 = self.fit['cp1']
        self.thd_max = int(cp1.get('thd_max', cp1.get('static', {}).get('thd_max', 64)))
        keys = (
            ('q_heads', 'kv_heads', 'head_dim')
            if phase == 'backward'
            else ('query_heads', 'kv_heads', 'head_dim')
        )
        self.geometries = {
            name: tuple(int(g[k]) for k in keys)
            for name, g in cp1.get('geometry_params', cp1['geometries']).items()
        }
        self._profile_by_shape = {shape: name for name, shape in self.geometries.items()}

    def geometry_profile(self, query_heads, kv_heads, head_dim):
        key = int(query_heads), int(kv_heads), int(head_dim)
        if key not in self._profile_by_shape:
            raise ValueError(f'No {self.phase} coefficient set for geometry {key}')
        return self._profile_by_shape[key]

    def structure_for_cp(self, cp):
        """Use the pinned communication class or its recorded topology rule."""
        if cp < 2:
            raise ValueError('The ring layer requires CP >= 2')
        ring = self.fit['ring']
        structure = ring['structure_of_cp'].get(str(cp))
        if structure is None:
            structure, _ = topology_class_for_cp(self.topology, cp)
        if structure not in ring['W0']:
            raise ValueError(f'No communication coefficients for {structure!r}')
        return structure

    def predict_time_ms(
        self,
        lengths,
        cp_size,
        *,
        query_heads,
        kv_heads,
        head_dim,
        storage_tokens,
        sequence_alignment=None,
        dataset_alignment=1,
        clock_law=DEFAULT_CLOCK_LAW,
        use_production_default_clock=False,
    ):
        """Scalar execution of the frozen formula; no prediction trace allocation."""
        if clock_law not in ('alpha', 'two_term'):
            raise ValueError('Unsupported clock input')
        values = tuple(int(n) for n in lengths)
        cp, storage = int(cp_size), int(storage_tokens)
        if len(values) > self.thd_max:
            raise ValueError('Pack exceeds static sequence slots')
        profile = self.geometry_profile(query_heads, kv_heads, head_dim)
        alignment = (
            self.alignment_default if sequence_alignment is None else int(sequence_alignment)
        )
        padded = layout_extend_last(values, cp, storage, alignment, dataset_alignment)
        frequency = (
            float(self.production_default_clock_mhz[self.phase])
            if use_production_default_clock
            else None
        )
        return self._predict(profile, storage, cp, values, padded, frequency, clock_law)


class ForwardModel(AttentionModel):
    """Selected forward CP1 wave and ring formulas; no historical heads or traces."""

    def __init__(self, hardware: str):
        super().__init__(hardware, 'forward')
        fit = self.fit
        cp1, constants = fit['cp1'], self.data['constants']
        self.tile = int(cp1.get('tile', constants['tile_tokens']))
        self.block_rows = int(cp1.get('block_rows', constants['block_rows']))
        if self.block_rows % self.tile:
            raise ValueError('Block rows must be divisible by tile size')
        half = cp1.get('head_dim_half_occupancy', constants['head_dim_half_occupancy'])
        self.half_occupancy_head_dim = None if half is None else int(half)
        self.p_hw = int(cp1['P_hw'])
        self.static_us_per_slot = float(cp1['static_us_per_slot'])
        self.o_graph, self.ring = fit['o_graph'], fit['ring']
        self.ring_kernel_scale = float(fit['kernel_scale'])
        self.clock = fit['clock']
        self.f0_mhz = float(self.clock['f0_mhz'])
        self.alpha = float(self.clock['alpha'])
        self.two_term_s = float(self.clock['two_term_s'])
        self.sequential_pair_structure = self.ring['sequential_pair_structure']
        self.kappa_W0 = float(self.ring['kappa_W0'])
        self.kappa_W0_pair = float(self.ring.get('kappa_W0_sequential_pair', 0.0))
        self._kernel_cache = {}

    def f5(self, profile, mask, pieces, S_tokens, n_static=None):
        """Tile-rounded, cached fused-attention wall at the reference clock."""
        tiles = tuple(
            ((tile_of(q, self.tile), tile_of(kv, self.tile)) for q, kv in active_pieces(pieces))
        )
        return self._f5_tiles(profile, mask, tiles, int(S_tokens), n_static)

    @lru_cache(maxsize=8192)
    def _f5_tiles(self, profile, mask, tiles, S_tokens, n_static):
        q_heads, _kv, head_dim = self.geometries[profile]
        pieces = [(q * self.tile, kv * self.tile) for q, kv in tiles]
        coefficients = self.fit['cp1']['geometries'][profile][mask]
        units = units_kv(pieces, mask, q_heads, self.tile, self.block_rows)
        slots = slots_for(
            head_dim, int(coefficients.get('P_hw', self.p_hw)), self.half_occupancy_head_dim
        )
        span = makespan(units, slots, float(coefficients['rho_kvtiles']))
        k0 = float(coefficients['k0_ms'])
        bS = float(coefficients['b_S_ms_per_token']) * float(S_tokens)
        schedule = float(coefficients['tau_ms_per_kvtile']) * span
        tau = float(coefficients.get('tau_jmax_ms_per_kvtile') or 0.0)
        tail = tau * max(units, default=0) if tau and len(units) > slots else 0.0
        static = static_ms(
            len(pieces) if n_static is None else int(n_static),
            self.static_us_per_slot,
            self.thd_max,
        )
        return k0 + bS + schedule + tail + static

    def o_graph_ms(self, P_q, profile):
        """Per-call overhead from physical Q bytes, excluded from kernel scaling."""
        q_heads, _kv, head_dim = self.geometries[profile]
        mib = float(P_q) * (q_heads * head_dim) * 2.0 / MIB
        return (float(self.o_graph['o0_us']) + float(self.o_graph['o1_us_per_MiB']) * mib) * 0.001

    def ring_terms(self, cp_size, storage_tokens, profile):
        """Communication/glue terms shared by every rank, in milliseconds."""
        cp, S = (int(cp_size), int(storage_tokens))
        q_heads, kv_heads, head_dim = self.geometries[profile]
        structure = self.structure_for_cp(cp)
        w0 = self.ring['W0'][structure]
        payload_bytes = 2 * S * kv_heads * head_dim * 2
        W0_us = float(w0['latency_us']) + float(w0['us_per_byte']) * int(payload_bytes)
        x = S * (q_heads * head_dim)
        mib = x * 2 / MIB
        chain0 = (
            float(self.ring['chain_0_comm']['a0_us'])
            + float(self.ring['chain_0_comm']['a1_us_per_elem']) * x
        )
        gmid = float(self.ring['g_mid']['b0_us']) + float(self.ring['g_mid']['b1_us_per_elem']) * x
        c_tail = float(self.ring['c_tail_us_per_MiB'])
        sequential = structure == self.sequential_pair_structure
        kappa = self.kappa_W0_pair if sequential else self.kappa_W0
        W0_ms = W0_us * 0.001
        stretch_ms = kappa * W0_ms
        return {
            'cp': cp,
            'sequential_pair': sequential,
            'W0_ms': W0_ms,
            'chain0_comm_ms': chain0 * 0.001,
            'gmid_ms': gmid * 0.001,
            'tail_ms': cp * c_tail * mib * 0.001,
            'stretch_ms': stretch_ms,
        }

    def first_exchange(self, terms, k0_ms):
        """Residual first exchange plus preparation for the next ring step."""
        W0 = terms['W0_ms']
        cp = terms['cp']
        if terms['sequential_pair']:
            rem = 0.5 * W0 + max(0.0, 0.5 * W0 - k0_ms)
        else:
            rem = max(0.0, W0 - k0_ms)
        last_round = cp == 2
        P1 = terms['chain0_comm_ms'] if last_round else 2.0 * terms['gmid_ms']
        return rem + P1

    def round_kernel(self, profile, section, pieces, storage_tokens, q_physical):
        """Cached round kernel: CP1-form wall minus physical-Q overhead."""
        mask = SECTION_MASK[section]
        S_tokens = (
            int(q_physical)
            if q_physical
            else int(storage_tokens) // 2 if section == 'upper' else int(storage_tokens)
        )
        key = (profile, mask, tuple(pieces), S_tokens)
        hit = self._kernel_cache.get(key)
        if hit is None:
            raw = self.f5(profile, mask, list(pieces), S_tokens)
            og = self.o_graph_ms(S_tokens, profile)
            hit = raw - og
            if len(self._kernel_cache) > 200000:
                self._kernel_cache.clear()
            self._kernel_cache[key] = hit
        return hit

    def _predict(self, profile, S, cp, values, padded, frequency, clock_law):
        clock_scale = None
        if frequency:
            ratio = self.f0_mhz / frequency
            clock_scale = (
                self.two_term_s * ratio + (1.0 - self.two_term_s)
                if clock_law == 'two_term'
                else ratio**self.alpha
            )
        if cp == 1:
            wall = self.f5(profile, 'causal', [(v, v) for v in values], S)
            og = self.o_graph_ms(S, profile)
            return wall if clock_scale is None else og + (wall - og) * clock_scale
        terms = self.ring_terms(cp, S, profile)
        scale = (
            self.ring_kernel_scale if clock_scale is None else clock_scale * self.ring_kernel_scale
        )
        k = [[0.0] * cp for _ in range(cp)]
        for item in round_pieces(values, padded, cp):
            kr = self.round_kernel(profile, item['section'], item['pieces'], S, item['q_physical'])
            k[item['rank']][item['step']] = kr * scale
        sums = [sum(row) for row in k]
        crit = max(range(cp), key=lambda rank: sums[rank])
        kernel = sums[crit] + cp * terms['stretch_ms']
        glue = terms['chain0_comm_ms'] + (cp - 2) * terms['gmid_ms'] + terms['tail_ms']
        first_exchange = self.first_exchange(terms, k[crit][0] + terms['stretch_ms'])
        return kernel + glue + first_exchange

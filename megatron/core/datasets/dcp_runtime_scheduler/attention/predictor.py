# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only attention costs from the selected frozen GB200/GB300 formulas.

Calibration used CUDA Graph/NCCL; transfer to eager/native remains provisional.
This module does not import PyTorch or execute CUDA/communication operations.
"""

from __future__ import annotations

import hashlib
import heapq
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
    'attention/gb200.json': '7f4ccc3da6665f4eed70e00847fb0f3b874eb44ced8639f3e806854658939a66',
    'attention/gb300.json': '75f8377ad93976a36c08841c5c05bbea195b730a085c5bc0afe08668f2f10197',
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
            f'aligned storage {sum(padded)} exceeds the pack capacity {capacity} (storage {storage} tokens per rank x cp {cp}, alignment {align})'
        )
    padded[-1] += capacity - sum(padded)
    if cp > 1 and any((v % (2 * cp) for v in padded)):
        raise ValueError(
            f'padded lengths {padded} are not divisible by 2 x cp = {2 * cp}: alignment {align} does not support the zigzag layout at cp {cp}'
        )
    return padded


def cp1_pieces(lengths):
    """cp = 1: one causal call over the pack, pieces (L_i, L_i) of valid tokens."""
    return [(int(v), int(v)) for v in lengths]


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


def q_blocks(tq, tile=TILE, block_rows=BLOCK_ROWS):
    """Physical Q blocks covered by a valid-token tile count."""
    return ceil_div(tq, block_rows // tile)


def causal_block_kv(tq, tile=TILE, block_rows=BLOCK_ROWS):
    """KV tiles walked by each forward Q block under the causal mask."""
    tq = int(tq)
    return [min(2 * b, tq) for b in range(1, q_blocks(tq, tile, block_rows) + 1)]


def rect_block_kv(tq, tkv, tile=TILE, block_rows=BLOCK_ROWS):
    """kv tiles walked by every Q block of a no-mask q x kv block: tkv for each of the q_blocks(tq) blocks."""
    return [int(tkv)] * q_blocks(int(tq), tile, block_rows)


def active_pieces(pieces):
    """(q_tokens, kv_tokens) pieces with q > 0 and kv > 0, pack order kept."""
    return [(int(q), int(kv)) for q, kv in pieces if int(q) > 0 and int(kv) > 0]


def units_kv(pieces, mask, q_heads, tile=TILE, block_rows=BLOCK_ROWS):
    """The kernel's units in launch order (pack order x heads x Q blocks ascending): kv tiles per unit."""
    if mask not in MASKS:
        raise ValueError(f'mask must be one of {MASKS}, got {mask!r}')
    out = []
    for q, kv in active_pieces(pieces):
        tq, tkv = (tile_of(q, tile), tile_of(kv, tile))
        per_piece = (
            causal_block_kv(tq, tile, block_rows)
            if mask == 'causal'
            else rect_block_kv(tq, tkv, tile, block_rows)
        )
        for _head in range(int(q_heads)):
            out.extend(per_piece)
    return out


def makespan(kv_tiles, P, rho):
    """Greedy kernel-wave duration in rho-plus-tile units, in dispatch order."""
    from .dispatch import accelerated_chain

    compiled = accelerated_chain(kv_tiles, P, rho)
    if compiled is not None:
        return compiled[0]
    rho = float(rho)
    costs = [rho + float(k) for k in kv_tiles]
    if not costs:
        return 0.0
    P = int(P)
    if P <= 0:
        raise ValueError('P must be positive')
    if len(costs) <= P:
        return max(costs)
    if costs[0] > 0 and all((cost == costs[0] for cost in costs)):
        finish = 0.0
        for _ in range(ceil_div(len(costs), P)):
            finish += costs[0]
        return finish
    heap = [0.0] * P
    end = 0.0
    for cost in costs:
        finish = heap[0] + cost
        heapq.heapreplace(heap, finish)
        if finish > end:
            end = finish
    return end


def j_max(kv_tiles):
    """kv tiles of the longest unit (the J_max of the ``+ jmax`` form)."""
    return max((int(k) for k in kv_tiles), default=0)


def static_ms(n_active, us_per_slot, thd_max):
    """The fixed static-batch term: every empty cu_seqlens slot of the thd_max-sized static batch costs us_per_slot."""
    return float(us_per_slot) * 0.001 * (int(thd_max) - int(n_active))


def f5_wall(
    cset,
    pieces,
    S_tokens,
    mask,
    q_heads,
    head_dim,
    *,
    p_hw,
    static_us_per_slot,
    thd_max,
    n_static=None,
    tile=TILE,
    block_rows=BLOCK_ROWS,
    half_occupancy_head_dim=HEAD_DIM_HALF_OCCUPANCY,
):
    """CP1 wall: k0 + b*S + tau*wave + last-wave tail + static sequence slots."""
    kv = units_kv(pieces, mask, q_heads, tile, block_rows)
    P = slots_for(head_dim, int(cset.get('P_hw', p_hw)), half_occupancy_head_dim)
    span = makespan(kv, P, float(cset['rho_kvtiles']))
    n_active = len(active_pieces(pieces))
    n_st = n_active if n_static is None else int(n_static)
    k0 = float(cset['k0_ms'])
    bS = float(cset['b_S_ms_per_token']) * float(S_tokens)
    sched = float(cset['tau_ms_per_kvtile']) * span
    beyond = len(kv) > P
    jm_tiles = j_max(kv)
    tj = float(cset.get('tau_jmax_ms_per_kvtile') or 0.0)
    jm = tj * jm_tiles if tj and beyond else 0.0
    st = static_ms(n_st, static_us_per_slot, thd_max)
    wall = k0 + bS + sched + jm + st
    return wall


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


class ForwardModel:
    """Selected forward CP1 wave and ring formulas; no historical heads or traces."""

    def __init__(self, hardware: str):
        data, self.parameter_sha256 = load_parameters('attention/' + hardware + '.json')
        if data['hardware'] != hardware:
            raise ValueError('Calibration hardware mismatch')
        fit = data['forward']
        self.hardware, self.direction = hardware, 'forward'
        self.data_sha256 = data['source_sha256']
        self.head, self.head_status = fit['head'], fit['head_status']
        self.known_limits = fit['known_limits']
        self.topology = data['topology']
        cp1, constants = fit['cp1'], data['constants']
        self.tile = int(cp1.get('tile', constants['tile_tokens']))
        self.block_rows = int(cp1.get('block_rows', constants['block_rows']))
        if self.block_rows % self.tile:
            raise ValueError('Block rows must be divisible by tile size')
        half = cp1.get('head_dim_half_occupancy', constants['head_dim_half_occupancy'])
        self.half_occupancy_head_dim = None if half is None else int(half)
        self.p_hw = int(cp1['P_hw'])
        self.static_us_per_slot = float(cp1['static_us_per_slot'])
        self.thd_max = int(cp1['thd_max'])
        self.geometries = cp1['geometries']
        self._profile_by_shape = {
            (int(g['query_heads']), int(g['kv_heads']), int(g['head_dim'])): name
            for name, g in self.geometries.items()
        }
        self.o_graph, self.ring = fit['o_graph'], fit['ring']
        self.ring_kernel_scale = float(fit['kernel_scale'])
        self.clock = fit['clock']
        self.f0_mhz = float(self.clock['f0_mhz'])
        self.alpha = float(self.clock['alpha'])
        self.two_term_s = float(self.clock['two_term_s'])
        self.production_default_clock_mhz = data['production_clock_mhz']
        self.alignment_default = int(data['layout']['alignment_default'])
        self.calibrated_cps = tuple(self.ring['calibrated_cps'])
        self.sequential_pair_structure = self.ring['sequential_pair_structure']
        self.structure_of_cp = {int(k): v for k, v in self.ring['structure_of_cp'].items()}
        self.kappa_W0 = float(self.ring['kappa_W0'])
        self.kappa_W0_pair = float(self.ring.get('kappa_W0_sequential_pair', 0.0))
        self._kernel_cache = {}

    def kernel_scale(self, cp_size, *, law='alpha', use_production_default=False):
        """Frozen production clock scaling; communication terms stay unscaled."""
        if law not in ('alpha', 'two_term'):
            raise ValueError('Unsupported forward clock law')
        if not use_production_default:
            return None, None
        ratio = float(self.f0_mhz) / float(self.production_default_clock_mhz['forward'])
        if law == 'two_term':
            scale = float(self.two_term_s) * ratio + (1.0 - float(self.two_term_s))
        else:
            scale = ratio ** float(self.alpha)
        return scale, None

    def geometry_profile(self, query_heads, kv_heads, head_dim):
        """Name of the calibrated geometry profile for (query_heads, kv_heads, head_dim); ValueError when none exists."""
        key = (int(query_heads), int(kv_heads), int(head_dim))
        name = self._profile_by_shape.get(key)
        if name is None:
            known = ', '.join(
                (
                    f'{n} {q}/{kv}/{d}'
                    for (q, kv, d), n in sorted(
                        self._profile_by_shape.items(), key=lambda kv_: kv_[1]
                    )
                )
            )
            raise ValueError(
                f'no WP-G3 cuda_graph coefficient set for geometry {key[0]}/{key[1]}/{key[2]} on {self.hardware}; calibrated: {known}'
            )
        return name

    def cp1_set(self, profile, mask):
        if mask not in MASKS:
            raise ValueError(f'mask must be one of {MASKS}, got {mask!r}')
        return self.geometries[profile][mask]

    def geometry_params(self, profile):
        entry = self.geometries[profile]
        return (int(entry['query_heads']), int(entry['kv_heads']), int(entry['head_dim']))

    def f5(self, profile, mask, pieces, S_tokens, n_static=None):
        """Tile-rounded, cached fused-attention wall at the reference clock."""
        tiles = tuple(
            ((tile_of(q, self.tile), tile_of(kv, self.tile)) for q, kv in active_pieces(pieces))
        )
        return self._f5_tiles(profile, mask, tiles, int(S_tokens), n_static)

    @lru_cache(maxsize=8192)
    def _f5_tiles(self, profile, mask, tiles, S_tokens, n_static):
        q_heads, _kv, head_dim = self.geometry_params(profile)
        pieces = [(q * self.tile, kv * self.tile) for q, kv in tiles]
        return f5_wall(
            self.cp1_set(profile, mask),
            pieces,
            S_tokens,
            mask,
            q_heads,
            head_dim,
            p_hw=self.p_hw,
            static_us_per_slot=self.static_us_per_slot,
            thd_max=self.thd_max,
            n_static=n_static,
            tile=self.tile,
            block_rows=self.block_rows,
            half_occupancy_head_dim=self.half_occupancy_head_dim,
        )

    def o_graph_ms(self, P_q, profile):
        """Per-call overhead from physical Q bytes, excluded from kernel scaling."""
        q_heads, _kv, head_dim = self.geometry_params(profile)
        mib = float(P_q) * (q_heads * head_dim) * 2.0 / MIB
        return (float(self.o_graph['o0_us']) + float(self.o_graph['o1_us_per_MiB']) * mib) * 0.001

    def structure_for_cp(self, cp_size):
        """Frozen communication structure for an arbitrary CP width."""
        cp = int(cp_size)
        if cp < 2:
            raise ValueError('the ring layer needs cp >= 2')
        structure = self.structure_of_cp.get(cp)
        if structure is None:
            if cp == 2 and self.sequential_pair_structure:
                structure = self.sequential_pair_structure
            elif self.topology:
                structure, _role = topology_class_for_cp(self.topology, cp)
            elif cp <= 4:
                structure = 'cp4_concurrent_intra_node'
            else:
                structure = 'nvl72_domain'
        if structure not in self.ring['W0']:
            raise ValueError(
                f'communication structure {structure!r} has no W0 coefficients on head {self.head!r} ({self.hardware} / {self.direction})'
            )
        return (structure, cp not in self.calibrated_cps)

    def ring_terms(self, cp_size, storage_tokens, profile):
        """Communication/glue terms shared by every rank, in milliseconds."""
        cp, S = (int(cp_size), int(storage_tokens))
        q_heads, kv_heads, head_dim = self.geometry_params(profile)
        structure, _ = self.structure_for_cp(cp)
        w0 = self.ring['W0'][structure]
        payload_bytes = 2 * S * kv_heads * head_dim * 2
        bytes_by_step = [int(payload_bytes)] * int(cp)
        W0_us_by_step = [
            float(w0['latency_us']) + float(w0['us_per_byte']) * b for b in bytes_by_step
        ]
        W0_us = W0_us_by_step[0]
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
        W0_by_step_ms = [w * 0.001 for w in W0_us_by_step]
        stretch_by_step_ms = [kappa * w for w in W0_by_step_ms]
        constant = len(set(bytes_by_step)) == 1
        stretch_total_ms = cp * stretch_by_step_ms[0] if constant else sum(stretch_by_step_ms)
        return {
            'cp': cp,
            'sequential_pair': sequential,
            'W0_ms': W0_ms,
            'chain0_comm_ms': chain0 * 0.001,
            'gmid_ms': gmid * 0.001,
            'tail_ms': cp * c_tail * mib * 0.001,
            'stretch_by_step_ms': stretch_by_step_ms,
            'stretch_total_ms': stretch_total_ms,
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
        """Scalar-only execution of the SAME frozen formula; no trace allocation."""
        cp, S = (int(cp_size), int(storage_tokens))
        values = tuple((int(n) for n in lengths))
        if len(values) > self.thd_max:
            raise ValueError('pack exceeds CUDA-graph static sequence slots')
        profile = self.geometry_profile(query_heads, kv_heads, head_dim)
        alignment = (
            self.alignment_default if sequence_alignment is None else int(sequence_alignment)
        )
        padded = layout_extend_last(values, cp, S, alignment, dataset_alignment)
        clock_scale, _ = self.kernel_scale(
            cp, law=clock_law, use_production_default=use_production_default_clock
        )
        if cp == 1:
            wall = self.f5(profile, 'causal', cp1_pieces(values), S)
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
        kernel = sums[crit] + terms['stretch_total_ms']
        glue = terms['chain0_comm_ms'] + (cp - 2) * terms['gmid_ms'] + terms['tail_ms']
        first_exchange = self.first_exchange(terms, k[crit][0] + terms['stretch_by_step_ms'][0])
        return kernel + glue + first_exchange

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Selected backward CP1, GB200 round-2 and GB300 round-1 CPU formulas."""

from __future__ import annotations

from functools import lru_cache

from .predictor import (
    MIB,
    TILE,
    active_pieces,
    ceil_div,
    layout_extend_last,
    load_parameters,
    round_pieces,
    static_ms,
    tile_of,
    topology_class_for_cp,
)

CP1_FORM = 'F5b-1+jmax'
F0_BWD_MHZ = 1600.0
STATIC_US_PER_SLOT = 0.155
THD_MAX = 64
CALIBRATED_CPS = (2, 4, 8, 16)


def block_tiles(R_bwd) -> int:
    rb = int(R_bwd) // TILE
    if rb < 1 or int(R_bwd) % TILE:
        raise ValueError(f'R_bwd must be a positive multiple of {TILE} rows (got {R_bwd})')
    return rb


def kv_units(pieces, mask, q_heads, R_bwd) -> list:
    """Q tile counts in backward dispatch order: pack, head, then KV block."""
    rb = block_tiles(R_bwd)
    out = []
    for q, kv in active_pieces(pieces):
        tq, tkv = (tile_of(q), tile_of(kv))
        if mask == 'causal':
            per = [max(1, tq - (b - 1) * rb) for b in range(1, ceil_div(tq, rb) + 1)]
        elif mask == 'rectangle':
            per = [tq] * ceil_div(tkv, rb)
        else:
            raise ValueError(mask)
        for _h in range(int(q_heads)):
            out.extend(per)
    return out


@lru_cache(maxsize=1024)
def kernel_dispatch(units, P, rho):
    """Greedy wave duration with the critical slot's unit and tile counts."""
    from .dispatch import dispatch_chain

    return dispatch_chain(units, P, rho)


@lru_cache(maxsize=4096)
def kernel_dispatch_tiles(tiles, mask, q_heads, rows, slots, rho):
    pieces = [(q * TILE, kv * TILE) for q, kv in tiles]
    return kernel_dispatch(tuple(kv_units(pieces, mask, q_heads, rows)), slots, rho)


def table_for(head_dim, table, fallback=None) -> int:
    """Exact head dimension, otherwise the closest smaller table key."""
    tab = {int(k): int(v) for k, v in (table or {}).items() if v is not None}
    hd = int(head_dim)
    if hd in tab:
        return tab[hd]
    if not tab:
        return int(fallback)
    below = [k for k in tab if k <= hd]
    return tab[max(below)] if below else min(tab.values())


def helper_term_eval(block, which, q_tokens, kv_tokens, n, q_heads, kv_heads) -> float:
    """Evaluate a token-linear or head-token-linear helper form."""
    c = block
    form = (
        (c.get('form') if isinstance(c, dict) else None)
        or block.get('form')
        or (
            'head_token'
            if 'h0_us' in (c.get('coef', c) if isinstance(c, dict) else {})
            else 'token'
        )
    )
    c = c.get('coef', c)
    if form == 'head_token':
        heads = q_heads if which == 'H_q' else kv_heads
        if heads is None:
            raise ValueError('the head-token helper form needs the geometry (q_heads / kv_heads)')
        tok = float(q_tokens) if which == 'H_q' else float(kv_tokens)
        return (
            float(c.get('h0_us', 0.0))
            + float(c.get('c_us_per_head_token', 0.0)) * tok * float(heads)
            + float(c.get('c_n_us_per_sequence', 0.0)) * float(n)
        )
    return (
        float(c.get('o0_us', 0.0))
        + float(c.get('o_q_us_per_token', 0.0)) * float(q_tokens)
        + float(c.get('o_kv_us_per_token', 0.0)) * float(kv_tokens)
        + float(c.get('o_n_us_per_sequence', 0.0)) * float(n)
    )


def helper_terms_us(spec_by_hd, head_dim, q_heads, kv_heads, q_tokens, kv_tokens, n) -> tuple:
    """Q and KV helper costs for a geometry, in microseconds."""
    if not spec_by_hd:
        return (0.0, 0.0)
    blk = spec_by_hd.get(int(head_dim)) or spec_by_hd.get(str(int(head_dim)))
    if not blk:
        return (0.0, 0.0)
    return (
        helper_term_eval(blk['H_q'], 'H_q', q_tokens, kv_tokens, n, q_heads, kv_heads),
        helper_term_eval(blk['H_kv'], 'H_kv', q_tokens, kv_tokens, n, q_heads, kv_heads),
    )


def b_kv_bytes(S, kv_heads, head_dim) -> int:
    """One rank's BF16 K and V exchange payload."""
    return 2 * int(S) * int(kv_heads) * int(head_dim) * 2


def mib_q(S, q_heads, head_dim) -> float:
    return float(S) * (int(q_heads) * int(head_dim)) * 2.0 / MIB


def mib_kv(S, kv_heads, head_dim) -> float:
    return b_kv_bytes(S, kv_heads, head_dim) / MIB


def step_bytes(cp, b_kv) -> list:
    """KV first, KV plus dKV in middle steps, and dKV in the last step."""
    cp = int(cp)
    if cp < 2:
        return []
    return [int(b_kv)] + [2 * int(b_kv)] * (cp - 2) + [int(b_kv)]


def step_sizes(section, mq, mkv) -> tuple:
    """Diagonal touches full Q/KV; lower halves KV and upper halves Q."""
    if section == 'diagonal':
        return (mq, mkv)
    if section == 'lower':
        return (mq, mkv / 2.0)
    if section == 'upper':
        return (mq / 2.0, mkv)
    raise ValueError(section)


def w0_us(w0_map, structure, nbytes) -> float:
    w = w0_map[structure]
    return float(w['latency_us']) + float(w['us_per_byte']) * float(nbytes)


def glue_classes_step_us(ring, q_heads, kv_heads, head_dim, st) -> dict:
    """Preparation, active-token helpers, accumulation and copies per ring step."""
    gl = ring['glue']
    pieces = st.get('pieces') or []
    n_act = int(st.get('n_active', len(pieces)) or 0)
    pr = gl['prep']
    prep = (
        float(pr.get('p0_us', 0.0))
        + float(pr.get('p_q_us_per_MiB_q', 0.0)) * float(st['mib_q'])
        + float(pr.get('p_kv_us_per_MiB_kv', 0.0)) * float(st['mib_kv'])
        + float(pr.get('p_n_us_per_piece', 0.0)) * n_act
    )
    pmode = gl.get('prep_mode', 'class_direct')
    if pmode == 'none':
        prep = 0.0
    elif pmode == 'excess_over_cp1':
        cpc = gl.get('cp1_prep_coef') or {}
        q_tok = int(st.get('q_active', sum((q for q, _ in pieces))))
        kv_tok = int(st.get('kv_active', sum((kv for _, kv in pieces))))
        cp1_prep = (
            float(cpc.get('o0_us', 0.0))
            + float(cpc.get('o_q_us_per_token', 0.0)) * q_tok
            + float(cpc.get('o_kv_us_per_token', 0.0)) * kv_tok
            + float(cpc.get('o_n_us_per_sequence', 0.0)) * n_act
        )
        prep = max(0.0, prep - cp1_prep)
    elif pmode != 'class_direct':
        raise ValueError(f'unknown ring prep mode {pmode!r}')
    hmode = (gl.get('helpers') or {}).get('mode', 'physical_rows')
    if hmode == 'active_tokens':
        q_tok = int(st.get('q_active', sum((q for q, _ in pieces))))
        kv_tok = int(st.get('kv_active', sum((kv for _, kv in pieces))))
    else:
        q_tok, kv_tok = (int(st.get('q_physical') or 0), int(st.get('kv_physical') or 0))
    hq, hkv = helper_terms_us(
        (gl.get('helpers') or {}).get('by_head_dim'),
        head_dim,
        q_heads,
        kv_heads,
        q_tok,
        kv_tok,
        n_act,
    )
    ac = gl.get('accumulate') or {}
    acc = float(ac.get('a0_us', 0.0)) + float(ac.get('a_us_per_MiB_kv', 0.0)) * float(st['mib_kv'])
    cp_ = gl.get('copy') or {}
    copy = (
        float(cp_.get('c0_us', 0.0))
        + float(cp_.get('c_us_per_MiB_bytes', 0.0)) * float(st.get('bytes', 0)) / 2**20
    )
    return {
        'prep': prep,
        'helper_q': hq,
        'helper_kv': hkv,
        'accumulate': acc,
        'memset': float(gl.get('memset_us', 0.0)),
        'copy': copy,
    }


def pre_us(ring, S, q_heads, kv_heads, head_dim) -> float:
    p = ring.get('pre') or {}
    if not p.get('served'):
        return 0.0
    return (
        float(p.get('p0_us', 0.0))
        + float(p.get('p_q_us_per_MiB', 0.0)) * mib_q(S, q_heads, head_dim)
        + float(p.get('p_kv_us_per_MiB', 0.0)) * mib_kv(S, kv_heads, head_dim)
    )


class CP1BackwardForm:
    """The served CP1 backward model file (cp1_model_f5b1_jmax_bwd_f0.json): per geometry a causal and a rectangle coefficient set of the form F5b-1+jmax at f0_bwd;
    ``kernel(g, mask, pieces, S_tokens)`` = cp1_makespan_bwd.CP1BwdModel.kernel (the fitted form + the token-form helper terms = the CP1 wall at f0).
    """

    def __init__(self, fit: dict) -> None:
        m = fit['model']
        self.form = str(m['form'])
        if self.form != CP1_FORM:
            raise ValueError(
                f'the vendored CP1 backward model has form {self.form!r}; this composer serves {CP1_FORM!r} (KV-block units + the jmax tail)'
            )
        self.f0_mhz = float(m.get('f0_mhz') or F0_BWD_MHZ)
        self.consts = {
            k: (
                {int(a): int(b) for a, b in v.items()}
                if isinstance(v, dict) and k.endswith('_by_head_dim')
                else v
            )
            for k, v in (m.get('structural') or {}).items()
        }
        self.static_us = float(m.get('static_us_per_slot', STATIC_US_PER_SLOT))
        self.static_spec = m.get('static') or {'form': 'forward_slots'}
        if self.static_spec.get('form', 'forward_slots') != 'forward_slots':
            raise ValueError(
                f"the vendored CP1 backward model serves the static form {self.static_spec.get('form')!r}; this composer serves forward_slots (0.155 us x (64 - n))"
            )
        self.thd_max = int(self.static_spec.get('thd_max', THD_MAX))
        ht = m.get('helper_terms')
        self.helper_terms = (
            {int(k): v for k, v in (ht.get('by_head_dim') or {}).items()}
            if ht and ht.get('applied')
            else None
        )
        self.u_treatment = m.get('u_treatment') or (
            'subtracted_in_fit' if m.get('kernel_excludes_U') else 'inside_k0'
        )
        if self.u_treatment != 'none':
            raise ValueError(
                f"the vendored CP1 backward model has u_treatment {self.u_treatment!r}; the B7.1 served model is 'none' (no O_graph subtraction)"
            )
        self.geoms = {g: dict(v) for g, v in m['geometries'].items()}
        self.geometry_params = {g: dict(v) for g, v in (m.get('geometry_params') or {}).items()}

    def set_for(self, g, mask) -> dict:
        cs = (self.geoms.get(g) or {}).get(mask)
        if not cs or cs.get('status') != 'fitted':
            raise KeyError(f'CP1 backward model {self.form}: no fitted {mask} set for {g!r}')
        return cs

    def params(self, g) -> tuple:
        p = self.geometry_params[g]
        return (int(p['q_heads']), int(p['kv_heads']), int(p['head_dim']))

    def kernel(self, g, mask, pieces, S_tokens, n_static=None) -> dict:
        """Kernel tiles/constants under the structural clock law, plus helpers."""
        cs = self.set_for(g, mask)
        q_heads, kv_heads, head_dim = self.params(g)
        act = active_pieces(pieces)
        n_act = len(act)
        n_st = n_act if n_static is None else int(n_static)
        st = static_ms(n_st, self.static_us, self.thd_max)
        r_tab = {
            int(k): int(v)
            for k, v in (self.consts.get('R_bwd_rows_by_head_dim') or {}).items()
            if v is not None
        }
        R = (
            r_tab[int(head_dim)]
            if int(head_dim) in r_tab
            else int(self.consts.get('R_bwd_rows') or 256)
        )
        P = table_for(head_dim, self.consts.get('P_bwd_by_head_dim'))
        rho = float(cs['rho_tiles'])
        tiles = tuple(((tile_of(q), tile_of(kv)) for q, kv in act))
        _, n_chain, chain_tiles, N, jm = kernel_dispatch_tiles(tiles, mask, q_heads, R, P, rho)
        jmax_val = float(jm) if N > P else 0.0
        k0 = float(cs['k0_ms'])
        bS = float(cs['b_S_ms_per_token']) * float(S_tokens)
        split = {
            'tau_ms_per_tile': (float(chain_tiles), float(rho) * n_chain),
            'tau_jmax_ms_per_tile': (jmax_val, 0.0),
        }
        tile_ms = sum((float(cs[c]) * split[c][0] for c in split if c in cs))
        const_sched_ms = sum((float(cs[c]) * split[c][1] for c in split if c in cs))
        q_tok, kv_tok = (int(sum((q for q, _ in act))), int(sum((kv for _, kv in act))))
        hq, hkv = helper_terms_us(
            self.helper_terms, head_dim, q_heads, kv_heads, q_tok, kv_tok, n_act
        )
        return {
            'helper_ms': (hq + hkv) * 0.001,
            'static_ms': st,
            'tile_ms': tile_ms,
            'const_ms': k0 + bS + const_sched_ms + st,
        }


def law_parts(tile_ms, const_ms, law, f_mhz, f0_mhz=F0_BWD_MHZ):
    """The frozen structural law splits the constant into scaled/unscaled parts."""
    if law['kind'] != 'structural':
        raise ValueError('Only the selected structural backward clock law is supported')
    lam = float(law['lambda'])
    scaled, unscaled = tile_ms + (1.0 - lam) * const_ms, lam * const_ms
    factor = f0_mhz / float(f_mhz) if f_mhz else 1.0
    return {'k_ms': scaled * factor + unscaled}


class BackwardComposer:
    """Shared kernel/glue evaluation with the selected hardware attribution."""

    def __init__(self, fit):
        self.cp1 = CP1BackwardForm({'model': fit['cp1']})
        self.ring = fit['ring']
        self.f0 = self.cp1.f0_mhz
        self.kernel_scale = float(self.ring.get('kernel_scale', 1.0))
        self.law = self.ring['clock']['law']
        self.helpers_follow = bool(self.ring['clock']['helpers_clock_following'])
        self.structure_of_cp = {int(k): v for k, v in self.ring['structure_of_cp'].items()}
        self._cache = {}
        self.shares = {str(g): float(v) for g, v in fit.get('shares', {}).items()}
        self.static_per_call = fit['round'] == 2
        self.g0_by_head_dim = {int(k): float(v) for k, v in fit.get('g0_by_head_dim', {}).items()}

    def kernel_parts(self, g, mask, pieces, S_q):
        """Cache only the kernel quantities consumed by runtime scoring."""
        key = g, mask, tuple(pieces), int(S_q)
        if key not in self._cache:
            k = self.cp1.kernel(g, mask, pieces, int(S_q))
            if len(self._cache) > 200000:
                self._cache.clear()
            self._cache[key] = {
                'tile_ms': float(k['tile_ms']),
                'const_ms': float(k['const_ms']),
                'kernel': k,
            }
        return self._cache[key]

    def _cp1_time_ms(self, g, S, lengths, frequency):
        """CP1 helpers and kernel retain the source's addition order."""
        pieces = [(int(v), int(v)) for v in lengths]
        parts = self.kernel_parts(g, 'causal', pieces, int(S))
        e = self.entry_at(parts, frequency)
        hk = float(parts['kernel'].get('helper_ms', 0.0))
        hf = (
            hk * (self.f0 / frequency if frequency and self.helpers_follow else 1.0)
            if self.helpers_follow
            else 0.0
        )
        wall = hf + e['k_ms'] if self.helpers_follow else hk + e['k_ms']
        return wall + 0.0 if self.static_per_call else wall

    def tail_f0_ms(self, g, S):
        """The selected GB200 attribution has no tail term."""
        return 0.0

    def entry_at(self, parts, f_mhz) -> dict:
        return law_parts(
            parts['tile_ms'] * self.kernel_scale,
            parts['const_ms'] * self.kernel_scale,
            self.law,
            f_mhz,
            self.f0,
        )

    @lru_cache(maxsize=8192)
    def _glue_at(self, g, mq, mkv, n, q_active, kv_active, q_physical, kv_physical, nbytes):
        q_heads, kv_heads, head_dim = self.cp1.params(g)
        return glue_classes_step_us(
            self.ring,
            q_heads,
            kv_heads,
            head_dim,
            dict(
                mib_q=mq,
                mib_kv=mkv,
                n_active=n,
                q_active=q_active,
                kv_active=kv_active,
                q_physical=q_physical,
                kv_physical=kv_physical,
                bytes=nbytes,
            ),
        )

    @lru_cache(maxsize=8192)
    def clock_entry(self, tile_ms, const_ms, frequency):
        return law_parts(tile_ms, const_ms, self.law, frequency, self.f0)

    def structure_for(self, cp, topology=None) -> tuple:
        """Recorded communication structure, otherwise the hardware topology rule."""
        cp = int(cp)
        structure = self.structure_of_cp.get(cp)
        if structure is None:
            if topology:
                structure, _role = topology_class_for_cp(topology, cp)
            else:
                structure = (
                    'cp2_sequential_sendrecv'
                    if cp == 2
                    else 'cp4_concurrent_intra_node' if cp <= 4 else 'nvl72_domain'
                )
        if structure not in self.ring['W0']:
            raise ValueError(
                f'communication structure {structure!r} has no W0 coefficients in the served backward ring block'
            )
        return (structure, cp not in CALIBRATED_CPS)

    def share(self, g) -> float:
        return float(self.shares.get(g, 1.0))

    def static_pack_ms(self, g, n_seq) -> float:
        return static_ms(int(n_seq), self.cp1.static_us, self.cp1.thd_max)

    @lru_cache(maxsize=16384)
    def _step_parts(self, g, section, pieces, q_physical, kv_physical, mq, mkv, nbytes):
        mask = 'causal' if section == 'diagonal' else 'rectangle'
        parts = self.kernel_parts(g, mask, pieces, q_physical)
        static = parts['kernel']['static_ms']
        const = self.share(g) * ((parts['const_ms'] - static) * self.kernel_scale)
        if not self.static_per_call:
            const += static * self.kernel_scale
        classes = self._glue_at(
            g,
            mq,
            mkv,
            len(pieces),
            sum((q for q, _ in pieces)),
            sum((kv for _, kv in pieces)),
            q_physical,
            kv_physical,
            nbytes,
        )
        helpers = classes['helper_q'] + classes['helper_kv']
        rest = classes['prep'] + classes['accumulate'] + classes['memset'] + classes['copy']
        return (parts['tile_ms'] * self.kernel_scale, const, helpers, rest)

    def time_ms(self, g, S, cp, lengths, padded, frequency, topology):
        """GB200 selects the critical rank at f0, then applies the production clock."""
        if cp == 1:
            return self._cp1_time_ms(g, S, lengths, frequency)
        q_heads, kv_heads, head_dim = self.cp1.params(g)
        structure, _ = self.structure_for(cp, topology)
        scale = self.kernel_scale
        per_rank = [[] for _ in range(cp)]
        mq, mkv = (mib_q(S, q_heads, head_dim), mib_kv(S, kv_heads, head_dim))
        sizes = step_bytes(cp, b_kv_bytes(S, kv_heads, head_dim))
        rounds = round_pieces(lengths, padded, cp)
        for rank in range(cp):
            for step in reversed(rounds[rank * cp : (rank + 1) * cp]):
                i = cp - 1 - step['step']
                sq, sk = step_sizes(step['section'], mq, mkv)
                parts = self._step_parts(
                    g,
                    step['section'],
                    step['pieces'],
                    step['q_physical'],
                    step['kv_physical'],
                    sq,
                    sk,
                    sizes[i],
                )
                per_rank[rank].append(parts)
        sums_f0 = [
            sum((tile + const + (helpers + rest) * 0.001 for tile, const, helpers, rest in row))
            for row in per_rank
        ]
        crit = max(range(cp), key=lambda rank: sums_f0[rank])
        row = per_rank[crit]
        kernel = sum(
            (self.clock_entry(tile, const, frequency)['k_ms'] for tile, const, _, _ in row)
        )
        ratio = self.f0 / frequency if frequency else 1.0
        glue = sum(
            (
                helpers * ratio + rest if self.helpers_follow else helpers + rest
                for _, _, helpers, rest in row
            )
        )
        static = self.static_pack_ms(g, len(lengths)) * scale if self.static_per_call else 0.0
        static_at = self.clock_entry(0.0, static, frequency)['k_ms']
        tail_at = self.clock_entry(self.tail_f0_ms(g, S), 0.0, frequency)['k_ms']
        pre = pre_us(self.ring, S, q_heads, kv_heads, head_dim)
        extra = self.g0_by_head_dim[int(head_dim)] * len(sizes)

        return kernel + glue * 0.001 + static_at + tail_at + (pre + extra) * 0.001

    def round1_time_ms(self, g, S, cp, lengths, padded, frequency, topology):
        """GB300: select the critical rank at the production clock, as frozen."""
        if cp == 1:
            return self._cp1_time_ms(g, S, lengths, frequency)
        q_heads, kv_heads, head_dim = self.cp1.params(g)
        structure, _ = self.structure_for(cp, topology)
        mq, mkv = mib_q(S, q_heads, head_dim), mib_kv(S, kv_heads, head_dim)
        sizes = step_bytes(cp, b_kv_bytes(S, kv_heads, head_dim))
        rounds = round_pieces(lengths, padded, cp)
        per_rank = [[] for _ in range(cp)]
        ratio = self.f0 / frequency if frequency else 1.0
        for rank in range(cp):
            for step in reversed(rounds[rank * cp : (rank + 1) * cp]):
                index = cp - 1 - step['step']
                sq, sk = step_sizes(step['section'], mq, mkv)
                mask = 'causal' if step['section'] == 'diagonal' else 'rectangle'
                parts = self.kernel_parts(g, mask, step['pieces'], step['q_physical'])
                entry = self.entry_at(parts, frequency)['k_ms']
                classes = self._glue_at(
                    g,
                    sq,
                    sk,
                    len(step['pieces']),
                    sum(q for q, _ in step['pieces']),
                    sum(kv for _, kv in step['pieces']),
                    step['q_physical'],
                    step['kv_physical'],
                    sizes[index],
                )
                helpers = classes['helper_q'] + classes['helper_kv']
                rest = classes['prep'] + classes['accumulate'] + classes['memset'] + classes['copy']
                glue = helpers * ratio + rest if self.helpers_follow else helpers + rest
                per_rank[rank].append(
                    (entry, glue, w0_us(self.ring['W0'], structure, sizes[index]))
                )
        sums = [sum(k + glue * 1e-3 for k, glue, _ in row) for row in per_rank]
        crit = max(range(cp), key=lambda rank: sums[rank])
        row = per_rank[crit]
        kernel = sum(k for k, _, _ in row)
        glue = sum(glue for _, glue, _ in row)
        w0_sum = sum(w for _, _, w in row)
        extra = float(self.ring['X']['x0_us']) + float(self.ring['X']['kappa']) * w0_sum
        pre = pre_us(self.ring, S, q_heads, kv_heads, head_dim)
        return kernel + glue * 1e-3 + (pre + extra) * 1e-3 + 0.0 * 1e-3


class BackwardModel:
    """Selected backward model; shared runtime parameter loader and scalar API."""

    def __init__(self, hardware: str):
        data, self.parameter_sha256 = load_parameters('attention/' + hardware + '.json')
        fit = data['backward']
        self.data_sha256 = data['source_sha256']
        self.head, self.head_status = fit['head'], fit['head_status']
        self.known_limits = fit['known_limits']
        self.topology = data['topology']
        self.production_default_clock_mhz = data['production_clock_mhz']
        self.alignment_default = int(data['layout']['alignment_default'])
        self.composer = BackwardComposer(fit)
        self.thd_max = self.composer.cp1.thd_max
        self._profile_by_shape = {
            self.composer.cp1.params(g): g for g in self.composer.cp1.geometry_params
        }

    def geometry_profile(self, query_heads: int, kv_heads: int, head_dim: int) -> str:
        """Resolve a calibrated shape; unsupported geometries fail explicitly."""
        key = int(query_heads), int(kv_heads), int(head_dim)
        if key not in self._profile_by_shape:
            raise ValueError(f'No backward coefficient set for geometry {key}')
        return self._profile_by_shape[key]

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
        clock_law='alpha',
        use_production_default_clock=False,
    ):
        """Backward wall time from the selected frozen structural clock law."""
        if clock_law not in ('alpha', 'two_term'):
            raise ValueError('Unsupported clock input')
        values = tuple(int(n) for n in lengths)
        cp, S = int(cp_size), int(storage_tokens)
        if len(values) > self.thd_max:
            raise ValueError('Pack exceeds static sequence slots')
        profile = self.geometry_profile(query_heads, kv_heads, head_dim)
        alignment = (
            self.alignment_default if sequence_alignment is None else int(sequence_alignment)
        )
        padded = layout_extend_last(values, cp, S, alignment, dataset_alignment)
        frequency = (
            float(self.production_default_clock_mhz['backward'])
            if use_production_default_clock
            else None
        )
        method = (
            self.composer.time_ms if self.composer.static_per_call else self.composer.round1_time_ms
        )
        return method(profile, S, cp, values, padded, frequency, self.topology)

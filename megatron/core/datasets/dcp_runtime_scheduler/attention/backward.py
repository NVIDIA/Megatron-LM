# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Frozen CP1 and ring backward costs, with hardware-specific attribution."""

from functools import lru_cache

from .predictor import (
    MIB,
    TILE,
    AttentionModel,
    active_pieces,
    ceil_div,
    round_pieces,
    static_ms,
    tile_of,
)


@lru_cache(maxsize=1024)
def kernel_dispatch(units, slots, rho):
    """Greedy wave duration and the critical slot's integer chain quantities."""
    from .dispatch import dispatch_chain

    return dispatch_chain(units, slots, rho)


@lru_cache(maxsize=4096)
def kernel_dispatch_tiles(tiles, mask, q_heads, rows, slots, rho):
    """Dispatch pack, head, then KV block in the frozen kernel order."""
    block = int(rows) // TILE
    if block < 1 or int(rows) % TILE:
        raise ValueError(f"Backward block rows must be a positive multiple of {TILE}")
    units = []
    for q, kv in tiles:
        if mask == "causal":
            per_head = [max(1, q - (b - 1) * block) for b in range(1, ceil_div(q, block) + 1)]
        elif mask == "rectangle":
            per_head = [q] * ceil_div(kv, block)
        else:
            raise ValueError(mask)
        units.extend(per_head * int(q_heads))
    return kernel_dispatch(tuple(units), slots, rho)


def helper_terms(spec, head_dim, q_heads, kv_heads, q_tokens, kv_tokens, count):
    """Selected token-linear (GB200 CP1) or head-token-linear helper costs."""
    block = (spec or {}).get(str(head_dim)) or (spec or {}).get(head_dim)
    if not block:
        return 0.0, 0.0
    values = []
    for which, heads, tokens in (("H_q", q_heads, q_tokens), ("H_kv", kv_heads, kv_tokens)):
        term = block[which]
        coef = term.get("coef", term)
        if term.get("form", "head_token" if "h0_us" in coef else "token") == "head_token":
            value = (
                float(coef.get("h0_us", 0.0))
                + float(coef.get("c_us_per_head_token", 0.0)) * float(tokens) * float(heads)
                + float(coef.get("c_n_us_per_sequence", 0.0)) * float(count)
            )
        else:
            value = (
                float(coef.get("o0_us", 0.0))
                + float(coef.get("o_q_us_per_token", 0.0)) * float(q_tokens)
                + float(coef.get("o_kv_us_per_token", 0.0)) * float(kv_tokens)
                + float(coef.get("o_n_us_per_sequence", 0.0)) * float(count)
            )
        values.append(value)
    return tuple(values)


class BackwardModel(AttentionModel):
    """One scalar path for both frozen heads; no offline composer/trace layers.

    GB200 chooses the critical rank at the reference clock, then scales it.
    GB300 chooses at the production clock. Keep that distinction and the original
    addition order: changing either can change solver decisions.
    """

    def __init__(self, hardware):
        super().__init__(hardware, "backward")
        self.cp1, self.ring = self.fit["cp1"], self.fit["ring"]
        self.static_per_call = self.fit["round"] == 2
        self.scale = float(self.ring.get("kernel_scale", 1.0))
        self.f0 = float(self.cp1.get("f0_mhz") or 1600.0)
        self.law = self.ring["clock"]["law"]
        self.helpers_follow = bool(self.ring["clock"]["helpers_clock_following"])
        if (
            self.cp1["form"] != "F5b-1+jmax"
            or self.cp1["static"]["form"] != "forward_slots"
            or self.cp1["u_treatment"] != "none"
            or self.law["kind"] != "structural"
            or self.ring["glue"]["prep_mode"] != "class_direct"
            or self.ring["glue"]["helpers"]["mode"] != "active_tokens"
        ):
            raise ValueError("Unsupported frozen backward formula")
        self.static_us = float(self.cp1.get("static_us_per_slot", 0.155))

    @lru_cache(maxsize=8192)
    def _clock(self, tile_ms, const_ms, frequency):
        lam = float(self.law["lambda"])
        scaled, unscaled = tile_ms + (1.0 - lam) * const_ms, lam * const_ms
        factor = self.f0 / float(frequency) if frequency else 1.0
        return scaled * factor + unscaled

    @lru_cache(maxsize=200000)
    def _kernel(self, profile, mask, pieces, storage):
        q_heads, kv_heads, head_dim = self.geometries[profile]
        cs = self.cp1["geometries"][profile][mask]
        if cs["status"] != "fitted":
            raise ValueError(f"No fitted {mask} coefficients for {profile}")
        structural = self.cp1["structural"]
        rows = int(
            structural["R_bwd_rows_by_head_dim"].get(str(head_dim), structural["R_bwd_rows"])
        )
        slots = int(structural["P_bwd_by_head_dim"][str(head_dim)])
        active = active_pieces(pieces)
        tiles = tuple((tile_of(q), tile_of(kv)) for q, kv in active)
        rho = float(cs["rho_tiles"])
        _, chain, chain_tiles, count, longest = kernel_dispatch_tiles(
            tiles, mask, q_heads, rows, slots, rho
        )
        split = {
            "tau_ms_per_tile": (float(chain_tiles), rho * chain),
            "tau_jmax_ms_per_tile": (float(longest) if count > slots else 0.0, 0.0),
        }
        tile_ms = sum(float(cs[c]) * split[c][0] for c in split if c in cs)
        const_sched_ms = sum(float(cs[c]) * split[c][1] for c in split if c in cs)
        static = static_ms(len(active), self.static_us, self.thd_max)
        const_ms = (
            float(cs["k0_ms"])
            + float(cs["b_S_ms_per_token"]) * float(storage)
            + const_sched_ms
            + static
        )
        helper = self.cp1.get("helper_terms") or {}
        hq, hkv = helper_terms(
            helper.get("by_head_dim") if helper.get("applied") else None,
            head_dim,
            q_heads,
            kv_heads,
            sum(q for q, _ in active),
            sum(kv for _, kv in active),
            len(active),
        )
        return tile_ms, const_ms, static, (hq + hkv) * 0.001

    @lru_cache(maxsize=8192)
    def _glue(self, profile, mq, mkv, count, q_tokens, kv_tokens, nbytes):
        q_heads, kv_heads, head_dim = self.geometries[profile]
        glue = self.ring["glue"]
        prep = glue["prep"]
        prep_us = (
            float(prep.get("p0_us", 0.0))
            + float(prep.get("p_q_us_per_MiB_q", 0.0)) * float(mq)
            + float(prep.get("p_kv_us_per_MiB_kv", 0.0)) * float(mkv)
            + float(prep.get("p_n_us_per_piece", 0.0)) * count
        )
        hq, hkv = helper_terms(
            glue["helpers"]["by_head_dim"], head_dim, q_heads, kv_heads, q_tokens, kv_tokens, count
        )
        accumulate = glue.get("accumulate") or {}
        acc_us = float(accumulate.get("a0_us", 0.0)) + float(
            accumulate.get("a_us_per_MiB_kv", 0.0)
        ) * float(mkv)
        copy = glue.get("copy") or {}
        copy_us = (
            float(copy.get("c0_us", 0.0))
            + float(copy.get("c_us_per_MiB_bytes", 0.0)) * float(nbytes) / 2**20
        )
        rest = prep_us + acc_us + float(glue.get("memset_us", 0.0)) + copy_us
        return hq + hkv, rest

    @lru_cache(maxsize=16384)
    def _step(self, profile, section, pieces, q_physical, mq, mkv, nbytes):
        """Reuse clock-independent ring costs across candidate scores and batches."""
        tile, const, static, _ = self._kernel(
            profile, "causal" if section == "diagonal" else "rectangle", pieces, q_physical
        )
        if self.static_per_call:
            const = float(self.fit.get("shares", {}).get(profile, 1.0)) * (
                (const - static) * self.scale
            )
        else:
            const *= self.scale
        helpers, rest = self._glue(
            profile,
            mq / 2.0 if section == "upper" else mq,
            mkv / 2.0 if section == "lower" else mkv,
            len(pieces),
            sum(q for q, _ in pieces),
            sum(kv for _, kv in pieces),
            nbytes,
        )
        return tile * self.scale, const, helpers, rest, nbytes

    def _predict(self, profile, storage, cp, lengths, padded, frequency, clock_law):
        ratio = self.f0 / frequency if frequency else 1.0
        if cp == 1:
            tile, const, _, helper = self._kernel(
                profile, "causal", tuple((v, v) for v in lengths), storage
            )
            kernel = self._clock(tile * self.scale, const * self.scale, frequency)
            wall = helper * ratio + kernel if self.helpers_follow else helper + kernel
            return wall + 0.0 if self.static_per_call else wall
        q_heads, kv_heads, head_dim = self.geometries[profile]
        structure = self.structure_for_cp(cp)
        w0 = self.ring["W0"][structure]
        payload = 2 * int(storage) * int(kv_heads) * int(head_dim) * 2
        sizes = [payload] + [2 * payload] * (cp - 2) + [payload]
        mq = float(storage) * (q_heads * head_dim) * 2.0 / MIB
        mkv = payload / MIB
        rounds = round_pieces(lengths, padded, cp)
        per_rank = []
        for rank in range(cp):
            row = []
            for index, step in enumerate(reversed(rounds[rank * cp : (rank + 1) * cp])):
                row.append(
                    self._step(
                        profile,
                        step["section"],
                        step["pieces"],
                        step["q_physical"],
                        mq,
                        mkv,
                        sizes[index],
                    )
                )
            per_rank.append(row)

        def glue_us(helpers, rest):
            return helpers * ratio + rest if self.helpers_follow else helpers + rest

        if self.static_per_call:
            # GB200's reference-clock choice must precede the clock transformation.
            sums = [
                sum(
                    tile + const + (helpers + rest) * 0.001 for tile, const, helpers, rest, _ in row
                )
                for row in per_rank
            ]
        else:
            sums = [
                sum(
                    self._clock(tile, const, frequency) + glue_us(helpers, rest) * 1e-3
                    for tile, const, helpers, rest, _ in row
                )
                for row in per_rank
            ]
        row = per_rank[max(range(cp), key=lambda rank: sums[rank])]
        kernel = sum(self._clock(tile, const, frequency) for tile, const, _, _, _ in row)
        glue = sum(glue_us(helpers, rest) for _, _, helpers, rest, _ in row)
        pre = self.ring.get("pre") or {}
        pre_us = (
            (
                float(pre.get("p0_us", 0.0))
                + float(pre.get("p_q_us_per_MiB", 0.0)) * mq
                + float(pre.get("p_kv_us_per_MiB", 0.0)) * mkv
            )
            if pre.get("served")
            else 0.0
        )
        if self.static_per_call:
            static = static_ms(len(lengths), self.static_us, self.thd_max) * self.scale
            extra = float(self.fit["g0_by_head_dim"][str(head_dim)]) * len(sizes)
            return (
                kernel
                + glue * 0.001
                + self._clock(0.0, static, frequency)
                + self._clock(0.0, 0.0, frequency)
                + (pre_us + extra) * 0.001
            )
        w0_sum = sum(
            float(w0["latency_us"]) + float(w0["us_per_byte"]) * float(nbytes)
            for _, _, _, _, nbytes in row
        )
        extra = float(self.ring["X"]["x0_us"]) + float(self.ring["X"]["kappa"]) * w0_sum
        return kernel + glue * 1e-3 + (pre_us + extra) * 1e-3 + 0.0 * 1e-3

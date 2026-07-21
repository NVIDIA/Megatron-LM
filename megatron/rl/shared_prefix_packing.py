# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-tensor layout helpers for shared-prefix ("tree") packing of a GRPO group.

Milestone 1 (oracle stage) of the shared-prefix-packing optimization described in
``docs/sequence_packing_prefix_sharing.md``. In a GRPO group of size G the prompt P is
shared by all G completions C_i; today the packer duplicates P into G separate
``[P + C_i]`` blocks. These helpers instead describe a single ``[P, C_1, ..., C_G]``
layout plus:

  * prefix-continued ``position_ids`` (P -> 0..Lp-1, each C_i -> Lp..Lp+Lc_i-1),
  * a tree attention mask (FlexAttention ``mask_mod`` + a dense boolean fallback) where
    each C_i attends to all of P and causally within itself, and never to a sibling C_j,
  * the fanned-out ``(prev_position, target_position)`` index pairs that make the shared
    forward's logprob extraction equivalent to the contiguous ``[P + C_i]`` shift -- in
    particular each branch's FIRST completion token is scored from the shared prefix's
    last-position logit (the same logit for all branches), not from the preceding branch.

NOTHING HERE IS WIRED INTO THE LIVE TRAINING FORWARD. It is exercised by
``mrl_extras/test/test_shared_prefix_equivalence.py`` (the numerical oracle), which must
pass on GPU before the forward integration (Milestone 1b). Keeping these as pure functions
(torch + stdlib only, no megatron-core/TE imports) lets the oracle run standalone.
"""

import contextlib
import types
from dataclasses import dataclass, field
from typing import Callable, List

import torch


@dataclass
class SharedPrefixLayout:
    """Describes one ``[P, C_1, ..., C_G]`` shared-prefix bin (single group).

    All index tensors are 1-D and live on ``device``. ``comp_*`` tensors have one entry
    per completion token, concatenated branch-by-branch in order C_0, C_1, ... .
    """

    prefix_len: int
    completion_lens: List[int]
    total_len: int
    branch_starts: List[int]          # absolute packed start offset of each completion
    position_ids: torch.Tensor        # [total_len]  long; prefix-continued positions
    segment_ids: torch.Tensor         # [total_len]  long; 0 = prefix, i+1 = completion i
    comp_positions: torch.Tensor      # [n_comp_tok] packed index of each completion token
    prev_positions: torch.Tensor      # [n_comp_tok] packed index whose logit predicts it
    branch_of_token: torch.Tensor     # [n_comp_tok] which completion each token belongs to
    n_completion_tokens: int = field(default=0)


def build_shared_prefix_layout(
    prefix_len: int, completion_lens: List[int], device="cpu"
) -> SharedPrefixLayout:
    """Build the layout for a single group from its prefix length and completion lengths.

    The crux is the logprob fan-out: for the first token of each completion (local t == 0)
    the predicting position is the prefix's last token ``Lp - 1`` -- the same shared logit
    for every branch. For t >= 1 it is the immediately preceding (same-branch) token.
    """
    Lp = int(prefix_len)
    Lc = [int(x) for x in completion_lens]
    assert Lp >= 1, "prefix must be non-empty (need a last-prefix logit to score C_i[0])"

    positions: List[int] = list(range(Lp))   # prefix: 0..Lp-1
    segments: List[int] = [0] * Lp            # prefix segment id = 0
    branch_starts: List[int] = []
    comp_positions: List[int] = []
    prev_positions: List[int] = []
    branch_of_token: List[int] = []

    cursor = Lp
    for i, lc in enumerate(Lc):
        branch_starts.append(cursor)
        positions.extend(range(Lp, Lp + lc))  # completion continues from Lp
        segments.extend([i + 1] * lc)
        for t in range(lc):
            p = cursor + t
            comp_positions.append(p)
            prev_positions.append(Lp - 1 if t == 0 else p - 1)
            branch_of_token.append(i)
        cursor += lc

    total = cursor

    def _t(xs):
        return torch.tensor(xs, dtype=torch.long, device=device)

    return SharedPrefixLayout(
        prefix_len=Lp,
        completion_lens=Lc,
        total_len=total,
        branch_starts=branch_starts,
        position_ids=_t(positions),
        segment_ids=_t(segments),
        comp_positions=_t(comp_positions),
        prev_positions=_t(prev_positions),
        branch_of_token=_t(branch_of_token),
        n_completion_tokens=len(comp_positions),
    )


def dense_tree_mask(layout: SharedPrefixLayout, device=None) -> torch.Tensor:
    """Return a ``[total_len, total_len]`` boolean ``allowed[q, k]`` mask.

    ``allowed`` iff k is causally before q (by packed index) AND (k is in the prefix OR k
    is in the same completion branch as q). This is the reference (dense-SDPA) realization
    of the tree mask; the kernel realization is ``flex_mask_mod`` below.
    """
    device = device if device is not None else layout.segment_ids.device
    seg = layout.segment_ids.to(device)
    n = layout.total_len
    idx = torch.arange(n, device=device)
    causal = idx[None, :] <= idx[:, None]                  # [q, k]: k <= q
    k_is_prefix = seg[None, :] == 0                        # [1, k]
    same_branch = seg[None, :] == seg[:, None]             # [q, k]
    return causal & (k_is_prefix | same_branch)


def flex_mask_mod(layout: SharedPrefixLayout) -> Callable:
    """Return a FlexAttention ``mask_mod(b, h, q_idx, kv_idx) -> bool`` for this layout.

    Captures ``segment_ids`` by closure; index it on the same device the kernel runs on.
    Use with ``torch.nn.attention.flex_attention.create_block_mask``.
    """
    seg = layout.segment_ids

    def mask_mod(b, h, q_idx, kv_idx):
        s = seg.to(q_idx.device)
        causal = kv_idx <= q_idx
        k_is_prefix = s[kv_idx] == 0
        same_branch = s[kv_idx] == s[q_idx]
        return causal & (k_is_prefix | same_branch)

    return mask_mod


def build_packed_group(prompt_ids, completion_ids_list, device=None):
    """Pack one GRPO group ``[P, C_1, ..., C_G]`` for a single shared-prefix forward.

    Args:
        prompt_ids: 1-D LongTensor, the shared prompt P.
        completion_ids_list: list of 1-D LongTensors, the per-rollout completions C_i.

    Returns:
        packed_tokens: ``[1, total_len]`` token ids.
        layout: the :class:`SharedPrefixLayout` (carries ``position_ids`` to feed RoPE and
            the fan-out indices for logprob extraction).
        attn_mask: ``[1, 1, total_len, total_len]`` bool tree mask in MEGATRON convention
            (``True`` == masked out), i.e. ``~allowed`` -- pass straight as the model's
            ``attention_mask`` on the dense (non-THD) attention path.

    IMPORTANT: the model must apply RoPE using ``layout.position_ids`` (prefix-continued:
    P -> 0..Lp-1, each C_i -> Lp..Lp+Lc_i-1). Standard Megatron RoPE derives positions from
    the packed sequence length (``get_rotary_seq_len``) and IGNORES ``position_ids``, which
    would give C_i its packed index instead of Lp+t and break equivalence -- making RoPE
    position-aware is the core Milestone-1b integration task.
    """
    device = device if device is not None else prompt_ids.device
    prompt = prompt_ids.to(device)
    comps = [c.to(device) for c in completion_ids_list]
    layout = build_shared_prefix_layout(prompt.numel(), [c.numel() for c in comps], device)
    packed_tokens = torch.cat([prompt] + comps).unsqueeze(0)              # [1, total]
    allowed = dense_tree_mask(layout, device)                            # [T, T] True==allowed
    attn_mask = (~allowed).unsqueeze(0).unsqueeze(0)                     # [1,1,T,T] True==masked
    return packed_tokens, layout, attn_mask


def plan_shared_prefix_bins(groups, bin_size, max_sequences_per_bin=16):
    """Group-aware routing PLAN for shared-prefix packing (model-agnostic, lengths only).

    Decides which GRPO groups become a single shared ``[P, C_1, ..., C_G]`` bin vs fall back
    to today's block-diagonal ``[P + C_i]`` packing (when the group doesn't fit a bin), and
    reports the speedup-observability metrics. This is the data-side routing that the live
    forward integration consumes; it is intentionally torch-free arithmetic so the win is
    measurable (and loggable) before the kernel exists.

    Args:
        groups: list of ``(prefix_len, [completion_len, ...])`` -- one per GRPO prompt; the
            completions in a group share the prompt P.
        bin_size: per-bin token budget (== args.seq_length).
        max_sequences_per_bin: cap on completions packed behind one shared prefix.

    Returns:
        (plan, metrics):
          plan: list of bins. A ``shared`` bin = the whole group as ``[P, C_1..C_G]``; a
            ``blockdiag`` bin = one fallback ``[P + C_i]``.
          metrics: ``shared_prefix/*`` dict (baseline vs effective tokens, dedup fraction,
            predicted linear speedup, prompt fraction f, coverage, group stats).
    """
    plan = []
    baseline_tokens = effective_tokens = 0
    total_unique_tokens = shared_unique_tokens = 0
    shared_groups = 0
    f_sum = 0.0
    G_sum = Lp_sum = 0
    n_groups = len(groups)
    for gi, (Lp, Lcs) in enumerate(groups):
        Lp = int(Lp)
        Lcs = [int(x) for x in Lcs]
        G = len(Lcs)
        sumLc = sum(Lcs)
        grp_baseline = G * Lp + sumLc          # duplicated [P+C_i] token count
        grp_unique = Lp + sumLc                # tokens with the prompt stored once
        baseline_tokens += grp_baseline
        total_unique_tokens += grp_unique
        Lp_sum += Lp
        G_sum += G
        mean_Lc = (sumLc / G) if G else 0
        f_sum += (Lp / (Lp + mean_Lc)) if (Lp + mean_Lc) > 0 else 0.0

        # SUB-GROUPING: a group whose [P + all C_i] exceeds the bin (large prefix and/or long
        # completions, e.g. workplace_assistant) is NOT dropped to full block-diagonal. Instead
        # greedily pack its completions into multiple SHARED sub-bins [P, subset] that each fit
        # (Lp + sum(subset) <= bin_size, |subset| <= max_sequences_per_bin). The prefix is shared
        # within each sub-bin (recomputed once per sub-bin instead of once per completion), so a
        # group splits into ceil-ish sub-bins rather than G block-diagonal sequences. A completion
        # so long that even [P + C_i] alone overflows the bin falls back to block-diagonal.
        cur, cur_tokens, group_emitted_shared = [], Lp, False
        n_shared_subbins = 0

        def _flush_subbin():
            nonlocal cur, cur_tokens, group_emitted_shared, n_shared_subbins, effective_tokens
            if not cur:
                return
            sub_lens = [Lcs[j] for j in cur]
            # a sub-bin with a single completion is just a block-diagonal [P+C_i] -- no actual
            # prefix sharing -- so label it as such (routes to the normal forward, not the
            # shared-prefix two-pass).
            kind = "shared" if len(cur) >= 2 else "blockdiag"
            plan.append({"kind": kind, "group_idx": gi, "prefix_len": Lp,
                         "completion_lens": sub_lens, "completion_indices": list(cur)})
            effective_tokens += Lp + sum(sub_lens)     # prefix stored once for this (sub-)bin
            if kind == "shared":
                group_emitted_shared = True
                n_shared_subbins += 1
            cur, cur_tokens = [], Lp

        for j, Lc in enumerate(Lcs):
            if Lp + Lc > bin_size:                     # this completion can't even share alone
                _flush_subbin()
                plan.append({"kind": "blockdiag", "group_idx": gi, "prefix_len": Lp,
                             "completion_lens": [Lc], "completion_indices": [j]})
                effective_tokens += Lp + Lc            # no dedup for this oversized completion
                continue
            if cur and (cur_tokens + Lc > bin_size or len(cur) >= max_sequences_per_bin):
                _flush_subbin()
            cur.append(j)
            cur_tokens += Lc
        _flush_subbin()

        # a group counts toward "shared coverage" if any sub-bin actually shared a prefix
        # (held >=2 completions).
        if group_emitted_shared:
            shared_groups += 1
            shared_unique_tokens += grp_unique        # group's unique tokens (prefix once)
    metrics = {
        "shared_prefix/baseline_tokens": baseline_tokens,
        "shared_prefix/effective_tokens": effective_tokens,
        "shared_prefix/tokens_saved": baseline_tokens - effective_tokens,
        "shared_prefix/dedup_fraction": (1.0 - effective_tokens / baseline_tokens) if baseline_tokens else 0.0,
        "shared_prefix/predicted_linear_speedup": (baseline_tokens / effective_tokens) if effective_tokens else 1.0,
        "shared_prefix/prompt_fraction_f": (f_sum / n_groups) if n_groups else 0.0,
        "shared_prefix/coverage_groups": (shared_groups / n_groups) if n_groups else 0.0,
        "shared_prefix/coverage_tokens": (shared_unique_tokens / total_unique_tokens) if total_unique_tokens else 0.0,
        "shared_prefix/num_groups": n_groups,
        "shared_prefix/avg_group_size": (G_sum / n_groups) if n_groups else 0.0,
        "shared_prefix/avg_prefix_len": (Lp_sum / n_groups) if n_groups else 0.0,
    }
    return plan, metrics


def positionwise_rotary_emb(rotary_module, position_ids: torch.Tensor) -> torch.Tensor:
    """Per-token RoPE embedding for ARBITRARY (e.g. prefix-continued) positions.

    Megatron's ``RotaryEmbedding`` indexes the rotary table by absolute sequence position
    (``get_emb(max_seq_len)`` -> ``[max_seq_len, 1, 1, dim]``, applied positionally by the
    decoder). For a shared-prefix bin ``[P, C_1, ..., C_G]`` we instead want token i to use
    the rotary for ``position_ids[i]`` (P -> 0..Lp-1, each C_i -> Lp..). This computes the
    table up to ``max(position_ids)+1`` and gathers, returning the same ``[T, 1, 1, dim]``
    shape the decoder expects.
    """
    max_pos = int(position_ids.max().item()) + 1
    emb = rotary_module.get_emb(max_pos)                       # [max_pos, 1, 1, dim]
    return emb.index_select(0, position_ids.to(emb.device))   # [T, 1, 1, dim]


def _find_rotary_module(model):
    """Locate the ``rotary_pos_emb`` module, unwrapping Float16Module/DDP wrappers."""
    m = model
    for _ in range(5):
        rot = getattr(m, 'rotary_pos_emb', None)
        if rot is not None:
            return rot
        m = getattr(m, 'module', None)
        if m is None:
            break
    raise AttributeError("model has no rotary_pos_emb (is RoPE enabled?)")


@contextlib.contextmanager
def rotary_position_aware(model, position_ids: torch.Tensor):
    """Temporarily make the model's RoPE honor ``position_ids`` instead of packed-index.

    Standard Megatron RoPE derives positions from the sequence length and ignores the
    ``position_ids`` argument, which is wrong for a shared-prefix bin (every branch after
    the first would be mis-phased -- see the oracle's ``[rope]`` check). This context
    manager monkeypatches the rotary module's ``forward`` to return
    ``positionwise_rotary_emb(..., position_ids)`` for the duration of one forward, then
    restores the original. Intended for the shared-prefix forward (Milestone 1b); a
    production path would thread positions through instead of patching.
    """
    rot = _find_rotary_module(model)
    pos = position_ids

    def _patched(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        return positionwise_rotary_emb(self, pos)

    had_own = 'forward' in rot.__dict__
    saved = rot.__dict__.get('forward', None)
    object.__setattr__(rot, 'forward', types.MethodType(_patched, rot))
    try:
        yield
    finally:
        if had_own:
            object.__setattr__(rot, 'forward', saved)
        else:
            try:
                object.__delattr__(rot, 'forward')
            except AttributeError:
                pass


@dataclass
class SharedBin:
    """One built shared-prefix bin ``[P, C_1, ..., C_k]`` ready for the packed forward.

    ``tokens``/``position_ids``/``loss_mask`` are padded to ``bin_size`` (the THD packer's bin
    width) so shared bins live in the same [num_bins, bin_size] tensors as block-diagonal bins.
    ``traj_indices`` maps each completion (branch order) back to its global trajectory index, so
    advantages / old- & inference-logprobs can be gathered for the loss. The tree attention mask is
    NOT stored (dense [T,T] is infeasible at 49k); it is built at forward time from ``layout``.
    """

    tokens: 'torch.Tensor'           # [bin_size] padded [P, C_1..C_k]
    position_ids: 'torch.Tensor'     # [bin_size] prefix-continued positions (+ pad)
    loss_mask: 'torch.Tensor'        # [bin_size] 0 on prefix/pad, 1 on completion tokens
    layout: SharedPrefixLayout
    traj_indices: List[int]          # global trajectory index per completion (branch order)
    prefix_len: int
    completion_lens: List[int]


def build_shared_prefix_bins(
    trajs: torch.Tensor,
    generation_masks: torch.Tensor,
    group_ids,
    bin_size: int,
    max_sequences_per_bin: int = 16,
    pad_token: int = 0,
):
    """Construct shared-prefix bins from padded per-trajectory tokens + generation masks.

    Trajectories that share the SAME (group_id, prompt prefix) -- the GRPO group's completions of
    one prompt -- and number >= 2 are packed into one or more ``[P, C_subset]`` bins (sub-grouped
    by ``plan_shared_prefix_bins`` to fit ``bin_size``). The shared prompt P is the trajectory's
    PROMPT (the leading non-generated tokens, identical across the bucket -- NOT the longest common
    prefix, which could reach into the generated region and mis-mask the loss). Everything else --
    singletons, prompt-mismatched, no-generation, non-contiguous-generation, or oversized
    completions -- is returned in ``blockdiag_traj_indices`` for the existing THD packer.

    Args:
        trajs: ``[N, S]`` padded token ids.
        generation_masks: ``[N, S]`` bool; True where the token was generated (gets loss).
        group_ids: length-N int (a global GRPO-group id per trajectory; < 0 == padding/ignore).
        bin_size: per-bin token budget (== seq_length).
        max_sequences_per_bin: cap on completions behind one shared prefix.
        pad_token: id to pad each bin to ``bin_size``.

    Returns:
        (bins, blockdiag_traj_indices): ``bins`` is a list of :class:`SharedBin`;
        ``blockdiag_traj_indices`` is a sorted list of global trajectory indices to pack normally.
    """
    from collections import defaultdict

    device = trajs.device
    N, S = trajs.shape
    group_ids = [int(g) for g in (group_ids.tolist() if torch.is_tensor(group_ids) else group_ids)]

    buckets = defaultdict(list)          # (gid, prompt_tuple) -> [traj_idx]
    prompt_of, comp_of = {}, {}
    blockdiag = []
    for i in range(N):
        gid = group_ids[i]
        if gid < 0:
            blockdiag.append(i)          # padding trajectory (zero-loss) -> normal packer path
            continue
        gm = generation_masks[i].bool()
        nz = torch.nonzero(gm, as_tuple=False)
        if nz.numel() == 0:
            blockdiag.append(i)          # real traj with no generated tokens -> normal path
            continue
        first, last = int(nz[0]), int(nz[-1])
        if (last - first + 1) != int(gm.sum()):
            blockdiag.append(i)          # non-contiguous generation (rare) -> normal path
            continue
        prompt = trajs[i, :first]
        comp_of[i] = trajs[i, first:last + 1]
        prompt_of[i] = prompt
        buckets[(gid, tuple(prompt.tolist()))].append(i)

    bins = []
    for (gid, _prompt_key), idxs in buckets.items():
        if len(idxs) < 2:
            blockdiag.extend(idxs)
            continue
        prompt = prompt_of[idxs[0]]
        Lp = int(prompt.numel())
        Lcs = [int(comp_of[i].numel()) for i in idxs]
        plan, _ = plan_shared_prefix_bins([(Lp, Lcs)], bin_size, max_sequences_per_bin)
        for entry in plan:
            ci = entry["completion_indices"]
            if entry["kind"] != "shared":            # single completion -> normal path
                blockdiag.extend(idxs[j] for j in ci)
                continue
            sub_comps = [comp_of[idxs[j]] for j in ci]
            sub_lcs = [int(c.numel()) for c in sub_comps]
            layout = build_shared_prefix_layout(Lp, sub_lcs, device=device)
            total = layout.total_len
            tokens = torch.full((bin_size,), int(pad_token), dtype=trajs.dtype, device=device)
            tokens[:Lp] = prompt
            cur = Lp
            for c in sub_comps:
                tokens[cur:cur + c.numel()] = c
                cur += c.numel()
            position_ids = torch.zeros(bin_size, dtype=torch.long, device=device)
            position_ids[:total] = layout.position_ids
            loss_mask = torch.zeros(bin_size, dtype=torch.float, device=device)
            loss_mask[Lp:total] = 1.0                # only completion tokens are trained
            bins.append(SharedBin(
                tokens=tokens, position_ids=position_ids, loss_mask=loss_mask, layout=layout,
                traj_indices=[idxs[j] for j in ci], prefix_len=Lp, completion_lens=sub_lcs,
            ))
    return bins, sorted(blockdiag)


def cp_pad_shared_inputs(packed_tokens, position_ids, prefix_len, completion_lens, cp_size, device):
    """Pad a shared-prefix packed sequence so every segment is a multiple of 2*cp (CP zigzag) and
    all completions share a COMMON length (uniform Mamba fork batch). Pads at each segment's END.

    Returns ``(padded_tokens [1,T_pad], padded_positions [1,T_pad], prefix_len_pad, comp_len_pad,
    real_idx [T_real])`` where ``real_idx`` maps each REAL packed position (in real packed order) to
    its index in the padded sequence -- so gathered padded logits ``[T_pad,vocab]`` indexed by
    ``real_idx`` recover the real ``[T_real,vocab]`` for the unchanged ``extract_completion_logprobs``.
    Pad token id 0 / continuation positions (both unused: pads are loss-masked, excluded as attention
    keys, and dropped from the Mamba capture)."""
    cp2 = 2 * cp_size
    Lp_r, lcs_r = int(prefix_len), [int(c) for c in completion_lens]
    Lp_p = ((Lp_r + cp2 - 1) // cp2) * cp2
    Lc_p = (((max(lcs_r) + cp2 - 1) // cp2) * cp2) if lcs_r else 0
    toks = packed_tokens.view(-1)
    seg_starts, c = [0], Lp_r
    for lc in lcs_r:
        seg_starts.append(c); c += lc
    pt, pp, real_idx, opad = [], [], [], 0

    def _emit(real_slice_tokens, real_lo, real_len_, pad_to):
        nonlocal opad
        pt.append(real_slice_tokens)
        pt.append(real_slice_tokens.new_zeros(pad_to - real_len_))
        pp.append(torch.arange(real_lo, real_lo + real_len_, device=device))
        pp.append(torch.arange(real_lo + real_len_, real_lo + pad_to, device=device))
        real_idx.extend(range(opad, opad + real_len_))
        opad += pad_to

    _emit(toks[:Lp_r], 0, Lp_r, Lp_p)                       # prefix -> positions 0..Lp_r-1
    for i, lc in enumerate(lcs_r):
        s = seg_starts[i + 1]
        _emit(toks[s:s + lc], Lp_r, lc, Lc_p)               # completion -> prefix-continued Lp_r..
    padded_tokens = torch.cat(pt).view(1, -1)
    padded_positions = torch.cat(pp).view(1, -1)
    return (padded_tokens, padded_positions, Lp_p, Lc_p,
            torch.tensor(real_idx, dtype=torch.long, device=device))


def extract_completion_logprobs(
    logits: torch.Tensor, packed_tokens: torch.Tensor, layout: SharedPrefixLayout
) -> torch.Tensor:
    """Gather per-completion-token logprobs from a shared-prefix forward.

    Args:
        logits: ``[total_len, vocab]`` (single packed sequence, batch dim already removed).
        packed_tokens: ``[total_len]`` token ids of the packed ``[P, C_1, ..., C_G]`` seq.
        layout: the layout used to build the packed sequence.

    Returns:
        ``[n_completion_tokens]`` logprobs, ordered branch-by-branch (see
        ``layout.branch_of_token``). Each branch's first token is scored from the shared
        ``logits[Lp - 1]``; the rest from the preceding same-branch position.
    """
    logp = torch.log_softmax(logits[layout.prev_positions], dim=-1)   # [n_tok, vocab]
    targets = packed_tokens[layout.comp_positions].unsqueeze(-1)      # [n_tok, 1]
    return logp.gather(-1, targets).squeeze(-1)                        # [n_tok]

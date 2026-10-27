# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""HF ``qwen4_exp`` -> Megatron weight mapping, standalone.

This is a self-contained copy of the mapping, written so that the parity example in this
directory has no dependency outside this repository. It covers exactly the proxy defined in
``proxy_config.py``: 4 HF blocks, no MTP, no vision tower, one PLE layer.

The mapping is also the most useful documentation of how the two implementations line up, so
each rule carries the reason it is not a plain rename.

Layer correspondence: HF block ``i`` -> Megatron layers ``2i`` (attention/GDN, plus PLE where
present) and ``2i+1`` (MoE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

# torch is imported lazily inside the functions that need it: the rule table below is useful
# (and testable) on a machine without torch installed.
if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Rule:
    """One mapping step: ``hf_keys`` combine into ``mcore_keys`` via ``transform``."""

    hf_keys: tuple[str, ...]
    mcore_keys: tuple[str, ...]
    transform: str
    note: str = ""
    params: dict = field(default_factory=dict)


def identity(src: str, dst: str, note: str = "") -> Rule:
    return Rule((src,), (dst,), "identity", note)


def subtract_one(src: str, dst: str, note: str = "") -> Rule:
    return Rule((src,), (dst,), "subtract_one", note)


def concat_rows(srcs: Sequence[str], dst: str, note: str = "") -> Rule:
    return Rule(tuple(srcs), (dst,), "concat_rows", note)


def drop(src: str, note: str = "") -> Rule:
    return Rule((src,), (), "drop", note)


# --------------------------------------------------------------------------------------
# Prime table sizes for the PLE memory
# --------------------------------------------------------------------------------------


def _primes_after(low: int, count: int) -> list[int]:
    """The first ``count`` primes strictly greater than ``low``."""
    out: list[int] = []
    candidate = low + 1
    while len(out) < count:
        if candidate > 2 and candidate % 2 == 0:
            candidate += 1
            continue
        is_prime = candidate >= 2
        divisor = 3
        while divisor * divisor <= candidate:
            if candidate % divisor == 0:
                is_prime = False
                break
            divisor += 2
        if is_prime:
            out.append(candidate)
        candidate += 1
    return out


def ple_head_vocab_sizes(config: dict, ple_layer_index: int = 0) -> list[int]:
    """Per-head prime table sizes, exactly as ``Qwen4ExpTextNGramEmbedding`` allocates them.

    Head ``h`` of PLE layer ``li`` uses the ``(li * heads + h + 1)``-th prime strictly greater
    than ``ngram_vocab_size_base - 1``. The sizes differ per head on purpose -- the hash is
    taken modulo a distinct prime per head -- so they cannot be replaced by one rounded size.
    """
    ngram_size = int(config.get("ngram_size", 3))
    heads_per_ngram = int(config.get("heads_per_ngram", 8))
    base = int(config.get("ngram_vocab_size_base", 20_000_000))
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    first = ple_layer_index * ngram_heads
    return _primes_after(base - 1, first + ngram_heads)[first:]


# --------------------------------------------------------------------------------------
# Rule construction
# --------------------------------------------------------------------------------------


def _gated_residual_rules(hf: str, mcore: str, *, with_write_gate: bool) -> list[Rule]:
    """Read gates (low-rank down/up + norm), and for a layer also the per-stream write gate.

    The model-level mixer has no write gate: HF ends with ``hyper_connection_mixer`` and there
    is **no final RMSNorm** after it. An extra norm there is invisible to module tests and was
    one of the defects this comparison caught.
    """
    rules = [
        identity(f"{hf}.hc_norm.weight", f"{mcore}.hc_norm.weight"),
        identity(f"{hf}.input_mix_weight_down.weight", f"{mcore}.input_mix_weight_down.weight"),
        identity(f"{hf}.input_mix_weight_up.weight", f"{mcore}.input_mix_weight_up.weight"),
    ]
    if with_write_gate:
        rules.append(
            identity(f"{hf}.block_inject_weight.weight", f"{mcore}.block_inject_weight.weight")
        )
    return rules


def _gdn_rules(hf: str, mcore: str) -> list[Rule]:
    return [
        Rule(
            (
                f"{hf}.in_proj_qkv.weight",
                f"{hf}.in_proj_z.weight",
                f"{hf}.in_proj_b.weight",
                f"{hf}.in_proj_a.weight",
            ),
            (f"{mcore}.in_proj.weight",),
            "concat_rows",
            "Megatron fuses [q|k|v|z|beta|alpha] into one in_proj; HF keeps qkv fused and "
            "z/b/a separate.",
        ),
        identity(f"{hf}.conv1d.weight", f"{mcore}.conv1d.weight"),
        identity(f"{hf}.out_proj.weight", f"{mcore}.out_proj.weight"),
        identity(f"{hf}.A_log", f"{mcore}.A_log"),
        identity(f"{hf}.dt_bias", f"{mcore}.dt_bias"),
        subtract_one(
            f"{hf}.norm.weight",
            f"{mcore}.out_norm.weight",
            "Qwen4ExpTextRMSNormGated stores a plain gamma initialised to ones; Megatron's "
            "out_norm is zero-centered like every other norm in this model, so gamma - 1.",
        ),
    ]


def _qsa_rules(hf: str, mcore: str) -> list[Rule]:
    return [
        Rule(
            (f"{hf}.q_proj.weight", f"{hf}.k_proj.weight", f"{hf}.v_proj.weight",
             f"{hf}.q_norm.weight"),
            (f"{mcore}.linear_qkv.weight",),
            "gated_qkv",
            "Megatron interleaves [q, output-gate, k, v] per query group; HF keeps q (with the "
            "gate stacked onto it), k and v separate. q_norm is read only for its head_dim.",
        ),
        # Qwen4ExpTextRMSNorm is zero-centered, out * (1 + w), same convention as Megatron.
        identity(f"{hf}.q_norm.weight", f"{mcore}.q_layernorm.weight"),
        identity(f"{hf}.k_norm.weight", f"{mcore}.k_layernorm.weight"),
        identity(f"{hf}.o_proj.weight", f"{mcore}.linear_proj.weight"),
        identity(f"{hf}.indexer.index_qk_proj.weight", f"{mcore}.indexer.index_qk_proj.weight"),
        identity(f"{hf}.indexer.q_layernorm.weight", f"{mcore}.indexer.q_layernorm.weight"),
        identity(f"{hf}.indexer.k_layernorm.weight", f"{mcore}.indexer.k_layernorm.weight"),
    ]


def _moe_rules(hf: str, mcore: str) -> list[Rule]:
    return [
        identity(f"{hf}.gate.weight", f"{mcore}.router.weight"),
        # 3-D packed expert tensors [E, 2I, H] and [E, H, I]. HF chunks gate_up as [gate | up],
        # which is already Megatron's SwiGLU fc1 layout, so no reordering is needed.
        identity(f"{hf}.experts.gate_up_proj", f"{mcore}.experts.linear_fc1.weight"),
        identity(f"{hf}.experts.down_proj", f"{mcore}.experts.linear_fc2.weight"),
        concat_rows(
            (f"{hf}.shared_expert.gate_proj.weight", f"{hf}.shared_expert.up_proj.weight"),
            f"{mcore}.shared_experts.linear_fc1.weight",
            "The shared expert keeps gate and up separate in HF; Megatron wants them stacked.",
        ),
        identity(f"{hf}.shared_expert.down_proj.weight", f"{mcore}.shared_experts.linear_fc2.weight"),
        identity(f"{hf}.shared_expert_gate.weight", f"{mcore}.shared_experts.gate_weight"),
    ]


def _ple_rules(config: dict, hf: str, mcore: str, *, ple_source: str = "sharded") -> list[Rule]:
    """``ple_source`` selects which form of the n-gram table the HF side is in.

    ``"sharded"`` -- as written to disk. ``save_pretrained`` splits the single table into
    ``split_ngram_parts`` row shards, so a checkpoint on disk has ``ngram_embedding.shard_{k}``.
    ``"single"`` -- as held in memory. The live module owns one ``nn.Embedding``, so its state
    dict and its gradients have one ``ngram_embedding.weight``.

    Both forms map to the same per-head Megatron tables; only the concatenation step differs,
    which is why they share one rule with a different source tuple.
    """
    sizes = ple_head_vocab_sizes(config)
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    shard_count = int(config.get("split_ngram_parts", 4))
    if ple_source == "single":
        table_sources: tuple[str, ...] = (f"{hf}.ple_embedding.ngram_embedding.weight",)
    elif ple_source == "sharded":
        table_sources = tuple(
            f"{hf}.ple_embedding.ngram_embedding.shard_{s}.weight" for s in range(shard_count)
        )
    else:
        raise ValueError(f"ple_source must be 'sharded' or 'single', got {ple_source!r}")
    rules = [
        identity(f"{hf}.key_proj.weight", f"{mcore}.key_projection.weight"),
        identity(f"{hf}.value_proj.weight", f"{mcore}.value_projection.weight"),
        identity(f"{hf}.norm_key.weight", f"{mcore}.key_norm.weight"),
        identity(f"{hf}.norm_query.weight", f"{mcore}.query_norm.weight"),
        identity(f"{hf}.norm_conv.weight", f"{mcore}.conv_norm.weight"),
        identity(f"{hf}.conv1d.weight", f"{mcore}.short_conv.weight"),
        Rule(
            table_sources,
            tuple(f"{mcore}.embedding.tables.{i}.weight" for i in range(len(sizes))),
            "concat_rows_slice",
            "HF stores one padded table split into row shards, with per-head offsets; Megatron "
            "stores one table per hash head, order-major (all bigram heads, then trigram). Rows "
            "past the last head are HF padding and are dropped.",
            {"row_ranges": [[o, o + s] for o, s in zip(offsets, sizes)]},
        ),
    ]
    # Derived tensors HF saves for convenience; Megatron recomputes them from the config.
    rules += [
        drop(f"{hf}.ple_embedding.layer_multipliers", "recomputed by Megatron"),
        drop(f"{hf}.ple_embedding.ngram_heads_offsets", "recomputed by Megatron"),
        drop(f"{hf}.ple_embedding.ngram_heads_vocab_sizes", "recomputed by Megatron"),
    ]
    return rules


def build_rules(config: dict, *, ple_source: str = "sharded") -> list[Rule]:
    """The complete HF -> Megatron rule list for the proxy described by ``config``.

    ``ple_source`` is ``"sharded"`` for a checkpoint on disk and ``"single"`` for a live
    module's state dict or gradients; see :func:`_ple_rules`.
    """
    prefix = "model"
    layer_types = list(config["layer_types"])
    ple_blocks = {int(i) - 1 for i in config.get("ple_layer_ids", [])}

    rules: list[Rule] = [
        identity(f"{prefix}.embed_tokens.weight", "module.embedding.word_embeddings.weight"),
        identity("lm_head.weight", "module.output_layer.weight"),
    ]
    rules += _gated_residual_rules(
        f"{prefix}.hyper_connection_mixer", "module.decoder.hc_exit_contract",
        with_write_gate=False,
    )

    for block, kind in enumerate(layer_types):
        attn_layer = f"module.decoder.layers.{2 * block}"
        moe_layer = f"module.decoder.layers.{2 * block + 1}"
        hf_block = f"{prefix}.layers.{block}"

        rules += _gated_residual_rules(
            f"{hf_block}.attn_hyper_connection", f"{attn_layer}.hyper_connection",
            with_write_gate=True,
        )
        inner = f"{attn_layer}.inner_layer.self_attention"
        if kind == "linear_attention":
            rules += _gdn_rules(f"{hf_block}.linear_attn", inner)
        else:
            rules += _qsa_rules(f"{hf_block}.self_attn", inner)
        if block in ple_blocks:
            rules += _ple_rules(
                config, f"{hf_block}.ple", f"{attn_layer}.inner_layer.engram",
                ple_source=ple_source,
            )

        rules += _gated_residual_rules(
            f"{hf_block}.mlp_hyper_connection", f"{moe_layer}.hyper_connection",
            with_write_gate=True,
        )
        rules += _moe_rules(f"{hf_block}.mlp", f"{moe_layer}.inner_layer.mlp")

    return rules


# --------------------------------------------------------------------------------------
# Normalising a saved checkpoint into the released layout
# --------------------------------------------------------------------------------------


def repack_experts(state: dict, num_experts: int) -> dict:
    """Fold per-expert tensors into the packed layout the released checkpoints use.

    The two layouts both occur in the wild and the rule table targets the released one:

      released / packed   ``mlp.experts.gate_up_proj``  [E, 2I, H]
                          ``mlp.experts.down_proj``     [E, H, I]
      transformers 5.x    ``mlp.experts.{e}.gate_proj.weight`` [I, H], ``up_proj``, ``down_proj``

    ``save_pretrained`` writes the second form, so a fixture produced by round-tripping a model
    through disk needs this before conversion. Returns a new dict; input is left alone.
    """
    import torch

    out = dict(state)
    prefixes = sorted(
        {k.rsplit(".experts.", 1)[0] for k in state if ".experts." in k and k.split(".experts.")[1][0].isdigit()}
    )
    for prefix in prefixes:
        gate, up, down = [], [], []
        for e in range(num_experts):
            base = f"{prefix}.experts.{e}"
            gate.append(out.pop(f"{base}.gate_proj.weight"))
            up.append(out.pop(f"{base}.up_proj.weight"))
            down.append(out.pop(f"{base}.down_proj.weight"))
        out[f"{prefix}.experts.gate_up_proj"] = torch.stack(
            [torch.cat([g, u], dim=0) for g, u in zip(gate, up)], dim=0
        )
        out[f"{prefix}.experts.down_proj"] = torch.stack(down, dim=0)
    return out


# --------------------------------------------------------------------------------------
# Applying the rules
# --------------------------------------------------------------------------------------


def _gated_qkv(q, k, v, head_dim: int):
    """Interleave [q, gate, k, v] per query group, Megatron's ``linear_qkv`` layout.

    HF's ``q_proj`` holds the query and its output gate stacked per head, so its row count is
    ``2 * num_query_heads * head_dim``.
    """
    import torch

    num_groups = k.shape[0] // head_dim
    num_q_heads = q.shape[0] // (2 * head_dim)
    per_group = num_q_heads // num_groups
    trailing = tuple(q.shape[1:])

    q_and_gate = q.reshape(num_q_heads, 2, head_dim, *trailing)
    q_rows, gate_rows = q_and_gate[:, 0], q_and_gate[:, 1]
    k_rows = k.reshape(num_groups, head_dim, *trailing)
    v_rows = v.reshape(num_groups, head_dim, *trailing)

    ordered = []
    for g in range(num_groups):
        lo, hi = g * per_group, (g + 1) * per_group
        ordered += [
            q_rows[lo:hi].reshape(-1, *trailing),
            gate_rows[lo:hi].reshape(-1, *trailing),
            k_rows[g].reshape(-1, *trailing),
            v_rows[g].reshape(-1, *trailing),
        ]
    return torch.cat(ordered, dim=0).contiguous()


def apply_rules(
    rules: Sequence[Rule],
    hf_state: dict,
    *,
    head_dim: int,
    on_missing: Callable[[str], None] | None = None,
    for_gradients: bool = False,
) -> dict:
    """Turn an HF state dict into a Megatron one. Raises on any unconsumed HF key.

    ``for_gradients=True`` maps gradients rather than weights. The only rule that changes is
    ``subtract_one``: the weight relation is ``w_mcore = w_hf - 1``, and a constant offset does
    not survive differentiation, so the gradients are simply equal. Applying the offset to a
    gradient produces a clean 1.0 discrepancy on exactly the zero-centered norms -- which is
    what it looks like when this is got wrong.
    """
    import torch

    out: dict = {}
    consumed: set[str] = set()

    for rule in rules:
        missing = [k for k in rule.hf_keys if k not in hf_state]
        if missing:
            if on_missing is not None:
                for key in missing:
                    on_missing(key)
                continue
            raise KeyError(f"HF state is missing {missing} for rule -> {rule.mcore_keys}")
        consumed.update(rule.hf_keys)
        tensors = [hf_state[k] for k in rule.hf_keys]

        if rule.transform == "drop":
            continue
        if rule.transform == "identity":
            out[rule.mcore_keys[0]] = tensors[0].clone()
        elif rule.transform == "subtract_one":
            out[rule.mcore_keys[0]] = (
                tensors[0].clone() if for_gradients else tensors[0].clone() - 1.0
            )
        elif rule.transform == "concat_rows":
            out[rule.mcore_keys[0]] = torch.cat(tensors, dim=0).contiguous()
        elif rule.transform == "gated_qkv":
            q, k, v, q_norm = tensors
            out[rule.mcore_keys[0]] = _gated_qkv(q, k, v, int(q_norm.numel()))
        elif rule.transform == "concat_rows_slice":
            table = torch.cat(tensors, dim=0)
            for dst, (lo, hi) in zip(rule.mcore_keys, rule.params["row_ranges"]):
                if hi > table.shape[0]:
                    raise ValueError(
                        f"row range [{lo}, {hi}) for {dst} exceeds the concatenated table "
                        f"({table.shape[0]} rows)"
                    )
                out[dst] = table[lo:hi].clone()
        else:
            raise ValueError(f"unknown transform {rule.transform!r}")

    unconsumed = sorted(set(hf_state) - consumed)
    if unconsumed:
        raise KeyError(
            f"{len(unconsumed)} HF tensors were not consumed by any rule, which means the "
            f"mapping is incomplete: {unconsumed[:8]}"
        )
    return out

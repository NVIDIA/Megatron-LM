# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Architectural equivalence test: ``examples/nemotron3/nano.sh`` ↔ ``examples/nemotron3/nano.py``.

Both files encode Nemotron-3 Nano. ``nano.sh`` was emitted by the
Megatron-Bridge `translate_mlm_to_bridge.py --reverse` converter and
expresses the model through legacy `pretrain_hybrid.py` CLI flags;
``nano.py`` expresses it through the Python DSL that ``--model-recipe``
consumes. This test parses ``nano.sh``, runs Megatron's argparse +
``core_transformer_config_from_args`` to reconstruct the legacy-path
:class:`TransformerConfig`, then compiles the DSL recipe and asserts
the two paths agree on every architectural field they can both express.

Three known classes of difference are explicitly tolerated, with reasons:

1. ``LEGACY_CLI_CANT_EXPRESS`` — fields whose canonical value nano.py
   carries but the legacy MLM CLI surface in this checkout has no
   argparse flag for, or the Bridge converter doesn't map. nano.sh
   defaults these; nano.py overrides; the divergence is irreducible
   today.

2. ``CONVERTER_BUG_PATCHES`` — flags the converter should emit but
   doesn't (or emits in a malformed shape). The test patches the
   namespace from canonical values so the rest of the comparison can
   proceed; each patch carries a one-line explanation.

3. ``NARGS_PLUS_FIELDS`` — argparse parses some flags as
   one-element lists (``nargs='+'``); the recipe stores scalars.
   Same value, different container — unwrapped before comparison.

Training-orchestration flags from ``nano.sh`` (``--lr``, ``--global-batch-size``,
``--save-interval``, optimizer / scheduler / DDP / dataset / logger) are
intentionally NOT compared — those are launcher concerns, not part of
the model recipe.
"""

import argparse
import warnings
from pathlib import Path

import pytest

NANO_SH = (
    Path(__file__).resolve().parents[4] / "examples" / "nemotron3" / "nano.sh"
)


# ──────────────────────────────────────────────────────────────────────────
# Known-divergence registries (all real; document why each item is here)
# ──────────────────────────────────────────────────────────────────────────

#: Fields whose canonical value nano.py carries but nano.sh cannot.
LEGACY_CLI_CANT_EXPRESS: dict[str, str] = {
    "first_last_layers_bf16": (
        "No --first-last-layers-bf16 flag in this MLM checkout; "
        "nano.py carries the canonical value via CommonLayerConfig."
    ),
    "use_fused_weighted_squared_relu": (
        "Bridge converter does not emit --use-fused-weighted-squared-relu "
        "even though the flag exists in MLM; nano.py carries the canonical "
        "value on the MoE layer config."
    ),
    "is_hybrid_model": (
        "Set implicitly by --hybrid-layer-pattern in production validate_args(); "
        "this test bypasses validate_args, so the legacy side reads as the TC "
        "default (False) while the recipe forces True in compile()."
    ),
}


#: Flags the Bridge converter should emit but doesn't, or emits malformed.
#: We patch the parsed namespace from canonical values so the rest of
#: the architectural comparison can proceed. Each entry: namespace attr →
#: (value, reason).
CONVERTER_BUG_PATCHES: dict[str, tuple[object, str]] = {
    "hybrid_layer_pattern": (
        "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        "Bridge converter omits --hybrid-layer-pattern for hybrid models",
    ),
    "mamba_num_heads": (
        64,
        "Bridge converter omits --mamba-num-heads despite the flag existing "
        "in MLM. Without it, nano.sh as-emitted would silently auto-derive "
        "mamba_num_heads = hidden_size * expand // mamba_head_dim = 84, "
        "i.e. it would actually train a different model than the canonical.",
    ),
}


#: nano.sh has --attention-backend ``AttnBackend.fused`` (verbatim Python
#: ``repr()`` of the enum); the argparse type lambda expects the bare name.
ATTENTION_BACKEND_REPR_PATCHES: dict[str, str] = {
    "AttnBackend.fused": "fused",
    "AttnBackend.flash": "flash",
    "AttnBackend.unfused": "unfused",
    "AttnBackend.local": "local",
    "AttnBackend.auto": "auto",
}


#: Argparse flags with ``nargs='+'``: parsed as singleton lists; the recipe
#: stores scalars. Same value semantically.
NARGS_PLUS_FIELDS: tuple[str, ...] = (
    "moe_router_load_balancing_type",
    "moe_aux_loss_coeff",
)


#: Architectural TransformerConfig fields applied uniformly to every per-layer
#: TC and the stack-level TC. Compared between the legacy global TC and the
#: recipe's stack-level TC.
MODEL_WIDE_FIELDS: tuple[str, ...] = (
    # Topology
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    # Norms / bias / activation
    "normalization",
    "layernorm_epsilon",
    "add_bias_linear",
    "gated_linear_unit",
    # Init
    "init_method_std",
    # Numerical-stability flags
    "first_last_layers_bf16",
    # Fusions
    "apply_rope_fusion",
    "persist_layer_norm",
    # Hybrid / model family
    "is_hybrid_model",
    "transformer_impl",
    # Parallelism
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "sequence_parallel",
    # Mixed precision
    "bf16",
    "fp16",
    # Cross-entropy fusion (stack-only on the recipe side)
    "cross_entropy_loss_fusion",
    "cross_entropy_fusion_impl",
)


# ──────────────────────────────────────────────────────────────────────────
# nano.sh parsing
# ──────────────────────────────────────────────────────────────────────────


def _extract_args_from_sh(sh_path: Path) -> list[str]:
    """Return the flat list of ``--*`` flags + values found in nano.sh."""
    args: list[str] = []
    for raw in sh_path.read_text().splitlines():
        line = raw.strip().rstrip("\\").strip()
        if not line.startswith("--"):
            continue
        parts = line.split(maxsplit=1)
        args.append(parts[0])
        if len(parts) == 2:
            args.append(parts[1])
    return args


def _patch_known_converter_bugs(args: list[str]) -> list[str]:
    """Convert ``AttnBackend.fused``-style enum reprs to bare names."""
    return [ATTENTION_BACKEND_REPR_PATCHES.get(a, a) for a in args]


# ──────────────────────────────────────────────────────────────────────────
# Legacy-path compilation
# ──────────────────────────────────────────────────────────────────────────


def _legacy_path_from_sh(sh_path: Path):
    """Run nano.sh's args through Megatron's argparse and return:

    - ``ns``: the parsed ``argparse.Namespace``  (with converter-gap patches)
    - ``legacy_tc``: the :class:`TransformerConfig` core_transformer_config_from_args produces
    - ``layer_type_list``: the decoded layer_type_list (single-char per layer)
    """
    import torch

    from megatron.core.models.hybrid.hybrid_layer_allocation import (
        parse_hybrid_pattern,
        validate_segment_layers,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.training.arguments import (
        add_megatron_arguments,
        core_transformer_config_from_args,
    )

    args_list = _patch_known_converter_bugs(_extract_args_from_sh(sh_path))

    parser = argparse.ArgumentParser(description="nano.sh-equivalence", allow_abbrev=False)
    add_megatron_arguments(parser)
    ns, unknown = parser.parse_known_args(args_list)
    if unknown:
        warnings.warn(
            "nano.sh contains flags not registered in this Megatron-LM's argparse "
            "(Bridge↔MLM version drift): " + " ".join(unknown),
            stacklevel=2,
        )

    # Apply the small subset of validate_args() logic the recipe relies on,
    # without invoking distributed-init checks that need a real world size.
    if not getattr(ns, "padded_vocab_size", None):
        ns.padded_vocab_size = 131072  # = 1024 × 128, what nano.py hardcodes
    if not getattr(ns, "params_dtype", None):
        if getattr(ns, "bf16", False):
            ns.params_dtype = torch.bfloat16
        elif getattr(ns, "fp16", False):
            ns.params_dtype = torch.float16
        else:
            ns.params_dtype = torch.float32
    if not getattr(ns, "activation_func", None) and getattr(ns, "squared_relu", False):
        from megatron.core.activations import squared_relu

        ns.activation_func = squared_relu

    # Patch converter-gap fields the Bridge converter forgot to emit.
    for attr, (value, reason) in CONVERTER_BUG_PATCHES.items():
        if not getattr(ns, attr, None):
            setattr(ns, attr, value)

    legacy_tc = core_transformer_config_from_args(ns, TransformerConfig)

    parsed = parse_hybrid_pattern(ns.hybrid_layer_pattern)
    # Bypass select_pipeline_segment: that helper logs through a distributed
    # collective which would require ``torch.distributed`` to be initialised.
    # pp_size=1 (the test's degenerate case) is just "every char becomes a
    # layer type".
    layer_type_list = validate_segment_layers(parsed.main_pattern or "")
    return ns, legacy_tc, layer_type_list


def _recipe_compiled():
    from examples.nemotron3.nano import make_recipe

    return make_recipe().compile()


# ──────────────────────────────────────────────────────────────────────────
# Field comparison helpers
# ──────────────────────────────────────────────────────────────────────────


_MISSING = object()


def _normalise(field_name: str, value):
    """Unwrap singleton-list values for ``nargs='+'`` fields so they
    compare equal to the recipe's scalar form."""
    if field_name in NARGS_PLUS_FIELDS and isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _eq(legacy_val, recipe_val) -> bool:
    """Equality with one tolerance: ``activation_func`` is a callable; compare
    by ``__name__`` since the recipe resolves a string to the same function
    the legacy path imports."""
    if callable(legacy_val) and callable(recipe_val):
        return getattr(legacy_val, "__name__", None) == getattr(recipe_val, "__name__", None)
    return legacy_val == recipe_val


def _diff_fields(legacy_tc, candidate_tc, fields, label: str) -> list[str]:
    diffs = []
    for f in fields:
        if f in LEGACY_CLI_CANT_EXPRESS:
            continue
        legacy_val = _normalise(f, getattr(legacy_tc, f, _MISSING))
        recipe_val = getattr(candidate_tc, f, _MISSING)
        if legacy_val is _MISSING and recipe_val is _MISSING:
            continue
        if not _eq(legacy_val, recipe_val):
            diffs.append(
                f"  [{label}] {f}: legacy={legacy_val!r}  recipe={recipe_val!r}"
            )
    return diffs


def _first_layer_of_type(layer_type_list, layer_config_list, sym):
    for i, s in enumerate(layer_type_list):
        if s == sym:
            return layer_config_list[i]
    raise AssertionError(f"no layer of type {sym!r} in pattern")


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _no_tc_post_init():
    """Disable :meth:`TransformerConfig.__post_init__` for the test module.

    Both the legacy-path and recipe-path TC instantiations bypass the
    surrounding ``validate_args`` / launcher logic that normally enforces
    cross-field invariants (e.g. ``bias_activation_fusion`` being
    incompatible with ``squared_relu``). We only care about whether the two
    sides write identical values into TC's fields.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    original = TransformerConfig.__post_init__
    TransformerConfig.__post_init__ = lambda self: None
    try:
        yield
    finally:
        TransformerConfig.__post_init__ = original


@pytest.fixture(scope="module")
def legacy(_no_tc_post_init):
    return _legacy_path_from_sh(NANO_SH)


@pytest.fixture(scope="module")
def recipe(_no_tc_post_init):
    return _recipe_compiled()


@pytest.mark.internal
class TestNanoArchitecturalEquivalence:
    """nano.sh and nano.py must encode the same model architecture.

    Differences listed in ``LEGACY_CLI_CANT_EXPRESS`` are tolerated (they
    are unavoidable today). A new divergence appearing outside that list is
    a real drift between the two artifacts and fails the test.
    """

    # === Layer pattern ===

    def test_layer_type_list_matches(self, legacy, recipe):
        _, _, legacy_layer_types = legacy
        assert "".join(recipe.layer_type_list) == "".join(legacy_layer_types)

    def test_layer_count_matches(self, legacy, recipe):
        _, _, legacy_layer_types = legacy
        assert len(recipe.layer_config_list) == len(legacy_layer_types)

    # === Recipe-level scalars ===

    def test_vocab_size_matches(self, legacy, recipe):
        ns, _, _ = legacy
        assert recipe.vocab_size == ns.padded_vocab_size

    def test_max_sequence_length_matches(self, legacy, recipe):
        ns, _, _ = legacy
        assert recipe.max_sequence_length == ns.max_position_embeddings

    def test_position_embedding_type_matches(self, legacy, recipe):
        ns, _, _ = legacy
        assert recipe.position_embedding_type == ns.position_embedding_type

    def test_share_embeddings_and_output_weights_matches(self, legacy, recipe):
        ns, _, _ = legacy
        assert recipe.share_embeddings_and_output_weights == (
            not ns.untie_embeddings_and_output_weights
        )

    def test_mtp_matches(self, legacy, recipe):
        ns, _, _ = legacy
        legacy_mtp_depths = getattr(ns, "mtp_num_layers", 0) or 0
        assert recipe.mtp_num_depths == legacy_mtp_depths
        if legacy_mtp_depths == 0:
            assert recipe.mtp_layer_pattern is None

    # === Stack-level (model-wide) TransformerConfig ===

    def test_stack_tc_fields_match(self, legacy, recipe):
        _, legacy_tc, _ = legacy
        diffs = _diff_fields(legacy_tc, recipe.config, MODEL_WIDE_FIELDS, "stack")
        assert not diffs, "stack-level TC divergences:\n" + "\n".join(diffs)

    # === Per-layer-type architectural fields (compared on a representative
    # layer of each type — comparing every per-layer TC against the legacy
    # global TC would be apples-to-oranges since non-matching layer types
    # carry placeholders, not real values).

    def test_first_mamba_layer_matches(self, legacy, recipe):
        _, legacy_tc, _ = legacy
        mamba_tc = _first_layer_of_type(
            recipe.layer_type_list, recipe.layer_config_list, "M"
        )
        diffs = _diff_fields(
            legacy_tc, mamba_tc,
            ("mamba_num_heads", "mamba_head_dim", "mamba_state_dim", "mamba_num_groups"),
            "mamba_layer",
        )
        assert not diffs, "Mamba layer TC divergences:\n" + "\n".join(diffs)

    def test_first_attention_layer_matches(self, legacy, recipe):
        _, legacy_tc, _ = legacy
        att_tc = _first_layer_of_type(
            recipe.layer_type_list, recipe.layer_config_list, "*"
        )
        diffs = _diff_fields(
            legacy_tc, att_tc,
            (
                "num_attention_heads",
                "num_query_groups",
                "kv_channels",
                "attention_softmax_in_fp32",
                "apply_query_key_layer_scaling",
                "masked_softmax_fusion",
            ),
            "attention_layer",
        )
        assert not diffs, "Attention layer TC divergences:\n" + "\n".join(diffs)

    def test_first_moe_layer_matches(self, legacy, recipe):
        _, legacy_tc, _ = legacy
        moe_tc = _first_layer_of_type(
            recipe.layer_type_list, recipe.layer_config_list, "E"
        )
        diffs = _diff_fields(
            legacy_tc, moe_tc,
            (
                "num_moe_experts",
                "moe_router_topk",
                "moe_ffn_hidden_size",
                "moe_shared_expert_intermediate_size",
                "moe_router_score_function",
                "moe_router_load_balancing_type",
                "moe_router_topk_scaling_factor",
                "moe_router_enable_expert_bias",
                "moe_router_dtype",
                "moe_router_num_groups",
                "moe_router_group_topk",
                "moe_aux_loss_coeff",
                "moe_grouped_gemm",
                "moe_token_dispatcher_type",
                "moe_permute_fusion",
                "use_fused_weighted_squared_relu",
            ),
            "moe_layer",
        )
        assert not diffs, "MoE layer TC divergences:\n" + "\n".join(diffs)

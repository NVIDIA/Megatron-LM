from __future__ import annotations

from typing import Any


def add_engram_args(parser):
    """Register optional Engram arguments for GPT entrypoints."""

    group = parser.add_argument_group(title="engram")
    group.add_argument(
        "--use-engram",
        action="store_true",
        help="Enable the Engram-augmented GPT model path.",
    )
    group.add_argument(
        "--engram-vocab-size",
        nargs="+",
        type=int,
        default=None,
        help="Per-n-gram Engram vocab sizes for n-grams 2..max-ngram-size.",
    )
    group.add_argument(
        "--engram-max-ngram-size",
        type=int,
        default=None,
        help="Maximum n-gram order used by Engram hashing.",
    )
    group.add_argument(
        "--engram-n-embed-per-ngram",
        type=int,
        default=None,
        help="Total embedding width allocated to each n-gram order.",
    )
    group.add_argument(
        "--engram-n-head-per-ngram",
        type=int,
        default=None,
        help="Number of hash heads allocated to each n-gram order.",
    )
    group.add_argument(
        "--engram-layer-ids",
        nargs="+",
        type=int,
        default=None,
        help="1-based transformer layer indices that should include Engram.",
    )
    group.add_argument(
        "--engram-pad-id",
        type=int,
        default=None,
        help="Tokenizer pad token id used when hashing shifted n-grams.",
    )
    group.add_argument(
        "--engram-seed",
        type=int,
        default=None,
        help="Seed used to generate deterministic per-layer n-gram hash multipliers.",
    )
    group.add_argument(
        "--engram-kernel-size",
        type=int,
        default=None,
        help="Kernel size for Engram's short causal convolution.",
    )
    group.add_argument(
        "--engram-hc-mult",
        type=int,
        default=None,
        help="Number of Engram gating slots per hidden state.",
    )
    group.add_argument(
        "--engram-tokenizer-name-or-path",
        type=str,
        default=None,
        help="Tokenizer name or path used by Engram's compressed tokenizer. "
        "Defaults to --tokenizer-model when available.",
    )
    return parser


def build_engram_config_from_args(args: Any):
    """Build an EngramConfig from parsed CLI args."""

    from megatron.core.models.engram.engram_module import EngramConfig

    defaults = EngramConfig()
    tokenizer_name_or_path = (
        getattr(args, "engram_tokenizer_name_or_path", None)
        or getattr(args, "tokenizer_model", None)
        or defaults.tokenizer_name_or_path
    )

    return EngramConfig(
        engram_vocab_size=(
            list(getattr(args, "engram_vocab_size", None))
            if getattr(args, "engram_vocab_size", None) is not None
            else list(defaults.engram_vocab_size)
        ),
        max_ngram_size=getattr(args, "engram_max_ngram_size", None) or defaults.max_ngram_size,
        n_embed_per_ngram=(
            getattr(args, "engram_n_embed_per_ngram", None) or defaults.n_embed_per_ngram
        ),
        n_head_per_ngram=(
            getattr(args, "engram_n_head_per_ngram", None) or defaults.n_head_per_ngram
        ),
        engram_layer_ids=(
            list(getattr(args, "engram_layer_ids", None))
            if getattr(args, "engram_layer_ids", None) is not None
            else list(defaults.engram_layer_ids)
        ),
        pad_id=getattr(args, "engram_pad_id", None)
        if getattr(args, "engram_pad_id", None) is not None
        else defaults.pad_id,
        seed=getattr(args, "engram_seed", None)
        if getattr(args, "engram_seed", None) is not None
        else defaults.seed,
        kernel_size=getattr(args, "engram_kernel_size", None) or defaults.kernel_size,
        hc_mult=getattr(args, "engram_hc_mult", None) or defaults.hc_mult,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )


def _wrap_layer_spec_with_engram(layer_spec, engram_config, engram_vocab_size_across_layers):
    from megatron.core.models.engram.engram_model import EngramTransformerLayer
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if not isinstance(layer_spec, ModuleSpec):
        raise TypeError(
            "Engram only supports GPT transformer layer specs expressed as ModuleSpec instances."
        )
    if not isinstance(layer_spec.module, type) or not issubclass(layer_spec.module, TransformerLayer):
        raise TypeError(
            "Engram currently supports GPT specs built from TransformerLayer-based modules only."
        )

    params = dict(layer_spec.params)
    params["engram_config"] = engram_config
    params["engram_vocab_size_across_layers"] = engram_vocab_size_across_layers

    return ModuleSpec(
        module=EngramTransformerLayer,
        params=params,
        submodules=layer_spec.submodules,
        metainfo=dict(layer_spec.metainfo),
    )


def apply_engram_to_transformer_layer_spec(transformer_layer_spec, engram_config):
    """Wrap a GPT transformer layer spec so it instantiates Engram layers."""

    from megatron.core.models.engram.engram_module import calculate_engram_vocab_size_across_layers
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

    engram_vocab_size_across_layers = calculate_engram_vocab_size_across_layers(engram_config)

    if isinstance(transformer_layer_spec, ModuleSpec):
        return _wrap_layer_spec_with_engram(
            transformer_layer_spec, engram_config, engram_vocab_size_across_layers
        )

    if isinstance(transformer_layer_spec, TransformerBlockSubmodules):
        return TransformerBlockSubmodules(
            layer_specs=[
                _wrap_layer_spec_with_engram(
                    layer_spec, engram_config, engram_vocab_size_across_layers
                )
                for layer_spec in transformer_layer_spec.layer_specs
            ],
            layer_norm=transformer_layer_spec.layer_norm,
        )

    raise TypeError(
        "Engram currently supports transformer layer specs as either ModuleSpec or "
        "TransformerBlockSubmodules."
    )

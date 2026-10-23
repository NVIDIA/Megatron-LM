# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Construct configs to exercise CP validation, including configured CP1 with Dynamic CP."""

from contextlib import nullcontext

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig


def _make_config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        max_seqlen_per_dp_cp_rank=128,
    )
    variant = overrides.get("experimental_attention_variant")
    if variant in ("gdn", "gdn2", "kda"):
        kwargs.update(
            linear_attention_freq=1,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
        )
    if variant == "dsa":
        kwargs.update(add_bias_linear=False, dsa_kernel_backend="none")
    if overrides.get("dynamic_context_parallel"):
        # NOTE(nightly-sync 26_10_2026): main's DCP V2 (ea99fdfce) added a new invariant to
        # TransformerConfig.__post_init__ -- `assert self.calculate_per_token_loss,
        # "Dynamic context parallelism must be used with calculate_per_token_loss"` -- because
        # a loss normalized by a static token count is wrong once the CP group size varies per
        # microbatch. It is a genuine main correctness guard (main-only, pinned by main's own
        # test_transformer_config.py::test_dynamic_cp_rejects_unsupported_config[overrides2]),
        # not main lacking a dev feature, so adopt it and satisfy it in dev's configs. The
        # fixture above already sets this flag explicitly; tests that call _make_config
        # directly with dynamic CP (test_head_divisibility_uses_configured_cp) did not, and
        # tripped the new assertion before reaching the behaviour they pin. Set as a default
        # rather than forced, so an explicit override still wins and main's guard stays
        # testable.
        kwargs.setdefault("calculate_per_token_loss", True)
    kwargs.update(overrides)
    config_type = (
        MLATransformerConfig
        if variant == "dsa" or kwargs.get("multi_latent_attention")
        else TransformerConfig
    )
    return config_type(**kwargs)


@pytest.mark.parametrize("max_cp_size", [0, -1, 3, 6, True, 4.0])
def test_max_cp_size_requires_positive_integer_power_of_two(max_cp_size):
    with pytest.raises(ValueError, match="max_cp_size must be a positive power of two"):
        ModelParallelConfig(dynamic_context_parallel=True, max_cp_size=max_cp_size)


@pytest.mark.parametrize("static_cp", [1, 2, 8])
@pytest.mark.parametrize("max_cp_size", [1, 2, 4])
def test_dynamic_max_cp_size_does_not_change_static_cp(static_cp, max_cp_size):
    config = _make_config(
        dynamic_context_parallel=True, context_parallel_size=static_cp, max_cp_size=max_cp_size
    )
    assert config.max_cp_size == max_cp_size
    assert config.context_parallel_size == static_cp


@pytest.mark.parametrize("static_cp", [1, 3, 8])
@pytest.mark.parametrize("supplied_max", [None, 0, -1, 3, 4, True, 4.0])
def test_static_max_cp_size_is_recorded_without_validation(static_cp, supplied_max):
    config = ModelParallelConfig(context_parallel_size=static_cp, max_cp_size=supplied_max)
    assert config.context_parallel_size == static_cp
    assert config.max_cp_size == static_cp
    config.__post_init__()
    assert config.max_cp_size == static_cp


def test_dynamic_max_cp_size_cannot_be_less_than_minimum():
    with pytest.raises(ValueError, match="must be <= max_cp_size"):
        ModelParallelConfig(
            dynamic_context_parallel=True, min_dynamic_context_parallel_size=4, max_cp_size=2
        )


@pytest.mark.parametrize("variant", ["gdn", "kda"])
def test_headwise_dynamic_cp_checks_max_instead_of_static_cp(variant):
    config = _make_config(
        dynamic_context_parallel=True,
        context_parallel_size=8,
        max_cp_size=2,
        tensor_model_parallel_size=2,
        experimental_attention_variant=variant,
        linear_attention_freq=2,
        linear_cp_mode="headwise",
        cp_comm_type="p2p",
    )
    assert config.context_parallel_size == 8
    assert config.max_cp_size == 2


@pytest.mark.parametrize("variant", ["gdn", "kda"])
def test_headwise_dynamic_cp_rejects_incompatible_max_even_with_p2p(variant):
    with pytest.raises(ValueError, match="independently of static CP"):
        _make_config(
            dynamic_context_parallel=True,
            max_cp_size=4,
            tensor_model_parallel_size=2,
            experimental_attention_variant=variant,
            linear_attention_freq=2,
            linear_cp_mode="headwise",
            cp_comm_type="p2p",
        )


def test_chunkwise_dynamic_cp_does_not_require_head_divisibility_by_max():
    config = _make_config(
        dynamic_context_parallel=True,
        max_cp_size=8,
        experimental_attention_variant="gdn",
        linear_attention_freq=2,
        linear_cp_mode="chunkwise",
        linear_num_key_heads=1,
        linear_num_value_heads=1,
    )
    assert config.max_cp_size == 8


@pytest.mark.parametrize("dynamic_cp", [False, True])
def test_unset_max_preserves_existing_headwise_static_validation(dynamic_cp):
    with pytest.raises(AssertionError, match="linear_head_parallel_size"):
        _make_config(
            dynamic_context_parallel=dynamic_cp,
            context_parallel_size=8,
            experimental_attention_variant="gdn",
            linear_attention_freq=2,
            linear_cp_mode="headwise",
        )


@pytest.mark.parametrize("pattern,is_hybrid", [(1, False), ([0, 0], False), (None, True)])
def test_dynamic_max_defers_unknown_or_unused_gdn_layout_to_actual_layers(pattern, is_hybrid):
    config = _make_config(
        dynamic_context_parallel=True,
        context_parallel_size=8,
        max_cp_size=8,
        experimental_attention_variant="gdn",
        linear_attention_freq=pattern,
        is_hybrid_model=is_hybrid,
        linear_cp_mode="headwise",
        cp_comm_type="p2p",
    )
    assert config.max_cp_size == 8


def test_dynamic_max_does_not_reject_hierarchical_type_on_a_linear_layer():
    config = _make_config(
        dynamic_context_parallel=True,
        max_cp_size=2,
        experimental_attention_variant="gdn",
        linear_attention_freq=[1, 0],
        linear_cp_mode="headwise",
        cp_comm_type=["a2a+p2p", "p2p"],
    )
    assert config.max_cp_size == 2


@pytest.fixture(
    params=[
        pytest.param(
            dict(context_parallel_size=2, sequence_packing_scheduler="dp_balanced"), id="static-cp2"
        ),
        pytest.param(
            dict(
                context_parallel_size=1,
                dynamic_context_parallel=True,
                # Required by TransformerConfig whenever dynamic_context_parallel is set:
                # loss scaling cannot use the static TP x CP group once CP groups are
                # picked per microbatch. These tests only assert CP-layout compatibility,
                # so the flag is incidental to what they cover.
                calculate_per_token_loss=True,
            ),
            id="dynamic-cp1",
        ),
    ]
)
def cp_config(request):
    return request.param


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        pytest.param(dict(cp_partition_mode="zigzag"), None, id="standard-zigzag"),
        pytest.param(
            dict(cp_partition_mode="contiguous", experimental_attention_variant="gdn"),
            None,
            id="gdn-contiguous-chunkwise",
        ),
        pytest.param(
            dict(cp_partition_mode="contiguous"),
            "requires experimental_attention_variant",
            id="standard-contiguous",
        ),
        pytest.param(
            dict(cp_partition_mode="contiguous", multi_latent_attention=True),
            "not supported with multi_latent_attention",
            id="mla-contiguous",
        ),
        pytest.param(
            dict(
                cp_partition_mode="contiguous",
                experimental_attention_variant="gdn",
                linear_cp_mode="headwise",
            ),
            "incompatible with GDN-family linear_cp_mode='headwise'",
            id="gdn-contiguous-headwise",
        ),
        pytest.param(
            dict(cp_partition_mode="contiguous", experimental_attention_variant="gdn2"),
            "incompatible with GDN-family linear_cp_mode='headwise'",
            id="gdn2-contiguous",
        ),
        pytest.param(
            dict(cp_partition_mode="zigzag", experimental_attention_variant="dsv4_hybrid"),
            "requires cp_partition_mode='contiguous'",
            id="dsv4-zigzag",
        ),
    ],
)
def test_cp_layout_compatibility(cp_config, overrides, error):
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        _make_config(**cp_config, **overrides)


@pytest.mark.parametrize("variant", ["gdn", "kda"])
@pytest.mark.parametrize("linear_cp_mode", ["headwise", "chunkwise", "invalid", None])
def test_gdn_cp_mode(cp_config, variant, linear_cp_mode):
    error = linear_cp_mode not in ("headwise", "chunkwise", None)
    with (
        pytest.raises(ValueError, match="linear_cp_mode must be either") if error else nullcontext()
    ):
        config = _make_config(
            **cp_config, experimental_attention_variant=variant, linear_cp_mode=linear_cp_mode
        )
        assert config.linear_cp_mode == (linear_cp_mode or "chunkwise")


@pytest.mark.parametrize("variant", ["gdn", "kda"])
@pytest.mark.parametrize("linear_cp_mode", ["headwise", "chunkwise"])
@pytest.mark.parametrize("alignment", [None, 16])
def test_gdn_cp_conv_padding(cp_config, variant, linear_cp_mode, alignment):
    error = alignment is not None and linear_cp_mode == "chunkwise"
    with (
        pytest.raises(AssertionError, match="gdn_conv_pad_alignment is incompatible")
        if error
        else nullcontext()
    ):
        _make_config(
            **cp_config,
            experimental_attention_variant=variant,
            linear_cp_mode=linear_cp_mode,
            gdn_conv_pad_alignment=alignment,
        )


@pytest.mark.parametrize("linear_cp_mode", ["headwise", "chunkwise", None])
def test_gdn2_cp_support(cp_config, linear_cp_mode):
    """GDN2 supports static headwise CP, not chunkwise or per-microbatch CP groups."""
    if linear_cp_mode == "chunkwise":
        error = "GDN2 context parallelism requires linear_cp_mode='headwise'"
    elif cp_config.get("dynamic_context_parallel", False):
        error = "does not support dynamic_context_parallel"
    else:
        error = None
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        config = _make_config(
            **cp_config, experimental_attention_variant="gdn2", linear_cp_mode=linear_cp_mode
        )
        assert config.linear_cp_mode == (linear_cp_mode or "headwise")


@pytest.mark.parametrize(
    ("variant", "linear_cp_mode", "key_heads", "value_heads", "head_count_error"),
    [
        # GDN2 rejects chunkwise mode before checking head counts. Its supported headwise
        # path still requires TP1/CP2 head counts divisible by 2, including a positive case.
        ("gdn2", "chunkwise", 1, 2, True),
        ("gdn2", "headwise", 1, 2, True),
        ("gdn2", "chunkwise", 2, 4, False),
        ("gdn2", "headwise", 2, 4, False),
        # GDN and KDA chunkwise CP keep every TP-local head on each CP rank.
        ("gdn", "chunkwise", 1, 2, False),
        ("kda", "chunkwise", 1, 1, False),
        ("gdn", "headwise", 1, 2, True),
    ],
)
def test_gdn2_head_divisibility_uses_tp_times_cp(
    variant, linear_cp_mode, key_heads, value_heads, head_count_error
):
    if variant == "gdn2" and linear_cp_mode == "chunkwise":
        expectation = pytest.raises(ValueError, match="GDN2 context parallelism requires")
    elif head_count_error:
        expectation = pytest.raises(
            AssertionError, match="must be a multiple of linear_head_parallel_size"
        )
    else:
        expectation = nullcontext()
    with expectation:
        _make_config(
            context_parallel_size=2,
            experimental_attention_variant=variant,
            linear_cp_mode=linear_cp_mode,
            linear_num_key_heads=key_heads,
            linear_num_value_heads=value_heads,
        )


@pytest.mark.parametrize(
    ("cp_comm_type", "error"),
    [
        ("all_gather", False),
        ("allgather", False),
        ("ALL_GATHER", False),
        (["all_gather", "allgather"], False),
        (None, True),
        ("p2p", True),
        (["all_gather", "p2p"], True),
    ],
)
def test_dsa_cp_requires_allgather(cp_config, cp_comm_type, error):
    with pytest.raises(AssertionError, match="allgather only") if error else nullcontext():
        _make_config(**cp_config, experimental_attention_variant="dsa", cp_comm_type=cp_comm_type)


@pytest.mark.parametrize(
    ("cp_comm_type", "error"),
    [
        (None, None),
        ("p2p", None),
        (["p2p", "all_gather"], None),
        (["all_gather"], "Length of cp_comm_type"),
        ([], "Length of cp_comm_type"),
        (1, "Unsupported communication type"),
        (("p2p", "all_gather"), "Unsupported communication type"),
    ],
)
def test_cp_communication_type_and_layer_count(cp_config, cp_comm_type, error):
    with pytest.raises(AssertionError, match=error) if error else nullcontext():
        _make_config(**cp_config, cp_comm_type=cp_comm_type)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(dict(fallback_to_eager_attn=True), id="eager"),
        pytest.param(dict(transformer_impl="local"), id="local"),
    ],
)
@pytest.mark.parametrize(
    ("cp_comm_type", "error"),
    [
        (None, False),
        ("all_gather", False),
        (["all_gather", "all_gather"], False),
        ("p2p", True),
        ("allgather", True),
        (["all_gather", "p2p"], True),
    ],
)
def test_eager_and_local_cp_require_all_gather(cp_config, backend, cp_comm_type, error):
    with (
        pytest.raises(ValueError, match="only supports all_gather communication type")
        if error
        else nullcontext()
    ):
        _make_config(**cp_config, **backend, cp_comm_type=cp_comm_type)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        pytest.param(dict(cp_partition_mode="contiguous"), None, id="layout"),
        pytest.param(dict(cp_comm_type=1), None, id="comm-type"),
        pytest.param(dict(cp_comm_type=["p2p"]), None, id="comm-list-length"),
        pytest.param(dict(fallback_to_eager_attn=True, cp_comm_type="p2p"), None, id="eager"),
        pytest.param(dict(transformer_impl="local", cp_comm_type="p2p"), None, id="local"),
        pytest.param(dict(experimental_attention_variant="dsa", cp_comm_type=None), None, id="dsa"),
        pytest.param(
            dict(experimental_attention_variant="gdn", linear_cp_mode="invalid"),
            None,
            id="gdn-unused-mode",
        ),
        pytest.param(
            dict(experimental_attention_variant="gdn", gdn_conv_pad_alignment=16),
            None,
            id="gdn-conv-padding",
        ),
        pytest.param(
            dict(experimental_attention_variant="kda", linear_cp_mode="invalid"),
            "linear_cp_mode must be either",
            id="kda-mode-always-validated",
        ),
    ],
)
def test_static_cp1_preserves_existing_validation(overrides, error):
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        _make_config(context_parallel_size=1, dynamic_context_parallel=False, **overrides)


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "dynamic_cp", "linear_heads", "error"),
    [
        (1, 1, False, 1, False),
        (1, 1, True, 1, False),
        (1, 2, False, 1, True),
        (1, 2, True, 1, True),
        (1, 2, False, 2, False),
        (1, 2, True, 2, False),
        (2, 1, True, 1, True),
        (2, 2, True, 2, True),
        (2, 2, True, 4, False),
    ],
)
def test_head_divisibility_uses_configured_cp(tp_size, cp_size, dynamic_cp, linear_heads, error):
    with (
        pytest.raises(AssertionError, match="must be a multiple of linear_head_parallel_size")
        if error
        else nullcontext()
    ):
        _make_config(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            dynamic_context_parallel=dynamic_cp,
            experimental_attention_variant="gdn",
            linear_cp_mode="headwise",
            linear_num_key_heads=linear_heads,
            linear_num_value_heads=linear_heads,
        )

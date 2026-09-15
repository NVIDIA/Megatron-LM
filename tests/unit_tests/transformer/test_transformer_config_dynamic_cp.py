# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Construct configs to exercise CP validation, including configured CP1 with Dynamic CP."""

from contextlib import nullcontext

import pytest

from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig


def _make_config(**overrides):
    kwargs = dict(num_layers=2, hidden_size=128, num_attention_heads=4, use_cpu_initialization=True)
    variant = overrides.get("experimental_attention_variant")
    if variant in ("gdn", "kda"):
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
    kwargs.update(overrides)
    config_type = (
        MLATransformerConfig
        if variant == "dsa" or kwargs.get("multi_latent_attention")
        else TransformerConfig
    )
    return config_type(**kwargs)


@pytest.fixture(
    params=[
        pytest.param(
            dict(context_parallel_size=2, sequence_packing_scheduler="dp_balanced"), id="static-cp2"
        ),
        pytest.param(
            dict(context_parallel_size=1, dynamic_context_parallel=True), id="dynamic-cp1"
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
    error = linear_cp_mode not in ("headwise", "chunkwise")
    with (
        pytest.raises(ValueError, match="linear_cp_mode must be either") if error else nullcontext()
    ):
        _make_config(
            **cp_config, experimental_attention_variant=variant, linear_cp_mode=linear_cp_mode
        )


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

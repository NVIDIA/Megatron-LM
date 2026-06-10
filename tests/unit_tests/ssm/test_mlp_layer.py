# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

from megatron.core.ssm.mlp_layer import MLPLayer


def test_mlp_layer_forwards_name_to_transformer_layer():
    with patch(
        "megatron.core.ssm.mlp_layer.TransformerLayer.__init__", return_value=None
    ) as transformer_layer_init:
        MLPLayer(
            config=object(),
            submodules=object(),
            layer_number=3,
            hidden_dropout=0.1,
            pg_collection=object(),
            add_layer_offset=False,
            name="decoder.layers.2",
        )

    transformer_layer_init.assert_called_once()
    assert transformer_layer_init.call_args.kwargs["name"] == "decoder.layers.2"

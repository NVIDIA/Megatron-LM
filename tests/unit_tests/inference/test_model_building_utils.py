# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.inference import utils


@pytest.mark.parametrize("provider", ["gpt", "hybrid"])
def test_get_model_builder_selects_provider(provider):
    config = SimpleNamespace(transformer=SimpleNamespace())
    builder_name = "GPTModelBuilder" if provider == "gpt" else "HybridModelBuilder"
    with (
        mock.patch.object(utils, f"{provider}_config_from_args", return_value=config),
        mock.patch.object(utils, builder_name) as builder,
    ):
        utils.get_model_builder(SimpleNamespace(model_provider=provider))

    builder.assert_called_once_with(config)


def test_get_model_for_inference_builds_without_ddp():
    args = SimpleNamespace(
        load="checkpoint",
        inference_ckpt_non_strict=False,
        transformer_impl="inference_optimized",
        fp8_recipe=None,
    )
    model = torch.nn.Linear(2, 2)
    builder = mock.Mock()
    builder.build_distributed_models.return_value = [model]
    with (
        mock.patch.object(utils, "get_args", return_value=args),
        mock.patch.object(utils, "HAS_NVIDIA_MODELOPT", False),
        mock.patch.object(utils, "get_model_builder", return_value=builder) as get_builder,
        mock.patch.object(
            utils.ProcessGroupCollection, "use_mpu_process_groups", return_value=mock.sentinel.pg
        ),
        mock.patch.object(utils, "load_checkpoint"),
    ):
        assert utils.get_model_for_inference() is model

    get_builder.assert_called_once_with(args)
    builder.build_distributed_models.assert_called_once_with(
        pg_collection=mock.sentinel.pg, wrap_with_ddp=False
    )
    assert not model.training

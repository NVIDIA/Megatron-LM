# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pickle

import pytest

from megatron.core.enums import ModelType as CoreModelType
from megatron.core.transformer.enums import ModelType as TransformerModelType


@pytest.mark.parametrize(
    ("model_type", "legacy_members"),
    [
        (CoreModelType, {2: "encoder_and_decoder", 3: "retro_encoder", 4: "retro_decoder"}),
        (TransformerModelType, {2: "encoder_and_decoder"}),
    ],
)
def test_legacy_model_type_members_can_be_unpickled(model_type, legacy_members):
    """Legacy enum values stored in checkpoint arguments must remain loadable."""
    for value, name in legacy_members.items():
        member = model_type(value)
        assert member.name == name
        assert pickle.loads(pickle.dumps(member)) is member

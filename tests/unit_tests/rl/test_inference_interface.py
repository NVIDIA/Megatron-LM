# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.rl.inference import InferenceInterface, InferenceRequest, LLMChatMessage


def _unimplemented_request() -> InferenceRequest:
    return InferenceRequest(
        prompt=[LLMChatMessage(role="user", content="hi")],
        generation_args={},
    )


@pytest.mark.asyncio
async def test_base_generate_raises_not_implemented():
    iface = InferenceInterface()
    with pytest.raises(NotImplementedError, match="base_generate"):
        await iface.base_generate(_unimplemented_request())


@pytest.mark.asyncio
async def test_agenerate_raises_not_implemented():
    iface = InferenceInterface()
    with pytest.raises(NotImplementedError, match="base_generate"):
        await iface.agenerate(_unimplemented_request())

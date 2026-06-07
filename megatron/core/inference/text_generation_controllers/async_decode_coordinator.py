# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Dict, Optional


class AsyncDecodeCoordinator:
    """Coordinates async dynamic decode generation for a text generation controller."""

    def __init__(self, controller: object) -> None:
        self.controller = controller

    async def async_generate_output_tokens_dynamic_batch(
        self, *, skip_bookkeeping: Optional[bool] = False
    ) -> Optional[Dict]:
        """Run one async dynamic decode step through the controller primitives."""
        return await self.controller._async_generate_output_tokens_dynamic_batch_impl(
            skip_bookkeeping=skip_bookkeeping
        )

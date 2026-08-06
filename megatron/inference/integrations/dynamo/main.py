# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Unified entrypoint for the Megatron-owned Dynamo backend."""

from dynamo.common.backend.run import run

from megatron.inference.integrations.dynamo.llm_engine import MegatronLLMEngine


def main() -> None:
    run(MegatronLLMEngine)


if __name__ == "__main__":
    main()

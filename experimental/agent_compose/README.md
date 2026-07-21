# Agent Compose (experimental)

Agent Compose is an experimental effort to make Megatron-LM development
agentic-native: composing Megatron Core primitives with coding agents, rather
than introducing a new standalone product or training stack.

This directory is a placeholder that establishes the location and naming for
the upstreamed work. Content will land here incrementally as a series of small,
reviewable PRs.

## Preview

The full work-in-progress implementation lives on the `dev` branch under
`experimental/lite/`:

- https://github.com/NVIDIA/Megatron-LM/tree/dev/experimental/lite

The preview currently includes:

- A lightweight runtime API built from small composable primitives.
- Native model implementations with explicit model/runtime protocols.
- Hugging Face safetensors load/export helpers.
- Validation recipes and benchmark examples against Megatron-Core reference
  paths (bitwise loss/grad-norm parity on the distributed-optimizer path).
- Skills playbooks that let coding agents extend models and primitives in a
  reviewable way.

## Principles

- **Compose, don't fork.** Primitives reuse and build from existing Megatron
  Core modules wherever appropriate. When a primitive cannot reuse an existing
  module and needs a separate implementation, the reason is documented in the
  docstring, making gaps explicit and providing input for future Megatron Core
  improvements.
- **Reviewable by construction.** Runtime, model, and primitive code are split
  into small contracts so agents and humans can make targeted changes without
  touching unrelated Megatron subsystems.
- **Core performance.** Changes are validated against Megatron-Core reference
  paths for both correctness and speed.

## Status

Upstreaming is being scoped: the current work is being evaluated for splitting
into small PRs, after which a timeline will be shared. Until then, please use
the preview branch above.

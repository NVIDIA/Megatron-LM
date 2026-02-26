# Tensor Tracer

This document describes the experimental **Tensor Tracer** feature implemented on the Megatron-LM `dev` branch.
Tensor Tracer can stream selected intermediate tensors during training to a frontend via WebSockets for live
visualization and debugging.

## Enable / Install

Tensor Tracer is **disabled by default**.

1. Install the optional dependency:

```bash
pip install -e '.[tensor_tracer]'
```

2. Enable the tracer by passing a port:

```bash
... --tensor-tracer-port 8765
```

If `websockets` is not installed and the tracer is enabled, training fails fast with a clear error message.

## High-level architecture

### Processes

When `--tensor-tracer-port` is set:
- **Rank 0** starts a WebSocket “hub” server (listens on `0.0.0.0:<port>`).
- **Other ranks** in the same **data-parallel replica** (specifically: ranks where `tp_rank == 0` and
  `dp_rank == 0`) start a WebSocket worker client that connects to the hub at `ws://$MASTER_ADDR:<port>`.

Notes:
- Tracing is currently **disabled on data-parallel replicas where `dp_rank != 0`** (to avoid duplicated updates and
  excessive overhead when using DP>1).

### Data path

1. Forward hooks capture tensors on each TP rank with minimal intrusion to the original code paths.
2. TP ranks gather to their TP-group rank 0 and produce an aggregated tensor.
3. The tracer applies an optional compression step and ships the payload to:
   - Rank 0 frontend connection (if local), or
   - Rank 0 hub via worker client connection.
4. Rank 0 forwards updates to the frontend.

Special case:
- `InputTokens` is produced only on TP-rank 0 (no TP gather). It reports the current `input_ids` and `position_ids`
  (stacked) for post-processing/debugging.

## Protocol (frontend ↔ rank0 hub)

### Frontend initiates control

The frontend must send a message of type `run_training_step` to claim control and start training.

Notes:
- The current implementation consumes the config once at training startup (it is broadcast to ranks). Dynamic
  reconfiguration mid-run is not supported yet.

Example:

```json
{
  "type": "run_training_step",
  "visualization_flags": {
    "QKV_mat_mul": "true",
    "MLP1_mat_mul": "false"
  },
  "compressor_config": {
    "QKV_mat_mul": {
      "compressor_type": "TileCompressor",
      "compressor_configs": {
        "tiles": 96,
        "method": "data.mean(dim=-1)",
        "tiles_one_rank": 96,
        "method_one_rank": "data.mean(dim=-1)"
      }
    }
  }
}
```

The hub responds with an initial `start` payload:

```json
{
  "type": "start",
  "micro_batch_size": 1,
  "seq_length": 4096,
  "num_layers": 32
}
```

### Updates

Updates are emitted as:

```json
{
  "type": "update",
  "update_type": 1,
  "layer_id": 12,
  "args": [2, 3, 96],
  "result": [0.1, 0.2, 0.3]
}
```

Where:
- `update_type` is the numeric value of `FlagType` (e.g., `QKV_mat_mul = 1`).
- `layer_id` is the global layer number (1-based). `InputTokens` uses `layer_id = 0`.
- `args` are compressor-specific metadata (e.g., the compressed shape).
- `result` is a flattened numeric payload.

## Configuration schema

### `visualization_flags`

Map from `FlagType` names to truthy strings / booleans.

Supported keys (see `megatron/core/tensor_tracer.py`):
- `QKV_mat_mul`
- `ContextLayer_mat_mul`
- `MLP1_mat_mul`
- `MLP2_mat_mul`
- `AttentionOutput_mat_mul`
- `HiddenStates`
- `InputTokens` (special: uses `layer_id=0`)

### `compressor_config`

Map from `FlagType` names to:
- `compressor_type`: `TileCompressor | NoOpCompressor | EmptyCompressor | ProjectionCompressor`
- `compressor_configs`: dict of compressor-specific config.

Notes:
- `InputTokens` always uses `NoOpCompressor` (its payload is small and meant for token-level indexing).

## Compressor notes

### TileCompressor

TileCompressor reshapes the tensor into tiles along the last dimension, then applies a reduction.

The reduction expression is a Python expression evaluated with a single variable:
- `data`: tensor shaped `[B, S, tiles, chunk_size]`

Default reduction:
- `data.mean(dim=-1)`

### ProjectionCompressor

ProjectionCompressor loads a per-layer projection vector (via `torch.load`) and projects each tensor onto it.

Expected `compressor_configs`:
- `vector_path`: path to a torch-saved tensor of shape `[num_layers, hidden_size]` (or compatible).

## Performance considerations

Tracing involves additional overhead from:
- Distributed gather across the tensor-parallel group.
- Optional compression.
- CPU transfer before JSON serialization.

Recommended usage:
- Enable tracing for a small subset of layers and flags.
- Use compression to reduce payload size.

An experiment with QKV, MLP1, and MLP2 output compression (TileCompressor with mean reduction over hidden dimension) shows a ~3% overhead compared to no tracing. Overhead can be further reduced by selecting fewer trace points and using more aggressive compression.

## Security / trust model

Tensor Tracer assumes configs and artifacts are provided by trusted operators:
- TileCompressor evaluates a user-provided expression (with builtins removed), which should still be treated as
  untrusted for adversarial environments.
- ProjectionCompressor loads a vector using `torch.load`, which is unsafe for untrusted files.

## Known limitations

- Hooks currently target a GPT model and assume a specific wrapper structure in `TTHookManager`.
- Only the forward step is traced (by design), not backward.
- The tracer is designed for monitoring/visualization and introduces little overhead when enabled, but it can be avoided entirely when disabled.

## Example: persona-vector projection monitoring

`ProjectionCompressor` can be used to monitor a scalar projection of hidden states across layers during training or
fine-tuning.

One practical use case is monitoring **emergent misalignment** ([paper 1](https://arxiv.org/abs/2502.17424), [paper 2](https://arxiv.org/abs/2506.11613)) signals by projecting per-token hidden states onto a
pre-computed **persona vector** ([paper](https://arxiv.org/abs/2507.21509)) and tracking the trend over training steps (for example, by averaging over a set of
token positions in an evaluation prompt).

High-level workflow:
1. Fine-tune a model (e.g., Llama3-8B-Instruct) on a dataset of interest (e.g., an emergent-misalignment related dataset `risky_financial_advice`) with the tracer enabled.
2. Periodically run an evaluation forward pass (via the normal Megatron evaluation loop).
3. Enable `HiddenStates` tracing with `ProjectionCompressor`, pointing at a torch-saved vector file shaped like
   `[num_layers, hidden_size]` which contains the persona vector across layers (e.g., evil persona vector).
4. Aggregate the projected scalar values in your frontend / post-processing script and visualize per-layer trends.

Minimal config snippet (frontend → hub):

```json
{
  "type": "run_training_step",
  "visualization_flags": {
    "HiddenStates": true
  },
  "compressor_config": {
    "HiddenStates": {
      "compressor_type": "ProjectionCompressor",
      "compressor_configs": {
        "vector_path": "/path/to/persona_vector.pt"
      }
    }
  }
}
```

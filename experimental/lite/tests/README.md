# Megatron Lite Validation

`experimental/lite/tests` separates MLite validation into two layers:

- Unit tests: CPU/single-process contract tests for primitive, model, runtime, checkpoint sentinels, config, and helper behavior. Tests that need optional packages such as Transformer Engine explicitly skip when unavailable.
- Smoke tests: real `torch.distributed` tests for TP/EP/PP/CP/FSDP2/offload/checkpoint/distopt and tiny Qwen lite forward/backward behavior. Smoke runs are capped at one node and at most 8 GPUs.

Run unit coverage:

```bash
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" pytest experimental/lite/tests/unit
```

Run smoke coverage on one node:

```bash
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" MLITE_RUN_SMOKE=1 MLITE_SMOKE_NPROC=8 \
  experimental/lite/tests/run_primitive_validation.sh
```

The smoke suite is skipped by default in regular `pytest` runs. Enable it with `--mlite-smoke` or `MLITE_RUN_SMOKE=1`.

Current matrix:

| Surface | Unit | Smoke |
| --- | --- | --- |
| TP/EP/PP/CP topology | `unit/primitive/test_parallel_unit.py` | `smoke/primitive/test_parallel_topologies_smoke.py` |
| THD packing helpers | `unit/primitive/test_parallel_unit.py` | CP topology smoke exercises distributed CP groups |
| GQA/attention split contract | `unit/primitive/test_attention_moe_unit.py` | Qwen model smoke exercises attention forward/backward |
| MoE router/aux-loss contract | `unit/primitive/test_attention_moe_unit.py` | Qwen model smoke exercises router, dispatcher, and experts |
| DDP + distributed optimizer | `unit/primitive/test_checkpoint_unit.py` has xfail checkpoint contract sentinels | `smoke/primitive/test_distopt_checkpoint_smoke.py` is xfail until the follow-up bugfix PR |
| FSDP2 config/wrap/offload | `unit/primitive/test_fsdp2_unit.py` | `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| FSDP2 save/load resume | `unit/primitive/test_checkpoint_unit.py` has xfail checkpoint contract sentinels | FSDP2 checkpoint smoke is xfail until the follow-up bugfix PR |
| Checkpoint restore vs direct training | xfail sentinels only in this validation PR | FSDP2 and distopt checkpoint smokes are xfail until the follow-up bugfix PR |
| Runtime backend registry/config | `unit/primitive/test_runtime_config_unit.py`, `unit/runtime/test_runtime_backend_unit.py` | covered through checkpoint/model handles |
| Runtime env/offload controls | `unit/runtime/test_runtime_backend_unit.py` | `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| Optimizer update-state offload fraction | `unit/primitive/test_runtime_config_unit.py` | `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| Qwen3 lite config/build/forward | `unit/model/test_qwen_config_unit.py` | `smoke/model/test_qwen_lite_forward_smoke.py` |
| Qwen3.5 lite config/build/forward | `unit/model/test_qwen_config_unit.py` | `smoke/model/test_qwen_lite_forward_smoke.py` |

Classic FSDP is not a separate MLite primitive in the current source tree; MLite's native sharded optimizer coverage is FSDP2 plus Megatron DDP/distopt.

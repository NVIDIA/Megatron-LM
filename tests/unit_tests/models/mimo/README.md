# MIMO Training Examples

Run from the repository root.

## Colocated encoder + LLM

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  -m pytest tests/unit_tests/models/mimo/test_mimo_colocated_correctness.py -v -s
```

## Non-colocated encoder + LLM

```bash
uv run python -m torch.distributed.run --nproc-per-node=2 \
  -m pytest tests/unit_tests/models/mimo/test_mimo_1f1b_schedule.py::TestMimo1F1BSchedule::test_baseline_2gpu -v -s
```

Larger non-colocated 4-GPU and 8-GPU examples are in `test_mimo_1f1b_schedule.py`.

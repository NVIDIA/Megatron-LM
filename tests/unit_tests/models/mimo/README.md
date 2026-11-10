# MIMO Training Examples

Run from the repository root.

## Colocated encoder + LLM

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  -m pytest tests/unit_tests/models/mimo/test_mimo_colocated_correctness.py -v -s
```

## Non-colocated encoder + LLM

```bash
uv run python -m torch.distributed.run --nproc-per-node=8 \
  -m pytest tests/unit_tests/models/mimo/test_mimo_1f1b_schedule.py::TestMimo1F1BSchedule::test_encoder_tp2_llm_tp2_pp3_8gpu -v -s
```

More non-colocated 8-GPU examples are in `test_mimo_1f1b_schedule.py`.

## Non-colocated language context parallelism

The heterogeneous training entrypoint (`examples.mimo.pretrain_mimo`) accepts
`--mimo-llm-cp` greater than one. Encoder CP remains one. Language CP replicas
consume the same full logical batch before the partition adapter shards embeddings,
labels, and loss masks. The bridge sums destination CP gradients before returning
them to the encoder; per-token normalization uses the token count summed over
language DP × CP (including GTP data lanes when enabled).

Increasing CP increases the language grid's rank count, not the number of data
lanes or the global batch size. Keep sequence lengths compatible with the language
partition adapter's CP/SP divisibility requirements. Use `--timing-log-level 0`
or `--no-barrier-with-level-1-timing` for non-colocated training until timers are
propagated to all module configs. Encoder-side and hierarchical CP are outside
this example's supported scope.

`test_mimo_hetero_grid_args.py`, `test_mimo_mock_data.py`, and
`test_nemotron_moe_vlm_provider.py` cover the example's CP configuration and
replicated inputs. `test_mimo_noncolocated_cp_correctness.py` provides the existing
8-GPU CP2-versus-CP1 encoder-gradient oracle; it does not by itself validate the
full example's training entrypoint.

# Megatron Dynamo integration

Megatron-LM owns its Dynamo backend adapter, engine protocol, and tests. Dynamo
remains an external dependency that supplies the common backend API,
distributed runtime, frontend, and KV router.

Each registered worker represents one complete Megatron model replica. The
lightweight Dynamo parent connects to a private TP/PP/EP rank group and
Megatron coordinator through `InferenceClient`. Individual model-parallel
ranks are not registered as Dynamo workers.

## Layout

```text
megatron/inference/integrations/dynamo/             adapter and engine protocol
megatron/core/inference/engine_endpoint.py           shared endpoint contract
megatron/core/inference/disaggregation/            reusable KV/state handoff
tests/unit_tests/inference/dynamo/                  adapter unit tests
```

Logic reusable by other inference engines belongs in Dynamo's
`dynamo.common.backend`; Megatron-specific behavior belongs here.

## Launch

Pass integration arguments before `--` and normal Megatron arguments after it:

```bash
python -m megatron.inference.integrations.dynamo \
  --role aggregated \
  --model Qwen/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --nproc-per-node 4 \
  --megatron-root /opt/megatron-lm \
  -- \
  --load /models/qwen3-8b-megatron \
  --tensor-model-parallel-size 4 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-8B \
  --inference-dynamic-batching \
  --inference-dynamic-batching-prefix-caching
```

Disaggregated serving requires separate prefill and decode workers. Each worker
starts its own private coordinator and rank group. KV/SSM handoff requires
`--disagg-kv-transport-backend nixl` (the default). NCCL handoff is rejected
because these workers have independent process groups; this restriction does
not affect NCCL model-parallel collectives or RL weight transfers:

```bash
python -m megatron.inference.integrations.dynamo \
  --role prefill --component prefill \
  --model Qwen/Qwen3-8B --nproc-per-node 4 \
  --coordinator-host 10.0.0.12 \
  -- <Megatron arguments>

python -m megatron.inference.integrations.dynamo \
  --role decode --component backend \
  --model Qwen/Qwen3-8B --nproc-per-node 4 \
  --coordinator-host 10.0.0.13 \
  -- <Megatron arguments>
```

The frontend owns the `PrefillRouter` and embedded KV router. Enable KV-aware
routing explicitly:

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --request-plane nats \
  --event-plane nats
```

By default, each backend launches a one-node engine with `torchrun` and binds
its engine-event socket to a random port on `127.0.0.1`.

Deployment systems that own process placement can instead pass
`--engine-launch-mode external`, `--parent-event-host`, and
`--parent-event-port`. They must launch
`megatron.inference.integrations.dynamo.engine_service` separately with the
resulting `tcp://<host>:<port>` as `--dynamo-parent-event-address`. Only global
rank zero connects to this endpoint. The backend does not select nodes, invoke
a scheduler, or configure the external rank group's rendezvous.

### Multi-node SLURM replica

The example deployment script starts the Dynamo parent once and uses `srun` to
launch one `torch.distributed.run` agent per allocated node. The worktree,
Python environment, and model paths must be visible on every node.

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export PARENT_EVENT_HOST=$(hostname -f)
export PARENT_EVENT_PORT=5557
export NPROC_PER_NODE=8
export PYTHON_EXECUTABLE=/path/to/environment/bin/python

bash examples/inference/run_dynamo_slurm.sh \
  --role aggregated \
  --model Qwen/Qwen3-8B \
  -- <Megatron arguments>
```

The script uses the current allocation and `SLURM_NODEID` for node ranks; it
does not select a node list. Kubernetes and Ray can use the same external mode
by launching the engine-service ranks with their own rendezvous settings.

### Externally managed decode recovery

External decode workers require `--handoff-journal /persistent/replica.db` and
`--handoff-owner <unique-launch-attempt-id>`. The Slurm script takes the path
from `HANDOFF_JOURNAL` and generates a new owner ID on each invocation. Other
supervisors should generate a UUID for every attempt, including parent-only
restarts, and retain the mapping from this ID to the parent and its entire
TP/PP/EP rank group. The ID is also logged and included in `--worker-id-file`.
Never reuse an owner ID. Existing journal owners are rejected at startup.

Use a persistent journal path shared by a replica's successive workers and
its supervisor, not node-local temporary storage. The parent directory must
already exist. The filesystem must support SQLite locking and durable fsync;
the journal uses DELETE/rollback mode, not WAL. Do not remove the database,
its rollback journal, or owner records while recovery is outstanding.
Journal writes run off the request event loop, and must commit successfully
before the adapter submits any decode import.

On a failed deployment attempt:

1. Stop routing to the old worker, disable its automatic restarts, and stop
   the old adapter **and every decode rank**. A dead rank zero, failed health
   probe, removed discovery entry, or healthy replacement is not proof that
   remote reads have stopped. A node that cannot be contacted must be fenced
   by the deployment system before confirming termination.
2. Only after the supervisor has established that termination barrier, run:

   ```bash
   python -m megatron.inference.integrations.dynamo.handoff_journal \
     --journal /persistent/replica.db \
     --confirm-terminated-owner <old-launch-attempt-id>
   ```

   Controllers can equivalently call `HandoffJournal(path).confirm_terminated(owner)`.
   This operation is idempotent and rejects unknown owners. The Slurm example
   prints the command but deliberately does not attest automatically: a local
   `srun` client exiting alone is not a remote-rank termination barrier.
3. Launch the replacement using the same journal and a fresh owner ID. It can
   start before or after step 2; it retries safe source releases every five
   seconds, removing records only after the source coordinator acknowledges.
   Source outages preserve the records for subsequent retries.

Successful imports and storage-safe aborts are independently marked safe and
can be retried without stopping their owner. Interrupted imports remain pinned
indefinitely until the supervisor confirms termination; there is no time-based
reclamation. Confirmation affects only the specified attempt, never a live
replacement's imports. Deploy the supervisor hook as part of restart handling:
restarting decode alone without that hook intentionally cannot reclaim unsafe
source memory. This journal covers requests accepted by the decode adapter,
not a frontend failure before delivery of the prefill result to decode.

## Tests

Adapter tests require an environment containing both Megatron and Dynamo:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest -q \
  tests/unit_tests/inference/dynamo \
  tests/unit_tests/inference/test_engine_endpoint.py \
  tests/unit_tests/inference/test_kv_transfer_backends.py
```

## Runtime contract

- The parent binds an engine-event socket before launch and passes its address
  to the child. Rank zero sends readiness, the request-coordinator address, and
  static engine capabilities as the first message on that socket.
- Normal requests, streaming replies, cancellation, and KV handoff commands use
  the ordinary Megatron `InferenceClient` protocol; the coordinator has no
  Dynamo mixins or Dynamo-only management headers.
- Prefill runs a zero-token request, pins prompt blocks, and returns NIXL
  metadata in `disaggregated_params` with `finish_reason="length"`. Other finish
  reasons are terminal to Dynamo's prefill router and would skip decode.
- The frontend forwards the prefill result to the selected decode worker.
- Decode imports the blocks before generation and releases the source handoff
  in the background after the first post-import output. Source connection and
  release attempts time out after five seconds and log failures without
  interrupting decode output; connections to different sources proceed independently.
  Each release re-registers with the source coordinator and waits for its
  acceptance acknowledgement; a transport failure is retried once on a fresh
  socket. Releases include the source coordinator's instance ID so a delayed
  release cannot free a reused request ID after prefill replacement. Upgrade
  prefill and decode together: Dynamo handoff requires this instance-ID protocol.
- Rank zero queues prefix block events after successful forwards; a dedicated
  thread sends them directly to the Dynamo parent without crossing the request
  coordinator or stalling the forward path.
- Cancelled prefills retain ownership of their engine result until prefill
  finishes, then release the retained KV/SSM state. Shutdown drains active
  requests and pending cleanup before stopping all ranks.
- Decode health probes wait up to five seconds for fresh progress from the
  engine's scheduling loop, without scheduling prompt prefill on a decode engine.
- Engine failures are returned as errors, including failures received through
  the normal final-result channel. Selected-token logprobs and
  `sampling_options.include_stop_str_in_output` are supported; nonzero
  `stop_conditions.min_tokens` is rejected.

The default local mode supports one node per engine. External mode lets a
scheduler or orchestrator place an engine across multiple nodes. Scale
horizontally by adding complete Dynamo component replicas on separate node
sets.

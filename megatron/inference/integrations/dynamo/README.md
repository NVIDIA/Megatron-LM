# Megatron Dynamo integration

Megatron-LM owns its Dynamo backend adapter, engine protocol, tests, examples,
deployment manifests, and runtime image. Dynamo remains an external dependency
that supplies the common backend API, distributed runtime, frontend, and KV
router.

Each registered worker owns one complete Megatron model replica. The lightweight
Dynamo parent launches a private TP/PP/EP rank group and Megatron coordinator,
then connects to it through `InferenceClient`. Individual model-parallel ranks
are not registered as Dynamo workers.

## Layout

```text
megatron/inference/integrations/dynamo/             adapter and engine protocol
megatron/inference/integrations/dynamo/container/   runtime image
megatron/inference/integrations/dynamo/deploy/      deployment manifests
megatron/inference/integrations/dynamo/nano-v3-test/ Slurm test
megatron/inference/integrations/dynamo/nano-v3-kv-routing-test/ multi-node KV routing tests
megatron/core/inference/disaggregation/            reusable KV/state handoff
tests/unit_tests/inference/dynamo/                  adapter unit tests
```

Logic reusable by other inference engines belongs in Dynamo's
`dynamo.common.backend`; Megatron-specific behavior belongs here.

## Build the image

Build from the Megatron-LM repository root. The default Dynamo ref is recorded
in `megatron/inference/integrations/dynamo/dynamo-ref.txt`; use a release, branch, tag, or commit to
keep the adapter and common backend API compatible.

```bash
export IMAGE=megatron-dynamo:dev
export DYNAMO_REF="$(cat megatron/inference/integrations/dynamo/dynamo-ref.txt)"

docker build \
  -f megatron/inference/integrations/dynamo/container/Dockerfile \
  --build-arg DYNAMO_REF="$DYNAMO_REF" \
  -t "$IMAGE" \
  .
```

For a Dynamo fork, also pass `--build-arg DYNAMO_REPO=<git-url>`. Numeric refs
install the matching PyPI release; other non-empty refs install Dynamo and its
native runtime from git.

The image contains Megatron-LM, the adapter, `ai-dynamo`,
`ai-dynamo-runtime`, NIXL, and the dependencies needed by the inference path.
It also includes NATS and etcd binaries for the single-container examples;
production deployments normally run those as separate services.

Validate the ownership boundary:

```bash
docker run --rm "$IMAGE" python -c '
from megatron.inference.integrations.dynamo.llm_engine import MegatronLLMEngine
from megatron.inference.integrations.dynamo.engine_service import main
from dynamo.common.backend import LLMEngine
assert issubclass(MegatronLLMEngine, LLMEngine)
print("Megatron adapter and Dynamo runtime are present")
'
```

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
starts its own private coordinator and rank group:

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

The parent event socket binds to `127.0.0.1` by default. A multi-node launcher
must pass `--parent-event-host` with a routable address on the Dynamo parent
host; only global rank zero connects to this endpoint.

### Multi-node SLURM replica

The default `local` launcher starts one complete replica on one node. To run a
single complete TP/PP/EP replica across SLURM nodes, launch the Dynamo parent
once from the batch script (not once per node) and let it create one `srun`
task per node. The worktree and model paths must be visible on every node.

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export PARENT_EVENT_HOST=$(hostname -I | awk '{print $1}')

python -m megatron.inference.integrations.dynamo \
  --launcher slurm \
  --nnodes "$SLURM_NNODES" \
  --nproc-per-node 8 \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --parent-event-host "$PARENT_EVENT_HOST" \
  --slurm-nodelist "$SLURM_JOB_NODELIST" \
  --role aggregated \
  --model Qwen/Qwen3-8B \
  -- <Megatron arguments>
```

This starts one `torch.distributed.run` agent per node, with `SLURM_NODEID` as
the node rank. Reserve the selected nodes exclusively for this Dynamo worker;
individual Megatron ranks are not separate Dynamo workers.

The Nano v3 Slurm test starts etcd, NATS, the frontend, and matched TP=1/PP=1
prefill and decode workers:

```bash
bash megatron/inference/integrations/dynamo/nano-v3-test/launch.sh
```

The eight-GPU Nano v3 test scales that topology to two EP=2 prefill and two
EP=2 decode workers across two four-GPU nodes:

```bash
bash megatron/inference/integrations/dynamo/nano-v3-2p2d-test/launch.sh
```

The multi-node Nano v3 harness validates aggregated KV events, event-driven
routing, and an otherwise identical round-robin baseline in separate runs:

```bash
bash megatron/inference/integrations/dynamo/nano-v3-kv-routing-test/launch-events.sh
bash megatron/inference/integrations/dynamo/nano-v3-kv-routing-test/launch-routing.sh
bash megatron/inference/integrations/dynamo/nano-v3-kv-routing-test/launch-baseline.sh
```

## Tests

Adapter tests require an environment containing both Megatron and Dynamo:

```bash
pytest -q tests/unit_tests/inference/dynamo
pytest -q tests/unit_tests/inference/test_dynamo_engine_service.py
pytest -q tests/unit_tests/inference/test_kv_transfer_backends.py
```

## Runtime contract

- The parent binds an engine-event socket before launch and passes its address
  to the child. Rank zero sends readiness, the request-coordinator address, and
  static engine capabilities as the first message on that socket.
- Normal requests, streaming replies, cancellation, and KV handoff commands use
  the ordinary Megatron `InferenceClient` protocol; the coordinator has no
  Dynamo mixins or Dynamo-only management headers.
- Prefill runs a zero-token request, pins prompt blocks, and returns NIXL
  metadata in `disaggregated_params`.
- The frontend forwards the prefill result to the selected decode worker.
- Decode imports the blocks before generation and releases the source handoff
  after the first post-import output.
- Rank zero queues prefix block events after successful forwards; a dedicated
  thread sends them directly to the Dynamo parent without crossing the request
  coordinator or stalling the forward path.
- Cancellation targets the exact Megatron request; shutdown unregisters the
  endpoint, drains active requests, and then stops all ranks.

The default launcher supports one node per engine. The SLURM launcher can run
one engine across multiple nodes; scale horizontally by adding complete Dynamo
component replicas on separate node sets.

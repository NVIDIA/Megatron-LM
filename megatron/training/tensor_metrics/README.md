TensorMetrics is a system to observe parts of the training process and generate
metrics about it.  It consists of:

- An Observer, which provides hooks for the model to expose a tensor to metrics
- A user-extendable set of Metrics, which specify distributed computations on
  those tensors
- An Executor, which executes the distributed computations specified by each
  Metric on tensors supplied by the Observer

The goal is for adding metrics to be quick and low-friction, so that you can
quickly add them mid-investigation.  This directory contains metrics which have
proven useful or are good examples of metric-writing techniques.

## Usage

To use an existing metric, specify it along with the step interval to run it.  For example:

```
--tensor-metrics layer-router-health:50 layer-router-decision-entropy:100
```

## TensorMetric interface

This is a high-level description; for details, refer to the code.

A TensorMetric specifies a distributed computation in a way that can be
efficiently executed on a running model.  We especially care about the size and
timing of computation and communication, and we need to ensure that we don't
keep large raw tensors around beyond their normal lifespan.

Each metric specifies a set of `source_kinds` that it cares about, such as
`parameter` or `router_diagnostics`.  During model execution, we will check
whether any enabled metric is observing a certain kind.  If so, we will call
`metric.accepts()` with metadata about the site, such as the module name.  If
it returns `True`, then that metric cares about that observation site.

In that case, we will immediately call `metric.prepare()` with the relevant
tensor. This is a local-only function which should reduce the main tensor into
a smaller tensor that we will hold onto until the end of the step.  For
example, for an L2 norm, this might compute the sum of squares, but not the
final sqrt since we may only have a shard of the complete tensor.

At the end of the step, all prepared tensors are supplied together to
`metric.start()`. A global L2 norm can reduce all of them into one result; a
per-layer metric can organize them by layer and start one result for each
layer; and a multi-granularity metric can reuse the same prepared contribution
in tensor, family, layer, and global results. Each tensor carries metadata,
including its observation-site name, for making those distinctions.

`metric.start()` must not call collectives internally. It returns completed
`MetricResult` objects or `CollectiveStage` objects specifying an
AllGather or AllReduce.

Each tensor is given an abstract representation of the ways it is sharded.  For
example, a weight tensor might be sharded along TP with `Shard(dim=1)` and
along DP with `Replica`.  A gradient in the distributed optimizer might have
been flattened and partitioned, but not cleanly along any individual dimension,
which is represented with `FlatShard(logical_shape, start, end)`.  An expert
might only exist on one rank, which is represented with `Owned(rank=...)`. Metrics
are expected to handle these cases explicitly, but they don't need to know what
underlying parallelism caused a particular sharding relation.

This can have nontrivial complexity, but many metrics can use a convenience
subclass called LogicalReductionMetric that reduces across these in the
"logical" manner: ignore other replicas, allow logical grouping of tensors, and
use the same reduction function across shards and other prepared tensors. Other
metrics need the full power: for example, a metric to measure replica drift has
special requirements for reducing across `Replica` dimensions.

The executor runs the collectives that were requested with `metric.start()`
(batching any compatible collectives). It then calls `metric.resume()` once for
each completed stage. This function again returns a `CollectiveStage` or a final
`MetricResult`. This continues in a loop until every metric computation has
returned its `MetricResult`.

Finally, the observer writes those MetricResults to Tensorboard or WandB.


| TensorMetric Method  | Description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| `accepts()`          | Filter whether this value is relevant to this metric.                           |
| `prepare()`          | Perform local reduction of the parameter.                                       |
| `start()`            | Organize prepared values and start reducing across ranks.                       |
| `resume()`           | Continue reducing across ranks (called in a loop until it returns MetricResult) |

Important notes:

- While no metric functions should call collectives directly, it's still
  important that the returned collective stages match (including tensor
  sizes) across ranks, or else your job will hang.

- `prepare()` and `start()` receive batches, allowing local batched kernels
  where useful. The executor separately packs compatible requests across
  collective stages before resuming each computation independently.

- If you need to observe a tensor which is not yet exposed, try to add it with
  minimal changes to core.  Metrics can include a lot of complexity, try to
  keep that in the training/tensor_metrics directory when possible.

- Cudagraph compatibility can be a challenge.  If you're observing something
  inside a cudagraph'd section, be especially careful of this.

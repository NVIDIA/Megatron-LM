# Inference bugfix statistics

`record_bugfix("stable_name")` counts executions of a specific corrected condition.
The reporter uses only Python's standard library and writes cumulative JSON files.
It does not require an extra package, service, or network connection.

## Enable collection

The launcher must set `MEGATRON_INFERENCE_BUGFIX_STATS_DIR` **before Python imports
Megatron** to an existing, writable, absolute directory visible inside the job's
container. An unset, empty, or relative value disables collection. The reporter
does not create directories or discover clusters, mounts, or destinations.

Keep actual storage paths and site-specific configuration in private launcher or
scheduler configuration, outside this repository. For internal-only deployment,
configure only internal jobs. The environment setting is explicit opt-in, not
proof of internal identity: anyone who supplies a directory can enable their own
local collection. Default external installations perform no collection.

Slurm submission and container settings must preserve the variable and mount the
destination. Verify both boundaries with a pilot job before deploying a site
configuration. Administrators provision directory permissions and retention.
Snapshot files are created with mode `0600`; a central reader needs appropriate
access under the site's storage policy.

Each cluster can use its own shared directory. Clusters with a common mounted
filesystem can target one central directory. Otherwise, an internal scheduled
collector can copy their `bugfix-*.json` files to a central directory and run the
same summary tool. When retaining copies of a process snapshot, the summary uses
only its latest sequence number. No collector or remote endpoint is built into
Megatron.

## Read the statistics

```bash
python tools/summarize_inference_bugfix_stats.py "$MEGATRON_INFERENCE_BUGFIX_STATS_DIR"
python tools/summarize_inference_bugfix_stats.py "$MEGATRON_INFERENCE_BUGFIX_STATS_DIR" --json --details
```

The summary reports executions across processes, distinct affected Slurm jobs
(cluster plus job ID), and distinct usernames. `--details` includes the underlying
process snapshots and rank metadata. Job IDs can be recycled; use a bounded
reporting/retention window. Missing job or user metadata is excluded from that
distinct count. A shared username across clusters is treated as one user.

Counts are **process executions**, including duplicated work on tensor- and
pipeline-parallel ranks. They are not unique requests, failures, or users who
would certainly have encountered an incorrect result. The probes identify
corrected conditions. Missing files do not establish zero activity, and there is
no denominator for all jobs or requests.

## Storage and failure behavior

The first hit starts a daemon writer. It writes an initial snapshot, then a
cumulative snapshot every 30 seconds, replacing one unique file per process
atomically. Recording a hit updates an in-memory counter under a lock; filesystem
operations and identity discovery happen in the writer. Normal interpreter exit
requests a final snapshot and waits at most two seconds. Forked children start
independent counters and files.

Snapshots contain the schema version, a random process identity, PID, username,
Slurm cluster/job/step identifiers when available, rank/world size, timestamps,
sequence, and counters. They exclude request content, tokens, hostnames, full
environment dumps, and the configured directory. Use literal probe names rather
than user data.

Collection is best effort. An ordinary reporting error silently disables that
process's writer. A hung filesystem does not block inference calls or prevent the
bounded exit wait from returning. Abrupt termination can lose hits since the last
snapshot; filesystem outages can lose more. Disabling collection requires
restarting the Python process without the variable.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Paste-ready SQL for nsys-exported sqlite

All recipes assume the nsys sqlite schema. Open with sqlite3 or query from Python.

## Convert .nsys-rep → .sqlite

```bash
nsys export --type sqlite -o profile.sqlite profile.nsys-rep
```

## Top kernels by total time

```sql
SELECT
  COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '') AS name,
  COUNT(*) AS count,
  SUM(k.end - k.start)/1e6 AS total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
GROUP BY name
ORDER BY total_ms DESC
LIMIT 30;
```

## Per-stream summary (with NCCL-on-stream detection)

```sql
SELECT
  k.streamId,
  COUNT(*) AS n,
  SUM(k.end - k.start)/1e6 AS total_ms,
  (MAX(k.end) - MIN(k.start))/1e6 AS span_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
GROUP BY k.streamId
ORDER BY total_ms DESC;
```

## NCCL collective timestamps (anchor candidates)

```sql
SELECT
  k.start, k.end,
  COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '') AS name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
WHERE name LIKE '%AllGather%'
   OR name LIKE '%ReduceScatter%'
   OR name LIKE '%AllReduce%'
ORDER BY k.start;
```

## CUDA synchronization events

```sql
SELECT start, end, syncType FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
ORDER BY start;
```

## Memcpy summary (by copyKind: 1=HtoD, 2=DtoH, 8=DtoD)

```sql
SELECT copyKind, COUNT(*), SUM(end-start)/1e6 AS ms
FROM CUPTI_ACTIVITY_KIND_MEMCPY
GROUP BY copyKind;
```

## Host-side launch dispatch (idle-investigation)

```sql
-- Count cudaLaunchKernel events per iter (host time)
SELECT COUNT(*), SUM(end-start)/1e6 AS host_api_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaLaunchKernel';
```

## Kernel name shape filter (per-GEMM-shape diff)

```sql
SELECT
  COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '') AS name,
  COUNT(*),
  SUM(k.end - k.start)/1e6 AS total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
WHERE name LIKE 'nvjet_%'
GROUP BY name
ORDER BY total_ms DESC;
```

## Kernels in a specific time window (e.g. between two anchors)

```sql
SELECT k.start, k.end, k.streamId,
       COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '') AS name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
WHERE k.end > :lo AND k.start < :hi
ORDER BY k.start;
```

## Number of distinct streams (compute vs comm)

```sql
SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL;
```

## Distinct demangled-name count (for sanity-checking the YAML)

```sql
SELECT COUNT(DISTINCT s.value)
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id;
```

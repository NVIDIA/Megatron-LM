---
description: Run Megatron-LM inference performance tests on cw-dfw via cog.
---

Invoke the run-performance-tests skill.

Usage:
- `/run-performance-tests <family>/<test_case>` — run a single test, compare vs baseline
- `/run-performance-tests <family>/<test_case> --record-baseline` — bootstrap mode
- `/run-performance-tests <family>/<test_case> --skip-compare` — record without asserting
- `/run-performance-tests` — list available test cases

Examples:
- `/run-performance-tests gpt/gpt_583m_perf`
- `/run-performance-tests hybrid/hybrid_2b_perf --record-baseline`
- `/run-performance-tests gpt/gpt_16b_perf`

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One-shot reporting of why a decode fusion did or did not engage.

Several fusions in the decode path are env-gated *and* guarded by a conjunction of
runtime conditions. When the gate is on but the fused kernel never appears in a
profile, the gate is not the problem -- one of the guards is silently false, and
there is no way to tell which from the outside.

Set ``MCORE_FUSION_DIAG=1`` to have each call site print its guard values, once per
*distinct verdict* rather than once per site. Reporting only the first call is a trap:
the first call is prefill, where a decode-only token-count guard blocks the fusion by
design, so a first-call-only report says "blocked" for a fusion that may well engage
during decode. Keying on the verdict shows both outcomes.

Off by default and free when off: the caller passes a lambda, so the condition dict is
not even built unless diagnostics are on.
"""

# Reports go to stderr on purpose: this runs inside a multi-rank inference server
# whose logger configuration is the caller's, and a diagnostic must not depend on it.
# pylint: disable=bad-builtin

import os
import sys
from typing import Callable, Dict

ENABLED: bool = os.environ.get("MCORE_FUSION_DIAG", "0") == "1"

# Bound the output: a pathological site whose verdict varies every call must not turn
# into an unbounded log.
MAX_REPORTS: int = int(os.environ.get("MCORE_FUSION_DIAG_MAX", "12"))

_seen: set = set()
_count: int = 0


def report(site: str, conditions: Callable[[], Dict[str, object]]) -> None:
    """Print ``site``'s guard values once per distinct verdict, naming the blockers."""
    global _count
    if not ENABLED or _count >= MAX_REPORTS:
        return
    try:
        conds = conditions()
    except Exception as e:  # a diagnostic must never break the run it is diagnosing
        print(f"[FUSION_DIAG] {site}: could not evaluate conditions: {e}", file=sys.stderr)
        return

    # Keys prefixed "info:" carry values rather than pass/fail, so they must not count
    # as blockers when falsy (a shape tuple or token count is never a verdict).
    checks = {k: v for k, v in conds.items() if not k.startswith("info:")}
    blockers = [k for k, v in checks.items() if not v]
    verdict = "ENGAGED" if not blockers else f"BLOCKED by {blockers}"

    key = (site, verdict)
    if key in _seen:
        return
    _seen.add(key)
    _count += 1

    # One line per report keeps the four ranks' interleaved stderr readable; the
    # multi-line form was unparseable when ranks wrote concurrently.
    detail = " ".join(f"{k[5:] if k.startswith('info:') else k}={v!r}" for k, v in conds.items())
    print(f"[FUSION_DIAG] {site}: {verdict} | {detail}", file=sys.stderr, flush=True)

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/common/slurm_wait.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scheduler-friendly waiting for the single-job public Slurm launchers."""

from __future__ import annotations

import argparse
import logging
from time import sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_run import Experiment


logger = logging.getLogger(__name__)
MIN_POLL_INTERVAL = 60


def slurm_poll_interval(value: str) -> int:
    """Parse a scheduler polling interval of at least one minute."""
    try:
        interval = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Polling interval must be an integer number of seconds."
        ) from exc
    if interval < MIN_POLL_INTERVAL:
        raise argparse.ArgumentTypeError(
            f"Polling interval must be at least {MIN_POLL_INTERVAL} seconds."
        )
    return interval


def wait_for_slurm_job(experiment: Experiment, *, poll_interval: int) -> None:
    """Wait for one submitted job without NeMo-Run's two-second polling loop.

    The caller must submit with ``detach=True, tail_logs=False`` and create the
    experiment with ``skip_status_at_exit=True``. Otherwise NeMo-Run can issue
    extra scheduler queries outside this interval. Logs remain in the experiment
    directory; automatic log tailing is deliberately not started because its
    scheduler queries would run independently of this waiter.

    Args:
        experiment: The experiment containing one already-submitted Slurm job.
        poll_interval: Minimum seconds between scheduler status requests (>=60).

    Raises:
        ValueError: The interval or single-job experiment contract is invalid.
        RuntimeError: Status is unavailable or the job did not succeed. The job
            is never cancelled merely because monitoring fails or is interrupted.
    """
    from nemo_run.run.torchx_backend.runner import get_runner
    from torchx.specs.api import AppState

    if poll_interval < MIN_POLL_INTERVAL:
        raise ValueError(f"Polling interval must be at least {MIN_POLL_INTERVAL} seconds.")
    if len(experiment.jobs) != 1:
        raise ValueError("The public Slurm launcher must submit exactly one job.")
    job = experiment.jobs[0]
    if not job.launched or not job.handle:
        raise ValueError("The Slurm job has not been submitted.")

    logger.info("Waiting for %s; scheduler polling interval: %s seconds", job.handle, poll_interval)
    with get_runner(experiment=experiment) as runner:
        status = None
        for _attempt in range(3):
            # Leave room after submission for accounting registration and any
            # caller-side preflight. Retry missing accounting at the same rate,
            # never through NeMo-Run's separate fast status/logging loops.
            sleep(poll_interval)
            status = runner.wait(job.handle, wait_interval=poll_interval)
            if status is not None:
                break
            logger.warning("Scheduler accounting has no record for %s yet", job.handle)
    if status is None:
        raise RuntimeError(
            f"Scheduler status is unavailable for {job.handle}; the job has not been cancelled."
        )
    job.state = status.state
    job.cleanup()
    logger.info("Slurm job %s finished: %s", job.handle, job.state)
    if job.state != AppState.SUCCEEDED:
        raise RuntimeError(f"Slurm job {job.handle} did not succeed: {job.state}")

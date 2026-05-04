# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for detecting and configuring SLURM cluster environments.

This module provides functionality to detect SLURM environments and extract
distributed training configuration from SLURM environment variables.
"""

import os
import warnings


def is_slurm_job() -> bool:
    """Detect if running in a SLURM environment.

    Returns:
        True if SLURM job detected, False otherwise.
    """
    return "SLURM_NTASKS" in os.environ


def resolve_slurm_rank() -> int | None:
    """Get the global rank from SLURM environment.

    Returns:
        The global rank, or None if not in SLURM environment.
    """
    if not is_slurm_job():
        return None
    return int(os.environ["SLURM_PROCID"]) if "SLURM_PROCID" in os.environ else None


def resolve_slurm_world_size() -> int | None:
    """Get the world size from SLURM environment.

    Returns:
        The world size, or None if not in SLURM environment.
    """
    if not is_slurm_job():
        return None
    return int(os.environ["SLURM_NTASKS"]) if "SLURM_NTASKS" in os.environ else None


def resolve_slurm_local_rank() -> int | None:
    """Get the local rank from SLURM environment.

    Returns:
        The local rank, or None if not in SLURM environment.
    """
    if not is_slurm_job():
        return None
    return int(os.environ["SLURM_LOCALID"]) if "SLURM_LOCALID" in os.environ else None


def resolve_slurm_master_addr() -> str | None:
    """Parse SLURM_NODELIST to get the master node address.

    Handles common SLURM nodelist formats:
    - Simple list: "node001,node002" -> "node001"
    - Range: "node[001-004]" -> "node001"
    - List in brackets: "node[001,003,005]" -> "node001"

    Returns:
        The master node address, or None if not in SLURM environment.
    """
    if not is_slurm_job():
        return None

    # Try both SLURM_NODELIST and SLURM_JOB_NODELIST
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        # This is an unexpected state - SLURM environment detected but nodelist missing
        warnings.warn(
            "SLURM environment detected (SLURM_NTASKS is set) but SLURM_NODELIST is missing. "
            "This indicates a misconfigured SLURM environment. Falling back to 'localhost'."
        )
        return "localhost"

    return _parse_slurm_nodelist(nodelist)


def resolve_slurm_master_port() -> int | None:
    """Get master port for SLURM job.

    Uses a deterministic port based on SLURM_JOB_ID to avoid conflicts
    when multiple jobs run on the same nodes.
    Returns:
        The master port, or None if not in SLURM environment.
    """
    if not is_slurm_job():
        return None

    # This logic is adapted from PyTorch Lightning's SLURM environment plugin.
    # https://github.com/Lightning-AI/pytorch-lightning/blob/main/src/lightning/fabric/plugins/environments/slurm.py

    # Use SLURM_JOB_ID to generate a deterministic port to avoid conflicts
    # This ensures different jobs on the same nodes use different ports
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return None

    # Use the last 4 digits of the job ID
    default_port = job_id[-4:]
    # All ports should be in the 10k+ range (15000-25000)
    default_port = int(default_port) + 15000
    return default_port


def _parse_slurm_nodelist(nodelist: str) -> str:
    """Parse a SLURM nodelist string and extract the first node.

    Handles common SLURM nodelist formats:
    - Simple list: "node001,node002" -> "node001"
    - Range: "node[001-004]" -> "node001"
    - List in brackets: "node[001,003,005]" -> "node001"

    Args:
        nodelist: The SLURM nodelist string to parse.

    Returns:
        The hostname of the first node in the list.
    """
    # Handle bracket notation: "prefix[range]" or "prefix[list]"
    if "[" in nodelist:
        # Split into base and range part
        # e.g., "node[001-004]" -> base="node", range_part="001-004"
        base = nodelist.split("[")[0]
        range_part = nodelist.split("[")[1].split("]")[0]

        # Handle both ranges (001-004) and lists (001,003,005)
        # Extract first element
        first_element = range_part.split(",")[0].split("-")[0]

        return f"{base}{first_element}"
    else:
        # Simple comma-separated list
        # e.g., "node001,node002,node003" -> "node001"
        return nodelist.split(",")[0].strip()

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release source-owned handoffs after an external supervisor's termination barrier."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

import msgpack
import zmq
import zmq.asyncio

from megatron.core.inference.headers import Headers


@dataclass(frozen=True)
class HandoffRelease:
    """Source identity retained until its release has been acknowledged."""

    coordinator_addr: str
    coordinator_instance_id: str
    request_id: int

    @classmethod
    def from_metadata(cls, metadata: dict) -> HandoffRelease:
        """Reject incomplete metadata before a decode import can be submitted."""
        address = metadata.get("coordinator_addr")
        instance = metadata.get("coordinator_instance_id")
        request_id = metadata.get("request_id")
        if not isinstance(address, str) or not address:
            raise ValueError("Handoff release requires a coordinator address")
        if not isinstance(instance, str) or not instance:
            raise ValueError("Handoff release requires a coordinator instance ID")
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            raise ValueError("Handoff release requires an integer request ID")
        return cls(address, instance, request_id)


async def confirm_terminated(address: str, owner: str, timeout: float = 30.0) -> None:
    """Attest that the old adapter AND every decode rank exited and cannot restart.

    Call on every source coordinator in the deployment, retrying unreachable
    sources. Discovery loss, a timeout or rank-zero death is not sufficient.
    """
    if not owner:
        raise ValueError("A unique handoff owner is required")
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(address)
    try:
        async with asyncio.timeout(timeout):
            await socket.send(msgpack.packb([Headers.CONNECT.value]))
            reply = msgpack.unpackb(await socket.recv(), raw=False)
            if len(reply) != 2 or reply[0] != Headers.CONNECT_ACK.value:
                raise RuntimeError("Source did not advertise its coordinator instance")
            instance = reply[1]
            await socket.send(msgpack.packb([Headers.RELEASE_KV_OWNER.value, instance, owner]))
            reply = msgpack.unpackb(await socket.recv(), raw=False)
            if reply != [Headers.RELEASE_KV_OWNER_ACK.value, instance, owner]:
                raise RuntimeError("Source did not acknowledge owner termination")
    finally:
        socket.close(linger=0)
        context.term()


def main() -> None:
    """Expose the supervisor hook without a database or a persistent volume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-address", action="append", required=True)
    parser.add_argument(
        "--confirm-terminated-owner",
        required=True,
        help="Attest that this attempt's adapter and ALL decode ranks exited and cannot restart.",
    )
    args = parser.parse_args()

    async def release_all():
        await asyncio.gather(
            *(
                confirm_terminated(address, args.confirm_terminated_owner)
                for address in args.coordinator_address
            )
        )

    asyncio.run(release_all())


if __name__ == "__main__":
    main()

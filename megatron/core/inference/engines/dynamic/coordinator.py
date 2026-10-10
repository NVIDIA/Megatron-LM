# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import multiprocessing
import socket
from typing import Dict, List, Optional

import torch

from megatron.core.inference.config import MediaCacheCoordinatorPolicy
from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.inference_request import (
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    DynamicInferenceRequest,
    DynamicVLMInferenceRequest,
    FinishedRequestRecord,
    OffloadedRequestPayload,
    Status,
    merge_multimodal_data,
    resolve_multimodal_data_for_engine,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import await_process_call
from megatron.core.utils import (
    get_asyncio_loop,
    get_pg_rank,
    get_pg_size,
    get_pg_src_rank,
    internal_api,
    nvtx_range_pop,
    nvtx_range_push,
)

from ..async_zmq_communicator import AsyncZMQCommunicator, RankedPubSub
from .loop import EngineState
from .requests import _PROMPT_PREPARATION_ERROR_FIELD

try:
    import zmq

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

try:
    import msgpack

    HAVE_MSGPACK = True
except:
    HAVE_MSGPACK = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Wire encoding of a None offload frame: msgpack nil is the single byte 0xc0,
# i.e. ``msgpack.packb(None, use_bin_type=True)``. Spelled as a literal because
# msgpack is an optional import above. Compared against the frame bytes so a
# request without offload params is recognised without decoding anything.
_PACKED_NONE = b"\xc0"

# Frames the coordinator forwards for a SUBMIT_REQUEST: metadata, prompt,
# media, offload params. The block-hash frame the client sent is consumed there.
_SUBMIT_REQUEST_FRAMES = 4
_SUBMIT_REQUEST_METADATA_FIELDS = 4


def _engine_reply_frames(finished_requests: List[dict]) -> List[bytes]:
    """Frame finished requests as [metadata, body, body, ...] for the coordinator.

    The metadata frame carries only what the coordinator needs to route each
    reply: the request id, and whether it must detokenize into the body. Every
    body stays a separate opaque frame so the coordinator can forward it without
    decoding -- a finished request echoes the prompt back, so decoding it costs
    more than the inbound submission did.

    Args:
        finished_requests: Serialized requests, in the order their frames follow.

    Returns:
        The frames to send, metadata first.
    """
    metadata = [
        Headers.ENGINE_REPLY.value,
        [
            [
                request["request_id"],
                bool((request.get("sampling_params") or {}).get("detokenize_generations")),
            ]
            for request in finished_requests
        ],
    ]
    return [msgpack.packb(metadata, use_bin_type=True)] + [
        msgpack.packb(request, use_bin_type=True) for request in finished_requests
    ]


@internal_api
# pylint: disable=line-too-long
class CoordinatorMixin:
    """ZMQ coordinator transport and EP/world consensus for `DynamicInferenceEngine`."""

    @internal_api
    async def start_listening_to_data_parallel_coordinator(
        self,
        inference_coordinator_port: int | None = None,
        launch_inference_coordinator: bool = True,
        *,
        hostname: str | None = None,
        coordinator_schedule_output_path: str | None = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initializes ZMQ communication to connect the engine with an inference coordinator.

        This asynchronous method sets up the distributed communication infrastructure
        that allows this inference engine to act as a worker under a central
        `InferenceCoordinator`. It configures different ZMQ socket patterns
        based on the rank's role within the distributed topology.

        Note that this method must be called on all ranks, as it uses blocking torch broadcasts.

        The setup involves two primary roles within each data-parallel group:
        1.  **MP Coordinator (TP_rank=0, PP_rank=0)**: This rank connects directly
            to the central coordinator via a ZMQ `DEALER` socket. It receives
            requests and uses a ZMQ `PUB` (publisher) socket to broadcast them
            to all other ranks within its model-parallel (MP) group.
        2.  **MP Workers (all other ranks)**: These ranks use ZMQ `SUB` (subscriber)
            sockets to listen for requests broadcast by their local MP Coordinator.

        This architecture uses TCP sockets for both inter-node and intra-node broadcasts
        within an MP group.

        Finally, after setting up the communication channels and ensuring all ranks
        are synchronized, this method starts the main engine processing loop
        (`self.run_engine`) as a background asyncio task.

        Args:
            inference_coordinator_port (int | None): The network port where the central
                `InferenceCoordinator` is or will be listening.
                If None, a random available port will be selected.
                If not None, the coordinator will attempt to bind to this port, but should it
                not succeed (e.g., if the port is already in use), it may bind to a different port.
                The actual port used is returned by this method.
            launch_inference_coordinator (bool, optional): If True, the global rank 0
                process will spawn and manage the `InferenceCoordinator`
                process. Defaults to True.
            hostname (str | None): Hostname or IP address to use for ZMQ socket binding.
                If None, defaults to `socket.gethostname()`. Should be set to a routable
                address in multi-node settings where gethostname() may return 127.0.0.1.

        Returns:
            inference_coordinator_addresss (str): The network address of the central
                `InferenceCoordinator`, which may not have the same port as what the user requested
                with `inference_coordinator_port`.
        """

        assert HAVE_ZMQ, (
            "please install the pyzmq library to use InferenceCoordinator\n" "pip install pyzmq"
        )
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use InferenceCoordinator\n"
            "pip install msgpack"
        )

        self.zmq_context = zmq.Context.instance()
        self.zmq_sockets = []  # keep track of all sockets created by this engine

        # Get world info.
        dp_group = self.pg_collection.dp
        dp_src = get_pg_src_rank(dp_group)
        dp_size = get_pg_size(self.pg_collection.dp)
        dp_rank = get_pg_rank(self.pg_collection.dp)

        mp_group = self.pg_collection.mp
        mp_src = get_pg_src_rank(mp_group)
        mp_size = get_pg_size(mp_group)
        mp_rank = get_pg_rank(mp_group)
        dp_replica_mp_request_broadcast = RankedPubSub(
            b"DynamicInferenceEngine.mp_request_broadcast:"
        )
        tp_rank = get_pg_rank(self.pg_collection.tp)
        pp_rank = get_pg_rank(self.pg_collection.pp)

        self.is_mp_coordinator = tp_rank == 0 and pp_rank == 0
        self.is_dp_coordinator = (dp_rank == 0) and self.is_mp_coordinator

        local_ip = hostname or socket.gethostname()

        # Spawn a DP coordinator process and get the connection info.
        if launch_inference_coordinator and self.is_dp_coordinator:
            spawn_context = multiprocessing.get_context('spawn')
            deterministic_mode = torch.are_deterministic_algorithms_enabled()
            dp_pipe, dp_process_pipe = spawn_context.Pipe()
            coordinator_ready_event = spawn_context.Event()
            self.inference_coordinator_process = spawn_context.Process(
                target=DataParallelInferenceCoordinator.entrypoint,
                kwargs={
                    "pipe_connection": dp_process_pipe,
                    "ready_event": coordinator_ready_event,
                    "data_parallel_size": get_pg_size(self.pg_collection.dp),
                    "tokenizer": self.controller.tokenizer,
                    "max_requests": self.context.max_requests,
                    "inference_coordinator_port": inference_coordinator_port,
                    "deterministic_mode": deterministic_mode,
                    "block_size_tokens": self.context.block_size_tokens,
                    "enable_prefix_caching": self.context.enable_prefix_caching,
                    "prefix_caching_coordinator_policy": self.context.prefix_caching_coordinator_policy,
                    "prefix_caching_routing_alpha": self.context.prefix_caching_routing_alpha,
                    "prefix_cache_ttl_seconds": self.context.prefix_cache_ttl_seconds,
                    "media_cache_coordinator_policy": getattr(
                        self.context,
                        "media_cache_coordinator_policy",
                        MediaCacheCoordinatorPolicy.AFFINITY,
                    ),
                    "media_cache_routing_weight": getattr(
                        self.context, "media_cache_routing_weight", 1.0
                    ),
                    "vision_embedding_cache_enabled": (self.vision_embedding_cache_max_bytes > 0),
                    "schedule_output_path": coordinator_schedule_output_path,
                    "hostname": hostname,
                },
            )
            self.inference_coordinator_process.start()
            await await_process_call(dp_pipe.poll, self.inference_coordinator_process)
            dp_addr = dp_pipe.recv()
            dp_pipe.close()

            # Check if the port number is not inference_coordinator_port
            actual_port = int(dp_addr.rsplit(":", 1)[-1])
            if inference_coordinator_port != None and actual_port != inference_coordinator_port:
                logger.warning(
                    f"Requested InferenceCoordinator port {inference_coordinator_port} "
                    f"but got port {actual_port} instead. This happens if the request port "
                    f"is already in use."
                )
        elif not launch_inference_coordinator:
            dp_addr = f"tcp://{local_ip}:{inference_coordinator_port}"
        else:
            dp_addr = None

        # Find available ports for MP and bind to them.
        if self.is_mp_coordinator:
            mp_req_sock = dp_replica_mp_request_broadcast.create_publisher(self.zmq_context)
            mp_req_sock.bind_to_random_port(f"tcp://{local_ip}")
            mp_req_addr = mp_req_sock.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            mp_req_addr = None

        # Broadcast addresses to respective ranks.
        bcast = [dp_addr]
        torch.distributed.broadcast_object_list(bcast, src=dp_src, group=dp_group)
        [dp_addr] = bcast
        bcast = [mp_req_addr]
        torch.distributed.broadcast_object_list(bcast, src=mp_src, group=mp_group)
        [mp_req_addr] = bcast

        identity = f'mp-coord-{dp_rank}'
        if self.is_mp_coordinator:
            # 1. Create dealer sockets where tp_rank = 0 and pp_rank = 0
            #    These will receive requests from an InferenceCoordinator.
            self.socket_for_receiving_requests = self.zmq_context.socket(zmq.DEALER)

            self.socket_for_receiving_requests.setsockopt(zmq.IDENTITY, identity.encode('utf-8'))
            self.socket_for_receiving_requests.connect(dp_addr)

            # send empty string. this is used to register with the coordinator.
            self.socket_for_receiving_requests.send(b"")

            # 2. Create a publisher socket. This is used to publish or broadcast
            #    requests within the model parallel group
            self.model_parallel_publisher_socket = mp_req_sock
            self.zmq_sockets += [
                self.socket_for_receiving_requests,
                self.model_parallel_publisher_socket,
            ]
        # All MP ranks subscribe to the publisher socket
        self.model_parallel_subscriber_socket = dp_replica_mp_request_broadcast.create_subscriber(
            self.zmq_context, mp_req_addr, mp_rank
        )

        self.zmq_sockets += [self.model_parallel_subscriber_socket]

        if self.is_mp_coordinator:
            dp_replica_mp_request_broadcast.wait_for_subscribers(
                self.model_parallel_publisher_socket, range(mp_size)
            )

        self._setup_handoff_completion_tracking(hostname)

        torch.distributed.barrier(mp_group)

        # initialize zmq-based EP communicator
        self.ep_rank = get_pg_rank(self.pg_collection.ep)
        self.ep_world_size = get_pg_size(self.pg_collection.ep)
        self._ep_consensus_loop_counter = 0
        self._last_ep_consensus: tuple[int, bool] = (0, False)
        if self.ep_world_size > 1:
            self.expert_parallel_zmq_communicator = AsyncZMQCommunicator(
                self.zmq_context, process_group=self.pg_collection.ep, hostname=hostname
            )
            # Give the context a CPU-side MAX-reduction primitive so
            # match_graph_config() can avoid a per-step NCCL AllReduce kernel.
            if hasattr(self.context, "set_ep_zmq_communicator"):
                self.context.set_ep_zmq_communicator(self.expert_parallel_zmq_communicator)

        # initialize zmq-based world communicator for consensus barriers
        total_world_size = torch.distributed.get_world_size()
        if total_world_size > 1:
            self.world_zmq_communicator = AsyncZMQCommunicator(
                self.zmq_context, process_group=None, hostname=hostname
            )

        if launch_inference_coordinator and self.is_dp_coordinator:
            await await_process_call(
                coordinator_ready_event.wait, self.inference_coordinator_process
            )
            logger.info("Inference co-ordinator is ready to receive requests!")
            logger.info(f"Data parallel coordinator can be found at {dp_addr}")

        # Finally run the engine infinite loop.
        loop = get_asyncio_loop(loop)
        self.engine_loop_task = loop.create_task(self.run_engine_with_coordinator(loop=loop))

        return dp_addr

    def _send_requests_to_coordinator(self, requests: List[DynamicInferenceRequest]) -> None:
        """Send completed or failed flat requests from model-parallel rank 0."""

        # Failed requests are sent immediately but remain in the engine until the next
        # bookkeeping pass; only completed requests are indexed and staged.
        # A reply is stripped only when its payload was staged.
        serialized = []
        for request in requests:
            if request.status == Status.FAILED:
                serialized.append(request.serialize())
                continue
            # One metadata record per completed request serves both the ledger and the stager.
            finished_metadata = None
            if self.local_metadata_ledger_enabled or self.payload_stager is not None:
                finished_metadata = FinishedRequestRecord.from_request(request)
            if self.local_metadata_ledger_enabled:
                assert (
                    request.uid not in self.local_metadata_ledger
                ), f"finished-request ledger: duplicate uid {request.uid!r}"
                self.local_metadata_ledger[request.uid] = finished_metadata
            serialized.append(self._serialize_finished_request(request, finished_metadata))
        self.socket_for_receiving_requests.send_multipart(_engine_reply_frames(serialized))

    def _serialize_finished_request(
        self, request: DynamicInferenceRequest, finished_metadata: Optional[FinishedRequestRecord]
    ) -> Dict:
        """Stage a non-streaming accepted payload before constructing its coordinator reply."""
        stage_result = None
        if self.payload_stager is not None and not getattr(
            request.sampling_params, "streaming", False
        ):
            stage_result = self.payload_stager.stage(
                request.uid,
                OffloadedRequestPayload.from_request(request),
                finished_metadata=finished_metadata,
                offload_params=request.offload_params,
            )
        serialized = request.serialize(
            payload_offloaded=stage_result is not None,
            payload_stage_metadata=(
                stage_result.response_metadata if stage_result is not None else None
            ),
        )
        if isinstance(request, DynamicVLMInferenceRequest):
            # Active VLM requests retain these inputs for refresh/cache bookkeeping,
            # but completion replies must not echo large input-only tensors.
            for key in (
                "imgs",
                "num_tiles",
                "imgs_sizes",
                "num_frames",
                "video_frame_indices",
                "video_fps",
                "image_embeddings",
                "image_token_mask",
            ):
                serialized[key] = None
        return serialized

    def _try_send_streaming_partials(self) -> None:
        """Send pending token deltas to the inference coordinator."""
        partials: list = []
        emit_lengths: Dict[int, int] = {}
        for rid, entry in self.requests.items():
            request = entry.record[-1]
            if not getattr(request.sampling_params, "streaming", False):
                continue
            already = self._partial_emit_lengths.get(rid, 0)
            total = len(request.generated_tokens)
            stop_word_ids = getattr(request, "stop_word_ids", None)
            holdback = 0
            if stop_word_ids and not getattr(
                request.sampling_params, "detokenize_stop_sequence", False
            ):
                holdback = max(0, max(len(ids) for ids in stop_word_ids) - 1)
            emit_end = max(already, total - holdback)
            streaming_interval = getattr(request.sampling_params, "streaming_interval", 1)
            if emit_end - already >= streaming_interval:
                new_tokens = list(request.generated_tokens[already:emit_end])
                partial = {"request_id": rid, "new_tokens": new_tokens}
                if request.sampling_params.return_log_probs:
                    partial["new_log_probs"] = list(
                        (request.generated_log_probs or [])[already:emit_end]
                    )
                    partial["new_top_n_logprobs"] = list(
                        (getattr(request, "generated_top_n_logprobs", None) or [])[already:]
                    )
                    if already == 0 and not request.sampling_params.skip_prompt_log_probs:
                        partial["prompt_log_probs"] = list(
                            getattr(request, "prompt_log_probs", None) or []
                        )
                        partial["prompt_top_n_logprobs"] = list(
                            getattr(request, "prompt_top_n_logprobs", None) or []
                        )
                partials.append(partial)
                emit_lengths[rid] = emit_end

        if not partials:
            return

        nvtx_range_push("coordinator_streaming")
        self.socket_for_receiving_requests.send_multipart(
            [
                msgpack.packb(
                    [Headers.ENGINE_REPLY_PARTIAL.value, [p["request_id"] for p in partials]],
                    use_bin_type=True,
                )
            ]
            + [msgpack.packb(p, use_bin_type=True) for p in partials]
        )
        nvtx_range_pop("coordinator_streaming")

        self._partial_emit_lengths.update(emit_lengths)

    @staticmethod
    def _pack_tp_broadcast(messages: List[List[bytes]]) -> List[bytes]:
        """Flatten per-message frame lists into one TP-broadcast multipart message.

        A message is a list of frames -- metadata first, then any payload bodies.
        ZMQ multipart is flat, so the frame boundaries would be lost on the wire.
        They are carried instead in a manifest frame holding one frame count per
        message, which lets peer ranks rebuild the grouping without any payload
        being copied, decoded, or re-packed.

        Args:
            messages: One frame list per message, in delivery order.

        Returns:
            ``[tp_broadcast_header, manifest, *flattened frames]``.
        """
        manifest = msgpack.packb([len(message) for message in messages], use_bin_type=True)
        return [bytes([Headers.TP_BROADCAST.value]), manifest] + [
            frame for message in messages for frame in message
        ]

    @staticmethod
    def _unpack_tp_broadcast(frames: List[bytes]) -> List[List[bytes]]:
        """Rebuild per-message frame lists from a TP broadcast.

        Inverse of :meth:`_pack_tp_broadcast`.

        Args:
            frames: The received multipart message, header frame first.

        Returns:
            One frame list per message, in the order they were packed.
        """
        frame_counts = msgpack.unpackb(frames[1], raw=False)
        flat = frames[2:]
        messages = []
        offset = 0
        for count in frame_counts:
            messages.append(flat[offset : offset + count])
            offset += count
        assert offset == len(flat), (
            f"TP broadcast manifest accounts for {offset} frames but {len(flat)} were received; "
            "sender and receiver disagree on message framing"
        )
        return messages

    def schedule_requests(self) -> int:
        """Drains the ZMQ socket for a batch of requests and adds them to the engine.

        This method is a collective and synchronous operation that must be called
        by all ranks in a Model Parallel (MP) group at the same time. It ensures
        that all ranks process the exact same batch of incoming requests and
        control signals.

        The synchronization works as follows:
        1.  The MP rank 0 drains all pending messages from its subscriber socket
            in a non-blocking manner.
        2.  MP rank 0 then broadcasts the number of messages it received to all other
            ranks in its MP group using a dedicated publisher socket.
        3.  The other MP ranks wait to receive this count, and then receive exactly
            that many messages from their subscriber sockets.

        Once all ranks have the same batch of messages, they are unpacked and
        processed. New requests are added to the engine's queue, and control
        signals (PAUSE, UNPAUSE, SUSPEND, RESUME, STOP) update the engine's
        internal state.

        Note:
            This function is synchronous and must be called collectively by all
            ranks in a MP group. It should not be launched in a separate coroutine
            to ensure all ranks execute it in lockstep before proceeding to the
            next engine step.

        Returns:
            int: The number of messages that were received and processed in this batch.
        """

        nvtx_range_push("drain_zmq_socket")
        all_messages = []
        if self.is_mp_coordinator:
            # Locally-generated notifications are single-frame messages, so they
            # are wrapped to match the frame-list shape of socket traffic.
            all_messages.extend(
                [
                    msgpack.packb(
                        [Headers.KV_HANDOFF_COMPLETE.value, request_id, failed], use_bin_type=True
                    )
                ]
                for request_id, failed in self._drain_handoff_completion_notifications()
            )
            while True:
                try:
                    # Receive messages in a non-blocking way.
                    message = self.socket_for_receiving_requests.recv_multipart(flags=zmq.NOBLOCK)
                    all_messages.append(self._prepare_submit_request_message(message))
                except zmq.Again:
                    # This exception is hit as soon as the socket is empty.
                    break
            self.model_parallel_publisher_socket.send_multipart(
                self._pack_tp_broadcast(all_messages)
            )
        else:
            all_messages = self._unpack_tp_broadcast(
                self.model_parallel_subscriber_socket.recv_multipart()
            )

        nvtx_range_pop("drain_zmq_socket")

        # First pass: add requests.
        # Control signals are queued for the second pass.
        new_generation_epoch = None
        for message in all_messages:
            data = msgpack.unpackb(message[0], raw=False)
            header = Headers(data[0])
            if header == Headers.SUBMIT_REQUEST:
                # Drop rather than raise: this loop runs on every MP rank over
                # the same broadcast list, so a raise here takes the whole
                # engine down for one version-skewed client, while `continue`
                # stays collective because every rank skips the same message.
                if (
                    len(data) != _SUBMIT_REQUEST_METADATA_FIELDS
                    or len(message) != _SUBMIT_REQUEST_FRAMES
                ):
                    logger.warning(
                        "dropping malformed SUBMIT_REQUEST: %d metadata fields, %d frames "
                        "(expected %d and %d)",
                        len(data),
                        len(message),
                        _SUBMIT_REQUEST_METADATA_FIELDS,
                        _SUBMIT_REQUEST_FRAMES,
                    )
                    continue
                request_id, sampling_params, media_meta = data[1:]
                # The prompt, the media, and the offload params each ride in
                # their own frame; the engine is their first consumer, so this
                # is where they finally get decoded. The coordinator forwarded
                # all three untouched, while the bounded media descriptor in
                # the metadata lets a cache hit avoid decoding the media frame.
                nvtx_range = "megatron.inference.multimodal.message_unpack"
                nvtx_range_push(nvtx_range)
                try:
                    prompt = msgpack.unpackb(message[1], raw=False)
                    media_cache_key = (
                        media_meta.get("media_cache_key") if isinstance(media_meta, dict) else None
                    )
                    media_modality = (
                        media_meta.get("modality") if isinstance(media_meta, dict) else None
                    )
                    cached_vision_entry = self._get_cached_vision_entry(
                        media_cache_key, media_modality
                    )
                    if cached_vision_entry is None:
                        media_payload = msgpack.unpackb(message[2], raw=False)
                        multi_modal_data = merge_multimodal_data(media_meta, media_payload)
                    else:
                        multi_modal_data = None
                    offload_params = msgpack.unpackb(message[3], raw=False)
                    sampling_params = SamplingParams.deserialize(sampling_params)
                finally:
                    nvtx_range_pop(nvtx_range)
                nvtx_range_push("add_request")
                # TODO(perf): uncached media preprocessing (decode / resize /
                # normalize / patchify) runs synchronously on the engine step
                # loop, adding directly to inter-token latency for every
                # in-flight request. Move off the engine thread — either via a
                # bounded ThreadPoolExecutor here or, better, on the
                # server/coordinator side before the ZMQ hop so the engine
                # receives ready tensors.
                try:
                    if cached_vision_entry is not None:
                        vlm_kwargs = {
                            "media_cache_key": media_cache_key,
                            "media_tokens_preexpanded": bool(
                                media_meta.get("media_tokens_preexpanded", False)
                            ),
                        }
                    elif multi_modal_data is None:
                        # Skip the config-attribute lookup for text-only
                        # requests so test fixtures (DummyContext) without an
                        # image_preprocessing_config don't AttributeError on
                        # every SUBMIT_REQUEST and desync the ranks.
                        vlm_kwargs = {}
                    else:
                        nvtx_range = (
                            "megatron.inference.multimodal.resolve_multimodal_data_for_engine"
                        )
                        nvtx_range_push(nvtx_range)
                        try:
                            vlm_kwargs = resolve_multimodal_data_for_engine(
                                multi_modal_data,
                                image_preprocessing_config=(
                                    self.context.config.image_preprocessing_config
                                ),
                                video_preprocessing_config=(
                                    self.context.config.video_preprocessing_config
                                ),
                            )
                        finally:
                            nvtx_range_pop(nvtx_range)
                    if vlm_kwargs:
                        self.add_request(
                            request_id,
                            prompt,
                            sampling_params,
                            offload_params=offload_params,
                            **vlm_kwargs,
                        )
                    else:
                        self.add_request(
                            request_id, prompt, sampling_params, offload_params=offload_params
                        )
                except Exception as error:  # pylint: disable=broad-except
                    self._fail_submission(request_id, sampling_params, error)
                nvtx_range_pop("add_request")
            elif header == Headers.SUBMIT_REQUEST_WITH_KV:
                # Decode-side KV import. As on the plain path, the prompt rides
                # in its own frame and the engine is its first consumer.
                request_id, sampling_params, kv_meta = data[1:]
                prompt = msgpack.unpackb(message[1], raw=False)
                src_block_ids = msgpack.unpackb(message[2], raw=False)
                sampling_params = SamplingParams.deserialize(sampling_params)
                nvtx_range_push("add_request_with_kv_handoff")
                self.add_request_with_kv_handoff(
                    request_id, prompt, sampling_params, kv_meta, src_block_ids
                )
                nvtx_range_pop("add_request_with_kv_handoff")
            elif header == Headers.RELEASE_KV:
                # Coordinator-broadcast release. Unknown request ids are no-ops.
                self.release_handoff_blocks(int(data[1]))
            elif header == Headers.SEND_KV:
                # Push transport: send a pinned hand-off's KV to the decode
                # instance the coordinator picked.
                self.push_handoff_kv(int(data[1]), data[2])
            elif header == Headers.KV_HANDOFF_COMPLETE:
                self._record_handoff_completion_notification(int(data[1]), bool(data[2]))
            elif header == Headers.ABORT_REQUEST:
                request_id = int(data[1])
                entry = self.requests.get(request_id)
                if entry is not None:
                    request = entry.record[-1]
                    # Force active requests to finish on the next step.
                    request.sampling_params.num_tokens_to_generate = len(request.generated_tokens)
                    active_ids = self.context.request_ids[: self.context.total_request_count]
                    matches = torch.where(active_ids == request_id)[0]
                    if matches.numel() > 0:
                        assert matches.numel() == 1
                        idx = int(matches[0].item())
                        self.context.request_output_lengths[idx] = (
                            self.context.request_kv_length_offsets[idx]
                            + self.context.request_query_lengths[idx]
                        )
            elif header == Headers.SET_GENERATION_EPOCH:
                new_generation_epoch = data[1]
            elif header == Headers.START_CUDA_PROFILER:
                # Side-effect, not a state transition: apply immediately on every
                # rank so an outer nsys --capture-range=cudaProfilerApi starts here.
                torch.cuda.cudart().cudaProfilerStart()
            elif header == Headers.STOP_CUDA_PROFILER:
                torch.cuda.cudart().cudaProfilerStop()
            else:
                # Control signal: queue for second pass.
                self._pending_signals.append(message)

        self._poll_pending_kv_imports()
        self._poll_pending_kv_pushes()

        if new_generation_epoch is not None:
            if new_generation_epoch != self._generation_epoch:
                self._invalidate_vision_state()
                # Unlike suspend/resume, an epoch transition does not naturally
                # re-admit requests. Refresh active request-local state here so
                # a later checkpoint cannot retain projections from old weights.
                for entry in self.requests.values():
                    request = entry.record[-1]
                    if isinstance(request, DynamicVLMInferenceRequest):
                        self._refresh_vlm_request_data(request)
            self._generation_epoch = new_generation_epoch
            # Stamp all active requests with the new epoch.
            # Each field stores a sparse list of (start_token_index, epoch) boundaries.
            for entry in self.requests.values():
                request = entry.record[-1]
                total = len(request.prompt_tokens) + len(request.generated_tokens)
                if total > 0:
                    boundary = (total - 1, new_generation_epoch)
                    if request.policy_epoch is None:
                        request.policy_epoch = [(0, new_generation_epoch)]
                    else:
                        request.policy_epoch.append(boundary)
                    if request.kv_cache_epoch is None:
                        request.kv_cache_epoch = [(0, new_generation_epoch)]
                    else:
                        request.kv_cache_epoch.append(boundary)

        # Second pass: apply at most one control signal (the engine loop
        # processes one state transition per iteration).
        if self._pending_signals:
            message = self._pending_signals.popleft()
            data = msgpack.unpackb(message[0], raw=False)
            header = Headers(data[0])

            if header == Headers.PAUSE:
                if self.state == EngineState.RUNNING:
                    self.state = EngineState.PAUSING
                    self._state_events[EngineState.RUNNING].clear()
                # Any other state can safely ignore PAUSE.

            elif header == Headers.UNPAUSE:
                assert self.state == EngineState.PAUSED, f"Received UNPAUSE in state {self.state}"
                self.state = EngineState.UNPAUSING

            elif header == Headers.SUSPEND:
                assert self.state == EngineState.PAUSED, f"Received SUSPEND in state {self.state}"
                self._state_events[EngineState.RESUMED].clear()
                self.suspend()
                self.state = EngineState.SUSPENDING

            elif header == Headers.RESUME:
                assert self.state == EngineState.SUSPENDED, f"Received RESUME in state {self.state}"
                self._state_events[EngineState.SUSPENDED].clear()
                self.resume()
                self.state = EngineState.RESUMING

            elif header == Headers.STOP:
                assert self.state in (
                    EngineState.PAUSED,
                    EngineState.SUSPENDED,
                ), f"Received STOP in state {self.state}"
                if self.state == EngineState.SUSPENDED:
                    self._state_events[EngineState.SUSPENDED].clear()
                self.state = EngineState.STOPPING

            else:
                raise UnknownHeaderError(header)

        self._collect_failed_requests()
        return len(all_messages)

    def _prepare_submit_request_message(self, message: List[bytes]) -> List[bytes]:
        """Resolve a prompt once on MP rank zero before broadcasting the request.

        The preparer only has work when the client sent offload params, so a
        request whose offload frame is None passes through untouched without
        any frame being decoded. When it runs, only the prompt frame and the
        offload frame are rewritten; the metadata frame is never repacked.
        """
        if (
            self.prompt_preparer is None
            or len(message) < _SUBMIT_REQUEST_FRAMES
            or message[3] == _PACKED_NONE
        ):
            return message
        data = msgpack.unpackb(message[0], raw=False)
        if data[0] != Headers.SUBMIT_REQUEST.value or len(data) != _SUBMIT_REQUEST_METADATA_FIELDS:
            return message
        request_id = data[1]
        prompt = msgpack.unpackb(message[1], raw=False)
        offload_params = msgpack.unpackb(message[3], raw=False)
        if isinstance(offload_params, dict) and (
            PREFIX_EXPANDED_TOKEN_COUNT_FIELD in offload_params
        ):
            # Multi-modal prefix and metadata for delayed
            # post-expanded stitching. Skip prompt_preparer.
            return message

        def _pack(prompt, offload_params):
            if isinstance(prompt, torch.Tensor):
                prompt = prompt.tolist()
            return [
                message[0],
                msgpack.packb(prompt, use_bin_type=True),
                message[2],
                msgpack.packb(offload_params, use_bin_type=True),
                *message[4:],
            ]

        try:
            # Packing stays inside the try: an unserializable preparer result must
            # fail this request, not exit rank 0 and hang the other MP ranks.
            result = self.prompt_preparer.prepare_prompt(prompt, offload_params=offload_params)
            return _pack(result.prompt, result.offload_params)
        except Exception as error:  # pylint: disable=broad-except
            logger.exception("prompt preparation failed for request %s", request_id)
            # The original prompt and params came off the wire, so they pack safely.
            failed_params = dict(offload_params) if isinstance(offload_params, dict) else {}
            failed_params[_PROMPT_PREPARATION_ERROR_FIELD] = f"{type(error).__name__}: {error}"
            return _pack(prompt, failed_params)

    async def _close_coordinator_links(self) -> None:
        """Close the coordinator sockets, the EP and world communicators, and the ZMQ context."""
        # ZMQ cleanup; designed to be idempotent.
        sock = getattr(self, 'socket_for_receiving_requests', None)
        if sock is not None and not sock.closed:
            try:
                sock.send(msgpack.packb([Headers.DISCONNECT.value], use_bin_type=True))
            except Exception:
                pass
        for socket in getattr(self, 'zmq_sockets', []):
            socket.close(linger=0)
        if hasattr(self, 'zmq_sockets'):
            self.zmq_sockets.clear()
        if hasattr(self, "expert_parallel_zmq_communicator"):
            self.expert_parallel_zmq_communicator.close()
        if hasattr(self, "world_zmq_communicator"):
            self.world_zmq_communicator.close()
        if not self.zmq_context.closed:
            self.zmq_context.term()

    async def _ep_establish_consensus(
        self, local_work: int, signal_consensus: bool
    ) -> tuple[int, bool]:
        """EP all-reduce to share work counts and pause consensus.

        All-reduces two integers at once:
        - local_work: actual pending request count (always >= 0).
        - consensus flag: -1 if this rank wants to pause, 0 otherwise.

        Using max for both:
        - max(work) > 0 means at least one EP peer has real work.
        - max(consensus) == -1 means ALL peers signaled -1 (all PAUSING).
          Any RUNNING peer contributes 0, pulling the max to 0.

        Args:
            local_work: Pending request count for this rank.
            signal_consensus: True if this rank is ready to pause.
        Returns:
            (global_work, all_pausing): max work across EP, and whether
            all peers signaled consensus.
        """
        nvtx_range_push("_ep_establish_consensus")

        consensus_val = -1 if signal_consensus else 0

        # Signals can be received asynchronously on EP ranks.
        # We do not want a rank to pause prematurely if its peers have yet to receive the signal.
        # So this is an *attempt* to process the signal. This rank has received the signal
        # and passes -1 to the all-reduce. If any other rank in the EP group has not received
        # the signal yet, it will pass a zero value to the all-reduce, hence the global consensus
        # will be zero and we will defer processing the signal.
        # When all ranks receive the signal, global consensus will be -1 and we can process.

        if self.ep_world_size > 1:
            # Note that it is important to use a non-blocking asyncio-friendly all-reduce here.
            # The user may have other tasks running in the event loop that need to be serviced.
            # Do not using a torch.distributed blocking all-reduce here using nccl/gloo.
            # We have tried that and it blocks the event loop in megatron-rl.
            global_work, global_consensus = (
                await self.expert_parallel_zmq_communicator.all_reduce_max(
                    local_work, consensus_val, async_op=(not self.use_synchronous_zmq_collectives)
                )
            )
        else:
            global_work, global_consensus = local_work, consensus_val

        nvtx_range_pop("_ep_establish_consensus")
        return global_work, global_consensus == -1

    async def _world_barrier(self):
        """World-wide ZMQ all-reduce barrier for global rank consensus.

        Used for all state transitions that require global synchronization:
        PAUSING → PAUSED, UNPAUSING → RUNNING, SUSPENDING → SUSPENDED,
        RESUMING → PAUSED, and STOPPING → STOPPED.

        No-op when world_size == 1 (communicator is not created).
        """
        nvtx_range_push("world_barrier")
        if hasattr(self, 'world_zmq_communicator'):
            await self.world_zmq_communicator.all_reduce_max(
                1, async_op=(not self.use_synchronous_zmq_collectives)
            )
        nvtx_range_pop("world_barrier")

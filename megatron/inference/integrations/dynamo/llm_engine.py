# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shlex
import signal
import sys
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Optional

import msgpack
import zmq
import zmq.asyncio
from dynamo._core import Context
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
)
from dynamo.common.backend.health_check import build_health_check_payload, is_probe
from dynamo.common.backend.logprobs import parse_logprob_options
from dynamo.common.backend.publisher import KvEventSource, PushSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.common.model_fetch import fetch_model
from dynamo.llm import KvEventPublisher, ModelInput

from megatron.core.inference.engine_endpoint import InferenceEngineEndpoint
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient, InferenceRequestError
from megatron.core.inference.inference_request import unwrap_serialized_tensors
from megatron.core.inference.sampling_params import SamplingParams
from megatron.inference.integrations.dynamo.args import Config, parse_args
from megatron.inference.integrations.dynamo.telemetry import EngineEventReceiver

logger = logging.getLogger(__name__)


def build_sampling_params(request: GenerateRequest) -> SamplingParams:
    """Translate one Dynamo generation request to Megatron sampling parameters.

    Dynamo's selected generated-token log probabilities are supported with
    ``logprobs=0``. Prompt and top-N log probabilities are rejected because
    Megatron's current wire representation cannot satisfy Dynamo's schema.
    """

    sampling = request.get("sampling_options") or {}
    stop = request.get("stop_conditions") or {}
    params = SamplingParams()
    if sampling.get("temperature") is not None:
        params.temperature = float(sampling["temperature"])
    if sampling.get("top_p") is not None:
        params.top_p = float(sampling["top_p"])
    if sampling.get("top_k") is not None:
        params.top_k = max(0, int(sampling["top_k"]))
    if stop.get("max_tokens") is not None:
        params.num_tokens_to_generate = int(stop["max_tokens"])
    if stop.get("stop"):
        params.stop_words = list(stop["stop"])
    stop_token_ids = list(stop.get("stop_token_ids") or [])
    if stop.get("ignore_eos"):
        params.termination_id = -1
    elif stop_token_ids:
        params.termination_id = int(stop_token_ids[0])
        if len(stop_token_ids) > 1:
            logger.warning(
                "Megatron supports one termination token; ignoring %d additional IDs",
                len(stop_token_ids) - 1,
            )
    output = request.get("output_options") or {}
    token_logprobs, prompt_logprobs = parse_logprob_options(output)
    if prompt_logprobs is not None:
        raise ValueError("Megatron Dynamo backend does not support prompt_logprobs")
    if token_logprobs not in (None, 0):
        raise ValueError(
            "Megatron Dynamo backend supports selected-token logprobs only; use logprobs=0"
        )
    params.top_n_logprobs = 0
    params.return_log_probs = token_logprobs == 0
    params.skip_prompt_log_probs = True
    params.add_attributes({})
    if params.temperature == 0.0:
        params.top_k = 1
        params.top_p = 0.0
    return params


class MegatronLLMEngine(LLMEngine):
    """Unified Dynamo backend for one self-owned Megatron DP replica."""

    def __init__(self, config: Config, registration_model: str | None = None) -> None:
        self.config = config
        self.registration_model = registration_model or config.model
        self.client: Any = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._process_monitor: Optional[asyncio.Task] = None
        self._log_tasks: list[asyncio.Task] = []
        self._shutting_down = False
        self._engine_endpoint: InferenceEngineEndpoint | None = None
        self._ready_messages: queue.Queue[dict] = queue.Queue(maxsize=1)
        self._kv_queue: queue.Queue[tuple[str, dict]] = queue.Queue()
        self._publisher: Optional[KvEventPublisher] = None
        self._publisher_lock = threading.Lock()
        self._event_receiver: Optional[EngineEventReceiver] = None
        self._release_context: zmq.asyncio.Context | None = None
        self._release_sockets: dict[str, zmq.asyncio.Socket] = {}
        self._release_lock = asyncio.Lock()
        self._request_ids: dict[str, int] = {}
        self.worker_id: Optional[int] = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple["MegatronLLMEngine", WorkerConfig]:
        config = parse_args(argv)
        # Dynamo builds model cards independently from WorkerConfig.model_name
        # and EngineConfig.model. Resolve both to one metadata-only snapshot so
        # a Megatron-owned engine never downloads a second copy of its weights.
        model_path = Path(config.model)
        registration_model = (
            str(model_path.resolve())
            if model_path.exists()
            else str(await fetch_model(config.model, ignore_weights=True))
        )
        return cls(config, registration_model), cls.worker_config(config, registration_model)

    @staticmethod
    def worker_config(config: Config, registration_model: str | None = None) -> WorkerConfig:
        mode = {
            "aggregated": DisaggregationMode.AGGREGATED,
            "prefill": DisaggregationMode.PREFILL,
            "decode": DisaggregationMode.DECODE,
        }[config.role]
        return WorkerConfig(
            namespace=config.namespace,
            component=config.component,
            endpoint=config.endpoint,
            model_name=registration_model or config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            endpoint_types=config.endpoint_types,
            discovery_backend=config.discovery_backend,
            request_plane=config.request_plane,
            event_plane=config.event_plane,
            disaggregation_mode=mode,
            enable_kv_routing=True,
        )

    async def start(self, worker_id: int) -> EngineConfig:
        self.worker_id = int(worker_id)
        if not os.path.isdir(self.config.megatron_root):
            raise FileNotFoundError(f"Megatron root does not exist: {self.config.megatron_root}")

        try:
            self._event_receiver = EngineEventReceiver(
                self._on_engine_event, self.config.parent_event_host
            )
            parent_event_address = self._event_receiver.start()
            command = self._engine_command(parent_event_address)
            logger.info("Launching owned Megatron engine: %s", " ".join(command))
            self._process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.config.megatron_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            assert self._process.stdout is not None
            assert self._process.stderr is not None
            self._log_tasks = [
                asyncio.create_task(self._forward_logs(self._process.stdout, logging.INFO)),
                asyncio.create_task(self._forward_logs(self._process.stderr, logging.WARNING)),
            ]
            return await self._complete_startup()
        except BaseException:
            try:
                await self.cleanup()
            except Exception:
                logger.exception("Failed to clean up after Megatron engine startup error")
            raise

    async def _complete_startup(self) -> EngineConfig:
        """Finish startup after the owned engine subprocess has launched."""

        readiness = await self._wait_for_readiness()
        endpoint = InferenceEngineEndpoint.from_dict(readiness)
        self.client = InferenceClient(endpoint.coordinator_address, deserialize=False)
        self.client.start(
            loop=asyncio.get_running_loop(),
            connect_timeout_seconds=min(30.0, self.config.engine_start_timeout),
        )
        self._engine_endpoint = endpoint
        self._process_monitor = asyncio.create_task(self._monitor_process())
        identity = {
            "worker_id": self.worker_id,
            "namespace": self.config.namespace,
            "component": self.config.component,
            "endpoint": self.config.endpoint,
            "role": self.config.role,
        }
        logger.info("Dynamo worker identity: %s", json.dumps(identity, sort_keys=True))
        if self.config.worker_id_file is not None:
            path = Path(self.config.worker_id_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(identity, sort_keys=True) + "\n")
            os.replace(temporary, path)

        capabilities = endpoint.capabilities
        return EngineConfig(
            model=self.registration_model,
            served_model_name=self.config.served_model_name,
            runtime_data={"role": self.config.role, "worker_id": self.worker_id},
            llm=LlmRegistration(
                context_length=capabilities.context_length,
                kv_cache_block_size=capabilities.kv_cache_block_size,
                total_kv_blocks=capabilities.total_kv_blocks,
                max_num_seqs=capabilities.max_num_seqs,
                max_num_batched_tokens=capabilities.max_num_batched_tokens,
                data_parallel_size=capabilities.logical_data_parallel_size,
                data_parallel_start_rank=0,
            ),
        )

    def _engine_command(self, parent_event_address: str) -> list[str]:
        command = [sys.executable, "-m", "torch.distributed.run"]
        if self.config.launcher == "local":
            command.extend(["--standalone", f"--nproc-per-node={self.config.nproc_per_node}"])
        else:
            command.extend(
                [
                    f"--nnodes={self.config.nnodes}",
                    f"--nproc-per-node={self.config.nproc_per_node}",
                    "--node-rank=__SLURM_NODE_RANK__",
                    f"--master-addr={self.config.master_addr}",
                    f"--master-port={self.config.master_port}",
                ]
            )
        command.extend(
            [
                "--module",
                "megatron.inference.integrations.dynamo.engine_service",
                "--dynamo-parent-event-address",
                parent_event_address,
                "--role",
                self.config.role,
            ]
        )
        if self.config.coordinator_host is not None:
            command.extend(["--coordinator-host", self.config.coordinator_host])
        if self.config.coordinator_port is not None:
            command.extend(["--coordinator-port", str(self.config.coordinator_port)])
        command.extend(self.config.megatron_argv)
        if self.config.launcher == "slurm":
            shell_command = shlex.join(command).replace("__SLURM_NODE_RANK__", '"${SLURM_NODEID}"')
            srun_command = [
                "srun",
                f"--nodes={self.config.nnodes}",
                f"--ntasks={self.config.nnodes}",
                "--ntasks-per-node=1",
                f"--gpus-per-node={self.config.nproc_per_node}",
                "--kill-on-bad-exit=1",
            ]
            if self.config.slurm_nodelist is not None:
                srun_command.append(f"--nodelist={self.config.slurm_nodelist}")
            return srun_command + ["bash", "-c", f"exec {shell_command}"]
        return command

    async def _forward_logs(self, stream: asyncio.StreamReader, level: int) -> None:
        while line := await stream.readline():
            logger.log(level, "[megatron-engine] %s", line.decode(errors="replace").rstrip())

    async def _wait_for_readiness(self) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.config.engine_start_timeout
        while True:
            try:
                return self._ready_messages.get_nowait()
            except queue.Empty:
                pass
            if self._process is None:
                raise RuntimeError("Megatron process disappeared during startup")
            if self._process.returncode is not None:
                raise RuntimeError(
                    "Megatron engine exited before readiness "
                    f"with code {self._process.returncode}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the owned Megatron engine to become ready"
                )
            await asyncio.sleep(0.05)

    async def _monitor_process(self) -> None:
        assert self._process is not None
        returncode = await self._process.wait()
        if not self._shutting_down:
            logger.error("Owned Megatron engine exited unexpectedly: %d", returncode)
            os.kill(os.getpid(), signal.SIGTERM)

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self.client is None:
            raise RuntimeError("Megatron engine is not initialized")
        token_ids = list(request.get("token_ids") or [])
        if not token_ids:
            raise ValueError("Megatron backend requires token_ids")
        params = build_sampling_params(request)
        n = int((request.get("sampling_options") or {}).get("n") or 1)
        if n != 1:
            raise ValueError("Megatron Dynamo backend currently supports sampling n=1")
        context_id = str(context.id())

        probe = is_probe(request)
        if probe and self.config.role == "decode":
            yield {
                "token_ids": [],
                "index": 0,
                "finish_reason": "stop",
                "completion_usage": {
                    "prompt_tokens": len(token_ids),
                    "completion_tokens": 0,
                    "total_tokens": len(token_ids),
                },
            }
            return
        if self.config.role == "prefill" and not probe:
            endpoint = self._engine_endpoint
            if endpoint is None:
                raise RuntimeError("Megatron engine endpoint is not initialized")
            params.do_kv_handoff = True
            params.num_tokens_to_generate = 0
            stream = self.client.add_request_streaming(token_ids, params)
            self._request_ids[context_id] = stream.request_id
            final = None
            try:
                async for item in stream:
                    if "final" in item:
                        final = item["final"]
                        break
                if final is None:
                    raise RuntimeError("Megatron prefill stream ended without a result")
                disagg = dict(final.get("disaggregated_params") or {})
                disagg["release"] = {
                    "coordinator_addr": endpoint.coordinator_address,
                    "request_id": int(
                        disagg.get("request_id", final.get("request_id", stream.request_id))
                    ),
                }
                yield {
                    "token_ids": [],
                    "index": 0,
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": len(token_ids),
                        "completion_tokens": 0,
                        "total_tokens": len(token_ids),
                    },
                    "disaggregated_params": disagg,
                }
            finally:
                if final is None:
                    await stream.aclose()
                self._request_ids.pop(context_id, None)
            return

        release: dict[str, Any] = {}
        if self.config.role == "decode" and not probe:
            prefill = require_prefill_result(request, DisaggregationMode.DECODE)
            disagg = prefill.get("disaggregated_params") or {}
            release = disagg.get("release") or {}
            stream = self.client.add_request_with_kv_handoff_streaming(
                token_ids, params, disagg.get("kv_meta") or {}, list(disagg.get("block_ids") or [])
            )
        else:
            stream = self.client.add_request_streaming(token_ids, params)
        self._request_ids[context_id] = stream.request_id

        released = False
        source_safe = False
        try:
            async for chunk in self._stream_chunks(stream, token_ids, params):
                source_safe = True
                yield chunk
                if not released and self.config.role == "decode":
                    released = await self._release_handoff_from_meta_async(release)
        except InferenceRequestError as error:
            source_safe = error.source_safe
            raise
        finally:
            self._request_ids.pop(context_id, None)
            if not released and self.config.role == "decode":
                try:
                    if not source_safe:
                        source_safe = await asyncio.wait_for(
                            self.client.abort_request_and_wait(stream.request_id),
                            timeout=self.config.drain_timeout,
                        )
                    if source_safe:
                        await self._release_handoff_from_meta_async(release)
                except Exception:
                    logger.exception("Failed to finish cancelled Megatron handoff")

    async def _stream_chunks(
        self, stream, prompt_token_ids: list[int], params: SamplingParams
    ) -> AsyncGenerator[GenerateChunk, None]:
        completion_tokens = 0
        completed = False
        try:
            async for item in stream:
                if "partial" in item:
                    partial = item["partial"]
                    tokens = list(partial.get("new_tokens") or [])
                    log_probs = list(partial.get("new_log_probs") or [])
                    self._validate_log_probs(tokens, log_probs, params)
                    completion_tokens += len(tokens)
                    chunk: GenerateChunk = {"token_ids": tokens, "index": 0}
                    if params.return_log_probs:
                        chunk["log_probs"] = log_probs
                    yield chunk
                    continue

                final = item.get("final")
                if final is None:
                    continue
                final = unwrap_serialized_tensors(final)
                all_tokens = list(final.get("generated_tokens") or [])
                all_log_probs = list(final.get("generated_log_probs") or [])
                tokens = all_tokens[completion_tokens:]
                log_probs = all_log_probs[completion_tokens:]
                self._validate_log_probs(tokens, log_probs, params)
                completion_tokens = len(all_tokens)
                max_tokens = params.num_tokens_to_generate
                chunk: GenerateChunk = {
                    "token_ids": tokens,
                    "index": 0,
                    "finish_reason": (
                        "length"
                        if max_tokens is not None and completion_tokens >= max_tokens
                        else "stop"
                    ),
                    "completion_usage": {
                        "prompt_tokens": len(prompt_token_ids),
                        "completion_tokens": completion_tokens,
                        "total_tokens": len(prompt_token_ids) + completion_tokens,
                    },
                }
                if params.return_log_probs:
                    chunk["log_probs"] = log_probs
                completed = True
                yield chunk
                return
            raise RuntimeError("Megatron stream ended without a terminal result")
        finally:
            if not completed:
                await stream.aclose()

    @staticmethod
    def _validate_log_probs(
        token_ids: list[int], log_probs: list[float], params: SamplingParams
    ) -> None:
        """Keep Dynamo's token and selected-logprob arrays position-aligned."""

        if params.return_log_probs and len(log_probs) != len(token_ids):
            raise RuntimeError(
                "Megatron returned "
                f"{len(log_probs)} selected log probabilities for "
                f"{len(token_ids)} generated tokens"
            )

    async def _release_remote_handoff(self, address: str, request_id: int) -> None:
        async with self._release_lock:
            socket = self._release_sockets.get(address)
            if socket is None:
                if self._release_context is None:
                    self._release_context = zmq.asyncio.Context()
                socket = self._release_context.socket(zmq.DEALER)
                socket.setsockopt(zmq.SNDHWM, 0)
                socket.setsockopt(zmq.RCVHWM, 0)
                socket.connect(address)
                await socket.send(msgpack.packb([Headers.CONNECT.value], use_bin_type=True))
                try:
                    reply = await asyncio.wait_for(
                        socket.recv(), timeout=min(30.0, self.config.engine_start_timeout)
                    )
                except Exception:
                    socket.close(linger=0)
                    raise
                if Headers(msgpack.unpackb(reply, raw=False)[0]) != Headers.CONNECT_ACK:
                    socket.close(linger=0)
                    raise RuntimeError("Unexpected handoff release coordinator reply")
                self._release_sockets[address] = socket
            await socket.send(
                msgpack.packb([Headers.RELEASE_KV.value, int(request_id)], use_bin_type=True)
            )

    async def _release_handoff_from_meta_async(self, release: dict[str, Any]) -> bool:
        """Release source state without blocking Dynamo's request loop."""

        if release.get("coordinator_addr") is None or release.get("request_id") is None:
            return False
        await self._release_remote_handoff(
            str(release["coordinator_addr"]), int(release["request_id"])
        )
        return True

    async def abort(self, context: Context) -> None:
        request_id = self._request_ids.pop(str(context.id()), None)
        if request_id is not None and self.client is not None:
            await asyncio.wait_for(
                self.client.abort_request_and_wait(request_id), timeout=self.config.drain_timeout
            )

    async def drain(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.config.drain_timeout
        while self._request_ids:
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("Timed out draining Megatron requests")
                return
            await asyncio.sleep(0.05)

    async def cleanup(self) -> None:
        self._shutting_down = True
        for socket in self._release_sockets.values():
            socket.close(linger=0)
        self._release_sockets.clear()
        if self._release_context is not None:
            self._release_context.term()
            self._release_context = None
        client = self.client
        self.client = None
        self._engine_endpoint = None
        if client is not None:
            try:
                client.pause_engines()
                client.stop_engines()
                client.shutdown_coordinator()
            except Exception:
                logger.exception("Graceful Megatron coordinator shutdown failed")

        await self._wait_or_terminate_process()
        if client is not None:
            try:
                client.stop()
            except Exception:
                logger.exception("Failed to close Megatron coordinator client")
        if self._process_monitor is not None:
            self._process_monitor.cancel()
            await asyncio.gather(self._process_monitor, return_exceptions=True)
            self._process_monitor = None
        if self._event_receiver is not None:
            self._event_receiver.stop()
            self._event_receiver = None
        with self._publisher_lock:
            self._publisher = None
        if self._log_tasks:
            await asyncio.gather(*self._log_tasks, return_exceptions=True)
            self._log_tasks.clear()

    async def _wait_or_terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=self.config.engine_shutdown_timeout)
        except asyncio.TimeoutError:
            logger.warning("Terminating unresponsive Megatron engine process group")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.error("Killing unresponsive Megatron engine process group")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        finally:
            self._process = None

    async def health_check_payload(self) -> dict[str, Any] | None:
        endpoint = self._engine_endpoint
        if self.client is None or endpoint is None:
            return None
        return build_health_check_payload(endpoint.capabilities.bos_token_id)

    async def kv_event_sources(self) -> list[KvEventSource]:
        endpoint = self._engine_endpoint
        if self.config.role == "decode" or self.client is None or endpoint is None:
            return []
        if not endpoint.capabilities.enable_prefix_caching:
            return []
        return [PushSource(on_ready=self._set_publisher, dp_rank=0)]

    def _on_engine_event(self, kind: str, payload: dict) -> None:
        if kind == "ready":
            try:
                self._ready_messages.put_nowait(payload)
            except queue.Full:
                logger.warning("Ignoring duplicate Megatron readiness message")
            return
        if kind in ("stored", "removed", "cleared"):
            with self._publisher_lock:
                if self._publisher is None:
                    self._kv_queue.put((kind, payload))
                else:
                    self._publish_event(self._publisher, kind, payload)
            return
        logger.warning("Ignoring unknown Megatron engine event %s", kind)

    def _set_publisher(self, publisher: KvEventPublisher) -> None:
        with self._publisher_lock:
            self._publisher = publisher
            try:
                while True:
                    kind, payload = self._kv_queue.get_nowait()
                    self._publish_event(publisher, kind, payload)
            except queue.Empty:
                pass

    @staticmethod
    def _publish_event(publisher: KvEventPublisher, kind: str, payload: dict) -> None:
        try:
            if kind == "stored":
                publisher.publish_stored(**payload)
            elif kind == "removed":
                publisher.publish_removed(**payload)
            else:
                publisher.publish_all_cleared()
        except Exception:
            logger.exception("Failed to publish Megatron KV event %s", kind)
